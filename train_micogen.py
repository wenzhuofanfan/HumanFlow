import argparse
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from omegaconf import OmegaConf
from copy import deepcopy
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_dream_and_update_latents, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange
from src.flux.sampling import denoise, denoise_controlnet, get_noise, get_schedule, prepare, unpack
from src.flux.util import (configs, load_ae, load_clip,
                       load_flow_model2, load_controlnet, load_t5)
from PIL import Image
if is_wandb_available():
    import wandb
logger = get_logger(__name__, log_level="INFO")

def get_models(name: str, device, offload: bool, is_schnell: bool):
    t5 = load_t5(device, max_length=256 if is_schnell else 512)
    clip = load_clip(device)
    model = load_flow_model2(name, device="cpu")
    vae = load_ae(name, device="cpu" if offload else device)
    return model, vae, t5, clip

def get_dataloader(args):
    """Load appropriate dataset based on config."""
    data_config = args.data_config
    
    # Determine dataset type
    if 'data_root' in data_config and 'control_type' in data_config:
        # Use MiCoGen dataset
        from image_datasets.micogen_dataset import loader
        logger.info(f"Using MiCoGen dataset with control_type='{data_config.control_type}'")
    else:
        # Use original canny dataset
        from image_datasets.canny_dataset import loader
        logger.info("Using original Canny dataset")
    
    return loader(**data_config)

def sample_validation_data_from_jsonl(data_root, control_type, num_samples=3, seed=42, enhance_prompts=True):
    """
    Sample validation prompts and control images from the dataset jsonl file.
    
    Args:
        data_root: path to dataset root
        control_type: type of control signal
        num_samples: number of validation samples
        seed: random seed for reproducibility
        enhance_prompts: if True, add full-body keywords to prompts
        
    Returns:
        validation_prompts: list of text prompts
        validation_control_images: list of control image paths
    """
    import json
    import random
    
    jsonl_path = os.path.join(data_root, 'img_text_pair.jsonl')
    
    # Load all samples
    samples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    
    # Set seed for reproducibility
    random.seed(seed)
    selected_samples = random.sample(samples, min(num_samples, len(samples)))
    
    validation_prompts = []
    validation_control_images = []
    
    for sample in selected_samples:
        img_name = sample['img_path']
        prompt = sample['prompt']
        control_path = os.path.join(data_root, 'control_signals', control_type, img_name)
        
        # Enhance prompt with full-body keywords
        if enhance_prompts:
            # Add full-body prefix if not already mentioned
            full_body_keywords = ['full body', 'whole body', 'standing', 'from head to toe']
            has_full_body = any(kw in prompt.lower() for kw in full_body_keywords)
            if not has_full_body:
                prompt = f"Full body shot of a person. {prompt}"
        
        validation_prompts.append(prompt)
        validation_control_images.append(control_path)
    
    logger.info(f"Sampled {len(validation_prompts)} validation samples from dataset")
    if enhance_prompts:
        logger.info("Prompts enhanced with full-body keywords")
    for i, (prompt, path) in enumerate(zip(validation_prompts, validation_control_images)):
        logger.info(f"  [{i}] {prompt[:80]}...")
    
    return validation_prompts, validation_control_images

@torch.no_grad()
def generate_validation_images(
    controlnet,
    dit,
    vae,
    t5,
    clip,
    device,
    output_dir,
    global_step,
    validation_prompts=None,
    validation_control_images=None,
    img_size=512,
    num_inference_steps=20,
    control_weight=0.9
):
    """
    Generate validation images during training.
    
    Args:
        controlnet: ControlNet model
        dit: Main Flux model
        vae: VAE model
        t5: T5 text encoder
        clip: CLIP text encoder
        device: Device to run on
        output_dir: Output directory for saving images
        global_step: Current training step
        validation_prompts: List of prompts (if None, use default)
        validation_control_images: List of control image paths (if None, use first batch)
        img_size: Image size
        num_inference_steps: Number of sampling steps
        control_weight: Control guidance strength
    """
    # Set models to eval mode
    controlnet.eval()
    dit.eval()
    
    # Use bfloat16 for inference (matching training)
    weight_dtype = torch.bfloat16
    
    # Ensure all models are in the correct dtype
    original_controlnet_dtype = next(controlnet.parameters()).dtype
    original_dit_dtype = next(dit.parameters()).dtype
    
    if original_controlnet_dtype != weight_dtype:
        controlnet = controlnet.to(weight_dtype)
    if original_dit_dtype != weight_dtype:
        dit = dit.to(weight_dtype)
    
    # Default validation prompts
    if validation_prompts is None:
        validation_prompts = [
            "a person in elegant dress",
            "a young woman with long hair",
            "a man in casual clothing"
        ]
    
    # Create output directory
    sample_dir = os.path.join(output_dir, "validation_samples")
    os.makedirs(sample_dir, exist_ok=True)
    
    validation_images = []
    
    for idx, prompt in enumerate(validation_prompts):
        try:
            # Load or use provided control image
            if validation_control_images and idx < len(validation_control_images):
                control_image_path = validation_control_images[idx]
                control_image = Image.open(control_image_path).convert('RGB')
                # Use same preprocessing as training (no crop, just resize)
                control_image = control_image.resize((img_size, img_size), Image.LANCZOS)
                control_image = torch.from_numpy((np.array(control_image) / 127.5) - 1.0)
                control_image = control_image.permute(2, 0, 1).unsqueeze(0).to(device, dtype=weight_dtype)
            else:
                # Skip if no control image available
                logger.warning(f"No control image for prompt {idx}, skipping")
                continue
            
            # Generate noise (in weight_dtype)
            x = get_noise(
                1, img_size, img_size,
                device=device,
                dtype=weight_dtype,
                seed=42 + idx
            )
            
            # Prepare text conditioning
            inp = prepare(t5=t5, clip=clip, img=x, prompt=prompt)
            
            # Get timesteps
            timesteps = get_schedule(
                num_inference_steps,
                (img_size // 8) * (img_size // 8) // (16 * 16),
                shift=True,
            )
            
            # Denoise with ControlNet using autocast for consistent dtype
            with torch.autocast(device_type='cuda', dtype=weight_dtype):
                x = denoise_controlnet(
                    model=dit,
                    controlnet=controlnet,
                    img=inp['img'],
                    img_ids=inp['img_ids'],
                    txt=inp['txt'],
                    txt_ids=inp['txt_ids'],
                    vec=inp['vec'],
                    neg_txt=inp['txt'],  # Use same as positive for simplicity
                    neg_txt_ids=inp['txt_ids'],
                    neg_vec=inp['vec'],
                    controlnet_cond=control_image,
                    timesteps=timesteps,  # timesteps is a list, not a tensor
                    guidance=4.0,
                    true_gs=1.0,
                    controlnet_gs=control_weight,
                    timestep_to_start_cfg=0,
                )
            
            # Decode (convert to float32 for VAE)
            x = unpack(x.float(), img_size, img_size)
            x = vae.decode(x.to(torch.float32))
            
            # Convert to image
            x = x.clamp(-1, 1)
            x = rearrange(x[0], "c h w -> h w c")
            output_img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
            
            # Save image
            save_path = os.path.join(sample_dir, f"step_{global_step}_prompt_{idx}.png")
            output_img.save(save_path)
            
            # Also save control image for comparison
            control_save_path = os.path.join(sample_dir, f"step_{global_step}_control_{idx}.png")
            control_img_np = ((control_image[0].float().permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).astype(np.uint8)
            Image.fromarray(control_img_np).save(control_save_path)
            
            logger.info(f"Saved validation image {idx} to {save_path}")
            
            # Prepare for W&B logging
            if is_wandb_available():
                validation_images.append(wandb.Image(
                    output_img,
                    caption=f"Step {global_step} - {prompt[:50]}"
                ))
                validation_images.append(wandb.Image(
                    Image.fromarray(control_img_np),
                    caption=f"Control {idx}"
                ))
        
        except Exception as e:
            logger.warning(f"Failed to generate validation image {idx}: {e}")
            import traceback
            logger.warning(traceback.format_exc())
            continue
    
    # Restore original dtypes
    if original_controlnet_dtype != weight_dtype:
        controlnet = controlnet.to(original_controlnet_dtype)
    if original_dit_dtype != weight_dtype:
        dit = dit.to(original_dit_dtype)
    
    # Restore training mode
    controlnet.train()
    
    # Log to W&B
    if is_wandb_available() and len(validation_images) > 0:
        wandb.log({f"Validation Images": validation_images}, step=global_step)
    
    # Return models to training mode
    controlnet.train()
    dit.eval()  # Keep dit in eval mode (frozen during training)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="path to config",
    )
    args = parser.parse_args()


    return args.config
def main():

    args = OmegaConf.load(parse_args())
    is_schnell = args.model_name == "flux-schnell"
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    print("DEVICE", accelerator.device)
    dit, vae, t5, clip = get_models(name=args.model_name, device=accelerator.device, offload=False, is_schnell=is_schnell)

    vae.requires_grad_(False)
    t5.requires_grad_(False)
    clip.requires_grad_(False)
    dit.requires_grad_(False)
    dit.to(accelerator.device)

    # Read control_type from data_config
    control_type = args.data_config.get('control_type', 'canny')
    logger.info(f"Initializing ControlNet with control_type='{control_type}'")
    
    controlnet = load_controlnet(
        name=args.model_name, 
        device=accelerator.device, 
        transformer=dit,
        control_type=control_type  # pass control type
    )
    controlnet = controlnet.to(torch.float32)
    controlnet.train()

    optimizer_cls = torch.optim.AdamW

    print(sum([p.numel() for p in controlnet.parameters() if p.requires_grad]) / 1000000, 'parameters')
    optimizer = optimizer_cls(
        [p for p in controlnet.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataloader = get_dataloader(args)
    
    # Sample validation data from real dataset if using MiCoGen
    if 'data_root' in args.data_config and 'control_type' in args.data_config:
        validation_prompts, validation_control_images = sample_validation_data_from_jsonl(
            data_root=args.data_config['data_root'],
            control_type=args.data_config['control_type'],
            num_samples=args.get('num_validation_samples', 3),
            seed=args.get('validation_seed', 42),
            enhance_prompts=args.get('enhance_validation_prompts', True)  # Default: enhance prompts
        )
        # Override config values with real data
        args.validation_prompts = validation_prompts
        args.validation_control_images = validation_control_images
        logger.info(f"Set {len(validation_prompts)} validation prompts and control images")
    else:
        logger.warning("Not using MiCoGen dataset, validation data not auto-sampled")
    
    # Initialize HTCL if enabled (Improved version with multi-templates)
    htcl_wrapper = None
    use_htcl = args.get('use_htcl', False)
    if use_htcl:
        try:
            from htcl import HTCLWrapper
            htcl_template_path = args.get('htcl_template_path', 'ckpts/htcl_multi_templates.pt')
            htcl_loss_weight = args.get('htcl_loss_weight', 0.2)
            htcl_conf_threshold = args.get('htcl_conf_threshold', 0.3)
            htcl_conf_penalty_weight = args.get('htcl_conf_penalty_weight', 1.0)
            htcl_bone_loss_weight = args.get('htcl_bone_loss_weight', 1.0)
            htcl_angle_loss_weight = args.get('htcl_angle_loss_weight', 0.5)
            htcl_symmetry_loss_weight = args.get('htcl_symmetry_loss_weight', 0.3)
            
            logger.info(f"Initializing improved HTCL with multi-templates")
            logger.info(f"  Template path: {htcl_template_path}")
            logger.info(f"  Loss weight: {htcl_loss_weight}")
            logger.info(f"  Conf threshold: {htcl_conf_threshold}")
            
            htcl_wrapper = HTCLWrapper(
                template_path=htcl_template_path,  # Updated API: template_path instead of template_matrix_path
                pose_model_type='hrnet',
                device=accelerator.device,
                loss_weight=htcl_loss_weight,
                conf_threshold=htcl_conf_threshold,
                conf_penalty_weight=htcl_conf_penalty_weight,
                bone_loss_weight=htcl_bone_loss_weight,
                angle_loss_weight=htcl_angle_loss_weight,
                symmetry_loss_weight=htcl_symmetry_loss_weight
            )
            htcl_wrapper.eval()
            for param in htcl_wrapper.parameters():
                param.requires_grad = False
            logger.info("✓ Improved HTCL initialized and frozen")
        except Exception as e:
            logger.warning(f"Failed to initialize HTCL: {e}. Training without HTCL.")
            logger.warning(f"Please ensure you have generated multi-templates using: python compute_htcl_template.py")
            htcl_wrapper = None
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    controlnet, optimizer, _, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, deepcopy(train_dataloader), lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision


    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, {"test": None})

    timesteps = list(torch.linspace(1, 0, 1000).numpy())
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                img, control_image, prompts = batch
                control_image = control_image.to(accelerator.device)
                
                # Store original img for HTCL (before VAE encoding)
                img_for_htcl = img.clone() if htcl_wrapper is not None else None
                
                with torch.no_grad():
                    x_1 = vae.encode(img.to(accelerator.device).to(torch.float32))
                    inp = prepare(t5=t5, clip=clip, img=x_1, prompt=prompts)

                    x_1 = rearrange(x_1, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

                bs = img.shape[0]
                t = torch.sigmoid(torch.randn((bs,), device=accelerator.device))

                x_0 = torch.randn_like(x_1).to(accelerator.device)
                x_t = (1 - t.unsqueeze(1).unsqueeze(2).repeat(1, x_1.shape[1], x_1.shape[2])) * x_1 + t.unsqueeze(1).unsqueeze(2).repeat(1, x_1.shape[1], x_1.shape[2]) * x_0
                bsz = x_1.shape[0]
                guidance_vec = torch.full((x_t.shape[0],), 4, device=x_t.device, dtype=x_t.dtype)

                block_res_samples = controlnet(
                    img=x_t.to(weight_dtype),
                    img_ids=inp['img_ids'].to(weight_dtype),
                    controlnet_cond=control_image.to(weight_dtype),
                    txt=inp['txt'].to(weight_dtype),
                    txt_ids=inp['txt_ids'].to(weight_dtype),
                    y=inp['vec'].to(weight_dtype),
                    timesteps=t.to(weight_dtype),
                    guidance=guidance_vec.to(weight_dtype),
                )
                # Predict the noise residual and compute loss
                model_pred = dit(
                    img=x_t.to(weight_dtype),
                    img_ids=inp['img_ids'].to(weight_dtype),
                    txt=inp['txt'].to(weight_dtype),
                    txt_ids=inp['txt_ids'].to(weight_dtype),
                    block_controlnet_hidden_states=[
                        sample.to(dtype=weight_dtype) for sample in block_res_samples
                    ],
                    y=inp['vec'].to(weight_dtype),
                    timesteps=t.to(weight_dtype),
                    guidance=guidance_vec.to(weight_dtype),
                )

                # Flow matching velocity loss
                velocity_loss = F.mse_loss(model_pred.float(), (x_0 - x_1).float(), reduction="mean")
                loss = velocity_loss
                
                # Add HTCL loss if enabled (Improved version)
                htcl_loss_value = 0.0
                htcl_info = {}
                htcl_compute_every = args.get('htcl_compute_every', 1)
                if htcl_wrapper is not None and global_step % htcl_compute_every == 0:
                    try:
                        # Decode predicted latent at t=0 (clean state) for HTCL
                        # Use model prediction to estimate x_0_pred
                        # NOTE: Keep gradients enabled for HTCL to work!
                        x_0_pred = x_t - t.unsqueeze(1).unsqueeze(2).repeat(1, x_1.shape[1], x_1.shape[2]) * model_pred
                        x_0_pred_unpacked = unpack(x_0_pred.float(), img.shape[2], img.shape[3])
                        decoded_imgs = vae.decode(x_0_pred_unpacked)
                        
                        # Compute improved HTCL on decoded images
                        # New API returns (loss, info_dict)
                        htcl_loss, htcl_info = htcl_wrapper.compute_loss(decoded_imgs.float())
                        loss = loss + htcl_loss
                        htcl_loss_value = htcl_loss.item()
                        
                        # Log detailed HTCL info every 100 steps
                        if global_step % 100 == 0:
                            logger.info(f"HTCL Info - Topology: {htcl_info.get('topology_loss', 0):.4f}, "
                                      f"Conf: {htcl_info.get('avg_confidence', 0):.3f}, "
                                      f"Valid bones: {htcl_info.get('bone_valid_ratio', 0):.3f}, "
                                      f"Valid angles: {htcl_info.get('angle_valid_ratio', 0):.3f}")
                    except Exception as e:
                        # If HTCL computation fails, continue without it
                        logger.warning(f"HTCL computation failed at step {global_step}: {e}")
                        htcl_loss_value = 0.0
                        htcl_info = {}

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                log_dict = {"train_loss": train_loss, "velocity_loss": velocity_loss.item()}
                if htcl_wrapper is not None:
                    log_dict["htcl_loss"] = htcl_loss_value
                    # Add detailed HTCL metrics if available
                    if htcl_info:
                        log_dict["htcl_topology_loss"] = htcl_info.get('topology_loss', 0)
                        log_dict["htcl_confidence_penalty"] = htcl_info.get('confidence_penalty', 0)
                        log_dict["htcl_avg_confidence"] = htcl_info.get('avg_confidence', 0)
                        log_dict["htcl_bone_valid_ratio"] = htcl_info.get('bone_valid_ratio', 0)
                        log_dict["htcl_angle_valid_ratio"] = htcl_info.get('angle_valid_ratio', 0)
                accelerator.log(log_dict, step=global_step)
                train_loss = 0.0
                
                # Generate validation images
                sample_every = args.get('sample_every', 5000)
                if global_step % sample_every == 0 and global_step > 0:
                    if accelerator.is_main_process:
                        val_prompts = args.get('validation_prompts', None)
                        val_images = args.get('validation_control_images', None)
                        logger.info(f"Generating validation images at step {global_step}...")
                        logger.info(f"Validation prompts: {val_prompts is not None}, images: {val_images is not None}")
                        
                        try:
                            generate_validation_images(
                                controlnet=accelerator.unwrap_model(controlnet),
                                dit=dit,
                                vae=vae,
                                t5=t5,
                                clip=clip,
                                device=accelerator.device,
                                output_dir=args.output_dir,
                                global_step=global_step,
                                validation_prompts=val_prompts,
                                validation_control_images=val_images,
                                img_size=args.data_config.get('img_size', 512),
                                num_inference_steps=args.get('num_inference_steps', 20),
                                control_weight=args.get('validation_control_weight', 0.9)
                            )
                            logger.info(f"Successfully generated validation images at step {global_step}")
                        except Exception as e:
                            logger.error(f"Failed to generate validation images: {e}")
                            import traceback
                            logger.error(traceback.format_exc())

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    #if not os.path.exists(save_path):
                    #        os.mkdir(save_path)

                    accelerator.save_state(save_path)
                    unwrapped_model = accelerator.unwrap_model(controlnet)

                    torch.save(unwrapped_model.state_dict(), os.path.join(save_path, 'controlnet.bin'))
                    logger.info(f"Saved state to {save_path}")


            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if htcl_wrapper is not None and htcl_loss_value > 0:
                logs["htcl"] = htcl_loss_value
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
