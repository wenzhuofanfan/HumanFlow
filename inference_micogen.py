#!/usr/bin/env python3
"""
MiCoGen inference script.

Supports loading checkpoints for different control conditions and optional
SDXL Refiner post-processing.

Author: fwz
Date: 2025.10
"""

import argparse
import os
import json
import time
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionXLImg2ImgPipeline
from src.flux.humanflow_pipeline import HumanFlowPipeline


def load_jsonl(jsonl_path):
    """Load samples from a jsonl file."""
    samples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return samples


def get_gpu_memory():
    """Return peak GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3  # convert to GB
    return 0


def clear_gpu_cache():
    """Clear GPU cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def create_argparser():
    parser = argparse.ArgumentParser(description='MiCoGen ControlNet inference script')
    
    # Model and checkpoint paths
    parser.add_argument(
        '--checkpoint_path', 
        type=str, 
        default=None,
        help='Path to trained ControlNet checkpoint (for single-model inference)'
    )
    parser.add_argument(
        '--control_type', 
        type=str, 
        default=None,
        choices=['canny', 'densepose', 'depth', 'keypoint', 'normal', 'seg'],
        help='Control condition type (for single-model inference)'
    )
    parser.add_argument(
        '--batch_inference',
        action='store_true',
        default=True,
        help='Whether to run batch inference for all control conditions'
    )
    
    # Data paths
    parser.add_argument(
        '--test_data_root',
        type=str,
        default='/root/autodl-tmp/HumanFlow/datasets/MiCoGen_test_512',
        help='Test dataset root directory (512x512 preprocessed version)'
    )
    parser.add_argument(
        '--jsonl_path',
        type=str,
        default=None,
        help='Path to jsonl file; defaults to test_data_root/img_text_pair.jsonl'
    )
    
    # Inference parameters
    parser.add_argument(
        '--model_type',
        type=str,
        default='flux-dev',
        choices=['flux-dev', 'flux-dev-fp8', 'flux-schnell'],
        help='FLUX model type'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to run on'
    )
    parser.add_argument(
        '--offload',
        action='store_true',
        help='Whether to offload the model to CPU to save VRAM'
    )
    parser.add_argument(
        '--num_steps',
        type=int,
        default=28,
        help='Number of inference steps'
    )
    parser.add_argument(
        '--guidance',
        type=float,
        default=4.0,  # aligned with training validation sampling
        help='Guidance scale (CFG)'
    )
    parser.add_argument(
        '--true_gs',
        type=float,
        default=1.0,  # aligned with training validation sampling
        help='True guidance scale'
    )
    parser.add_argument(
        '--timestep_to_start_cfg',
        type=int,
        default=0,  # aligned with training
        help='Timestep at which to start applying CFG'
    )
    parser.add_argument(
        '--control_weight',
        type=float,
        default=1.0,  # aligned with training (no scaling during training)
        help='ControlNet strength'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=512,  # must match training resolution
        help='Image size'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=2485,
        help='Random seed'
    )
    
    # Output parameters
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/root/autodl-tmp/HumanFlow/generation_demo',
        help='Output directory'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='Number of samples to infer; defaults to all'
    )
    parser.add_argument(
        '--start_idx',
        type=int,
        default=0,
        help='Index of the first sample to infer'
    )
    parser.add_argument(
        '--save_control_image',
        action='store_true',
        help='Whether to also save control images'
    )
    
    # SDXL Refiner parameters
    parser.add_argument(
        '--use_refiner',
        action='store_true',
        default=True,
        help='Whether to use SDXL Refiner post-processing (enabled by default)'
    )
    parser.add_argument(
        '--no_refiner',
        action='store_false',
        dest='use_refiner',
        help='Disable SDXL Refiner post-processing'
    )
    parser.add_argument(
        '--refiner_path',
        type=str,
        default='/root/autodl-tmp/stable-diffusion-xl-refiner-1.0',
        help='Path to SDXL Refiner model'
    )
    parser.add_argument(
        '--refiner_strength',
        type=float,
        default=0.3,
        help='Refiner strength in [0, 1]; higher values apply stronger edits (default 0.3)'
    )
    parser.add_argument(
        '--refiner_steps',
        type=int,
        default=20,
        help='Number of Refiner inference steps (default 20)'
    )
    
    return parser


def process_single_control_type(args, control_type, checkpoint_path):
    """Run inference for a single control type."""
    print(f"\n{'='*60}")
    print(f"Processing control type: {control_type}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"{'='*60}\n")
    
    # Resolve jsonl path
    if args.jsonl_path is None:
        jsonl_path = os.path.join(args.test_data_root, 'img_text_pair.jsonl')
    else:
        jsonl_path = args.jsonl_path
    
    # Load sample list
    print(f"Loading sample list: {jsonl_path}")
    samples = load_jsonl(jsonl_path)
    print(f"Loaded {len(samples)} samples in total")
    
    # Select inference range
    end_idx = len(samples)
    if args.num_samples is not None:
        end_idx = min(args.start_idx + args.num_samples, len(samples))
    samples_to_process = samples[args.start_idx:end_idx]
    print(f"Will infer {len(samples_to_process)} samples (index: {args.start_idx} to {end_idx-1})")
    
    # Create output directory named by control condition
    output_dir = os.path.join(args.output_dir, control_type)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Initialize pipeline
    print(f"\nInitializing {args.model_type} pipeline...")
    humanflow_pipeline = HumanFlowPipeline(args.model_type, args.device, args.offload)
    
    # Load ControlNet checkpoint
    print(f"Loading ControlNet checkpoint: {checkpoint_path}")
    
    # Inspect checkpoint file format
    if os.path.isdir(checkpoint_path):
        # If a directory is provided, locate the weight file (prefer controlnet.bin)
        possible_files = [
            os.path.join(checkpoint_path, 'controlnet.bin'),  # ControlNet weights saved during training
            os.path.join(checkpoint_path, 'controlnet.safetensors'),
            os.path.join(checkpoint_path, 'pytorch_model.bin'),
            os.path.join(checkpoint_path, 'model.safetensors'),
            os.path.join(checkpoint_path, 'diffusion_pytorch_model.safetensors'),
        ]
        checkpoint_file = None
        for f in possible_files:
            if os.path.exists(f):
                checkpoint_file = f
                break
        if checkpoint_file is None:
            raise FileNotFoundError(f"No model file found in directory {checkpoint_path}")
        print(f"Found checkpoint file: {checkpoint_file}")
    else:
        checkpoint_file = checkpoint_path
    
    # Configure ControlNet (uses the matching condition-aware encoder automatically)
    humanflow_pipeline.set_controlnet(
        control_type, 
        local_path=checkpoint_file,
        repo_id=None,
        name=None
    )
    print(f"ControlNet loaded successfully")
    
    # Initialize SDXL Refiner when enabled
    refiner = None
    if args.use_refiner:
        print(f"\nLoading SDXL Refiner: {args.refiner_path}")
        try:
            refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                args.refiner_path,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True
            )
            refiner.to(args.device)
            refiner.enable_model_cpu_offload() if args.offload else None
            print(f"SDXL Refiner loaded successfully")
            print(f"   - Strength: {args.refiner_strength}")
            print(f"   - Steps: {args.refiner_steps}")
        except Exception as e:
            print(f"Warning: failed to load SDXL Refiner: {e}")
            print(f"   Skipping refiner post-processing")
            refiner = None
            args.use_refiner = False
    
    print(f"\nStarting inference...")
    print(f"Control type: {control_type}")
    print(f"Inference steps: {args.num_steps}")
    print(f"Guidance scale: {args.guidance}")
    print(f"Control strength: {args.control_weight}")
    print(f"Image size: {args.img_size}x{args.img_size}")
    print(f"Random seed: {args.seed}")
    if args.use_refiner and refiner is not None:
        print(f"Refiner enabled: only refined images will be saved")
    else:
        print(f"Refiner disabled: original generated images will be saved")
    print()
    
    # Save run configuration
    config_info = {
        'control_type': control_type,
        'checkpoint_path': checkpoint_path,
        'model_type': args.model_type,
        'num_steps': args.num_steps,
        'guidance': args.guidance,
        'control_weight': args.control_weight,
        'img_size': args.img_size,
        'seed': args.seed,
    }
    config_save_path = os.path.join(output_dir, 'config.json')
    with open(config_save_path, 'w', encoding='utf-8') as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)
    print(f"Configuration saved to: {config_save_path}\n")
    
    # Inference loop
    success_count = 0
    error_count = 0
    total_inference_time = 0
    total_refiner_time = 0
    
    # Reset GPU memory statistics
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    for idx, sample in enumerate(tqdm(samples_to_process, desc="Inference progress")):
        try:
            img_name = sample['img_path']
            prompt = sample['prompt']
            actual_idx = args.start_idx + idx
            
            # Build control image path
            control_img_path = os.path.join(
                args.test_data_root, 
                'control_signals', 
                control_type, 
                img_name
            )
            
            # Check whether the control image exists
            if not os.path.exists(control_img_path):
                print(f"\nWarning: control image does not exist: {control_img_path}")
                error_count += 1
                continue
            
            # Load control image
            try:
                control_image = Image.open(control_img_path).convert('RGB')
                original_size = control_image.size
                
                # Resize image with high-quality resampling
                if control_image.size != (args.img_size, args.img_size):
                    control_image = control_image.resize(
                        (args.img_size, args.img_size), 
                        Image.Resampling.LANCZOS
                    )
            except Exception as e:
                print(f"\nError: failed to load control image {control_img_path}: {e}")
                error_count += 1
                continue
            
            # Run inference
            inference_start = time.time()
            result = humanflow_pipeline(
                prompt=prompt,
                controlnet_image=control_image,
                width=args.img_size,
                height=args.img_size,
                guidance=args.guidance,
                num_steps=args.num_steps,
                seed=247,  
                true_gs=args.true_gs,  # pass true_gs to match training validation
                control_weight=args.control_weight,
                neg_prompt=prompt,  # aligned with training validation: same prompt as negative
                timestep_to_start_cfg=args.timestep_to_start_cfg,  # pass CFG start timestep
            )
            inference_time = time.time() - inference_start
            total_inference_time += inference_time
            
            # Post-process with SDXL Refiner and save
            if args.use_refiner and refiner is not None:
                try:
                    refiner_start = time.time()
                    
                    # Refiner requires image dimensions to be multiples of 8
                    refine_width = (args.img_size // 8) * 8
                    refine_height = (args.img_size // 8) * 8
                    
                    # Resize first when dimensions do not match
                    if result.size != (refine_width, refine_height):
                        result_for_refine = result.resize(
                            (refine_width, refine_height), 
                            Image.Resampling.LANCZOS
                        )
                    else:
                        result_for_refine = result
                    
                    # Refine the generated image
                    refined_result = refiner(
                        prompt=prompt,
                        image=result_for_refine,
                        strength=args.refiner_strength,
                        num_inference_steps=args.refiner_steps,
                        generator=torch.Generator(device=args.device).manual_seed(247)
                    ).images[0]
                    
                    refiner_time = time.time() - refiner_start
                    total_refiner_time += refiner_time
                    
                    # Resize back when the original size differs
                    if refined_result.size != result.size:
                        refined_result = refined_result.resize(
                            result.size, 
                            Image.Resampling.LANCZOS
                        )
                    
                    # Save only the refined result using the original filename
                    result_path = os.path.join(output_dir, img_name)
                    refined_result.save(result_path, quality=95, optimize=True)
                    
                except Exception as e:
                    print(f"\nWarning: refiner processing failed ({img_name}): {e}")
                    # Fall back to the original result when refiner fails
                    result_path = os.path.join(output_dir, img_name)
                    result.save(result_path, quality=95, optimize=True)
            else:
                # Save the original result when refiner is disabled
                result_path = os.path.join(output_dir, img_name)
                result.save(result_path, quality=95, optimize=True)
            
            # Optionally save the control image with the original filename
            if args.save_control_image:
                control_save_path = os.path.join(output_dir, img_name.replace('.png', '_control.png'))
                control_image.save(control_save_path, quality=95, optimize=True)
            
            success_count += 1
            
            # Clear GPU cache every 10 images
            if (idx + 1) % 10 == 0:
                clear_gpu_cache()
            
        except Exception as e:
            print(f"\nError: failed to process sample {img_name}: {str(e)}")
            error_count += 1
            continue
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"Inference complete! [{control_type}]")
    print(f"Success: {success_count} samples")
    print(f"Failed: {error_count} samples")
    print(f"Results saved to: {output_dir}")
    
    # Performance statistics
    if success_count > 0:
        avg_inference_time = total_inference_time / success_count
        print(f"\nPerformance statistics:")
        print(f"  Average inference time: {avg_inference_time:.2f}s/image")
        print(f"  Total inference time: {total_inference_time:.2f}s")
        
        if args.use_refiner and refiner is not None and total_refiner_time > 0:
            avg_refiner_time = total_refiner_time / success_count
            print(f"  Average refiner time: {avg_refiner_time:.2f}s/image")
            print(f"  Total refiner time: {total_refiner_time:.2f}s")
            print(f"  Average total time: {(total_inference_time + total_refiner_time) / success_count:.2f}s/image")
        
        # GPU memory statistics
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  Peak GPU memory usage: {peak_memory:.2f}GB")
    
    print(f"{'='*60}\n")
    
    # Final GPU cache cleanup
    clear_gpu_cache()
    
    return success_count, error_count


def main(args):
    """Main entry point; supports batch processing across control conditions."""
    
    # Default checkpoint paths for all control conditions
    control_configs = {
        'canny': '/root/autodl-tmp/HumanFlow/ckpts-train/saves_micogen_canny/checkpoint-15000',
        'seg': '/root/autodl-tmp/HumanFlow/ckpts-train/saves_micogen_seg/checkpoint-15000',
        'normal': '/root/autodl-tmp/HumanFlow/ckpts-train/saves_micogen_normal/checkpoint-15000',
        'keypoint': '/root/autodl-tmp/HumanFlow/ckpts-train/saves_micogen_keypoint/checkpoint-7500',
        'depth': '/root/autodl-tmp/HumanFlow/ckpts-train/saves_micogen_depth/checkpoint-30000',
        'densepose': '/root/autodl-tmp/HumanFlow/ckpts-train/saves_micogen_densepose/checkpoint-35000',
    }
    
    # Single-control mode when both control_type and checkpoint_path are provided
    if args.control_type is not None and args.checkpoint_path is not None:
        print("Single control condition inference mode")
        process_single_control_type(args, args.control_type, args.checkpoint_path)
    # Batch mode: process all control conditions sequentially
    elif args.batch_inference:
        print("Batch inference mode: processing all control conditions sequentially\n")
        
        total_stats = {}
        for control_type, checkpoint_path in control_configs.items():
            try:
                success_count, error_count = process_single_control_type(
                    args, control_type, checkpoint_path
                )
                total_stats[control_type] = {
                    'success': success_count,
                    'error': error_count
                }
            except Exception as e:
                print(f"\nError while processing {control_type}: {e}")
                total_stats[control_type] = {
                    'success': 0,
                    'error': 0,
                    'exception': str(e)
                }
                continue
        
        # Print overall statistics
        print(f"\n{'='*60}")
        print(f"All inference complete! Summary:")
        print(f"{'='*60}")
        for control_type, stats in total_stats.items():
            if 'exception' in stats:
                print(f"{control_type:12} - error: {stats['exception']}")
            else:
                print(f"{control_type:12} - success: {stats['success']:3d}, failed: {stats['error']:3d}")
        print(f"{'='*60}")
    else:
        print("Error: specify --control_type and --checkpoint_path, or enable --batch_inference")
        return


if __name__ == '__main__':
    args = create_argparser().parse_args()
    main(args)

