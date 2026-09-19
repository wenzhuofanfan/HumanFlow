#!/usr/bin/env python3
"""
Compute improved HTCL multi-templates from MiCoGen training dataset.
This should be run once before training with HTCL.

Improvements over original:
- Clusters poses into multiple templates (not single average)
- Uses skeleton-based features (not full connectivity matrix)
- Scale-normalized bone ratios (not absolute distances)
"""
import sys
sys.path.insert(0, '.')

import torch
from image_datasets.micogen_dataset import MiCoGenDataset
from htcl import PoseExtractor, compute_multi_templates

def main():
    print("="*60)
    print("Computing Improved HTCL Multi-Templates from MiCoGen Dataset")
    print("="*60)
    
    # Configuration
    data_root = "/root/autodl-tmp/HumanFlow/datasets/MiCoGen"
    save_path = "/root/autodl-tmp/HumanFlow/ckpts/htcl_multi_templates.pt"
    max_samples = 10000  # Use 10k samples for templates (adjust as needed)
    batch_size = 32
    num_clusters = 5  # Number of pose families to cluster
    conf_threshold = 0.3  # Minimum confidence threshold
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nConfiguration:")
    print(f"  Data root: {data_root}")
    print(f"  Save path: {save_path}")
    print(f"  Max samples: {max_samples}")
    print(f"  Batch size: {batch_size}")
    print(f"  Num clusters: {num_clusters}")
    print(f"  Conf threshold: {conf_threshold}")
    print(f"  Device: {device}")
    print()
    
    # Create dataset
    print("Loading dataset...")
    dataset = MiCoGenDataset(
        data_root=data_root,
        control_type='canny',  # Doesn't matter which control type
        img_size=512,
        subset=max_samples
    )
    print(f"✓ Dataset loaded with {len(dataset)} samples\n")
    
    # Initialize pose extractor
    print("Initializing pose extractor...")
    pose_extractor = PoseExtractor(
        model_type='hrnet',
        pretrained=True,
        device=device
    )
    print("✓ Pose extractor initialized\n")
    
    # Compute multi-templates
    print("Computing multi-templates with clustering...")
    print("This may take several minutes...\n")
    
    templates = compute_multi_templates(
        dataset=dataset,
        pose_extractor=pose_extractor,
        device=device,
        max_samples=max_samples,
        batch_size=batch_size,
        num_clusters=num_clusters,
        conf_threshold=conf_threshold,
        save_path=save_path
    )
    
    print(f"\n{'='*60}")
    print("✓ Multi-templates computed and saved successfully!")
    print(f"  Number of clusters: {templates['num_clusters']}")
    print(f"  Bone ratio templates shape: {templates['bone_ratio_templates'].shape}")
    print(f"  Angle templates shape: {templates['angle_templates'].shape}")
    print(f"  Location: {save_path}")
    print(f"{'='*60}\n")
    print("You can now train with improved HTCL enabled.")
    print(f"\nUsage in training:")
    print(f"  htcl_wrapper = HTCLWrapper(")
    print(f"      template_path='{save_path}',")
    print(f"      device='cuda'")
    print(f"  )")


if __name__ == "__main__":
    main()

