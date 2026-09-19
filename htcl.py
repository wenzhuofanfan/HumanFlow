"""
Human Topology Consistency Loss (HTCL) Implementation - IMPROVED VERSION
for noise-free human image generation in ControlFlow.

Key improvements over original:
1. Skeleton-based topology (not fully connected)
2. Scale-normalized bone length ratios (not absolute distances)
3. Multi-template clustering (not single average)
4. Confidence filtering (not weighting)
5. Angular constraints for directionality
6. Left-right symmetry enforcement

This module provides:
1. PoseExtractor: Frozen pretrained pose estimation model
2. Skeleton topology computation
3. Multi-template computation from training data
4. HTCL loss function with ratio + angle constraints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict
from sklearn.cluster import KMeans
import os


# COCO 17 keypoints definition (MUST match pose estimator output order)
# Index: Name
# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
# 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
# 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
# 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

# COCO skeleton definition (anatomical connections)
# Format: (parent_idx, child_idx)
COCO_SKELETON = [
    # Head
    (0, 1), (0, 2),  # nose -> eyes
    (1, 3), (2, 4),  # eyes -> ears
    # Torso
    (0, 5), (0, 6),  # nose -> shoulders
    (5, 6),  # shoulder connection
    (5, 11), (6, 12),  # shoulders -> hips
    (11, 12),  # hip connection
    # Left arm
    (5, 7), (7, 9),  # shoulder -> elbow -> wrist
    # Right arm
    (6, 8), (8, 10),  # shoulder -> elbow -> wrist
    # Left leg
    (11, 13), (13, 15),  # hip -> knee -> ankle
    # Right leg
    (12, 14), (14, 16),  # hip -> knee -> ankle
]

def verify_coco_keypoint_order(keypoints_names):
    """
    Verify that the pose estimator outputs COCO keypoints in the expected order.
    
    Args:
        keypoints_names: list of keypoint names from pose estimator
    
    Raises:
        ValueError if order doesn't match COCO standard
    """
    expected_order = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    if len(keypoints_names) != 17:
        raise ValueError(f"Expected 17 keypoints, got {len(keypoints_names)}")
    
    for idx, (expected, actual) in enumerate(zip(expected_order, keypoints_names)):
        if expected.lower() != actual.lower():
            raise ValueError(
                f"Keypoint order mismatch at index {idx}: "
                f"expected '{expected}', got '{actual}'"
            )

# Core bones for torso size normalization
TORSO_BONES = [
    (5, 6),   # shoulders
    (11, 12), # hips
    (5, 11),  # left shoulder-hip
    (6, 12),  # right shoulder-hip
]

# Symmetric bone pairs (left, right)
SYMMETRIC_PAIRS = [
    # Arms
    ((5, 7), (6, 8)),   # upper arms
    ((7, 9), (8, 10)),  # forearms
    # Legs
    ((11, 13), (12, 14)),  # thighs
    ((13, 15), (14, 16)),  # calves
]

# Joint angle triplets (joint1, vertex, joint2) for angle constraints
ANGLE_TRIPLETS = [
    # Arms
    (5, 7, 9),   # left arm angle
    (6, 8, 10),  # right arm angle
    # Legs
    (11, 13, 15),  # left leg angle
    (12, 14, 16),  # right leg angle
    # Torso
    (5, 0, 6),   # shoulder-nose-shoulder
    (11, 5, 7),  # hip-shoulder-elbow (left)
    (12, 6, 8),  # hip-shoulder-elbow (right)
]


class PoseExtractor(nn.Module):
    """
    Frozen pretrained pose estimation model.
    Uses HRNet or ViTPose to extract human keypoints.
    
    Input: RGB image tensor (B, 3, H, W) in range [-1, 1]
    Output: 
        - keypoints: (B, 17, 2) in pixel coordinates
        - confidences: (B, 17) in range [0, 1]
    """
    
    def __init__(self, model_type='hrnet', pretrained=True, device='cuda'):
        """
        Args:
            model_type: 'hrnet' or 'vitpose'
            pretrained: whether to load pretrained weights
            device: device to load model on
        """
        super().__init__()
        self.model_type = model_type
        self.device = device
        self.num_joints = 17  # COCO keypoints format
        
        # Load pretrained pose estimation model
        if model_type == 'hrnet':
            self._init_hrnet(pretrained)
        elif model_type == 'vitpose':
            self._init_vitpose(pretrained)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        
        # Move model to specified device
        self.to(device)
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        
        self.eval()
    
    def _init_hrnet(self, pretrained):
        """Initialize HRNet model"""
        try:
            # Try to use MMPose's HRNet
            from mmpose.apis import init_model
            config_file = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py'
            checkpoint_file = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
            
            if pretrained:
                self.model = init_model(config_file, checkpoint_file, device=self.device)
            else:
                self.model = init_model(config_file, None, device=self.device)
            
            self.use_mmpose = True
            print("PoseExtractor: Using MMPose HRNet")
            
        except ImportError:
            # Fallback: Use a simple HRNet implementation
            print("PoseExtractor: MMPose not available, using fallback HRNet")
            self.use_mmpose = False
            self._init_fallback_hrnet(pretrained)
    
    def _init_vitpose(self, pretrained):
        """Initialize ViTPose model"""
        try:
            from mmpose.apis import init_model
            config_file = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitpose_base_coco_256x192.py'
            checkpoint_file = 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/vitpose_base_coco_256x192-216eae50_20230314.pth'
            
            if pretrained:
                self.model = init_model(config_file, checkpoint_file, device=self.device)
            else:
                self.model = init_model(config_file, None, device=self.device)
            
            self.use_mmpose = True
            print("PoseExtractor: Using MMPose ViTPose")
            
        except ImportError:
            print("PoseExtractor: MMPose not available, using fallback model")
            self.use_mmpose = False
            self._init_fallback_hrnet(pretrained)
    
    def _init_fallback_hrnet(self, pretrained):
        """Fallback HRNet using timm or torchvision"""
        # Simple fallback: Use a pretrained ResNet backbone + heatmap head
        import torchvision.models as models
        
        # Use ResNet50 as backbone
        backbone = models.resnet50(pretrained=pretrained)
        
        # Remove final layers
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # Add heatmap prediction head
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.num_joints, kernel_size=1)
        )
        
        self.model = nn.Sequential(self.backbone, self.heatmap_head)
        print("PoseExtractor: Using fallback ResNet50-based pose estimator")
    
    def preprocess(self, images):
        """
        Preprocess images for pose estimation.
        Input: (B, 3, H, W) in range [-1, 1]
        Output: (B, 3, H, W) in range [0, 1] or normalized
        """
        # Convert from [-1, 1] to [0, 1]
        images = (images + 1.0) / 2.0
        
        # Normalize to ImageNet stats if needed
        # mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        # std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        # images = (images - mean) / std
        
        return images
    
    def extract_keypoints_from_heatmap(self, heatmaps, use_soft_argmax=True):
        """
        Extract keypoint coordinates from heatmaps using soft-argmax (differentiable).
        
        Args:
            heatmaps: (B, num_joints, H, W)
            use_soft_argmax: if True, use differentiable soft-argmax; else use hard argmax
        
        Returns:
            keypoints: (B, num_joints, 2) in pixel coordinates
            confidences: (B, num_joints) in range [0, 1]
        """
        B, num_joints, H, W = heatmaps.shape
        device = heatmaps.device
        
        if use_soft_argmax:
            # Soft-argmax: compute expected coordinates (differentiable)
            # This allows gradients to flow back to the image
            
            # Apply softmax to get probability distribution
            heatmaps_flat = heatmaps.view(B, num_joints, -1)
            probs = F.softmax(heatmaps_flat * 10.0, dim=2)  # temperature=0.1 for sharper peaks
            probs = probs.view(B, num_joints, H, W)
            
            # Create coordinate grids
            y_grid = torch.arange(H, device=device).float().view(1, 1, H, 1)
            x_grid = torch.arange(W, device=device).float().view(1, 1, 1, W)
            
            # Compute expected coordinates (weighted average)
            y_coords = (probs * y_grid).sum(dim=(2, 3))  # (B, num_joints)
            x_coords = (probs * x_grid).sum(dim=(2, 3))  # (B, num_joints)
            
            # Stack to get keypoints (B, num_joints, 2)
            keypoints = torch.stack([x_coords, y_coords], dim=2)
            
            # Confidence: max of heatmap (before softmax)
            confidences = heatmaps.view(B, num_joints, -1).max(dim=2)[0]
            confidences = torch.sigmoid(confidences)
            
        else:
            # Hard argmax (non-differentiable, for debugging/comparison only)
            heatmaps_flat = heatmaps.view(B, num_joints, -1)
            confidences, max_indices = torch.max(heatmaps_flat, dim=2)
            
            y_coords = (max_indices // W).float()
            x_coords = (max_indices % W).float()
            
            keypoints = torch.stack([x_coords, y_coords], dim=2)
            confidences = torch.sigmoid(confidences)
        
        return keypoints, confidences
    
    def forward(self, images):
        """
        Extract keypoints from images.
        
        Note: Although this model is frozen (requires_grad=False),
        we don't use @torch.no_grad() to allow gradients to flow
        through for loss computation.
        
        Args:
            images: (B, 3, H, W) in range [-1, 1]
        
        Returns:
            keypoints: (B, 17, 2) in pixel coordinates
            confidences: (B, 17) in range [0, 1]
        """
        # Ensure model is in eval mode and frozen
        self.eval()
        
        B, C, H, W = images.shape
        
        # Preprocess images
        images_processed = self.preprocess(images)
        
        if self.use_mmpose:
            # MMPose models require complex inference pipeline (person detection, bbox, etc.)
            # For now, we don't support direct MMPose integration in differentiable mode
            # Fall back to simple heatmap-based model
            raise NotImplementedError(
                "MMPose integration is not yet supported for differentiable HTCL. "
                "Please use fallback model (will auto-use if mmpose import fails)."
            )
        
        # Fallback: Direct inference with simple heatmap model
        heatmaps = self.model(images_processed)
        
        # Upsample heatmaps to match input resolution if needed
        if heatmaps.shape[2] != H or heatmaps.shape[3] != W:
            heatmaps = F.interpolate(heatmaps, size=(H, W), mode='bilinear', align_corners=False)
        
        # Extract keypoints and confidences from heatmaps (using soft-argmax for differentiability)
        keypoints, confidences = self.extract_keypoints_from_heatmap(heatmaps, use_soft_argmax=True)
        
        return keypoints, confidences


def compute_bone_length(keypoints, bone_idx):
    """
    Compute length of a bone (edge in skeleton).
    
    Args:
        keypoints: (B, 17, 2) or (17, 2)
        bone_idx: tuple (parent_idx, child_idx)
    
    Returns:
        length: (B,) or scalar
    """
    parent_idx, child_idx = bone_idx
    diff = keypoints[..., child_idx, :] - keypoints[..., parent_idx, :]
    length = torch.norm(diff, p=2, dim=-1)
    return length


def compute_torso_size(keypoints, confidences=None, conf_threshold=0.3, min_size=10.0):
    """
    Compute torso size for normalization with robust handling.
    Uses masked average of torso bone lengths (only valid bones).
    
    Args:
        keypoints: (B, 17, 2) or (17, 2)
        confidences: (B, 17) or (17,), optional. If provided, filter by conf_threshold
        conf_threshold: minimum confidence to consider bone valid
        min_size: minimum torso size (clamp to avoid division by near-zero)
    
    Returns:
        torso_size: (B,) or scalar, clamped to [min_size, inf)
    """
    is_batched = keypoints.ndim == 3
    
    if confidences is not None:
        # Compute valid torso bones (both endpoints must be confident)
        valid_lengths = []
        valid_masks = []
        
        for bone in TORSO_BONES:
            parent_idx, child_idx = bone
            
            if is_batched:
                parent_valid = confidences[:, parent_idx] > conf_threshold
                child_valid = confidences[:, child_idx] > conf_threshold
                bone_valid = parent_valid & child_valid  # (B,)
            else:
                parent_valid = confidences[parent_idx] > conf_threshold
                child_valid = confidences[child_idx] > conf_threshold
                bone_valid = parent_valid & child_valid  # scalar
            
            length = compute_bone_length(keypoints, bone)
            valid_lengths.append(length)
            valid_masks.append(bone_valid.float() if isinstance(bone_valid, torch.Tensor) else float(bone_valid))
        
        valid_lengths = torch.stack(valid_lengths, dim=-1)  # (B, 4) or (4,)
        valid_masks = torch.stack(valid_masks, dim=-1)      # (B, 4) or (4,)
        
        # Masked mean: only average over valid bones
        sum_lengths = (valid_lengths * valid_masks).sum(dim=-1)
        count_valid = valid_masks.sum(dim=-1)
        
        # If no valid bones, use mean of all (fallback)
        torso_size = torch.where(
            count_valid > 0,
            sum_lengths / (count_valid + 1e-6),
            valid_lengths.mean(dim=-1)
        )
    else:
        # No confidence filtering, use simple mean
        torso_lengths = []
        for bone in TORSO_BONES:
            length = compute_bone_length(keypoints, bone)
            torso_lengths.append(length)
        
        torso_size = torch.stack(torso_lengths, dim=-1).mean(dim=-1)
    
    # Clamp to avoid division by near-zero
    torso_size = torch.clamp(torso_size, min=min_size)
    
    return torso_size


def compute_angle(keypoints, triplet, eps=1e-6):
    """
    Compute angle at vertex formed by three joints.
    
    Args:
        keypoints: (B, 17, 2) or (17, 2)
        triplet: (idx1, vertex_idx, idx2)
        eps: small value for numerical stability
    
    Returns:
        angle: (B,) or scalar, in radians [0, π]
    """
    idx1, vertex_idx, idx2 = triplet
    
    # Get vectors
    v1 = keypoints[..., idx1, :] - keypoints[..., vertex_idx, :]
    v2 = keypoints[..., idx2, :] - keypoints[..., vertex_idx, :]
    
    # Compute angle using dot product
    dot = (v1 * v2).sum(dim=-1)
    norm1 = torch.norm(v1, p=2, dim=-1)
    norm2 = torch.norm(v2, p=2, dim=-1)
    
    cos_angle = dot / (norm1 * norm2 + eps)
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    angle = torch.acos(cos_angle)
    
    return angle


def build_skeleton_features(keypoints, confidences, conf_threshold=0.3):
    """
    Build skeleton-based topology features (replaces full connectivity matrix).
    
    Features:
    1. Normalized bone length ratios (relative to torso)
    2. Joint angles
    3. Valid masks based on confidence
    
    Args:
        keypoints: (B, 17, 2) keypoint coordinates in pixels
        confidences: (B, 17) keypoint confidences in [0, 1]
        conf_threshold: minimum confidence to consider keypoint valid
    
    Returns:
        features: dict with keys
            - 'bone_ratios': (B, num_bones) normalized bone lengths
            - 'angles': (B, num_angles) joint angles
            - 'bone_masks': (B, num_bones) validity mask for bones
            - 'angle_masks': (B, num_angles) validity mask for angles
    """
    B = keypoints.shape[0]
    device = keypoints.device
    
    # Compute torso size for normalization (with robust masking)
    torso_size = compute_torso_size(keypoints, confidences, conf_threshold)  # (B,)
    
    # Compute bone length ratios
    bone_ratios = []
    bone_masks = []
    
    for bone in COCO_SKELETON:
        parent_idx, child_idx = bone
        
        # Check if both endpoints are valid
        parent_valid = confidences[:, parent_idx] > conf_threshold
        child_valid = confidences[:, child_idx] > conf_threshold
        bone_valid = parent_valid & child_valid  # (B,)
        
        # Compute bone length
        bone_length = compute_bone_length(keypoints, bone)  # (B,)
        
        # Normalize by torso size
        bone_ratio = bone_length / (torso_size + 1e-6)
        
        bone_ratios.append(bone_ratio)
        bone_masks.append(bone_valid.float())
    
    bone_ratios = torch.stack(bone_ratios, dim=1)  # (B, num_bones)
    bone_masks = torch.stack(bone_masks, dim=1)    # (B, num_bones)
    
    # Compute joint angles
    angles = []
    angle_masks = []
    
    for triplet in ANGLE_TRIPLETS:
        idx1, vertex_idx, idx2 = triplet
        
        # Check if all three joints are valid
        valid1 = confidences[:, idx1] > conf_threshold
        valid2 = confidences[:, vertex_idx] > conf_threshold
        valid3 = confidences[:, idx2] > conf_threshold
        angle_valid = valid1 & valid2 & valid3  # (B,)
        
        # Compute angle
        angle = compute_angle(keypoints, triplet)  # (B,)
        
        angles.append(angle)
        angle_masks.append(angle_valid.float())
    
    angles = torch.stack(angles, dim=1)           # (B, num_angles)
    angle_masks = torch.stack(angle_masks, dim=1)  # (B, num_angles)
    
    features = {
        'bone_ratios': bone_ratios,
        'angles': angles,
        'bone_masks': bone_masks,
        'angle_masks': angle_masks,
    }
    
    return features


def compute_multi_templates(dataset, pose_extractor, device='cuda', 
                           max_samples=None, batch_size=32,
                           num_clusters=5, conf_threshold=0.3,
                           save_path=None):
    """
    Compute multiple template clusters from training dataset.
    
    Instead of averaging all poses (creating a meaningless "mean pose"),
    we cluster poses into common pose families (standing, sitting, action, etc.)
    and compute a template for each cluster.
    
    Args:
        dataset: training dataset
        pose_extractor: PoseExtractor instance
        device: device to run computation on
        max_samples: maximum number of samples to process (None = all)
        batch_size: batch size for processing
        num_clusters: number of pose clusters (default: 5)
        conf_threshold: minimum confidence threshold
        save_path: path to save the templates (optional)
    
    Returns:
        templates: dict with keys
            - 'bone_ratio_templates': (num_clusters, num_bones)
            - 'angle_templates': (num_clusters, num_angles)
            - 'cluster_weights': (num_clusters,) frequency of each cluster
    """
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    
    print(f"Computing {num_clusters} pose templates from training data...")
    
    pose_extractor.eval()
    pose_extractor.to(device)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        drop_last=False
    )
    
    # Accumulate features
    all_bone_ratios = []
    all_angles = []
    all_bone_masks = []
    all_angle_masks = []
    num_processed = 0
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Extracting features")):
            # Extract images from batch
            # Handle different dataset formats
            if len(batch_data) == 3:
                # MiCoGenDataset format: (img, control, prompt)
                images = batch_data[0]
            elif len(batch_data) == 6:
                # Format: [z, token_embedding, token_mask, token, caption, img]
                images = batch_data[5]
            elif len(batch_data) == 7:
                # Format: [z, token_embedding, token_mask, token, caption, img, control]
                images = batch_data[5]
            else:
                print(f"Warning: Unexpected batch format with {len(batch_data)} items")
                continue
            
            images = images.to(device)
            B, C, H, W = images.shape
            
            # Extract keypoints
            keypoints, confidences = pose_extractor(images)
            
            # Build skeleton features
            features = build_skeleton_features(keypoints, confidences, conf_threshold)
            
            # Store features
            all_bone_ratios.append(features['bone_ratios'].cpu())
            all_angles.append(features['angles'].cpu())
            all_bone_masks.append(features['bone_masks'].cpu())
            all_angle_masks.append(features['angle_masks'].cpu())
            
            num_processed += B
            
            if max_samples is not None and num_processed >= max_samples:
                break
    
    # Concatenate all features
    all_bone_ratios = torch.cat(all_bone_ratios, dim=0)  # (N, num_bones)
    all_angles = torch.cat(all_angles, dim=0)            # (N, num_angles)
    all_bone_masks = torch.cat(all_bone_masks, dim=0)    # (N, num_bones)
    all_angle_masks = torch.cat(all_angle_masks, dim=0)  # (N, num_angles)
    
    N = all_bone_ratios.shape[0]
    print(f"Extracted features from {N} images")
    
    # Filter out samples with too many invalid keypoints
    bone_valid_ratio = all_bone_masks.mean(dim=1)  # (N,)
    angle_valid_ratio = all_angle_masks.mean(dim=1)  # (N,)
    overall_valid = (bone_valid_ratio > 0.5) & (angle_valid_ratio > 0.5)
    
    valid_bone_ratios = all_bone_ratios[overall_valid]
    valid_angles = all_angles[overall_valid]
    valid_bone_masks = all_bone_masks[overall_valid]
    valid_angle_masks = all_angle_masks[overall_valid]
    
    N_valid = valid_bone_ratios.shape[0]
    print(f"Filtered to {N_valid} valid samples ({N_valid/N*100:.1f}%)")
    
    if N_valid < num_clusters:
        print(f"Warning: Not enough valid samples ({N_valid}) for {num_clusters} clusters")
        num_clusters = max(1, N_valid // 10)
        print(f"Reducing to {num_clusters} clusters")
    
    # Concatenate features for clustering
    # First normalize angles to similar range as bone ratios
    normalized_angles = valid_angles / np.pi
    clustering_features_raw = torch.cat([valid_bone_ratios, normalized_angles], dim=1)
    
    # Z-score standardization: (x - mean) / std
    # This ensures all dimensions contribute equally to clustering
    feature_mean = clustering_features_raw.mean(dim=0, keepdim=True)
    feature_std = clustering_features_raw.std(dim=0, keepdim=True)
    clustering_features_standardized = (clustering_features_raw - feature_mean) / (feature_std + 1e-6)
    
    # Convert to numpy for sklearn
    clustering_features = clustering_features_standardized.numpy()
    
    # Perform K-means clustering on standardized features
    print(f"Clustering into {num_clusters} pose families (with z-score standardization)...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(clustering_features)
    
    # Compute template for each cluster
    bone_ratio_templates = []
    angle_templates = []
    cluster_weights = []
    
    for k in range(num_clusters):
        cluster_mask = cluster_labels == k
        cluster_size = cluster_mask.sum()
        
        if cluster_size == 0:
            continue
        
        # Get samples in this cluster
        cluster_bone_ratios = valid_bone_ratios[cluster_mask]
        cluster_angles = valid_angles[cluster_mask]
        cluster_bone_masks = valid_bone_masks[cluster_mask]
        cluster_angle_masks = valid_angle_masks[cluster_mask]
        
        # Compute masked mean (only average over valid entries)
        bone_ratio_sum = (cluster_bone_ratios * cluster_bone_masks).sum(dim=0)
        bone_mask_sum = cluster_bone_masks.sum(dim=0)
        bone_ratio_template = bone_ratio_sum / (bone_mask_sum + 1e-6)
        
        angle_sum = (cluster_angles * cluster_angle_masks).sum(dim=0)
        angle_mask_sum = cluster_angle_masks.sum(dim=0)
        angle_template = angle_sum / (angle_mask_sum + 1e-6)
        
        bone_ratio_templates.append(bone_ratio_template)
        angle_templates.append(angle_template)
        cluster_weights.append(cluster_size / N_valid)
        
        print(f"  Cluster {k}: {cluster_size} samples ({cluster_size/N_valid*100:.1f}%)")
    
    bone_ratio_templates = torch.stack(bone_ratio_templates)  # (num_clusters, num_bones)
    angle_templates = torch.stack(angle_templates)            # (num_clusters, num_angles)
    cluster_weights = torch.tensor(cluster_weights)           # (num_clusters,)
    
    templates = {
        'bone_ratio_templates': bone_ratio_templates,
        'angle_templates': angle_templates,
        'cluster_weights': cluster_weights,
        'num_clusters': len(bone_ratio_templates),
    }
    
    print(f"\nComputed {len(bone_ratio_templates)} templates")
    print(f"  Bone ratios shape: {bone_ratio_templates.shape}")
    print(f"  Angles shape: {angle_templates.shape}")
    
    # Save templates if path provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(templates, save_path)
        print(f"\nSaved templates to {save_path}")
    
    return templates


def compute_symmetry_loss(features, relaxed=True):
    """
    Compute left-right symmetry loss with relaxed constraints.
    Symmetric body parts should have similar bone lengths (but allow for perspective).
    
    Args:
        features: dict with 'bone_ratios' and 'bone_masks'
        relaxed: if True, only apply to core symmetric pairs (arms/legs), 
                 and use softer loss to tolerate perspective distortion
    
    Returns:
        symmetry_loss: scalar
    """
    bone_ratios = features['bone_ratios']  # (B, num_bones)
    bone_masks = features['bone_masks']    # (B, num_bones)
    
    losses = []
    
    # Map bones to their indices in COCO_SKELETON
    bone_to_idx = {bone: idx for idx, bone in enumerate(COCO_SKELETON)}
    
    # Select which pairs to enforce based on relaxed mode
    if relaxed:
        # Only enforce symmetry on upper arms and thighs (most robust to perspective)
        symmetric_pairs_to_use = [
            ((5, 7), (6, 8)),   # upper arms
            ((11, 13), (12, 14)),  # thighs
        ]
    else:
        # Use all symmetric pairs
        symmetric_pairs_to_use = SYMMETRIC_PAIRS
    
    for left_bone, right_bone in symmetric_pairs_to_use:
        if left_bone in bone_to_idx and right_bone in bone_to_idx:
            left_idx = bone_to_idx[left_bone]
            right_idx = bone_to_idx[right_bone]
            
            left_ratio = bone_ratios[:, left_idx]
            right_ratio = bone_ratios[:, right_idx]
            left_valid = bone_masks[:, left_idx]
            right_valid = bone_masks[:, right_idx]
            
            # Only compute loss if both sides are valid
            both_valid = left_valid * right_valid
            
            # Relative difference (allows up to 2x difference before linear penalty)
            # This tolerates perspective distortion better than strict equality
            ratio = torch.maximum(left_ratio, right_ratio) / (torch.minimum(left_ratio, right_ratio) + 1e-6)
            diff = torch.relu(ratio - 1.5)  # Only penalize if ratio > 1.5 (50% difference)
            masked_diff = diff * both_valid
            
            losses.append(masked_diff)
    
    if len(losses) > 0:
        all_losses = torch.stack(losses, dim=1)  # (B, num_pairs)
        symmetry_loss = all_losses.mean()
    else:
        symmetry_loss = torch.tensor(0.0, device=bone_ratios.device)
    
    return symmetry_loss


class HumanTopologyConsistencyLoss(nn.Module):
    """
    Improved Human Topology Consistency Loss (HTCL).
    
    Uses skeleton-based features with multi-template matching:
    1. Bone length ratios (scale-normalized)
    2. Joint angles (directional constraints)
    3. Left-right symmetry
    """
    
    def __init__(self, templates, device='cuda', 
                 bone_loss_weight=1.0, angle_loss_weight=0.5, 
                 symmetry_loss_weight=0.3):
        """
        Args:
            templates: dict with multi-template data from compute_multi_templates()
                - 'bone_ratio_templates': (K, num_bones)
                - 'angle_templates': (K, num_angles)
                - 'cluster_weights': (K,)
            device: device to store tensors on
            bone_loss_weight: weight for bone ratio loss
            angle_loss_weight: weight for angle loss
            symmetry_loss_weight: weight for symmetry loss
        """
        super().__init__()
        
        # Register templates as buffers
        self.register_buffer('bone_ratio_templates', 
                           templates['bone_ratio_templates'].to(device))
        self.register_buffer('angle_templates', 
                           templates['angle_templates'].to(device))
        self.register_buffer('cluster_weights', 
                           templates['cluster_weights'].to(device))
        
        self.num_clusters = templates['num_clusters']
        self.bone_loss_weight = bone_loss_weight
        self.angle_loss_weight = angle_loss_weight
        self.symmetry_loss_weight = symmetry_loss_weight
        
        # Define bone importance weights (structural bones more important)
        self.register_buffer('bone_weights', self._get_bone_weights().to(device))
        self.register_buffer('angle_weights', self._get_angle_weights().to(device))
    
    def _get_bone_weights(self):
        """Get importance weights for each bone in skeleton."""
        weights = torch.ones(len(COCO_SKELETON))
        
        # Higher weights for torso bones
        for idx, bone in enumerate(COCO_SKELETON):
            if bone in TORSO_BONES:
                weights[idx] = 3.0
            # Medium weights for limb bones
            elif bone in [(5, 7), (6, 8), (11, 13), (12, 14),  # upper arm/thigh
                         (7, 9), (8, 10), (13, 15), (14, 16)]: # forearm/calf
                weights[idx] = 2.0
            # Lower weights for head connections
            elif bone[0] in [0, 1, 2, 3, 4] or bone[1] in [0, 1, 2, 3, 4]:
                weights[idx] = 0.5
        
        # Normalize
        weights = weights / weights.mean()
        return weights
    
    def _get_angle_weights(self):
        """Get importance weights for each angle."""
        weights = torch.ones(len(ANGLE_TRIPLETS))
        
        # Higher weights for major joint angles (elbows, knees)
        for idx, triplet in enumerate(ANGLE_TRIPLETS):
            if triplet in [(5, 7, 9), (6, 8, 10), (11, 13, 15), (12, 14, 16)]:
                weights[idx] = 2.0
        
        weights = weights / weights.mean()
        return weights
    
    def compute_distance_to_templates(self, features):
        """
        Compute distance from generated features to each template.
        
        Args:
            features: dict with 'bone_ratios', 'angles', 'bone_masks', 'angle_masks'
        
        Returns:
            distances: (B, K) distance to each of K templates
        """
        B = features['bone_ratios'].shape[0]
        K = self.num_clusters
        
        bone_ratios = features['bone_ratios']  # (B, num_bones)
        angles = features['angles']            # (B, num_angles)
        bone_masks = features['bone_masks']    # (B, num_bones)
        angle_masks = features['angle_masks']  # (B, num_angles)
        
        distances = []
        
        for k in range(K):
            # Bone ratio distance
            bone_template = self.bone_ratio_templates[k:k+1, :]  # (1, num_bones)
            bone_diff = (bone_ratios - bone_template) ** 2
            bone_diff_weighted = bone_diff * self.bone_weights * bone_masks
            bone_dist = bone_diff_weighted.sum(dim=1) / (bone_masks.sum(dim=1) + 1e-6)
            
            # Angle distance
            # Note: angles are in [0, π] (unsigned angles), no wrapping needed
            angle_template = self.angle_templates[k:k+1, :]  # (1, num_angles)
            angle_diff = torch.abs(angles - angle_template)
            angle_diff_squared = angle_diff ** 2
            angle_diff_weighted = angle_diff_squared * self.angle_weights * angle_masks
            angle_dist = angle_diff_weighted.sum(dim=1) / (angle_masks.sum(dim=1) + 1e-6)
            
            # Combined distance
            total_dist = (self.bone_loss_weight * bone_dist + 
                         self.angle_loss_weight * angle_dist)
            distances.append(total_dist)
        
        distances = torch.stack(distances, dim=1)  # (B, K)
        return distances
    
    def forward(self, features, reduction='mean', use_soft_min=True, temperature=0.1):
        """
        Compute HTCL loss using multi-template matching with soft minimum.
        
        Args:
            features: dict with skeleton features from build_skeleton_features()
            reduction: 'mean', 'sum', or 'none'
            use_soft_min: if True, use differentiable soft minimum; else use hard min
            temperature: temperature for soft minimum (lower = sharper, closer to hard min)
        
        Returns:
            loss: scalar loss or (B,) losses if reduction='none'
        """
        B = features['bone_ratios'].shape[0]
        
        # Compute distance to each template
        distances = self.compute_distance_to_templates(features)  # (B, K)
        
        if use_soft_min:
            # Soft minimum using log-sum-exp trick (differentiable)
            # softmin(x) = -τ * log(Σ exp(-x/τ))
            # This is smoother than hard min and avoids gradient discontinuities
            
            # Apply temperature scaling
            scaled_distances = distances / (temperature + 1e-8)
            
            # Compute soft minimum
            # Using logsumexp for numerical stability
            max_dist = scaled_distances.max(dim=1, keepdim=True)[0]
            exp_distances = torch.exp(-scaled_distances + max_dist)
            soft_min = -temperature * (torch.log(exp_distances.sum(dim=1) + 1e-8) - max_dist.squeeze(1))
            
            topology_loss = soft_min  # (B,)
        else:
            # Hard minimum (for comparison/debugging)
            topology_loss = distances.min(dim=1)[0]  # (B,)
        
        # Add symmetry loss (relaxed to tolerate perspective)
        symmetry_loss = compute_symmetry_loss(features, relaxed=True)
        
        # Combine losses
        total_loss = topology_loss + self.symmetry_loss_weight * symmetry_loss
        
        # Apply reduction
        if reduction == 'mean':
            return total_loss.mean()
        elif reduction == 'sum':
            return total_loss.sum()
        elif reduction == 'none':
            return total_loss
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")


class HTCLWrapper(nn.Module):
    """
    Improved wrapper module that combines PoseExtractor and HTCL loss.
    Uses skeleton-based features and multi-template matching.
    """
    
    def __init__(self, template_path, pose_model_type='hrnet', 
                 device='cuda', loss_weight=1.0, 
                 conf_threshold=0.3, conf_penalty_weight=1.0,
                 bone_loss_weight=1.0, angle_loss_weight=0.5, 
                 symmetry_loss_weight=0.3):
        """
        Args:
            template_path: path to saved multi-templates (from compute_multi_templates)
            pose_model_type: 'hrnet' or 'vitpose'
            device: device to run on
            loss_weight: overall weight for HTCL loss in total loss
            conf_threshold: minimum confidence to consider keypoint valid
            conf_penalty_weight: weight for confidence penalty
            bone_loss_weight: weight for bone ratio loss
            angle_loss_weight: weight for angle loss
            symmetry_loss_weight: weight for symmetry loss
        """
        super().__init__()
        
        # Load multi-templates
        if os.path.exists(template_path):
            templates = torch.load(template_path, map_location=device)
            print(f"Loaded {templates['num_clusters']} pose templates from {template_path}")
        else:
            print(f"Warning: Templates not found at {template_path}")
            print("Creating dummy templates. Please compute templates first!")
            # Create dummy templates as fallback
            num_bones = len(COCO_SKELETON)
            num_angles = len(ANGLE_TRIPLETS)
            templates = {
                'bone_ratio_templates': torch.ones(1, num_bones),
                'angle_templates': torch.ones(1, num_angles) * (np.pi / 2),
                'cluster_weights': torch.ones(1),
                'num_clusters': 1,
            }
        
        # Initialize pose extractor (frozen)
        self.pose_extractor = PoseExtractor(
            model_type=pose_model_type, 
            pretrained=True, 
            device=device
        )
        
        # Ensure pose extractor is in float32 and on correct device
        self.pose_extractor = self.pose_extractor.float().to(device)
        
        # Initialize improved HTCL loss
        self.htcl_loss = HumanTopologyConsistencyLoss(
            templates=templates,
            device=device,
            bone_loss_weight=bone_loss_weight,
            angle_loss_weight=angle_loss_weight,
            symmetry_loss_weight=symmetry_loss_weight
        )
        
        self.loss_weight = loss_weight
        self.conf_threshold = conf_threshold
        self.conf_penalty_weight = conf_penalty_weight
        self.device = device
        
        # Freeze pose extractor
        for param in self.pose_extractor.parameters():
            param.requires_grad = False
    
    def compute_loss(self, generated_images):
        """
        Compute improved HTCL loss for generated images.
        
        Args:
            generated_images: (B, 3, H, W) in range [-1, 1]
        
        Returns:
            loss: scalar HTCL loss
            info: dict with debugging information
        """
        B = generated_images.shape[0]
        
        # Extract keypoints (frozen model, gradients still flow for loss backprop)
        keypoints, confidences = self.pose_extractor(generated_images)
        
        # Build skeleton features (scale-normalized, with validity masks)
        features = build_skeleton_features(keypoints, confidences, self.conf_threshold)
        
        # Compute topology consistency loss
        topology_loss = self.htcl_loss(features)
        
        # Improved confidence penalty:
        # Instead of penalizing low confidence directly (which creates escape route),
        # we penalize when VALID keypoints are too few
        bone_valid_ratio = features['bone_masks'].mean()
        angle_valid_ratio = features['angle_masks'].mean()
        
        # High penalty when less than 50% keypoints are detected
        # This encourages the model to generate clear, detectable humans
        # but doesn't create escape route through low confidence
        detection_quality = (bone_valid_ratio + angle_valid_ratio) / 2
        confidence_penalty = torch.relu(0.5 - detection_quality) ** 2
        
        # Total loss
        total_loss = topology_loss + self.conf_penalty_weight * confidence_penalty
        
        # Debug info
        info = {
            'topology_loss': topology_loss.item(),
            'confidence_penalty': confidence_penalty.item(),
            'avg_confidence': confidences.mean().item(),
            'bone_valid_ratio': bone_valid_ratio.item(),
            'angle_valid_ratio': angle_valid_ratio.item(),
        }
        
        return total_loss * self.loss_weight, info
    
    def forward(self, generated_images):
        """
        Forward pass computes HTCL loss.
        
        Returns only loss (for backward compatibility).
        Use compute_loss() to get additional info.
        """
        loss, _ = self.compute_loss(generated_images)
        return loss

