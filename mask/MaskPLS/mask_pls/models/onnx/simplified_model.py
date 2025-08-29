"""
Fixed MaskPLS ONNX model that actually works
This version simplifies the architecture for ONNX compatibility
Location: mask/MaskPLS/mask_pls/models/onnx/simplified_model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math


class SimpleVoxelEncoder(nn.Module):
    """
    Simplified voxel-based encoder that's ONNX-friendly
    Pre-computes voxel grid outside the model
    """
    def __init__(self, cfg):
        super().__init__()
        
        input_dim = cfg.INPUT_DIM
        cs = cfg.CHANNELS
        
        # Simple 3D CNN backbone - no dynamic operations
        self.encoder = nn.Sequential(
            # Initial processing
            nn.Conv3d(input_dim, cs[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(cs[0]),
            nn.ReLU(inplace=True),
            nn.Conv3d(cs[0], cs[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(cs[0]),
            nn.ReLU(inplace=True),
            
            # Downsample 1
            nn.Conv3d(cs[0], cs[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(cs[1]),
            nn.ReLU(inplace=True),
            
            # Downsample 2
            nn.Conv3d(cs[1], cs[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(cs[2]),
            nn.ReLU(inplace=True),
            
            # Downsample 3
            nn.Conv3d(cs[2], cs[3], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(cs[3]),
            nn.ReLU(inplace=True),
            
            # Downsample 4
            nn.Conv3d(cs[3], cs[4], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(cs[4]),
            nn.ReLU(inplace=True),
        )
        
        # Feature dimension after encoding
        self.feat_dim = cs[4]
        
    def forward(self, voxel_features):
        """
        Args:
            voxel_features: [B, C, D, H, W] voxel grid
        Returns:
            encoded_features: [B, C', D', H', W'] encoded voxels
        """
        return self.encoder(voxel_features)


class SimplePointDecoderV11(nn.Module):
    """
    Decoder compatible with ONNX (no grid_sample)
    Uses nearest neighbor sampling which is fully supported
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.feature_proj = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.mlp = nn.Sequential(
            nn.Linear(out_channels, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_channels)
        )
        
    def forward(self, voxel_features, point_coords):
        B, C, D, H, W = voxel_features.shape
        B_p, N, _ = point_coords.shape
        
        # Project features
        voxel_features = self.feature_proj(voxel_features)
        C_out = voxel_features.shape[1]
        
        # Flatten voxel features
        voxel_flat = voxel_features.view(B, C_out, -1)
        
        # Convert coordinates to indices
        voxel_coords = point_coords * torch.tensor([D, H, W], device=point_coords.device, dtype=point_coords.dtype)
        
        # Clamp each dimension separately for ONNX compatibility
        voxel_coords[..., 0] = torch.clamp(voxel_coords[..., 0], 0, D-1)
        voxel_coords[..., 1] = torch.clamp(voxel_coords[..., 1], 0, H-1)
        voxel_coords[..., 2] = torch.clamp(voxel_coords[..., 2], 0, W-1)
        
        voxel_coords = voxel_coords.long()
        
        # Flat indices
        flat_indices = (voxel_coords[..., 0] * H * W + 
                       voxel_coords[..., 1] * W + 
                       voxel_coords[..., 2])
        flat_indices = flat_indices.unsqueeze(1).expand(-1, C_out, -1)
        
        # Gather features
        point_features = torch.gather(voxel_flat, 2, flat_indices)
        point_features = point_features.transpose(1, 2)
        
        # MLP
        point_features = self.mlp(point_features)
        return point_features


class SimplifiedMaskDecoder(nn.Module):
    """
    Simplified mask decoder without dynamic operations
    """
    def __init__(self, cfg, feat_dim, num_classes):
        super().__init__()
        
        hidden_dim = cfg.HIDDEN_DIM
        self.num_queries = cfg.NUM_QUERIES
        self.num_classes = num_classes
        
        # Fixed learned queries
        self.query_embed = nn.Parameter(torch.randn(1, self.num_queries, hidden_dim))
        
        # Simple transformer-like attention (ONNX-friendly)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, 
            num_heads=8, 
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, hidden_dim)
        )
        
        # Output heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Project input features
        self.input_proj = nn.Linear(feat_dim, hidden_dim)
        
    def forward(self, point_features):
        """
        Args:
            point_features: [B, N, C] point features
        Returns:
            pred_logits: [B, Q, num_classes+1]
            pred_masks: [B, N, Q]
        """
        B, N, C = point_features.shape
        
        # Project features
        point_features = self.input_proj(point_features)  # [B, N, hidden_dim]
        
        # Expand queries for batch
        queries = self.query_embed.expand(B, -1, -1)  # [B, Q, hidden_dim]
        
        # Cross attention (simplified - single layer)
        queries = self.norm1(queries)
        attn_out, _ = self.cross_attn(
            queries, 
            point_features, 
            point_features
        )
        queries = queries + attn_out
        
        # FFN
        queries = self.norm2(queries)
        queries = queries + self.ffn(queries)
        
        # Generate outputs
        pred_logits = self.class_embed(queries)  # [B, Q, num_classes+1]
        
        # Generate masks
        mask_embed = self.mask_embed(queries)  # [B, Q, hidden_dim]
        # Simple dot product attention for masks
        pred_masks = torch.einsum('bqc,bnc->bnq', mask_embed, point_features)
        
        return pred_logits, pred_masks


class MaskPLSSimplifiedONNX(nn.Module):
    """
    Complete simplified MaskPLS model for ONNX export
    This version pre-voxelizes the data and uses simpler operations
    """
    def __init__(self, cfg):
        super().__init__()
        
        # Configuration
        self.cfg = cfg
        dataset = cfg.MODEL.DATASET
        self.data_cfg = cfg[dataset]
        self.num_classes = self.data_cfg.NUM_CLASSES
        
        # Spatial configuration for voxelization
        self.spatial_shape = (32, 32, 16)  # Reduced size for efficiency
        
        # Register coordinate bounds as a buffer so it gets exported to ONNX
        coord_bounds = self.data_cfg.SPACE
        # Convert to tensor and register as buffer
        self.register_buffer('coordinate_bounds', 
                           torch.tensor(coord_bounds, dtype=torch.float32))
        
        # Encoder
        self.encoder = SimpleVoxelEncoder(cfg.BACKBONE)
        
        # Point decoder - ONLY use the ONNX-compatible version
        self.point_decoder = SimplePointDecoderV11(
            self.encoder.feat_dim,
            cfg.DECODER.HIDDEN_DIM
        )
        
        # Mask decoder
        self.mask_decoder = SimplifiedMaskDecoder(
            cfg.DECODER,
            cfg.DECODER.HIDDEN_DIM,
            self.num_classes
        )
        
        # Semantic head (optional)
        self.sem_head = nn.Linear(cfg.DECODER.HIDDEN_DIM, self.num_classes)
        
    def voxelize_points(self, points, features):
        """
        Pre-voxelization step (should be done outside model for ONNX)
        This is included for completeness but won't be traced
        """
        B = points.shape[0]
        D, H, W = self.spatial_shape
        C = features.shape[2]
        
        voxel_grids = torch.zeros(B, C, D, H, W, device=points.device)
        
        for b in range(B):
            # Normalize coordinates to [0, 1]
            norm_coords = torch.zeros_like(points[b])
            for i in range(3):
                min_val = self.coordinate_bounds[i, 0]
                max_val = self.coordinate_bounds[i, 1]
                norm_coords[:, i] = (points[b, :, i] - min_val) / (max_val - min_val)
            
            # Clip to valid range
            norm_coords = torch.clamp(norm_coords, 0, 0.999)
            
            # Convert to voxel indices
            voxel_indices = (norm_coords * torch.tensor([D, H, W], device=points.device)).long()
            
            # Accumulate features (simple average)
            for idx in range(points.shape[1]):
                d, h, w = voxel_indices[idx]
                voxel_grids[b, :, d, h, w] += features[b, idx]
        
        return voxel_grids
    
    def get_coordinate_bounds(self):
        """
        Get the coordinate bounds as a tensor
        This will be preserved in ONNX export
        
        Returns:
            coordinate_bounds: [3, 2] tensor with [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
        """
        return self.coordinate_bounds
    
    def forward(self, voxel_features, point_coords):
        """
        Forward pass for ONNX export
        
        Args:
            voxel_features: [B, C, D, H, W] pre-voxelized features
            point_coords: [B, N, 3] normalized point coordinates in [0, 1]
        
        Returns:
            pred_logits: [B, Q, num_classes+1]
            pred_masks: [B, N, Q]
            sem_logits: [B, N, num_classes]
        """
        # Encode voxel features
        encoded_voxels = self.encoder(voxel_features)
        
        # Decode to point features
        point_features = self.point_decoder(encoded_voxels, point_coords)
        
        # Generate masks and classes
        pred_logits, pred_masks = self.mask_decoder(point_features)
        
        # Semantic segmentation (optional)
        sem_logits = self.sem_head(point_features)
        
        return pred_logits, pred_masks, sem_logits


def create_onnx_model(cfg):
    """Create the simplified ONNX model"""
    return MaskPLSSimplifiedONNX(cfg)


def export_model_to_onnx(model, output_path, batch_size=1, num_points=10000):
    """
    Export the model to ONNX format
    """
    # Set to eval mode
    model.eval()
    
    # Create dummy inputs
    D, H, W = model.spatial_shape
    C = 4  # XYZI features
    
    # Pre-voxelized features
    dummy_voxels = torch.randn(batch_size, C, D, H, W)
    # Normalized point coordinates
    dummy_coords = torch.rand(batch_size, num_points, 3)
    
    # Export
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_voxels, dummy_coords),
            output_path,
            export_params=True,
            opset_version=16,  # Use 16 or higher
            do_constant_folding=True,
            input_names=['voxel_features', 'point_coords'],
            output_names=['pred_logits', 'pred_masks', 'sem_logits'],
            dynamic_axes={
                'voxel_features': {0: 'batch_size'},
                'point_coords': {0: 'batch_size', 1: 'num_points'},
                'pred_logits': {0: 'batch_size'},
                'pred_masks': {0: 'batch_size', 1: 'num_points'},
                'sem_logits': {0: 'batch_size', 1: 'num_points'}
            }
        )
    
    print(f"Model exported to {output_path}")
    return True


# Example usage
if __name__ == "__main__":
    from easydict import EasyDict as edict
    
    # Create config
    cfg = edict({
        'MODEL': {
            'DATASET': 'KITTI',
            'OVERLAP_THRESHOLD': 0.8
        },
        'KITTI': {
            'NUM_CLASSES': 20,
            'MIN_POINTS': 10,
            'SPACE': [[-48.0, 48.0], [-48.0, 48.0], [-4.0, 1.5]],
        },
        'BACKBONE': {
            'INPUT_DIM': 4,
            'CHANNELS': [32, 32, 64, 128, 256],
        },
        'DECODER': {
            'HIDDEN_DIM': 256,
            'NUM_QUERIES': 100,
        }
    })
    
    # Create and export model
    model = create_onnx_model(cfg)
    export_model_to_onnx(model, "maskpls_simplified.onnx")
