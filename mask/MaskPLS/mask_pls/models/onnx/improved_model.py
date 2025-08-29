"""
Improved MaskPLS ONNX model with residual connections and better decoder
This version maintains ONNX compatibility while improving gradient flow
Location: mask/MaskPLS/mask_pls/models/onnx/improved_model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math


class ResidualBlock3D(nn.Module):
    """3D Residual block for better gradient flow"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride),
                nn.BatchNorm3d(out_channels)
            )
        
    def forward(self, x):
        identity = self.skip(x)
        
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        
        out = out + identity
        out = F.relu(out, inplace=True)
        
        return out


class ImprovedVoxelEncoder(nn.Module):
    """
    Improved voxel encoder with residual connections
    Better gradient flow while maintaining ONNX compatibility
    """
    def __init__(self, cfg):
        super().__init__()
        
        input_dim = cfg.INPUT_DIM
        cs = cfg.CHANNELS  # [64, 128, 256, 256, 256]
        
        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv3d(input_dim, cs[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(cs[0]),
            nn.ReLU(inplace=True),
            nn.Conv3d(cs[0], cs[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(cs[0]),
            nn.ReLU(inplace=True),
        )
        
        # Residual stages with downsampling
        self.stage1 = nn.Sequential(
            ResidualBlock3D(cs[0], cs[1], kernel_size=3, stride=2),
            ResidualBlock3D(cs[1], cs[1], kernel_size=3)
        )
        
        self.stage2 = nn.Sequential(
            ResidualBlock3D(cs[1], cs[2], kernel_size=3, stride=2),
            ResidualBlock3D(cs[2], cs[2], kernel_size=3)
        )
        
        self.stage3 = nn.Sequential(
            ResidualBlock3D(cs[2], cs[3], kernel_size=3, stride=2),
            ResidualBlock3D(cs[3], cs[3], kernel_size=3)
        )
        
        self.stage4 = nn.Sequential(
            ResidualBlock3D(cs[3], cs[4], kernel_size=3, stride=2),
            ResidualBlock3D(cs[4], cs[4], kernel_size=3)
        )
        
        # Store intermediate features for multi-scale
        self.feat_dims = cs
        
    def forward(self, voxel_features):
        """
        Args:
            voxel_features: [B, C, D, H, W] voxel grid
        Returns:
            features: List of multi-scale features
        """
        x0 = self.stem(voxel_features)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        
        # Return multi-scale features for decoder
        return [x2, x3, x4]  # Use last 3 levels like original


class ImprovedPointDecoder(nn.Module):
    """
    Improved point decoder with multi-scale feature fusion
    """
    def __init__(self, feat_dims, out_channels):
        super().__init__()
        
        # Project each feature level
        self.feat_projs = nn.ModuleList([
            nn.Conv3d(dim, out_channels, kernel_size=1)
            for dim in feat_dims
        ])
        
        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(out_channels * len(feat_dims), 512),
            nn.ReLU(inplace=True),
            nn.LayerNorm(512),
            nn.Linear(512, out_channels)
        )
        
    def forward(self, multi_scale_features, point_coords):
        """
        Args:
            multi_scale_features: List of [B, C_i, D_i, H_i, W_i] features
            point_coords: [B, N, 3] normalized coordinates
        Returns:
            point_features: [B, N, C] features at point locations
        """
        B, N, _ = point_coords.shape
        
        all_features = []
        
        for features, proj in zip(multi_scale_features, self.feat_projs):
            # Project features
            features = proj(features)
            B_f, C, D, H, W = features.shape
            
            # Sample features at points (using nearest neighbor for ONNX)
            # Scale coordinates to feature resolution
            scale = torch.tensor([D, H, W], device=point_coords.device, dtype=point_coords.dtype)
            scaled_coords = point_coords * scale
            
            # Clamp coordinates
            scaled_coords[..., 0] = torch.clamp(scaled_coords[..., 0], 0, D-1)
            scaled_coords[..., 1] = torch.clamp(scaled_coords[..., 1], 0, H-1)
            scaled_coords[..., 2] = torch.clamp(scaled_coords[..., 2], 0, W-1)
            
            coords_int = scaled_coords.long()
            
            # Flatten and gather
            flat_features = features.view(B_f, C, -1)
            flat_indices = (coords_int[..., 0] * H * W + 
                          coords_int[..., 1] * W + 
                          coords_int[..., 2])
            flat_indices = flat_indices.unsqueeze(1).expand(-1, C, -1)
            
            sampled = torch.gather(flat_features, 2, flat_indices)
            sampled = sampled.transpose(1, 2)  # [B, N, C]
            
            all_features.append(sampled)
        
        # Concatenate and fuse
        concat_features = torch.cat(all_features, dim=-1)
        fused_features = self.fusion_mlp(concat_features)
        
        return fused_features


class ImprovedMaskDecoder(nn.Module):
    """
    Improved mask decoder with multi-layer attention
    """
    def __init__(self, cfg, feat_dim, num_classes):
        super().__init__()
        
        hidden_dim = cfg.HIDDEN_DIM
        self.num_queries = cfg.NUM_QUERIES
        self.num_classes = num_classes
        self.num_layers = 3  # Multiple decoder layers
        
        # Learned queries
        self.query_embed = nn.Parameter(torch.randn(1, self.num_queries, hidden_dim))
        self.query_pos = nn.Parameter(torch.randn(1, self.num_queries, hidden_dim))
        
        # Multi-layer transformer decoder
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(hidden_dim, num_heads=8)
            for _ in range(self.num_layers)
        ])
        
        # Normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Output heads
        self.class_embed = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes + 1)
            for _ in range(self.num_layers)
        ])
        
        self.mask_embed = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, hidden_dim, 3)
            for _ in range(self.num_layers)
        ])
        
        # Point feature projection
        self.point_proj = nn.Linear(feat_dim, hidden_dim)
        
    def forward(self, point_features, padding_mask=None):
        """
        Args:
            point_features: [B, N, C] point features
            padding_mask: [B, N] padding mask
        Returns:
            outputs: Dict with pred_logits and pred_masks
        """
        B, N, _ = point_features.shape
        
        # Project point features
        memory = self.point_proj(point_features)
        
        # Initialize queries
        queries = self.query_embed.expand(B, -1, -1)
        query_pos = self.query_pos.expand(B, -1, -1)
        
        # Store intermediate predictions for deep supervision
        predictions_class = []
        predictions_mask = []
        
        # Multi-layer decoding
        for i, layer in enumerate(self.layers):
            queries = layer(
                queries,
                memory,
                query_pos=query_pos,
                memory_key_padding_mask=padding_mask
            )
            
            # Intermediate predictions
            queries_norm = self.norm(queries)
            
            # Class prediction
            outputs_class = self.class_embed[i](queries_norm)
            predictions_class.append(outputs_class)
            
            # Mask prediction
            mask_embed = self.mask_embed[i](queries_norm)
            outputs_mask = torch.einsum("bqc,bnc->bnq", mask_embed, memory)
            predictions_mask.append(outputs_mask)
        
        # Use last layer predictions as main output
        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': [
                {'pred_logits': c, 'pred_masks': m}
                for c, m in zip(predictions_class[:-1], predictions_mask[:-1])
            ]
        }
        
        return out


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer with self and cross attention"""
    def __init__(self, d_model, num_heads=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        # Self attention
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Cross attention
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, tgt, memory, query_pos=None, memory_key_padding_mask=None):
        # Self attention
        q = k = tgt + query_pos if query_pos is not None else tgt
        tgt2, _ = self.self_attn(q, k, tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross attention
        q = tgt + query_pos if query_pos is not None else tgt
        tgt2, _ = self.cross_attn(q, memory, memory, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # FFN
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


class MLP(nn.Module):
    """Simple MLP for mask embedding"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


class ImprovedMaskPLSONNX(nn.Module):
    """
    Improved MaskPLS model with better architecture
    Maintains ONNX compatibility while improving performance
    """
    def __init__(self, cfg):
        super().__init__()
        
        # Get dataset config
        dataset = cfg.MODEL.DATASET
        num_classes = cfg[dataset].NUM_CLASSES
        
        # Encoder with residual connections
        self.encoder = ImprovedVoxelEncoder(cfg.BACKBONE)
        
        # Get feature dimensions from last 3 stages
        feat_dims = [cfg.BACKBONE.CHANNELS[i] for i in [2, 3, 4]]
        
        # Point decoder with multi-scale fusion
        decoder_dim = cfg.DECODER.HIDDEN_DIM
        self.point_decoder = ImprovedPointDecoder(feat_dims, decoder_dim)
        
        # Mask decoder with multi-layer attention
        self.mask_decoder = ImprovedMaskDecoder(cfg.DECODER, decoder_dim, num_classes)
        
        # Semantic head for auxiliary supervision
        self.sem_head = nn.Linear(decoder_dim, num_classes)
        
        # Store config
        self.num_classes = num_classes
        self.spatial_shape = None  # Set dynamically
        
    def forward(self, voxel_features, point_coords):
        """
        Args:
            voxel_features: [B, C, D, H, W] pre-voxelized features
            point_coords: [B, N, 3] normalized point coordinates
        Returns:
            pred_logits: [B, Q, num_classes+1] class predictions
            pred_masks: [B, N, Q] mask predictions
            sem_logits: [B, N, num_classes] semantic predictions
        """
        # Encode voxels with multi-scale features
        multi_scale_features = self.encoder(voxel_features)
        
        # Decode point features with multi-scale fusion
        point_features = self.point_decoder(multi_scale_features, point_coords)
        
        # Generate padding mask
        B, N, _ = point_coords.shape
        padding_mask = torch.zeros(B, N, dtype=torch.bool, device=point_coords.device)
        
        # Decode masks and classes
        outputs = self.mask_decoder(point_features, padding_mask)
        
        # Semantic predictions for auxiliary supervision
        sem_logits = self.sem_head(point_features)
        
        return outputs['pred_logits'], outputs['pred_masks'], sem_logits