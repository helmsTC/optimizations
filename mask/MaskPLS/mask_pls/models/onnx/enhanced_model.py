# mask/MaskPLS/mask_pls/models/onnx/enhanced_model_fixed.py
"""
Fixed Enhanced MaskPLS ONNX model with correct channel dimensions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class MultiScaleEncoder(nn.Module):
    """Enhanced encoder with multi-scale feature extraction"""
    def __init__(self, cfg):
        super().__init__()
        
        input_dim = cfg.INPUT_DIM
        # Use the actual channel configuration from backbone
        cs = cfg.CHANNELS  # [32, 32, 64, 128, 256, 256, 128, 96, 96]
        
        # For simplified architecture, use fewer stages
        # We'll use: 32 -> 64 -> 128 -> 256 -> 512
        self.channels = [32, 64, 128, 256, 512]
        
        # Initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv3d(input_dim, self.channels[0], kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(self.channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.channels[0], self.channels[0], kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(self.channels[0]),
            nn.ReLU(inplace=True),
        )
        
        # Encoder stages with residual connections
        self.encoder_stages = nn.ModuleList()
        for i in range(len(self.channels) - 1):
            self.encoder_stages.append(
                EncoderStage(self.channels[i], self.channels[i+1], stride=2)
            )
        
        # Decoder stages with skip connections
        self.decoder_stages = nn.ModuleList()
        # Decoder goes from deep to shallow: 512 -> 256 -> 128
        decoder_configs = [
            (512, 256, 256),  # (in_channels, out_channels, skip_channels)
            (256, 128, 128),
            (128, 64, 64),
        ]
        
        for in_ch, out_ch, skip_ch in decoder_configs:
            self.decoder_stages.append(
                DecoderStage(in_ch, out_ch, skip_ch)
            )
        
        # Store output channels for point decoder
        self.output_channels = [256, 128, 64]
        
    def forward(self, x):
        # Encoder with skip connections
        encoder_features = []
        
        x = self.stem(x)
        encoder_features.append(x)
        
        for stage in self.encoder_stages:
            x = stage(x)
            encoder_features.append(x)
        
        # Decoder with skip connections
        decoder_features = []
        
        # Start from the deepest features
        x = encoder_features[-1]  # 512 channels
        
        # Upsample and merge with skip connections
        for i, stage in enumerate(self.decoder_stages):
            # Get the corresponding encoder feature for skip connection
            # encoder_features[-2] = 256 channels (for first decoder stage)
            # encoder_features[-3] = 128 channels (for second decoder stage)
            # encoder_features[-4] = 64 channels (for third decoder stage)
            skip_idx = -(i+2)
            if abs(skip_idx) <= len(encoder_features):
                skip = encoder_features[skip_idx]
            else:
                # If we don't have enough encoder features, use zero skip
                skip = torch.zeros_like(x)
            x = stage(x, skip)
            decoder_features.append(x)
        
        # Return multi-scale features for the transformer decoder
        return decoder_features


class EncoderStage(nn.Module):
    """Encoder stage with residual connection"""
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, 
                              kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, 
                              kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        
        # Residual connection
        self.downsample = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 
                     kernel_size=1, stride=stride, bias=False),
            nn.InstanceNorm3d(out_channels)
        )
        
    def forward(self, x):
        identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out, inplace=True)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        out += identity
        out = F.relu(out, inplace=True)
        
        return out


class DecoderStage(nn.Module):
    """Decoder stage with skip connection - FIXED"""
    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()
        
        # First upsample to out_channels
        self.upsample = nn.ConvTranspose3d(
            in_channels, out_channels,
            kernel_size=2, stride=2, bias=False
        )
        self.norm1 = nn.InstanceNorm3d(out_channels)
        
        # Skip connection processing
        # Make sure skip has same channels as upsampled features
        if skip_channels != out_channels:
            self.skip_conv = nn.Conv3d(skip_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.skip_conv = nn.Identity()
        
        # Process concatenated features
        # After concatenation we have out_channels + out_channels = 2*out_channels
        self.conv = nn.Conv3d(out_channels * 2, out_channels,
                             kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        
    def forward(self, x, skip):
        # Upsample
        x = self.upsample(x)
        x = self.norm1(x)
        x = F.relu(x, inplace=True)
        
        # Process skip connection to match channels
        skip = self.skip_conv(skip)
        
        # Handle size mismatch by cropping or padding
        if x.shape != skip.shape:
            # Ensure spatial dimensions match
            _, _, D_x, H_x, W_x = x.shape
            _, _, D_s, H_s, W_s = skip.shape
            
            if D_s != D_x or H_s != H_x or W_s != W_x:
                # Crop or pad skip to match x
                # Use 'trilinear' for F.interpolate (different from grid_sample)
                skip = F.interpolate(skip, size=(D_x, H_x, W_x), mode='trilinear', align_corners=False)
        
        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Process combined features
        x = self.conv(x)
        x = self.norm2(x)
        x = F.relu(x, inplace=True)
        
        return x


class EnhancedPointDecoder(nn.Module):
    """Point decoder with multi-scale feature fusion and trilinear interpolation - FIXED"""
    def __init__(self, feat_dims, hidden_dim):
        super().__init__()
        
        self.feat_dims = feat_dims
        self.hidden_dim = hidden_dim
        
        # Feature projections for each scale
        self.scale_projs = nn.ModuleList([
            nn.Conv3d(dim, hidden_dim, kernel_size=1, bias=False)
            for dim in feat_dims
        ])
        
        # Feature fusion - fixed input dimension calculation
        total_features = hidden_dim * len(feat_dims)
        self.fusion = nn.Sequential(
            nn.Linear(total_features, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, multi_scale_features, point_coords):
        B, N, _ = point_coords.shape
        
        all_features = []
        
        for features, proj in zip(multi_scale_features, self.scale_projs):
            # Project features to hidden_dim
            features = proj(features)
            
            # Trilinear interpolation for smooth feature sampling
            features_sampled = self.trilinear_sample(features, point_coords)
            all_features.append(features_sampled)
        
        # Concatenate and fuse
        concat_features = torch.cat(all_features, dim=-1)
        fused_features = self.fusion(concat_features)
        
        return fused_features
    
    def trilinear_sample(self, features, coords):
        """
        Trilinear interpolation for feature sampling
        Args:
            features: [B, C, D, H, W]
            coords: [B, N, 3] normalized to [0, 1]
        Returns:
            sampled: [B, N, C]
        """
        B, C, D, H, W = features.shape
        B_p, N, _ = coords.shape
        
        # Convert normalized coords to grid coordinates [-1, 1]
        grid_coords = coords * 2.0 - 1.0
        
        # Reshape for grid_sample: [B, N, 1, 1, 3]
        grid_coords = grid_coords.view(B, N, 1, 1, 3)
        
        # Swap coordinates for grid_sample (expects z, y, x order)
        grid_coords = torch.stack([
            grid_coords[..., 2],  # z -> x
            grid_coords[..., 1],  # y -> y  
            grid_coords[..., 0]   # x -> z
        ], dim=-1)
        
        # Sample features: [B, C, N, 1, 1]
        # For 5D tensors (3D spatial), use 'bilinear' mode (which does trilinear for 3D)
        sampled = F.grid_sample(features, grid_coords, 
                               mode='bilinear',  # 'bilinear' for 3D data
                               padding_mode='border',
                               align_corners=True)
        
        # Reshape to [B, N, C]
        sampled = sampled.squeeze(-1).squeeze(-1).permute(0, 2, 1)
        
        return sampled


class ImprovedTransformerDecoder(nn.Module):
    """Transformer decoder with better attention mechanisms"""
    def __init__(self, cfg, hidden_dim, num_classes):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_queries = cfg.NUM_QUERIES
        self.num_heads = cfg.NHEADS
        self.num_layers = cfg.DEC_BLOCKS
        
        # Learnable query embeddings
        self.query_embed = nn.Parameter(torch.randn(1, self.num_queries, hidden_dim))
        self.query_pos = nn.Parameter(torch.randn(1, self.num_queries, hidden_dim))
        
        # Initialize properly
        nn.init.normal_(self.query_embed, std=0.02)
        nn.init.normal_(self.query_pos, std=0.02)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerDecoderLayer(
                hidden_dim, 
                self.num_heads,
                dim_feedforward=cfg.DIM_FFN,
                dropout=0.1
            )
            for _ in range(self.num_layers)
        ])
        
        # Output heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        
        # Mask embedding with MLP
        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Normalization
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, memory, memory_padding_mask=None):
        B = memory.shape[0]
        
        # Initialize queries
        queries = self.query_embed.expand(B, -1, -1)
        query_pos = self.query_pos.expand(B, -1, -1)
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            queries = layer(
                queries, 
                memory,
                query_pos=query_pos,
                memory_key_padding_mask=memory_padding_mask
            )
        
        # Normalize output
        queries = self.decoder_norm(queries)
        
        # Generate class predictions
        outputs_class = self.class_embed(queries)
        
        # Generate mask embeddings
        mask_embed = self.mask_embed(queries)
        
        # Compute mask predictions through dot product
        outputs_mask = torch.einsum('bqc,bnc->bnq', mask_embed, memory)
        
        return outputs_class, outputs_mask


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        # Self attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, 
                                               dropout=dropout, 
                                               batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross attention
        self.cross_attn = nn.MultiheadAttention(d_model, nhead,
                                                dropout=dropout,
                                                batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(self, tgt, memory, query_pos=None, memory_key_padding_mask=None):
        # Self-attention with pre-norm
        tgt_norm = self.norm1(tgt)
        q = k = self._with_pos_embed(tgt_norm, query_pos)
        tgt2 = self.self_attn(q, k, tgt_norm)[0]
        tgt = tgt + self.dropout1(tgt2)
        
        # Cross-attention with pre-norm
        tgt_norm = self.norm2(tgt)
        q = self._with_pos_embed(tgt_norm, query_pos)
        tgt2 = self.cross_attn(
            q, memory, memory,
            key_padding_mask=memory_key_padding_mask
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        
        # FFN with pre-norm
        tgt_norm = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt_norm))))
        tgt = tgt + self.dropout3(tgt2)
        
        return tgt
    
    def _with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

# In enhanced_model.py, add this efficient interpolation
class EfficientPointInterpolation(nn.Module):
    """
    Efficient point feature interpolation from voxel features
    Mimics the original KNN interpolation
    """
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        
    def forward(self, voxel_features, voxel_coords, point_coords):
        """
        Efficiently interpolate features from voxels to points
        Uses trilinear interpolation instead of KNN for ONNX compatibility
        """
        B, C, D, H, W = voxel_features.shape
        B_p, N, _ = point_coords.shape
        
        # Use grid_sample for efficient interpolation
        # Convert point coords to grid coordinates [-1, 1]
        grid_coords = point_coords * 2.0 - 1.0
        grid_coords = grid_coords.view(B, N, 1, 1, 3)
        
        # Reorder for grid_sample (expects z, y, x)
        grid_coords = torch.stack([
            grid_coords[..., 2],  # z
            grid_coords[..., 1],  # y  
            grid_coords[..., 0]   # x
        ], dim=-1)
        
        # Sample features
        sampled = F.grid_sample(
            voxel_features, 
            grid_coords,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        # Reshape to [B, N, C]
        sampled = sampled.squeeze(-1).squeeze(-1).permute(0, 2, 1)
        
        return sampled
        
class EnhancedMaskPLSONNX(nn.Module):
    """Complete enhanced MaskPLS model for ONNX export - FIXED"""
    def __init__(self, cfg):
        super().__init__()
        
        dataset = cfg.MODEL.DATASET
        self.num_classes = cfg[dataset].NUM_CLASSES
        
        # Multi-scale encoder with fixed channels
        self.encoder = MultiScaleEncoder(cfg.BACKBONE)
        
        # Point decoder with multi-scale fusion
        # Use the actual output channels from encoder
        feat_dims = self.encoder.output_channels  # [256, 128, 64]
        self.point_decoder = EnhancedPointDecoder(
            feat_dims,
            cfg.DECODER.HIDDEN_DIM
        )
        
        # Transformer decoder
        self.mask_decoder = ImprovedTransformerDecoder(
            cfg.DECODER,
            cfg.DECODER.HIDDEN_DIM,
            self.num_classes
        )
        
        # Semantic segmentation head
        self.semantic_head = nn.Sequential(
            nn.Linear(cfg.DECODER.HIDDEN_DIM, cfg.DECODER.HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.DECODER.HIDDEN_DIM, self.num_classes)
        )
        
    def forward(self, voxel_features, point_coords):
        """
        Args:
            voxel_features: [B, C, D, H, W] voxelized point cloud
            point_coords: [B, N, 3] normalized point coordinates [0, 1]
        Returns:
            pred_logits: [B, Q, num_classes+1] class predictions
            pred_masks: [B, N, Q] mask predictions
            sem_logits: [B, N, num_classes] semantic predictions
        """
        # Extract multi-scale features
        multi_scale_features = self.encoder(voxel_features)
        
        # Decode point features with multi-scale fusion
        point_features = self.point_decoder(multi_scale_features, point_coords)
        
        # Generate mask predictions through transformer
        pred_logits, pred_masks = self.mask_decoder(point_features)
        
        # Semantic segmentation
        sem_logits = self.semantic_head(point_features)
        
        return pred_logits, pred_masks, sem_logits