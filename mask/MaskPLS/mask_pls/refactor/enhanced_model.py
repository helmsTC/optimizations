# mask/MaskPLS/mask_pls/models/onnx/enhanced_model.py
"""
Enhanced MaskPLS ONNX model with multi-scale features and better spatial resolution
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
        cs = cfg.CHANNELS  # [32, 64, 128, 256, 512]
        
        # Initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv3d(input_dim, cs[0], kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(cs[0]),  # More stable than BatchNorm for varying sizes
            nn.ReLU(inplace=True),
            nn.Conv3d(cs[0], cs[0], kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(cs[0]),
            nn.ReLU(inplace=True),
        )
        
        # Encoder stages with residual connections
        self.encoder_stages = nn.ModuleList()
        for i in range(len(cs) - 1):
            self.encoder_stages.append(
                EncoderStage(cs[i], cs[i+1], stride=2)
            )
        
        # Decoder stages with skip connections
        self.decoder_stages = nn.ModuleList()
        decoder_channels = [cs[-1], cs[-2], cs[-3], cs[-4]]
        for i in range(len(decoder_channels) - 1):
            in_ch = decoder_channels[i]
            out_ch = decoder_channels[i+1]
            skip_ch = cs[-(i+2)]  # Skip connection from encoder
            self.decoder_stages.append(
                DecoderStage(in_ch + skip_ch, out_ch)
            )
        
        self.output_channels = decoder_channels
        
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
        x = encoder_features[-1]
        decoder_features.append(x)
        
        # Upsample and merge with skip connections
        for i, stage in enumerate(self.decoder_stages):
            skip = encoder_features[-(i+2)]
            x = stage(x, skip)
            decoder_features.append(x)
        
        # Return multi-scale features for the transformer decoder
        # Use the last 3 decoder features
        return decoder_features[-3:]


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
    """Decoder stage with skip connection"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.upsample = nn.ConvTranspose3d(
            in_channels, out_channels,
            kernel_size=2, stride=2, bias=False
        )
        self.norm = nn.InstanceNorm3d(out_channels)
        
        self.conv = nn.Conv3d(out_channels, out_channels,
                             kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        
    def forward(self, x, skip):
        # Upsample
        x = self.upsample(x)
        x = self.norm(x)
        x = F.relu(x, inplace=True)
        
        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Process combined features
        x = self.conv(x)
        x = self.norm2(x)
        x = F.relu(x, inplace=True)
        
        return x


class EnhancedPointDecoder(nn.Module):
    """Point decoder with multi-scale feature fusion and trilinear interpolation"""
    def __init__(self, feat_dims, hidden_dim):
        super().__init__()
        
        self.feat_dims = feat_dims
        self.hidden_dim = hidden_dim
        
        # Feature projections for each scale
        self.scale_projs = nn.ModuleList([
            nn.Conv3d(dim, hidden_dim, kernel_size=1, bias=False)
            for dim in feat_dims
        ])
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * len(feat_dims), hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, multi_scale_features, point_coords):
        B, N, _ = point_coords.shape
        
        all_features = []
        
        for features, proj in zip(multi_scale_features, self.scale_projs):
            # Project features
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
        sampled = F.grid_sample(features, grid_coords, 
                               mode='trilinear', 
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
        # Self-attention
        q = k = self._with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention
        q = self._with_pos_embed(tgt, query_pos)
        tgt2 = self.cross_attn(
            q, memory, memory,
            key_padding_mask=memory_key_padding_mask
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # FFN
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt
    
    def _with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos


class EnhancedMaskPLSONNX(nn.Module):
    """Complete enhanced MaskPLS model for ONNX export"""
    def __init__(self, cfg):
        super().__init__()
        
        dataset = cfg.MODEL.DATASET
        self.num_classes = cfg[dataset].NUM_CLASSES
        
        # Better channel progression for features
        cfg.BACKBONE.CHANNELS = [32, 64, 128, 256, 512]
        
        # Multi-scale encoder
        self.encoder = MultiScaleEncoder(cfg.BACKBONE)
        
        # Point decoder with multi-scale fusion
        feat_dims = self.encoder.output_channels[-3:]  # Last 3 scales
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
