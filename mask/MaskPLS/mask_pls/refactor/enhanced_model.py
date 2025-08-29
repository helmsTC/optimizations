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

