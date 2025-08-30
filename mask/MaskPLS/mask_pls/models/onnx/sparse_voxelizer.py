# mask/MaskPLS/mask_pls/models/onnx/sparse_voxelizer.py
"""
Sparse voxelizer that mimics MinkowskiEngine efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional


class SparseVoxelizer:
    """
    Efficient sparse voxelizer that only processes occupied voxels
    Similar to MinkowskiEngine but ONNX-compatible
    """
    def __init__(self, resolution=0.05, spatial_shape=(96, 96, 48), 
                 coordinate_bounds=[[-48.0, 48.0], [-48.0, 48.0], [-4.0, 1.5]], 
                 device='cuda'):
        self.resolution = resolution  # Match original: 0.05
        self.spatial_shape = spatial_shape  # Match original spatial extent
        self.coordinate_bounds = coordinate_bounds
        self.device = device
        
        # Precompute bounds
        self.bounds_min = torch.tensor(
            [coordinate_bounds[i][0] for i in range(3)], 
            device=device
        )
        self.bounds_max = torch.tensor(
            [coordinate_bounds[i][1] for i in range(3)],
            device=device
        )
        self.bounds_range = self.bounds_max - self.bounds_min
        
        # Precompute voxel grid dimensions based on resolution
        # Limit grid size to prevent memory explosion
        raw_grid_size = (self.bounds_range / resolution).long()
        max_grid_size = 512  # Reasonable limit for dense grids
        self.grid_size = torch.clamp(raw_grid_size, max=max_grid_size)
        
        print(f"Grid size limited to: {self.grid_size.tolist()} (raw would be: {raw_grid_size.tolist()})")
        
    def voxelize_batch(self, points_list, features_list, max_points=80000):
        """
        Efficient voxelization using sparse operations
        """
        B = len(points_list)
        
        # Process each point cloud
        sparse_voxels = []
        point_coords = []
        valid_indices = []
        
        for b in range(B):
            pts = torch.from_numpy(points_list[b]).to(self.device)
            feat = torch.from_numpy(features_list[b]).to(self.device)
            
            # Filter valid points
            valid_mask = ((pts >= self.bounds_min) & 
                         (pts < self.bounds_max)).all(dim=1)
            valid_idx = torch.where(valid_mask)[0]
            
            if valid_idx.numel() == 0:
                # Empty point cloud
                sparse_voxels.append({
                    'indices': torch.zeros((0, 3), device=self.device, dtype=torch.long),
                    'features': torch.zeros((0, feat.shape[1]), device=self.device)
                })
                point_coords.append(torch.zeros((0, 3), device=self.device))
                valid_indices.append(valid_idx)
                continue
            
            valid_pts = pts[valid_idx]
            valid_feat = feat[valid_idx]
            
            # Subsample if needed
            if valid_pts.shape[0] > max_points:
                perm = torch.randperm(valid_pts.shape[0])[:max_points]
                valid_pts = valid_pts[perm]
                valid_feat = valid_feat[perm]
                valid_idx = valid_idx[perm]
            
            # Quantize points to voxel indices
            voxel_coords = ((valid_pts - self.bounds_min) / self.resolution).long()
            # Convert tensor to individual scalars for clamp
            max_x, max_y, max_z = (self.grid_size - 1).tolist()
            voxel_coords[:, 0] = torch.clamp(voxel_coords[:, 0], 0, max_x)
            voxel_coords[:, 1] = torch.clamp(voxel_coords[:, 1], 0, max_y)  
            voxel_coords[:, 2] = torch.clamp(voxel_coords[:, 2], 0, max_z)
            
            # Create unique voxel indices and aggregate features
            # This is the key optimization - only store occupied voxels
            unique_voxels, inverse_indices = torch.unique(
                voxel_coords, return_inverse=True, dim=0
            )
            
            # Aggregate features for each voxel using scatter operations
            num_voxels = unique_voxels.shape[0]
            num_features = valid_feat.shape[1]
            
            voxel_features = torch.zeros(
                (num_voxels, num_features), 
                device=self.device, 
                dtype=valid_feat.dtype
            )
            voxel_counts = torch.zeros(
                num_voxels, 
                device=self.device, 
                dtype=torch.float32
            )
            
            # Efficient aggregation
            voxel_features.scatter_add_(0, 
                inverse_indices.unsqueeze(1).expand(-1, num_features), 
                valid_feat
            )
            voxel_counts.scatter_add_(0, inverse_indices, 
                torch.ones_like(inverse_indices, dtype=torch.float32)
            )
            
            # Average pooling
            voxel_features = voxel_features / voxel_counts.unsqueeze(1)
            
            sparse_voxels.append({
                'indices': unique_voxels,
                'features': voxel_features,
                'inverse': inverse_indices,  # For upsampling
                'points': valid_pts
            })
            
            # Normalized coordinates for point decoder
            norm_coords = (valid_pts - self.bounds_min) / self.bounds_range
            point_coords.append(norm_coords)
            valid_indices.append(valid_idx)
        
        return sparse_voxels, point_coords, valid_indices
    
    def sparse_to_dense(self, sparse_voxels, batch_size):
        """
        Convert sparse voxels to dense grid for ONNX compatibility
        Only create dense grid when necessary - with memory safety
        """
        D, H, W = self.grid_size.tolist()
        
        # Safety check for memory usage
        total_voxels = D * H * W
        if total_voxels > 134217728:  # 512^3 voxels max
            print(f"Warning: Grid too large ({D}x{H}x{W} = {total_voxels:,} voxels), using smaller grid")
            # Use a smaller, reasonable grid size
            D = min(D, 256)
            H = min(H, 256) 
            W = min(W, 256)
            
        C = sparse_voxels[0]['features'].shape[1] if sparse_voxels[0]['features'].shape[0] > 0 else 4
        
        # Create dense grid with safety bounds
        dense_grid = torch.zeros(
            (batch_size, C, D, H, W), 
            device=self.device, 
            dtype=torch.float32
        )
        
        for b, sparse in enumerate(sparse_voxels):
            if sparse['indices'].shape[0] > 0:
                indices = sparse['indices']
                features = sparse['features']
                
                # Clamp indices to grid bounds for safety
                indices = torch.clamp(indices, min=0, max=torch.tensor([D-1, H-1, W-1], device=self.device))
                
                # Efficient dense assignment
                dense_grid[b, :, indices[:, 0], indices[:, 1], indices[:, 2]] = features.T
        
        return dense_grid


class EfficientSparseBackbone(nn.Module):
    """
    Backbone that processes sparse voxels efficiently
    Mimics the original MinkowskiEngine backbone
    """
    def __init__(self, cfg):
        super().__init__()
        
        input_dim = cfg.INPUT_DIM
        cs = cfg.CHANNELS  # [32, 32, 64, 128, 256, 256, 128, 96, 96]
        self.resolution = cfg.RESOLUTION
        
        # Match the original architecture exactly
        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(input_dim, cs[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(cs[0]),
            nn.ReLU(inplace=True),
            nn.Conv3d(cs[0], cs[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(cs[0]),
            nn.ReLU(inplace=True),
        )
        
        # Encoder stages (matching original)
        self.stage1 = nn.Sequential(
            nn.Conv3d(cs[0], cs[0], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm3d(cs[0]),
            nn.ReLU(inplace=True),
            ResidualBlock3D(cs[0], cs[1], kernel_size=3),
            ResidualBlock3D(cs[1], cs[1], kernel_size=3),
        )
        
        self.stage2 = nn.Sequential(
            nn.Conv3d(cs[1], cs[1], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm3d(cs[1]),
            nn.ReLU(inplace=True),
            ResidualBlock3D(cs[1], cs[2], kernel_size=3),
            ResidualBlock3D(cs[2], cs[2], kernel_size=3),
        )
        
        self.stage3 = nn.Sequential(
            nn.Conv3d(cs[2], cs[2], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm3d(cs[2]),
            nn.ReLU(inplace=True),
            ResidualBlock3D(cs[2], cs[3], kernel_size=3),
            ResidualBlock3D(cs[3], cs[3], kernel_size=3),
        )
        
        self.stage4 = nn.Sequential(
            nn.Conv3d(cs[3], cs[3], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm3d(cs[3]),
            nn.ReLU(inplace=True),
            ResidualBlock3D(cs[3], cs[4], kernel_size=3),
            ResidualBlock3D(cs[4], cs[4], kernel_size=3),
        )
        
        # Decoder stages (matching original)
        self.up1 = nn.ModuleList([
            nn.ConvTranspose3d(cs[4], cs[5], kernel_size=2, stride=2, bias=False),
            nn.Sequential(
                ResidualBlock3D(cs[5] + cs[3], cs[5], kernel_size=3),
                ResidualBlock3D(cs[5], cs[5], kernel_size=3),
            )
        ])
        
        self.up2 = nn.ModuleList([
            nn.ConvTranspose3d(cs[5], cs[6], kernel_size=2, stride=2, bias=False),
            nn.Sequential(
                ResidualBlock3D(cs[6] + cs[2], cs[6], kernel_size=3),
                ResidualBlock3D(cs[6], cs[6], kernel_size=3),
            )
        ])
        
        self.up3 = nn.ModuleList([
            nn.ConvTranspose3d(cs[6], cs[7], kernel_size=2, stride=2, bias=False),
            nn.Sequential(
                ResidualBlock3D(cs[7] + cs[1], cs[7], kernel_size=3),
                ResidualBlock3D(cs[7], cs[7], kernel_size=3),
            )
        ])
        
        self.up4 = nn.ModuleList([
            nn.ConvTranspose3d(cs[7], cs[8], kernel_size=2, stride=2, bias=False),
            nn.Sequential(
                ResidualBlock3D(cs[8] + cs[0], cs[8], kernel_size=3),
                ResidualBlock3D(cs[8], cs[8], kernel_size=3),
            )
        ])
        
        # Output normalization (matching original)
        self.out_channels = [cs[5], cs[6], cs[7], cs[8]]
        self.out_bnorm = nn.ModuleList([
            nn.BatchNorm1d(ch) for ch in self.out_channels
        ])
        
        # Semantic head
        self.sem_head = nn.Linear(cs[-1], 20)
        
    def forward(self, dense_voxels, sparse_info=None):
        """
        Process with efficiency optimizations
        """
        # Encoder
        x0 = self.stem(dense_voxels)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        
        # Decoder with skip connections
        y1 = self.up1[0](x4)
        y1 = torch.cat([y1, x3], dim=1)
        y1 = self.up1[1](y1)
        
        y2 = self.up2[0](y1)
        y2 = torch.cat([y2, x2], dim=1)
        y2 = self.up2[1](y2)
        
        y3 = self.up3[0](y2)
        y3 = torch.cat([y3, x1], dim=1)
        y3 = self.up3[1](y3)
        
        y4 = self.up4[0](y3)
        y4 = torch.cat([y4, x0], dim=1)
        y4 = self.up4[1](y4)
        
        return [y1, y2, y3, y4]


class ResidualBlock3D(nn.Module):
    """Residual block matching original"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, 
                               padding=padding, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, 
                               padding=padding, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.downsample = nn.Sequential()
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        
    def forward(self, x):
        identity = self.downsample(x)
        
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        
        out = out + identity
        out = F.relu(out, inplace=True)
        
        return out