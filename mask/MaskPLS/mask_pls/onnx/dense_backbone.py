"""
Dense 3D CNN backbone for ONNX compatibility
Replaces MinkowskiEngine sparse convolutions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


class DenseConv3DBackbone(nn.Module):
    """
    Dense 3D convolution backbone to replace MinkowskiEngine operations
    for ONNX compatibility.
    """
    def __init__(self, cfg, data_cfg):
        super().__init__()
        self.cfg = cfg
        self.data_cfg = data_cfg
        
        input_dim = cfg.INPUT_DIM
        cs = cfg.CHANNELS
        n_classes = data_cfg.NUM_CLASSES
        self.resolution = cfg.RESOLUTION
        
        # Voxelization parameters
        self.spatial_shape = (96, 96, 8)  # (D, H, W)
        self.coordinate_bounds = tuple(data_cfg.SPACE)
        
        # Encoder blocks - using regular 3D convolutions
        self.stem = nn.Sequential(
            nn.Conv3d(input_dim, cs[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(cs[0]),
            nn.ReLU(inplace=True),
            nn.Conv3d(cs[0], cs[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(cs[0]),
            nn.ReLU(inplace=True),
        )
        
        # Downsampling stages
        self.stage1 = nn.Sequential(
            nn.Conv3d(cs[0], cs[0], kernel_size=2, stride=2),
            nn.BatchNorm3d(cs[0]),
            nn.ReLU(inplace=True),
            ResidualBlock3D(cs[0], cs[1]),
            ResidualBlock3D(cs[1], cs[1]),
        )
        
        self.stage2 = nn.Sequential(
            nn.Conv3d(cs[1], cs[1], kernel_size=2, stride=2),
            nn.BatchNorm3d(cs[1]),
            nn.ReLU(inplace=True),
            ResidualBlock3D(cs[1], cs[2]),
            ResidualBlock3D(cs[2], cs[2]),
        )
        
        self.stage3 = nn.Sequential(
            nn.Conv3d(cs[2], cs[2], kernel_size=2, stride=2),
            nn.BatchNorm3d(cs[2]),
            nn.ReLU(inplace=True),
            ResidualBlock3D(cs[2], cs[3]),
            ResidualBlock3D(cs[3], cs[3]),
        )
        
        self.stage4 = nn.Sequential(
            nn.Conv3d(cs[3], cs[3], kernel_size=2, stride=2),
            nn.BatchNorm3d(cs[3]),
            nn.ReLU(inplace=True),
            ResidualBlock3D(cs[3], cs[4]),
            ResidualBlock3D(cs[4], cs[4]),
        )
        
        # Decoder blocks with skip connections
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(cs[4], cs[5], kernel_size=2, stride=2),
            nn.BatchNorm3d(cs[5]),
            nn.ReLU(inplace=True),
        )
        self.up1_merge = nn.Sequential(
            ResidualBlock3D(cs[5] + cs[3], cs[5]),
            ResidualBlock3D(cs[5], cs[5]),
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(cs[5], cs[6], kernel_size=2, stride=2),
            nn.BatchNorm3d(cs[6]),
            nn.ReLU(inplace=True),
        )
        self.up2_merge = nn.Sequential(
            ResidualBlock3D(cs[6] + cs[2], cs[6]),
            ResidualBlock3D(cs[6], cs[6]),
        )
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(cs[6], cs[7], kernel_size=2, stride=2),
            nn.BatchNorm3d(cs[7]),
            nn.ReLU(inplace=True),
        )
        self.up3_merge = nn.Sequential(
            ResidualBlock3D(cs[7] + cs[1], cs[7]),
            ResidualBlock3D(cs[7], cs[7]),
        )
        
        self.up4 = nn.Sequential(
            nn.ConvTranspose3d(cs[7], cs[8], kernel_size=2, stride=2),
            nn.BatchNorm3d(cs[8]),
            nn.ReLU(inplace=True),
        )
        self.up4_merge = nn.Sequential(
            ResidualBlock3D(cs[8] + cs[0], cs[8]),
            ResidualBlock3D(cs[8], cs[8]),
        )
        
        # Output projections for multi-scale features
        levels = [cs[-i] for i in range(4, 0, -1)]
        self.out_convs = nn.ModuleList([
            nn.Conv3d(l, l, kernel_size=1) for l in levels
        ])
        
        # Semantic segmentation head
        self.sem_head = nn.Linear(cs[-1], n_classes)
        
        # KNN interpolation module (ONNX-compatible)
        self.knn_up = KNNInterpolationONNX(k=3)
        
    def voxelize_batch(self, points_batch: List[torch.Tensor], 
                      features_batch: List[torch.Tensor]) -> torch.Tensor:
        """
        Convert batch of point clouds to voxel grids
        """
        batch_size = len(points_batch)
        C = features_batch[0].shape[1]
        D, H, W = self.spatial_shape
        
        voxel_batch = torch.zeros(batch_size, C, D, H, W, 
                                  device=features_batch[0].device)
        
        for b, (points, features) in enumerate(zip(points_batch, features_batch)):
            voxel_batch[b] = self.voxelize_single(points, features)
        
        return voxel_batch
    
    def voxelize_single(self, points: torch.Tensor, 
                       features: torch.Tensor) -> torch.Tensor:
        """
        Convert single point cloud to voxel grid
        """
        D, H, W = self.spatial_shape
        C = features.shape[1]
        
        # Initialize voxel grid
        voxel_grid = torch.zeros(C, D, H, W, device=features.device)
        count_grid = torch.zeros(D, H, W, device=features.device)
        
        # Normalize coordinates to voxel indices
        x_min, x_max = self.coordinate_bounds[0]
        y_min, y_max = self.coordinate_bounds[1]
        z_min, z_max = self.coordinate_bounds[2]
        
        # Compute voxel indices
        voxel_x = ((points[:, 0] - x_min) / (x_max - x_min) * D).long()
        voxel_y = ((points[:, 1] - y_min) / (y_max - y_min) * H).long()
        voxel_z = ((points[:, 2] - z_min) / (z_max - z_min) * W).long()
        
        # Clip to valid range
        voxel_x = torch.clamp(voxel_x, 0, D - 1)
        voxel_y = torch.clamp(voxel_y, 0, H - 1)
        voxel_z = torch.clamp(voxel_z, 0, W - 1)
        
        # Accumulate features (average pooling)
        for i in range(points.shape[0]):
            voxel_grid[:, voxel_x[i], voxel_y[i], voxel_z[i]] += features[i]
            count_grid[voxel_x[i], voxel_y[i], voxel_z[i]] += 1
        
        # Average pooling
        mask = count_grid > 0
        for c in range(C):
            voxel_grid[c][mask] /= count_grid[mask]
        
        return voxel_grid
    
    def extract_voxel_features(self, voxel_features: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract features at each voxel location and flatten
        """
        B, C, D, H, W = voxel_features.shape
        # Flatten spatial dimensions
        features = voxel_features.view(B, C, -1).transpose(1, 2)  # [B, D*H*W, C]
        return features
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: dict with 'pt_coord' and 'feats' OR tuple (voxel_features, points)
        """
        if isinstance(x, dict):
            # Batch input from dataloader
            points_batch = [torch.from_numpy(p).cuda() for p in x['pt_coord']]
            features_batch = [torch.from_numpy(f).cuda() for f in x['feats']]
            
            # Voxelize
            voxel_features = self.voxelize_batch(points_batch, features_batch)
            
            # Store original coordinates for interpolation
            coords = points_batch
        else:
            # Direct tensor input (for ONNX export)
            voxel_features, coords = x
            if not isinstance(coords, list):
                coords = [coords[b] for b in range(coords.shape[0])]
        
        # Encoder
        x0 = self.stem(voxel_features)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        
        # Decoder with skip connections
        y1 = self.up1(x4)
        y1 = torch.cat([y1, x3], dim=1)
        y1 = self.up1_merge(y1)
        
        y2 = self.up2(y1)
        y2 = torch.cat([y2, x2], dim=1)
        y2 = self.up2_merge(y2)
        
        y3 = self.up3(y2)
        y3 = torch.cat([y3, x1], dim=1)
        y3 = self.up3_merge(y3)
        
        y4 = self.up4(y3)
        y4 = torch.cat([y4, x0], dim=1)
        y4 = self.up4_merge(y4)
        
        # Extract multi-scale features
        multi_scale_voxels = [y1, y2, y3, y4]
        feats = []
        coors = []
        pad_masks = []
        
        for i, (voxel_feat, conv) in enumerate(zip(multi_scale_voxels, self.out_convs)):
            # Apply output convolution
            voxel_feat = conv(voxel_feat)
            
            # Extract features at voxel locations
            feat = self.extract_voxel_features(voxel_feat)
            
            # Interpolate to original points
            batch_feats = []
            batch_coors = []
            batch_masks = []
            
            for b in range(voxel_feat.shape[0]):
                # Get voxel coordinates
                voxel_coords = self.get_voxel_coords(voxel_feat.shape[2:])
                
                # Interpolate to original points
                if b < len(coords):
                    interp_feat = self.knn_up(
                        voxel_coords.to(voxel_feat.device),
                        feat[b],
                        coords[b].to(voxel_feat.device)
                    )
                    
                    # Pad to max size in batch
                    max_pts = max(c.shape[0] for c in coords)
                    pad_size = max_pts - coords[b].shape[0]
                    
                    if pad_size > 0:
                        interp_feat = F.pad(interp_feat, (0, 0, 0, pad_size))
                        padded_coords = F.pad(coords[b], (0, 0, 0, pad_size))
                        mask = torch.cat([
                            torch.zeros(coords[b].shape[0], device=voxel_feat.device, dtype=torch.bool),
                            torch.ones(pad_size, device=voxel_feat.device, dtype=torch.bool)
                        ])
                    else:
                        padded_coords = coords[b]
                        mask = torch.zeros(coords[b].shape[0], device=voxel_feat.device, dtype=torch.bool)
                    
                    batch_feats.append(interp_feat)
                    batch_coors.append(padded_coords)
                    batch_masks.append(mask)
            
            if batch_feats:
                feats.append(torch.stack(batch_feats))
                coors.append(torch.stack(batch_coors))
                pad_masks.append(torch.stack(batch_masks))
        
        # Semantic segmentation logits
        if feats:
            logits = self.sem_head(feats[-1])
        else:
            logits = None
        
        return feats, coors, pad_masks, logits
    
    def get_voxel_coords(self, shape):
        """Generate voxel center coordinates"""
        D, H, W = shape
        x = torch.linspace(-48, 48, D)
        y = torch.linspace(-48, 48, H)
        z = torch.linspace(-4, 1.5, W)
        
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        coords = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
        
        return coords


class ResidualBlock3D(nn.Module):
    """3D Residual Block for dense convolutions"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, 
                               stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, 
                               stride=1, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = out + identity
        out = self.relu(out)
        
        return out


class KNNInterpolationONNX(nn.Module):
    """
    ONNX-compatible KNN interpolation to replace PyKeOps
    """
    def __init__(self, k=3):
        super().__init__()
        self.k = k
    
    def forward(self, source_points: torch.Tensor, 
                source_features: torch.Tensor,
                target_points: torch.Tensor) -> torch.Tensor:
        """
        Interpolate features from source points to target points using KNN
        """
        # Handle empty inputs
        if source_points.shape[0] == 0 or target_points.shape[0] == 0:
            return torch.zeros(target_points.shape[0], source_features.shape[1], 
                              device=source_features.device)
        
        # Compute pairwise distances
        # [M, 1, 3] - [1, N, 3] -> [M, N]
        dists = torch.sum(
            (target_points.unsqueeze(1) - source_points.unsqueeze(0)) ** 2,
            dim=-1
        )
        
        # Find k-nearest neighbors
        k = min(self.k, source_points.shape[0])
        knn_dists, knn_indices = torch.topk(dists, k, dim=1, largest=False)
        
        # Compute weights (inverse distance weighting)
        weights = 1.0 / (knn_dists + 1e-8)
        weights = weights / weights.sum(dim=1, keepdim=True)
        
        # Gather features and interpolate
        M = target_points.shape[0]
        C = source_features.shape[1]
        
        knn_features = source_features[knn_indices.view(-1)].view(M, k, C)
        interpolated = torch.sum(knn_features * weights.unsqueeze(-1), dim=1)
        
        return interpolated