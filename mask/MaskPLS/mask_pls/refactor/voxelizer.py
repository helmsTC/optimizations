# mask/MaskPLS/mask_pls/models/onnx/voxelizer.py
"""
High-resolution voxelizer for better spatial accuracy
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict


class HighResVoxelizer:
    """High-resolution voxelizer with efficient memory usage"""
    def __init__(self, spatial_shape, coordinate_bounds, device='cuda'):
        # Use higher resolution for better accuracy
        self.spatial_shape = spatial_shape  # (128, 128, 64) or higher
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
        
    def voxelize_batch(self, points_list, features_list, max_points=None):
        """
        Voxelize batch of point clouds with high resolution
        
        Returns:
            voxel_grids: [B, C, D, H, W]
            point_coords: List of [N, 3] normalized coordinates
            valid_indices: List of valid point indices
        """
        B = len(points_list)
        D, H, W = self.spatial_shape
        C = features_list[0].shape[1]
        
        # Use half precision for memory efficiency with large grids
        dtype = torch.float16 if D * H * W > 1e6 else torch.float32
        
        voxel_grids = torch.zeros(B, C, D, H, W, device=self.device, dtype=dtype)
        voxel_counts = torch.zeros(B, D, H, W, device=self.device, dtype=dtype)
        
        point_coords = []
        valid_indices = []
        
        for b in range(B):
            # Process each point cloud
            pts = torch.from_numpy(points_list[b]).to(self.device)
            feat = torch.from_numpy(features_list[b]).to(self.device)
            
            # Filter valid points
            valid_mask = self._get_valid_mask(pts)
            valid_idx = torch.where(valid_mask)[0]
            
            if valid_idx.numel() == 0:
                point_coords.append(torch.zeros(0, 3, device=self.device))
                valid_indices.append(valid_idx)
                continue
            
            valid_pts = pts[valid_idx]
            valid_feat = feat[valid_idx]
            
            # Subsample if needed
            if max_points and valid_pts.shape[0] > max_points:
                perm = torch.randperm(valid_pts.shape[0])[:max_points]
                valid_pts = valid_pts[perm]
                valid_feat = valid_feat[perm]
                valid_idx = valid_idx[perm]
            
            # Normalize coordinates
            norm_coords = (valid_pts - self.bounds_min) / self.bounds_range
            norm_coords = torch.clamp(norm_coords, 0, 0.999999)
            
            # Voxelize with trilinear splatting for better accuracy
            self._trilinear_voxelize(
                voxel_grids[b], 
                voxel_counts[b],
                valid_pts,
                valid_feat,
                norm_coords
            )
            
            point_coords.append(norm_coords)
            valid_indices.append(valid_idx)
        
        # Average the voxels
        mask = voxel_counts > 0
        for b in range(B):
            for c in range(C):
                voxel_grids[b, c][mask[b]] /= voxel_counts[b][mask[b]]
        
        # Convert back to float32 if needed
        if dtype == torch.float16:
            voxel_grids = voxel_grids.float()

