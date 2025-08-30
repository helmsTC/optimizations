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
        
        return voxel_grids, point_coords, valid_indices
    
    def _get_valid_mask(self, points):
        """Get mask for points within bounds"""
        return ((points >= self.bounds_min) & 
                (points < self.bounds_max)).all(dim=1)
    
    def _trilinear_voxelize(self, voxel_grid, voxel_count, points, features, norm_coords):
        """
        Trilinear splatting for smooth voxelization
        """
        D, H, W = self.spatial_shape
        
        # Get voxel coordinates
        voxel_coords = norm_coords * torch.tensor([D, H, W], device=self.device)
        
        # Get integer and fractional parts
        voxel_coords_floor = torch.floor(voxel_coords).long()
        voxel_coords_ceil = torch.ceil(voxel_coords).long()
        voxel_coords_frac = voxel_coords - voxel_coords_floor.float()
        
        # Clamp coordinates
        voxel_coords_floor[:, 0] = torch.clamp(voxel_coords_floor[:, 0], 0, D-1)
        voxel_coords_floor[:, 1] = torch.clamp(voxel_coords_floor[:, 1], 0, H-1)
        voxel_coords_floor[:, 2] = torch.clamp(voxel_coords_floor[:, 2], 0, W-1)
        
        voxel_coords_ceil[:, 0] = torch.clamp(voxel_coords_ceil[:, 0], 0, D-1)
        voxel_coords_ceil[:, 1] = torch.clamp(voxel_coords_ceil[:, 1], 0, H-1)
        voxel_coords_ceil[:, 2] = torch.clamp(voxel_coords_ceil[:, 2], 0, W-1)
        
        # Compute trilinear weights
        weights = torch.stack([
            (1 - voxel_coords_frac[:, 0]) * (1 - voxel_coords_frac[:, 1]) * (1 - voxel_coords_frac[:, 2]),
            voxel_coords_frac[:, 0] * (1 - voxel_coords_frac[:, 1]) * (1 - voxel_coords_frac[:, 2]),
            (1 - voxel_coords_frac[:, 0]) * voxel_coords_frac[:, 1] * (1 - voxel_coords_frac[:, 2]),
            voxel_coords_frac[:, 0] * voxel_coords_frac[:, 1] * (1 - voxel_coords_frac[:, 2]),
            (1 - voxel_coords_frac[:, 0]) * (1 - voxel_coords_frac[:, 1]) * voxel_coords_frac[:, 2],
            voxel_coords_frac[:, 0] * (1 - voxel_coords_frac[:, 1]) * voxel_coords_frac[:, 2],
            (1 - voxel_coords_frac[:, 0]) * voxel_coords_frac[:, 1] * voxel_coords_frac[:, 2],
            voxel_coords_frac[:, 0] * voxel_coords_frac[:, 1] * voxel_coords_frac[:, 2]
        ], dim=0).T
        
        # Get 8 corner coordinates
        corners = torch.stack([
            voxel_coords_floor,
            torch.stack([voxel_coords_ceil[:, 0], voxel_coords_floor[:, 1], voxel_coords_floor[:, 2]], dim=1),
            torch.stack([voxel_coords_floor[:, 0], voxel_coords_ceil[:, 1], voxel_coords_floor[:, 2]], dim=1),
            torch.stack([voxel_coords_ceil[:, 0], voxel_coords_ceil[:, 1], voxel_coords_floor[:, 2]], dim=1),
            torch.stack([voxel_coords_floor[:, 0], voxel_coords_floor[:, 1], voxel_coords_ceil[:, 2]], dim=1),
            torch.stack([voxel_coords_ceil[:, 0], voxel_coords_floor[:, 1], voxel_coords_ceil[:, 2]], dim=1),
            torch.stack([voxel_coords_floor[:, 0], voxel_coords_ceil[:, 1], voxel_coords_ceil[:, 2]], dim=1),
            voxel_coords_ceil
        ], dim=0)
        
        # Splat to voxels
        for i in range(8):
            corner = corners[i]
            weight = weights[:, i:i+1]
            
            # Flatten indices
            flat_idx = corner[:, 0] * H * W + corner[:, 1] * W + corner[:, 2]
            
            # Accumulate weighted features
            for c in range(features.shape[1]):
                # Ensure dtypes match
                weighted_features = (features[:, c] * weight.squeeze()).to(voxel_grid.dtype)
                voxel_grid[c].view(-1).scatter_add_(
                    0, flat_idx, weighted_features
                )
            
            # Accumulate weights
            # Ensure weight dtype matches voxel_count dtype
            weight_to_add = weight.squeeze().to(voxel_count.dtype)
            voxel_count.view(-1).scatter_add_(
                0, flat_idx, weight_to_add
            ) D*H*W]
        
        # Vectorized coordinate conversion (all batches at once)
        spatial_dims = torch.tensor([D, H, W], device=point_coords.device, dtype=point_coords.dtype)
        voxel_coords = point_coords * spatial_dims.unsqueeze(0).unsqueeze(0)
        
        # Efficient clamping - clamp each dimension separately
        voxel_coords[..., 0] = torch.clamp(voxel_coords[..., 0], 0, D - 1)
        voxel_coords[..., 1] = torch.clamp(voxel_coords[..., 1], 0, H - 1)
        voxel_coords[..., 2] = torch.clamp(voxel_coords[..., 2], 0, W - 1)
        voxel_coords = voxel_coords.long()
        
        # Vectorized flat indexing
        flat_indices = (voxel_coords[..., 0] * H * W + 
                       voxel_coords[..., 1] * W + 
                       voxel_coords[..., 2])
        
        # Efficient gather operation
        flat_indices_expanded = flat_indices.unsqueeze(1).expand(-1, C_out, -1)
        point_features = torch.gather(voxel_flat, 2, flat_indices_expanded)
        point_features = point_features.transpose(1, 2)  # [B, N, C]
        
        # Efficient MLP with chunking for large inputs
        if N > 10000:
            # Process in chunks to avoid memory issues
            chunk_size = 5000
            output_chunks = []
            for i in range(0, N, chunk_size):
                end_idx = min(i + chunk_size, N)
                chunk = self.mlp(point_features[:, i:end_idx])
                output_chunks.append(chunk)
            point_features = torch.cat(output_chunks, dim=1)
        else:
            point_features = self.mlp(point_features)
        
        return point_features