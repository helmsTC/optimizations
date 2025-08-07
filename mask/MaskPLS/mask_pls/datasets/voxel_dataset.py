"""
Dataset wrapper that handles voxelization for the simplified model
Save as: mask/MaskPLS/mask_pls/datasets/voxel_dataset.py
"""

import torch
import numpy as np
from torch.utils.data import Dataset


class VoxelizationWrapper(Dataset):
    """
    Wraps the existing MaskSemanticDataset to add voxelization
    """
    def __init__(self, 
                 base_dataset,
                 spatial_shape=(32, 32, 16),
                 coordinate_bounds=[[-48.0, 48.0], [-48.0, 48.0], [-4.0, 1.5]],
                 max_points=50000):
        self.base_dataset = base_dataset
        self.spatial_shape = spatial_shape
        self.coordinate_bounds = np.array(coordinate_bounds)
        self.max_points = max_points
        
        # Precompute normalization factors
        self.coord_min = self.coordinate_bounds[:, 0]
        self.coord_range = self.coordinate_bounds[:, 1] - self.coordinate_bounds[:, 0]
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, index):
        # Get original data
        data = self.base_dataset[index]
        xyz, feats, sem_labels, ins_labels, masks, masks_cls, masks_ids, fname, pose, token = data
        
        # Convert to tensors
        xyz_tensor = torch.from_numpy(xyz).float()
        feats_tensor = torch.from_numpy(feats).float()
        
        # Filter points within bounds
        valid_mask = torch.ones(xyz_tensor.shape[0], dtype=torch.bool)
        for dim in range(3):
            valid_mask &= (xyz_tensor[:, dim] >= self.coord_min[dim])
            valid_mask &= (xyz_tensor[:, dim] < self.coord_min[dim] + self.coord_range[dim])
        
        # Get valid points
        valid_indices = torch.where(valid_mask)[0]
        valid_xyz = xyz_tensor[valid_mask]
        valid_feats = feats_tensor[valid_mask]
        
        # Limit points if needed
        if len(valid_xyz) > self.max_points:
            perm = torch.randperm(len(valid_xyz))[:self.max_points]
            valid_xyz = valid_xyz[perm]
            valid_feats = valid_feats[perm]
            valid_indices = valid_indices[perm]
        
        # Normalize coordinates to [0, 1]
        norm_coords = (valid_xyz - torch.from_numpy(self.coord_min).float()) / torch.from_numpy(self.coord_range).float()
        norm_coords = torch.clamp(norm_coords, 0, 0.999)
        
        # Voxelize
        voxel_grid = self.voxelize(valid_xyz, valid_feats)
        
        # Prepare output - need to handle masks for valid points only
        valid_masks = []
        valid_masks_ids = []
        
        for mask, mask_id in zip(masks, masks_ids):
            # Convert mask indices to valid indices
            mask_valid = torch.zeros(len(valid_indices), dtype=mask.dtype)
            for i, vi in enumerate(valid_indices):
                if vi < len(mask):
                    mask_valid[i] = mask[vi]
            
            # Only keep mask if it has enough points
            if mask_valid.sum() > 10:  # min points threshold
                valid_masks.append(mask_valid)
                
                # Remap mask IDs to valid indices
                valid_mask_id = torch.where(mask_valid)[0]
                valid_masks_ids.append(valid_mask_id)
        
        if len(valid_masks) > 0:
            valid_masks = torch.stack(valid_masks)
        else:
            valid_masks = torch.zeros(0, len(valid_indices))
        
        # Return voxelized data
        return {
            'voxel_grid': voxel_grid,
            'norm_coords': norm_coords,
            'valid_indices': valid_indices,
            'num_valid': len(valid_indices),
            'xyz': xyz,  # Original coordinates
            'feats': feats,
            'sem_labels': sem_labels,
            'ins_labels': ins_labels,
            'masks': valid_masks,
            'masks_cls': masks_cls,
            'masks_ids': valid_masks_ids,
            'fname': fname,
            'pose': pose,
            'token': token
        }
    
    def voxelize(self, points, features):
        """Voxelize points and features"""
        D, H, W = self.spatial_shape
        C = features.shape[1]
        
        voxel_grid = torch.zeros(C, D, H, W)
        count_grid = torch.zeros(D, H, W)
        
        # Convert to voxel indices
        voxel_coords = (points - torch.from_numpy(self.coord_min).float()) / torch.from_numpy(self.coord_range).float()
        voxel_coords = (voxel_coords * torch.tensor([D, H, W], dtype=torch.float32)).long()
        voxel_coords = torch.clamp(voxel_coords, 0, torch.tensor([D-1, H-1, W-1]))
        
        # Accumulate features
        for i in range(len(points)):
            d, h, w = voxel_coords[i]
            voxel_grid[:, d, h, w] += features[i]
            count_grid[d, h, w] += 1
        
        # Average
        mask = count_grid > 0
        for c in range(C):
            voxel_grid[c][mask] /= count_grid[mask]
        
        return voxel_grid


class VoxelBatchCollation:
    """Custom collation for voxelized data"""
    def __init__(self):
        pass
    
    def __call__(self, batch):
        # Stack voxel grids
        voxel_grids = torch.stack([item['voxel_grid'] for item in batch])
        
        # Pad coordinates to same size
        max_points = max(item['num_valid'] for item in batch)
        
        norm_coords = []
        padding_masks = []
        
        for item in batch:
            coords = item['norm_coords']
            n_pts = item['num_valid']
            
            # Pad if needed
            if n_pts < max_points:
                pad_size = max_points - n_pts
                coords = F.pad(coords, (0, 0, 0, pad_size))
            
            norm_coords.append(coords)
            
            # Create padding mask
            mask = torch.zeros(max_points, dtype=torch.bool)
            mask[n_pts:] = True
            padding_masks.append(mask)
        
        norm_coords = torch.stack(norm_coords)
        padding_masks = torch.stack(padding_masks)
        
        # Collect other data
        return {
            'voxel_grids': voxel_grids,
            'norm_coords': norm_coords,
            'padding_masks': padding_masks,
            'valid_indices': [item['valid_indices'] for item in batch],
            'pt_coord': [item['xyz'] for item in batch],
            'feats': [item['feats'] for item in batch],
            'sem_label': [item['sem_labels'] for item in batch],
            'ins_label': [item['ins_labels'] for item in batch],
            'masks': [item['masks'] for item in batch],
            'masks_cls': [item['masks_cls'] for item in batch],
            'masks_ids': [item['masks_ids'] for item in batch],
            'fname': [item['fname'] for item in batch],
            'pose': [item['pose'] for item in batch],
            'token': [item['token'] for item in batch],
        }
