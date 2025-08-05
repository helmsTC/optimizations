"""
Complete ONNX-compatible MaskPLS model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

from .dense_backbone import DenseConv3DBackbone
from .onnx_decoder import ONNXCompatibleDecoder


class MaskPLSONNX(nn.Module):
    """
    ONNX-compatible version of MaskPS model
    This model can be exported to ONNX and run without MinkowskiEngine
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Determine dataset configuration
        dataset = cfg.MODEL.DATASET
        self.data_cfg = cfg[dataset]
        self.num_classes = self.data_cfg.NUM_CLASSES
        
        # Initialize backbone
        self.backbone = DenseConv3DBackbone(
            cfg.BACKBONE, 
            self.data_cfg
        )
        
        # Initialize decoder
        self.decoder = ONNXCompatibleDecoder(
            cfg.DECODER,
            cfg.BACKBONE,
            self.data_cfg
        )
        
        # Store configuration
        self.voxel_size = cfg.BACKBONE.RESOLUTION
        self.spatial_shape = getattr(cfg, 'SPATIAL_SHAPE', (96, 96, 8))
        self.coordinate_bounds = self.data_cfg.SPACE
        
        # Things/stuff IDs for panoptic segmentation
        self.things_ids = self.get_things_ids(dataset)
        self.overlap_threshold = cfg.MODEL.OVERLAP_THRESHOLD
        
    def get_things_ids(self, dataset):
        """Get thing class IDs based on dataset"""
        if dataset == "KITTI":
            return [1, 2, 3, 4, 5, 6, 7, 8]
        elif dataset == "NUSCENES":
            return [2, 3, 4, 5, 6, 7, 9, 10]
        else:
            return []
    
    def forward(self, points_or_dict, features=None):
        """
        Forward pass
        Args:
            points_or_dict: Either:
                - Dict from dataloader with 'pt_coord' and 'feats'
                - Tensor of points [B, N, 3]
            features: Tensor of features [B, N, C] (if points_or_dict is tensor)
        Returns:
            Dict with:
                - pred_logits: [B, Q, num_classes+1] class predictions
                - pred_masks: [B, N, Q] mask predictions
                - aux_outputs: List of intermediate predictions
                - sem_logits: [B, N, num_classes] semantic segmentation
        """
        # Handle different input formats
        if isinstance(points_or_dict, dict):
            # Input from dataloader
            x = points_or_dict
            
            # Get backbone features
            feats, coors, pad_masks, sem_logits = self.backbone(x)
            
        else:
            # Direct tensor input (for ONNX export)
            points = points_or_dict
            
            # Combine points and features for voxelization
            if features is None:
                # If no features provided, use coordinates + zeros for intensity
                features = torch.cat([
                    points,
                    torch.zeros(points.shape[0], points.shape[1], 1, device=points.device)
                ], dim=-1)
            
            # Voxelize and process through backbone
            B = points.shape[0]
            voxel_grids = []
            
            for b in range(B):
                voxel_grid = self.voxelize_points(points[b], features[b])
                voxel_grids.append(voxel_grid)
            
            voxel_features = torch.stack(voxel_grids)
            
            # Process through backbone
            feats, coors, pad_masks, sem_logits = self.backbone((voxel_features, points))
        
        # Process through decoder
        if feats and len(feats) > 0:
            # Use last feature level as mask features
            mask_features = feats[-1]
            
            # Decoder forward pass
            outputs = self.decoder(feats, mask_features, pad_masks)
            
            # Add semantic logits to output
            outputs['sem_logits'] = sem_logits
        else:
            # Empty output
            outputs = {
                'pred_logits': torch.zeros(1, 100, self.num_classes + 1),
                'pred_masks': torch.zeros(1, 1, 100),
                'aux_outputs': [],
                'sem_logits': torch.zeros(1, 1, self.num_classes)
            }
        
        return outputs
    
    def voxelize_points(self, points: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Convert points to voxel grid
        Args:
            points: [N, 3] point coordinates
            features: [N, C] point features
        Returns:
            voxel_grid: [C, D, H, W] voxelized features
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
        
        # Accumulate features
        for i in range(points.shape[0]):
            voxel_grid[:, voxel_x[i], voxel_y[i], voxel_z[i]] += features[i]
            count_grid[voxel_x[i], voxel_y[i], voxel_z[i]] += 1
        
        # Average pooling
        mask = count_grid > 0
        voxel_grid[:, mask] = voxel_grid[:, mask] / count_grid[mask].unsqueeze(0)
        
        return voxel_grid
    
    def panoptic_postprocess(self, outputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert model outputs to panoptic segmentation
        Args:
            outputs: Model outputs with pred_logits and pred_masks
        Returns:
            semantic: [B, N] semantic predictions
            instance: [B, N] instance predictions
        """
        mask_cls = outputs['pred_logits']  # [B, Q, C+1]
        mask_pred = outputs['pred_masks']  # [B, N, Q]
        
        B = mask_cls.shape[0]
        semantic_preds = []
        instance_preds = []
        
        for b in range(B):
            # Get class predictions
            scores, labels = mask_cls[b].max(-1)  # [Q]
            masks = mask_pred[b].sigmoid()  # [N, Q]
            
            # Filter out no-object predictions
            keep = labels.ne(self.num_classes)
            
            if keep.sum() == 0:
                # No valid predictions
                semantic = torch.zeros(masks.shape[0], dtype=torch.long, device=masks.device)
                instance = torch.zeros(masks.shape[0], dtype=torch.long, device=masks.device)
            else:
                cur_scores = scores[keep]
                cur_classes = labels[keep]
                cur_masks = masks[:, keep]  # [N, K]
                
                # Assign each point to highest scoring mask
                mask_scores = cur_scores.unsqueeze(0) * cur_masks  # [N, K]
                mask_ids = mask_scores.argmax(1)  # [N]
                
                # Get semantic and instance predictions
                semantic = cur_classes[mask_ids]
                
                # Instance IDs for thing classes
                instance = torch.zeros_like(semantic)
                instance_id = 1
                
                for k in range(cur_classes.shape[0]):
                    if cur_classes[k].item() in self.things_ids:
                        mask = (mask_ids == k)
                        if mask.sum() > 0:
                            instance[mask] = instance_id
                            instance_id += 1
            
            semantic_preds.append(semantic)
            instance_preds.append(instance)
        
        return torch.stack(semantic_preds), torch.stack(instance_preds)


class MaskPLSExportWrapper(nn.Module):
    """
    Wrapper for clean ONNX export with simplified interface
    """
    def __init__(self, model: MaskPLSONNX):
        super().__init__()
        self.model = model
    
    def forward(self, points: torch.Tensor, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simplified forward for ONNX export
        Args:
            points: [B, N, 3] point coordinates
            features: [B, N, 4] point features (xyz + intensity)
        Returns:
            pred_logits: [B, Q, C+1] class predictions
            pred_masks: [B, N, Q] mask predictions  
            sem_logits: [B, N, C] semantic predictions
        """
        outputs = self.model(points, features)
        
        return (
            outputs['pred_logits'],
            outputs['pred_masks'],
            outputs.get('sem_logits', torch.zeros(1, 1, self.model.num_classes))
        )


def create_onnx_model(cfg) -> MaskPLSONNX:
    """
    Factory function to create ONNX-compatible model
    """
    model = MaskPLSONNX(cfg)
    
    # Initialize weights properly
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    return model


def load_checkpoint_weights(model: MaskPLSONNX, checkpoint_path: str, strict: bool = False):
    """
    Load weights from original MaskPLS checkpoint
    """
    import torch
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    # Create mapping from old names to new names
    name_mapping = {
        # Backbone mappings
        'backbone.stem': 'backbone.stem',
        'backbone.stage': 'backbone.stage',
        'backbone.up': 'backbone.up',
        
        # Decoder mappings
        'decoder.query_feat': 'decoder.query_feat',
        'decoder.query_embed': 'decoder.query_pos',
        'decoder.class_embed': 'decoder.class_embed',
        'decoder.mask_embed': 'decoder.mask_embed',
    }
    
    # Filter and map weights
    new_state_dict = {}
    model_state = model.state_dict()
    
    for old_name, param in state_dict.items():
        # Remove 'module.' prefix if present
        old_name = old_name.replace('module.', '')
        
        # Skip MinkowskiEngine specific parameters
        if 'kernel' in old_name or 'MinkowskiConvolution' in old_name:
            continue
        
        # Try direct mapping
        if old_name in model_state:
            if param.shape == model_state[old_name].shape:
                new_state_dict[old_name] = param
        else:
            # Try mapped names
            for old_pattern, new_pattern in name_mapping.items():
                if old_pattern in old_name:
                    new_name = old_name.replace(old_pattern, new_pattern)
                    if new_name in model_state and param.shape == model_state[new_name].shape:
                        new_state_dict[new_name] = param
                        break
    
    # Load weights
    model.load_state_dict(new_state_dict, strict=strict)
    
    print(f"Loaded {len(new_state_dict)}/{len(model_state)} parameters from checkpoint")
    
    return model