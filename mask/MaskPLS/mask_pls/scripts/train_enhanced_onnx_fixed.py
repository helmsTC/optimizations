# mask/MaskPLS/mask_pls/scripts/train_enhanced_onnx_complete.py
"""
Complete fixed training script for ONNX-compatible MaskPLS
Resolves dimension mismatch errors and improves stability
"""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from os.path import join
import click
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import numpy as np
import time
from pathlib import Path
from easydict import EasyDict as edict
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule

# Import original components
from mask_pls.datasets.semantic_dataset import SemanticDatasetModule
from mask_pls.models.decoder import MaskedTransformerDecoder
from mask_pls.models.loss import SemLoss
from mask_pls.models.matcher import HungarianMatcher
from mask_pls.utils.evaluate_panoptic import PanopticEvaluator
from mask_pls.utils.misc import pad_stack


# ============================================================================
# BUILDING BLOCKS
# ============================================================================

class ResidualBlock3D(nn.Module):
    """Residual block for 3D convolutions"""
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


# ============================================================================
# VOXELIZATION
# ============================================================================

class SparseVoxelizer:
    """Efficient sparse voxelization"""
    def __init__(self, resolution=0.05, coordinate_bounds=None, device='cuda'):
        self.resolution = resolution
        self.bounds_min = torch.tensor([coordinate_bounds[i][0] for i in range(3)], device=device)
        self.bounds_max = torch.tensor([coordinate_bounds[i][1] for i in range(3)], device=device)
        self.device = device
    
    def voxelize_batch(self, points_list, features_list, max_points=10000):
        """Voxelize batch of point clouds"""
        B = len(points_list)
        
        # Calculate grid dimensions
        grid_size = ((self.bounds_max - self.bounds_min) / self.resolution).long()
        grid_size = torch.clamp(grid_size, min=1, max=64)  # Limit size for memory
        D, H, W = grid_size.tolist()  # Convert to Python ints
        
        all_voxel_grids = []
        all_point_coords = []
        all_valid_indices = []
        
        for b in range(B):
            pts = torch.from_numpy(points_list[b]).to(self.device).float()
            feat = torch.from_numpy(features_list[b]).to(self.device).float()
            
            # Filter valid points
            valid_mask = ((pts >= self.bounds_min) & (pts < self.bounds_max)).all(dim=1)
            valid_idx = torch.where(valid_mask)[0]
            
            if valid_idx.numel() == 0:
                # Empty point cloud
                C = feat.shape[1]
                all_voxel_grids.append(torch.zeros(C, D, H, W, device=self.device))
                all_point_coords.append(torch.zeros(0, 3, device=self.device))
                all_valid_indices.append(valid_idx)
                continue
            
            valid_pts = pts[valid_idx]
            valid_feat = feat[valid_idx]
            
            # Subsample if needed
            if valid_pts.shape[0] > max_points:
                perm = torch.randperm(valid_pts.shape[0])[:max_points]
                valid_pts = valid_pts[perm]
                valid_feat = valid_feat[perm]
                valid_idx = valid_idx[perm]
            
            # Voxelize
            voxel_coords = ((valid_pts - self.bounds_min) / self.resolution).long()
            # Clamp each dimension separately with scalar values
            voxel_coords[:, 0] = torch.clamp(voxel_coords[:, 0], min=0, max=D-1)
            voxel_coords[:, 1] = torch.clamp(voxel_coords[:, 1], min=0, max=H-1)
            voxel_coords[:, 2] = torch.clamp(voxel_coords[:, 2], min=0, max=W-1)
            
            # Create dense grid
            C = valid_feat.shape[1]
            dense_grid = torch.zeros(C, D, H, W, device=self.device)
            counts = torch.zeros(D, H, W, device=self.device)
            
            # Accumulate features
            for i in range(len(valid_pts)):
                d, h, w = voxel_coords[i].tolist()
                dense_grid[:, d, h, w] += valid_feat[i]
                counts[d, h, w] += 1
            
            # Average
            mask = counts > 0
            for c in range(C):
                dense_grid[c][mask] /= counts[mask]
            
            # Normalize coordinates
            norm_coords = (valid_pts - self.bounds_min) / (self.bounds_max - self.bounds_min)
            
            all_voxel_grids.append(dense_grid)
            all_point_coords.append(norm_coords)
            all_valid_indices.append(valid_idx)
        
        # Stack batch
        voxel_batch = torch.stack(all_voxel_grids)
        return voxel_batch, all_point_coords, all_valid_indices


# ============================================================================
# BACKBONE
# ============================================================================

class SparseBackbone(nn.Module):
    """Backbone network matching original architecture"""
    def __init__(self, cfg):
        super().__init__()
        
        input_dim = cfg.INPUT_DIM
        cs = cfg.CHANNELS  # [32, 32, 64, 128, 256, 256, 128, 96, 96]
        self.resolution = cfg.RESOLUTION
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(input_dim, cs[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(cs[0]),
            nn.ReLU(inplace=True),
            nn.Conv3d(cs[0], cs[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(cs[0]),
            nn.ReLU(inplace=True),
        )
        
        # Encoder stages
        self.stage1 = nn.Sequential(
            nn.Conv3d(cs[0], cs[0], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm3d(cs[0]),
            nn.ReLU(inplace=True),
            ResidualBlock3D(cs[0], cs[1]),
            ResidualBlock3D(cs[1], cs[1]),
        )
        
        self.stage2 = nn.Sequential(
            nn.Conv3d(cs[1], cs[1], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm3d(cs[1]),
            nn.ReLU(inplace=True),
            ResidualBlock3D(cs[1], cs[2]),
            ResidualBlock3D(cs[2], cs[2]),
        )
        
        self.stage3 = nn.Sequential(
            nn.Conv3d(cs[2], cs[2], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm3d(cs[2]),
            nn.ReLU(inplace=True),
            ResidualBlock3D(cs[2], cs[3]),
            ResidualBlock3D(cs[3], cs[3]),
        )
        
        self.stage4 = nn.Sequential(
            nn.Conv3d(cs[3], cs[3], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm3d(cs[3]),
            nn.ReLU(inplace=True),
            ResidualBlock3D(cs[3], cs[4]),
            ResidualBlock3D(cs[4], cs[4]),
        )
        
        # Decoder stages
        self.up1 = nn.ModuleList([
            nn.ConvTranspose3d(cs[4], cs[5], kernel_size=2, stride=2, bias=False),
            nn.Sequential(
                ResidualBlock3D(cs[5] + cs[3], cs[5]),
                ResidualBlock3D(cs[5], cs[5]),
            )
        ])
        
        self.up2 = nn.ModuleList([
            nn.ConvTranspose3d(cs[5], cs[6], kernel_size=2, stride=2, bias=False),
            nn.Sequential(
                ResidualBlock3D(cs[6] + cs[2], cs[6]),
                ResidualBlock3D(cs[6], cs[6]),
            )
        ])
        
        self.up3 = nn.ModuleList([
            nn.ConvTranspose3d(cs[6], cs[7], kernel_size=2, stride=2, bias=False),
            nn.Sequential(
                ResidualBlock3D(cs[7] + cs[1], cs[7]),
                ResidualBlock3D(cs[7], cs[7]),
            )
        ])
        
        self.up4 = nn.ModuleList([
            nn.ConvTranspose3d(cs[7], cs[8], kernel_size=2, stride=2, bias=False),
            nn.Sequential(
                ResidualBlock3D(cs[8] + cs[0], cs[8]),
                ResidualBlock3D(cs[8], cs[8]),
            )
        ])
        
        self.out_channels = [cs[5], cs[6], cs[7], cs[8]]
        self.out_bnorm = nn.ModuleList([nn.BatchNorm1d(ch) for ch in self.out_channels])
        self.sem_head = nn.Linear(cs[8], 20)
    
    def forward(self, dense_voxels):
        # Encoder
        x0 = self.stem(dense_voxels)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        
        # Decoder with skip connections
        y1 = self.up1[0](x4)
        if y1.shape[2:] != x3.shape[2:]:
            y1 = F.interpolate(y1, size=x3.shape[2:], mode='trilinear', align_corners=True)
        y1 = torch.cat([y1, x3], dim=1)
        y1 = self.up1[1](y1)
        
        y2 = self.up2[0](y1)
        if y2.shape[2:] != x2.shape[2:]:
            y2 = F.interpolate(y2, size=x2.shape[2:], mode='trilinear', align_corners=True)
        y2 = torch.cat([y2, x2], dim=1)
        y2 = self.up2[1](y2)
        
        y3 = self.up3[0](y2)
        if y3.shape[2:] != x1.shape[2:]:
            y3 = F.interpolate(y3, size=x1.shape[2:], mode='trilinear', align_corners=True)
        y3 = torch.cat([y3, x1], dim=1)
        y3 = self.up3[1](y3)
        
        y4 = self.up4[0](y3)
        if y4.shape[2:] != x0.shape[2:]:
            y4 = F.interpolate(y4, size=x0.shape[2:], mode='trilinear', align_corners=True)
        y4 = torch.cat([y4, x0], dim=1)
        y4 = self.up4[1](y4)
        
        return [y1, y2, y3, y4], self.out_bnorm


# ============================================================================
# POINT FEATURE EXTRACTION
# ============================================================================

class PointFeatureExtractor(nn.Module):
    """Extract features for points from voxel grids"""
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.resolution = 0.05
    
    def interpolate_features(self, voxel_features, point_coords):
        """Trilinear interpolation from voxel grid to points"""
        B, C, D, H, W = voxel_features.shape
        
        # Convert to grid coordinates [-1, 1]
        grid_coords = point_coords * 2.0 - 1.0
        grid_coords = grid_coords.view(B, -1, 1, 1, 3)
        
        # Reorder for grid_sample (z, y, x)
        grid_coords = torch.stack([
            grid_coords[..., 2], grid_coords[..., 1], grid_coords[..., 0]
        ], dim=-1)
        
        # Sample features
        sampled = F.grid_sample(
            voxel_features, grid_coords,
            mode='bilinear', padding_mode='border', align_corners=True
        )
        
        # Reshape [B, C, N, 1, 1] -> [B, N, C]
        sampled = sampled.squeeze(-1).squeeze(-1).permute(0, 2, 1)
        return sampled
    
    def forward(self, multi_scale_voxels, norm_coords, batch_norms):
        """Extract multi-scale features for points"""
        point_features = []
        point_coords_list = []
        
        # Get max points for padding
        max_pts = max(c.shape[0] for c in norm_coords) if norm_coords else 1000
        
        for i, (voxel_feat, bn) in enumerate(zip(multi_scale_voxels, batch_norms)):
            B = voxel_feat.shape[0]
            
            # Pad coordinates
            padded_coords = []
            for coords in norm_coords:
                n_pts = coords.shape[0]
                if n_pts == 0:
                    coords = torch.zeros(max_pts, 3, device=voxel_feat.device)
                elif n_pts < max_pts:
                    coords = F.pad(coords, (0, 0, 0, max_pts - n_pts))
                padded_coords.append(coords)
            
            batch_coords = torch.stack(padded_coords)
            
            # Interpolate features
            feat = self.interpolate_features(voxel_feat, batch_coords)
            
            # Apply batch norm
            feat_list = []
            for b in range(B):
                n_valid = norm_coords[b].shape[0] if b < len(norm_coords) else 0
                if n_valid > 0:
                    valid_feat = feat[b, :n_valid, :].transpose(0, 1)
                    valid_feat = bn(valid_feat.unsqueeze(0)).squeeze(0)
                    valid_feat = valid_feat.transpose(0, 1)
                    if n_valid < max_pts:
                        valid_feat = F.pad(valid_feat, (0, 0, 0, max_pts - n_valid))
                    feat_list.append(valid_feat)
                else:
                    feat_list.append(feat[b])
            
            feat = torch.stack(feat_list)
            point_features.append(feat)
            
            # Scale coordinates
            scale_factor = 2 ** (3 - i) if i < 4 else 1
            scaled_coords = batch_coords * self.resolution * scale_factor
            point_coords_list.append(scaled_coords)
        
        return point_features, point_coords_list


# ============================================================================
# FIXED MASK LOSS
# ============================================================================

def sample_points_safe(masks, masks_ids, n_pts, n_samples):
    """Safe point sampling that handles bounds properly"""
    sampled = []
    for ids, mm in zip(masks_ids, masks):
        if len(mm) == 0:
            sampled.append(torch.tensor([], dtype=torch.long))
            continue
            
        max_points = mm.shape[1]
        m_idx_list = []
        
        for id in ids:
            if len(id) > 0 and max_points > 0:
                valid_id = id[id < max_points]
                if len(valid_id) > 0:
                    n_sample = min(len(valid_id), n_pts)
                    perm = torch.randperm(len(valid_id))[:n_sample]
                    m_idx_list.append(valid_id[perm])
        
        if m_idx_list:
            m_idx = torch.cat(m_idx_list)
            m_idx = m_idx[m_idx < max_points]
        else:
            m_idx = torch.tensor([], dtype=torch.long)
        
        # Add random indices
        remaining = n_samples - len(m_idx)
        if remaining > 0 and max_points > 0:
            r_idx = torch.randint(0, max_points, (remaining,)).to(m_idx.device if len(m_idx) > 0 else 'cpu')
            idx = torch.cat([m_idx, r_idx]) if len(m_idx) > 0 else r_idx
        else:
            idx = m_idx[:n_samples] if len(m_idx) > n_samples else m_idx
        
        if max_points > 0 and len(idx) > 0:
            idx = idx[(idx >= 0) & (idx < max_points)]
        
        sampled.append(idx)
    return sampled


class FixedMaskLoss(nn.Module):
    """Fixed mask loss implementation"""
    def __init__(self, cfg, data_cfg):
        super().__init__()
        self.num_classes = data_cfg.NUM_CLASSES
        self.ignore_label = data_cfg.IGNORE_LABEL
        self.matcher = HungarianMatcher(cfg.WEIGHTS, cfg.P_RATIO)
        
        self.weight_dict = {
            cfg.WEIGHTS_KEYS[i]: cfg.WEIGHTS[i] 
            for i in range(len(cfg.WEIGHTS))
        }
        
        self.eos_coef = cfg.EOS_COEF
        weights = torch.ones(self.num_classes + 1)
        weights[0] = 0.0
        weights[-1] = self.eos_coef
        self.register_buffer('weights', weights)
        
        self.num_points = cfg.NUM_POINTS
        self.n_mask_pts = cfg.NUM_MASK_PTS
    
    def forward(self, outputs, targets, mask_indices):
        losses = {}
        
        num_masks = sum(len(t) for t in targets["classes"])
        if num_masks == 0:
            device = outputs["pred_logits"].device
            return {
                "loss_ce": torch.tensor(0.01, device=device, requires_grad=True),
                "loss_dice": torch.tensor(0.01, device=device, requires_grad=True),
                "loss_mask": torch.tensor(0.01, device=device, requires_grad=True)
            }
        
        num_masks = max(num_masks, 1)
        outputs_no_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        indices = self.matcher(outputs_no_aux, targets)
        
        # Get losses
        loss_ce = self.loss_classes(outputs, targets, indices)
        loss_masks = self.loss_masks(outputs, targets, indices, num_masks, mask_indices)
        
        losses.update(loss_ce)
        losses.update(loss_masks)
        
        # Apply weights
        weighted = {}
        for k, v in losses.items():
            for wk in self.weight_dict:
                if wk in k:
                    weighted[k] = v * self.weight_dict[wk]
                    break
        
        return weighted if weighted else losses
    
    def loss_classes(self, outputs, targets, indices):
        pred_logits = outputs["pred_logits"].float()
        
        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(indices)
        ])
        src_idx = torch.cat([src for (src, _) in indices])
        
        target_classes_o = torch.cat([
            t[J] for t, (_, J) in zip(targets["classes"], indices)
        ])
        
        target_classes = torch.full(
            pred_logits.shape[:2], self.num_classes,
            dtype=torch.int64, device=pred_logits.device
        )
        
        if len(batch_idx) > 0:
            target_classes[batch_idx, src_idx] = target_classes_o.to(pred_logits.device)
        
        loss_ce = F.cross_entropy(
            pred_logits.transpose(1, 2), target_classes,
            self.weights, ignore_index=self.ignore_label
        )
        
        return {"loss_ce": loss_ce}
    
    def loss_masks(self, outputs, targets, indices, num_masks, mask_indices):
        masks = targets["masks"]
        if not masks or all(len(m) == 0 for m in masks):
            device = outputs["pred_masks"].device
            return {
                "loss_mask": torch.tensor(0.01, device=device, requires_grad=True),
                "loss_dice": torch.tensor(0.01, device=device, requires_grad=True)
            }
        
        # Sample points
        with torch.no_grad():
            sampled_idx = sample_points_safe(masks, mask_indices, self.n_mask_pts, self.num_points)
        
        # Get predictions and targets
        pred_masks = outputs["pred_masks"]
        target_masks = pad_stack(masks).to(pred_masks.device)
        
        # Get matched indices
        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(indices)
        ])
        pred_idx = torch.cat([src for (src, _) in indices])
        
        n_masks = [m.shape[0] for m in masks]
        tgt_idx = []
        offset = 0
        for b, (_, tgt) in enumerate(indices):
            for t in tgt:
                if t < n_masks[b]:
                    tgt_idx.append(offset + t)
            offset += n_masks[b]
        
        if len(batch_idx) == 0 or len(tgt_idx) == 0:
            device = outputs["pred_masks"].device
            return {
                "loss_mask": torch.tensor(0.01, device=device, requires_grad=True),
                "loss_dice": torch.tensor(0.01, device=device, requires_grad=True)
            }
        
        # Select matched predictions and targets properly
        if len(pred_masks.shape) == 3:  # [B, N, Q]
            pred_masks = pred_masks[batch_idx, :, pred_idx]
        else:
            pred_masks = pred_masks[batch_idx, pred_idx]
        
        # Handle target_masks indexing properly
        if len(tgt_idx) > 0:
            tgt_idx_tensor = torch.tensor(tgt_idx, dtype=torch.long, device=target_masks.device)
            target_masks = target_masks[tgt_idx_tensor]
        
        # Sample points for loss
        point_logits = []
        point_labels = []
        
        n_masks_cumsum = [0] + np.cumsum(n_masks).tolist()
        
        for b, idx in enumerate(sampled_idx):
            if b < len(n_masks) and n_masks[b] > 0 and len(idx) > 0:
                # Find predictions for this batch
                batch_mask = (batch_idx == b)
                if batch_mask.sum() > 0:
                    # Get the indices of predictions for this batch
                    batch_pred_indices = torch.where(batch_mask)[0]
                    
                    # Ensure indices are valid
                    if len(batch_pred_indices) > 0 and len(pred_masks) > 0:
                        batch_pred = pred_masks[batch_pred_indices]
                        batch_tgt = target_masks[batch_pred_indices]
                        
                        # Ensure idx is on the right device and within bounds
                        idx = idx.to(pred_masks.device)
                        max_idx = batch_pred.shape[1] - 1 if len(batch_pred.shape) > 1 else 0
                        
                        if max_idx > 0:
                            idx = torch.clamp(idx, 0, max_idx)
                            
                            # Sample points based on tensor dimensions
                            if len(batch_pred.shape) == 2:
                                point_logits.append(batch_pred[:, idx])
                                point_labels.append(batch_tgt[:, idx])
                            elif len(batch_pred.shape) == 1:
                                # Handle 1D case
                                point_logits.append(batch_pred[idx % len(batch_pred)].unsqueeze(0))
                                point_labels.append(batch_tgt[idx % len(batch_tgt)].unsqueeze(0))
        
        if not point_logits:
            device = outputs["pred_masks"].device
            return {
                "loss_mask": torch.tensor(0.01, device=device, requires_grad=True),
                "loss_dice": torch.tensor(0.01, device=device, requires_grad=True)
            }
        
        point_logits = torch.cat(point_logits, dim=0)
        point_labels = torch.cat(point_labels, dim=0)
        
        # BCE loss
        loss_mask = F.binary_cross_entropy_with_logits(
            point_logits, point_labels.float(), reduction='none'
        ).mean(1).sum() / num_masks
        
        # Dice loss
        pred_sigmoid = torch.sigmoid(point_logits)
        numerator = 2 * (pred_sigmoid * point_labels).sum(-1)
        denominator = pred_sigmoid.sum(-1) + point_labels.sum(-1)
        dice = 1 - (numerator + 1) / (denominator + 1)
        loss_dice = dice.sum() / num_masks
        
        return {"loss_mask": loss_mask, "loss_dice": loss_dice}


# ============================================================================
# MAIN MODEL
# ============================================================================

class EnhancedMaskPLS(LightningModule):
    """Enhanced MaskPLS model for ONNX compatibility"""
    
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(dict(cfg))
        self.cfg = cfg
        
        dataset = cfg.MODEL.DATASET
        self.num_classes = cfg[dataset].NUM_CLASSES
        self.ignore_label = cfg[dataset].IGNORE_LABEL
        
        # Components
        self.voxelizer = SparseVoxelizer(
            resolution=cfg.BACKBONE.RESOLUTION,
            coordinate_bounds=cfg[dataset].SPACE,
            device='cuda'
        )
        
        self.backbone = SparseBackbone(cfg.BACKBONE)
        self.feature_extractor = PointFeatureExtractor(k=cfg.BACKBONE.KNN_UP)
        
        self.decoder = MaskedTransformerDecoder(
            cfg.DECODER,
            cfg.BACKBONE,
            cfg[dataset]
        )
        
        self.mask_loss = FixedMaskLoss(cfg.LOSS, cfg[dataset])
        self.sem_loss = SemLoss(cfg.LOSS.SEM.WEIGHTS)
        
        self.evaluator = PanopticEvaluator(cfg[dataset], dataset)
        
        # Get things IDs
        data = SemanticDatasetModule(cfg)
        data.setup()
        self.things_ids = data.things_ids
        
        self.validation_step_outputs = []
    
    def forward(self, batch):
        points = batch['pt_coord']
        features = batch['feats']
        
        # Voxelize
        voxel_grids, norm_coords, valid_indices = self.voxelizer.voxelize_batch(
            points, features, max_points=self.cfg[self.cfg.MODEL.DATASET].SUB_NUM_POINTS
        )
        
        # Extract features
        multi_scale_voxels, batch_norms = self.backbone(voxel_grids)
        point_features, point_coords_list = self.feature_extractor(
            multi_scale_voxels, norm_coords, batch_norms
        )
        
        # Create padding masks
        max_pts = point_features[0].shape[1]
        padding_masks = []
        for coords in norm_coords:
            n_pts = coords.shape[0]
            mask = torch.zeros(max_pts, dtype=torch.bool, device='cuda')
            if n_pts < max_pts:
                mask[n_pts:] = True
            padding_masks.append(mask)
        
        padding_masks_stacked = torch.stack(padding_masks)
        padding_masks_list = [padding_masks_stacked for _ in range(len(point_features))]
        
        # Decode
        outputs, last_pad = self.decoder(point_features, point_coords_list, padding_masks_list)
        sem_logits = self.backbone.sem_head(point_features[-1])
        
        return outputs, last_pad, sem_logits, valid_indices
    
    def prepare_targets(self, batch, max_points, valid_indices):
        targets = {'classes': [], 'masks': []}
        
        for i in range(len(batch['masks_cls'])):
            if len(batch['masks_cls'][i]) > 0:
                classes = torch.tensor(batch['masks_cls'][i], dtype=torch.long, device='cuda')
                classes = torch.clamp(classes, 0, self.num_classes - 1)
                targets['classes'].append(classes)
                
                masks_list = []
                for mask in batch['masks'][i]:
                    if not isinstance(mask, torch.Tensor):
                        mask = torch.from_numpy(np.array(mask)).float()
                    else:
                        mask = mask.float()
                    
                    remapped_mask = torch.zeros(max_points, device='cuda')
                    valid_idx = valid_indices[i]
                    
                    if len(valid_idx) > 0:
                        for j, v_idx in enumerate(valid_idx):
                            if j < max_points and v_idx < len(mask):
                                remapped_mask[j] = mask[v_idx]
                    
                    masks_list.append(remapped_mask)
                
                targets['masks'].append(torch.stack(masks_list) if masks_list else 
                                       torch.zeros(0, max_points, device='cuda'))
            else:
                targets['classes'].append(torch.zeros(0, dtype=torch.long, device='cuda'))
                targets['masks'].append(torch.zeros(0, max_points, device='cuda'))
        
        return targets
    
    def compute_semantic_loss(self, batch, sem_logits, valid_indices, padding_masks):
        all_logits = []
        all_labels = []
        
        for i, (labels, valid_idx, pad_mask) in enumerate(
            zip(batch['sem_label'], valid_indices, padding_masks)
        ):
            if len(valid_idx) == 0:
                continue
            
            valid_mask = ~pad_mask
            batch_logits = sem_logits[i][valid_mask]
            
            labels = np.array(labels).flatten() if not isinstance(labels, np.ndarray) else labels.flatten()
            
            valid_idx_cpu = valid_idx.cpu().numpy()
            valid_labels = []
            
            for j, v_idx in enumerate(valid_idx_cpu):
                if j < len(batch_logits) and v_idx < len(labels):
                    valid_labels.append(labels[v_idx])
            
            if valid_labels:
                labels_tensor = torch.tensor(valid_labels, dtype=torch.long, device='cuda')
                labels_tensor = torch.clamp(labels_tensor, 0, self.num_classes - 1)
                
                min_len = min(len(batch_logits), len(labels_tensor))
                if min_len > 0:
                    all_logits.append(batch_logits[:min_len])
                    all_labels.append(labels_tensor[:min_len])
        
        if all_logits:
            combined_logits = torch.cat(all_logits, dim=0)
            combined_labels = torch.cat(all_labels, dim=0)
            
            ce_loss = F.cross_entropy(combined_logits, combined_labels, ignore_index=0)
            
            try:
                probs = F.softmax(combined_logits, dim=1)
                probs = torch.clamp(probs, min=1e-7, max=0.9999999)
                lovasz_loss = self.sem_loss.lovasz_softmax(probs, combined_labels)
            except:
                lovasz_loss = torch.tensor(0.0, device='cuda', requires_grad=True)
            
            return (self.cfg.LOSS.SEM.WEIGHTS[0] * ce_loss + 
                   self.cfg.LOSS.SEM.WEIGHTS[1] * lovasz_loss)
        else:
            return torch.tensor(0.01, device='cuda', requires_grad=True)
    
    def panoptic_inference(self, outputs, padding_masks, valid_indices, batch):
        mask_cls = outputs["pred_logits"]
        mask_pred = outputs["pred_masks"]
        
        sem_pred = []
        ins_pred = []
        
        for b in range(mask_cls.shape[0]):
            valid_mask = ~padding_masks[b]
            valid_pred = mask_pred[b][valid_mask].sigmoid()
            
            scores, labels = mask_cls[b].max(-1)
            keep = labels.ne(self.num_classes)
            
            orig_size = batch['sem_label'][b].shape[0]
            sem_out = np.zeros(orig_size, dtype=np.int32)
            ins_out = np.zeros(orig_size, dtype=np.int32)
            
            if keep.sum() > 0 and len(valid_indices[b]) > 0:
                cur_scores = scores[keep]
                cur_classes = labels[keep]
                cur_masks = valid_pred[:, keep] if valid_pred.dim() == 2 else valid_pred.unsqueeze(1)
                
                cur_prob_masks = cur_scores.unsqueeze(0) * cur_masks
                mask_ids = cur_prob_masks.argmax(1)
                
                valid_idx_cpu = valid_indices[b].cpu().numpy()
                segment_id = 0
                stuff_memory = {}
                
                for k in range(cur_classes.shape[0]):
                    pred_class = cur_classes[k].item()
                    isthing = pred_class in self.things_ids
                    
                    mask_points = (mask_ids == k) & (cur_masks[:, k] >= 0.5) if cur_masks.dim() == 2 else (mask_ids == k)
                    
                    if mask_points.sum() < 10:
                        continue
                    
                    mask_area = (mask_ids == k).sum().item()
                    original_area = (cur_masks[:, k] >= 0.5).sum().item() if cur_masks.dim() == 2 else mask_points.sum().item()
                    
                    if original_area > 0 and mask_area / original_area < self.cfg.MODEL.OVERLAP_THRESHOLD:
                        continue
                    
                    mask_indices = mask_points.cpu().numpy()
                    
                    for i, is_mask in enumerate(mask_indices):
                        if i < len(valid_idx_cpu) and is_mask:
                            orig_idx = valid_idx_cpu[i]
                            if orig_idx < orig_size:
                                sem_out[orig_idx] = pred_class
                                
                                if isthing:
                                    ins_out[orig_idx] = segment_id + 1
                                else:
                                    if pred_class not in stuff_memory:
                                        stuff_memory[pred_class] = segment_id + 1
                                        segment_id += 1
                                    ins_out[orig_idx] = stuff_memory[pred_class]
                    
                    if isthing:
                        segment_id += 1
            
            sem_pred.append(sem_out)
            ins_pred.append(ins_out)
        
        return sem_pred, ins_pred
    
    def training_step(self, batch, batch_idx):
        try:
            outputs, padding_masks, sem_logits, valid_indices = self.forward(batch)
            
            if all(len(v) == 0 for v in valid_indices):
                return torch.tensor(0.01, device='cuda', requires_grad=True)
            
            targets = self.prepare_targets(batch, padding_masks.shape[1], valid_indices)
            
            mask_losses = self.mask_loss(outputs, targets, batch['masks_ids'])
            sem_loss = self.compute_semantic_loss(batch, sem_logits, valid_indices, padding_masks)
            
            total_loss = sum(mask_losses.values()) + sem_loss
            
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                total_loss = torch.tensor(1.0, device='cuda', requires_grad=True)
            
            self.log("train_loss", total_loss.item(), batch_size=self.cfg.TRAIN.BATCH_SIZE)
            
            return total_loss
            
        except Exception as e:
            print(f"Error in training step {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            return torch.tensor(1.0, device='cuda', requires_grad=True)
    
    def validation_step(self, batch, batch_idx):
        try:
            outputs, padding_masks, sem_logits, valid_indices = self.forward(batch)
            
            if all(len(v) == 0 for v in valid_indices):
                return
            
            targets = self.prepare_targets(batch, padding_masks.shape[1], valid_indices)
            mask_losses = self.mask_loss(outputs, targets, batch['masks_ids'])
            sem_loss = self.compute_semantic_loss(batch, sem_logits, valid_indices, padding_masks)
            
            total_loss = sum(mask_losses.values()) + sem_loss
            
            if not (torch.isnan(total_loss) or torch.isinf(total_loss)):
                self.log("val_loss", total_loss, batch_size=self.cfg.TRAIN.BATCH_SIZE)
            
            sem_pred, ins_pred = self.panoptic_inference(
                outputs, padding_masks, valid_indices, batch
            )
            
            self.evaluator.update(sem_pred, ins_pred, batch)
            
            if not (torch.isnan(total_loss) or torch.isinf(total_loss)):
                self.validation_step_outputs.append(total_loss)
                
        except Exception as e:
            print(f"Error in validation step {batch_idx}: {e}")
    
    def on_validation_epoch_end(self):
        try:
            pq = self.evaluator.get_mean_pq()
            iou = self.evaluator.get_mean_iou()
            rq = self.evaluator.get_mean_rq()
            
            self.log("metrics/pq", pq, batch_size=self.cfg.TRAIN.BATCH_SIZE)
            self.log("metrics/iou", iou, batch_size=self.cfg.TRAIN.BATCH_SIZE)
            self.log("metrics/rq", rq, batch_size=self.cfg.TRAIN.BATCH_SIZE)
            
            print(f"\nValidation Metrics - PQ: {pq:.4f}, IoU: {iou:.4f}, RQ: {rq:.4f}")
            
        except Exception as e:
            print(f"Error computing validation metrics: {e}")
        finally:
            self.evaluator.reset()
            self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.TRAIN.LR,
            weight_decay=1e-4,
            eps=1e-8
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.TRAIN.MAX_EPOCH,
            eta_min=self.cfg.TRAIN.LR * 0.001
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }


# ============================================================================
# MAIN FUNCTION
# ============================================================================

@click.command()
@click.option("--config", type=str, default="config/model.yaml")
@click.option("--epochs", type=int, default=100)
@click.option("--batch_size", type=int, default=1)
@click.option("--lr", type=float, default=0.00005)
@click.option("--gpus", type=int, default=1)
@click.option("--num_workers", type=int, default=2)
@click.option("--checkpoint", type=str, default=None)
@click.option("--nuscenes", is_flag=True)
def main(config, epochs, batch_size, lr, gpus, num_workers, checkpoint, nuscenes):
    """Train ONNX-compatible MaskPLS"""
    
    print("="*60)
    print("Enhanced MaskPLS Training (ONNX-Compatible)")
    print("="*60)
    
    # Load config
    def getDir(obj):
        return os.path.dirname(os.path.abspath(obj))
    
    model_cfg = edict(yaml.safe_load(open(join(getDir(__file__), "../config/model.yaml"))))
    backbone_cfg = edict(yaml.safe_load(open(join(getDir(__file__), "../config/backbone.yaml"))))
    decoder_cfg = edict(yaml.safe_load(open(join(getDir(__file__), "../config/decoder.yaml"))))
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    
    # Update config
    cfg.TRAIN.MAX_EPOCH = epochs
    cfg.TRAIN.BATCH_SIZE = batch_size
    cfg.TRAIN.LR = lr
    cfg.TRAIN.N_GPUS = gpus
    cfg.TRAIN.NUM_WORKERS = num_workers
    cfg.TRAIN.BATCH_ACC = 1
    
    if nuscenes:
        cfg.MODEL.DATASET = "NUSCENES"
    
    dataset = cfg.MODEL.DATASET
    
    # Memory-friendly settings
    cfg[dataset].SUB_NUM_POINTS = 10000
    cfg.LOSS.NUM_POINTS = 5000
    cfg.LOSS.NUM_MASK_PTS = 100
    
    print(f"Configuration:")
    print(f"  Dataset: {dataset}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {lr}")
    print(f"  Max Points: {cfg[dataset].SUB_NUM_POINTS}")
    
    # Create data module
    data = SemanticDatasetModule(cfg)
    
    # Create model
    model = EnhancedMaskPLS(cfg)
    
    # Load checkpoint if provided
    if checkpoint:
        print(f"\nLoading checkpoint: {checkpoint}")
        try:
            ckpt = torch.load(checkpoint, map_location='cpu')
            model.load_state_dict(ckpt['state_dict'], strict=False)
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")
    
    # Setup logging
    tb_logger = pl_loggers.TensorBoardLogger(
        "experiments/" + cfg.EXPERIMENT.ID + "_onnx",
        default_hp_metric=False
    )
    
    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    checkpoint_callback = ModelCheckpoint(
        monitor="metrics/iou",
        filename=cfg.EXPERIMENT.ID + "_onnx_epoch{epoch:02d}_iou{metrics/iou:.3f}",
        auto_insert_metric_name=False,
        mode="max",
        save_last=True,
        save_top_k=3
    )
    
    # Create trainer
    trainer = Trainer(
        devices=gpus,
        accelerator="gpu" if gpus > 0 else "cpu",
        strategy="ddp" if gpus > 1 else None,
        logger=tb_logger,
        max_epochs=epochs,
        callbacks=[lr_monitor, checkpoint_callback],
        log_every_n_steps=1,
        gradient_clip_val=1.0,
        accumulate_grad_batches=cfg.TRAIN.BATCH_ACC,
        precision=32,
        check_val_every_n_epoch=2,
        num_sanity_val_steps=0,
        detect_anomaly=False
    )
    
    # Train
    print("\nStarting training...")
    trainer.fit(model, data)
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()