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
from pytorch_lightning import Trainer, seed_everything
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
        
        D, H, W = int(grid_size[0].item()), int(grid_size[1].item()), int(grid_size[2].item())
        
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
            nonzero = counts > 0
            dense_grid[:, nonzero] = dense_grid[:, nonzero] / counts[nonzero]
            
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
        
        # Downsample stages
        self.stage1 = nn.Sequential(
            nn.Conv3d(cs[0], cs[1], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm3d(cs[1]),
            nn.ReLU(inplace=True),
            ResidualBlock3D(cs[1], cs[1]),
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
        
        # Heads
        self.mask_feat_proj = nn.Conv1d(cs[8], 256, kernel_size=1)
        self.sem_head = nn.Conv1d(cs[8], 20, kernel_size=1)  # dummy replaced later
    
    def forward(self, x):
        B, C, D, H, W = x.shape
        x0 = self.stem(x)
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
        
        # Project to per-point features
        B, C, D, H, W = y4.shape
        y4_flat = y4.view(B, C, -1)  # [B, C, P]
        
        return [y1.view(B, y1.shape[1], -1).permute(0, 2, 1),
                y2.view(B, y2.shape[1], -1).permute(0, 2, 1),
                y3.view(B, y3.shape[1], -1).permute(0, 2, 1),
                y4_flat.permute(0, 2, 1)], [y1, y2, y3, y4]


# ============================================================================
# POINT FEATURE EXTRACTOR
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
        
        # Voxel grid [B,C,D,H,W] -> [B,C,D,H,W]
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
            
            # Sample features
            feat = self.interpolate_features(voxel_feat, batch_coords)
            
            # Normalize features
            feat_list = []
            for b in range(B):
                if hasattr(bn, 'weight'):
                    feat_list.append(F.layer_norm(feat[b], feat[b].shape[-1:]))
                else:
                    feat_list.append(feat[b])
            
            feat = torch.stack(feat_list)
            point_features.append(feat)
            
            # (coords returned here are ignored; world coords are built in the model forward)
            point_coords_list.append(batch_coords)
        
        return point_features, point_coords_list


# ============================================================================
# LOSS UTILS
# ============================================================================
def sample_points_safe(masks, masks_ids, n_pts, n_samples):
    """Safe point sampling that handles bounds properly (no random top-up)."""
    sampled = []
    for ids, mm in zip(masks_ids, masks):
        if len(mm) == 0:
            sampled.append(torch.tensor([], dtype=torch.long))
            continue
        max_points = mm.shape[1]
        m_idx_list = []
        for id in ids:
            if isinstance(id, torch.Tensor):
                id = id
            else:
                id = torch.tensor(id, dtype=torch.long)
            if id.numel() > 0 and max_points > 0:
                valid_id = id[id >= 0]
                valid_id = valid_id[valid_id < max_points]
                if valid_id.numel() > 0:
                    n_sample = min(valid_id.numel(), n_pts)
                    perm = torch.randperm(valid_id.numel(), device=valid_id.device)[:n_sample]
                    m_idx_list.append(valid_id[perm])
        if m_idx_list:
            m_idx = torch.cat(m_idx_list)
            # bound check
            m_idx = m_idx[(m_idx >= 0) & (m_idx < max_points)]
            # limit to at most n_samples
            if m_idx.numel() > n_samples:
                perm = torch.randperm(m_idx.numel(), device=m_idx.device)[:n_samples]
                m_idx = m_idx[perm]
            sampled.append(m_idx)
        else:
            sampled.append(torch.tensor([], dtype=torch.long))
    return sampled


def remap_mask_ids(masks_ids, valid_indices):
    """
    Remap original point indices in masks_ids to the compact [0..max_points) indexing
    used after voxelization/subsampling, per batch element.
    """
    remapped = []
    inv_maps = []
    # Build per-batch inverse maps: orig_idx -> compact_idx
    for valid in valid_indices:
        inv = {int(orig.item()): i for i, orig in enumerate(valid.cpu())}
        inv_maps.append(inv)
    for b, ids_per_inst in enumerate(masks_ids):
        inst_list = []
        for ids in ids_per_inst:
            if isinstance(ids, torch.Tensor):
                ids = ids.cpu().tolist()
            new_ids = [inv_maps[b].get(int(x), -1) for x in ids]
            new_ids = [i for i in new_ids if i >= 0]
            inst_list.append(torch.tensor(new_ids, device='cuda', dtype=torch.long))
        remapped.append(inst_list)
    return remapped


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

    def _get_pred_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices, n_masks):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        # From [B,id] to [id] of stacked masks
        cont_id = torch.cat([torch.arange(n, device=batch_idx.device) for n in n_masks])
        b_id = torch.stack((batch_idx, cont_id), axis=1)
        map_m = torch.zeros((int(torch.max(batch_idx).item()) + 1, max(n_masks)), device=batch_idx.device)
        for i in range(len(b_id)):
            map_m[int(b_id[i, 0].item()), int(b_id[i, 1].item())] = i
        stack_ids = [int(map_m[int(batch_idx[i].item()), int(tgt_idx[i].item())].item()) for i in range(len(batch_idx))]
        return stack_ids
    
    def forward(self, outputs, targets, mask_indices):
        losses = {}
        
        # Number of masks (for normalization)
        num_masks = sum(len(t) for t in targets["classes"])
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        num_masks = torch.clamp(num_masks, min=1.0).item()
        
        # Compute matching on logits/masks (no aux)
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
            pred_logits.transpose(1, 2),
            target_classes,
            self.weights.to(pred_logits),
            ignore_index=self.ignore_label,
        )
        return {"loss_ce": loss_ce}
    
    def loss_masks(self, outputs, targets, indices, num_masks, mask_indices):
        masks = [t for t in targets["masks"]]
        
        # Sample points (per-batch indices)
        with torch.no_grad():
            sampled_idx = sample_points_safe(masks, mask_indices, self.n_mask_pts, self.num_points)

        # Stack masks and select matched predictions/targets
        pred_masks_all = outputs["pred_masks"]  # [B,P,Q]
        masks_stacked = pad_stack(masks)        # [sum_gt,P]

        # Get permutation indices
        batch_idx, src_idx = self._get_pred_permutation_idx(indices)
        n_masks = [m.shape[0] for m in masks]
        if sum(n_masks) == 0 or len(src_idx) == 0:
            device = pred_masks_all.device
            return {
                "loss_mask": torch.tensor(0.01, device=device, requires_grad=True),
                "loss_dice": torch.tensor(0.01, device=device, requires_grad=True)
            }

        pred_masks = pred_masks_all[batch_idx, :, src_idx]  # [M,P]
        tgt_idx = self._get_tgt_permutation_idx(indices, n_masks)
        target_masks = masks_stacked.to(pred_masks.device)[tgt_idx]  # [M,P]

        # Build cumulative offsets for each batch's matched masks
        n_masks_insert0 = [0] + n_masks
        nm = torch.cumsum(torch.tensor(n_masks_insert0, device=pred_masks.device), 0)

        # Gather logits/labels at sampled points per batch
        point_logits = []
        point_labels = []
        for b, p in enumerate(sampled_idx):
            if n_masks[b] == 0 or p.numel() == 0:
                continue
            sl = slice(nm[b].item(), nm[b+1].item())
            point_logits.append(pred_masks[sl][:, p.to(pred_masks.device)])
            point_labels.append(target_masks[sl][:, p.to(pred_masks.device)])

        if len(point_logits) == 0:
            device = pred_masks_all.device
            return {
                "loss_mask": torch.tensor(0.01, device=device, requires_grad=True),
                "loss_dice": torch.tensor(0.01, device=device, requires_grad=True)
            }

        point_logits = torch.cat(point_logits, dim=0)
        point_labels = torch.cat(point_labels, dim=0)

        # Compute BCE + Dice on sampled points
        inputs = point_logits
        targets = point_labels
        inputs_sigmoid = inputs.sigmoid()
        numerator = 2 * (inputs_sigmoid * targets).sum(-1)
        denominator = inputs_sigmoid.sum(-1) + targets.sum(-1)
        loss_dice = (1 - (numerator + 1) / (denominator + 1)).sum() / num_masks
        loss_mask = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none').mean(1).sum() / num_masks

        return {"loss_mask": loss_mask, "loss_dice": loss_dice}


# ============================================================================
# MODEL
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
        
        # Build world-space coordinates (same across levels)
        # Convert normalized coords [0,1] back to world using voxelizer bounds
        bounds_min = self.voxelizer.bounds_min
        bounds_max = self.voxelizer.bounds_max
        world_coords = []
        for coords in norm_coords:
            if coords.numel() == 0:
                world_coords.append(torch.zeros(0, 3, device='cuda', dtype=torch.float32))
            else:
                world_coords.append(coords * (bounds_max - bounds_min) + bounds_min)
        max_pts = max(c.shape[0] for c in world_coords) if world_coords else 0
        padded_world = []
        for c in world_coords:
            n = c.shape[0]
            if n == 0 and max_pts > 0:
                padded_world.append(torch.zeros(max_pts, 3, device='cuda', dtype=torch.float32))
            elif n < max_pts:
                padded_world.append(F.pad(c, (0,0,0, max_pts - n)))
            else:
                padded_world.append(c)
        point_coords_batch = torch.stack(padded_world) if max_pts > 0 else torch.zeros(len(world_coords), 0, 3, device='cuda')
        # Use identical coords for each feature level as in original MaskPLS
        point_coords_list = [point_coords_batch for _ in range(len(point_features))]
        
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
        sem_logits = self.backbone.mask_feat_proj(point_features[-1].permute(0, 2, 1))
        
        return outputs, sem_logits, padding_masks_stacked, valid_indices
    
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
                
                # Cross entropy
                ce = F.cross_entropy(batch_logits, labels_tensor, ignore_index=self.ignore_label)
                all_logits.append(ce)
        
        if all_logits:
            return torch.stack(all_logits).mean()
        return torch.tensor(0.0, device='cuda', requires_grad=True)
    
    def training_step(self, batch, batch_idx):
        try:
            outputs, sem_logits, padding_masks, valid_indices = self.forward(batch)
            
            # Targets
            targets = self.prepare_targets(batch, padding_masks.shape[1], valid_indices)
            
            remapped_ids = remap_mask_ids(batch['masks_ids'], valid_indices)
            mask_losses = self.mask_loss(outputs, targets, remapped_ids)
            sem_loss = self.compute_semantic_loss(batch, sem_logits, valid_indices, padding_masks)
            
            total_loss = sum(mask_losses.values()) + sem_loss
            
            # Logging
            log_dict = {f"train/{k}": v for k, v in mask_losses.items()}
            log_dict["train/sem_ce"] = sem_loss
            self.log_dict(log_dict, prog_bar=True, on_step=True, batch_size=self.cfg.TRAIN.BATCH_SIZE)
            return total_loss
        except Exception as e:
            print(f"Error in training step {batch_idx}: {e}")
            return torch.tensor(0.0, device='cuda', requires_grad=True)
    
    def validation_step(self, batch, batch_idx):
        try:
            outputs, sem_logits, padding_masks, valid_indices = self.forward(batch)
            
            # Targets
            targets = self.prepare_targets(batch, padding_masks.shape[1], valid_indices)
            remapped_ids = remap_mask_ids(batch['masks_ids'], valid_indices)
            mask_losses = self.mask_loss(outputs, targets, remapped_ids)
            sem_loss = self.compute_semantic_loss(batch, sem_logits, valid_indices, padding_masks)
            
            total_loss = sum(mask_losses.values()) + sem_loss
            
            # Logging
            log_dict = {f"val/{k}": v for k, v in mask_losses.items()}
            log_dict["val/sem_ce"] = sem_loss
            self.log_dict(log_dict, prog_bar=True, on_epoch=True, batch_size=self.cfg.TRAIN.BATCH_SIZE)
            
            # Panoptic evaluation
            self.panoptic_postprocess(batch, outputs, padding_masks, valid_indices)
            self.validation_step_outputs.append(total_loss.detach())
            
            return total_loss
        except Exception as e:
            print(f"Error in validation step {batch_idx}: {e}")
    
    # ----- Panoptic postprocessing (unchanged from your script) -----
    def panoptic_postprocess(self, batch, outputs, padding_masks, valid_indices):
        try:
            pred_logits = outputs['pred_logits']
            pred_masks = outputs['pred_masks']
            pred_probs = pred_logits.softmax(-1)[..., :-1]
            scores, labels = pred_probs.max(-1)
            mask_probs = pred_masks.sigmoid()
            
            B = pred_logits.shape[0]
            for b in range(B):
                # prepare tensors
                s = scores[b]
                l = labels[b]
                m = mask_probs[b]  # [P,Q]
                pad = padding_masks[b]  # [P]
                valid_mask = ~pad
                m = m[valid_mask]  # shrink to valid P
                s = s
                l = l
                
                # threshold and select
                keep = s > self.cfg.MODEL.CONFIDENCE_THR
                if keep.sum() == 0 or m.shape[0] == 0:
                    continue
                m = m[:, keep]
                l = l[keep]
                s = s[keep]
                
                # predicted semantic/instance maps for valid points
                mask_ids = torch.zeros(m.shape[0], dtype=torch.long)
                max_points = m.shape[0]
                cur_masks = m
                
                # CPU numpy arrays from batch
                mask_ids_np = np.array(batch['masks_ids'][b], dtype=object)
                orig_size = batch['size'][b][0]
                valid_idx_cpu = valid_indices[b].cpu().numpy() if len(valid_indices[b]) > 0 else np.array([])
                
                sem_out = np.full(orig_size, fill_value=self.ignore_label, dtype=np.int32)
                ins_out = np.zeros(orig_size, dtype=np.int32)
                segment_id = 0
                stuff_memory = {}
                
                # construct simple argmax over queries for each point
                if cur_masks.dim() == 2:
                    # take best query per point
                    best_q = (cur_masks >= 0.5).float().argmax(dim=1)
                else:
                    best_q = None
                
                for k in range(cur_masks.shape[1]):
                    pred_class = int(l[k].item())
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
                
                self.evaluator.update(sem_out, ins_out, batch['sem_label'][b], batch['ins_label'][b])
                
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
        
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.cfg.TRAIN.LR_DECAY_STEPS,
            gamma=self.cfg.TRAIN.LR_DECAY_GAMMA
        )
        
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


# ============================================================================
# CLI
# ============================================================================
@click.command()
@click.option("--config", type=str, required=True)
@click.option("--gpus", type=int, default=1)
@click.option("--epochs", type=int, default=None)
@click.option("--seed", type=int, default=42)
def main(config, gpus, epochs, seed):
    # Seed
    seed_everything(seed)
    
    # Load config
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)
    from easydict import EasyDict as edict
    cfg = edict(cfg)
    
    # Data
    data = SemanticDatasetModule(cfg)
    data.setup()
    
    # Model
    model = EnhancedMaskPLS(cfg)
    
    # Logging & Callbacks
    tb_logger = False
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(
        dirpath=join(cfg.EXPERIMENT.OUT_DIR, cfg.EXPERIMENT.ID),
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
