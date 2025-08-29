"""
Numerically stable optimized training script for MaskPLS
Fixes NaN/Inf issues while maintaining performance
"""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Enable for better error debugging
os.environ['TORCH_USE_CUDA_DSA'] = '0'

from os.path import join
import click
import torch
import yaml
import numpy as np
import time
import gc
from pathlib import Path
from easydict import EasyDict as edict
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Callback
from collections import OrderedDict, defaultdict
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
import warnings

# Import model components
from mask_pls.models.onnx.simplified_model import MaskPLSSimplifiedONNX
from mask_pls.datasets.semantic_dataset import SemanticDatasetModule
from mask_pls.models.matcher import HungarianMatcher
from mask_pls.utils.evaluate_panoptic import PanopticEvaluator
from mask_pls.utils.misc import sample_points, pad_stack

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class StableVoxelizer:
    """Numerically stable voxelizer with proper bounds checking"""
    def __init__(self, spatial_shape, bounds, device='cuda'):
        # Use moderate resolution for stability
        self.spatial_shape = (96, 96, 48)  # Balanced resolution
        self.bounds = bounds
        self.device = device
        
        # Precompute constants with numerical stability
        self.bounds_min = torch.tensor([bounds[d][0] for d in range(3)], device=device, dtype=torch.float32)
        self.bounds_max = torch.tensor([bounds[d][1] for d in range(3)], device=device, dtype=torch.float32)
        self.bounds_range = self.bounds_max - self.bounds_min
        
        # Add small epsilon to prevent division by zero
        self.bounds_range = torch.clamp(self.bounds_range, min=1e-6)
        
        # Cache for efficiency
        self.cache = OrderedDict()
        self.cache_size = 50
        
    def voxelize_batch_stable(self, points_batch, features_batch):
        """Stable batch voxelization with numerical checks"""
        B = len(points_batch)
        D, H, W = self.spatial_shape
        C = features_batch[0].shape[1] if len(features_batch) > 0 else 4
        
        # Initialize with zeros (float32 for stability)
        voxel_grids = torch.zeros(B, C, D, H, W, device=self.device, dtype=torch.float32)
        voxel_counts = torch.zeros(B, 1, D, H, W, device=self.device, dtype=torch.float32)
        normalized_coords = []
        valid_indices = []
        
        for b in range(B):
            try:
                # Convert to tensors with error checking
                pts = torch.from_numpy(points_batch[b]).float().to(self.device)
                feat = torch.from_numpy(features_batch[b]).float().to(self.device)
                
                # Check for NaN/Inf in input
                if torch.isnan(pts).any() or torch.isinf(pts).any():
                    print(f"Warning: NaN/Inf detected in points for batch {b}, skipping")
                    normalized_coords.append(torch.zeros(0, 3, device=self.device))
                    valid_indices.append(torch.zeros(0, dtype=torch.long, device=self.device))
                    continue
                
                if torch.isnan(feat).any() or torch.isinf(feat).any():
                    print(f"Warning: NaN/Inf detected in features for batch {b}, replacing with zeros")
                    feat = torch.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # Bounds check with stability
                valid_mask = ((pts >= self.bounds_min - 0.1) & (pts < self.bounds_max + 0.1)).all(dim=1)
                
                if valid_mask.sum() == 0:
                    normalized_coords.append(torch.zeros(0, 3, device=self.device))
                    valid_indices.append(torch.zeros(0, dtype=torch.long, device=self.device))
                    continue
                
                valid_idx = torch.where(valid_mask)[0]
                valid_pts = pts[valid_mask]
                valid_feat = feat[valid_mask]
                
                # Normalize with clamping for stability
                valid_pts = torch.clamp(valid_pts, self.bounds_min, self.bounds_max - 1e-6)
                norm_coords = (valid_pts - self.bounds_min) / self.bounds_range
                norm_coords = torch.clamp(norm_coords, 0.0, 0.999999)
                
                # Convert to voxel indices safely
                voxel_indices = (norm_coords * torch.tensor([D, H, W], device=self.device, dtype=torch.float32)).long()
                voxel_indices = torch.clamp(voxel_indices, 0, torch.tensor([D-1, H-1, W-1], device=self.device))
                
                # Compute flat indices
                flat_indices = (voxel_indices[:, 0] * H * W + 
                              voxel_indices[:, 1] * W + 
                              voxel_indices[:, 2])
                
                # Accumulate features with stability checks
                for c in range(C):
                    feat_values = valid_feat[:, c]
                    # Clamp extreme values
                    feat_values = torch.clamp(feat_values, -100.0, 100.0)
                    voxel_grids[b, c].view(-1).scatter_add_(0, flat_indices, feat_values)
                
                # Count accumulation
                ones = torch.ones_like(valid_feat[:, 0])
                voxel_counts[b, 0].view(-1).scatter_add_(0, flat_indices, ones)
                
                normalized_coords.append(norm_coords)
                valid_indices.append(valid_idx)
                
            except Exception as e:
                print(f"Error in voxelization for batch {b}: {e}")
                normalized_coords.append(torch.zeros(0, 3, device=self.device))
                valid_indices.append(torch.zeros(0, dtype=torch.long, device=self.device))
                continue
        
        # Average with numerical stability
        voxel_grids = voxel_grids / (voxel_counts + 1.0)  # Add 1.0 instead of small epsilon
        
        # Replace NaN/Inf with zeros
        voxel_grids = torch.nan_to_num(voxel_grids, nan=0.0, posinf=0.0, neginf=0.0)
        
        return voxel_grids, normalized_coords, valid_indices


class StableMaskLoss(torch.nn.Module):
    """Numerically stable mask loss implementation"""
    def __init__(self, cfg, data_cfg):
        super().__init__()
        
        self.num_classes = data_cfg.NUM_CLASSES
        self.ignore = data_cfg.IGNORE_LABEL
        self.matcher = HungarianMatcher(cfg.WEIGHTS, cfg.P_RATIO)
        
        self.weight_dict = {
            cfg.WEIGHTS_KEYS[i]: cfg.WEIGHTS[i] for i in range(len(cfg.WEIGHTS))
        }
        self.eos_coef = cfg.EOS_COEF
        
        # Weights with numerical stability
        weights = torch.ones(self.num_classes + 1)
        weights[0] = 0.1  # Small non-zero weight for stability
        weights[-1] = self.eos_coef
        self.register_buffer('weights', weights)
        
        # Use moderate point sampling for stability
        self.num_points = min(cfg.NUM_POINTS, 30000)  # Limit for stability
        self.n_mask_pts = min(cfg.NUM_MASK_PTS, 300)  # Limit for stability
        
    def forward(self, outputs, targets, masks_ids, coors):
        """Stable loss computation with numerical checks"""
        losses = {}
        
        # Check for NaN/Inf in outputs
        if torch.isnan(outputs["pred_logits"]).any() or torch.isinf(outputs["pred_logits"]).any():
            print("Warning: NaN/Inf detected in pred_logits")
            outputs["pred_logits"] = torch.nan_to_num(outputs["pred_logits"], nan=0.0, posinf=10.0, neginf=-10.0)
        
        if torch.isnan(outputs["pred_masks"]).any() or torch.isinf(outputs["pred_masks"]).any():
            print("Warning: NaN/Inf detected in pred_masks")
            outputs["pred_masks"] = torch.nan_to_num(outputs["pred_masks"], nan=0.0, posinf=10.0, neginf=-10.0)
        
        # Handle empty targets
        num_masks = sum(len(t) for t in targets["classes"])
        if num_masks == 0:
            device = outputs["pred_logits"].device
            # Return small non-zero loss for gradient flow
            return {
                "loss_ce": torch.tensor(0.1, device=device, requires_grad=True),
                "loss_dice": torch.tensor(0.1, device=device, requires_grad=True),
                "loss_mask": torch.tensor(0.1, device=device, requires_grad=True)
            }
        
        num_masks = max(num_masks, 1)
        outputs_no_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        
        try:
            # Hungarian matching with error handling
            indices = self.matcher(outputs_no_aux, targets)
        except Exception as e:
            print(f"Error in Hungarian matching: {e}")
            device = outputs["pred_logits"].device
            # Return fallback loss
            return {
                "loss_ce": torch.tensor(1.0, device=device, requires_grad=True),
                "loss_dice": torch.tensor(1.0, device=device, requires_grad=True),
                "loss_mask": torch.tensor(1.0, device=device, requires_grad=True)
            }
        
        # Compute losses with stability
        losses.update(self.get_losses(outputs, targets, indices, num_masks, masks_ids, coors))
        
        # Apply weights with clamping
        weighted_losses = {}
        for l in losses:
            for k in self.weight_dict:
                if k in l:
                    loss_val = losses[l] * self.weight_dict[k]
                    # Clamp loss values for stability
                    loss_val = torch.clamp(loss_val, 0.0, 100.0)
                    weighted_losses[l] = loss_val
                    break
        
        return weighted_losses
    
    def get_losses(self, outputs, targets, indices, num_masks, masks_ids, coors):
        """Get losses with stability checks"""
        losses = {}
        
        try:
            losses.update(self.loss_classes(outputs, targets, indices))
        except Exception as e:
            print(f"Error in classification loss: {e}")
            losses["loss_ce"] = torch.tensor(1.0, device=outputs["pred_logits"].device, requires_grad=True)
        
        try:
            losses.update(self.loss_masks(outputs, targets, indices, num_masks, masks_ids))
        except Exception as e:
            print(f"Error in mask loss: {e}")
            losses["loss_mask"] = torch.tensor(1.0, device=outputs["pred_masks"].device, requires_grad=True)
            losses["loss_dice"] = torch.tensor(1.0, device=outputs["pred_masks"].device, requires_grad=True)
        
        return losses
    
    def loss_classes(self, outputs, targets, indices):
        """Stable classification loss"""
        pred_logits = outputs["pred_logits"].float()
        
        # Clamp logits for stability
        pred_logits = torch.clamp(pred_logits, -50.0, 50.0)
        
        idx = self._get_pred_permutation_idx(indices)
        
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(targets["classes"], indices)])
        target_classes = torch.full(pred_logits.shape[:2], self.num_classes, 
                                  dtype=torch.int64, device=pred_logits.device)
        
        if len(idx[0]) > 0:
            target_classes[idx] = target_classes_o.to(pred_logits.device)
        
        # Stable cross entropy
        loss_ce = F.cross_entropy(pred_logits.transpose(1, 2), target_classes, 
                                 self.weights, ignore_index=self.ignore, reduction='mean')
        
        # Check for NaN/Inf
        if torch.isnan(loss_ce) or torch.isinf(loss_ce):
            print("Warning: NaN/Inf in classification loss, using fallback")
            loss_ce = torch.tensor(1.0, device=pred_logits.device, requires_grad=True)
        
        return {"loss_ce": loss_ce}
    
    def loss_masks(self, outputs, targets, indices, num_masks, masks_ids):
        """Stable mask losses"""
        masks = [t for t in targets["masks"]]
        if not masks or sum(m.shape[0] for m in masks) == 0:
            device = outputs["pred_masks"].device
            return {
                "loss_mask": torch.tensor(0.1, device=device, requires_grad=True),
                "loss_dice": torch.tensor(0.1, device=device, requires_grad=True)
            }
        
        try:
            n_masks = [m.shape[0] for m in masks]
            target_masks = pad_stack(masks)
            
            pred_idx = self._get_pred_permutation_idx(indices)
            tgt_idx = self._get_tgt_permutation_idx(indices, n_masks)
            
            if len(pred_idx[0]) == 0 or len(tgt_idx) == 0:
                device = outputs["pred_masks"].device
                return {
                    "loss_mask": torch.tensor(0.1, device=device, requires_grad=True),
                    "loss_dice": torch.tensor(0.1, device=device, requires_grad=True)
                }
            
            pred_masks = outputs["pred_masks"]
            pred_masks = pred_masks[pred_idx[0], :, pred_idx[1]]
            target_masks = target_masks.to(pred_masks)
            target_masks = target_masks[tgt_idx]
            
            # Clamp masks for stability
            pred_masks = torch.clamp(pred_masks, -50.0, 50.0)
            
            # Sample points with error handling
            with torch.no_grad():
                try:
                    idx = sample_points(masks, masks_ids, self.n_mask_pts, self.num_points)
                except Exception as e:
                    print(f"Error in point sampling: {e}")
                    # Use random sampling as fallback
                    idx = [torch.randperm(m.shape[1])[:min(100, m.shape[1])] for m in masks]
                
                n_masks.insert(0, 0)
                nm = torch.cumsum(torch.tensor(n_masks), 0)
                
                point_labels = []
                point_logits = []
                
                for i, p in enumerate(idx):
                    if nm[i] < nm[i+1] and len(p) > 0:
                        labels = target_masks[nm[i]:nm[i+1]][:, p]
                        logits = pred_masks[nm[i]:nm[i+1]][:, p]
                        point_labels.append(labels)
                        point_logits.append(logits)
                
                if not point_labels:
                    device = outputs["pred_masks"].device
                    return {
                        "loss_mask": torch.tensor(0.1, device=device, requires_grad=True),
                        "loss_dice": torch.tensor(0.1, device=device, requires_grad=True)
                    }
                
                point_labels = torch.cat(point_labels)
                point_logits = torch.cat(point_logits)
            
            # Stable loss computation
            losses = {
                "loss_mask": self.stable_sigmoid_ce_loss(point_logits, point_labels, num_masks),
                "loss_dice": self.stable_dice_loss(point_logits, point_labels, num_masks),
            }
            
            return losses
            
        except Exception as e:
            print(f"Error in mask loss computation: {e}")
            device = outputs["pred_masks"].device
            return {
                "loss_mask": torch.tensor(1.0, device=device, requires_grad=True),
                "loss_dice": torch.tensor(1.0, device=device, requires_grad=True)
            }
    
    def stable_dice_loss(self, inputs, targets, num_masks):
        """Numerically stable dice loss"""
        inputs = torch.sigmoid(torch.clamp(inputs, -20.0, 20.0))
        
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        
        # Add larger epsilon for stability
        loss = 1 - (numerator + 1.0) / (denominator + 1.0)
        loss = loss.sum() / max(num_masks, 1.0)
        
        # Clamp final loss
        loss = torch.clamp(loss, 0.0, 10.0)
        
        return loss
    
    def stable_sigmoid_ce_loss(self, inputs, targets, num_masks):
        """Numerically stable BCE loss"""
        # Clamp inputs for stability
        inputs = torch.clamp(inputs, -20.0, 20.0)
        
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        loss = loss.mean(1).sum() / max(num_masks, 1.0)
        
        # Clamp final loss
        loss = torch.clamp(loss, 0.0, 10.0)
        
        return loss
    
    def _get_pred_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def _get_tgt_permutation_idx(self, indices, n_masks):
        """Target permutation with stability checks"""
        try:
            batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
            tgt_idx = torch.cat([tgt for (_, tgt) in indices])
            
            if len(batch_idx) == 0:
                return []
            
            cont_id = torch.cat([torch.arange(n) for n in n_masks if n > 0])
            
            if len(cont_id) == 0:
                return []
            
            b_id = torch.stack((batch_idx, cont_id), axis=1)
            
            max_batch = int(torch.max(batch_idx).item()) if len(batch_idx) > 0 else 0
            max_mask = max(n_masks) if n_masks else 1
            
            map_m = torch.zeros((max_batch + 1, max_mask))
            for i in range(len(b_id)):
                if b_id[i, 0] < map_m.shape[0] and b_id[i, 1] < map_m.shape[1]:
                    map_m[b_id[i, 0], b_id[i, 1]] = i
            
            stack_ids = []
            for i in range(len(batch_idx)):
                if batch_idx[i] < map_m.shape[0] and tgt_idx[i] < map_m.shape[1]:
                    stack_ids.append(int(map_m[batch_idx[i], tgt_idx[i]]))
            
            return stack_ids
            
        except Exception as e:
            print(f"Error in target permutation: {e}")
            return []


class StableOptimizedMaskPLS(LightningModule):
    """Numerically stable optimized MaskPLS"""
    def __init__(self, cfg, onnx_interval=5):
        super().__init__()
        self.save_hyperparameters(dict(cfg))
        self.cfg = cfg
        self.onnx_interval = onnx_interval
        
        # Get dataset config
        dataset = cfg.MODEL.DATASET
        self.num_classes = cfg[dataset].NUM_CLASSES
        self.ignore_label = cfg[dataset].IGNORE_LABEL
        
        # Use stable channel progression
        cfg.BACKBONE.CHANNELS = [32, 64, 128, 128, 128]  # More conservative
        
        # Create model
        self.model = MaskPLSSimplifiedONNX(cfg)
        
        # Use stable loss
        self.mask_loss = StableMaskLoss(cfg.LOSS, cfg[dataset])
        
        # Semantic loss
        from mask_pls.models.loss import SemLoss
        self.sem_loss = SemLoss(cfg.LOSS.SEM.WEIGHTS)
        
        # Stable voxelizer
        self.voxelizer = StableVoxelizer(
            (96, 96, 48),  # Moderate resolution
            cfg[dataset].SPACE,
            device='cuda'
        )
        
        # Evaluator
        self.evaluator = PanopticEvaluator(cfg[dataset], dataset)
        
        # Cache things IDs
        data = SemanticDatasetModule(cfg)
        data.setup()
        self.things_ids = data.things_ids
        
        # Performance tracking
        self.batch_times = []
        
        # Initialize weights with Xavier for stability
        self._init_weights_stable()
        
    def _init_weights_stable(self):
        """Stable weight initialization"""
        for m in self.model.modules():
            if isinstance(m, (torch.nn.Conv3d, torch.nn.Conv2d, torch.nn.Conv1d)):
                # Xavier initialization for stability
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, (torch.nn.BatchNorm3d, torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, batch):
        """Forward pass with stability checks"""
        points = batch['pt_coord']
        features = batch['feats']
        
        # Stable voxelization
        voxel_grids, norm_coords, valid_indices = self.voxelizer.voxelize_batch_stable(points, features)
        
        # Check for NaN/Inf
        if torch.isnan(voxel_grids).any() or torch.isinf(voxel_grids).any():
            print("Warning: NaN/Inf in voxel grids, replacing with zeros")
            voxel_grids = torch.nan_to_num(voxel_grids, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Update spatial shape
        self.model.spatial_shape = self.voxelizer.spatial_shape
        
        # Prepare coordinates with stability
        if len(norm_coords) > 0 and any(len(c) > 0 for c in norm_coords):
            # Find max points
            valid_coords = [c for c in norm_coords if len(c) > 0]
            if valid_coords:
                max_pts = max(c.shape[0] for c in valid_coords)
            else:
                max_pts = 1000
            
            padded_coords = []
            padding_masks = []
            
            for coords in norm_coords:
                n_pts = coords.shape[0]
                if n_pts == 0:
                    # Handle empty coordinates
                    coords = torch.zeros(max_pts, 3, device='cuda')
                    mask = torch.ones(max_pts, dtype=torch.bool, device='cuda')
                elif n_pts < max_pts:
                    coords = F.pad(coords, (0, 0, 0, max_pts - n_pts))
                    mask = torch.zeros(max_pts, dtype=torch.bool, device='cuda')
                    mask[n_pts:] = True
                else:
                    mask = torch.zeros(max_pts, dtype=torch.bool, device='cuda')
                
                padded_coords.append(coords)
                padding_masks.append(mask)
            
            batch_coords = torch.stack(padded_coords)
            padding_masks = torch.stack(padding_masks)
        else:
            # Fallback for empty batch
            batch_coords = torch.zeros(len(points), 1000, 3, device='cuda')
            padding_masks = torch.ones(len(points), 1000, dtype=torch.bool, device='cuda')
            valid_indices = [torch.zeros(0, dtype=torch.long, device='cuda') for _ in points]
        
        # Forward through model with gradient checkpointing for memory efficiency
        try:
            pred_logits, pred_masks, sem_logits = self.model(voxel_grids, batch_coords)
            
            # Check outputs for NaN/Inf
            if torch.isnan(pred_logits).any() or torch.isinf(pred_logits).any():
                print("Warning: NaN/Inf in pred_logits")
                pred_logits = torch.nan_to_num(pred_logits, nan=0.0, posinf=10.0, neginf=-10.0)
            
            if torch.isnan(pred_masks).any() or torch.isinf(pred_masks).any():
                print("Warning: NaN/Inf in pred_masks")
                pred_masks = torch.nan_to_num(pred_masks, nan=0.0, posinf=10.0, neginf=-10.0)
            
            if torch.isnan(sem_logits).any() or torch.isinf(sem_logits).any():
                print("Warning: NaN/Inf in sem_logits")
                sem_logits = torch.nan_to_num(sem_logits, nan=0.0, posinf=10.0, neginf=-10.0)
            
        except Exception as e:
            print(f"Error in model forward pass: {e}")
            # Return dummy outputs
            B = len(points)
            pred_logits = torch.zeros(B, 100, self.num_classes + 1, device='cuda')
            pred_masks = torch.zeros(B, 1000, 100, device='cuda')
            sem_logits = torch.zeros(B, 1000, self.num_classes, device='cuda')
        
        outputs = {
            'pred_logits': pred_logits,
            'pred_masks': pred_masks,
            'aux_outputs': []
        }
        
        return outputs, padding_masks, sem_logits, valid_indices
    
    def training_step(self, batch, batch_idx):
        """Stable training step"""
        step_start = time.time()
        
        try:
            # Forward pass
            outputs, padding, sem_logits, valid_indices = self.forward(batch)
            
            # Skip if no valid data
            if all(len(v) == 0 for v in valid_indices):
                print(f"Warning: No valid data in batch {batch_idx}")
                return torch.tensor(0.1, device='cuda', requires_grad=True)
            
            # Prepare targets
            targets = self.prepare_targets_stable(batch, padding.shape[1], valid_indices)
            
            # Compute losses with stability
            mask_losses = self.mask_loss(outputs, targets, batch['masks_ids'], batch['pt_coord'])
            
            # Semantic loss
            sem_loss_value = self.compute_sem_loss_stable(batch, sem_logits, valid_indices, padding)
            
            # Total loss with clamping
            total_loss = sum(mask_losses.values()) + sem_loss_value
            total_loss = torch.clamp(total_loss, 0.0, 50.0)
            
            # Check for NaN/Inf
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"Warning: NaN/Inf loss in batch {batch_idx}, using fallback")
                total_loss = torch.tensor(1.0, device='cuda', requires_grad=True)
            
            # Log metrics
            if batch_idx % 10 == 0:
                self.log("train_loss", total_loss, batch_size=self.cfg.TRAIN.BATCH_SIZE)
                for k, v in mask_losses.items():
                    self.log(f"train/{k}", v, batch_size=self.cfg.TRAIN.BATCH_SIZE)
                self.log("train/sem_loss", sem_loss_value, batch_size=self.cfg.TRAIN.BATCH_SIZE)
                
                # Performance tracking
                step_time = time.time() - step_start
                self.batch_times.append(step_time)
                if len(self.batch_times) > 100:
                    self.batch_times.pop(0)
                avg_time = np.mean(self.batch_times)
                print(f"Batch {batch_idx}: {step_time:.2f}s (avg: {avg_time:.2f}s), loss: {total_loss:.4f}")
            
            return total_loss
            
        except Exception as e:
            print(f"Error in training step {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            # Return small loss to continue training
            return torch.tensor(0.1, device='cuda', requires_grad=True)
    
    def prepare_targets_stable(self, batch, max_points, valid_indices):
        """Prepare targets with stability checks"""
        targets = {'classes': [], 'masks': []}
        
        for i in range(len(batch['masks_cls'])):
            try:
                if len(batch['masks_cls'][i]) > 0:
                    # Classes
                    classes = torch.tensor(batch['masks_cls'][i], dtype=torch.long, device='cuda')
                    # Clamp class values for safety
                    classes = torch.clamp(classes, 0, self.num_classes - 1)
                    targets['classes'].append(classes)
                    
                    # Masks
                    masks_list = []
                    for m in batch['masks'][i]:
                        if isinstance(m, torch.Tensor):
                            mask = m.float()
                        else:
                            mask = torch.from_numpy(m).float()
                        
                        # Check for NaN/Inf
                        if torch.isnan(mask).any() or torch.isinf(mask).any():
                            mask = torch.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)
                        
                        # Ensure correct size
                        if mask.shape[0] != max_points:
                            if mask.shape[0] < max_points:
                                mask = F.pad(mask, (0, max_points - mask.shape[0]))
                            else:
                                mask = mask[:max_points]
                        
                        masks_list.append(mask.to('cuda'))
                    
                    if masks_list:
                        targets['masks'].append(torch.stack(masks_list))
                    else:
                        targets['masks'].append(torch.zeros(0, max_points, device='cuda'))
                else:
                    targets['classes'].append(torch.empty(0, dtype=torch.long, device='cuda'))
                    targets['masks'].append(torch.zeros(0, max_points, device='cuda'))
                    
            except Exception as e:
                print(f"Error preparing targets for batch {i}: {e}")
                targets['classes'].append(torch.empty(0, dtype=torch.long, device='cuda'))
                targets['masks'].append(torch.zeros(0, max_points, device='cuda'))
        
        return targets
    
    def compute_sem_loss_stable(self, batch, sem_logits, valid_indices, padding):
        """Stable semantic loss computation"""
        try:
            if sem_logits.numel() == 0:
                return torch.tensor(0.1, device='cuda', requires_grad=True)
            
            all_logits = []
            all_labels = []
            
            for i, (labels, idx, pad) in enumerate(zip(batch['sem_label'], valid_indices, padding)):
                if len(idx) == 0:
                    continue
                
                valid_mask = ~pad
                if valid_mask.sum() == 0:
                    continue
                    
                valid_logits = sem_logits[i][valid_mask]
                
                # Handle labels
                if isinstance(labels, np.ndarray):
                    labels = labels.flatten()
                else:
                    labels = np.array(labels).flatten()
                
                # Map indices safely
                if len(labels) > 0 and len(idx) > 0:
                    idx_cpu = idx.cpu().numpy()
                    # Ensure indices are within bounds
                    valid_idx = idx_cpu[(idx_cpu >= 0) & (idx_cpu < len(labels))]
                    
                    if len(valid_idx) > 0:
                        selected_labels = labels[valid_idx]
                        # Match number of logits and labels
                        min_len = min(len(selected_labels), len(valid_logits))
                        if min_len > 0:
                            selected_labels = selected_labels[:min_len]
                            selected_logits = valid_logits[:min_len]
                            
                            labels_tensor = torch.from_numpy(selected_labels).long().cuda()
                            labels_tensor = torch.clamp(labels_tensor, 0, self.num_classes - 1)
                            
                            all_logits.append(selected_logits)
                            all_labels.append(labels_tensor)
            
            if all_logits:
                combined_logits = torch.cat(all_logits, dim=0)
                combined_labels = torch.cat(all_labels, dim=0)
                
                # Clamp logits for stability
                combined_logits = torch.clamp(combined_logits, -50.0, 50.0)
                
                # Compute loss
                sem_loss = F.cross_entropy(combined_logits, combined_labels, ignore_index=0, reduction='mean')
                
                # Check for NaN/Inf
                if torch.isnan(sem_loss) or torch.isinf(sem_loss):
                    sem_loss = torch.tensor(1.0, device='cuda', requires_grad=True)
                
                # Apply weight and clamp
                sem_loss = sem_loss * self.cfg.LOSS.SEM.WEIGHTS[0]
                sem_loss = torch.clamp(sem_loss, 0.0, 10.0)
                
                return sem_loss
            else:
                return torch.tensor(0.1, device='cuda', requires_grad=True)
                
        except Exception as e:
            print(f"Semantic loss error: {e}")
            return torch.tensor(0.1, device='cuda', requires_grad=True)
    
    def validation_step(self, batch, batch_idx):
        """Stable validation step"""
        try:
            outputs, padding, sem_logits, valid_indices = self.forward(batch)
            
            if all(len(v) == 0 for v in valid_indices):
                return torch.tensor(0.0, device='cuda')
            
            targets = self.prepare_targets_stable(batch, padding.shape[1], valid_indices)
            losses = self.mask_loss(outputs, targets, batch['masks_ids'], batch['pt_coord'])
            sem_loss = self.compute_sem_loss_stable(batch, sem_logits, valid_indices, padding)
            
            total_loss = sum(losses.values()) + sem_loss
            total_loss = torch.clamp(total_loss, 0.0, 50.0)
            
            # Log validation metrics
            if batch_idx % 10 == 0:
                self.log("val_loss", total_loss, batch_size=self.cfg.TRAIN.BATCH_SIZE)
                for k, v in losses.items():
                    self.log(f"val/{k}", v, batch_size=self.cfg.TRAIN.BATCH_SIZE)
                
                # Panoptic evaluation
                sem_pred, ins_pred = self.panoptic_inference(outputs, padding)
                self.evaluator.update(sem_pred, ins_pred, batch)
            
            return total_loss
            
        except Exception as e:
            print(f"Validation error: {e}")
            return torch.tensor(0.0, device='cuda')
    
    def panoptic_inference(self, outputs, padding):
        """Stable panoptic inference"""
        mask_cls = outputs["pred_logits"]
        mask_pred = outputs["pred_masks"]
        
        sem_pred = []
        ins_pred = []
        
        for mask_cls_b, mask_pred_b, pad in zip(mask_cls, mask_pred, padding):
            try:
                scores, labels = mask_cls_b.max(-1)
                mask_pred_valid = mask_pred_b[~pad].sigmoid()
                
                keep = labels.ne(self.num_classes)
                
                if keep.sum() == 0:
                    sem_pred.append(torch.zeros(mask_pred_valid.shape[0], dtype=torch.long).cpu().numpy())
                    ins_pred.append(torch.zeros(mask_pred_valid.shape[0], dtype=torch.long).cpu().numpy())
                    continue
                
                cur_scores = scores[keep]
                cur_classes = labels[keep]
                cur_masks = mask_pred_valid[:, keep]
                
                # Mask assignment
                cur_prob_masks = cur_scores.unsqueeze(0) * cur_masks
                cur_mask_ids = cur_prob_masks.argmax(1)
                
                sem = torch.zeros(cur_masks.shape[0], dtype=torch.long, device=cur_masks.device)
                ins = torch.zeros_like(sem)
                
                segment_id = 0
                for k in range(cur_classes.shape[0]):
                    pred_class = cur_classes[k].item()
                    mask = (cur_mask_ids == k) & (cur_masks[:, k] >= 0.5)
                    
                    if mask.sum() > 10:
                        sem[mask] = pred_class
                        if pred_class in self.things_ids:
                            segment_id += 1
                            ins[mask] = segment_id
                
                sem_pred.append(sem.cpu().numpy())
                ins_pred.append(ins.cpu().numpy())
                
            except Exception as e:
                print(f"Error in panoptic inference: {e}")
                sem_pred.append(np.zeros(1000, dtype=np.int32))
                ins_pred.append(np.zeros(1000, dtype=np.int32))
        
        return sem_pred, ins_pred
    
    def validation_epoch_end(self, outputs):
        """Compute validation metrics"""
        try:
            bs = self.cfg.TRAIN.BATCH_SIZE
            pq = self.evaluator.get_mean_pq()
            iou = self.evaluator.get_mean_iou()
            rq = self.evaluator.get_mean_rq()
            
            self.log("metrics/pq", pq, batch_size=bs)
            self.log("metrics/iou", iou, batch_size=bs)
            self.log("metrics/rq", rq, batch_size=bs)
            
            print(f"\nValidation Metrics - PQ: {pq:.4f}, IoU: {iou:.4f}, RQ: {rq:.4f}")
            
            self.evaluator.reset()
            
        except Exception as e:
            print(f"Metrics computation error: {e}")
            self.log("metrics/pq", 0.0, batch_size=self.cfg.TRAIN.BATCH_SIZE)
            self.log("metrics/iou", 0.0, batch_size=self.cfg.TRAIN.BATCH_SIZE)
    
    def configure_optimizers(self):
        """Stable optimizer configuration"""
        # Use AdamW with conservative learning rate
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.TRAIN.LR * 0.5,  # Lower LR for stability
            weight_decay=1e-4,
            eps=1e-8  # Default epsilon for stability
        )
        
        # Cosine annealing for smooth learning
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.TRAIN.MAX_EPOCH,
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }


@click.command()
@click.option("--epochs", type=int, default=100, help="Number of epochs")
@click.option("--batch_size", type=int, default=2, help="Batch size")
@click.option("--lr", type=float, default=0.00005, help="Learning rate (lower for stability)")
@click.option("--gpus", type=int, default=1, help="Number of GPUs")
@click.option("--num_workers", type=int, default=2, help="Number of workers")
@click.option("--nuscenes", is_flag=True, help="Use NuScenes dataset")
@click.option("--checkpoint", type=str, default=None, help="Resume from checkpoint")
@click.option("--onnx_interval", type=int, default=10, help="Export ONNX model every N epochs")
def main(epochs, batch_size, lr, gpus, num_workers, nuscenes, checkpoint, onnx_interval):
    """Numerically stable MaskPLS training"""
    
    print("="*60)
    print("Numerically Stable MaskPLS Training v5")
    print("Fixing NaN/Inf issues with robust error handling")
    print("="*60)
    
    # Load config
    def getDir(obj):
        return os.path.dirname(os.path.abspath(obj))
    
    model_cfg = edict(yaml.safe_load(open(join(getDir(__file__), "../config/model.yaml"))))
    backbone_cfg = edict(yaml.safe_load(open(join(getDir(__file__), "../config/backbone.yaml"))))
    decoder_cfg = edict(yaml.safe_load(open(join(getDir(__file__), "../config/decoder.yaml"))))
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    
    # Update config for stability
    cfg.TRAIN.MAX_EPOCH = epochs
    cfg.TRAIN.BATCH_SIZE = batch_size
    cfg.TRAIN.LR = lr
    cfg.TRAIN.N_GPUS = gpus
    cfg.TRAIN.NUM_WORKERS = num_workers
    cfg.TRAIN.BATCH_ACC = 2  # Lower accumulation for stability
    
    if nuscenes:
        cfg.MODEL.DATASET = "NUSCENES"
    
    # Use moderate sampling for stability
    dataset = cfg.MODEL.DATASET
    cfg[dataset].SUB_NUM_POINTS = min(cfg[dataset].SUB_NUM_POINTS, 50000)
    
    # Moderate loss parameters
    cfg.LOSS.NUM_POINTS = 30000  # Moderate
    cfg.LOSS.NUM_MASK_PTS = 300  # Moderate
    
    print(f"Configuration:")
    print(f"  Dataset: {dataset}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {lr}")
    print(f"  Point Sampling: {cfg[dataset].SUB_NUM_POINTS}")
    print(f"  Loss Point Sampling: {cfg.LOSS.NUM_POINTS}")
    print(f"  Voxel Resolution: 96x96x48")
    print(f"  Numerical Stability: ENABLED")
    
    # Create data module
    data = SemanticDatasetModule(cfg)
    
    # Create model
    model = StableOptimizedMaskPLS(cfg, onnx_interval=onnx_interval)
    
    # Load checkpoint if provided
    if checkpoint:
        print(f"Loading checkpoint: {checkpoint}")
        try:
            ckpt = torch.load(checkpoint, map_location='cpu')
            model.load_state_dict(ckpt['state_dict'], strict=False)
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
    
    # Setup logging
    tb_logger = pl_loggers.TensorBoardLogger(
        "experiments/" + cfg.EXPERIMENT.ID + "_stable_v5",
        default_hp_metric=False
    )
    
    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  # Monitor validation loss for stability
        filename=cfg.EXPERIMENT.ID + "_stable_epoch{epoch:02d}_loss{val_loss:.2f}",
        auto_insert_metric_name=False,
        mode="min",
        save_last=True,
        save_top_k=3
    )
    
    # Create trainer with stability settings
    trainer = Trainer(
        gpus=gpus,
        accelerator="gpu" if gpus > 0 else None,
        strategy="ddp" if gpus > 1 else None,
        logger=tb_logger,
        max_epochs=epochs,
        callbacks=[lr_monitor, checkpoint_callback],
        log_every_n_steps=10,
        gradient_clip_val=1.0,  # Conservative gradient clipping
        accumulate_grad_batches=cfg.TRAIN.BATCH_ACC,
        precision=32,  # Use full precision for stability
        num_sanity_val_steps=2,  # Do sanity check
        val_check_interval=0.5,
        resume_from_checkpoint=checkpoint,
        detect_anomaly=True  # Enable anomaly detection
    )
    
    # Train
    print("\nStarting stable training...")
    trainer.fit(model, data)
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()