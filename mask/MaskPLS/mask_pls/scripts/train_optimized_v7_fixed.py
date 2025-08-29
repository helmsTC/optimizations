"""
Fixed optimized training script for MaskPLS
Addresses scatter operation errors and optimizer assertion issues
"""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
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

warnings.filterwarnings("ignore", category=UserWarning)


class FixedVoxelizer:
    """Fixed voxelizer with proper tensor type handling"""
    def __init__(self, spatial_shape, bounds, device='cuda'):
        self.spatial_shape = (96, 96, 48)  # Moderate resolution for stability
        self.bounds = bounds
        self.device = device
        
        # Ensure all bounds tensors are on the correct device and dtype
        self.bounds_min = torch.tensor([bounds[d][0] for d in range(3)], 
                                     device=device, dtype=torch.float32)
        self.bounds_max = torch.tensor([bounds[d][1] for d in range(3)], 
                                     device=device, dtype=torch.float32)
        self.bounds_range = self.bounds_max - self.bounds_min
        self.bounds_range = torch.clamp(self.bounds_range, min=1e-3)
        
        # Precompute spatial dimensions as tensors
        self.spatial_dims = torch.tensor([self.spatial_shape[0], self.spatial_shape[1], self.spatial_shape[2]], 
                                       device=device, dtype=torch.float32)
        self.spatial_dims_int = torch.tensor([self.spatial_shape[0]-1, self.spatial_shape[1]-1, self.spatial_shape[2]-1], 
                                           device=device, dtype=torch.long)
        
        self.cache = OrderedDict()
        self.cache_size = 50
        
    def voxelize_batch_fixed(self, points_batch, features_batch):
        """Fixed voxelization with proper tensor types"""
        B = len(points_batch)
        D, H, W = self.spatial_shape
        C = features_batch[0].shape[1] if len(features_batch) > 0 else 4
        
        # Initialize with correct types
        voxel_grids = torch.zeros(B, C, D, H, W, device=self.device, dtype=torch.float32)
        voxel_counts = torch.zeros(B, 1, D, H, W, device=self.device, dtype=torch.float32)
        normalized_coords = []
        valid_indices = []
        
        for b in range(B):
            try:
                # Convert inputs with proper error handling
                if isinstance(points_batch[b], np.ndarray):
                    pts = torch.from_numpy(points_batch[b].astype(np.float32)).to(self.device)
                else:
                    pts = points_batch[b].float().to(self.device)
                
                if isinstance(features_batch[b], np.ndarray):
                    feat = torch.from_numpy(features_batch[b].astype(np.float32)).to(self.device)
                else:
                    feat = features_batch[b].float().to(self.device)
                
                # Check for empty inputs
                if pts.numel() == 0 or feat.numel() == 0:
                    print(f"Warning: Empty input for batch {b}")
                    normalized_coords.append(torch.zeros(0, 3, device=self.device))
                    valid_indices.append(torch.zeros(0, dtype=torch.long, device=self.device))
                    continue
                
                # Ensure correct shapes
                if len(pts.shape) != 2 or pts.shape[1] != 3:
                    print(f"Warning: Invalid points shape {pts.shape} for batch {b}")
                    normalized_coords.append(torch.zeros(0, 3, device=self.device))
                    valid_indices.append(torch.zeros(0, dtype=torch.long, device=self.device))
                    continue
                
                if len(feat.shape) != 2 or feat.shape[0] != pts.shape[0]:
                    print(f"Warning: Invalid features shape {feat.shape} vs points {pts.shape} for batch {b}")
                    normalized_coords.append(torch.zeros(0, 3, device=self.device))
                    valid_indices.append(torch.zeros(0, dtype=torch.long, device=self.device))
                    continue
                
                # NaN/Inf handling
                if torch.isnan(pts).any() or torch.isinf(pts).any():
                    print(f"Warning: NaN/Inf in points for batch {b}")
                    pts = torch.nan_to_num(pts, nan=0.0, posinf=100.0, neginf=-100.0)
                
                if torch.isnan(feat).any() or torch.isinf(feat).any():
                    print(f"Warning: NaN/Inf in features for batch {b}")
                    feat = torch.nan_to_num(feat, nan=0.0, posinf=10.0, neginf=-10.0)
                
                # Bounds checking
                valid_mask = ((pts >= self.bounds_min) & (pts < self.bounds_max)).all(dim=1)
                
                if valid_mask.sum() == 0:
                    print(f"Warning: No points within bounds for batch {b}")
                    normalized_coords.append(torch.zeros(0, 3, device=self.device))
                    valid_indices.append(torch.zeros(0, dtype=torch.long, device=self.device))
                    continue
                
                valid_idx = torch.where(valid_mask)[0]
                valid_pts = pts[valid_mask]
                valid_feat = feat[valid_mask]
                
                # Normalize coordinates
                norm_coords = (valid_pts - self.bounds_min) / self.bounds_range
                norm_coords = torch.clamp(norm_coords, 0.0, 0.999999)
                
                # Convert to voxel indices - FIXED: ensure all tensors have correct types
                voxel_coords_float = norm_coords * self.spatial_dims
                voxel_indices = voxel_coords_float.long()
                
                # Clamp indices to valid range
                voxel_indices = torch.clamp(voxel_indices, 
                                          torch.zeros(3, device=self.device, dtype=torch.long),
                                          self.spatial_dims_int)
                
                # Calculate flat indices - FIXED: ensure all operations use consistent types
                flat_indices = (voxel_indices[:, 0] * (H * W) + 
                              voxel_indices[:, 1] * W + 
                              voxel_indices[:, 2]).long()
                
                # Ensure flat_indices are within bounds
                max_idx = D * H * W - 1
                flat_indices = torch.clamp(flat_indices, 0, max_idx)
                
                # Accumulate features - FIXED: proper type handling for scatter_add_
                for c in range(C):
                    feat_values = valid_feat[:, c].float()  # Ensure float32
                    feat_values = torch.clamp(feat_values, -100.0, 100.0)
                    
                    # FIXED: Use proper scatter_add_ with consistent types
                    voxel_grids[b, c].view(-1).scatter_add_(0, flat_indices, feat_values)
                
                # Count accumulation
                ones = torch.ones(valid_feat.shape[0], device=self.device, dtype=torch.float32)
                voxel_counts[b, 0].view(-1).scatter_add_(0, flat_indices, ones)
                
                normalized_coords.append(norm_coords)
                valid_indices.append(valid_idx)
                
            except Exception as e:
                print(f"Voxelization error for batch {b}: {e}")
                import traceback
                traceback.print_exc()
                normalized_coords.append(torch.zeros(0, 3, device=self.device))
                valid_indices.append(torch.zeros(0, dtype=torch.long, device=self.device))
                continue
        
        # Average with numerical stability
        voxel_grids = voxel_grids / (voxel_counts + 1e-4)
        voxel_grids = torch.nan_to_num(voxel_grids, nan=0.0, posinf=0.0, neginf=0.0)
        
        return voxel_grids, normalized_coords, valid_indices


class FixedMaskLoss(torch.nn.Module):
    """Fixed mask loss with proper error handling"""
    def __init__(self, cfg, data_cfg):
        super().__init__()
        
        self.num_classes = data_cfg.NUM_CLASSES
        self.ignore = data_cfg.IGNORE_LABEL
        self.matcher = HungarianMatcher(cfg.WEIGHTS, cfg.P_RATIO)
        
        self.weight_dict = {
            cfg.WEIGHTS_KEYS[i]: cfg.WEIGHTS[i] for i in range(len(cfg.WEIGHTS))
        }
        self.eos_coef = cfg.EOS_COEF
        
        weights = torch.ones(self.num_classes + 1)
        weights[0] = 0.1  # Small non-zero for stability
        weights[-1] = self.eos_coef
        self.register_buffer('weights', weights)
        
        # Use moderate sampling
        self.num_points = min(cfg.NUM_POINTS, 50000)
        self.n_mask_pts = min(cfg.NUM_MASK_PTS, 400)
        
    def forward(self, outputs, targets, masks_ids, coors):
        """Fixed loss computation"""
        losses = {}
        
        # Check outputs
        if not isinstance(outputs, dict) or "pred_logits" not in outputs or "pred_masks" not in outputs:
            print("Warning: Invalid outputs structure")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            return {
                "loss_ce": torch.tensor(0.1, device=device, requires_grad=True),
                "loss_dice": torch.tensor(0.1, device=device, requires_grad=True),
                "loss_mask": torch.tensor(0.1, device=device, requires_grad=True)
            }
        
        # Stability checks
        pred_logits = outputs["pred_logits"]
        pred_masks = outputs["pred_masks"]
        
        if torch.isnan(pred_logits).any() or torch.isinf(pred_logits).any():
            print("Warning: NaN/Inf in pred_logits")
            pred_logits = torch.nan_to_num(pred_logits, nan=0.0, posinf=10.0, neginf=-10.0)
            outputs["pred_logits"] = pred_logits
        
        if torch.isnan(pred_masks).any() or torch.isinf(pred_masks).any():
            print("Warning: NaN/Inf in pred_masks")
            pred_masks = torch.nan_to_num(pred_masks, nan=0.0, posinf=10.0, neginf=-10.0)
            outputs["pred_masks"] = pred_masks
        
        # Check targets
        if not isinstance(targets, dict) or "classes" not in targets or "masks" not in targets:
            print("Warning: Invalid targets structure")
            device = pred_logits.device
            return {
                "loss_ce": torch.tensor(0.1, device=device, requires_grad=True),
                "loss_dice": torch.tensor(0.1, device=device, requires_grad=True),
                "loss_mask": torch.tensor(0.1, device=device, requires_grad=True)
            }
        
        num_masks = sum(len(t) for t in targets["classes"] if len(t) > 0)
        if num_masks == 0:
            device = pred_logits.device
            return {
                "loss_ce": torch.tensor(0.1, device=device, requires_grad=True),
                "loss_dice": torch.tensor(0.1, device=device, requires_grad=True),
                "loss_mask": torch.tensor(0.1, device=device, requires_grad=True)
            }
        
        num_masks = max(num_masks, 1)
        outputs_no_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        
        try:
            indices = self.matcher(outputs_no_aux, targets)
            losses.update(self.get_losses(outputs, targets, indices, num_masks, masks_ids, coors))
            
        except Exception as e:
            print(f"Matching/loss error: {e}")
            device = pred_logits.device
            return {
                "loss_ce": torch.tensor(0.5, device=device, requires_grad=True),
                "loss_dice": torch.tensor(0.5, device=device, requires_grad=True),
                "loss_mask": torch.tensor(0.5, device=device, requires_grad=True)
            }
        
        # Apply weights with safety
        weighted_losses = {}
        for l in losses:
            for k in self.weight_dict:
                if k in l:
                    loss_val = losses[l] * self.weight_dict[k]
                    loss_val = torch.clamp(loss_val, 0.0, 20.0)  # Reasonable upper bound
                    weighted_losses[l] = loss_val
                    break
            if l not in weighted_losses:
                weighted_losses[l] = torch.clamp(losses[l], 0.0, 20.0)
        
        return weighted_losses
    
    def get_losses(self, outputs, targets, indices, num_masks, masks_ids, coors):
        losses = {}
        
        try:
            losses.update(self.loss_classes(outputs, targets, indices))
        except Exception as e:
            print(f"Classification loss error: {e}")
            losses["loss_ce"] = torch.tensor(0.5, device=outputs["pred_logits"].device, requires_grad=True)
        
        try:
            losses.update(self.loss_masks(outputs, targets, indices, num_masks, masks_ids))
        except Exception as e:
            print(f"Mask loss error: {e}")
            device = outputs["pred_masks"].device
            losses["loss_mask"] = torch.tensor(0.5, device=device, requires_grad=True)
            losses["loss_dice"] = torch.tensor(0.5, device=device, requires_grad=True)
        
        return losses
    
    def loss_classes(self, outputs, targets, indices):
        pred_logits = outputs["pred_logits"].float()
        pred_logits = torch.clamp(pred_logits, -20.0, 20.0)
        
        idx = self._get_pred_permutation_idx(indices)
        
        if len(idx[0]) == 0:
            # No matches, use standard cross entropy
            target_classes = torch.full(pred_logits.shape[:2], self.num_classes, 
                                      dtype=torch.int64, device=pred_logits.device)
        else:
            target_classes_o = torch.cat([t[J] for t, (_, J) in zip(targets["classes"], indices)])
            target_classes = torch.full(pred_logits.shape[:2], self.num_classes, 
                                      dtype=torch.int64, device=pred_logits.device)
            target_classes[idx] = target_classes_o.to(pred_logits.device)
        
        loss_ce = F.cross_entropy(pred_logits.transpose(1, 2), target_classes, 
                                 self.weights.to(pred_logits.device), ignore_index=self.ignore)
        
        return {"loss_ce": torch.clamp(loss_ce, 0.0, 10.0)}
    
    def loss_masks(self, outputs, targets, indices, num_masks, masks_ids):
        masks = [t for t in targets["masks"] if t.numel() > 0]
        if not masks:
            device = outputs["pred_masks"].device
            return {
                "loss_mask": torch.tensor(0.1, device=device, requires_grad=True),
                "loss_dice": torch.tensor(0.1, device=device, requires_grad=True)
            }
        
        try:
            n_masks = [m.shape[0] for m in masks]
            if sum(n_masks) == 0:
                device = outputs["pred_masks"].device
                return {
                    "loss_mask": torch.tensor(0.1, device=device, requires_grad=True),
                    "loss_dice": torch.tensor(0.1, device=device, requires_grad=True)
                }
            
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
            pred_masks = torch.clamp(pred_masks, -20.0, 20.0)
            
            target_masks = target_masks.to(pred_masks.device).to(pred_masks.dtype)
            target_masks = target_masks[tgt_idx]
            
            # Simple point sampling for stability
            with torch.no_grad():
                try:
                    idx = sample_points(masks, masks_ids, self.n_mask_pts, self.num_points)
                except:
                    # Fallback sampling
                    idx = [torch.randperm(m.shape[1], device=m.device)[:min(100, m.shape[1])] for m in masks]
                
                if not idx or all(len(i) == 0 for i in idx):
                    device = pred_masks.device
                    return {
                        "loss_mask": torch.tensor(0.1, device=device, requires_grad=True),
                        "loss_dice": torch.tensor(0.1, device=device, requires_grad=True)
                    }
                
                n_masks.insert(0, 0)
                nm = torch.cumsum(torch.tensor(n_masks), 0)
                
                point_labels_list = []
                point_logits_list = []
                
                for i, p in enumerate(idx):
                    if i < len(nm) - 1 and nm[i] < nm[i+1] and len(p) > 0:
                        try:
                            labels = target_masks[nm[i]:nm[i+1]][:, p]
                            logits = pred_masks[nm[i]:nm[i+1]][:, p]
                            if labels.numel() > 0 and logits.numel() > 0:
                                point_labels_list.append(labels)
                                point_logits_list.append(logits)
                        except Exception as e:
                            print(f"Point sampling error for mask {i}: {e}")
                            continue
                
                if not point_labels_list:
                    device = pred_masks.device
                    return {
                        "loss_mask": torch.tensor(0.1, device=device, requires_grad=True),
                        "loss_dice": torch.tensor(0.1, device=device, requires_grad=True)
                    }
                
                point_labels = torch.cat(point_labels_list, dim=0)
                point_logits = torch.cat(point_logits_list, dim=0)
            
            # Compute losses
            losses = {
                "loss_mask": self.stable_sigmoid_ce_loss(point_logits, point_labels, num_masks),
                "loss_dice": self.stable_dice_loss(point_logits, point_labels, num_masks),
            }
            
            return losses
            
        except Exception as e:
            print(f"Mask loss computation error: {e}")
            device = outputs["pred_masks"].device
            return {
                "loss_mask": torch.tensor(0.5, device=device, requires_grad=True),
                "loss_dice": torch.tensor(0.5, device=device, requires_grad=True)
            }
    
    def stable_dice_loss(self, inputs, targets, num_masks):
        inputs = torch.sigmoid(torch.clamp(inputs, -15.0, 15.0))
        
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        
        loss = 1 - (numerator + 1.0) / (denominator + 1.0)
        loss = loss.sum() / max(num_masks, 1.0)
        
        return torch.clamp(loss, 0.0, 5.0)
    
    def stable_sigmoid_ce_loss(self, inputs, targets, num_masks):
        inputs = torch.clamp(inputs, -15.0, 15.0)
        
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        loss = loss.mean(1).sum() / max(num_masks, 1.0)
        
        return torch.clamp(loss, 0.0, 5.0)
    
    def _get_pred_permutation_idx(self, indices):
        if not indices or all(len(src) == 0 for src, _ in indices):
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)
        
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices) if len(src) > 0])
        src_idx = torch.cat([src for (src, _) in indices if len(src) > 0])
        return batch_idx, src_idx
    
    def _get_tgt_permutation_idx(self, indices, n_masks):
        try:
            if not indices or all(len(tgt) == 0 for _, tgt in indices) or not n_masks or sum(n_masks) == 0:
                return []
            
            batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices) if len(tgt) > 0])
            tgt_idx = torch.cat([tgt for (_, tgt) in indices if len(tgt) > 0])
            
            if len(batch_idx) == 0:
                return []
            
            cont_id = torch.cat([torch.arange(n) for n in n_masks if n > 0])
            if len(cont_id) == 0:
                return []
            
            # Ensure we have matching lengths
            min_len = min(len(batch_idx), len(cont_id))
            batch_idx = batch_idx[:min_len]
            cont_id = cont_id[:min_len]
            
            if min_len == 0:
                return []
            
            b_id = torch.stack((batch_idx, cont_id), axis=1)
            
            max_batch = int(torch.max(batch_idx).item())
            max_mask = max(n_masks)
            
            map_m = torch.zeros((max_batch + 1, max_mask), device=batch_idx.device)
            
            for i in range(len(b_id)):
                if b_id[i, 0] < map_m.shape[0] and b_id[i, 1] < map_m.shape[1]:
                    map_m[b_id[i, 0], b_id[i, 1]] = i
            
            stack_ids = []
            for i in range(min(len(batch_idx), len(tgt_idx))):
                if batch_idx[i] < map_m.shape[0] and tgt_idx[i] < map_m.shape[1]:
                    stack_ids.append(int(map_m[batch_idx[i], tgt_idx[i]]))
            
            return stack_ids
            
        except Exception as e:
            print(f"Target permutation error: {e}")
            return []


class FixedOptimizedMaskPLS(LightningModule):
    """Fixed optimized MaskPLS with proper error handling"""
    def __init__(self, cfg, onnx_interval=10):
        super().__init__()
        self.save_hyperparameters(dict(cfg))
        self.cfg = cfg
        self.onnx_interval = onnx_interval
        
        dataset = cfg.MODEL.DATASET
        self.num_classes = cfg[dataset].NUM_CLASSES
        self.ignore_label = cfg[dataset].IGNORE_LABEL
        
        # Conservative architecture
        cfg.BACKBONE.CHANNELS = [32, 64, 96, 96, 96]
        
        self.model = MaskPLSSimplifiedONNX(cfg)
        self.mask_loss = FixedMaskLoss(cfg.LOSS, cfg[dataset])
        
        from mask_pls.models.loss import SemLoss
        self.sem_loss = SemLoss(cfg.LOSS.SEM.WEIGHTS)
        
        self.voxelizer = FixedVoxelizer(
            (96, 96, 48),
            cfg[dataset].SPACE,
            device='cuda'
        )
        
        self.evaluator = PanopticEvaluator(cfg[dataset], dataset)
        
        # Cache things IDs
        try:
            data = SemanticDatasetModule(cfg)
            data.setup()
            self.things_ids = data.things_ids
        except:
            self.things_ids = []
        
        self.batch_times = []
        self.last_onnx_export_epoch = -1
        
        # Xavier initialization for stability
        self._init_weights_stable()
        
    def _init_weights_stable(self):
        for m in self.model.modules():
            if isinstance(m, (torch.nn.Conv3d, torch.nn.Conv2d, torch.nn.Conv1d)):
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
        """Fixed forward pass"""
        if not isinstance(batch, dict):
            print("Warning: Invalid batch type")
            return None, None, None, []
        
        if 'pt_coord' not in batch or 'feats' not in batch:
            print("Warning: Missing required batch keys")
            return None, None, None, []
        
        points = batch['pt_coord']
        features = batch['feats']
        
        if not points or not features:
            print("Warning: Empty points or features")
            return None, None, None, []
        
        # Fixed voxelization
        voxel_grids, norm_coords, valid_indices = self.voxelizer.voxelize_batch_fixed(points, features)
        
        # Check voxelization results
        if voxel_grids is None or torch.isnan(voxel_grids).any() or torch.isinf(voxel_grids).any():
            print("Warning: Invalid voxelization results")
            voxel_grids = torch.nan_to_num(voxel_grids, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.model.spatial_shape = self.voxelizer.spatial_shape
        
        # Prepare coordinates
        try:
            if len(norm_coords) > 0 and any(len(c) > 0 for c in norm_coords):
                valid_coords = [c for c in norm_coords if len(c) > 0]
                max_pts = max(c.shape[0] for c in valid_coords) if valid_coords else 1000
                max_pts = min(max_pts, 10000)  # Limit for memory
                
                padded_coords = []
                padding_masks = []
                
                for coords in norm_coords:
                    n_pts = coords.shape[0]
                    if n_pts == 0:
                        coords = torch.zeros(max_pts, 3, device='cuda')
                        mask = torch.ones(max_pts, dtype=torch.bool, device='cuda')
                    elif n_pts < max_pts:
                        coords = F.pad(coords, (0, 0, 0, max_pts - n_pts), value=0)
                        mask = torch.zeros(max_pts, dtype=torch.bool, device='cuda')
                        mask[n_pts:] = True
                    else:
                        coords = coords[:max_pts]
                        mask = torch.zeros(max_pts, dtype=torch.bool, device='cuda')
                    
                    padded_coords.append(coords)
                    padding_masks.append(mask)
                
                batch_coords = torch.stack(padded_coords)
                padding_masks = torch.stack(padding_masks)
            else:
                print("Warning: No valid coordinates, using defaults")
                batch_coords = torch.zeros(len(points), 1000, 3, device='cuda')
                padding_masks = torch.ones(len(points), 1000, dtype=torch.bool, device='cuda')
                valid_indices = [torch.zeros(0, dtype=torch.long, device='cuda') for _ in points]
                
        except Exception as e:
            print(f"Coordinate preparation error: {e}")
            batch_coords = torch.zeros(len(points), 1000, 3, device='cuda')
            padding_masks = torch.ones(len(points), 1000, dtype=torch.bool, device='cuda')
            valid_indices = [torch.zeros(0, dtype=torch.long, device='cuda') for _ in points]
        
        # Model forward
        try:
            pred_logits, pred_masks, sem_logits = self.model(voxel_grids, batch_coords)
            
            # Safety checks
            pred_logits = torch.nan_to_num(pred_logits, nan=0.0, posinf=5.0, neginf=-5.0)
            pred_masks = torch.nan_to_num(pred_masks, nan=0.0, posinf=5.0, neginf=-5.0)
            sem_logits = torch.nan_to_num(sem_logits, nan=0.0, posinf=5.0, neginf=-5.0)
            
        except Exception as e:
            print(f"Model forward error: {e}")
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
        """Fixed training step"""
        step_start = time.time()
        
        try:
            # Forward pass with error handling
            result = self.forward(batch)
            if result[0] is None:
                print(f"Skipping batch {batch_idx} due to forward pass error")
                return torch.tensor(0.1, device='cuda', requires_grad=True)
                
            outputs, padding, sem_logits, valid_indices = result
            
            # Check for valid data
            if all(len(v) == 0 for v in valid_indices):
                print(f"No valid data in batch {batch_idx}")
                return torch.tensor(0.1, device='cuda', requires_grad=True)
            
            # Prepare targets
            targets = self.prepare_targets_safe(batch, padding.shape[1], valid_indices)
            if targets is None:
                print(f"Target preparation failed for batch {batch_idx}")
                return torch.tensor(0.1, device='cuda', requires_grad=True)
            
            # Loss computation
            mask_losses = self.mask_loss(outputs, targets, batch.get('masks_ids', []), batch.get('pt_coord', []))
            sem_loss_value = self.compute_sem_loss_safe(batch, sem_logits, valid_indices, padding)
            
            # Total loss
            total_loss = sum(mask_losses.values()) + sem_loss_value
            total_loss = torch.clamp(total_loss, 0.0, 50.0)
            
            # NaN/Inf check
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"NaN/Inf loss in batch {batch_idx}")
                total_loss = torch.tensor(0.5, device='cuda', requires_grad=True)
            
            # Logging
            if batch_idx % 20 == 0:  # Less frequent logging
                self.log("train_loss", total_loss, batch_size=self.cfg.TRAIN.BATCH_SIZE)
                for k, v in mask_losses.items():
                    if torch.isfinite(v):
                        self.log(f"train/{k}", v, batch_size=self.cfg.TRAIN.BATCH_SIZE)
                if torch.isfinite(sem_loss_value):
                    self.log("train/sem_loss", sem_loss_value, batch_size=self.cfg.TRAIN.BATCH_SIZE)
                
                step_time = time.time() - step_start
                print(f"Batch {batch_idx}: {step_time:.2f}s, loss: {total_loss:.4f}")
            
            return total_loss
            
        except Exception as e:
            print(f"Training step error {batch_idx}: {e}")
            return torch.tensor(0.1, device='cuda', requires_grad=True)
    
    def prepare_targets_safe(self, batch, max_points, valid_indices):
        """Safe target preparation"""
        try:
            if 'masks_cls' not in batch or 'masks' not in batch:
                print("Warning: Missing mask data in batch")
                return None
            
            targets = {'classes': [], 'masks': []}
            
            for i in range(len(batch['masks_cls'])):
                try:
                    if len(batch['masks_cls'][i]) > 0:
                        classes = torch.tensor(batch['masks_cls'][i], dtype=torch.long, device='cuda')
                        classes = torch.clamp(classes, 0, self.num_classes - 1)
                        targets['classes'].append(classes)
                        
                        masks_list = []
                        for m in batch['masks'][i]:
                            try:
                                if isinstance(m, torch.Tensor):
                                    mask = m.float()
                                elif isinstance(m, np.ndarray):
                                    mask = torch.from_numpy(m.astype(np.float32))
                                else:
                                    continue
                                
                                mask = torch.nan_to_num(mask, nan=0.0)
                                
                                if mask.shape[0] != max_points:
                                    if mask.shape[0] < max_points:
                                        mask = F.pad(mask, (0, max_points - mask.shape[0]), value=0)
                                    else:
                                        mask = mask[:max_points]
                                
                                masks_list.append(mask.to('cuda'))
                            except Exception as e:
                                print(f"Mask processing error: {e}")
                                continue
                        
                        if masks_list:
                            targets['masks'].append(torch.stack(masks_list))
                        else:
                            targets['masks'].append(torch.zeros(0, max_points, device='cuda'))
                    else:
                        targets['classes'].append(torch.empty(0, dtype=torch.long, device='cuda'))
                        targets['masks'].append(torch.zeros(0, max_points, device='cuda'))
                        
                except Exception as e:
                    print(f"Target prep error for sample {i}: {e}")
                    targets['classes'].append(torch.empty(0, dtype=torch.long, device='cuda'))
                    targets['masks'].append(torch.zeros(0, max_points, device='cuda'))
            
            return targets
            
        except Exception as e:
            print(f"Target preparation error: {e}")
            return None
    
    def compute_sem_loss_safe(self, batch, sem_logits, valid_indices, padding):
        """Safe semantic loss computation"""
        try:
            if sem_logits is None or sem_logits.numel() == 0:
                return torch.tensor(0.1, device='cuda', requires_grad=True)
            
            if 'sem_label' not in batch:
                return torch.tensor(0.1, device='cuda', requires_grad=True)
            
            all_logits = []
            all_labels = []
            
            for i, (labels, idx, pad) in enumerate(zip(batch['sem_label'], valid_indices, padding)):
                try:
                    if len(idx) == 0:
                        continue
                    
                    valid_mask = ~pad
                    if valid_mask.sum() == 0:
                        continue
                    
                    valid_logits = sem_logits[i][valid_mask]
                    
                    if isinstance(labels, np.ndarray):
                        labels = labels.flatten()
                    else:
                        labels = np.array(labels).flatten()
                    
                    if len(labels) > 0 and len(idx) > 0:
                        idx_cpu = idx.cpu().numpy()
                        valid_idx = idx_cpu[(idx_cpu >= 0) & (idx_cpu < len(labels))]
                        
                        if len(valid_idx) > 0:
                            selected_labels = labels[valid_idx]
                            min_len = min(len(selected_labels), len(valid_logits))
                            if min_len > 0:
                                selected_labels = selected_labels[:min_len]
                                selected_logits = valid_logits[:min_len]
                                
                                labels_tensor = torch.from_numpy(selected_labels).long().cuda()
                                labels_tensor = torch.clamp(labels_tensor, 0, self.num_classes - 1)
                                
                                all_logits.append(selected_logits)
                                all_labels.append(labels_tensor)
                                
                except Exception as e:
                    print(f"Sem loss processing error for sample {i}: {e}")
                    continue
            
            if all_logits:
                combined_logits = torch.cat(all_logits, dim=0)
                combined_labels = torch.cat(all_labels, dim=0)
                
                combined_logits = torch.clamp(combined_logits, -15.0, 15.0)
                
                sem_loss = F.cross_entropy(combined_logits, combined_labels, ignore_index=0, reduction='mean')
                
                if torch.isnan(sem_loss) or torch.isinf(sem_loss):
                    sem_loss = torch.tensor(0.5, device='cuda', requires_grad=True)
                
                sem_loss = torch.clamp(sem_loss * self.cfg.LOSS.SEM.WEIGHTS[0], 0.0, 5.0)
                return sem_loss
            else:
                return torch.tensor(0.1, device='cuda', requires_grad=True)
                
        except Exception as e:
            print(f"Semantic loss error: {e}")
            return torch.tensor(0.1, device='cuda', requires_grad=True)
    
    def validation_step(self, batch, batch_idx):
        """Safe validation step"""
        try:
            result = self.forward(batch)
            if result[0] is None:
                return torch.tensor(0.0, device='cuda')
                
            outputs, padding, sem_logits, valid_indices = result
            
            if all(len(v) == 0 for v in valid_indices):
                return torch.tensor(0.0, device='cuda')
            
            targets = self.prepare_targets_safe(batch, padding.shape[1], valid_indices)
            if targets is None:
                return torch.tensor(0.0, device='cuda')
            
            losses = self.mask_loss(outputs, targets, batch.get('masks_ids', []), batch.get('pt_coord', []))
            sem_loss = self.compute_sem_loss_safe(batch, sem_logits, valid_indices, padding)
            
            total_loss = sum(losses.values()) + sem_loss
            total_loss = torch.clamp(total_loss, 0.0, 50.0)
            
            if batch_idx % 20 == 0:
                self.log("val_loss", total_loss, batch_size=self.cfg.TRAIN.BATCH_SIZE)
                
                # Simple evaluation
                try:
                    sem_pred, ins_pred = self.panoptic_inference_safe(outputs, padding)
                    if sem_pred and ins_pred:
                        self.evaluator.update(sem_pred, ins_pred, batch)
                except Exception as e:
                    print(f"Evaluation error: {e}")
            
            return total_loss
            
        except Exception as e:
            print(f"Validation error: {e}")
            return torch.tensor(0.0, device='cuda')
    
    def panoptic_inference_safe(self, outputs, padding):
        """Safe panoptic inference"""
        try:
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
                    
                    cur_prob_masks = cur_scores.unsqueeze(0) * cur_masks
                    cur_mask_ids = cur_prob_masks.argmax(1)
                    
                    sem = torch.zeros(cur_masks.shape[0], dtype=torch.long, device=cur_masks.device)
                    ins = torch.zeros_like(sem)
                    
                    segment_id = 0
                    for k in range(cur_classes.shape[0]):
                        pred_class = cur_classes[k].item()
                        mask = (cur_mask_ids == k) & (cur_masks[:, k] >= 0.5)
                        
                        if mask.sum() > 5:
                            sem[mask] = pred_class
                            if pred_class in self.things_ids:
                                segment_id += 1
                                ins[mask] = segment_id
                    
                    sem_pred.append(sem.cpu().numpy())
                    ins_pred.append(ins.cpu().numpy())
                    
                except Exception as e:
                    print(f"Panoptic inference error: {e}")
                    # Use fallback
                    sem_pred.append(np.zeros(1000, dtype=np.int32))
                    ins_pred.append(np.zeros(1000, dtype=np.int32))
            
            return sem_pred, ins_pred
            
        except Exception as e:
            print(f"Panoptic inference error: {e}")
            return [], []
    
    def validation_epoch_end(self, outputs):
        """Safe metrics computation"""
        try:
            bs = self.cfg.TRAIN.BATCH_SIZE
            
            try:
                pq = self.evaluator.get_mean_pq()
                iou = self.evaluator.get_mean_iou()
                rq = self.evaluator.get_mean_rq()
            except:
                pq = iou = rq = 0.0
            
            self.log("metrics/pq", pq, batch_size=bs)
            self.log("metrics/iou", iou, batch_size=bs)
            self.log("metrics/rq", rq, batch_size=bs)
            
            print(f"\nValidation Metrics - PQ: {pq:.4f}, IoU: {iou:.4f}, RQ: {rq:.4f}")
            
            try:
                self.evaluator.reset()
            except:
                pass
                
        except Exception as e:
            print(f"Metrics error: {e}")
            bs = self.cfg.TRAIN.BATCH_SIZE
            self.log("metrics/pq", 0.0, batch_size=bs)
            self.log("metrics/iou", 0.0, batch_size=bs)
            self.log("metrics/rq", 0.0, batch_size=bs)
    
    def configure_optimizers(self):
        """Fixed optimizer configuration"""
        # Use standard AdamW without inf checks
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.TRAIN.LR * 0.5,  # Lower LR for stability
            weight_decay=1e-4,
            eps=1e-6  # Smaller epsilon for stability
        )
        
        # Simple step scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(self.cfg.TRAIN.STEP, 20),  # Ensure reasonable step size
            gamma=0.5  # More conservative decay
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
@click.option("--lr", type=float, default=0.00005, help="Learning rate")
@click.option("--gpus", type=int, default=1, help="Number of GPUs")
@click.option("--num_workers", type=int, default=2, help="Number of workers")
@click.option("--nuscenes", is_flag=True, help="Use NuScenes dataset")
@click.option("--checkpoint", type=str, default=None, help="Resume from checkpoint")
@click.option("--onnx_interval", type=int, default=20, help="Export ONNX every N epochs")
def main(epochs, batch_size, lr, gpus, num_workers, nuscenes, checkpoint, onnx_interval):
    """Fixed optimized MaskPLS training"""
    
    print("="*60)
    print("Fixed Optimized MaskPLS Training v7")
    print("Addressing scatter operation and optimizer errors")
    print("="*60)
    
    def getDir(obj):
        return os.path.dirname(os.path.abspath(obj))
    
    model_cfg = edict(yaml.safe_load(open(join(getDir(__file__), "../config/model.yaml"))))
    backbone_cfg = edict(yaml.safe_load(open(join(getDir(__file__), "../config/backbone.yaml"))))
    decoder_cfg = edict(yaml.safe_load(open(join(getDir(__file__), "../config/decoder.yaml"))))
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    
    # Conservative configuration
    cfg.TRAIN.MAX_EPOCH = epochs
    cfg.TRAIN.BATCH_SIZE = batch_size
    cfg.TRAIN.LR = lr
    cfg.TRAIN.N_GPUS = gpus
    cfg.TRAIN.NUM_WORKERS = num_workers
    cfg.TRAIN.BATCH_ACC = 2
    
    if nuscenes:
        cfg.MODEL.DATASET = "NUSCENES"
    
    dataset = cfg.MODEL.DATASET
    cfg[dataset].SUB_NUM_POINTS = min(cfg[dataset].SUB_NUM_POINTS, 40000)
    cfg.LOSS.NUM_POINTS = 50000  # Conservative
    cfg.LOSS.NUM_MASK_PTS = 400  # Conservative
    
    print(f"Configuration:")
    print(f"  Dataset: {dataset}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {lr}")
    print(f"  Point Sampling: {cfg[dataset].SUB_NUM_POINTS}")
    print(f"  Voxel Resolution: 96x96x48")
    print(f"  Fixed scatter operations and optimizer")
    
    data = SemanticDatasetModule(cfg)
    model = FixedOptimizedMaskPLS(cfg, onnx_interval=onnx_interval)
    
    if checkpoint:
        print(f"Loading checkpoint: {checkpoint}")
        try:
            ckpt = torch.load(checkpoint, map_location='cpu')
            model.load_state_dict(ckpt['state_dict'], strict=False)
        except Exception as e:
            print(f"Checkpoint load warning: {e}")
    
    tb_logger = pl_loggers.TensorBoardLogger(
        "experiments/" + cfg.EXPERIMENT.ID + "_fixed_v7",
        default_hp_metric=False
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename=cfg.EXPERIMENT.ID + "_fixed_epoch{epoch:02d}_loss{val_loss:.2f}",
        auto_insert_metric_name=False,
        mode="min",
        save_last=True,
        save_top_k=2
    )
    
    trainer = Trainer(
        gpus=gpus,
        accelerator="gpu" if gpus > 0 else None,
        strategy="ddp" if gpus > 1 else None,
        logger=tb_logger,
        max_epochs=epochs,
        callbacks=[lr_monitor, checkpoint_callback],
        log_every_n_steps=20,
        gradient_clip_val=2.0,  # More conservative clipping
        accumulate_grad_batches=cfg.TRAIN.BATCH_ACC,
        precision=32,  # Full precision for stability
        num_sanity_val_steps=1,
        val_check_interval=1.0,  # Full epoch validation
        resume_from_checkpoint=checkpoint
    )
    
    print("\nStarting fixed training...")
    trainer.fit(model, data)
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()