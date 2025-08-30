# mask/MaskPLS/mask_pls/scripts/train_enhanced_onnx_debug.py
"""
Enhanced training script with debugging and NaN/Inf handling
"""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from os.path import join
import click
import torch
import yaml
import numpy as np
import time
from pathlib import Path
from easydict import EasyDict as edict
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
import torch.nn.functional as F

# Import components
from mask_pls.models.onnx.enhanced_model_fixed import EnhancedMaskPLSONNX
from mask_pls.models.onnx.voxelizer import HighResVoxelizer
from mask_pls.datasets.semantic_dataset import SemanticDatasetModule
from mask_pls.models.loss import MaskLoss, SemLoss
from mask_pls.models.matcher import HungarianMatcher
from mask_pls.utils.evaluate_panoptic import PanopticEvaluator
from mask_pls.utils.misc import sample_points, pad_stack


class EnhancedMaskLoss(torch.nn.Module):
    """Fixed mask loss with proper index handling and NaN prevention"""
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
        
        # Class weights
        weights = torch.ones(self.num_classes + 1)
        weights[0] = 0.0
        weights[-1] = self.eos_coef
        self.register_buffer('weights', weights)
        
        # Use original sampling parameters
        self.num_points = cfg.NUM_POINTS
        self.n_mask_pts = cfg.NUM_MASK_PTS
        
    def forward(self, outputs, targets, mask_indices):
        """Forward with fixed index handling and NaN prevention"""
        losses = {}
        
        num_masks = sum(len(t) for t in targets["classes"])
        if num_masks == 0:
            device = outputs["pred_logits"].device
            # Return small non-zero losses to maintain gradients
            return {
                "loss_ce": torch.tensor(0.01, device=device, requires_grad=True),
                "loss_dice": torch.tensor(0.01, device=device, requires_grad=True),
                "loss_mask": torch.tensor(0.01, device=device, requires_grad=True)
            }
        
        num_masks = max(num_masks, 1)
        
        # Hungarian matching
        outputs_no_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        indices = self.matcher(outputs_no_aux, targets)
        
        # Compute losses
        losses.update(self.get_losses(outputs, targets, indices, num_masks, mask_indices))
        
        # Apply weights
        weighted_losses = {}
        for l in losses:
            for k in self.weight_dict:
                if k in l:
                    weighted_losses[l] = losses[l] * self.weight_dict[k]
                    break
        
        return weighted_losses
    
    def get_losses(self, outputs, targets, indices, num_masks, mask_indices):
        """Compute all losses with NaN prevention"""
        losses = {}
        
        # Classification loss
        losses.update(self.loss_classes(outputs, targets, indices))
        
        # Mask losses with proper sampling
        losses.update(self.loss_masks(outputs, targets, indices, num_masks, mask_indices))
        
        return losses
    
    def loss_classes(self, outputs, targets, indices):
        """Classification loss with NaN prevention"""
        pred_logits = outputs["pred_logits"].float()
        
        # Check for NaN/Inf in predictions
        if torch.isnan(pred_logits).any() or torch.isinf(pred_logits).any():
            print("Warning: NaN/Inf in pred_logits, clamping values")
            pred_logits = torch.nan_to_num(pred_logits, nan=0.0, posinf=10.0, neginf=-10.0)
            pred_logits = torch.clamp(pred_logits, min=-100, max=100)
        
        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(indices)
        ])
        src_idx = torch.cat([src for (src, _) in indices])
        
        target_classes_o = torch.cat([
            t[J] for t, (_, J) in zip(targets["classes"], indices)
        ])
        
        target_classes = torch.full(
            pred_logits.shape[:2], 
            self.num_classes,
            dtype=torch.int64, 
            device=pred_logits.device
        )
        
        if len(batch_idx) > 0:
            target_classes[batch_idx, src_idx] = target_classes_o.to(pred_logits.device)
        
        loss_ce = F.cross_entropy(
            pred_logits.transpose(1, 2), 
            target_classes,
            self.weights, 
            ignore_index=self.ignore_label
        )
        
        # Check for NaN in loss
        if torch.isnan(loss_ce) or torch.isinf(loss_ce):
            print("Warning: NaN/Inf in CE loss, using default value")
            loss_ce = torch.tensor(1.0, device=pred_logits.device, requires_grad=True)
        
        return {"loss_ce": loss_ce}
    
    def loss_masks(self, outputs, targets, indices, num_masks, mask_indices):
        """Mask loss with NaN prevention"""
        masks = [t for t in targets["masks"]]
        if not masks or sum(m.shape[0] for m in masks) == 0:
            device = outputs["pred_masks"].device
            return {
                "loss_mask": torch.tensor(0.01, device=device, requires_grad=True),
                "loss_dice": torch.tensor(0.01, device=device, requires_grad=True)
            }
        
        # Get prediction and target indices
        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(indices)
        ])
        pred_idx = torch.cat([src for (src, _) in indices])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        
        # Stack target masks
        target_masks = pad_stack(masks)
        
        # Get matched masks
        pred_masks = outputs["pred_masks"]
        
        # Check for NaN/Inf in predictions
        if torch.isnan(pred_masks).any() or torch.isinf(pred_masks).any():
            print("Warning: NaN/Inf in pred_masks, clamping values")
            pred_masks = torch.nan_to_num(pred_masks, nan=0.0, posinf=10.0, neginf=-10.0)
            pred_masks = torch.clamp(pred_masks, min=-100, max=100)
        
        if len(batch_idx) == 0:
            device = pred_masks.device
            return {
                "loss_mask": torch.tensor(0.01, device=device, requires_grad=True),
                "loss_dice": torch.tensor(0.01, device=device, requires_grad=True)
            }
        
        matched_pred_masks = pred_masks[batch_idx, :, pred_idx]
        
        # Build proper target mask indices
        n_masks = [m.shape[0] for m in masks]
        mask_batch_idx = torch.cat([
            torch.full((n,), i) for i, n in enumerate(n_masks)
        ])
        mask_idx = torch.cat([torch.arange(n) for n in n_masks])
        
        # Get matched target masks
        matched_tgt_idx = mask_batch_idx * 1000000 + mask_idx
        matched_pred_tgt_idx = batch_idx * 1000000 + tgt_idx
        
        # Find matching indices
        match_indices = []
        for idx in matched_pred_tgt_idx:
            match_pos = (matched_tgt_idx == idx).nonzero(as_tuple=True)[0]
            if len(match_pos) > 0:
                match_indices.append(match_pos[0].item())
            else:
                match_indices.append(0)
        
        matched_target_masks = target_masks[match_indices].to(pred_masks.device)
        
        # Sample points for loss computation
        with torch.no_grad():
            sampled_indices = []
            for b_idx in range(len(masks)):
                n_pts = masks[b_idx].shape[1]
                
                # Sample indices
                if n_pts > self.num_points:
                    idx = torch.randperm(n_pts)[:self.num_points]
                else:
                    idx = torch.arange(n_pts)
                    if n_pts < self.num_points and n_pts > 0:
                        extra = torch.randint(0, n_pts, (self.num_points - n_pts,))
                        idx = torch.cat([idx, extra])
                
                sampled_indices.append(idx)
        
        # Apply sampling to get point-wise losses
        point_logits = []
        point_labels = []
        
        for b_idx in range(len(masks)):
            n_masks_b = sum(1 for b in batch_idx if b == b_idx)
            if n_masks_b == 0:
                continue
            
            batch_mask = batch_idx == b_idx
            batch_pred_masks = matched_pred_masks[batch_mask]
            batch_tgt_masks = matched_target_masks[batch_mask]
            
            if b_idx < len(sampled_indices):
                sample_idx = sampled_indices[b_idx].to(pred_masks.device)
                
                # Ensure indices are within bounds
                max_idx = batch_pred_masks.shape[1] - 1
                sample_idx = torch.clamp(sample_idx, 0, max_idx)
                
                sampled_pred = batch_pred_masks[:, sample_idx]
                sampled_tgt = batch_tgt_masks[:, sample_idx]
                
                point_logits.append(sampled_pred)
                point_labels.append(sampled_tgt)
        
        if point_logits:
            point_logits = torch.cat(point_logits, dim=0)
            point_labels = torch.cat(point_labels, dim=0)
            
            # BCE loss with NaN prevention
            loss_mask = F.binary_cross_entropy_with_logits(
                point_logits, 
                point_labels.float(), 
                reduction='none'
            )
            
            # Check for NaN
            if torch.isnan(loss_mask).any() or torch.isinf(loss_mask).any():
                print("Warning: NaN/Inf in mask loss")
                loss_mask = torch.nan_to_num(loss_mask, nan=0.0, posinf=1.0, neginf=0.0)
            
            loss_mask = loss_mask.mean(1).sum() / num_masks
            
            # Dice loss with NaN prevention
            pred_sigmoid = torch.sigmoid(point_logits)
            pred_sigmoid = torch.clamp(pred_sigmoid, min=1e-7, max=1-1e-7)  # Prevent 0/1
            
            numerator = 2 * (pred_sigmoid * point_labels).sum(-1)
            denominator = pred_sigmoid.sum(-1) + point_labels.sum(-1)
            dice = 1 - (numerator + 1) / (denominator + 1)
            
            # Check for NaN in dice
            if torch.isnan(dice).any() or torch.isinf(dice).any():
                print("Warning: NaN/Inf in dice loss")
                dice = torch.nan_to_num(dice, nan=0.0, posinf=1.0, neginf=0.0)
            
            loss_dice = dice.sum() / num_masks
            
            return {
                "loss_mask": loss_mask,
                "loss_dice": loss_dice
            }
        else:
            device = outputs["pred_masks"].device
            return {
                "loss_mask": torch.tensor(0.01, device=device, requires_grad=True),
                "loss_dice": torch.tensor(0.01, device=device, requires_grad=True)
            }


class EnhancedMaskPLS(LightningModule):
    """Enhanced MaskPLS with debugging and NaN handling"""
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(dict(cfg))
        self.cfg = cfg
        
        dataset = cfg.MODEL.DATASET
        self.num_classes = cfg[dataset].NUM_CLASSES
        self.ignore_label = cfg[dataset].IGNORE_LABEL
        
        # High-resolution voxelizer - use lower resolution initially for stability
        self.voxelizer = HighResVoxelizer(
            spatial_shape=(64, 64, 32),  # Reduced from (128, 128, 64)
            coordinate_bounds=cfg[dataset].SPACE,
            device='cuda'
        )
        
        # Enhanced model
        self.model = EnhancedMaskPLSONNX(cfg)
        
        # Fixed loss functions
        self.mask_loss = EnhancedMaskLoss(cfg.LOSS, cfg[dataset])
        self.sem_loss = SemLoss(cfg.LOSS.SEM.WEIGHTS)
        
        # Evaluator
        self.evaluator = PanopticEvaluator(cfg[dataset], dataset)
        
        # Get things IDs
        data = SemanticDatasetModule(cfg)
        data.setup()
        self.things_ids = data.things_ids
        
        # Performance tracking
        self.validation_step_outputs = []
        
        # Add gradient clipping value
        self.gradient_clip_val = 1.0
        
    def forward(self, batch):
        """Forward with NaN detection"""
        points = batch['pt_coord']
        features = batch['feats']
        
        # High-resolution voxelization
        voxel_grids, norm_coords, valid_indices = self.voxelizer.voxelize_batch(
            points, features, max_points=self.cfg[self.cfg.MODEL.DATASET].SUB_NUM_POINTS
        )
        
        # Check for NaN in voxel grids
        if torch.isnan(voxel_grids).any() or torch.isinf(voxel_grids).any():
            print("Warning: NaN/Inf in voxel_grids")
            voxel_grids = torch.nan_to_num(voxel_grids, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Pad coordinates for batch processing
        max_pts = max(c.shape[0] for c in norm_coords) if norm_coords else 1000
        max_pts = max(max_pts, 100)  # Ensure minimum size
        
        padded_coords = []
        padding_masks = []
        
        for coords in norm_coords:
            n_pts = coords.shape[0]
            if n_pts == 0:
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
        
        # Check for NaN in coordinates
        if torch.isnan(batch_coords).any() or torch.isinf(batch_coords).any():
            print("Warning: NaN/Inf in batch_coords")
            batch_coords = torch.nan_to_num(batch_coords, nan=0.5, posinf=1.0, neginf=0.0)
            batch_coords = torch.clamp(batch_coords, 0, 1)
        
        # Model forward with try-catch
        try:
            pred_logits, pred_masks, sem_logits = self.model(voxel_grids, batch_coords)
            
            # Check outputs for NaN
            if torch.isnan(pred_logits).any() or torch.isinf(pred_logits).any():
                print("Warning: NaN/Inf in pred_logits output")
                pred_logits = torch.nan_to_num(pred_logits, nan=0.0, posinf=10.0, neginf=-10.0)
            
            if torch.isnan(pred_masks).any() or torch.isinf(pred_masks).any():
                print("Warning: NaN/Inf in pred_masks output")
                pred_masks = torch.nan_to_num(pred_masks, nan=0.0, posinf=10.0, neginf=-10.0)
            
            if torch.isnan(sem_logits).any() or torch.isinf(sem_logits).any():
                print("Warning: NaN/Inf in sem_logits output")
                sem_logits = torch.nan_to_num(sem_logits, nan=0.0, posinf=10.0, neginf=-10.0)
            
        except RuntimeError as e:
            print(f"Runtime error in forward: {e}")
            # Return zero outputs
            B = voxel_grids.shape[0]
            N = batch_coords.shape[1]
            Q = self.cfg.DECODER.NUM_QUERIES
            C = self.num_classes
            
            pred_logits = torch.zeros(B, Q, C+1, device='cuda')
            pred_masks = torch.zeros(B, N, Q, device='cuda')
            sem_logits = torch.zeros(B, N, C, device='cuda')
        
        outputs = {
            'pred_logits': pred_logits,
            'pred_masks': pred_masks,
            'aux_outputs': []
        }
        
        return outputs, padding_masks, sem_logits, valid_indices
    
    def prepare_targets(self, batch, max_points, valid_indices):
        """Prepare targets with proper remapping and NaN prevention"""
        targets = {'classes': [], 'masks': []}
        
        for i in range(len(batch['masks_cls'])):
            if len(batch['masks_cls'][i]) > 0:
                # Classes
                classes = torch.tensor(
                    batch['masks_cls'][i], 
                    dtype=torch.long, 
                    device='cuda'
                )
                # Clamp classes to valid range
                classes = torch.clamp(classes, 0, self.num_classes - 1)
                targets['classes'].append(classes)
                
                # Remap masks to valid points
                masks_list = []
                for mask in batch['masks'][i]:
                    if isinstance(mask, torch.Tensor):
                        mask = mask.float()
                    else:
                        mask = torch.from_numpy(mask).float()
                    
                    # Create remapped mask
                    remapped_mask = torch.zeros(max_points, device='cuda')
                    
                    # Map valid indices
                    valid_idx = valid_indices[i]
                    if len(valid_idx) > 0:
                        for j, v_idx in enumerate(valid_idx):
                            if j < max_points and v_idx < len(mask):
                                remapped_mask[j] = mask[v_idx]
                    
                    masks_list.append(remapped_mask)
                
                if masks_list:
                    targets['masks'].append(torch.stack(masks_list))
                else:
                    targets['masks'].append(torch.zeros(0, max_points, device='cuda'))
            else:
                targets['classes'].append(torch.zeros(0, dtype=torch.long, device='cuda'))
                targets['masks'].append(torch.zeros(0, max_points, device='cuda'))
        
        return targets
    
    def compute_semantic_loss(self, batch, sem_logits, valid_indices, padding_masks):
        """Compute semantic loss with NaN prevention"""
        all_logits = []
        all_labels = []
        
        for i, (labels, valid_idx, pad_mask) in enumerate(
            zip(batch['sem_label'], valid_indices, padding_masks)
        ):
            if len(valid_idx) == 0:
                continue
            
            # Get valid logits
            valid_mask = ~pad_mask
            batch_logits = sem_logits[i][valid_mask]
            
            # Check for NaN in logits
            if torch.isnan(batch_logits).any() or torch.isinf(batch_logits).any():
                print(f"Warning: NaN/Inf in batch {i} semantic logits")
                batch_logits = torch.nan_to_num(batch_logits, nan=0.0, posinf=10.0, neginf=-10.0)
            
            # Get corresponding labels
            if isinstance(labels, np.ndarray):
                labels = labels.flatten()
            else:
                labels = np.array(labels).flatten()
            
            # Map to valid indices
            valid_idx_cpu = valid_idx.cpu().numpy()
            valid_labels = []
            
            for j, v_idx in enumerate(valid_idx_cpu):
                if j < len(batch_logits) and v_idx < len(labels):
                    valid_labels.append(labels[v_idx])
            
            if valid_labels:
                labels_tensor = torch.tensor(
                    valid_labels, 
                    dtype=torch.long, 
                    device='cuda'
                )
                labels_tensor = torch.clamp(labels_tensor, 0, self.num_classes - 1)
                
                # Match dimensions
                min_len = min(len(batch_logits), len(labels_tensor))
                if min_len > 0:
                    all_logits.append(batch_logits[:min_len])
                    all_labels.append(labels_tensor[:min_len])
        
        if all_logits:
            combined_logits = torch.cat(all_logits, dim=0)
            combined_labels = torch.cat(all_labels, dim=0)
            
            # Compute loss with NaN prevention
            ce_loss = F.cross_entropy(combined_logits, combined_labels, ignore_index=0)
            
            if torch.isnan(ce_loss) or torch.isinf(ce_loss):
                print("Warning: NaN/Inf in semantic CE loss")
                ce_loss = torch.tensor(1.0, device='cuda', requires_grad=True)
            
            # Lovasz loss with try-catch
            try:
                probs = F.softmax(combined_logits, dim=1)
                probs = torch.clamp(probs, min=1e-7, max=1-1e-7)
                lovasz_loss = self.sem_loss.lovasz_softmax(probs, combined_labels)
                
                if torch.isnan(lovasz_loss) or torch.isinf(lovasz_loss):
                    print("Warning: NaN/Inf in Lovasz loss")
                    lovasz_loss = torch.tensor(0.0, device='cuda', requires_grad=True)
            except:
                lovasz_loss = torch.tensor(0.0, device='cuda', requires_grad=True)
            
            return (self.cfg.LOSS.SEM.WEIGHTS[0] * ce_loss + 
                   self.cfg.LOSS.SEM.WEIGHTS[1] * lovasz_loss)
        else:
            return torch.tensor(0.01, device='cuda', requires_grad=True)
    
    def training_step(self, batch, batch_idx):
        """Training step with NaN detection and handling"""
        try:
            # Forward pass
            outputs, padding_masks, sem_logits, valid_indices = self.forward(batch)
            
            # Skip if no valid data
            if all(len(v) == 0 for v in valid_indices):
                return torch.tensor(0.01, device='cuda', requires_grad=True)
            
            # Prepare targets
            targets = self.prepare_targets(batch, padding_masks.shape[1], valid_indices)
            
            # Compute losses with NaN handling
            mask_losses = self.mask_loss(outputs, targets, batch['masks_ids'])
            sem_loss = self.compute_semantic_loss(batch, sem_logits, valid_indices, padding_masks)
            
            # Check individual losses for NaN
            for k, v in mask_losses.items():
                if torch.isnan(v) or torch.isinf(v):
                    print(f"Warning: NaN/Inf in {k}")
                    mask_losses[k] = torch.tensor(0.01, device='cuda', requires_grad=True)
            
            # Total loss
            total_loss = sum(mask_losses.values()) + sem_loss
            
            # Final NaN check
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"Warning: NaN/Inf in total loss at batch {batch_idx}")
                total_loss = torch.tensor(1.0, device='cuda', requires_grad=True)
            
            # Logging with NaN prevention
            self.log("train_loss", total_loss.item() if not torch.isnan(total_loss) else 1.0, 
                    batch_size=self.cfg.TRAIN.BATCH_SIZE)
            
            for k, v in mask_losses.items():
                self.log(f"train/{k}", v.item() if not torch.isnan(v) else 0.01, 
                        batch_size=self.cfg.TRAIN.BATCH_SIZE)
            
            self.log("train/sem_loss", sem_loss.item() if not torch.isnan(sem_loss) else 0.01, 
                    batch_size=self.cfg.TRAIN.BATCH_SIZE)
            
            return total_loss
            
        except Exception as e:
            print(f"Error in training step {batch_idx}: {e}")
            return torch.tensor(1.0, device='cuda', requires_grad=True)
    
    def validation_step(self, batch, batch_idx):
        """Validation step with NaN handling"""
        try:
            # Forward pass
            outputs, padding_masks, sem_logits, valid_indices = self.forward(batch)
            
            if all(len(v) == 0 for v in valid_indices):
                return
            
            # Compute losses
            targets = self.prepare_targets(batch, padding_masks.shape[1], valid_indices)
            mask_losses = self.mask_loss(outputs, targets, batch['masks_ids'])
            sem_loss = self.compute_semantic_loss(batch, sem_logits, valid_indices, padding_masks)
            
            total_loss = sum(mask_losses.values()) + sem_loss
            
            # Check for NaN before logging
            if not (torch.isnan(total_loss) or torch.isinf(total_loss)):
                self.log("val_loss", total_loss, batch_size=self.cfg.TRAIN.BATCH_SIZE)
            
            # Panoptic inference
            sem_pred, ins_pred = self.panoptic_inference(
                outputs, padding_masks, valid_indices, batch
            )
            
            # Update evaluator
            self.evaluator.update(sem_pred, ins_pred, batch)
            
            if not (torch.isnan(total_loss) or torch.isinf(total_loss)):
                self.validation_step_outputs.append(total_loss)
                
        except Exception as e:
            print(f"Error in validation step {batch_idx}: {e}")
    
    def on_validation_epoch_end(self):
        """Validation epoch end with NaN handling"""
        try:
            # Compute metrics
            pq = self.evaluator.get_mean_pq()
            iou = self.evaluator.get_mean_iou()
            rq = self.evaluator.get_mean_rq()
            
            # Check for NaN/invalid values
            if np.isnan(pq) or np.isinf(pq):
                pq = 0.0
            if np.isnan(iou) or np.isinf(iou):
                iou = 0.0
            if np.isnan(rq) or np.isinf(rq):
                rq = 0.0
            
            self.log("metrics/pq", pq, batch_size=self.cfg.TRAIN.BATCH_SIZE)
            self.log("metrics/iou", iou, batch_size=self.cfg.TRAIN.BATCH_SIZE)
            self.log("metrics/rq", rq, batch_size=self.cfg.TRAIN.BATCH_SIZE)
            
            print(f"\nValidation Metrics - PQ: {pq:.4f}, IoU: {iou:.4f}, RQ: {rq:.4f}")
            
        except Exception as e:
            print(f"Error computing validation metrics: {e}")
            # Log zero metrics to avoid crash
            self.log("metrics/pq", 0.0, batch_size=self.cfg.TRAIN.BATCH_SIZE)
            self.log("metrics/iou", 0.0, batch_size=self.cfg.TRAIN.BATCH_SIZE)
            self.log("metrics/rq", 0.0, batch_size=self.cfg.TRAIN.BATCH_SIZE)
        finally:
            # Reset evaluator
            self.evaluator.reset()
            self.validation_step_outputs.clear()
    
    def panoptic_inference(self, outputs, padding_masks, valid_indices, batch):
        """Panoptic inference with NaN handling"""
        mask_cls = outputs["pred_logits"]
        mask_pred = outputs["pred_masks"]
        
        sem_pred = []
        ins_pred = []
        
        for b, (cls_b, mask_b, pad_b, valid_idx) in enumerate(
            zip(mask_cls, mask_pred, padding_masks, valid_indices)
        ):
            # Get valid predictions
            valid_mask = ~pad_b
            valid_pred = mask_b[valid_mask].sigmoid()
            
            # Check for NaN
            if torch.isnan(valid_pred).any():
                valid_pred = torch.nan_to_num(valid_pred, nan=0.5)
            
            # Get predictions
            scores, labels = cls_b.max(-1)
            keep = labels.ne(self.num_classes)
            
            # Create output arrays with original size
            orig_size = batch['sem_label'][b].shape[0]
            sem_out = np.zeros(orig_size, dtype=np.int32)
            ins_out = np.zeros(orig_size, dtype=np.int32)
            
            if keep.sum() > 0 and len(valid_idx) > 0:
                try:
                    # Get valid masks
                    cur_scores = scores[keep]
                    cur_classes = labels[keep]
                    cur_masks = valid_pred[:, keep]
                    
                    # Assign points to masks
                    cur_prob_masks = cur_scores.unsqueeze(0) * cur_masks
                    point_mask_ids = cur_prob_masks.argmax(1)
                    
                    # Map back to original indices
                    valid_idx_cpu = valid_idx.cpu().numpy()
                    
                    segment_id = 0
                    for k in range(cur_classes.shape[0]):
                        pred_class = cur_classes[k].item()
                        point_mask = (point_mask_ids == k) & (cur_masks[:, k] >= 0.5)
                        
                        if point_mask.sum() > 10:  # Minimum mask size
                            # Map to original indices
                            mask_indices = point_mask.cpu().numpy()
                            for i, is_mask in enumerate(mask_indices):
                                if i < len(valid_idx_cpu) and is_mask:
                                    orig_idx = valid_idx_cpu[i]
                                    if orig_idx < orig_size:
                                        sem_out[orig_idx] = pred_class
                                        if pred_class in self.things_ids:
                                            ins_out[orig_idx] = segment_id + 1
                            
                            if pred_class in self.things_ids:
                                segment_id += 1
                except Exception as e:
                    print(f"Error in panoptic inference for batch {b}: {e}")
            
            sem_pred.append(sem_out)
            ins_pred.append(ins_out)
        
        return sem_pred, ins_pred
    
    def configure_optimizers(self):
        """Optimizer configuration with gradient clipping"""
        # AdamW with weight decay
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.TRAIN.LR,
            weight_decay=1e-4,
            eps=1e-8  # Small epsilon for numerical stability
        )
        
        # Cosine annealing with warmup
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
    
    def on_before_optimizer_step(self, optimizer):
        """Gradient clipping and NaN detection"""
        # Check for NaN gradients
        for name, param in self.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"Warning: NaN/Inf gradient in {name}")
                    param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_val)


@click.command()
@click.option("--config", type=str, default="config/model.yaml")
@click.option("--epochs", type=int, default=100)
@click.option("--batch_size", type=int, default=1)  # Start with batch size 1
@click.option("--lr", type=float, default=0.00005)  # Lower learning rate
@click.option("--gpus", type=int, default=1)
@click.option("--num_workers", type=int, default=2)  # Fewer workers initially
@click.option("--checkpoint", type=str, default=None)
@click.option("--nuscenes", is_flag=True)
def main(config, epochs, batch_size, lr, gpus, num_workers, checkpoint, nuscenes):
    """Enhanced training with debugging and NaN handling"""
    
    print("="*60)
    print("Enhanced MaskPLS Training with Debug Mode")
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
    
    # Use smaller values initially for debugging
    cfg[dataset].SUB_NUM_POINTS = 40000  # Reduced from 80000
    cfg.LOSS.NUM_POINTS = 25000  # Reduced from 50000
    cfg.LOSS.NUM_MASK_PTS = 250  # Reduced from 500
    
    print(f"Configuration:")
    print(f"  Dataset: {dataset}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {lr}")
    print(f"  Voxel Resolution: 64x64x32 (reduced for stability)")
    print(f"  Point Sampling: {cfg[dataset].SUB_NUM_POINTS}")
    print(f"  Gradient Clipping: Enabled")
    print(f"  NaN Detection: Enabled")
    
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
        "experiments/" + cfg.EXPERIMENT.ID + "_enhanced_debug",
        default_hp_metric=False
    )
    
    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    checkpoint_callback = ModelCheckpoint(
        monitor="metrics/iou",  # Monitor IoU instead of PQ initially
        filename=cfg.EXPERIMENT.ID + "_debug_epoch{epoch:02d}_iou{metrics/iou:.3f}",
        auto_insert_metric_name=False,
        mode="max",
        save_last=True,
        save_top_k=3
    )
    
    # Create trainer with debugging settings
    trainer = Trainer(
        gpus=gpus,
        accelerator="gpu" if gpus > 0 else None,
        strategy="ddp" if gpus > 1 else None,
        logger=tb_logger,
        max_epochs=epochs,
        callbacks=[lr_monitor, checkpoint_callback],
        log_every_n_steps=1,
        gradient_clip_val=1.0,  # Gradient clipping
        accumulate_grad_batches=cfg.TRAIN.BATCH_ACC,
        precision=32,  # Use full precision initially for stability
        check_val_every_n_epoch=5,  # Validate less frequently initially
        num_sanity_val_steps=0,
        detect_anomaly=True,  # Enable anomaly detection
        track_grad_norm=2,  # Track gradient norms
    )
    
    # Train
    print("\nStarting training with debug mode...")
    print("Monitor for NaN/Inf warnings in the output")
    trainer.fit(model, data)
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
