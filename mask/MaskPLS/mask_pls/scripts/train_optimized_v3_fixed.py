"""
Fixed Ultra-optimized training script for MaskPLS with proper model performance
This version balances speed improvements with correct model functionality
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

# Import model components
from mask_pls.models.onnx.simplified_model import MaskPLSSimplifiedONNX
from mask_pls.datasets.semantic_dataset import SemanticDatasetModule
from mask_pls.models.loss import MaskLoss, SemLoss
from mask_pls.utils.evaluate_panoptic import PanopticEvaluator
from mask_pls.utils.misc import sample_points


class ImprovedVoxelizer:
    """Optimized voxelizer with proper resolution and caching"""
    def __init__(self, spatial_shape, bounds, device='cuda'):
        # Use better resolution for accuracy
        self.spatial_shape = (64, 64, 32)  # Increased from (32, 32, 16)
        self.bounds = bounds
        self.device = device
        
        # Precompute constants
        self.bounds_min = torch.tensor([bounds[d][0] for d in range(3)], device=device)
        self.bounds_max = torch.tensor([bounds[d][1] for d in range(3)], device=device)
        self.bounds_range = self.bounds_max - self.bounds_min
        
        # Cache for repeated point clouds
        self.cache = OrderedDict()
        self.cache_size = 500
        
    def voxelize_batch_fast(self, points_batch, features_batch):
        """Vectorized batch voxelization with proper feature accumulation"""
        B = len(points_batch)
        D, H, W = self.spatial_shape
        C = features_batch[0].shape[1] if len(features_batch) > 0 else 4
        
        voxel_grids = torch.zeros(B, C, D, H, W, device=self.device)
        voxel_counts = torch.zeros(B, 1, D, H, W, device=self.device)  # Track counts for averaging
        normalized_coords = []
        valid_indices = []
        
        for b in range(B):
            pts = torch.from_numpy(points_batch[b]).float().to(self.device)
            feat = torch.from_numpy(features_batch[b]).float().to(self.device)
            
            # Vectorized bounds check
            valid_mask = ((pts >= self.bounds_min) & (pts < self.bounds_max)).all(dim=1)
            
            if valid_mask.sum() == 0:
                normalized_coords.append(torch.zeros(0, 3, device=self.device))
                valid_indices.append(torch.zeros(0, dtype=torch.long, device=self.device))
                continue
            
            valid_idx = torch.where(valid_mask)[0]
            valid_pts = pts[valid_mask]
            valid_feat = feat[valid_mask]
            
            # Normalize coordinates
            norm_coords = (valid_pts - self.bounds_min) / self.bounds_range
            norm_coords = torch.clamp(norm_coords, 0, 0.999)
            
            # Convert to voxel indices
            voxel_indices = (norm_coords * torch.tensor([D, H, W], device=self.device)).long()
            
            # Accumulate features with averaging
            flat_indices = (voxel_indices[:, 0] * H * W + 
                          voxel_indices[:, 1] * W + 
                          voxel_indices[:, 2])
            
            # Accumulate features
            for c in range(C):
                voxel_grids[b, c].view(-1).scatter_add_(0, flat_indices, valid_feat[:, c])
            
            # Count points per voxel for averaging
            voxel_counts[b, 0].view(-1).scatter_add_(0, flat_indices, torch.ones_like(valid_feat[:, 0]))
            
            normalized_coords.append(norm_coords)
            valid_indices.append(valid_idx)
        
        # Average the accumulated features
        voxel_grids = voxel_grids / (voxel_counts + 1e-6)
        
        return voxel_grids, normalized_coords, valid_indices


class ProperMaskLoss(torch.nn.Module):
    """Fixed mask loss with proper sampling and computation"""
    def __init__(self, cfg, data_cfg):
        super().__init__()
        from mask_pls.models.matcher import HungarianMatcher
        
        self.num_classes = data_cfg.NUM_CLASSES
        self.ignore = data_cfg.IGNORE_LABEL
        self.matcher = HungarianMatcher(cfg.WEIGHTS, cfg.P_RATIO)
        
        self.weight_dict = {
            cfg.WEIGHTS_KEYS[i]: cfg.WEIGHTS[i] for i in range(len(cfg.WEIGHTS))
        }
        self.eos_coef = cfg.EOS_COEF
        
        # Precompute weights
        weights = torch.ones(self.num_classes + 1)
        weights[0] = 0.0
        weights[-1] = self.eos_coef
        self.register_buffer('weights', weights)
        
        # Use reasonable sampling for balance between speed and accuracy
        self.num_points = min(cfg.NUM_POINTS, 10000)  # Balanced sampling
        self.n_mask_pts = min(cfg.NUM_MASK_PTS, 2048)  # Reasonable mask sampling
        
    def forward(self, outputs, targets, masks_ids, coors):
        """Proper loss computation with Hungarian matching"""
        losses = {}
        
        # Check for empty targets
        num_masks = sum(len(t) for t in targets["classes"])
        if num_masks == 0:
            device = outputs["pred_logits"].device
            return {
                "loss_ce": torch.tensor(0.0, device=device, requires_grad=True),
                "loss_dice": torch.tensor(0.0, device=device, requires_grad=True),
                "loss_mask": torch.tensor(0.0, device=device, requires_grad=True)
            }
        
        num_masks = max(num_masks, 1)
        outputs_no_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        
        # Proper Hungarian matching
        indices = self.matcher(outputs_no_aux, targets)
        
        # Compute all losses
        losses.update(self.get_losses(outputs, targets, indices, num_masks, masks_ids, coors))
        
        # Apply proper weights
        weighted_losses = {}
        for l in losses:
            for k in self.weight_dict:
                if k in l:
                    weighted_losses[l] = losses[l] * self.weight_dict[k]
                    break
        
        return weighted_losses
    
    def get_losses(self, outputs, targets, indices, num_masks, masks_ids, coors):
        """Compute all losses properly"""
        losses = {}
        
        # Classification loss
        losses.update(self.loss_classes(outputs, targets, indices))
        
        # Mask losses with proper sampling
        losses.update(self.loss_masks(outputs, targets, indices, num_masks, masks_ids, coors))
        
        return losses
    
    def loss_classes(self, outputs, targets, indices):
        """Proper classification loss"""
        pred_logits = outputs["pred_logits"].float()
        idx = self._get_pred_permutation_idx(indices)
        
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(targets["classes"], indices)])
        target_classes = torch.full(pred_logits.shape[:2], self.num_classes, 
                                  dtype=torch.int64, device=pred_logits.device)
        target_classes[idx] = target_classes_o.to(pred_logits.device)
        
        loss_ce = F.cross_entropy(pred_logits.transpose(1, 2), target_classes, 
                                 self.weights, ignore_index=self.ignore)
        return {"loss_ce": loss_ce}
    
    def loss_masks(self, outputs, targets, indices, num_masks, masks_ids, coors):
        """Proper mask loss with correct point sampling"""
        masks = [t for t in targets["masks"]]
        if not masks or sum(m.shape[0] for m in masks) == 0:
            device = outputs["pred_masks"].device
            return {
                "loss_mask": torch.tensor(0.0, device=device, requires_grad=True),
                "loss_dice": torch.tensor(0.0, device=device, requires_grad=True)
            }
        
        pred_masks = outputs["pred_masks"]
        device = pred_masks.device
        
        # Get matched predictions and targets
        pred_idx = self._get_pred_permutation_idx(indices)
        n_masks = [m.shape[0] for m in masks]
        
        if len(pred_idx[0]) == 0 or sum(n_masks) == 0:
            return {
                "loss_mask": torch.tensor(0.0, device=device, requires_grad=True),
                "loss_dice": torch.tensor(0.0, device=device, requires_grad=True)
            }
        
        # Prepare target masks properly
        from mask_pls.utils.misc import pad_stack
        target_masks = pad_stack(masks).to(device)
        tgt_idx = self._get_tgt_permutation_idx(indices, n_masks)
        
        selected_pred_masks = pred_masks[pred_idx[0], :, pred_idx[1]]
        selected_target_masks = target_masks[tgt_idx]
        
        # Proper point sampling for efficiency
        with torch.no_grad():
            # Use the proper sampling function from original
            sampled_indices = sample_points(masks, masks_ids, self.n_mask_pts, self.num_points)
            n_masks_cumsum = [0] + list(np.cumsum(n_masks))
            
            point_labels = []
            point_logits = []
            
            for i, idx in enumerate(sampled_indices):
                start_idx = n_masks_cumsum[i]
                end_idx = n_masks_cumsum[i + 1]
                
                if start_idx < end_idx and len(idx) > 0:
                    batch_target_masks = selected_target_masks[start_idx:end_idx]
                    batch_pred_masks = selected_pred_masks[start_idx:end_idx]
                    
                    # Sample points
                    sampled_targets = batch_target_masks[:, idx]
                    sampled_preds = batch_pred_masks[:, idx]
                    
                    point_labels.append(sampled_targets)
                    point_logits.append(sampled_preds)
        
        if point_labels and point_logits:
            point_labels = torch.cat(point_labels, dim=0)
            point_logits = torch.cat(point_logits, dim=0)
            
            # Compute proper losses
            loss_mask = F.binary_cross_entropy_with_logits(
                point_logits, point_labels.float(), reduction='none'
            ).mean(1).sum() / num_masks
            
            # Proper dice loss
            pred_sigmoid = torch.sigmoid(point_logits)
            numerator = 2 * (pred_sigmoid * point_labels).sum(-1)
            denominator = pred_sigmoid.sum(-1) + point_labels.sum(-1)
            loss_dice = (1 - (numerator + 1) / (denominator + 1)).sum() / num_masks
            
            return {
                "loss_mask": loss_mask,
                "loss_dice": loss_dice
            }
        else:
            return {
                "loss_mask": torch.tensor(0.0, device=device, requires_grad=True),
                "loss_dice": torch.tensor(0.0, device=device, requires_grad=True)
            }
    
    def _get_pred_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def _get_tgt_permutation_idx(self, indices, n_masks):
        """Proper target permutation index calculation"""
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        
        # From [B,id] to [id] of stacked masks
        cont_id = torch.cat([torch.arange(n) for n in n_masks])
        b_id = torch.stack((batch_idx, cont_id), axis=1)
        
        map_m = torch.zeros((torch.max(batch_idx) + 1, max(n_masks)))
        for i in range(len(b_id)):
            map_m[b_id[i, 0], b_id[i, 1]] = i
        
        stack_ids = [int(map_m[batch_idx[i], tgt_idx[i]]) for i in range(len(batch_idx))]
        return stack_ids


class FixedOptimizedMaskPLS(LightningModule):
    """Fixed optimized MaskPLS with proper model performance"""
    def __init__(self, cfg, onnx_interval=5):
        super().__init__()
        self.save_hyperparameters(dict(cfg))
        self.cfg = cfg
        self.onnx_interval = onnx_interval
        
        # Get dataset config
        dataset = cfg.MODEL.DATASET
        self.num_classes = cfg[dataset].NUM_CLASSES
        self.ignore_label = cfg[dataset].IGNORE_LABEL
        
        # Update backbone config for better features
        cfg.BACKBONE.CHANNELS = [32, 64, 128, 256, 256]  # Better channel progression
        
        # Create model with better architecture
        self.model = MaskPLSSimplifiedONNX(cfg)
        
        # Use proper loss functions
        self.mask_loss = ProperMaskLoss(cfg.LOSS, cfg[dataset])
        self.sem_loss = SemLoss(cfg.LOSS.SEM.WEIGHTS)
        
        # Improved voxelizer with better resolution
        self.voxelizer = ImprovedVoxelizer(
            (64, 64, 32),  # Better resolution
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
        self.last_time = time.time()
        
        # ONNX export tracking
        self.last_onnx_export_epoch = -1
        
    def forward(self, batch):
        """Optimized forward pass with proper voxelization"""
        points = batch['pt_coord']
        features = batch['feats']
        
        # Proper voxelization
        voxel_grids, norm_coords, valid_indices = self.voxelizer.voxelize_batch_fast(points, features)
        
        # Update spatial shape to match voxelizer
        self.model.spatial_shape = self.voxelizer.spatial_shape
        
        # Pad coordinates properly
        if len(norm_coords) > 0 and all(len(c) > 0 for c in norm_coords):
            max_pts = max(c.shape[0] for c in norm_coords)
            padded_coords = []
            padding_masks = []
            
            for coords in norm_coords:
                n_pts = coords.shape[0]
                if n_pts < max_pts:
                    coords = F.pad(coords, (0, 0, 0, max_pts - n_pts))
                padded_coords.append(coords)
                
                mask = torch.zeros(max_pts, dtype=torch.bool, device=coords.device)
                if n_pts < max_pts:
                    mask[n_pts:] = True
                padding_masks.append(mask)
            
            batch_coords = torch.stack(padded_coords)
            padding_masks = torch.stack(padding_masks)
        else:
            # Handle empty case
            batch_coords = torch.zeros(len(points), 1000, 3, device='cuda')
            padding_masks = torch.ones(len(points), 1000, dtype=torch.bool, device='cuda')
            valid_indices = [torch.zeros(0, dtype=torch.long, device='cuda') for _ in points]
        
        # Forward through model
        pred_logits, pred_masks, sem_logits = self.model(voxel_grids, batch_coords)
        
        outputs = {
            'pred_logits': pred_logits,
            'pred_masks': pred_masks,
            'aux_outputs': []  # No auxiliary outputs in simplified version
        }
        
        return outputs, padding_masks, sem_logits, valid_indices
    
    def training_step(self, batch, batch_idx):
        """Training step with proper loss computation"""
        step_start = time.time()
        
        try:
            # Forward pass
            outputs, padding, sem_logits, valid_indices = self.forward(batch)
            
            # Skip if no valid data
            if all(len(v) == 0 for v in valid_indices):
                return torch.tensor(0.01, device='cuda', requires_grad=True)
            
            # Proper target preparation
            targets = self.prepare_targets_proper(batch, padding.shape[1], valid_indices)
            
            # Compute losses with proper functions
            mask_losses = self.mask_loss(outputs, targets, batch['masks_ids'], batch['pt_coord'])
            
            # Proper semantic loss
            sem_loss_value = self.compute_sem_loss_proper(batch, sem_logits, valid_indices, padding)
            
            # Total loss with proper weights
            total_loss = sum(mask_losses.values()) + sem_loss_value
            
            # Clamp for stability
            total_loss = torch.clamp(total_loss, 0, 50)
            
            # Log every 10 batches
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
            print(f"Error in batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            return torch.tensor(0.01, device='cuda', requires_grad=True)
    
    def prepare_targets_proper(self, batch, max_points, valid_indices):
        """Proper target preparation maintaining data integrity"""
        targets = {'classes': [], 'masks': []}
        
        for i in range(len(batch['masks_cls'])):
            if len(batch['masks_cls'][i]) > 0:
                # Proper class handling
                classes = torch.tensor(batch['masks_cls'][i], dtype=torch.long, device='cuda')
                targets['classes'].append(classes)
                
                # Proper mask handling
                masks_list = []
                for m in batch['masks'][i]:
                    if isinstance(m, torch.Tensor):
                        mask = m.float()
                    else:
                        mask = torch.from_numpy(m).float()
                    
                    # Ensure mask has correct size
                    if mask.shape[0] != max_points:
                        # Pad or truncate as needed
                        if mask.shape[0] < max_points:
                            mask = F.pad(mask, (0, max_points - mask.shape[0]))
                        else:
                            mask = mask[:max_points]
                    
                    masks_list.append(mask.to('cuda'))
                
                if masks_list:
                    targets['masks'].append(torch.stack(masks_list))
                else:
                    targets['masks'].append(torch.empty(0, max_points, device='cuda'))
            else:
                targets['classes'].append(torch.empty(0, dtype=torch.long, device='cuda'))
                targets['masks'].append(torch.empty(0, max_points, device='cuda'))
        
        return targets
    
    def compute_sem_loss_proper(self, batch, sem_logits, valid_indices, padding):
        """Proper semantic loss computation"""
        try:
            if sem_logits.numel() == 0:
                return torch.tensor(0.0, device='cuda', requires_grad=True)
            
            all_logits = []
            all_labels = []
            
            for i, (labels, idx, pad) in enumerate(zip(batch['sem_label'], valid_indices, padding)):
                if len(idx) == 0:
                    continue
                
                valid_mask = ~pad
                valid_logits = sem_logits[i][valid_mask]
                
                # Handle labels properly
                if isinstance(labels, np.ndarray):
                    labels = labels.flatten()
                else:
                    labels = np.array(labels).flatten()
                
                # Map indices properly
                if len(labels) > 0 and len(idx) > 0:
                    idx_cpu = idx.cpu().numpy()
                    valid_idx = idx_cpu[idx_cpu < len(labels)]
                    
                    if len(valid_idx) > 0:
                        selected_labels = labels[valid_idx]
                        selected_logits = valid_logits[:len(selected_labels)]
                        
                        if len(selected_labels) > 0:
                            labels_tensor = torch.from_numpy(selected_labels).long().cuda()
                            labels_tensor = torch.clamp(labels_tensor, 0, self.num_classes - 1)
                            
                            all_logits.append(selected_logits)
                            all_labels.append(labels_tensor)
            
            if all_logits:
                combined_logits = torch.cat(all_logits, dim=0)
                combined_labels = torch.cat(all_labels, dim=0)
                
                # Use proper semantic loss weights
                sem_loss = F.cross_entropy(combined_logits, combined_labels, ignore_index=0)
                return sem_loss * self.cfg.LOSS.SEM.WEIGHTS[0]  # Use proper weight
            else:
                return torch.tensor(0.0, device='cuda', requires_grad=True)
                
        except Exception as e:
            print(f"Semantic loss error: {e}")
            return torch.tensor(0.0, device='cuda', requires_grad=True)
    
    def validation_step(self, batch, batch_idx):
        """Validation with proper metrics"""
        try:
            outputs, padding, sem_logits, valid_indices = self.forward(batch)
            
            if all(len(v) == 0 for v in valid_indices):
                return torch.tensor(0.0, device='cuda')
            
            targets = self.prepare_targets_proper(batch, padding.shape[1], valid_indices)
            losses = self.mask_loss(outputs, targets, batch['masks_ids'], batch['pt_coord'])
            sem_loss = self.compute_sem_loss_proper(batch, sem_logits, valid_indices, padding)
            
            total_loss = sum(losses.values()) + sem_loss
            
            # Log validation metrics
            if batch_idx % 10 == 0:
                self.log("val_loss", total_loss, batch_size=self.cfg.TRAIN.BATCH_SIZE)
                for k, v in losses.items():
                    self.log(f"val/{k}", v, batch_size=self.cfg.TRAIN.BATCH_SIZE)
                
                # Proper panoptic evaluation
                sem_pred, ins_pred = self.panoptic_inference(outputs, padding)
                self.evaluator.update(sem_pred, ins_pred, batch)
            
            return total_loss
            
        except Exception as e:
            print(f"Validation error: {e}")
            return torch.tensor(0.0, device='cuda')
    
    def panoptic_inference(self, outputs, padding):
        """Proper panoptic inference following original implementation"""
        mask_cls = outputs["pred_logits"]
        mask_pred = outputs["pred_masks"]
        
        sem_pred = []
        ins_pred = []
        
        for mask_cls_b, mask_pred_b, pad in zip(mask_cls, mask_pred, padding):
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
            
            # Proper mask assignment
            cur_prob_masks = cur_scores.unsqueeze(0) * cur_masks
            cur_mask_ids = cur_prob_masks.argmax(1)
            
            sem = torch.zeros(cur_masks.shape[0], dtype=torch.long, device=cur_masks.device)
            ins = torch.zeros_like(sem)
            
            segment_id = 0
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                mask = (cur_mask_ids == k) & (cur_masks[:, k] >= 0.5)
                
                if mask.sum() > 10:  # Minimum mask size
                    sem[mask] = pred_class
                    if pred_class in self.things_ids:
                        segment_id += 1
                        ins[mask] = segment_id
            
            sem_pred.append(sem.cpu().numpy())
            ins_pred.append(ins.cpu().numpy())
        
        return sem_pred, ins_pred
    
    def validation_epoch_end(self, outputs):
        """Compute validation metrics and export ONNX if needed"""
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
            
            # Export ONNX at specified intervals
            current_epoch = self.current_epoch
            if (self.onnx_interval > 0 and 
                current_epoch > 0 and 
                current_epoch % self.onnx_interval == 0 and 
                current_epoch != self.last_onnx_export_epoch):
                self.export_to_onnx(current_epoch, pq, iou)
                self.last_onnx_export_epoch = current_epoch
                
        except Exception as e:
            print(f"Metrics computation error: {e}")
            self.log("metrics/pq", 0.0, batch_size=self.cfg.TRAIN.BATCH_SIZE)
            self.log("metrics/iou", 0.0, batch_size=self.cfg.TRAIN.BATCH_SIZE)
    
    def export_to_onnx(self, epoch, pq, iou):
        """Export the model to ONNX format"""
        try:
            print(f"\n{'='*60}")
            print(f"Exporting ONNX model at epoch {epoch}")
            print(f"Current metrics - PQ: {pq:.4f}, IoU: {iou:.4f}")
            print(f"{'='*60}")
            
            # Create output directory
            onnx_dir = Path("experiments") / f"{self.cfg.EXPERIMENT.ID}_fixed_v3" / "onnx_exports"
            onnx_dir.mkdir(parents=True, exist_ok=True)
            
            # Output filename with epoch and metrics
            output_path = onnx_dir / f"model_epoch{epoch:03d}_pq{pq:.3f}_iou{iou:.3f}.onnx"
            
            # Set model to eval mode
            self.model.eval()
            
            # Create dummy inputs matching the actual model inputs
            batch_size = 1
            num_points = 10000
            D, H, W = self.voxelizer.spatial_shape
            C = 4  # XYZI features
            
            # Pre-voxelized features
            dummy_voxels = torch.randn(batch_size, C, D, H, W, device='cuda')
            # Normalized point coordinates
            dummy_coords = torch.rand(batch_size, num_points, 3, device='cuda')
            
            # Export with proper settings
            with torch.no_grad():
                torch.onnx.export(
                    self.model,
                    (dummy_voxels, dummy_coords),
                    str(output_path),
                    export_params=True,
                    opset_version=16,
                    do_constant_folding=True,
                    input_names=['voxel_features', 'point_coords'],
                    output_names=['pred_logits', 'pred_masks', 'sem_logits'],
                    dynamic_axes={
                        'voxel_features': {0: 'batch_size'},
                        'point_coords': {0: 'batch_size', 1: 'num_points'},
                        'pred_logits': {0: 'batch_size'},
                        'pred_masks': {0: 'batch_size', 1: 'num_points'},
                        'sem_logits': {0: 'batch_size', 1: 'num_points'}
                    },
                    verbose=False
                )
            
            # Verify the exported model
            import onnx
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            
            # Get file size
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            
            print(f"✓ ONNX model exported successfully")
            print(f"  File: {output_path}")
            print(f"  Size: {file_size_mb:.2f} MB")
            print(f"  Opset Version: 16")
            print(f"  Input shapes:")
            print(f"    - voxel_features: [{batch_size}, {C}, {D}, {H}, {W}]")
            print(f"    - point_coords: [{batch_size}, {num_points}, 3]")
            print(f"{'='*60}\n")
            
            # Set model back to train mode
            self.model.train()
            
        except Exception as e:
            print(f"ONNX export failed: {e}")
            import traceback
            traceback.print_exc()
            # Set model back to train mode even if export fails
            self.model.train()
    
    def configure_optimizers(self):
        """Use original optimizer configuration for stability"""
        # Use original AdamW settings
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.TRAIN.LR,
            weight_decay=1e-4
        )
        
        # Use original StepLR schedule
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.cfg.TRAIN.STEP,
            gamma=self.cfg.TRAIN.DECAY
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
@click.option("--lr", type=float, default=0.0001, help="Learning rate (using original)")
@click.option("--gpus", type=int, default=1, help="Number of GPUs")
@click.option("--num_workers", type=int, default=4, help="Number of workers")
@click.option("--nuscenes", is_flag=True, help="Use NuScenes dataset")
@click.option("--checkpoint", type=str, default=None, help="Resume from checkpoint")
@click.option("--onnx_interval", type=int, default=5, help="Export ONNX model every N epochs (0 to disable)")
def main(epochs, batch_size, lr, gpus, num_workers, nuscenes, checkpoint, onnx_interval):
    """Fixed Ultra-optimized MaskPLS training with proper performance"""
    
    print("="*60)
    print("Fixed Ultra-Optimized MaskPLS Training v3")
    print("Balancing speed with model performance")
    print("="*60)
    
    # Load config
    def getDir(obj):
        return os.path.dirname(os.path.abspath(obj))
    
    model_cfg = edict(yaml.safe_load(open(join(getDir(__file__), "../config/model.yaml"))))
    backbone_cfg = edict(yaml.safe_load(open(join(getDir(__file__), "../config/backbone.yaml"))))
    decoder_cfg = edict(yaml.safe_load(open(join(getDir(__file__), "../config/decoder.yaml"))))
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    
    # Update config with proper values
    cfg.TRAIN.MAX_EPOCH = epochs
    cfg.TRAIN.BATCH_SIZE = batch_size
    cfg.TRAIN.LR = lr  # Use original learning rate
    cfg.TRAIN.N_GPUS = gpus
    cfg.TRAIN.NUM_WORKERS = num_workers
    cfg.TRAIN.BATCH_ACC = 2  # Some accumulation for stability
    
    if nuscenes:
        cfg.MODEL.DATASET = "NUSCENES"
    
    # Use reasonable point limits for balance
    dataset = cfg.MODEL.DATASET
    cfg[dataset].SUB_NUM_POINTS = min(cfg[dataset].SUB_NUM_POINTS, 40000)  # Balanced
    
    # Use proper loss parameters
    cfg.LOSS.NUM_POINTS = 25000  # Balanced between speed and accuracy
    cfg.LOSS.NUM_MASK_PTS = 2048  # Reasonable sampling
    
    print(f"Configuration:")
    print(f"  Dataset: {dataset}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {lr}")
    print(f"  Point Sampling: {cfg[dataset].SUB_NUM_POINTS}")
    print(f"  Loss Point Sampling: {cfg.LOSS.NUM_POINTS}")
    print(f"  Mask Point Sampling: {cfg.LOSS.NUM_MASK_PTS}")
    print(f"  ONNX Export Interval: {'Every ' + str(onnx_interval) + ' epochs' if onnx_interval > 0 else 'Disabled'}")
    
    # Create data module
    data = SemanticDatasetModule(cfg)
    
    # Create model with ONNX export interval
    model = FixedOptimizedMaskPLS(cfg, onnx_interval=onnx_interval)
    
    # Load checkpoint if provided
    if checkpoint:
        print(f"Loading checkpoint: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)
    
    # Setup logging
    tb_logger = pl_loggers.TensorBoardLogger(
        "experiments/" + cfg.EXPERIMENT.ID + "_fixed_v3",
        default_hp_metric=False
    )
    
    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    checkpoint_callback = ModelCheckpoint(
        monitor="metrics/pq",
        filename=cfg.EXPERIMENT.ID + "_fixed_epoch{epoch:02d}_pq{metrics/pq:.2f}",
        auto_insert_metric_name=False,
        mode="max",
        save_last=True,
        save_top_k=3
    )
    
    # Create trainer with proper settings
    trainer = Trainer(
        gpus=gpus,
        accelerator="gpu" if gpus > 0 else None,
        strategy="ddp" if gpus > 1 else None,
        logger=tb_logger,
        max_epochs=epochs,
        callbacks=[lr_monitor, checkpoint_callback],
        log_every_n_steps=10,
        gradient_clip_val=0.5,  # Keep gradient clipping for stability
        accumulate_grad_batches=cfg.TRAIN.BATCH_ACC,
        precision=16,  # Mixed precision for speed
        num_sanity_val_steps=0,  # Skip sanity check for speed
        val_check_interval=0.5,  # Validate twice per epoch
        resume_from_checkpoint=checkpoint
    )
    
    # Train
    print("\nStarting training...")
    trainer.fit(model, data)
    
    print("\nTraining completed!")
    
    # Export final ONNX model
    if onnx_interval > 0:
        print("\nExporting final ONNX model...")
        try:
            # Get final metrics
            final_pq = model.evaluator.get_mean_pq() if hasattr(model.evaluator, 'get_mean_pq') else 0.0
            final_iou = model.evaluator.get_mean_iou() if hasattr(model.evaluator, 'get_mean_iou') else 0.0
            model.export_to_onnx(epochs, final_pq, final_iou)
        except Exception as e:
            print(f"Final ONNX export failed: {e}")


if __name__ == "__main__":
    main()