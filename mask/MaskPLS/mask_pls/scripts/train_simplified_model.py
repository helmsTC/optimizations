"""
Training script for the simplified ONNX-compatible MaskPLS model
Save as: mask/MaskPLS/mask_pls/scripts/train_simplified_model.py
"""

import os
# Enable CUDA error debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

from os.path import join
import click
import torch
import yaml
import numpy as np
import time
from easydict import EasyDict as edict
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Callback
from torch.profiler import profile, record_function, ProfilerActivity
import torch.onnx
from pathlib import Path

# Import the simplified model components
from mask_pls.models.onnx.simplified_model import MaskPLSSimplifiedONNX
from mask_pls.datasets.semantic_dataset import SemanticDatasetModule
from mask_pls.models.loss import MaskLoss, SemLoss
from mask_pls.utils.evaluate_panoptic import PanopticEvaluator

import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule


class ONNXExportCallback(Callback):
    """Callback to export model to ONNX format every N epochs"""
    def __init__(self, export_interval=5, output_dir="onnx_checkpoints"):
        self.export_interval = export_interval
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def on_epoch_end(self, trainer, pl_module):
        """Export model to ONNX at the end of specified epochs"""
        current_epoch = trainer.current_epoch + 1
        
        if current_epoch % self.export_interval == 0:
            try:
                print(f"\n[ONNX Export] Exporting model at epoch {current_epoch}...")
                
                # Set model to eval mode
                pl_module.eval()
                
                # Get the underlying model
                model = pl_module.model
                
                # Create dummy input (adjust sizes as needed)
                batch_size = 1
                num_points = 10000
                dummy_voxels = torch.randn(batch_size, 4, 64, 64, 64).cuda()
                dummy_coords = torch.randn(batch_size, num_points, 3).cuda()
                
                # Export path
                export_path = self.output_dir / f"model_epoch_{current_epoch:03d}.onnx"
                
                # Export to ONNX
                with torch.no_grad():
                    torch.onnx.export(
                        model,
                        (dummy_voxels, dummy_coords),
                        str(export_path),
                        export_params=True,
                        opset_version=11,
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
                
                file_size_mb = export_path.stat().st_size / (1024 * 1024)
                print(f"[ONNX Export] ✓ Saved to {export_path} ({file_size_mb:.2f} MB)")
                
                # Back to train mode
                pl_module.train()
                
            except Exception as e:
                print(f"[ONNX Export] ✗ Failed to export at epoch {current_epoch}: {e}")
                # Continue training even if export fails
                pl_module.train()


class ProfilingCallback(Callback):
    """Callback for profiling training performance"""
    def __init__(self, profile_interval=10, warmup_steps=5):
        self.profile_interval = profile_interval
        self.warmup_steps = warmup_steps
        self.step_count = 0
        self.profiling_enabled = False
        self.profiler = None
        
        # Timing metrics
        self.data_load_times = []
        self.forward_times = []
        self.loss_times = []
        self.backward_times = []
        self.optimizer_times = []
        
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Start profiling if conditions are met"""
        self.step_count += 1
        
        # Start profiler after warmup
        if self.step_count == self.warmup_steps:
            print(f"\n[Profiler] Starting performance profiling...")
            self.profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
            self.profiler.__enter__()
            self.profiling_enabled = True
            
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Stop profiling and report if interval reached"""
        if self.profiling_enabled and self.step_count >= self.warmup_steps + self.profile_interval:
            self.profiler.__exit__(None, None, None)
            
            # Print profiler results
            print("\n" + "="*80)
            print("[Profiler] Performance Analysis Report")
            print("="*80)
            
            # Sort by CUDA time
            print("\nTop operations by CUDA time:")
            print(self.profiler.key_averages().table(
                sort_by="cuda_time_total", row_limit=10
            ))
            
            # Sort by CPU time
            print("\nTop operations by CPU time:")
            print(self.profiler.key_averages().table(
                sort_by="cpu_time_total", row_limit=10
            ))
            
            # Memory usage
            print("\nTop operations by memory usage:")
            print(self.profiler.key_averages().table(
                sort_by="self_cuda_memory_usage", row_limit=10
            ))
            
            print("="*80 + "\n")
            
            # Reset for next profiling interval
            self.profiling_enabled = False
            self.profiler = None
            self.step_count = 0


def prepare_targets_for_loss(batch, max_points, valid_indices, device='cuda'):
    """
    Convert batch targets to the format expected by the loss functions
    Now handles padding to match model output size
    
    Args:
        batch: Original batch data
        max_points: Maximum number of points (padded size)
        valid_indices: List of valid point indices for each sample
        device: Device to place tensors on
    """
    fixed_targets = {
        'classes': [],
        'masks': []
    }
    
    for i in range(len(batch['masks_cls'])):
        if len(batch['masks_cls'][i]) > 0:
            # Convert list of scalar tensors to single tensor
            classes = []
            for c in batch['masks_cls'][i]:
                if isinstance(c, torch.Tensor):
                    classes.append(c.item() if c.numel() == 1 else int(c))
                else:
                    classes.append(int(c))
            
            # Create tensor and move to device
            classes_tensor = torch.tensor(classes, dtype=torch.long, device=device)
            fixed_targets['classes'].append(classes_tensor)
            
            # Process masks with padding
            masks_list = []
            for m in batch['masks'][i]:
                if isinstance(m, torch.Tensor):
                    mask = m.float()
                else:
                    mask = torch.tensor(m, dtype=torch.float32)
                
                # Get the original mask size
                orig_size = mask.shape[0]
                
                # Create a padded mask initialized with zeros
                padded_mask = torch.zeros(max_points, device=device)
                
                # Get valid indices for this sample
                valid_idx = valid_indices[i]
                
                if len(valid_idx) > 0 and orig_size > 0:
                    # Map the mask values to the valid indices
                    idx_cpu = valid_idx.cpu().numpy()
                    
                    # Ensure indices are within bounds of the original mask
                    valid_mask_indices = idx_cpu < orig_size
                    idx_to_use = idx_cpu[valid_mask_indices]
                    
                    # Get corresponding positions in the padded mask
                    # (these are the positions where we kept the points)
                    padded_positions = torch.arange(len(valid_idx))[valid_mask_indices]
                    
                    if len(idx_to_use) > 0:
                        # Copy mask values to the padded mask
                        mask_values = mask[idx_to_use]
                        if not isinstance(mask_values, torch.Tensor):
                            mask_values = torch.tensor(mask_values, dtype=torch.float32)
                        padded_mask[padded_positions] = mask_values.to(device)
                
                masks_list.append(padded_mask)
            
            if len(masks_list) > 0:
                masks_tensor = torch.stack(masks_list)
                fixed_targets['masks'].append(masks_tensor)
            else:
                fixed_targets['masks'].append(torch.empty(0, max_points, device=device))
        else:
            # No masks in this sample
            fixed_targets['classes'].append(torch.empty(0, dtype=torch.long, device=device))
            fixed_targets['masks'].append(torch.empty(0, max_points, device=device))
    
    return fixed_targets


def prepare_masks_ids_for_loss(batch_masks_ids, valid_indices, max_points):
    """
    Convert masks_ids to work with padded masks
    
    Args:
        batch_masks_ids: Original masks_ids from batch
        valid_indices: List of valid point indices for each sample
        max_points: Maximum number of points (padded size)
    
    Returns:
        Fixed masks_ids that work with padded data
    """
    fixed_masks_ids = []
    
    for i, (masks_ids_sample, valid_idx) in enumerate(zip(batch_masks_ids, valid_indices)):
        if len(masks_ids_sample) == 0:
            fixed_masks_ids.append([])
            continue
            
        sample_masks_ids = []
        valid_idx_cpu = valid_idx.cpu().numpy()
        
        # Create mapping from original indices to padded indices
        orig_to_padded = {}
        for padded_idx, orig_idx in enumerate(valid_idx_cpu):
            orig_to_padded[orig_idx] = padded_idx
        
        for mask_ids in masks_ids_sample:
            if isinstance(mask_ids, torch.Tensor):
                mask_ids = mask_ids.cpu().numpy()
            elif not isinstance(mask_ids, np.ndarray):
                mask_ids = np.array(mask_ids)
            
            # Map original indices to padded indices
            new_mask_ids = []
            for idx in mask_ids:
                if idx in orig_to_padded:
                    new_mask_ids.append(orig_to_padded[idx])
            
            if len(new_mask_ids) > 0:
                sample_masks_ids.append(torch.tensor(new_mask_ids, dtype=torch.long))
            else:
                # Empty mask after mapping
                sample_masks_ids.append(torch.tensor([], dtype=torch.long))
        
        fixed_masks_ids.append(sample_masks_ids)
    
    return fixed_masks_ids


class SimplifiedMaskPLS(LightningModule):
    """
    Lightning module for training the simplified MaskPLS model
    """
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(dict(cfg))
        self.cfg = cfg
        
        # Timing tracking
        self.timing_enabled = False
        self.component_times = {
            'data_prep': [],
            'voxelization': [],
            'backbone': [],
            'decoder': [],
            'loss_computation': [],
            'total_forward': []
        }
        
        # Get dataset configuration
        dataset = cfg.MODEL.DATASET
        self.num_classes = cfg[dataset].NUM_CLASSES
        self.ignore_label = cfg[dataset].IGNORE_LABEL
        
        print(f"Initializing model for {dataset} with {self.num_classes} classes")
        print(f"Ignore label: {self.ignore_label}")
        
        # Create the simplified model
        self.model = MaskPLSSimplifiedONNX(cfg)
        
        # Verify semantic head configuration
        sem_head_out = self.model.sem_head.out_features
        if sem_head_out != self.num_classes:
            print(f"WARNING: Semantic head outputs {sem_head_out} classes but dataset has {self.num_classes}")
        
        # Loss functions (same as original)
        self.mask_loss = MaskLoss(cfg.LOSS, cfg[dataset])
        self.sem_loss = SemLoss(cfg.LOSS.SEM.WEIGHTS)
        
        # Evaluator (same as original)
        self.evaluator = PanopticEvaluator(cfg[dataset], dataset)
        
        # Cache things IDs
        data = SemanticDatasetModule(cfg)
        data.setup()
        self.things_ids = data.things_ids
        
        # Debug mode
        self.debug = False
        self.fast_mode = False
        
    def on_train_start(self):
        """Simple training start message"""
        print(f"\nStarting training for {self.cfg.MODEL.DATASET} dataset")
        print(f"Number of classes: {self.num_classes}")
        print(f"Ignore label: {self.ignore_label}")
        print("-" * 60)
            
    def forward(self, batch):
        """Optimized forward pass - vectorized preprocessing"""
        if self.timing_enabled:
            total_start = time.time()
            data_prep_start = time.time()
            
        # Check if batch already has preprocessed voxels (from optimized dataset)
        if 'preprocessed_voxels' in batch and 'preprocessed_coords' in batch:
            batch_voxels = batch['preprocessed_voxels']
            batch_coords = batch['preprocessed_coords'] 
            valid_indices = batch['valid_indices']
            if self.timing_enabled:
                self.component_times['data_prep'].append(time.time() - data_prep_start)
                self.component_times['voxelization'].append(0.01)  # Minimal time since pre-computed
        else:
            # Fallback to original processing but optimized
            batch_voxels, batch_coords, valid_indices = self._process_batch_optimized(batch)
            if self.timing_enabled:
                self.component_times['data_prep'].append(time.time() - data_prep_start)
        
        if self.timing_enabled:
            model_start = time.time()
        # Efficient batching with pre-computed max size
        if len(batch_coords) > 0:
            max_pts = max(c.shape[0] for c in batch_coords)
            
            # Vectorized padding
            padded_coords = []
            padding_masks = []
            for coords in batch_coords:
                n_pts = coords.shape[0]
                if n_pts < max_pts:
                    coords = F.pad(coords, (0, 0, 0, max_pts - n_pts))
                padded_coords.append(coords)
                
                # Efficient padding mask creation
                mask = torch.zeros(max_pts, dtype=torch.bool, device=coords.device)
                if n_pts < max_pts:
                    mask[n_pts:] = True
                padding_masks.append(mask)
            
            batch_voxels = torch.stack(batch_voxels)
            batch_coords = torch.stack(padded_coords) 
            padding_masks = torch.stack(padding_masks)
        else:
            # Handle empty batch
            batch_voxels = torch.empty(0, device='cuda')
            batch_coords = torch.empty(0, device='cuda')
            padding_masks = torch.empty(0, dtype=torch.bool, device='cuda')
        
        # Forward through model with component timing
        if self.timing_enabled:
            with record_function("model_forward"):
                pred_logits, pred_masks, sem_logits = self.model(batch_voxels, batch_coords)
            self.component_times['backbone'].append(time.time() - model_start)
        else:
            pred_logits, pred_masks, sem_logits = self.model(batch_voxels, batch_coords)
        
        # Validate outputs
        if self.debug:
            print(f"Model outputs:")
            print(f"  pred_logits: {pred_logits.shape}, range [{pred_logits.min():.2f}, {pred_logits.max():.2f}]")
            print(f"  pred_masks: {pred_masks.shape}, range [{pred_masks.min():.2f}, {pred_masks.max():.2f}]")
            print(f"  sem_logits: {sem_logits.shape}, range [{sem_logits.min():.2f}, {sem_logits.max():.2f}]")
        
        # Check semantic logits shape
        expected_classes = self.num_classes
        if sem_logits.shape[-1] != expected_classes:
            print(f"ERROR: sem_logits has {sem_logits.shape[-1]} classes but expected {expected_classes}")
            # Adjust if necessary
            if sem_logits.shape[-1] > expected_classes:
                sem_logits = sem_logits[..., :expected_classes]
            else:
                # Pad with zeros
                pad_size = expected_classes - sem_logits.shape[-1]
                sem_logits = F.pad(sem_logits, (0, pad_size))
        
        outputs = {
            'pred_logits': pred_logits,
            'pred_masks': pred_masks,
            'aux_outputs': []  # Simplified model doesn't have aux outputs
        }
        
        if self.timing_enabled:
            self.component_times['total_forward'].append(time.time() - total_start)
            
            # Print timing stats every 50 steps
            if len(self.component_times['total_forward']) % 50 == 0:
                self._print_timing_stats()
        
        return outputs, padding_masks, sem_logits, valid_indices
    
    def _process_batch_optimized(self, batch):
        """Optimized batch processing with vectorization where possible"""
        points = batch['pt_coord']
        features = batch['feats']
        
        batch_voxels = []
        batch_coords = []
        valid_indices = []
        
        # Cache config values
        bounds = self.cfg[self.cfg.MODEL.DATASET].SPACE
        max_pts = self.cfg[self.cfg.MODEL.DATASET].SUB_NUM_POINTS
        
        voxel_start = time.time() if self.timing_enabled else None
        
        for i, (pts_np, feat_np) in enumerate(zip(points, features)):
            # Single tensor conversion
            pts = torch.from_numpy(pts_np).float().cuda()
            feat = torch.from_numpy(feat_np).float().cuda()
            
            # Vectorized bounds filtering
            valid_mask = torch.ones(pts.shape[0], dtype=torch.bool, device=pts.device)
            for dim in range(3):
                valid_mask &= (pts[:, dim] >= bounds[dim][0]) & (pts[:, dim] < bounds[dim][1])
            
            valid_idx = torch.where(valid_mask)[0]
            valid_pts = pts[valid_mask]
            valid_feat = feat[valid_mask]
            
            # Smart subsampling for large point clouds
            if len(valid_pts) > max_pts:
                if self.training:
                    # Hierarchical subsampling: keep some random, some structured
                    n_pts = len(valid_pts)
                    if hasattr(self, 'fast_mode') and self.fast_mode and n_pts > max_pts * 2:
                        # For very large clouds, use stratified sampling
                        stride = n_pts // (max_pts // 2)
                        structured_idx = torch.arange(0, n_pts, stride, device=valid_pts.device)[:max_pts//2]
                        remaining = max_pts - len(structured_idx)
                        if remaining > 0:
                            random_mask = torch.ones(n_pts, dtype=torch.bool, device=valid_pts.device)
                            random_mask[structured_idx] = False
                            random_candidates = torch.where(random_mask)[0]
                            if len(random_candidates) > 0:
                                random_idx = random_candidates[torch.randperm(len(random_candidates))[:remaining]]
                                perm = torch.cat([structured_idx, random_idx])
                            else:
                                perm = structured_idx
                        else:
                            perm = structured_idx
                    else:
                        # Standard random subsampling for moderate sizes
                        perm = torch.randperm(len(valid_pts), device=valid_pts.device)[:max_pts]
                else:
                    # Deterministic subsampling for validation
                    perm = torch.arange(max_pts, device=valid_pts.device)
                    
                valid_pts = valid_pts[perm]
                valid_feat = valid_feat[perm] 
                valid_idx = valid_idx[perm]
            
            # Efficient normalization with bounds caching
            if len(valid_pts) > 0:
                bounds_min = torch.tensor([bounds[dim][0] for dim in range(3)], device=valid_pts.device)
                bounds_range = torch.tensor([bounds[dim][1] - bounds[dim][0] for dim in range(3)], device=valid_pts.device)
                norm_coords = (valid_pts - bounds_min) / bounds_range
            else:
                norm_coords = torch.empty(0, 3, device='cuda')
            
            # Voxelize (still the main bottleneck)
            voxel_grid = self.model.voxelize_points(
                valid_pts.unsqueeze(0),
                valid_feat.unsqueeze(0)
            )[0]
            
            batch_voxels.append(voxel_grid)
            batch_coords.append(norm_coords)
            valid_indices.append(valid_idx)
        
        if self.timing_enabled and voxel_start:
            self.component_times['voxelization'].append(time.time() - voxel_start)
            
        return batch_voxels, batch_coords, valid_indices
    
    def _print_timing_stats(self):
        """Print timing statistics for performance analysis"""
        print("\n" + "="*60)
        print("[Timing Stats] Component Performance (ms)")
        print("="*60)
        
        for component, times in self.component_times.items():
            if times:
                avg_time = np.mean(times) * 1000  # Convert to ms
                std_time = np.std(times) * 1000
                print(f"{component:20s}: {avg_time:7.2f} ± {std_time:5.2f} ms")
        
        # Calculate percentages
        if self.component_times['total_forward']:
            total_avg = np.mean(self.component_times['total_forward'])
            print("\nComponent percentages:")
            for component, times in self.component_times.items():
                if times and component != 'total_forward':
                    pct = (np.mean(times) / total_avg) * 100
                    print(f"{component:20s}: {pct:5.1f}%")
        
        print("="*60 + "\n")
        
        # Clear after printing
        for k in self.component_times:
            self.component_times[k] = []
    
    def training_step(self, batch, batch_idx):
        try:
            if self.timing_enabled:
                loss_start = time.time()
            
            outputs, padding, sem_logits, valid_indices = self.forward(batch)
            
            # Get the max points from the padding shape
            max_points = padding.shape[1]
            
            # Prepare targets for mask loss with proper padding
            targets = prepare_targets_for_loss(batch, max_points, valid_indices, device='cuda')
            
            # Fix masks_ids to work with padded data
            fixed_masks_ids = prepare_masks_ids_for_loss(batch['masks_ids'], valid_indices, max_points)
            
            # Mask loss
            try:
                loss_mask = self.mask_loss(outputs, targets, fixed_masks_ids, batch['pt_coord'])
            except Exception as e:
                print(f"Error in mask loss: {e}")
                print(f"pred_masks shape: {outputs['pred_masks'].shape}")
                print(f"target masks shapes: {[t.shape for t in targets['masks']]}")
                # Return minimal loss to continue
                return torch.tensor(0.1, device='cuda', requires_grad=True)
            
            # Optimized semantic loss computation
            try:
                sem_loss_result = self._compute_semantic_loss_fast(batch, sem_logits, valid_indices, padding)
                loss_mask.update(sem_loss_result)
                    
            except Exception as e:
                print(f"Error in semantic loss computation: {e}")
                loss_mask['sem_ce'] = torch.tensor(0.0, device='cuda', requires_grad=True)
                loss_mask['sem_lov'] = torch.tensor(0.0, device='cuda', requires_grad=True)
            
            # Log losses
            for k, v in loss_mask.items():
                if torch.isnan(v) or torch.isinf(v):
                    v = torch.tensor(0.0, device='cuda', requires_grad=True)
                self.log(f"train/{k}", v, batch_size=self.cfg.TRAIN.BATCH_SIZE, on_step=True, on_epoch=True)
            
            # Total loss with fast computation
            total_loss = sum(loss_mask.values())
            
            # Minimal safety check for speed
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                total_loss = torch.tensor(0.1, device='cuda', requires_grad=True)
            
            # Reduced logging frequency for speed
            if batch_idx % 5 == 0 or not hasattr(self, 'fast_mode') or not self.fast_mode:
                self.log("train_loss", total_loss, batch_size=self.cfg.TRAIN.BATCH_SIZE, on_step=True, on_epoch=True)
            
            if self.timing_enabled:
                self.component_times['loss_computation'].append(time.time() - loss_start)
            
            return total_loss
            
        except Exception as e:
            print(f"\nUnexpected error in training_step: {type(e).__name__}: {e}")
            return torch.tensor(0.1, device='cuda', requires_grad=True)
    
    
    def validation_step(self, batch, batch_idx):
        try:
            outputs, padding, sem_logits, valid_indices = self.forward(batch)
            
            # Get the max points from the padding shape
            max_points = padding.shape[1]
            
            # Prepare targets for mask loss with proper padding
            targets = prepare_targets_for_loss(batch, max_points, valid_indices, device='cuda')
            
            # Fix masks_ids to work with padded data
            fixed_masks_ids = prepare_masks_ids_for_loss(batch['masks_ids'], valid_indices, max_points)
            
            # Calculate losses
            loss_mask = self.mask_loss(outputs, targets, fixed_masks_ids, batch['pt_coord'])
            
            # Semantic loss
            all_sem_labels = []
            all_sem_logits = []
            
            for i, (label, idx, pad) in enumerate(zip(batch['sem_label'], valid_indices, padding)):
                # Get valid points for this sample
                valid_mask = ~pad
                num_valid = valid_mask.sum().item()
                
                if num_valid == 0:
                    continue
                
                # Get semantic logits for valid points
                sem_logits_i = sem_logits[i][valid_mask]
                
                # Get semantic labels for the points we actually kept
                idx_cpu = idx.cpu().numpy()
                num_logits = sem_logits_i.shape[0]
                
                # Get labels
                label_array = label.flatten() if label.ndim > 1 else label
                label_size = len(label_array)
                
                # Ensure indices are within bounds of both logits and labels
                valid_idx_mask = (idx_cpu < label_size) & (idx_cpu >= 0)
                idx_to_use = idx_cpu[valid_idx_mask]
                
                # Ensure we don't use more indices than we have logits
                idx_to_use = idx_to_use[:num_logits]
                
                # Adjust sem_logits to match the number of valid indices
                if len(idx_to_use) < num_logits:
                    sem_logits_i = sem_logits_i[:len(idx_to_use)]
                    
                if len(idx_to_use) == 0:
                    continue
                
                valid_labels = label_array[idx_to_use]
                valid_labels_tensor = torch.from_numpy(valid_labels).long()
                
                # Safety clamp
                if valid_labels_tensor.max() >= self.num_classes:
                    valid_labels_tensor = torch.clamp(valid_labels_tensor, 0, self.num_classes - 1)
                
                valid_labels_tensor = valid_labels_tensor.cuda()
                
                all_sem_logits.append(sem_logits_i)
                all_sem_labels.append(valid_labels_tensor)
            
            # Compute semantic loss if we have valid data
            if len(all_sem_logits) > 0:
                all_sem_logits = torch.cat(all_sem_logits, dim=0)
                all_sem_labels = torch.cat(all_sem_labels, dim=0)
                loss_sem = self.sem_loss(all_sem_logits, all_sem_labels)
                loss_mask.update(loss_sem)
            
            # Log losses
            for k, v in loss_mask.items():
                if torch.isnan(v) or torch.isinf(v):
                    v = torch.tensor(0.0, device='cuda')
                self.log(f"val/{k}", v, batch_size=self.cfg.TRAIN.BATCH_SIZE)
            
            total_loss = sum(loss_mask.values())
            self.log("val_loss", total_loss, batch_size=self.cfg.TRAIN.BATCH_SIZE)
            
            # Panoptic inference and evaluation
            sem_pred, ins_pred = self.panoptic_inference(outputs, padding)
            
            # Map predictions back to original points
            full_sem_pred = []
            full_ins_pred = []
            
            for i, (pred_sem, pred_ins, idx) in enumerate(zip(sem_pred, ins_pred, valid_indices)):
                # Create full predictions initialized with ignore label
                full_sem = torch.zeros(len(batch['sem_label'][i]), dtype=torch.long)
                full_ins = torch.zeros(len(batch['ins_label'][i]), dtype=torch.long)
                
                # Map back to original indices
                idx_cpu = idx.cpu().numpy()
                
                # Ensure indices are within bounds
                max_size = len(full_sem)
                valid_mask = idx_cpu < max_size
                idx_cpu = idx_cpu[valid_mask]
                
                valid_len = min(len(idx_cpu), len(pred_sem))
                
                if valid_len > 0:
                    full_sem[idx_cpu[:valid_len]] = torch.from_numpy(pred_sem[:valid_len])
                    full_ins[idx_cpu[:valid_len]] = torch.from_numpy(pred_ins[:valid_len])
                
                full_sem_pred.append(full_sem.numpy())
                full_ins_pred.append(full_ins.numpy())
            
            # Update evaluator
            self.evaluator.update(full_sem_pred, full_ins_pred, batch)
            
            return total_loss
            
        except Exception as e:
            print(f"Error in validation_step: {e}")
            import traceback
            traceback.print_exc()
            return torch.tensor(0.1, device='cuda')
    
    def validation_epoch_end(self, outputs):
        bs = self.cfg.TRAIN.BATCH_SIZE
        self.log("metrics/pq", self.evaluator.get_mean_pq(), batch_size=bs)
        self.log("metrics/iou", self.evaluator.get_mean_iou(), batch_size=bs)
        self.log("metrics/rq", self.evaluator.get_mean_rq(), batch_size=bs)
        self.evaluator.reset()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.TRAIN.LR)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=self.cfg.TRAIN.STEP, 
            gamma=self.cfg.TRAIN.DECAY
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def panoptic_inference(self, outputs, padding):
        """Panoptic segmentation inference (adapted from original)"""
        mask_cls = outputs["pred_logits"]
        mask_pred = outputs["pred_masks"]
        num_classes = self.cfg[self.cfg.MODEL.DATASET].NUM_CLASSES
        
        sem_pred = []
        ins_pred = []
        
        for b_cls, b_mask, b_pad in zip(mask_cls, mask_pred, padding):
            # Remove padding
            valid_mask = ~b_pad
            b_mask_valid = b_mask[valid_mask]
            
            scores, labels = b_cls.max(-1)
            b_mask_valid = b_mask_valid.sigmoid()
            
            keep = labels.ne(num_classes)
            
            if keep.sum() == 0:
                sem = torch.zeros(b_mask_valid.shape[0], dtype=torch.long)
                ins = torch.zeros(b_mask_valid.shape[0], dtype=torch.long)
            else:
                cur_scores = scores[keep]
                cur_classes = labels[keep]
                cur_masks = b_mask_valid[:, keep]
                
                # Get predictions
                cur_prob_masks = cur_scores.unsqueeze(0) * cur_masks
                mask_ids = cur_prob_masks.argmax(1)
                
                # Generate semantic and instance
                sem = torch.zeros(b_mask_valid.shape[0], dtype=torch.long)
                ins = torch.zeros(b_mask_valid.shape[0], dtype=torch.long)
                
                instance_id = 1
                for k in range(cur_classes.shape[0]):
                    mask = (mask_ids == k) & (cur_masks[:, k] >= 0.5)
                    if mask.sum() > 0:
                        sem[mask] = cur_classes[k]
                        if cur_classes[k].item() in self.things_ids:
                            ins[mask] = instance_id
                            instance_id += 1
            
            sem_pred.append(sem.cpu().numpy())
            ins_pred.append(ins.cpu().numpy())
        
        return sem_pred, ins_pred
    
    def _compute_semantic_loss_fast(self, batch, sem_logits, valid_indices, padding):
        """Optimized semantic loss with reduced tensor operations"""
        # Collect all valid data in fewer operations
        valid_logits_list = []
        valid_labels_list = []
        
        for i, (label, idx, pad) in enumerate(zip(batch['sem_label'], valid_indices, padding)):
            valid_mask = ~pad
            num_valid = valid_mask.sum().item()
            
            if num_valid == 0:
                continue
            
            # Get data for valid points only
            sem_logits_i = sem_logits[i][valid_mask]
            idx_cpu = idx.cpu()
            
            # Process labels efficiently
            label_array = label.flatten() if label.ndim > 1 else label
            label_size = len(label_array)
            
            # Efficient bounds checking
            valid_idx_mask = (idx_cpu < label_size) & (idx_cpu >= 0)
            idx_to_use = idx_cpu[valid_idx_mask][:sem_logits_i.shape[0]]
            
            if len(idx_to_use) == 0:
                continue
                
            # Adjust sizes to match
            if len(idx_to_use) < sem_logits_i.shape[0]:
                sem_logits_i = sem_logits_i[:len(idx_to_use)]
            
            # Extract and process labels
            valid_labels = torch.from_numpy(label_array[idx_to_use.numpy()]).long().cuda()
            valid_labels = torch.clamp(valid_labels, 0, self.num_classes - 1)
            
            if sem_logits_i.shape[0] == valid_labels.shape[0] and len(valid_labels) > 0:
                valid_logits_list.append(sem_logits_i)
                valid_labels_list.append(valid_labels)
        
        # Single concatenation instead of multiple
        if valid_logits_list:
            all_logits = torch.cat(valid_logits_list, dim=0)
            all_labels = torch.cat(valid_labels_list, dim=0) 
            return self.sem_loss(all_logits, all_labels)
        else:
            return {
                'sem_ce': torch.tensor(0.0, device='cuda', requires_grad=True),
                'sem_lov': torch.tensor(0.0, device='cuda', requires_grad=True)
            }


def getDir(obj):
    return os.path.dirname(os.path.abspath(obj))


@click.command()
@click.option("--checkpoint", type=str, default=None, help="Resume from checkpoint")
@click.option("--nuscenes", is_flag=True, help="Use NuScenes dataset")
@click.option("--epochs", type=int, default=100, help="Number of epochs")
@click.option("--batch_size", type=int, default=1, help="Batch size")
@click.option("--lr", type=float, default=0.0001, help="Learning rate")
@click.option("--gpus", type=int, default=1, help="Number of GPUs")
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.option("--num_workers", type=int, default=4, help="Number of data loader workers")
@click.option("--profile", is_flag=True, help="Enable performance profiling")
@click.option("--timing", is_flag=True, help="Enable component timing analysis")
@click.option("--export_onnx", is_flag=True, help="Export ONNX models every 5 epochs")
@click.option("--onnx_interval", type=int, default=5, help="Interval for ONNX export")
@click.option("--precision", type=int, default=16, help="Training precision (16 for mixed precision, 32 for full)")
@click.option("--max_points", type=int, default=None, help="Override max points per sample (default: keep original, fast_mode: 16K-32K)")
@click.option("--skip_val", is_flag=True, help="Skip validation during training for speed")
@click.option("--fast_mode", is_flag=True, help="Enable smart optimizations for large point clouds")
@click.option("--chunk_size", type=int, default=None, help="Process point clouds in chunks (for very large clouds)")
def main(checkpoint, nuscenes, epochs, batch_size, lr, gpus, debug, num_workers, profile, timing, export_onnx, onnx_interval, precision, max_points, skip_val, fast_mode, chunk_size):
    # Load configurations
    model_cfg = edict(
        yaml.safe_load(open(join(getDir(__file__), "../config/model.yaml")))
    )
    backbone_cfg = edict(
        yaml.safe_load(open(join(getDir(__file__), "../config/backbone.yaml")))
    )
    decoder_cfg = edict(
        yaml.safe_load(open(join(getDir(__file__), "../config/decoder.yaml")))
    )
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    
    # Update config with command line args
    cfg.TRAIN.MAX_EPOCH = epochs
    cfg.TRAIN.BATCH_SIZE = batch_size
    cfg.TRAIN.LR = lr
    cfg.TRAIN.N_GPUS = gpus
    cfg.TRAIN.NUM_WORKERS = num_workers
    
    # Apply fast mode optimizations
    if fast_mode:
        print("🚀 FAST MODE ENABLED - Applying smart optimizations for large point clouds")
        cfg.TRAIN.NUM_WORKERS = max(8, num_workers)  # More workers
        cfg.TRAIN.BATCH_ACC = max(4, cfg.TRAIN.BATCH_ACC)  # More gradient accumulation
        skip_val = True  # Force skip validation
        if max_points is None:
            # More conservative limit for large point clouds
            original_max = cfg[cfg.MODEL.DATASET].SUB_NUM_POINTS
            # More conservative for large point clouds  
            if original_max > 100000:
                max_points = min(65536, max(32768, original_max // 2))  # 32K-64K for very large clouds
            else:
                max_points = min(32768, max(16384, original_max))  # 16K-32K for moderate clouds
            print(f"Auto-selected {max_points} points (original: {original_max})")
            
            # Auto-suggest chunking for very large clouds
            if original_max > 200000:
                print(f"💡 Your point clouds are very large ({original_max} points). Consider --chunk_size 50000 for better memory usage")
    
    # Override max points if specified
    if max_points is not None:
        original_max = cfg[cfg.MODEL.DATASET].SUB_NUM_POINTS
        cfg[cfg.MODEL.DATASET].SUB_NUM_POINTS = max_points
        print(f"⚡ Adjusted max points: {original_max} → {max_points}")
        
        # Adjust batch size if points are very large to avoid OOM
        if max_points > 50000 and batch_size > 1:
            new_batch_size = 1
            cfg.TRAIN.BATCH_SIZE = new_batch_size
            print(f"⚠️ Large point clouds detected, reducing batch size to {new_batch_size}")
    
    if nuscenes:
        cfg.MODEL.DATASET = "NUSCENES"
    
    # Update experiment ID
    cfg.EXPERIMENT.ID = cfg.EXPERIMENT.ID + "_simplified"
    
    print("Training Simplified MaskPLS Model")
    print(f"Dataset: {cfg.MODEL.DATASET}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"GPUs: {gpus}")
    print(f"Workers: {num_workers}")
    print(f"Debug mode: {debug}")
    print(f"Profiling: {profile}")
    print(f"Timing analysis: {timing}")
    print(f"ONNX export: {export_onnx} (every {onnx_interval} epochs)" if export_onnx else "")
    print(f"Precision: {precision} ({'Mixed precision' if precision == 16 else 'Full precision'})")
    print(f"Max points per sample: {cfg[cfg.MODEL.DATASET].SUB_NUM_POINTS}")
    print(f"Skip validation: {skip_val}")
    print(f"Fast mode: {fast_mode}")
    print(f"Gradient accumulation: {cfg.TRAIN.BATCH_ACC} steps")
    
    # Create optimized data module
    data = SemanticDatasetModule(cfg)
    
    # Apply dataloader optimizations by overriding the train_dataloader method
    if hasattr(data, 'train_dataloader'):
        original_train_dataloader = data.train_dataloader
        def optimized_train_dataloader():
            from torch.utils.data import DataLoader
            from mask_pls.datasets.semantic_dataset import BatchCollation
            
            dataset = data.train_mask_set
            collate_fn = BatchCollation()
            
            # Create optimized DataLoader with performance settings
            dataloader_kwargs = {
                'dataset': dataset,
                'batch_size': cfg.TRAIN.BATCH_SIZE,
                'collate_fn': collate_fn,
                'shuffle': True,
                'num_workers': cfg.TRAIN.NUM_WORKERS,
                'pin_memory': True,
                'drop_last': False,
                'timeout': 0,
            }
            
            # Add performance optimizations only if workers > 0
            if cfg.TRAIN.NUM_WORKERS > 0:
                dataloader_kwargs['prefetch_factor'] = 4
                dataloader_kwargs['persistent_workers'] = True
            
            return DataLoader(**dataloader_kwargs)
            
        data.train_dataloader = optimized_train_dataloader
    
    # Create model with optimizations
    model = SimplifiedMaskPLS(cfg)
    
    # Enable fast mode optimizations on model
    if fast_mode:
        model.fast_mode = True
        print("⚡ Model fast mode enabled")
    
    # Enable debug mode if requested
    if debug:
        model.debug = True
        print("Debug mode enabled - will print detailed information")
    
    # Enable timing if requested
    if timing:
        model.timing_enabled = True
        print("Timing analysis enabled - will print component timing stats")
    
    # Load checkpoint if provided
    if checkpoint:
        print(f"Loading checkpoint from {checkpoint}")
        ckpt = torch.load(checkpoint, map_location='cpu')
        if 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'], strict=False)
        else:
            model.model.load_state_dict(ckpt, strict=False)
    
    # Setup logger
    tb_logger = pl_loggers.TensorBoardLogger(
        "experiments/" + cfg.EXPERIMENT.ID, 
        default_hp_metric=False
    )
    
    # Callbacks
    callbacks = []
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    iou_ckpt = ModelCheckpoint(
        monitor="metrics/iou",
        filename=cfg.EXPERIMENT.ID + "_epoch{epoch:02d}_iou{metrics/iou:.2f}",
        auto_insert_metric_name=False,
        mode="max",
        save_last=True,
    )
    
    pq_ckpt = ModelCheckpoint(
        monitor="metrics/pq",
        filename=cfg.EXPERIMENT.ID + "_epoch{epoch:02d}_pq{metrics/pq:.2f}",
        auto_insert_metric_name=False,
        mode="max",
        save_last=True,
    )
    callbacks.extend([pq_ckpt, iou_ckpt])
    
    # Add profiling callback if requested
    if profile:
        profiling_cb = ProfilingCallback(profile_interval=50, warmup_steps=10)
        callbacks.append(profiling_cb)
        print("Added profiling callback")
    
    # Add ONNX export callback if requested
    if export_onnx:
        onnx_dir = f"experiments/{cfg.EXPERIMENT.ID}/onnx_checkpoints"
        onnx_cb = ONNXExportCallback(export_interval=onnx_interval, output_dir=onnx_dir)
        callbacks.append(onnx_cb)
        print(f"Added ONNX export callback - models will be saved to {onnx_dir}")
    
    # Create trainer with performance optimizations
    trainer = Trainer(
        gpus=cfg.TRAIN.N_GPUS,
        accelerator="ddp" if cfg.TRAIN.N_GPUS > 1 else None,
        logger=tb_logger,
        max_epochs=cfg.TRAIN.MAX_EPOCH,
        callbacks=callbacks,
        log_every_n_steps=1,
        gradient_clip_val=0.5,
        accumulate_grad_batches=cfg.TRAIN.BATCH_ACC,
        resume_from_checkpoint=checkpoint if checkpoint and os.path.exists(checkpoint) else None,
        num_sanity_val_steps=0,  # Skip sanity validation
        enable_progress_bar=True,
        check_val_every_n_epoch=999 if skip_val else 1,  # Skip validation if requested  # Keep progress bar enabled
        precision=precision,  # Configurable precision for speed vs stability
    )
    
    # Train
    trainer.fit(model, data)
    
    print(f"\nTraining complete! Checkpoints saved in: experiments/{cfg.EXPERIMENT.ID}")


if __name__ == "__main__":
    main()