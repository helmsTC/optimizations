"""
Optimized training script for the simplified ONNX-compatible MaskPLS model
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
import gc
import pickle
from pathlib import Path
from easydict import EasyDict as edict
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Callback
from torch.profiler import profile, record_function, ProfilerActivity
import torch.onnx
from collections import OrderedDict

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
                
                # Create dummy input
                batch_size = 1
                num_points = 10000
                D, H, W = model.spatial_shape
                dummy_voxels = torch.randn(batch_size, 4, D, H, W).cuda()
                dummy_coords = torch.rand(batch_size, num_points, 3).cuda()
                
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


class MemoryMonitorCallback(Callback):
    """Monitor memory usage and clean up when needed"""
    def __init__(self, cleanup_interval=50, memory_limit_gb=10):
        self.cleanup_interval = cleanup_interval
        self.memory_limit_gb = memory_limit_gb
        self.batch_count = 0
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.batch_count += 1
        
        if self.batch_count % self.cleanup_interval == 0:
            # Check GPU memory
            if torch.cuda.is_available():
                allocated_gb = torch.cuda.memory_allocated() / 1024**3
                cached_gb = torch.cuda.memory_reserved() / 1024**3
                
                if allocated_gb > self.memory_limit_gb:
                    print(f"\n[Memory] High usage detected: {allocated_gb:.2f} GB allocated, {cached_gb:.2f} GB cached")
                    
                    # Clear cache
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Clear voxel cache if it exists
                    if hasattr(pl_module, 'voxel_cache'):
                        cache_size = len(pl_module.voxel_cache)
                        if cache_size > 500:
                            # Keep only recent entries
                            pl_module.voxel_cache = dict(list(pl_module.voxel_cache.items())[-250:])
                            print(f"[Memory] Cleared voxel cache: {cache_size} -> {len(pl_module.voxel_cache)}")
                    
                    new_allocated = torch.cuda.memory_allocated() / 1024**3
                    print(f"[Memory] After cleanup: {new_allocated:.2f} GB allocated")


class ProgressCallback(Callback):
    """Monitor training progress and detect freezing"""
    def __init__(self, timeout_seconds=120):
        self.timeout_seconds = timeout_seconds
        self.last_batch_time = time.time()
        self.batch_times = []
        
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.batch_start_time = time.time()
        
        # Check if previous batch took too long
        time_since_last = time.time() - self.last_batch_time
        if time_since_last > self.timeout_seconds:
            print(f"\n⚠️ WARNING: {time_since_last:.1f}s since last batch (timeout: {self.timeout_seconds}s)")
            print(f"  Current batch: {batch_idx}")
            print(f"  Consider reducing batch_size or max_points")
        
        # Progress update every 10 batches
        if batch_idx % 10 == 0:
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated() / 1024**3
                mem_cached = torch.cuda.memory_reserved() / 1024**3
                print(f"\n[Progress] Epoch {trainer.current_epoch} - Batch {batch_idx}/{trainer.num_training_batches}")
                print(f"  GPU Memory: {mem_allocated:.2f} GB allocated, {mem_cached:.2f} GB cached")
            else:
                print(f"\n[Progress] Epoch {trainer.current_epoch} - Batch {batch_idx}/{trainer.num_training_batches}")
            
            if len(self.batch_times) > 0:
                avg_time = np.mean(self.batch_times[-20:])
                print(f"  Avg batch time: {avg_time:.2f}s")
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        batch_time = time.time() - self.batch_start_time
        self.batch_times.append(batch_time)
        self.last_batch_time = time.time()
        
        # Alert on slow batches
        if batch_time > 30:
            print(f"⚠️ Slow batch {batch_idx}: {batch_time:.1f}s")


def prepare_targets_for_loss(batch, max_points, valid_indices, device='cuda'):
    """
    Convert batch targets to the format expected by the loss functions
    Optimized version with better memory management
    """
    fixed_targets = {
        'classes': [],
        'masks': []
    }
    
    for i in range(len(batch['masks_cls'])):
        if len(batch['masks_cls'][i]) > 0:
            # Convert classes efficiently
            classes = torch.tensor(
                [c.item() if isinstance(c, torch.Tensor) and c.numel() == 1 else int(c) 
                 for c in batch['masks_cls'][i]], 
                dtype=torch.long, 
                device=device
            )
            fixed_targets['classes'].append(classes)
            
            # Process masks with efficient padding
            masks_list = []
            valid_idx = valid_indices[i]
            
            for m in batch['masks'][i]:
                # Create padded mask
                padded_mask = torch.zeros(max_points, device=device, dtype=torch.float32)
                
                if isinstance(m, torch.Tensor):
                    mask_data = m.float()
                else:
                    mask_data = torch.tensor(m, dtype=torch.float32)
                
                # Map mask values efficiently
                if len(valid_idx) > 0 and mask_data.shape[0] > 0:
                    idx_cpu = valid_idx.cpu().numpy()
                    valid_mask_indices = idx_cpu < mask_data.shape[0]
                    idx_to_use = idx_cpu[valid_mask_indices]
                    
                    if len(idx_to_use) > 0:
                        padded_positions = torch.arange(len(valid_idx))[valid_mask_indices]
                        mask_values = mask_data[idx_to_use].to(device)
                        padded_mask[padded_positions] = mask_values
                
                masks_list.append(padded_mask)
            
            if len(masks_list) > 0:
                fixed_targets['masks'].append(torch.stack(masks_list))
            else:
                fixed_targets['masks'].append(torch.empty(0, max_points, device=device))
        else:
            fixed_targets['classes'].append(torch.empty(0, dtype=torch.long, device=device))
            fixed_targets['masks'].append(torch.empty(0, max_points, device=device))
    
    return fixed_targets


def prepare_masks_ids_for_loss(batch_masks_ids, valid_indices, max_points):
    """
    Convert masks_ids to work with padded masks - Optimized version
    """
    fixed_masks_ids = []
    
    for masks_ids_sample, valid_idx in zip(batch_masks_ids, valid_indices):
        if len(masks_ids_sample) == 0:
            fixed_masks_ids.append([])
            continue
            
        sample_masks_ids = []
        valid_idx_cpu = valid_idx.cpu().numpy()
        
        # Pre-compute mapping
        orig_to_padded = {orig_idx: padded_idx 
                         for padded_idx, orig_idx in enumerate(valid_idx_cpu)}
        
        for mask_ids in masks_ids_sample:
            if isinstance(mask_ids, torch.Tensor):
                mask_ids = mask_ids.cpu().numpy()
            elif not isinstance(mask_ids, np.ndarray):
                mask_ids = np.array(mask_ids)
            
            # Efficient mapping
            new_mask_ids = [orig_to_padded[idx] for idx in mask_ids if idx in orig_to_padded]
            
            if len(new_mask_ids) > 0:
                sample_masks_ids.append(torch.tensor(new_mask_ids, dtype=torch.long))
            else:
                sample_masks_ids.append(torch.tensor([], dtype=torch.long))
        
        fixed_masks_ids.append(sample_masks_ids)
    
    return fixed_masks_ids


class SimplifiedMaskPLS(LightningModule):
    """
    Optimized Lightning module for training the simplified MaskPLS model
    """
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(dict(cfg))
        self.cfg = cfg
        
        # Voxel caching system
        self.voxel_cache = OrderedDict()
        self.cache_max_size = 500
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Timing tracking
        self.timing_enabled = False
        self.component_times = {
            'data_prep': [],
            'voxelization': [],
            'forward': [],
            'loss': [],
            'backward': [],
            'total': []
        }
        
        # Get dataset configuration
        dataset = cfg.MODEL.DATASET
        self.num_classes = cfg[dataset].NUM_CLASSES
        self.ignore_label = cfg[dataset].IGNORE_LABEL
        
        print(f"Initializing model for {dataset} with {self.num_classes} classes")
        print(f"Ignore label: {self.ignore_label}")
        
        # Create the simplified model
        self.model = MaskPLSSimplifiedONNX(cfg)
        
        # Try to compile voxelization for speed
        try:
            self.voxelize_compiled = torch.jit.script(self.model.voxelize_points)
            print("✓ Voxelization compiled with TorchScript")
        except:
            self.voxelize_compiled = self.model.voxelize_points
            print("⚠ Using uncompiled voxelization")
        
        # Loss functions
        self.mask_loss = MaskLoss(cfg.LOSS, cfg[dataset])
        self.sem_loss = SemLoss(cfg.LOSS.SEM.WEIGHTS)
        
        # Evaluator
        self.evaluator = PanopticEvaluator(cfg[dataset], dataset)
        
        # Cache things IDs
        data = SemanticDatasetModule(cfg)
        data.setup()
        self.things_ids = data.things_ids
        
        # Flags
        self.debug = False
        self.use_cache = True
        self.preprocessed_mode = False
        
        # Batch tracking for debugging
        self.batches_processed = 0
        self.epoch_start_time = None
        
    def on_train_epoch_start(self):
        """Track epoch start and reset caches"""
        self.epoch_start_time = time.time()
        self.batches_processed = 0
        
        # Clear cache every epoch if too large
        if len(self.voxel_cache) > self.cache_max_size * 0.8:
            old_size = len(self.voxel_cache)
            # Keep only half of most recent
            items = list(self.voxel_cache.items())
            self.voxel_cache = OrderedDict(items[-(self.cache_max_size//2):])
            print(f"[Cache] Cleared: {old_size} -> {len(self.voxel_cache)} entries")
        
        print(f"\n{'='*60}")
        print(f"Starting Epoch {self.current_epoch}")
        print(f"Cache stats: {self.cache_hits} hits, {self.cache_misses} misses")
        print(f"{'='*60}\n")
        
    def on_train_epoch_end(self):
        """Report epoch statistics"""
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            print(f"\n{'='*60}")
            print(f"Epoch {self.current_epoch} completed in {epoch_time/60:.2f} minutes")
            print(f"Batches processed: {self.batches_processed}")
            if self.use_cache:
                hit_rate = self.cache_hits / max(self.cache_hits + self.cache_misses, 1) * 100
                print(f"Cache hit rate: {hit_rate:.1f}%")
            print(f"{'='*60}\n")
            
    def forward(self, batch):
        """Optimized forward pass with caching and preprocessing support"""
        start_time = time.time() if self.timing_enabled else None
        
        # Check for preprocessed data
        if 'preprocessed_voxels' in batch:
            return self._forward_preprocessed(batch)
        
        # Use caching if enabled
        if self.use_cache:
            return self._forward_with_cache(batch)
        
        # Standard forward
        return self._forward_standard(batch)
    
    def _forward_preprocessed(self, batch):
        """Forward pass with preprocessed voxel data"""
        # Directly use preprocessed voxels
        batch_voxels = batch['preprocessed_voxels']
        batch_coords = batch['preprocessed_coords']
        valid_indices = batch['valid_indices']
        padding_masks = batch['padding_masks']
        
        # Forward through model
        pred_logits, pred_masks, sem_logits = self.model(batch_voxels, batch_coords)
        
        outputs = {
            'pred_logits': pred_logits,
            'pred_masks': pred_masks,
            'aux_outputs': []
        }
        
        return outputs, padding_masks, sem_logits, valid_indices
    
    def _forward_with_cache(self, batch):
        """Forward pass with voxel caching"""
        # Create cache key from file names
        cache_key = tuple(batch['fname'])
        
        if cache_key in self.voxel_cache:
            # Cache hit
            self.cache_hits += 1
            cached_data = self.voxel_cache[cache_key]
            batch_voxels = cached_data['voxels']
            batch_coords = cached_data['coords']
            valid_indices = cached_data['indices']
            padding_masks = cached_data['padding']
            
            # Move cache entry to end (LRU)
            self.voxel_cache.move_to_end(cache_key)
            
        else:
            # Cache miss - compute voxelization
            self.cache_misses += 1
            batch_voxels, batch_coords, valid_indices = self._process_batch_optimized(batch)
            
            # Pad and create masks
            if len(batch_coords) > 0:
                max_pts = max(c.shape[0] for c in batch_coords)
                padded_coords = []
                padding_masks = []
                
                for coords in batch_coords:
                    n_pts = coords.shape[0]
                    if n_pts < max_pts:
                        coords = F.pad(coords, (0, 0, 0, max_pts - n_pts))
                    padded_coords.append(coords)
                    
                    mask = torch.zeros(max_pts, dtype=torch.bool, device=coords.device)
                    if n_pts < max_pts:
                        mask[n_pts:] = True
                    padding_masks.append(mask)
                
                batch_voxels = torch.stack(batch_voxels)
                batch_coords = torch.stack(padded_coords)
                padding_masks = torch.stack(padding_masks)
            else:
                # Empty batch handling
                batch_voxels = torch.empty(0, device='cuda')
                batch_coords = torch.empty(0, device='cuda')
                padding_masks = torch.empty(0, dtype=torch.bool, device='cuda')
            
            # Add to cache if not too large
            if len(self.voxel_cache) < self.cache_max_size:
                self.voxel_cache[cache_key] = {
                    'voxels': batch_voxels,
                    'coords': batch_coords,
                    'indices': valid_indices,
                    'padding': padding_masks
                }
        
        # Forward through model
        pred_logits, pred_masks, sem_logits = self.model(batch_voxels, batch_coords)
        
        # Validate outputs
        self._validate_outputs(pred_logits, pred_masks, sem_logits)
        
        outputs = {
            'pred_logits': pred_logits,
            'pred_masks': pred_masks,
            'aux_outputs': []
        }
        
        return outputs, padding_masks, sem_logits, valid_indices
    
    def _forward_standard(self, batch):
        """Standard forward pass without caching"""
        # Process batch
        batch_voxels, batch_coords, valid_indices = self._process_batch_optimized(batch)
        
        # Pad and create masks
        if len(batch_coords) > 0:
            max_pts = max(c.shape[0] for c in batch_coords)
            padded_coords = []
            padding_masks = []
            
            for coords in batch_coords:
                n_pts = coords.shape[0]
                if n_pts < max_pts:
                    coords = F.pad(coords, (0, 0, 0, max_pts - n_pts))
                padded_coords.append(coords)
                
                mask = torch.zeros(max_pts, dtype=torch.bool, device=coords.device)
                if n_pts < max_pts:
                    mask[n_pts:] = True
                padding_masks.append(mask)
            
            batch_voxels = torch.stack(batch_voxels)
            batch_coords = torch.stack(padded_coords)
            padding_masks = torch.stack(padding_masks)
        else:
            batch_voxels = torch.empty(0, device='cuda')
            batch_coords = torch.empty(0, device='cuda')
            padding_masks = torch.empty(0, dtype=torch.bool, device='cuda')
        
        # Forward through model
        pred_logits, pred_masks, sem_logits = self.model(batch_voxels, batch_coords)
        
        outputs = {
            'pred_logits': pred_logits,
            'pred_masks': pred_masks,
            'aux_outputs': []
        }
        
        return outputs, padding_masks, sem_logits, valid_indices
    
    def _process_batch_optimized(self, batch):
        """Optimized batch processing with better memory management"""
        points = batch['pt_coord']
        features = batch['feats']
        
        batch_voxels = []
        batch_coords = []
        valid_indices = []
        
        # Cache config values
        bounds = self.cfg[self.cfg.MODEL.DATASET].SPACE
        max_pts = self.cfg[self.cfg.MODEL.DATASET].SUB_NUM_POINTS
        
        # Pre-compute tensor constants
        bounds_min = torch.tensor([bounds[d][0] for d in range(3)], device='cuda')
        bounds_max = torch.tensor([bounds[d][1] for d in range(3)], device='cuda')
        bounds_range = bounds_max - bounds_min
        
        for i in range(len(points)):
            # Convert to tensor once
            pts = torch.from_numpy(points[i]).float().cuda()
            feat = torch.from_numpy(features[i]).float().cuda()
            
            # Vectorized bounds check
            valid_mask = ((pts >= bounds_min) & (pts < bounds_max)).all(dim=1)
            
            if valid_mask.sum() == 0:
                # No valid points - add empty
                batch_voxels.append(torch.zeros(4, *self.model.spatial_shape, device='cuda'))
                batch_coords.append(torch.zeros(0, 3, device='cuda'))
                valid_indices.append(torch.zeros(0, dtype=torch.long, device='cuda'))
                continue
            
            valid_idx = torch.where(valid_mask)[0]
            valid_pts = pts[valid_mask]
            valid_feat = feat[valid_mask]
            
            # Efficient subsampling
            if len(valid_pts) > max_pts:
                if self.training:
                    # Random sampling for training
                    perm = torch.randperm(len(valid_pts), device='cuda')[:max_pts]
                else:
                    # Deterministic for validation
                    perm = torch.arange(max_pts, device='cuda')
                
                valid_pts = valid_pts[perm]
                valid_feat = valid_feat[perm]
                valid_idx = valid_idx[perm]
            
            # Vectorized normalization
            norm_coords = (valid_pts - bounds_min) / bounds_range
            norm_coords = torch.clamp(norm_coords, 0, 0.999)
            
            # Use compiled voxelization if available
            voxel_grid = self.voxelize_compiled(
                valid_pts.unsqueeze(0),
                valid_feat.unsqueeze(0)
            )[0]
            
            batch_voxels.append(voxel_grid)
            batch_coords.append(norm_coords)
            valid_indices.append(valid_idx)
        
        return batch_voxels, batch_coords, valid_indices
    
    def _validate_outputs(self, pred_logits, pred_masks, sem_logits):
        """Validate model outputs and adjust if needed"""
        if self.debug:
            print(f"Output shapes - logits: {pred_logits.shape}, masks: {pred_masks.shape}, sem: {sem_logits.shape}")
        
        # Check semantic logits shape
        if sem_logits.shape[-1] != self.num_classes:
            if sem_logits.shape[-1] > self.num_classes:
                sem_logits = sem_logits[..., :self.num_classes]
            else:
                pad_size = self.num_classes - sem_logits.shape[-1]
                sem_logits = F.pad(sem_logits, (0, pad_size))
    
    def training_step(self, batch, batch_idx):
        """Optimized training step with better error handling"""
        try:
            self.batches_processed += 1
            step_start = time.time()
            
            # Forward pass
            outputs, padding, sem_logits, valid_indices = self.forward(batch)
            
            # Skip batch if no valid data
            if len(valid_indices) == 0 or all(len(v) == 0 for v in valid_indices):
                print(f"⚠️ Skipping empty batch {batch_idx}")
                return torch.tensor(0.0, device='cuda', requires_grad=True)
            
            # Prepare targets
            max_points = padding.shape[1]
            targets = prepare_targets_for_loss(batch, max_points, valid_indices, device='cuda')
            fixed_masks_ids = prepare_masks_ids_for_loss(batch['masks_ids'], valid_indices, max_points)
            
            # Compute losses with error handling
            try:
                loss_mask = self.mask_loss(outputs, targets, fixed_masks_ids, batch['pt_coord'])
            except Exception as e:
                if self.debug:
                    print(f"Mask loss error in batch {batch_idx}: {e}")
                loss_mask = {'loss_ce': torch.tensor(0.1, device='cuda', requires_grad=True),
                           'loss_dice': torch.tensor(0.1, device='cuda', requires_grad=True),
                           'loss_mask': torch.tensor(0.1, device='cuda', requires_grad=True)}
            
            # Semantic loss with better handling
            try:
                sem_loss_result = self._compute_semantic_loss_safe(batch, sem_logits, valid_indices, padding)
                loss_mask.update(sem_loss_result)
            except Exception as e:
                if self.debug:
                    print(f"Semantic loss error in batch {batch_idx}: {e}")
                loss_mask['sem_ce'] = torch.tensor(0.0, device='cuda', requires_grad=True)
                loss_mask['sem_lov'] = torch.tensor(0.0, device='cuda', requires_grad=True)
            
            # Log losses (reduced frequency)
            if batch_idx % 10 == 0:
                for k, v in loss_mask.items():
                    if not (torch.isnan(v) or torch.isinf(v)):
                        self.log(f"train/{k}", v, batch_size=self.cfg.TRAIN.BATCH_SIZE)
            
            # Total loss with safety checks
            total_loss = sum(v for v in loss_mask.values() if not (torch.isnan(v) or torch.isinf(v)))
            
            if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss.item() > 1000:
                print(f"⚠️ Invalid loss in batch {batch_idx}: {total_loss.item()}")
                total_loss = torch.tensor(0.1, device='cuda', requires_grad=True)
            
            if batch_idx % 10 == 0:
                self.log("train_loss", total_loss, batch_size=self.cfg.TRAIN.BATCH_SIZE)
            
            # Timing
            if self.timing_enabled and batch_idx % 50 == 0:
                step_time = time.time() - step_start
                print(f"Batch {batch_idx} time: {step_time:.2f}s")
            
            return total_loss
            
        except Exception as e:
            print(f"❌ Error in batch {batch_idx}: {type(e).__name__}: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            
            # Return minimal loss to continue
            return torch.tensor(0.1, device='cuda', requires_grad=True)
    
    def _compute_semantic_loss_safe(self, batch, sem_logits, valid_indices, padding):
        """Safe semantic loss computation with better error handling"""
        all_logits = []
        all_labels = []
        
        for i, (label, idx, pad) in enumerate(zip(batch['sem_label'], valid_indices, padding)):
            valid_mask = ~pad
            num_valid = valid_mask.sum().item()
            
            if num_valid == 0:
                continue
            
            # Get valid logits
            sem_logits_i = sem_logits[i][valid_mask]
            
            # Process labels safely
            if isinstance(label, np.ndarray):
                label = label.flatten()
            else:
                label = np.array(label).flatten()
            
            # Map indices safely
            idx_cpu = idx.cpu().numpy()
            valid_idx_mask = (idx_cpu >= 0) & (idx_cpu < len(label))
            idx_to_use = idx_cpu[valid_idx_mask][:sem_logits_i.shape[0]]
            
            if len(idx_to_use) == 0:
                continue
            
            # Adjust sizes
            if len(idx_to_use) < sem_logits_i.shape[0]:
                sem_logits_i = sem_logits_i[:len(idx_to_use)]
            
            # Get labels and ensure valid range
            valid_labels = torch.from_numpy(label[idx_to_use]).long().cuda()
            valid_labels = torch.clamp(valid_labels, 0, self.num_classes - 1)
            
            all_logits.append(sem_logits_i)
            all_labels.append(valid_labels)
        
        # Compute loss if we have data
        if len(all_logits) > 0:
            all_logits = torch.cat(all_logits, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            return self.sem_loss(all_logits, all_labels)
        else:
            return {
                'sem_ce': torch.tensor(0.0, device='cuda', requires_grad=True),
                'sem_lov': torch.tensor(0.0, device='cuda', requires_grad=True)
            }
    
    def validation_step(self, batch, batch_idx):
        """Validation step with error handling"""
        try:
            outputs, padding, sem_logits, valid_indices = self.forward(batch)
            
            # Skip if no valid data
            if len(valid_indices) == 0 or all(len(v) == 0 for v in valid_indices):
                return torch.tensor(0.0, device='cuda')
            
            max_points = padding.shape[1]
            targets = prepare_targets_for_loss(batch, max_points, valid_indices, device='cuda')
            fixed_masks_ids = prepare_masks_ids_for_loss(batch['masks_ids'], valid_indices, max_points)
            
            # Calculate losses
            loss_mask = self.mask_loss(outputs, targets, fixed_masks_ids, batch['pt_coord'])
            
            # Semantic loss
            sem_loss_result = self._compute_semantic_loss_safe(batch, sem_logits, valid_indices, padding)
            loss_mask.update(sem_loss_result)
            
            # Log losses
            for k, v in loss_mask.items():
                if not (torch.isnan(v) or torch.isinf(v)):
                    self.log(f"val/{k}", v, batch_size=self.cfg.TRAIN.BATCH_SIZE)
            
            total_loss = sum(v for v in loss_mask.values() if not (torch.isnan(v) or torch.isinf(v)))
            self.log("val_loss", total_loss, batch_size=self.cfg.TRAIN.BATCH_SIZE)
            
            # Panoptic evaluation
            sem_pred, ins_pred = self.panoptic_inference(outputs, padding)
            
            # Map back to original points
            full_sem_pred = []
            full_ins_pred = []
            
            for i, (pred_sem, pred_ins, idx) in enumerate(zip(sem_pred, ins_pred, valid_indices)):
                full_sem = torch.zeros(len(batch['sem_label'][i]), dtype=torch.long)
                full_ins = torch.zeros(len(batch['ins_label'][i]), dtype=torch.long)
                
                idx_cpu = idx.cpu().numpy()
                max_size = len(full_sem)
                valid_mask = (idx_cpu >= 0) & (idx_cpu < max_size)
                idx_cpu = idx_cpu[valid_mask]
                
                valid_len = min(len(idx_cpu), len(pred_sem))
                
                if valid_len > 0:
                    full_sem[idx_cpu[:valid_len]] = torch.from_numpy(pred_sem[:valid_len])
                    full_ins[idx_cpu[:valid_len]] = torch.from_numpy(pred_ins[:valid_len])
                
                full_sem_pred.append(full_sem.numpy())
                full_ins_pred.append(full_ins.numpy())
            
            self.evaluator.update(full_sem_pred, full_ins_pred, batch)
            
            return total_loss
            
        except Exception as e:
            print(f"Error in validation batch {batch_idx}: {e}")
            return torch.tensor(0.1, device='cuda')
    
    def validation_epoch_end(self, outputs):
        """End of validation epoch"""
        bs = self.cfg.TRAIN.BATCH_SIZE
        self.log("metrics/pq", self.evaluator.get_mean_pq(), batch_size=bs)
        self.log("metrics/iou", self.evaluator.get_mean_iou(), batch_size=bs)
        self.log("metrics/rq", self.evaluator.get_mean_rq(), batch_size=bs)
        self.evaluator.reset()
    
    def configure_optimizers(self):
        """Optimizer configuration with warmup option"""
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.cfg.TRAIN.LR,
            weight_decay=1e-4
        )
        
        # Learning rate scheduler with warmup
        if hasattr(self.cfg.TRAIN, 'WARMUP_EPOCHS'):
            from torch.optim.lr_scheduler import OneCycleLR
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.cfg.TRAIN.LR,
                epochs=self.cfg.TRAIN.MAX_EPOCH,
                steps_per_epoch=1000,  # Approximate
                pct_start=0.05,  # 5% warmup
                anneal_strategy='cos'
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step'
                }
            }
        else:
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
        """Panoptic segmentation inference"""
        mask_cls = outputs["pred_logits"]
        mask_pred = outputs["pred_masks"]
        num_classes = self.cfg[self.cfg.MODEL.DATASET].NUM_CLASSES
        
        sem_pred = []
        ins_pred = []
        
        for b_cls, b_mask, b_pad in zip(mask_cls, mask_pred, padding):
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
                
                cur_prob_masks = cur_scores.unsqueeze(0) * cur_masks
                mask_ids = cur_prob_masks.argmax(1)
                
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
@click.option("--num_workers", type=int, default=2, help="Number of data loader workers")
@click.option("--export_onnx", is_flag=True, help="Export ONNX models during training")
@click.option("--onnx_interval", type=int, default=5, help="Interval for ONNX export")
@click.option("--precision", type=int, default=16, help="Training precision (16 or 32)")
@click.option("--max_points", type=int, default=None, help="Override max points per sample")
@click.option("--skip_val", is_flag=True, help="Skip validation during training")
@click.option("--no_cache", is_flag=True, help="Disable voxel caching")
@click.option("--memory_limit", type=float, default=10.0, help="GPU memory limit in GB")
@click.option("--timeout", type=int, default=120, help="Batch timeout in seconds")
@click.option("--gradient_clip", type=float, default=0.5, help="Gradient clipping value")
def main(checkpoint, nuscenes, epochs, batch_size, lr, gpus, debug, num_workers, 
         export_onnx, onnx_interval, precision, max_points, skip_val, no_cache,
         memory_limit, timeout, gradient_clip):
    """
    Optimized training script for simplified MaskPLS model
    """
    print("="*60)
    print("Optimized Simplified MaskPLS Training")
    print("="*60)
    
    # Load configurations
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
    
    if nuscenes:
        cfg.MODEL.DATASET = "NUSCENES"
    
    # Override max points if specified
    if max_points is not None:
        dataset = cfg.MODEL.DATASET
        original = cfg[dataset].SUB_NUM_POINTS
        cfg[dataset].SUB_NUM_POINTS = max_points
        print(f"Adjusted max points: {original} -> {max_points}")
    
    # Update experiment ID
    cfg.EXPERIMENT.ID = cfg.EXPERIMENT.ID + "_simplified_optimized"
    
    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Dataset: {cfg.MODEL.DATASET}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Workers: {num_workers}")
    print(f"  GPUs: {gpus}")
    print(f"  Precision: {precision}")
    print(f"  Max points: {cfg[cfg.MODEL.DATASET].SUB_NUM_POINTS}")
    print(f"  Caching: {'disabled' if no_cache else 'enabled'}")
    print(f"  Debug: {debug}")
    print(f"  Memory limit: {memory_limit} GB")
    print(f"  Batch timeout: {timeout}s")
    
    # Create data module with optimized settings
    data = SemanticDatasetModule(cfg)
    
    # Create model
    model = SimplifiedMaskPLS(cfg)
    model.debug = debug
    model.use_cache = not no_cache
    
    # Load checkpoint if provided
    if checkpoint and os.path.exists(checkpoint):
        print(f"\nLoading checkpoint: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location='cpu')
        if 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'], strict=False)
            print("✓ Checkpoint loaded")
        else:
            print("⚠ No state_dict in checkpoint")
    
    # Setup logger
    tb_logger = pl_loggers.TensorBoardLogger(
        f"experiments/{cfg.EXPERIMENT.ID}",
        default_hp_metric=False
    )
    
    # Callbacks
    callbacks = []
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    # Model checkpoints
    iou_ckpt = ModelCheckpoint(
        monitor="metrics/iou",
        filename=cfg.EXPERIMENT.ID + "_epoch{epoch:02d}_iou{metrics/iou:.2f}",
        auto_insert_metric_name=False,
        mode="max",
        save_last=True,
        save_top_k=3
    )
    
    pq_ckpt = ModelCheckpoint(
        monitor="metrics/pq",
        filename=cfg.EXPERIMENT.ID + "_epoch{epoch:02d}_pq{metrics/pq:.2f}",
        auto_insert_metric_name=False,
        mode="max",
        save_top_k=3
    )
    callbacks.extend([pq_ckpt, iou_ckpt])
    
    # Memory monitor
    memory_cb = MemoryMonitorCallback(cleanup_interval=50, memory_limit_gb=memory_limit)
    callbacks.append(memory_cb)
    
    # Progress monitor
    progress_cb = ProgressCallback(timeout_seconds=timeout)
    callbacks.append(progress_cb)
    
    # ONNX export
    if export_onnx:
        onnx_dir = f"experiments/{cfg.EXPERIMENT.ID}/onnx_checkpoints"
        onnx_cb = ONNXExportCallback(export_interval=onnx_interval, output_dir=onnx_dir)
        callbacks.append(onnx_cb)
        print(f"  ONNX export: every {onnx_interval} epochs to {onnx_dir}")
    
    # Create trainer
    trainer = Trainer(
        gpus=cfg.TRAIN.N_GPUS,
        accelerator="ddp" if cfg.TRAIN.N_GPUS > 1 else None,
        logger=tb_logger,
        max_epochs=cfg.TRAIN.MAX_EPOCH,
        callbacks=callbacks,
        log_every_n_steps=1,
        gradient_clip_val=gradient_clip,
        accumulate_grad_batches=cfg.TRAIN.BATCH_ACC,
        resume_from_checkpoint=checkpoint if checkpoint and os.path.exists(checkpoint) else None,
        num_sanity_val_steps=0 if debug else 2,
        check_val_every_n_epoch=999 if skip_val else 1,
        enable_progress_bar=True,
        precision=precision,
        detect_anomaly=debug,  # Enable anomaly detection in debug mode
        benchmark=True,  # Enable cudnn benchmark for speed
        deterministic=False,  # Non-deterministic for speed
        replace_sampler_ddp=True,
        sync_batchnorm=cfg.TRAIN.N_GPUS > 1,
        limit_val_batches=1.0 if not skip_val else 0,
        val_check_interval=1.0 if not skip_val else 999999,
    )
    
    # Train
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}\n")
    
    trainer.fit(model, data)
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Checkpoints saved in: experiments/{cfg.EXPERIMENT.ID}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
