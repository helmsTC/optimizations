#!/usr/bin/env python3
"""
Evaluator for original MaskPLS model with MinkowskiEngine backbone
Provides timing, metrics, and optional prediction saving
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import yaml
from pathlib import Path
from easydict import EasyDict as edict
import click
from tqdm import tqdm
import time
from collections import defaultdict

# Add the original model path to Python path
original_path = os.path.join(os.path.dirname(__file__), "../original/MaskPLS")
if original_path not in sys.path:
    sys.path.insert(0, original_path)

# Import original components
from mask_pls.models.mink import MinkEncoderDecoder
from mask_pls.models.decoder import MaskedTransformerDecoder
from mask_pls.datasets.semantic_dataset import SemanticDatasetModule
from mask_pls.utils.evaluate_panoptic import PanopticEvaluator
import MinkowskiEngine as ME


def get_config():
    """Load configuration for the original model"""
    def getDir(obj):
        return os.path.dirname(os.path.abspath(obj))
    
    # Load from original config location
    config_base = os.path.join(getDir(__file__), "../original/MaskPLS/mask_pls/config")
    
    model_cfg = edict(yaml.safe_load(open(os.path.join(config_base, "model.yaml"))))
    backbone_cfg = edict(yaml.safe_load(open(os.path.join(config_base, "backbone.yaml"))))
    decoder_cfg = edict(yaml.safe_load(open(os.path.join(config_base, "decoder.yaml"))))
    
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    
    # Evaluation settings
    cfg.TRAIN.BATCH_SIZE = 1
    cfg.TRAIN.NUM_WORKERS = 4
    cfg.TRAIN.SUBSAMPLE = False
    cfg.TRAIN.AUG = False
    
    return cfg


class OriginalStandaloneModel(nn.Module):
    """Standalone inference model using original MinkowskiEngine architecture"""
    
    def __init__(self, cfg, things_ids):
        super().__init__()
        
        dataset = cfg.MODEL.DATASET
        self.cfg = cfg
        self.num_classes = cfg[dataset].NUM_CLASSES
        self.ignore_label = cfg[dataset].IGNORE_LABEL
        self.things_ids = things_ids
        self.overlap_threshold = cfg.MODEL.OVERLAP_THRESHOLD
        
        # Create the ORIGINAL components
        backbone = MinkEncoderDecoder(cfg.BACKBONE, cfg[dataset])
        self.backbone = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(backbone)
        
        self.decoder = MaskedTransformerDecoder(
            cfg.DECODER,
            cfg.BACKBONE,
            cfg[dataset]
        )
    
    @torch.no_grad()
    def forward(self, batch_dict):
        """Forward pass matching the original training model"""
        feats, coords, pad_masks, bb_logits = self.backbone(batch_dict)
        outputs, padding = self.decoder(feats, coords, pad_masks)
        return outputs, padding, bb_logits
    
    @torch.no_grad()
    def forward_with_timing(self, batch_dict):
        """Forward pass with detailed timing"""
        # Backbone timing
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        backbone_start = time.perf_counter()
        
        feats, coords, pad_masks, bb_logits = self.backbone(batch_dict)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        backbone_time = time.perf_counter() - backbone_start
        
        # Decoder timing
        decoder_start = time.perf_counter()
        
        outputs, padding = self.decoder(feats, coords, pad_masks)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        decoder_time = time.perf_counter() - decoder_start
        
        inference_time = backbone_time + decoder_time
        
        return outputs, padding, bb_logits, {
            'backbone_time': backbone_time,
            'decoder_time': decoder_time,
            'inference_time': inference_time
        }
    
    def panoptic_inference(self, outputs, padding):
        """Panoptic segmentation inference"""
        mask_cls = outputs["pred_logits"]
        mask_pred = outputs["pred_masks"]
        things_ids = self.things_ids
        num_classes = self.num_classes
        
        sem_pred = []
        ins_pred = []
        
        for mask_cls, mask_pred, pad in zip(mask_cls, mask_pred, padding):
            scores, labels = mask_cls.max(-1)
            mask_pred = mask_pred[~pad].sigmoid()
            keep = labels.ne(num_classes)
            
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred[:, keep]
            
            cur_prob_masks = cur_scores.unsqueeze(0) * cur_masks
            
            panoptic_seg = torch.zeros(
                (cur_masks.shape[0]), dtype=torch.int32, device=cur_masks.device
            )
            sem = torch.zeros_like(panoptic_seg)
            ins = torch.zeros_like(panoptic_seg)
            
            segment_id = 0
            
            if cur_masks.shape[1] == 0:
                sem_pred.append(sem.cpu().numpy())
                ins_pred.append(ins.cpu().numpy())
            else:
                cur_mask_ids = cur_prob_masks.argmax(1)
                stuff_memory_list = {}
                
                for k in range(cur_classes.shape[0]):
                    pred_class = cur_classes[k].item()
                    isthing = pred_class in things_ids
                    mask_area = (cur_mask_ids == k).sum().item()
                    original_area = (cur_masks[:, k] >= 0.5).sum().item()
                    mask = (cur_mask_ids == k) & (cur_masks[:, k] >= 0.5)
                    
                    if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                        if mask_area / original_area < self.overlap_threshold:
                            continue
                        if not isthing:
                            if int(pred_class) in stuff_memory_list.keys():
                                panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                                continue
                            else:
                                stuff_memory_list[int(pred_class)] = segment_id + 1
                        segment_id += 1
                        panoptic_seg[mask] = segment_id
                        
                        sem[mask] = pred_class
                        if isthing:
                            ins[mask] = segment_id
                        else:
                            ins[mask] = 0
                            
                sem_pred.append(sem.cpu().numpy())
                ins_pred.append(ins.cpu().numpy())
        
        return sem_pred, ins_pred


class TimingStats:
    """Track and report timing statistics"""
    def __init__(self):
        self.backbone_times = []
        self.decoder_times = []
        self.inference_times = []
        self.postprocess_times = []
        self.total_times = []
        
    def add_sample(self, backbone_time, decoder_time, inference_time, postprocess_time, total_time):
        self.backbone_times.append(backbone_time)
        self.decoder_times.append(decoder_time)
        self.inference_times.append(inference_time)
        self.postprocess_times.append(postprocess_time)
        self.total_times.append(total_time)
    
    def get_stats(self):
        """Get timing statistics"""
        if not self.total_times:
            return {}
        
        def get_stat(times):
            times = np.array(times) * 1000  # Convert to ms
            return {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'median': np.median(times)
            }
        
        return {
            'backbone': get_stat(self.backbone_times),
            'decoder': get_stat(self.decoder_times),
            'inference': get_stat(self.inference_times),
            'postprocess': get_stat(self.postprocess_times),
            'total': get_stat(self.total_times),
            'fps': 1000.0 / np.mean(np.array(self.total_times) * 1000)
        }
    
    def print_stats(self):
        """Print formatted timing statistics"""
        stats = self.get_stats()
        if not stats:
            return
        
        print("\n" + "="*60)
        print("TIMING STATISTICS (ms per point cloud)")
        print("="*60)
        
        components = ['backbone', 'decoder', 'inference', 'postprocess', 'total']
        headers = ['Component', 'Mean±Std', 'Min', 'Max', 'Median']
        
        row_format = "{:<15} {:<15} {:<10} {:<10} {:<10}"
        print(row_format.format(*headers))
        print("-" * 60)
        
        for comp in components:
            if comp in stats:
                s = stats[comp]
                mean_std = f"{s['mean']:.2f}±{s['std']:.2f}"
                print(row_format.format(
                    comp.capitalize(),
                    mean_std,
                    f"{s['min']:.2f}",
                    f"{s['max']:.2f}",
                    f"{s['median']:.2f}"
                ))
        
        print("-" * 60)
        print(f"Average FPS: {stats['fps']:.2f} Hz")
        print("="*60)


def create_save_dirs(dataset, output_dir=None, sequences=None):
    """Create directories for saving predictions"""
    def getDir(obj):
        return os.path.dirname(os.path.abspath(obj))
    
    if output_dir:
        base_dir = output_dir
    else:
        base_dir = os.path.join(getDir(__file__), "..", "output")
    
    if dataset == "NUSCENES":
        results_dir = os.path.join(base_dir, "nuscenes_test")
        os.makedirs(results_dir, exist_ok=True)
    else:  # KITTI
        results_dir = os.path.join(base_dir, "test", "sequences")
        os.makedirs(results_dir, exist_ok=True)
        
        if sequences:
            for seq in sequences:
                seq_str = str(seq).zfill(2)
                sub_dir = os.path.join(results_dir, seq_str, "predictions")
                os.makedirs(sub_dir, exist_ok=True)
                print(f"  Created directory for sequence {seq_str}")
        else:
            # Default test sequences
            for i in range(11, 22):
                sub_dir = os.path.join(results_dir, str(i).zfill(2), "predictions")
                os.makedirs(sub_dir, exist_ok=True)
    
    return results_dir


def save_predictions(sem_pred, ins_pred, batch, results_dir, class_inv_lut, dataset="KITTI"):
    """Save predictions in KITTI/NuScenes format"""
    for i in range(len(sem_pred)):
        sem = sem_pred[i]
        ins = ins_pred[i]
        
        # Convert to original labels
        sem_inv = class_inv_lut[sem].astype(np.uint32)
        label = sem_inv.reshape(-1, 1) + ((ins.astype(np.uint32) << 16) & 0xFFFF0000).reshape(-1, 1)
        
        fname = batch['fname'][i]
        
        if dataset == "KITTI":
            # Extract sequence and frame number
            seq = fname.split("/")[-3]
            pcd_fname = fname.split("/")[-1].split(".")[-2] + ".label"
            save_path = os.path.join(results_dir, seq, "predictions", pcd_fname)
        else:  # NUSCENES
            token = batch['token'][i] if 'token' in batch else fname
            name = token + "_panoptic.npz"
            save_path = os.path.join(results_dir, name)
            
        # Save the file
        if dataset == "KITTI":
            label.reshape(-1).astype(np.uint32).tofile(save_path)
        else:
            np.savez_compressed(save_path, data=label.reshape(-1).astype(np.uint16))


def test_model(model_path, cfg, data_module, max_batches=None, 
               save_predictions_flag=False, output_dir=None,
               warmup_batches=3, timing_enabled=True):
    """Test the original model with comprehensive evaluation"""
    
    print(f"\nLoading model from {model_path}")
    
    # Load saved model
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Print checkpoint information
    print("\nCheckpoint contents:")
    for key in checkpoint.keys():
        if key not in ['model_state_dict', 'backbone_state_dict', 'decoder_state_dict', 'config']:
            if isinstance(checkpoint[key], (list, tuple, int, float, str)):
                print(f"  {key}: {checkpoint[key]}")
            elif isinstance(checkpoint[key], dict):
                print(f"  {key}: dict with {len(checkpoint[key])} items")
    
    # Handle configuration properly
    # Always use the passed cfg as base, don't replace it
    model_cfg = cfg  # Use the configuration passed to the function
    
    # Get dataset name
    dataset = model_cfg.MODEL.DATASET if hasattr(model_cfg, 'MODEL') and hasattr(model_cfg.MODEL, 'DATASET') else 'KITTI'
    
    # Ensure dataset config exists
    if dataset not in model_cfg:
        # Create dataset configuration if missing
        if dataset == 'KITTI':
            model_cfg[dataset] = edict({
                'PATH': 'data/kitti',
                'CONFIG': 'mask_pls/datasets/semantic-kitti.yaml',
                'NUM_CLASSES': 20,
                'IGNORE_LABEL': 0,
                'MIN_POINTS': 10,
                'SPACE': [[-48.0, 48.0], [-48.0, 48.0], [-4.0, 1.5]],
                'SUB_NUM_POINTS': 80000
            })
        elif dataset == 'NUSCENES':
            model_cfg[dataset] = edict({
                'PATH': 'data/nuscenes',
                'CONFIG': 'mask_pls/datasets/semantic-nuscenes.yaml',
                'NUM_CLASSES': 17,
                'IGNORE_LABEL': 0,
                'MIN_POINTS': 10,
                'SPACE': [[-50.0, 50.0], [-50.0, 50.0], [-5.0, 3]],
                'SUB_NUM_POINTS': 50000
            })
    
    dataset_cfg = model_cfg[dataset]
    
    things_ids = checkpoint.get('things_ids', data_module.things_ids)
    print(f"\nUsing things_ids: {things_ids}")
    
    # Create model
    model = OriginalStandaloneModel(model_cfg, things_ids)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Loaded full model state dict")
    else:
        # Load individual components
        if 'backbone_state_dict' in checkpoint:
            model.backbone.load_state_dict(checkpoint['backbone_state_dict'])
            print("✓ Loaded backbone state dict")
        if 'decoder_state_dict' in checkpoint:
            model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
            print("✓ Loaded decoder state dict")
    
    # Move to GPU and set eval mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"\nModel Configuration:")
    print(f"  Device: {device}")
    print(f"  Architecture: MinkowskiEngine")
    print(f"  Dataset: {dataset}")
    print(f"  Num classes: {model.num_classes}")
    print(f"  Overlap threshold: {model.overlap_threshold}")
    
    # Setup dataloader
    dataloader = data_module.val_dataloader()
    
    # Create save directory if needed
    results_dir = None
    if save_predictions_flag:
        sequences = None
        if hasattr(data_module, 'val_mask_set'):
            dataset_obj = data_module.val_mask_set.dataset
            if hasattr(dataset_obj, 'split'):
                sequences = dataset_obj.split.get('valid', None)
        
        results_dir = create_save_dirs(dataset, output_dir, sequences)
        print(f"\nSaving predictions to: {results_dir}")
    
    # Setup evaluator with proper configuration
    evaluator = PanopticEvaluator(dataset_cfg, dataset)
    evaluator.reset()
    
    # Get class inverse lookup table
    class_inv_lut = evaluator.get_class_inv_lut()
    
    # Timing statistics
    timing_stats = TimingStats() if timing_enabled else None
    
    # Warmup runs
    if warmup_batches > 0 and timing_enabled:
        print(f"\nWarming up with {warmup_batches} batches...")
        for i, batch in enumerate(dataloader):
            if i >= warmup_batches:
                break
            with torch.no_grad():
                outputs, padding, bb_logits = model(batch)
                sem_pred, ins_pred = model.panoptic_inference(outputs, padding)
        torch.cuda.empty_cache()
    
    # Main evaluation loop
    print("\nRunning evaluation...")
    
    total_batches = len(dataloader)
    if max_batches:
        total_batches = min(total_batches, max_batches)
    
    successful = 0
    saved_count = 0
    
    with tqdm(total=total_batches, desc="Evaluating") as pbar:
        for i, batch in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break
            
            try:
                # Total timing
                total_start = time.perf_counter()
                
                with torch.no_grad():
                    if timing_enabled:
                        # Inference with timing
                        outputs, padding, bb_logits, timings = model.forward_with_timing(batch)
                        
                        # Post-processing timing
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                        postprocess_start = time.perf_counter()
                        
                        sem_pred, ins_pred = model.panoptic_inference(outputs, padding)
                        
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                        postprocess_time = time.perf_counter() - postprocess_start
                        
                        total_time = time.perf_counter() - total_start
                        
                        # Record timing
                        batch_size = len(sem_pred)
                        for _ in range(batch_size):
                            timing_stats.add_sample(
                                timings['backbone_time'] / batch_size,
                                timings['decoder_time'] / batch_size,
                                timings['inference_time'] / batch_size,
                                postprocess_time / batch_size,
                                total_time / batch_size
                            )
                    else:
                        # Standard inference without timing
                        outputs, padding, bb_logits = model(batch)
                        sem_pred, ins_pred = model.panoptic_inference(outputs, padding)
                
                # Save predictions if requested
                if save_predictions_flag and results_dir:
                    save_predictions(sem_pred, ins_pred, batch, results_dir, 
                                   class_inv_lut, dataset)
                    saved_count += len(sem_pred)
                
                # Update evaluator
                evaluator.update(sem_pred, ins_pred, batch)
                successful += 1
                
                # Clear cache periodically
                if i % 10 == 0:
                    torch.cuda.empty_cache()
                
                # Update progress bar
                if successful > 0 and successful % 10 == 0:
                    try:
                        pq = evaluator.get_mean_pq()
                        iou = evaluator.get_mean_iou()
                        
                        postfix = {
                            'PQ': f'{pq:.3f}',
                            'IoU': f'{iou:.3f}'
                        }
                        
                        if timing_enabled and timing_stats:
                            stats = timing_stats.get_stats()
                            postfix['FPS'] = f'{stats["fps"]:.1f}'
                        
                        if save_predictions_flag:
                            postfix['Saved'] = saved_count
                        
                        pbar.set_postfix(postfix)
                    except:
                        pass
                
            except Exception as e:
                print(f"\nError in batch {i}: {e}")
                if i < 3:  # Print detailed error for first few batches
                    import traceback
                    traceback.print_exc()
            
            pbar.update(1)
    
    print(f"\n✓ Processed {successful}/{total_batches} batches successfully")
    
    # Print timing statistics
    if timing_enabled and timing_stats:
        timing_stats.print_stats()
    
    # Print evaluation results
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    evaluator.print_results()
    
    if save_predictions_flag:
        print(f"\n✓ Saved {saved_count} predictions to: {results_dir}")
    
    # Return metrics
    metrics = {
        'PQ': evaluator.get_mean_pq(),
        'IoU': evaluator.get_mean_iou(),
        'RQ': evaluator.get_mean_rq()
    }
    
    if timing_enabled and timing_stats:
        metrics['timing'] = timing_stats.get_stats()
    
    return metrics


@click.command()
@click.option('--model', required=True, type=str, help='Path to converted .pt model')
@click.option('--dataset', default='KITTI', type=click.Choice(['KITTI', 'NUSCENES']))
@click.option('--batch-size', default=1, type=int, help='Batch size for evaluation')
@click.option('--max-batches', default=None, type=int, help='Max batches to evaluate (None = all)')
@click.option('--num-workers', default=4, type=int, help='Number of data loading workers')
@click.option('--save-predictions', is_flag=True, help='Save predictions as .label files')
@click.option('--output-dir', default=None, type=str, help='Output directory for predictions')
@click.option('--warmup-batches', default=3, type=int, help='Number of warmup batches for timing')
@click.option('--no-timing', is_flag=True, help='Disable detailed timing measurements')
@click.option('--data-path', default=None, type=str, help='Override data path')
def main(model, dataset, batch_size, max_batches, num_workers, save_predictions, 
         output_dir, warmup_batches, no_timing, data_path):
    """Evaluate original MaskPLS model with MinkowskiEngine backbone"""
    
    print("="*60)
    print("ORIGINAL MASKPLS MODEL EVALUATOR")
    print("MinkowskiEngine Architecture")
    print("="*60)
    
    # Check MinkowskiEngine
    try:
        import MinkowskiEngine as ME
        print(f"✓ MinkowskiEngine available")
    except ImportError:
        print("✗ Error: MinkowskiEngine not installed!")
        print("  Install with: pip install MinkowskiEngine")
        return
    
    # Verify model file exists
    if not os.path.exists(model):
        print(f"✗ Error: Model file not found: {model}")
        return
    
    print(f"\nConfiguration:")
    print(f"  Model: {model}")
    print(f"  Dataset: {dataset}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max batches: {max_batches if max_batches else 'all'}")
    print(f"  Save predictions: {save_predictions}")
    print(f"  Timing: {'disabled' if no_timing else 'enabled'}")
    
    # Load configuration
    cfg = get_config()
    cfg.MODEL.DATASET = dataset
    cfg.TRAIN.BATCH_SIZE = batch_size
    cfg.TRAIN.NUM_WORKERS = num_workers
    
    # Override data path if provided
    if data_path:
        cfg[dataset].PATH = data_path
        print(f"  Data path: {data_path}")
    
    # Setup data module
    print("\nSetting up dataset...")
    try:
        data_module = SemanticDatasetModule(cfg)
        data_module.setup()
        print(f"✓ Dataset loaded")
        print(f"  Things IDs: {data_module.things_ids}")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return
    
    # Run evaluation
    try:
        metrics = test_model(
            model, cfg, data_module,
            max_batches=max_batches,
            save_predictions_flag=save_predictions,
            output_dir=output_dir,
            warmup_batches=warmup_batches if not no_timing else 0,
            timing_enabled=not no_timing
        )
        
        # Print summary
        print(f"\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Panoptic Quality (PQ): {metrics['PQ']:.4f}")
        print(f"Mean IoU: {metrics['IoU']:.4f}")
        print(f"Recognition Quality (RQ): {metrics['RQ']:.4f}")
        
        if 'timing' in metrics and metrics['timing']:
            timing = metrics['timing']
            print(f"\nInference Performance:")
            print(f"  Mean inference time: {timing['inference']['mean']:.2f} ms")
            print(f"  Mean total time: {timing['total']['mean']:.2f} ms")
            print(f"  Average FPS: {timing['fps']:.2f} Hz")
        
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())