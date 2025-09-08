#!/usr/bin/env python3
"""
Enhanced standalone .pt model evaluator with timing and label saving
"""

import os
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

from mask_pls.datasets.semantic_dataset import SemanticDatasetModule
from mask_pls.utils.evaluate_panoptic import PanopticEvaluator
from mask_pls.models.dgcnn.dgcnn_backbone_efficient import FixedDGCNNBackbone
from mask_pls.models.decoder import MaskedTransformerDecoder


def get_config():
    """Load configuration exactly as in training"""
    def getDir(obj):
        return os.path.dirname(os.path.abspath(obj))
    
    model_cfg = edict(yaml.safe_load(open(os.path.join(getDir(__file__), "../config/model.yaml"))))
    backbone_cfg = edict(yaml.safe_load(open(os.path.join(getDir(__file__), "../config/backbone.yaml"))))
    decoder_cfg = edict(yaml.safe_load(open(os.path.join(getDir(__file__), "../config/decoder.yaml"))))
    
    # Try to load dataset-specific config (e.g., semantic-kitti-custom.yaml)
    dataset_cfg = {}
    possible_dataset_configs = [
        "../config/dataset/semantic-kitti-custom.yaml",
        "../config/dataset/semantic-kitti.yaml",
        "../config/semantic-kitti-custom.yaml",
        "../config/semantic-kitti.yaml"
    ]
    
    for config_path in possible_dataset_configs:
        full_path = os.path.join(getDir(__file__), config_path)
        if os.path.exists(full_path):
            print(f"Loading dataset config from: {config_path}")
            dataset_cfg = edict(yaml.safe_load(open(full_path)))
            break
    
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg, **dataset_cfg})
    
    cfg.TRAIN.BATCH_SIZE = 2
    cfg.TRAIN.ACCUMULATE_GRAD_BATCHES = 2
    cfg.TRAIN.WARMUP_STEPS = 500
    cfg.TRAIN.SUBSAMPLE = True
    cfg.TRAIN.AUG = False  # No augmentation for evaluation
    cfg.TRAIN.LR = 0.0001
    
    return cfg


def detect_sequences_from_dataloader(dataloader, max_batches=5):
    """Detect which sequences are in the dataset by examining filenames"""
    sequences = set()
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        for fname in batch['fname']:
            if '_' in fname:
                seq = fname.split('_')[0]
                sequences.add(seq)
    return sorted(list(sequences))


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
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
    else:  # KITTI
        results_dir = os.path.join(base_dir, "test", "sequences")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
        
        # If sequences provided, create dirs for those; otherwise use default test sequences
        if sequences:
            for seq in sequences:
                # Ensure sequence is 2 digits
                if isinstance(seq, int):
                    seq_str = str(seq).zfill(2)
                else:
                    seq_str = str(seq).zfill(2) if seq.isdigit() else seq
                sub_dir = os.path.join(results_dir, seq_str, "predictions")
                if not os.path.exists(sub_dir):
                    os.makedirs(sub_dir, exist_ok=True)
                print(f"  Created directory for sequence {seq_str}")
        else:
            # Default to standard test sequences
            print("  No sequences detected, using default test sequences 11-21")
            for i in range(11, 22):
                sub_dir = os.path.join(results_dir, str(i).zfill(2), "predictions")
                if not os.path.exists(sub_dir):
                    os.makedirs(sub_dir, exist_ok=True)
    
    return results_dir


def save_predictions(sem_pred, ins_pred, batch, results_dir, dataset="KITTI"):
    """Save predictions as .label files"""
    for b in range(len(sem_pred)):
        fname = batch['fname'][b]
        
        # Combine semantic and instance predictions into panoptic labels
        # Format: upper 16 bits = instance, lower 16 bits = semantic
        panoptic_labels = (ins_pred[b].astype(np.uint32) << 16) | sem_pred[b].astype(np.uint32)
        
        if dataset == "KITTI":
            # Extract sequence and frame from filename
            # Expected format: sequence_frame (e.g., "0_000000" or "00_000000")
            if '_' in fname:
                parts = fname.split('_')
                if len(parts) == 2:
                    sequence = parts[0].zfill(2)  # Ensure 2-digit sequence
                    frame = parts[1]
                    save_path = os.path.join(results_dir, sequence, "predictions", f"{frame}.label")
                else:
                    # Fallback for unexpected format
                    save_path = os.path.join(results_dir, f"{fname}.label")
            else:
                # No underscore, might be just a number or other format
                save_path = os.path.join(results_dir, f"{fname}.label")
        else:  # NUSCENES
            save_path = os.path.join(results_dir, f"{fname}.label")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save as binary file
        panoptic_labels.astype(np.uint32).tofile(save_path)


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
            'fps': 1000.0 / np.mean(np.array(self.total_times) * 1000)  # Hz
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
        
        # Print header
        row_format = "{:<15} {:<15} {:<10} {:<10} {:<10}"
        print(row_format.format(*headers))
        print("-" * 60)
        
        # Print each component
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


class StandaloneInferenceModel(nn.Module):
    """Standalone inference model without Lightning dependencies"""
    
    def __init__(self, cfg, things_ids):
        super().__init__()
        
        dataset = cfg.MODEL.DATASET
        self.num_classes = cfg[dataset].NUM_CLASSES
        self.ignore_label = cfg[dataset].IGNORE_LABEL
        self.things_ids = things_ids
        
        # Create the EXACT same components as in training
        self.backbone = FixedDGCNNBackbone(cfg.BACKBONE, None)
        self.backbone.set_num_classes(self.num_classes)
        
        self.decoder = MaskedTransformerDecoder(
            cfg.DECODER,
            cfg.BACKBONE,
            cfg[dataset]
        )
    
    @torch.no_grad()
    def forward_with_timing(self, batch_dict):
        """Forward pass with detailed timing"""
        # Backbone timing
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        backbone_start = time.perf_counter()
        
        feats, coords, pad_masks, sem_logits = self.backbone(batch_dict)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        backbone_time = time.perf_counter() - backbone_start
        
        # Decoder timing
        decoder_start = time.perf_counter()
        
        outputs, padding = self.decoder(feats, coords, pad_masks)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        decoder_time = time.perf_counter() - decoder_start
        
        inference_time = backbone_time + decoder_time
        
        return outputs, padding, sem_logits, {
            'backbone_time': backbone_time,
            'decoder_time': decoder_time,
            'inference_time': inference_time
        }
    
    @torch.no_grad()
    def forward(self, batch_dict):
        """Standard forward pass"""
        feats, coords, pad_masks, sem_logits = self.backbone(batch_dict)
        outputs, padding = self.decoder(feats, coords, pad_masks)
        return outputs, padding, sem_logits
    
    def panoptic_inference(self, outputs, padding):
        """Panoptic segmentation inference"""
        mask_cls = outputs["pred_logits"]
        mask_pred = outputs["pred_masks"]
        
        batch_size = mask_cls.shape[0]
        sem_pred = []
        ins_pred = []
        
        for b in range(batch_size):
            valid_mask = ~padding[b]
            num_valid = valid_mask.sum().item()
            
            if num_valid == 0:
                sem_pred.append(np.zeros(0, dtype=np.int32))
                ins_pred.append(np.zeros(0, dtype=np.int32))
                continue
            
            scores, labels = mask_cls[b].max(-1)
            mask_pred_b = mask_pred[b][valid_mask].sigmoid()
            
            # Filter valid predictions
            keep = labels.ne(self.num_classes)
            
            if keep.sum() == 0:
                sem_pred.append(np.zeros(num_valid, dtype=np.int32))
                ins_pred.append(np.zeros(num_valid, dtype=np.int32))
                continue
            
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred_b[:, keep]
            
            # Instance assignment
            cur_prob_masks = cur_scores.unsqueeze(0) * cur_masks
            
            semantic_seg = torch.zeros(num_valid, dtype=torch.int32, device=cur_masks.device)
            instance_seg = torch.zeros(num_valid, dtype=torch.int32, device=cur_masks.device)
            
            current_segment_id = 0
            
            if cur_masks.shape[1] > 0:
                cur_mask_ids = cur_prob_masks.argmax(1)
                stuff_memory_list = {}
                
                for k in range(cur_classes.shape[0]):
                    pred_class = cur_classes[k].item()
                    isthing = pred_class in self.things_ids
                    
                    mask = (cur_mask_ids == k) & (cur_masks[:, k] >= 0.5)
                    mask_area = mask.sum().item()
                    
                    if mask_area > 0:
                        if isthing:
                            current_segment_id += 1
                            instance_seg[mask] = current_segment_id
                            semantic_seg[mask] = pred_class
                        else:
                            if pred_class not in stuff_memory_list:
                                current_segment_id += 1
                                stuff_memory_list[pred_class] = current_segment_id
                            semantic_seg[mask] = pred_class
            
            sem_pred.append(semantic_seg.cpu().numpy())
            ins_pred.append(instance_seg.cpu().numpy())
        
        return sem_pred, ins_pred
    
    def prepare_batch_for_eval(self, batch, sem_pred):
        """Prepare batch for evaluation with subsampling handling"""
        subsample_indices = self.backbone.subsample_indices
        
        eval_batch = {'fname': batch['fname']}
        eval_sem_labels = []
        eval_ins_labels = []
        
        for b in range(len(sem_pred)):
            sem_label = batch['sem_label'][b]
            ins_label = batch['ins_label'][b]
            
            # Convert to numpy if needed
            if isinstance(sem_label, torch.Tensor):
                sem_label = sem_label.cpu().numpy()
            if isinstance(ins_label, torch.Tensor):
                ins_label = ins_label.cpu().numpy()
            
            # Apply subsampling if needed
            if b in subsample_indices:
                indices = subsample_indices[b].cpu().numpy()
                if indices.shape[0] < sem_label.shape[0]:
                    sem_label = sem_label[indices]
                    ins_label = ins_label[indices]
            
            # Match prediction size
            pred_size = len(sem_pred[b])
            if pred_size < len(sem_label):
                sem_label = sem_label[:pred_size]
                ins_label = ins_label[:pred_size]
            
            eval_sem_labels.append(sem_label)
            eval_ins_labels.append(ins_label)
        
        eval_batch['sem_label'] = eval_sem_labels
        eval_batch['ins_label'] = eval_ins_labels
        
        return eval_batch


def test_standalone_model(model_path, cfg, data_module, max_batches=None, 
                         save_predictions_flag=False, output_dir=None,
                         warmup_batches=3):
    """Test standalone model with timing and optional prediction saving"""
    
    print(f"Loading standalone model from {model_path}")
    
    # Load saved model
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Print checkpoint contents
    print("\nCheckpoint contents:")
    for key in checkpoint.keys():
        if key not in ['model_state_dict', 'backbone_state_dict', 'decoder_state_dict', 'config']:
            if isinstance(checkpoint[key], (list, tuple)):
                print(f"  {key}: {checkpoint[key]}")
            elif isinstance(checkpoint[key], dict):
                print(f"  {key}: dict with {len(checkpoint[key])} items")
            else:
                print(f"  {key}: {checkpoint[key]}")
    
    # Get things_ids from checkpoint or data module
    if 'things_ids' in checkpoint:
        things_ids = checkpoint['things_ids']
        print(f"Using things_ids from checkpoint: {things_ids}")
    else:
        things_ids = data_module.things_ids
        print(f"Using things_ids from data module: {things_ids}")
    
    # Create model
    model = StandaloneInferenceModel(cfg, things_ids)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Try loading individual components
        if 'backbone_state_dict' in checkpoint:
            model.backbone.load_state_dict(checkpoint['backbone_state_dict'])
        if 'decoder_state_dict' in checkpoint:
            model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    # Move to GPU and eval mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"\nModel loaded successfully")
    print(f"  Device: {device}")
    print(f"  Things IDs: {model.things_ids}")
    print(f"  Num classes: {model.num_classes}")
    
    # Get dataloader
    dataloader = data_module.val_dataloader()
    
    # Detect sequences from the actual data
    sequences = None
    if cfg.MODEL.DATASET == "KITTI" and save_predictions_flag:
        # First try to get from config
        if hasattr(cfg, 'KITTI') and hasattr(cfg.KITTI, 'TEST'):
            sequences = cfg.KITTI.TEST
            # Convert to strings if they're integers
            sequences = [str(s) for s in sequences]
            print(f"\nTest sequences from config: {sequences}")
        else:
            # Detect from dataloader
            print("\nDetecting sequences from dataloader...")
            sequences = detect_sequences_from_dataloader(dataloader)
            if sequences:
                print(f"  Detected sequences: {sequences}")
            else:
                print("  Warning: Could not detect sequences from data")
    
    # Create save directory if needed
    results_dir = None
    if save_predictions_flag:
        results_dir = create_save_dirs(cfg.MODEL.DATASET, output_dir, sequences)
        print(f"\nSaving predictions to: {results_dir}")
    
    # Setup evaluator
    dataset = cfg.MODEL.DATASET
    evaluator = PanopticEvaluator(cfg[dataset], dataset)
    evaluator.reset()
    
    # Timing statistics
    timing_stats = TimingStats()
    
    # Warmup runs (important for accurate timing)
    print(f"\nWarming up with {warmup_batches} batches...")
    for i, batch in enumerate(dataloader):
        if i >= warmup_batches:
            break
        with torch.no_grad():
            outputs, padding, sem_logits = model(batch)
            sem_pred, ins_pred = model.panoptic_inference(outputs, padding)
    
    torch.cuda.empty_cache()
    
    # Full evaluation with timing
    print("\nRunning full evaluation with timing...")
    
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
                    # Inference with timing
                    outputs, padding, sem_logits, timings = model.forward_with_timing(batch)
                    
                    # Post-processing timing
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    postprocess_start = time.perf_counter()
                    
                    sem_pred, ins_pred = model.panoptic_inference(outputs, padding)
                    
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    postprocess_time = time.perf_counter() - postprocess_start
                
                total_time = time.perf_counter() - total_start
                
                # Record timing for each sample in the batch
                batch_size = len(sem_pred)
                for _ in range(batch_size):
                    timing_stats.add_sample(
                        timings['backbone_time'] / batch_size,
                        timings['decoder_time'] / batch_size,
                        timings['inference_time'] / batch_size,
                        postprocess_time / batch_size,
                        total_time / batch_size
                    )
                
                # Save predictions if requested
                if save_predictions_flag and results_dir:
                    save_predictions(sem_pred, ins_pred, batch, results_dir, dataset)
                    saved_count += batch_size
                
                # Prepare batch for evaluation
                eval_batch = model.prepare_batch_for_eval(batch, sem_pred)
                
                # Update evaluator
                evaluator.update(sem_pred, ins_pred, eval_batch)
                successful += 1
                
                # Clear cache periodically
                if i % 10 == 0:
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"\nError in batch {i}: {e}")
                import traceback
                if i < 3:
                    traceback.print_exc()
            
            pbar.update(1)
            
            # Show metrics periodically
            if successful > 0 and successful % 10 == 0:
                try:
                    pq = evaluator.get_mean_pq()
                    iou = evaluator.get_mean_iou()
                    rq = evaluator.get_mean_rq()
                    stats = timing_stats.get_stats()
                    pbar.set_postfix({
                        'PQ': f'{pq:.3f}',
                        'IoU': f'{iou:.3f}',
                        'FPS': f'{stats["fps"]:.1f}',
                        'Saved': saved_count if save_predictions_flag else 0
                    })
                except:
                    pass
    
    print(f"\nProcessed {successful}/{total_batches} batches successfully")
    
    # Print timing statistics
    timing_stats.print_stats()
    
    # Final metrics
    print("\n" + "="*60)
    print("FINAL EVALUATION METRICS")
    print("="*60)
    
    pq = evaluator.get_mean_pq()
    iou = evaluator.get_mean_iou()
    rq = evaluator.get_mean_rq()
    
    print(f"PQ (Panoptic Quality): {pq:.4f}")
    print(f"IoU (Mean IoU): {iou:.4f}")
    print(f"RQ (Recognition Quality): {rq:.4f}")
    
    evaluator.print_results()
    
    if save_predictions_flag:
        print(f"\nSaved {saved_count} predictions to: {results_dir}")
        if sequences:
            print(f"Sequences processed: {sequences}")
    
    return {'PQ': pq, 'IoU': iou, 'RQ': rq, 'timing': timing_stats.get_stats()}


@click.command()
@click.option('--model', required=True, type=str, help='Path to standalone .pt model')
@click.option('--dataset', default='KITTI', type=click.Choice(['KITTI', 'NUSCENES']))
@click.option('--batch-size', default=1, type=int, help='Batch size for evaluation')
@click.option('--max-batches', default=None, type=int, help='Max batches to evaluate')
@click.option('--num-workers', default=4, type=int, help='Number of data loading workers')
@click.option('--save-predictions', is_flag=True, help='Save predictions as .label files')
@click.option('--output-dir', default=None, type=str, help='Output directory for predictions')
@click.option('--warmup-batches', default=3, type=int, help='Number of warmup batches for timing')
@click.option('--sequences', default=None, type=str, help='Comma-separated list of sequences (e.g., "0,1,2" or "00,01,02")')
@click.option('--dataset-config', default=None, type=str, help='Path to dataset config YAML file')
def main(model, dataset, batch_size, max_batches, num_workers, save_predictions, output_dir, warmup_batches, sequences, dataset_config):
    """Test standalone .pt model with timing and optional prediction saving"""
    
    print("="*60)
    print("STANDALONE MODEL TESTING")
    print("="*60)
    print(f"Model path: {model}")
    print(f"Dataset: {dataset}")
    print(f"Batch size: {batch_size}")
    print(f"Save predictions: {save_predictions}")
    
    # Parse sequences if provided
    if sequences:
        sequences = [s.strip() for s in sequences.split(',')]
        print(f"Manual sequences: {sequences}")
    
    # Load configuration
    cfg = get_config()
    
    # Load custom dataset config if provided
    if dataset_config and os.path.exists(dataset_config):
        print(f"Loading custom dataset config from: {dataset_config}")
        custom_cfg = edict(yaml.safe_load(open(dataset_config)))
        cfg.update(custom_cfg)
    
    cfg.MODEL.DATASET = dataset
    cfg.TRAIN.BATCH_SIZE = batch_size
    cfg.TRAIN.NUM_WORKERS = num_workers
    
    # Override sequences in config if manually specified
    if sequences and dataset == "KITTI":
        if 'KITTI' not in cfg:
            cfg.KITTI = edict()
        cfg.KITTI.TEST = sequences
        print(f"Overriding test sequences to: {sequences}")
    
    # Print detected test sequences
    if dataset == "KITTI" and hasattr(cfg, 'KITTI') and hasattr(cfg.KITTI, 'TEST'):
        print(f"Test sequences from config: {cfg.KITTI.TEST}")
    
    # Create data module
    print("\nSetting up dataset...")
    data_module = SemanticDatasetModule(cfg)
    data_module.setup()
    
    print(f"Data module things_ids: {data_module.things_ids}")
    
    # Test model
    metrics = test_standalone_model(
        model, cfg, data_module, 
        max_batches=max_batches,
        save_predictions_flag=save_predictions,
        output_dir=output_dir,
        warmup_batches=warmup_batches
    )
    
    if metrics:
        print(f"\n" + "="*60)
        print(f"SUMMARY")
        print("="*60)
        print(f"Accuracy Metrics:")
        print(f"  PQ  = {metrics['PQ']:.4f}")
        print(f"  IoU = {metrics['IoU']:.4f}")
        print(f"  RQ  = {metrics['RQ']:.4f}")
        
        if 'timing' in metrics and metrics['timing']:
            timing = metrics['timing']
            print(f"\nTiming Performance:")
            print(f"  Inference: {timing['inference']['mean']:.2f} ms/cloud")
            print(f"  Total:     {timing['total']['mean']:.2f} ms/cloud")
            print(f"  FPS:       {timing['fps']:.2f} Hz")
        print("="*60)


if __name__ == "__main__":
    main()