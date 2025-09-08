#!/usr/bin/env python3
"""
Exact replication of training inference - matching train_efficient_dgcnn.py exactly
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

# Import the EXACT model used in training - from maskpls_dgcnn_optimized.py
from mask_pls.models.dgcnn.maskpls_dgcnn_optimized import MaskPLSDGCNNFixed
from mask_pls.datasets.semantic_dataset import SemanticDatasetModule
from mask_pls.utils.evaluate_panoptic import PanopticEvaluator


def get_config():
    """Load configuration exactly as in train_efficient_dgcnn.py"""
    def getDir(obj):
        return os.path.dirname(os.path.abspath(obj))
    
    model_cfg = edict(yaml.safe_load(open(os.path.join(getDir(__file__), "../config/model.yaml"))))
    backbone_cfg = edict(yaml.safe_load(open(os.path.join(getDir(__file__), "../config/backbone.yaml"))))
    decoder_cfg = edict(yaml.safe_load(open(os.path.join(getDir(__file__), "../config/decoder.yaml"))))
    
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    
    # Match training configuration exactly
    cfg.TRAIN.BATCH_SIZE = 2  # Small batch size
    cfg.TRAIN.ACCUMULATE_GRAD_BATCHES = 2  # Effective batch size of 4
    cfg.TRAIN.WARMUP_STEPS = 500
    cfg.TRAIN.SUBSAMPLE = True
    cfg.TRAIN.AUG = True
    cfg.TRAIN.LR = 0.0001  # Lower learning rate
    
    return cfg


class ExactInferenceWrapper:
    """Wrapper that uses the exact training model"""
    
    def __init__(self, checkpoint_path, cfg, data_module, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.cfg = cfg
        
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # Create the EXACT model used in training
        self.model = MaskPLSDGCNNFixed(cfg)
        
        # CRITICAL: Set things_ids from data module BEFORE loading checkpoint
        # This matches exactly what train_efficient_dgcnn.py does
        self.model.things_ids = data_module.things_ids
        print(f"Set model.things_ids from data_module: {self.model.things_ids}")
        
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
            print(f"Checkpoint info:")
            if 'epoch' in ckpt:
                print(f"  Epoch: {ckpt['epoch']}")
            if 'global_step' in ckpt:
                print(f"  Global step: {ckpt['global_step']}")
        elif 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        else:
            state_dict = ckpt
        
        # Load state dict
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        
        print(f"Loaded checkpoint:")
        print(f"  Missing keys: {len(missing_keys)}")
        print(f"  Unexpected keys: {len(unexpected_keys)}")
        
        # Only print details if there are issues
        if missing_keys and len(missing_keys) < 10:
            print(f"  Missing: {missing_keys}")
        if unexpected_keys and len(unexpected_keys) < 10:
            print(f"  Unexpected: {unexpected_keys}")
        
        # Move to device and set to eval
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Verify weights are loaded
        self._verify_weights()
    
    def _verify_weights(self):
        """Verify that weights are actually loaded"""
        print("\nVerifying loaded weights...")
        
        # Check backbone weights (FixedDGCNNBackbone from dgcnn_backbone_efficient.py)
        if hasattr(self.model, 'backbone'):
            backbone = self.model.backbone
            
            # Check critical layers based on the actual backbone structure
            checks = []
            
            # The FixedDGCNNBackbone uses edge_conv layers as Sequential modules
            if hasattr(backbone, 'edge_conv1'):
                if isinstance(backbone.edge_conv1, nn.Sequential):
                    weight = backbone.edge_conv1[0].weight
                    checks.append(('edge_conv1', weight.abs().max().item(), weight.abs().mean().item()))
            
            if hasattr(backbone, 'edge_conv2'):
                if isinstance(backbone.edge_conv2, nn.Sequential):
                    weight = backbone.edge_conv2[0].weight
                    checks.append(('edge_conv2', weight.abs().max().item(), weight.abs().mean().item()))
            
            if hasattr(backbone, 'conv5'):
                if isinstance(backbone.conv5, nn.Sequential):
                    weight = backbone.conv5[0].weight
                    checks.append(('conv5', weight.abs().max().item(), weight.abs().mean().item()))
            
            for name, max_val, mean_val in checks:
                if max_val < 1e-6:
                    print(f"  ❌ {name}: appears uninitialized (max={max_val:.6f})")
                else:
                    print(f"  ✓ {name}: max={max_val:.4f}, mean={mean_val:.4f}")
        
        # Check decoder weights
        if hasattr(self.model, 'decoder'):
            decoder = self.model.decoder
            if hasattr(decoder, 'query_feat'):
                weight = decoder.query_feat.weight
                max_val = weight.abs().max().item()
                mean_val = weight.abs().mean().item()
                if max_val < 1e-6:
                    print(f"  ❌ decoder.query_feat: appears uninitialized")
                else:
                    print(f"  ✓ decoder.query_feat: max={max_val:.4f}, mean={mean_val:.4f}")
            
            if hasattr(decoder, 'class_embed'):
                weight = decoder.class_embed.weight
                max_val = weight.abs().max().item()
                print(f"  ✓ decoder.class_embed: max={max_val:.4f}")
    
    @torch.no_grad()
    def process_batch(self, batch):
        """Process batch using exact training pipeline"""
        
        # Clear GPU cache like in training
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Use the model's forward method directly
        outputs, padding, sem_logits = self.model(batch)
        
        # Use the model's panoptic_inference method
        sem_pred, ins_pred = self.model.panoptic_inference(outputs, padding)
        
        return sem_pred, ins_pred, outputs, padding
    
    def test_single_batch(self, batch):
        """Test a single batch and print detailed diagnostics"""
        print("\nTesting single batch...")
        
        # Get predictions
        with torch.no_grad():
            outputs, padding, sem_logits = self.model(batch)
        
        print(f"Output shapes:")
        print(f"  pred_logits: {outputs['pred_logits'].shape}")
        print(f"  pred_masks: {outputs['pred_masks'].shape}")
        print(f"  sem_logits: {sem_logits.shape}")
        
        # Check values
        print(f"\nOutput statistics:")
        print(f"  pred_logits: min={outputs['pred_logits'].min():.3f}, max={outputs['pred_logits'].max():.3f}, mean={outputs['pred_logits'].mean():.3f}")
        print(f"  pred_masks: min={outputs['pred_masks'].min():.3f}, max={outputs['pred_masks'].max():.3f}, mean={outputs['pred_masks'].mean():.3f}")
        print(f"  sem_logits: min={sem_logits.min():.3f}, max={sem_logits.max():.3f}, mean={sem_logits.mean():.3f}")
        
        # Check class predictions from decoder
        pred_classes = outputs['pred_logits'].argmax(dim=-1)
        unique_pred_classes, counts = torch.unique(pred_classes, return_counts=True)
        print(f"\nDecoder class predictions (queries):")
        for cls, cnt in zip(unique_pred_classes, counts):
            print(f"  Class {cls}: {cnt} queries")
        
        # Check semantic predictions
        for b in range(sem_logits.shape[0]):
            sem_pred = torch.argmax(sem_logits[b], dim=-1)
            valid_mask = ~padding[b]
            if valid_mask.sum() > 0:
                unique_classes, counts = torch.unique(sem_pred[valid_mask], return_counts=True)
            else:
                unique_classes, counts = torch.unique(sem_pred, return_counts=True)
            
            print(f"\nSemantic predictions for sample {b} (per point):")
            for cls, cnt in zip(unique_classes, counts):
                print(f"  Class {cls}: {cnt} points")
        
        # Get panoptic predictions
        sem_pred, ins_pred = self.model.panoptic_inference(outputs, padding)
        
        print(f"\nPanoptic predictions:")
        for i, (sp, ip) in enumerate(zip(sem_pred, ins_pred)):
            if len(sp) > 0:
                unique_sem = np.unique(sp)
                unique_ins = np.unique(ip[ip > 0])  # Only count non-zero instances
                print(f"  Sample {i}: {len(unique_sem)} semantic classes, {len(unique_ins)} instances")
                print(f"    Semantic classes: {unique_sem[:10]}...")  # Show first 10
                if len(unique_ins) > 0:
                    print(f"    Instance IDs: {unique_ins[:10]}...")  # Show first 10


def evaluate_with_exact_model(checkpoint_path, cfg, data_module, max_batches=None):
    """Evaluate using exact training model setup"""
    
    # Get dataloader
    dataloader = data_module.val_dataloader()
    
    # Create wrapper with data module
    wrapper = ExactInferenceWrapper(checkpoint_path, cfg, data_module)
    
    # Test on one batch first
    print("\n" + "="*60)
    print("Testing on single batch...")
    print("="*60)
    
    for batch in dataloader:
        wrapper.test_single_batch(batch)
        break
    
    # Setup evaluator - same as used in training
    dataset = cfg.MODEL.DATASET
    evaluator = PanopticEvaluator(cfg[dataset], dataset)
    evaluator.reset()
    
    # Evaluation loop
    print("\n" + "="*60)
    print("Starting full evaluation...")
    print("="*60)
    
    total_batches = len(dataloader)
    if max_batches:
        total_batches = min(total_batches, max_batches)
    
    successful = 0
    failed = 0
    
    with tqdm(total=total_batches, desc="Evaluating") as pbar:
        for i, batch in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break
            
            try:
                # Process batch
                sem_pred, ins_pred, outputs, padding = wrapper.process_batch(batch)
                
                # Use the model's prepare_batch_for_eval method if available
                if hasattr(wrapper.model, 'prepare_batch_for_eval'):
                    eval_batch = wrapper.model.prepare_batch_for_eval(batch, sem_pred)
                else:
                    # Fallback to simple approach
                    eval_batch = {
                        'fname': batch['fname'],
                        'sem_label': batch['sem_label'],
                        'ins_label': batch['ins_label']
                    }
                
                # Update evaluator
                evaluator.update(sem_pred, ins_pred, eval_batch)
                successful += 1
                
                # Clear cache periodically like in training
                if i % 10 == 0:
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"\nError in batch {i}: {e}")
                failed += 1
                if failed < 3:
                    import traceback
                    traceback.print_exc()
            
            pbar.update(1)
            
            # Show metrics periodically
            if successful > 0 and successful % 10 == 0:
                try:
                    pq = evaluator.get_mean_pq()
                    iou = evaluator.get_mean_iou()
                    rq = evaluator.get_mean_rq()
                    pbar.set_postfix({
                        'PQ': f'{pq:.3f}',
                        'IoU': f'{iou:.3f}',
                        'RQ': f'{rq:.3f}'
                    })
                except:
                    pass
    
    print(f"\nProcessed {successful}/{total_batches} batches successfully")
    
    if successful == 0:
        print("No batches were processed successfully!")
        return None
    
    # Final metrics
    print("\n" + "="*60)
    print("Final Evaluation Metrics")
    print("="*60)
    
    pq = evaluator.get_mean_pq()
    iou = evaluator.get_mean_iou()
    rq = evaluator.get_mean_rq()
    
    print(f"PQ (Panoptic Quality): {pq:.4f}")
    print(f"IoU (Mean IoU): {iou:.4f}")
    print(f"RQ (Recognition Quality): {rq:.4f}")
    
    evaluator.print_results()
    
    return {'PQ': pq, 'IoU': iou, 'RQ': rq}


@click.command()
@click.option('--checkpoint', required=True, help='Path to checkpoint (.ckpt)')
@click.option('--dataset', default='KITTI', type=click.Choice(['KITTI', 'NUSCENES']))
@click.option('--batch-size', default=1, type=int)
@click.option('--max-batches', type=int, help='Max batches to evaluate')
@click.option('--num-workers', default=4, type=int)
@click.option('--pretrained', type=str, default=None, help='Pre-trained DGCNN weights (unused in eval)')
def main(checkpoint, dataset, batch_size, max_batches, num_workers, pretrained):
    """Evaluate using exact training model setup"""
    
    print("="*60)
    print("Model Evaluation - Exact Training Replication")
    print("="*60)
    print(f"Using model: MaskPLSDGCNNFixed from maskpls_dgcnn_optimized.py")
    print(f"Using backbone: FixedDGCNNBackbone from dgcnn_backbone_efficient.py")
    
    # Load configuration EXACTLY as in training
    cfg = get_config()
    
    # Update with command line args (but keep training defaults where important)
    cfg.MODEL.DATASET = dataset
    cfg.TRAIN.BATCH_SIZE = batch_size
    cfg.TRAIN.NUM_WORKERS = num_workers
    
    # Keep other training settings
    print(f"\nConfiguration:")
    print(f"  Dataset: {cfg.MODEL.DATASET}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Workers: {num_workers}")
    print(f"  Subsample: {cfg.TRAIN.SUBSAMPLE}")
    print(f"  Aug: {cfg.TRAIN.AUG} (disabled for eval)")
    
    # Create data module EXACTLY as in training
    print("\nSetting up dataset...")
    data_module = SemanticDatasetModule(cfg)
    data_module.setup()
    
    print(f"Data module things_ids: {data_module.things_ids}")
    print(f"Data module has {len(data_module.val_mask_set)} validation samples")
    
    # Note: Augmentation should not affect validation
    # The val_mask_set already has aug=False in its setup
    
    # Evaluate
    metrics = evaluate_with_exact_model(checkpoint, cfg, data_module, max_batches)
    
    if metrics:
        print(f"\n" + "="*60)
        print(f"Final Results:")
        print(f"  PQ = {metrics['PQ']:.4f}")
        print(f"  IoU = {metrics['IoU']:.4f}")
        print(f"  RQ = {metrics['RQ']:.4f}")
        print("="*60)


if __name__ == "__main__":
    main()