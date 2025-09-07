#!/usr/bin/env python3
"""
Exact replication of training inference - no modifications
"""

import torch
import torch.nn as nn
import numpy as np
import yaml
from pathlib import Path
from easydict import EasyDict as edict
import click
from tqdm import tqdm

# Import the EXACT model used in training
from mask_pls.models.dgcnn.maskpls_dgcnn_optimized import MaskPLSDGCNNFixed
from mask_pls.datasets.semantic_dataset import SemanticDatasetModule
from mask_pls.utils.evaluate_panoptic import PanopticEvaluator


class ExactInferenceWrapper:
    """Wrapper that uses the exact training model"""
    
    def __init__(self, checkpoint_path, cfg, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # Create the EXACT model used in training
        self.model = MaskPLSDGCNNFixed(cfg)
        
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        
        # Load state dict
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        
        print(f"Loaded checkpoint:")
        print(f"  Missing keys: {len(missing_keys)}")
        print(f"  Unexpected keys: {len(unexpected_keys)}")
        
        if missing_keys:
            print(f"  First 5 missing: {missing_keys[:5]}")
        
        # Move to device and set to eval
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Set things_ids if not set
        if not hasattr(self.model, 'things_ids') or not self.model.things_ids:
            dataset = cfg.MODEL.DATASET
            if dataset == 'KITTI':
                self.model.things_ids = [1, 2, 3, 4, 5, 6, 7, 8]
            else:
                self.model.things_ids = [2, 3, 4, 5, 6, 7, 9, 10]
        
        # Verify weights are loaded
        self._verify_weights()
    
    def _verify_weights(self):
        """Verify that weights are actually loaded"""
        print("\nVerifying loaded weights...")
        
        # Check backbone weights
        if hasattr(self.model, 'backbone'):
            backbone = self.model.backbone
            
            # Check critical layers
            checks = []
            if hasattr(backbone, 'edge_conv1'):
                weight = backbone.edge_conv1[0].weight
                checks.append(('edge_conv1', weight.abs().max().item(), weight.abs().mean().item()))
            
            if hasattr(backbone, 'edge_conv2'):
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
    
    @torch.no_grad()
    def process_batch(self, batch):
        """Process batch using exact training pipeline"""
        
        # Use the model's forward method directly
        outputs, padding, sem_logits = self.model(batch)
        
        # Use the model's panoptic_inference method
        sem_pred, ins_pred = self.model.panoptic_inference(outputs, padding)
        
        return sem_pred, ins_pred
    
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
        print(f"  pred_logits: min={outputs['pred_logits'].min():.3f}, max={outputs['pred_logits'].max():.3f}")
        print(f"  pred_masks: min={outputs['pred_masks'].min():.3f}, max={outputs['pred_masks'].max():.3f}")
        print(f"  sem_logits: min={sem_logits.min():.3f}, max={sem_logits.max():.3f}")
        
        # Check semantic predictions
        sem_pred = torch.argmax(sem_logits[0], dim=-1)
        unique_classes, counts = torch.unique(sem_pred[~padding[0]], return_counts=True)
        
        print(f"\nSemantic predictions:")
        for cls, cnt in zip(unique_classes, counts):
            print(f"  Class {cls}: {cnt} points")
        
        # Check if predictions are reasonable
        if len(unique_classes) == 1:
            print("  ⚠️ WARNING: Only predicting a single class!")
        
        if outputs['pred_logits'].abs().max() < 0.1:
            print("  ⚠️ WARNING: pred_logits values are very small!")
        
        # Get panoptic predictions
        sem_pred, ins_pred = self.model.panoptic_inference(outputs, padding)
        
        print(f"\nPanoptic predictions:")
        for i, (sp, ip) in enumerate(zip(sem_pred, ins_pred)):
            unique_sem = np.unique(sp)
            unique_ins = np.unique(ip)
            print(f"  Sample {i}: {len(unique_sem)} semantic classes, {len(unique_ins)} instances")


def evaluate_with_exact_model(checkpoint_path, cfg, dataloader, max_batches=None):
    """Evaluate using exact training model"""
    
    # Create wrapper
    wrapper = ExactInferenceWrapper(checkpoint_path, cfg)
    
    # Test on one batch first
    print("\n" + "="*60)
    print("Testing on single batch...")
    print("="*60)
    
    for batch in dataloader:
        wrapper.test_single_batch(batch)
        break
    
    # Setup evaluator
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
                sem_pred, ins_pred = wrapper.process_batch(batch)
                
                # Fix size mismatches
                for j in range(len(batch['sem_label'])):
                    gt_sem = batch['sem_label'][j]
                    gt_ins = batch['ins_label'][j]
                    
                    if isinstance(gt_sem, np.ndarray):
                        gt_sem = gt_sem.reshape(-1)
                    if isinstance(gt_ins, np.ndarray):
                        gt_ins = gt_ins.reshape(-1)
                    
                    if j < len(sem_pred):
                        pred_size = len(sem_pred[j])
                        gt_size = len(gt_sem)
                        
                        if pred_size != gt_size:
                            min_size = min(pred_size, gt_size)
                            sem_pred[j] = sem_pred[j][:min_size]
                            ins_pred[j] = ins_pred[j][:min_size]
                            batch['sem_label'][j] = gt_sem[:min_size]
                            batch['ins_label'][j] = gt_ins[:min_size]
                
                # Update evaluator
                evaluator.update(sem_pred, ins_pred, batch)
                successful += 1
                
            except Exception as e:
                print(f"\nError in batch {i}: {e}")
                failed += 1
                if failed < 3:
                    import traceback
                    traceback.print_exc()
            
            pbar.update(1)
            
            # Show metrics
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
@click.option('--checkpoint', required=True, help='Path to checkpoint')
@click.option('--dataset', default='KITTI', type=click.Choice(['KITTI', 'NUSCENES']))
@click.option('--batch-size', default=1, type=int)
@click.option('--max-batches', type=int, help='Max batches to evaluate')
@click.option('--num-workers', default=4, type=int)
def main(checkpoint, dataset, batch_size, max_batches, num_workers):
    """Evaluate using exact training model"""
    
    print("="*60)
    print("Exact Model Evaluation (No Export)")
    print("="*60)
    
    # Load configuration
    config_dir = Path(__file__).parent.parent / "config"
    cfg = edict()
    
    for config_file in ['model.yaml', 'backbone.yaml', 'decoder.yaml']:
        config_path = config_dir / config_file
        if config_path.exists():
            with open(config_path, 'r') as f:
                cfg.update(yaml.safe_load(f))
            print(f"✓ Loaded: {config_path}")
    
    # Update config
    cfg.MODEL.DATASET = dataset
    cfg.TRAIN.BATCH_SIZE = batch_size
    cfg.TRAIN.NUM_WORKERS = num_workers
    
    # Create data module
    print("\nSetting up dataset...")
    data_module = SemanticDatasetModule(cfg)
    data_module.setup()
    dataloader = data_module.val_dataloader()
    
    # Evaluate
    metrics = evaluate_with_exact_model(checkpoint, cfg, dataloader, max_batches)
    
    if metrics:
        print(f"\n" + "="*60)
        print(f"Final Results:")
        print(f"  PQ = {metrics['PQ']:.4f}")
        print(f"  IoU = {metrics['IoU']:.4f}")
        print(f"  RQ = {metrics['RQ']:.4f}")
        print("="*60)


if __name__ == "__main__":
    main()