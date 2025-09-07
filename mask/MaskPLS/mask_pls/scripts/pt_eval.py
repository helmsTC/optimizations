#!/usr/bin/env python3
"""
Fixed evaluation script with proper size handling
"""

import torch
import numpy as np
import yaml
import click
from pathlib import Path
from tqdm import tqdm
from easydict import EasyDict as edict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mask_pls.datasets.semantic_dataset import SemanticDatasetModule
from mask_pls.utils.evaluate_panoptic import PanopticEvaluator


class DirectModelEvaluator:
    """Evaluate direct model with size handling"""
    
    def __init__(self, model_path, cfg, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading model from {model_path}")
        
        # Load saved model
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get config
        if 'config' in checkpoint:
            cfg = checkpoint['config']
        
        dataset = cfg.MODEL.DATASET
        
        # Create components
        from mask_pls.models.dgcnn.dgcnn_backbone_efficient import FixedDGCNNBackbone
        from mask_pls.models.decoder import MaskedTransformerDecoder
        
        self.backbone = FixedDGCNNBackbone(cfg.BACKBONE, None)
        self.backbone.set_num_classes(cfg[dataset].NUM_CLASSES)
        
        self.decoder = MaskedTransformerDecoder(
            cfg.DECODER,
            cfg.BACKBONE,
            cfg[dataset]
        )
        
        # Load state dicts
        if 'backbone_state_dict' in checkpoint:
            self.backbone.load_state_dict(checkpoint['backbone_state_dict'])
            print(f"  Loaded backbone weights")
        if 'decoder_state_dict' in checkpoint:
            self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
            print(f"  Loaded decoder weights")
        
        self.backbone = self.backbone.to(self.device)
        self.decoder = self.decoder.to(self.device)
        self.backbone.eval()
        self.decoder.eval()
        
        # Setup evaluator
        self.evaluator = PanopticEvaluator(cfg[dataset], dataset)
        self.num_classes = cfg[dataset].NUM_CLASSES
        self.things_ids = [1, 2, 3, 4, 5, 6, 7, 8] if dataset == 'KITTI' else [2, 3, 4, 5, 6, 7, 9, 10]
        
        print(f"Model loaded on {self.device}")
        print(f"Dataset: {dataset}, Classes: {self.num_classes}")
    
    def process_batch(self, batch):
        """Process batch with size tracking"""
        
        # Get original sizes
        original_sizes = []
        for i in range(len(batch['pt_coord'])):
            original_sizes.append(len(batch['pt_coord'][i]))
        
        with torch.no_grad():
            # Run through backbone
            feats, coords, pad_masks, sem_logits = self.backbone(batch)
            
            # Run through decoder
            outputs, padding = self.decoder(feats, coords, pad_masks)
        
        # Process outputs
        sem_pred, ins_pred = self.panoptic_inference(outputs, padding, original_sizes)
        
        # Ensure predictions match ground truth size
        sem_pred_fixed = []
        ins_pred_fixed = []
        
        for i in range(len(batch['sem_label'])):
            # Get ground truth size
            gt_sem = batch['sem_label'][i]
            if isinstance(gt_sem, torch.Tensor):
                gt_sem = gt_sem.cpu().numpy()
            gt_size = len(gt_sem.reshape(-1))
            
            # Get prediction
            pred_sem = sem_pred[i] if i < len(sem_pred) else np.zeros(0, dtype=np.int32)
            pred_ins = ins_pred[i] if i < len(ins_pred) else np.zeros(0, dtype=np.int32)
            
            # Adjust sizes
            if len(pred_sem) > gt_size:
                # Truncate prediction
                pred_sem = pred_sem[:gt_size]
                pred_ins = pred_ins[:gt_size]
            elif len(pred_sem) < gt_size:
                # Pad prediction with zeros
                pad_size = gt_size - len(pred_sem)
                pred_sem = np.concatenate([pred_sem, np.zeros(pad_size, dtype=np.int32)])
                pred_ins = np.concatenate([pred_ins, np.zeros(pad_size, dtype=np.int32)])
            
            sem_pred_fixed.append(pred_sem)
            ins_pred_fixed.append(pred_ins)
        
        return sem_pred_fixed, ins_pred_fixed
    
    def panoptic_inference(self, outputs, padding, original_sizes):
        """Panoptic inference with size handling"""
        mask_cls = outputs["pred_logits"]
        mask_pred = outputs["pred_masks"]
        
        batch_size = mask_cls.shape[0]
        sem_pred = []
        ins_pred = []
        
        for b in range(batch_size):
            # Get valid mask
            if b < len(padding):
                valid_mask = ~padding[b]
            else:
                valid_mask = torch.ones(mask_pred.shape[-1], dtype=torch.bool, device=mask_pred.device)
            
            num_valid = valid_mask.sum().item()
            
            # Limit to original size
            if b < len(original_sizes):
                target_size = original_sizes[b]
                # Handle subsampling by backbone
                if hasattr(self.backbone, 'subsample_indices') and b in self.backbone.subsample_indices:
                    subsample_idx = self.backbone.subsample_indices[b]
                    target_size = min(target_size, len(subsample_idx))
                num_valid = min(num_valid, target_size)
            
            if num_valid == 0:
                sem_pred.append(np.zeros(0, dtype=np.int32))
                ins_pred.append(np.zeros(0, dtype=np.int32))
                continue
            
            # Get predictions
            scores, labels = mask_cls[b].max(-1)
            
            # Handle mask dimensions
            if mask_pred.shape[1] == mask_cls.shape[1]:  # [B, Q, N]
                mask_pred_b = mask_pred[b].transpose(0, 1)  # [N, Q]
            else:  # [B, N, Q]
                mask_pred_b = mask_pred[b]
            
            # Extract only valid points (up to num_valid)
            if mask_pred_b.shape[0] > num_valid:
                mask_pred_b = mask_pred_b[:num_valid]
                valid_mask = torch.ones(num_valid, dtype=torch.bool, device=mask_pred.device)
            else:
                valid_mask = valid_mask[:mask_pred_b.shape[0]]
                num_valid = mask_pred_b.shape[0]
            
            mask_pred_b = mask_pred_b[valid_mask].sigmoid()
            actual_valid = mask_pred_b.shape[0]
            
            # Filter valid predictions
            keep = labels.ne(self.num_classes)
            
            if keep.sum() == 0 or actual_valid == 0:
                sem_pred.append(np.zeros(actual_valid, dtype=np.int32))
                ins_pred.append(np.zeros(actual_valid, dtype=np.int32))
                continue
            
            # Process predictions
            keep_indices = torch.where(keep)[0]
            cur_scores = scores[keep_indices]
            cur_classes = labels[keep_indices]
            
            # Ensure mask dimensions match
            if keep_indices.shape[0] > mask_pred_b.shape[1]:
                keep_indices = keep_indices[:mask_pred_b.shape[1]]
                cur_scores = cur_scores[:mask_pred_b.shape[1]]
                cur_classes = cur_classes[:mask_pred_b.shape[1]]
            
            cur_masks = mask_pred_b[:, keep_indices]
            
            # Weighted masks
            cur_prob_masks = cur_scores.unsqueeze(0) * cur_masks
            
            # Initialize outputs
            semantic_seg = torch.zeros(actual_valid, dtype=torch.int32, device=cur_masks.device)
            instance_seg = torch.zeros(actual_valid, dtype=torch.int32, device=cur_masks.device)
            
            if cur_masks.shape[1] > 0:
                cur_mask_ids = cur_prob_masks.argmax(1)
                
                current_segment_id = 0
                
                for k in range(cur_classes.shape[0]):
                    pred_class = cur_classes[k].item()
                    isthing = pred_class in self.things_ids
                    
                    mask = (cur_mask_ids == k) & (cur_masks[:, k] >= 0.5)
                    mask_area = mask.sum().item()
                    
                    if mask_area > 0:
                        semantic_seg[mask] = pred_class
                        
                        if isthing:
                            current_segment_id += 1
                            instance_seg[mask] = current_segment_id
            
            sem_pred.append(semantic_seg.cpu().numpy())
            ins_pred.append(instance_seg.cpu().numpy())
        
        return sem_pred, ins_pred
    
    def evaluate(self, dataloader, max_batches=None):
        """Evaluate on dataloader"""
        
        print("Starting evaluation...")
        self.evaluator.reset()
        
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
                    sem_pred, ins_pred = self.process_batch(batch)
                    
                    # Prepare batch for evaluation
                    eval_batch = {
                        'sem_label': [],
                        'ins_label': [],
                        'fname': batch['fname']
                    }
                    
                    for j in range(len(batch['sem_label'])):
                        # Get ground truth
                        sem_gt = batch['sem_label'][j]
                        ins_gt = batch['ins_label'][j]
                        
                        if isinstance(sem_gt, torch.Tensor):
                            sem_gt = sem_gt.cpu().numpy()
                        if isinstance(ins_gt, torch.Tensor):
                            ins_gt = ins_gt.cpu().numpy()
                        
                        # Flatten
                        sem_gt = sem_gt.reshape(-1)
                        ins_gt = ins_gt.reshape(-1)
                        
                        # Ensure same size as predictions
                        if j < len(sem_pred):
                            pred_size = len(sem_pred[j])
                            if len(sem_gt) != pred_size:
                                min_size = min(len(sem_gt), pred_size)
                                sem_gt = sem_gt[:min_size]
                                ins_gt = ins_gt[:min_size]
                                sem_pred[j] = sem_pred[j][:min_size]
                                ins_pred[j] = ins_pred[j][:min_size]
                        
                        eval_batch['sem_label'].append(sem_gt)
                        eval_batch['ins_label'].append(ins_gt)
                    
                    # Update evaluator
                    self.evaluator.update(sem_pred, ins_pred, eval_batch)
                    successful += 1
                    
                except Exception as e:
                    print(f"\nError in batch {i}: {e}")
                    failed += 1
                    if failed < 3:  # Only print traceback for first few errors
                        import traceback
                        traceback.print_exc()
                
                pbar.update(1)
                
                # Show intermediate metrics
                if successful > 0 and successful % 10 == 0:
                    try:
                        pq = self.evaluator.get_mean_pq()
                        iou = self.evaluator.get_mean_iou()
                        rq = self.evaluator.get_mean_rq()
                        pbar.set_postfix({
                            'PQ': f'{pq:.3f}',
                            'IoU': f'{iou:.3f}',
                            'RQ': f'{rq:.3f}',
                            'OK': successful,
                            'Fail': failed
                        })
                    except:
                        # Metrics not ready yet
                        pbar.set_postfix({'OK': successful, 'Fail': failed})
        
        print(f"\nProcessed {successful} batches successfully, {failed} failed")
        
        if successful == 0:
            print("No batches were successfully processed!")
            return None
        
        # Final metrics
        print("\n" + "="*60)
        print("Final Metrics")
        print("="*60)
        
        try:
            pq = self.evaluator.get_mean_pq()
            iou = self.evaluator.get_mean_iou()
            rq = self.evaluator.get_mean_rq()
            
            print(f"PQ (Panoptic Quality): {pq:.4f}")
            print(f"IoU (Mean IoU): {iou:.4f}")
            print(f"RQ (Recognition Quality): {rq:.4f}")
            
            # Print detailed results
            self.evaluator.print_results()
            
            return {
                'PQ': float(pq),
                'IoU': float(iou),
                'RQ': float(rq),
                'successful_batches': successful,
                'failed_batches': failed
            }
        except Exception as e:
            print(f"Error computing final metrics: {e}")
            print("This usually means no valid predictions were made")
            return None


@click.command()
@click.option('--model', required=True, help='Path to model')
@click.option('--dataset', default='KITTI', type=click.Choice(['KITTI', 'NUSCENES']))
@click.option('--batch-size', default=1, type=int)
@click.option('--max-batches', type=int, help='Max batches to evaluate')
@click.option('--debug', is_flag=True, help='Debug mode')
def main(model, dataset, batch_size, max_batches, debug):
    """Evaluate direct model"""
    
    # Load config
    config_dir = Path(__file__).parent.parent / "config"
    cfg = edict()
    
    for config_file in ['model.yaml', 'backbone.yaml', 'decoder.yaml']:
        config_path = config_dir / config_file
        if config_path.exists():
            with open(config_path, 'r') as f:
                cfg.update(yaml.safe_load(f))
    
    cfg.MODEL.DATASET = dataset
    cfg.TRAIN.BATCH_SIZE = batch_size
    cfg.TRAIN.NUM_WORKERS = 0 if debug else 4
    
    # Setup data
    print("Setting up dataset...")
    data_module = SemanticDatasetModule(cfg)
    data_module.setup()
    dataloader = data_module.val_dataloader()
    
    # Debug mode - test one batch
    if debug:
        print("\nDebug mode - testing one batch...")
        evaluator = DirectModelEvaluator(model, cfg)
        
        for batch in dataloader:
            try:
                sem_pred, ins_pred = evaluator.process_batch(batch)
                print(f"Success!")
                print(f"  Predictions: {[p.shape for p in sem_pred]}")
                print(f"  Ground truth: {[batch['sem_label'][i].shape for i in range(len(batch['sem_label']))]}")
                
                # Check values
                for i, sp in enumerate(sem_pred):
                    unique_classes = np.unique(sp)
                    print(f"  Sample {i}: predicted classes = {unique_classes}")
                    
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
            break
        return
    
    # Full evaluation
    evaluator = DirectModelEvaluator(model, cfg)
    metrics = evaluator.evaluate(dataloader, max_batches)
    
    if metrics:
        print(f"\nFinal Results:")
        print(f"  PQ = {metrics['PQ']:.4f}")
        print(f"  IoU = {metrics['IoU']:.4f}")
        print(f"  RQ = {metrics['RQ']:.4f}")
        
        # Save metrics
        output_file = Path(model).stem + '_metrics.yaml'
        with open(output_file, 'w') as f:
            yaml.dump(metrics, f, default_flow_style=False)
        print(f"\nMetrics saved to {output_file}")
    else:
        print("\nEvaluation failed - no valid metrics computed")


if __name__ == "__main__":
    main()