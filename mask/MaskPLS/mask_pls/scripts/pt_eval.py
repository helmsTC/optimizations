#!/usr/bin/env python3
"""
Evaluation script for inference model (not traced)
"""

import torch
import numpy as np
import yaml
import click
from pathlib import Path
from tqdm import tqdm
from easydict import EasyDict as edict

from mask_pls.datasets.semantic_dataset import SemanticDatasetModule
from mask_pls.utils.evaluate_panoptic import PanopticEvaluator


class InferenceEvaluator:
    """Evaluate inference model"""
    
    def __init__(self, model_path, cfg, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading inference model from {model_path}")
        saved = torch.load(model_path, map_location=self.device)
        
        self.model = saved['model']
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Setup evaluator
        dataset = cfg.MODEL.DATASET
        self.evaluator = PanopticEvaluator(cfg[dataset], dataset)
        self.num_classes = self.model.num_classes
        self.things_ids = self.model.things_ids
        
        print(f"Model loaded on {self.device}")
    
    def process_batch(self, batch):
        """Process batch through model"""
        
        # The model expects lists of numpy arrays
        points_list = batch['pt_coord']
        features_list = batch['feats']
        
        # Process through model
        with torch.no_grad():
            pred_logits, pred_masks, sem_logits, padding = self.model.forward(
                points_list, features_list
            )
        
        # Convert to CPU for processing
        pred_logits = pred_logits.cpu()
        pred_masks = pred_masks.cpu()
        sem_logits = sem_logits.cpu()
        padding = padding.cpu()
        
        # Process predictions
        sem_pred = []
        ins_pred = []
        
        batch_size = len(points_list)
        
        for b in range(batch_size):
            # Get valid mask
            valid_mask = ~padding[b]
            n_valid = valid_mask.sum().item()
            
            if n_valid == 0:
                sem_pred.append(np.zeros(0, dtype=np.int32))
                ins_pred.append(np.zeros(0, dtype=np.int32))
                continue
            
            # Semantic prediction
            sem_logit = sem_logits[b][valid_mask]
            sem = torch.argmax(sem_logit, dim=-1).numpy()
            
            # Instance prediction
            ins = self.process_instance(
                pred_logits[b],
                pred_masks[b],
                valid_mask
            )
            
            # Match ground truth size
            gt_size = len(batch['sem_label'][b])
            if len(sem) > gt_size:
                sem = sem[:gt_size]
                ins = ins[:gt_size]
            elif len(sem) < gt_size:
                # This shouldn't happen if model is working correctly
                print(f"Warning: Prediction size {len(sem)} < GT size {gt_size}")
            
            sem_pred.append(sem)
            ins_pred.append(ins)
        
        return sem_pred, ins_pred
    
    def process_instance(self, query_logits, mask_logits, valid_mask):
        """Process instance predictions"""
        
        # Get query predictions
        query_classes = torch.argmax(query_logits, dim=-1)
        query_scores = torch.softmax(query_logits, dim=-1).max(dim=-1)[0]
        
        # Filter valid queries
        valid_queries = query_classes < self.num_classes
        
        if not valid_queries.any():
            return np.zeros(valid_mask.sum().item(), dtype=np.int32)
        
        # Handle mask dimensions
        if mask_logits.shape[0] == query_logits.shape[0]:  # [Q, N]
            mask_logits = mask_logits.transpose(0, 1)  # [N, Q]
        
        # Extract valid points
        mask_logits_valid = mask_logits[valid_mask]
        
        # Get valid query indices
        valid_query_indices = torch.where(valid_queries)[0]
        
        if len(valid_query_indices) == 0:
            return np.zeros(valid_mask.sum().item(), dtype=np.int32)
        
        # Select valid query masks
        mask_logits_valid = mask_logits_valid[:, valid_query_indices]
        valid_query_scores = query_scores[valid_query_indices]
        
        # Apply sigmoid
        mask_probs = torch.sigmoid(mask_logits_valid)
        
        # Weight by query scores
        weighted_masks = mask_probs * valid_query_scores.unsqueeze(0)
        
        # Assign instances
        max_scores, instance_ids = weighted_masks.max(dim=1)
        
        ins = instance_ids + 1
        ins[max_scores < 0.5] = 0
        
        return ins.numpy()
    
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
                    
                    # Update evaluator
                    self.evaluator.update(sem_pred, ins_pred, batch)
                    
                    successful += 1
                    
                except Exception as e:
                    print(f"\nError in batch {i}: {e}")
                    failed += 1
                    import traceback
                    traceback.print_exc()
                
                pbar.update(1)
                
                # Show intermediate metrics
                if successful > 0 and successful % 10 == 0:
                    pq = self.evaluator.get_mean_pq()
                    iou = self.evaluator.get_mean_iou()
                    rq = self.evaluator.get_mean_rq()
                    pbar.set_postfix({
                        'PQ': f'{pq:.3f}',
                        'IoU': f'{iou:.3f}',
                        'RQ': f'{rq:.3f}'
                    })
        
        print(f"\nProcessed {successful} batches successfully, {failed} failed")
        
        # Final metrics
        print("\n" + "="*60)
        print("Final Evaluation Metrics")
        print("="*60)
        
        pq = self.evaluator.get_mean_pq()
        iou = self.evaluator.get_mean_iou()
        rq = self.evaluator.get_mean_rq()
        
        print(f"PQ (Panoptic Quality): {pq:.4f}")
        print(f"IoU (Mean IoU): {iou:.4f}")
        print(f"RQ (Recognition Quality): {rq:.4f}")
        
        self.evaluator.print_results()
        
        return {
            'PQ': float(pq),
            'IoU': float(iou),
            'RQ': float(rq)
        }


@click.command()
@click.option('--model', required=True, help='Path to inference model (.pth)')
@click.option('--dataset', default='KITTI', type=click.Choice(['KITTI', 'NUSCENES']))
@click.option('--batch-size', default=1, type=int)
@click.option('--max-batches', type=int, help='Max batches to evaluate')
@click.option('--split', default='valid', type=click.Choice(['train', 'valid', 'test']))
def main(model, dataset, batch_size, max_batches, split):
    """Evaluate inference model"""
    
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
    
    # Setup data
    data_module = SemanticDatasetModule(cfg)
    data_module.setup()
    
    if split == 'valid':
        dataloader = data_module.val_dataloader()
    else:
        dataloader = data_module.train_dataloader()
    
    # Evaluate
    evaluator = InferenceEvaluator(model, cfg)
    metrics = evaluator.evaluate(dataloader, max_batches)
    
    # Save metrics
    output_file = Path(model).stem + '_metrics.yaml'
    with open(output_file, 'w') as f:
        yaml.dump(metrics, f)
    
    print(f"\nMetrics saved to {output_file}")


if __name__ == "__main__":
    main()