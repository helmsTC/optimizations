#!/usr/bin/env python3
"""
Evaluate .pt model and compute metrics
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


class PTModelEvaluator:
    """Evaluate exported .pt model"""
    
    def __init__(self, model_path, cfg, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading .pt model from {model_path}")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        
        # Setup evaluator
        dataset = cfg.MODEL.DATASET
        self.evaluator = PanopticEvaluator(cfg[dataset], dataset)
        self.num_classes = cfg[dataset].NUM_CLASSES
        self.things_ids = []  # Will be set from datamodule
        
        print(f"Model loaded on {self.device}")
    
    def process_batch(self, batch):
        """Process a batch through the model"""
        
        # Prepare inputs
        points_list = []
        features_list = []
        
        for i in range(len(batch['pt_coord'])):
            points = torch.from_numpy(batch['pt_coord'][i]).float()
            feats = torch.from_numpy(batch['feats'][i]).float()
            
            # Subsample if needed
            max_points = 30000
            if points.shape[0] > max_points:
                indices = torch.randperm(points.shape[0])[:max_points]
                points = points[indices]
                feats = feats[indices]
            
            points_list.append(points)
            features_list.append(feats)
        
        # Pad to same size
        max_pts = max(p.shape[0] for p in points_list)
        
        padded_points = []
        padded_features = []
        valid_masks = []
        
        for points, feats in zip(points_list, features_list):
            n_pts = points.shape[0]
            if n_pts < max_pts:
                pad_size = max_pts - n_pts
                points = torch.nn.functional.pad(points, (0, 0, 0, pad_size))
                feats = torch.nn.functional.pad(feats, (0, 0, 0, pad_size))
            
            padded_points.append(points)
            padded_features.append(feats)
            valid_masks.append(n_pts)
        
        # Stack and move to device
        points_tensor = torch.stack(padded_points).to(self.device)
        features_tensor = torch.stack(padded_features).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(points_tensor, features_tensor)
        
        pred_logits = outputs[0]  # [B, Q, C+1]
        pred_masks = outputs[1]   # [B, N, Q] or [B, Q, N]
        sem_logits = outputs[2]   # [B, N, C]
        
        # Process predictions
        sem_pred = []
        ins_pred = []
        
        for b in range(len(batch['pt_coord'])):
            # Get valid points
            n_valid = valid_masks[b]
            
            # Semantic prediction
            sem_logit = sem_logits[b, :n_valid]
            sem = torch.argmax(sem_logit, dim=-1).cpu().numpy()
            
            # Instance prediction (simplified)
            mask_logit = pred_masks[b]
            if mask_logit.shape[0] != n_valid:
                mask_logit = mask_logit.transpose(0, 1)
            mask_logit = mask_logit[:n_valid]
            
            query_logit = pred_logits[b]
            query_classes = torch.argmax(query_logit, dim=-1)
            query_scores = torch.softmax(query_logit, dim=-1).max(dim=-1)[0]
            
            # Simple instance assignment
            valid_queries = query_classes < self.num_classes
            
            if valid_queries.sum() > 0:
                mask_probs = torch.sigmoid(mask_logit[:, valid_queries])
                weighted_masks = mask_probs * query_scores[valid_queries]
                
                ins = torch.argmax(weighted_masks, dim=1) + 1
                ins[weighted_masks.max(dim=1)[0] < 0.5] = 0
                ins = ins.cpu().numpy()
            else:
                ins = np.zeros_like(sem)
            
            sem_pred.append(sem)
            ins_pred.append(ins)
        
        return sem_pred, ins_pred
    
    def evaluate(self, dataloader, max_batches=None):
        """Evaluate on dataloader"""
        
        print("Starting evaluation...")
        self.evaluator.reset()
        
        total_batches = len(dataloader)
        if max_batches:
            total_batches = min(total_batches, max_batches)
        
        with tqdm(total=total_batches, desc="Evaluating") as pbar:
            for i, batch in enumerate(dataloader):
                if max_batches and i >= max_batches:
                    break
                
                try:
                    # Process batch
                    sem_pred, ins_pred = self.process_batch(batch)
                    
                    # Update evaluator
                    self.evaluator.update(sem_pred, ins_pred, batch)
                    
                except Exception as e:
                    print(f"Error processing batch {i}: {e}")
                    continue
                
                pbar.update(1)
                
                # Print intermediate metrics
                if (i + 1) % 10 == 0:
                    pq = self.evaluator.get_mean_pq()
                    iou = self.evaluator.get_mean_iou()
                    rq = self.evaluator.get_mean_rq()
                    pbar.set_postfix({
                        'PQ': f'{pq:.3f}',
                        'IoU': f'{iou:.3f}',
                        'RQ': f'{rq:.3f}'
                    })
        
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
        
        # Per-class metrics
        self.evaluator.print_results()
        
        return {
            'PQ': pq,
            'IoU': iou,
            'RQ': rq,
            'class_metrics': self.evaluator.get_class_metrics()
        }


@click.command()
@click.option('--model', required=True, help='Path to .pt model')
@click.option('--dataset', default='KITTI', type=click.Choice(['KITTI', 'NUSCENES']))
@click.option('--data-path', help='Path to dataset')
@click.option('--batch-size', default=1, type=int)
@click.option('--num-workers', default=4, type=int)
@click.option('--max-batches', type=int, help='Maximum batches to evaluate')
@click.option('--split', default='valid', type=click.Choice(['train', 'valid', 'test']))
def main(model, dataset, data_path, batch_size, num_workers, max_batches, split):
    """Evaluate .pt model on dataset"""
    
    # Load configuration
    config_dir = Path(__file__).parent.parent / "config"
    cfg = edict()
    
    for config_file in ['model.yaml', 'backbone.yaml', 'decoder.yaml']:
        config_path = config_dir / config_file
        if config_path.exists():
            with open(config_path, 'r') as f:
                cfg.update(yaml.safe_load(f))
    
    # Update config
    cfg.MODEL.DATASET = dataset
    cfg.TRAIN.BATCH_SIZE = batch_size
    cfg.TRAIN.NUM_WORKERS = num_workers
    
    if data_path:
        cfg[dataset].PATH = data_path
    
    # Create data module
    print("Setting up dataset...")
    data_module = SemanticDatasetModule(cfg)
    data_module.setup()
    
    # Get dataloader
    if split == 'train':
        dataloader = data_module.train_dataloader()
    elif split == 'valid':
        dataloader = data_module.val_dataloader()
    else:
        dataloader = data_module.test_dataloader()
    
    # Create evaluator
    evaluator = PTModelEvaluator(model, cfg)
    evaluator.things_ids = data_module.things_ids
    
    # Evaluate
    metrics = evaluator.evaluate(dataloader, max_batches)
    
    # Save metrics
    output_file = Path(model).stem + '_metrics.yaml'
    with open(output_file, 'w') as f:
        yaml.dump(metrics, f, default_flow_style=False)
    
    print(f"\nMetrics saved to {output_file}")


if __name__ == "__main__":
    main()