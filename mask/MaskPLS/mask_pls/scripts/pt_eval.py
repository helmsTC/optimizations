#!/usr/bin/env python3
"""
Fixed evaluation script for .pt model with proper tensor indexing
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
    """Evaluate exported .pt model with fixed tensor operations"""
    
    def __init__(self, model_path, cfg, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading .pt model from {model_path}")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        
        # Setup evaluator
        dataset = cfg.MODEL.DATASET
        self.evaluator = PanopticEvaluator(cfg[dataset], dataset)
        self.num_classes = cfg[dataset].NUM_CLASSES
        self.things_ids = [1, 2, 3, 4, 5, 6, 7, 8] if dataset == 'KITTI' else [2, 3, 4, 5, 6, 7, 9, 10]
        
        print(f"Model loaded on {self.device}")
        print(f"Dataset: {dataset}, Num classes: {self.num_classes}")
    
    def process_batch(self, batch):
        """Process a batch through the model with fixed indexing"""
        
        # Prepare inputs
        points_list = []
        features_list = []
        original_sizes = []
        
        for i in range(len(batch['pt_coord'])):
            points = torch.from_numpy(batch['pt_coord'][i]).float()
            feats = torch.from_numpy(batch['feats'][i]).float()
            
            original_sizes.append(points.shape[0])
            
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
            
            # Create padding mask
            mask = torch.ones(max_pts, dtype=torch.bool)
            mask[:n_pts] = False
            valid_masks.append(mask)
            
            if n_pts < max_pts:
                pad_size = max_pts - n_pts
                points = torch.nn.functional.pad(points, (0, 0, 0, pad_size))
                feats = torch.nn.functional.pad(feats, (0, 0, 0, pad_size))
            
            padded_points.append(points)
            padded_features.append(feats)
        
        # Stack and move to device
        points_tensor = torch.stack(padded_points).to(self.device)
        features_tensor = torch.stack(padded_features).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(points_tensor, features_tensor)
        
        pred_logits = outputs[0]  # [B, Q, C+1]
        pred_masks = outputs[1]   # [B, N, Q] or [B, Q, N]
        sem_logits = outputs[2]   # [B, N, C]
        
        # Process predictions for each sample in batch
        sem_pred = []
        ins_pred = []
        
        for b in range(len(batch['pt_coord'])):
            # Get the actual number of valid points
            valid_mask = ~valid_masks[b]
            n_valid = valid_mask.sum().item()
            
            # Process semantic prediction
            sem_logit = sem_logits[b][valid_mask]  # Extract only valid points
            sem = torch.argmax(sem_logit, dim=-1).cpu().numpy()
            
            # Process instance prediction
            ins = self.process_instance_prediction(
                pred_logits[b], 
                pred_masks[b],
                valid_mask,
                n_valid
            )
            
            # Truncate to original size if it was subsampled
            orig_size = min(original_sizes[b], len(sem))
            sem = sem[:orig_size]
            ins = ins[:orig_size]
            
            sem_pred.append(sem)
            ins_pred.append(ins)
        
        return sem_pred, ins_pred
    
    def process_instance_prediction(self, query_logits, mask_logits, valid_mask, n_valid):
        """Process instance predictions with proper tensor indexing"""
        
        # Get query predictions
        query_classes = torch.argmax(query_logits, dim=-1)  # [Q]
        query_scores = torch.softmax(query_logits, dim=-1).max(dim=-1)[0]  # [Q]
        
        # Filter valid queries (not background/no-object class)
        valid_queries = query_classes < self.num_classes  # [Q] boolean mask
        
        if not valid_queries.any():
            return np.zeros(n_valid, dtype=np.int32)
        
        # Handle mask dimensions
        if mask_logits.shape[0] == query_logits.shape[0]:  # [Q, N]
            mask_logits = mask_logits.transpose(0, 1)  # Convert to [N, Q]
        
        # Extract valid points and valid queries
        mask_logits_valid = mask_logits[valid_mask]  # [n_valid, Q]
        
        # Get indices of valid queries instead of boolean indexing
        valid_query_indices = torch.where(valid_queries)[0]  # [num_valid_queries]
        
        if len(valid_query_indices) == 0:
            return np.zeros(n_valid, dtype=np.int32)
        
        # Select only valid query masks using indices
        mask_logits_valid = mask_logits_valid[:, valid_query_indices]  # [n_valid, num_valid_queries]
        
        # Get scores for valid queries
        valid_query_scores = query_scores[valid_query_indices]  # [num_valid_queries]
        
        # Apply sigmoid to get probabilities
        mask_probs = torch.sigmoid(mask_logits_valid)  # [n_valid, num_valid_queries]
        
        # Weight by query scores
        weighted_masks = mask_probs * valid_query_scores.unsqueeze(0)  # [n_valid, num_valid_queries]
        
        # Assign each point to the highest scoring mask
        max_scores, instance_ids = weighted_masks.max(dim=1)  # [n_valid]
        
        # Create instance predictions (1-indexed)
        ins = instance_ids + 1
        
        # Points with low scores get instance 0 (no instance)
        ins[max_scores < 0.5] = 0
        
        return ins.cpu().numpy()
    
    def panoptic_inference_safe(self, outputs, padding):
        """Safe panoptic inference with proper indexing"""
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
            
            # Get predictions for this sample
            scores, labels = mask_cls[b].max(-1)
            
            # Handle mask dimensions
            if mask_pred.dim() == 3:
                if mask_pred.shape[1] == mask_cls.shape[1]:  # [B, Q, N]
                    mask_pred_b = mask_pred[b].transpose(0, 1)  # [N, Q]
                else:  # [B, N, Q]
                    mask_pred_b = mask_pred[b]  # [N, Q]
            else:
                mask_pred_b = mask_pred[b]
            
            # Extract valid points
            mask_pred_b = mask_pred_b[valid_mask].sigmoid()  # [num_valid, Q]
            
            # Filter valid predictions (not background)
            keep = labels.ne(self.num_classes)
            
            if keep.sum() == 0:
                sem_pred.append(np.zeros(num_valid, dtype=np.int32))
                ins_pred.append(np.zeros(num_valid, dtype=np.int32))
                continue
            
            # Get indices of valid queries
            keep_indices = torch.where(keep)[0]
            
            cur_scores = scores[keep_indices]
            cur_classes = labels[keep_indices]
            cur_masks = mask_pred_b[:, keep_indices]  # [num_valid, num_kept_queries]
            
            # Weighted masks
            cur_prob_masks = cur_scores.unsqueeze(0) * cur_masks
            
            # Initialize outputs
            semantic_seg = torch.zeros(num_valid, dtype=torch.int32, device=cur_masks.device)
            instance_seg = torch.zeros(num_valid, dtype=torch.int32, device=cur_masks.device)
            
            if cur_masks.shape[1] > 0:
                # Assign points to masks
                cur_mask_ids = cur_prob_masks.argmax(1)
                
                current_segment_id = 0
                stuff_memory_list = {}
                
                for k in range(cur_classes.shape[0]):
                    pred_class = cur_classes[k].item()
                    isthing = pred_class in self.things_ids
                    
                    # Points belonging to this mask
                    mask = (cur_mask_ids == k) & (cur_masks[:, k] >= 0.5)
                    mask_area = mask.sum().item()
                    
                    if mask_area > 0:
                        semantic_seg[mask] = pred_class
                        
                        if isthing:
                            current_segment_id += 1
                            instance_seg[mask] = current_segment_id
                        else:
                            if pred_class not in stuff_memory_list:
                                current_segment_id += 1
                                stuff_memory_list[pred_class] = current_segment_id
                            # For stuff classes, we typically don't assign instances
                            # but you can uncomment the line below if needed
                            # instance_seg[mask] = stuff_memory_list[pred_class]
            
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
                    print(f"\nError processing batch {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                pbar.update(1)
                
                # Print intermediate metrics every 10 batches
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
            'PQ': float(pq),
            'IoU': float(iou),
            'RQ': float(rq),
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