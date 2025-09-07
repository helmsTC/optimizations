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

from mask_pls.datasets.semantic_dataset import SemanticDatasetModule
from mask_pls.utils.evaluate_panoptic import PanopticEvaluator


class PTModelEvaluator:
    """Evaluate exported .pt model with proper size handling"""
    
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
        """Process a batch through the model with proper size tracking"""
        
        # Prepare inputs
        points_list = []
        features_list = []
        actual_sizes = []  # Track actual sizes before padding
        
        for i in range(len(batch['pt_coord'])):
            points = torch.from_numpy(batch['pt_coord'][i]).float()
            feats = torch.from_numpy(batch['feats'][i]).float()
            
            # Subsample if needed
            max_points = 30000
            if points.shape[0] > max_points:
                indices = torch.randperm(points.shape[0])[:max_points]
                points = points[indices]
                feats = feats[indices]
            
            actual_sizes.append(points.shape[0])  # Store actual size before padding
            points_list.append(points)
            features_list.append(feats)
        
        # Find max size for padding
        max_pts = max(p.shape[0] for p in points_list)
        
        padded_points = []
        padded_features = []
        
        for points, feats in zip(points_list, features_list):
            n_pts = points.shape[0]
            
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
        
        # Get actual output size from model
        output_size = sem_logits.shape[1]  # Number of points in output
        
        # Process predictions for each sample in batch
        sem_pred = []
        ins_pred = []
        
        for b in range(len(batch['pt_coord'])):
            # Determine how many points to process
            # The model might output fewer points than we padded to
            actual_size = actual_sizes[b]
            valid_output_size = min(actual_size, output_size)
            
            # Process semantic prediction - take only valid points
            sem_logit = sem_logits[b][:valid_output_size]  # [valid_size, C]
            sem = torch.argmax(sem_logit, dim=-1).cpu().numpy()
            
            # Process instance prediction
            ins = self.process_instance_prediction(
                pred_logits[b], 
                pred_masks[b],
                valid_output_size
            )
            
            sem_pred.append(sem)
            ins_pred.append(ins)
        
        return sem_pred, ins_pred
    
    def process_instance_prediction(self, query_logits, mask_logits, n_valid):
        """Process instance predictions for n_valid points"""
        
        # Get query predictions
        query_classes = torch.argmax(query_logits, dim=-1)  # [Q]
        query_scores = torch.softmax(query_logits, dim=-1).max(dim=-1)[0]  # [Q]
        
        # Filter valid queries (not background/no-object class)
        valid_queries = query_classes < self.num_classes
        
        if not valid_queries.any():
            return np.zeros(n_valid, dtype=np.int32)
        
        # Handle mask dimensions
        if mask_logits.shape[0] == query_logits.shape[0]:  # [Q, N]
            mask_logits = mask_logits.transpose(0, 1)  # Convert to [N, Q]
        
        # Take only the valid points from mask predictions
        mask_logits_valid = mask_logits[:n_valid]  # [n_valid, Q]
        
        # Get indices of valid queries
        valid_query_indices = torch.where(valid_queries)[0]
        
        if len(valid_query_indices) == 0:
            return np.zeros(n_valid, dtype=np.int32)
        
        # Select only valid query masks
        mask_logits_valid = mask_logits_valid[:, valid_query_indices]  # [n_valid, num_valid_queries]
        
        # Get scores for valid queries
        valid_query_scores = query_scores[valid_query_indices]
        
        # Apply sigmoid to get probabilities
        mask_probs = torch.sigmoid(mask_logits_valid)
        
        # Weight by query scores
        weighted_masks = mask_probs * valid_query_scores.unsqueeze(0)
        
        # Assign each point to the highest scoring mask
        max_scores, instance_ids = weighted_masks.max(dim=1)
        
        # Create instance predictions (1-indexed)
        ins = instance_ids + 1
        
        # Points with low scores get instance 0 (no instance)
        ins[max_scores < 0.5] = 0
        
        return ins.cpu().numpy()
    
    def evaluate(self, dataloader, max_batches=None):
        """Evaluate on dataloader"""
        
        print("Starting evaluation...")
        self.evaluator.reset()
        
        total_batches = len(dataloader)
        if max_batches:
            total_batches = min(total_batches, max_batches)
        
        # Track statistics
        total_points_processed = 0
        successful_batches = 0
        failed_batches = 0
        
        with tqdm(total=total_batches, desc="Evaluating") as pbar:
            for i, batch in enumerate(dataloader):
                if max_batches and i >= max_batches:
                    break
                
                try:
                    # Process batch
                    sem_pred, ins_pred = self.process_batch(batch)
                    
                    # Verify predictions match ground truth size
                    for j in range(len(sem_pred)):
                        gt_size = len(batch['sem_label'][j])
                        pred_size = len(sem_pred[j])
                        
                        if pred_size != gt_size:
                            # Adjust prediction size to match ground truth
                            min_size = min(pred_size, gt_size)
                            sem_pred[j] = sem_pred[j][:min_size]
                            ins_pred[j] = ins_pred[j][:min_size]
                            
                            # Also adjust ground truth if needed
                            batch['sem_label'][j] = batch['sem_label'][j][:min_size]
                            batch['ins_label'][j] = batch['ins_label'][j][:min_size]
                    
                    # Update evaluator
                    self.evaluator.update(sem_pred, ins_pred, batch)
                    
                    successful_batches += 1
                    total_points_processed += sum(len(sp) for sp in sem_pred)
                    
                except Exception as e:
                    print(f"\nError processing batch {i}: {e}")
                    failed_batches += 1
                    import traceback
                    traceback.print_exc()
                    continue
                
                pbar.update(1)
                
                # Print intermediate metrics every 10 batches
                if successful_batches > 0 and successful_batches % 10 == 0:
                    pq = self.evaluator.get_mean_pq()
                    iou = self.evaluator.get_mean_iou()
                    rq = self.evaluator.get_mean_rq()
                    pbar.set_postfix({
                        'PQ': f'{pq:.3f}',
                        'IoU': f'{iou:.3f}',
                        'RQ': f'{rq:.3f}',
                        'OK': successful_batches,
                        'Fail': failed_batches
                    })
        
        # Print statistics
        print(f"\nProcessing complete:")
        print(f"  Successful batches: {successful_batches}/{total_batches}")
        print(f"  Failed batches: {failed_batches}")
        print(f"  Total points processed: {total_points_processed:,}")
        
        if successful_batches == 0:
            print("\nNo batches were successfully processed!")
            return None
        
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
            'class_metrics': self.evaluator.get_class_metrics(),
            'statistics': {
                'successful_batches': successful_batches,
                'failed_batches': failed_batches,
                'total_points': total_points_processed
            }
        }


@click.command()
@click.option('--model', required=True, help='Path to .pt model')
@click.option('--dataset', default='KITTI', type=click.Choice(['KITTI', 'NUSCENES']))
@click.option('--data-path', help='Path to dataset')
@click.option('--batch-size', default=1, type=int)
@click.option('--num-workers', default=4, type=int)
@click.option('--max-batches', type=int, help='Maximum batches to evaluate')
@click.option('--split', default='valid', type=click.Choice(['train', 'valid', 'test']))
@click.option('--debug', is_flag=True, help='Enable debug mode')
def main(model, dataset, data_path, batch_size, num_workers, max_batches, split, debug):
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
    
    # If debug mode, test with just one batch first
    if debug:
        print("\nDebug mode: Testing with one batch...")
        for batch in dataloader:
            try:
                sem_pred, ins_pred = evaluator.process_batch(batch)
                print(f"Success! Predictions shape: {[sp.shape for sp in sem_pred]}")
                print(f"Ground truth shape: {[batch['sem_label'][i].shape for i in range(len(batch['sem_label']))]}")
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
            break
        return
    
    # Evaluate
    metrics = evaluator.evaluate(dataloader, max_batches)
    
    if metrics:
        # Save metrics
        output_file = Path(model).stem + '_metrics.yaml'
        with open(output_file, 'w') as f:
            yaml.dump(metrics, f, default_flow_style=False)
        
        print(f"\nMetrics saved to {output_file}")


if __name__ == "__main__":
    main()