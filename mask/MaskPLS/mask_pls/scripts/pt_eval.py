#!/usr/bin/env python3
"""
Evaluate direct model without Lightning
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

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mask_pls.datasets.semantic_dataset import SemanticDatasetModule
from mask_pls.utils.evaluate_panoptic import PanopticEvaluator

# Import the model class
from save_direct_model import DirectInferenceModel


class DirectModelEvaluator:
    """Evaluate direct model"""
    
    def __init__(self, model_path, cfg, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading model from {model_path}")
        
        # Load saved model
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Recreate model
        if 'config' in checkpoint:
            cfg = checkpoint['config']
        
        # Create model directly (not from checkpoint this time)
        from mask_pls.models.dgcnn.dgcnn_backbone_efficient import FixedDGCNNBackbone
        from mask_pls.models.decoder import MaskedTransformerDecoder
        
        dataset = cfg.MODEL.DATASET
        
        # Create components
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
        if 'decoder_state_dict' in checkpoint:
            self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        
        self.backbone = self.backbone.to(self.device)
        self.decoder = self.decoder.to(self.device)
        self.backbone.eval()
        self.decoder.eval()
        
        # Setup evaluator
        self.evaluator = PanopticEvaluator(cfg[dataset], dataset)
        self.num_classes = cfg[dataset].NUM_CLASSES
        self.things_ids = [1, 2, 3, 4, 5, 6, 7, 8] if dataset == 'KITTI' else [2, 3, 4, 5, 6, 7, 9, 10]
        
        print(f"Model loaded on {self.device}")
    
    def process_batch(self, batch):
        """Process batch through model"""
        
        with torch.no_grad():
            # Run through backbone
            feats, coords, pad_masks, sem_logits = self.backbone(batch)
            
            # Run through decoder
            outputs, padding = self.decoder(feats, coords, pad_masks)
        
        # Process outputs
        sem_pred, ins_pred = self.panoptic_inference(outputs, padding)
        
        return sem_pred, ins_pred
    
    def panoptic_inference(self, outputs, padding):
        """Panoptic inference"""
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
            
            # Handle mask dimensions
            if mask_pred.shape[1] == mask_cls.shape[1]:  # [B, Q, N]
                mask_pred_b = mask_pred[b].transpose(0, 1)  # [N, Q]
            else:  # [B, N, Q]
                mask_pred_b = mask_pred[b]
            
            # Extract valid points
            mask_pred_b = mask_pred_b[valid_mask].sigmoid()
            
            # Filter valid predictions
            keep = labels.ne(self.num_classes)
            
            if keep.sum() == 0:
                sem_pred.append(np.zeros(num_valid, dtype=np.int32))
                ins_pred.append(np.zeros(num_valid, dtype=np.int32))
                continue
            
            # Process predictions
            keep_indices = torch.where(keep)[0]
            cur_scores = scores[keep_indices]
            cur_classes = labels[keep_indices]
            cur_masks = mask_pred_b[:, keep_indices]
            
            # Weighted masks
            cur_prob_masks = cur_scores.unsqueeze(0) * cur_masks
            
            # Assign classes
            semantic_seg = torch.zeros(num_valid, dtype=torch.int32, device=cur_masks.device)
            instance_seg = torch.zeros(num_valid, dtype=torch.int32, device=cur_masks.device)
            
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
                    print(f"\nError in batch {i}: {e}")
                    import traceback
                    traceback.print_exc()
                
                pbar.update(1)
                
                # Show metrics
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
        print("Final Metrics")
        print("="*60)
        
        pq = self.evaluator.get_mean_pq()
        iou = self.evaluator.get_mean_iou()
        rq = self.evaluator.get_mean_rq()
        
        print(f"PQ: {pq:.4f}")
        print(f"IoU: {iou:.4f}")
        print(f"RQ: {rq:.4f}")
        
        self.evaluator.print_results()
        
        return {'PQ': pq, 'IoU': iou, 'RQ': rq}


@click.command()
@click.option('--model', required=True, help='Path to model')
@click.option('--dataset', default='KITTI')
@click.option('--batch-size', default=1)
@click.option('--max-batches', type=int)
def main(model, dataset, batch_size, max_batches):
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
    
    # Setup data
    data_module = SemanticDatasetModule(cfg)
    data_module.setup()
    dataloader = data_module.val_dataloader()
    
    # Evaluate
    evaluator = DirectModelEvaluator(model, cfg)
    metrics = evaluator.evaluate(dataloader, max_batches)
    
    print(f"\nFinal: PQ={metrics['PQ']:.4f}, IoU={metrics['IoU']:.4f}, RQ={metrics['RQ']:.4f}")


if __name__ == "__main__":
    main()