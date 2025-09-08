#!/usr/bin/env python3
"""
Test standalone .pt model to verify it matches checkpoint performance
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
    
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    
    cfg.TRAIN.BATCH_SIZE = 2
    cfg.TRAIN.ACCUMULATE_GRAD_BATCHES = 2
    cfg.TRAIN.WARMUP_STEPS = 500
    cfg.TRAIN.SUBSAMPLE = True
    cfg.TRAIN.AUG = True
    cfg.TRAIN.LR = 0.0001
    
    return cfg


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
    def forward(self, batch_dict):
        """Forward pass matching the training model"""
        # Run through backbone
        feats, coords, pad_masks, sem_logits = self.backbone(batch_dict)
        
        # Run through decoder
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


def test_standalone_model(model_path, cfg, data_module, max_batches=None):
    """Test standalone model"""
    
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
    print(f"  Things IDs: {model.things_ids}")
    print(f"  Num classes: {model.num_classes}")
    
    # Verify weights
    print("\nVerifying weights:")
    if hasattr(model.backbone, 'edge_conv1'):
        weight = model.backbone.edge_conv1[0].weight
        print(f"  ✓ edge_conv1: max={weight.abs().max().item():.4f}")
    if hasattr(model.backbone, 'conv5'):
        weight = model.backbone.conv5[0].weight  
        print(f"  ✓ conv5: max={weight.abs().max().item():.4f}")
    if hasattr(model.decoder, 'query_feat'):
        weight = model.decoder.query_feat.weight
        print(f"  ✓ query_feat: max={weight.abs().max().item():.4f}")
    
    # Get dataloader
    dataloader = data_module.val_dataloader()
    
    # Setup evaluator
    dataset = cfg.MODEL.DATASET
    evaluator = PanopticEvaluator(cfg[dataset], dataset)
    evaluator.reset()
    
    # Test one batch first
    print("\nTesting single batch...")
    for batch in dataloader:
        with torch.no_grad():
            outputs, padding, sem_logits = model(batch)
            sem_pred, ins_pred = model.panoptic_inference(outputs, padding)
        
        print(f"  Batch processed successfully")
        print(f"  pred_logits shape: {outputs['pred_logits'].shape}")
        print(f"  pred_masks shape: {outputs['pred_masks'].shape}")
        print(f"  sem_logits shape: {sem_logits.shape}")
        print(f"  Semantic predictions: {len(np.unique(sem_pred[0]))} classes")
        break
    
    # Full evaluation
    print("\nRunning full evaluation...")
    
    total_batches = len(dataloader)
    if max_batches:
        total_batches = min(total_batches, max_batches)
    
    successful = 0
    
    with tqdm(total=total_batches, desc="Evaluating") as pbar:
        for i, batch in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break
            
            try:
                with torch.no_grad():
                    outputs, padding, sem_logits = model(batch)
                    sem_pred, ins_pred = model.panoptic_inference(outputs, padding)
                
                # Prepare batch for evaluation
                eval_batch = model.prepare_batch_for_eval(batch, sem_pred)
                
                # Update evaluator
                evaluator.update(sem_pred, ins_pred, eval_batch)
                successful += 1
                
                # Clear cache
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
                    pbar.set_postfix({
                        'PQ': f'{pq:.3f}',
                        'IoU': f'{iou:.3f}',
                        'RQ': f'{rq:.3f}'
                    })
                except:
                    pass
    
    print(f"\nProcessed {successful}/{total_batches} batches successfully")
    
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
@click.option('--model', required=True, type=str, help='Path to standalone .pt model')
@click.option('--dataset', default='KITTI', type=click.Choice(['KITTI', 'NUSCENES']))
@click.option('--batch-size', default=1, type=int)
@click.option('--max-batches', default=None, type=int, help='Max batches to evaluate')
@click.option('--num-workers', default=4, type=int)
def main(model, dataset, batch_size, max_batches, num_workers):
    """Test standalone .pt model"""
    
    print("="*60)
    print("Standalone Model Testing")
    print("="*60)
    print(f"Model path: {model}")
    
    # Load configuration
    cfg = get_config()
    cfg.MODEL.DATASET = dataset
    cfg.TRAIN.BATCH_SIZE = batch_size
    cfg.TRAIN.NUM_WORKERS = num_workers
    
    # Create data module
    print("\nSetting up dataset...")
    data_module = SemanticDatasetModule(cfg)
    data_module.setup()
    
    print(f"Data module things_ids: {data_module.things_ids}")
    
    # Test model
    metrics = test_standalone_model(model, cfg, data_module, max_batches)
    
    if metrics:
        print(f"\n" + "="*60)
        print(f"Final Results:")
        print(f"  PQ = {metrics['PQ']:.4f}")
        print(f"  IoU = {metrics['IoU']:.4f}")
        print(f"  RQ = {metrics['RQ']:.4f}")
        print("="*60)


if __name__ == "__main__":
    main()