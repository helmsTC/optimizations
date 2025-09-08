#!/usr/bin/env python3
"""
Test standalone .pt model to verify it matches checkpoint performance
"""

import os
import torch
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


# Import the StandaloneInferenceModel class from the converter
exec(open('pt_converter_improved.py').read())  # Or import it properly


def test_standalone_model(model_path, cfg, data_module, max_batches=None):
    """Test standalone model"""
    
    print(f"Loading standalone model from {model_path}")
    
    # Load saved model
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Verify checkpoint contents
    print("Checkpoint contents:")
    for key in checkpoint.keys():
        if key != 'model_state_dict':
            print(f"  {key}: {checkpoint[key] if not isinstance(checkpoint[key], dict) else f'dict with {len(checkpoint[key])} items'}")
    
    # Create model
    things_ids = checkpoint['things_ids']
    model = StandaloneInferenceModel(cfg, things_ids)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to GPU and eval mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"  Things IDs: {model.things_ids}")
    print(f"  Num classes: {model.num_classes}")
    
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
        print(f"  Outputs shape: {outputs['pred_logits'].shape}")
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
            
            pbar.update(1)
            
            # Show metrics periodically
            if successful > 0 and successful % 10 == 0:
                pq = evaluator.get_mean_pq()
                iou = evaluator.get_mean_iou()
                rq = evaluator.get_mean_rq()
                pbar.set_postfix({
                    'PQ': f'{pq:.3f}',
                    'IoU': f'{iou:.3f}',
                    'RQ': f'{rq:.3f}'
                })
    
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
@click.option('--model', required=True, help='Path to standalone .pt model')
@click.option('--dataset', default='KITTI', type=click.Choice(['KITTI', 'NUSCENES']))
@click.option('--batch-size', default=1, type=int)
@click.option('--max-batches', type=int, help='Max batches to evaluate')
@click.option('--num-workers', default=4, type=int)
def main(model, dataset, batch_size, max_batches, num_workers):
    """Test standalone .pt model"""
    
    print("="*60)
    print("Standalone Model Testing")
    print("="*60)
    
    # Load configuration
    cfg = get_config()
    cfg.MODEL.DATASET = dataset
    cfg.TRAIN.BATCH_SIZE = batch_size
    cfg.TRAIN.NUM_WORKERS = num_workers
    
    # Create data module
    print("\nSetting up dataset...")
    data_module = SemanticDatasetModule(cfg)
    data_module.setup()
    
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