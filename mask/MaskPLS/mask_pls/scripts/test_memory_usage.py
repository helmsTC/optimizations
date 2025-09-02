# test_memory.py
import torch
import torch.nn as nn
import numpy as np
import gc
import os
from pathlib import Path
import yaml
from easydict import EasyDict as edict
from mask_pls.models.dgcnn_backbone_efficient import EfficientDGCNNBackbone
from mask_pls.models.mask_dgcnn_optimized import MaskPLSDGCNNOptimized


def test_memory():
    """Test memory usage of the model"""
    # Force CUDA memory management
    torch.cuda.empty_cache()
    gc.collect()
    
    # Get initial memory
    initial_memory = torch.cuda.memory_allocated() / 1024**2
    print(f"Initial GPU memory: {initial_memory:.2f} MB")
    
    # Load config
    from train_efficient_dgcnn import get_config
    cfg = get_config()
    cfg.TRAIN.MIXED_PRECISION = False  # Disable for testing
    
    # Test backbone only first
    print("\nTesting Backbone Memory Usage:")
    backbone = EfficientDGCNNBackbone(cfg.BACKBONE)
    backbone = backbone.cuda()
    backbone.eval()
    
    test_sizes = [1000, 5000, 10000, 20000, 50000]
    
    for num_points in test_sizes:
        torch.cuda.empty_cache()
        gc.collect()
        
        before_memory = torch.cuda.memory_allocated() / 1024**2
        
        try:
            # Create dummy input
            batch_size = 1
            coords = np.random.randn(num_points, 3).astype(np.float32)
            feats = np.random.randn(num_points, 4).astype(np.float32)
            
            x = {
                'pt_coord': [coords],
                'feats': [feats]
            }
            
            # Forward pass
            with torch.no_grad():
                outputs = backbone(x)
            
            after_memory = torch.cuda.memory_allocated() / 1024**2
            memory_used = after_memory - before_memory
            
            print(f"{num_points} points: {memory_used:.2f} MB used")
            
        except RuntimeError as e:
            print(f"{num_points} points: Failed - {str(e)}")
        
        # Clean up
        del x
        if 'outputs' in locals():
            del outputs
        torch.cuda.empty_cache()
        gc.collect()
    
    # Test full model
    print("\nTesting Full Model Memory Usage:")
    del backbone
    torch.cuda.empty_cache()
    gc.collect()
    
    model = MaskPLSDGCNNOptimized(cfg)
    model = model.cuda()
    model.eval()
    model.things_ids = [1, 2, 3, 4, 5, 6, 7, 8]  # Mock things IDs
    
    for num_points in [1000, 5000, 10000]:
        torch.cuda.empty_cache()
        gc.collect()
        
        before_memory = torch.cuda.memory_allocated() / 1024**2
        
        try:
            # Create dummy batch
            batch = create_dummy_batch(num_points, cfg)
            
            # Forward pass
            with torch.no_grad():
                outputs, padding, sem_logits = model(batch)
            
            after_memory = torch.cuda.memory_allocated() / 1024**2
            memory_used = after_memory - before_memory
            
            print(f"{num_points} points: {memory_used:.2f} MB used")
            
        except RuntimeError as e:
            print(f"{num_points} points: Failed - {str(e)}")
        
        # Clean up
        del batch
        if 'outputs' in locals():
            del outputs, padding, sem_logits
        torch.cuda.empty_cache()
        gc.collect()


def create_dummy_batch(num_points, cfg):
    """Create a dummy batch for testing"""
    batch_size = 1
    num_masks = 10
    
    batch = {
        'pt_coord': [np.random.randn(num_points, 3).astype(np.float32)],
        'feats': [np.random.randn(num_points, 4).astype(np.float32)],
        'sem_label': [np.random.randint(0, 20, (num_points, 1)).astype(np.int64)],
        'ins_label': [np.random.randint(0, 100, (num_points, 1)).astype(np.int64)],
        'masks': [torch.randint(0, 2, (num_masks, num_points), dtype=torch.float32)],
        'masks_cls': [torch.randint(0, 20, (num_masks,))],
        'masks_ids': [[torch.randint(0, num_points, (100,)) for _ in range(num_masks)]],
        'fname': ['dummy.bin'],
        'pose': [np.eye(4)],
        'token': ['0']
    }
    
    return batch


if __name__ == "__main__":
    test_memory()