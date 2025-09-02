# test_memory.py
import torch
import torch.nn as nn
import numpy as np
import gc
import os
import sys
from pathlib import Path
import yaml
from easydict import EasyDict as edict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from mask_pls.models.dgcnn.dgcnn_backbone_efficient import EfficientDGCNNBackbone
from mask_pls.models.dgcnn.maskpls_dgcnn_optimized import MaskPLSDGCNNOptimized


def get_config():
    """Load and merge configuration files"""
    def getDir(obj):
        return os.path.dirname(os.path.abspath(obj))
    
    config_dir = Path(__file__).parent.parent / "config"
    
    model_cfg = edict(yaml.safe_load(open(config_dir / "model.yaml")))
    backbone_cfg = edict(yaml.safe_load(open(config_dir / "backbone.yaml")))
    decoder_cfg = edict(yaml.safe_load(open(config_dir / "decoder.yaml")))
    
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    cfg.TRAIN.MIXED_PRECISION = False
    
    return cfg


def test_memory():
    """Test memory usage of the model"""
    # Force CUDA memory management
    torch.cuda.empty_cache()
    gc.collect()
    
    # Ensure we're using float32
    torch.set_default_dtype(torch.float32)
    
    # Get initial memory
    initial_memory = torch.cuda.memory_allocated() / 1024**2
    print(f"Initial GPU memory: {initial_memory:.2f} MB")
    
    # Load config
    cfg = get_config()
    
    # Test backbone only first
    print("\nTesting Backbone Memory Usage:")
    backbone = EfficientDGCNNBackbone(cfg.BACKBONE)
    backbone = backbone.cuda()
    backbone.eval()
    
    # Ensure float32 mode
    backbone = backbone.float()
    
    test_sizes = [1000, 5000, 10000, 20000, 50000]
    
    for num_points in test_sizes:
        torch.cuda.empty_cache()
        gc.collect()
        
        before_memory = torch.cuda.memory_allocated() / 1024**2
        
        try:
            # Create dummy input - ensure float32
            coords = np.random.randn(num_points, 3).astype(np.float32)
            feats = np.random.randn(num_points, 4).astype(np.float32)
            
            x = {
                'pt_coord': [coords],
                'feats': [feats]
            }
            
            # Forward pass
            with torch.no_grad():
                # Ensure no autocast
                with torch.cuda.amp.autocast(enabled=False):
                    outputs = backbone(x)
            
            after_memory = torch.cuda.memory_allocated() / 1024**2
            memory_used = after_memory - before_memory
            
            print(f"{num_points} points: {memory_used:.2f} MB used")
            
        except RuntimeError as e:
            print(f"{num_points} points: Failed - {str(e)}")
            
            # Additional debugging
            if "size of tensor" in str(e):
                print("  Tensor size mismatch detected")
            elif "Half" in str(e) and "float" in str(e):
                print("  Mixed precision issue detected")
                # Check model dtypes
                for name, param in backbone.named_parameters():
                    if param.dtype != torch.float32:
                        print(f"    {name}: {param.dtype}")
        
        # Clean up
        del x
        if 'outputs' in locals():
            del outputs
        torch.cuda.empty_cache()
        gc.collect()
    
    print("\nBackbone test completed.")


def create_dummy_batch(num_points, cfg):
    """Create a dummy batch for testing"""
    num_masks = 10
    
    batch = {
        'pt_coord': [np.random.randn(num_points, 3).astype(np.float32)],
        'feats': [np.random.randn(num_points, 4).astype(np.float32)],
        'sem_label': [np.random.randint(0, 20, (num_points, 1)).astype(np.int64)],
        'ins_label': [np.random.randint(0, 100, (num_points, 1)).astype(np.int64)],
        'masks': [torch.randint(0, 2, (num_masks, num_points), dtype=torch.float32)],
        'masks_cls': [torch.randint(0, 20, (num_masks,))],
        'masks_ids': [[torch.randint(0, min(100, num_points), (min(100, num_points),)) for _ in range(num_masks)]],
        'fname': ['dummy.bin'],
        'pose': [np.eye(4)],
        'token': ['0']
    }
    
    return batch


if __name__ == "__main__":
    test_memory()