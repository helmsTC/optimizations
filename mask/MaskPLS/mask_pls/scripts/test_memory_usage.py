# Save this as: mask/MaskPLS/mask_pls/scripts/test_memory_usage.py
"""
Test memory usage of the optimized model
"""

import torch
import numpy as np
import gc
from pathlib import Path
import yaml
from easydict import EasyDict as edict
import os

def get_gpu_memory_usage():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0

def test_model_memory():
    """Test memory usage with different configurations"""
    
    # Load config
    def getDir(obj):
        return os.path.dirname(os.path.abspath(obj))
    
    model_cfg = edict(yaml.safe_load(open(os.path.join(getDir(__file__), "../config/model.yaml"))))
    backbone_cfg = edict(yaml.safe_load(open(os.path.join(getDir(__file__), "../config/backbone.yaml"))))
    decoder_cfg = edict(yaml.safe_load(open(os.path.join(getDir(__file__), "../config/decoder.yaml"))))
    
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    cfg.CHUNK_SIZE = 10000
    cfg.VAL_BATCH_SIZE = 1
    cfg.MODEL.DATASET = "KITTI"
    
    print("Testing memory usage...")
    print(f"Initial GPU memory: {get_gpu_memory_usage():.2f} MB")
    
    # Import models
    from mask_pls.models.dgcnn.dgcnn_backbone import DGCNNBackbone
    from mask_pls.models.dgcnn.dgcnn_backbone_efficient import EfficientDGCNNBackbone
    
    # Test different point cloud sizes
    point_cloud_sizes = [10000, 50000, 80000, 100000]
    
    print("\n1. Testing original DGCNN backbone:")
    for num_points in point_cloud_sizes:
        try:
            # Clear memory
            torch.cuda.empty_cache()
            gc.collect()
            
            # Create model
            model = DGCNNBackbone(cfg.BACKBONE).cuda()
            model.eval()
            
            # Create dummy input
            batch = {
                'pt_coord': [np.random.randn(num_points, 3).astype(np.float32)],
                'feats': [np.random.randn(num_points, 4).astype(np.float32)]
            }
            
            # Measure memory before
            mem_before = get_gpu_memory_usage()
            
            # Forward pass
            with torch.no_grad():
                _ = model(batch)
            
            # Measure memory after
            mem_after = get_gpu_memory_usage()
            
            print(f"  {num_points} points: {mem_after - mem_before:.2f} MB used")
            
            # Cleanup
            del model
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  {num_points} points: OOM!")
                torch.cuda.empty_cache()
            else:
                raise e
    
    print("\n2. Testing efficient DGCNN backbone:")
    for num_points in point_cloud_sizes:
        try:
            # Clear memory
            torch.cuda.empty_cache()
            gc.collect()
            
            # Create model
            model = EfficientDGCNNBackbone(cfg.BACKBONE).cuda()
            model.eval()
            
            # Create dummy input
            batch = {
                'pt_coord': [np.random.randn(num_points, 3).astype(np.float32)],
                'feats': [np.random.randn(num_points, 4).astype(np.float32)]
            }
            
            # Measure memory before
            mem_before = get_gpu_memory_usage()
            
            # Forward pass
            with torch.no_grad():
                _ = model(batch)
            
            # Measure memory after
            mem_after = get_gpu_memory_usage()
            
            print(f"  {num_points} points: {mem_after - mem_before:.2f} MB used")
            
            # Cleanup
            del model
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  {num_points} points: OOM!")
                torch.cuda.empty_cache()
            else:
                raise e
    
    print("\n3. Testing full model with validation:")
    from mask_pls.models.dgcnn.maskpls_dgcnn_optimized import MaskPLSDGCNNOptimized
    
    # Create full model
    model = MaskPLSDGCNNOptimized(cfg).cuda()
    model.eval()
    model.things_ids = [1, 2, 3, 4, 5, 6, 7, 8]
    
    for num_points in [10000, 50000, 80000]:
        try:
            # Clear memory
            torch.cuda.empty_cache()
            gc.collect()
            
            # Create dummy batch
            batch = {
                'pt_coord': [np.random.randn(num_points, 3).astype(np.float32)],
                'feats': [np.random.randn(num_points, 4).astype(np.float32)],
                'sem_label': [np.random.randint(0, 20, (num_points, 1))],
                'ins_label': [np.random.randint(0, 100, (num_points, 1))],
                'masks': [torch.rand(5, num_points)],
                'masks_cls': [torch.randint(0, 20, (5,))],
                'masks_ids': [[torch.arange(100) for _ in range(5)]],
                'fname': ['test.bin'],
                'pose': [np.eye(4)],
                'token': ['0']
            }
            
            # Measure memory before
            mem_before = get_gpu_memory_usage()
            
            # Forward pass (validation)
            with torch.no_grad():
                outputs, padding, sem_logits = model(batch)
                
                # Test panoptic inference
                sem_pred, ins_pred = model.panoptic_inference(outputs, padding)
            
            # Measure memory after
            mem_after = get_gpu_memory_usage()
            
            print(f"  {num_points} points: {mem_after - mem_before:.2f} MB used")
            
            # Cleanup
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  {num_points} points: OOM!")
                torch.cuda.empty_cache()
            else:
                raise e
    
    print("\nMemory test completed!")


if __name__ == "__main__":
    test_model_memory()