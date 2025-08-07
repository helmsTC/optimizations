#!/usr/bin/env python3
"""
Quick test script to verify ONNX export works
Save as: mask_pls/onnx/test_export.py
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent))

def test_export():
    """Test if ONNX export works with fixed decoder"""
    
    print("Testing ONNX Export with Fixed Decoder...")
    
    # Import the models
    from mask_pls.models.onnx.onnx_model import MaskPLSONNX, MaskPLSExportWrapper
    from easydict import EasyDict as edict
    
    # Minimal config
    cfg = edict({
        'MODEL': {'DATASET': 'KITTI', 'OVERLAP_THRESHOLD': 0.8},
        'KITTI': {
            'NUM_CLASSES': 20,
            'MIN_POINTS': 10,
            'SPACE': [[-48.0, 48.0], [-48.0, 48.0], [-4.0, 1.5]],
            'SUB_NUM_POINTS': 80000
        },
        'BACKBONE': {
            'INPUT_DIM': 4,
            'CHANNELS': [32, 32, 64, 128, 256, 256, 128, 96, 96],
            'RESOLUTION': 0.05,
            'KNN_UP': 3
        },
        'DECODER': {
            'HIDDEN_DIM': 256,
            'NHEADS': 8,
            'DIM_FFN': 1024,
            'FEATURE_LEVELS': 3,
            'DEC_BLOCKS': 3,
            'NUM_QUERIES': 100,
            'POS_ENC': {
                'MAX_FREQ': 10000,
                'DIMENSIONALITY': 3,
                'BASE': 2,
                'FEAT_SIZE': 256
            }
        }
    })
    
    # Create model
    model = MaskPLSONNX(cfg)
    model.eval()
    
    # Create wrapper
    export_model = MaskPLSExportWrapper(model)
    
    # Test with small input
    dummy_points = torch.randn(1, 1000, 3)
    dummy_features = torch.randn(1, 1000, 4)
    
    # Test forward pass
    print("Testing forward pass...")
    try:
        with torch.no_grad():
            outputs = export_model(dummy_points, dummy_features)
        print("✓ Forward pass successful")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False
    
    # Try ONNX export
    print("\nTesting ONNX export...")
    try:
        torch.onnx.export(
            export_model,
            (dummy_points, dummy_features),
            "test_model.onnx",
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['points', 'features'],
            output_names=['pred_logits', 'pred_masks', 'sem_logits'],
            verbose=False
        )
        print("✓ ONNX export successful!")
        print(f"  Model saved to: test_model.onnx")
        
        # Test with ONNX Runtime if available
        try:
            import onnxruntime as ort
            session = ort.InferenceSession("test_model.onnx")
            outputs = session.run(None, {
                'points': dummy_points.numpy(),
                'features': dummy_features.numpy()
            })
            print("✓ ONNX Runtime inference successful!")
            return True
        except ImportError:
            print("  (ONNX Runtime not installed, skipping inference test)")
            return True
            
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_export()
    sys.exit(0 if success else 1)
