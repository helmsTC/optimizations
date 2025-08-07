#!/usr/bin/env python3
"""
Fixed MaskPLS to ONNX converter
This version works with the corrected decoder
Replace mask_pls/onnx/convert_to_onnx.py with this
"""

import os
import sys
import torch
import torch.onnx
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def simple_convert():
    """
    Simple conversion that actually works
    """
    print("=" * 50)
    print("MaskPLS to ONNX Converter")
    print("=" * 50)
    
    # Import the ONNX model
    try:
        from models.onnx.onnx_model import MaskPLSONNX, MaskPLSExportWrapper
    except ImportError:
        # Try alternative import path
        from mask_pls.models.onnx.onnx_model import MaskPLSONNX, MaskPLSExportWrapper
    
    # Create config
    from easydict import EasyDict as edict
    
    cfg = edict({
        'MODEL': {
            'DATASET': 'KITTI',
            'OVERLAP_THRESHOLD': 0.8
        },
        'KITTI': {
            'NUM_CLASSES': 20,
            'MIN_POINTS': 10,
            'SPACE': [[-48.0, 48.0], [-48.0, 48.0], [-4.0, 1.5]],
            'SUB_NUM_POINTS': 80000,
            'IGNORE_LABEL': 0
        },
        'NUSCENES': {
            'NUM_CLASSES': 17,
            'MIN_POINTS': 10,
            'SPACE': [[-50.0, 50.0], [-50.0, 50.0], [-5.0, 3.0]],
            'SUB_NUM_POINTS': 50000,
            'IGNORE_LABEL': 0
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
                'BASE': 2
            }
        }
    })
    
    # Ask for dataset type
    dataset_choice = input("\nDataset type (1=KITTI, 2=NUSCENES) [default=1]: ").strip()
    if dataset_choice == "2":
        cfg.MODEL.DATASET = 'NUSCENES'
        print("Using NuScenes configuration")
    else:
        cfg.MODEL.DATASET = 'KITTI'
        print("Using KITTI configuration")
    
    print("\n1. Creating ONNX model...")
    model = MaskPLSONNX(cfg)
    
    print("\n2. Setting to eval mode...")
    model.eval()
    
    # Disable gradients
    for param in model.parameters():
        param.requires_grad = False
    
    # Optional: Load checkpoint weights if available
    checkpoint_path = input("\n3. Enter checkpoint path (or press Enter to skip): ").strip()
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"   Loading weights from {checkpoint_path}...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            
            # Filter compatible weights
            model_dict = model.state_dict()
            compatible_dict = {}
            
            for k, v in state_dict.items():
                k_new = k.replace('module.', '')
                # Skip MinkowskiEngine specific weights
                if 'kernel' not in k and 'MinkowskiConvolution' not in k:
                    if k_new in model_dict:
                        if v.shape == model_dict[k_new].shape:
                            compatible_dict[k_new] = v
            
            model.load_state_dict(compatible_dict, strict=False)
            print(f"   ✓ Loaded {len(compatible_dict)}/{len(model_dict)} weights")
        except Exception as e:
            print(f"   ⚠ Warning: Could not load weights: {e}")
            print("   Continuing with random weights...")
    else:
        print("   Using random initialization...")
    
    # Export to ONNX
    output_path = input("\n4. Enter output path (default: mask_pls.onnx): ").strip()
    if not output_path:
        output_path = "mask_pls.onnx"
    
    # Export parameters
    batch_size = input("   Batch size for export (default: 1): ").strip()
    batch_size = int(batch_size) if batch_size else 1
    
    num_points = input("   Number of points for export (default: 10000): ").strip()
    num_points = int(num_points) if num_points else 10000
    
    print(f"\n5. Exporting to {output_path}...")
    print(f"   Batch size: {batch_size}")
    print(f"   Number of points: {num_points}")
    
    # Create wrapper for clean export
    export_model = MaskPLSExportWrapper(model)
    
    # Create dummy input
    dummy_points = torch.randn(batch_size, num_points, 3)
    dummy_features = torch.randn(batch_size, num_points, 4)
    
    # Test forward pass first
    print("\n   Testing forward pass...")
    try:
        with torch.no_grad():
            test_output = export_model(dummy_points, dummy_features)
        print(f"   ✓ Forward pass successful: {[o.shape for o in test_output]}")
    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Try export
    print("\n   Attempting ONNX export...")
    try:
        with torch.no_grad():
            # Try with simpler settings first
            torch.onnx.export(
                export_model,
                (dummy_points, dummy_features),
                output_path,
                export_params=True,
                opset_version=11,  # Most compatible
                do_constant_folding=True,
                input_names=['points', 'features'],
                output_names=['pred_logits', 'pred_masks', 'sem_logits'],
                dynamic_axes={
                    'points': {0: 'batch_size', 1: 'num_points'},
                    'features': {0: 'batch_size', 1: 'num_points'},
                    'pred_logits': {0: 'batch_size'},
                    'pred_masks': {0: 'batch_size', 1: 'num_points'},
                    'sem_logits': {0: 'batch_size', 1: 'num_points'}
                },
                verbose=False
            )
        
        print(f"   ✓ Model exported successfully!")
        print(f"   Output file: {output_path}")
        print(f"   File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"   ✗ Export failed: {e}")
        print("\n   Trying alternative export method...")
        
        # Try tracing instead
        try:
            with torch.no_grad():
                traced = torch.jit.trace(export_model, (dummy_points, dummy_features), check_trace=False)
                torch.onnx.export(
                    traced,
                    (dummy_points, dummy_features),
                    output_path,
                    opset_version=11,
                    input_names=['points', 'features'],
                    output_names=['pred_logits', 'pred_masks', 'sem_logits']
                )
            print(f"   ✓ Export successful with tracing!")
        except Exception as e2:
            print(f"   ✗ Tracing also failed: {e2}")
            return False
    
    # Test with ONNX Runtime if available
    try:
        import onnxruntime as ort
        
        print("\n6. Testing with ONNX Runtime...")
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(output_path, providers=providers)
        
        print(f"   Using providers: {session.get_providers()}")
        
        # Test inference
        test_points = np.random.randn(1, 1000, 3).astype(np.float32)
        test_features = np.random.randn(1, 1000, 4).astype(np.float32)
        
        outputs = session.run(None, {
            'points': test_points,
            'features': test_features
        })
        
        print(f"   ✓ Inference test passed!")
        print(f"   Output shapes: {[o.shape for o in outputs]}")
        
    except ImportError:
        print("\n6. ONNX Runtime not installed. Cannot test inference.")
        print("   Install with: pip install onnxruntime-gpu")
    except Exception as e:
        print(f"\n6. Inference test error: {e}")
    
    print("\n" + "=" * 50)
    print("Conversion Complete!")
    print("=" * 50)
    
    return True


if __name__ == "__main__":
    success = simple_convert()
    sys.exit(0 if success else 1)
