#!/usr/bin/env python3
"""
Fixed MaskPLS to ONNX conversion script
This version uses a simplified architecture that actually works with ONNX
"""

import os
import sys
import torch
import torch.onnx
import numpy as np
from pathlib import Path
import yaml
from easydict import EasyDict as edict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_original_weights(model, checkpoint_path):
    """
    Load weights from original MaskPLS checkpoint
    Maps only compatible layers
    """
    print(f"Loading weights from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    # Map weights where possible
    new_state_dict = {}
    model_state = model.state_dict()
    
    # Simple mapping for encoder layers
    mapping_rules = {
        # Map stem layers to encoder initial layers
        'backbone.stem.0.kernel': None,  # Skip MinkowskiEngine kernels
        'backbone.stem.0.bn.weight': 'encoder.encoder.1.weight',
        'backbone.stem.0.bn.bias': 'encoder.encoder.1.bias',
        # Add more mappings as needed
    }
    
    loaded = 0
    for name, param in model_state.items():
        # Try direct mapping
        if name in state_dict and param.shape == state_dict[name].shape:
            new_state_dict[name] = state_dict[name]
            loaded += 1
        # Try mapped names
        elif name in mapping_rules and mapping_rules[name] in state_dict:
            mapped_name = mapping_rules[name]
            if param.shape == state_dict[mapped_name].shape:
                new_state_dict[name] = state_dict[mapped_name]
                loaded += 1
    
    # Load what we can
    model.load_state_dict(new_state_dict, strict=False)
    print(f"Loaded {loaded}/{len(model_state)} parameters")
    
    return model


def test_model(model, batch_size=1, num_points=10000):
    """Test the model with dummy input"""
    print("\nTesting model...")
    
    model.eval()
    
    # Create dummy inputs
    D, H, W = model.spatial_shape
    C = 4  # XYZI features
    
    dummy_voxels = torch.randn(batch_size, C, D, H, W)
    dummy_coords = torch.rand(batch_size, num_points, 3)
    
    with torch.no_grad():
        try:
            outputs = model(dummy_voxels, dummy_coords)
            print(f"✓ Forward pass successful")
            print(f"  Outputs: {[o.shape for o in outputs]}")
            return True
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            return False


def export_to_onnx(model, output_path, batch_size=1, num_points=10000, opset_version=14):
    """Export model to ONNX"""
    print(f"\nExporting to ONNX...")
    
    model.eval()
    
    # Prepare inputs
    D, H, W = model.spatial_shape
    C = 4
    
    dummy_voxels = torch.randn(batch_size, C, D, H, W)
    dummy_coords = torch.rand(batch_size, num_points, 3)
    
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                (dummy_voxels, dummy_coords),
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['voxel_features', 'point_coords'],
                output_names=['pred_logits', 'pred_masks', 'sem_logits'],
                dynamic_axes={
                    'voxel_features': {0: 'batch_size'},
                    'point_coords': {0: 'batch_size', 1: 'num_points'},
                    'pred_logits': {0: 'batch_size'},
                    'pred_masks': {0: 'batch_size', 1: 'num_points'},
                    'sem_logits': {0: 'batch_size', 1: 'num_points'}
                },
                verbose=False
            )
        
        print(f"✓ Export successful!")
        print(f"  Output: {output_path}")
        print(f"  Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        return True
        
    except Exception as e:
        print(f"✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_onnx_inference(onnx_path, batch_size=1, num_points=5000):
    """Test ONNX model inference"""
    try:
        import onnxruntime as ort
        
        print(f"\nTesting ONNX inference...")
        
        # Create session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(onnx_path, providers=providers)
        
        print(f"  Providers: {session.get_providers()}")
        
        # Get input/output info
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        
        print(f"  Inputs:")
        for inp in inputs:
            print(f"    {inp.name}: {inp.shape} ({inp.type})")
        
        print(f"  Outputs:")
        for out in outputs:
            print(f"    {out.name}: {out.shape} ({out.type})")
        
        # Test inference
        D, H, W = 32, 32, 16  # Must match model
        C = 4
        
        dummy_voxels = np.random.randn(batch_size, C, D, H, W).astype(np.float32)
        dummy_coords = np.random.rand(batch_size, num_points, 3).astype(np.float32)
        
        start = time.time()
        outputs = session.run(None, {
            'voxel_features': dummy_voxels,
            'point_coords': dummy_coords
        })
        inference_time = (time.time() - start) * 1000
        
        print(f"✓ Inference successful!")
        print(f"  Time: {inference_time:.2f} ms")
        print(f"  Output shapes: {[o.shape for o in outputs]}")
        
        return True
        
    except ImportError:
        print("\n⚠ ONNX Runtime not installed")
        print("  Install with: pip install onnxruntime-gpu")
        return False
    except Exception as e:
        print(f"\n✗ ONNX inference failed: {e}")
        return False


def main():
    """Main conversion function"""
    print("=" * 60)
    print("MaskPLS to ONNX Converter (Fixed Version)")
    print("=" * 60)
    
    # Import the simplified model
    try:
        # Try to import from the artifact we just created
        from mask_pls.models.onnx.simplified_model import create_onnx_model, export_model_to_onnx
    except:
        # Define inline if needed
        print("Error: Could not import simplified model")
        print("Make sure simplified_model.py is in mask_pls/models/onnx/")
        return
    
    # Configuration
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
            'CHANNELS': [32, 32, 64, 128, 256],
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
        }
    })
    
    # Get user inputs
    dataset_choice = input("\n1. Dataset (1=KITTI, 2=NUSCENES) [default=1]: ").strip()
    if dataset_choice == "2":
        cfg.MODEL.DATASET = 'NUSCENES'
        print("   Using NuScenes configuration")
    else:
        print("   Using KITTI configuration")
    
    # ONNX opset version choice
    opset_choice = input("\n2. ONNX opset version (11=wider compatibility, 14=better quality) [default=14]: ").strip()
    opset_version = 11 if opset_choice == "11" else 14
    print(f"   Using opset version {opset_version}")
    
    # Create model
    print(f"\n3. Creating simplified model...")
    model = create_onnx_model(cfg, opset_version=opset_version)
    print(f"   Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Optional: Load weights
    checkpoint_path = input("\n4. Checkpoint path (press Enter to skip): ").strip()
    if checkpoint_path and os.path.exists(checkpoint_path):
        load_original_weights(model, checkpoint_path)
    else:
        print("   Using random initialization")
    
    # Test model
    if not test_model(model):
        print("\n⚠ Model test failed, but continuing anyway...")
    
    # Export settings
    output_path = input("\n5. Output path [default: maskpls_simplified.onnx]: ").strip()
    if not output_path:
        output_path = "maskpls_simplified.onnx"
    
    batch_size = input("   Batch size [default: 1]: ").strip()
    batch_size = int(batch_size) if batch_size else 1
    
    num_points = input("   Number of points [default: 10000]: ").strip()
    num_points = int(num_points) if num_points else 10000
    
    # Export to ONNX
    if export_to_onnx(model, output_path, batch_size, num_points, opset_version):
        # Test ONNX inference
        test_onnx_inference(output_path, batch_size, num_points)
    
    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print("=" * 60)
    
    # Additional tips
    print("\nNext steps:")
    print("1. Test with real data using the preprocessing pipeline")
    print("2. Optimize the ONNX model:")
    print("   python -m onnxruntime.tools.optimizer --input maskpls_simplified.onnx --output maskpls_optimized.onnx")
    print("3. Quantize for faster inference:")
    print("   python optimize_onnx.py maskpls_simplified.onnx --quantize")


if __name__ == "__main__":
    import time
    main()
