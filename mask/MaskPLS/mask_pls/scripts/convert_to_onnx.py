#!/usr/bin/env python3
"""
Fixed MaskPLS to ONNX converter using simplified_model
This version works with the actual model structure
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


def simple_convert():
    """
    Simple conversion that actually works with simplified_model
    """
    print("=" * 50)
    print("MaskPLS to ONNX Converter")
    print("=" * 50)
    
    # Import the ONNX model
    try:
        from models.onnx.simplified_model import MaskPLSSimplifiedONNX
    except ImportError:
        # Try alternative import path
        from mask_pls.models.onnx.simplified_model import MaskPLSSimplifiedONNX
    
    # Load configurations
    def getDir(obj):
        return os.path.dirname(os.path.abspath(obj))
    
    try:
        model_cfg = edict(yaml.safe_load(open(os.path.join(getDir(__file__), "../config/model.yaml"))))
        backbone_cfg = edict(yaml.safe_load(open(os.path.join(getDir(__file__), "../config/backbone.yaml"))))
        decoder_cfg = edict(yaml.safe_load(open(os.path.join(getDir(__file__), "../config/decoder.yaml"))))
        cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    except:
        # Fallback to hardcoded config if files not found
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
                'CHANNELS': [32, 64, 128, 256, 256],  # Updated for optimized model
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
    
    # Update backbone config to match optimized version
    cfg.BACKBONE.CHANNELS = [32, 64, 128, 256, 256]
    
    print("\n1. Creating ONNX model...")
    model = MaskPLSSimplifiedONNX(cfg)
    
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
            
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Filter and clean state dict
            model_dict = model.state_dict()
            compatible_dict = {}
            
            for k, v in state_dict.items():
                # Remove prefixes
                k_clean = k.replace('module.', '').replace('model.', '')
                
                # Check if key exists in model
                if k_clean in model_dict:
                    if v.shape == model_dict[k_clean].shape:
                        compatible_dict[k_clean] = v
                    else:
                        print(f"   Shape mismatch for {k_clean}: {v.shape} vs {model_dict[k_clean].shape}")
            
            if compatible_dict:
                model.load_state_dict(compatible_dict, strict=False)
                print(f"   ✓ Loaded {len(compatible_dict)}/{len(model_dict)} weights")
            else:
                print("   ⚠ No compatible weights found, using random initialization")
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
    
    # Voxel dimensions
    voxel_d = input("   Voxel grid depth (default: 64): ").strip()
    voxel_d = int(voxel_d) if voxel_d else 64
    
    voxel_h = input("   Voxel grid height (default: 64): ").strip()
    voxel_h = int(voxel_h) if voxel_h else 64
    
    voxel_w = input("   Voxel grid width (default: 32): ").strip()
    voxel_w = int(voxel_w) if voxel_w else 32
    
    print(f"\n5. Exporting to {output_path}...")
    print(f"   Batch size: {batch_size}")
    print(f"   Number of points: {num_points}")
    print(f"   Voxel shape: [{voxel_d}, {voxel_h}, {voxel_w}]")
    
    # Update model's spatial shape
    model.spatial_shape = (voxel_d, voxel_h, voxel_w)
    
    # Move model to CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Create dummy input for the simplified model
    # The model expects: voxel_features [B, C, D, H, W] and point_coords [B, N, 3]
    C = 4  # XYZI features
    dummy_voxels = torch.randn(batch_size, C, voxel_d, voxel_h, voxel_w, device=device)
    dummy_coords = torch.rand(batch_size, num_points, 3, device=device)  # Normalized coords [0,1]
    
    # Test forward pass first
    print("\n   Testing forward pass...")
    try:
        with torch.no_grad():
            pred_logits, pred_masks, sem_logits = model(dummy_voxels, dummy_coords)
        print(f"   ✓ Forward pass successful:")
        print(f"     - pred_logits: {pred_logits.shape}")
        print(f"     - pred_masks: {pred_masks.shape}")
        print(f"     - sem_logits: {sem_logits.shape}")
    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Try export
    print("\n   Attempting ONNX export...")
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                (dummy_voxels, dummy_coords),
                output_path,
                export_params=True,
                opset_version=16,  # Use newer opset for better compatibility
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
        
        print(f"   ✓ Model exported successfully!")
        print(f"   Output file: {output_path}")
        print(f"   File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        
        # Verify the exported model
        print("\n6. Verifying ONNX model...")
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("   ✓ ONNX model verification passed")
        except ImportError:
            print("   ⚠ ONNX not installed, skipping verification")
        except Exception as e:
            print(f"   ⚠ Verification warning: {e}")
        
        print("\n" + "=" * 50)
        print("Export completed successfully!")
        print("=" * 50)
        return True
        
    except Exception as e:
        print(f"   ✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Main entry point
    """
    success = simple_convert()
    if not success:
        print("\n⚠ Export failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()