# mask_pls/scripts/export_onnx_fixed_shapes.py
"""
ONNX converter using fixed shapes to avoid shape inference issues
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import onnx
import yaml
from pathlib import Path
from easydict import EasyDict as edict

# Force CPU mode
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.Tensor.cuda = lambda self, *args, **kwargs: self
torch.cuda.is_available = lambda: False

sys.path.append(str(Path(__file__).parent.parent))

from mask_pls.models.dgcnn.maskpls_dgcnn_optimized import MaskPLSDGCNNFixed


class FixedPointsWrapper(nn.Module):
    """Wrapper that ensures fixed number of points throughout"""
    
    def __init__(self, model, fixed_points=10000):
        super().__init__()
        self.model = model.cpu()
        self.fixed_points = fixed_points
        self.num_queries = model.decoder.num_queries
        self.num_classes = model.num_classes
        
    def forward(self, coords, feats):
        B, N, _ = coords.shape
        
        # Ensure we have exactly fixed_points
        if N != self.fixed_points:
            print(f"Warning: Expected {self.fixed_points} points, got {N}")
            # Pad or truncate
            if N < self.fixed_points:
                pad_n = self.fixed_points - N
                coords = F.pad(coords, (0, 0, 0, pad_n), value=0)
                feats = F.pad(feats, (0, 0, 0, pad_n), value=0)
            else:
                coords = coords[:, :self.fixed_points]
                feats = feats[:, :self.fixed_points]
        
        # Convert to expected format
        x = {
            'pt_coord': [coords[b].cpu().numpy() for b in range(B)],
            'feats': [feats[b].cpu().numpy() for b in range(B)]
        }
        
        # Forward through model
        outputs, padding, sem_logits = self.model(x)
        
        pred_logits = outputs['pred_logits']  # [B, Q, C+1]
        pred_masks = outputs['pred_masks']    # [B, N_backbone, Q]
        
        # Ensure outputs have fixed dimensions
        B_out = pred_logits.shape[0]
        Q = pred_logits.shape[1]
        C = pred_logits.shape[2]
        
        # Fix pred_masks to have exactly fixed_points
        N_mask = pred_masks.shape[1]
        if N_mask != self.fixed_points:
            if N_mask < self.fixed_points:
                pad_n = self.fixed_points - N_mask
                pred_masks = F.pad(pred_masks, (0, 0, 0, pad_n), value=0)
            else:
                pred_masks = pred_masks[:, :self.fixed_points, :]
        
        # Fix sem_logits similarly
        N_sem = sem_logits.shape[1]
        if N_sem != self.fixed_points:
            if N_sem < self.fixed_points:
                pad_n = self.fixed_points - N_sem
                sem_logits = F.pad(sem_logits, (0, 0, 0, pad_n), value=0)
            else:
                sem_logits = sem_logits[:, :self.fixed_points, :]
        
        # Ensure fixed output shapes
        pred_logits = pred_logits.reshape(B, Q, C)
        pred_masks = pred_masks.reshape(B, self.fixed_points, Q)
        sem_logits = sem_logits.reshape(B, self.fixed_points, self.num_classes)
        
        return pred_logits, pred_masks, sem_logits


def simplify_model(input_path, output_path):
    """Try to simplify the ONNX model"""
    try:
        import onnxsim
        print("\nSimplifying ONNX model...")
        
        model = onnx.load(input_path)
        model_simp, check = onnxsim.simplify(
            model,
            input_shapes={
                'coords': [1, 10000, 3],
                'features': [1, 10000, 4]
            },
            dynamic_input_shape=False
        )
        
        if check:
            onnx.save(model_simp, output_path)
            print("  ✓ Simplification successful")
            return True
        else:
            print("  ✗ Simplification check failed")
            return False
            
    except ImportError:
        print("  ⚠ onnx-simplifier not installed")
        print("    Install with: pip install onnx-simplifier")
        return False
    except Exception as e:
        print(f"  ✗ Simplification failed: {e}")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Export MaskPLS to ONNX with fixed shapes"
    )
    parser.add_argument('checkpoint', help='Path to checkpoint')
    parser.add_argument('--output', '-o', help='Output ONNX file')
    parser.add_argument('--batch-size', type=int, default=1, help='Fixed batch size')
    parser.add_argument('--num-points', type=int, default=10000, help='Fixed number of points')
    parser.add_argument('--simplify', action='store_true', help='Simplify model after export')
    
    args = parser.parse_args()
    
    output_path = args.output or Path(args.checkpoint).stem + "_fixed.onnx"
    
    print("="*60)
    print("Fixed Shape ONNX Export")
    print("="*60)
    print(f"Batch size: {args.batch_size} (fixed)")
    print(f"Num points: {args.num_points} (fixed)")
    
    # Load config
    config_dir = Path(__file__).parent.parent / "config"
    cfg = {}
    for cfg_file in ['model.yaml', 'backbone.yaml', 'decoder.yaml']:
        cfg_path = config_dir / cfg_file
        if cfg_path.exists():
            with open(cfg_path, 'r') as f:
                cfg.update(yaml.safe_load(f))
    cfg = edict(cfg)
    
    # Create model
    print(f"\n1. Loading model from: {args.checkpoint}")
    model = MaskPLSDGCNNFixed(cfg)
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model = model.cpu()
    
    # Create wrapper
    print("\n2. Creating fixed-shape wrapper...")
    wrapped_model = FixedPointsWrapper(model, fixed_points=args.num_points)
    wrapped_model.eval()
    
    # Create fixed-size inputs
    print(f"\n3. Creating fixed inputs...")
    dummy_coords = torch.randn(args.batch_size, args.num_points, 3, dtype=torch.float32)
    dummy_feats = torch.randn(args.batch_size, args.num_points, 4, dtype=torch.float32)
    
    # Test forward
    print("\n4. Testing forward pass...")
    try:
        with torch.no_grad():
            outputs = wrapped_model(dummy_coords, dummy_feats)
        print(f"   ✓ Success! Output shapes:")
        for i, out in enumerate(outputs):
            print(f"     Output {i}: {out.shape}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return
    
    # Export with fixed shapes (no dynamic axes)
    print(f"\n5. Exporting to: {output_path}")
    try:
        torch.onnx.export(
            wrapped_model,
            (dummy_coords, dummy_feats),
            output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['coords', 'features'],
            output_names=['pred_logits', 'pred_masks', 'sem_logits'],
            # NO dynamic axes - everything is fixed
            verbose=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX
        )
        print("   ✓ Export successful!")
        
    except Exception as e:
        print(f"   ✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Try with ATEN fallback
        print("\n   Trying with ATEN fallback...")
        try:
            torch.onnx.export(
                wrapped_model,
                (dummy_coords, dummy_feats),
                output_path,
                export_params=True,
                opset_version=17,
                input_names=['coords', 'features'],
                output_names=['pred_logits', 'pred_masks', 'sem_logits'],
                operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
            )
            print("   ✓ Export with ATEN fallback successful!")
        except:
            print("   ✗ ATEN fallback also failed")
            return
    
    # Verify basic structure
    print("\n6. Verifying ONNX model...")
    try:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("   ✓ Model structure valid")
        
        size_mb = Path(output_path).stat().st_size / 1024 / 1024
        print(f"   Size: {size_mb:.2f} MB")
        
    except Exception as e:
        print(f"   ✗ Verification failed: {e}")
    
    # Simplify if requested
    if args.simplify:
        simplified_path = output_path.replace('.onnx', '_simplified.onnx')
        if simplify_model(output_path, simplified_path):
            output_path = simplified_path
    
    # Test with ONNX Runtime
    print("\n7. Testing with ONNX Runtime...")
    try:
        import onnxruntime as ort
        
        # More detailed error info
        so = ort.SessionOptions()
        so.log_severity_level = 1
        
        try:
            session = ort.InferenceSession(output_path, so, providers=['CPUExecutionProvider'])
            print("   ✓ Model loaded successfully")
            
            # Print input/output info
            print("\n   Input shapes:")
            for inp in session.get_inputs():
                print(f"     {inp.name}: {inp.shape} ({inp.type})")
            
            print("\n   Output shapes:")
            for out in session.get_outputs():
                print(f"     {out.name}: {out.shape} ({out.type})")
            
            # Test inference
            test_outputs = session.run(None, {
                'coords': dummy_coords.numpy(),
                'features': dummy_feats.numpy()
            })
            
            print("\n   ✓ Inference successful!")
            
        except ort.capi.onnxruntime_pybind11_state.InvalidGraph as e:
            print(f"   ✗ Graph validation error: {e}")
            print("\n   This usually means there's a shape mismatch in the graph.")
            print("   Try using --simplify flag or different num-points value")
            
    except ImportError:
        print("   ⚠ ONNX Runtime not installed")
    except Exception as e:
        print(f"   ✗ Runtime error: {e}")
    
    print("\n✅ Done!")
    print(f"   Output: {output_path}")
    print(f"   Fixed shape: batch={args.batch_size}, points={args.num_points}")


if __name__ == "__main__":
    main()