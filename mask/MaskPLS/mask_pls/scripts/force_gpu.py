# mask_pls/scripts/export_onnx_force_cpu.py
"""
ONNX converter that forces everything to CPU
This avoids all device mismatch issues
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import onnx
import yaml
from pathlib import Path
from easydict import EasyDict as edict

# Force CPU mode before importing anything else
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Monkey patch cuda() to always return CPU tensors
torch.Tensor.cuda = lambda self, *args, **kwargs: self
torch.cuda.is_available = lambda: False

# Now import the model
from mask_pls.models.dgcnn.maskpls_dgcnn_optimized import MaskPLSDGCNNFixed


class CPUWrapper(nn.Module):
    """Wrapper that ensures everything runs on CPU"""
    
    def __init__(self, model):
        super().__init__()
        # Move model to CPU and keep it there
        self.model = model.cpu()
        
        # Ensure all submodules are on CPU
        for module in self.model.modules():
            module.to('cpu')
    
    def forward(self, coords, feats):
        # Ensure inputs are on CPU
        coords = coords.cpu()
        feats = feats.cpu()
        
        B = coords.shape[0]
        
        # Convert to expected format
        x = {
            'pt_coord': [coords[b].numpy() for b in range(B)],
            'feats': [feats[b].numpy() for b in range(B)]
        }
        
        # Forward through model
        outputs, _, sem_logits = self.model(x)
        
        # Ensure outputs are on CPU (they should already be)
        pred_logits = outputs['pred_logits'].cpu()
        pred_masks = outputs['pred_masks'].cpu()
        sem_logits = sem_logits.cpu()
        
        return pred_logits, pred_masks, sem_logits


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Export MaskPLS DGCNN to ONNX (CPU only)"
    )
    parser.add_argument('checkpoint', help='Path to checkpoint file')
    parser.add_argument('--output', '-o', help='Output ONNX file')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-points', type=int, default=10000)
    
    args = parser.parse_args()
    
    output_path = args.output or Path(args.checkpoint).stem + "_cpu.onnx"
    
    print("="*60)
    print("Force CPU ONNX Converter")
    print("="*60)
    print("CUDA disabled - using CPU only")
    
    # Load config
    print("\n1. Loading configuration...")
    config_dir = Path(__file__).parent.parent / "config"
    cfg = {}
    for cfg_file in ['model.yaml', 'backbone.yaml', 'decoder.yaml']:
        cfg_path = config_dir / cfg_file
        if cfg_path.exists():
            with open(cfg_path, 'r') as f:
                cfg.update(yaml.safe_load(f))
    cfg = edict(cfg)
    
    # Create model
    print(f"\n2. Loading model from: {args.checkpoint}")
    model = MaskPLSDGCNNFixed(cfg)
    
    # Load checkpoint - always map to CPU
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    # Load state dict
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # Force model to CPU
    model = model.cpu()
    
    # Double-check all parameters are on CPU
    for name, param in model.named_parameters():
        if param.device.type != 'cpu':
            print(f"   Warning: {name} not on CPU, moving...")
            param.data = param.data.cpu()
    
    # Create wrapper
    print("\n3. Creating CPU wrapper...")
    wrapped_model = CPUWrapper(model)
    wrapped_model.eval()
    
    # Test forward pass
    print("\n4. Testing forward pass...")
    test_coords = torch.randn(args.batch_size, args.num_points, 3, device='cpu')
    test_feats = torch.randn(args.batch_size, args.num_points, 4, device='cpu')
    
    try:
        with torch.no_grad():
            outputs = wrapped_model(test_coords, test_feats)
        print(f"   ✓ Success! Output shapes: {[o.shape for o in outputs]}")
        
        # Verify all outputs are on CPU
        for i, o in enumerate(outputs):
            if o.device.type != 'cpu':
                print(f"   Warning: Output {i} not on CPU!")
        
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Export to ONNX
    print(f"\n5. Exporting to: {output_path}")
    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapped_model,
                (test_coords, test_feats),
                output_path,
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=['coords', 'features'],
                output_names=['pred_logits', 'pred_masks', 'sem_logits'],
                dynamic_axes={
                    'coords': {0: 'batch', 1: 'points'},
                    'features': {0: 'batch', 1: 'points'},
                    'pred_logits': {0: 'batch'},
                    'pred_masks': {0: 'batch', 1: 'points'},
                    'sem_logits': {0: 'batch', 1: 'points'}
                },
                verbose=False
            )
        print("   ✓ Export successful!")
        
        # Verify
        print("\n6. Verifying ONNX model...")
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        size_mb = Path(output_path).stat().st_size / 1024 / 1024
        print(f"   ✓ Model verified! Size: {size_mb:.2f} MB")
        
        # Try ONNX Runtime
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
            
            # Test inference
            test_outputs = session.run(None, {
                'coords': test_coords.numpy(),
                'features': test_feats.numpy()
            })
            print(f"   ✓ ONNX Runtime test passed!")
            
        except ImportError:
            print("   ⚠ ONNX Runtime not installed (optional)")
        
        print("\n✅ Conversion completed successfully!")
        print("   (All operations forced to CPU)")
        
    except Exception as e:
        print(f"   ✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()