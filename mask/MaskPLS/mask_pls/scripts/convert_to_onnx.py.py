# mask_pls/scripts/export_onnx_fixed_device.py
"""
ONNX converter with proper device handling
Ensures all components are on the same device
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

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


class DeviceConsistentModel(nn.Module):
    """Wrapper that ensures device consistency"""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.device = next(model.parameters()).device
        
        # Store original method
        self._original_forward = model.backbone.forward.__func__
        
        # Patch the backbone
        self._patch_backbone()
    
    def _patch_backbone(self):
        """Patch backbone to handle device consistently"""
        model = self.model
        device = self.device
        
        # Define a new forward that handles devices properly
        def consistent_forward(self, x):
            coords_list = x['pt_coord']
            feats_list = x['feats']
            
            batch_size = len(coords_list)
            self.subsample_indices = {}
            
            all_features = []
            all_coords = []
            all_masks = []
            
            for b in range(batch_size):
                # Convert to tensors on the model's device
                coords = torch.from_numpy(coords_list[b]).float().to(device)
                feats = torch.from_numpy(feats_list[b]).float().to(device)
                
                # Subsample if needed
                max_points = 50000 if self.training else 30000
                if coords.shape[0] > max_points:
                    indices = torch.randperm(coords.shape[0], device=device)[:max_points]
                    indices = indices.sort()[0]
                    coords = coords[indices]
                    feats = feats[indices]
                    self.subsample_indices[b] = indices
                else:
                    self.subsample_indices[b] = torch.arange(coords.shape[0], device=device)
                
                # Process through DGCNN
                point_features = self.process_single_cloud(coords, feats)
                
                all_features.append(point_features)
                all_coords.append(coords)
                all_masks.append(torch.zeros(coords.shape[0], dtype=torch.bool, device=device))
            
            # Generate multi-scale features with padding
            ms_features = []
            ms_coords = []
            ms_masks = []
            
            for i in range(len(self.feat_layers)):
                level_features, level_coords, level_masks = self.pad_batch_level(
                    [f[i] for f in all_features],
                    all_coords,
                    all_masks
                )
                ms_features.append(level_features)
                ms_coords.append(level_coords)
                ms_masks.append(level_masks)
            
            # Semantic predictions
            sem_logits = self.compute_semantic_logits(ms_features[-1], ms_masks[-1])
            
            return ms_features, ms_coords, ms_masks, sem_logits
        
        # Replace the method
        model.backbone.forward = consistent_forward.__get__(model.backbone, model.backbone.__class__)
    
    def to(self, device):
        """Override to() to update our device tracking"""
        self.device = device
        self.model = self.model.to(device)
        self._patch_backbone()  # Re-patch with new device
        return super().to(device)
    
    def forward(self, coords, feats):
        # Ensure inputs are on correct device
        coords = coords.to(self.device)
        feats = feats.to(self.device)
        
        B = coords.shape[0]
        
        # Convert to expected format
        x = {
            'pt_coord': [coords[b].detach().cpu().numpy() for b in range(B)],
            'feats': [feats[b].detach().cpu().numpy() for b in range(B)]
        }
        
        # Forward through model
        outputs, _, sem_logits = self.model(x)
        
        # Ensure outputs are on correct device
        outputs['pred_logits'] = outputs['pred_logits'].to(self.device)
        outputs['pred_masks'] = outputs['pred_masks'].to(self.device)
        sem_logits = sem_logits.to(self.device)
        
        return outputs['pred_logits'], outputs['pred_masks'], sem_logits


def convert_to_onnx(checkpoint_path, output_path=None, batch_size=1, num_points=10000):
    """Main conversion function"""
    
    if output_path is None:
        output_path = Path(checkpoint_path).stem + ".onnx"
    
    print("="*60)
    print("ONNX Export with Fixed Device Handling")
    print("="*60)
    
    # Import model
    from mask_pls.models.dgcnn.maskpls_dgcnn_optimized import MaskPLSDGCNNFixed
    
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
    print(f"\n2. Loading model from: {checkpoint_path}")
    model = MaskPLSDGCNNFixed(cfg)
    
    # Load checkpoint on CPU first
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Initial device: {device}")
    
    # For ONNX export, always use CPU to avoid issues
    print("\n3. Setting up for ONNX export (CPU mode)...")
    model = model.cpu()
    
    # Create wrapper
    wrapped_model = DeviceConsistentModel(model)
    wrapped_model.eval()
    wrapped_model = wrapped_model.cpu()  # Ensure everything is on CPU
    
    # Test forward pass
    print("\n4. Testing forward pass...")
    test_coords = torch.randn(batch_size, num_points, 3, device='cpu')
    test_feats = torch.randn(batch_size, num_points, 4, device='cpu')
    
    try:
        with torch.no_grad():
            outputs = wrapped_model(test_coords, test_feats)
        print(f"   ✓ Success! Output shapes: {[o.shape for o in outputs]}")
        print(f"   Output devices: {[o.device for o in outputs]}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
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
        
        # Test with ONNX Runtime if available
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
            
            # Test inference
            test_outputs = session.run(None, {
                'coords': test_coords.numpy(),
                'features': test_feats.numpy()
            })
            print(f"   ✓ ONNX Runtime test passed!")
            print(f"     Output shapes: {[o.shape for o in test_outputs]}")
            
        except ImportError:
            print("   ⚠ ONNX Runtime not installed (optional)")
        except Exception as e:
            print(f"   ⚠ ONNX Runtime test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Export MaskPLS DGCNN to ONNX with proper device handling"
    )
    parser.add_argument('checkpoint', help='Path to checkpoint file')
    parser.add_argument('--output', '-o', help='Output ONNX file')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-points', type=int, default=10000)
    
    args = parser.parse_args()
    
    success = convert_to_onnx(
        args.checkpoint,
        args.output,
        args.batch_size,
        args.num_points
    )
    
    if success:
        print("\n✅ Conversion completed successfully!")
        print("   No source files were modified.")
    else:
        print("\n❌ Conversion failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()