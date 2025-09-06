# mask_pls/scripts/export_onnx_no_modify.py
"""
ONNX converter that doesn't modify the original backbone
Uses runtime patching to handle CUDA dependencies
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
import types

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def create_patched_forward(original_forward, use_cuda=True):
    """Create a patched forward method that handles device placement"""
    
    def patched_forward(self, x):
        # Temporarily patch the backbone's methods
        coords_list = x['pt_coord']
        feats_list = x['feats']
        
        batch_size = len(coords_list)
        
        # Clear subsample tracking
        self.subsample_indices = {}
        
        # Process each point cloud
        all_features = []
        all_coords = []
        all_masks = []
        
        for b in range(batch_size):
            # Convert to tensors - handle device placement
            if use_cuda and torch.cuda.is_available():
                coords = torch.from_numpy(coords_list[b]).float().cuda()
                feats = torch.from_numpy(feats_list[b]).float().cuda()
            else:
                coords = torch.from_numpy(coords_list[b]).float()
                feats = torch.from_numpy(feats_list[b]).float()
            
            # Subsample if needed
            max_points = 50000 if self.training else 30000
            if coords.shape[0] > max_points:
                indices = torch.randperm(coords.shape[0], device=coords.device)[:max_points]
                indices = indices.sort()[0]
                coords = coords[indices]
                feats = feats[indices]
                self.subsample_indices[b] = indices
            else:
                self.subsample_indices[b] = torch.arange(coords.shape[0], device=coords.device)
            
            # Process through DGCNN
            point_features = self.process_single_cloud(coords, feats)
            
            all_features.append(point_features)
            all_coords.append(coords)
            all_masks.append(torch.zeros(coords.shape[0], dtype=torch.bool, device=coords.device))
        
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
    
    return patched_forward


class ONNXExportWrapper(nn.Module):
    """Wrapper that temporarily patches the model for export"""
    
    def __init__(self, model, use_cuda=False):
        super().__init__()
        self.model = model
        self.use_cuda = use_cuda
        
        # Save original methods
        self._original_backbone_forward = self.model.backbone.forward
        
        # Apply temporary patch
        self._apply_patch()
        
    def _apply_patch(self):
        """Apply temporary patches for ONNX export"""
        # Patch the backbone forward method
        patched_forward = create_patched_forward(
            self._original_backbone_forward, 
            self.use_cuda
        )
        self.model.backbone.forward = types.MethodType(patched_forward, self.model.backbone)
        
    def _restore_original(self):
        """Restore original methods"""
        self.model.backbone.forward = self._original_backbone_forward
        
    def forward(self, coords, feats):
        B = coords.shape[0]
        
        # Convert to expected format
        x = {
            'pt_coord': [coords[b].detach().cpu().numpy() for b in range(B)],
            'feats': [feats[b].detach().cpu().numpy() for b in range(B)]
        }
        
        # Forward through patched model
        outputs, _, sem_logits = self.model(x)
        
        return outputs['pred_logits'], outputs['pred_masks'], sem_logits
    
    def __del__(self):
        """Restore original methods when wrapper is deleted"""
        if hasattr(self, '_original_backbone_forward'):
            self._restore_original()


def load_config():
    """Load configuration"""
    config_dir = Path(__file__).parent.parent / "config"
    cfg = {}
    for cfg_file in ['model.yaml', 'backbone.yaml', 'decoder.yaml']:
        cfg_path = config_dir / cfg_file
        if cfg_path.exists():
            with open(cfg_path, 'r') as f:
                cfg.update(yaml.safe_load(f))
    return edict(cfg)


def convert_checkpoint_to_onnx(checkpoint_path, output_path=None, 
                              batch_size=1, num_points=10000):
    """Main conversion function"""
    
    if output_path is None:
        output_path = Path(checkpoint_path).stem + ".onnx"
    
    print("="*60)
    print("ONNX Export (No Backbone Modification)")
    print("="*60)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    device = 'cuda' if cuda_available else 'cpu'
    print(f"\nDevice status: {device}")
    
    # Load config
    print("\n1. Loading configuration...")
    cfg = load_config()
    
    # Import model AFTER setting up environment
    from mask_pls.models.dgcnn.maskpls_dgcnn_optimized import MaskPLSDGCNNFixed
    
    # Create model
    print(f"\n2. Loading model from: {checkpoint_path}")
    model = MaskPLSDGCNNFixed(cfg)
    
    # Load checkpoint
    map_location = device if not cuda_available else None
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # Move model to device
    model = model.to(device)
    
    # Create wrapper with patching
    print("\n3. Creating patched wrapper for export...")
    wrapped_model = ONNXExportWrapper(model, use_cuda=cuda_available)
    wrapped_model.eval()
    
    # For ONNX export, we need to use CPU
    export_device = 'cpu'
    if device == 'cuda':
        print("   Moving to CPU for ONNX export...")
        wrapped_model = wrapped_model.cpu()
        # Re-create wrapper for CPU
        model = model.cpu()
        wrapped_model = ONNXExportWrapper(model, use_cuda=False)
        wrapped_model.eval()
    
    # Test forward pass
    print("\n4. Testing forward pass...")
    test_coords = torch.randn(batch_size, num_points, 3, device=export_device)
    test_feats = torch.randn(batch_size, num_points, 4, device=export_device)
    
    try:
        with torch.no_grad():
            outputs = wrapped_model(test_coords, test_feats)
        print(f"   ✓ Success! Output shapes: {[o.shape for o in outputs]}")
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
        
        # Try ONNX Runtime if available
        try:
            import onnxruntime as ort
            providers = ['CPUExecutionProvider']
            if cuda_available:
                providers.insert(0, 'CUDAExecutionProvider')
            
            session = ort.InferenceSession(output_path, providers=providers)
            print(f"   ✓ ONNX Runtime loaded successfully")
            print(f"     Providers: {session.get_providers()}")
            
            # Test inference
            test_outputs = session.run(None, {
                'coords': test_coords.cpu().numpy(),
                'features': test_feats.cpu().numpy()
            })
            print(f"     Inference test passed!")
            
        except ImportError:
            print("   ⚠ ONNX Runtime not installed (optional)")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # The wrapper will restore original methods when deleted
        del wrapped_model
        print("\n   Original model restored (no modifications)")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Export MaskPLS DGCNN to ONNX without modifying backbone"
    )
    parser.add_argument('checkpoint', help='Path to checkpoint file')
    parser.add_argument('--output', '-o', help='Output ONNX file')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--num-points', type=int, default=10000, help='Number of points')
    
    args = parser.parse_args()
    
    success = convert_checkpoint_to_onnx(
        args.checkpoint,
        args.output,
        args.batch_size,
        args.num_points
    )
    
    if success:
        print("\n✅ Conversion completed successfully!")
        print("   Your backbone file was NOT modified.")
    else:
        print("\n❌ Conversion failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()