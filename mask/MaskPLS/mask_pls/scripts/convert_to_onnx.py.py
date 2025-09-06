# mask_pls/scripts/convert_onnx_fixed.py
"""
Fixed ONNX converter that properly handles device placement
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import onnx
import yaml
import click
from pathlib import Path
from easydict import EasyDict as edict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from mask_pls.models.dgcnn.maskpls_dgcnn_optimized import MaskPLSDGCNNFixed
from mask_pls.models.dgcnn.dgcnn_backbone_efficient import FixedDGCNNBackbone


class CPUWrapper(nn.Module):
    """
    Wrapper that ensures all operations happen on CPU to avoid device conflicts
    """
    def __init__(self, model):
        super().__init__()
        # Move entire model to CPU
        self.model = model.cpu()
        self.num_classes = model.num_classes
        
    def forward(self, coords, feats):
        """
        Forward pass with explicit CPU operations
        
        Args:
            coords: [B, N, 3] point coordinates (CPU tensor)
            feats: [B, N, 4] point features (CPU tensor)
        """
        # Ensure inputs are on CPU
        coords = coords.cpu()
        feats = feats.cpu()
        
        B, N, _ = coords.shape
        
        # Create input dict for the model
        x = {
            'pt_coord': [],
            'feats': []
        }
        
        for b in range(B):
            # Convert to numpy arrays (always on CPU)
            coords_np = coords[b].detach().numpy()
            feats_np = feats[b].detach().numpy()
            
            x['pt_coord'].append(coords_np)
            x['feats'].append(feats_np)
        
        # Forward through model (all operations on CPU)
        with torch.no_grad():
            outputs, padding, sem_logits = self.model(x)
        
        # Ensure outputs are on CPU
        pred_logits = outputs['pred_logits'].cpu()
        pred_masks = outputs['pred_masks'].cpu()
        sem_logits = sem_logits.cpu()
        
        return pred_logits, pred_masks, sem_logits


def load_config(config_dir=None):
    """Load configuration"""
    if config_dir is None:
        config_dir = Path(__file__).parent.parent / "config"
    
    cfg = {}
    for cfg_file in ['model.yaml', 'backbone.yaml', 'decoder.yaml']:
        cfg_path = config_dir / cfg_file
        if cfg_path.exists():
            with open(cfg_path, 'r') as f:
                cfg.update(yaml.safe_load(f))
    
    return edict(cfg)


@click.command()
@click.option('--checkpoint', '-c', required=True, help='Path to checkpoint')
@click.option('--output', '-o', help='Output ONNX path')
@click.option('--batch-size', default=1, help='Batch size')
@click.option('--num-points', default=10000, help='Number of points')
@click.option('--dataset', default='KITTI', type=click.Choice(['KITTI', 'NUSCENES']))
def convert(checkpoint, output, batch_size, num_points, dataset):
    """Convert MaskPLS checkpoint to ONNX with proper device handling"""
    
    print("="*60)
    print("MaskPLS DGCNN to ONNX Converter (Fixed)")
    print("="*60)
    
    # Set output path
    if not output:
        output = Path(checkpoint).stem + "_fixed.onnx"
    
    # Force CPU mode to avoid device conflicts
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    device = 'cpu'
    print(f"\nForcing CPU mode for stable export")
    
    # Load config
    print("\nLoading configuration...")
    cfg = load_config()
    cfg.MODEL.DATASET = dataset
    
    # Create model
    print(f"\nLoading model from: {checkpoint}")
    model = MaskPLSDGCNNFixed(cfg)
    
    # Load checkpoint
    checkpoint_data = torch.load(checkpoint, map_location='cpu')
    if 'state_dict' in checkpoint_data:
        state_dict = checkpoint_data['state_dict']
        print(f"  Epoch: {checkpoint_data.get('epoch', 'unknown')}")
    else:
        state_dict = checkpoint_data
    
    # Load weights
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model = model.cpu()
    
    # Create CPU wrapper
    print("\nCreating CPU wrapper...")
    wrapped_model = CPUWrapper(model)
    wrapped_model.eval()
    
    # Prepare dummy inputs on CPU
    print(f"\nPreparing inputs (batch_size={batch_size}, num_points={num_points})...")
    coords = torch.randn(batch_size, num_points, 3) * 20.0
    feats = torch.randn(batch_size, num_points, 4)
    
    # Test forward pass
    print("\nTesting forward pass...")
    try:
        with torch.no_grad():
            test_outputs = wrapped_model(coords, feats)
        print("  ✓ Forward pass successful")
        print(f"  Output shapes: {[t.shape for t in test_outputs]}")
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        return
    
    # Export to ONNX
    print(f"\nExporting to ONNX: {output}")
    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapped_model,
                (coords, feats),
                output,
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
        print(f"  ✓ Export successful!")
    except Exception as e:
        print(f"  ✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Verify the model
    print("\nVerifying ONNX model...")
    try:
        onnx_model = onnx.load(output)
        onnx.checker.check_model(onnx_model)
        print("  ✓ ONNX model is valid")
        
        file_size = Path(output).stat().st_size / (1024 * 1024)
        print(f"  ✓ Model size: {file_size:.2f} MB")
        
        # Try ONNX Runtime
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(output, providers=['CPUExecutionProvider'])
            
            # Test inference
            test_coords = coords.numpy()
            test_feats = feats.numpy()
            
            outputs = session.run(None, {
                'coords': test_coords,
                'features': test_feats
            })
            
            print("  ✓ ONNX Runtime inference successful")
            print(f"  Output shapes: {[o.shape for o in outputs]}")
            
        except ImportError:
            print("  ⚠ ONNX Runtime not installed, skipping inference test")
        except Exception as e:
            print(f"  ⚠ ONNX Runtime test failed: {e}")
            
    except Exception as e:
        print(f"  ✗ Verification failed: {e}")
    
    print("\n✅ Conversion complete!")


if __name__ == "__main__":
    convert()