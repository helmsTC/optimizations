# mask_pls/scripts/simple_convert_onnx.py
"""
Simplified ONNX converter for MaskPLS DGCNN
Handles PyTorch 2.4 and ONNX 1.19 compatibility
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

from mask_pls.models.dgcnn.maskpls_dgcnn_optimized import MaskPLSDGCNNFixed


class TracingWrapper(nn.Module):
    """
    Wrapper that makes the model traceable by handling dict inputs
    """
    def __init__(self, model, max_points=50000):
        super().__init__()
        self.model = model
        self.max_points = max_points
        # Store model components we need
        self.backbone = model.backbone
        self.decoder = model.decoder
        self.num_classes = model.num_classes
        
    def forward(self, coords_batch, feats_batch, valid_lengths):
        """
        Args:
            coords_batch: [B, N, 3] padded coordinates
            feats_batch: [B, N, 4] padded features  
            valid_lengths: [B] number of valid points per sample
            
        Returns:
            Tuple of (pred_logits, pred_masks, sem_logits)
        """
        B = coords_batch.shape[0]
        
        # Convert to the format expected by the model
        x = {'pt_coord': [], 'feats': []}
        
        # Process each sample in the batch
        for b in range(B):
            n_valid = int(valid_lengths[b].item())
            # Extract valid points only
            coords = coords_batch[b, :n_valid].cpu().numpy()
            feats = feats_batch[b, :n_valid].cpu().numpy()
            
            x['pt_coord'].append(coords)
            x['feats'].append(feats)
        
        # Forward pass through original model
        outputs, padding, sem_logits = self.model(x)
        
        # Ensure outputs are padded to max_points
        B, N, Q = outputs['pred_masks'].shape
        if N < self.max_points:
            # Pad masks to max_points
            pad_size = self.max_points - N
            pred_masks = torch.nn.functional.pad(
                outputs['pred_masks'], 
                (0, 0, 0, pad_size), 
                value=0
            )
            sem_logits_padded = torch.nn.functional.pad(
                sem_logits,
                (0, 0, 0, pad_size),
                value=0
            )
        else:
            pred_masks = outputs['pred_masks']
            sem_logits_padded = sem_logits
        
        return outputs['pred_logits'], pred_masks, sem_logits_padded


def load_config(config_dir=None):
    """Load configuration"""
    if config_dir is None:
        config_dir = Path(__file__).parent.parent / "config"
    
    cfg_files = {
        'model': config_dir / "model.yaml",
        'backbone': config_dir / "backbone.yaml", 
        'decoder': config_dir / "decoder.yaml"
    }
    
    cfg = {}
    for name, path in cfg_files.items():
        if path.exists():
            with open(path, 'r') as f:
                cfg.update(yaml.safe_load(f))
    
    return edict(cfg)


def convert_checkpoint_to_onnx(
    checkpoint_path: str,
    output_path: str = None,
    batch_size: int = 1,
    max_points: int = 30000,
    dataset: str = 'KITTI',
    verify: bool = True
):
    """
    Main conversion function
    
    Args:
        checkpoint_path: Path to .ckpt file
        output_path: Output ONNX file path
        batch_size: Batch size for export
        max_points: Maximum points per cloud
        dataset: 'KITTI' or 'NUSCENES'
        verify: Whether to verify the converted model
    """
    print("="*60)
    print("MaskPLS DGCNN to ONNX Converter")
    print("="*60)
    
    # Set output path
    if output_path is None:
        output_path = str(Path(checkpoint_path).with_suffix('.onnx'))
    
    # Load config
    print("\n1. Loading configuration...")
    cfg = load_config()
    cfg.MODEL.DATASET = dataset
    
    # Create and load model
    print("\n2. Loading PyTorch model...")
    model = MaskPLSDGCNNFixed(cfg)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print(f"   Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    else:
        state_dict = checkpoint
    
    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"   Missing keys: {len(missing)}")
    if unexpected:
        print(f"   Unexpected keys: {len(unexpected)}")
    
    model.eval()
    
    # Move to appropriate device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"   Model on device: {device}")
    
    # Create wrapper
    print("\n3. Creating ONNX-compatible wrapper...")
    wrapped_model = TracingWrapper(model, max_points)
    wrapped_model.eval()
    
    # Prepare dummy inputs
    print("\n4. Preparing dummy inputs...")
    print(f"   Batch size: {batch_size}")
    print(f"   Max points: {max_points}")
    
    # Create dummy data with varying point counts
    coords = torch.randn(batch_size, max_points, 3, device=device) * 20.0
    feats = torch.randn(batch_size, max_points, 4, device=device)
    valid_lengths = torch.randint(
        max_points // 2, max_points, 
        (batch_size,), 
        device=device
    )
    
    # Export to ONNX
    print("\n5. Exporting to ONNX...")
    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapped_model,
                (coords, feats, valid_lengths),
                output_path,
                export_params=True,
                opset_version=17,  # Compatible with PyTorch 2.4
                do_constant_folding=True,
                input_names=['coords', 'features', 'valid_lengths'],
                output_names=['pred_logits', 'pred_masks', 'sem_logits'],
                dynamic_axes={
                    'coords': {0: 'batch', 1: 'points'},
                    'features': {0: 'batch', 1: 'points'},
                    'valid_lengths': {0: 'batch'},
                    'pred_logits': {0: 'batch'},
                    'pred_masks': {0: 'batch', 1: 'points'},
                    'sem_logits': {0: 'batch', 1: 'points'}
                },
                verbose=False
            )
        
        print(f"   ✓ Successfully exported to: {output_path}")
        
    except Exception as e:
        print(f"   ✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Verify the model
    if verify:
        print("\n6. Verifying ONNX model...")
        try:
            # Check model validity
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("   ✓ ONNX model structure is valid")
            
            # Check file size
            file_size = Path(output_path).stat().st_size / (1024 * 1024)
            print(f"   ✓ Model size: {file_size:.2f} MB")
            
            # Try to create inference session
            try:
                import onnxruntime as ort
                session = ort.InferenceSession(output_path)
                print("   ✓ ONNX Runtime can load the model")
                
                # Print input/output info
                print("\n   Model inputs:")
                for inp in session.get_inputs():
                    print(f"     - {inp.name}: {inp.shape} ({inp.type})")
                
                print("\n   Model outputs:")
                for out in session.get_outputs():
                    print(f"     - {out.name}: {out.shape} ({out.type})")
                    
            except ImportError:
                print("   ⚠ ONNX Runtime not installed, skipping runtime verification")
                
        except Exception as e:
            print(f"   ✗ Verification failed: {e}")
            return False
    
    print("\n✅ Conversion completed successfully!")
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert MaskPLS DGCNN checkpoint to ONNX"
    )
    parser.add_argument(
        "checkpoint", 
        help="Path to PyTorch checkpoint (.ckpt file)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output ONNX file path (default: same name as checkpoint)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int, 
        default=1,
        help="Batch size for export (default: 1)"
    )
    parser.add_argument(
        "--max-points", "-p",
        type=int,
        default=30000,
        help="Maximum points per cloud (default: 30000)"
    )
    parser.add_argument(
        "--dataset", "-d",
        choices=['KITTI', 'NUSCENES'],
        default='KITTI',
        help="Dataset configuration (default: KITTI)"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification step"
    )
    
    args = parser.parse_args()
    
    # Run conversion
    success = convert_checkpoint_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        batch_size=args.batch_size,
        max_points=args.max_points,
        dataset=args.dataset,
        verify=not args.no_verify
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
