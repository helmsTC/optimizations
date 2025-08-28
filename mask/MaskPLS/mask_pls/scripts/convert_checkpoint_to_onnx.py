"""
Standalone script to convert a trained checkpoint to ONNX format
"""

import os
from os.path import join
import click
import torch
import yaml
import onnx
from pathlib import Path
from easydict import EasyDict as edict

# Import model
from mask_pls.models.onnx.simplified_model import MaskPLSSimplifiedONNX


def export_checkpoint_to_onnx(checkpoint_path, output_path=None, batch_size=1, num_points=10000, 
                             voxel_shape=(64, 64, 32), verify=True):
    """
    Export a checkpoint to ONNX format
    
    Args:
        checkpoint_path: Path to the checkpoint file
        output_path: Output ONNX file path (optional)
        batch_size: Batch size for export
        num_points: Number of points for export
        voxel_shape: Voxel grid dimensions (D, H, W)
        verify: Whether to verify the exported model
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract hyperparameters
    if 'hyper_parameters' in checkpoint:
        cfg = edict(checkpoint['hyper_parameters'])
    else:
        # Load default config if not in checkpoint
        def getDir(obj):
            return os.path.dirname(os.path.abspath(obj))
        
        model_cfg = edict(yaml.safe_load(open(join(getDir(__file__), "../config/model.yaml"))))
        backbone_cfg = edict(yaml.safe_load(open(join(getDir(__file__), "../config/backbone.yaml"))))
        decoder_cfg = edict(yaml.safe_load(open(join(getDir(__file__), "../config/decoder.yaml"))))
        cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    
    # Update backbone config to match training
    cfg.BACKBONE.CHANNELS = [32, 64, 128, 256, 256]
    
    # Create model
    print("Creating model...")
    model = MaskPLSSimplifiedONNX(cfg)
    
    # Load state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Remove 'model.' prefix if present
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        # Only load matching keys
        model_state = model.state_dict()
        filtered_state = {k: v for k, v in state_dict.items() if k in model_state}
        model.load_state_dict(filtered_state, strict=False)
        print(f"Loaded {len(filtered_state)}/{len(model_state)} parameters")
    else:
        print("Warning: No state_dict found in checkpoint")
    
    # Set output path
    if output_path is None:
        checkpoint_path = Path(checkpoint_path)
        output_path = checkpoint_path.parent / f"{checkpoint_path.stem}.onnx"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set model to eval mode
    model.eval()
    
    # Update spatial shape
    model.spatial_shape = voxel_shape
    D, H, W = voxel_shape
    C = 4  # XYZI features
    
    # Create dummy inputs
    print(f"Exporting with input shapes:")
    print(f"  voxel_features: [{batch_size}, {C}, {D}, {H}, {W}]")
    print(f"  point_coords: [{batch_size}, {num_points}, 3]")
    
    dummy_voxels = torch.randn(batch_size, C, D, H, W)
    dummy_coords = torch.rand(batch_size, num_points, 3)
    
    # Move to CUDA if available
    if torch.cuda.is_available():
        model = model.cuda()
        dummy_voxels = dummy_voxels.cuda()
        dummy_coords = dummy_coords.cuda()
    
    # Export to ONNX
    print(f"Exporting to: {output_path}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_voxels, dummy_coords),
            str(output_path),
            export_params=True,
            opset_version=16,
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
    
    # Verify the exported model
    if verify:
        print("Verifying ONNX model...")
        try:
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            print("✓ ONNX model verification passed")
        except Exception as e:
            print(f"⚠ ONNX verification failed: {e}")
    
    # Get file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    
    print(f"\n{'='*60}")
    print(f"✓ ONNX export successful!")
    print(f"  Output: {output_path}")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"  Opset Version: 16")
    print(f"{'='*60}")
    
    return str(output_path)


@click.command()
@click.argument('checkpoint_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=str, default=None, help='Output ONNX file path')
@click.option('--batch_size', type=int, default=1, help='Batch size for export')
@click.option('--num_points', type=int, default=10000, help='Number of points for export')
@click.option('--voxel_d', type=int, default=64, help='Voxel grid depth')
@click.option('--voxel_h', type=int, default=64, help='Voxel grid height')
@click.option('--voxel_w', type=int, default=32, help='Voxel grid width')
@click.option('--no_verify', is_flag=True, help='Skip ONNX verification')
def main(checkpoint_path, output, batch_size, num_points, voxel_d, voxel_h, voxel_w, no_verify):
    """
    Convert a MaskPLS checkpoint to ONNX format
    
    Example:
        python convert_checkpoint_to_onnx.py path/to/checkpoint.ckpt -o model.onnx
    """
    voxel_shape = (voxel_d, voxel_h, voxel_w)
    verify = not no_verify
    
    export_checkpoint_to_onnx(
        checkpoint_path=checkpoint_path,
        output_path=output,
        batch_size=batch_size,
        num_points=num_points,
        voxel_shape=voxel_shape,
        verify=verify
    )


if __name__ == "__main__":
    main()