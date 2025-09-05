# mask_pls/scripts/convert_to_onnx.py
"""
Convert trained MaskPLS DGCNN checkpoint to ONNX format
Supports PyTorch 2.4 and ONNX 1.19
"""

import os
import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
import yaml
from pathlib import Path
from easydict import EasyDict as edict
import click
from typing import Dict, Tuple, List

# Import the model
from mask_pls.models.dgcnn.maskpls_dgcnn_optimized import MaskPLSDGCNNFixed


class ONNXWrapper(nn.Module):
    """
    Wrapper to make the model ONNX-compatible by handling the dict input/output
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
        
    def forward(self, coords_tensor: torch.Tensor, feats_tensor: torch.Tensor, 
                valid_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ONNX-compatible forward pass
        
        Args:
            coords_tensor: [B, N, 3] point coordinates
            feats_tensor: [B, N, 4] point features (xyz + intensity)
            valid_mask: [B, N] boolean mask for valid points
            
        Returns:
            pred_logits: [B, Q, num_classes+1] class predictions
            pred_masks: [B, N, Q] mask predictions  
            sem_logits: [B, N, num_classes] semantic predictions
        """
        # Convert tensors back to the format expected by the model
        batch_size = coords_tensor.shape[0]
        
        # Create the input dict expected by the model
        x = {
            'pt_coord': [],
            'feats': []
        }
        
        for b in range(batch_size):
            # Get valid points for this sample
            valid_idx = valid_mask[b]
            coords_b = coords_tensor[b][valid_idx].cpu().numpy()
            feats_b = feats_tensor[b][valid_idx].cpu().numpy()
            
            x['pt_coord'].append(coords_b)
            x['feats'].append(feats_b)
        
        # Forward through the model
        outputs, padding, sem_logits = self.model(x)
        
        # Extract outputs
        pred_logits = outputs['pred_logits']
        pred_masks = outputs['pred_masks']
        
        return pred_logits, pred_masks, sem_logits


class SimplifiedONNXModel(nn.Module):
    """
    Simplified ONNX-compatible version that processes tensors directly
    """
    def __init__(self, original_model):
        super().__init__()
        self.backbone = original_model.backbone
        self.decoder = original_model.decoder
        self.num_classes = original_model.num_classes
        
    def forward(self, coords: torch.Tensor, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simplified forward for ONNX export
        
        Args:
            coords: [B, N, 3] normalized coordinates in [0, 1]
            feats: [B, N, 4] point features
            
        Returns:
            pred_logits: [B, Q, num_classes+1]
            pred_masks: [B, N, Q]
            sem_logits: [B, N, num_classes]
        """
        B, N, _ = coords.shape
        
        # Create dummy batch data to simulate the original input format
        x = {
            'pt_coord': [coords[b].cpu().numpy() for b in range(B)],
            'feats': [feats[b].cpu().numpy() for b in range(B)]
        }
        
        # Use the backbone directly
        ms_features, ms_coords, ms_masks, sem_logits = self.backbone(x)
        
        # Decoder forward
        outputs, padding = self.decoder(ms_features, ms_coords, ms_masks)
        
        return outputs['pred_logits'], outputs['pred_masks'], sem_logits


def load_config():
    """Load configuration files"""
    def getDir(obj):
        return os.path.dirname(os.path.abspath(obj))
    
    base_dir = Path(getDir(__file__)).parent
    
    model_cfg = yaml.safe_load(open(base_dir / "config" / "model.yaml"))
    backbone_cfg = yaml.safe_load(open(base_dir / "config" / "backbone.yaml"))
    decoder_cfg = yaml.safe_load(open(base_dir / "config" / "decoder.yaml"))
    
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    return cfg


def prepare_dummy_input(batch_size: int = 1, num_points: int = 10000, device: str = 'cuda'):
    """
    Prepare dummy input for ONNX export
    """
    # Random coordinates in [-50, 50] range
    coords = torch.randn(batch_size, num_points, 3, device=device) * 30.0
    
    # Features: xyz + intensity
    intensity = torch.rand(batch_size, num_points, 1, device=device)
    feats = torch.cat([coords, intensity], dim=-1)
    
    # Valid mask (all points are valid in this case)
    valid_mask = torch.ones(batch_size, num_points, dtype=torch.bool, device=device)
    
    return coords, feats, valid_mask


def export_to_onnx(model: nn.Module, output_path: str, 
                   batch_size: int = 1, num_points: int = 10000,
                   opset_version: int = 17):
    """
    Export model to ONNX format
    """
    print(f"\nExporting to ONNX format...")
    print(f"  Output path: {output_path}")
    print(f"  Batch size: {batch_size}")
    print(f"  Num points: {num_points}")
    print(f"  Opset version: {opset_version}")
    
    # Set model to eval mode
    model.eval()
    
    # Prepare dummy input
    device = next(model.parameters()).device
    coords, feats, valid_mask = prepare_dummy_input(batch_size, num_points, device)
    
    # Create a simplified forward pass for ONNX
    class ONNXModel(nn.Module):
        def __init__(self, backbone, decoder, num_classes):
            super().__init__()
            self.backbone = backbone
            self.decoder = decoder
            self.num_classes = num_classes
            
        def forward(self, coords, feats):
            # Simulate the original input format
            B = coords.shape[0]
            x = {
                'pt_coord': [coords[b].cpu().numpy() for b in range(B)],
                'feats': [feats[b].cpu().numpy() for b in range(B)]
            }
            
            # Backbone forward
            ms_features, ms_coords, ms_masks, sem_logits = self.backbone(x)
            
            # Decoder forward  
            outputs, _ = self.decoder(ms_features, ms_coords, ms_masks)
            
            return outputs['pred_logits'], outputs['pred_masks'], sem_logits
    
    # Create ONNX-compatible model
    onnx_model = ONNXModel(model.backbone, model.decoder, model.num_classes)
    onnx_model.eval()
    
    # Export to ONNX
    try:
        # Use torch.onnx.export with dynamic axes
        torch.onnx.export(
            onnx_model,
            (coords, feats),
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['point_coords', 'point_features'],
            output_names=['pred_logits', 'pred_masks', 'sem_logits'],
            dynamic_axes={
                'point_coords': {0: 'batch_size', 1: 'num_points'},
                'point_features': {0: 'batch_size', 1: 'num_points'},
                'pred_logits': {0: 'batch_size'},
                'pred_masks': {0: 'batch_size', 1: 'num_points'},
                'sem_logits': {0: 'batch_size', 1: 'num_points'}
            },
            verbose=False
        )
        
        print(f"✓ Successfully exported to {output_path}")
        
        # Verify the exported model
        print("\nVerifying ONNX model...")
        onnx_model_loaded = onnx.load(output_path)
        onnx.checker.check_model(onnx_model_loaded)
        print("✓ ONNX model is valid")
        
        # Print model info
        print("\nModel Information:")
        print(f"  File size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_onnx_inference(onnx_path: str, batch_size: int = 1, num_points: int = 10000):
    """
    Verify ONNX model inference
    """
    print("\nVerifying ONNX inference...")
    
    try:
        # Create inference session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(onnx_path, providers=providers)
        
        print(f"  Providers: {session.get_providers()}")
        
        # Get input/output info
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        
        print("\n  Inputs:")
        for inp in inputs:
            print(f"    {inp.name}: {inp.shape} ({inp.type})")
            
        print("\n  Outputs:")
        for out in outputs:
            print(f"    {out.name}: {out.shape} ({out.type})")
        
        # Prepare test input
        coords = np.random.randn(batch_size, num_points, 3).astype(np.float32) * 30.0
        feats = np.random.randn(batch_size, num_points, 4).astype(np.float32)
        
        # Run inference
        print("\n  Running test inference...")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        outputs = session.run(None, {
            'point_coords': coords,
            'point_features': feats
        })
        end.record()
        torch.cuda.synchronize()
        
        inference_time = start.elapsed_time(end)
        
        print(f"  ✓ Inference successful!")
        print(f"  Inference time: {inference_time:.2f} ms")
        
        # Check outputs
        pred_logits, pred_masks, sem_logits = outputs
        print(f"\n  Output shapes:")
        print(f"    pred_logits: {pred_logits.shape}")
        print(f"    pred_masks: {pred_masks.shape}")
        print(f"    sem_logits: {sem_logits.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Verification failed: {e}")
        return False


@click.command()
@click.option('--checkpoint', '-c', required=True, help='Path to PyTorch checkpoint')
@click.option('--output', '-o', help='Output ONNX file path (default: model.onnx)')
@click.option('--batch_size', default=1, help='Batch size for export')
@click.option('--num_points', default=10000, help='Number of points for export')
@click.option('--opset', default=17, help='ONNX opset version')
@click.option('--simplify', is_flag=True, help='Simplify the ONNX model')
@click.option('--optimize', is_flag=True, help='Optimize the ONNX model')
@click.option('--nuscenes', is_flag=True, help='Use NuScenes configuration')
@click.option('--verify', is_flag=True, help='Verify the exported model')
def main(checkpoint, output, batch_size, num_points, opset, simplify, optimize, nuscenes, verify):
    """Convert MaskPLS DGCNN checkpoint to ONNX format"""
    
    print("="*60)
    print("MaskPLS DGCNN to ONNX Converter")
    print("="*60)
    
    # Set output path
    if not output:
        output = Path(checkpoint).stem + ".onnx"
    
    # Load configuration
    cfg = load_config()
    if nuscenes:
        cfg.MODEL.DATASET = "NUSCENES"
    
    # Create model
    print(f"\nLoading model from: {checkpoint}")
    model = MaskPLSDGCNNFixed(cfg)
    
    # Load checkpoint
    ckpt = torch.load(checkpoint, map_location='cpu')
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # Move to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"Model loaded on: {device}")
    
    # Export to ONNX
    success = export_to_onnx(
        model, output, 
        batch_size=batch_size,
        num_points=num_points,
        opset_version=opset
    )
    
    if not success:
        return
    
    # Simplify if requested
    if simplify:
        try:
            import onnxsim
            print("\nSimplifying ONNX model...")
            
            model_onnx = onnx.load(output)
            model_simplified, check = onnxsim.simplify(model_onnx)
            
            if check:
                output_simplified = output.replace('.onnx', '_simplified.onnx')
                onnx.save(model_simplified, output_simplified)
                print(f"✓ Simplified model saved to: {output_simplified}")
                output = output_simplified
            else:
                print("✗ Simplification check failed")
                
        except ImportError:
            print("⚠ onnx-simplifier not installed. Install with: pip install onnx-simplifier")
    
    # Optimize if requested
    if optimize:
        from mask_pls.utils.onnx.optimization import optimize_onnx_model
        
        output_optimized = output.replace('.onnx', '_optimized.onnx')
        if optimize_onnx_model(output, output_optimized, 'extended'):
            output = output_optimized
    
    # Verify if requested
    if verify:
        verify_onnx_inference(output, batch_size, num_points)
    
    print(f"\n✅ Conversion complete!")
    print(f"   Output: {output}")
    print(f"   Size: {Path(output).stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()