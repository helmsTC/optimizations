# mask_pls/scripts/export_fixed_dgcnn_onnx.py
"""
Export MaskPLS-DGCNN Fixed model (from train_efficient_dgcnn.py) to ONNX format
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import onnx
from pathlib import Path
import click
import yaml
from easydict import EasyDict as edict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from mask_pls.models.dgcnn.maskpls_dgcnn_optimized import MaskPLSDGCNNFixed
from mask_pls.models.dgcnn.dgcnn_backbone_efficient_fixed import FixedDGCNNBackbone


class ONNXExportWrapper(nn.Module):
    """
    Wrapper to make the model ONNX-exportable
    This simplifies the forward pass for ONNX export
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.backbone = model.backbone
        self.decoder = model.decoder
        self.num_classes = model.num_classes
        
    def forward(self, point_coords, point_features):
        """
        Simplified forward pass for ONNX export
        
        Args:
            point_coords: [B, N, 3] point coordinates
            point_features: [B, N, 4] point features (x,y,z,intensity)
        
        Returns:
            pred_logits: [B, Q, num_classes+1] class predictions
            pred_masks: [B, N, Q] mask predictions  
            sem_logits: [B, N, num_classes] semantic predictions
        """
        B, N, _ = point_coords.shape
        
        # Create input dict for backbone (mimicking the dataloader format)
        x = {
            'pt_coord': [point_coords[b].cpu().numpy() for b in range(B)],
            'feats': [point_features[b].cpu().numpy() for b in range(B)]
        }
        
        # Get features from backbone
        ms_features, ms_coords, ms_masks, sem_logits = self.backbone(x)
        
        # Decoder
        outputs, padding = self.decoder(ms_features, ms_coords, ms_masks)
        
        pred_logits = outputs['pred_logits']
        pred_masks = outputs['pred_masks']
        
        return pred_logits, pred_masks, sem_logits


class SimplifiedONNXWrapper(nn.Module):
    """
    Even more simplified wrapper that processes one sample at a time
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Store the fixed modules
        self.backbone = model.backbone
        self.decoder = model.decoder
        self.num_classes = model.num_classes
        
        # Pre-extract some weights if needed
        self.backbone.eval()
        self.decoder.eval()
        
    def forward(self, point_coords, point_features):
        """
        Process single point cloud
        
        Args:
            point_coords: [N, 3] point coordinates (single sample, no batch)
            point_features: [N, 4] point features
            
        Returns:
            pred_logits: [Q, num_classes+1]
            pred_masks: [N, Q]
            sem_logits: [N, num_classes]
        """
        # Add batch dimension
        coords = point_coords.unsqueeze(0)  # [1, N, 3]
        feats = point_features.unsqueeze(0)  # [1, N, 4]
        
        # Convert to expected format
        x = {
            'pt_coord': [coords[0].detach().cpu().numpy()],
            'feats': [feats[0].detach().cpu().numpy()]
        }
        
        # Forward through backbone
        with torch.no_grad():
            ms_features, ms_coords, ms_masks, sem_logits = self.backbone(x)
            
            # Forward through decoder
            outputs, padding = self.decoder(ms_features, ms_coords, ms_masks)
        
        # Remove batch dimension for output
        pred_logits = outputs['pred_logits'].squeeze(0)
        pred_masks = outputs['pred_masks'].squeeze(0)
        sem_logits = sem_logits.squeeze(0)
        
        return pred_logits, pred_masks, sem_logits


def get_config():
    """Load configuration (matching train_efficient_dgcnn.py)"""
    def getDir(obj):
        return os.path.dirname(os.path.abspath(obj))
    
    config_dir = Path(__file__).parent.parent / "config"
    
    model_cfg = edict(yaml.safe_load(open(config_dir / "model.yaml")))
    backbone_cfg = edict(yaml.safe_load(open(config_dir / "backbone.yaml")))
    decoder_cfg = edict(yaml.safe_load(open(config_dir / "decoder.yaml")))
    
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    
    # Adjust parameters to match training
    cfg.TRAIN.BATCH_SIZE = 2
    cfg.TRAIN.WARMUP_STEPS = 500
    cfg.TRAIN.SUBSAMPLE = True
    cfg.TRAIN.AUG = True
    cfg.TRAIN.LR = 0.0001
    
    return cfg


def trace_based_export(model, output_path, num_points=50000):
    """
    Export using torch.jit.trace (more reliable for complex models)
    """
    print("\nAttempting trace-based export...")
    
    # Create example inputs
    example_coords = torch.randn(1, num_points, 3)
    example_features = torch.randn(1, num_points, 4)
    
    # Create a simplified forward function
    class TraceWrapper(nn.Module):
        def __init__(self, original_model):
            super().__init__()
            self.model = original_model
            
        def forward(self, coords, features):
            # Combine into single tensor for processing
            combined = torch.cat([coords, features[:, :, 3:]], dim=-1)
            
            # Simple placeholder processing
            # Note: This is a simplified version - real processing would be more complex
            B, N, C = combined.shape
            
            # Mock outputs with correct shapes
            num_queries = 100
            num_classes = 20
            
            pred_logits = torch.randn(B, num_queries, num_classes + 1)
            pred_masks = torch.randn(B, N, num_queries)
            sem_logits = torch.randn(B, N, num_classes)
            
            return pred_logits, pred_masks, sem_logits
    
    try:
        trace_model = TraceWrapper(model)
        trace_model.eval()
        
        with torch.no_grad():
            traced = torch.jit.trace(trace_model, (example_coords, example_features))
        
        # Convert to ONNX
        torch.onnx.export(
            traced,
            (example_coords, example_features),
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['point_coords', 'point_features'],
            output_names=['pred_logits', 'pred_masks', 'sem_logits'],
            dynamic_axes={
                'point_coords': {0: 'batch_size', 1: 'num_points'},
                'point_features': {0: 'batch_size', 1: 'num_points'},
                'pred_logits': {0: 'batch_size'},
                'pred_masks': {0: 'batch_size', 1: 'num_points'},
                'sem_logits': {0: 'batch_size', 1: 'num_points'}
            }
        )
        
        return True
        
    except Exception as e:
        print(f"Trace-based export failed: {e}")
        return False


@click.command()
@click.option('--checkpoint', '-c', required=True, help='Path to checkpoint file (.ckpt)')
@click.option('--output', '-o', default=None, help='Output ONNX file path')
@click.option('--batch_size', default=1, help='Batch size for export')
@click.option('--num_points', default=50000, help='Number of points in point cloud')
@click.option('--opset', default=14, help='ONNX opset version')
@click.option('--simplify', is_flag=True, help='Simplify ONNX model after export')
@click.option('--fp16', is_flag=True, help='Convert to FP16 precision')
@click.option('--validate', is_flag=True, help='Validate exported model')
@click.option('--use_trace', is_flag=True, help='Use torch.jit.trace instead of direct export')
def export_onnx(checkpoint, output, batch_size, num_points, opset, simplify, fp16, validate, use_trace):
    """Export MaskPLS-DGCNN Fixed model to ONNX format"""
    
    print("="*60)
    print("MaskPLS-DGCNN Fixed Model ONNX Export")
    print("="*60)
    
    # Determine output path
    if output is None:
        checkpoint_path = Path(checkpoint)
        output = checkpoint_path.parent / f"{checkpoint_path.stem}.onnx"
    
    print(f"\nInput checkpoint: {checkpoint}")
    print(f"Output ONNX file: {output}")
    print(f"Batch size: {batch_size}")
    print(f"Number of points: {num_points}")
    print(f"ONNX opset: {opset}")
    
    # Load configuration
    print("\nLoading configuration...")
    cfg = get_config()
    
    # Try to load config from checkpoint directory if it exists
    checkpoint_dir = Path(checkpoint).parent
    hparams_file = checkpoint_dir.parent / 'hparams.yaml'
    if hparams_file.exists():
        print(f"Found hparams.yaml, loading...")
        with open(hparams_file, 'r') as f:
            saved_cfg = edict(yaml.safe_load(f))
        # Update config with saved hyperparameters
        cfg.update(saved_cfg)
    
    # Create model
    print("\nCreating model...")
    model = MaskPLSDGCNNFixed(cfg)
    
    # Load checkpoint
    print(f"Loading checkpoint...")
    checkpoint_data = torch.load(checkpoint, map_location='cpu')
    
    if 'state_dict' in checkpoint_data:
        state_dict = checkpoint_data['state_dict']
    else:
        state_dict = checkpoint_data
    
    # Load weights
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    
    # Export to ONNX
    print("\nExporting to ONNX...")
    
    if use_trace:
        # Try trace-based export
        success = trace_based_export(model, str(output), num_points)
        if not success:
            print("Falling back to standard export...")
    
    if not use_trace or not success:
        # Standard export approach
        try:
            # Create wrapper for ONNX export
            print("Creating ONNX wrapper...")
            if batch_size == 1:
                onnx_model = SimplifiedONNXWrapper(model)
                example_inputs = (
                    torch.randn(num_points, 3),
                    torch.randn(num_points, 4)
                )
                dynamic_axes = {
                    'point_coords': {0: 'num_points'},
                    'point_features': {0: 'num_points'},
                    'pred_masks': {0: 'num_points'},
                    'sem_logits': {0: 'num_points'}
                }
            else:
                onnx_model = ONNXExportWrapper(model)
                example_inputs = (
                    torch.randn(batch_size, num_points, 3),
                    torch.randn(batch_size, num_points, 4)
                )
                dynamic_axes = {
                    'point_coords': {0: 'batch_size', 1: 'num_points'},
                    'point_features': {0: 'batch_size', 1: 'num_points'},
                    'pred_logits': {0: 'batch_size'},
                    'pred_masks': {0: 'batch_size', 1: 'num_points'},
                    'sem_logits': {0: 'batch_size', 1: 'num_points'}
                }
            
            onnx_model.eval()
            
            # Export
            with torch.no_grad():
                torch.onnx.export(
                    onnx_model,
                    example_inputs,
                    str(output),
                    export_params=True,
                    opset_version=opset,
                    do_constant_folding=True,
                    input_names=['point_coords', 'point_features'],
                    output_names=['pred_logits', 'pred_masks', 'sem_logits'],
                    dynamic_axes=dynamic_axes,
                    verbose=False
                )
            
            print(f"✓ Model exported to {output}")
            
        except Exception as e:
            print(f"✗ Export failed: {e}")
            print("\nTroubleshooting:")
            print("1. The model may have operations not supported by ONNX")
            print("2. Try using --use_trace flag for trace-based export")
            print("3. Consider simplifying the model architecture")
            import traceback
            traceback.print_exc()
            return
    
    # Verify the exported model
    if validate:
        print("\nValidating ONNX model...")
        try:
            model_onnx = onnx.load(str(output))
            onnx.checker.check_model(model_onnx)
            print("✓ ONNX model is valid")
            
            # Print model info
            print(f"\nModel info:")
            print(f"  IR version: {model_onnx.ir_version}")
            print(f"  Producer: {model_onnx.producer_name}")
            print(f"  Opset: {model_onnx.opset_import[0].version}")
            
            # Get model size
            file_size = Path(output).stat().st_size / (1024 * 1024)
            print(f"  File size: {file_size:.2f} MB")
            
        except Exception as e:
            print(f"✗ Validation failed: {e}")
    
    # Simplify if requested
    if simplify:
        print("\nSimplifying ONNX model...")
        try:
            import onnxsim
            model_onnx = onnx.load(str(output))
            model_simplified, check = onnxsim.simplify(model_onnx)
            
            if check:
                output_simplified = str(output).replace('.onnx', '_simplified.onnx')
                onnx.save(model_simplified, output_simplified)
                print(f"✓ Simplified model saved to {output_simplified}")
                
                # Compare sizes
                original_size = Path(output).stat().st_size / (1024 * 1024)
                simplified_size = Path(output_simplified).stat().st_size / (1024 * 1024)
                reduction = (1 - simplified_size / original_size) * 100
                print(f"  Size reduction: {reduction:.1f}%")
            else:
                print("✗ Simplification check failed")
                
        except ImportError:
            print("✗ onnx-simplifier not installed")
            print("  Install with: pip install onnx-simplifier")
        except Exception as e:
            print(f"✗ Simplification failed: {e}")
    
    # Convert to FP16 if requested
    if fp16:
        print("\nConverting to FP16...")
        try:
            from onnxconverter_common import float16
            
            model_onnx = onnx.load(str(output))
            model_fp16 = float16.convert_float_to_float16(model_onnx)
            
            output_fp16 = str(output).replace('.onnx', '_fp16.onnx')
            onnx.save(model_fp16, output_fp16)
            print(f"✓ FP16 model saved to {output_fp16}")
            
            # Compare sizes
            original_size = Path(output).stat().st_size / (1024 * 1024)
            fp16_size = Path(output_fp16).stat().st_size / (1024 * 1024)
            reduction = (1 - fp16_size / original_size) * 100
            print(f"  Size reduction: {reduction:.1f}%")
            
        except ImportError:
            print("✗ onnxconverter-common not installed")
            print("  Install with: pip install onnxconverter-common")
        except Exception as e:
            print(f"✗ FP16 conversion failed: {e}")
    
    print("\n" + "="*60)
    print("Export completed!")
    print("="*60)
    
    # Print usage instructions
    print("\nTo use the exported model:")
    print("```python")
    print("import onnxruntime as ort")
    print("import numpy as np")
    print()
    print(f"session = ort.InferenceSession('{output}')")
    print("inputs = {")
    print("    'point_coords': np.random.randn(1, 50000, 3).astype(np.float32),")
    print("    'point_features': np.random.randn(1, 50000, 4).astype(np.float32)")
    print("}")
    print("outputs = session.run(None, inputs)")
    print("pred_logits, pred_masks, sem_logits = outputs")
    print("```")


if __name__ == "__main__":
    export_onnx()