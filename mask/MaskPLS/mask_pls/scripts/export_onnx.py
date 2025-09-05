# mask_pls/scripts/export_efficient_dgcnn_onnx.py
"""
Export MaskPLS-DGCNN Fixed model checkpoint to ONNX
For checkpoints trained with train_efficient_dgcnn.py
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path
import click
import yaml
from easydict import EasyDict as edict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from mask_pls.models.dgcnn.maskpls_dgcnn_optimized import MaskPLSDGCNNFixed


class ONNXExportWrapper(nn.Module):
    """
    Wrapper for ONNX export that handles the MaskPLSDGCNN Fixed model
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.backbone = model.backbone
        self.decoder = model.decoder
        self.num_classes = model.num_classes
        
        # Set to eval mode
        self.model.eval()
        self.backbone.eval()
        self.decoder.eval()
    
    def forward(self, point_coords, point_features):
        """
        Simplified forward pass for ONNX export
        
        Args:
            point_coords: [B, N, 3] point coordinates
            point_features: [B, N, 4] point features (x, y, z, intensity)
            
        Returns:
            pred_logits: [B, Q, num_classes+1] class predictions for queries
            pred_masks: [B, N, Q] mask predictions
            sem_logits: [B, N, num_classes] semantic segmentation logits
        """
        B, N, _ = point_coords.shape
        
        # Create the expected input format
        x = {
            'pt_coord': [],
            'feats': []
        }
        
        # Convert batch to list format expected by backbone
        for b in range(B):
            x['pt_coord'].append(point_coords[b].cpu().numpy())
            x['feats'].append(point_features[b].cpu().numpy())
        
        # Forward through backbone
        with torch.no_grad():
            ms_features, ms_coords, ms_masks, sem_logits = self.backbone(x)
            
            # Forward through decoder
            outputs, padding = self.decoder(ms_features, ms_coords, ms_masks)
        
        # Extract outputs
        pred_logits = outputs['pred_logits']
        pred_masks = outputs['pred_masks']
        
        return pred_logits, pred_masks, sem_logits


class SimpleONNXWrapper(nn.Module):
    """
    Even simpler wrapper that processes one sample at a time
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.num_classes = model.num_classes
        
        # Extract components
        self.backbone = model.backbone
        self.decoder = model.decoder
        
        # Set to eval
        self.eval()
        
    def forward(self, point_coords, point_features):
        """
        Process single sample (no batch dimension)
        
        Args:
            point_coords: [N, 3] 
            point_features: [N, 4]
            
        Returns:
            pred_logits: [Q, num_classes+1]
            pred_masks: [N, Q]
            sem_logits: [N, num_classes]
        """
        # Add batch dimension
        coords_batch = point_coords.unsqueeze(0)
        feats_batch = point_features.unsqueeze(0)
        
        # Create input dict
        x = {
            'pt_coord': [coords_batch[0].detach().cpu().numpy()],
            'feats': [feats_batch[0].detach().cpu().numpy()]
        }
        
        # Process through model
        with torch.no_grad():
            # Backbone
            ms_features, ms_coords, ms_masks, sem_logits = self.backbone(x)
            
            # Decoder
            outputs, padding = self.decoder(ms_features, ms_coords, ms_masks)
        
        # Remove batch dimension
        pred_logits = outputs['pred_logits'].squeeze(0)
        pred_masks = outputs['pred_masks'].squeeze(0)
        sem_logits = sem_logits.squeeze(0)
        
        return pred_logits, pred_masks, sem_logits


def load_config(checkpoint_dir):
    """Load configuration from checkpoint directory"""
    config_dir = Path(__file__).parent.parent / "config"
    
    # Load base configs
    model_cfg = edict(yaml.safe_load(open(config_dir / "model.yaml")))
    backbone_cfg = edict(yaml.safe_load(open(config_dir / "backbone.yaml")))
    decoder_cfg = edict(yaml.safe_load(open(config_dir / "decoder.yaml")))
    
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    
    # Check for saved hyperparameters
    hparams_file = checkpoint_dir.parent / 'hparams.yaml'
    if hparams_file.exists():
        print(f"Loading saved hyperparameters from {hparams_file}")
        with open(hparams_file, 'r') as f:
            saved_cfg = yaml.safe_load(f)
            if saved_cfg:
                cfg.update(saved_cfg)
    
    # Set default training params if missing
    if 'TRAIN' not in cfg:
        cfg.TRAIN = edict({
            'BATCH_SIZE': 2,
            'WARMUP_STEPS': 500,
            'SUBSAMPLE': True,
            'AUG': True,
            'LR': 0.0001
        })
    
    return cfg


@click.command()
@click.option('--checkpoint', '-c', required=True, help='Path to checkpoint file (.ckpt)')
@click.option('--output', '-o', default=None, help='Output ONNX file path')
@click.option('--batch_size', '-b', default=1, help='Batch size for export')
@click.option('--num_points', '-n', default=50000, help='Number of points for dummy input')
@click.option('--opset', default=14, help='ONNX opset version')
@click.option('--simplify', is_flag=True, help='Simplify the exported model')
@click.option('--fp16', is_flag=True, help='Convert to FP16 precision')
@click.option('--validate', is_flag=True, help='Validate exported model')
@click.option('--simple', is_flag=True, help='Use simple single-sample wrapper')
def export_onnx(checkpoint, output, batch_size, num_points, opset, simplify, fp16, validate, simple):
    """Export MaskPLS-DGCNN Fixed model to ONNX"""
    
    print("="*60)
    print("MaskPLS-DGCNN ONNX Export")
    print("="*60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Set output path
    if output is None:
        checkpoint_path = Path(checkpoint)
        output = checkpoint_path.parent / f"{checkpoint_path.stem}.onnx"
    
    print(f"Checkpoint: {checkpoint}")
    print(f"Output: {output}")
    print(f"Batch size: {batch_size}")
    print(f"Points: {num_points}")
    
    # Load configuration
    print("\nLoading configuration...")
    checkpoint_dir = Path(checkpoint).parent
    cfg = load_config(checkpoint_dir)
    
    # Create model
    print("\nCreating model...")
    model = MaskPLSDGCNNFixed(cfg)
    
    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint_data = torch.load(checkpoint, map_location=device)
    
    if 'state_dict' in checkpoint_data:
        state_dict = checkpoint_data['state_dict']
        epoch = checkpoint_data.get('epoch', 'unknown')
        print(f"  Checkpoint from epoch: {epoch}")
    else:
        state_dict = checkpoint_data
    
    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"  Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"  Unexpected keys: {len(unexpected_keys)}")
    
    # Move to device and set to eval
    model = model.to(device)
    model.eval()
    
    # Create wrapper
    if simple:
        print("\nUsing simple single-sample wrapper...")
        onnx_model = SimpleONNXWrapper(model).to(device)
        
        # Create dummy input (no batch dimension)
        dummy_coords = torch.randn(num_points, 3, device=device)
        dummy_features = torch.randn(num_points, 4, device=device)
        dummy_input = (dummy_coords, dummy_features)
        
        input_names = ['point_coords', 'point_features']
        output_names = ['pred_logits', 'pred_masks', 'sem_logits']
        dynamic_axes = {
            'point_coords': {0: 'num_points'},
            'point_features': {0: 'num_points'},
            'pred_masks': {0: 'num_points'}
        }
    else:
        print("\nUsing batch wrapper...")
        onnx_model = ONNXExportWrapper(model).to(device)
        
        # Create dummy input (with batch dimension)
        dummy_coords = torch.randn(batch_size, num_points, 3, device=device)
        dummy_features = torch.randn(batch_size, num_points, 4, device=device)
        dummy_input = (dummy_coords, dummy_features)
        
        input_names = ['point_coords', 'point_features']
        output_names = ['pred_logits', 'pred_masks', 'sem_logits']
        dynamic_axes = {
            'point_coords': {0: 'batch_size', 1: 'num_points'},
            'point_features': {0: 'batch_size', 1: 'num_points'},
            'pred_logits': {0: 'batch_size'},
            'pred_masks': {0: 'batch_size', 1: 'num_points'},
            'sem_logits': {0: 'batch_size', 1: 'num_points'}
        }
    
    # Test forward pass
    print("\nTesting forward pass...")
    try:
        with torch.no_grad():
            test_output = onnx_model(*dummy_input)
        print(f"✓ Forward pass successful")
        print(f"  Output shapes: {[t.shape for t in test_output]}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Export to ONNX
    print("\nExporting to ONNX...")
    try:
        with torch.no_grad():
            torch.onnx.export(
                onnx_model,
                dummy_input,
                str(output),
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=False
            )
        
        print(f"✓ Model exported to {output}")
        
        # Get file size
        file_size = Path(output).stat().st_size / (1024 * 1024)
        print(f"  File size: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Check model
    print("\nChecking ONNX model...")
    try:
        model_onnx = onnx.load(str(output))
        onnx.checker.check_model(model_onnx)
        print("✓ ONNX model is valid")
    except Exception as e:
        print(f"✗ Model check failed: {e}")
    
    # Simplify if requested
    if simplify:
        print("\nSimplifying model...")
        try:
            import onnxsim
            model_onnx, check = onnxsim.simplify(model_onnx)
            if check:
                onnx.save(model_onnx, str(output))
                new_size = Path(output).stat().st_size / (1024 * 1024)
                print(f"✓ Model simplified")
                print(f"  New size: {new_size:.2f} MB (was {file_size:.2f} MB)")
            else:
                print("✗ Simplification check failed")
        except ImportError:
            print("  ⚠ onnx-simplifier not installed")
            print("  Install with: pip install onnx-simplifier")
    
    # Convert to FP16 if requested
    if fp16:
        print("\nConverting to FP16...")
        try:
            from onnxconverter_common import float16
            model_fp16 = float16.convert_float_to_float16(model_onnx)
            fp16_path = str(output).replace('.onnx', '_fp16.onnx')
            onnx.save(model_fp16, fp16_path)
            fp16_size = Path(fp16_path).stat().st_size / (1024 * 1024)
            print(f"✓ FP16 model saved to {fp16_path}")
            print(f"  Size: {fp16_size:.2f} MB")
        except ImportError:
            print("  ⚠ onnxconverter-common not installed")
            print("  Install with: pip install onnxconverter-common")
    
    # Validate if requested
    if validate:
        print("\nValidating ONNX model...")
        try:
            # Create inference session
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            session = ort.InferenceSession(str(output), providers=providers)
            
            print(f"  Provider: {session.get_providers()[0]}")
            
            # Test inference
            if simple:
                test_coords = np.random.randn(num_points, 3).astype(np.float32)
                test_features = np.random.randn(num_points, 4).astype(np.float32)
            else:
                test_coords = np.random.randn(batch_size, num_points, 3).astype(np.float32)
                test_features = np.random.randn(batch_size, num_points, 4).astype(np.float32)
            
            inputs = {
                'point_coords': test_coords,
                'point_features': test_features
            }
            
            outputs = session.run(None, inputs)
            
            print("✓ Inference successful")
            print(f"  Output shapes: {[o.shape for o in outputs]}")
            
        except Exception as e:
            print(f"✗ Validation failed: {e}")
    
    print("\n" + "="*60)
    print("Export completed!")
    print("="*60)
    
    # Print usage example
    print("\nUsage example:")
    print("```python")
    print("import onnxruntime as ort")
    print("import numpy as np")
    print()
    print(f"session = ort.InferenceSession('{output}')")
    if simple:
        print("inputs = {")
        print("    'point_coords': np.random.randn(50000, 3).astype(np.float32),")
        print("    'point_features': np.random.randn(50000, 4).astype(np.float32)")
        print("}")
    else:
        print("inputs = {")
        print(f"    'point_coords': np.random.randn({batch_size}, 50000, 3).astype(np.float32),")
        print(f"    'point_features': np.random.randn({batch_size}, 50000, 4).astype(np.float32)")
        print("}")
    print("outputs = session.run(None, inputs)")
    print("pred_logits, pred_masks, sem_logits = outputs")
    print("```")


if __name__ == "__main__":
    export_onnx()