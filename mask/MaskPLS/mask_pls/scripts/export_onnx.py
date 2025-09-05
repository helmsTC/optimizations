# mask_pls/scripts/export_dgcnn_cpu_compatible.py
"""
Export MaskPLS-DGCNN Fixed model to ONNX with CPU-compatible backbone
This version patches all CUDA calls in the backbone
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import onnx
from pathlib import Path
import click
import yaml
from easydict import EasyDict as edict
import types

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from mask_pls.models.dgcnn.maskpls_dgcnn_optimized import MaskPLSDGCNNFixed
from mask_pls.models.dgcnn.dgcnn_backbone_efficient_fixed import FixedDGCNNBackbone


def create_cpu_compatible_forward(original_model):
    """
    Create a CPU-compatible forward method for the backbone
    """
    def cpu_forward(self, x):  # Added self parameter
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
            # Convert to tensors WITHOUT .cuda()
            coords = torch.from_numpy(coords_list[b]).float()  # No .cuda()!
            feats = torch.from_numpy(feats_list[b]).float()    # No .cuda()!
            
            # Subsample if needed
            max_points = 50000 if self.training else 30000
            if coords.shape[0] > max_points:
                indices = torch.randperm(coords.shape[0])[:max_points]  # CPU tensor
                indices = indices.sort()[0]
                coords = coords[indices]
                feats = feats[indices]
                self.subsample_indices[b] = indices
            else:
                self.subsample_indices[b] = torch.arange(coords.shape[0])  # CPU tensor
            
            # Process through DGCNN
            point_features = self.process_single_cloud(coords, feats)
            
            all_features.append(point_features)
            all_coords.append(coords)
            # Create mask on CPU
            all_masks.append(torch.zeros(coords.shape[0], dtype=torch.bool))  # No device specified
        
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
    
    return cpu_forward


def patch_backbone_for_cpu(backbone):
    """
    Patch the backbone to work on CPU by replacing its forward method
    """
    # Replace the forward method
    backbone.forward = types.MethodType(create_cpu_compatible_forward(backbone), backbone)
    
    # Also patch the compute_semantic_logits method if it uses device
    original_compute_sem = backbone.compute_semantic_logits
    
    def cpu_compute_semantic_logits(self, features, masks):
        batch_size = features.shape[0]
        sem_logits = []
        
        for b in range(batch_size):
            valid_mask = ~masks[b]
            if valid_mask.sum() > 0:
                valid_features = features[b][valid_mask]
                logits = self.sem_head(valid_features)
            else:
                # Create on CPU instead of specifying device
                logits = torch.zeros(0, self.num_classes)
            
            # Pad back to full size
            full_logits = torch.zeros(features.shape[1], self.num_classes)
            if valid_mask.sum() > 0:
                full_logits[valid_mask] = logits
            
            sem_logits.append(full_logits)
        
        return torch.stack(sem_logits)
    
    backbone.compute_semantic_logits = types.MethodType(cpu_compute_semantic_logits, backbone)
    
    return backbone


class CPUONNXWrapper(nn.Module):
    """
    ONNX export wrapper that ensures everything runs on CPU
    """
    def __init__(self, model):
        super().__init__()
        # Move model to CPU
        self.model = model.cpu()
        self.backbone = model.backbone.cpu()
        self.decoder = model.decoder.cpu()
        self.num_classes = model.num_classes
        
        # Patch the backbone to work on CPU
        self.backbone = patch_backbone_for_cpu(self.backbone)
        
        # Set to eval mode
        self.model.eval()
        self.backbone.eval()
        self.decoder.eval()
    
    def forward(self, point_coords, point_features):
        """
        Forward pass for ONNX export (batch size 1)
        
        Args:
            point_coords: [N, 3] point coordinates
            point_features: [N, 4] point features
            
        Returns:
            pred_logits: [Q, num_classes+1]
            pred_masks: [N, Q]
            sem_logits: [N, num_classes]
        """
        # Ensure everything is on CPU
        point_coords = point_coords.cpu() if hasattr(point_coords, 'cpu') else point_coords
        point_features = point_features.cpu() if hasattr(point_features, 'cpu') else point_features
        
        # Add batch dimension
        coords = point_coords.unsqueeze(0)
        feats = point_features.unsqueeze(0)
        
        # Convert to expected format
        x = {
            'pt_coord': [coords[0].detach().numpy()],
            'feats': [feats[0].detach().numpy()]
        }
        
        # Forward through patched backbone
        with torch.no_grad():
            ms_features, ms_coords, ms_masks, sem_logits = self.backbone(x)
            
            # Forward through decoder
            outputs, padding = self.decoder(ms_features, ms_coords, ms_masks)
        
        # Remove batch dimension
        pred_logits = outputs['pred_logits'].squeeze(0)
        pred_masks = outputs['pred_masks'].squeeze(0)
        sem_logits = sem_logits.squeeze(0)
        
        return pred_logits, pred_masks, sem_logits


def get_config():
    """Load configuration"""
    config_dir = Path(__file__).parent.parent / "config"
    
    model_cfg = edict(yaml.safe_load(open(config_dir / "model.yaml")))
    backbone_cfg = edict(yaml.safe_load(open(config_dir / "backbone.yaml")))
    decoder_cfg = edict(yaml.safe_load(open(config_dir / "decoder.yaml")))
    
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    
    cfg.TRAIN.BATCH_SIZE = 2
    cfg.TRAIN.WARMUP_STEPS = 500
    cfg.TRAIN.SUBSAMPLE = True
    cfg.TRAIN.AUG = True
    cfg.TRAIN.LR = 0.0001
    
    return cfg


@click.command()
@click.option('--checkpoint', '-c', required=True, help='Path to checkpoint file')
@click.option('--output', '-o', default=None, help='Output ONNX file path')
@click.option('--num_points', default=10000, help='Number of points (start small!)')
@click.option('--opset', default=11, help='ONNX opset version')
@click.option('--validate', is_flag=True, help='Validate exported model')
@click.option('--debug', is_flag=True, help='Enable debug mode')
def export_onnx(checkpoint, output, num_points, opset, validate, debug):
    """Export MaskPLS-DGCNN to ONNX with CPU compatibility"""
    
    print("="*60)
    print("CPU-Compatible DGCNN ONNX Export")
    print("="*60)
    
    # Set up paths
    if output is None:
        checkpoint_path = Path(checkpoint)
        output = checkpoint_path.parent / f"{checkpoint_path.stem}_cpu.onnx"
    
    print(f"\nCheckpoint: {checkpoint}")
    print(f"Output: {output}")
    print(f"Points: {num_points}")
    
    # Load config
    print("\nLoading configuration...")
    cfg = get_config()
    
    # Check for saved hparams
    checkpoint_dir = Path(checkpoint).parent
    hparams_file = checkpoint_dir.parent / 'hparams.yaml'
    if hparams_file.exists():
        print(f"Loading saved hyperparameters...")
        with open(hparams_file, 'r') as f:
            saved_cfg = edict(yaml.safe_load(f))
        cfg.update(saved_cfg)
    
    # Create model on CPU
    print("\nCreating model...")
    
    # Temporarily disable CUDA to force CPU initialization
    original_cuda_available = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    
    try:
        model = MaskPLSDGCNNFixed(cfg)
    finally:
        # Restore original CUDA availability check
        torch.cuda.is_available = original_cuda_available
    
    # Load checkpoint to CPU
    print("Loading checkpoint (CPU only)...")
    checkpoint_data = torch.load(checkpoint, map_location='cpu')
    
    if 'state_dict' in checkpoint_data:
        state_dict = checkpoint_data['state_dict']
    else:
        state_dict = checkpoint_data
    
    # Load weights
    model.load_state_dict(state_dict, strict=False)
    model = model.cpu()
    model.eval()
    
    # Move all parameters to CPU explicitly
    for param in model.parameters():
        param.data = param.data.cpu()
    
    print("✓ Model loaded on CPU")
    
    # Create wrapper
    print("\nCreating CPU-compatible wrapper...")
    onnx_model = CPUONNXWrapper(model)
    
    # Create dummy inputs on CPU
    dummy_coords = torch.randn(num_points, 3)
    dummy_features = torch.randn(num_points, 4)
    
    if debug:
        print("\nTesting forward pass on CPU...")
        try:
            with torch.no_grad():
                test_output = onnx_model(dummy_coords, dummy_features)
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
                (dummy_coords, dummy_features),
                str(output),
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=['point_coords', 'point_features'],
                output_names=['pred_logits', 'pred_masks', 'sem_logits'],
                dynamic_axes={
                    'point_coords': {0: 'num_points'},
                    'point_features': {0: 'num_points'},
                    'pred_masks': {0: 'num_points'},
                    'sem_logits': {0: 'num_points'}
                },
                verbose=debug
            )
        
        print(f"✓ Model exported to {output}")
        
    except Exception as e:
        print(f"✗ Export failed: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        
        print("\nTips:")
        print("1. Try reducing --num_points (e.g., 1000)")
        print("2. Use --debug flag for more info")
        print("3. Check that all model operations are ONNX-compatible")
        return
    
    # Validate if requested
    if validate:
        print("\nValidating ONNX model...")
        try:
            import onnxruntime as ort
            
            # Check model structure
            model_onnx = onnx.load(str(output))
            onnx.checker.check_model(model_onnx)
            print("✓ Model structure valid")
            
            # Test inference
            session = ort.InferenceSession(str(output))
            inputs = {
                'point_coords': dummy_coords.numpy(),
                'point_features': dummy_features.numpy()
            }
            outputs = session.run(None, inputs)
            print(f"✓ Inference successful")
            print(f"  Output shapes: {[o.shape for o in outputs]}")
            
            # Check file size
            file_size = Path(output).stat().st_size / (1024 * 1024)
            print(f"  File size: {file_size:.2f} MB")
            
        except Exception as e:
            print(f"✗ Validation failed: {e}")
    
    print("\n" + "="*60)
    print("Export completed!")
    print("="*60)
    
    print("\n📌 IMPORTANT: GPU Inference")
    print("-" * 30)
    print("Although export was done on CPU, the model WILL run on GPU!")
    print("\nExample GPU inference:")
    print("```python")
    print("import onnxruntime as ort")
    print("import numpy as np")
    print()
    print("# This automatically uses GPU if available")
    print("providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']")
    print(f"session = ort.InferenceSession('{output}', providers=providers)")
    print()
    print("# Model runs on GPU!")
    print("inputs = {")
    print("    'point_coords': np.random.randn(50000, 3).astype(np.float32),")
    print("    'point_features': np.random.randn(50000, 4).astype(np.float32)")
    print("}")
    print("outputs = session.run(None, inputs)")
    print("```")
    print("\nTo test GPU inference, run:")
    print(f"python inference_onnx_gpu.py {output} --compare")


if __name__ == "__main__":
    export_onnx()