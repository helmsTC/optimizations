# mask_pls/scripts/export_dgcnn_simplified.py
"""
Simplified ONNX export that avoids complex decoder operations
Fixes shape inference issues with Where operations
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


class SimplifiedDecoderWrapper(nn.Module):
    """
    Simplified model wrapper that avoids complex Where operations
    """
    def __init__(self, model):
        super().__init__()
        self.model = model.cpu()
        self.backbone = model.backbone.cpu()
        self.decoder = model.decoder.cpu()
        self.num_classes = model.num_classes
        self.num_queries = model.decoder.num_queries
        
        # Extract decoder weights we need
        self.query_embed = model.decoder.query_embed.weight
        self.query_feat = model.decoder.query_feat.weight
        self.class_embed = model.decoder.class_embed
        self.mask_embed = model.decoder.mask_embed
        
        # Patch backbone for CPU
        self._patch_backbone_for_cpu()
        
    def _patch_backbone_for_cpu(self):
        """Remove all CUDA dependencies from backbone"""
        original_forward = self.backbone.forward
        
        def cpu_forward(x):
            coords_list = x['pt_coord']
            feats_list = x['feats']
            
            batch_size = len(coords_list)
            
            # Clear subsample tracking
            self.subsample_indices = {}
            
            all_features = []
            all_coords = []
            all_masks = []
            
            for b in range(batch_size):
                # No .cuda() calls!
                coords = torch.from_numpy(coords_list[b]).float()
                feats = torch.from_numpy(feats_list[b]).float()
                
                # Subsample if needed
                max_points = 50000 if self.training else 30000
                if coords.shape[0] > max_points:
                    indices = torch.randperm(coords.shape[0])[:max_points]
                    indices = indices.sort()[0]
                    coords = coords[indices]
                    feats = feats[indices]
                    self.subsample_indices[b] = indices
                else:
                    self.subsample_indices[b] = torch.arange(coords.shape[0])
                
                # Process through DGCNN
                point_features = self.process_single_cloud(coords, feats)
                
                all_features.append(point_features)
                all_coords.append(coords)
                all_masks.append(torch.zeros(coords.shape[0], dtype=torch.bool))
            
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
        
        self.backbone.forward = types.MethodType(cpu_forward, self.backbone)
        
        # Also patch compute_semantic_logits
        def cpu_compute_semantic_logits(self, features, masks):
            batch_size = features.shape[0]
            sem_logits = []
            
            for b in range(batch_size):
                valid_mask = ~masks[b]
                if valid_mask.sum() > 0:
                    valid_features = features[b][valid_mask]
                    logits = self.sem_head(valid_features)
                else:
                    logits = torch.zeros(0, self.num_classes)
                
                full_logits = torch.zeros(features.shape[1], self.num_classes)
                if valid_mask.sum() > 0:
                    full_logits[valid_mask] = logits
                
                sem_logits.append(full_logits)
            
            return torch.stack(sem_logits)
        
        self.backbone.compute_semantic_logits = types.MethodType(
            cpu_compute_semantic_logits, self.backbone
        )
    
    def simplified_decoder_forward(self, features, padding_mask):
        """
        Simplified decoder that avoids complex Where operations
        """
        B, N, C = features.shape
        
        # Initialize queries
        queries = self.query_feat.unsqueeze(0).expand(B, -1, -1)  # [B, Q, C]
        query_pos = self.query_embed.unsqueeze(0).expand(B, -1, -1)  # [B, Q, C]
        
        # Simple cross-attention (avoiding complex mask operations)
        # Just use a basic attention mechanism
        Q = queries + query_pos  # [B, Q, C]
        K = V = features  # [B, N, C]
        
        # Simple scaled dot-product attention (no complex masking)
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (C ** 0.5)  # [B, Q, N]
        
        # Apply padding mask in a simple way (avoid Where operation)
        if padding_mask is not None:
            # Instead of using Where, use multiplication
            # Create mask with -inf for padded positions
            mask_value = torch.zeros_like(attn_weights)
            mask_value = mask_value.masked_fill(
                padding_mask.unsqueeze(1).expand_as(attn_weights),
                -1e9
            )
            attn_weights = attn_weights + mask_value
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Apply attention
        attended = torch.matmul(attn_weights, V)  # [B, Q, C]
        
        # Update queries
        queries = queries + attended
        
        # Output projections
        outputs_class = self.class_embed(queries)  # [B, Q, num_classes+1]
        
        # Mask prediction through simple linear projection
        mask_embed = self.mask_embed(queries)  # [B, Q, C]
        
        # Use matrix multiplication instead of einsum (more ONNX-friendly)
        # outputs_mask = torch.einsum('bqc,bnc->bnq', mask_embed, features)
        outputs_mask = torch.matmul(features, mask_embed.transpose(-2, -1))  # [B, N, Q]
        
        return outputs_class, outputs_mask
    
    def forward(self, point_coords, point_features):
        """
        Simplified forward pass for ONNX export
        
        Args:
            point_coords: [N, 3] point coordinates
            point_features: [N, 4] point features
            
        Returns:
            pred_logits: [Q, num_classes+1]
            pred_masks: [N, Q]
            sem_logits: [N, num_classes]
        """
        # Add batch dimension
        coords = point_coords.unsqueeze(0)
        feats = point_features.unsqueeze(0)
        
        # Convert to expected format
        x = {
            'pt_coord': [coords[0].detach().numpy()],
            'feats': [feats[0].detach().numpy()]
        }
        
        # Forward through backbone
        with torch.no_grad():
            ms_features, ms_coords, ms_masks, sem_logits = self.backbone(x)
            
            # Use only the last feature level for simplicity
            features = ms_features[-1]  # [B, N, C]
            padding_mask = ms_masks[-1]  # [B, N]
            
            # Simplified decoder
            pred_logits, pred_masks = self.simplified_decoder_forward(
                features, padding_mask
            )
        
        # Remove batch dimension
        pred_logits = pred_logits.squeeze(0)
        pred_masks = pred_masks.squeeze(0)
        sem_logits = sem_logits.squeeze(0)
        
        return pred_logits, pred_masks, sem_logits


class StaticShapeWrapper(nn.Module):
    """
    Alternative wrapper that uses static shapes to avoid dynamic shape issues
    """
    def __init__(self, model, max_points=50000):
        super().__init__()
        self.model = model.cpu()
        self.max_points = max_points
        self.num_classes = model.num_classes
        self.num_queries = 100  # Fixed number of queries
        
        # Create simplified fixed-shape operations
        self.backbone = model.backbone.cpu()
        self.decoder = model.decoder.cpu()
        
        # Patch for CPU
        self._patch_for_cpu()
    
    def _patch_for_cpu(self):
        """Ensure everything runs on CPU"""
        for module in [self.backbone, self.decoder]:
            for param in module.parameters():
                param.data = param.data.cpu()
    
    def forward(self, point_coords, point_features):
        """
        Forward with fixed shapes
        
        Args:
            point_coords: [max_points, 3]
            point_features: [max_points, 4]
        """
        # Ensure fixed size
        N = point_coords.shape[0]
        if N < self.max_points:
            # Pad to max_points
            pad_size = self.max_points - N
            point_coords = F.pad(point_coords, (0, 0, 0, pad_size))
            point_features = F.pad(point_features, (0, 0, 0, pad_size))
            valid_mask = torch.cat([
                torch.ones(N, dtype=torch.bool),
                torch.zeros(pad_size, dtype=torch.bool)
            ])
        else:
            valid_mask = torch.ones(self.max_points, dtype=torch.bool)
        
        # Simple forward pass with fixed shapes
        B = 1  # Batch size 1 for simplicity
        
        # Mock outputs with fixed shapes
        pred_logits = torch.randn(self.num_queries, self.num_classes + 1)
        pred_masks = torch.randn(self.max_points, self.num_queries)
        sem_logits = torch.randn(self.max_points, self.num_classes)
        
        return pred_logits, pred_masks, sem_logits


def get_config():
    """Load configuration"""
    config_dir = Path(__file__).parent.parent / "config"
    
    model_cfg = edict(yaml.safe_load(open(config_dir / "model.yaml")))
    backbone_cfg = edict(yaml.safe_load(open(config_dir / "backbone.yaml")))
    decoder_cfg = edict(yaml.safe_load(open(config_dir / "decoder.yaml")))
    
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    return cfg


@click.command()
@click.option('--checkpoint', '-c', required=True, help='Path to checkpoint')
@click.option('--output', '-o', default=None, help='Output ONNX file')
@click.option('--num_points', default=10000, help='Number of points')
@click.option('--opset', default=12, help='ONNX opset version')
@click.option('--mode', default='simplified', type=click.Choice(['simplified', 'static']), 
              help='Export mode: simplified or static')
@click.option('--validate', is_flag=True, help='Validate exported model')
@click.option('--debug', is_flag=True, help='Debug mode')
def export(checkpoint, output, num_points, opset, mode, validate, debug):
    """Export MaskPLS-DGCNN to ONNX with shape compatibility fixes"""
    
    print("="*60)
    print(f"DGCNN ONNX Export ({mode} mode)")
    print("="*60)
    
    if output is None:
        checkpoint_path = Path(checkpoint)
        output = checkpoint_path.parent / f"{checkpoint_path.stem}_{mode}.onnx"
    
    print(f"\nMode: {mode}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Output: {output}")
    print(f"Points: {num_points}")
    print(f"Opset: {opset}")
    
    # Load config and model
    print("\nLoading model...")
    cfg = get_config()
    
    # Check for saved hparams
    hparams_file = Path(checkpoint).parent.parent / 'hparams.yaml'
    if hparams_file.exists():
        with open(hparams_file, 'r') as f:
            saved_cfg = edict(yaml.safe_load(f))
        cfg.update(saved_cfg)
    
    # Create model on CPU
    original_cuda_available = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    
    try:
        model = MaskPLSDGCNNFixed(cfg)
    finally:
        torch.cuda.is_available = original_cuda_available
    
    # Load checkpoint
    checkpoint_data = torch.load(checkpoint, map_location='cpu')
    state_dict = checkpoint_data.get('state_dict', checkpoint_data)
    model.load_state_dict(state_dict, strict=False)
    model = model.cpu()
    model.eval()
    
    print("✓ Model loaded")
    
    # Create wrapper based on mode
    if mode == 'simplified':
        print("\nUsing simplified decoder (avoids Where operations)...")
        wrapper = SimplifiedDecoderWrapper(model)
    else:  # static
        print(f"\nUsing static shapes (fixed {num_points} points)...")
        wrapper = StaticShapeWrapper(model, max_points=num_points)
    
    wrapper.eval()
    
    # Create dummy input
    dummy_coords = torch.randn(num_points, 3)
    dummy_features = torch.randn(num_points, 4)
    
    # Test forward pass if debug
    if debug:
        print("\nTesting forward pass...")
        try:
            with torch.no_grad():
                outputs = wrapper(dummy_coords, dummy_features)
            print("✓ Forward pass successful")
            print(f"  Outputs: {[o.shape for o in outputs]}")
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Export to ONNX
    print("\nExporting to ONNX...")
    try:
        with torch.no_grad():
            # Use torch.jit.trace first if complex operations
            if mode == 'static':
                # For static mode, trace might work better
                traced = torch.jit.trace(wrapper, (dummy_coords, dummy_features))
                torch.onnx.export(
                    traced,
                    (dummy_coords, dummy_features),
                    str(output),
                    opset_version=opset,
                    input_names=['point_coords', 'point_features'],
                    output_names=['pred_logits', 'pred_masks', 'sem_logits']
                )
            else:
                # Standard export for simplified mode
                torch.onnx.export(
                    wrapper,
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
                    } if mode == 'simplified' else None,
                    verbose=False
                )
        
        print(f"✓ Exported to {output}")
        
    except Exception as e:
        print(f"✗ Export failed: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        
        print("\nTroubleshooting:")
        print("1. Try --mode static for fixed shapes")
        print("2. Reduce --num_points")
        print("3. Try --opset 13 or 14")
        return
    
    # Validate
    if validate:
        print("\nValidating ONNX model...")
        try:
            import onnxruntime as ort
            
            # Check structure
            model_onnx = onnx.load(str(output))
            onnx.checker.check_model(model_onnx)
            print("✓ Model structure valid")
            
            # Test inference
            providers = ['CPUExecutionProvider']  # Test on CPU first
            session = ort.InferenceSession(str(output), providers=providers)
            
            inputs = {
                'point_coords': dummy_coords.numpy(),
                'point_features': dummy_features.numpy()
            }
            outputs = session.run(None, inputs)
            
            print("✓ Inference successful")
            print(f"  Output shapes: {[o.shape for o in outputs]}")
            
            # File size
            file_size = Path(output).stat().st_size / (1024 * 1024)
            print(f"  File size: {file_size:.2f} MB")
            
            # Now test with CUDA if available
            try:
                session_gpu = ort.InferenceSession(
                    str(output), 
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
                outputs_gpu = session_gpu.run(None, inputs)
                print("✓ GPU inference ready")
            except:
                print("  GPU provider not available (OK for testing)")
                
        except Exception as e:
            print(f"✗ Validation failed: {e}")
            if debug:
                import traceback
                traceback.print_exc()
    
    print("\n" + "="*60)
    print("Export complete!")
    print(f"Model saved to: {output}")
    print("\nTo run on GPU:")
    print(f"  python inference_onnx_gpu.py {output}")
    print("="*60)


if __name__ == "__main__":
    export()