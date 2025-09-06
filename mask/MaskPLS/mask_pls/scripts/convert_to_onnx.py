#!/usr/bin/env python3
"""
Fixed ONNX export script for MaskPLS with DGCNN backbone
Addresses shape inference errors in decoder
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import onnx
import yaml
from pathlib import Path
from easydict import EasyDict as edict
import warnings
warnings.filterwarnings("ignore")

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from mask_pls.models.dgcnn.dgcnn_backbone_efficient import FixedDGCNNBackbone
from mask_pls.models.decoder import MaskedTransformerDecoder
from mask_pls.models.loss import MaskLoss, SemLoss
from mask_pls.utils.evaluate_panoptic import PanopticEvaluator

# Import the model
from mask_pls.models.dgcnn.maskpls_dgcnn_optimized import MaskPLSDGCNNFixed


class ONNXCompatibleDecoder(MaskedTransformerDecoder):
    """Decoder with ONNX-compatible operations"""
    
    def pred_heads(self, output, mask_features, pad_mask=None):
        """Modified pred_heads to avoid shape inference issues"""
        decoder_output = self.decoder_norm(output)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bpc->bpq", mask_embed, mask_features)

        # Create attention mask with ONNX-compatible operations
        attn_mask = (outputs_mask.sigmoid() < 0.5).detach()
        
        if pad_mask is not None:
            # Use indexing instead of advanced indexing
            B, N = pad_mask.shape
            Q = attn_mask.shape[2]
            
            # Expand pad_mask to match attn_mask dimensions
            pad_mask_expanded = pad_mask.unsqueeze(2).expand(B, N, Q)
            attn_mask = attn_mask | pad_mask_expanded
        
        # ONNX-compatible reshaping for attention mask
        attn_mask = attn_mask.unsqueeze(1).expand(-1, self.nheads, -1, -1)
        attn_mask = attn_mask.reshape(B * self.nheads, N, Q)
        attn_mask = attn_mask.permute(0, 2, 1)

        return outputs_class, outputs_mask, attn_mask


class ONNXExportWrapper(nn.Module):
    """Wrapper that ensures ONNX compatibility"""
    
    def __init__(self, model, num_points=10000):
        super().__init__()
        self.model = model
        self.num_points = num_points
        self.num_queries = model.decoder.num_queries
        self.num_classes = model.num_classes
        
        # Replace decoder with ONNX-compatible version
        self.replace_decoder()
        
    def replace_decoder(self):
        """Replace the decoder with ONNX-compatible version"""
        # Copy decoder configuration
        cfg = self.model.cfg
        dataset = cfg.MODEL.DATASET
        
        # Create new decoder
        new_decoder = ONNXCompatibleDecoder(
            cfg.DECODER,
            cfg.BACKBONE,
            cfg[dataset]
        )
        
        # Copy weights
        new_decoder.load_state_dict(self.model.decoder.state_dict())
        
        # Replace decoder
        self.model.decoder = new_decoder
        
    def forward(self, coords, feats):
        """
        Forward pass with fixed shapes
        Args:
            coords: [B, N, 3] point coordinates
            feats: [B, N, 4] point features (XYZI)
        Returns:
            pred_logits: [B, Q, C+1]
            pred_masks: [B, N, Q]
            sem_logits: [B, N, C]
        """
        B, N, _ = coords.shape
        
        # Ensure fixed number of points
        if N != self.num_points:
            # Pad or truncate
            if N < self.num_points:
                pad_n = self.num_points - N
                coords = F.pad(coords, (0, 0, 0, pad_n), value=0)
                feats = F.pad(feats, (0, 0, 0, pad_n), value=0)
                # Create mask for valid points
                valid_mask = torch.cat([
                    torch.ones(B, N, dtype=torch.bool),
                    torch.zeros(B, pad_n, dtype=torch.bool)
                ], dim=1)
            else:
                coords = coords[:, :self.num_points]
                feats = feats[:, :self.num_points]
                valid_mask = torch.ones(B, self.num_points, dtype=torch.bool)
        else:
            valid_mask = torch.ones(B, N, dtype=torch.bool)
        
        # Convert to list format expected by the model
        coords_list = []
        feats_list = []
        
        for b in range(B):
            # Get valid points for this sample
            valid_b = valid_mask[b]
            coords_b = coords[b][valid_b].detach().cpu().numpy()
            feats_b = feats[b][valid_b].detach().cpu().numpy()
            
            coords_list.append(coords_b)
            feats_list.append(feats_b)
        
        # Create batch dictionary
        batch = {
            'pt_coord': coords_list,
            'feats': feats_list
        }
        
        # Forward through model
        outputs, padding, sem_logits = self.model(batch)
        
        # Get outputs
        pred_logits = outputs['pred_logits']  # [B, Q, C+1]
        pred_masks = outputs['pred_masks']    # [B, N_variable, Q]
        
        # Ensure fixed output dimensions
        B_out = pred_logits.shape[0]
        Q = self.num_queries
        C = self.num_classes + 1
        
        # Fix pred_masks to have exactly num_points
        N_mask = pred_masks.shape[1]
        if N_mask != self.num_points:
            if N_mask < self.num_points:
                pad_n = self.num_points - N_mask
                pred_masks = F.pad(pred_masks, (0, 0, 0, pad_n), value=0)
            else:
                pred_masks = pred_masks[:, :self.num_points, :]
        
        # Fix sem_logits similarly
        N_sem = sem_logits.shape[1]
        if N_sem != self.num_points:
            if N_sem < self.num_points:
                pad_n = self.num_points - N_sem
                sem_logits = F.pad(sem_logits, (0, 0, 0, pad_n), value=0)
            else:
                sem_logits = sem_logits[:, :self.num_points, :]
        
        # Ensure correct shapes
        pred_logits = pred_logits.reshape(B, Q, C)
        pred_masks = pred_masks.reshape(B, self.num_points, Q)
        sem_logits = sem_logits.reshape(B, self.num_points, self.num_classes)
        
        return pred_logits, pred_masks, sem_logits


def trace_and_export(model, output_path, batch_size=1, num_points=10000):
    """Export model using torch.jit.trace for better compatibility"""
    print("\nUsing torch.jit.trace for export...")
    
    # Create dummy inputs
    dummy_coords = torch.randn(batch_size, num_points, 3)
    dummy_feats = torch.randn(batch_size, num_points, 4)
    
    # Trace the model
    try:
        with torch.no_grad():
            traced_model = torch.jit.trace(model, (dummy_coords, dummy_feats))
        
        # Export to ONNX
        torch.onnx.export(
            traced_model,
            (dummy_coords, dummy_feats),
            output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['coords', 'features'],
            output_names=['pred_logits', 'pred_masks', 'sem_logits'],
            dynamic_axes={
                'coords': {0: 'batch_size'},
                'features': {0: 'batch_size'},
                'pred_logits': {0: 'batch_size'},
                'pred_masks': {0: 'batch_size'},
                'sem_logits': {0: 'batch_size'}
            },
            verbose=False
        )
        return True
    except Exception as e:
        print(f"Trace export failed: {e}")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fixed ONNX Export for MaskPLS")
    parser.add_argument('checkpoint', help='Path to checkpoint')
    parser.add_argument('--output', '-o', help='Output ONNX file', 
                       default='maskpls_dgcnn_fixed.onnx')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--num-points', type=int, default=10000, help='Fixed number of points')
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true', help='Simplify model after export')
    parser.add_argument('--use-trace', action='store_true', help='Use torch.jit.trace')
    
    args = parser.parse_args()
    
    print("="*60)
    print("MaskPLS DGCNN ONNX Export (Fixed)")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num points: {args.num_points}")
    print(f"Opset: {args.opset}")
    
    # Load configuration
    print("\n1. Loading configuration...")
    config_dir = Path(__file__).parent.parent / "config"
    cfg = {}
    
    for cfg_file in ['model.yaml', 'backbone.yaml', 'decoder.yaml']:
        cfg_path = config_dir / cfg_file
        if cfg_path.exists():
            with open(cfg_path, 'r') as f:
                cfg.update(yaml.safe_load(f))
        else:
            print(f"Warning: {cfg_file} not found")
    
    cfg = edict(cfg)
    
    # Create model
    print("\n2. Creating model...")
    model = MaskPLSDGCNNFixed(cfg)
    
    # Load checkpoint
    print(f"\n3. Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"Unexpected keys: {len(unexpected_keys)}")
    
    # IMPORTANT: Move model to CPU and set to eval mode
    model = model.cpu()
    model.eval()
    
    # Ensure no gradients are computed
    for param in model.parameters():
        param.requires_grad = False
    
    # Create wrapper
    print("\n4. Creating ONNX-compatible wrapper...")
    wrapped_model = ONNXExportWrapper(model, num_points=args.num_points)
    wrapped_model.eval()
    
    # Test forward pass
    print("\n5. Testing forward pass...")
    dummy_coords = torch.randn(args.batch_size, args.num_points, 3)
    dummy_feats = torch.randn(args.batch_size, args.num_points, 4)
    
    try:
        with torch.no_grad():
            outputs = wrapped_model(dummy_coords, dummy_feats)
        print("✓ Forward pass successful")
        for i, out in enumerate(outputs):
            print(f"  Output {i}: {out.shape}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Export to ONNX
    print(f"\n6. Exporting to ONNX (opset {args.opset})...")
    
    if args.use_trace:
        success = trace_and_export(wrapped_model, args.output, args.batch_size, args.num_points)
    else:
        try:
            torch.onnx.export(
                wrapped_model,
                (dummy_coords, dummy_feats),
                args.output,
                export_params=True,
                opset_version=args.opset,
                do_constant_folding=True,
                input_names=['coords', 'features'],
                output_names=['pred_logits', 'pred_masks', 'sem_logits'],
                dynamic_axes={
                    'coords': {0: 'batch_size'},
                    'features': {0: 'batch_size'},
                    'pred_logits': {0: 'batch_size'},
                    'pred_masks': {0: 'batch_size'},
                    'sem_logits': {0: 'batch_size'}
                },
                verbose=False,
                operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
            )
            success = True
            print("✓ Export successful")
        except Exception as e:
            print(f"✗ Export failed: {e}")
            success = False
            import traceback
            traceback.print_exc()
    
    if not success:
        return
    
    # Verify ONNX model
    print("\n7. Verifying ONNX model...")
    try:
        onnx_model = onnx.load(args.output)
        onnx.checker.check_model(onnx_model)
        print("✓ Model structure valid")
        
        # Print model info
        print(f"\nModel info:")
        print(f"  IR version: {onnx_model.ir_version}")
        print(f"  Opset version: {onnx_model.opset_import[0].version}")
        print(f"  Producer: {onnx_model.producer_name} {onnx_model.producer_version}")
        print(f"  Graph nodes: {len(onnx_model.graph.node)}")
        
        # File size
        size_mb = Path(args.output).stat().st_size / 1024 / 1024
        print(f"  File size: {size_mb:.2f} MB")
        
    except Exception as e:
        print(f"✗ Verification failed: {e}")
    
    # Simplify if requested
    if args.simplify:
        print("\n8. Simplifying model...")
        try:
            import onnxsim
            
            model_simp, check = onnxsim.simplify(
                onnx_model,
                input_shapes={
                    'coords': (args.batch_size, args.num_points, 3),
                    'features': (args.batch_size, args.num_points, 4)
                }
            )
            
            if check:
                output_simp = args.output.replace('.onnx', '_simplified.onnx')
                onnx.save(model_simp, output_simp)
                print(f"✓ Saved simplified model to: {output_simp}")
                
                # Update output path
                args.output = output_simp
            else:
                print("✗ Simplification check failed")
        except ImportError:
            print("⚠ onnx-simplifier not installed")
        except Exception as e:
            print(f"✗ Simplification failed: {e}")
    
    # Test with ONNX Runtime
    print("\n9. Testing with ONNX Runtime...")
    try:
        import onnxruntime as ort
        
        # Create session with detailed error info
        so = ort.SessionOptions()
        so.log_severity_level = 1  # Verbose logging
        
        providers = ['CPUExecutionProvider']
        session = ort.InferenceSession(args.output, so, providers=providers)
        print("✓ Model loaded successfully")
        
        # Print input/output info
        print("\nModel interface:")
        print("  Inputs:")
        for inp in session.get_inputs():
            print(f"    {inp.name}: {inp.shape} ({inp.type})")
        
        print("  Outputs:")
        for out in session.get_outputs():
            print(f"    {out.name}: {out.shape} ({out.type})")
        
        # Test inference
        print("\nTesting inference...")
        test_coords = dummy_coords.numpy()
        test_feats = dummy_feats.numpy()
        
        outputs = session.run(None, {
            'coords': test_coords,
            'features': test_feats
        })
        
        print("✓ Inference successful")
        for i, out in enumerate(outputs):
            print(f"  Output {i}: shape={out.shape}, dtype={out.dtype}")
            print(f"    Range: [{out.min():.3f}, {out.max():.3f}]")
        
    except ImportError:
        print("⚠ ONNX Runtime not installed")
        print("  Install with: pip install onnxruntime")
    except Exception as e:
        print(f"✗ Runtime test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✅ Done! Model exported to: {args.output}")


if __name__ == "__main__":
    main()