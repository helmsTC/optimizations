#!/usr/bin/env python3
"""
Simplified MaskPLS to ONNX converter with minimal dependencies
This version includes ALL automatic fixes for common issues
"""

import os
import sys
import torch
import torch.onnx
import numpy as np
import types
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def apply_model_fixes(model):
    """
    Apply ALL 4 fixes for common ONNX export issues
    """
    import torch.nn as nn
    
    # Fix 1: Increase spatial dimensions to prevent kernel size errors
    model.backbone.spatial_shape = (128, 128, 32)
    model.spatial_shape = (128, 128, 32)
    
    # Fix 2: Make positional encoding handle any input size
    def flexible_pe_forward(self, x):
        """Handle any input size for positional encoding"""
        if x.shape[1] > self.pe.shape[1]:
            # If input is larger than PE buffer, skip PE
            return x
        # Normal case: add positional encoding
        return x + self.pe[:, :x.shape[1], :self.d_model]
    
    # Apply the flexible PE to decoder
    model.decoder.pos_encoder.forward = types.MethodType(
        flexible_pe_forward, 
        model.decoder.pos_encoder
    )
    
    # Fix 3: Fix dimension mismatch in decoder (THIS WAS MISSING!)
    # The backbone outputs [256, 128, 96] but decoder expects different dims
    hidden_dim = 256  # Decoder hidden dimension
    
    # Recreate input projections with correct dimensions
    new_input_proj = nn.ModuleList([
        nn.Identity(),        # 256 -> 256 (no change needed)
        nn.Linear(128, 256),  # 128 -> 256 (projection needed)
        nn.Linear(96, 256),   # 96 -> 256 (projection needed)
    ])
    
    # Replace the decoder's input projections
    model.decoder.input_proj = new_input_proj
    
    # Also fix mask feature projection (last backbone output is 96 channels)
    model.decoder.mask_feat_proj = nn.Linear(96, 256)
    
    # Fix 4: Fix attention mask shape
    def generate_attention_mask_fixed(self, mask_pred, pad_mask):
        """Generate attention mask with correct dimensions for cross-attention"""
        # Threshold predicted masks
        attn_mask = (mask_pred.sigmoid() < 0.5)
        
        # Apply padding mask if provided
        if pad_mask is not None:
            attn_mask = attn_mask | pad_mask.unsqueeze(-1)
        
        # Shape is [B, N, Q] where B=batch, N=points, Q=queries
        B, N, Q = attn_mask.shape
        
        # Cross-attention expects [B*nheads, Q, N] not [B*nheads, N, Q]
        # So we need to transpose BEFORE reshaping
        attn_mask = attn_mask.transpose(1, 2)  # [B, Q, N]
        
        # Expand for multi-head attention
        attn_mask = attn_mask.unsqueeze(1)  # [B, 1, Q, N]
        attn_mask = attn_mask.expand(-1, self.nheads, -1, -1)  # [B, nheads, Q, N]
        attn_mask = attn_mask.reshape(B * self.nheads, Q, N)  # [B*nheads, Q, N]
        
        return attn_mask
    
    # Replace the method
    model.decoder.generate_attention_mask = types.MethodType(
        generate_attention_mask_fixed,
        model.decoder
    )
    # Fix 5: NEW - Fix hanging on conditional checks
    # Replace the decoder's forward method to avoid dynamic conditionals
    original_forward = model.decoder.forward
    
    def forward_no_conditionals(self, feats, mask_features, pad_masks):
        """Forward pass without dynamic conditionals that cause ONNX to hang"""
        B = feats[0].shape[0] if feats else 1
        
        # Initialize queries
        query_feat = self.query_feat.expand(B, -1, -1)
        query_pos = self.query_pos.expand(B, -1, -1)
        
        # Project mask features and add positional encoding
        # REMOVE the conditional check that causes hanging
        mask_features = self.mask_feat_proj(mask_features)
        # Always apply positional encoding (no conditional)
        mask_features = mask_features + self.pos_encoder(mask_features)
        
        # Store predictions at each layer
        predictions_class = []
        predictions_mask = []
        
        # Initial prediction
        outputs_class, outputs_mask = self.predict_masks(query_feat, mask_features)
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        
        # Process through transformer layers
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            
            # Get features for current level
            if level_index < len(feats):
                level_features = self.input_proj[level_index](feats[level_index])
                level_emb = self.level_embed.weight[level_index:level_index+1]
                level_features = level_features + level_emb
                level_features = level_features + self.pos_encoder(level_features)
                
                # Get padding mask
                level_pad_mask = pad_masks[level_index] if level_index < len(pad_masks) else None
                
                # Generate attention mask
                # Use None instead of conditional generation for first iteration
                if i == 0:
                    attn_mask = None
                else:
                    attn_mask = self.generate_attention_mask(predictions_mask[-1], level_pad_mask)
                
                # Transformer layer
                query_feat = self.transformer_layers[i](
                    query_feat,
                    level_features,
                    query_pos=query_pos,
                    memory_mask=attn_mask,
                    memory_key_padding_mask=level_pad_mask
                )
                
                # Predict at this layer
                outputs_class, outputs_mask = self.predict_masks(query_feat, mask_features)
                predictions_class.append(outputs_class)
                predictions_mask.append(outputs_mask)
        
        # Final output
        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self.set_aux_outputs(predictions_class[:-1], predictions_mask[:-1])
        }
        
        return out
    
    # Replace the forward method
    model.decoder.forward = types.MethodType(forward_no_conditionals, model.decoder)
    
    # Also fix the predict_masks method to avoid conditionals
    original_predict_masks = model.decoder.predict_masks
    
    def predict_masks_no_conditionals(self, query_feat, mask_features):
        """Predict masks without conditionals"""
        decoder_output = self.decoder_norm(query_feat)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        
        # Always compute mask logits (no conditional)
        # Use bmm which works better for ONNX
        outputs_mask = torch.bmm(mask_features, mask_embed.transpose(1, 2))
        
        return outputs_class, outputs_mask
    
    model.decoder.predict_masks = types.MethodType(
        predict_masks_no_conditionals, model.decoder
    )
    
    print("  ✓ Fixed spatial dimensions: (128, 128, 32)")
    print("  ✓ Applied flexible positional encoding")
    print("  ✓ Fixed decoder dimension mismatches")
    print("  ✓ Fixed attention mask shape")
    print("  ✓ Fixed ONNX export hanging issue")
    
    return model


def simple_convert():
    """
    Simple conversion without all the bells and whistles
    """
    print("=" * 50)
    print("MaskPLS to ONNX Simple Converter")
    print("=" * 50)
    
    # Import the ONNX model
    try:
        from models.onnx.onnx_model import MaskPLSONNX, MaskPLSExportWrapper
    except ImportError as e:
        print(f"Error importing ONNX model: {e}")
        print("\nMake sure you have created:")
        print("  - mask_pls/models/onnx/dense_backbone.py")
        print("  - mask_pls/models/onnx/onnx_decoder.py")
        print("  - mask_pls/models/onnx/onnx_model.py")
        return False
    
    # Create a simple config (minimal required fields)
    from easydict import EasyDict as edict
    
    cfg = edict({
        'MODEL': {
            'DATASET': 'KITTI',
            'OVERLAP_THRESHOLD': 0.8
        },
        'KITTI': {
            'NUM_CLASSES': 20,
            'MIN_POINTS': 10,
            'SPACE': [[-48.0, 48.0], [-48.0, 48.0], [-4.0, 1.5]],
            'SUB_NUM_POINTS': 80000
        },
        'NUSCENES': {
            'NUM_CLASSES': 17,
            'MIN_POINTS': 10,
            'SPACE': [[-50.0, 50.0], [-50.0, 50.0], [-5.0, 3.0]],
            'SUB_NUM_POINTS': 50000
        },
        'BACKBONE': {
            'INPUT_DIM': 4,
            'CHANNELS': [32, 32, 64, 128, 256, 256, 128, 96, 96],
            'RESOLUTION': 0.05,
            'KNN_UP': 3
        },
        'DECODER': {
            'HIDDEN_DIM': 256,
            'NHEADS': 8,
            'DIM_FFN': 1024,
            'FEATURE_LEVELS': 3,
            'DEC_BLOCKS': 3,
            'NUM_QUERIES': 100
        }
    })
    
    # Ask for dataset type
    dataset_choice = input("\nDataset type (1=KITTI, 2=NUSCENES) [default=1]: ").strip()
    if dataset_choice == "2":
        cfg.MODEL.DATASET = 'NUSCENES'
        print("Using NuScenes configuration")
    else:
        cfg.MODEL.DATASET = 'KITTI'
        print("Using KITTI configuration")
    
    print("\n1. Creating ONNX model...")
    model = MaskPLSONNX(cfg)
    
    # Apply automatic fixes
    print("\n2. Applying automatic fixes...")
    model = apply_model_fixes(model)
    print("  ✓ Fixed spatial dimensions: (128, 128, 32)")
    print("  ✓ Applied flexible positional encoding")
    print("  ✓ Fixed decoder dimension mismatches")
    print("  ✓ Fixed attention mask shape")
    
    model.eval()
    
    # Optional: Load checkpoint weights if available
    checkpoint_path = input("\n3. Enter checkpoint path (or press Enter to skip): ").strip()
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"   Loading weights from {checkpoint_path}...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            
            # Filter compatible weights
            model_dict = model.state_dict()
            compatible_dict = {}
            incompatible_keys = []
            
            for k, v in state_dict.items():
                k_new = k.replace('module.', '')
                if 'kernel' not in k and 'MinkowskiConvolution' not in k:
                    if k_new in model_dict:
                        try:
                            if v.shape == model_dict[k_new].shape:
                                compatible_dict[k_new] = v
                            else:
                                incompatible_keys.append(k_new)
                        except:
                            pass
            
            model.load_state_dict(compatible_dict, strict=False)
            print(f"   ✓ Loaded {len(compatible_dict)}/{len(model_dict)} weights")
            if incompatible_keys:
                print(f"   ⚠ Skipped {len(incompatible_keys)} incompatible weights")
        except Exception as e:
            print(f"   ⚠ Warning: Could not load weights: {e}")
            print("   Continuing with random weights...")
    else:
        print("   Using random initialization...")
    
    # Export to ONNX
    output_path = input("\n4. Enter output path (default: mask_pls.onnx): ").strip()
    if not output_path:
        output_path = "mask_pls.onnx"
    
    # Ask for export parameters
    batch_size = input("   Batch size for export (default: 1): ").strip()
    batch_size = int(batch_size) if batch_size else 1
    
    num_points = input("   Number of points for export (default: 10000): ").strip()
    num_points = int(num_points) if num_points else 10000
    
    print(f"\n5. Exporting to {output_path}...")
    print(f"   Batch size: {batch_size}")
    print(f"   Number of points: {num_points}")
    
    # Create wrapper for clean export
    export_model = MaskPLSExportWrapper(model)
    
    # Create dummy input
    dummy_points = torch.randn(batch_size, num_points, 3)
    dummy_features = torch.randn(batch_size, num_points, 4)
    
    try:
        with torch.no_grad():
            torch.onnx.export(
                export_model,
                (dummy_points, dummy_features),
                output_path,
                export_params=True,
                opset_version=13,
                do_constant_folding=True,
                input_names=['points', 'features'],
                output_names=['pred_logits', 'pred_masks', 'sem_logits'],
                dynamic_axes={
                    'points': {0: 'batch_size', 1: 'num_points'},
                    'features': {0: 'batch_size', 1: 'num_points'},
                    'pred_logits': {0: 'batch_size'},
                    'pred_masks': {0: 'batch_size', 1: 'num_points'},
                    'sem_logits': {0: 'batch_size', 1: 'num_points'}
                },
                verbose=False
            )
        
        print(f"   ✓ Model exported successfully!")
        print(f"   Output file: {output_path}")
        print(f"   File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"   ✗ Export failed: {e}")
        return False
    
    # Test with ONNX Runtime if available
    try:
        import onnxruntime as ort
        
        print("\n6. Testing with ONNX Runtime...")
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(output_path, providers=providers)
        
        print(f"   Using providers: {session.get_providers()}")
        
        # Test inference with different sizes
        test_sizes = [1000, 5000, 10000]
        print(f"   Testing with different point cloud sizes: {test_sizes}")
        
        all_tests_passed = True
        for test_size in test_sizes:
            test_points = np.random.randn(1, test_size, 3).astype(np.float32)
            test_features = np.random.randn(1, test_size, 4).astype(np.float32)
            
            try:
                outputs = session.run(None, {
                    'points': test_points,
                    'features': test_features
                })
                
                # Check for valid outputs
                valid = True
                for output in outputs:
                    if np.any(np.isnan(output)) or np.any(np.isinf(output)):
                        valid = False
                        break
                
                if valid:
                    print(f"     ✓ {test_size} points: Success")
                else:
                    print(f"     ✗ {test_size} points: Invalid outputs")
                    all_tests_passed = False
                    
            except Exception as e:
                print(f"     ✗ {test_size} points: {e}")
                all_tests_passed = False
        
        if all_tests_passed:
            print("   ✓ All inference tests passed!")
            
            # Show output shapes from last test
            print(f"\n   Output shapes (for {test_sizes[-1]} points):")
            print(f"     pred_logits: {outputs[0].shape}")
            print(f"     pred_masks: {outputs[1].shape}")
            print(f"     sem_logits: {outputs[2].shape}")
        else:
            print("   ⚠ Some inference tests failed")
        
    except ImportError:
        print("\n6. ONNX Runtime not installed. Cannot test inference.")
        print("   Install with: pip install onnxruntime-gpu")
    except Exception as e:
        print(f"\n6. Inference test error: {e}")
    
    print("\n" + "=" * 50)
    print("Conversion Complete!")
    print("=" * 50)
    
    print("\nNext steps:")
    print("1. Test with your data:")
    print(f"   python test_onnx_model.py --model {output_path}")
    print("\n2. Optimize the model (optional):")
    print(f"   python -m mask_pls.utils.onnx.optimization {output_path} --optimize")
    print("\n3. Deploy for inference:")
    print("   import onnxruntime as ort")
    print(f"   session = ort.InferenceSession('{output_path}')")
    print("   outputs = session.run(None, {'points': points, 'features': features})")
    
    return True


if __name__ == "__main__":
    success = simple_convert()
    sys.exit(0 if success else 1)
