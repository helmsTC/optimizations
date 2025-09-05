# mask_pls/scripts/export_efficient_dgcnn_onnx_final.py
"""
Final fixed ONNX export for MaskPLS-DGCNN
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path
import click
import yaml
from easydict import EasyDict as edict

sys.path.append(str(Path(__file__).parent.parent))

from mask_pls.models.dgcnn.maskpls_dgcnn_optimized import MaskPLSDGCNNFixed


class SimplifiedSemanticModel(nn.Module):
    """
    Simplified model for semantic segmentation only
    Properly handles the DGCNN architecture
    """
    def __init__(self, original_model):
        super().__init__()
        
        backbone = original_model.backbone
        
        # Copy all edge convolution layers
        self.edge_conv1 = backbone.edge_conv1
        self.edge_conv2 = backbone.edge_conv2
        self.edge_conv3 = backbone.edge_conv3
        self.edge_conv4 = backbone.edge_conv4
        self.conv5 = backbone.conv5
        
        # Output layers
        self.feat_layer = backbone.feat_layers[-1]  # Last feature layer
        self.out_bn = backbone.out_bn[-1]  # Last batch norm
        self.sem_head = backbone.sem_head
        self.num_classes = backbone.num_classes
        self.k = backbone.k
        
        self.eval()
    
    def forward(self, points, features):
        """
        Forward pass for semantic segmentation only
        Args:
            points: [N, 3]
            features: [N, 1] intensity
        Returns:
            sem_logits: [N, num_classes]
        """
        N = points.shape[0]
        
        # Subsample if needed
        max_points = 30000
        if N > max_points:
            stride = N // max_points
            indices = torch.arange(0, N, stride)[:max_points]
            points = points[indices]
            features = features[indices]
            N = points.shape[0]
        
        # Prepare input: [1, 4, N] (xyz + intensity)
        if features.dim() == 1:
            features = features.unsqueeze(1)
        x = torch.cat([points, features], dim=1).T.unsqueeze(0)
        
        # Import graph feature function
        from mask_pls.models.dgcnn.dgcnn_backbone_efficient_fixed import get_graph_feature
        
        # Process through edge convolutions
        # Conv1: 4 -> 64
        x1 = get_graph_feature(x, k=self.k)
        x1 = self.edge_conv1(x1)
        x1 = x1.max(dim=-1, keepdim=False)[0]
        
        # Conv2: 64 -> 64
        x2 = get_graph_feature(x1, k=self.k)
        x2 = self.edge_conv2(x2)
        x2 = x2.max(dim=-1, keepdim=False)[0]
        
        # Conv3: 64 -> 128
        x3 = get_graph_feature(x2, k=self.k)
        x3 = self.edge_conv3(x3)
        x3 = x3.max(dim=-1, keepdim=False)[0]
        
        # Conv4: 128 -> 256
        x4 = get_graph_feature(x3, k=self.k)
        x4 = self.edge_conv4(x4)
        x4 = x4.max(dim=-1, keepdim=False)[0]
        
        # Concatenate: 64+64+128+256 = 512
        x = torch.cat((x1, x2, x3, x4), dim=1)
        
        # Conv5: 512 -> 512
        x = self.conv5(x)
        
        # Project to final feature dimension
        feat = self.feat_layer(x)
        feat = self.out_bn(feat)
        feat = feat.squeeze(0).T  # [N, C]
        
        # Semantic head
        sem_logits = self.sem_head(feat)
        
        return sem_logits


class CPUBackboneWrapper(nn.Module):
    """
    Wrapper that patches the backbone's forward method to work on CPU for ONNX export
    """
    def __init__(self, original_backbone):
        super().__init__()
        # Copy all attributes from original backbone
        for name, module in original_backbone.named_children():
            self.add_module(name, module)
        
        # Copy non-module attributes
        for attr_name in dir(original_backbone):
            if not attr_name.startswith('_') and not hasattr(self, attr_name):
                attr = getattr(original_backbone, attr_name)
                if not callable(attr) and not isinstance(attr, nn.Module):
                    setattr(self, attr_name, attr)
        
        self.eval()
    
    def forward(self, x):
        """CPU-compatible version of backbone forward"""
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
            # Use CPU instead of hardcoded CUDA
            device = next(self.parameters()).device
            coords = torch.from_numpy(coords_list[b]).float().to(device)
            feats = torch.from_numpy(feats_list[b]).float().to(device)
            
            # Subsample if needed
            max_points = 30000  # Use smaller number for export
            if coords.shape[0] > max_points:
                indices = torch.randperm(coords.shape[0], device=coords.device)[:max_points]
                indices = indices.sort()[0]
                coords = coords[indices]
                feats = feats[indices]
                self.subsample_indices[b] = indices
            else:
                self.subsample_indices[b] = torch.arange(coords.shape[0], device=coords.device)
            
            # Process through DGCNN
            N = coords.shape[0]
            
            # Prepare input: [1, 4, N] (xyz + intensity)
            if feats.dim() == 1:
                feats = feats.unsqueeze(1)
            x_input = torch.cat([coords, feats], dim=1).T.unsqueeze(0)
            
            # Import graph feature function
            from mask_pls.models.dgcnn.dgcnn_backbone_efficient import get_graph_feature
            
            # Process through edge convolutions
            # Conv1: 4 -> 64
            x1 = get_graph_feature(x_input, k=self.k)
            x1 = self.edge_conv1(x1)
            x1 = x1.max(dim=-1, keepdim=False)[0]
            
            # Conv2: 64 -> 64
            x2 = get_graph_feature(x1, k=self.k)
            x2 = self.edge_conv2(x2)
            x2 = x2.max(dim=-1, keepdim=False)[0]
            
            # Conv3: 64 -> 128
            x3 = get_graph_feature(x2, k=self.k)
            x3 = self.edge_conv3(x3)
            x3 = x3.max(dim=-1, keepdim=False)[0]
            
            # Conv4: 128 -> 256
            x4 = get_graph_feature(x3, k=self.k)
            x4 = self.edge_conv4(x4)
            x4 = x4.max(dim=-1, keepdim=False)[0]
            
            # Concatenate: 64+64+128+256 = 512
            feat_concat = torch.cat((x1, x2, x3, x4), dim=1)
            
            # Conv5: 512 -> 512
            x5 = self.conv5(feat_concat)
            
            # Multi-scale features
            ms_features_b = []
            ms_coords_b = []
            ms_masks_b = []
            
            # Create features at different scales
            for i, (feat_layer, out_bn) in enumerate(zip(self.feat_layers, self.out_bn)):
                feat = feat_layer(x5)
                feat = out_bn(feat)
                ms_features_b.append(feat)
                ms_coords_b.append(coords.unsqueeze(0))  # Add batch dim
                ms_masks_b.append(torch.zeros(1, N, dtype=torch.bool, device=device))
            
            all_features.append(ms_features_b)
            all_coords.append(ms_coords_b)
            all_masks.append(ms_masks_b)
            
            # Semantic segmentation
            if hasattr(self, 'sem_head'):
                sem_feat = self.feat_layers[-1](x5)
                sem_feat = self.out_bn[-1](sem_feat)
                sem_feat = sem_feat.squeeze(0).T  # [N, C]
                sem_logits = self.sem_head(sem_feat).unsqueeze(0)  # Add batch dim
        
        # Combine batch results
        ms_features = []
        ms_coords = []
        ms_masks = []
        
        # Transpose from [batch][scale] to [scale][batch]
        num_scales = len(all_features[0])
        for scale in range(num_scales):
            scale_features = torch.cat([all_features[b][scale] for b in range(batch_size)], dim=0)
            scale_coords = torch.cat([all_coords[b][scale] for b in range(batch_size)], dim=0)
            scale_masks = torch.cat([all_masks[b][scale] for b in range(batch_size)], dim=0)
            
            ms_features.append(scale_features)
            ms_coords.append(scale_coords)
            ms_masks.append(scale_masks)
        
        return ms_features, ms_coords, ms_masks, sem_logits


class FullModelWithoutEinsum(nn.Module):
    """
    Full model that replaces einsum with matmul for ONNX compatibility
    """
    def __init__(self, original_model):
        super().__init__()
        
        self.num_classes = original_model.num_classes
        # Use CPU-compatible backbone wrapper
        self.backbone = CPUBackboneWrapper(original_model.backbone)
        
        # Copy decoder but we'll override the mask computation
        self.decoder = original_model.decoder
        
        self.eval()
    
    def forward(self, point_coords, point_features):
        """
        Forward pass with einsum replaced
        Args:
            point_coords: [N, 3]
            point_features: [N, 4]
        Returns:
            pred_logits: [Q, num_classes+1]
            pred_masks: [N, Q]
            sem_logits: [N, num_classes]
        """
        # Ensure tensors are on CPU for ONNX export
        point_coords = point_coords.cpu()
        point_features = point_features.cpu()
        
        # Create batch format expected by backbone
        x = {
            'pt_coord': [point_coords.detach().cpu().numpy()],
            'feats': [point_features.detach().cpu().numpy()]
        }
        
        # Process through backbone
        with torch.no_grad():
            ms_features, ms_coords, ms_masks, sem_logits = self.backbone(x)
            
            # Process through decoder layers manually
            # to replace einsum operation
            outputs = self.forward_decoder_without_einsum(
                ms_features, ms_coords, ms_masks
            )
        
        # Remove batch dimension
        pred_logits = outputs['pred_logits'].squeeze(0)
        pred_masks = outputs['pred_masks'].squeeze(0)
        sem_logits = sem_logits.squeeze(0)
        
        return pred_logits, pred_masks, sem_logits
    
    def forward_decoder_without_einsum(self, feats, coords, pad_masks):
        """
        Decoder forward pass with einsum replaced by matmul
        """
        decoder = self.decoder
        
        # Process features
        last_coords = coords.pop()
        mask_features = decoder.mask_feat_proj(feats.pop()) + decoder.pe_layer(last_coords)
        last_pad = pad_masks.pop()
        
        src = []
        pos = []
        
        for i in range(decoder.num_feature_levels):
            pos.append(decoder.pe_layer(coords[i]))
            feat = decoder.input_proj[i](feats[i])
            src.append(feat)
        
        bs = src[0].shape[0]
        query_embed = decoder.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        output = decoder.query_feat.weight.unsqueeze(0).repeat(bs, 1, 1)
        
        # Process through transformer layers
        for i in range(decoder.num_layers):
            level_index = i % decoder.num_feature_levels
            
            output = decoder.transformer_cross_attention_layers[i](
                output,
                src[level_index],
                padding_mask=pad_masks[level_index],
                pos=pos[level_index],
                query_pos=query_embed,
            )
            output = decoder.transformer_self_attention_layers[i](
                output, query_pos=query_embed
            )
            output = decoder.transformer_ffn_layers[i](output)
        
        # Final predictions
        decoder_output = decoder.decoder_norm(output)
        outputs_class = decoder.class_embed(decoder_output)
        mask_embed = decoder.mask_embed(decoder_output)
        
        # Replace einsum 'bqc,bpc->bpq' with equivalent operations
        # Original: torch.einsum("bqc,bpc->bpq", mask_embed, mask_features)
        # This computes: sum_c(mask_embed[b,q,c] * mask_features[b,p,c])
        
        # Equivalent using transpose and matmul:
        # mask_embed: [B, Q, C]
        # mask_features: [B, P, C]
        # Result: [B, P, Q]
        outputs_mask = torch.matmul(mask_features, mask_embed.transpose(-2, -1))
        
        return {
            'pred_logits': outputs_class,
            'pred_masks': outputs_mask
        }


def load_config(checkpoint_dir):
    """Load configuration"""
    config_dir = Path(__file__).parent.parent / "config"
    
    model_cfg = edict(yaml.safe_load(open(config_dir / "model.yaml")))
    backbone_cfg = edict(yaml.safe_load(open(config_dir / "backbone.yaml")))
    decoder_cfg = edict(yaml.safe_load(open(config_dir / "decoder.yaml")))
    
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    
    hparams_file = checkpoint_dir.parent / 'hparams.yaml'
    if hparams_file.exists():
        with open(hparams_file, 'r') as f:
            saved_cfg = yaml.safe_load(f)
            if saved_cfg:
                cfg.update(saved_cfg)
    
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
@click.option('--checkpoint', '-c', required=True, help='Path to checkpoint')
@click.option('--output', '-o', default=None, help='Output ONNX path')
@click.option('--num-points', '-n', default=10000, help='Number of test points')
@click.option('--opset', default=12, help='ONNX opset version (12+ for full model)')
@click.option('--mode', type=click.Choice(['semantic', 'full']), default='semantic',
              help='Export mode: semantic only or full model')
@click.option('--validate', is_flag=True, help='Validate after export')
def export_onnx(checkpoint, output, num_points, opset, mode, validate):
    """Export MaskPLS-DGCNN to ONNX - Final Fixed Version"""
    
    print("="*60)
    print("MaskPLS-DGCNN ONNX Export (Final)")
    print("="*60)
    
    # Check opset for full model
    if mode == 'full' and opset < 12:
        print(f"⚠ Warning: Full model requires opset 12+ (using {opset})")
        print("  The model uses einsum which is supported from opset 12")
        opset = 12
        print(f"  Automatically setting opset to {opset}")
    
    # Output path
    if output is None:
        checkpoint_path = Path(checkpoint)
        suffix = "_semantic.onnx" if mode == 'semantic' else "_full.onnx"
        output = checkpoint_path.parent / f"{checkpoint_path.stem}{suffix}"
    
    print(f"Checkpoint: {checkpoint}")
    print(f"Output: {output}")
    print(f"Mode: {mode}")
    print(f"Opset: {opset}")
    
    # Load config
    print("\nLoading configuration...")
    cfg = load_config(Path(checkpoint).parent)
    
    # Create and load model
    print("Creating model...")
    model = MaskPLSDGCNNFixed(cfg)
    
    print("Loading checkpoint...")
    checkpoint_data = torch.load(checkpoint, map_location='cpu')
    
    if 'state_dict' in checkpoint_data:
        state_dict = checkpoint_data['state_dict']
        print(f"  Epoch: {checkpoint_data.get('epoch', 'unknown')}")
    else:
        state_dict = checkpoint_data
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # Move model to CPU for ONNX export to avoid device mismatches
    model = model.cpu()
    
    # Create ONNX model based on mode
    if mode == 'semantic':
        print("\nCreating semantic segmentation model...")
        onnx_model = SimplifiedSemanticModel(model)
        
        # Test inputs
        dummy_points = torch.randn(num_points, 3)
        dummy_intensity = torch.randn(num_points, 1)
        
        input_names = ['points', 'intensity']
        output_names = ['sem_logits']
        
        dynamic_axes = {
            'points': {0: 'num_points'},
            'intensity': {0: 'num_points'},
            'sem_logits': {0: 'num_points'}
        }
        
        dummy_input = (dummy_points, dummy_intensity)
        
    else:  # full
        print("\nCreating full model (with instance segmentation)...")
        onnx_model = FullModelWithoutEinsum(model)
        
        # Test inputs
        dummy_points = torch.randn(num_points, 3)
        dummy_features = torch.randn(num_points, 4)
        
        input_names = ['point_coords', 'point_features']
        output_names = ['pred_logits', 'pred_masks', 'sem_logits']
        
        dynamic_axes = {
            'point_coords': {0: 'num_points'},
            'point_features': {0: 'num_points'},
            'pred_masks': {0: 'num_points'},
            'sem_logits': {0: 'num_points'}
        }
        
        dummy_input = (dummy_points, dummy_features)
    
    # Test forward pass
    print("\nTesting forward pass...")
    try:
        with torch.no_grad():
            test_outputs = onnx_model(*dummy_input)
            if isinstance(test_outputs, tuple):
                print(f"✓ Output shapes: {[t.shape for t in test_outputs]}")
            else:
                print(f"✓ Output shape: {test_outputs.shape}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Export
    print("\nExporting to ONNX...")
    try:
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
        
        print(f"✓ Exported to {output}")
        
        # Verify
        model_onnx = onnx.load(str(output))
        onnx.checker.check_model(model_onnx)
        print("✓ ONNX model is valid")
        
    except Exception as e:
        print(f"✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Validate if requested
    if validate:
        print("\nValidating with ONNX Runtime...")
        try:
            session = ort.InferenceSession(str(output))
            print("✓ Model loads in ONNX Runtime")
            
            # Prepare test input
            if mode == 'semantic':
                test_input = {
                    'points': dummy_points.numpy(),
                    'intensity': dummy_intensity.numpy()
                }
            else:
                test_input = {
                    'point_coords': dummy_points.numpy(),
                    'point_features': dummy_features.numpy()
                }
            
            outputs = session.run(None, test_input)
            print(f"✓ Inference successful")
            if isinstance(outputs, list):
                print(f"  Output shapes: {[o.shape for o in outputs]}")
            
        except Exception as e:
            print(f"✗ Validation failed: {e}")
    
    print("\n" + "="*60)
    print("Export completed successfully!")
    print("="*60)


if __name__ == "__main__":
    export_onnx()