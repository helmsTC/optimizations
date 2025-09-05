# mask_pls/scripts/export_efficient_dgcnn_onnx_fixed.py
"""
Fixed ONNX export for MaskPLS-DGCNN that properly handles shapes
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

sys.path.append(str(Path(__file__).parent.parent))

from mask_pls.models.dgcnn.maskpls_dgcnn_optimized import MaskPLSDGCNNFixed


class SimplifiedONNXModel(nn.Module):
    """
    Simplified ONNX-compatible version that bypasses complex list/dict processing
    """
    def __init__(self, original_model):
        super().__init__()
        self.num_classes = original_model.num_classes
        self.ignore_label = original_model.ignore_label
        self.things_ids = original_model.things_ids if hasattr(original_model, 'things_ids') else []
        
        # Extract the key components
        backbone = original_model.backbone
        decoder = original_model.decoder
        
        # Copy backbone components directly
        self.k = backbone.k
        self.edge_conv1 = backbone.edge_conv1
        self.edge_conv2 = backbone.edge_conv2
        self.edge_conv3 = backbone.edge_conv3
        self.edge_conv4 = backbone.edge_conv4
        self.conv5 = backbone.conv5
        self.feat_layers = backbone.feat_layers
        self.out_bn = backbone.out_bn
        self.sem_head = backbone.sem_head
        
        # Copy decoder
        self.decoder = decoder
        
        self.eval()
    
    def process_point_cloud(self, coords, feats):
        """Process a single point cloud through DGCNN backbone"""
        # Combine coordinates and intensity feature
        if feats.shape[-1] > 3:
            x = torch.cat([coords, feats[:, 3:4]], dim=1).transpose(0, 1).unsqueeze(0)
        else:
            x = coords.transpose(0, 1).unsqueeze(0)
        
        # Import the knn function
        from mask_pls.models.dgcnn.dgcnn_backbone_efficient_fixed import get_graph_feature
        
        # Edge convolutions
        x1 = get_graph_feature(x, k=self.k)
        x1 = self.edge_conv1(x1)
        x1 = x1.max(dim=-1, keepdim=False)[0]
        
        x2 = get_graph_feature(x1, k=self.k)
        x2 = self.edge_conv2(x2)
        x2 = x2.max(dim=-1, keepdim=False)[0]
        
        x3 = get_graph_feature(x2, k=self.k)
        x3 = self.edge_conv3(x3)
        x3 = x3.max(dim=-1, keepdim=False)[0]
        
        x4 = get_graph_feature(x3, k=self.k)
        x4 = self.edge_conv4(x4)
        x4 = x4.max(dim=-1, keepdim=False)[0]
        
        # Aggregate
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        
        # Generate multi-scale features
        features = []
        for feat_layer, bn_layer in zip(self.feat_layers, self.out_bn):
            feat = feat_layer(x)
            feat = bn_layer(feat)
            feat = feat.squeeze(0).transpose(0, 1)
            features.append(feat)
        
        return features
    
    def forward(self, point_coords, point_features):
        """
        Simplified forward for ONNX
        
        Args:
            point_coords: [N, 3] tensor
            point_features: [N, 4] tensor
        
        Returns:
            pred_logits: [Q, num_classes+1]
            pred_masks: [N, Q]
            sem_logits: [N, num_classes]
        """
        N = point_coords.shape[0]
        
        # Subsample if needed (deterministic)
        max_points = 30000
        if N > max_points:
            # Use stride sampling for determinism
            stride = N // max_points
            indices = torch.arange(0, N, stride)[:max_points]
            point_coords = point_coords[indices]
            point_features = point_features[indices]
            N = point_coords.shape[0]
        
        # Process through backbone
        multi_scale_features = self.process_point_cloud(point_coords, point_features)
        
        # Prepare for decoder (need batch dimension)
        ms_features = []
        ms_coords = []
        ms_masks = []
        
        for feat in multi_scale_features:
            ms_features.append(feat.unsqueeze(0))  # Add batch dim
            ms_coords.append(point_coords.unsqueeze(0))
            ms_masks.append(torch.zeros(1, N, dtype=torch.bool))
        
        # Decode
        outputs, padding = self.decoder(ms_features, ms_coords, ms_masks)
        
        # Remove batch dimension
        pred_logits = outputs['pred_logits'].squeeze(0)
        pred_masks = outputs['pred_masks'].squeeze(0)
        
        # Semantic predictions
        sem_logits = self.sem_head(multi_scale_features[-1])
        
        return pred_logits, pred_masks, sem_logits


class TraceableONNXModel(nn.Module):
    """
    Even simpler version for tracing
    """
    def __init__(self, original_model):
        super().__init__()
        
        # Extract essential components
        self.backbone_conv1 = original_model.backbone.edge_conv1
        self.backbone_conv2 = original_model.backbone.edge_conv2
        self.backbone_conv3 = original_model.backbone.edge_conv3
        self.backbone_conv4 = original_model.backbone.edge_conv4
        self.backbone_conv5 = original_model.backbone.conv5
        
        # Feature projections
        self.feat_projs = original_model.backbone.feat_layers
        self.feat_norms = original_model.backbone.out_bn
        
        # Decoder components
        self.decoder = original_model.decoder
        self.sem_head = original_model.backbone.sem_head
        self.num_classes = original_model.num_classes
        
        self.eval()
    
    def forward(self, points, features):
        """
        Direct forward pass
        Args:
            points: [N, 3]
            features: [N, 1] intensity only
        Returns:
            sem_logits: [N, num_classes]
        """
        # Prepare input [1, 4, N]
        N = points.shape[0]
        x = torch.cat([points, features], dim=1).T.unsqueeze(0)
        
        # Simple feature extraction (no graph operations for now)
        # This is a simplified version - you may need to implement graph ops differently
        
        # For now, just use conv1d operations as placeholder
        x = self.backbone_conv5(x)  # Process aggregated features
        
        # Project to output dimension
        feat = self.feat_projs[-1](x)
        feat = self.feat_norms[-1](feat)
        feat = feat.squeeze(0).T
        
        # Semantic segmentation
        sem_logits = self.sem_head(feat)
        
        # For simplified output, just return semantic logits
        # You can add instance predictions later
        return sem_logits


def load_config(checkpoint_dir):
    """Load configuration"""
    config_dir = Path(__file__).parent.parent / "config"
    
    model_cfg = edict(yaml.safe_load(open(config_dir / "model.yaml")))
    backbone_cfg = edict(yaml.safe_load(open(config_dir / "backbone.yaml")))
    decoder_cfg = edict(yaml.safe_load(open(config_dir / "decoder.yaml")))
    
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    
    # Load saved hyperparameters if available
    hparams_file = checkpoint_dir.parent / 'hparams.yaml'
    if hparams_file.exists():
        print(f"Loading hyperparameters from {hparams_file}")
        with open(hparams_file, 'r') as f:
            saved_cfg = yaml.safe_load(f)
            if saved_cfg:
                cfg.update(saved_cfg)
    
    # Set defaults
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
@click.option('--num-points', '-n', default=10000, help='Number of points for test')
@click.option('--opset', default=11, help='ONNX opset version')
@click.option('--simple', is_flag=True, help='Use simplified model (semantic only)')
@click.option('--debug', is_flag=True, help='Debug mode')
def export_onnx(checkpoint, output, num_points, opset, simple, debug):
    """Export MaskPLS-DGCNN to ONNX with fixed shape handling"""
    
    print("="*60)
    print("MaskPLS-DGCNN ONNX Export (Fixed)")
    print("="*60)
    
    # Output path
    if output is None:
        checkpoint_path = Path(checkpoint)
        suffix = "_simple.onnx" if simple else "_full.onnx"
        output = checkpoint_path.parent / f"{checkpoint_path.stem}{suffix}"
    
    print(f"Checkpoint: {checkpoint}")
    print(f"Output: {output}")
    print(f"Mode: {'Simple (semantic only)' if simple else 'Full model'}")
    
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
    
    # Create ONNX model
    if simple:
        print("\nUsing simplified model (semantic segmentation only)...")
        onnx_model = TraceableONNXModel(model)
        
        # Test input
        dummy_points = torch.randn(num_points, 3)
        dummy_features = torch.randn(num_points, 1)  # Just intensity
        
        input_names = ['points', 'features']
        output_names = ['sem_logits']
        
        dynamic_axes = {
            'points': {0: 'num_points'},
            'features': {0: 'num_points'},
            'sem_logits': {0: 'num_points'}
        }
        
    else:
        print("\nUsing full model...")
        onnx_model = SimplifiedONNXModel(model)
        
        # Test input
        dummy_points = torch.randn(num_points, 3)
        dummy_features = torch.randn(num_points, 4)  # xyz + intensity
        
        input_names = ['point_coords', 'point_features']
        output_names = ['pred_logits', 'pred_masks', 'sem_logits']
        
        dynamic_axes = {
            'point_coords': {0: 'num_points'},
            'point_features': {0: 'num_points'},
            'pred_masks': {0: 'num_points'},
            'sem_logits': {0: 'num_points'}
        }
    
    # Test forward pass
    print("\nTesting forward pass...")
    try:
        with torch.no_grad():
            if simple:
                test_output = onnx_model(dummy_points, dummy_features)
                print(f"✓ Output shape: {test_output.shape}")
            else:
                test_outputs = onnx_model(dummy_points, dummy_features)
                print(f"✓ Output shapes: {[t.shape for t in test_outputs]}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        return
    
    # Export
    print("\nExporting to ONNX...")
    try:
        torch.onnx.export(
            onnx_model,
            (dummy_points, dummy_features),
            str(output),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=debug
        )
        
        print(f"✓ Exported to {output}")
        
        # Verify
        model_onnx = onnx.load(str(output))
        onnx.checker.check_model(model_onnx)
        print("✓ ONNX model is valid")
        
        # Test loading
        print("\nTesting ONNX Runtime loading...")
        session = ort.InferenceSession(str(output))
        print("✓ Model loads in ONNX Runtime")
        
        # Test inference
        if simple:
            test_input = {
                'points': dummy_points.numpy(),
                'features': dummy_features.numpy()
            }
        else:
            test_input = {
                'point_coords': dummy_points.numpy(),
                'point_features': dummy_features.numpy()
            }
        
        outputs = session.run(None, test_input)
        print(f"✓ Inference successful")
        print(f"  Output shapes: {[o.shape for o in outputs]}")
        
    except Exception as e:
        print(f"✗ Export failed: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        
        print("\nTry using --simple flag for a simpler model")


if __name__ == "__main__":
    export_onnx()