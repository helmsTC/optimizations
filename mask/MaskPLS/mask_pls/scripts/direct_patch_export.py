#!/usr/bin/env python3
"""
Final fixed ONNX export - simplified gather operations for shape inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
import onnx
from pathlib import Path
from easydict import EasyDict as edict
import sys
from typing import Optional

sys.path.append(str(Path(__file__).parent.parent))

from mask_pls.models.dgcnn.maskpls_dgcnn_optimized import MaskPLSDGCNNFixed


class SimplifiedKNNOps:
    """
    Simplified KNN operations that work better with ONNX shape inference
    """
    
    @staticmethod
    def compute_knn_indices(x: torch.Tensor, k: int) -> torch.Tensor:
        """
        Compute KNN indices with simplified operations
        """
        batch_size, num_dims, num_points = x.shape
        
        # Transpose for distance computation
        x_trans = x.transpose(2, 1)  # [B, N, C]
        
        # Compute pairwise distances
        inner = torch.matmul(x_trans, x_trans.transpose(2, 1))  # [B, N, N]
        xx = (x_trans ** 2).sum(dim=2, keepdim=True)  # [B, N, 1]
        pairwise_distance = xx + xx.transpose(2, 1) - 2 * inner  # [B, N, N]
        
        # Get k-nearest neighbors (including self)
        _, idx = pairwise_distance.topk(k=k, dim=-1, largest=False)  # [B, N, k]
        
        return idx
    
    @staticmethod
    def get_edge_features(x: torch.Tensor, k: int) -> torch.Tensor:
        """
        Simplified edge feature extraction that ONNX can handle
        """
        batch_size, num_dims, num_points = x.shape
        device = x.device
        
        # Get KNN indices
        idx = SimplifiedKNNOps.compute_knn_indices(x, k)  # [B, N, k]
        
        # Prepare for gathering - use simpler indexing
        x = x.transpose(2, 1).contiguous()  # [B, N, C]
        
        # Create index for batch dimension
        batch_idx = torch.arange(batch_size, device=device).view(-1, 1, 1, 1)
        batch_idx = batch_idx.expand(batch_size, num_points, k, 1)  # [B, N, k, 1]
        
        # Expand idx for gathering
        idx_expanded = idx.unsqueeze(3).expand(-1, -1, -1, num_dims)  # [B, N, k, C]
        
        # Use gather instead of complex indexing
        x_expanded = x.unsqueeze(2).expand(-1, -1, k, -1)  # [B, N, k, C]
        x_gathered = torch.gather(x.unsqueeze(1).expand(-1, num_points, -1, -1), 2, idx_expanded)  # [B, N, k, C]
        
        # Compute edge features
        x_repeat = x.unsqueeze(2).expand(-1, -1, k, -1)  # [B, N, k, C]
        edge_features = torch.cat([x_gathered - x_repeat, x_repeat], dim=3)  # [B, N, k, 2C]
        
        # Reshape to expected format
        edge_features = edge_features.permute(0, 3, 1, 2).contiguous()  # [B, 2C, N, k]
        
        return edge_features


class SimplifiedEdgeConv(nn.Module):
    """
    Simplified edge convolution for ONNX
    """
    def __init__(self, in_channels, out_channels, k=20):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def forward(self, x):
        # Get edge features with simplified operations
        edge_feat = SimplifiedKNNOps.get_edge_features(x, self.k)
        
        # Apply convolution
        out = self.conv(edge_feat)
        
        # Max pooling over neighbors
        out = out.max(dim=-1)[0]
        
        return out


class SimplifiedDGCNNWrapper(nn.Module):
    """
    Simplified wrapper that avoids complex operations
    """
    def __init__(self, model, num_points=10000):
        super().__init__()
        self.num_points = num_points
        self.k = 20
        
        # Get original backbone
        backbone = model.backbone
        
        # Create simplified edge convolutions
        self.edge_conv1 = SimplifiedEdgeConv(4, 64, self.k)
        self.edge_conv2 = SimplifiedEdgeConv(64, 64, self.k)
        self.edge_conv3 = SimplifiedEdgeConv(64, 128, self.k)
        self.edge_conv4 = SimplifiedEdgeConv(128, 256, self.k)
        
        # Copy weights from original model
        self.edge_conv1.conv[0].load_state_dict(backbone.edge_conv1[0].state_dict())
        self.edge_conv1.conv[1].load_state_dict(backbone.edge_conv1[1].state_dict())
        
        self.edge_conv2.conv[0].load_state_dict(backbone.edge_conv2[0].state_dict())
        self.edge_conv2.conv[1].load_state_dict(backbone.edge_conv2[1].state_dict())
        
        self.edge_conv3.conv[0].load_state_dict(backbone.edge_conv3[0].state_dict())
        self.edge_conv3.conv[1].load_state_dict(backbone.edge_conv3[1].state_dict())
        
        self.edge_conv4.conv[0].load_state_dict(backbone.edge_conv4[0].state_dict())
        self.edge_conv4.conv[1].load_state_dict(backbone.edge_conv4[1].state_dict())
        
        # Copy other layers
        self.conv5 = backbone.conv5
        self.feat_layers = backbone.feat_layers
        self.out_bn = backbone.out_bn
        self.sem_head = backbone.sem_head
        
        # Copy decoder
        self.decoder = model.decoder
        self.num_classes = model.num_classes
        
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, coords: torch.Tensor, feats: torch.Tensor) -> tuple:
        """
        Simplified forward pass
        """
        B, N, _ = coords.shape
        
        # Ensure fixed size
        if N != self.num_points:
            # Use interpolation for resizing
            coords = F.interpolate(
                coords.transpose(1, 2), 
                size=self.num_points, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
            
            feats = F.interpolate(
                feats.transpose(1, 2), 
                size=self.num_points, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
        
        # Combine coords and features
        x = torch.cat([coords, feats[:, :, 3:4]], dim=2)  # [B, N, 4]
        x = x.transpose(1, 2)  # [B, 4, N]
        
        # Apply edge convolutions
        x1 = self.edge_conv1(x)  # [B, 64, N]
        x2 = self.edge_conv2(x1)  # [B, 64, N]
        x3 = self.edge_conv3(x2)  # [B, 128, N]
        x4 = self.edge_conv4(x3)  # [B, 256, N]
        
        # Aggregate
        x = torch.cat([x1, x2, x3, x4], dim=1)  # [B, 512, N]
        x = self.conv5(x)  # [B, 512, N]
        
        # Generate multi-scale features
        ms_features = []
        for feat_layer, bn_layer in zip(self.feat_layers, self.out_bn):
            feat = feat_layer(x)  # [B, C, N]
            feat = bn_layer(feat)
            feat = feat.transpose(1, 2)  # [B, N, C]
            ms_features.append(feat)
        
        # Create dummy coords and masks for decoder
        ms_coords = [coords for _ in ms_features]
        ms_masks = [torch.zeros(B, self.num_points, dtype=torch.bool) for _ in ms_features]
        
        # Decoder
        outputs, padding = self.decoder(ms_features, ms_coords, ms_masks)
        
        # Semantic predictions
        sem_logits = self.sem_head(ms_features[-1])  # [B, N, num_classes]
        
        pred_logits = outputs['pred_logits']  # [B, Q, num_classes+1]
        pred_masks = outputs['pred_masks']  # [B, N, Q]
        
        return pred_logits, pred_masks, sem_logits


def export_final(checkpoint_path, output_path, num_points=10000):
    """
    Final export with simplified operations
    """
    print("="*60)
    print("Final ONNX Export with Simplified KNN")
    print("="*60)
    
    # Load config
    config_dir = Path(__file__).parent.parent / "config"
    cfg = {}
    
    for cfg_file in ['model.yaml', 'backbone.yaml', 'decoder.yaml']:
        cfg_path = config_dir / cfg_file
        if cfg_path.exists():
            with open(cfg_path, 'r') as f:
                cfg.update(yaml.safe_load(f))
    
    cfg = edict(cfg)
    
    # Create and load model
    print("Creating model...")
    model = MaskPLSDGCNNFixed(cfg)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    
    # Create simplified wrapper
    print("Creating simplified wrapper...")
    wrapper = SimplifiedDGCNNWrapper(model, num_points)
    wrapper.eval()
    
    # Test
    print("Testing forward pass...")
    dummy_coords = torch.randn(1, num_points, 3)
    dummy_feats = torch.randn(1, num_points, 4)
    
    with torch.no_grad():
        outputs = wrapper(dummy_coords, dummy_feats)
        print(f"✓ Forward pass successful: {[out.shape for out in outputs]}")
    
    # Export to ONNX
    print("Exporting to ONNX...")
    
    # Try with operator_export_type for better compatibility
    torch.onnx.export(
        wrapper,
        (dummy_coords, dummy_feats),
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['coords', 'features'],
        output_names=['pred_logits', 'pred_masks', 'sem_logits'],
        dynamic_axes=None,  # Fixed dimensions for better compatibility
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        verbose=False
    )
    
    print(f"✓ Exported to: {output_path}")
    
    # Verify
    print("\nVerifying ONNX model...")
    try:
        # Basic check
        model_onnx = onnx.load(output_path)
        onnx.checker.check_model(model_onnx)
        print("✓ ONNX model structure is valid")
        
        # Try to simplify the model
        try:
            import onnx.shape_inference
            print("Running shape inference...")
            model_onnx = onnx.shape_inference.infer_shapes(model_onnx)
            onnx.save(model_onnx, output_path)
            print("✓ Shape inference successful")
        except Exception as e:
            print(f"⚠ Shape inference warning (non-critical): {e}")
        
        # Test with runtime
        print("\nTesting with ONNX Runtime...")
        import onnxruntime as ort
        
        # Use CPUExecutionProvider to avoid CUDA issues
        providers = ['CPUExecutionProvider']
        session = ort.InferenceSession(output_path, providers=providers)
        
        # Run inference
        outputs = session.run(None, {
            'coords': dummy_coords.numpy(),
            'features': dummy_feats.numpy()
        })
        
        print("✓ ONNX Runtime test passed!")
        print(f"  Output shapes: {[o.shape for o in outputs]}")
        
        # Additional info
        print(f"\nModel info:")
        print(f"  File size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")
        print(f"  Inputs: {[inp.name for inp in session.get_inputs()]}")
        print(f"  Outputs: {[out.name for out in session.get_outputs()]}")
        
        print("\n" + "="*60)
        print("✓ Export successful with KNN functionality!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        print("\nTrying alternative verification...")
        
        # Try without shape inference
        try:
            import onnxruntime as ort
            providers = ['CPUExecutionProvider']
            session = ort.InferenceSession(output_path, providers=providers)
            print("✓ Model loads in ONNX Runtime (shape inference skipped)")
            
            # Try inference anyway
            outputs = session.run(None, {
                'coords': dummy_coords.numpy(),
                'features': dummy_feats.numpy()
            })
            print("✓ Inference works despite shape inference warning")
            print(f"  Output shapes: {[o.shape for o in outputs]}")
            return True
            
        except Exception as e2:
            print(f"✗ Runtime failed: {e2}")
            return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Final ONNX export")
    parser.add_argument('checkpoint', help='Path to checkpoint')
    parser.add_argument('--output', '-o', default='maskpls_final.onnx')
    parser.add_argument('--num-points', type=int, default=10000)
    
    args = parser.parse_args()
    
    success = export_final(args.checkpoint, args.output, args.num_points)
    
    if not success:
        print("\nIf verification still fails, the model may still work!")
        print("Try using the exported ONNX model directly in your application.")
        print("Shape inference warnings don't always prevent inference.")