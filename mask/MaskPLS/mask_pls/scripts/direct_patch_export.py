#!/usr/bin/env python3
"""
Fixed ONNX export with KNN functionality - resolves type annotation issues
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


class ONNXKNNOperations:
    """
    Static methods for KNN operations that work with ONNX
    """
    
    @staticmethod
    def compute_knn(x: torch.Tensor, k: int) -> torch.Tensor:
        """
        KNN implementation using torch operations that ONNX can trace
        No default parameters to avoid torch.jit issues
        """
        batch_size = x.size(0)
        num_points = x.size(2)
        num_dims = x.size(1)
        
        # Reshape for distance computation
        x = x.view(batch_size, -1, num_points)  # [B, C, N]
        
        # Compute pairwise distances using torch operations
        inner = -2 * torch.matmul(x.transpose(2, 1), x)  # [B, N, N]
        xx = torch.sum(x**2, dim=1, keepdim=True)  # [B, 1, N]
        pairwise_distance = -xx.transpose(2, 1) - inner - xx  # [B, N, N]
        
        # Get k-nearest neighbors
        idx = pairwise_distance.topk(k=k, dim=-1, largest=True)[1]  # [B, N, k]
        
        return idx
    
    @staticmethod
    def get_graph_feature(x: torch.Tensor, k: int, idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Graph feature extraction using ONNX-compatible operations
        """
        batch_size = x.size(0)
        num_points = x.size(2)
        num_dims = x.size(1)
        
        x = x.view(batch_size, -1, num_points)
        
        if idx is None:
            idx = ONNXKNNOperations.compute_knn(x, k)  # [B, N, k]
        
        device = x.device
        
        # Use gather operations for ONNX compatibility
        x_flat = x.transpose(2, 1).contiguous()  # [B, N, C]
        
        # Create batch indices
        batch_idx = torch.arange(batch_size, device=device).view(-1, 1, 1)
        batch_idx = batch_idx.expand(batch_size, num_points, k).reshape(-1)  # [B*N*k]
        
        # Flatten indices
        idx_flat = idx.reshape(-1)  # [B*N*k]
        
        # Gather neighbor features
        # We need to gather from [B, N, C] using indices [B*N*k]
        x_gathered = []
        for b in range(batch_size):
            x_b = x_flat[b]  # [N, C]
            idx_b = idx[b].reshape(-1)  # [N*k]
            gathered = x_b[idx_b]  # [N*k, C]
            gathered = gathered.view(num_points, k, num_dims)  # [N, k, C]
            x_gathered.append(gathered)
        
        x_neighbors = torch.stack(x_gathered, dim=0)  # [B, N, k, C]
        x_neighbors = x_neighbors.permute(0, 3, 1, 2).contiguous()  # [B, C, N, k]
        
        # Get center features
        x_center = x.unsqueeze(-1).expand(-1, -1, -1, k)  # [B, C, N, k]
        
        # Compute edge features
        feature = torch.cat((x_center - x_neighbors, x_center), dim=1)  # [B, 2C, N, k]
        
        return feature


class ONNXCompatibleEdgeConv(nn.Module):
    """
    Edge convolution layer using ONNX-compatible operations
    """
    def __init__(self, in_channels, out_channels, k=20):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def forward(self, x, idx=None):
        # Use ONNX-compatible graph feature extraction
        # Pass k as integer, not as default parameter
        x = ONNXKNNOperations.get_graph_feature(x, self.k, idx)
        x = self.conv(x)
        x = x.max(dim=-1, keepdim=False)[0]
        return x


class ONNXDGCNNBackbone(nn.Module):
    """
    DGCNN backbone with ONNX-compatible KNN operations
    """
    def __init__(self, original_backbone):
        super().__init__()
        
        # Copy configuration from original
        self.k = 20  # Fixed k value
        self.num_classes = original_backbone.num_classes
        
        # Replace edge convolutions with ONNX-compatible versions
        self.edge_conv1 = ONNXCompatibleEdgeConv(4, 64, k=self.k)
        self.edge_conv2 = ONNXCompatibleEdgeConv(64, 64, k=self.k)
        self.edge_conv3 = ONNXCompatibleEdgeConv(64, 128, k=self.k)
        self.edge_conv4 = ONNXCompatibleEdgeConv(128, 256, k=self.k)
        
        # Copy weights from original
        self.edge_conv1.conv[0].load_state_dict(original_backbone.edge_conv1[0].state_dict())
        self.edge_conv1.conv[1].load_state_dict(original_backbone.edge_conv1[1].state_dict())
        
        self.edge_conv2.conv[0].load_state_dict(original_backbone.edge_conv2[0].state_dict())
        self.edge_conv2.conv[1].load_state_dict(original_backbone.edge_conv2[1].state_dict())
        
        self.edge_conv3.conv[0].load_state_dict(original_backbone.edge_conv3[0].state_dict())
        self.edge_conv3.conv[1].load_state_dict(original_backbone.edge_conv3[1].state_dict())
        
        self.edge_conv4.conv[0].load_state_dict(original_backbone.edge_conv4[0].state_dict())
        self.edge_conv4.conv[1].load_state_dict(original_backbone.edge_conv4[1].state_dict())
        
        # Copy other layers directly
        self.conv5 = original_backbone.conv5
        self.feat_layers = original_backbone.feat_layers
        self.out_bn = original_backbone.out_bn
        self.sem_head = original_backbone.sem_head
        
        # Track subsampling
        self.subsample_indices = {}
    
    def forward(self, x):
        """
        Forward pass with ONNX-compatible operations
        """
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
            coords = torch.from_numpy(coords_list[b]).float()
            feats = torch.from_numpy(feats_list[b]).float()
            
            # Subsample if needed
            max_points = 10000  # Fixed for ONNX
            if coords.shape[0] > max_points:
                # Use torch operations for subsampling
                perm = torch.randperm(coords.shape[0])[:max_points]
                perm = perm.sort()[0]
                coords = coords[perm]
                feats = feats[perm]
                self.subsample_indices[b] = perm
            else:
                self.subsample_indices[b] = torch.arange(coords.shape[0])
            
            # Process through DGCNN with ONNX-compatible KNN
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
    
    def process_single_cloud(self, coords, feats):
        """
        Process a single point cloud through DGCNN with real KNN
        """
        # Combine coordinates and intensity
        x = torch.cat([coords, feats[:, 3:4]], dim=1).transpose(0, 1).unsqueeze(0)
        
        # Edge convolutions with ONNX-compatible KNN
        x1 = self.edge_conv1(x)
        x2 = self.edge_conv2(x1)
        x3 = self.edge_conv3(x2)
        x4 = self.edge_conv4(x3)
        
        # Aggregate features
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
    
    def pad_batch_level(self, features, coords, masks):
        """Pad features, coordinates and masks to same size"""
        max_points = 10000  # Fixed for ONNX
        
        padded_features = []
        padded_coords = []
        padded_masks = []
        
        for feat, coord, mask in zip(features, coords, masks):
            n_points = feat.shape[0]
            if n_points < max_points:
                pad_size = max_points - n_points
                feat = F.pad(feat, (0, 0, 0, pad_size))
                coord = F.pad(coord, (0, 0, 0, pad_size))
                mask = F.pad(mask, (0, pad_size), value=True)
            else:
                feat = feat[:max_points]
                coord = coord[:max_points]
                mask = mask[:max_points]
            
            padded_features.append(feat)
            padded_coords.append(coord)
            padded_masks.append(mask)
        
        return (torch.stack(padded_features), 
                torch.stack(padded_coords), 
                torch.stack(padded_masks))
    
    def compute_semantic_logits(self, features, masks):
        """Compute semantic logits for valid points"""
        batch_size = features.shape[0]
        sem_logits = []
        
        for b in range(batch_size):
            valid_mask = ~masks[b]
            if valid_mask.sum() > 0:
                valid_features = features[b][valid_mask]
                logits = self.sem_head(valid_features)
            else:
                logits = torch.zeros(0, self.num_classes, device=features.device)
            
            # Pad back to full size
            full_logits = torch.zeros(features.shape[1], self.num_classes, 
                                     device=features.device)
            if valid_mask.sum() > 0:
                full_logits[valid_mask] = logits
            
            sem_logits.append(full_logits)
        
        return torch.stack(sem_logits)


class KNNPreservingWrapper(nn.Module):
    """
    Wrapper that preserves KNN functionality for ONNX export
    """
    def __init__(self, model, num_points=10000):
        super().__init__()
        self.num_points = num_points
        
        # Replace backbone with ONNX-compatible version
        self.backbone = ONNXDGCNNBackbone(model.backbone)
        self.decoder = model.decoder
        self.num_classes = model.num_classes
        
        # Move to CPU and eval
        self.cpu()
        self.eval()
        
        # Disable gradients
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, coords: torch.Tensor, feats: torch.Tensor) -> tuple:
        """
        Forward pass with preserved KNN
        Type annotations help torch.jit
        """
        B, N, _ = coords.shape
        
        # Ensure fixed size
        if N != self.num_points:
            if N < self.num_points:
                pad = self.num_points - N
                coords = F.pad(coords, (0, 0, 0, pad))
                feats = F.pad(feats, (0, 0, 0, pad))
            else:
                coords = coords[:, :self.num_points]
                feats = feats[:, :self.num_points]
        
        # Convert to expected format
        batch = {
            'pt_coord': [coords[b].cpu().numpy() for b in range(B)],
            'feats': [feats[b].cpu().numpy() for b in range(B)]
        }
        
        # Forward through backbone with KNN
        ms_features, ms_coords, ms_masks, sem_logits = self.backbone(batch)
        
        # Forward through decoder
        outputs, padding = self.decoder(ms_features, ms_coords, ms_masks)
        
        pred_logits = outputs['pred_logits']
        pred_masks = outputs['pred_masks']
        
        # Ensure correct output sizes
        if pred_masks.shape[1] != self.num_points:
            if pred_masks.shape[1] < self.num_points:
                pad = self.num_points - pred_masks.shape[1]
                pred_masks = F.pad(pred_masks, (0, 0, 0, pad))
            else:
                pred_masks = pred_masks[:, :self.num_points]
        
        if sem_logits.shape[1] != self.num_points:
            if sem_logits.shape[1] < self.num_points:
                pad = self.num_points - sem_logits.shape[1]
                sem_logits = F.pad(sem_logits, (0, 0, 0, pad))
            else:
                sem_logits = sem_logits[:, :self.num_points]
        
        return pred_logits, pred_masks, sem_logits


def export_with_knn(checkpoint_path, output_path, num_points=10000):
    """
    Export with preserved KNN functionality
    """
    print("="*60)
    print("ONNX Export with Preserved KNN (Fixed)")
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
    
    # Create wrapper with KNN preserved
    print("Creating KNN-preserving wrapper...")
    wrapper = KNNPreservingWrapper(model, num_points)
    
    # Test
    print("Testing forward pass...")
    dummy_coords = torch.randn(1, num_points, 3)
    dummy_feats = torch.randn(1, num_points, 4)
    
    with torch.no_grad():
        outputs = wrapper(dummy_coords, dummy_feats)
        print(f"✓ Forward pass successful: {[out.shape for out in outputs]}")
    
    # Export WITHOUT torch.jit.script - direct ONNX export
    print("Exporting to ONNX (without scripting)...")
    
    try:
        torch.onnx.export(
            wrapper,
            (dummy_coords, dummy_feats),
            output_path,
            export_params=True,
            opset_version=11,  # Use stable opset
            do_constant_folding=True,
            input_names=['coords', 'features'],
            output_names=['pred_logits', 'pred_masks', 'sem_logits'],
            dynamic_axes={
                'coords': {0: 'batch'},
                'features': {0: 'batch'},
                'pred_logits': {0: 'batch'},
                'pred_masks': {0: 'batch'},
                'sem_logits': {0: 'batch'}
            },
            verbose=False
        )
        
        print(f"✓ Exported to: {output_path}")
        
    except Exception as e:
        print(f"✗ Direct export failed: {e}")
        print("\nTrying alternative export method...")
        
        # Alternative: Export with fixed batch size
        torch.onnx.export(
            wrapper,
            (dummy_coords, dummy_feats),
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['coords', 'features'],
            output_names=['pred_logits', 'pred_masks', 'sem_logits'],
            dynamic_axes=None,  # Fixed dimensions
            verbose=False
        )
        
        print(f"✓ Exported with fixed dimensions to: {output_path}")
    
    # Verify
    print("\nVerifying ONNX model...")
    try:
        model_onnx = onnx.load(output_path)
        onnx.checker.check_model(model_onnx)
        print("✓ ONNX model is valid")
        
        # Test with runtime
        import onnxruntime as ort
        session = ort.InferenceSession(output_path)
        
        outputs = session.run(None, {
            'coords': dummy_coords.numpy(),
            'features': dummy_feats.numpy()
        })
        
        print("✓ ONNX Runtime test passed")
        print(f"  Output shapes: {[o.shape for o in outputs]}")
        
        print("\n✓ Successfully exported with preserved KNN functionality!")
        
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ONNX export with KNN preserved")
    parser.add_argument('checkpoint', help='Path to checkpoint')
    parser.add_argument('--output', '-o', default='maskpls_with_knn.onnx')
    parser.add_argument('--num-points', type=int, default=10000)
    
    args = parser.parse_args()
    
    export_with_knn(args.checkpoint, args.output, args.num_points)