#!/usr/bin/env python3
"""
Direct patch export - modifies only the specific lines causing issues
"""

import torch
import torch.nn as nn
import yaml
import onnx
from pathlib import Path
from easydict import EasyDict as edict
import sys

sys.path.append(str(Path(__file__).parent.parent))

from mask_pls.models.dgcnn.maskpls_dgcnn_optimized import MaskPLSDGCNNFixed


def patch_backbone_for_cpu(backbone):
    """Directly patch the backbone methods that use .cuda()"""
    
    # Patch the forward method
    original_forward = backbone.forward
    
    def patched_forward(x):
        coords_list = x['pt_coord']
        feats_list = x['feats']
        
        batch_size = len(coords_list)
        
        # Clear subsample tracking
        backbone.subsample_indices = {}
        
        # Process each point cloud
        all_features = []
        all_coords = []
        all_masks = []
        
        for b in range(batch_size):
            # CHANGE: Remove .cuda() calls
            coords = torch.from_numpy(coords_list[b]).float()  # No .cuda()
            feats = torch.from_numpy(feats_list[b]).float()    # No .cuda()
            
            # Subsample if needed
            max_points = 50000 if backbone.training else 30000
            if coords.shape[0] > max_points:
                perm = torch.randperm(coords.shape[0], device=coords.device)[:max_points]
                perm = perm.sort()[0]
                coords = coords[perm]
                feats = feats[perm]
                backbone.subsample_indices[b] = perm
            else:
                backbone.subsample_indices[b] = torch.arange(coords.shape[0], device=coords.device)
            
            # Process through DGCNN
            point_features = backbone.process_single_cloud(coords, feats)
            
            all_features.append(point_features)
            all_coords.append(coords)
            # CHANGE: device parameter for zeros
            all_masks.append(torch.zeros(coords.shape[0], dtype=torch.bool, device=coords.device))
        
        # Generate multi-scale features with padding
        ms_features = []
        ms_coords = []
        ms_masks = []
        
        for i in range(len(backbone.feat_layers)):
            level_features, level_coords, level_masks = backbone.pad_batch_level(
                [f[i] for f in all_features],
                all_coords,
                all_masks
            )
            ms_features.append(level_features)
            ms_coords.append(level_coords)
            ms_masks.append(level_masks)
        
        # Semantic predictions
        sem_logits = backbone.compute_semantic_logits(ms_features[-1], ms_masks[-1])
        
        return ms_features, ms_coords, ms_masks, sem_logits
    
    # Replace the forward method
    backbone.forward = patched_forward
    
    return backbone


class DirectPatchWrapper(nn.Module):
    """Wrapper with patched backbone"""
    
    def __init__(self, model, num_points=10000):
        super().__init__()
        self.model = model
        self.num_points = num_points
        
        # Patch the backbone
        self.model.backbone = patch_backbone_for_cpu(self.model.backbone)
        
        # Ensure CPU
        self.model = self.model.cpu()
        self.model.eval()
    
    def forward(self, coords, feats):
        B, N, _ = coords.shape
        
        # Ensure fixed size
        if N != self.num_points:
            if N < self.num_points:
                pad = self.num_points - N
                coords = torch.nn.functional.pad(coords, (0, 0, 0, pad))
                feats = torch.nn.functional.pad(feats, (0, 0, 0, pad))
            else:
                coords = coords[:, :self.num_points]
                feats = feats[:, :self.num_points]
        
        # Convert to expected format
        batch = {
            'pt_coord': [coords[b].cpu().numpy() for b in range(B)],
            'feats': [feats[b].cpu().numpy() for b in range(B)]
        }
        
        # Forward
        outputs, padding, sem_logits = self.model(batch)
        
        pred_logits = outputs['pred_logits']
        pred_masks = outputs['pred_masks']
        
        # Fix output sizes
        if pred_masks.shape[1] != self.num_points:
            if pred_masks.shape[1] < self.num_points:
                pad = self.num_points - pred_masks.shape[1]
                pred_masks = torch.nn.functional.pad(pred_masks, (0, 0, 0, pad))
            else:
                pred_masks = pred_masks[:, :self.num_points]
        
        if sem_logits.shape[1] != self.num_points:
            if sem_logits.shape[1] < self.num_points:
                pad = self.num_points - sem_logits.shape[1]
                sem_logits = torch.nn.functional.pad(sem_logits, (0, 0, 0, pad))
            else:
                sem_logits = sem_logits[:, :self.num_points]
        
        return pred_logits, pred_masks, sem_logits


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Direct patch ONNX export")
    parser.add_argument('checkpoint', help='Path to checkpoint')
    parser.add_argument('--output', '-o', default='maskpls_patched.onnx')
    parser.add_argument('--num-points', type=int, default=10000)
    
    args = parser.parse_args()
    
    # Load config
    print("Loading configuration...")
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
    
    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    
    # Create wrapper
    print("Creating patched wrapper...")
    wrapper = DirectPatchWrapper(model, args.num_points)
    
    # Test
    print("Testing forward pass...")
    dummy_coords = torch.randn(1, args.num_points, 3, device='cpu')
    dummy_feats = torch.randn(1, args.num_points, 4, device='cpu')
    
    with torch.no_grad():
        outputs = wrapper(dummy_coords, dummy_feats)
        print(f"✓ Forward pass successful: {[out.shape for out in outputs]}")
    
    # Export
    print("Exporting to ONNX...")
    torch.onnx.export(
        wrapper,
        (dummy_coords, dummy_feats),
        args.output,
        export_params=True,
        opset_version=13,  # Middle ground opset
        do_constant_folding=True,
        input_names=['coords', 'features'],
        output_names=['pred_logits', 'pred_masks', 'sem_logits']
    )
    
    print(f"✓ Exported to: {args.output}")
    
    # Verify
    try:
        model = onnx.load(args.output)
        onnx.checker.check_model(model)
        print("✓ ONNX model is valid")
        
        # Test with runtime
        import onnxruntime as ort
        session = ort.InferenceSession(args.output)
        
        outputs = session.run(None, {
            'coords': dummy_coords.numpy(),
            'features': dummy_feats.numpy()
        })
        
        print("✓ ONNX Runtime test passed")
        
    except Exception as e:
        print(f"✗ Verification failed: {e}")


if __name__ == "__main__":
    main()