#!/usr/bin/env python3
"""
Fallback export that bypasses the decoder entirely
Save as: mask_pls/onnx/fallback_export.py
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

class BackboneOnlyModel(nn.Module):
    """Export just the backbone, bypass decoder"""
    def __init__(self, original_model):
        super().__init__()
        self.backbone = original_model.backbone
        self.num_classes = original_model.num_classes
        
        # Simple output heads
        self.sem_head = nn.Linear(96, self.num_classes)
        self.mask_head = nn.Linear(96, 100)  # 100 queries
    
    def forward(self, points, features):
        # Process through backbone
        feats, coors, pad_masks, _ = self.backbone((features, points))
        
        if feats and len(feats) > 0:
            # Use last feature level
            last_feat = feats[-1]  # [B, N, C]
            
            # Simple predictions
            sem_logits = self.sem_head(last_feat)  # [B, N, num_classes]
            mask_logits = self.mask_head(last_feat)  # [B, N, 100]
            
            # Fake class predictions
            B = last_feat.shape[0]
            pred_logits = torch.zeros(B, 100, self.num_classes + 1)
            
            return pred_logits, mask_logits, sem_logits
        else:
            B = points.shape[0]
            N = points.shape[1]
            return (
                torch.zeros(B, 100, self.num_classes + 1),
                torch.zeros(B, N, 100),
                torch.zeros(B, N, self.num_classes)
            )

def export_backbone_only():
    """Export only the backbone, skip problematic decoder"""
    from mask_pls.models.onnx.onnx_model import MaskPLSONNX
    from easydict import EasyDict as edict
    
    # Config
    cfg = edict({
        'MODEL': {'DATASET': 'KITTI', 'OVERLAP_THRESHOLD': 0.8},
        'KITTI': {
            'NUM_CLASSES': 20,
            'MIN_POINTS': 10,
            'SPACE': [[-48.0, 48.0], [-48.0, 48.0], [-4.0, 1.5]],
            'SUB_NUM_POINTS': 80000
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
    
    print("Creating backbone-only model...")
    original_model = MaskPLSONNX(cfg)
    backbone_model = BackboneOnlyModel(original_model)
    backbone_model.eval()
    
    # Export
    dummy_points = torch.randn(1, 5000, 3)
    dummy_features = torch.randn(1, 5000, 4)
    
    print("Exporting backbone-only model...")
    torch.onnx.export(
        backbone_model,
        (dummy_points, dummy_features),
        "mask_pls_backbone_only.onnx",
        export_params=True,
        opset_version=11,
        input_names=['points', 'features'],
        output_names=['pred_logits', 'pred_masks', 'sem_logits']
    )
    
    print("✓ Backbone export successful!")
    print("  Note: This model bypasses the decoder entirely")
    print("  You'll need to implement the decoder logic separately")

if __name__ == "__main__":
    export_backbone_only()
