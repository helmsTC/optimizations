# mask_pls/scripts/export_dgcnn_onnx.py
"""
Export MaskPLS-DGCNN model to ONNX format
"""

import torch
import torch.nn as nn
import numpy as np
import onnx
from pathlib import Path
import click
import yaml
from easydict import EasyDict as edict

from mask_pls.models.dgcnn.maskpls_dgcnn import MaskPLSDGCNN


class ONNXWrapper(nn.Module):
    """Wrapper to handle dynamic shapes in ONNX"""
    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.decoder = model.decoder
    
    def forward(self, point_coords, point_features):
        """
        Args:
            point_coords: [B, N, 3]
            point_features: [B, N, 4]
        Returns:
            pred_logits: [B, Q, num_classes+1]
            pred_masks: [B, N, Q]
            sem_logits: [B, N, num_classes]
        """
        # Create input dict
        B, N, _ = point_coords.shape
        
        # Process each sample (ONNX doesn't handle lists well)
        ms_features = []
        ms_coords = []
        ms_masks = []
        
        for b in range(B):
            x = {
                'pt_coord': [point_coords[b].cpu().numpy()],
                'feats': [point_features[b].cpu().numpy()]
            }
            
            # Get features from backbone
            feats, coords, masks, sem = self.backbone(x)
            
            ms_features.append(feats)
            ms_coords.append(coords)
            ms_masks.append(masks)
        
        # Stack features
        stacked_features = []
        stacked_coords = []
        stacked_masks = []
        
        for level in range(len(ms_features[0])):
            level_feats = torch.cat([f[level] for f in ms_features], dim=0)
            level_coords = torch.cat([c[level] for c in ms_coords], dim=0)
            level_masks = torch.cat([m[level] for m in ms_masks], dim=0)
            
            stacked_features.append(level_feats)
            stacked_coords.append(level_coords)
            stacked_masks.append(level_masks)
        
        # Decode
        outputs, _ = self.decoder(stacked_features, stacked_coords, stacked_masks)
        
        # Semantic predictions
        sem_logits = self.backbone.sem_head(stacked_features[-1])
        
        return outputs['pred_logits'], outputs['pred_masks'], sem_logits


@click.command()
@click.option('--checkpoint', required=True, help='Path to trained checkpoint')
@click.option('--output', default='maskpls_dgcnn.onnx', help='Output ONNX file')
@click.option('--batch_size', default=1, help='Batch size for export')
@click.option('--num_points', default=50000, help='Number of points')
@click.option('--opset', default=14, help='ONNX opset version')
@click.option('--simplify', is_flag=True, help='Simplify ONNX model')
def export_onnx(checkpoint, output, batch_size, num_points, opset, simplify):
    """Export MaskPLS-DGCNN to ONNX"""
    
    print("Exporting MaskPLS-DGCNN to ONNX...")
    
    # Load config
    cfg_path = Path(checkpoint).parent.parent / 'hparams.yaml'
    if cfg_path.exists():
        cfg = edict(yaml.safe_load(open(cfg_path)))
    else:
        # Use default config
        from mask_pls.scripts.train_dgcnn import get_config
        cfg = get_config()
    
    # Load model
    model = MaskPLSDGCNN(cfg)
    checkpoint_data = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint_data['state_dict'])
    model.eval()
    
    # Create wrapper
    onnx_model = ONNXWrapper(model)
    onnx_model.eval()
    
    # Create dummy input
    dummy_coords = torch.randn(batch_size, num_points, 3)
    dummy_features = torch.randn(batch_size, num_points, 4)
    
    # Export
    with torch.no_grad():
        torch.onnx.export(
            onnx_model,
            (dummy_coords, dummy_features),
            output,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=['point_coords', 'point_features'],
            output_names=['pred_logits', 'pred_masks', 'sem_logits'],
            dynamic_axes={
                'point_coords': {0: 'batch_size', 1: 'num_points'},
                'point_features': {0: 'batch_size', 1: 'num_points'},
                'pred_logits': {0: 'batch_size'},
                'pred_masks': {0: 'batch_size', 1: 'num_points'},
                'sem_logits': {0: 'batch_size', 1: 'num_points'}
            }
        )
    
    print(f"Model exported to {output}")
    
    # Verify
    model_onnx = onnx.load(output)
    onnx.checker.check_model(model_onnx)
    print("✓ ONNX model is valid")
    
    # Simplify if requested
    if simplify:
        try:
            import onnxsim
            model_onnx, check = onnxsim.simplify(model_onnx)
            if check:
                onnx.save(model_onnx, output)
                print("✓ Model simplified")
        except ImportError:
            print("! onnx-simplifier not installed")
    
    print(f"Export complete! Model saved to: {output}")


if __name__ == "__main__":
    export_onnx()