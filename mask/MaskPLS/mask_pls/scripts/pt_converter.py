#!/usr/bin/env python3
"""
Fixed export script that properly uses the trained DGCNN backbone
"""

import torch
import torch.nn as nn
import numpy as np
import yaml
from pathlib import Path
from easydict import EasyDict as edict
import click


class ProperMaskPLSExport(nn.Module):
    """Properly export MaskPLS with actual trained weights"""
    
    def __init__(self, checkpoint_path, cfg, device='cuda'):
        super().__init__()
        
        # Load the full model properly
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # Import the actual model
        from mask_pls.models.dgcnn.maskpls_dgcnn_optimized import MaskPLSDGCNNFixed
        
        # Create model instance
        self.model = MaskPLSDGCNNFixed(cfg)
        
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        
        # Load weights
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(device)
        self.model.eval()
        
        # Store config
        dataset = cfg.MODEL.DATASET
        self.num_classes = cfg[dataset].NUM_CLASSES
        self.things_ids = cfg[dataset].get('THINGS_IDS', [])
    
    @torch.no_grad()
    def forward(self, points: torch.Tensor, features: torch.Tensor):
        """
        Forward pass with proper input format
        Args:
            points: [B, N, 3]
            features: [B, N, 4]
        """
        batch_size = points.shape[0]
        
        # Convert to the format expected by the model
        batch_dict = {
            'pt_coord': [],
            'feats': []
        }
        
        for b in range(batch_size):
            batch_dict['pt_coord'].append(points[b].cpu().numpy())
            batch_dict['feats'].append(features[b].cpu().numpy())
        
        # Run through the actual model
        outputs, padding, sem_logits = self.model(batch_dict)
        
        # Get predictions
        pred_logits = outputs['pred_logits']
        pred_masks = outputs['pred_masks']
        
        return pred_logits, pred_masks, sem_logits


def export_properly(checkpoint_path, output_path, cfg):
    """Export with actual model processing"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model with actual weights
    model = ProperMaskPLSExport(checkpoint_path, cfg, device.type)
    model = model.to(device)
    model.eval()
    
    # Create realistic example inputs
    batch_size = 1
    num_points = 10000
    
    example_points = torch.randn(batch_size, num_points, 3, device=device) * 20
    example_features = torch.randn(batch_size, num_points, 4, device=device)
    example_features[:, :, :3] = example_points  # xyz should match
    
    print("Tracing model with actual weights...")
    with torch.no_grad():
        # Test first
        outputs = model(example_points, example_features)
        print(f"Test output shapes: {[o.shape for o in outputs]}")
        
        # Now trace
        traced = torch.jit.trace(model, (example_points, example_features))
    
    # Save
    traced.save(output_path)
    print(f"✓ Model saved to {output_path}")
    
    return traced


@click.command()
@click.option('--checkpoint', required=True, help='Path to checkpoint')
@click.option('--output', default='model_fixed.pt', help='Output file')
@click.option('--config', help='Config directory')
def main(checkpoint, output, config):
    """Export with proper weights"""
    
    # Load configuration
    if config:
        config_dir = Path(config)
    else:
        config_dir = Path(__file__).parent.parent / "config"
    
    cfg = edict()
    for config_file in ['model.yaml', 'backbone.yaml', 'decoder.yaml']:
        config_path = config_dir / config_file
        if config_path.exists():
            with open(config_path, 'r') as f:
                cfg.update(yaml.safe_load(f))
    
    # Export properly
    export_properly(checkpoint, output, cfg)


if __name__ == "__main__":
    main()