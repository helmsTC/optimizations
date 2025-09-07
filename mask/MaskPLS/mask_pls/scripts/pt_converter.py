#!/usr/bin/env python3
"""
Save model with exact inference preservation - no tracing
"""

import torch
import torch.nn as nn
import numpy as np
import yaml
from pathlib import Path
from easydict import EasyDict as edict
import click
import pickle


class InferenceModel(nn.Module):
    """Model wrapper that preserves exact inference behavior"""
    
    def __init__(self, checkpoint_path, cfg):
        super().__init__()
        
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # Load the full Lightning model to get exact behavior
        from mask_pls.models.dgcnn.maskpls_dgcnn_optimized import MaskPLSDGCNNFixed
        
        # Create model
        self.model = MaskPLSDGCNNFixed(cfg)
        
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        
        # Load state dict
        self.model.load_state_dict(state_dict, strict=False)
        
        # Extract components we need
        self.backbone = self.model.backbone
        self.decoder = self.model.decoder
        self.num_classes = self.model.num_classes
        self.things_ids = getattr(self.model, 'things_ids', [1, 2, 3, 4, 5, 6, 7, 8])
        
        # Remove Lightning-specific attributes
        if hasattr(self.model, 'trainer'):
            delattr(self.model, 'trainer')
        
        self.eval()
    
    @torch.no_grad()
    def forward(self, points_list, features_list):
        """
        Forward pass with lists of numpy arrays (like original)
        Args:
            points_list: List of numpy arrays [N, 3]
            features_list: List of numpy arrays [N, 4]
        """
        # Create input dict as expected by backbone
        x = {
            'pt_coord': points_list,
            'feats': features_list
        }
        
        # Use the actual model forward
        outputs, padding, sem_logits = self.model(x)
        
        return outputs['pred_logits'], outputs['pred_masks'], sem_logits, padding
    
    def process_single(self, points, features):
        """Process single point cloud"""
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        
        # Call with single-item lists
        outputs = self.forward([points], [features])
        
        # Extract single batch results
        return (
            outputs[0][0:1],  # pred_logits [1, Q, C]
            outputs[1][0:1],  # pred_masks [1, N, Q]
            outputs[2][0:1],  # sem_logits [1, N, C]
            outputs[3][0:1]   # padding [1, N]
        )


def save_inference_model(checkpoint_path, output_path, cfg):
    """Save model for inference without tracing"""
    
    # Create inference model
    model = InferenceModel(checkpoint_path, cfg)
    model.eval()
    
    # Save the entire model (not traced)
    print(f"Saving inference model to {output_path}")
    torch.save({
        'model': model,
        'config': cfg,
        'type': 'inference_model'
    }, output_path)
    
    print(f"✓ Model saved successfully")
    return model


@click.command()
@click.option('--checkpoint', required=True, help='Path to checkpoint')
@click.option('--output', default='model_inference.pth', help='Output file')
@click.option('--config', help='Config directory path')
@click.option('--test', is_flag=True, help='Test the saved model')
def main(checkpoint, output, config, test):
    """Save MaskPLS model for inference"""
    
    print("="*60)
    print("MaskPLS Inference Model Export")
    print("="*60)
    
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
    
    # Save model
    model = save_inference_model(checkpoint, output, cfg)
    
    if test:
        print("\n" + "="*60)
        print("Testing saved model...")
        print("="*60)
        
        # Load saved model
        saved = torch.load(output, map_location='cpu')
        saved_model = saved['model']
        saved_model.eval()
        
        # Test with random data
        points = np.random.randn(10000, 3).astype(np.float32) * 20
        features = np.random.randn(10000, 4).astype(np.float32)
        features[:, :3] = points
        
        with torch.no_grad():
            outputs = saved_model.process_single(points, features)
        
        print(f"Output shapes:")
        for i, out in enumerate(outputs):
            print(f"  Output {i}: {out.shape}")
        
        # Check semantic predictions
        sem_logits = outputs[2][0]
        sem_pred = torch.argmax(sem_logits, dim=-1)
        unique_classes = torch.unique(sem_pred)
        print(f"Predicted classes: {unique_classes.numpy()}")
    
    print(f"\n✓ Export complete!")
    print(f"Model size: {Path(output).stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()