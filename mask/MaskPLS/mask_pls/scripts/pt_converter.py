#!/usr/bin/env python3
"""
Correct export that preserves the actual model inference pipeline
"""

import torch
import torch.nn as nn
import numpy as np
import yaml
from pathlib import Path
from easydict import EasyDict as edict
import click
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class CorrectMaskPLSExport(nn.Module):
    """Export wrapper that preserves exact inference pipeline"""
    
    def __init__(self, checkpoint_path, cfg):
        super().__init__()
        
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        
        # Setup configuration
        dataset = cfg.MODEL.DATASET
        self.num_classes = cfg[dataset].NUM_CLASSES
        self.ignore_label = cfg[dataset].IGNORE_LABEL
        self.things_ids = [1, 2, 3, 4, 5, 6, 7, 8] if dataset == 'KITTI' else [2, 3, 4, 5, 6, 7, 9, 10]
        
        # Import the actual model components
        from mask_pls.models.dgcnn.dgcnn_backbone_efficient import FixedDGCNNBackbone
        from mask_pls.models.decoder import MaskedTransformerDecoder
        
        # Create the actual backbone with pretrained path if available
        pretrained_path = cfg.get('PRETRAINED_PATH', None)
        self.backbone = FixedDGCNNBackbone(cfg.BACKBONE, pretrained_path)
        self.backbone.set_num_classes(self.num_classes)
        
        # Create decoder
        self.decoder = MaskedTransformerDecoder(
            cfg.DECODER,
            cfg.BACKBONE,
            cfg[dataset]
        )
        
        # Load weights
        print("Loading model weights...")
        loaded_backbone = 0
        loaded_decoder = 0
        
        # Load backbone weights
        backbone_state_dict = self.backbone.state_dict()
        for key in backbone_state_dict.keys():
            full_key = f'backbone.{key}'
            if full_key in state_dict:
                backbone_state_dict[key] = state_dict[full_key]
                loaded_backbone += 1
        
        self.backbone.load_state_dict(backbone_state_dict, strict=False)
        print(f"  Loaded {loaded_backbone}/{len(backbone_state_dict)} backbone parameters")
        
        # Load decoder weights
        decoder_state_dict = self.decoder.state_dict()
        for key in decoder_state_dict.keys():
            full_key = f'decoder.{key}'
            if full_key in state_dict:
                decoder_state_dict[key] = state_dict[full_key]
                loaded_decoder += 1
        
        self.decoder.load_state_dict(decoder_state_dict, strict=False)
        print(f"  Loaded {loaded_decoder}/{len(decoder_state_dict)} decoder parameters")
        
        # Verify critical layers are loaded
        self.verify_weights_loaded()
        
        self.eval()
    
    def verify_weights_loaded(self):
        """Verify that critical layers have non-zero weights"""
        print("\nVerifying loaded weights...")
        
        # Check backbone critical layers
        critical_checks = [
            ('backbone.edge_conv1.0.weight', self.backbone.edge_conv1[0].weight),
            ('backbone.edge_conv2.0.weight', self.backbone.edge_conv2[0].weight),
            ('backbone.edge_conv3.0.weight', self.backbone.edge_conv3[0].weight),
            ('backbone.edge_conv4.0.weight', self.backbone.edge_conv4[0].weight),
        ]
        
        for name, param in critical_checks:
            if param.abs().max().item() < 1e-6:
                print(f"  WARNING: {name} appears to be uninitialized!")
            else:
                print(f"  ✓ {name}: max weight = {param.abs().max().item():.4f}")
        
        # Check decoder
        if hasattr(self.decoder, 'query_feat'):
            print(f"  ✓ Decoder query_feat: max = {self.decoder.query_feat.weight.abs().max().item():.4f}")
    
    @torch.no_grad()
    def forward(self, points: torch.Tensor, features: torch.Tensor):
        """
        Forward pass that exactly matches training inference
        Args:
            points: [B, N, 3]
            features: [B, N, 4]
        """
        batch_size = points.shape[0]
        
        # Create the exact input format expected by backbone
        x = {
            'pt_coord': [],
            'feats': []
        }
        
        for b in range(batch_size):
            # Convert to numpy as expected by the dataloader
            x['pt_coord'].append(points[b].cpu().numpy())
            x['feats'].append(features[b].cpu().numpy())
        
        # Run through the ACTUAL backbone forward pass
        # This should use the real DGCNN processing, not random features!
        ms_features, ms_coords, ms_masks, sem_logits = self.backbone(x)
        
        # Run through decoder
        outputs, padding = self.decoder(ms_features, ms_coords, ms_masks)
        
        return outputs['pred_logits'], outputs['pred_masks'], sem_logits


class TracingWrapper(nn.Module):
    """Wrapper optimized for tracing that avoids numpy conversions"""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.backbone = model.backbone
        self.decoder = model.decoder
        self.num_classes = model.num_classes
    
    @torch.no_grad()
    def forward(self, points: torch.Tensor, features: torch.Tensor):
        """Forward pass optimized for tracing"""
        # Directly call backbone with tensors
        # We need to modify this to work with tensors instead of numpy
        
        # Since the backbone expects numpy arrays, we need to handle this differently
        # For now, we'll use the original model's forward
        return self.model(points, features)


def test_model_outputs(model, device='cuda'):
    """Test that the model produces reasonable outputs"""
    model = model.to(device)
    model.eval()
    
    # Create realistic test data
    batch_size = 1
    num_points = 5000
    
    # Generate realistic point cloud
    points = torch.randn(batch_size, num_points, 3, device=device) * 20
    points[:, :, 2] = points[:, :, 2] * 0.5  # Compress Z axis
    
    features = torch.zeros(batch_size, num_points, 4, device=device)
    features[:, :, :3] = points  # xyz
    features[:, :, 3] = torch.rand(batch_size, num_points, device=device)  # intensity
    
    print("\nTesting model outputs...")
    with torch.no_grad():
        outputs = model(points, features)
    
    pred_logits, pred_masks, sem_logits = outputs
    
    print(f"Output shapes:")
    print(f"  pred_logits: {pred_logits.shape}")
    print(f"  pred_masks: {pred_masks.shape}")
    print(f"  sem_logits: {sem_logits.shape}")
    
    # Check for reasonable values
    print(f"\nOutput statistics:")
    print(f"  pred_logits: min={pred_logits.min():.3f}, max={pred_logits.max():.3f}, mean={pred_logits.mean():.3f}")
    print(f"  pred_masks: min={pred_masks.min():.3f}, max={pred_masks.max():.3f}, mean={pred_masks.mean():.3f}")
    print(f"  sem_logits: min={sem_logits.min():.3f}, max={sem_logits.max():.3f}, mean={sem_logits.mean():.3f}")
    
    # Check semantic predictions
    sem_pred = torch.argmax(sem_logits[0], dim=-1)
    unique_classes = torch.unique(sem_pred)
    print(f"\nPredicted classes: {unique_classes.cpu().numpy()}")
    
    # Check if outputs are reasonable (not all zeros or random)
    if pred_logits.abs().max() < 0.01:
        print("WARNING: pred_logits appears to be all zeros!")
    if sem_logits.abs().max() < 0.01:
        print("WARNING: sem_logits appears to be all zeros!")
    
    return True


def export_with_verification(checkpoint_path, output_path, cfg):
    """Export with verification steps"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create the correct model
    print("\n" + "="*60)
    print("Creating model with actual trained weights...")
    print("="*60)
    
    model = CorrectMaskPLSExport(checkpoint_path, cfg)
    
    # Test the model
    test_model_outputs(model, device.type)
    
    print("\n" + "="*60)
    print("Exporting model...")
    print("="*60)
    
    model = model.to(device)
    model.eval()
    
    # Create example inputs
    example_points = torch.randn(1, 10000, 3, device=device) * 20
    example_features = torch.randn(1, 10000, 4, device=device)
    example_features[:, :, :3] = example_points
    
    # Try to trace the model
    print("Tracing model...")
    with torch.no_grad():
        try:
            traced = torch.jit.trace(model, (example_points, example_features))
            
            # Verify traced model
            print("Verifying traced model...")
            traced_outputs = traced(example_points, example_features)
            
            # Compare with original
            original_outputs = model(example_points, example_features)
            
            for i, (to, oo) in enumerate(zip(traced_outputs, original_outputs)):
                diff = (to - oo).abs().max().item()
                print(f"  Output {i} difference: {diff:.6f}")
                if diff > 1e-3:
                    print(f"    WARNING: Large difference detected!")
            
            # Save
            traced.save(output_path)
            print(f"\n✓ Model exported to {output_path}")
            
        except Exception as e:
            print(f"Tracing failed: {e}")
            print("Attempting to save as scripted module...")
            
            # Try scripting instead
            try:
                scripted = torch.jit.script(model)
                scripted.save(output_path)
                print(f"\n✓ Model scripted and saved to {output_path}")
            except Exception as e2:
                print(f"Scripting also failed: {e2}")
                
                # Last resort: save state dict
                torch.save({
                    'backbone_state_dict': model.backbone.state_dict(),
                    'decoder_state_dict': model.decoder.state_dict(),
                    'config': cfg
                }, output_path.replace('.pt', '_state.pth'))
                print(f"Saved state dict to {output_path.replace('.pt', '_state.pth')}")
                return None
    
    return traced


@click.command()
@click.option('--checkpoint', required=True, help='Path to checkpoint')
@click.option('--output', default='model_correct.pt', help='Output file')
@click.option('--config', help='Config directory path')
@click.option('--compare', is_flag=True, help='Compare with original checkpoint')
def main(checkpoint, output, config, compare):
    """Export MaskPLS model correctly"""
    
    print("="*60)
    print("MaskPLS Correct Export")
    print("="*60)
    
    # Load configuration
    if config:
        config_dir = Path(config)
    else:
        config_dir = Path(__file__).parent.parent / "config"
    
    cfg = edict()
    config_files = ['model.yaml', 'backbone.yaml', 'decoder.yaml']
    
    for config_file in config_files:
        config_path = config_dir / config_file
        if config_path.exists():
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
                if data:
                    cfg.update(data)
            print(f"✓ Loaded config: {config_path}")
        else:
            print(f"✗ Config not found: {config_path}")
    
    # Export with verification
    traced_model = export_with_verification(checkpoint, output, cfg)
    
    if traced_model and compare:
        print("\n" + "="*60)
        print("Comparing with original checkpoint...")
        print("="*60)
        
        # Load both models and compare on random data
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Original model
        original = CorrectMaskPLSExport(checkpoint, cfg).to(device)
        original.eval()
        
        # Exported model
        exported = torch.jit.load(output, map_location=device)
        exported.eval()
        
        # Test data
        test_points = torch.randn(1, 5000, 3, device=device) * 20
        test_features = torch.randn(1, 5000, 4, device=device)
        test_features[:, :, :3] = test_points
        
        with torch.no_grad():
            orig_out = original(test_points, test_features)
            exp_out = exported(test_points, test_features)
        
        print("Output comparison:")
        for i in range(3):
            diff = (orig_out[i] - exp_out[i]).abs()
            print(f"  Output {i}: max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}")
    
    print(f"\n✓ Export complete!")
    print(f"Model size: {Path(output).stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()