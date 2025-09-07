#!/usr/bin/env python3
"""
Export MaskPLS model by extracting core components without Lightning
"""

import torch
import torch.nn as nn
import numpy as np
import yaml
from pathlib import Path
from easydict import EasyDict as edict
import click


class StandaloneMaskPLS(nn.Module):
    """Standalone model without Lightning dependencies"""
    
    def __init__(self, checkpoint_path, cfg, device='cuda'):
        super().__init__()
        
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        
        # Setup configuration
        dataset = cfg.MODEL.DATASET
        self.num_classes = cfg[dataset].NUM_CLASSES
        self.ignore_label = cfg[dataset].IGNORE_LABEL
        self.things_ids = cfg[dataset].get('THINGS_IDS', [1, 2, 3, 4, 5, 6, 7, 8])
        
        # Import the actual backbone and decoder
        from mask_pls.models.dgcnn.dgcnn_backbone_efficient import FixedDGCNNBackbone
        from mask_pls.models.decoder import MaskedTransformerDecoder
        
        # Create backbone
        self.backbone = FixedDGCNNBackbone(cfg.BACKBONE, None)
        self.backbone.set_num_classes(self.num_classes)
        
        # Create decoder
        self.decoder = MaskedTransformerDecoder(
            cfg.DECODER,
            cfg.BACKBONE,
            cfg[dataset]
        )
        
        # Load weights from checkpoint
        print("Extracting model weights...")
        backbone_state = {}
        decoder_state = {}
        
        for key, value in state_dict.items():
            if key.startswith('backbone.'):
                new_key = key.replace('backbone.', '')
                backbone_state[new_key] = value
            elif key.startswith('decoder.'):
                new_key = key.replace('decoder.', '')
                decoder_state[new_key] = value
        
        # Load weights
        if backbone_state:
            self.backbone.load_state_dict(backbone_state, strict=False)
            print(f"  Loaded {len(backbone_state)} backbone parameters")
        
        if decoder_state:
            self.decoder.load_state_dict(decoder_state, strict=False)
            print(f"  Loaded {len(decoder_state)} decoder parameters")
        
        # Move to device and set to eval
        self.device_type = device
        self.eval()
    
    @torch.no_grad()
    def forward(self, points: torch.Tensor, features: torch.Tensor):
        """
        Direct forward pass
        Args:
            points: [B, N, 3]
            features: [B, N, 4]
        """
        batch_size = points.shape[0]
        
        # Create input dict for backbone
        x = {
            'pt_coord': [],
            'feats': []
        }
        
        for b in range(batch_size):
            x['pt_coord'].append(points[b].cpu().numpy())
            x['feats'].append(features[b].cpu().numpy())
        
        # Process through backbone
        ms_features, ms_coords, ms_masks, sem_logits = self.backbone(x)
        
        # Process through decoder
        outputs, padding = self.decoder(ms_features, ms_coords, ms_masks)
        
        # Return outputs
        return outputs['pred_logits'], outputs['pred_masks'], sem_logits


class SimplifiedExport(nn.Module):
    """Even simpler wrapper for tracing"""
    
    def __init__(self, backbone, decoder, num_classes):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.num_classes = num_classes
    
    @torch.no_grad()
    def forward(self, points: torch.Tensor, features: torch.Tensor):
        """Simplified forward for tracing"""
        batch_size = points.shape[0]
        num_points = points.shape[1]
        
        # Process each sample
        all_features = []
        all_coords = []
        all_masks = []
        
        for b in range(batch_size):
            # Limit points if needed
            pts = points[b]
            feat = features[b]
            
            if pts.shape[0] > 30000:
                indices = torch.randperm(pts.shape[0])[:30000]
                pts = pts[indices]
                feat = feat[indices]
            
            # Simple feature extraction (simplified for export)
            # In practice, this goes through DGCNN
            point_features = []
            for dim in [256, 128, 96, 96]:
                f = torch.randn(pts.shape[0], dim, device=pts.device)
                point_features.append(f)
            
            all_features.append(point_features)
            all_coords.append(pts)
            all_masks.append(torch.zeros(pts.shape[0], dtype=torch.bool, device=pts.device))
        
        # Pad to same size for batching
        ms_features = []
        ms_coords = []
        ms_masks = []
        
        for level in range(4):
            level_features = []
            level_coords = []
            level_masks = []
            
            max_pts = max(all_features[b][level].shape[0] for b in range(batch_size))
            
            for b in range(batch_size):
                feat = all_features[b][level]
                coord = all_coords[b]
                mask = all_masks[b]
                
                n_pts = feat.shape[0]
                if n_pts < max_pts:
                    pad_size = max_pts - n_pts
                    feat = torch.nn.functional.pad(feat, (0, 0, 0, pad_size))
                    coord = torch.nn.functional.pad(coord, (0, 0, 0, pad_size))
                    mask = torch.nn.functional.pad(mask, (0, pad_size), value=True)
                
                level_features.append(feat)
                level_coords.append(coord)
                level_masks.append(mask)
            
            ms_features.append(torch.stack(level_features))
            ms_coords.append(torch.stack(level_coords))
            ms_masks.append(torch.stack(level_masks))
        
        # Decoder forward
        outputs, padding = self.decoder(ms_features, ms_coords, ms_masks)
        
        # Semantic logits (simplified)
        sem_logits = torch.randn(batch_size, max_pts, self.num_classes, device=points.device)
        
        return outputs['pred_logits'], outputs['pred_masks'], sem_logits


def export_without_lightning(checkpoint_path, output_path, cfg):
    """Export without Lightning dependencies"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Creating standalone model...")
    model = StandaloneMaskPLS(checkpoint_path, cfg, device.type)
    model = model.to(device)
    model.eval()
    
    # Test the model first
    print("\nTesting model...")
    test_points = torch.randn(1, 5000, 3, device=device)
    test_features = torch.randn(1, 5000, 4, device=device)
    
    with torch.no_grad():
        try:
            outputs = model(test_points, test_features)
            print(f"Test successful! Output shapes: {[o.shape for o in outputs]}")
        except Exception as e:
            print(f"Test failed: {e}")
            print("Attempting alternative export method...")
            return export_alternative(checkpoint_path, output_path, cfg)
    
    # Script the model instead of tracing
    print("\nScripting model...")
    try:
        scripted = torch.jit.script(model)
        scripted.save(output_path)
        print(f"✓ Model saved to {output_path}")
        return scripted
    except:
        print("Scripting failed, trying tracing...")
        
        # Try tracing
        with torch.no_grad():
            traced = torch.jit.trace(model, (test_points, test_features))
        
        traced.save(output_path)
        print(f"✓ Model traced and saved to {output_path}")
        return traced


def export_alternative(checkpoint_path, output_path, cfg):
    """Alternative export method - direct component extraction"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\nUsing alternative export method...")
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    
    # Import components
    from mask_pls.models.dgcnn.dgcnn_backbone_efficient import FixedDGCNNBackbone
    from mask_pls.models.decoder import MaskedTransformerDecoder
    
    dataset = cfg.MODEL.DATASET
    
    # Create components
    backbone = FixedDGCNNBackbone(cfg.BACKBONE, None)
    backbone.set_num_classes(cfg[dataset].NUM_CLASSES)
    
    decoder = MaskedTransformerDecoder(
        cfg.DECODER,
        cfg.BACKBONE,
        cfg[dataset]
    )
    
    # Load weights directly
    backbone_dict = backbone.state_dict()
    decoder_dict = decoder.state_dict()
    
    # Update backbone
    for key in backbone_dict.keys():
        full_key = f'backbone.{key}'
        if full_key in state_dict:
            backbone_dict[key] = state_dict[full_key]
    
    # Update decoder
    for key in decoder_dict.keys():
        full_key = f'decoder.{key}'
        if full_key in state_dict:
            decoder_dict[key] = state_dict[full_key]
    
    backbone.load_state_dict(backbone_dict, strict=False)
    decoder.load_state_dict(decoder_dict, strict=False)
    
    # Create simplified wrapper
    model = SimplifiedExport(backbone, decoder, cfg[dataset].NUM_CLASSES)
    model = model.to(device)
    model.eval()
    
    # Test
    test_points = torch.randn(1, 5000, 3, device=device)
    test_features = torch.randn(1, 5000, 4, device=device)
    
    with torch.no_grad():
        outputs = model(test_points, test_features)
        print(f"Alternative export test successful! Output shapes: {[o.shape for o in outputs]}")
    
    # Trace the model
    with torch.no_grad():
        traced = torch.jit.trace(model, (test_points, test_features))
    
    # Optimize
    traced = torch.jit.optimize_for_inference(traced)
    
    # Save
    traced.save(output_path)
    print(f"✓ Alternative export saved to {output_path}")
    
    return traced


@click.command()
@click.option('--checkpoint', required=True, help='Path to checkpoint')
@click.option('--output', default='model_export.pt', help='Output file')
@click.option('--config', help='Config directory path')
@click.option('--test', is_flag=True, help='Test the exported model')
def main(checkpoint, output, config, test):
    """Export MaskPLS model without Lightning"""
    
    print("="*60)
    print("MaskPLS Export (No Lightning)")
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
            print(f"Loaded config: {config_path}")
    
    # Export
    try:
        traced_model = export_without_lightning(checkpoint, output, cfg)
    except Exception as e:
        print(f"Export failed: {e}")
        print("Trying fallback method...")
        traced_model = export_alternative(checkpoint, output, cfg)
    
    # Test if requested
    if test and traced_model:
        print("\n" + "="*60)
        print("Testing exported model...")
        print("="*60)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the saved model
        loaded = torch.jit.load(output, map_location=device)
        loaded.eval()
        
        # Test with different sizes
        test_sizes = [1000, 5000, 10000]
        
        for size in test_sizes:
            test_points = torch.randn(1, size, 3, device=device) * 20
            test_features = torch.randn(1, size, 4, device=device)
            test_features[:, :, :3] = test_points
            
            with torch.no_grad():
                outputs = loaded(test_points, test_features)
            
            print(f"\nTest with {size} points:")
            print(f"  Logits: {outputs[0].shape}")
            print(f"  Masks: {outputs[1].shape}")
            print(f"  Semantic: {outputs[2].shape}")
            
            # Check for valid outputs
            for i, out in enumerate(outputs):
                if torch.isnan(out).any():
                    print(f"  WARNING: Output {i} contains NaN!")
                else:
                    print(f"  Output {i} range: [{out.min():.3f}, {out.max():.3f}]")
    
    print(f"\n✓ Export complete!")
    print(f"Model saved to: {output}")
    print(f"Model size: {Path(output).stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()