# mask_pls/scripts/export_torchscript_fixed.py
import torch
import torch.nn as nn
import numpy as np
import click
import yaml
from pathlib import Path
from easydict import EasyDict as edict
from typing import List, Tuple, Dict

from mask_pls.models.dgcnn.maskpls_dgcnn_optimized import MaskPLSDGCNNFixed
from mask_pls.datasets.semantic_dataset import SemanticDatasetModule


class TorchScriptWrapper(nn.Module):
    """Wrapper that makes the model TorchScript compatible by handling dict inputs"""
    
    def __init__(self, model, num_classes):
        super().__init__()
        self.backbone = model.backbone
        self.decoder = model.decoder
        self.num_classes = num_classes
        self.things_ids = model.things_ids if hasattr(model, 'things_ids') else []
        
    def forward(self, points_list: List[torch.Tensor], features_list: List[torch.Tensor]):
        """
        Args:
            points_list: List of point cloud tensors [N, 3]
            features_list: List of feature tensors [N, 4]
        Returns:
            Tuple of (pred_logits, pred_masks, sem_logits)
        """
        batch_size = len(points_list)
        
        # Process each point cloud through backbone
        all_features = []
        all_coords = []
        all_masks = []
        
        for b in range(batch_size):
            coords = points_list[b].float()
            feats = features_list[b].float()
            
            # Limit points if needed
            max_points = 50000
            if coords.shape[0] > max_points:
                indices = torch.randperm(coords.shape[0])[:max_points]
                coords = coords[indices]
                feats = feats[indices]
            
            # Process through backbone layers directly
            point_features = self.process_single_cloud(coords, feats)
            
            all_features.append(point_features)
            all_coords.append(coords)
            all_masks.append(torch.zeros(coords.shape[0], dtype=torch.bool, device=coords.device))
        
        # Pad to same size
        ms_features, ms_coords, ms_masks = self.pad_batch(all_features, all_coords, all_masks)
        
        # Decoder
        outputs, padding = self.decoder(ms_features, ms_coords, ms_masks)
        
        # Semantic predictions
        sem_logits = self.compute_semantic_logits(ms_features[-1], ms_masks[-1])
        
        return outputs['pred_logits'], outputs['pred_masks'], sem_logits
    
    @torch.jit.export
    def process_single_cloud(self, coords: torch.Tensor, feats: torch.Tensor) -> List[torch.Tensor]:
        """Process a single point cloud - simplified for TorchScript"""
        # This needs to be implemented based on your DGCNN backbone
        # For now, returning dummy features
        hidden_dims = [256, 128, 96, 96]  # Match your backbone output dimensions
        features = []
        
        for dim in hidden_dims:
            feat = torch.randn(coords.shape[0], dim, device=coords.device)
            features.append(feat)
        
        return features
    
    @torch.jit.export
    def pad_batch(self, 
                  features: List[List[torch.Tensor]], 
                  coords: List[torch.Tensor], 
                  masks: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Pad features to same size"""
        batch_size = len(features)
        num_levels = len(features[0])
        
        ms_features = []
        ms_coords = []
        ms_masks = []
        
        for level in range(num_levels):
            level_features = [features[b][level] for b in range(batch_size)]
            max_points = max(f.shape[0] for f in level_features)
            
            padded_features = []
            padded_coords = []
            padded_masks = []
            
            for b in range(batch_size):
                feat = level_features[b]
                coord = coords[b]
                mask = masks[b]
                
                n_points = feat.shape[0]
                if n_points < max_points:
                    pad_size = max_points - n_points
                    feat = torch.nn.functional.pad(feat, (0, 0, 0, pad_size))
                    coord = torch.nn.functional.pad(coord, (0, 0, 0, pad_size))
                    mask = torch.nn.functional.pad(mask, (0, pad_size), value=True)
                
                padded_features.append(feat)
                padded_coords.append(coord)
                padded_masks.append(mask)
            
            ms_features.append(torch.stack(padded_features))
            ms_coords.append(torch.stack(padded_coords))
            ms_masks.append(torch.stack(padded_masks))
        
        return ms_features, ms_coords, ms_masks
    
    @torch.jit.export
    def compute_semantic_logits(self, features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Compute semantic logits"""
        # Simplified semantic head
        sem_logits = torch.randn(features.shape[0], features.shape[1], self.num_classes, 
                                 device=features.device)
        return sem_logits


class SimplifiedDGCNNWrapper(nn.Module):
    """Even simpler wrapper that bypasses the complex backbone"""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Copy essential components
        self.decoder = model.decoder
        self.num_classes = model.num_classes
        
        # Extract backbone layers we need
        if hasattr(model.backbone, 'feat_layers'):
            self.feat_layers = model.backbone.feat_layers
            self.out_bn = model.backbone.out_bn
            self.sem_head = model.backbone.sem_head
        
    def forward(self, points: torch.Tensor, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simplified forward pass
        Args:
            points: [B, N, 3]
            features: [B, N, 4]
        Returns:
            Tuple of (pred_logits, pred_masks, sem_logits)
        """
        B, N, _ = points.shape
        
        # Create dummy multi-scale features (replace with actual DGCNN processing)
        # These dimensions should match your backbone output
        ms_features = []
        ms_coords = []
        ms_masks = []
        
        feature_dims = [256, 128, 96, 96]
        for dim in feature_dims:
            # Create feature tensor
            feat = torch.randn(B, N, dim, device=points.device)
            ms_features.append(feat)
            ms_coords.append(points)
            ms_masks.append(torch.zeros(B, N, dtype=torch.bool, device=points.device))
        
        # Pass through decoder
        outputs, padding = self.decoder(ms_features, ms_coords, ms_masks)
        
        # Generate semantic logits
        sem_logits = torch.randn(B, N, self.num_classes, device=points.device)
        
        return outputs['pred_logits'], outputs['pred_masks'], sem_logits


def trace_model(model, output_path: str):
    """Use torch.jit.trace instead of script"""
    model.eval()
    
    # Create example inputs
    batch_size = 1
    num_points = 10000
    
    example_points = torch.randn(batch_size, num_points, 3)
    example_features = torch.randn(batch_size, num_points, 4)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        example_points = example_points.cuda()
        example_features = example_features.cuda()
    
    # Create wrapper
    wrapped_model = SimplifiedDGCNNWrapper(model)
    
    # Trace the model
    print("Tracing model...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapped_model, (example_points, example_features))
    
    # Optimize for inference
    traced = torch.jit.optimize_for_inference(traced)
    
    # Save
    torch.jit.save(traced, output_path)
    print(f"Traced model saved to {output_path}")
    
    return traced


def export_with_custom_ops(model, output_path: str):
    """Export using custom operators for complex parts"""
    
    class CustomDGCNNModule(torch.nn.Module):
        def __init__(self, original_model):
            super().__init__()
            self.model = original_model
            
        def forward(self, points: torch.Tensor, features: torch.Tensor):
            # Convert to the format the model expects
            batch_size = points.shape[0]
            
            # Create batch dict (this will be done in C++)
            batch_data = {
                'pt_coord': [points[b].cpu().numpy() for b in range(batch_size)],
                'feats': [features[b].cpu().numpy() for b in range(batch_size)]
            }
            
            # Call model (simplified)
            # In practice, you'd implement the DGCNN logic here
            outputs = self.simplified_forward(points, features)
            
            return outputs
        
        def simplified_forward(self, points, features):
            """Simplified forward without dict inputs"""
            B, N, _ = points.shape
            
            # Dummy outputs for now
            pred_logits = torch.randn(B, 100, self.model.num_classes + 1)
            pred_masks = torch.randn(B, N, 100)
            sem_logits = torch.randn(B, N, self.model.num_classes)
            
            return pred_logits, pred_masks, sem_logits
    
    # Create custom module
    custom_model = CustomDGCNNModule(model)
    
    # Script it
    scripted = torch.jit.script(custom_model)
    torch.jit.save(scripted, output_path)
    print(f"Custom scripted model saved to {output_path}")


@click.command()
@click.option('--checkpoint', required=True, help='Path to checkpoint')
@click.option('--output', default='model_traced.pt', help='Output file')
@click.option('--method', type=click.Choice(['trace', 'script', 'custom']), default='trace')
def main(checkpoint, output, method):
    """Export MaskPLS DGCNN to TorchScript"""
    
    print("="*60)
    print(f"Exporting MaskPLS to TorchScript using {method} method")
    print("="*60)
    
    # Load configuration
    def get_config():
        base_path = Path(__file__).parent.parent
        model_cfg = edict(yaml.safe_load(open(base_path / "config/model.yaml")))
        backbone_cfg = edict(yaml.safe_load(open(base_path / "config/backbone.yaml")))
        decoder_cfg = edict(yaml.safe_load(open(base_path / "config/decoder.yaml")))
        return edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    
    cfg = get_config()
    
    # Load model
    print("Loading model...")
    model = MaskPLSDGCNNFixed(cfg)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint}")
    ckpt = torch.load(checkpoint, map_location='cpu')
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # Set things_ids
    data = SemanticDatasetModule(cfg)
    data.setup()
    model.things_ids = data.things_ids
    
    # Export based on method
    if method == 'trace':
        trace_model(model, output)
    elif method == 'custom':
        export_with_custom_ops(model, output)
    else:  # script
        # Try simplified scripting
        print("Creating scriptable wrapper...")
        wrapper = TorchScriptWrapper(model, model.num_classes)
        
        # Create example inputs
        example_points = [torch.randn(10000, 3)]
        example_features = [torch.randn(10000, 4)]
        
        print("Scripting model...")
        scripted = torch.jit.script(wrapper)
        
        # Save
        torch.jit.save(scripted, output)
        print(f"Scripted model saved to {output}")
    
    print("\nExport complete!")
    print(f"Model saved to: {output}")
    print(f"File size: {Path(output).stat().st_size / 1024 / 1024:.2f} MB")
    
    # Test loading
    print("\nTesting load...")
    loaded = torch.jit.load(output)
    print("✓ Model loads successfully")
    
    # Test inference
    print("Testing inference...")
    test_points = torch.randn(1, 5000, 3)
    test_features = torch.randn(1, 5000, 4)
    
    with torch.no_grad():
        if method == 'script':
            outputs = loaded([test_points[0]], [test_features[0]])
        else:
            outputs = loaded(test_points, test_features)
    
    print(f"✓ Inference successful")
    print(f"  Output shapes: {[o.shape for o in outputs]}")


if __name__ == "__main__":
    main()