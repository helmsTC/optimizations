# mask_pls/scripts/export_torchscript.py
import torch
import torch.jit
import click
import yaml
from pathlib import Path
from easydict import EasyDict as edict
import numpy as np

from mask_pls.models.dgcnn.maskpls_dgcnn_optimized import MaskPLSDGCNNFixed
from mask_pls.datasets.semantic_dataset import SemanticDatasetModule

class TorchScriptWrapper(torch.nn.Module):
    """Wrapper to make the model TorchScript compatible"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
        
    @torch.jit.export
    def forward(self, points_list, features_list):
        """
        Args:
            points_list: List of point clouds [B x [N, 3]]
            features_list: List of features [B x [N, 4]]
        Returns:
            Dictionary with predictions
        """
        # Convert lists to the expected format
        batch = {
            'pt_coord': [p.cpu().numpy() for p in points_list],
            'feats': [f.cpu().numpy() for f in features_list]
        }
        
        # Get model outputs
        outputs, padding, sem_logits = self.model(batch)
        
        # Process outputs for C++
        results = {
            'pred_logits': outputs['pred_logits'],
            'pred_masks': outputs['pred_masks'],
            'sem_logits': sem_logits,
            'padding': padding
        }
        
        return results

@click.command()
@click.option('--checkpoint', required=True, help='Path to checkpoint')
@click.option('--output', default='model.pt', help='Output TorchScript file')
@click.option('--trace', is_flag=True, help='Use tracing instead of scripting')
def export_torchscript(checkpoint, output, trace):
    """Export MaskPLS DGCNN to TorchScript"""
    
    print("Loading configuration...")
    def get_config():
        base_path = Path(__file__).parent.parent
        model_cfg = edict(yaml.safe_load(open(base_path / "config/model.yaml")))
        backbone_cfg = edict(yaml.safe_load(open(base_path / "config/backbone.yaml")))
        decoder_cfg = edict(yaml.safe_load(open(base_path / "config/decoder.yaml")))
        return edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    
    cfg = get_config()
    
    print("Loading model...")
    model = MaskPLSDGCNNFixed(cfg)
    
    # Load checkpoint
    ckpt = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.eval()
    
    # Set things_ids if available
    data = SemanticDatasetModule(cfg)
    data.setup()
    model.things_ids = data.things_ids
    
    if trace:
        print("Tracing model...")
        # Create example inputs
        example_points = [torch.randn(10000, 3) for _ in range(2)]
        example_features = [torch.randn(10000, 4) for _ in range(2)]
        
        # Wrap model
        wrapped_model = TorchScriptWrapper(model)
        
        # Trace
        traced = torch.jit.trace(wrapped_model, (example_points, example_features))
        
        # Save
        traced.save(output)
        print(f"Traced model saved to {output}")
        
    else:
        print("Scripting model...")
        # Create a simplified scriptable version
        scripted = create_scriptable_model(model, cfg)
        torch.jit.save(scripted, output)
        print(f"Scripted model saved to {output}")

def create_scriptable_model(model, cfg):
    """Create a scriptable version of the model"""
    
    class ScriptableMaskPLS(torch.nn.Module):
        def __init__(self, original_model):
            super().__init__()
            # Copy essential components
            self.backbone = original_model.backbone
            self.decoder = original_model.decoder
            self.num_classes = original_model.num_classes
            self.things_ids = original_model.things_ids
            
        def forward(self, points: torch.Tensor, features: torch.Tensor):
            """
            Simplified forward for TorchScript
            Args:
                points: [B, N, 3]
                features: [B, N, 4]
            """
            batch_size = points.shape[0]
            
            # Process through backbone (simplified)
            all_features = []
            all_coords = []
            all_masks = []
            
            for b in range(batch_size):
                pts = points[b]
                feat = features[b]
                
                # Process single cloud
                cloud_features = self.process_cloud(pts, feat)
                all_features.append(cloud_features)
                all_coords.append(pts)
                all_masks.append(torch.zeros(pts.shape[0], dtype=torch.bool))
            
            # Pad to same size
            ms_features, ms_coords, ms_masks = self.pad_features(
                all_features, all_coords, all_masks
            )
            
            # Decoder
            outputs, padding = self.decoder(ms_features, ms_coords, ms_masks)
            
            return outputs['pred_logits'], outputs['pred_masks'], padding
        
        @torch.jit.export
        def process_cloud(self, points: torch.Tensor, features: torch.Tensor):
            """Process single point cloud"""
            # Simplified processing without dynamic graph
            # This is a placeholder - implement based on your needs
            hidden_dim = 256
            return torch.randn(points.shape[0], hidden_dim)
        
        @torch.jit.export
        def pad_features(self, features, coords, masks):
            """Pad features to same size"""
            # Implement padding logic
            return features, coords, masks
    
    return torch.jit.script(ScriptableMaskPLS(model))

if __name__ == "__main__":
    export_torchscript()