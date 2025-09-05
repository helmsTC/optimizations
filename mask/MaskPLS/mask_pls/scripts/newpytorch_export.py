# export_onnx_pt24.py
import torch
import torch.nn as nn
import numpy as np
import onnx
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from mask_pls.models.dgcnn.maskpls_dgcnn_optimized import MaskPLSDGCNNFixed

class SemanticONNXModel(nn.Module):
    """Simplified model for ONNX export with PyTorch 2.4"""
    def __init__(self, original_model):
        super().__init__()
        backbone = original_model.backbone
        
        # Only copy essential layers
        self.edge_conv1 = backbone.edge_conv1
        self.edge_conv2 = backbone.edge_conv2
        self.edge_conv3 = backbone.edge_conv3
        self.edge_conv4 = backbone.edge_conv4
        self.conv5 = backbone.conv5
        
        self.feat_layer = backbone.feat_layers[-1]
        self.out_bn = backbone.out_bn[-1]
        self.sem_head = backbone.sem_head
        self.k = 20
        
    def forward(self, points, features):
        # Simplified forward without complex operations
        x = torch.cat([points, features], dim=1)
        x = x.transpose(0, 1).unsqueeze(0)
        
        # Process through layers (simplified - no KNN)
        # This loses accuracy but ensures ONNX compatibility
        x1 = self.edge_conv1.conv(x.unsqueeze(-1)).squeeze(-1)
        x2 = self.edge_conv2.conv(x1.unsqueeze(-1)).squeeze(-1)
        x3 = self.edge_conv3.conv(x2.unsqueeze(-1)).squeeze(-1)
        x4 = self.edge_conv4.conv(x3.unsqueeze(-1)).squeeze(-1)
        
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        
        feat = self.feat_layer(x)
        feat = self.out_bn(feat)
        feat = feat.squeeze(0).transpose(0, 1)
        
        return self.sem_head(feat)

def export_with_pytorch24(checkpoint_path, output_path):
    """Export using PyTorch 2.4 compatible method"""
    import yaml
    from easydict import EasyDict as edict
    
    # Load config
    config_dir = Path(__file__).parent.parent / "config"
    model_cfg = edict(yaml.safe_load(open(config_dir / "model.yaml")))
    backbone_cfg = edict(yaml.safe_load(open(config_dir / "backbone.yaml")))
    decoder_cfg = edict(yaml.safe_load(open(config_dir / "decoder.yaml")))
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    
    # Load model
    model = MaskPLSDGCNNFixed(cfg)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # Create simplified model
    onnx_model = SemanticONNXModel(model)
    onnx_model.eval()
    
    # Test inputs
    dummy_points = torch.randn(1000, 3)
    dummy_features = torch.randn(1000, 1)
    
    # For PyTorch 2.4, use dynamo export if available
    try:
        # Try new export API
        export_program = torch.onnx.dynamo_export(
            onnx_model, 
            (dummy_points, dummy_features)
        )
        export_program.save(output_path)
        print(f"✓ Exported using dynamo_export to {output_path}")
        return True
    except:
        pass
    
    # Fallback to legacy export with PyTorch 2.4 fixes
    try:
        # Use TorchScript as intermediate
        traced = torch.jit.trace(onnx_model, (dummy_points, dummy_features))
        
        torch.onnx.export(
            traced,
            (dummy_points, dummy_features),
            output_path,
            export_params=True,
            opset_version=17,  # Use latest opset for PyTorch 2.4
            do_constant_folding=True,
            input_names=['points', 'features'],
            output_names=['sem_logits'],
            dynamic_axes={
                'points': {0: 'num_points'},
                'features': {0: 'num_points'},
                'sem_logits': {0: 'num_points'}
            }
        )
        print(f"✓ Exported using legacy method to {output_path}")
        
        # Verify
        model_onnx = onnx.load(output_path)
        onnx.checker.check_model(model_onnx)
        print("✓ ONNX model is valid")
        return True
        
    except Exception as e:
        print(f"✗ Export failed: {e}")
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python export_onnx_pt24.py <checkpoint_path>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    output_path = "semantic_model_pt24.onnx"
    
    export_with_pytorch24(checkpoint_path, output_path)