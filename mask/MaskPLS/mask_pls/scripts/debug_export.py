# debug_export_onnx.py
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory
sys.path.append(str(Path(__file__).parent.parent))

from mask_pls.models.dgcnn.maskpls_dgcnn_optimized import MaskPLSDGCNNFixed

def trace_debug():
    """Debug which operations are causing issues"""
    
    # Create a minimal test model
    class MinimalDGCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(4, 64, 1)
            self.bn = nn.BatchNorm1d(64)
            
        def forward(self, x):
            # x shape: [N, 4]
            x = x.transpose(0, 1).unsqueeze(0)  # [1, 4, N]
            x = self.conv(x)
            x = self.bn(x)
            return x.squeeze(0).transpose(0, 1)  # [N, 64]
    
    # Test export
    model = MinimalDGCNN()
    model.eval()
    
    dummy_input = torch.randn(1000, 4)
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            "test_minimal.onnx",
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'num_points'}}
        )
        print("✓ Minimal export succeeded")
        
        # Check if it's actually ONNX
        with open("test_minimal.onnx", 'rb') as f:
            header = f.read(16)
            if header.startswith(b'\x08\x07\x12\x07pyto'):
                print("✗ Still creating PyTorch file!")
            else:
                print("✓ Created actual ONNX file")
                
    except Exception as e:
        print(f"✗ Export failed: {e}")

if __name__ == "__main__":
    trace_debug()