# check_environment.py
import torch
import onnx
import onnxruntime

print("PyTorch version:", torch.__version__)
print("ONNX version:", onnx.__version__)
print("ONNX Runtime version:", onnxruntime.__version__)
print("CUDA available:", torch.cuda.is_available())
print("PyTorch built with CUDA:", torch.version.cuda)

# Check if torch.onnx module exists
try:
    import torch.onnx
    print("torch.onnx module: Available")
    print("torch.onnx export function:", hasattr(torch.onnx, 'export'))
except ImportError:
    print("torch.onnx module: NOT AVAILABLE")

# Test the most basic ONNX export
import numpy as np

class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
    
    def forward(self, x):
        return self.linear(x)

model = TinyModel()
x = torch.randn(1, 3)

# Try different export methods
print("\nTrying torch.onnx.export...")
try:
    torch.onnx.export(model, x, "tiny_test.onnx", verbose=True)
    
    # Check what was actually created
    with open("tiny_test.onnx", "rb") as f:
        header = f.read(32)
        print("File header (hex):", header.hex())
        print("File header (ascii):", header[:16])
        
        # ONNX files should start with specific bytes
        if header[:4] == b'\x08\x01\x12\x00':
            print("This IS an ONNX file!")
        elif b'pytorch' in header or b'torch' in header:
            print("This is a PyTorch file!")
        else:
            print("Unknown file format")
            
except Exception as e:
    print(f"Export failed: {e}")