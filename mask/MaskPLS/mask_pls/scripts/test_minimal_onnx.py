#!/usr/bin/env python3
"""
Minimal ONNX export test to identify what's breaking
"""
import torch
import torch.nn as nn
from pathlib import Path

class MinimalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 10)
        
    def forward(self, x):
        return self.linear(x)

class TestConv2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 64, 1)
        
    def forward(self, x):
        return self.conv(x)

class TestBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 64, 1)
        self.bn = nn.BatchNorm2d(64)
        
    def forward(self, x):
        x = self.conv(x)
        return self.bn(x)

def test_export(model, dummy_input, name):
    """Test if a model can export to ONNX"""
    output_path = f"test_{name}.onnx"
    
    print(f"\n=== Testing {name} ===")
    
    try:
        # Test forward pass
        with torch.no_grad():
            output = model(dummy_input)
            if isinstance(output, tuple):
                shapes = [o.shape for o in output]
                print(f"[OK] Forward pass successful: {shapes}")
            else:
                print(f"[OK] Forward pass successful: {output.shape}")
    except Exception as e:
        print(f"[FAIL] Forward pass failed: {e}")
        return False
    
    try:
        # Test ONNX export
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=False,
            input_names=['input'],
            output_names=['output'],
            verbose=False
        )
        
        # Check if file was created and is ONNX format
        if Path(output_path).exists():
            with open(output_path, 'rb') as f:
                header = f.read(16)
                print(f"File header: {header.hex()}")
                
                if header.startswith(b'\x08\x07\x12\x07pyto'):
                    print(f"[FAIL] {name}: Created PyTorch file, not ONNX")
                    Path(output_path).unlink()
                    return False
                else:
                    print(f"[OK] {name}: Successfully created ONNX file")
                    Path(output_path).unlink()  # Clean up
                    return True
        else:
            print(f"[FAIL] {name}: No output file created")
            return False
            
    except Exception as e:
        print(f"[FAIL] {name}: Export failed with error: {e}")
        return False

def main():
    print("Testing minimal ONNX export scenarios...")
    
    # Test 1: Basic linear layer
    model1 = MinimalModel()
    dummy1 = torch.randn(100, 4)
    test_export(model1, dummy1, "linear")
    
    # Test 2: Conv2D layer
    model2 = TestConv2D()
    dummy2 = torch.randn(1, 8, 100, 20)
    test_export(model2, dummy2, "conv2d")
    
    # Test 3: Conv2D + BatchNorm
    model3 = TestBatchNorm()
    dummy3 = torch.randn(1, 8, 100, 20)
    test_export(model3, dummy3, "conv2d_bn")
    
    # Test 4: Multiple outputs
    class MultiOutput(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 5)
            self.linear2 = nn.Linear(10, 3)
            
        def forward(self, x):
            return self.linear1(x), self.linear2(x)
    
    model4 = MultiOutput()
    dummy4 = torch.randn(100, 10)
    test_export(model4, dummy4, "multi_output")
    
    print("\n" + "="*50)
    print("If all tests pass, the issue is in the complex model structure")
    print("If tests fail, there's a fundamental PyTorch/ONNX setup issue")

if __name__ == "__main__":
    main()