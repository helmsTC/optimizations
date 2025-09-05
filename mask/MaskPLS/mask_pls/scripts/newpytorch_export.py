# new_export_method.py
import torch
import torch.onnx

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
    
    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
model.eval()
dummy_input = torch.randn(1, 10)

# Method 1: Try the new dynamo export (PyTorch 2.0+)
try:
    print("Trying torch.onnx.dynamo_export...")
    export_output = torch.onnx.dynamo_export(model, dummy_input)
    export_output.save("dynamo_export.onnx")
    print("✓ Dynamo export succeeded")
except Exception as e:
    print(f"✗ Dynamo export failed: {e}")

# Method 2: Try with explicit ONNX program
try:
    print("\nTrying with ExportOptions...")
    export_options = torch.onnx.ExportOptions(opset_version=17)
    export_output = torch.onnx.export(
        model, 
        (dummy_input,), 
        "export_with_options.onnx",
        export_options=export_options
    )
    print("✓ Export with options succeeded")
except Exception as e:
    print(f"✗ Export with options failed: {e}")

# Method 3: Use the legacy export with specific parameters for PyTorch 2.4
try:
    print("\nTrying legacy export with PyTorch 2.4 fixes...")
    # For PyTorch 2.4, we need to explicitly set some parameters
    torch.onnx.export(
        model,
        dummy_input,
        "legacy_export_fixed.onnx",
        export_params=True,
        opset_version=17,  # Use newer opset for PyTorch 2.4
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH
    )
    
    # Check if it worked
    import onnx
    model_onnx = onnx.load("legacy_export_fixed.onnx")
    onnx.checker.check_model(model_onnx)
    print("✓ Legacy export with fixes succeeded")
except Exception as e:
    print(f"✗ Legacy export failed: {e}")