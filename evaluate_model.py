# Add these imports at the top
from mask_pls.models.mask_model import MaskPSONNX

# Modify the click command to add ONNX options
@click.option("--use_onnx", is_flag=True, help="Use ONNX-optimized decoder")
@click.option("--onnx_decoder", type=str, default=None, help="Path to ONNX decoder")
def main(w, save_testset, nuscenes, use_onnx, onnx_decoder):  # Add the new parameters
    # ... existing config loading code ...
    
    data = SemanticDatasetModule(cfg)
    
    # Modified model loading
    if use_onnx and onnx_decoder:
        model = MaskPSONNX.from_checkpoint_with_onnx(w, onnx_decoder)
        print("✅ Using ONNX-optimized model")
    else:
        model = MaskPS(cfg)
        w = torch.load(w, map_location="cpu")
        model.load_state_dict(w["state_dict"])
        print("Using original PyTorch model")
    
    # ... rest of existing code unchanged ...
