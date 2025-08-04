#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Add mask_pls to path
sys.path.append(str(Path(__file__).parent.parent))

from mask_pls.models.mask_model import MaskPS, MaskPSONNX
from mask_pls.utils.onnx_converter import MaskPLSONNXConverter

def main():
    parser = argparse.ArgumentParser(description="MaskPLS ONNX Optimization")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--output_dir", default="./onnx_models", help="Output directory")
    
    args = parser.parse_args()
    
    print("🚀 MaskPLS ONNX Optimization")
    print("="*40)
    
    # Load model
    print("📂 Loading model...")
    model = MaskPS.load_from_checkpoint(args.checkpoint)
    model.eval()
    
    # Convert decoder
    print("🔄 Converting decoder to ONNX...")
    converter = MaskPLSONNXConverter(model, args.output_dir)
    onnx_path = converter.convert_decoder()
    
    if onnx_path and converter.validate_onnx_model(onnx_path):
        print(f"✅ Success! ONNX decoder: {onnx_path}")
        
        # Test optimized model
        print("🔗 Creating optimized model...")
        optimized_model = MaskPSONNX.from_checkpoint_with_onnx(args.checkpoint, onnx_path)
        print("✅ Optimized model ready!")
        
        print(f"\n💡 Usage:")
        print(f"   Use {onnx_path} with MaskPSONNX class")
    else:
        print("❌ ONNX conversion failed")

if __name__ == "__main__":
    main()
