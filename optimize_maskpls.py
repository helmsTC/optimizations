#!/usr/bin/env python3
"""
Fixed optimize_maskpls.py - handles the hparams issue properly
"""
import argparse
import sys
import torch
import yaml
from pathlib import Path
from easydict import EasyDict as edict

def find_config_dir():
    """Find the config directory - tries multiple locations"""
    possible_locations = [
        Path("config"),  # Current directory
        Path("../config"),  # Parent directory
        Path(__file__).parent.parent / "config",  # Relative to script
        Path.cwd() / "config",  # Working directory
    ]
    
    for config_dir in possible_locations:
        if config_dir.exists() and (config_dir / "model.yaml").exists():
            return config_dir
    
    print("❌ Could not find config directory with model.yaml")
    print("   Searched in:", [str(p) for p in possible_locations])
    print("   Please run from the MaskPLS root directory")
    return None

def load_maskpls_config():
    """Load MaskPLS configuration files"""
    config_dir = find_config_dir()
    if not config_dir:
        return None
    
    try:
        model_cfg = edict(yaml.safe_load(open(config_dir / "model.yaml")))
        backbone_cfg = edict(yaml.safe_load(open(config_dir / "backbone.yaml")))
        decoder_cfg = edict(yaml.safe_load(open(config_dir / "decoder.yaml")))
        cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
        return cfg
    except Exception as e:
        print(f"❌ Failed to load config files: {e}")
        return None

def load_maskpls_model(checkpoint_path):
    """Load MaskPLS model with proper configuration"""
    # Load configuration
    cfg = load_maskpls_config()
    if cfg is None:
        return None
    
    try:
        # Import here to avoid path issues
        from mask_pls.models.mask_model import MaskPS
        
        # Create model with config
        model = MaskPS(cfg)
        
        # Load checkpoint weights
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        
        print("✅ Model loaded successfully")
        return model
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you're running from the MaskPLS directory")
        print("   And mask_pls package is in your Python path")
        return None
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

class SimpleMaskPLSONNXConverter:
    """Simplified ONNX converter that doesn't rely on complex imports"""
    
    def __init__(self, model, output_dir="./onnx_models"):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def convert_decoder_simple(self, max_points=30000):
        """Simple decoder conversion using real backbone features"""
        print(f"🔄 Converting decoder to ONNX (using real backbone, {max_points} points)...")
        
        try:
            # Step 1: Generate realistic features using the backbone
            print("   🔧 Generating realistic features using backbone...")
            
            # Create sample input for backbone
            sample_input = self.create_sample_input(max_points)
            
            # Run backbone to get real features
            with torch.no_grad():
                self.model.eval()
                feats, coors, pad_masks, bb_logits = self.model.backbone(sample_input)
            
            print(f"   ✅ Backbone generated features:")
            for i, feat in enumerate(feats):
                print(f"      Level {i}: {feat.shape}")
            
            # Step 2: Create decoder wrapper
            class RealisticDecoderWrapper(torch.nn.Module):
                def __init__(self, original_decoder):
                    super().__init__()
                    self.decoder = original_decoder
                    
                def forward(self, feats_0, feats_1, feats_2, feats_3, 
                           coors_0, coors_1, coors_2, coors_3,
                           pad_masks_0, pad_masks_1, pad_masks_2, pad_masks_3):
                    """Forward with correct feature dimensions"""
                    feats = [feats_0, feats_1, feats_2, feats_3]
                    coors = [coors_0, coors_1, coors_2, coors_3]
                    pad_masks = [pad_masks_0, pad_masks_1, pad_masks_2, pad_masks_3]
                    
                    # Run decoder (should work now with correct dimensions)
                    outputs, padding = self.decoder(feats, coors, pad_masks)
                    return outputs['pred_logits'], outputs['pred_masks'], padding
            
            wrapper = RealisticDecoderWrapper(self.model.decoder)
            
            # Step 3: Reshape features to fixed size for ONNX
            print("   🔧 Preparing ONNX inputs with correct dimensions...")
            dummy_inputs = []
            input_names = []
            
            # Pad/trim features to fixed size
            batch_size = feats[0].shape[0]
            
            for i, (feat, coor, mask) in enumerate(zip(feats, coors, pad_masks)):
                current_points = feat.shape[1]
                target_channels = feat.shape[2]
                
                if current_points >= max_points:
                    # Trim to max_points
                    feat_fixed = feat[:, :max_points, :]
                    coor_fixed = coor[:, :max_points, :]
                    mask_fixed = mask[:, :max_points]
                else:
                    # Pad to max_points
                    pad_points = max_points - current_points
                    feat_fixed = torch.cat([feat, torch.zeros(batch_size, pad_points, target_channels)], dim=1)
                    coor_fixed = torch.cat([coor, torch.zeros(batch_size, pad_points, 3)], dim=1)
                    mask_fixed = torch.cat([mask, torch.ones(batch_size, pad_points, dtype=torch.bool)], dim=1)
                
                dummy_inputs.extend([feat_fixed, coor_fixed, mask_fixed])
                input_names.extend([f'feats_{i}', f'coors_{i}', f'pad_masks_{i}'])
            
            # Step 4: Export to ONNX
            onnx_path = self.output_dir / f"realistic_decoder_{max_points}pts.onnx"
            
            # Test the wrapper first
            print("   🧪 Testing wrapper before ONNX export...")
            test_inputs = []
            for i in range(0, len(dummy_inputs), 3):  # Group by 3 (feat, coor, mask)
                test_inputs.extend(dummy_inputs[i:i+3])
            
            with torch.no_grad():
                try:
                    test_outputs = wrapper(*test_inputs)
                    print(f"   ✅ Wrapper test successful, output shapes: {[o.shape for o in test_outputs]}")
                except Exception as e:
                    print(f"   ❌ Wrapper test failed: {e}")
                    return None
            
            # Now export to ONNX
            torch.onnx.export(
                wrapper,
                tuple(test_inputs),
                str(onnx_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=input_names,
                output_names=['pred_logits', 'pred_masks', 'padding'],
                dynamic_axes={name: {0: 'batch_size'} for name in input_names + ['pred_logits', 'pred_masks', 'padding']},
                verbose=False
            )
            
            print(f"✅ Realistic decoder exported to {onnx_path}")
            return onnx_path
            
        except Exception as e:
            print(f"❌ ONNX export failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_sample_input(self, num_points):
        """Create sample input for the backbone"""
        # Generate realistic point cloud
        points = torch.randn(num_points, 3) * 10  # 20m x 20m x 20m scene
        features = torch.randn(num_points, 4)     # XYZI features
        
        return {
            'pt_coord': [points.numpy()],
            'feats': [features.numpy()],
        }
    
    def validate_onnx_model(self, onnx_path):
        """Validate the ONNX model"""
        try:
            import onnx
            import onnxruntime as ort
            
            # Load and check ONNX model
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # Test with ONNX Runtime
            session = ort.InferenceSession(str(onnx_path))
            print(f"✅ ONNX model validated: {onnx_path}")
            return True
            
        except ImportError:
            print("⚠️  ONNX/ONNX Runtime not available - install with:")
            print("   pip install onnx onnxruntime-gpu")
            return False
        except Exception as e:
            print(f"❌ ONNX validation failed: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="MaskPLS ONNX Optimization (Fixed)")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (.ckpt file)")
    parser.add_argument("--output_dir", default="./onnx_models", help="Output directory")
    parser.add_argument("--max_points", type=int, default=30000, help="Max points (start small)")
    
    args = parser.parse_args()
    
    print("🚀 MaskPLS ONNX Optimization (Fixed Version)")
    print("="*50)
    
    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return
    
    # Load model with proper configuration
    print("📂 Loading model with configuration...")
    model = load_maskpls_model(args.checkpoint)
    if model is None:
        return
    
    # Simple ONNX conversion
    print("🔄 Converting decoder using simple approach...")
    converter = SimpleMaskPLSONNXConverter(model, args.output_dir)
    onnx_path = converter.convert_decoder_simple(args.max_points)
    
    if onnx_path:
        if converter.validate_onnx_model(onnx_path):
            print(f"\n🎉 Success!")
            print(f"   ONNX decoder: {onnx_path}")
            print(f"\n💡 Next steps:")
            print(f"   1. Test the ONNX model with ONNX Runtime")
            print(f"   2. Integrate with your inference pipeline")
            print(f"   3. Try larger max_points if this worked")
        else:
            print("⚠️  ONNX export succeeded but validation had issues")
    else:
        print("❌ ONNX conversion failed")
        print("💡 Try with smaller --max_points (e.g., 10000)")

if __name__ == "__main__":
    main()
