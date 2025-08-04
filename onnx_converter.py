import torch
import torch.onnx
import onnx
import onnxruntime as ort
from pathlib import Path

class MaskPLSONNXConverter:
    """Main converter class for MaskPLS ONNX optimization"""
    
    def __init__(self, model, output_dir="./onnx_models"):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def convert_decoder(self, max_points=50000):
        """Convert decoder to ONNX"""
        from ..models.decoder import ONNXFriendlyDecoder
        
        # Create ONNX-friendly version
        onnx_decoder = ONNXFriendlyDecoder(
            self.model.cfg.DECODER,
            self.model.cfg.BACKBONE, 
            self.model.cfg[self.model.cfg.MODEL.DATASET]
        )
        
        # Copy weights from original decoder
        onnx_decoder.load_state_dict(self.model.decoder.state_dict())
        onnx_decoder.eval()
        
        # Create dummy inputs
        batch_size = 1
        hidden_dim = 256
        feature_levels = 4
        
        dummy_feats = [torch.randn(batch_size, max_points, hidden_dim) for _ in range(feature_levels)]
        dummy_coors = [torch.randn(batch_size, max_points, 3) for _ in range(feature_levels)]
        dummy_pad_masks = [torch.zeros(batch_size, max_points, dtype=torch.bool) for _ in range(feature_levels)]
        
        # Export to ONNX
        onnx_path = self.output_dir / f"decoder_{max_points}pts.onnx"
        
        torch.onnx.export(
            onnx_decoder,
            (dummy_feats, dummy_coors, dummy_pad_masks),
            str(onnx_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=[f'feats_{i}' for i in range(feature_levels)] + 
                       [f'coors_{i}' for i in range(feature_levels)] + 
                       [f'pad_masks_{i}' for i in range(feature_levels)],
            output_names=['pred_logits', 'pred_masks', 'padding'],
            dynamic_axes={
                **{f'feats_{i}': {0: 'batch_size'} for i in range(feature_levels)},
                **{f'coors_{i}': {0: 'batch_size'} for i in range(feature_levels)},
                **{f'pad_masks_{i}': {0: 'batch_size'} for i in range(feature_levels)},
            }
        )
        
        print(f"✅ Decoder exported to {onnx_path}")
        return onnx_path
    
    def validate_onnx_model(self, onnx_path):
        """Validate ONNX model"""
        try:
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            session = ort.InferenceSession(str(onnx_path))
            print(f"✅ ONNX model validated: {onnx_path}")
            return True
        except Exception as e:
            print(f"❌ ONNX validation failed: {e}")
            return False
