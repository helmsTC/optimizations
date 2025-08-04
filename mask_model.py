# Add this import at the top
import onnxruntime as ort

# Add this class after the existing MaskPS class
class MaskPSONNX(MaskPS):
    """ONNX-optimized version of MaskPS"""
    
    def __init__(self, hparams, onnx_decoder_path=None):
        super().__init__(hparams)
        self.onnx_decoder_path = onnx_decoder_path
        self.onnx_session = None
        
        if onnx_decoder_path:
            self.load_onnx_decoder(onnx_decoder_path)
    
    def load_onnx_decoder(self, onnx_path):
        """Load ONNX decoder for optimized inference"""
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.onnx_session = ort.InferenceSession(str(onnx_path), providers=providers)
            print(f"✅ ONNX decoder loaded: {onnx_path}")
        except Exception as e:
            print(f"❌ Failed to load ONNX decoder: {e}")
            self.onnx_session = None
    
    def forward(self, x):
        """Forward pass with optional ONNX decoder"""
        # Stage 1: PyTorch sparse backbone
        feats, coors, pad_masks, bb_logits = self.backbone(x)
        
        # Stage 2: ONNX decoder if available
        if self.onnx_session is not None:
            outputs, padding = self.forward_onnx_decoder(feats, coors, pad_masks)
        else:
            outputs, padding = self.decoder(feats, coors, pad_masks)
        
        return outputs, padding, bb_logits
    
    def forward_onnx_decoder(self, feats, coors, pad_masks):
        """Run decoder using ONNX Runtime"""
        # Prepare inputs for ONNX
        onnx_inputs = {}
        for i, (feat, coor, mask) in enumerate(zip(feats, coors, pad_masks)):
            onnx_inputs[f'feats_{i}'] = feat.detach().cpu().numpy()
            onnx_inputs[f'coors_{i}'] = coor.detach().cpu().numpy()
            onnx_inputs[f'pad_masks_{i}'] = mask.detach().cpu().numpy()
        
        # Run ONNX inference
        onnx_outputs = self.onnx_session.run(None, onnx_inputs)
        
        # Convert back to PyTorch tensors
        outputs = {
            'pred_logits': torch.from_numpy(onnx_outputs[0]).cuda(),
            'pred_masks': torch.from_numpy(onnx_outputs[1]).cuda(),
        }
        padding = torch.from_numpy(onnx_outputs[2]).cuda()
        
        return outputs, padding

    @classmethod
    def from_checkpoint_with_onnx(cls, checkpoint_path, onnx_decoder_path):
        """Load model from checkpoint and attach ONNX decoder"""
        original_model = cls.load_from_checkpoint(checkpoint_path)
        optimized_model = cls(original_model.cfg, onnx_decoder_path)
        optimized_model.load_state_dict(original_model.state_dict(), strict=False)
        return optimized_model
