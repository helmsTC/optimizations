# Add this class after the existing MaskedTransformerDecoder class
class ONNXFriendlyDecoder(MaskedTransformerDecoder):
    """ONNX-compatible version of MaskedTransformerDecoder"""
    
    def __init__(self, cfg, bb_cfg, data_cfg):
        super().__init__(cfg, bb_cfg, data_cfg)
    
    def forward(self, feats, coors, pad_masks):
        """ONNX-friendly forward pass - removes dynamic operations"""
        # Process features (avoid pop() operations that cause ONNX issues)
        last_coors = coors[-1]
        mask_features = self.mask_feat_proj(feats[-1]) + self.pe_layer(last_coors)
        last_pad = pad_masks[-1]
        
        src = []
        pos = []
        
        for i in range(self.num_feature_levels):
            pos.append(self.pe_layer(coors[i]))
            feat = self.input_proj[i](feats[i])
            src.append(feat)
        
        bs = src[0].shape[0]
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        output = self.query_feat.weight.unsqueeze(0).repeat(bs, 1, 1)
        
        # Simplified transformer layers
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            
            # Cross-attention (no dynamic attention masks)
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index], attn_mask=None,
                padding_mask=pad_masks[level_index], pos=pos[level_index],
                query_pos=query_embed,
            )
            
            # Self-attention
            output = self.transformer_self_attention_layers[i](
                output, attn_mask=None, padding_mask=None, query_pos=query_embed
            )
            
            # FFN
            output = self.transformer_ffn_layers[i](output)
        
        # Final prediction heads
        decoder_output = self.decoder_norm(output)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bpc->bpq", mask_embed, mask_features)
        
        return outputs_class, outputs_mask, last_pad
