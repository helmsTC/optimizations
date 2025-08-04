# ============================================================================  
# FILE: mask_pls/models/mink.py (MODIFICATIONS)
# ============================================================================

"""
Add mixed precision and optimization hints for the sparse backbone
"""

# ADD TO mink.py:

class OptimizedMinkEncoderDecoder(MinkEncoderDecoder):
    """
    Optimized version of the sparse backbone
    Applies mixed precision and memory optimizations
    """
    
    def __init__(self, cfg, data_cfg):
        super().__init__(cfg, data_cfg)
        self.use_mixed_precision = getattr(cfg, 'MIXED_PRECISION', True)
    
    def forward(self, x):
        """Forward pass with mixed precision optimization"""
        
        if self.use_mixed_precision:
            # Use automatic mixed precision for speedup
            with torch.cuda.amp.autocast():
                return self._forward_impl(x)
        else:
            return self._forward_impl(x)
    
    def _forward_impl(self, x):
        """Implementation that can be wrapped with AMP"""
        # Same as original forward pass
        in_field = self.TensorField(x)

        x0 = self.stem(in_field.sparse())
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        y1 = self.up1[0](x4)
        y1 = ME.cat(y1, x3)
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = ME.cat(y2, x2)
        y2 = self.up2[1](y2)

        y3 = self.up3[0](y2)
        y3 = ME.cat(y3, x1)
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        y4 = ME.cat(y4, x0)
        y4 = self.up4[1](y4)

        out_feats = [y1, y2, y3, y4]

        # Optimized coordinate processing
        coors = [in_field.decomposed_coordinates for _ in range(len(out_feats))]
        coors = [[c * self.res for c in coors[i]] for i in range(len(coors))]
        bs = in_field.coordinate_manager.number_of_unique_batch_indices()
        vox_coors = [
            [l.coordinates_at(i) * self.res for i in range(bs)] for l in out_feats
        ]
        
        # Apply batch normalization with mixed precision support
        feats = [
            [
                bn(self.knn_up(vox_c, vox_f, pt_c))
                for vox_c, vox_f, pt_c in zip(vc, vf.decomposed_features, pc)
            ]
            for vc, vf, pc, bn in zip(vox_coors, out_feats, coors, self.out_bnorm)
        ]

        feats, coors, pad_masks = self.pad_batch(coors, feats)
        logits = self.sem_head(feats[-1])
        return feats, coors, pad_masks, logits
