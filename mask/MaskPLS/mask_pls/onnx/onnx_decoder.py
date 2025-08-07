"""
ONNX-compatible decoder with CORRECT attribute/method names
Replace mask_pls/onnx/onnx_decoder.py with this
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict
import math


class ONNXCompatibleDecoder(nn.Module):
    """
    ONNX-compatible version matching EXACTLY the original decoder interface
    """
    def __init__(self, cfg, bb_cfg, data_cfg):
        super().__init__()
        hidden_dim = cfg.HIDDEN_DIM
        
        self.hidden_dim = hidden_dim
        self.num_layers = cfg.FEATURE_LEVELS * cfg.DEC_BLOCKS
        self.nheads = cfg.NHEADS
        self.num_queries = cfg.NUM_QUERIES
        self.num_feature_levels = cfg.FEATURE_LEVELS
        self.num_classes = data_cfg.NUM_CLASSES
        
        # Initialize positional encoder FIRST (matching original)
        # Check if POS_ENC config exists
        if hasattr(cfg, 'POS_ENC'):
            cfg.POS_ENC.FEAT_SIZE = cfg.HIDDEN_DIM
            self.pe_layer = PositionalEncoder(cfg.POS_ENC)
        else:
            # Create default config
            pos_cfg = type('obj', (object,), {
                'MAX_FREQ': 10000,
                'DIMENSIONALITY': 3,
                'FEAT_SIZE': hidden_dim,
                'NUM_BANDS': hidden_dim // 6,  # Approximate
                'BASE': 2
            })()
            self.pe_layer = PositionalEncoder(pos_cfg)
        
        # Transformer layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        
        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(d_model=hidden_dim, nhead=self.nheads, dropout=0.0)
            )
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(d_model=hidden_dim, nhead=self.nheads, dropout=0.0)
            )
            self.transformer_ffn_layers.append(
                FFNLayer(d_model=hidden_dim, dim_feedforward=cfg.DIM_FFN, dropout=0.0)
            )
        
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        
        # Query embeddings (matching original names)
        self.query_feat = nn.Embedding(cfg.NUM_QUERIES, hidden_dim)
        self.query_embed = nn.Embedding(cfg.NUM_QUERIES, hidden_dim)
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        
        # Projections
        self.mask_feat_proj = nn.Sequential()
        in_channels = bb_cfg.CHANNELS
        if in_channels[-1] != hidden_dim:
            self.mask_feat_proj = nn.Linear(in_channels[-1], hidden_dim)
        
        # Input projections for each feature level
        in_channels = in_channels[:-1][-self.num_feature_levels:]
        self.input_proj = nn.ModuleList()
        for ch in in_channels:
            if ch != hidden_dim:
                self.input_proj.append(nn.Linear(ch, hidden_dim))
            else:
                self.input_proj.append(nn.Sequential())
        
        # Output FFNs (matching original)
        self.class_embed = nn.Linear(hidden_dim, data_cfg.NUM_CLASSES + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
    
    def forward(self, feats, coors, pad_masks):
        """
        Forward matching original decoder interface
        Args:
            feats: List of [B, N, C] features
            coors: List of [B, N, 3] coordinates  
            pad_masks: List of [B, N] padding masks
        Returns:
            outputs: dict with predictions
            last_pad: padding mask
        """
        # Handle the features similar to original
        if not feats or len(feats) == 0:
            B = 1
            out = {
                'pred_logits': torch.zeros(B, self.num_queries, self.num_classes + 1),
                'pred_masks': torch.zeros(B, 1, self.num_queries),
                'aux_outputs': []
            }
            return out, torch.zeros(B, 1, dtype=torch.bool)
        
        # Pop last level for mask features (like original)
        last_coors = coors.pop() if coors else None
        mask_features = self.mask_feat_proj(feats.pop())
        
        # Add positional encoding if we have coordinates
        if last_coors is not None and mask_features.shape[1] > 0:
            mask_features = mask_features + self.pe_layer(last_coors)
        
        last_pad = pad_masks.pop() if pad_masks else None
        
        # Prepare source features
        src = []
        pos = []
        size_list = []
        
        for i in range(self.num_feature_levels):
            if i < len(feats):
                size_list.append(feats[i].shape[1])
                # Add positional encoding
                if i < len(coors) and coors[i] is not None:
                    pos.append(self.pe_layer(coors[i]))
                else:
                    pos.append(torch.zeros_like(feats[i]))
                feat = self.input_proj[i](feats[i])
                src.append(feat)
        
        # Initialize queries
        bs = feats[0].shape[0] if feats else 1
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        output = self.query_feat.weight.unsqueeze(0).repeat(bs, 1, 1)
        
        predictions_class = []
        predictions_mask = []
        
        # Initial prediction (matching original method name)
        outputs_class, outputs_mask, _ = self.pred_heads(
            output, mask_features, pad_mask=last_pad
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        
        # Process through transformer layers
        for i in range(min(self.num_layers, 9)):  # Cap at 9 for ONNX
            level_index = i % self.num_feature_levels
            
            if level_index < len(src):
                # Cross-attention first (like original)
                output = self.transformer_cross_attention_layers[i](
                    output,
                    src[level_index],
                    pos=pos[level_index] if level_index < len(pos) else None,
                    query_pos=query_embed,
                    padding_mask=None  # Don't use padding mask for ONNX
                )
                
                # Self-attention
                output = self.transformer_self_attention_layers[i](
                    output, 
                    query_pos=query_embed
                )
                
                # FFN
                output = self.transformer_ffn_layers[i](output)
                
                # Get predictions
                outputs_class, outputs_mask, _ = self.pred_heads(
                    output, mask_features, pad_mask=last_pad
                )
                predictions_class.append(outputs_class)
                predictions_mask.append(outputs_mask)
        
        # Format output
        out = {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1]
        }
        
        # Set auxiliary outputs
        out["aux_outputs"] = self.set_aux(predictions_class, predictions_mask)
        
        return out, last_pad if last_pad is not None else torch.zeros(bs, mask_features.shape[1], dtype=torch.bool)
    
    def pred_heads(self, output, mask_features, pad_mask=None):
        """
        Prediction heads - matching original name and interface
        """
        decoder_output = self.decoder_norm(output)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        
        # Compute masks
        if mask_features.shape[1] == 0:
            B, Q = output.shape[:2]
            outputs_mask = torch.zeros(B, 1, Q, device=output.device)
        else:
            outputs_mask = torch.einsum("bqc,bpc->bpq", mask_embed, mask_features)
        
        # For ONNX, don't generate dynamic attention mask
        # Just return None for attn_mask
        attn_mask = None
        
        return outputs_class, outputs_mask, attn_mask
    
    def predict_masks(self, query_feat, mask_features):
        """
        Alias for compatibility if something calls predict_masks
        """
        decoder_output = self.decoder_norm(query_feat)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        
        if mask_features.shape[1] == 0:
            B, Q = query_feat.shape[:2]
            outputs_mask = torch.zeros(B, 1, Q, device=query_feat.device)
        else:
            outputs_mask = torch.einsum("bqc,bpc->bpq", mask_embed, mask_features)
        
        return outputs_class, outputs_mask
    
    @torch.jit.unused
    def set_aux(self, outputs_class, outputs_seg_masks):
        """Set auxiliary outputs - matching original"""
        return [
            {"pred_logits": a, "pred_masks": b}
            for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ]


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q_embed, query_pos=None, attn_mask=None, padding_mask=None):
        """Forward with optional args for compatibility"""
        q = k = self.with_pos_embed(q_embed, query_pos)
        q_embed2 = self.self_attn(
            q, k, value=q_embed,
            attn_mask=None,  # Don't use dynamic masks for ONNX
            key_padding_mask=None
        )[0]
        q_embed = q_embed + self.dropout(q_embed2)
        q_embed = self.norm(q_embed)
        return q_embed
    
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q_embed, bb_feat, attn_mask=None, padding_mask=None, 
                pos=None, query_pos=None):
        """Forward with all original args for compatibility"""
        q_embed = self.norm(q_embed)
        q = self.with_pos_embed(q_embed, query_pos)
        k = v = self.with_pos_embed(bb_feat, pos)
        
        q_embed2 = self.multihead_attn(
            query=q,
            key=k,
            value=v,
            attn_mask=None,  # Don't use for ONNX
            key_padding_mask=None
        )[0]
        q_embed = q_embed + self.dropout(q_embed2)
        return q_embed
    
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos


class FFNLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = F.relu
    
    def forward(self, tgt):
        tgt = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        return tgt


class MLP(nn.Module):
    """MLP as in original"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PositionalEncoder(nn.Module):
    """Matching the original positional encoder interface"""
    def __init__(self, cfg):
        super().__init__()
        # Handle both config object and dict
        if hasattr(cfg, 'MAX_FREQ'):
            self.max_freq = cfg.MAX_FREQ
            self.dimensionality = cfg.DIMENSIONALITY
            self.num_bands = math.floor(cfg.FEAT_SIZE / cfg.DIMENSIONALITY / 2)
            self.base = cfg.BASE
            feat_size = cfg.FEAT_SIZE
        else:
            # Default values
            self.max_freq = 10000
            self.dimensionality = 3
            feat_size = 256
            self.num_bands = math.floor(feat_size / self.dimensionality / 2)
            self.base = 2
        
        pad = feat_size - self.num_bands * 2 * self.dimensionality
        self.zero_pad = nn.ZeroPad2d((pad, 0, 0, 0))  # left padding
    
    def forward(self, _x):
        """
        _x [B,N,3]: batched point coordinates
        returns: [B,N,C]: positional encoding
        """
        # Handle different input shapes
        if len(_x.shape) == 2:
            _x = _x.unsqueeze(0)
        
        x = _x.clone()
        
        # Normalize coordinates (like original)
        x[:, :, 0] = x[:, :, 0] / 48
        x[:, :, 1] = x[:, :, 1] / 48
        if x.shape[2] > 2:
            x[:, :, 2] = x[:, :, 2] / 4
        
        x = x.unsqueeze(-1)
        
        scales = torch.logspace(
            0.0,
            math.log(self.max_freq / 2) / math.log(self.base),
            self.num_bands,
            base=self.base,
            device=x.device,
            dtype=x.dtype,
        )
        
        scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]
        x = x * scales * math.pi
        x = torch.cat([x.sin(), x.cos()], dim=-1)
        x = x.flatten(2)
        enc = self.zero_pad(x)
        return enc


# Also add compatibility for if something expects PositionalEncodingONNX
PositionalEncodingONNX = PositionalEncoder
