"""
ONNX-compatible decoder for MaskPLS
This version matches EXACTLY the original decoder's attribute names
Replace mask_pls/onnx/onnx_decoder.py with this
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict
import math


class PositionalEncoder(nn.Module):
    """MUST match the original from mask_pls/models/positional_encoder.py"""
    def __init__(self, cfg):
        super().__init__()
        self.max_freq = cfg.MAX_FREQ
        self.dimensionality = cfg.DIMENSIONALITY
        self.num_bands = math.floor(cfg.FEAT_SIZE / cfg.DIMENSIONALITY / 2)
        self.base = cfg.BASE
        pad = cfg.FEAT_SIZE - self.num_bands * 2 * cfg.DIMENSIONALITY
        self.zero_pad = nn.ZeroPad2d((pad, 0, 0, 0))  # left padding

    def forward(self, _x):
        """
        _x [B,N,3]: batched point coordinates
        returns: [B,N,C]: positional encoding of dimension C
        """
        x = _x.clone()
        x[:, :, 0] = x[:, :, 0] / 48
        x[:, :, 1] = x[:, :, 1] / 48
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
        # reshaping
        scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]
        x = x * scales * math.pi
        x = torch.cat([x.sin(), x.cos()], dim=-1)
        x = x.flatten(2)
        enc = self.zero_pad(x)
        return enc


class ONNXCompatibleDecoder(nn.Module):
    """
    ONNX-compatible version of the MaskedTransformerDecoder
    Maintains exact same attribute names as original
    """
    def __init__(self, cfg, bb_cfg, data_cfg):
        super().__init__()
        hidden_dim = cfg.HIDDEN_DIM
        
        # Add POS_ENC config with FEAT_SIZE
        cfg.POS_ENC.FEAT_SIZE = cfg.HIDDEN_DIM
        
        # MUST use pe_layer like original, not pos_encoder!
        self.pe_layer = PositionalEncoder(cfg.POS_ENC)
        
        self.num_layers = cfg.FEATURE_LEVELS * cfg.DEC_BLOCKS
        self.nheads = cfg.NHEADS
        self.num_queries = cfg.NUM_QUERIES
        self.num_feature_levels = cfg.FEATURE_LEVELS
        
        # Transformer layers - same names as original
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        
        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=self.nheads,
                    dropout=0.0
                )
            )
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=self.nheads,
                    dropout=0.0
                )
            )
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=cfg.DIM_FFN,
                    dropout=0.0
                )
            )
        
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        
        # Query embeddings - MUST be nn.Embedding like original
        self.query_feat = nn.Embedding(cfg.NUM_QUERIES, hidden_dim)
        self.query_embed = nn.Embedding(cfg.NUM_QUERIES, hidden_dim)
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        
        # Mask feature projection
        self.mask_feat_proj = nn.Sequential()
        in_channels = bb_cfg.CHANNELS
        if in_channels[-1] != hidden_dim:
            self.mask_feat_proj = nn.Linear(in_channels[-1], hidden_dim)
        
        # Input projections
        in_channels = in_channels[:-1][-self.num_feature_levels:]
        self.input_proj = nn.ModuleList()
        for ch in in_channels:
            if ch != hidden_dim:
                self.input_proj.append(nn.Linear(ch, hidden_dim))
            else:
                self.input_proj.append(nn.Sequential())
        
        # Output FFNs
        self.class_embed = nn.Linear(hidden_dim, data_cfg.NUM_CLASSES + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
    
    def forward(self, feats, coors, pad_masks):
        """
        Forward pass matching original decoder interface
        Returns: (outputs dict, padding mask)
        """
        # Pop last level for mask features (like original)
        last_coors = coors.pop()
        mask_features = self.mask_feat_proj(feats.pop()) + self.pe_layer(last_coors)
        last_pad = pad_masks.pop()
        
        src = []
        pos = []
        size_list = []
        
        for i in range(self.num_feature_levels):
            size_list.append(feats[i].shape[1])
            pos.append(self.pe_layer(coors[i]))
            feat = self.input_proj[i](feats[i])
            src.append(feat)
        
        bs = src[0].shape[0]
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        output = self.query_feat.weight.unsqueeze(0).repeat(bs, 1, 1)
        
        predictions_class = []
        predictions_mask = []
        
        # Initial prediction
        outputs_class, outputs_mask, _ = self.pred_heads(
            output, mask_features, pad_mask=last_pad
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        
        # Process through transformer layers (cap at 9 for ONNX)
        for i in range(min(self.num_layers, 9)):
            level_index = i % self.num_feature_levels
            
            # REMOVE the attention mask generation for ONNX
            # No: if attn_mask is not None: ...
            
            # Cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output,
                src[level_index],
                attn_mask=None,  # Don't use dynamic masks
                padding_mask=None,  # Don't use padding masks for ONNX
                pos=pos[level_index],
                query_pos=query_embed,
            )
            
            # Self-attention
            output = self.transformer_self_attention_layers[i](
                output,
                attn_mask=None,
                padding_mask=None,
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
        
        assert len(predictions_class) == min(self.num_layers + 1, 10)
        
        out = {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1]
        }
        
        out["aux_outputs"] = self.set_aux(predictions_class, predictions_mask)
        
        return out, last_pad
    
    def pred_heads(self, output, mask_features, pad_mask=None):
        """Prediction heads matching original"""
        decoder_output = self.decoder_norm(output)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bpc->bpq", mask_embed, mask_features)
        
        # For ONNX, return None for attention mask
        attn_mask = None
        
        return outputs_class, outputs_mask, attn_mask
    
    @torch.jit.unused
    def set_aux(self, outputs_class, outputs_seg_masks):
        """Set auxiliary outputs"""
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
        self.activation = F.relu
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, q_embed, attn_mask=None, padding_mask=None, query_pos=None):
        q = k = self.with_pos_embed(q_embed, query_pos)
        q_embed2 = self.self_attn(
            q, k, value=q_embed,
            attn_mask=None,  # Don't use for ONNX
            key_padding_mask=None
        )[0]
        q_embed = q_embed + self.dropout(q_embed2)
        q_embed = self.norm(q_embed)
        return q_embed


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, q_embed, bb_feat, attn_mask=None, padding_mask=None,
                pos=None, query_pos=None):
        q_embed = self.norm(q_embed)
        q_embed2 = self.multihead_attn(
            query=self.with_pos_embed(q_embed, query_pos),
            key=self.with_pos_embed(bb_feat, pos),
            value=self.with_pos_embed(bb_feat, pos),
            attn_mask=None,  # Don't use for ONNX
            key_padding_mask=None
        )[0]
        q_embed = q_embed + self.dropout(q_embed2)
        return q_embed


class FFNLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = F.relu
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt):
        tgt = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        return tgt


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""
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


# Alias for compatibility
PositionalEncodingONNX = PositionalEncoder
