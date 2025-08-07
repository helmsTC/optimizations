"""
Minimal working ONNX decoder - fixes all interface issues
Replace mask_pls/onnx/onnx_decoder.py with this
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict


class ONNXCompatibleDecoder(nn.Module):
    """
    ONNX-compatible MaskedTransformerDecoder
    This version properly handles the backbone output format
    """
    def __init__(self, cfg, bb_cfg, data_cfg):
        super().__init__()
        
        hidden_dim = cfg.HIDDEN_DIM
        self.hidden_dim = hidden_dim
        self.num_queries = cfg.NUM_QUERIES
        self.num_classes = data_cfg.NUM_CLASSES
        self.num_feature_levels = cfg.FEATURE_LEVELS
        self.num_layers = min(cfg.FEATURE_LEVELS * cfg.DEC_BLOCKS, 9)  # Cap at 9
        self.nheads = cfg.NHEADS
        
        # Query embeddings
        self.query_feat = nn.Parameter(torch.randn(1, cfg.NUM_QUERIES, hidden_dim))
        self.query_embed = nn.Parameter(torch.randn(1, cfg.NUM_QUERIES, hidden_dim))
        
        # Level embedding
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        
        # Input projections - match backbone output channels
        in_channels = bb_cfg.CHANNELS
        # The backbone outputs features at different scales
        # Typically: [256, 128, 96, 96] for the last 4 levels
        self.input_proj = nn.ModuleList()
        
        # Take the last num_feature_levels channels
        start_idx = len(in_channels) - self.num_feature_levels - 1
        for i in range(self.num_feature_levels):
            ch = in_channels[start_idx + i] if start_idx + i < len(in_channels) else hidden_dim
            if ch != hidden_dim:
                self.input_proj.append(nn.Linear(ch, hidden_dim))
            else:
                self.input_proj.append(nn.Identity())
        
        # Mask feature projection (last feature level)
        last_ch = in_channels[-1]
        if last_ch != hidden_dim:
            self.mask_feat_proj = nn.Linear(last_ch, hidden_dim)
        else:
            self.mask_feat_proj = nn.Identity()
        
        # Transformer layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        
        for _ in range(self.num_layers):
            # Self attention
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(hidden_dim, self.nheads)
            )
            # Cross attention
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(hidden_dim, self.nheads)
            )
            # FFN
            self.transformer_ffn_layers.append(
                FFNLayer(hidden_dim, cfg.DIM_FFN)
            )
        
        # Output heads
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.class_embed = nn.Linear(hidden_dim, data_cfg.NUM_CLASSES + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        
        # Simple positional encoding
        self.pos_enc = PositionalEncoding(hidden_dim)
    
    def forward(self, feats: List[torch.Tensor], 
                coors: List[torch.Tensor],
                pad_masks: List[torch.Tensor]) -> tuple:
        """
        Args:
            feats: List of feature tensors [B, N, C] at different scales
            coors: List of coordinate tensors [B, N, 3]  
            pad_masks: List of padding masks [B, N]
        Returns:
            outputs: dict with pred_logits and pred_masks
            padding: padding mask for last level
        """
        # Handle empty input
        if not feats or len(feats) == 0:
            B = 1
            outputs = {
                'pred_logits': torch.zeros(B, self.num_queries, self.num_classes + 1),
                'pred_masks': torch.zeros(B, 1, self.num_queries),
                'aux_outputs': []
            }
            padding = torch.zeros(B, 1, dtype=torch.bool)
            return outputs, padding
        
        # Extract mask features from last level
        last_coors = coors[-1] if coors else None
        mask_features = self.mask_feat_proj(feats[-1])
        
        # Add positional encoding
        if mask_features.shape[1] > 0:
            mask_features = mask_features + self.pos_enc(mask_features)
        
        last_pad = pad_masks[-1] if pad_masks else None
        
        # Prepare multi-scale features
        src = []
        pos = []
        
        # Only use first num_feature_levels features
        for i in range(min(self.num_feature_levels, len(feats) - 1)):
            feat = self.input_proj[i](feats[i])
            src.append(feat)
            # Simple positional encoding
            pos.append(self.pos_enc(feat))
        
        # Initialize queries
        bs = feats[0].shape[0]
        query_embed = self.query_embed.expand(bs, -1, -1)
        output = self.query_feat.expand(bs, -1, -1)
        
        predictions_class = []
        predictions_mask = []
        
        # Initial prediction
        outputs_class, outputs_mask = self.pred_heads(
            output, mask_features, last_pad
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        
        # Transformer layers
        for i in range(self.num_layers):
            level_index = i % len(src) if src else 0
            
            if level_index < len(src):
                # Self-attention
                output = self.transformer_self_attention_layers[i](
                    output, query_pos=query_embed
                )
                
                # Cross-attention
                output = self.transformer_cross_attention_layers[i](
                    output, 
                    src[level_index],
                    pos=pos[level_index] if level_index < len(pos) else None,
                    query_pos=query_embed
                )
                
                # FFN
                output = self.transformer_ffn_layers[i](output)
                
                # Predict
                outputs_class, outputs_mask = self.pred_heads(
                    output, mask_features, last_pad
                )
                predictions_class.append(outputs_class)
                predictions_mask.append(outputs_mask)
        
        # Output
        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self.set_aux(predictions_class[:-1], predictions_mask[:-1])
        }
        
        return out, last_pad if last_pad is not None else torch.zeros(bs, mask_features.shape[1], dtype=torch.bool)
    
    def pred_heads(self, output, mask_features, pad_mask=None):
        """Predict classes and masks"""
        decoder_output = self.decoder_norm(output)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        
        # Compute mask logits
        if mask_features.shape[1] == 0:
            B, Q = output.shape[:2]
            outputs_mask = torch.zeros(B, 1, Q, device=output.device)
        else:
            outputs_mask = torch.einsum("bqc,bpc->bpq", mask_embed, mask_features)
        
        # No dynamic attention mask generation for ONNX
        return outputs_class, outputs_mask
    
    def set_aux(self, outputs_class, outputs_seg_masks):
        """Set auxiliary outputs"""
        return [
            {"pred_logits": a, "pred_masks": b}
            for a, b in zip(outputs_class, outputs_seg_masks)
        ]


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q_embed, query_pos=None):
        if query_pos is not None:
            q = k = q_embed + query_pos
        else:
            q = k = q_embed
        q_embed2 = self.self_attn(q, k, value=q_embed)[0]
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
    
    def forward(self, q_embed, bb_feat, pos=None, query_pos=None):
        q_embed = self.norm(q_embed)
        q = q_embed + query_pos if query_pos is not None else q_embed
        k = v = bb_feat + pos if pos is not None else bb_feat
        
        q_embed2 = self.multihead_attn(
            query=q,
            key=k,
            value=v
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
    
    def forward(self, tgt):
        tgt = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        return tgt


class MLP(nn.Module):
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        """Add positional encoding"""
        if x.shape[1] > self.pe.shape[1]:
            return x
        return x + self.pe[:, :x.shape[1], :self.d_model]
