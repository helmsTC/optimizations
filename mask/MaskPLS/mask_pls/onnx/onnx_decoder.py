"""
ONNX-compatible decoder for MaskPLS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict


class ONNXCompatibleDecoder(nn.Module):
    """
    ONNX-compatible version of the MaskedTransformerDecoder
    Uses standard PyTorch operations without custom CUDA kernels
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
        
        # Learnable query embeddings
        self.query_feat = nn.Parameter(torch.randn(1, cfg.NUM_QUERIES, hidden_dim))
        self.query_pos = nn.Parameter(torch.randn(1, cfg.NUM_QUERIES, hidden_dim))
        
        # Level embeddings for multi-scale features
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncodingONNX(hidden_dim)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            layer = TransformerDecoderLayerONNX(
                d_model=hidden_dim,
                nhead=self.nheads,
                dim_feedforward=cfg.DIM_FFN,
                dropout=0.0
            )
            self.transformer_layers.append(layer)
        
        # Feature projection layers
        in_channels = bb_cfg.CHANNELS
        self.input_proj = nn.ModuleList()
        
        # Project each feature level to hidden_dim
        for i in range(self.num_feature_levels):
            ch_idx = -(self.num_feature_levels - i)
            ch = in_channels[ch_idx]
            
            if ch != hidden_dim:
                self.input_proj.append(nn.Linear(ch, hidden_dim))
            else:
                self.input_proj.append(nn.Identity())
        
        # Mask feature projection
        if in_channels[-1] != hidden_dim:
            self.mask_feat_proj = nn.Linear(in_channels[-1], hidden_dim)
        else:
            self.mask_feat_proj = nn.Identity()
        
        # Output heads
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.class_embed = nn.Linear(hidden_dim, data_cfg.NUM_CLASSES + 1)
        
        # MLP for mask embedding
        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, feats: List[torch.Tensor], 
                mask_features: torch.Tensor,
                pad_masks: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        Args:
            feats: List of [B, N, C] feature tensors at different scales
            mask_features: [B, N, C] features for mask prediction
            pad_masks: List of [B, N] padding masks
        Returns:
            Dictionary with:
                - pred_logits: [B, Q, num_classes+1]
                - pred_masks: [B, N, Q]
                - aux_outputs: List of intermediate predictions
        """
        B = feats[0].shape[0] if feats else 1
        
        # Initialize queries
        query_feat = self.query_feat.expand(B, -1, -1)
        query_pos = self.query_pos.expand(B, -1, -1)
        
        # Project mask features and add positional encoding
        mask_features = self.mask_feat_proj(mask_features)
        if mask_features.shape[1] > 0:
            mask_features = mask_features + self.pos_encoder(mask_features)
        
        # Store predictions at each layer
        predictions_class = []
        predictions_mask = []
        
        # Initial prediction
        outputs_class, outputs_mask = self.predict_masks(
            query_feat, mask_features
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        
        # Process through transformer layers
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            
            # Get features for current level
            if level_index < len(feats):
                level_features = self.input_proj[level_index](feats[level_index])
                
                # Add level embedding and positional encoding
                level_emb = self.level_embed.weight[level_index:level_index+1]
                level_features = level_features + level_emb
                level_features = level_features + self.pos_encoder(level_features)
                
                # Get padding mask for current level
                if level_index < len(pad_masks):
                    level_pad_mask = pad_masks[level_index]
                else:
                    level_pad_mask = None
                
                # Generate attention mask from previous predictions
                if i > 0 and predictions_mask[-1] is not None:
                    attn_mask = self.generate_attention_mask(
                        predictions_mask[-1], 
                        level_pad_mask
                    )
                else:
                    attn_mask = None
                
                # Transformer layer
                query_feat = self.transformer_layers[i](
                    query_feat,
                    level_features,
                    query_pos=query_pos,
                    memory_mask=attn_mask,
                    memory_key_padding_mask=level_pad_mask
                )
                
                # Predict at this layer
                outputs_class, outputs_mask = self.predict_masks(
                    query_feat, mask_features
                )
                predictions_class.append(outputs_class)
                predictions_mask.append(outputs_mask)
        
        # Final output
        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self.set_aux_outputs(
                predictions_class[:-1], 
                predictions_mask[:-1]
            )
        }
        
        return out
    
    def predict_masks(self, query_feat: torch.Tensor, 
                     mask_features: torch.Tensor) -> tuple:
        """
        Predict class logits and masks from query features
        """
        # Normalize query features
        decoder_output = self.decoder_norm(query_feat)
        
        # Predict classes
        outputs_class = self.class_embed(decoder_output)
        
        # Generate mask embeddings
        mask_embed = self.mask_embed(decoder_output)
        
        # Compute mask logits
        if mask_features.shape[1] > 0:
            # [B, N, Q] = [B, N, C] x [B, Q, C]^T
            outputs_mask = torch.bmm(mask_features, mask_embed.transpose(1, 2))
        else:
            # Empty mask
            B, Q = query_feat.shape[:2]
            outputs_mask = torch.zeros(B, 1, Q, device=query_feat.device)
        
        return outputs_class, outputs_mask
    
    def generate_attention_mask(self, mask_pred: torch.Tensor, 
                               pad_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Generate attention mask from predicted masks
        """
        # Threshold predicted masks
        attn_mask = (mask_pred.sigmoid() < 0.5)
        
        # Apply padding mask if provided
        if pad_mask is not None:
            attn_mask = attn_mask | pad_mask.unsqueeze(-1)
        
        # Reshape for multi-head attention
        B, N, Q = attn_mask.shape
        attn_mask = attn_mask.unsqueeze(1).expand(-1, self.nheads, -1, -1)
        attn_mask = attn_mask.reshape(B * self.nheads, N, Q)
        
        return attn_mask
    
    def set_aux_outputs(self, class_preds: list, mask_preds: list) -> list:
        """
        Format auxiliary outputs for loss computation
        """
        aux_outputs = []
        for cls_pred, mask_pred in zip(class_preds, mask_preds):
            aux_outputs.append({
                'pred_logits': cls_pred,
                'pred_masks': mask_pred
            })
        return aux_outputs


class TransformerDecoderLayerONNX(nn.Module):
    """
    ONNX-compatible transformer decoder layer
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, tgt, memory, query_pos=None, memory_mask=None,
                memory_key_padding_mask=None):
        """
        Forward pass
        Args:
            tgt: [B, Q, C] query features
            memory: [B, N, C] memory features
            query_pos: [B, Q, C] query positional encoding
            memory_mask: [B*nhead, N, Q] attention mask
            memory_key_padding_mask: [B, N] padding mask
        """
        # Self-attention
        q = k = tgt + query_pos if query_pos is not None else tgt
        tgt2, _ = self.self_attn(q, k, tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention
        q = tgt + query_pos if query_pos is not None else tgt
        tgt2, _ = self.cross_attn(
            q, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feed-forward
        tgt2 = self.ffn(tgt)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


class PositionalEncodingONNX(nn.Module):
    """
    ONNX-compatible positional encoding
    """
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        """
        Add positional encoding to input
        Args:
            x: [B, N, C] input features
        Returns:
            [B, N, C] features with positional encoding
        """
        return x + self.pe[:, :x.shape[1], :self.d_model]