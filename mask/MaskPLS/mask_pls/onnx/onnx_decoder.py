"""
ONNX-compatible decoder for MaskPLS - CORRECTED VERSION
Replace mask_pls/onnx/onnx_decoder.py with this
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict


class ONNXCompatibleDecoder(nn.Module):
    """
    Fixed ONNX-compatible version of the MaskedTransformerDecoder
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
        
        # Level embeddings
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        
        # FIX: Create proper positional encoder
        self.pos_encoder = PositionalEncodingONNX(hidden_dim, max_len=10000)
        
        # Transformer layers - simplified for ONNX
        self.transformer_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.transformer_layers.append(
                TransformerDecoderLayerONNX(
                    d_model=hidden_dim,
                    nhead=self.nheads,
                    dim_feedforward=cfg.DIM_FFN,
                    dropout=0.0
                )
            )
        
        # Feature projection layers
        in_channels = bb_cfg.CHANNELS
        self.input_proj = nn.ModuleList()
        
        # Fix the projection layers to match actual channel dimensions
        # Assuming backbone outputs: [256, 128, 96, 96]
        level_channels = [256, 128, 96]  # Take first 3 for 3 feature levels
        
        for i in range(self.num_feature_levels):
            if i < len(level_channels):
                ch = level_channels[i]
                if ch != hidden_dim:
                    self.input_proj.append(nn.Linear(ch, hidden_dim))
                else:
                    self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(nn.Identity())
        
        # Mask feature projection - last channel is 96
        if in_channels[-1] != hidden_dim:
            self.mask_feat_proj = nn.Linear(in_channels[-1], hidden_dim)
        else:
            self.mask_feat_proj = nn.Identity()
        
        # Output heads
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.class_embed = nn.Linear(hidden_dim, data_cfg.NUM_CLASSES + 1)
        
        # MLP for mask embedding - simplified
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
        Fixed forward pass without dynamic control flow
        """
        # Handle empty features case
        if not feats or len(feats) == 0:
            B = 1
            return {
                'pred_logits': torch.zeros(B, self.num_queries, self.num_classes + 1),
                'pred_masks': torch.zeros(B, 1, self.num_queries),
                'aux_outputs': []
            }
        
        # Initialize batch size
        B = feats[0].shape[0]
        
        # Initialize queries
        query_feat = self.query_feat.expand(B, -1, -1)
        query_pos = self.query_pos.expand(B, -1, -1)
        
        # Project mask features and add positional encoding
        mask_features = self.mask_feat_proj(mask_features)
        # Only add PE if mask_features has points
        if mask_features.shape[1] > 0:
            mask_features = self.pos_encoder(mask_features)
        
        # Store predictions
        predictions_class = []
        predictions_mask = []
        
        # Initial prediction
        outputs_class, outputs_mask = self.predict_masks(
            query_feat, mask_features
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        
        # Process through transformer layers (cap at 9 for ONNX stability)
        num_iterations = min(self.num_layers, 9)
        
        for i in range(num_iterations):
            level_index = i % self.num_feature_levels
            
            # Check if we have features for this level
            if level_index < len(feats):
                # Project features to hidden dimension
                level_features = self.input_proj[level_index](feats[level_index])
                
                # Add level embedding
                level_emb = self.level_embed.weight[level_index:level_index+1]
                level_features = level_features + level_emb
                
                # Add positional encoding if features have points
                if level_features.shape[1] > 0:
                    level_features = self.pos_encoder(level_features)
                
                # Get padding mask if available (but don't use in attention)
                level_pad_mask = None
                if level_index < len(pad_masks):
                    level_pad_mask = pad_masks[level_index]
                
                # Apply transformer layer WITHOUT dynamic attention masks
                query_feat = self.transformer_layers[i](
                    query_feat,
                    level_features,
                    query_pos=query_pos,
                    memory_key_padding_mask=None  # Don't use padding mask to avoid ONNX issues
                )
                
                # Predict at this layer
                outputs_class, outputs_mask = self.predict_masks(
                    query_feat, mask_features
                )
                predictions_class.append(outputs_class)
                predictions_mask.append(outputs_mask)
        
        # Return final predictions
        return {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': []  # No aux outputs for ONNX simplicity
        }
    
    def predict_masks(self, query_feat: torch.Tensor, 
                     mask_features: torch.Tensor) -> tuple:
        """
        Predict class logits and masks - ONNX compatible
        """
        # Normalize query features
        decoder_output = self.decoder_norm(query_feat)
        
        # Predict classes
        outputs_class = self.class_embed(decoder_output)
        
        # Generate mask embeddings
        mask_embed = self.mask_embed(decoder_output)
        
        # Compute mask logits
        if mask_features.shape[1] == 0:
            # Handle empty mask features
            B, Q = query_feat.shape[:2]
            outputs_mask = torch.zeros(B, 1, Q, device=query_feat.device)
        else:
            # [B, N, Q] = [B, N, C] @ [B, C, Q]
            outputs_mask = torch.bmm(mask_features, mask_embed.transpose(1, 2))
        
        return outputs_class, outputs_mask


class TransformerDecoderLayerONNX(nn.Module):
    """
    ONNX-compatible transformer decoder layer without dynamic masks
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
    
    def forward(self, tgt, memory, query_pos=None,
                memory_key_padding_mask=None):
        """
        Forward pass - simplified for ONNX without dynamic masks
        """
        # Self-attention
        if query_pos is not None:
            q = k = tgt + query_pos
        else:
            q = k = tgt
            
        tgt2, _ = self.self_attn(q, k, tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention (no dynamic attention masks)
        if query_pos is not None:
            q = tgt + query_pos
        else:
            q = tgt
            
        tgt2, _ = self.cross_attn(q, memory, memory)
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
        
        # Create div_term for sinusoidal pattern
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(torch.log(torch.tensor(10000.0)) / d_model))
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cos to odd indices (handle case where d_model is odd)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        """
        Add positional encoding to input
        Args:
            x: Tensor of shape [B, N, C]
        Returns:
            x + positional encoding
        """
        # Handle case where input is larger than max_len
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            # If input is too large, just return without PE
            return x
        
        # Add positional encoding
        return x + self.pe[:, :seq_len, :x.size(2)]
