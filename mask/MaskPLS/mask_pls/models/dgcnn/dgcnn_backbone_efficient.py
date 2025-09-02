# Save this as: mask/MaskPLS/mask_pls/models/dgcnn/dgcnn_backbone_efficient.py
"""
Memory-efficient DGCNN backbone for point cloud feature extraction
Optimized for large point clouds without OOM errors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


def knn_chunked(x, k, chunk_size=10000):
    """
    Memory-efficient k-nearest neighbors using chunked processing
    Args:
        x: [B, C, N] input features
        k: number of neighbors
        chunk_size: points to process at once
    Returns:
        idx: [B, N, k] indices of k nearest neighbors
    """
    batch_size, channels, num_points = x.shape
    device = x.device
    
    # Process in chunks to avoid OOM
    idx_list = []
    
    for b in range(batch_size):
        x_b = x[b:b+1]  # [1, C, N]
        idx_b_list = []
        
        # Process query points in chunks
        for i in range(0, num_points, chunk_size):
            end_i = min(i + chunk_size, num_points)
            x_chunk = x_b[:, :, i:end_i]  # [1, C, chunk]
            
            # Compute distances to all points
            inner = -2 * torch.matmul(x_chunk.transpose(2, 1), x_b)
            xx_chunk = torch.sum(x_chunk**2, dim=1, keepdim=True)
            xx = torch.sum(x_b**2, dim=1, keepdim=True)
            pairwise_distance = -xx_chunk - inner - xx.transpose(2, 1)
            
            # Get k nearest neighbors for this chunk
            idx_chunk = pairwise_distance.topk(k=k, dim=-1)[1]
            idx_b_list.append(idx_chunk.squeeze(0))
            
            # Clear intermediate tensors
            del inner, xx_chunk, pairwise_distance
            
        idx_b = torch.cat(idx_b_list, dim=0)
        idx_list.append(idx_b)
    
    idx = torch.stack(idx_list, dim=0)
    return idx


def get_graph_feature_efficient(x, k=20, idx=None, chunk_size=10000):
    """
    Memory-efficient edge feature construction
    """
    batch_size, num_dims, num_points = x.shape
    device = x.device
    
    if idx is None:
        if num_points > chunk_size:
            idx = knn_chunked(x, k=k, chunk_size=chunk_size)
        else:
            # Original implementation for small point clouds
            inner = -2 * torch.matmul(x.transpose(2, 1), x)
            xx = torch.sum(x**2, dim=1, keepdim=True)
            pairwise_distance = -xx - inner - xx.transpose(2, 1)
            idx = pairwise_distance.topk(k=k, dim=-1)[1]
    
    # Process features in chunks
    x = x.transpose(2, 1).contiguous()  # [B, N, C]
    
    feature_list = []
    for b in range(batch_size):
        x_b = x[b]  # [N, C]
        idx_b = idx[b]  # [N, k]
        
        # Process in chunks
        feat_chunks = []
        for i in range(0, num_points, chunk_size):
            end_i = min(i + chunk_size, num_points)
            
            # Get features for this chunk
            idx_chunk = idx_b[i:end_i]  # [chunk, k]
            x_chunk = x_b[i:end_i].unsqueeze(1).repeat(1, k, 1)  # [chunk, k, C]
            
            # Gather neighbor features
            idx_flat = idx_chunk.reshape(-1)
            neighbors = x_b[idx_flat].reshape(end_i - i, k, num_dims)
            
            # Compute edge features
            feat_chunk = torch.cat((neighbors - x_chunk, x_chunk), dim=2)
            feat_chunks.append(feat_chunk)
            
            del neighbors, x_chunk
        
        feature_b = torch.cat(feat_chunks, dim=0)
        feature_list.append(feature_b)
    
    feature = torch.stack(feature_list, dim=0)
    feature = feature.permute(0, 3, 1, 2).contiguous()
    
    return feature


class EfficientEdgeConv(nn.Module):
    """
    Memory-efficient Edge convolution layer
    """
    def __init__(self, in_channels, out_channels, k=20):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
    
    def forward(self, x, idx=None):
        # Use efficient graph feature extraction
        x = get_graph_feature_efficient(x, k=self.k, idx=idx)
        x = self.conv(x)
        x = x.max(dim=-1, keepdim=False)[0]
        return x


class EfficientDGCNNBackbone(nn.Module):
    """
    Memory-efficient DGCNN backbone
    """
    def __init__(self, cfg):
        super().__init__()
        
        self.k = 20  # number of nearest neighbors
        self.chunk_size = cfg.get('CHUNK_SIZE', 10000)  # Process points in chunks
        input_dim = cfg.INPUT_DIM
        output_channels = cfg.CHANNELS
        
        # EdgeConv layers
        self.conv1 = EfficientEdgeConv(input_dim, 64, k=self.k)
        self.conv2 = EfficientEdgeConv(64, 64, k=self.k)
        self.conv3 = EfficientEdgeConv(64, 128, k=self.k)
        self.conv4 = EfficientEdgeConv(128, 256, k=self.k)
        
        # Aggregation
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        
        # Multi-scale feature extraction - expecting 1024 input channels (512 + 512 global)
        self.feat_layers = nn.ModuleList([
            nn.Conv1d(1024, output_channels[5], kernel_size=1),  # 256
            nn.Conv1d(1024, output_channels[6], kernel_size=1),  # 128
            nn.Conv1d(1024, output_channels[7], kernel_size=1),  # 96
            nn.Conv1d(1024, output_channels[8], kernel_size=1),  # 96
        ])
        
        # Batch normalization for outputs
        self.out_bn = nn.ModuleList([
            nn.BatchNorm1d(output_channels[i]) for i in range(5, 9)
        ])
        
        # Semantic head
        self.sem_head = nn.Linear(output_channels[8], 20)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def process_batch_chunked(self, coords_list, feats_list):
        """Process each point cloud in chunks to avoid OOM"""
        batch_size = len(coords_list)
        all_features = []
        all_coords = []
        all_masks = []
        
        for b in range(batch_size):
            coords = torch.from_numpy(coords_list[b]).float().cuda()
            feats = torch.from_numpy(feats_list[b]).float().cuda()
            
            # Process large point clouds in chunks
            if coords.shape[0] > self.chunk_size * 2:
                # Split into overlapping chunks for better feature extraction
                chunk_features = []
                overlap = self.chunk_size // 4
                
                for i in range(0, coords.shape[0], self.chunk_size - overlap):
                    end_i = min(i + self.chunk_size, coords.shape[0])
                    
                    # Extract chunk
                    coords_chunk = coords[i:end_i]
                    feats_chunk = feats[i:end_i]
                    
                    # Process chunk
                    x_in = torch.cat([coords_chunk, feats_chunk[:, 3:]], dim=1)
                    x_in = x_in.transpose(0, 1).unsqueeze(0)
                    
                    # Extract features for chunk
                    with torch.cuda.amp.autocast():
                        chunk_feat = self.forward_chunk(x_in)
                    
                    # Only keep non-overlapping part (except for last chunk)
                    if i > 0 and end_i < coords.shape[0]:
                        start_idx = overlap // 2
                        end_idx = -overlap // 2
                        chunk_feat = chunk_feat[:, :, start_idx:end_idx]
                    elif i > 0:
                        start_idx = overlap // 2
                        chunk_feat = chunk_feat[:, :, start_idx:]
                    elif end_i < coords.shape[0]:
                        end_idx = -overlap // 2
                        chunk_feat = chunk_feat[:, :, :end_idx]
                    
                    chunk_features.append(chunk_feat)
                
                # Concatenate chunk features
                x = torch.cat(chunk_features, dim=2)
            else:
                # Process normally for smaller point clouds
                x_in = torch.cat([coords, feats[:, 3:]], dim=1).transpose(0, 1).unsqueeze(0)
                x = self.forward_chunk(x_in)
            
            all_features.append(x)
            all_coords.append(coords)
            all_masks.append(torch.zeros(coords.shape[0], dtype=torch.bool).cuda())
        
        return all_features, all_coords, all_masks
    
    def forward_chunk(self, x_in):
        """Forward pass for a single chunk"""
        # DGCNN forward
        x1 = self.conv1(x_in)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        
        # Concatenate all edge conv outputs
        x = torch.cat((x1, x2, x3, x4), dim=1)  # Should be 64+64+128+256 = 512 channels
        x = self.conv5(x)  # Still 512 channels after conv5
        
        # Global features - use chunked max pooling for very large inputs
        if x.shape[2] > 50000:
            # Process global pooling in chunks
            chunk_size = 25000
            global_chunks = []
            for i in range(0, x.shape[2], chunk_size):
                chunk = x[:, :, i:i+chunk_size]
                global_chunks.append(F.adaptive_max_pool1d(chunk, 1))
            x_global = torch.max(torch.cat(global_chunks, dim=2), dim=2, keepdim=True)[0]
            x_global = x_global.expand(-1, -1, x.size(2))
        else:
            x_global = F.adaptive_max_pool1d(x, 1).expand(-1, -1, x.size(2))
        
        # Concatenate with global features to get 1024 channels total
        x = torch.cat((x, x_global), dim=1)  # 512 + 512 = 1024 channels
        return x
    
    def forward(self, x):
        """
        Args:
            x: dict with 'pt_coord' and 'feats' lists
        Returns:
            multi-scale features, coordinates, padding masks, semantic logits
        """
        coords_list = x['pt_coord']
        feats_list = x['feats']
        batch_size = len(coords_list)
        
        # Process with chunking for memory efficiency
        all_features, all_coords, all_masks = self.process_batch_chunked(coords_list, feats_list)
        
        # Generate multi-scale features
        ms_features = []
        ms_coords = []
        ms_masks = []
        
        # Process each scale level
        for i, (feat_layer, bn_layer) in enumerate(zip(self.feat_layers, self.out_bn)):
            level_features = []
            level_coords = []
            level_masks = []
            
            for b in range(batch_size):
                # Extract features for this level
                feat = feat_layer(all_features[b]).squeeze(0).transpose(0, 1)
                feat = bn_layer(feat.transpose(0, 1)).transpose(0, 1)
                
                level_features.append(feat)
                level_coords.append(all_coords[b])
                level_masks.append(all_masks[b])
            
            # Pad to same size
            max_points = max(f.shape[0] for f in level_features)
            padded_features = []
            padded_coords = []
            padded_masks = []
            
            for feat, coord, mask in zip(level_features, level_coords, level_masks):
                n_points = feat.shape[0]
                if n_points < max_points:
                    pad_size = max_points - n_points
                    feat = F.pad(feat, (0, 0, 0, pad_size))
                    coord = F.pad(coord, (0, 0, 0, pad_size))
                    mask = F.pad(mask, (0, pad_size), value=True)
                
                padded_features.append(feat)
                padded_coords.append(coord)
                padded_masks.append(mask)
            
            ms_features.append(torch.stack(padded_features))
            ms_coords.append(torch.stack(padded_coords))
            ms_masks.append(torch.stack(padded_masks))
        
        # Semantic predictions
        sem_features = ms_features[-1]
        sem_logits = []
        
        for b in range(batch_size):
            valid_mask = ~ms_masks[-1][b]
            valid_features = sem_features[b][valid_mask]
            
            # Process semantic predictions in chunks if needed
            if valid_features.shape[0] > 50000:
                sem_chunks = []
                for i in range(0, valid_features.shape[0], 50000):
                    chunk = valid_features[i:i+50000]
                    sem_chunks.append(self.sem_head(chunk))
                sem_logit = torch.cat(sem_chunks, dim=0)
            else:
                sem_logit = self.sem_head(valid_features)
            
            # Pad back
            if valid_mask.sum() < sem_features.shape[1]:
                full_logit = torch.zeros(sem_features.shape[1], 20).cuda()
                full_logit[valid_mask] = sem_logit
                sem_logit = full_logit
            
            sem_logits.append(sem_logit)
        
        sem_logits = torch.stack(sem_logits)
        
        # Clear intermediate tensors
        del all_features
        torch.cuda.empty_cache()
        
        return ms_features, ms_coords, ms_masks, sem_logits