# mask_pls/models/dgcnn/dgcnn_backbone_efficient.py
"""
Efficient DGCNN backbone for point cloud feature extraction
Optimized for memory usage and ONNX compatibility
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


def knn(x, k):
    """
    Get k nearest neighbors index
    Args:
        x: [B, C, N] input features
        k: number of neighbors
    Returns:
        idx: [B, N, k] indices of k nearest neighbors
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None):
    """
    Construct edge features for each point
    Args:
        x: [B, C, N]
        k: number of neighbors
        idx: pre-computed indices
    Returns:
        edge features: [B, 2*C, N, k]
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    
    if idx is None:
        idx = knn(x, k=k)
    
    device = x.device
    
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    
    _, num_dims, _ = x.size()
    
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    
    return feature


class EdgeConv(nn.Module):
    """
    Edge convolution layer
    """
    def __init__(self, in_channels, out_channels, k=20):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def forward(self, x, idx=None):
        x = get_graph_feature(x, k=self.k, idx=idx)
        x = self.conv(x)
        x = x.max(dim=-1, keepdim=False)[0]
        return x


class EfficientDGCNNBackbone(nn.Module):
    """
    Efficient DGCNN backbone for point cloud feature extraction
    """
    def __init__(self, cfg):
        super().__init__()
        
        self.k = 20  # number of nearest neighbors
        input_dim = cfg.INPUT_DIM
        output_channels = cfg.CHANNELS
        
        # Ensure all operations use float32
        self.dtype = torch.float32
        
        # EdgeConv layers
        self.conv1 = EdgeConv(input_dim, 64, k=self.k)
        self.conv2 = EdgeConv(64, 64, k=self.k)
        self.conv3 = EdgeConv(64, 128, k=self.k)
        self.conv4 = EdgeConv(128, 256, k=self.k)
        
        # Aggregation
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # Multi-scale feature extraction matching MaskPLS architecture
        self.feat_layers = nn.ModuleList([
            nn.Conv1d(512, output_channels[5], kernel_size=1),  # 256
            nn.Conv1d(512, output_channels[6], kernel_size=1),  # 128
            nn.Conv1d(512, output_channels[7], kernel_size=1),  # 96
            nn.Conv1d(512, output_channels[8], kernel_size=1),  # 96
        ])
        
        # Batch normalization for outputs
        self.out_bn = nn.ModuleList([
            nn.BatchNorm1d(output_channels[i]) for i in range(5, 9)
        ])
        
        # Semantic head
        self.sem_head = nn.Linear(output_channels[8], 20)
        
        # Initialize weights
        self.init_weights()
        
        # Ensure float32
        self.to(self.dtype)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: dict with 'pt_coord' and 'feats' lists
        Returns:
            multi-scale features, coordinates, padding masks, semantic logits
        """
        # Prepare batch
        coords_list = x['pt_coord']
        feats_list = x['feats']
        
        batch_size = len(coords_list)
        
        # Process each point cloud
        all_features = []
        all_coords = []
        all_masks = []
        original_sizes = []
        
        for b in range(batch_size):
            coords = torch.from_numpy(coords_list[b]).float().cuda()
            feats = torch.from_numpy(feats_list[b]).float().cuda()
            
            # Ensure float32 dtype
            coords = coords.to(self.dtype)
            feats = feats.to(self.dtype)
            
            # Store original size
            n_points = coords.shape[0]
            original_sizes.append(n_points)
            
            # DGCNN forward pass
            point_features = self.process_single_cloud(coords, feats)
            
            all_features.append(point_features)
            all_coords.append(coords)
            all_masks.append(torch.zeros(n_points, dtype=torch.bool, device=coords.device))
        
        # Generate multi-scale features with consistent sizing
        ms_features = []
        ms_coords = []
        ms_masks = []
        
        for i in range(len(self.feat_layers)):
            level_features = []
            level_coords = []
            level_masks = []
            
            for b in range(batch_size):
                feat = all_features[b][i]
                coord = all_coords[b]
                mask = all_masks[b]
                
                # Ensure features and coords have same number of points
                n_feat = feat.shape[0]
                n_coord = coord.shape[0]
                
                if n_feat != n_coord:
                    # This handles the size mismatch error
                    # Use the original size to determine correct size
                    target_size = original_sizes[b]
                    
                    if n_feat > target_size:
                        feat = feat[:target_size]
                    elif n_feat < target_size:
                        # This shouldn't happen, but handle it gracefully
                        pad_size = target_size - n_feat
                        feat = F.pad(feat, (0, 0, 0, pad_size))
                    
                    if n_coord > target_size:
                        coord = coord[:target_size]
                        mask = mask[:target_size]
                    elif n_coord < target_size:
                        # This shouldn't happen, but handle it gracefully
                        pad_size = target_size - n_coord
                        coord = F.pad(coord, (0, 0, 0, pad_size))
                        mask = F.pad(mask, (0, pad_size), value=True)
                
                level_features.append(feat)
                level_coords.append(coord)
                level_masks.append(mask)
            
            # Pad to same size within batch
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
            
            if valid_features.shape[0] > 0:
                # Ensure float32 for semantic head
                valid_features = valid_features.to(self.dtype)
                sem_logit = self.sem_head(valid_features)
            else:
                sem_logit = torch.zeros(0, 20, device=sem_features.device, dtype=self.dtype)
            
            # Pad back to full size
            if valid_mask.sum() < sem_features.shape[1]:
                full_logit = torch.zeros(sem_features.shape[1], 20, device=sem_features.device, dtype=self.dtype)
                if valid_features.shape[0] > 0:
                    full_logit[valid_mask] = sem_logit
                sem_logit = full_logit
            
            sem_logits.append(sem_logit)
        
        sem_logits = torch.stack(sem_logits)
        
        return ms_features, ms_coords, ms_masks, sem_logits
    
    def process_single_cloud(self, coords, feats):
        """Process a single point cloud and return multi-scale features"""
        # Get number of points
        n_points = coords.shape[0]
        
        # Combine coordinates and features
        x_in = torch.cat([coords, feats[:, 3:]], dim=1).transpose(0, 1).unsqueeze(0)
        
        # Ensure float32
        x_in = x_in.to(self.dtype)
        
        # DGCNN forward with efficient memory usage
        with torch.cuda.amp.autocast(enabled=False):  # Disable autocast to prevent half precision
            x1 = self.conv1(x_in)
            x2 = self.conv2(x1)
            x3 = self.conv3(x2)
            x4 = self.conv4(x3)
            
            # Concatenate all edge conv outputs
            x = torch.cat((x1, x2, x3, x4), dim=1)
            x = self.conv5(x)
            
            # Global max pooling
            x_global = F.adaptive_max_pool1d(x, 1)
            x_global = x_global.expand(-1, -1, n_points)
            
            # Concatenate with global features
            x = torch.cat((x, x_global), dim=1)
        
        # Generate features at each scale
        features = []
        for feat_layer, bn_layer in zip(self.feat_layers, self.out_bn):
            # Apply layer and batch norm
            feat = feat_layer(x)  # [1, C, N]
            feat = bn_layer(feat)
            feat = feat.squeeze(0).transpose(0, 1)  # [N, C]
            
            # Ensure correct size
            if feat.shape[0] != n_points:
                # This shouldn't happen, but handle it gracefully
                if feat.shape[0] > n_points:
                    feat = feat[:n_points]
                else:
                    pad_size = n_points - feat.shape[0]
                    feat = F.pad(feat, (0, 0, 0, pad_size))
            
            features.append(feat)
        
        return features
    
    def to(self, *args, **kwargs):
        """Override to ensure float32 is maintained"""
        super().to(*args, **kwargs)
        # Force float32 for all parameters
        if len(args) > 0 and args[0] == torch.float16:
            return super().to(torch.float32)
        return self