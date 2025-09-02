# mask_pls/models/dgcnn/dgcnn_backbone.py
"""
Dynamic Graph CNN (DGCNN) backbone for point cloud feature extraction
Complete version with memory optimizations
MIT Licensed - Compatible with ONNX export
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


def knn(x, k):
    """Get k nearest neighbors index"""
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None):
    """Construct edge features for each point"""
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
    """Edge convolution layer"""
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


class DGCNNBackbone(nn.Module):
    """DGCNN backbone for point cloud feature extraction with memory optimization"""
    def __init__(self, cfg):
        super().__init__()
        
        self.k = 20  # number of nearest neighbors
        input_dim = cfg.INPUT_DIM
        output_channels = cfg.CHANNELS
        
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
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
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
        
        # Process each point cloud with memory optimization
        all_features = []
        all_coords = []
        all_masks = []
        all_sem_logits = []
        
        for b in range(batch_size):
            coords = torch.from_numpy(coords_list[b]).float().cuda()
            feats = torch.from_numpy(feats_list[b]).float().cuda()
            
            # Subsample if too many points during validation
            if not self.training and coords.shape[0] > 30000:
                indices = torch.randperm(coords.shape[0])[:30000]
                coords = coords[indices]
                feats = feats[indices]
            
            # Combine coordinates and features
            x_in = torch.cat([coords, feats[:, 3:]], dim=1).transpose(0, 1).unsqueeze(0)
            
            # DGCNN forward with memory optimization for validation
            if not self.training:
                with torch.no_grad():
                    x1 = self.conv1(x_in)
                    x2 = self.conv2(x1)
                    x3 = self.conv3(x2)
                    x4 = self.conv4(x3)
                    x = torch.cat((x1, x2, x3, x4), dim=1)
                    x = self.conv5(x)
            else:
                x1 = self.conv1(x_in)
                x2 = self.conv2(x1)
                x3 = self.conv3(x2)
                x4 = self.conv4(x3)
                x = torch.cat((x1, x2, x3, x4), dim=1)
                x = self.conv5(x)
            
            all_features.append(x)
            all_coords.append(coords)
            all_masks.append(torch.zeros(coords.shape[0], dtype=torch.bool).cuda())
            
            # Clear intermediate variables to save memory
            if not self.training:
                del x1, x2, x3, x4, x_in
        
        # Clear cache periodically during validation
        if not self.training and batch_size > 1:
            torch.cuda.empty_cache()
        
        # Generate multi-scale features
        ms_features = []
        ms_coords = []
        ms_masks = []
        
        for i, (feat_layer, bn_layer) in enumerate(zip(self.feat_layers, self.out_bn)):
            level_features = []
            level_coords = []
            level_masks = []
            
            for b in range(batch_size):
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
        
        # Generate semantic logits
        sem_logits = []
        for b in range(batch_size):
            # Use the last feature level for semantic prediction
            feat = self.feat_layers[-1](all_features[b]).squeeze(0).transpose(0, 1)
            feat = self.out_bn[-1](feat.transpose(0, 1)).transpose(0, 1)
            sem_logit = self.sem_head(feat)
            sem_logits.append(sem_logit)
        
        # Pad semantic logits
        max_points = max(logit.shape[0] for logit in sem_logits)
        padded_sem_logits = []
        
        for sem_logit in sem_logits:
            n_points = sem_logit.shape[0]
            if n_points < max_points:
                pad_size = max_points - n_points
                sem_logit = F.pad(sem_logit, (0, 0, 0, pad_size))
            padded_sem_logits.append(sem_logit)
        
        sem_logits = torch.stack(padded_sem_logits)
        
        # Clear features list if in validation mode
        if not self.training:
            del all_features
            torch.cuda.empty_cache()
        
        return ms_features, ms_coords, ms_masks, sem_logits


class DGCNNPretrainedBackbone(DGCNNBackbone):
    """DGCNN backbone with pre-trained weights support"""
    def __init__(self, cfg, pretrained_path=None):
        super().__init__(cfg)
        
        if pretrained_path:
            self.load_pretrained(pretrained_path)
    
    def load_pretrained(self, path):
        """Load pre-trained weights from classification/segmentation tasks"""
        print(f"Loading pre-trained weights from {path}")
        try:
            checkpoint = torch.load(path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Map pretrained weights to our architecture
            model_dict = self.state_dict()
            
            # Filter and map pretrained weights
            pretrained_dict = {}
            for key, value in state_dict.items():
                # Try direct mapping first
                if key in model_dict and value.shape == model_dict[key].shape:
                    pretrained_dict[key] = value
            
            # Update current model
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)
            
            print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers from pre-trained model")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights from {path}: {e}")
            print("Continuing with random initialization...")