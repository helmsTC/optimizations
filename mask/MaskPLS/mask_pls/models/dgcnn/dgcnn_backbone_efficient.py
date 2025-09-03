# mask_pls/models/dgcnn/dgcnn_backbone_efficient_fixed.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
import os

def knn(x, k):
    """Get k nearest neighbors index"""
    batch_size = x.size(0)
    num_points = x.size(2)
    
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
        # Match DGCNN naming convention for weight loading
        self.conv1 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv = nn.Sequential(
            self.conv1,
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def forward(self, x, idx=None):
        x = get_graph_feature(x, k=self.k, idx=idx)
        x = self.conv(x)
        x = x.max(dim=-1, keepdim=False)[0]
        return x

class FixedDGCNNBackbone(nn.Module):
    """Fixed DGCNN backbone with proper pretrained weight loading"""
    def __init__(self, cfg, pretrained_path=None):
        super().__init__()
        
        self.k = 20  # Keep original k=20 for better features
        input_dim = cfg.INPUT_DIM
        output_channels = cfg.CHANNELS
        self.num_classes = 20  # Will be overridden by dataset config
        
        # Match DGCNN architecture naming for pretrained weights
        # EdgeConv layers with proper naming
        self.edge_conv1 = nn.Sequential(
            nn.Conv2d(input_dim * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.edge_conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.edge_conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.edge_conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # Aggregation layers matching DGCNN
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # For compatibility with pretrained models that have these layers
        self.emb = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # Multi-scale feature extraction for MaskPLS
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
        
        # Semantic head - will be reconfigured
        self.sem_head = nn.Linear(output_channels[8], self.num_classes)
        
        # Track subsampling
        self.subsample_indices = {}
        
        # Initialize weights
        self.init_weights()
        
        # Load pretrained if available
        if pretrained_path and os.path.exists(pretrained_path):
            self.load_pretrained_dgcnn(pretrained_path)
    
    def set_num_classes(self, num_classes):
        """Update number of classes for semantic head"""
        self.num_classes = num_classes
        self.sem_head = nn.Linear(self.sem_head.in_features, num_classes)
    
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
    
    def load_pretrained_dgcnn(self, path):
        """Load pretrained DGCNN weights with proper mapping"""
        print(f"Loading pretrained DGCNN weights from {path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                pretrained_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                pretrained_dict = checkpoint['model_state_dict']
            else:
                pretrained_dict = checkpoint
            
            # Get current model state
            model_dict = self.state_dict()
            
            # Create mapping for DGCNN weights to our model
            mapped_dict = {}
            
            # Common DGCNN weight mappings
            weight_mappings = {
                # Edge convolution mappings
                'conv1.0.weight': 'edge_conv1.0.weight',
                'conv1.0.bias': 'edge_conv1.0.bias',
                'conv1.1.weight': 'edge_conv1.1.weight',
                'conv1.1.bias': 'edge_conv1.1.bias',
                'conv1.1.running_mean': 'edge_conv1.1.running_mean',
                'conv1.1.running_var': 'edge_conv1.1.running_var',
                
                'conv2.0.weight': 'edge_conv2.0.weight',
                'conv2.0.bias': 'edge_conv2.0.bias',
                'conv2.1.weight': 'edge_conv2.1.weight',
                'conv2.1.bias': 'edge_conv2.1.bias',
                'conv2.1.running_mean': 'edge_conv2.1.running_mean',
                'conv2.1.running_var': 'edge_conv2.1.running_var',
                
                'conv3.0.weight': 'edge_conv3.0.weight',
                'conv3.0.bias': 'edge_conv3.0.bias',
                'conv3.1.weight': 'edge_conv3.1.weight',
                'conv3.1.bias': 'edge_conv3.1.bias',
                'conv3.1.running_mean': 'edge_conv3.1.running_mean',
                'conv3.1.running_var': 'edge_conv3.1.running_var',
                
                'conv4.0.weight': 'edge_conv4.0.weight',
                'conv4.0.bias': 'edge_conv4.0.bias',
                'conv4.1.weight': 'edge_conv4.1.weight',
                'conv4.1.bias': 'edge_conv4.1.bias',
                'conv4.1.running_mean': 'edge_conv4.1.running_mean',
                'conv4.1.running_var': 'edge_conv4.1.running_var',
                
                # Conv5 mappings
                'conv5.0.weight': 'conv5.0.weight',
                'conv5.0.bias': 'conv5.0.bias',
                'conv5.1.weight': 'conv5.1.weight',
                'conv5.1.bias': 'conv5.1.bias',
                'conv5.1.running_mean': 'conv5.1.running_mean',
                'conv5.1.running_var': 'conv5.1.running_var',
                
                # Embedding layers (if present in pretrained)
                'emb.0.weight': 'emb.0.weight',
                'emb.0.bias': 'emb.0.bias',
                'emb.1.weight': 'emb.1.weight',
                'emb.1.bias': 'emb.1.bias',
                'emb.1.running_mean': 'emb.1.running_mean',
                'emb.1.running_var': 'emb.1.running_var',
            }
            
            # Also try direct mappings for edge_conv layers
            for key in pretrained_dict.keys():
                # Handle edge_conv naming
                if key.startswith('edge_conv'):
                    if key in model_dict and pretrained_dict[key].shape == model_dict[key].shape:
                        mapped_dict[key] = pretrained_dict[key]
                
                # Handle conv layer mappings
                elif key in weight_mappings:
                    target_key = weight_mappings[key]
                    if target_key in model_dict:
                        if pretrained_dict[key].shape == model_dict[target_key].shape:
                            mapped_dict[target_key] = pretrained_dict[key]
                        else:
                            print(f"  Shape mismatch: {key} -> {target_key}: {pretrained_dict[key].shape} vs {model_dict[target_key].shape}")
                
                # Try direct mapping
                elif key in model_dict:
                    if pretrained_dict[key].shape == model_dict[key].shape:
                        mapped_dict[key] = pretrained_dict[key]
            
            # Special handling for models with different input dimensions
            # If the first conv layer has different input channels, skip it
            if 'edge_conv1.0.weight' in model_dict:
                model_in_channels = model_dict['edge_conv1.0.weight'].shape[1]
                if 'conv1.0.weight' in pretrained_dict:
                    pretrained_in_channels = pretrained_dict['conv1.0.weight'].shape[1]
                    if model_in_channels != pretrained_in_channels:
                        print(f"  Input dimension mismatch ({pretrained_in_channels} vs {model_in_channels}), skipping first conv layer")
                        # Remove first conv from mapped_dict if it was added
                        keys_to_remove = ['edge_conv1.0.weight', 'edge_conv1.0.bias']
                        for key in keys_to_remove:
                            if key in mapped_dict:
                                del mapped_dict[key]
            
            # Update model with mapped weights
            model_dict.update(mapped_dict)
            self.load_state_dict(model_dict, strict=False)
            
            print(f"Successfully loaded {len(mapped_dict)}/{len(model_dict)} layers from pretrained model")
            
            # Print which layers were loaded
            if len(mapped_dict) > 0:
                print("Loaded layers:")
                loaded_prefixes = set()
                for key in mapped_dict.keys():
                    prefix = key.split('.')[0]
                    loaded_prefixes.add(prefix)
                for prefix in sorted(loaded_prefixes):
                    count = sum(1 for k in mapped_dict.keys() if k.startswith(prefix))
                    print(f"  {prefix}: {count} parameters")
            
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            print("Continuing with randomly initialized weights")
    
    def forward(self, x):
        """Forward pass with edge convolutions"""
        coords_list = x['pt_coord']
        feats_list = x['feats']
        
        batch_size = len(coords_list)
        
        # Clear subsample tracking
        self.subsample_indices = {}
        
        # Process each point cloud
        all_features = []
        all_coords = []
        all_masks = []
        
        for b in range(batch_size):
            coords = torch.from_numpy(coords_list[b]).float().cuda()
            feats = torch.from_numpy(feats_list[b]).float().cuda()
            
            original_size = coords.shape[0]
            
            # Subsample if needed
            max_points = 50000 if self.training else 30000
            if coords.shape[0] > max_points:
                indices = torch.randperm(coords.shape[0], device=coords.device)[:max_points]
                indices = indices.sort()[0]
                coords = coords[indices]
                feats = feats[indices]
                self.subsample_indices[b] = indices
            else:
                self.subsample_indices[b] = torch.arange(coords.shape[0], device=coords.device)
            
            # Process through DGCNN
            point_features = self.process_single_cloud(coords, feats)
            
            all_features.append(point_features)
            all_coords.append(coords)
            all_masks.append(torch.zeros(coords.shape[0], dtype=torch.bool, device=coords.device))
        
        # Generate multi-scale features with padding
        ms_features = []
        ms_coords = []
        ms_masks = []
        
        for i in range(len(self.feat_layers)):
            level_features, level_coords, level_masks = self.pad_batch_level(
                [f[i] for f in all_features],
                all_coords,
                all_masks
            )
            ms_features.append(level_features)
            ms_coords.append(level_coords)
            ms_masks.append(level_masks)
        
        # Semantic predictions
        sem_logits = self.compute_semantic_logits(ms_features[-1], ms_masks[-1])
        
        return ms_features, ms_coords, ms_masks, sem_logits
    
    def process_single_cloud(self, coords, feats):
        """Process a single point cloud through DGCNN"""
        # Combine coordinates and intensity
        x = torch.cat([coords, feats[:, 3:4]], dim=1).transpose(0, 1).unsqueeze(0)
        
        # Edge convolutions
        x1 = get_graph_feature(x, k=self.k)
        x1 = self.edge_conv1(x1)
        x1 = x1.max(dim=-1, keepdim=False)[0]
        
        x2 = get_graph_feature(x1, k=self.k)
        x2 = self.edge_conv2(x2)
        x2 = x2.max(dim=-1, keepdim=False)[0]
        
        x3 = get_graph_feature(x2, k=self.k)
        x3 = self.edge_conv3(x3)
        x3 = x3.max(dim=-1, keepdim=False)[0]
        
        x4 = get_graph_feature(x3, k=self.k)
        x4 = self.edge_conv4(x4)
        x4 = x4.max(dim=-1, keepdim=False)[0]
        
        # Aggregate features
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        
        # Generate multi-scale features
        features = []
        for feat_layer, bn_layer in zip(self.feat_layers, self.out_bn):
            feat = feat_layer(x)
            feat = bn_layer(feat)
            feat = feat.squeeze(0).transpose(0, 1)
            features.append(feat)
        
        return features
    
    def pad_batch_level(self, features, coords, masks):
        """Pad features, coordinates and masks to same size"""
        max_points = max(f.shape[0] for f in features)
        
        padded_features = []
        padded_coords = []
        padded_masks = []
        
        for feat, coord, mask in zip(features, coords, masks):
            n_points = feat.shape[0]
            if n_points < max_points:
                pad_size = max_points - n_points
                feat = F.pad(feat, (0, 0, 0, pad_size))
                coord = F.pad(coord, (0, 0, 0, pad_size))
                mask = F.pad(mask, (0, pad_size), value=True)
            
            padded_features.append(feat)
            padded_coords.append(coord)
            padded_masks.append(mask)
        
        return (torch.stack(padded_features), 
                torch.stack(padded_coords), 
                torch.stack(padded_masks))
    
    def compute_semantic_logits(self, features, masks):
        """Compute semantic logits for valid points"""
        batch_size = features.shape[0]
        sem_logits = []
        
        for b in range(batch_size):
            valid_mask = ~masks[b]
            if valid_mask.sum() > 0:
                valid_features = features[b][valid_mask]
                logits = self.sem_head(valid_features)
            else:
                logits = torch.zeros(0, self.num_classes, device=features.device)
            
            # Pad back to full size
            full_logits = torch.zeros(features.shape[1], self.num_classes, 
                                     device=features.device)
            if valid_mask.sum() > 0:
                full_logits[valid_mask] = logits
            
            sem_logits.append(full_logits)
        
        return torch.stack(sem_logits)