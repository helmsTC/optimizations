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

class FixedDGCNNBackbone(nn.Module):
    """Fixed DGCNN backbone with improved pretrained weight loading"""
    def __init__(self, cfg, pretrained_path=None):
        super().__init__()
        
        self.k = 20  # Keep original k=20 for better features
        input_dim = cfg.INPUT_DIM
        output_channels = cfg.CHANNELS
        self.num_classes = 20  # Will be overridden by dataset config
        
        # EdgeConv layers
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
        
        # Aggregation layers
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
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
        
        # Semantic head
        self.sem_head = nn.Linear(output_channels[8], self.num_classes)
        
        # Track subsampling
        self.subsample_indices = {}
        
        # Initialize weights
        self.init_weights()
        
        # Load pretrained if available
        if pretrained_path and os.path.exists(pretrained_path):
            self.load_pretrained_partial(pretrained_path)
    
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
    
    def load_pretrained_partial(self, path):
        """Load pretrained weights with partial loading strategy"""
        print(f"\n{'='*60}")
        print(f"Loading pretrained DGCNN weights from {path}")
        print(f"{'='*60}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(path, map_location='cpu')
            
            # Extract state dict
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    pretrained_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    pretrained_dict = checkpoint['model_state_dict']
                elif 'model' in checkpoint:
                    pretrained_dict = checkpoint['model']
                else:
                    pretrained_dict = checkpoint
            else:
                pretrained_dict = checkpoint
            
            # Get current model state
            model_dict = self.state_dict()
            
            # Analyze pretrained model structure
            print("\nAnalyzing pretrained model structure...")
            pretrained_layers = self._analyze_checkpoint(pretrained_dict)
            
            # Strategy 1: Try direct mapping for feature extraction layers only
            mapped_dict = {}
            
            # Define mapping strategies
            mapping_strategies = [
                # Strategy 1: Direct DGCNN naming
                {
                    'conv1': 'edge_conv1.0',
                    'bn1': 'edge_conv1.1',
                    'conv2': 'edge_conv2.0',
                    'bn2': 'edge_conv2.1',
                    'conv3': 'edge_conv3.0',
                    'bn3': 'edge_conv3.1',
                    'conv4': 'edge_conv4.0',
                    'bn4': 'edge_conv4.1',
                },
                # Strategy 2: EdgeConv naming
                {
                    'edge_conv1.conv.0': 'edge_conv1.0',
                    'edge_conv1.conv.1': 'edge_conv1.1',
                    'edge_conv2.conv.0': 'edge_conv2.0',
                    'edge_conv2.conv.1': 'edge_conv2.1',
                    'edge_conv3.conv.0': 'edge_conv3.0',
                    'edge_conv3.conv.1': 'edge_conv3.1',
                    'edge_conv4.conv.0': 'edge_conv4.0',
                    'edge_conv4.conv.1': 'edge_conv4.1',
                }
            ]
            
            # Try each mapping strategy
            for strategy_idx, mapping in enumerate(mapping_strategies, 1):
                print(f"\nTrying mapping strategy {strategy_idx}...")
                strategy_mapped = self._apply_mapping_strategy(
                    pretrained_dict, model_dict, mapping
                )
                mapped_dict.update(strategy_mapped)
                
                if len(strategy_mapped) > 0:
                    print(f"  Strategy {strategy_idx} mapped {len(strategy_mapped)} parameters")
            
            # Handle dimension mismatches for first conv layer
            if len(mapped_dict) > 0:
                mapped_dict = self._handle_dimension_mismatch(
                    pretrained_dict, model_dict, mapped_dict
                )
            
            # Load aggregation layer (conv5) if possible - this is usually safe
            conv5_mapped = self._load_conv5(pretrained_dict, model_dict)
            mapped_dict.update(conv5_mapped)
            
            # Apply the mapped weights
            if len(mapped_dict) > 0:
                # Only load weights that exist and match
                filtered_dict = {}
                for key, value in mapped_dict.items():
                    if key in model_dict:
                        if value.shape == model_dict[key].shape:
                            filtered_dict[key] = value
                        else:
                            print(f"  Skipping {key}: shape mismatch {value.shape} vs {model_dict[key].shape}")
                
                if len(filtered_dict) > 0:
                    model_dict.update(filtered_dict)
                    self.load_state_dict(model_dict, strict=False)
                    
                    print(f"\n✓ Successfully loaded {len(filtered_dict)}/{len(model_dict)} parameters")
                    
                    # Summary of loaded layers
                    self._print_loading_summary(filtered_dict)
                    
                    # Print which important layers were loaded
                    important_layers = ['edge_conv1', 'edge_conv2', 'edge_conv3', 'edge_conv4', 'conv5']
                    print("\nStatus of important layers:")
                    for layer in important_layers:
                        loaded = any(key.startswith(layer) for key in filtered_dict.keys())
                        status = "✓ Loaded" if loaded else "✗ Not loaded"
                        print(f"  {layer}: {status}")
                else:
                    print("\n✗ No compatible weights found after filtering")
            else:
                print("\n✗ No weights could be mapped from pretrained model")
                print("  Continuing with random initialization...")
                
        except Exception as e:
            print(f"\n✗ Error loading pretrained weights: {e}")
            print("  Continuing with random initialization...")
            import traceback
            traceback.print_exc()
    
    def _analyze_checkpoint(self, state_dict):
        """Analyze the structure of the checkpoint"""
        layers = {
            'conv': [],
            'bn': [],
            'linear': [],
            'other': []
        }
        
        for key in state_dict.keys():
            if 'conv' in key.lower() and 'weight' in key:
                layers['conv'].append(key)
            elif ('bn' in key.lower() or 'norm' in key.lower()) and 'weight' in key:
                layers['bn'].append(key)
            elif ('linear' in key.lower() or 'fc' in key.lower()) and 'weight' in key:
                layers['linear'].append(key)
            else:
                layers['other'].append(key)
        
        print(f"  Found {len(layers['conv'])} conv layers")
        print(f"  Found {len(layers['bn'])} batch norm layers")
        print(f"  Found {len(layers['linear'])} linear layers")
        
        return layers
    
    def _apply_mapping_strategy(self, pretrained_dict, model_dict, mapping):
        """Apply a specific mapping strategy"""
        mapped = {}
        
        for pretrained_prefix, model_prefix in mapping.items():
            # Map all parameters with this prefix
            for key in pretrained_dict.keys():
                if key.startswith(pretrained_prefix):
                    # Create new key
                    suffix = key[len(pretrained_prefix):]
                    new_key = model_prefix + suffix
                    
                    if new_key in model_dict:
                        if pretrained_dict[key].shape == model_dict[new_key].shape:
                            mapped[new_key] = pretrained_dict[key]
        
        return mapped
    
    def _handle_dimension_mismatch(self, pretrained_dict, model_dict, mapped_dict):
        """Handle dimension mismatches, especially for first conv layer"""
        first_conv_key = 'edge_conv1.0.weight'
        
        if first_conv_key not in mapped_dict and first_conv_key in model_dict:
            model_shape = model_dict[first_conv_key].shape
            print(f"\nHandling first conv layer dimension mismatch...")
            print(f"  Target shape: {model_shape}")
            
            # Look for conv1 in pretrained model
            candidates = ['conv1.weight', 'edge_conv1.conv.0.weight', 'conv1.0.weight']
            
            for candidate in candidates:
                if candidate in pretrained_dict:
                    pretrained_shape = pretrained_dict[candidate].shape
                    print(f"  Found {candidate} with shape {pretrained_shape}")
                    
                    # Check if we can adapt it
                    if pretrained_shape[0] == model_shape[0]:  # Same output channels
                        if pretrained_shape[1] == 6 and model_shape[1] == 8:
                            # 3D input (6 after edge features) -> 4D input (8 after edge features)
                            print(f"  Adapting from 3D to 4D input...")
                            adapted_weight = torch.zeros(model_shape)
                            # Initialize with small random values
                            adapted_weight.normal_(0, 0.01)
                            # Copy the existing 6 channels
                            adapted_weight[:, :6, :, :] = pretrained_dict[candidate]
                            # The last 2 channels for intensity will learn from scratch
                            mapped_dict[first_conv_key] = adapted_weight
                            print(f"  ✓ Adapted first conv layer")
                            break
                        elif pretrained_shape == model_shape:
                            # Direct match
                            mapped_dict[first_conv_key] = pretrained_dict[candidate]
                            print(f"  ✓ Loaded first conv layer directly")
                            break
        
        return mapped_dict
    
    def _load_conv5(self, pretrained_dict, model_dict):
        """Try to load conv5 aggregation layer"""
        mapped = {}
        
        # Conv5 mappings
        conv5_mappings = [
            ('conv5.weight', 'conv5.0.weight'),
            ('conv5.bias', 'conv5.0.bias'),
            ('bn5.weight', 'conv5.1.weight'),
            ('bn5.bias', 'conv5.1.bias'),
            ('bn5.running_mean', 'conv5.1.running_mean'),
            ('bn5.running_var', 'conv5.1.running_var'),
            ('conv5.0.weight', 'conv5.0.weight'),  # Direct mapping
            ('conv5.1.weight', 'conv5.1.weight'),
            ('conv5.1.bias', 'conv5.1.bias'),
            ('conv5.1.running_mean', 'conv5.1.running_mean'),
            ('conv5.1.running_var', 'conv5.1.running_var'),
        ]
        
        for old_key, new_key in conv5_mappings:
            if old_key in pretrained_dict and new_key in model_dict:
                if pretrained_dict[old_key].shape == model_dict[new_key].shape:
                    mapped[new_key] = pretrained_dict[old_key]
        
        return mapped
    
    def _print_loading_summary(self, loaded_dict):
        """Print a summary of what was loaded"""
        layer_counts = {}
        
        for key in loaded_dict.keys():
            layer = key.split('.')[0]
            if layer not in layer_counts:
                layer_counts[layer] = 0
            layer_counts[layer] += 1
        
        print("\nLoaded parameters by layer:")
        for layer, count in sorted(layer_counts.items()):
            print(f"  {layer}: {count} parameters")
    
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