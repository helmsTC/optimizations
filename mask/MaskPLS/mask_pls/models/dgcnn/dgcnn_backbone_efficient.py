# mask_pls/models/dgcnn/dgcnn_backbone_efficient.py
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
        """Load pretrained weights with comprehensive mapping strategies"""
        print(f"\n{'='*60}")
        print(f"Loading pretrained DGCNN weights from {path}")
        print(f"{'='*60}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(path, map_location='cpu')
            
            # Extract state dict (handle various formats)
            if isinstance(checkpoint, dict):
                # Try different possible keys
                possible_keys = ['state_dict', 'model_state_dict', 'model', 'net']
                pretrained_dict = None
                for key in possible_keys:
                    if key in checkpoint:
                        pretrained_dict = checkpoint[key]
                        print(f"Found state dict under key: '{key}'")
                        break
                
                if pretrained_dict is None:
                    # Assume the checkpoint itself is the state dict
                    pretrained_dict = checkpoint
            else:
                pretrained_dict = checkpoint
            
            # Get current model state
            model_dict = self.state_dict()
            
            # Analyze pretrained model
            print(f"\nPretrained model has {len(pretrained_dict)} parameters")
            print(f"Current model has {len(model_dict)} parameters")
            
            # Show first few pretrained layer names to understand naming pattern
            print("\nFirst 10 pretrained layer names:")
            for i, (name, param) in enumerate(pretrained_dict.items()):
                if i >= 10:
                    break
                print(f"  {name:50} {str(param.shape):20}")
            
            # Create mapping based on common DGCNN architectures
            mapped_dict = {}
            
            # Strategy 1: Try exact name matches first
            print("\n[Strategy 1] Checking for exact name matches...")
            exact_matches = 0
            for key, value in pretrained_dict.items():
                if key in model_dict:
                    if value.shape == model_dict[key].shape:
                        mapped_dict[key] = value
                        exact_matches += 1
            print(f"  Found {exact_matches} exact matches")
            
            # Strategy 2: Common DGCNN patterns from classification/segmentation models
            print("\n[Strategy 2] Trying common DGCNN patterns...")
            
            # Pattern mappings for different DGCNN variants
            mapping_patterns = [
                # Pattern 1: Standard DGCNN (classification)
                {
                    'conv1.weight': 'edge_conv1.0.weight',
                    'conv1.bias': 'edge_conv1.0.bias',
                    'bn1.weight': 'edge_conv1.1.weight',
                    'bn1.bias': 'edge_conv1.1.bias',
                    'bn1.running_mean': 'edge_conv1.1.running_mean',
                    'bn1.running_var': 'edge_conv1.1.running_var',
                    'bn1.num_batches_tracked': 'edge_conv1.1.num_batches_tracked',
                    
                    'conv2.weight': 'edge_conv2.0.weight',
                    'conv2.bias': 'edge_conv2.0.bias',
                    'bn2.weight': 'edge_conv2.1.weight',
                    'bn2.bias': 'edge_conv2.1.bias',
                    'bn2.running_mean': 'edge_conv2.1.running_mean',
                    'bn2.running_var': 'edge_conv2.1.running_var',
                    
                    'conv3.weight': 'edge_conv3.0.weight',
                    'conv3.bias': 'edge_conv3.0.bias',
                    'bn3.weight': 'edge_conv3.1.weight',
                    'bn3.bias': 'edge_conv3.1.bias',
                    'bn3.running_mean': 'edge_conv3.1.running_mean',
                    'bn3.running_var': 'edge_conv3.1.running_var',
                    
                    'conv4.weight': 'edge_conv4.0.weight',
                    'conv4.bias': 'edge_conv4.0.bias',
                    'bn4.weight': 'edge_conv4.1.weight',
                    'bn4.bias': 'edge_conv4.1.bias',
                    'bn4.running_mean': 'edge_conv4.1.running_mean',
                    'bn4.running_var': 'edge_conv4.1.running_var',
                    
                    'conv5.weight': 'conv5.0.weight',
                    'conv5.bias': 'conv5.0.bias',
                    'bn5.weight': 'conv5.1.weight',
                    'bn5.bias': 'conv5.1.bias',
                    'bn5.running_mean': 'conv5.1.running_mean',
                    'bn5.running_var': 'conv5.1.running_var',
                },
                # Pattern 2: DGCNN with T-Net (transformation network)
                {
                    'transform_net.conv1.weight': None,  # Skip T-Net
                    'transform_net.bn1.weight': None,
                    'edge_conv1.weight': 'edge_conv1.0.weight',
                    'edge_bn1.weight': 'edge_conv1.1.weight',
                    'edge_conv2.weight': 'edge_conv2.0.weight',
                    'edge_bn2.weight': 'edge_conv2.1.weight',
                    'edge_conv3.weight': 'edge_conv3.0.weight',
                    'edge_bn3.weight': 'edge_conv3.1.weight',
                    'edge_conv4.weight': 'edge_conv4.0.weight',
                    'edge_bn4.weight': 'edge_conv4.1.weight',
                },
                # Pattern 3: DGCNN part segmentation style
                {
                    'transform.conv1.weight': None,  # Skip transform
                    'conv1.conv.weight': 'edge_conv1.0.weight',
                    'conv1.bn.weight': 'edge_conv1.1.weight',
                    'conv2.conv.weight': 'edge_conv2.0.weight',
                    'conv2.bn.weight': 'edge_conv2.1.weight',
                    'conv3.conv.weight': 'edge_conv3.0.weight',
                    'conv3.bn.weight': 'edge_conv3.1.weight',
                    'conv4.conv.weight': 'edge_conv4.0.weight',
                    'conv4.bn.weight': 'edge_conv4.1.weight',
                    'conv5.conv.weight': 'conv5.0.weight',
                    'conv5.bn.weight': 'conv5.1.weight',
                },
            ]
            
            # Try each pattern
            for pattern_idx, pattern in enumerate(mapping_patterns, 1):
                print(f"\n  Pattern {pattern_idx}: ", end='')
                pattern_matches = 0
                
                for src_key, dst_key in pattern.items():
                    if dst_key is None:  # Skip this layer
                        continue
                    
                    # Try to find matching keys with this pattern
                    for pretrained_key in pretrained_dict.keys():
                        if src_key in pretrained_key or pretrained_key == src_key:
                            if dst_key in model_dict and dst_key not in mapped_dict:
                                src_shape = pretrained_dict[pretrained_key].shape
                                dst_shape = model_dict[dst_key].shape
                                
                                # Handle dimension mismatch for first conv layer
                                if 'edge_conv1.0.weight' in dst_key and src_shape != dst_shape:
                                    if self._handle_conv1_mismatch(src_shape, dst_shape):
                                        adapted_weight = self._adapt_conv1_weight(
                                            pretrained_dict[pretrained_key], 
                                            dst_shape
                                        )
                                        mapped_dict[dst_key] = adapted_weight
                                        pattern_matches += 1
                                        continue
                                
                                # Direct shape match
                                if src_shape == dst_shape:
                                    mapped_dict[dst_key] = pretrained_dict[pretrained_key]
                                    pattern_matches += 1
                
                print(f"matched {pattern_matches} parameters")
            
            # Strategy 3: Fuzzy matching based on layer shapes
            print("\n[Strategy 3] Fuzzy matching by shape...")
            fuzzy_matches = self._fuzzy_match_by_shape(pretrained_dict, model_dict, mapped_dict)
            print(f"  Found {fuzzy_matches} additional matches by shape")
            
            # Load the mapped weights
            if len(mapped_dict) > 0:
                # Update model
                model_dict.update(mapped_dict)
                self.load_state_dict(model_dict, strict=False)
                
                print(f"\n{'='*60}")
                print(f"✓ Successfully loaded {len(mapped_dict)}/{len(model_dict)} parameters")
                print(f"{'='*60}")
                
                # Detailed summary
                self._print_loading_summary(mapped_dict)
                
                # Check critical layers
                critical_layers = ['edge_conv1', 'edge_conv2', 'edge_conv3', 'edge_conv4', 'conv5']
                print("\nCritical layer status:")
                for layer in critical_layers:
                    layer_params = sum(1 for k in mapped_dict.keys() if k.startswith(layer))
                    if layer_params > 0:
                        print(f"  ✓ {layer}: {layer_params} parameters loaded")
                    else:
                        print(f"  ✗ {layer}: NOT loaded (will train from scratch)")
            else:
                print(f"\n{'='*60}")
                print("✗ No compatible weights found")
                print("  The model will train from scratch")
                print(f"{'='*60}")
                print("\nTroubleshooting tips:")
                print("  1. Check if the pretrained model is for ModelNet40 (classification)")
                print("  2. Try a segmentation-specific DGCNN checkpoint")
                print("  3. Verify the checkpoint file is not corrupted")
                print("  4. Consider training from scratch - you're already getting decent results!")
                
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"✗ Error loading pretrained weights: {e}")
            print("  Continuing with random initialization...")
            print(f"{'='*60}")
            import traceback
            traceback.print_exc()
    
    def _handle_conv1_mismatch(self, src_shape, dst_shape):
        """Check if conv1 dimension mismatch can be handled"""
        # Check if it's a 3D->4D adaptation scenario
        if len(src_shape) == 4 and len(dst_shape) == 4:
            if src_shape[0] == dst_shape[0]:  # Same output channels
                if src_shape[1] == 6 and dst_shape[1] == 8:  # 3D to 4D input
                    return True
                elif src_shape[1] == 3 and dst_shape[1] == 8:  # Direct 3D coords to 4D
                    return True
        return False
    
    def _adapt_conv1_weight(self, src_weight, dst_shape):
        """Adapt conv1 weight from 3D to 4D input"""
        adapted_weight = torch.zeros(dst_shape)
        src_channels = src_weight.shape[1]
        
        if src_channels == 6:  # Edge features for 3D
            # Copy the 6 channels and initialize the rest
            adapted_weight[:, :6, :, :] = src_weight
            adapted_weight[:, 6:, :, :].normal_(0, 0.01)
        elif src_channels == 3:  # Direct 3D coordinates
            # Duplicate for edge features and add intensity channels
            adapted_weight[:, :3, :, :] = src_weight
            adapted_weight[:, 3:6, :, :] = src_weight  # Duplicate for edge features
            adapted_weight[:, 6:, :, :].normal_(0, 0.01)  # Initialize intensity channels
        
        print(f"    Adapted conv1 from {src_weight.shape} to {dst_shape}")
        return adapted_weight
    
    def _fuzzy_match_by_shape(self, pretrained_dict, model_dict, mapped_dict):
        """Try to match layers by shape when names don't match"""
        matches = 0
        
        # Group model layers by shape
        shape_to_model_keys = {}
        for key, param in model_dict.items():
            shape = tuple(param.shape)
            if shape not in shape_to_model_keys:
                shape_to_model_keys[shape] = []
            shape_to_model_keys[shape].append(key)
        
        # Try to match pretrained layers by shape
        for p_key, p_param in pretrained_dict.items():
            shape = tuple(p_param.shape)
            
            # Skip if already mapped
            if any(p_key in k for k in mapped_dict.keys()):
                continue
            
            # Look for matching shape in model
            if shape in shape_to_model_keys:
                candidates = shape_to_model_keys[shape]
                
                # Try to find a good match based on layer type
                for m_key in candidates:
                    if m_key not in mapped_dict:
                        # Match conv to conv, bn to bn, etc.
                        if self._layer_types_match(p_key, m_key):
                            mapped_dict[m_key] = p_param
                            matches += 1
                            break
        
        return matches
    
    def _layer_types_match(self, key1, key2):
        """Check if two layer keys are of compatible types"""
        # Extract layer type indicators
        type_indicators = {
            'conv': ['conv', 'Conv'],
            'bn': ['bn', 'batch_norm', 'BatchNorm'],
            'linear': ['linear', 'fc', 'classifier'],
        }
        
        for layer_type, indicators in type_indicators.items():
            key1_is_type = any(ind in key1 for ind in indicators)
            key2_is_type = any(ind in key2 for ind in indicators)
            if key1_is_type and key2_is_type:
                return True
        
        return False
    
    def _print_loading_summary(self, mapped_dict):
        """Print a detailed summary of loaded layers"""
        # Group by layer prefix
        layer_groups = {}
        for key in mapped_dict.keys():
            prefix = key.split('.')[0]
            if prefix not in layer_groups:
                layer_groups[prefix] = []
            layer_groups[prefix].append(key)
        
        print("\nLoaded parameters by layer group:")
        for prefix in sorted(layer_groups.keys()):
            params = layer_groups[prefix]
            # Count different parameter types
            weights = sum(1 for p in params if 'weight' in p)
            biases = sum(1 for p in params if 'bias' in p)
            others = len(params) - weights - biases
            
            print(f"  {prefix:15} - Total: {len(params):3} (weights: {weights}, biases: {biases}, other: {others})")
    
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