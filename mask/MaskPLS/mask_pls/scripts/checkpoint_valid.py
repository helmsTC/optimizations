#!/usr/bin/env python
"""
Script to inspect and optionally convert DGCNN checkpoints for MaskPLS
Usage: python inspect_checkpoint.py --checkpoint path/to/dgcnn.pth
"""

import torch
import argparse
from collections import defaultdict, OrderedDict
import json

def analyze_checkpoint(checkpoint_path):
    """Comprehensive checkpoint analysis"""
    
    print(f"\n{'='*70}")
    print(f"CHECKPOINT ANALYSIS: {checkpoint_path}")
    print(f"{'='*70}\n")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Determine checkpoint type
    if isinstance(checkpoint, dict):
        print("✓ Checkpoint is a dictionary with keys:")
        for key in checkpoint.keys():
            if key != 'state_dict' and key != 'model_state_dict':
                if isinstance(checkpoint[key], (int, float, str, bool)):
                    print(f"  - {key}: {checkpoint[key]}")
                else:
                    print(f"  - {key}: {type(checkpoint[key])}")
        
        # Extract state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("\n✓ Using 'state_dict' from checkpoint")
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("\n✓ Using 'model_state_dict' from checkpoint")
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
            print("\n✓ Using 'model' from checkpoint")
        else:
            state_dict = checkpoint
            print("\n✓ Using entire checkpoint as state_dict")
    else:
        state_dict = checkpoint
        print("✓ Checkpoint is directly a state_dict")
    
    return state_dict

def analyze_dgcnn_structure(state_dict):
    """Analyze DGCNN model structure"""
    
    print(f"\n{'='*70}")
    print("MODEL STRUCTURE ANALYSIS")
    print(f"{'='*70}\n")
    
    print(f"Total parameters: {len(state_dict)}")
    
    # Group by layer type
    structure = {
        'edge_conv_layers': [],
        'conv_layers': [],
        'bn_layers': [],
        'linear_layers': [],
        'embedding_layers': [],
        'other': []
    }
    
    # Analyze each parameter
    for key in state_dict.keys():
        if 'edge_conv' in key:
            structure['edge_conv_layers'].append(key)
        elif 'conv' in key and 'weight' in key:
            structure['conv_layers'].append(key)
        elif ('bn' in key or 'batch_norm' in key) and 'weight' in key:
            structure['bn_layers'].append(key)
        elif ('linear' in key or 'fc' in key) and 'weight' in key:
            structure['linear_layers'].append(key)
        elif 'emb' in key:
            structure['embedding_layers'].append(key)
        else:
            structure['other'].append(key)
    
    # Print structure
    for category, layers in structure.items():
        if layers:
            print(f"\n{category.upper().replace('_', ' ')} ({len(layers)} parameters):")
            for layer in layers[:5]:  # Show first 5
                shape = state_dict[layer].shape
                print(f"  {layer}: {shape}")
            if len(layers) > 5:
                print(f"  ... and {len(layers) - 5} more")
    
    return structure

def check_compatibility(state_dict):
    """Check compatibility with MaskPLS-DGCNN"""
    
    print(f"\n{'='*70}")
    print("MASKPLS-DGCNN COMPATIBILITY CHECK")
    print(f"{'='*70}\n")
    
    compatibility = {
        'edge_conv1': False,
        'edge_conv2': False,
        'edge_conv3': False,
        'edge_conv4': False,
        'conv5': False,
        'input_dim': None,
        'can_load': False
    }
    
    # Check for standard DGCNN layers
    conv_patterns = {
        'conv1': 'edge_conv1',
        'conv2': 'edge_conv2',
        'conv3': 'edge_conv3',
        'conv4': 'edge_conv4',
        'conv5': 'conv5'
    }
    
    for pattern, target in conv_patterns.items():
        found = False
        for key in state_dict.keys():
            if pattern in key and 'weight' in key and '.' + pattern + '.' not in key:
                compatibility[target] = True
                found = True
                
                # Check input dimension for first conv
                if pattern == 'conv1':
                    shape = state_dict[key].shape
                    input_channels = shape[1]
                    compatibility['input_dim'] = input_channels // 2  # Edge features double the input
                    print(f"✓ Found {pattern} with input dimension {input_channels} (implies {input_channels//2}D input)")
                else:
                    print(f"✓ Found {pattern}")
                break
        
        if not found:
            print(f"✗ Missing {pattern}")
    
    # Determine if we can partially load
    essential_layers = ['edge_conv1', 'edge_conv2', 'edge_conv3', 'edge_conv4']
    compatibility['can_load'] = sum(compatibility[layer] for layer in essential_layers) >= 2
    
    print(f"\n{'Summary':^20}")
    print("-" * 20)
    
    if compatibility['can_load']:
        print("✓ Can partially load pretrained weights")
        print(f"  Input dimension: {compatibility['input_dim']}D")
        if compatibility['input_dim'] == 3:
            print("  Note: Will need to adapt for 4D input (x,y,z,intensity)")
    else:
        print("✗ Incompatible model structure")
        print("  Consider training from scratch")
    
    return compatibility

def create_mapping_dict(state_dict):
    """Create a mapping dictionary for weight transfer"""
    
    print(f"\n{'='*70}")
    print("CREATING WEIGHT MAPPING")
    print(f"{'='*70}\n")
    
    mapping = OrderedDict()
    
    # Standard DGCNN to MaskPLS-DGCNN mappings
    mappings = [
        # Conv layers
        ('conv1.weight', 'edge_conv1.0.weight'),
        ('bn1.weight', 'edge_conv1.1.weight'),
        ('bn1.bias', 'edge_conv1.1.bias'),
        ('bn1.running_mean', 'edge_conv1.1.running_mean'),
        ('bn1.running_var', 'edge_conv1.1.running_var'),
        
        ('conv2.weight', 'edge_conv2.0.weight'),
        ('bn2.weight', 'edge_conv2.1.weight'),
        ('bn2.bias', 'edge_conv2.1.bias'),
        ('bn2.running_mean', 'edge_conv2.1.running_mean'),
        ('bn2.running_var', 'edge_conv2.1.running_var'),
        
        ('conv3.weight', 'edge_conv3.0.weight'),
        ('bn3.weight', 'edge_conv3.1.weight'),
        ('bn3.bias', 'edge_conv3.1.bias'),
        ('bn3.running_mean', 'edge_conv3.1.running_mean'),
        ('bn3.running_var', 'edge_conv3.1.running_var'),
        
        ('conv4.weight', 'edge_conv4.0.weight'),
        ('bn4.weight', 'edge_conv4.1.weight'),
        ('bn4.bias', 'edge_conv4.1.bias'),
        ('bn4.running_mean', 'edge_conv4.1.running_mean'),
        ('bn4.running_var', 'edge_conv4.1.running_var'),
        
        ('conv5.weight', 'conv5.0.weight'),
        ('bn5.weight', 'conv5.1.weight'),
        ('bn5.bias', 'conv5.1.bias'),
        ('bn5.running_mean', 'conv5.1.running_mean'),
        ('bn5.running_var', 'conv5.1.running_var'),
    ]
    
    found_mappings = []
    for src, dst in mappings:
        if src in state_dict:
            mapping[src] = dst
            found_mappings.append((src, dst))
    
    if found_mappings:
        print("Found mappable layers:")
        for src, dst in found_mappings:
            shape = state_dict[src].shape
            print(f"  {src:30} -> {dst:30} {shape}")
    else:
        print("✗ No directly mappable layers found")
        print("  This model may use different naming conventions")
    
    return mapping

def save_converted_checkpoint(state_dict, mapping, output_path):
    """Save a converted checkpoint with proper naming"""
    
    converted = OrderedDict()
    
    for old_key, new_key in mapping.items():
        if old_key in state_dict:
            converted[new_key] = state_dict[old_key]
    
    # Save as a proper checkpoint
    checkpoint = {
        'state_dict': converted,
        'original_keys': list(state_dict.keys()),
        'converted_keys': list(converted.keys()),
        'mapping': mapping
    }
    
    torch.save(checkpoint, output_path)
    print(f"\n✓ Saved converted checkpoint to: {output_path}")
    print(f"  Converted {len(converted)} parameters")

def main():
    parser = argparse.ArgumentParser(description='Inspect and convert DGCNN checkpoints')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to DGCNN checkpoint')
    parser.add_argument('--convert', action='store_true',
                       help='Convert checkpoint for MaskPLS-DGCNN')
    parser.add_argument('--output', type=str, default='dgcnn_converted.pth',
                       help='Output path for converted checkpoint')
    parser.add_argument('--save_mapping', type=str,
                       help='Save mapping to JSON file')
    
    args = parser.parse_args()
    
    # Analyze checkpoint
    state_dict = analyze_checkpoint(args.checkpoint)
    
    # Analyze structure
    structure = analyze_dgcnn_structure(state_dict)
    
    # Check compatibility
    compatibility = check_compatibility(state_dict)
    
    # Create mapping
    mapping = create_mapping_dict(state_dict)
    
    # Save mapping if requested
    if args.save_mapping:
        with open(args.save_mapping, 'w') as f:
            json.dump(mapping, f, indent=2)
        print(f"\n✓ Saved mapping to: {args.save_mapping}")
    
    # Convert if requested
    if args.convert and mapping:
        save_converted_checkpoint(state_dict, mapping, args.output)
    elif args.convert:
        print("\n✗ Cannot convert - no valid mappings found")
    
    # Final recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}\n")
    
    if compatibility['can_load']:
        print("1. This checkpoint can be partially loaded")
        print("2. Use the improved backbone script with partial loading")
        if compatibility['input_dim'] == 3:
            print("3. First conv layer will be adapted from 3D to 4D input")
        print("4. Expect faster convergence than training from scratch")
    else:
        print("1. This checkpoint appears incompatible")
        print("2. Consider training DGCNN from scratch on your data")
        print("3. Or look for a different pretrained DGCNN model")
    
    print("\nTo use with your training script:")
    print(f"  python train_efficient_dgcnn.py --pretrained {args.checkpoint}")

if __name__ == "__main__":
    main()