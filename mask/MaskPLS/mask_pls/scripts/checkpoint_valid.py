#!/usr/bin/env python
"""
Utility script to inspect DGCNN checkpoint and understand its structure
Usage: python inspect_checkpoint.py --checkpoint path/to/dgcnn.pth
"""

import torch
import argparse
from collections import defaultdict

def inspect_checkpoint(checkpoint_path):
    """Inspect a checkpoint file and print its structure"""
    
    print(f"\n{'='*60}")
    print(f"Inspecting checkpoint: {checkpoint_path}")
    print(f"{'='*60}\n")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Check what's in the checkpoint
    print("Checkpoint keys:", list(checkpoint.keys()) if isinstance(checkpoint, dict) else "Direct state_dict")
    print()
    
    # Get the state dict
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("Using 'state_dict' from checkpoint")
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("Using 'model_state_dict' from checkpoint")
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
            print("Using 'model' from checkpoint")
        else:
            # Assume the checkpoint itself is the state dict
            state_dict = checkpoint
            print("Using checkpoint directly as state_dict")
    else:
        state_dict = checkpoint
        print("Checkpoint is directly a state_dict")
    
    print(f"\nTotal parameters: {len(state_dict)}")
    print("="*60)
    
    # Group parameters by prefix
    param_groups = defaultdict(list)
    for key in state_dict.keys():
        prefix = key.split('.')[0]
        param_groups[prefix].append(key)
    
    # Print grouped parameters
    print("\nParameters grouped by prefix:")
    print("-"*40)
    for prefix, keys in sorted(param_groups.items()):
        print(f"\n{prefix}: ({len(keys)} parameters)")
        for key in sorted(keys)[:5]:  # Show first 5 of each group
            shape = state_dict[key].shape
            print(f"  {key}: {shape}")
        if len(keys) > 5:
            print(f"  ... and {len(keys)-5} more")
    
    # Analyze layer types
    print("\n" + "="*60)
    print("Layer Analysis:")
    print("-"*40)
    
    conv_layers = [k for k in state_dict.keys() if 'conv' in k.lower() and 'weight' in k]
    bn_layers = [k for k in state_dict.keys() if ('bn' in k.lower() or 'batch' in k.lower()) and 'weight' in k]
    linear_layers = [k for k in state_dict.keys() if ('linear' in k.lower() or 'fc' in k.lower()) and 'weight' in k]
    
    print(f"Convolutional layers: {len(conv_layers)}")
    for layer in conv_layers[:10]:
        print(f"  {layer}: {state_dict[layer].shape}")
    
    print(f"\nBatch Norm layers: {len(bn_layers)}")
    for layer in bn_layers[:10]:
        print(f"  {layer}: {state_dict[layer].shape}")
    
    print(f"\nLinear/FC layers: {len(linear_layers)}")
    for layer in linear_layers[:10]:
        print(f"  {layer}: {state_dict[layer].shape}")
    
    # Check for DGCNN specific patterns
    print("\n" + "="*60)
    print("DGCNN Pattern Check:")
    print("-"*40)
    
    # Check for EdgeConv patterns
    edge_conv_patterns = ['edge_conv', 'edgeconv']
    has_edge_conv = any(any(pattern in k.lower() for pattern in edge_conv_patterns) for k in state_dict.keys())
    print(f"Has EdgeConv layers: {has_edge_conv}")
    
    # Check for standard DGCNN conv naming
    standard_convs = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
    for conv_name in standard_convs:
        matching = [k for k in state_dict.keys() if conv_name in k and 'weight' in k]
        if matching:
            print(f"{conv_name}: Found - {matching[0]} with shape {state_dict[matching[0]].shape}")
        else:
            print(f"{conv_name}: Not found")
    
    # Check input dimensions (first conv layer)
    print("\n" + "="*60)
    print("Input/Output Dimensions:")
    print("-"*40)
    
    first_conv_candidates = ['conv1.weight', 'edge_conv1.0.weight', 'conv1.conv.weight']
    for candidate in first_conv_candidates:
        if candidate in state_dict:
            weight = state_dict[candidate]
            print(f"First conv layer: {candidate}")
            print(f"  Shape: {weight.shape}")
            print(f"  Input channels: {weight.shape[1]}")
            print(f"  Output channels: {weight.shape[0]}")
            break
    
    # Check for classification head
    print("\nClassification head:")
    classifier_candidates = ['linear3.weight', 'fc3.weight', 'classifier.weight', 'head.weight']
    for candidate in classifier_candidates:
        if candidate in state_dict:
            weight = state_dict[candidate]
            print(f"  {candidate}: {weight.shape} (predicts {weight.shape[0]} classes)")
    
    return state_dict

def compare_with_target_model():
    """Compare pretrained model with target MaskPLS-DGCNN model structure"""
    print("\n" + "="*60)
    print("Expected MaskPLS-DGCNN structure:")
    print("-"*40)
    
    expected_structure = {
        "edge_conv1": "EdgeConv layer 1 (input_dim*2 -> 64)",
        "edge_conv2": "EdgeConv layer 2 (64*2 -> 64)",
        "edge_conv3": "EdgeConv layer 3 (64*2 -> 128)",
        "edge_conv4": "EdgeConv layer 4 (128*2 -> 256)",
        "conv5": "Aggregation layer (512 -> 512)",
        "feat_layers": "Multi-scale feature extraction",
        "sem_head": "Semantic segmentation head"
    }
    
    for key, desc in expected_structure.items():
        print(f"  {key}: {desc}")
    
    print("\nTo successfully load pretrained weights, you need to map:")
    print("  - conv1/bn1 -> edge_conv1.0/edge_conv1.1")
    print("  - conv2/bn2 -> edge_conv2.0/edge_conv2.1")
    print("  - conv3/bn3 -> edge_conv3.0/edge_conv3.1")
    print("  - conv4/bn4 -> edge_conv4.0/edge_conv4.1")
    print("  - conv5/bn5 -> conv5.0/conv5.1")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inspect DGCNN checkpoint structure')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    args = parser.parse_args()
    
    state_dict = inspect_checkpoint(args.checkpoint)
    compare_with_target_model()
