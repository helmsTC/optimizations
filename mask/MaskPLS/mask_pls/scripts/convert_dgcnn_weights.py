# convert_dgcnn_checkpoint.py
import torch
import argparse

def convert_dgcnn_checkpoint(input_path, output_path):
    """Convert DGCNN checkpoint to MaskPLS-compatible format"""
    
    # Load original checkpoint
    checkpoint = torch.load(input_path, map_location='cpu')
    
    # Extract state dict
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    else:
        state_dict = checkpoint
    
    # Create new state dict with proper naming
    new_state_dict = {}
    
    # Define the mapping (adjust based on actual checkpoint structure)
    mapping = {
        # Add your specific mappings here based on the checkpoint analysis
        'conv1.weight': 'edge_conv1.0.weight',
        'bn1.weight': 'edge_conv1.1.weight',
        'bn1.bias': 'edge_conv1.1.bias',
        'bn1.running_mean': 'edge_conv1.1.running_mean',
        'bn1.running_var': 'edge_conv1.1.running_var',
        # ... add more mappings
    }
    
    for old_name, new_name in mapping.items():
        if old_name in state_dict:
            new_state_dict[new_name] = state_dict[old_name]
    
    # Save converted checkpoint
    torch.save({'state_dict': new_state_dict}, output_path)
    print(f"Converted {len(new_state_dict)} parameters")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input DGCNN checkpoint')
    parser.add_argument('output', help='Output converted checkpoint')
    args = parser.parse_args()
    
    convert_dgcnn_checkpoint(args.input, args.output)