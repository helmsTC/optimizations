# mask_pls/scripts/export_efficient_dgcnn_onnx.py
"""
Export MaskPLS-DGCNN Fixed model checkpoint to ONNX
For checkpoints trained with train_efficient_dgcnn.py
Fixed version that handles randperm issue
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path
import click
import yaml
from easydict import EasyDict as edict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from mask_pls.models.dgcnn.maskpls_dgcnn_optimized import MaskPLSDGCNNFixed
from mask_pls.models.dgcnn.dgcnn_backbone_efficient_fixed import FixedDGCNNBackbone


class ONNXSafeBackbone(nn.Module):
    """
    ONNX-safe version of the DGCNN backbone that avoids randperm
    """
    def __init__(self, original_backbone):
        super().__init__()
        # Copy all attributes from original
        self.__dict__.update(original_backbone.__dict__)
        
        # Store original modules
        self.edge_conv1 = original_backbone.edge_conv1
        self.edge_conv2 = original_backbone.edge_conv2
        self.edge_conv3 = original_backbone.edge_conv3
        self.edge_conv4 = original_backbone.edge_conv4
        self.conv5 = original_backbone.conv5
        self.feat_layers = original_backbone.feat_layers
        self.out_bn = original_backbone.out_bn
        self.sem_head = original_backbone.sem_head
        self.num_classes = original_backbone.num_classes
        self.k = original_backbone.k
        
    def forward(self, x):
        """Forward pass with deterministic subsampling for ONNX"""
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
            coords = torch.from_numpy(coords_list[b]).float()
            feats = torch.from_numpy(feats_list[b]).float()
            
            # DETERMINISTIC subsampling for ONNX (use stride instead of randperm)
            max_points = 50000 if self.training else 30000
            if coords.shape[0] > max_points:
                # Use deterministic stride-based sampling instead of randperm
                stride = coords.shape[0] // max_points
                indices = torch.arange(0, coords.shape[0], stride)[:max_points]
                indices = indices.sort()[0]
                coords = coords[indices]
                feats = feats[indices]
                self.subsample_indices[b] = indices
            else:
                self.subsample_indices[b] = torch.arange(coords.shape[0])
            
            # Process through DGCNN
            point_features = self.process_single_cloud(coords, feats)
            
            all_features.append(point_features)
            all_coords.append(coords)
            all_masks.append(torch.zeros(coords.shape[0], dtype=torch.bool))
        
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
        
        # Get graph features - need to import the functions
        from mask_pls.models.dgcnn.dgcnn_backbone_efficient_fixed import get_graph_feature
        
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
                feat = torch.nn.functional.pad(feat, (0, 0, 0, pad_size))
                coord = torch.nn.functional.pad(coord, (0, 0, 0, pad_size))
                mask = torch.nn.functional.pad(mask, (0, pad_size), value=True)
            
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
                logits = torch.zeros(0, self.num_classes)
            
            # Pad back to full size
            full_logits = torch.zeros(features.shape[1], self.num_classes)
            if valid_mask.sum() > 0:
                full_logits[valid_mask] = logits
            
            sem_logits.append(full_logits)
        
        return torch.stack(sem_logits)


class ONNXExportWrapper(nn.Module):
    """
    Wrapper for ONNX export that handles the MaskPLSDGCNN Fixed model
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Replace backbone with ONNX-safe version
        self.backbone = ONNXSafeBackbone(model.backbone)
        self.decoder = model.decoder
        self.num_classes = model.num_classes
        
        # Set to eval mode
        self.eval()
        self.backbone.eval()
        self.decoder.eval()
    
    def forward(self, point_coords, point_features):
        """
        Simplified forward pass for ONNX export
        
        Args:
            point_coords: [B, N, 3] point coordinates
            point_features: [B, N, 4] point features (x, y, z, intensity)
            
        Returns:
            pred_logits: [B, Q, num_classes+1] class predictions for queries
            pred_masks: [B, N, Q] mask predictions
            sem_logits: [B, N, num_classes] semantic segmentation logits
        """
        B, N, _ = point_coords.shape
        
        # Create the expected input format
        x = {
            'pt_coord': [],
            'feats': []
        }
        
        # Convert batch to list format expected by backbone
        for b in range(B):
            x['pt_coord'].append(point_coords[b].detach().cpu().numpy())
            x['feats'].append(point_features[b].detach().cpu().numpy())
        
        # Forward through backbone (ONNX-safe version)
        with torch.no_grad():
            ms_features, ms_coords, ms_masks, sem_logits = self.backbone(x)
            
            # Forward through decoder
            outputs, padding = self.decoder(ms_features, ms_coords, ms_masks)
        
        # Extract outputs
        pred_logits = outputs['pred_logits']
        pred_masks = outputs['pred_masks']
        
        return pred_logits, pred_masks, sem_logits


def load_config(checkpoint_dir):
    """Load configuration from checkpoint directory"""
    config_dir = Path(__file__).parent.parent / "config"
    
    # Load base configs
    model_cfg = edict(yaml.safe_load(open(config_dir / "model.yaml")))
    backbone_cfg = edict(yaml.safe_load(open(config_dir / "backbone.yaml")))
    decoder_cfg = edict(yaml.safe_load(open(config_dir / "decoder.yaml")))
    
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    
    # Check for saved hyperparameters
    hparams_file = checkpoint_dir.parent / 'hparams.yaml'
    if hparams_file.exists():
        print(f"Loading saved hyperparameters from {hparams_file}")
        with open(hparams_file, 'r') as f:
            saved_cfg = yaml.safe_load(f)
            if saved_cfg:
                cfg.update(saved_cfg)
    
    # Set default training params if missing
    if 'TRAIN' not in cfg:
        cfg.TRAIN = edict({
            'BATCH_SIZE': 2,
            'WARMUP_STEPS': 500,
            'SUBSAMPLE': True,
            'AUG': True,
            'LR': 0.0001
        })
    
    return cfg


@click.command()
@click.option('--checkpoint', '-c', required=True, help='Path to checkpoint file (.ckpt)')
@click.option('--output', '-o', default=None, help='Output ONNX file path')
@click.option('--batch_size', '-b', default=1, help='Batch size for export')
@click.option('--num_points', '-n', default=10000, help='Number of points for dummy input')
@click.option('--opset', default=14, help='ONNX opset version')
@click.option('--simplify', is_flag=True, help='Simplify the exported model')
@click.option('--validate', is_flag=True, help='Validate exported model')
def export_onnx(checkpoint, output, batch_size, num_points, opset, simplify, validate):
    """Export MaskPLS-DGCNN Fixed model to ONNX"""
    
    print("="*60)
    print("MaskPLS-DGCNN ONNX Export (Fixed for randperm)")
    print("="*60)
    
    # Set device to CPU for ONNX export to avoid device issues
    device = torch.device("cpu")
    print(f"\nDevice: {device} (CPU export for ONNX compatibility)")
    
    # Set output path
    if output is None:
        checkpoint_path = Path(checkpoint)
        output = checkpoint_path.parent / f"{checkpoint_path.stem}.onnx"
    
    print(f"Checkpoint: {checkpoint}")
    print(f"Output: {output}")
    print(f"Batch size: {batch_size}")
    print(f"Points: {num_points}")
    
    # Load configuration
    print("\nLoading configuration...")
    checkpoint_dir = Path(checkpoint).parent
    cfg = load_config(checkpoint_dir)
    
    # Create model
    print("\nCreating model...")
    model = MaskPLSDGCNNFixed(cfg)
    
    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint_data = torch.load(checkpoint, map_location=device)
    
    if 'state_dict' in checkpoint_data:
        state_dict = checkpoint_data['state_dict']
        epoch = checkpoint_data.get('epoch', 'unknown')
        print(f"  Checkpoint from epoch: {epoch}")
    else:
        state_dict = checkpoint_data
    
    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"  Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"  Unexpected keys: {len(unexpected_keys)}")
    
    # Move to device and set to eval
    model = model.to(device)
    model.eval()
    
    # Create ONNX-safe wrapper
    print("\nCreating ONNX-safe wrapper...")
    onnx_model = ONNXExportWrapper(model).to(device)
    
    # Create dummy input
    dummy_coords = torch.randn(batch_size, num_points, 3, device=device)
    dummy_features = torch.randn(batch_size, num_points, 4, device=device)
    dummy_input = (dummy_coords, dummy_features)
    
    input_names = ['point_coords', 'point_features']
    output_names = ['pred_logits', 'pred_masks', 'sem_logits']
    dynamic_axes = {
        'point_coords': {0: 'batch_size', 1: 'num_points'},
        'point_features': {0: 'batch_size', 1: 'num_points'},
        'pred_logits': {0: 'batch_size'},
        'pred_masks': {0: 'batch_size', 1: 'num_points'},
        'sem_logits': {0: 'batch_size', 1: 'num_points'}
    }
    
    # Test forward pass
    print("\nTesting forward pass...")
    try:
        with torch.no_grad():
            test_output = onnx_model(*dummy_input)
        print(f"✓ Forward pass successful")
        print(f"  Output shapes: {[t.shape for t in test_output]}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Export to ONNX
    print("\nExporting to ONNX...")
    print("  Note: Using deterministic subsampling to avoid randperm")
    
    try:
        with torch.no_grad():
            torch.onnx.export(
                onnx_model,
                dummy_input,
                str(output),
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=False,
                # Disable some checks that might fail with our custom ops
                operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
            )
        
        print(f"✓ Model exported to {output}")
        
        # Get file size
        file_size = Path(output).stat().st_size / (1024 * 1024)
        print(f"  File size: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Check model
    print("\nChecking ONNX model...")
    try:
        model_onnx = onnx.load(str(output))
        onnx.checker.check_model(model_onnx)
        print("✓ ONNX model is valid")
    except Exception as e:
        print(f"✗ Model check failed: {e}")
    
    # Simplify if requested
    if simplify:
        print("\nSimplifying model...")
        try:
            import onnxsim
            model_onnx, check = onnxsim.simplify(model_onnx)
            if check:
                onnx.save(model_onnx, str(output))
                new_size = Path(output).stat().st_size / (1024 * 1024)
                print(f"✓ Model simplified")
                print(f"  New size: {new_size:.2f} MB (was {file_size:.2f} MB)")
            else:
                print("✗ Simplification check failed")
        except ImportError:
            print("  ⚠ onnx-simplifier not installed")
            print("  Install with: pip install onnx-simplifier")
    
    # Validate if requested
    if validate:
        print("\nValidating ONNX model...")
        try:
            # Create inference session
            providers = ['CPUExecutionProvider']  # Use CPU for validation
            session = ort.InferenceSession(str(output), providers=providers)
            
            print(f"  Provider: {session.get_providers()[0]}")
            
            # Test inference
            test_coords = np.random.randn(batch_size, num_points, 3).astype(np.float32)
            test_features = np.random.randn(batch_size, num_points, 4).astype(np.float32)
            
            inputs = {
                'point_coords': test_coords,
                'point_features': test_features
            }
            
            outputs = session.run(None, inputs)
            
            print("✓ Inference successful")
            print(f"  Output shapes: {[o.shape for o in outputs]}")
            
        except Exception as e:
            print(f"✗ Validation failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("Export completed!")
    print("="*60)
    print("\n⚠ Note: This export uses deterministic subsampling instead of random")
    print("  to ensure ONNX compatibility. Performance may slightly differ from")
    print("  the original model with random subsampling.")


if __name__ == "__main__":
    export_onnx()