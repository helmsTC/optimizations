#!/usr/bin/env python3
"""
Extract model components directly without Lightning
"""

import torch
import torch.nn as nn
import numpy as np
import yaml
from pathlib import Path
from easydict import EasyDict as edict
import click


class DirectInferenceModel(nn.Module):
    """Direct inference without Lightning dependencies"""
    
    def __init__(self, checkpoint_path, cfg):
        super().__init__()
        
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        
        # Setup configuration
        dataset = cfg.MODEL.DATASET
        self.num_classes = cfg[dataset].NUM_CLASSES
        self.ignore_label = cfg[dataset].IGNORE_LABEL
        self.things_ids = [1, 2, 3, 4, 5, 6, 7, 8] if dataset == 'KITTI' else [2, 3, 4, 5, 6, 7, 9, 10]
        
        # Import ONLY the backbone and decoder (not the Lightning wrapper)
        from mask_pls.models.dgcnn.dgcnn_backbone_efficient import FixedDGCNNBackbone
        from mask_pls.models.decoder import MaskedTransformerDecoder
        from mask_pls.models.loss import MaskLoss, SemLoss
        
        # Create components directly
        print("Creating model components...")
        self.backbone = FixedDGCNNBackbone(cfg.BACKBONE, None)
        self.backbone.set_num_classes(self.num_classes)
        
        self.decoder = MaskedTransformerDecoder(
            cfg.DECODER,
            cfg.BACKBONE,
            cfg[dataset]
        )
        
        # Create loss modules (for panoptic inference logic)
        self.mask_loss = MaskLoss(cfg.LOSS, cfg[dataset])
        
        # Load weights from checkpoint
        print("Loading weights from checkpoint...")
        self._load_weights_from_checkpoint(state_dict)
        
        self.eval()
    
    def _load_weights_from_checkpoint(self, state_dict):
        """Load weights from checkpoint state dict"""
        
        # Load backbone weights
        backbone_state = {}
        decoder_state = {}
        
        for key, value in state_dict.items():
            if key.startswith('backbone.'):
                new_key = key.replace('backbone.', '')
                backbone_state[new_key] = value
            elif key.startswith('decoder.'):
                new_key = key.replace('decoder.', '')
                decoder_state[new_key] = value
        
        # Load into modules
        if backbone_state:
            missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)
            print(f"  Backbone: loaded {len(backbone_state)} params")
            if missing:
                print(f"    Missing keys: {len(missing)}")
            if unexpected:
                print(f"    Unexpected keys: {len(unexpected)}")
        
        if decoder_state:
            missing, unexpected = self.decoder.load_state_dict(decoder_state, strict=False)
            print(f"  Decoder: loaded {len(decoder_state)} params")
            if missing:
                print(f"    Missing keys: {len(missing)}")
            if unexpected:
                print(f"    Unexpected keys: {len(unexpected)}")
        
        # Verify critical weights are loaded
        self._verify_weights()
    
    def _verify_weights(self):
        """Verify critical weights are non-zero"""
        print("\nVerifying loaded weights...")
        
        # Check backbone
        critical_params = [
            ('edge_conv1', self.backbone.edge_conv1[0].weight if hasattr(self.backbone, 'edge_conv1') else None),
            ('edge_conv2', self.backbone.edge_conv2[0].weight if hasattr(self.backbone, 'edge_conv2') else None),
            ('edge_conv3', self.backbone.edge_conv3[0].weight if hasattr(self.backbone, 'edge_conv3') else None),
            ('edge_conv4', self.backbone.edge_conv4[0].weight if hasattr(self.backbone, 'edge_conv4') else None),
        ]
        
        for name, param in critical_params:
            if param is not None:
                max_val = param.abs().max().item()
                if max_val < 1e-6:
                    print(f"  WARNING: {name} appears uninitialized!")
                else:
                    print(f"  ✓ {name}: max weight = {max_val:.4f}")
        
        # Check decoder
        if hasattr(self.decoder, 'query_feat'):
            max_val = self.decoder.query_feat.weight.abs().max().item()
            print(f"  ✓ Decoder query_feat: max = {max_val:.4f}")
    
    @torch.no_grad()
    def forward(self, batch_dict):
        """
        Forward pass matching the original model
        Args:
            batch_dict: Dictionary with 'pt_coord' and 'feats' lists
        """
        # Run through backbone
        feats, coords, pad_masks, sem_logits = self.backbone(batch_dict)
        
        # Run through decoder
        outputs, padding = self.decoder(feats, coords, pad_masks)
        
        return outputs, padding, sem_logits
    
    @torch.no_grad()
    def inference(self, points, features):
        """
        Inference for single point cloud
        Args:
            points: numpy array or tensor [N, 3]
            features: numpy array or tensor [N, 4]
        """
        # Convert to numpy if needed
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        
        # Create batch dict
        batch_dict = {
            'pt_coord': [points],
            'feats': [features]
        }
        
        # Forward pass
        outputs, padding, sem_logits = self.forward(batch_dict)
        
        # Process outputs for single sample
        pred_logits = outputs['pred_logits'][0]  # [Q, C+1]
        pred_masks = outputs['pred_masks'][0]    # [N, Q]
        sem_logits = sem_logits[0]               # [N, C]
        padding = padding[0]                      # [N]
        
        return pred_logits, pred_masks, sem_logits, padding
    
    def panoptic_inference(self, outputs, padding):
        """Panoptic segmentation inference (from original model)"""
        mask_cls = outputs["pred_logits"]
        mask_pred = outputs["pred_masks"]
        
        batch_size = mask_cls.shape[0]
        sem_pred = []
        ins_pred = []
        
        for b in range(batch_size):
            valid_mask = ~padding[b]
            num_valid = valid_mask.sum().item()
            
            if num_valid == 0:
                sem_pred.append(np.zeros(0, dtype=np.int32))
                ins_pred.append(np.zeros(0, dtype=np.int32))
                continue
            
            scores, labels = mask_cls[b].max(-1)
            
            # Handle mask dimensions
            if mask_pred.dim() == 3:
                if mask_pred.shape[1] == mask_cls.shape[1]:  # [B, Q, N]
                    mask_pred_b = mask_pred[b].transpose(0, 1)  # [N, Q]
                else:  # [B, N, Q]
                    mask_pred_b = mask_pred[b]
            else:
                mask_pred_b = mask_pred[b]
            
            # Extract valid points
            mask_pred_b = mask_pred_b[valid_mask].sigmoid()
            
            # Filter valid predictions
            keep = labels.ne(self.num_classes)
            
            if keep.sum() == 0:
                sem_pred.append(np.zeros(num_valid, dtype=np.int32))
                ins_pred.append(np.zeros(num_valid, dtype=np.int32))
                continue
            
            # Get valid queries
            keep_indices = torch.where(keep)[0]
            cur_scores = scores[keep_indices]
            cur_classes = labels[keep_indices]
            cur_masks = mask_pred_b[:, keep_indices]
            
            # Weighted masks
            cur_prob_masks = cur_scores.unsqueeze(0) * cur_masks
            
            # Initialize outputs
            semantic_seg = torch.zeros(num_valid, dtype=torch.int32, device=cur_masks.device)
            instance_seg = torch.zeros(num_valid, dtype=torch.int32, device=cur_masks.device)
            
            if cur_masks.shape[1] > 0:
                cur_mask_ids = cur_prob_masks.argmax(1)
                
                current_segment_id = 0
                stuff_memory_list = {}
                
                for k in range(cur_classes.shape[0]):
                    pred_class = cur_classes[k].item()
                    isthing = pred_class in self.things_ids
                    
                    mask = (cur_mask_ids == k) & (cur_masks[:, k] >= 0.5)
                    mask_area = mask.sum().item()
                    
                    if mask_area > 0:
                        semantic_seg[mask] = pred_class
                        
                        if isthing:
                            current_segment_id += 1
                            instance_seg[mask] = current_segment_id
                        else:
                            if pred_class not in stuff_memory_list:
                                current_segment_id += 1
                                stuff_memory_list[pred_class] = current_segment_id
            
            sem_pred.append(semantic_seg.cpu().numpy())
            ins_pred.append(instance_seg.cpu().numpy())
        
        return sem_pred, ins_pred


def save_model(checkpoint_path, output_path, cfg):
    """Save model for inference"""
    
    # Create model
    model = DirectInferenceModel(checkpoint_path, cfg)
    model.eval()
    
    # Test the model
    print("\nTesting model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Test input
    test_points = np.random.randn(5000, 3).astype(np.float32) * 20
    test_features = np.random.randn(5000, 4).astype(np.float32)
    test_features[:, :3] = test_points
    
    with torch.no_grad():
        pred_logits, pred_masks, sem_logits, padding = model.inference(test_points, test_features)
    
    print(f"Test output shapes:")
    print(f"  pred_logits: {pred_logits.shape}")
    print(f"  pred_masks: {pred_masks.shape}")
    print(f"  sem_logits: {sem_logits.shape}")
    
    # Check outputs
    sem_pred = torch.argmax(sem_logits, dim=-1)
    unique_classes = torch.unique(sem_pred[~padding])
    print(f"Predicted classes: {unique_classes.cpu().numpy()}")
    
    # Save
    print(f"\nSaving model to {output_path}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'backbone_state_dict': model.backbone.state_dict(),
        'decoder_state_dict': model.decoder.state_dict(),
        'config': cfg,
        'model_class': 'DirectInferenceModel'
    }, output_path)
    
    print(f"✓ Model saved successfully")
    print(f"File size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")
    
    return model


@click.command()
@click.option('--checkpoint', required=True, help='Path to checkpoint')
@click.option('--output', default='model_direct.pth', help='Output file')
@click.option('--config', help='Config directory path')
def main(checkpoint, output, config):
    """Extract model without Lightning"""
    
    print("="*60)
    print("Direct Model Extraction (No Lightning)")
    print("="*60)
    
    # Load configuration
    if config:
        config_dir = Path(config)
    else:
        config_dir = Path(__file__).parent.parent / "config"
    
    cfg = edict()
    for config_file in ['model.yaml', 'backbone.yaml', 'decoder.yaml']:
        config_path = config_dir / config_file
        if config_path.exists():
            with open(config_path, 'r') as f:
                cfg.update(yaml.safe_load(f))
            print(f"✓ Loaded: {config_path}")
    
    # Save model
    model = save_model(checkpoint, output, cfg)
    
    print("\n✓ Export complete!")


if __name__ == "__main__":
    main()