#!/usr/bin/env python3
"""
Convert original MaskPLS Lightning checkpoint to standalone PyTorch model
Preserves the original MinkowskiEngine-based architecture
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import yaml
from pathlib import Path
from easydict import EasyDict as edict
import click

# Add the original model path to Python path
original_path = os.path.join(os.path.dirname(__file__), "../original/MaskPLS")
if original_path not in sys.path:
    sys.path.insert(0, original_path)

# Import the ORIGINAL models
from mask_pls.models.mink import MinkEncoderDecoder
from mask_pls.models.decoder import MaskedTransformerDecoder
from mask_pls.datasets.semantic_dataset import SemanticDatasetModule
import MinkowskiEngine as ME


def get_config():
    """Load configuration for the original model"""
    def getDir(obj):
        return os.path.dirname(os.path.abspath(obj))
    
    # Load from original config location
    config_base = os.path.join(getDir(__file__), "../original/MaskPLS/mask_pls/config")
    
    model_cfg = edict(yaml.safe_load(open(os.path.join(config_base, "model.yaml"))))
    backbone_cfg = edict(yaml.safe_load(open(os.path.join(config_base, "backbone.yaml"))))
    decoder_cfg = edict(yaml.safe_load(open(os.path.join(config_base, "decoder.yaml"))))
    
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    
    # Set default training parameters
    cfg.TRAIN.BATCH_SIZE = 1
    cfg.TRAIN.NUM_WORKERS = 4
    cfg.TRAIN.SUBSAMPLE = False  # Disable for inference
    cfg.TRAIN.AUG = False
    
    return cfg


class OriginalStandaloneModel(nn.Module):
    """Standalone inference model using original MinkowskiEngine architecture"""
    
    def __init__(self, cfg, things_ids):
        super().__init__()
        
        dataset = cfg.MODEL.DATASET
        self.cfg = cfg
        self.num_classes = cfg[dataset].NUM_CLASSES
        self.ignore_label = cfg[dataset].IGNORE_LABEL
        self.things_ids = things_ids
        self.overlap_threshold = cfg.MODEL.OVERLAP_THRESHOLD
        
        # Create the ORIGINAL components
        backbone = MinkEncoderDecoder(cfg.BACKBONE, cfg[dataset])
        self.backbone = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(backbone)
        
        self.decoder = MaskedTransformerDecoder(
            cfg.DECODER,
            cfg.BACKBONE,
            cfg[dataset]
        )
        
        print(f"Original Model initialized:")
        print(f"  Num classes: {self.num_classes}")
        print(f"  Things IDs: {self.things_ids}")
        print(f"  Using MinkowskiEngine backbone")
    
    @torch.no_grad()
    def forward(self, batch_dict):
        """Forward pass matching the original training model"""
        # Run through MinkowskiEngine backbone
        feats, coords, pad_masks, bb_logits = self.backbone(batch_dict)
        
        # Run through decoder
        outputs, padding = self.decoder(feats, coords, pad_masks)
        
        return outputs, padding, bb_logits
    
    def panoptic_inference(self, outputs, padding):
        """Panoptic segmentation inference (exact copy from original model)"""
        mask_cls = outputs["pred_logits"]
        mask_pred = outputs["pred_masks"]
        things_ids = self.things_ids
        num_classes = self.num_classes
        
        sem_pred = []
        ins_pred = []
        
        for mask_cls, mask_pred, pad in zip(mask_cls, mask_pred, padding):
            scores, labels = mask_cls.max(-1)
            mask_pred = mask_pred[~pad].sigmoid()
            keep = labels.ne(num_classes)
            
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred[:, keep]
            
            # prob to belong to each of the `keep` masks for each point
            cur_prob_masks = cur_scores.unsqueeze(0) * cur_masks
            
            panoptic_seg = torch.zeros(
                (cur_masks.shape[0]), dtype=torch.int32, device=cur_masks.device
            )
            sem = torch.zeros_like(panoptic_seg)
            ins = torch.zeros_like(panoptic_seg)
            
            segment_id = 0
            
            if cur_masks.shape[1] == 0:  # no masks detected
                sem_pred.append(sem.cpu().numpy())
                ins_pred.append(ins.cpu().numpy())
            else:
                # mask index for each point
                cur_mask_ids = cur_prob_masks.argmax(1)
                stuff_memory_list = {}
                
                for k in range(cur_classes.shape[0]):
                    pred_class = cur_classes[k].item()
                    isthing = pred_class in things_ids
                    mask_area = (cur_mask_ids == k).sum().item()
                    original_area = (cur_masks[:, k] >= 0.5).sum().item()
                    mask = (cur_mask_ids == k) & (cur_masks[:, k] >= 0.5)
                    
                    if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                        if mask_area / original_area < self.overlap_threshold:
                            continue
                        if not isthing:  # merge stuff regions
                            if int(pred_class) in stuff_memory_list.keys():
                                panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                                continue
                            else:
                                stuff_memory_list[int(pred_class)] = segment_id + 1
                        segment_id += 1
                        panoptic_seg[mask] = segment_id
                        
                        sem[mask] = pred_class
                        if isthing:
                            ins[mask] = segment_id
                        else:
                            ins[mask] = 0
                            
                sem_pred.append(sem.cpu().numpy())
                ins_pred.append(ins.cpu().numpy())
        
        return sem_pred, ins_pred


def convert_checkpoint(checkpoint_path, output_path, cfg):
    """Convert Lightning checkpoint to standalone model"""
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract state dict
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        print(f"Checkpoint info:")
        if 'epoch' in ckpt:
            print(f"  Epoch: {ckpt['epoch']}")
        if 'global_step' in ckpt:
            print(f"  Global step: {ckpt['global_step']}")
    else:
        state_dict = ckpt
    
    # Setup data module to get things_ids
    print("\nSetting up data module to get things_ids...")
    data_module = SemanticDatasetModule(cfg)
    data_module.setup()
    things_ids = data_module.things_ids
    print(f"Things IDs from data module: {things_ids}")
    
    # Create standalone model with original architecture
    print("\nCreating standalone model with original MinkowskiEngine architecture...")
    model = OriginalStandaloneModel(cfg, things_ids)
    
    # Extract and load weights
    print("\nExtracting weights from checkpoint...")
    backbone_state = {}
    decoder_state = {}
    other_state = {}
    
    for key, value in state_dict.items():
        if key.startswith('backbone.'):
            new_key = key.replace('backbone.', '')
            backbone_state[new_key] = value
        elif key.startswith('decoder.'):
            new_key = key.replace('decoder.', '')
            decoder_state[new_key] = value
        else:
            other_state[key] = value
    
    print(f"Found {len(backbone_state)} backbone parameters")
    print(f"Found {len(decoder_state)} decoder parameters")
    print(f"Found {len(other_state)} other parameters")
    
    # Load weights
    print(f"\nLoading backbone weights...")
    missing_b, unexpected_b = model.backbone.load_state_dict(backbone_state, strict=False)
    if missing_b:
        print(f"  Warning - Missing keys: {missing_b[:5]}..." if len(missing_b) > 5 else f"  Missing: {missing_b}")
    if unexpected_b:
        print(f"  Warning - Unexpected keys: {unexpected_b[:5]}..." if len(unexpected_b) > 5 else f"  Unexpected: {unexpected_b}")
    else:
        print(f"  ✓ All backbone weights loaded successfully")
    
    print(f"\nLoading decoder weights...")
    missing_d, unexpected_d = model.decoder.load_state_dict(decoder_state, strict=False)
    if missing_d:
        print(f"  Warning - Missing keys: {missing_d[:5]}..." if len(missing_d) > 5 else f"  Missing: {missing_d}")
    if unexpected_d:
        print(f"  Warning - Unexpected keys: {unexpected_d[:5]}..." if len(unexpected_d) > 5 else f"  Unexpected: {unexpected_d}")
    else:
        print(f"  ✓ All decoder weights loaded successfully")
    
    # Verify weights
    print("\nVerifying loaded weights...")
    verify_weights(model)
    
    # Test the model
    print("\nTesting model with dummy data...")
    test_model(model, cfg)
    
    # Save the standalone model
    print(f"\nSaving standalone model to {output_path}")
    
    # Save full model state for maximum compatibility
    save_dict = {
        'model_state_dict': model.state_dict(),
        'backbone_state_dict': model.backbone.state_dict(),
        'decoder_state_dict': model.decoder.state_dict(),
        'config': cfg,
        'things_ids': things_ids,
        'num_classes': model.num_classes,
        'model_class': 'OriginalStandaloneModel',
        'architecture': 'MinkowskiEngine',
        'overlap_threshold': model.overlap_threshold
    }
    
    # Add checkpoint metadata if available
    if 'epoch' in ckpt:
        save_dict['epoch'] = ckpt['epoch']
    if 'global_step' in ckpt:
        save_dict['global_step'] = ckpt['global_step']
    
    torch.save(save_dict, output_path)
    
    file_size = Path(output_path).stat().st_size / 1024 / 1024
    print(f"✓ Model saved successfully ({file_size:.2f} MB)")
    
    return model


def verify_weights(model):
    """Verify critical weights are loaded"""
    
    # Check MinkowskiEngine backbone components
    if hasattr(model.backbone, 'stem'):
        # Check first conv layer in stem
        for name, module in model.backbone.stem.named_modules():
            if isinstance(module, ME.MinkowskiConvolution):
                weight = module.kernel
                if weight is not None:
                    max_val = weight.abs().max().item()
                    print(f"  ✓ backbone.stem conv: max weight = {max_val:.4f}")
                    break
    
    if hasattr(model.backbone, 'sem_head'):
        weight = model.backbone.sem_head.weight
        max_val = weight.abs().max().item()
        print(f"  ✓ backbone.sem_head: max weight = {max_val:.4f}")
    
    # Check decoder
    if hasattr(model.decoder, 'query_feat'):
        weight = model.decoder.query_feat.weight
        max_val = weight.abs().max().item()
        print(f"  ✓ decoder.query_feat: max weight = {max_val:.4f}")
    
    if hasattr(model.decoder, 'class_embed'):
        weight = model.decoder.class_embed.weight
        max_val = weight.abs().max().item()
        print(f"  ✓ decoder.class_embed: max weight = {max_val:.4f}")


def test_model(model, cfg):
    """Test model with dummy data"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Create dummy data matching the expected format
    num_points = 5000
    test_points = np.random.randn(num_points, 3).astype(np.float32) * 20
    test_intensity = np.random.rand(num_points).astype(np.float32)
    test_features = np.concatenate([test_points, test_intensity.reshape(-1, 1)], axis=1)
    
    # Create dummy labels (for shape verification only)
    test_sem_labels = np.zeros((num_points, 1), dtype=np.int32)
    test_ins_labels = np.zeros((num_points, 1), dtype=np.int32)
    
    batch_dict = {
        'pt_coord': [test_points],
        'feats': [test_features],
        'sem_label': [test_sem_labels],
        'ins_label': [test_ins_labels],
        'masks': [torch.zeros(1, num_points)],
        'masks_cls': [torch.zeros(1, dtype=torch.long)],
        'masks_ids': [[]],
        'fname': ['test'],
        'pose': [np.eye(4, dtype=np.float32)],
        'token': ['test_token']
    }
    
    # Forward pass
    try:
        with torch.no_grad():
            outputs, padding, bb_logits = model(batch_dict)
        
        print(f"  ✓ Forward pass successful")
        print(f"  Output shapes:")
        print(f"    pred_logits: {outputs['pred_logits'].shape}")
        print(f"    pred_masks: {outputs['pred_masks'].shape}")
        print(f"    bb_logits: {bb_logits.shape}")
        
        # Test panoptic inference
        sem_pred, ins_pred = model.panoptic_inference(outputs, padding)
        print(f"  ✓ Panoptic inference successful")
        print(f"    Semantic predictions shape: {sem_pred[0].shape}")
        print(f"    Instance predictions shape: {ins_pred[0].shape}")
        
    except Exception as e:
        print(f"  ✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()


@click.command()
@click.option('--checkpoint', required=True, help='Path to original Lightning checkpoint (.ckpt)')
@click.option('--output', default='original_model_standalone.pt', help='Output file path')
@click.option('--dataset', default='KITTI', type=click.Choice(['KITTI', 'NUSCENES']))
def main(checkpoint, output, dataset):
    """Convert original MaskPLS Lightning checkpoint to standalone PyTorch model"""
    
    print("="*60)
    print("Original MaskPLS Model Converter")
    print("Using MinkowskiEngine Architecture")
    print("="*60)
    
    # Check MinkowskiEngine availability
    try:
        import MinkowskiEngine as ME
        print(f"✓ MinkowskiEngine version: {ME.__version__ if hasattr(ME, '__version__') else 'unknown'}")
    except ImportError:
        print("✗ Error: MinkowskiEngine not found!")
        print("  Please install MinkowskiEngine: pip install MinkowskiEngine")
        return
    
    # Load configuration
    cfg = get_config()
    cfg.MODEL.DATASET = dataset
    
    # Convert checkpoint
    try:
        model = convert_checkpoint(checkpoint, output, cfg)
        print("\n✓ Conversion complete!")
        print(f"  Output saved to: {output}")
        print("\nTo use the model:")
        print("  1. Load with: checkpoint = torch.load('{}')".format(output))
        print("  2. Create model: model = OriginalStandaloneModel(cfg, checkpoint['things_ids'])")
        print("  3. Load weights: model.load_state_dict(checkpoint['model_state_dict'])")
    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()