#!/usr/bin/env python3
"""
Convert Lightning checkpoint to standalone PyTorch model
Ensures exact architecture match with training
"""

import os
import torch
import torch.nn as nn
import numpy as np
import yaml
from pathlib import Path
from easydict import EasyDict as edict
import click

# Import the EXACT models used in training
from mask_pls.models.dgcnn.dgcnn_backbone_efficient import FixedDGCNNBackbone
from mask_pls.models.decoder import MaskedTransformerDecoder
from mask_pls.datasets.semantic_dataset import SemanticDatasetModule


def get_config():
    """Load configuration exactly as in training"""
    def getDir(obj):
        return os.path.dirname(os.path.abspath(obj))
    
    model_cfg = edict(yaml.safe_load(open(os.path.join(getDir(__file__), "../config/model.yaml"))))
    backbone_cfg = edict(yaml.safe_load(open(os.path.join(getDir(__file__), "../config/backbone.yaml"))))
    decoder_cfg = edict(yaml.safe_load(open(os.path.join(getDir(__file__), "../config/decoder.yaml"))))
    
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    
    # Match training configuration
    cfg.TRAIN.BATCH_SIZE = 2
    cfg.TRAIN.ACCUMULATE_GRAD_BATCHES = 2
    cfg.TRAIN.WARMUP_STEPS = 500
    cfg.TRAIN.SUBSAMPLE = True
    cfg.TRAIN.AUG = True
    cfg.TRAIN.LR = 0.0001
    
    return cfg


class StandaloneInferenceModel(nn.Module):
    """Standalone inference model without Lightning dependencies"""
    
    def __init__(self, cfg, things_ids):
        super().__init__()
        
        dataset = cfg.MODEL.DATASET
        self.num_classes = cfg[dataset].NUM_CLASSES
        self.ignore_label = cfg[dataset].IGNORE_LABEL
        self.things_ids = things_ids
        
        # Create the EXACT same components as in training
        self.backbone = FixedDGCNNBackbone(cfg.BACKBONE, None)
        self.backbone.set_num_classes(self.num_classes)
        
        self.decoder = MaskedTransformerDecoder(
            cfg.DECODER,
            cfg.BACKBONE,
            cfg[dataset]
        )
        
        print(f"Model initialized:")
        print(f"  Num classes: {self.num_classes}")
        print(f"  Things IDs: {self.things_ids}")
    
    @torch.no_grad()
    def forward(self, batch_dict):
        """Forward pass matching the training model"""
        # Run through backbone
        feats, coords, pad_masks, sem_logits = self.backbone(batch_dict)
        
        # Run through decoder
        outputs, padding = self.decoder(feats, coords, pad_masks)
        
        return outputs, padding, sem_logits
    
    def panoptic_inference(self, outputs, padding):
        """Panoptic segmentation inference (exact copy from training model)"""
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
            mask_pred_b = mask_pred[b][valid_mask].sigmoid()
            
            # Filter valid predictions
            keep = labels.ne(self.num_classes)
            
            if keep.sum() == 0:
                sem_pred.append(np.zeros(num_valid, dtype=np.int32))
                ins_pred.append(np.zeros(num_valid, dtype=np.int32))
                continue
            
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred_b[:, keep]
            
            # Instance assignment
            cur_prob_masks = cur_scores.unsqueeze(0) * cur_masks
            
            semantic_seg = torch.zeros(num_valid, dtype=torch.int32, device=cur_masks.device)
            instance_seg = torch.zeros(num_valid, dtype=torch.int32, device=cur_masks.device)
            
            current_segment_id = 0
            
            if cur_masks.shape[1] > 0:
                cur_mask_ids = cur_prob_masks.argmax(1)
                stuff_memory_list = {}
                
                for k in range(cur_classes.shape[0]):
                    pred_class = cur_classes[k].item()
                    isthing = pred_class in self.things_ids
                    
                    mask = (cur_mask_ids == k) & (cur_masks[:, k] >= 0.5)
                    mask_area = mask.sum().item()
                    
                    if mask_area > 0:
                        if isthing:
                            current_segment_id += 1
                            instance_seg[mask] = current_segment_id
                            semantic_seg[mask] = pred_class
                        else:
                            if pred_class not in stuff_memory_list:
                                current_segment_id += 1
                                stuff_memory_list[pred_class] = current_segment_id
                            semantic_seg[mask] = pred_class
            
            sem_pred.append(semantic_seg.cpu().numpy())
            ins_pred.append(instance_seg.cpu().numpy())
        
        return sem_pred, ins_pred
    
    def prepare_batch_for_eval(self, batch, sem_pred):
        """Prepare batch for evaluation with subsampling handling"""
        subsample_indices = self.backbone.subsample_indices
        
        eval_batch = {'fname': batch['fname']}
        eval_sem_labels = []
        eval_ins_labels = []
        
        for b in range(len(sem_pred)):
            sem_label = batch['sem_label'][b]
            ins_label = batch['ins_label'][b]
            
            # Convert to numpy if needed
            if isinstance(sem_label, torch.Tensor):
                sem_label = sem_label.cpu().numpy()
            if isinstance(ins_label, torch.Tensor):
                ins_label = ins_label.cpu().numpy()
            
            # Apply subsampling if needed
            if b in subsample_indices:
                indices = subsample_indices[b].cpu().numpy()
                if indices.shape[0] < sem_label.shape[0]:
                    sem_label = sem_label[indices]
                    ins_label = ins_label[indices]
            
            # Match prediction size
            pred_size = len(sem_pred[b])
            if pred_size < len(sem_label):
                sem_label = sem_label[:pred_size]
                ins_label = ins_label[:pred_size]
            
            eval_sem_labels.append(sem_label)
            eval_ins_labels.append(ins_label)
        
        eval_batch['sem_label'] = eval_sem_labels
        eval_batch['ins_label'] = eval_ins_labels
        
        return eval_batch


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
    
    # Create standalone model
    print("\nCreating standalone model...")
    model = StandaloneInferenceModel(cfg, things_ids)
    
    # Extract and load weights
    print("\nExtracting weights from checkpoint...")
    backbone_state = {}
    decoder_state = {}
    
    for key, value in state_dict.items():
        if key.startswith('backbone.'):
            new_key = key.replace('backbone.', '')
            backbone_state[new_key] = value
        elif key.startswith('decoder.'):
            new_key = key.replace('decoder.', '')
            decoder_state[new_key] = value
    
    # Load weights
    print(f"Loading backbone weights ({len(backbone_state)} parameters)...")
    missing_b, unexpected_b = model.backbone.load_state_dict(backbone_state, strict=False)
    if missing_b:
        print(f"  Missing: {len(missing_b)}")
    if unexpected_b:
        print(f"  Unexpected: {len(unexpected_b)}")
    
    print(f"Loading decoder weights ({len(decoder_state)} parameters)...")
    missing_d, unexpected_d = model.decoder.load_state_dict(decoder_state, strict=False)
    if missing_d:
        print(f"  Missing: {len(missing_d)}")
    if unexpected_d:
        print(f"  Unexpected: {len(unexpected_d)}")
    
    # Verify weights
    print("\nVerifying loaded weights...")
    verify_weights(model)
    
    # Test the model
    print("\nTesting model with dummy data...")
    test_model(model)
    
    # Save the standalone model
    print(f"\nSaving standalone model to {output_path}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'backbone_state_dict': model.backbone.state_dict(),
        'decoder_state_dict': model.decoder.state_dict(),
        'config': cfg,
        'things_ids': things_ids,
        'num_classes': model.num_classes,
        'model_class': 'StandaloneInferenceModel'
    }, output_path)
    
    file_size = Path(output_path).stat().st_size / 1024 / 1024
    print(f"✓ Model saved successfully ({file_size:.2f} MB)")
    
    return model


def verify_weights(model):
    """Verify critical weights are loaded"""
    
    # Check backbone
    if hasattr(model.backbone, 'edge_conv1'):
        weight = model.backbone.edge_conv1[0].weight
        max_val = weight.abs().max().item()
        print(f"  ✓ edge_conv1: max weight = {max_val:.4f}")
    
    if hasattr(model.backbone, 'conv5'):
        weight = model.backbone.conv5[0].weight
        max_val = weight.abs().max().item()
        print(f"  ✓ conv5: max weight = {max_val:.4f}")
    
    # Check decoder
    if hasattr(model.decoder, 'query_feat'):
        weight = model.decoder.query_feat.weight
        max_val = weight.abs().max().item()
        print(f"  ✓ decoder.query_feat: max weight = {max_val:.4f}")


def test_model(model):
    """Test model with dummy data"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Create dummy data
    test_points = np.random.randn(5000, 3).astype(np.float32) * 20
    test_features = np.random.randn(5000, 4).astype(np.float32)
    test_features[:, :3] = test_points
    
    batch_dict = {
        'pt_coord': [test_points],
        'feats': [test_features]
    }
    
    # Forward pass
    with torch.no_grad():
        outputs, padding, sem_logits = model(batch_dict)
    
    print(f"  Output shapes:")
    print(f"    pred_logits: {outputs['pred_logits'].shape}")
    print(f"    pred_masks: {outputs['pred_masks'].shape}")
    print(f"    sem_logits: {sem_logits.shape}")
    
    # Check semantic predictions
    sem_pred = torch.argmax(sem_logits[0], dim=-1)
    unique_classes = torch.unique(sem_pred[~padding[0]])
    print(f"  Predicted classes: {unique_classes.cpu().numpy()}")


@click.command()
@click.option('--checkpoint', required=True, help='Path to Lightning checkpoint (.ckpt)')
@click.option('--output', default='model_standalone.pt', help='Output file path')
@click.option('--dataset', default='KITTI', type=click.Choice(['KITTI', 'NUSCENES']))
def main(checkpoint, output, dataset):
    """Convert Lightning checkpoint to standalone PyTorch model"""
    
    print("="*60)
    print("Lightning Checkpoint to Standalone Model Converter")
    print("="*60)
    
    # Load configuration
    cfg = get_config()
    cfg.MODEL.DATASET = dataset
    
    # Convert checkpoint
    model = convert_checkpoint(checkpoint, output, cfg)
    
    print("\n✓ Conversion complete!")


if __name__ == "__main__":
    main()