"""
Test the actual training pipeline to find where CUDA error occurs
Save as: mask_pls/scripts/test_training_pipeline.py
"""
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

import torch
import yaml
import numpy as np
from easydict import EasyDict as edict
from os.path import join, dirname, abspath

# Import actual components
from mask_pls.models.onnx.simplified_model import MaskPLSSimplifiedONNX
from mask_pls.datasets.semantic_dataset import SemanticDatasetModule
from mask_pls.models.loss import MaskLoss, SemLoss


def test_actual_pipeline():
    """Test the actual training pipeline step by step"""
    print("Testing Actual Training Pipeline")
    print("=" * 60)
    
    # Load configs
    base_dir = dirname(dirname(abspath(__file__)))
    model_cfg = edict(yaml.safe_load(open(join(base_dir, "config/model.yaml"))))
    backbone_cfg = edict(yaml.safe_load(open(join(base_dir, "config/backbone.yaml"))))
    decoder_cfg = edict(yaml.safe_load(open(join(base_dir, "config/decoder.yaml"))))
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    
    # Settings
    cfg.TRAIN.BATCH_SIZE = 1
    cfg.TRAIN.NUM_WORKERS = 0
    dataset = cfg.MODEL.DATASET
    num_classes = cfg[dataset].NUM_CLASSES
    
    print(f"Configuration:")
    print(f"  Dataset: {dataset}")
    print(f"  Num classes: {num_classes}")
    print(f"  Ignore label: {cfg[dataset].IGNORE_LABEL}")
    
    # Create model and data
    print("\n1. Creating model...")
    model = MaskPLSSimplifiedONNX(cfg).cuda()
    model.eval()
    
    print("\n2. Creating dataset...")
    data = SemanticDatasetModule(cfg)
    data.setup()
    train_loader = data.train_dataloader()
    
    # Get one batch
    print("\n3. Getting batch...")
    batch = next(iter(train_loader))
    
    print(f"\nBatch info:")
    print(f"  Batch size: {len(batch['pt_coord'])}")
    for i in range(len(batch['pt_coord'])):
        print(f"  Sample {i}:")
        print(f"    Points: {batch['pt_coord'][i].shape}")
        print(f"    Sem labels: {batch['sem_label'][i].shape}, range [{batch['sem_label'][i].min()}, {batch['sem_label'][i].max()}]")
        print(f"    Masks: {len(batch['masks_cls'][i])}")
        if len(batch['masks_cls'][i]) > 0:
            cls_values = [c.item() if hasattr(c, 'item') else c for c in batch['masks_cls'][i]]
            print(f"    Mask classes: {cls_values}")
    
    # Process through model (similar to forward in training)
    print("\n4. Processing through model...")
    
    with torch.no_grad():
        # Pre-voxelize the batch (from training script)
        batch_voxels = []
        batch_coords = []
        valid_indices = []
        
        for i in range(len(batch['pt_coord'])):
            pts = torch.from_numpy(batch['pt_coord'][i]).float().cuda()
            feat = torch.from_numpy(batch['feats'][i]).float().cuda()
            
            # Filter by bounds
            bounds = cfg[dataset].SPACE
            valid_mask = torch.ones(pts.shape[0], dtype=torch.bool, device=pts.device)
            for dim in range(3):
                valid_mask &= (pts[:, dim] >= bounds[dim][0]) & (pts[:, dim] < bounds[dim][1])
            
            valid_idx = torch.where(valid_mask)[0]
            valid_pts = pts[valid_mask]
            valid_feat = feat[valid_mask]
            
            print(f"\n  Sample {i} preprocessing:")
            print(f"    Original points: {pts.shape[0]}")
            print(f"    Valid points: {valid_pts.shape[0]}")
            print(f"    Valid indices: {valid_idx.shape[0]}")
            
            # Subsample if needed
            max_pts = cfg[dataset].SUB_NUM_POINTS
            if len(valid_pts) > max_pts:
                perm = torch.randperm(len(valid_pts))[:max_pts]
                valid_pts = valid_pts[perm]
                valid_feat = valid_feat[perm]
                valid_idx = valid_idx[perm]
                print(f"    After subsampling: {valid_pts.shape[0]}")
            
            # Normalize coordinates
            norm_coords = torch.zeros_like(valid_pts)
            for dim in range(3):
                norm_coords[:, dim] = (valid_pts[:, dim] - bounds[dim][0]) / (bounds[dim][1] - bounds[dim][0])
            
            # Voxelize
            voxel_grid = model.voxelize_points(valid_pts.unsqueeze(0), valid_feat.unsqueeze(0))[0]
            
            batch_voxels.append(voxel_grid)
            batch_coords.append(norm_coords)
            valid_indices.append(valid_idx)
        
        # Pad and stack
        max_pts = max(c.shape[0] for c in batch_coords)
        padded_coords = []
        padding_masks = []
        
        for coords in batch_coords:
            n_pts = coords.shape[0]
            if n_pts < max_pts:
                coords = torch.nn.functional.pad(coords, (0, 0, 0, max_pts - n_pts))
            padded_coords.append(coords)
            
            mask = torch.zeros(max_pts, dtype=torch.bool, device=coords.device)
            mask[n_pts:] = True
            padding_masks.append(mask)
        
        batch_voxels = torch.stack(batch_voxels)
        batch_coords = torch.stack(padded_coords)
        padding_masks = torch.stack(padding_masks)
        
        print(f"\n  Batched tensors:")
        print(f"    Voxels: {batch_voxels.shape}")
        print(f"    Coords: {batch_coords.shape}")
        print(f"    Padding: {[p.sum().item() for p in padding_masks]}")
        
        # Forward through model
        print("\n5. Forward pass...")
        pred_logits, pred_masks, sem_logits = model(batch_voxels, batch_coords)
        
        print(f"\n  Model outputs:")
        print(f"    pred_logits: {pred_logits.shape}, range [{pred_logits.min():.2f}, {pred_logits.max():.2f}]")
        print(f"    pred_masks: {pred_masks.shape}, range [{pred_masks.min():.2f}, {pred_masks.max():.2f}]")
        print(f"    sem_logits: {sem_logits.shape}, range [{sem_logits.min():.2f}, {sem_logits.max():.2f}]")
        
        # Check dimensions
        assert pred_logits.shape[-1] == num_classes + 1, f"Wrong pred_logits classes: {pred_logits.shape[-1]} vs {num_classes + 1}"
        assert sem_logits.shape[-1] == num_classes, f"Wrong sem_logits classes: {sem_logits.shape[-1]} vs {num_classes}"
        
        # Prepare for loss
        outputs = {
            'pred_logits': pred_logits,
            'pred_masks': pred_masks,
            'aux_outputs': []
        }
        
        targets = {
            'classes': batch['masks_cls'],
            'masks': batch['masks']
        }
        
        print("\n6. Testing mask loss...")
        mask_loss = MaskLoss(cfg.LOSS, cfg[dataset])
        
        # Check target classes
        print("\n  Checking target classes:")
        for i, cls_list in enumerate(targets['classes']):
            if len(cls_list) > 0:
                cls_tensor = torch.stack(cls_list) if isinstance(cls_list[0], torch.Tensor) else torch.tensor(cls_list)
                print(f"    Sample {i}: {len(cls_list)} masks, classes: {cls_tensor}")
                if cls_tensor.max() >= num_classes:
                    print(f"    ERROR: Class {cls_tensor.max().item()} >= {num_classes}!")
        
        try:
            losses = mask_loss(outputs, targets, batch['masks_ids'], batch['pt_coord'])
            print("\n  Mask loss SUCCESS!")
            for k, v in losses.items():
                print(f"    {k}: {v.item():.4f}")
        except Exception as e:
            print(f"\n  Mask loss FAILED: {e}")
            import traceback
            traceback.print_exc()
            
            # Debug the exact failure point
            print("\n7. Debugging mask loss internals...")
            
            # Check if it's the matcher
            try:
                indices = mask_loss.matcher(outputs, targets)
                print("  Matcher OK, indices:", indices)
            except Exception as e2:
                print(f"  Matcher failed: {e2}")
                return
            
            # Check if it's loss_classes
            try:
                class_losses = mask_loss.loss_classes(outputs, targets, indices)
                print("  loss_classes OK")
            except Exception as e3:
                print(f"  loss_classes failed: {e3}")
                
                # Manual debug
                print("\n  Manual check of loss_classes:")
                # Get matched target classes
                target_classes_o = torch.cat([t[J] for t, (_, J) in zip(targets["classes"], indices)])
                print(f"    Matched target classes: {target_classes_o}")
                print(f"    Max target class: {target_classes_o.max() if len(target_classes_o) > 0 else 'N/A'}")
                
                # Check the pred_logits shape
                print(f"    pred_logits shape: {outputs['pred_logits'].shape}")
                print(f"    Expected: [batch_size, num_queries, {num_classes + 1}]")


if __name__ == "__main__":
    test_actual_pipeline()
