"""
Debug the mask loss computation in detail
Save as: mask_pls/scripts/debug_mask_loss.py
"""
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

import torch
import yaml
import numpy as np
from easydict import EasyDict as edict
from os.path import join, dirname, abspath

from mask_pls.models.onnx.simplified_model import MaskPLSSimplifiedONNX
from mask_pls.datasets.semantic_dataset import SemanticDatasetModule
from mask_pls.models.loss import MaskLoss
from mask_pls.models.matcher import HungarianMatcher


def debug_mask_loss_detailed():
    """Debug mask loss computation step by step"""
    print("Debugging Mask Loss in Detail")
    print("=" * 60)
    
    # Load configs
    base_dir = dirname(dirname(abspath(__file__)))
    model_cfg = edict(yaml.safe_load(open(join(base_dir, "config/model.yaml"))))
    backbone_cfg = edict(yaml.safe_load(open(join(base_dir, "config/backbone.yaml"))))
    decoder_cfg = edict(yaml.safe_load(open(join(base_dir, "config/decoder.yaml"))))
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    
    cfg.TRAIN.BATCH_SIZE = 1
    cfg.TRAIN.NUM_WORKERS = 0
    dataset = cfg.MODEL.DATASET
    num_classes = cfg[dataset].NUM_CLASSES
    
    # Get real data
    print("1. Loading real data...")
    data = SemanticDatasetModule(cfg)
    data.setup()
    train_loader = data.train_dataloader()
    batch = next(iter(train_loader))
    
    # Process through model
    print("\n2. Processing through model...")
    model = MaskPLSSimplifiedONNX(cfg).cuda()
    model.eval()
    
    with torch.no_grad():
        # Simplified preprocessing (just get the outputs)
        batch_voxels = []
        batch_coords = []
        valid_indices = []
        
        for i in range(len(batch['pt_coord'])):
            pts = torch.from_numpy(batch['pt_coord'][i]).float().cuda()
            feat = torch.from_numpy(batch['feats'][i]).float().cuda()
            
            bounds = cfg[dataset].SPACE
            valid_mask = torch.ones(pts.shape[0], dtype=torch.bool, device=pts.device)
            for dim in range(3):
                valid_mask &= (pts[:, dim] >= bounds[dim][0]) & (pts[:, dim] < bounds[dim][1])
            
            valid_idx = torch.where(valid_mask)[0]
            valid_pts = pts[valid_mask]
            valid_feat = feat[valid_mask]
            
            max_pts = cfg[dataset].SUB_NUM_POINTS
            if len(valid_pts) > max_pts:
                perm = torch.randperm(len(valid_pts))[:max_pts]
                valid_pts = valid_pts[perm]
                valid_feat = valid_feat[perm]
                valid_idx = valid_idx[perm]
            
            norm_coords = torch.zeros_like(valid_pts)
            for dim in range(3):
                norm_coords[:, dim] = (valid_pts[:, dim] - bounds[dim][0]) / (bounds[dim][1] - bounds[dim][0])
            
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
        
        # Forward
        pred_logits, pred_masks, sem_logits = model(batch_voxels, batch_coords)
        
        print(f"Model outputs:")
        print(f"  pred_logits: {pred_logits.shape}")
        print(f"  pred_masks: {pred_masks.shape}")
        print(f"  padding mask sum: {padding_masks[0].sum()}")
    
    # Prepare loss inputs
    outputs = {
        'pred_logits': pred_logits,
        'pred_masks': pred_masks,
        'aux_outputs': []
    }
    
    targets = {
        'classes': batch['masks_cls'],
        'masks': batch['masks']
    }
    
    print(f"\n3. Target information:")
    for i, (cls_list, mask_list) in enumerate(zip(targets['classes'], targets['masks'])):
        print(f"  Sample {i}:")
        print(f"    Number of masks: {len(cls_list)}")
        if len(cls_list) > 0:
            # Extract class values
            cls_values = []
            for c in cls_list:
                if isinstance(c, torch.Tensor):
                    cls_values.append(c.item() if c.numel() == 1 else c)
                else:
                    cls_values.append(c)
            print(f"    Classes: {cls_values}")
            print(f"    Mask shapes: {[m.shape for m in mask_list]}")
    
    # Create mask loss
    print("\n4. Creating mask loss...")
    mask_loss = MaskLoss(cfg.LOSS, cfg[dataset])
    
    # Debug the matcher first
    print("\n5. Testing matcher...")
    matcher = mask_loss.matcher
    
    try:
        indices = matcher(outputs, targets)
        print("  Matcher success!")
        for i, (src_idx, tgt_idx) in enumerate(indices):
            print(f"    Batch {i}: matched {len(src_idx)} queries to {len(tgt_idx)} targets")
            if len(src_idx) > 0:
                print(f"      Query indices: {src_idx[:5]}... (showing first 5)")
                print(f"      Target indices: {tgt_idx[:5]}... (showing first 5)")
    except Exception as e:
        print(f"  Matcher failed: {e}")
        return
    
    # Now debug loss_classes
    print("\n6. Testing loss_classes...")
    
    # Get the permutation indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    
    print(f"  Permutation indices:")
    print(f"    batch_idx shape: {batch_idx.shape}, values: {batch_idx[:10] if len(batch_idx) > 0 else 'empty'}")
    print(f"    src_idx shape: {src_idx.shape}, values: {src_idx[:10] if len(src_idx) > 0 else 'empty'}")
    
    # Get matched target classes
    target_classes_list = []
    for i, (t_list, (_, J)) in enumerate(zip(targets["classes"], indices)):
        print(f"\n  Processing batch {i}:")
        print(f"    Target indices J: {J}")
        print(f"    Number of targets: {len(t_list)}")
        
        if len(J) > 0:
            for j in J:
                if j < len(t_list):
                    cls = t_list[j]
                    if isinstance(cls, torch.Tensor):
                        cls_val = cls.item() if cls.numel() == 1 else cls
                    else:
                        cls_val = cls
                    target_classes_list.append(cls_val)
                    print(f"      Target {j}: class {cls_val}")
                else:
                    print(f"      ERROR: Target index {j} >= number of targets {len(t_list)}")
    
    if len(target_classes_list) > 0:
        target_classes_o = torch.tensor(target_classes_list, device=pred_logits.device)
        print(f"\n  Matched target classes: {target_classes_o}")
        print(f"  Max class: {target_classes_o.max()}")
        print(f"  Min class: {target_classes_o.min()}")
    else:
        print("\n  No matched targets!")
        target_classes_o = torch.tensor([], device=pred_logits.device, dtype=torch.int64)
    
    # Create target tensor
    print(f"\n7. Creating target tensor for cross_entropy...")
    target_classes = torch.full(
        outputs['pred_logits'].shape[:2],
        num_classes,  # no-object class
        dtype=torch.int64,
        device=outputs['pred_logits'].device,
    )
    print(f"  Initialized target_classes shape: {target_classes.shape}")
    print(f"  Initialized with no-object class: {num_classes}")
    
    # Assign matched classes
    if len(batch_idx) > 0:
        print(f"\n  Assigning {len(batch_idx)} matched targets...")
        print(f"  batch_idx max: {batch_idx.max()}, min: {batch_idx.min()}")
        print(f"  src_idx max: {src_idx.max()}, min: {src_idx.min()}")
        print(f"  target_classes_o: {target_classes_o}")
        
        # Check if indices are valid
        if batch_idx.max() >= target_classes.shape[0]:
            print(f"  ERROR: batch_idx max {batch_idx.max()} >= batch size {target_classes.shape[0]}")
        if src_idx.max() >= target_classes.shape[1]:
            print(f"  ERROR: src_idx max {src_idx.max()} >= num queries {target_classes.shape[1]}")
        
        try:
            target_classes[batch_idx, src_idx] = target_classes_o
            print("  Assignment successful!")
        except Exception as e:
            print(f"  Assignment failed: {e}")
            return
    
    print(f"\n  Final target_classes:")
    print(f"    Shape: {target_classes.shape}")
    print(f"    Min: {target_classes.min()}, Max: {target_classes.max()}")
    print(f"    Unique values: {torch.unique(target_classes).cpu().numpy()}")
    
    # Test cross_entropy
    print("\n8. Testing cross_entropy...")
    print(f"  pred_logits shape: {outputs['pred_logits'].shape}")
    print(f"  target_classes shape: {target_classes.shape}")
    
    try:
        # Get weights
        weights = mask_loss.weights.to(outputs['pred_logits'].device)
        print(f"  Weights shape: {weights.shape}")
        print(f"  Weights: {weights}")
        
        # Compute loss
        loss_ce = torch.nn.functional.cross_entropy(
            outputs['pred_logits'].transpose(1, 2),
            target_classes,
            weights,
            ignore_index=mask_loss.ignore
        )
        print(f"  Cross entropy SUCCESS! Loss: {loss_ce.item():.4f}")
    except Exception as e:
        print(f"  Cross entropy FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        # Additional debugging
        print("\n  Additional debugging:")
        print(f"  ignore_index: {mask_loss.ignore}")
        print(f"  num_classes from config: {mask_loss.num_classes}")
        print(f"  pred_logits last dim: {outputs['pred_logits'].shape[-1]}")
        print(f"  Expected: {mask_loss.num_classes + 1}")


if __name__ == "__main__":
    debug_mask_loss_detailed()
