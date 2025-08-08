"""
Debug the Hungarian Matcher specifically
Save as: mask_pls/scripts/debug_matcher.py
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
from mask_pls.models.matcher import HungarianMatcher


def debug_matcher():
    """Debug the Hungarian Matcher"""
    print("Debugging Hungarian Matcher")
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
    
    # First, let's test with simple dummy data
    print("1. Testing with simple dummy data...")
    
    # Create matcher
    matcher = HungarianMatcher(cfg.LOSS.WEIGHTS, cfg.LOSS.P_RATIO)
    print(f"  Matcher weights: {cfg.LOSS.WEIGHTS}")
    print(f"  P_ratio: {cfg.LOSS.P_RATIO}")
    
    # Simple test
    batch_size = 1
    num_queries = 100
    num_points = 1000
    num_targets = 3
    
    outputs = {
        'pred_logits': torch.randn(batch_size, num_queries, num_classes + 1).cuda(),
        'pred_masks': torch.randn(batch_size, num_points, num_queries).cuda(),
    }
    
    # Create targets - note the structure!
    targets = {
        'classes': [torch.tensor([1, 5, 10]).cuda()],  # List of tensors
        'masks': [torch.rand(num_targets, num_points).cuda() > 0.5],  # List of mask tensors
    }
    
    print(f"\n  Dummy data shapes:")
    print(f"    pred_logits: {outputs['pred_logits'].shape}")
    print(f"    pred_masks: {outputs['pred_masks'].shape}")
    print(f"    target classes: {[t.shape for t in targets['classes']]}")
    print(f"    target masks: {[t.shape for t in targets['masks']]}")
    
    try:
        indices = matcher(outputs, targets)
        print("  Dummy data SUCCESS!")
        print(f"  Indices: {indices}")
    except Exception as e:
        print(f"  Dummy data FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Now test with real data
    print("\n2. Testing with real data...")
    
    # Get real data
    data = SemanticDatasetModule(cfg)
    data.setup()
    train_loader = data.train_dataloader()
    batch = next(iter(train_loader))
    
    # Check the format of real targets
    print("\n  Real data format:")
    print(f"    Batch size: {len(batch['masks_cls'])}")
    
    for i in range(len(batch['masks_cls'])):
        print(f"\n    Sample {i}:")
        print(f"      masks_cls type: {type(batch['masks_cls'][i])}")
        print(f"      masks_cls length: {len(batch['masks_cls'][i])}")
        if len(batch['masks_cls'][i]) > 0:
            print(f"      First class type: {type(batch['masks_cls'][i][0])}")
            if isinstance(batch['masks_cls'][i][0], torch.Tensor):
                print(f"      First class shape: {batch['masks_cls'][i][0].shape}")
                print(f"      First class value: {batch['masks_cls'][i][0]}")
        
        print(f"      masks type: {type(batch['masks'][i])}")
        print(f"      masks length: {len(batch['masks'][i])}")
        if len(batch['masks'][i]) > 0:
            print(f"      First mask type: {type(batch['masks'][i][0])}")
            print(f"      First mask shape: {batch['masks'][i][0].shape}")
            print(f"      Mask dtype: {batch['masks'][i][0].dtype}")
            print(f"      Mask device: {batch['masks'][i][0].device}")
    
    # Process through model to get predictions
    print("\n3. Getting model predictions...")
    model = MaskPLSSimplifiedONNX(cfg).cuda()
    model.eval()
    
    with torch.no_grad():
        # Quick processing (simplified)
        pts = torch.from_numpy(batch['pt_coord'][0]).float().cuda()
        feat = torch.from_numpy(batch['feats'][0]).float().cuda()
        
        # Take a subset for speed
        if len(pts) > 10000:
            pts = pts[:10000]
            feat = feat[:10000]
        
        # Simple normalization
        pts_norm = (pts - pts.min(0)[0]) / (pts.max(0)[0] - pts.min(0)[0])
        
        # Create dummy voxel grid
        voxel_grid = torch.randn(1, 4, 32, 32, 16).cuda()
        coords = pts_norm.unsqueeze(0)
        
        # Pad coordinates
        max_pts = 10000
        if coords.shape[1] < max_pts:
            coords = torch.nn.functional.pad(coords, (0, 0, 0, max_pts - coords.shape[1]))
        
        pred_logits, pred_masks, _ = model(voxel_grid, coords)
        
        # Adjust pred_masks to match number of points
        num_points = len(batch['pt_coord'][0])
        if pred_masks.shape[1] != num_points:
            print(f"  Adjusting pred_masks from {pred_masks.shape[1]} to {num_points} points")
            pred_masks = torch.nn.functional.interpolate(
                pred_masks.transpose(1, 2).unsqueeze(2),
                size=(1, num_points),
                mode='nearest'
            ).squeeze(2).transpose(1, 2)
    
    # Prepare for matcher
    outputs = {
        'pred_logits': pred_logits,
        'pred_masks': pred_masks,
    }
    
    # Fix the targets format
    print("\n4. Preparing targets for matcher...")
    
    # The issue might be that masks_cls contains individual tensors
    # Let's convert them to the right format
    fixed_targets = {
        'classes': [],
        'masks': []
    }
    
    for i in range(len(batch['masks_cls'])):
        if len(batch['masks_cls'][i]) > 0:
            # Convert list of scalar tensors to single tensor
            classes = []
            for c in batch['masks_cls'][i]:
                if isinstance(c, torch.Tensor):
                    classes.append(c.item() if c.numel() == 1 else c.cpu().numpy())
                else:
                    classes.append(c)
            
            # Create tensor and move to GPU
            classes_tensor = torch.tensor(classes, dtype=torch.long).cuda()
            fixed_targets['classes'].append(classes_tensor)
            
            # Stack masks and move to GPU
            masks_list = []
            for m in batch['masks'][i]:
                if isinstance(m, torch.Tensor):
                    masks_list.append(m.float())
                else:
                    masks_list.append(torch.tensor(m, dtype=torch.float32))
            
            if len(masks_list) > 0:
                masks_tensor = torch.stack(masks_list).cuda()
                fixed_targets['masks'].append(masks_tensor)
            else:
                fixed_targets['masks'].append(torch.empty(0, num_points).cuda())
        else:
            # Empty sample
            fixed_targets['classes'].append(torch.empty(0, dtype=torch.long).cuda())
            fixed_targets['masks'].append(torch.empty(0, num_points).cuda())
    
    print(f"  Fixed targets:")
    print(f"    classes: {[t.shape for t in fixed_targets['classes']]}")
    print(f"    masks: {[t.shape for t in fixed_targets['masks']]}")
    
    # Test matcher with fixed targets
    print("\n5. Testing matcher with fixed targets...")
    try:
        indices = matcher(outputs, fixed_targets)
        print("  SUCCESS!")
        for i, (src, tgt) in enumerate(indices):
            print(f"    Batch {i}: matched {len(src)} queries to {len(tgt)} targets")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to debug inside matcher
        print("\n6. Debugging inside matcher...")
        
        # Check the cost computation
        print("  Testing cost computation...")
        out_prob = outputs["pred_logits"][0].softmax(-1)
        tgt_ids = fixed_targets["classes"][0]
        
        print(f"    out_prob shape: {out_prob.shape}")
        print(f"    tgt_ids: {tgt_ids}")
        
        if len(tgt_ids) > 0:
            try:
                cost_class = -out_prob[:, tgt_ids]
                print(f"    cost_class shape: {cost_class.shape}")
            except Exception as e2:
                print(f"    cost_class computation failed: {e2}")


if __name__ == "__main__":
    debug_matcher()
