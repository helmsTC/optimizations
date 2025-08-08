"""
Minimal test to find CUDA error
Save as: mask_pls/scripts/test_minimal.py
"""
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

import torch
import torch.nn.functional as F
from mask_pls.models.loss import MaskLoss
from mask_pls.models.matcher import HungarianMatcher
from easydict import EasyDict as edict

def test_original_loss():
    """Test the original MaskLoss to see where it fails"""
    print("Testing Original MaskLoss")
    print("-" * 60)
    
    # Configuration
    num_classes = 20  # KITTI has 20 classes
    batch_size = 1
    num_queries = 100
    num_points = 1000
    
    # Create loss configuration
    cfg_loss = edict({
        'WEIGHTS': [2.0, 5.0, 5.0],
        'WEIGHTS_KEYS': ['loss_ce', 'loss_dice', 'loss_mask'],
        'EOS_COEF': 0.1,
        'NUM_POINTS': 50000,
        'NUM_MASK_PTS': 500,
        'P_RATIO': 0.4
    })
    
    data_cfg = edict({
        'NUM_CLASSES': num_classes,
        'IGNORE_LABEL': 0
    })
    
    # Test 1: Check what the matcher expects
    print("\n1. Testing Matcher:")
    matcher = HungarianMatcher(cfg_loss.WEIGHTS, cfg_loss.P_RATIO)
    
    # Create outputs with correct shape
    outputs = {
        'pred_logits': torch.randn(batch_size, num_queries, num_classes + 1).cuda(),
        'pred_masks': torch.randn(batch_size, num_points, num_queries).cuda(),
    }
    
    # Create targets - let's make 3 objects
    targets = {
        'classes': [torch.tensor([1, 5, 10]).cuda()],  # Classes in range [0, 19]
        'masks': [torch.rand(3, num_points).cuda() > 0.5],  # 3 binary masks
    }
    
    print(f"Output shapes:")
    print(f"  pred_logits: {outputs['pred_logits'].shape} (expecting {num_classes + 1} classes)")
    print(f"  pred_masks: {outputs['pred_masks'].shape}")
    print(f"Target shapes:")
    print(f"  classes: {targets['classes'][0]} (values in range 0-{num_classes-1})")
    print(f"  masks: {targets['masks'][0].shape}")
    
    try:
        indices = matcher(outputs, targets)
        print(f"Matcher success! Indices: {indices}")
    except Exception as e:
        print(f"Matcher failed: {e}")
        return
    
    # Test 2: Check the loss computation
    print("\n2. Testing Loss Computation:")
    
    # Create the loss module
    loss_module = MaskLoss(cfg_loss, data_cfg)
    
    # Create dummy mask IDs
    masks_ids = [[torch.where(mask)[0] for mask in targets['masks'][0]]]
    pt_coord = [torch.randn(num_points, 3).numpy()]
    
    try:
        losses = loss_module(outputs, targets, masks_ids, pt_coord)
        print("Loss computation success!")
        for k, v in losses.items():
            print(f"  {k}: {v.item():.4f}")
    except Exception as e:
        print(f"Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to pinpoint the exact failure
        print("\n3. Testing loss_classes directly:")
        try:
            # Call loss_classes directly
            class_losses = loss_module.loss_classes(outputs, targets, indices)
            print("loss_classes success!")
        except Exception as e2:
            print(f"loss_classes failed: {e2}")
            
            # Let's manually check what's happening
            print("\n4. Manual debugging of loss_classes:")
            
            # Get the permutation indices
            batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
            src_idx = torch.cat([src for (src, _) in indices])
            print(f"Permutation indices:")
            print(f"  batch_idx: {batch_idx}")
            print(f"  src_idx: {src_idx}")
            
            # Get target classes
            target_classes_o = torch.cat([t[J] for t, (_, J) in zip(targets["classes"], indices)])
            print(f"Matched target classes: {target_classes_o}")
            
            # Create target tensor
            target_classes = torch.full(
                outputs['pred_logits'].shape[:2],
                num_classes,  # This should be 20
                dtype=torch.int64,
                device=outputs['pred_logits'].device,
            )
            print(f"Target classes initialized to: {num_classes}")
            print(f"Target classes shape: {target_classes.shape}")
            
            # Assign matched classes
            if len(batch_idx) > 0:
                target_classes[batch_idx, src_idx] = target_classes_o
            
            print(f"Target classes after assignment:")
            print(f"  Min: {target_classes.min()}, Max: {target_classes.max()}")
            print(f"  Unique values: {torch.unique(target_classes).cpu().numpy()}")
            
            # Check if this will work with cross_entropy
            print(f"\n5. Testing cross_entropy directly:")
            print(f"  Logits shape: {outputs['pred_logits'].shape}")
            print(f"  Targets shape: {target_classes.shape}")
            print(f"  Targets range: [{target_classes.min()}, {target_classes.max()}]")
            
            try:
                # The error likely happens here
                loss_ce = F.cross_entropy(
                    outputs['pred_logits'].transpose(1, 2),
                    target_classes,
                    ignore_index=data_cfg.IGNORE_LABEL
                )
                print(f"  Cross entropy success! Loss: {loss_ce.item():.4f}")
            except Exception as e3:
                print(f"  Cross entropy failed: {e3}")
                torch.cuda.synchronize()


if __name__ == "__main__":
    test_original_loss()
