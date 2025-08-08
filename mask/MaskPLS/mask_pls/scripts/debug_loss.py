"""
Debug version of training script to find CUDA error
Save as: mask/MaskPLS/mask_pls/scripts/debug_cuda_error.py
"""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

import torch
import numpy as np
from mask_pls.models.loss import MaskLoss, SemLoss

def debug_mask_loss():
    """Test mask loss in isolation"""
    print("\n=== Testing MaskLoss ===")
    
    # Create dummy data
    batch_size = 1
    num_queries = 100
    num_points = 1000
    num_classes = 20
    
    # Create loss function
    from easydict import EasyDict as edict
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
    
    mask_loss = MaskLoss(cfg_loss, data_cfg)
    
    # Create dummy outputs
    outputs = {
        'pred_logits': torch.randn(batch_size, num_queries, num_classes + 1).cuda(),
        'pred_masks': torch.randn(batch_size, num_points, num_queries).cuda(),
        'aux_outputs': []
    }
    
    # Create dummy targets
    # Let's create 5 masks with different classes
    num_masks = 5
    targets = {
        'classes': [torch.randint(0, num_classes, (num_masks,)).cuda()],  # Random classes 0-19
        'masks': [torch.randint(0, 2, (num_masks, num_points)).float().cuda()]  # Binary masks
    }
    
    # Create dummy mask IDs (indices of points in each mask)
    masks_ids = [[torch.where(mask)[0] for mask in targets['masks'][0]]]
    
    # Dummy coordinates
    pt_coord = [np.random.randn(num_points, 3)]
    
    print("Input shapes:")
    print(f"  pred_logits: {outputs['pred_logits'].shape}")
    print(f"  pred_masks: {outputs['pred_masks'].shape}")
    print(f"  target classes: {targets['classes'][0]}")
    print(f"  target masks: {targets['masks'][0].shape}")
    
    try:
        # Test loss computation
        losses = mask_loss(outputs, targets, masks_ids, pt_coord)
        print("\nSuccess! Losses computed:")
        for k, v in losses.items():
            print(f"  {k}: {v.item():.4f}")
    except Exception as e:
        print(f"\nError in mask loss: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to sync CUDA to get more info
        torch.cuda.synchronize()


def debug_sem_loss():
    """Test semantic loss in isolation"""
    print("\n=== Testing SemLoss ===")
    
    num_points = 1000
    num_classes = 20
    
    sem_loss = SemLoss([2, 6])  # CE weight, Lovasz weight
    
    # Create dummy data
    logits = torch.randn(num_points, num_classes).cuda()
    
    # Test different label values
    for test_name, labels in [
        ("Valid labels (0-19)", torch.randint(0, num_classes, (num_points,)).cuda()),
        ("With ignore class 0", torch.cat([torch.zeros(100), torch.randint(1, num_classes, (num_points-100,))]).long().cuda()),
        ("All zeros", torch.zeros(num_points).long().cuda()),
    ]:
        print(f"\nTesting: {test_name}")
        print(f"  Labels range: [{labels.min().item()}, {labels.max().item()}]")
        print(f"  Unique labels: {torch.unique(labels).cpu().numpy()}")
        
        try:
            losses = sem_loss(logits, labels)
            print("  Success! Losses:")
            for k, v in losses.items():
                print(f"    {k}: {v.item():.4f}")
        except Exception as e:
            print(f"  Error: {type(e).__name__}: {e}")
            torch.cuda.synchronize()


def check_data_ranges():
    """Check actual data from the dataset"""
    print("\n=== Checking Dataset ===")
    
    from mask_pls.datasets.semantic_dataset import SemanticDatasetModule
    import yaml
    from easydict import EasyDict as edict
    from os.path import join, dirname, abspath
    
    # Load configs
    base_dir = dirname(dirname(abspath(__file__)))
    model_cfg = edict(yaml.safe_load(open(join(base_dir, "config/model.yaml"))))
    backbone_cfg = edict(yaml.safe_load(open(join(base_dir, "config/backbone.yaml"))))
    decoder_cfg = edict(yaml.safe_load(open(join(base_dir, "config/decoder.yaml"))))
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    
    # Create data module
    data = SemanticDatasetModule(cfg)
    data.setup()
    
    # Get one batch
    train_loader = data.train_dataloader()
    batch = next(iter(train_loader))
    
    print("\nBatch contents:")
    print(f"  Number of samples: {len(batch['pt_coord'])}")
    
    for i in range(len(batch['pt_coord'])):
        print(f"\n  Sample {i}:")
        print(f"    Points shape: {batch['pt_coord'][i].shape}")
        print(f"    Sem labels shape: {batch['sem_label'][i].shape}")
        print(f"    Sem labels range: [{batch['sem_label'][i].min()}, {batch['sem_label'][i].max()}]")
        print(f"    Unique sem labels: {np.unique(batch['sem_label'][i])}")
        
        if len(batch['masks_cls'][i]) > 0:
            mask_classes = torch.stack(batch['masks_cls'][i]) if isinstance(batch['masks_cls'][i][0], torch.Tensor) else torch.tensor(batch['masks_cls'][i])
            print(f"    Mask classes: {mask_classes}")
            if mask_classes.max() >= cfg[cfg.MODEL.DATASET].NUM_CLASSES:
                print(f"    ERROR: Mask class {mask_classes.max().item()} >= {cfg[cfg.MODEL.DATASET].NUM_CLASSES}")


def test_cross_entropy():
    """Test cross entropy with different configurations"""
    print("\n=== Testing Cross Entropy ===")
    
    # Test 1: Basic test
    print("\nTest 1: Basic cross entropy")
    logits = torch.randn(1, 21, 100).cuda()  # [batch, classes, queries]
    targets = torch.randint(0, 21, (1, 100)).cuda()  # Values 0-20
    
    try:
        loss = torch.nn.functional.cross_entropy(logits, targets)
        print(f"  Success! Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 2: With ignore index
    print("\nTest 2: With ignore_index=0")
    try:
        loss = torch.nn.functional.cross_entropy(logits, targets, ignore_index=0)
        print(f"  Success! Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 3: With out of bounds target
    print("\nTest 3: Out of bounds target (21 for 21 classes)")
    targets_bad = torch.full((1, 100), 21).cuda()
    try:
        loss = torch.nn.functional.cross_entropy(logits, targets_bad)
        print(f"  Success! Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"  Error: {e}")
        torch.cuda.synchronize()


if __name__ == "__main__":
    print("CUDA Debugging Script")
    print("=" * 60)
    
    # Test each component
    test_cross_entropy()
    debug_sem_loss()
    debug_mask_loss()
    check_data_ranges()
    
    print("\n" + "=" * 60)
    print("Debugging complete!")
