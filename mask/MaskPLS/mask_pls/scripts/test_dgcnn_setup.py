# test_dgcnn_setup.py
"""
Test script to verify DGCNN setup and data loading
"""

import torch
import numpy as np
from pathlib import Path
import sys
import yaml
from easydict import EasyDict as edict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from mask_pls.datasets.semantic_dataset import SemanticDatasetModule
from mask_pls.models.dgcnn.dgcnn_backbone import DGCNNBackbone
from mask_pls.models.dgcnn.maskpls_dgcnn import MaskPLSDGCNN


def test_data_loading():
    """Test if data module loads correctly"""
    print("Testing data loading...")
    
    # Load config
    from mask_pls.scripts.train_dgcnn import get_config
    cfg = get_config()
    
    # Create data module
    data = SemanticDatasetModule(cfg)
    data.setup()
    
    # Get a sample batch
    train_loader = data.train_dataloader()
    batch = next(iter(train_loader))
    
    print(f"✓ Data loaded successfully")
    print(f"  Batch keys: {list(batch.keys())}")
    print(f"  Batch size: {len(batch['pt_coord'])}")
    print(f"  Points shape: {batch['pt_coord'][0].shape}")
    print(f"  Features shape: {batch['feats'][0].shape}")
    
    return batch, cfg


def test_backbone(batch, cfg):
    """Test DGCNN backbone"""
    print("\nTesting DGCNN backbone...")
    
    # Create backbone
    backbone = DGCNNBackbone(cfg.BACKBONE)
    backbone.eval()
    
    # Test forward pass
    with torch.no_grad():
        try:
            feats, coords, masks, sem_logits = backbone(batch)
            print(f"✓ Backbone forward pass successful")
            print(f"  Number of feature levels: {len(feats)}")
            for i, feat in enumerate(feats):
                print(f"  Level {i} shape: {feat.shape}")
            print(f"  Semantic logits shape: {sem_logits.shape}")
        except Exception as e:
            print(f"✗ Backbone forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    return feats, coords, masks, sem_logits


def test_full_model(batch, cfg):
    """Test full MaskPLS-DGCNN model"""
    print("\nTesting full model...")
    
    # Create model
    model = MaskPLSDGCNN(cfg)
    model.eval()
    
    # Set things_ids (normally done by data module)
    model.things_ids = [1, 2, 3, 4, 5, 6, 7, 8]  # KITTI things
    
    # Test forward pass
    with torch.no_grad():
        try:
            outputs, padding, sem_logits = model(batch)
            print(f"✓ Full model forward pass successful")
            print(f"  Outputs keys: {list(outputs.keys())}")
            print(f"  Pred logits shape: {outputs['pred_logits'].shape}")
            print(f"  Pred masks shape: {outputs['pred_masks'].shape}")
            print(f"  Semantic logits shape: {sem_logits.shape}")
        except Exception as e:
            print(f"✗ Full model forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # Test loss computation
    try:
        losses = model.get_losses(batch, outputs, padding, sem_logits)
        print(f"\n✓ Loss computation successful")
        for k, v in losses.items():
            print(f"  {k}: {v.item():.4f}")
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
    
    return outputs


def test_memory_usage():
    """Test memory usage"""
    print("\nChecking memory usage...")
    
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"  Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    else:
        print("  Running on CPU")


def main():
    """Run all tests"""
    print("="*60)
    print("Testing DGCNN Setup for MaskPLS")
    print("="*60)
    
    # Test data loading
    batch, cfg = test_data_loading()
    
    # Test backbone
    backbone_outputs = test_backbone(batch, cfg)
    
    # Test full model
    if backbone_outputs is not None:
        model_outputs = test_full_model(batch, cfg)
    
    # Check memory
    test_memory_usage()
    
    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60)


if __name__ == "__main__":
    main()