"""
Test script to debug DGCNN setup and shape issues
"""

import torch
import numpy as np
import yaml
from pathlib import Path
from easydict import EasyDict as edict

# Add the parent directory to the path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from mask_pls.datasets.semantic_dataset import SemanticDatasetModule
from mask_pls.models.dgcnn.maskpls_dgcnn import MaskPLSDGCNN


def test_data_shapes():
    """Test the shapes of data coming from the dataset"""
    print("="*60)
    print("Testing Data Shapes")
    print("="*60)
    
    # Load config
    cfg = get_test_config()
    
    # Create data module
    data = SemanticDatasetModule(cfg)
    data.setup()
    
    # Get a single batch
    train_loader = data.train_dataloader()
    batch = next(iter(train_loader))
    
    print("\nBatch contents:")
    for key, value in batch.items():
        if isinstance(value, list):
            print(f"  {key}: list of {len(value)} items")
            if len(value) > 0:
                if isinstance(value[0], np.ndarray):
                    print(f"    First item shape: {value[0].shape}")
                    print(f"    First item dtype: {value[0].dtype}")
                elif isinstance(value[0], torch.Tensor):
                    print(f"    First item shape: {value[0].shape}")
                    print(f"    First item dtype: {value[0].dtype}")
        else:
            print(f"  {key}: {type(value)}")
    
    # Specifically check sem_label
    print("\nSemantic labels detail:")
    for i, sem_label in enumerate(batch['sem_label']):
        print(f"  Sample {i}:")
        print(f"    Type: {type(sem_label)}")
        print(f"    Shape: {sem_label.shape}")
        print(f"    Dtype: {sem_label.dtype}")
        print(f"    Min/Max: {sem_label.min()}/{sem_label.max()}")
        print(f"    Unique values: {np.unique(sem_label)[:10]}...")  # First 10 unique values
    
    return batch


def test_model_forward():
    """Test model forward pass with real data"""
    print("\n" + "="*60)
    print("Testing Model Forward Pass")
    print("="*60)
    
    # Get config and data
    cfg = get_test_config()
    cfg.PRETRAINED_PATH = None  # Skip pretrained for this test
    
    # Create model
    model = MaskPLSDGCNN(cfg)
    model.cuda()
    model.eval()
    
    # Get a batch
    data = SemanticDatasetModule(cfg)
    data.setup()
    model.things_ids = data.things_ids
    
    train_loader = data.train_dataloader()
    batch = next(iter(train_loader))
    
    print("\nRunning forward pass...")
    try:
        with torch.no_grad():
            outputs, padding, sem_logits = model(batch)
        print("✓ Forward pass successful")
        
        print(f"\nOutputs shapes:")
        print(f"  pred_logits: {outputs['pred_logits'].shape}")
        print(f"  pred_masks: {outputs['pred_masks'].shape}")
        print(f"  padding: {padding.shape}")
        print(f"  sem_logits: {sem_logits.shape}")
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, batch
    
    return outputs, padding, sem_logits, batch


def test_loss_computation():
    """Test loss computation with detailed debugging"""
    print("\n" + "="*60)
    print("Testing Loss Computation")
    print("="*60)
    
    cfg = get_test_config()
    cfg.PRETRAINED_PATH = None
    
    # Create model
    model = MaskPLSDGCNN(cfg)
    model.cuda()
    model.eval()
    
    # Get data
    outputs, padding, sem_logits, batch = test_model_forward()
    if outputs is None:
        return
    
    print("\nTesting loss computation with debugging...")
    
    # Add detailed debugging to get_losses
    try:
        # Manually run through the loss computation with prints
        print("\n--- Semantic Label Processing ---")
        sem_labels = []
        batch_size = len(batch['sem_label'])
        
        for i in range(batch_size):
            labels = batch['sem_label'][i]
            print(f"\nSample {i}:")
            print(f"  Original type: {type(labels)}")
            print(f"  Original shape: {labels.shape}")
            print(f"  Original dtype: {labels.dtype}")
            
            if isinstance(labels, np.ndarray):
                labels = torch.from_numpy(labels).long().cuda()
                print(f"  After torch conversion: {labels.shape}, {labels.dtype}")
            
            # Handle label shape
            print(f"  Dimensions: {labels.dim()}")
            if labels.dim() == 2:
                print(f"  2D tensor with shape {labels.shape}")
                if labels.shape[1] == 1:
                    labels = labels.squeeze(1)
                    print(f"  After squeeze(1): {labels.shape}")
                else:
                    labels = labels.reshape(-1)
                    print(f"  After reshape(-1): {labels.shape}")
            elif labels.dim() > 2:
                print(f"  >2D tensor with shape {labels.shape}")
                labels = labels.reshape(-1)
                print(f"  After reshape(-1): {labels.shape}")
            
            # Get valid mask for this sample
            valid_mask = ~padding[i]
            print(f"  Valid mask shape: {valid_mask.shape}")
            print(f"  Valid mask sum: {valid_mask.sum()}")
            
            # Apply valid mask
            if valid_mask.sum() > 0:
                min_len = min(labels.shape[0], valid_mask.shape[0])
                print(f"  Min length: {min_len}")
                valid_labels = labels[:min_len][valid_mask[:min_len]]
                print(f"  Valid labels shape: {valid_labels.shape}")
                print(f"  Valid labels sample: {valid_labels[:10]}")
                sem_labels.append(valid_labels)
        
        if sem_labels:
            sem_labels = torch.cat(sem_labels)
            print(f"\nConcatenated sem_labels shape: {sem_labels.shape}")
            print(f"Concatenated sem_labels dim: {sem_labels.dim()}")
            
            # Get valid semantic logits
            sem_logits_valid = []
            for i in range(batch_size):
                valid_mask = ~padding[i]
                if valid_mask.sum() > 0:
                    sem_logits_valid.append(sem_logits[i][valid_mask])
            
            if sem_logits_valid:
                sem_logits_valid = torch.cat(sem_logits_valid, dim=0)
                print(f"\nsem_logits_valid shape: {sem_logits_valid.shape}")
                print(f"sem_logits_valid dim: {sem_logits_valid.dim()}")
                
                # Now try the actual loss computation
                print("\nComputing semantic loss...")
                loss_sem = model.sem_loss(sem_logits_valid, sem_labels)
                print("✓ Semantic loss computed successfully")
                print(f"  sem_ce: {loss_sem['sem_ce'].item()}")
                print(f"  sem_lov: {loss_sem['sem_lov'].item()}")
    
    except Exception as e:
        print(f"\n✗ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()


def get_test_config():
    """Get a minimal test configuration"""
    cfg = edict({
        'MODEL': {
            'DATASET': 'KITTI'
        },
        'KITTI': {
            'PATH': 'data/kitti',
            'CONFIG': 'mask_pls/datasets/semantic-kitti.yaml',
            'NUM_CLASSES': 20,
            'IGNORE_LABEL': 0,
            'MIN_POINTS': 10,
            'SPACE': [[-48.0, 48.0], [-48.0, 48.0], [-4.0, 1.5]],
            'SUB_NUM_POINTS': 80000
        },
        'BACKBONE': {
            'INPUT_DIM': 4,
            'CHANNELS': [32, 32, 64, 128, 256, 256, 128, 96, 96],
            'RESOLUTION': 0.05,
            'KNN_UP': 3
        },
        'DECODER': {
            'HIDDEN_DIM': 256,
            'NHEADS': 8,
            'DIM_FFN': 1024,
            'FEATURE_LEVELS': 3,
            'DEC_BLOCKS': 3,
            'NUM_QUERIES': 100,
            'POS_ENC': {
                'MAX_FREQ': 10000,
                'DIMENSIONALITY': 3,
                'BASE': 2
            }
        },
        'LOSS': {
            'WEIGHTS_KEYS': ['loss_ce', 'loss_dice', 'loss_mask'],
            'WEIGHTS': [2.0, 5.0, 5.0],
            'EOS_COEF': 0.1,
            'NUM_POINTS': 50000,
            'NUM_MASK_PTS': 500,
            'P_RATIO': 0.4,
            'SEM': {
                'WEIGHTS': [2, 6]
            }
        },
        'TRAIN': {
            'BATCH_SIZE': 2,
            'NUM_WORKERS': 0,  # Set to 0 for debugging
            'SUBSAMPLE': True,
            'AUG': False,  # Disable augmentation for testing
            'LR': 0.001,
            'MAX_EPOCH': 100,
            'WARMUP_STEPS': 1000
        }
    })
    
    return cfg


if __name__ == "__main__":
    # Run tests
    print("Starting DGCNN Setup Tests\n")
    
    # Test 1: Check data shapes
    batch = test_data_shapes()
    
    # Test 2: Check model forward pass
    test_model_forward()
    
    # Test 3: Test loss computation with debugging
    test_loss_computation()
    
    print("\n" + "="*60)
    print("Tests completed!")