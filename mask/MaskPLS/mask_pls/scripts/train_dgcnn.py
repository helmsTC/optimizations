
helmsTC/optimizations
main

GITHUB

we are actively working on a conversion of the network found in the original/ directory to work with onnx. we ran into the issue of not having a pretrained backbone to replace the functionality that minkowskiengine was delivering. currently train_dgcnn.py is being used with models/dgcnn. please help me with this conversion. when running train im currently getting: error in line 132 in train_dgcnn: attributeerror EasyDict object has no attribute SUBSAMPLE

Looking at the error, it seems the configuration is missing the SUBSAMPLE attribute under the TRAIN section. Let me help you fix this issue and provide a comprehensive solution for the DGCNN conversion.

Here's the fix for the immediate error and improvements to the training script:

S


str_replace_editor
Request

{
  `command`: `str_replace`,
  `path`: `mask/MaskPLS/mask_pls/scripts/train_dgcnn.py`,
  `old_str`: `def get_config():
    \"\"\"Load and merge configuration files\"\"\"
    def getDir(obj):
        return os.path.dirname(os.path.abspath(obj))
    
    model_cfg = edict(yaml.safe_load(open(join(getDir(__file__), \"../config/model.yaml\"))))
    backbone_cfg = edict(yaml.safe_load(open(join(getDir(__file__), \"../config/backbone.yaml\"))))
    decoder_cfg = edict(yaml.safe_load(open(join(getDir(__file__), \"../config/decoder.yaml\"))))
    
    # DGCNN-specific configuration
    dgcnn_cfg = edict({
        'AUGMENTATION': {
            'rotation': True,
            'scaling': True,
            'flipping': True,
            'jittering': True,
            'dropout': True
        },
        'PRETRAINED_PATH': None,  # Path to pre-trained DGCNN weights
        'TRAIN': {
            'WARMUP_STEPS': 1000,
            'GRADIENT_CLIP': 1.0,
            'MIXED_PRECISION': True
        }
    })
    
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg, **dgcnn_cfg})
    return cfg`,
  `new_str`: `def get_config():
    \"\"\"Load and merge configuration files\"\"\"
    def getDir(obj):
        return os.path.dirname(os.path.abspath(obj))
    
    model_cfg = edict(yaml.safe_load(open(join(getDir(__file__), \"../config/model.yaml\"))))
    backbone_cfg = edict(yaml.safe_load(open(join(getDir(__file__), \"../config/backbone.yaml\"))))
    decoder_cfg = edict(yaml.safe_load(open(join(getDir(__file__), \"../config/decoder.yaml\"))))
    
    # DGCNN-specific configuration
    dgcnn_cfg = edict({
        'AUGMENTATION': {
            'rotation': True,
            'scaling': True,
            'flipping': True,
            'jittering': True,
            'dropout': True
        },
        'PRETRAINED_PATH': None,  # Path to pre-trained DGCNN weights
        'TRAIN': {
            'WARMUP_STEPS': 1000,
            'GRADIENT_CLIP': 1.0,
            'MIXED_PRECISION': True
        }
    })
    
    # Merge configurations with priority to existing values
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    
    # Only update TRAIN section with new fields, keeping existing ones
    if 'TRAIN' in cfg:
        cfg.TRAIN.update(dgcnn_cfg['TRAIN'])
    else:
        cfg.TRAIN = dgcnn_cfg['TRAIN']
    
    # Add DGCNN-specific fields
    cfg.AUGMENTATION = dgcnn_cfg['AUGMENTATION']
    cfg.PRETRAINED_PATH = dgcnn_cfg['PRETRAINED_PATH']
    
    return cfg`
}
Response

[{"type": "text", "text": "Tool 'str_replace_editor' not found.", "uuid": "e06db49e-b400-49af-9b59-b7526108be3e"}]
S


str_replace_based_edit_tool


Let me create a complete fixed version of the training script:

Fixed train_dgcnn.py script
Text 
Now, let me also fix a potential issue in the DGCNN backbone related to handling the input format properly:

Fixed DGCNN Backbone with proper input handling
Text 
Now let me also create a simple test script to help you debug the training process:

Test script for DGCNN setup
Code 
Summary of the fixes:
Fixed the SUBSAMPLE error: The main issue was that cfg.TRAIN.SUBSAMPLE was referenced in the data module but not defined in the configuration. I've added it with a default value of True.
Improved configuration handling: The configuration merging now properly preserves existing values from the YAML files while adding DGCNN-specific parameters.
Fixed the DGCNN backbone:
Improved input handling to work with both numpy arrays and tensors
Fixed device placement issues
Improved padding and masking logic
Removed global pooling for point-wise tasks (can be re-enabled if needed)
Added debugging features:
Added a --debug flag to the training script for easier debugging
Created a test script to verify each component works correctly
To use the fixed version:
Replace your train_dgcnn.py with the fixed version from the first artifact
Replace your dgcnn_backbone.py with the fixed version from the second artifact
Run the test script first to verify everything works:
bash
python test_dgcnn_setup.py
Start training with debug mode first:
bash
python scripts/train_dgcnn.py --debug --batch_size 1 --epochs 2
If successful, run full training:
bash
python scripts/train_dgcnn.py --batch_size 2 --lr 0.001 --epochs 100
Additional recommendations:
Start with smaller batch sizes (1-2) to avoid memory issues
Disable mixed precision initially (already done in the fixed version)
Monitor GPU memory usage during training
Consider using gradient accumulation if you need larger effective batch sizes
The key insight is that the DGCNN backbone is a good replacement for MinkowskiEngine's sparse convolutions, but it requires careful handling of the input format and proper configuration of all the expected parameters. The fixed version should now w





# mask_pls/scripts/train_dgcnn.py
"""
Training script for MaskPLS with DGCNN backbone
Optimized for better performance and ONNX compatibility
"""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from os.path import join
import click
import torch
import yaml
import numpy as np
from pathlib import Path
from easydict import EasyDict as edict
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy

from mask_pls.datasets.semantic_dataset import SemanticDatasetModule
from mask_pls.models.dgcnn.maskpls_dgcnn import MaskPLSDGCNN
from mask_pls.models.dgcnn.augmentation import PointCloudAugmentation


class AugmentedDataModule(SemanticDatasetModule):
    """Data module with augmentation support"""
    def __init__(self, cfg):
        super().__init__(cfg)
        # Only apply augmentation if specified
        if hasattr(cfg, 'AUGMENTATION'):
            self.augmentation = PointCloudAugmentation(cfg.AUGMENTATION)
        else:
            self.augmentation = None
    
    def setup(self, stage=None):
        super().setup(stage)
        
        # Apply augmentation to training dataset if enabled
        if hasattr(self, 'train_mask_set') and self.augmentation is not None:
            original_getitem = self.train_mask_set.__getitem__
            
            def augmented_getitem(idx):
                data = original_getitem(idx)
                if self.cfg.TRAIN.get('AUG', True) and self.augmentation:
                    xyz, feats = data[0], data[1]
                    # Apply augmentation
                    xyz_aug, feats_aug = self.augmentation(xyz, feats)
                    data = (xyz_aug, feats_aug) + data[2:]
                return data
            
            self.train_mask_set.__getitem__ = augmented_getitem


def get_config():
    """Load and merge configuration files"""
    def getDir(obj):
        return os.path.dirname(os.path.abspath(obj))
    
    model_cfg = edict(yaml.safe_load(open(join(getDir(__file__), "../config/model.yaml"))))
    backbone_cfg = edict(yaml.safe_load(open(join(getDir(__file__), "../config/backbone.yaml"))))
    decoder_cfg = edict(yaml.safe_load(open(join(getDir(__file__), "../config/decoder.yaml"))))
    
    # Merge base configurations
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    
    # DGCNN-specific configuration additions (only add what's missing)
    dgcnn_additions = {
        'AUGMENTATION': {
            'rotation': True,
            'scaling': True,
            'flipping': True,
            'jittering': True,
            'dropout': False  # Start with dropout disabled
        },
        'PRETRAINED_PATH': None,
    }
    
    # Add DGCNN-specific fields
    cfg.update(dgcnn_additions)
    
    # Add training parameters if missing
    if 'TRAIN' not in cfg:
        cfg.TRAIN = edict()
    
    # Add missing TRAIN fields with defaults
    train_defaults = {
        'WARMUP_STEPS': 1000,
        'GRADIENT_CLIP': 1.0,
        'MIXED_PRECISION': False,  # Start with FP32 for stability
        'SUBSAMPLE': True,  # This was missing!
        'AUG': True,  # This was also referenced but missing
    }
    
    for key, value in train_defaults.items():
        if key not in cfg.TRAIN:
            cfg.TRAIN[key] = value
    
    return cfg


@click.command()
@click.option("--config", type=str, default=None, help="Custom config file")
@click.option("--pretrained", type=str, default=None, help="Pre-trained DGCNN weights")
@click.option("--checkpoint", type=str, default=None, help="Resume from checkpoint")
@click.option("--epochs", type=int, default=100)
@click.option("--batch_size", type=int, default=2)
@click.option("--lr", type=float, default=0.001)
@click.option("--gpus", type=int, default=1)
@click.option("--num_workers", type=int, default=4)
@click.option("--nuscenes", is_flag=True)
@click.option("--experiment_name", type=str, default="maskpls_dgcnn")
@click.option("--debug", is_flag=True, help="Enable debug mode")
def main(config, pretrained, checkpoint, epochs, batch_size, lr, gpus, 
         num_workers, nuscenes, experiment_name, debug):
    """Train MaskPLS with DGCNN backbone"""
    
    print("="*60)
    print("MaskPLS Training with DGCNN Backbone")
    print("="*60)
    
    # Load configuration
    cfg = get_config()
    
    if config:
        custom_cfg = edict(yaml.safe_load(open(config)))
        cfg.update(custom_cfg)
    
    # Update configuration with command line arguments
    cfg.TRAIN.MAX_EPOCH = epochs
    cfg.TRAIN.BATCH_SIZE = batch_size
    cfg.TRAIN.LR = lr
    cfg.TRAIN.N_GPUS = gpus
    cfg.TRAIN.NUM_WORKERS = num_workers
    cfg.EXPERIMENT.ID = experiment_name
    
    if pretrained:
        cfg.PRETRAINED_PATH = pretrained
    
    if nuscenes:
        cfg.MODEL.DATASET = "NUSCENES"
    
    dataset = cfg.MODEL.DATASET
    
    print(f"Configuration:")
    print(f"  Dataset: {dataset}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {lr}")
    print(f"  Epochs: {epochs}")
    print(f"  Workers: {num_workers}")
    print(f"  GPUs: {gpus}")
    print(f"  Subsample: {cfg.TRAIN.SUBSAMPLE}")
    print(f"  Augmentation: {cfg.TRAIN.AUG}")
    print(f"  Pre-trained: {pretrained is not None}")
    
    if debug:
        print("\nDebug mode enabled - using reduced settings")
        cfg.TRAIN.NUM_WORKERS = 0  # Use main thread for debugging
        cfg.TRAIN.BATCH_SIZE = 1
        epochs = 2
    
    # Create data module with augmentation
    print("\nInitializing data module...")
    data = AugmentedDataModule(cfg)
    data.setup()
    
    # Create model
    print("Creating model...")
    model = MaskPLSDGCNN(cfg)
    
    # Set things_ids from data module
    model.things_ids = data.things_ids
    
    # Load checkpoint if provided
    if checkpoint:
        print(f"\nLoading checkpoint: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)
    
    # Setup logging
    tb_logger = pl_loggers.TensorBoardLogger(
        "experiments/" + experiment_name,
        default_hp_metric=False
    )
    
    # Callbacks
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        
        ModelCheckpoint(
            monitor="metrics/pq",
            filename=f"{experiment_name}_epoch{{epoch:02d}}_pq{{metrics/pq:.3f}}",
            auto_insert_metric_name=False,
            mode="max",
            save_last=True,
            save_top_k=3
        ),
        
        ModelCheckpoint(
            monitor="metrics/iou", 
            filename=f"{experiment_name}_epoch{{epoch:02d}}_iou{{metrics/iou:.3f}}",
            auto_insert_metric_name=False,
            mode="max",
            save_top_k=3
        ),
        
        EarlyStopping(
            monitor="val_loss",
            patience=20,
            mode="min"
        )
    ]
    
    # Training strategy
    if gpus > 1:
        strategy = DDPStrategy(find_unused_parameters=False)
    else:
        strategy = None
    
    # Create trainer
    trainer = Trainer(
        devices=gpus if gpus > 0 else 1,
        accelerator="gpu" if gpus > 0 else "cpu",
        strategy=strategy,
        logger=tb_logger,
        max_epochs=epochs,
        callbacks=callbacks,
        log_every_n_steps=10,
        gradient_clip_val=cfg.TRAIN.get('GRADIENT_CLIP', 1.0),
        accumulate_grad_batches=cfg.TRAIN.get('BATCH_ACC', 1),
        precision=16 if cfg.TRAIN.get('MIXED_PRECISION', False) else 32,
        check_val_every_n_epoch=2,
        num_sanity_val_steps=2 if not debug else 0,
        detect_anomaly=debug,
        benchmark=True if not debug else False,
        sync_batchnorm=True if gpus > 1 else False,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    # Train
    print("\nStarting training...")
    try:
        trainer.fit(model, data)
        print("\nTraining completed successfully!")
        print(f"Best checkpoints saved in: experiments/{experiment_name}")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
