# mask_pls/scripts/train_dgcnn.py
"""
Training script for MaskPLS with DGCNN backbone
Complete version with memory optimizations
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

try:
    from pytorch_lightning import Trainer
    from pytorch_lightning import loggers as pl_loggers
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
    from pytorch_lightning.strategies import DDPStrategy
except ImportError:
    from lightning.pytorch import Trainer
    from lightning.pytorch import loggers as pl_loggers
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
    from lightning.pytorch.strategies import DDPStrategy

from mask_pls.datasets.semantic_dataset import SemanticDatasetModule
from mask_pls.models.dgcnn.maskpls_dgcnn import MaskPLSDGCNN
from mask_pls.models.dgcnn.augmentation import PointCloudAugmentation


class AugmentedDataModule(SemanticDatasetModule):
    """Data module with augmentation support"""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.augmentation = PointCloudAugmentation(cfg.get('AUGMENTATION', {}))
    
    def setup(self, stage=None):
        super().setup(stage)
        
        # Apply augmentation to training dataset
        if hasattr(self, 'train_mask_set') and self.cfg.TRAIN.get('AUG', True):
            original_getitem = self.train_mask_set.__getitem__
            
            def augmented_getitem(idx):
                data = original_getitem(idx)
                xyz, feats = data[0], data[1]
                xyz_aug, feats_aug = self.augmentation(xyz, feats)
                data = (xyz_aug, feats_aug) + data[2:]
                return data
            
            self.train_mask_set.__getitem__ = augmented_getitem


def get_config():
    """Load and merge configuration files with memory optimizations"""
    def getDir(obj):
        return os.path.dirname(os.path.abspath(obj))
    
    model_cfg = edict(yaml.safe_load(open(join(getDir(__file__), "../config/model.yaml"))))
    backbone_cfg = edict(yaml.safe_load(open(join(getDir(__file__), "../config/backbone.yaml"))))
    decoder_cfg = edict(yaml.safe_load(open(join(getDir(__file__), "../config/decoder.yaml"))))
    
    # DGCNN-specific configuration with memory optimizations
    dgcnn_cfg = edict({
        'AUGMENTATION': {
            'rotation': True,
            'scaling': True,
            'flipping': True,
            'jittering': False,  # Disable jittering to save memory
            'dropout': False  # Disable dropout during validation
        },
        'PRETRAINED_PATH': None,  # Will be set by command line
        'TRAIN': {
            'WARMUP_STEPS': 500,  # Reduced warmup
            'GRADIENT_CLIP': 1.0,
            'MIXED_PRECISION': False,  # Keep disabled for stability
            'SUBSAMPLE': True,  # Enable subsampling
        },
        'LOSS': {
            'NUM_POINTS': 30000,  # Reduced from 50000
            'NUM_MASK_PTS': 300,  # Reduced from 500
            'P_RATIO': 0.3,  # Reduced from 0.4
        }
    })
    
    # Merge configs
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    
    # Update with DGCNN config
    for key, value in dgcnn_cfg.items():
        if key in cfg and isinstance(cfg[key], dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    
    # Override dataset-specific subsample settings for memory
    if 'KITTI' in cfg:
        cfg.KITTI.SUB_NUM_POINTS = 50000  # Reduced from 80000
    if 'NUSCENES' in cfg:
        cfg.NUSCENES.SUB_NUM_POINTS = 40000  # Reduced from 50000
    
    # Update loss config if it exists
    if 'LOSS' in cfg:
        cfg.LOSS.NUM_POINTS = dgcnn_cfg['LOSS']['NUM_POINTS']
        cfg.LOSS.NUM_MASK_PTS = dgcnn_cfg['LOSS']['NUM_MASK_PTS']
        cfg.LOSS.P_RATIO = dgcnn_cfg['LOSS']['P_RATIO']
    
    return cfg


@click.command()
@click.option("--config", type=str, default=None, help="Custom config file")
@click.option("--pretrained", type=str, default=None, help="Pre-trained DGCNN weights (.pth file)")
@click.option("--checkpoint", type=str, default=None, help="Resume from checkpoint")
@click.option("--epochs", type=int, default=100)
@click.option("--batch_size", type=int, default=2)
@click.option("--lr", type=float, default=0.001)
@click.option("--gpus", type=int, default=1)
@click.option("--num_workers", type=int, default=4)
@click.option("--nuscenes", is_flag=True)
@click.option("--experiment_name", type=str, default="maskpls_dgcnn")
def main(config, pretrained, checkpoint, epochs, batch_size, lr, gpus, 
         num_workers, nuscenes, experiment_name):
    """Train MaskPLS with DGCNN backbone"""
    
    print("="*60)
    print("MaskPLS Training with DGCNN Backbone (Memory Optimized)")
    print("="*60)
    
    # Load configuration
    cfg = get_config()
    
    if config:
        custom_cfg = edict(yaml.safe_load(open(config)))
        cfg.update(custom_cfg)
    
    # Update configuration
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
    print(f"  Pre-trained weights: {pretrained if pretrained else 'None'}")
    print(f"  Memory optimizations: Enabled")
    print(f"  Subsampling: {cfg.KITTI.SUB_NUM_POINTS if dataset == 'KITTI' else cfg.NUSCENES.SUB_NUM_POINTS} points")
    print(f"  Loss computation points: {cfg.LOSS.NUM_POINTS}")
    print(f"  Mask points: {cfg.LOSS.NUM_MASK_PTS}")
    
    # Create data module with augmentation
    data = AugmentedDataModule(cfg)
    data.setup()
    
    # Create model with pretrained weights
    model = MaskPLSDGCNN(cfg)
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
    strategy = DDPStrategy(find_unused_parameters=False) if gpus > 1 else None
    
    # Create trainer with memory optimizations
    trainer = Trainer(
        devices=gpus,
        accelerator="gpu" if gpus > 0 else "cpu",
        strategy=strategy,
        logger=tb_logger,
        max_epochs=epochs,
        callbacks=callbacks,
        log_every_n_steps=10,
        gradient_clip_val=cfg.TRAIN.get('GRADIENT_CLIP', 1.0),
        accumulate_grad_batches=cfg.TRAIN.get('BATCH_ACC', 4),
        precision=32,  # Keep at 32 for stability
        check_val_every_n_epoch=2,
        num_sanity_val_steps=0,  # Skip sanity checks to save memory
        detect_anomaly=False,
        benchmark=True,
        sync_batchnorm=True if gpus > 1 else False,
        # Memory optimization parameters
        limit_val_batches=0.5,  # Only validate on 50% of validation data
        val_check_interval=1.0,  # Check validation once per epoch
        enable_model_summary=False,  # Disable model summary to save memory
        enable_progress_bar=True,
        enable_checkpointing=True,
        reload_dataloaders_every_n_epochs=1,  # Reload to clear memory
    )
    
    # Train
    print("\nStarting training with memory optimizations...")
    print("Memory saving features:")
    print("  - Validation on 50% of data")
    print("  - Periodic CUDA cache clearing")
    print("  - Gradient detachment in validation")
    print("  - Point cloud subsampling")
    print("  - Reduced loss computation points")
    
    trainer.fit(model, data)
    
    print("\nTraining completed!")
    print(f"Best checkpoints saved in: experiments/{experiment_name}")


if __name__ == "__main__":
    # Clear CUDA cache before starting
    torch.cuda.empty_cache()
    main()