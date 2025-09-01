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
        self.augmentation = PointCloudAugmentation(cfg.get('AUGMENTATION', {}))
    
    def setup(self, stage=None):
        super().setup(stage)
        
        # Apply augmentation to training dataset
        if hasattr(self, 'train_mask_set'):
            original_getitem = self.train_mask_set.__getitem__
            
            def augmented_getitem(idx):
                data = original_getitem(idx)
                if self.cfg.TRAIN.get('AUG', True):
                    xyz, feats = data[0], data[1]
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
def main(config, pretrained, checkpoint, epochs, batch_size, lr, gpus, 
         num_workers, nuscenes, experiment_name):
    """Train MaskPLS with DGCNN backbone"""
    
    print("="*60)
    print("MaskPLS Training with DGCNN Backbone")
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
    print(f"  Pre-trained: {pretrained is not None}")
    
    # Create data module with augmentation
    data = AugmentedDataModule(cfg)
    data.setup()
    
    # Set things_ids in model
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
    
    # Create trainer
    trainer = Trainer(
        devices=gpus,
        accelerator="gpu" if gpus > 0 else "cpu",
        strategy=strategy,
        logger=tb_logger,
        max_epochs=epochs,
        callbacks=callbacks,
        log_every_n_steps=10,
        gradient_clip_val=cfg.TRAIN.get('GRADIENT_CLIP', 1.0),
        accumulate_grad_batches=cfg.TRAIN.get('BATCH_ACC', 1),
        precision=16 if cfg.TRAIN.get('MIXED_PRECISION', True) else 32,
        check_val_every_n_epoch=2,
        num_sanity_val_steps=0,
        detect_anomaly=False,
        benchmark=True,  # Enable cuDNN benchmark for speed
        sync_batchnorm=True if gpus > 1 else False
    )
    
    # Train
    print("\nStarting training...")
    trainer.fit(model, data)
    
    print("\nTraining completed!")
    print(f"Best checkpoints saved in: experiments/{experiment_name}")


if __name__ == "__main__":
    main()