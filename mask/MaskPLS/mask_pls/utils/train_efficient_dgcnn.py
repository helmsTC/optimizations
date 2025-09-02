# Save this as: mask/MaskPLS/mask_pls/scripts/train_dgcnn_optimized.py
"""
Optimized training script for MaskPLS with DGCNN backbone
Prevents OOM errors during validation
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
from mask_pls.models.dgcnn.maskpls_dgcnn_optimized import MaskPLSDGCNNOptimized


def get_config():
    """Load and merge configuration files"""
    def getDir(obj):
        return os.path.dirname(os.path.abspath(obj))
    
    model_cfg = edict(yaml.safe_load(open(join(getDir(__file__), "../config/model.yaml"))))
    backbone_cfg = edict(yaml.safe_load(open(join(getDir(__file__), "../config/backbone.yaml"))))
    decoder_cfg = edict(yaml.safe_load(open(join(getDir(__file__), "../config/decoder.yaml"))))
    
    # DGCNN-specific configuration with memory optimizations
    dgcnn_cfg = edict({
        'CHUNK_SIZE': 10000,  # Process points in chunks
        'VAL_BATCH_SIZE': 1,  # Validation batch size
        'TRAIN': {
            'WARMUP_STEPS': 1000,
            'GRADIENT_CLIP': 1.0,
            'MIXED_PRECISION': True,
            'SUBSAMPLE': True,  # Enable subsampling
            'AUG': True,  # Enable augmentation
            'VAL_CHECK_INTERVAL': 0.5,  # Check validation every half epoch
            'ACCUMULATE_GRAD_BATCHES': 4,  # Gradient accumulation
        }
    })
    
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg, **dgcnn_cfg})
    return cfg


@click.command()
@click.option("--config", type=str, default=None, help="Custom config file")
@click.option("--checkpoint", type=str, default=None, help="Resume from checkpoint")
@click.option("--epochs", type=int, default=100)
@click.option("--batch_size", type=int, default=1)
@click.option("--val_batch_size", type=int, default=1)
@click.option("--lr", type=float, default=0.0005)
@click.option("--gpus", type=int, default=1)
@click.option("--num_workers", type=int, default=4)
@click.option("--nuscenes", is_flag=True)
@click.option("--experiment_name", type=str, default="maskpls_dgcnn_optimized")
@click.option("--chunk_size", type=int, default=10000, help="Point cloud chunk size")
@click.option("--subsample_points", type=int, default=50000, help="Subsample points during training")
def main(config, checkpoint, epochs, batch_size, val_batch_size, lr, gpus, 
         num_workers, nuscenes, experiment_name, chunk_size, subsample_points):
    """Train MaskPLS with optimized DGCNN backbone"""
    
    print("="*60)
    print("MaskPLS Training with Optimized DGCNN Backbone")
    print("="*60)
    
    # Load configuration
    cfg = get_config()
    
    if config:
        custom_cfg = edict(yaml.safe_load(open(config)))
        cfg.update(custom_cfg)
    
    # Update configuration
    cfg.TRAIN.MAX_EPOCH = epochs
    cfg.TRAIN.BATCH_SIZE = batch_size
    cfg.VAL_BATCH_SIZE = val_batch_size
    cfg.TRAIN.LR = lr
    cfg.TRAIN.N_GPUS = gpus
    cfg.TRAIN.NUM_WORKERS = num_workers
    cfg.EXPERIMENT.ID = experiment_name
    cfg.CHUNK_SIZE = chunk_size
    
    # Update subsampling for memory efficiency
    if nuscenes:
        cfg.MODEL.DATASET = "NUSCENES"
        cfg.NUSCENES.SUB_NUM_POINTS = subsample_points
    else:
        cfg.KITTI.SUB_NUM_POINTS = subsample_points
    
    dataset = cfg.MODEL.DATASET
    
    print(f"Configuration:")
    print(f"  Dataset: {dataset}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Val Batch Size: {val_batch_size}")
    print(f"  Learning Rate: {lr}")
    print(f"  Epochs: {epochs}")
    print(f"  Chunk Size: {chunk_size}")
    print(f"  Subsample Points: {subsample_points}")
    print(f"  Mixed Precision: {cfg.TRAIN.MIXED_PRECISION}")
    
    # Create data module
    data = SemanticDatasetModule(cfg)
    data.setup()
    
    # Set things_ids in model
    model = MaskPLSDGCNNOptimized(cfg)
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
        strategy = DDPStrategy(
            find_unused_parameters=False,
            gradient_as_bucket_view=True  # Memory optimization
        )
    else:
        strategy = None
    
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
        accumulate_grad_batches=cfg.TRAIN.get('ACCUMULATE_GRAD_BATCHES', 4),
        precision=16 if cfg.TRAIN.get('MIXED_PRECISION', True) else 32,
        val_check_interval=cfg.TRAIN.get('VAL_CHECK_INTERVAL', 0.5),
        num_sanity_val_steps=0,
        detect_anomaly=False,
        benchmark=True,
        sync_batchnorm=True if gpus > 1 else False,
        # Memory optimizations
        enable_checkpointing=True,
        enable_model_summary=True,
        limit_val_batches=20,  # Limit validation batches to prevent OOM
    )
    
    # Train
    print("\nStarting training...")
    print("Memory optimizations enabled:")
    print("  - Chunked point cloud processing")
    print("  - Validation sub-batching")
    print("  - Limited validation batches")
    print("  - Mixed precision training")
    print("  - Gradient accumulation")
    
    trainer.fit(model, data)
    
    print("\nTraining completed!")
    print(f"Best checkpoints saved in: experiments/{experiment_name}")


if __name__ == "__main__":
    main()