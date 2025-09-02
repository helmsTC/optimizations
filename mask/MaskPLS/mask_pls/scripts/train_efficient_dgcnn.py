# mask_pls/scripts/train_efficient_dgcnn.py
"""
Training script for efficient DGCNN-based MaskPLS
"""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from os.path import join
import click
import torch
import yaml
from pathlib import Path
from easydict import EasyDict as edict
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy

from mask_pls.datasets.semantic_dataset import SemanticDatasetModule
from mask_pls.models.mask_dgcnn_optimized import MaskPLSDGCNNOptimized


def get_config():
    """Load and merge configuration files"""
    def getDir(obj):
        return os.path.dirname(os.path.abspath(obj))
    
    model_cfg = edict(yaml.safe_load(open(join(getDir(__file__), "../config/model.yaml"))))
    backbone_cfg = edict(yaml.safe_load(open(join(getDir(__file__), "../config/backbone.yaml"))))
    decoder_cfg = edict(yaml.safe_load(open(join(getDir(__file__), "../config/decoder.yaml"))))
    
    # Efficient DGCNN configuration
    dgcnn_cfg = edict({
        'TRAIN': {
            'WARMUP_STEPS': 1000,
            'GRADIENT_CLIP': 1.0,
            'MIXED_PRECISION': False,  # Disabled for now
            'ACCUMULATE_GRAD_BATCHES': 4,
        }
    })
    
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg, **dgcnn_cfg})
    
    # Override some settings for efficiency
    cfg.TRAIN.BATCH_SIZE = 2  # Smaller batch size for memory
    cfg.TRAIN.SUBSAMPLE = True
    cfg.TRAIN.AUG = True
    
    return cfg


@click.command()
@click.option("--checkpoint", type=str, default=None, help="Resume from checkpoint")
@click.option("--epochs", type=int, default=100)
@click.option("--batch_size", type=int, default=2)
@click.option("--lr", type=float, default=0.001)
@click.option("--gpus", type=int, default=1)
@click.option("--num_workers", type=int, default=4)
@click.option("--nuscenes", is_flag=True)
@click.option("--experiment_name", type=str, default="maskpls_dgcnn_efficient")
@click.option("--mixed_precision", is_flag=True, help="Use mixed precision training")
@click.option("--pretrained", type=str, default=None, help="Pre-trained DGCNN weights")

def main(checkpoint, epochs, batch_size, lr, gpus, num_workers, 
         nuscenes, experiment_name, mixed_precision):
    """Train efficient MaskPLS with DGCNN backbone"""
    
    print("="*60)
    print("MaskPLS Training with Efficient DGCNN Backbone")
    print("="*60)
    
    # Load configuration
    cfg = get_config()
    
    # Update configuration
    cfg.TRAIN.MAX_EPOCH = epochs
    cfg.TRAIN.BATCH_SIZE = batch_size
    cfg.TRAIN.LR = lr
    cfg.TRAIN.N_GPUS = gpus
    cfg.TRAIN.NUM_WORKERS = num_workers
    cfg.TRAIN.MIXED_PRECISION = mixed_precision
    cfg.EXPERIMENT.ID = experiment_name
    if pretrained:
    cfg.PRETRAINED_PATH = pretrained
    print(f"  Using pretrained weights: {pretrained}")
    if nuscenes:
        cfg.MODEL.DATASET = "NUSCENES"
    
    dataset = cfg.MODEL.DATASET
    
    print(f"Configuration:")
    print(f"  Dataset: {dataset}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {lr}")
    print(f"  Epochs: {epochs}")
    print(f"  Mixed Precision: {mixed_precision}")
    print(f"  GPUs: {gpus}")
    print(f"  Workers: {num_workers}")
    
    # Create data module
    data = SemanticDatasetModule(cfg)
    data.setup()
    
    # Create model
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
    strategy = 'auto' if gpus <= 1 else DDPStrategy(find_unused_parameters=False)
    
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
        accumulate_grad_batches=cfg.TRAIN.get('ACCUMULATE_GRAD_BATCHES', 1),
        precision=16 if mixed_precision else 32,  # Use 16-bit if mixed precision enabled
        check_val_every_n_epoch=2,
        num_sanity_val_steps=0,
        benchmark=True,
        sync_batchnorm=True if gpus > 1 else False
    )
    
    # Train
    print("\nStarting training...")
    trainer.fit(model, data)
    
    print("\nTraining completed!")
    print(f"Best checkpoints saved in: experiments/{experiment_name}")


if __name__ == "__main__":
    main()