# mask_pls/scripts/train_dgcnn_fixed.py
import os
import click
import torch
import yaml
from pathlib import Path
from easydict import EasyDict as edict
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

from mask_pls.datasets.semantic_dataset import SemanticDatasetModule
from mask_pls.models.dgcnn.maskpls_dgcnn_optimized import MaskPLSDGCNNFixed

def get_config():
    """Load and merge configuration files"""
    def getDir(obj):
        return os.path.dirname(os.path.abspath(obj))
    
    model_cfg = edict(yaml.safe_load(open(os.path.join(getDir(__file__), "../config/model.yaml"))))
    backbone_cfg = edict(yaml.safe_load(open(os.path.join(getDir(__file__), "../config/backbone.yaml"))))
    decoder_cfg = edict(yaml.safe_load(open(os.path.join(getDir(__file__), "../config/decoder.yaml"))))
    
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    
    # Adjust training parameters
    cfg.TRAIN.BATCH_SIZE = 2  # Small batch size
    cfg.TRAIN.ACCUMULATE_GRAD_BATCHES = 2  # Effective batch size of 4
    cfg.TRAIN.WARMUP_STEPS = 500
    cfg.TRAIN.SUBSAMPLE = True
    cfg.TRAIN.AUG = True
    cfg.TRAIN.LR = 0.0001  # Lower learning rate
    
    return cfg

@click.command()
@click.option("--checkpoint", type=str, default=None, help="Resume from checkpoint")
@click.option("--epochs", type=int, default=100)
@click.option("--batch_size", type=int, default=2)
@click.option("--lr", type=float, default=0.0001)
@click.option("--gpus", type=int, default=1)
@click.option("--num_workers", type=int, default=4)
@click.option("--nuscenes", is_flag=True)
@click.option("--experiment_name", type=str, default="maskpls_dgcnn_fixed")
@click.option("--pretrained", type=str, default=None, help="Pre-trained DGCNN weights")
def main(checkpoint, epochs, batch_size, lr, gpus, num_workers, 
         nuscenes, experiment_name, pretrained):
    """Train fixed MaskPLS with DGCNN backbone"""
    
    print("="*60)
    print("MaskPLS Training with Fixed DGCNN Backbone")
    print("="*60)
    
    # Load configuration
    cfg = get_config()
    
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
    
    print(f"Configuration:")
    print(f"  Dataset: {cfg.MODEL.DATASET}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {lr}")
    print(f"  Epochs: {epochs}")
    print(f"  GPUs: {gpus}")
    
    # Create data module
    data = SemanticDatasetModule(cfg)
    data.setup()
    
    # Create model
    model = MaskPLSDGCNNFixed(cfg)
    model.things_ids = data.things_ids
    
    # Load checkpoint if provided
    if checkpoint:
        print(f"Loading checkpoint: {checkpoint}")
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
            patience=15,
            mode="min"
        )
    ]
    
    # Create trainer
    trainer = Trainer(
        devices=gpus,
        accelerator="gpu" if gpus > 0 else "cpu",
        strategy="ddp" if gpus > 1 else "auto",
        logger=tb_logger,
        max_epochs=epochs,
        callbacks=callbacks,
        log_every_n_steps=10,
        gradient_clip_val=0.5,  # Gradient clipping
        accumulate_grad_batches=cfg.TRAIN.get('ACCUMULATE_GRAD_BATCHES', 2),
        precision=32,  # Use full precision for stability
        check_val_every_n_epoch=1,  # Validate every epoch
        num_sanity_val_steps=2,
        benchmark=True,
        sync_batchnorm=True if gpus > 1 else False
    )
    
    # Train
    print("\nStarting training...")
    trainer.fit(model, data)
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main()