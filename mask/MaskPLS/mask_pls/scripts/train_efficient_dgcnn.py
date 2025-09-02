# mask_pls/scripts/train_efficient_dgcnn.py
"""
Training script for efficient DGCNN-based MaskPLS
Memory optimized to prevent CUDA OOM during validation
"""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'  # Better memory allocation

from os.path import join
import click
import torch
import yaml
import gc
import numpy as np
from pathlib import Path
from easydict import EasyDict as edict
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy

from mask_pls.datasets.semantic_dataset import SemanticDatasetModule
# Note: Using the original DGCNN model, not the optimized one that might have issues
from mask_pls.models.dgcnn.maskpls_dgcnn import MaskPLSDGCNN


class MemoryOptimizedMaskPLSDGCNN(MaskPLSDGCNN):
    """Memory-optimized version to prevent OOM during validation"""
    
    def validation_step(self, batch, batch_idx):
        # Process smaller batches during validation
        with torch.cuda.amp.autocast(enabled=self.cfg.TRAIN.get('MIXED_PRECISION', False)):
            with torch.no_grad():
                # Clear cache before processing
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Subsample points for validation to prevent OOM
                for key in ['pt_coord', 'feats']:
                    if key in batch:
                        for i in range(len(batch[key])):
                            pts = batch[key][i]
                            if isinstance(pts, np.ndarray) and len(pts) > 20000:
                                # Subsample to 20k points for validation
                                idx = np.random.choice(len(pts), 20000, replace=False)
                                if key == 'pt_coord':
                                    batch['pt_coord'][i] = pts[idx]
                                    batch['feats'][i] = batch['feats'][i][idx]
                                    if i < len(batch['sem_label']):
                                        batch['sem_label'][i] = batch['sem_label'][i][idx]
                                    if i < len(batch['ins_label']):
                                        batch['ins_label'][i] = batch['ins_label'][i][idx]
                
                # Forward pass
                outputs, padding, sem_logits = self(batch)
                losses = self.get_losses(batch, outputs, padding, sem_logits)
                
                # Log losses
                for k, v in losses.items():
                    self.log(f'val/{k}', v, batch_size=self.cfg.TRAIN.BATCH_SIZE, sync_dist=True)
                
                total_loss = sum(losses.values())
                self.log('val_loss', total_loss, batch_size=self.cfg.TRAIN.BATCH_SIZE, sync_dist=True)
                
                # Skip panoptic inference for most batches to save memory
                if batch_idx % 10 == 0:  # Only evaluate every 10th batch
                    try:
                        sem_pred, ins_pred = self.panoptic_inference(outputs, padding)
                        self.evaluator.update(sem_pred, ins_pred, batch)
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"Skipping panoptic eval for batch {batch_idx} due to OOM")
                            torch.cuda.empty_cache()
                
                # Clear intermediate variables
                del outputs, padding, sem_logits, losses
                torch.cuda.empty_cache()
                
                self.validation_step_outputs.append(total_loss.detach().cpu())
                return total_loss
    
    def training_step(self, batch, batch_idx):
        # Clear cache periodically during training
        if batch_idx % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Subsample during training if needed
        for key in ['pt_coord', 'feats']:
            if key in batch:
                for i in range(len(batch[key])):
                    pts = batch[key][i]
                    if isinstance(pts, np.ndarray) and len(pts) > 40000:
                        # Subsample to 40k points for training
                        idx = np.random.choice(len(pts), 40000, replace=False)
                        if key == 'pt_coord':
                            batch['pt_coord'][i] = pts[idx]
                            batch['feats'][i] = batch['feats'][i][idx]
                            if i < len(batch['sem_label']):
                                batch['sem_label'][i] = batch['sem_label'][i][idx]
                            if i < len(batch['ins_label']):
                                batch['ins_label'][i] = batch['ins_label'][i][idx]
        
        # Regular training step
        return super().training_step(batch, batch_idx)


def get_config():
    """Load and merge configuration files with memory optimizations"""
    def getDir(obj):
        return os.path.dirname(os.path.abspath(obj))
    
    model_cfg = edict(yaml.safe_load(open(join(getDir(__file__), "../config/model.yaml"))))
    backbone_cfg = edict(yaml.safe_load(open(join(getDir(__file__), "../config/backbone.yaml"))))
    decoder_cfg = edict(yaml.safe_load(open(join(getDir(__file__), "../config/decoder.yaml"))))
    
    # Efficient DGCNN configuration with memory optimizations
    dgcnn_cfg = edict({
        'TRAIN': {
            'WARMUP_STEPS': 500,  # Reduced
            'GRADIENT_CLIP': 1.0,
            'MIXED_PRECISION': True,  # Enable mixed precision
            'ACCUMULATE_GRAD_BATCHES': 2,  # Reduced from 4
        }
    })
    
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg, **dgcnn_cfg})
    
    # Memory optimizations - reduce these parameters
    cfg.DECODER.NUM_QUERIES = 50  # Reduced from 100
    cfg.DECODER.DEC_BLOCKS = 2    # Reduced from 3
    cfg.LOSS.NUM_POINTS = 20000   # Reduced from 50000
    cfg.LOSS.NUM_MASK_PTS = 200   # Reduced from 500
    
    # Reduce subsample points
    cfg.KITTI.SUB_NUM_POINTS = 40000     # Reduced from 80000
    cfg.NUSCENES.SUB_NUM_POINTS = 30000  # Reduced from 50000
    
    # Force smaller batch size
    cfg.TRAIN.BATCH_SIZE = 1  # Force to 1 for memory
    cfg.TRAIN.SUBSAMPLE = True
    cfg.TRAIN.AUG = True
    
    return cfg


@click.command()
@click.option("--checkpoint", type=str, default=None, help="Resume from checkpoint")
@click.option("--epochs", type=int, default=100)
@click.option("--batch_size", type=int, default=1)  # Default to 1
@click.option("--lr", type=float, default=0.0005)  # Lower LR
@click.option("--gpus", type=int, default=1)
@click.option("--num_workers", type=int, default=2)  # Reduced workers
@click.option("--nuscenes", is_flag=True)
@click.option("--experiment_name", type=str, default="maskpls_dgcnn_efficient_mem")
@click.option("--mixed_precision", is_flag=True, default=True, help="Use mixed precision training")
@click.option("--pretrained", type=str, default=None, help="Pre-trained DGCNN weights")
def main(checkpoint, epochs, batch_size, lr, gpus, num_workers, 
         nuscenes, experiment_name, mixed_precision, pretrained):
    """Train efficient MaskPLS with DGCNN backbone - Memory Optimized"""
    
    print("="*60)
    print("MaskPLS Training with Efficient DGCNN Backbone")
    print("Memory Optimized Version")
    print("="*60)
    
    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()
    
    # Load configuration
    cfg = get_config()
    
    # Override batch size to 1 for memory safety
    cfg.TRAIN.MAX_EPOCH = epochs
    cfg.TRAIN.BATCH_SIZE = 1  # Force to 1
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
    
    print(f"Configuration (Memory Optimized):")
    print(f"  Dataset: {dataset}")
    print(f"  Batch Size: 1 (forced for memory)")
    print(f"  Learning Rate: {lr}")
    print(f"  Epochs: {epochs}")
    print(f"  Mixed Precision: {mixed_precision}")
    print(f"  GPUs: {gpus}")
    print(f"  Workers: {num_workers}")
    print(f"  Num Queries: {cfg.DECODER.NUM_QUERIES}")
    print(f"  Sub Points: {cfg[dataset].SUB_NUM_POINTS}")
    
    # Create data module
    data = SemanticDatasetModule(cfg)
    data.setup()
    
    # Create memory-optimized model
    model = MemoryOptimizedMaskPLSDGCNN(cfg)
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
    
    # Callbacks with reduced frequency
    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),  # Changed from "step"
        
        ModelCheckpoint(
            monitor="val_loss",  # Changed to val_loss for stability
            filename=f"{experiment_name}_epoch{{epoch:02d}}_loss{{val_loss:.3f}}",
            auto_insert_metric_name=False,
            mode="min",
            save_last=True,
            save_top_k=2  # Reduced from 3
        ),
        
        EarlyStopping(
            monitor="val_loss",
            patience=10,  # Reduced from 20
            mode="min"
        )
    ]
    
    # Training strategy - single GPU for memory efficiency
    strategy = 'auto' if gpus <= 1 else DDPStrategy(find_unused_parameters=False)
    
    # Create trainer with memory optimizations
    trainer = Trainer(
        devices=gpus,
        accelerator="gpu" if gpus > 0 else "cpu",
        strategy=strategy,
        logger=tb_logger,
        max_epochs=epochs,
        callbacks=callbacks,
        log_every_n_steps=50,  # Increased from 10
        gradient_clip_val=cfg.TRAIN.get('GRADIENT_CLIP', 1.0),
        accumulate_grad_batches=cfg.TRAIN.get('ACCUMULATE_GRAD_BATCHES', 2),
        precision=16 if mixed_precision else 32,
        check_val_every_n_epoch=5,  # Validate less frequently
        limit_val_batches=20,  # Only validate on 20 batches
        num_sanity_val_steps=0,
        enable_model_summary=False,  # Disable model summary
        enable_progress_bar=True,
        enable_checkpointing=True,
        benchmark=False,  # Disable benchmark to save memory
        sync_batchnorm=False  # Single GPU
    )
    
    # Train
    print("\nStarting training with memory optimizations...")
    print("Note: Validation uses subsampled points and runs less frequently")
    print("If you still get OOM errors, try:")
    print("  - Reducing SUB_NUM_POINTS further")
    print("  - Reducing NUM_QUERIES further")
    print("  - Disabling augmentations")
    
    try:
        trainer.fit(model, data)
        print("\nTraining completed successfully!")
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n" + "="*60)
            print("CUDA OUT OF MEMORY ERROR!")
            print("Try these additional steps:")
            print("  1. Reduce SUB_NUM_POINTS to 20000")
            print("  2. Reduce NUM_QUERIES to 25")
            print("  3. Set limit_val_batches=10")
            print("  4. Disable augmentations (--no-aug)")
            print("  5. Use CPU workers: --num_workers 0")
            print("="*60)
            raise
    
    print(f"\nBest checkpoints saved in: experiments/{experiment_name}")


if __name__ == "__main__":
    main()