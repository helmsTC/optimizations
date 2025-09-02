# mask_pls/scripts/train_efficient_dgcnn.py
"""
Training script for efficient DGCNN-based MaskPLS
Memory optimized to prevent CUDA OOM during validation
"""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

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

from mask_pls.datasets.semantic_dataset import SemanticDatasetModule
from mask_pls.models.dgcnn.maskpls_dgcnn import MaskPLSDGCNN
from mask_pls.models.loss import MaskLoss, SemLoss
from mask_pls.utils.evaluate_panoptic import PanopticEvaluator
from pytorch_lightning.core.lightning import LightningModule


class MaskPLSDGCNNOptimized(LightningModule):
    """Memory-optimized MaskPLS with DGCNN backbone"""
    
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(dict(cfg))
        self.cfg = cfg
        
        dataset = cfg.MODEL.DATASET
        self.num_classes = cfg[dataset].NUM_CLASSES
        self.things_ids = []
        
        # Import and create DGCNN backbone with pretrained support
        from mask_pls.models.dgcnn.dgcnn_backbone import DGCNNPretrainedBackbone, DGCNNBackbone
        
        # Use pretrained backbone if path provided
        pretrained_path = cfg.get('PRETRAINED_PATH', None)
        if pretrained_path and Path(pretrained_path).exists():
            print(f"Loading pretrained DGCNN from: {pretrained_path}")
            self.backbone = DGCNNPretrainedBackbone(cfg.BACKBONE, pretrained_path)
        else:
            if pretrained_path:
                print(f"Warning: Pretrained path {pretrained_path} not found, using random init")
            self.backbone = DGCNNBackbone(cfg.BACKBONE)
        
        # Create decoder
        from mask_pls.models.decoder import MaskedTransformerDecoder
        self.decoder = MaskedTransformerDecoder(
            cfg.DECODER,
            cfg.BACKBONE,
            cfg[dataset]
        )
        
        # Losses
        self.mask_loss = MaskLoss(cfg.LOSS, cfg[dataset])
        self.sem_loss = SemLoss(cfg.LOSS.SEM.WEIGHTS)
        
        # Evaluator
        self.evaluator = PanopticEvaluator(cfg[dataset], dataset)
        
        # For validation outputs
        self.validation_step_outputs = []
        
        # Learning rate warmup
        self.warmup_steps = cfg.TRAIN.get('WARMUP_STEPS', 500)
        self.current_step = 0
    
    def forward(self, x):
        # DGCNN feature extraction
        feats, coords, pad_masks, sem_logits = self.backbone(x)
        
        # Transformer decoder
        outputs, padding = self.decoder(feats, coords, pad_masks)
        
        return outputs, padding, sem_logits
    
    def get_losses(self, x, outputs, padding, sem_logits):
        """Compute all losses"""
        # Mask losses
        targets = {'classes': x['masks_cls'], 'masks': x['masks']}
        loss_mask = self.mask_loss(outputs, targets, x['masks_ids'], x['pt_coord'])
        
        # Semantic losses
        sem_labels = []
        for i, labels in enumerate(x['sem_label']):
            labels = torch.from_numpy(labels).long().cuda()
            if i < len(padding):
                valid_mask = ~padding[i]
                valid_labels = labels[valid_mask] if valid_mask.any() else labels
            else:
                valid_labels = labels
            sem_labels.append(valid_labels)
        
        if sem_labels and any(len(s) > 0 for s in sem_labels):
            sem_labels = torch.cat(sem_labels)
            if padding.numel() > 0:
                sem_logits_valid = sem_logits[~padding]
            else:
                sem_logits_valid = sem_logits.flatten(0, 1)
            
            if sem_logits_valid.numel() > 0 and sem_labels.numel() > 0:
                loss_sem = self.sem_loss(sem_logits_valid, sem_labels)
            else:
                loss_sem = {'sem_ce': torch.tensor(0.0).cuda(), 
                           'sem_lov': torch.tensor(0.0).cuda()}
        else:
            loss_sem = {'sem_ce': torch.tensor(0.0).cuda(), 
                       'sem_lov': torch.tensor(0.0).cuda()}
        
        loss_mask.update(loss_sem)
        return loss_mask
    
    def training_step(self, batch, batch_idx):
        # Clear cache periodically
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
        
        # Subsample if needed
        self._subsample_batch(batch, max_points=30000)
        
        outputs, padding, sem_logits = self(batch)
        losses = self.get_losses(batch, outputs, padding, sem_logits)
        
        # Log individual losses
        for k, v in losses.items():
            self.log(f'train/{k}', v, batch_size=self.cfg.TRAIN.BATCH_SIZE, on_step=True, on_epoch=False)
        
        # Total loss
        total_loss = sum(losses.values())
        self.log('train_loss', total_loss, batch_size=self.cfg.TRAIN.BATCH_SIZE, on_step=True, on_epoch=False)
        
        # Update step counter
        self.current_step += 1
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        # Clear cache
        torch.cuda.empty_cache()
        
        # Aggressive subsampling for validation
        self._subsample_batch(batch, max_points=15000)
        
        with torch.no_grad():
            outputs, padding, sem_logits = self(batch)
            losses = self.get_losses(batch, outputs, padding, sem_logits)
        
        # Log losses
        for k, v in losses.items():
            self.log(f'val/{k}', v, batch_size=self.cfg.TRAIN.BATCH_SIZE, on_step=False, on_epoch=True)
        
        total_loss = sum(losses.values())
        self.log('val_loss', total_loss, batch_size=self.cfg.TRAIN.BATCH_SIZE, on_step=False, on_epoch=True)
        
        # Skip most panoptic evaluations to save memory
        if batch_idx % 10 == 0:
            try:
                sem_pred, ins_pred = self.panoptic_inference(outputs, padding)
                self.evaluator.update(sem_pred, ins_pred, batch)
            except Exception as e:
                print(f"Skipping panoptic eval: {e}")
        
        self.validation_step_outputs.append(total_loss.detach().cpu())
        return total_loss
    
    def on_validation_epoch_end(self):
        # Compute metrics
        try:
            pq = self.evaluator.get_mean_pq()
            iou = self.evaluator.get_mean_iou()
            rq = self.evaluator.get_mean_rq()
            
            self.log('metrics/pq', pq, batch_size=self.cfg.TRAIN.BATCH_SIZE)
            self.log('metrics/iou', iou, batch_size=self.cfg.TRAIN.BATCH_SIZE)
            self.log('metrics/rq', rq, batch_size=self.cfg.TRAIN.BATCH_SIZE)
            
            print(f"\nValidation Metrics - PQ: {pq:.4f}, IoU: {iou:.4f}, RQ: {rq:.4f}")
        except:
            print("Metrics computation failed")
        
        # Reset
        self.evaluator.reset()
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        # Separate parameters for backbone and decoder with different LRs
        backbone_params = []
        decoder_params = []
        
        for name, param in self.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                decoder_params.append(param)
        
        # Different learning rates for pretrained backbone
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': self.cfg.TRAIN.LR * 0.1},  # Lower LR for pretrained
            {'params': decoder_params, 'lr': self.cfg.TRAIN.LR}
        ], weight_decay=1e-4)
        
        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.TRAIN.MAX_EPOCH
        )
        
        return [optimizer], [scheduler]
    
    def optimizer_step(self, *args, **kwargs):
        # Learning rate warmup
        if self.current_step < self.warmup_steps:
            lr_scale = min(1.0, float(self.current_step) / float(self.warmup_steps))
            for pg in self.optimizers().param_groups:
                pg['lr'] = lr_scale * pg['initial_lr'] if 'initial_lr' in pg else lr_scale * self.cfg.TRAIN.LR
        
        super().optimizer_step(*args, **kwargs)
    
    def _subsample_batch(self, batch, max_points):
        """Subsample points to prevent OOM"""
        for key in ['pt_coord', 'feats']:
            if key in batch:
                for i in range(len(batch[key])):
                    pts = batch[key][i]
                    if isinstance(pts, np.ndarray) and len(pts) > max_points:
                        idx = np.random.choice(len(pts), max_points, replace=False)
                        batch['pt_coord'][i] = pts[idx]
                        batch['feats'][i] = batch['feats'][i][idx]
                        if i < len(batch.get('sem_label', [])):
                            batch['sem_label'][i] = batch['sem_label'][i][idx]
                        if i < len(batch.get('ins_label', [])):
                            batch['ins_label'][i] = batch['ins_label'][i][idx]
    
    def panoptic_inference(self, outputs, padding):
        """Simplified panoptic segmentation inference"""
        mask_cls = outputs["pred_logits"]
        mask_pred = outputs["pred_masks"]
        
        sem_pred = []
        ins_pred = []
        
        for b in range(mask_cls.shape[0]):
            if b < len(padding):
                valid_mask = ~padding[b]
                # Simple semantic prediction
                sem = torch.argmax(mask_cls[b], dim=-1)
                sem_pred.append(sem.cpu().numpy())
                # Simple instance (zeros for now)
                ins = torch.zeros_like(sem)
                ins_pred.append(ins.cpu().numpy())
            else:
                # Fallback
                sem_pred.append(np.zeros(mask_pred.shape[1], dtype=np.int32))
                ins_pred.append(np.zeros(mask_pred.shape[1], dtype=np.int32))
        
        return sem_pred, ins_pred


def get_config():
    """Load and merge configuration files with memory optimizations"""
    def getDir(obj):
        return os.path.dirname(os.path.abspath(obj))
    
    model_cfg = edict(yaml.safe_load(open(join(getDir(__file__), "../config/model.yaml"))))
    backbone_cfg = edict(yaml.safe_load(open(join(getDir(__file__), "../config/backbone.yaml"))))
    decoder_cfg = edict(yaml.safe_load(open(join(getDir(__file__), "../config/decoder.yaml"))))
    
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    
    # Memory optimizations
    cfg.DECODER.NUM_QUERIES = 50
    cfg.DECODER.DEC_BLOCKS = 2
    cfg.LOSS.NUM_POINTS = 20000
    cfg.LOSS.NUM_MASK_PTS = 200
    
    cfg.KITTI.SUB_NUM_POINTS = 30000
    cfg.NUSCENES.SUB_NUM_POINTS = 25000
    
    cfg.TRAIN.BATCH_SIZE = 1
    cfg.TRAIN.WARMUP_STEPS = 500
    cfg.TRAIN.GRADIENT_CLIP = 1.0
    cfg.TRAIN.MIXED_PRECISION = True
    
    return cfg


@click.command()
@click.option("--checkpoint", type=str, default=None, help="Resume from checkpoint")
@click.option("--pretrained", type=str, default=None, help="Pre-trained DGCNN weights")
@click.option("--epochs", type=int, default=50)
@click.option("--batch_size", type=int, default=1)
@click.option("--lr", type=float, default=0.0005)
@click.option("--gpus", type=int, default=1)
@click.option("--num_workers", type=int, default=2)
@click.option("--nuscenes", is_flag=True)
@click.option("--experiment_name", type=str, default="maskpls_efficient")
@click.option("--mixed_precision", is_flag=True, default=True, help="Use mixed precision training")
def main(checkpoint, pretrained, epochs, batch_size, lr, gpus, num_workers, 
         nuscenes, experiment_name, mixed_precision):
    """Train efficient MaskPLS with DGCNN backbone"""
    
    print("="*60)
    print("Efficient MaskPLS-DGCNN Training")
    print("Memory Optimized Version")
    print("="*60)
    
    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()
    
    # Load config
    cfg = get_config()
    
    cfg.TRAIN.MAX_EPOCH = epochs
    cfg.TRAIN.BATCH_SIZE = 1  # Force to 1
    cfg.TRAIN.LR = lr
    cfg.TRAIN.N_GPUS = gpus
    cfg.TRAIN.NUM_WORKERS = num_workers
    cfg.TRAIN.MIXED_PRECISION = mixed_precision
    cfg.EXPERIMENT.ID = experiment_name
    
    # Add pretrained path to config
    if pretrained:
        cfg.PRETRAINED_PATH = pretrained
        print(f"  Using pretrained weights: {pretrained}")
    
    if nuscenes:
        cfg.MODEL.DATASET = "NUSCENES"
    
    print(f"Configuration:")
    print(f"  Dataset: {cfg.MODEL.DATASET}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: 1 (fixed)")
    print(f"  Learning Rate: {lr}")
    print(f"  Mixed Precision: {mixed_precision}")
    print(f"  Pretrained: {pretrained if pretrained else 'None'}")
    
    # Create data module
    data = SemanticDatasetModule(cfg)
    data.setup()
    
    # Create model
    model = MaskPLSDGCNNOptimized(cfg)
    model.things_ids = data.things_ids
    
    if checkpoint:
        print(f"Loading checkpoint: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)
    
    # Logger
    tb_logger = pl_loggers.TensorBoardLogger(
        "experiments/" + experiment_name,
        default_hp_metric=False
    )
    
    # Callbacks
    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            monitor="val_loss",
            filename=f"{experiment_name}_{{epoch:02d}}_{{val_loss:.3f}}",
            mode="min",
            save_last=True,
            save_top_k=2
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min"
        )
    ]
    
    # Trainer
    trainer = Trainer(
        devices=gpus,
        accelerator="gpu" if gpus > 0 else "cpu",
        logger=tb_logger,
        max_epochs=epochs,
        callbacks=callbacks,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        precision=16 if mixed_precision else 32,
        check_val_every_n_epoch=5,
        limit_val_batches=10,  # Only 10 validation batches
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        enable_checkpointing=True
    )
    
    # Train
    print("\nStarting training...")
    print("Note: Validation uses subsampled points and runs less frequently")
    
    try:
        trainer.fit(model, data)
        print("\nTraining completed successfully!")
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n" + "="*60)
            print("CUDA OUT OF MEMORY ERROR!")
            print("Try these steps:")
            print("  1. Reduce SUB_NUM_POINTS further")
            print("  2. Reduce NUM_QUERIES to 25")
            print("  3. Set limit_val_batches=5")
            print("  4. Use --num_workers 0")
            print("="*60)
            raise
    
    print(f"\nBest checkpoints saved in: experiments/{experiment_name}")


if __name__ == "__main__":
    main()