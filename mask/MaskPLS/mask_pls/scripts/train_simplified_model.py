"""
Training script for the simplified ONNX-compatible MaskPLS model
Save as: mask/MaskPLS/mask_pls/scripts/train_simplified_model.py
"""

import os
from os.path import join
import click
import torch
import yaml
from easydict import EasyDict as edict
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# Import the simplified model components
from mask_pls.models.onnx.simplified_model import MaskPLSSimplifiedONNX
from mask_pls.datasets.semantic_dataset import SemanticDatasetModule
from mask_pls.models.loss import MaskLoss, SemLoss
from mask_pls.utils.evaluate_panoptic import PanopticEvaluator

import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule


class SimplifiedMaskPLS(LightningModule):
    """
    Lightning module for training the simplified MaskPLS model
    """
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(dict(cfg))
        self.cfg = cfg
        
        # Create the simplified model
        self.model = MaskPLSSimplifiedONNX(cfg)
        
        # Loss functions (same as original)
        self.mask_loss = MaskLoss(cfg.LOSS, cfg[cfg.MODEL.DATASET])
        self.sem_loss = SemLoss(cfg.LOSS.SEM.WEIGHTS)
        
        # Evaluator (same as original)
        self.evaluator = PanopticEvaluator(
            cfg[cfg.MODEL.DATASET], 
            cfg.MODEL.DATASET
        )
        
        # Cache things IDs
        data = SemanticDatasetModule(cfg)
        data.setup()
        self.things_ids = data.things_ids
        
    def forward(self, batch):
        """Forward pass with pre-voxelization"""
        # Extract data from batch
        points = batch['pt_coord']
        features = batch['feats']
        
        # Pre-voxelize the batch
        batch_voxels = []
        batch_coords = []
        valid_indices = []
        
        for i in range(len(points)):
            # Get points and features
            pts = torch.from_numpy(points[i]).float().cuda()
            feat = torch.from_numpy(features[i]).float().cuda()
            
            # Pre-process points (filter by bounds and normalize)
            bounds = self.cfg[self.cfg.MODEL.DATASET].SPACE
            valid_mask = torch.ones(pts.shape[0], dtype=torch.bool, device=pts.device)
            
            for dim in range(3):
                valid_mask &= (pts[:, dim] >= bounds[dim][0]) & (pts[:, dim] < bounds[dim][1])
            
            valid_pts = pts[valid_mask]
            valid_feat = feat[valid_mask]
            valid_idx = torch.where(valid_mask)[0]
            
            # Subsample if needed
            max_pts = self.cfg[self.cfg.MODEL.DATASET].SUB_NUM_POINTS
            if len(valid_pts) > max_pts and self.training:
                perm = torch.randperm(len(valid_pts))[:max_pts]
                valid_pts = valid_pts[perm]
                valid_feat = valid_feat[perm]
                valid_idx = valid_idx[perm]
            
            # Normalize coordinates
            norm_coords = torch.zeros_like(valid_pts)
            for dim in range(3):
                norm_coords[:, dim] = (valid_pts[:, dim] - bounds[dim][0]) / (bounds[dim][1] - bounds[dim][0])
            
            # Voxelize
            voxel_grid = self.model.voxelize_points(
                valid_pts.unsqueeze(0), 
                valid_feat.unsqueeze(0)
            )[0]  # Remove batch dim
            
            batch_voxels.append(voxel_grid)
            batch_coords.append(norm_coords)
            valid_indices.append(valid_idx)
        
        # Stack batch
        max_pts = max(c.shape[0] for c in batch_coords)
        
        # Pad coordinates
        padded_coords = []
        padding_masks = []
        for coords in batch_coords:
            n_pts = coords.shape[0]
            if n_pts < max_pts:
                pad_size = max_pts - n_pts
                coords = F.pad(coords, (0, 0, 0, pad_size))
            padded_coords.append(coords)
            
            # Create padding mask
            mask = torch.zeros(max_pts, dtype=torch.bool, device=coords.device)
            mask[n_pts:] = True
            padding_masks.append(mask)
        
        batch_voxels = torch.stack(batch_voxels)
        batch_coords = torch.stack(padded_coords)
        padding_masks = torch.stack(padding_masks)
        
        # Forward through model
        pred_logits, pred_masks, sem_logits = self.model(batch_voxels, batch_coords)
        
        outputs = {
            'pred_logits': pred_logits,
            'pred_masks': pred_masks,
            'aux_outputs': []  # Simplified model doesn't have aux outputs
        }
        
        return outputs, padding_masks, sem_logits, valid_indices
    
    def training_step(self, batch, batch_idx):
        outputs, padding, sem_logits, valid_indices = self.forward(batch)
        
        # Prepare targets for mask loss
        targets = {
            'classes': batch['masks_cls'],
            'masks': batch['masks']
        }
        
        # Mask loss
        loss_mask = self.mask_loss(outputs, targets, batch['masks_ids'], batch['pt_coord'])
        
        # Semantic loss (on valid points)
        sem_labels = []
        for i, (label, idx) in enumerate(zip(batch['sem_label'], valid_indices)):
            # FIX: Convert torch tensor indices to CPU numpy for indexing
            idx_np = idx.cpu().numpy()
            valid_label = torch.from_numpy(label[idx_np]).long().cuda()
            
            # Pad if needed
            if len(valid_label) < sem_logits.shape[1]:
                valid_label = F.pad(valid_label, (0, sem_logits.shape[1] - len(valid_label)))
            sem_labels.append(valid_label[:sem_logits.shape[1]])
        
        sem_labels = torch.stack(sem_labels)
        sem_labels = sem_labels[~padding].squeeze(1)
        sem_logits_valid = sem_logits[~padding]
        
        loss_sem = self.sem_loss(sem_logits_valid, sem_labels)
        
        # Combine losses
        loss_mask.update(loss_sem)
        
        # Log losses
        for k, v in loss_mask.items():
            self.log(f"train/{k}", v, batch_size=self.cfg.TRAIN.BATCH_SIZE)
        
        total_loss = sum(loss_mask.values())
        self.log("train_loss", total_loss, batch_size=self.cfg.TRAIN.BATCH_SIZE)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        outputs, padding, sem_logits, valid_indices = self.forward(batch)
        
        # Calculate losses (same as training)
        targets = {'classes': batch['masks_cls'], 'masks': batch['masks']}
        loss_mask = self.mask_loss(outputs, targets, batch['masks_ids'], batch['pt_coord'])
        
        # Semantic loss
        sem_labels = []
        for i, (label, idx) in enumerate(zip(batch['sem_label'], valid_indices)):
            # FIX: Convert torch tensor indices to CPU numpy for indexing
            idx_np = idx.cpu().numpy()
            valid_label = torch.from_numpy(label[idx_np]).long().cuda()
            
            if len(valid_label) < sem_logits.shape[1]:
                valid_label = F.pad(valid_label, (0, sem_logits.shape[1] - len(valid_label)))
            sem_labels.append(valid_label[:sem_logits.shape[1]])
        
        sem_labels = torch.stack(sem_labels)
        sem_labels = sem_labels[~padding].squeeze(1)
        sem_logits_valid = sem_logits[~padding]
        
        loss_sem = self.sem_loss(sem_logits_valid, sem_labels)
        loss_mask.update(loss_sem)
        
        # Log losses
        for k, v in loss_mask.items():
            self.log(f"val/{k}", v, batch_size=self.cfg.TRAIN.BATCH_SIZE)
        
        total_loss = sum(loss_mask.values())
        self.log("val_loss", total_loss, batch_size=self.cfg.TRAIN.BATCH_SIZE)
        
        # Panoptic inference and evaluation
        sem_pred, ins_pred = self.panoptic_inference(outputs, padding)
        
        # Map predictions back to original points
        full_sem_pred = []
        full_ins_pred = []
        
        for i, (pred_sem, pred_ins, idx) in enumerate(zip(sem_pred, ins_pred, valid_indices)):
            # Create full predictions
            full_sem = torch.zeros(len(batch['sem_label'][i]), dtype=torch.long)
            full_ins = torch.zeros(len(batch['ins_label'][i]), dtype=torch.long)
            
            # FIX: Convert indices to CPU for indexing
            idx_cpu = idx.cpu()
            valid_len = min(len(idx_cpu), len(pred_sem))
            
            if valid_len > 0:
                full_sem[idx_cpu[:valid_len]] = torch.from_numpy(pred_sem[:valid_len])
                full_ins[idx_cpu[:valid_len]] = torch.from_numpy(pred_ins[:valid_len])
            
            full_sem_pred.append(full_sem.numpy())
            full_ins_pred.append(full_ins.numpy())
        
        # Update evaluator
        self.evaluator.update(full_sem_pred, full_ins_pred, batch)
        
        return total_loss
    
    def validation_epoch_end(self, outputs):
        bs = self.cfg.TRAIN.BATCH_SIZE
        self.log("metrics/pq", self.evaluator.get_mean_pq(), batch_size=bs)
        self.log("metrics/iou", self.evaluator.get_mean_iou(), batch_size=bs)
        self.log("metrics/rq", self.evaluator.get_mean_rq(), batch_size=bs)
        self.evaluator.reset()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.TRAIN.LR)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=self.cfg.TRAIN.STEP, 
            gamma=self.cfg.TRAIN.DECAY
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def panoptic_inference(self, outputs, padding):
        """Panoptic segmentation inference (adapted from original)"""
        mask_cls = outputs["pred_logits"]
        mask_pred = outputs["pred_masks"]
        num_classes = self.cfg[self.cfg.MODEL.DATASET].NUM_CLASSES
        
        sem_pred = []
        ins_pred = []
        
        for b_cls, b_mask, b_pad in zip(mask_cls, mask_pred, padding):
            # Remove padding
            valid_mask = ~b_pad
            b_mask_valid = b_mask[valid_mask]
            
            scores, labels = b_cls.max(-1)
            b_mask_valid = b_mask_valid.sigmoid()
            
            keep = labels.ne(num_classes)
            
            if keep.sum() == 0:
                sem = torch.zeros(b_mask_valid.shape[0], dtype=torch.long)
                ins = torch.zeros(b_mask_valid.shape[0], dtype=torch.long)
            else:
                cur_scores = scores[keep]
                cur_classes = labels[keep]
                cur_masks = b_mask_valid[:, keep]
                
                # Get predictions
                cur_prob_masks = cur_scores.unsqueeze(0) * cur_masks
                mask_ids = cur_prob_masks.argmax(1)
                
                # Generate semantic and instance
                sem = torch.zeros(b_mask_valid.shape[0], dtype=torch.long)
                ins = torch.zeros(b_mask_valid.shape[0], dtype=torch.long)
                
                instance_id = 1
                for k in range(cur_classes.shape[0]):
                    mask = (mask_ids == k) & (cur_masks[:, k] >= 0.5)
                    if mask.sum() > 0:
                        sem[mask] = cur_classes[k]
                        if cur_classes[k].item() in self.things_ids:
                            ins[mask] = instance_id
                            instance_id += 1
            
            sem_pred.append(sem.cpu().numpy())
            ins_pred.append(ins.cpu().numpy())
        
        return sem_pred, ins_pred


def getDir(obj):
    return os.path.dirname(os.path.abspath(obj))


@click.command()
@click.option("--checkpoint", type=str, default=None, help="Resume from checkpoint")
@click.option("--nuscenes", is_flag=True, help="Use NuScenes dataset")
@click.option("--epochs", type=int, default=100, help="Number of epochs")
@click.option("--batch_size", type=int, default=1, help="Batch size")
@click.option("--lr", type=float, default=0.0001, help="Learning rate")
@click.option("--gpus", type=int, default=1, help="Number of GPUs")
def main(checkpoint, nuscenes, epochs, batch_size, lr, gpus):
    # Load configurations
    model_cfg = edict(
        yaml.safe_load(open(join(getDir(__file__), "../config/model.yaml")))
    )
    backbone_cfg = edict(
        yaml.safe_load(open(join(getDir(__file__), "../config/backbone.yaml")))
    )
    decoder_cfg = edict(
        yaml.safe_load(open(join(getDir(__file__), "../config/decoder.yaml")))
    )
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    
    # Update config with command line args
    cfg.TRAIN.MAX_EPOCH = epochs
    cfg.TRAIN.BATCH_SIZE = batch_size
    cfg.TRAIN.LR = lr
    cfg.TRAIN.N_GPUS = gpus
    
    if nuscenes:
        cfg.MODEL.DATASET = "NUSCENES"
    
    # Update experiment ID
    cfg.EXPERIMENT.ID = cfg.EXPERIMENT.ID + "_simplified"
    
    print("Training Simplified MaskPLS Model")
    print(f"Dataset: {cfg.MODEL.DATASET}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"GPUs: {gpus}")
    
    # Create data module
    data = SemanticDatasetModule(cfg)
    
    # Create model
    model = SimplifiedMaskPLS(cfg)
    
    # Load checkpoint if provided
    if checkpoint:
        print(f"Loading checkpoint from {checkpoint}")
        ckpt = torch.load(checkpoint, map_location='cpu')
        if 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
        else:
            model.model.load_state_dict(ckpt)
    
    # Setup logger
    tb_logger = pl_loggers.TensorBoardLogger(
        "experiments/" + cfg.EXPERIMENT.ID, 
        default_hp_metric=False
    )
    
    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    iou_ckpt = ModelCheckpoint(
        monitor="metrics/iou",
        filename=cfg.EXPERIMENT.ID + "_epoch{epoch:02d}_iou{metrics/iou:.2f}",
        auto_insert_metric_name=False,
        mode="max",
        save_last=True,
    )
    
    pq_ckpt = ModelCheckpoint(
        monitor="metrics/pq",
        filename=cfg.EXPERIMENT.ID + "_epoch{epoch:02d}_pq{metrics/pq:.2f}",
        auto_insert_metric_name=False,
        mode="max",
        save_last=True,
    )
    
    # Create trainer
    trainer = Trainer(
        gpus=cfg.TRAIN.N_GPUS,
        accelerator="ddp" if cfg.TRAIN.N_GPUS > 1 else None,
        logger=tb_logger,
        max_epochs=cfg.TRAIN.MAX_EPOCH,
        callbacks=[lr_monitor, pq_ckpt, iou_ckpt],
        log_every_n_steps=1,
        gradient_clip_val=0.5,
        accumulate_grad_batches=cfg.TRAIN.BATCH_ACC,
        resume_from_checkpoint=checkpoint if checkpoint and os.path.exists(checkpoint) else None,
    )
    
    # Train
    trainer.fit(model, data)
    
    print(f"\nTraining complete! Checkpoints saved in: experiments/{cfg.EXPERIMENT.ID}")


if __name__ == "__main__":
    main()
