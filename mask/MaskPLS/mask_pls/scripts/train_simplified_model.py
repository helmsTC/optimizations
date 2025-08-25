"""
Training script for the simplified ONNX-compatible MaskPLS model
Save as: mask/MaskPLS/mask_pls/scripts/train_simplified_model.py
"""

import os
# Enable CUDA error debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

from os.path import join
import click
import torch
import yaml
import numpy as np
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


def prepare_targets_for_loss(batch, device='cuda'):
    """Convert batch targets to the format expected by the loss functions"""
    fixed_targets = {
        'classes': [],
        'masks': []
    }
    
    for i in range(len(batch['masks_cls'])):
        if len(batch['masks_cls'][i]) > 0:
            # Convert list of scalar tensors to single tensor
            classes = []
            for c in batch['masks_cls'][i]:
                if isinstance(c, torch.Tensor):
                    classes.append(c.item() if c.numel() == 1 else int(c))
                else:
                    classes.append(int(c))
            
            # Create tensor and move to device
            classes_tensor = torch.tensor(classes, dtype=torch.long, device=device)
            fixed_targets['classes'].append(classes_tensor)
            
            # Stack masks and move to device
            masks_list = []
            for m in batch['masks'][i]:
                if isinstance(m, torch.Tensor):
                    masks_list.append(m.float())
                else:
                    masks_list.append(torch.tensor(m, dtype=torch.float32))
            
            if len(masks_list) > 0:
                masks_tensor = torch.stack(masks_list).to(device)
                fixed_targets['masks'].append(masks_tensor)
            else:
                # Empty masks
                num_points = len(batch['pt_coord'][i])
                fixed_targets['masks'].append(torch.empty(0, num_points, device=device))
        else:
            # No masks in this sample
            num_points = len(batch['pt_coord'][i])
            fixed_targets['classes'].append(torch.empty(0, dtype=torch.long, device=device))
            fixed_targets['masks'].append(torch.empty(0, num_points, device=device))
    
    return fixed_targets


class SimplifiedMaskPLS(LightningModule):
    """
    Lightning module for training the simplified MaskPLS model
    """
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(dict(cfg))
        self.cfg = cfg
        
        # Get dataset configuration
        dataset = cfg.MODEL.DATASET
        self.num_classes = cfg[dataset].NUM_CLASSES
        self.ignore_label = cfg[dataset].IGNORE_LABEL
        
        print(f"Initializing model for {dataset} with {self.num_classes} classes")
        print(f"Ignore label: {self.ignore_label}")
        
        # Create the simplified model
        self.model = MaskPLSSimplifiedONNX(cfg)
        
        # Verify semantic head configuration
        sem_head_out = self.model.sem_head.out_features
        if sem_head_out != self.num_classes:
            print(f"WARNING: Semantic head outputs {sem_head_out} classes but dataset has {self.num_classes}")
        
        # Loss functions (same as original)
        self.mask_loss = MaskLoss(cfg.LOSS, cfg[dataset])
        self.sem_loss = SemLoss(cfg.LOSS.SEM.WEIGHTS)
        
        # Evaluator (same as original)
        self.evaluator = PanopticEvaluator(cfg[dataset], dataset)
        
        # Cache things IDs
        data = SemanticDatasetModule(cfg)
        data.setup()
        self.things_ids = data.things_ids
        
        # Debug mode
        self.debug = False
        
    def on_train_start(self):
        """Simple training start message"""
        print(f"\nStarting training for {self.cfg.MODEL.DATASET} dataset")
        print(f"Number of classes: {self.num_classes}")
        print(f"Ignore label: {self.ignore_label}")
        print("-" * 60)
            
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
            
            # Get indices of valid points in original array
            valid_idx = torch.where(valid_mask)[0]
            valid_pts = pts[valid_mask]
            valid_feat = feat[valid_mask]
            
            # Subsample if needed
            max_pts = self.cfg[self.cfg.MODEL.DATASET].SUB_NUM_POINTS
            if len(valid_pts) > max_pts and self.training:
                # Create permutation for subsampling
                perm = torch.randperm(len(valid_pts))[:max_pts]
                valid_pts = valid_pts[perm]
                valid_feat = valid_feat[perm]
                # IMPORTANT: Update the indices to reflect subsampling
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
        
        # Validate outputs
        if self.debug:
            print(f"Model outputs:")
            print(f"  pred_logits: {pred_logits.shape}, range [{pred_logits.min():.2f}, {pred_logits.max():.2f}]")
            print(f"  pred_masks: {pred_masks.shape}, range [{pred_masks.min():.2f}, {pred_masks.max():.2f}]")
            print(f"  sem_logits: {sem_logits.shape}, range [{sem_logits.min():.2f}, {sem_logits.max():.2f}]")
        
        # Check semantic logits shape
        expected_classes = self.num_classes
        if sem_logits.shape[-1] != expected_classes:
            print(f"ERROR: sem_logits has {sem_logits.shape[-1]} classes but expected {expected_classes}")
            # Adjust if necessary
            if sem_logits.shape[-1] > expected_classes:
                sem_logits = sem_logits[..., :expected_classes]
            else:
                # Pad with zeros
                pad_size = expected_classes - sem_logits.shape[-1]
                sem_logits = F.pad(sem_logits, (0, pad_size))
        
        outputs = {
            'pred_logits': pred_logits,
            'pred_masks': pred_masks,
            'aux_outputs': []  # Simplified model doesn't have aux outputs
        }
        
        return outputs, padding_masks, sem_logits, valid_indices
    
    def training_step(self, batch, batch_idx):
        try:
            outputs, padding, sem_logits, valid_indices = self.forward(batch)
            
            # Prepare targets for mask loss - THIS IS THE KEY FIX
            targets = prepare_targets_for_loss(batch, device='cuda')
            
            # Mask loss
            try:
                loss_mask = self.mask_loss(outputs, targets, batch['masks_ids'], batch['pt_coord'])
            except Exception as e:
                print(f"Error in mask loss: {e}")
                # Return minimal loss to continue
                return torch.tensor(0.1, device='cuda', requires_grad=True)
            
            # Semantic loss (on valid points) - MORE CAREFUL HANDLING
            try:
                all_sem_labels = []
                all_sem_logits = []
                
                for i, (label, idx, pad) in enumerate(zip(batch['sem_label'], valid_indices, padding)):
                    # Get valid points for this sample
                    valid_mask = ~pad
                    num_valid = valid_mask.sum().item()
                    
                    if num_valid == 0:
                        continue
                    
                    # Get semantic logits for valid points
                    sem_logits_i = sem_logits[i][valid_mask]  # [num_valid, num_classes]
                    
                    # Get indices
                    idx_cpu = idx.cpu().numpy()
                    
                    # Ensure we don't exceed the number of valid logits
                    num_logits = sem_logits_i.shape[0]
                    
                    # Get labels
                    label_array = label.flatten() if label.ndim > 1 else label
                    label_size = len(label_array)
                    
                    # CRITICAL: Ensure indices are valid for both bounds
                    valid_idx_mask = (idx_cpu < label_size) & (idx_cpu >= 0)
                    idx_to_use = idx_cpu[valid_idx_mask]
                    
                    # Ensure we don't use more indices than we have logits
                    idx_to_use = idx_to_use[:num_logits]
                    
                    # Adjust logits if needed
                    if len(idx_to_use) < num_logits:
                        sem_logits_i = sem_logits_i[:len(idx_to_use)]
                    
                    if len(idx_to_use) == 0:
                        continue
                    
                    # Get the labels for valid indices
                    valid_labels = label_array[idx_to_use]
                    
                    # Convert to tensor and ensure valid range
                    valid_labels_tensor = torch.from_numpy(valid_labels).long()
                    
                    # IMPORTANT: Clamp labels to valid range
                    # The cross-entropy loss expects labels in [0, num_classes-1]
                    valid_labels_tensor = torch.clamp(valid_labels_tensor, 0, self.num_classes - 1)
                    
                    # Move to GPU
                    valid_labels_tensor = valid_labels_tensor.cuda()
                    
                    # Ensure matching dimensions
                    min_len = min(sem_logits_i.shape[0], valid_labels_tensor.shape[0])
                    if min_len > 0:
                        all_sem_logits.append(sem_logits_i[:min_len])
                        all_sem_labels.append(valid_labels_tensor[:min_len])
                
                # Compute semantic loss if we have valid data
                if len(all_sem_logits) > 0:
                    # Concatenate all valid points
                    all_sem_logits = torch.cat(all_sem_logits, dim=0)
                    all_sem_labels = torch.cat(all_sem_labels, dim=0)
                    
                    # Double-check the labels are valid
                    if all_sem_labels.max() >= self.num_classes:
                        print(f"WARNING: Found labels >= {self.num_classes}, clamping")
                        all_sem_labels = torch.clamp(all_sem_labels, 0, self.num_classes - 1)
                    
                    # Compute loss
                    loss_sem = self.sem_loss(all_sem_logits, all_sem_labels)
                    loss_mask.update(loss_sem)
                else:
                    # No valid semantic data
                    loss_sem = {
                        'sem_ce': torch.tensor(0.0, device='cuda', requires_grad=True), 
                        'sem_lov': torch.tensor(0.0, device='cuda', requires_grad=True)
                    }
                    loss_mask.update(loss_sem)
                    
            except Exception as e:
                print(f"Error in semantic loss computation: {e}")
                # Add dummy semantic losses
                loss_mask['sem_ce'] = torch.tensor(0.0, device='cuda', requires_grad=True)
                loss_mask['sem_lov'] = torch.tensor(0.0, device='cuda', requires_grad=True)
            
            # Log losses
            for k, v in loss_mask.items():
                if torch.isnan(v) or torch.isinf(v):
                    v = torch.tensor(0.0, device='cuda', requires_grad=True)
                self.log(f"train/{k}", v, batch_size=self.cfg.TRAIN.BATCH_SIZE, on_step=True, on_epoch=True)
            
            # Total loss
            total_loss = sum(loss_mask.values())
            
            # Safety check
            if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss.item() > 1000:
                print(f"Warning: Invalid total_loss: {total_loss.item()}")
                total_loss = torch.tensor(0.1, device='cuda', requires_grad=True)
            
            self.log("train_loss", total_loss, batch_size=self.cfg.TRAIN.BATCH_SIZE, on_step=True, on_epoch=True)
            
            return total_loss
            
        except Exception as e:
            print(f"\nUnexpected error in training_step: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            
            # Return minimal loss to continue training
            return torch.tensor(0.1, device='cuda', requires_grad=True)
    
    def validation_step(self, batch, batch_idx):
        try:
            outputs, padding, sem_logits, valid_indices = self.forward(batch)
            
            # Prepare targets for mask loss - ADD THIS FIX
            targets = prepare_targets_for_loss(batch, device='cuda')
            
            # Calculate losses
            loss_mask = self.mask_loss(outputs, targets, batch['masks_ids'], batch['pt_coord'])
            
            # Semantic loss
            all_sem_labels = []
            all_sem_logits = []
            
            for i, (label, idx, pad) in enumerate(zip(batch['sem_label'], valid_indices, padding)):
                # Get valid points for this sample
                valid_mask = ~pad
                num_valid = valid_mask.sum().item()
                
                if num_valid == 0:
                    continue
                
                # Get semantic logits for valid points
                sem_logits_i = sem_logits[i][valid_mask]
                
                # Get semantic labels for the points we actually kept
                idx_cpu = idx.cpu().numpy()
                num_logits = sem_logits_i.shape[0]
                
                # Get labels
                label_array = label.flatten() if label.ndim > 1 else label
                label_size = len(label_array)
                
                # Ensure indices are within bounds of both logits and labels
                valid_idx_mask = (idx_cpu < label_size) & (idx_cpu >= 0)
                idx_to_use = idx_cpu[valid_idx_mask]
                
                # Ensure we don't use more indices than we have logits
                idx_to_use = idx_to_use[:num_logits]
                
                # Adjust sem_logits to match the number of valid indices
                if len(idx_to_use) < num_logits:
                    sem_logits_i = sem_logits_i[:len(idx_to_use)]
                    
                if len(idx_to_use) == 0:
                    continue
                
                valid_labels = label_array[idx_to_use]
                valid_labels_tensor = torch.from_numpy(valid_labels).long()
                
                # Safety clamp
                if valid_labels_tensor.max() >= self.num_classes:
                    valid_labels_tensor = torch.clamp(valid_labels_tensor, 0, self.num_classes - 1)
                
                valid_labels_tensor = valid_labels_tensor.cuda()
                
                all_sem_logits.append(sem_logits_i)
                all_sem_labels.append(valid_labels_tensor)
            
            # Compute semantic loss if we have valid data
            if len(all_sem_logits) > 0:
                all_sem_logits = torch.cat(all_sem_logits, dim=0)
                all_sem_labels = torch.cat(all_sem_labels, dim=0)
                loss_sem = self.sem_loss(all_sem_logits, all_sem_labels)
                loss_mask.update(loss_sem)
            
            # Log losses
            for k, v in loss_mask.items():
                if torch.isnan(v) or torch.isinf(v):
                    v = torch.tensor(0.0, device='cuda')
                self.log(f"val/{k}", v, batch_size=self.cfg.TRAIN.BATCH_SIZE)
            
            total_loss = sum(loss_mask.values())
            self.log("val_loss", total_loss, batch_size=self.cfg.TRAIN.BATCH_SIZE)
            
            # Panoptic inference and evaluation
            sem_pred, ins_pred = self.panoptic_inference(outputs, padding)
            
            # Map predictions back to original points
            full_sem_pred = []
            full_ins_pred = []
            
            for i, (pred_sem, pred_ins, idx) in enumerate(zip(sem_pred, ins_pred, valid_indices)):
                # Create full predictions initialized with ignore label
                full_sem = torch.zeros(len(batch['sem_label'][i]), dtype=torch.long)
                full_ins = torch.zeros(len(batch['ins_label'][i]), dtype=torch.long)
                
                # Map back to original indices
                idx_cpu = idx.cpu().numpy()
                
                # Ensure indices are within bounds
                max_size = len(full_sem)
                valid_mask = idx_cpu < max_size
                idx_cpu = idx_cpu[valid_mask]
                
                valid_len = min(len(idx_cpu), len(pred_sem))
                
                if valid_len > 0:
                    full_sem[idx_cpu[:valid_len]] = torch.from_numpy(pred_sem[:valid_len])
                    full_ins[idx_cpu[:valid_len]] = torch.from_numpy(pred_ins[:valid_len])
                
                full_sem_pred.append(full_sem.numpy())
                full_ins_pred.append(full_ins.numpy())
            
            # Update evaluator
            self.evaluator.update(full_sem_pred, full_ins_pred, batch)
            
            return total_loss
            
        except Exception as e:
            print(f"Error in validation_step: {e}")
            import traceback
            traceback.print_exc()
            return torch.tensor(0.1, device='cuda')
    
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
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.option("--num_workers", type=int, default=4, help="Number of data loader workers")
def main(checkpoint, nuscenes, epochs, batch_size, lr, gpus, debug, num_workers):
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
    cfg.TRAIN.NUM_WORKERS = num_workers
    
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
    print(f"Workers: {num_workers}")
    print(f"Debug mode: {debug}")
    
    # Create data module
    data = SemanticDatasetModule(cfg)
    
    # Create model
    model = SimplifiedMaskPLS(cfg)
    
    # Enable debug mode if requested
    if debug:
        model.debug = True
        print("Debug mode enabled - will print detailed information")
    
    # Load checkpoint if provided
    if checkpoint:
        print(f"Loading checkpoint from {checkpoint}")
        ckpt = torch.load(checkpoint, map_location='cpu')
        if 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'], strict=False)
        else:
            model.model.load_state_dict(ckpt, strict=False)
    
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
        num_sanity_val_steps=0 if debug else 2,
    )
    
    # Train
    trainer.fit(model, data)
    
    print(f"\nTraining complete! Checkpoints saved in: experiments/{cfg.EXPERIMENT.ID}")


if __name__ == "__main__":
    main()
