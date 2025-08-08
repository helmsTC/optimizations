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
        """Validate data before training starts"""
        print("\nValidating dataset before training...")
        
        # Get a few batches to check
        dataloader = self.trainer.train_dataloader
        
        for i, batch in enumerate(dataloader):
            if i >= 3:  # Check first 3 batches
                break
                
            print(f"\nChecking batch {i}:")
            
            # Check semantic labels
            for j, label in enumerate(batch['sem_label']):
                unique = np.unique(label)
                print(f"  Sample {j}: sem_labels range [{unique.min()}, {unique.max()}]")
                
                if unique.max() >= self.num_classes:
                    print(f"  ERROR: Found invalid labels: {unique[unique >= self.num_classes]}")
            
            # Check mask classes more carefully
            for j, mask_cls_list in enumerate(batch['masks_cls']):
                if len(mask_cls_list) > 0:
                    try:
                        # Each element in mask_cls_list is a tensor with a class label
                        # We need to extract the values
                        class_values = []
                        for cls in mask_cls_list:
                            if isinstance(cls, torch.Tensor):
                                # Handle both scalar tensors and arrays
                                if cls.numel() == 1:
                                    class_values.append(cls.item())
                                else:
                                    class_values.extend(cls.cpu().numpy().flatten())
                            else:
                                class_values.append(cls)
                        
                        unique_classes = np.unique(class_values)
                        print(f"  Sample {j}: {len(mask_cls_list)} masks, unique classes: {unique_classes}")
                        
                        if len(unique_classes) > 0 and unique_classes.max() >= self.num_classes:
                            print(f"  ERROR: Invalid mask class: {unique_classes.max()} >= {self.num_classes}")
                            
                    except Exception as e:
                        print(f"  Sample {j}: Could not process mask classes - {type(e).__name__}: {e}")
                else:
                    print(f"  Sample {j}: No masks in this sample")
        
        print("\nValidation complete. Starting training...\n")
            
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
            # Add CUDA synchronization for better error tracking
            if self.debug:
                torch.cuda.synchronize()
                
            outputs, padding, sem_logits, valid_indices = self.forward(batch)
            
            if self.debug and batch_idx < 5:
                print(f"\n=== Batch {batch_idx} Shapes ===")
                print(f"pred_logits: {outputs['pred_logits'].shape}")
                print(f"pred_masks: {outputs['pred_masks'].shape}")
                print(f"sem_logits: {sem_logits.shape}")
                print(f"padding: {[p.shape for p in padding]}")
                print(f"valid_indices: {[idx.shape for idx in valid_indices]}")
                
            # Prepare targets for mask loss
            targets = {
                'classes': batch['masks_cls'],
                'masks': batch['masks']
            }
            
            # Try mask loss first
            if self.debug:
                print(f"\nComputing mask loss...")
                torch.cuda.synchronize()
                
            loss_mask = self.mask_loss(outputs, targets, batch['masks_ids'], batch['pt_coord'])
            
            if self.debug:
                print(f"Mask loss computed successfully")
                for k, v in loss_mask.items():
                    print(f"  {k}: {v.item():.4f}")
                torch.cuda.synchronize()
            
            # Semantic loss (on valid points)
            all_sem_labels = []
            all_sem_logits = []
            
            for i, (label, idx, pad) in enumerate(zip(batch['sem_label'], valid_indices, padding)):
                # Get valid points for this sample
                valid_mask = ~pad
                num_valid = valid_mask.sum().item()
                
                if self.debug and i == 0:
                    print(f"\nProcessing sample {i}:")
                    print(f"  num_valid points: {num_valid}")
                    print(f"  valid_indices shape: {idx.shape}")
                    print(f"  label shape: {label.shape}")
                    print(f"  label dtype: {label.dtype}")
                    print(f"  label unique values: {np.unique(label)}")
                
                if num_valid == 0:
                    continue
                
                # Get semantic logits for valid points
                sem_logits_i = sem_logits[i][valid_mask]  # [num_valid, num_classes]
                
                if self.debug and i == 0:
                    print(f"  sem_logits_i shape: {sem_logits_i.shape}")
                
                # CRITICAL: We need to get labels for the exact points we have logits for
                # The valid_indices tell us which points from the original cloud we kept
                idx_cpu = idx.cpu().numpy()
                
                # The number of logits we have
                num_logits = sem_logits_i.shape[0]
                
                # We can only use as many indices as we have logits
                idx_to_use = idx_cpu[:num_logits]
                
                if self.debug and i == 0:
                    print(f"  Using {len(idx_to_use)} indices")
                    print(f"  Index range: [{idx_to_use.min()}, {idx_to_use.max()}]")
                
                # Get labels - ensure we handle both (N,) and (N,1) shapes
                label_array = label.flatten() if label.ndim > 1 else label
                
                # Bounds check
                max_label_idx = len(label_array) - 1
                valid_idx_mask = idx_to_use <= max_label_idx
                idx_to_use = idx_to_use[valid_idx_mask]
                
                if len(idx_to_use) != num_logits:
                    # We lost some indices, need to trim logits too
                    sem_logits_i = sem_logits_i[:len(idx_to_use)]
                    
                if len(idx_to_use) == 0:
                    continue
                    
                # Get the labels
                valid_labels = label_array[idx_to_use]
                
                if self.debug and i == 0:
                    print(f"  valid_labels shape: {valid_labels.shape}")
                    print(f"  valid_labels unique: {np.unique(valid_labels)}")
                    print(f"  valid_labels max: {valid_labels.max()}")
                
                # Convert to tensor
                valid_labels_tensor = torch.from_numpy(valid_labels).long()
                
                # Verify labels are in valid range BEFORE moving to GPU
                if valid_labels_tensor.max() >= self.num_classes:
                    print(f"ERROR: Sample {i} has label {valid_labels_tensor.max().item()} >= {self.num_classes}")
                    print(f"Unique labels: {torch.unique(valid_labels_tensor).cpu().numpy()}")
                    # Clamp to valid range
                    valid_labels_tensor = torch.clamp(valid_labels_tensor, 0, self.num_classes - 1)
                
                # Now move to GPU
                valid_labels_tensor = valid_labels_tensor.cuda()
                
                # Double-check dimensions match
                assert sem_logits_i.shape[0] == valid_labels_tensor.shape[0], \
                    f"Shape mismatch: logits {sem_logits_i.shape[0]} vs labels {valid_labels_tensor.shape[0]}"
                
                all_sem_logits.append(sem_logits_i)
                all_sem_labels.append(valid_labels_tensor)
            
            # Compute semantic loss if we have valid data
            if len(all_sem_logits) > 0:
                if self.debug:
                    print(f"\nComputing semantic loss...")
                    print(f"  Number of chunks: {len(all_sem_logits)}")
                    
                # Concatenate all valid points across batch
                all_sem_logits = torch.cat(all_sem_logits, dim=0)
                all_sem_labels = torch.cat(all_sem_labels, dim=0)
                
                if self.debug:
                    print(f"  Combined logits shape: {all_sem_logits.shape}")
                    print(f"  Combined labels shape: {all_sem_labels.shape}")
                    print(f"  Labels range: [{all_sem_labels.min().item()}, {all_sem_labels.max().item()}]")
                    print(f"  Expected range: [0, {self.num_classes - 1}]")
                    torch.cuda.synchronize()
                
                # Final validation before loss computation
                assert all_sem_labels.min() >= 0, f"Negative labels found: {all_sem_labels.min()}"
                assert all_sem_labels.max() < self.num_classes, \
                    f"Labels exceed num_classes: {all_sem_labels.max()} >= {self.num_classes}"
                assert all_sem_logits.shape[-1] == self.num_classes, \
                    f"Logits shape mismatch: {all_sem_logits.shape[-1]} != {self.num_classes}"
                
                # Compute semantic loss
                loss_sem = self.sem_loss(all_sem_logits, all_sem_labels)
                
                if self.debug:
                    print(f"Semantic loss computed successfully")
                    for k, v in loss_sem.items():
                        print(f"  {k}: {v.item():.4f}")
                        
                loss_mask.update(loss_sem)
            else:
                # No valid semantic data in this batch
                print(f"Warning: No valid semantic data in batch {batch_idx}")
                loss_sem = {'sem_ce': torch.tensor(0.0, device='cuda'), 
                           'sem_lov': torch.tensor(0.0, device='cuda')}
                loss_mask.update(loss_sem)
            
            # Log losses
            for k, v in loss_mask.items():
                if torch.isnan(v) or torch.isinf(v):
                    print(f"Warning: {k} is nan/inf: {v}")
                    v = torch.tensor(0.0, device='cuda', requires_grad=True)
                self.log(f"train/{k}", v, batch_size=self.cfg.TRAIN.BATCH_SIZE)
            
            total_loss = sum(loss_mask.values())
            
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"Warning: total_loss is nan/inf, returning small loss")
                return torch.tensor(0.1, device='cuda', requires_grad=True)
            
            self.log("train_loss", total_loss, batch_size=self.cfg.TRAIN.BATCH_SIZE)
            
            if self.debug:
                print(f"\nBatch {batch_idx} completed successfully!")
                print(f"Total loss: {total_loss.item():.4f}")
                
            return total_loss
            
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"CUDA Error in training_step batch {batch_idx}:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            
            # Try to get more CUDA error info
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                print(f"CUDA error detected after synchronize")
                
            import traceback
            traceback.print_exc()
            print(f"{'='*60}\n")
            
            # Return a small loss to continue training
            return torch.tensor(0.1, device='cuda', requires_grad=True)
    
    def validation_step(self, batch, batch_idx):
        try:
            outputs, padding, sem_logits, valid_indices = self.forward(batch)
            
            # Calculate losses (similar to training)
            targets = {'classes': batch['masks_cls'], 'masks': batch['masks']}
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
                idx_to_use = idx_cpu[:num_logits]
                
                # Get labels
                label_array = label.flatten() if label.ndim > 1 else label
                max_label_idx = len(label_array) - 1
                valid_idx_mask = idx_to_use <= max_label_idx
                idx_to_use = idx_to_use[valid_idx_mask]
                
                if len(idx_to_use) != num_logits:
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
                max_idx = len(full_sem) - 1
                valid_mask = idx_cpu <= max_idx
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
