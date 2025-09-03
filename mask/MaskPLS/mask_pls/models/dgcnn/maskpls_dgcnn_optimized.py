# mask_pls/models/dgcnn/maskpls_dgcnn_optimized.py
"""
Optimized MaskPLS with DGCNN backbone
Fixes mixed precision and target dimension issues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_lightning.core.lightning import LightningModule

from mask_pls.models.dgcnn.dgcnn_backbone_efficient import EfficientDGCNNBackbone
from mask_pls.models.decoder import MaskedTransformerDecoder
from mask_pls.models.loss import MaskLoss, SemLoss
from mask_pls.utils.evaluate_panoptic import PanopticEvaluator


class MaskPLSDGCNNOptimized(LightningModule):
    """
    Optimized MaskPLS model with DGCNN backbone
    """
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(dict(cfg))
        self.cfg = cfg
        
        dataset = cfg.MODEL.DATASET
        self.num_classes = cfg[dataset].NUM_CLASSES
        self.ignore_label = cfg[dataset].IGNORE_LABEL
        self.things_ids = []  # Will be set by datamodule
        
        # Initialize backbone
        pretrained_path = cfg.get('PRETRAINED_PATH', None)
        self.backbone = EfficientDGCNNBackbone(cfg.BACKBONE, pretrained_path)
        
        # Initialize decoder
        self.decoder = MaskedTransformerDecoder(
            cfg.DECODER,
            cfg.BACKBONE,
            cfg[dataset]
        )
        
        # Initialize losses
        self.mask_loss = MaskLoss(cfg.LOSS, cfg[dataset])
        self.sem_loss = SemLoss(cfg.LOSS.SEM.WEIGHTS)
        
        # Initialize evaluator
        self.evaluator = PanopticEvaluator(cfg[dataset], dataset)
        
        # For validation outputs
        self.validation_step_outputs = []
        
        # Learning rate warmup
        self.warmup_steps = cfg.TRAIN.get('WARMUP_STEPS', 1000)
        self.current_step = 0
        
        # Mixed precision handling
        self.use_amp = cfg.TRAIN.get('MIXED_PRECISION', False)
    
    def forward(self, x):
        # Forward pass without mixed precision for stability
        feats, coords, pad_masks, sem_logits = self.backbone(x)
        outputs, padding = self.decoder(feats, coords, pad_masks)
        
        return outputs, padding, sem_logits, coords  # Return coords for tracking subsampling
    
    def get_losses(self, x, outputs, padding, sem_logits, coords):
        """Compute all losses with proper target formatting and subsampling handling"""
        # Prepare mask loss targets
        targets = {'classes': x['masks_cls'], 'masks': x['masks']}
        
        # Compute mask loss - let the loss function handle the coordinate matching
        loss_mask = self.mask_loss(outputs, targets, x['masks_ids'], x['pt_coord'])
        
        # Prepare semantic loss targets with subsampling awareness
        sem_labels = []
        valid_sem_logits = []
        
        batch_size = len(x['sem_label'])
        
        for i in range(batch_size):
            # Get semantic labels
            labels = x['sem_label'][i]
            
            # Convert to tensor if numpy
            if isinstance(labels, np.ndarray):
                labels = torch.from_numpy(labels).long().cuda()
            
            # Ensure labels are 1D (flatten if needed)
            if labels.dim() > 1:
                labels = labels.squeeze(-1)
            
            # Get valid mask for this sample
            valid_mask = ~padding[i]
            num_valid = valid_mask.sum().item()
            
            # Handle subsampling: Use consistent approach for training and validation
            original_num_points = labels.shape[0]
            backbone_num_points = valid_mask.shape[0]
            
            if original_num_points > backbone_num_points:
                # The backbone subsampled - use deterministic subsampling for consistency
                indices = torch.linspace(0, original_num_points-1, backbone_num_points).long()
                labels = labels[indices]
            elif original_num_points < backbone_num_points:
                # This shouldn't happen, but handle it gracefully
                print(f"Warning: Labels have fewer points than backbone output")
                continue
            
            # Now apply the valid mask
            valid_labels = labels[valid_mask]
            valid_logits = sem_logits[i][valid_mask]
            
            # Collect for loss computation
            if valid_labels.numel() > 0:
                sem_labels.append(valid_labels)
                valid_sem_logits.append(valid_logits)
        
        # Compute semantic loss
        if sem_labels:
            # Concatenate all valid labels and logits
            sem_labels = torch.cat(sem_labels, dim=0)
            valid_sem_logits = torch.cat(valid_sem_logits, dim=0)
            
            # Ensure correct dimensions
            assert sem_labels.dim() == 1, f"Labels should be 1D, got {sem_labels.dim()}D"
            assert valid_sem_logits.dim() == 2, f"Logits should be 2D, got {valid_sem_logits.dim()}D"
            
            # Compute loss
            loss_sem = self.sem_loss(valid_sem_logits, sem_labels)
        else:
            # No valid points
            loss_sem = {
                'sem_ce': torch.tensor(0.0, device=self.device),
                'sem_lov': torch.tensor(0.0, device=self.device)
            }
        
        # Combine losses
        loss_mask.update(loss_sem)
        return loss_mask
    
    def training_step(self, batch, batch_idx):
        # Forward pass
        outputs, padding, sem_logits, coords = self(batch)
        
        # Compute losses
        losses = self.get_losses(batch, outputs, padding, sem_logits, coords)
        
        # Log individual losses
        for k, v in losses.items():
            self.log(f'train/{k}', v, batch_size=self.cfg.TRAIN.BATCH_SIZE, prog_bar=True)
        
        # Total loss
        total_loss = sum(losses.values())
        self.log('train_loss', total_loss, batch_size=self.cfg.TRAIN.BATCH_SIZE, prog_bar=True)
        
        # Update step counter
        self.current_step += 1
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        # Forward pass
        outputs, padding, sem_logits, coords = self(batch)
        
        # Compute losses
        losses = self.get_losses(batch, outputs, padding, sem_logits, coords)
        
        # Log losses
        for k, v in losses.items():
            self.log(f'val/{k}', v, batch_size=self.cfg.TRAIN.BATCH_SIZE)
        
        total_loss = sum(losses.values())
        self.log('val_loss', total_loss, batch_size=self.cfg.TRAIN.BATCH_SIZE)
        
        # Panoptic inference
        sem_pred, ins_pred = self.panoptic_inference(outputs, padding)
        
        # Need to subsample ground truth to match prediction size for evaluation
        # The evaluator expects predictions and ground truth to have same dimensions
        subsampled_batch = self.subsample_ground_truth_for_eval(batch, sem_pred)
        self.evaluator.update(sem_pred, ins_pred, subsampled_batch)
        
        self.validation_step_outputs.append(total_loss)
        return total_loss
    
    def on_validation_epoch_end(self):
        # Compute metrics
        pq = self.evaluator.get_mean_pq()
        iou = self.evaluator.get_mean_iou()
        rq = self.evaluator.get_mean_rq()
        
        self.log('metrics/pq', pq, batch_size=self.cfg.TRAIN.BATCH_SIZE)
        self.log('metrics/iou', iou, batch_size=self.cfg.TRAIN.BATCH_SIZE)
        self.log('metrics/rq', rq, batch_size=self.cfg.TRAIN.BATCH_SIZE)
        
        print(f"\nValidation Metrics - PQ: {pq:.4f}, IoU: {iou:.4f}, RQ: {rq:.4f}")
        
        # Reset
        self.evaluator.reset()
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        # Separate parameters for backbone and decoder
        backbone_params = []
        decoder_params = []
        
        for name, param in self.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                decoder_params.append(param)
        
        # Different learning rates
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': self.cfg.TRAIN.LR * 0.1},
            {'params': decoder_params, 'lr': self.cfg.TRAIN.LR}
        ], weight_decay=1e-4)
        
        # Scheduler
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.cfg.TRAIN.MAX_EPOCH
            ),
            'interval': 'epoch'
        }
        
        return [optimizer], [scheduler]
    
    def optimizer_step(self, *args, **kwargs):
        # Learning rate warmup
        if self.current_step < self.warmup_steps:
            lr_scale = min(1.0, float(self.current_step) / float(self.warmup_steps))
            for pg in self.optimizers().param_groups:
                pg['lr'] = lr_scale * self.cfg.TRAIN.LR
        
        super().optimizer_step(*args, **kwargs)
    
    def panoptic_inference(self, outputs, padding):
        """Panoptic segmentation inference"""
        mask_cls = outputs["pred_logits"]
        mask_pred = outputs["pred_masks"]
        
        batch_size = mask_cls.shape[0]
        sem_pred = []
        ins_pred = []
        
        for b in range(batch_size):
            # Get valid points (not padded)
            valid_mask = ~padding[b]
            num_valid = valid_mask.sum().item()
            
            if num_valid == 0:
                # No valid points
                sem_pred.append(np.zeros(0, dtype=np.int32))
                ins_pred.append(np.zeros(0, dtype=np.int32))
                continue
            
            # Get predictions for this sample
            scores, labels = mask_cls[b].max(-1)
            mask_pred_b = mask_pred[b][valid_mask].sigmoid()
            
            # Filter valid predictions (not background)
            keep = labels.ne(self.num_classes)
            
            if keep.sum() == 0:
                # No valid masks
                sem_pred.append(np.zeros(num_valid, dtype=np.int32))
                ins_pred.append(np.zeros(num_valid, dtype=np.int32))
                continue
            
            # Get valid masks
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred_b[:, keep]
            
            # Get instance masks
            cur_prob_masks = cur_scores.unsqueeze(0) * cur_masks
            
            # Initialize panoptic segmentation
            panoptic_seg = torch.zeros(num_valid, dtype=torch.int32, device=cur_masks.device)
            semantic_seg = torch.zeros(num_valid, dtype=torch.int32, device=cur_masks.device)
            instance_seg = torch.zeros(num_valid, dtype=torch.int32, device=cur_masks.device)
            
            current_segment_id = 0
            
            # Get mask assignments
            if cur_masks.shape[1] > 0:
                cur_mask_ids = cur_prob_masks.argmax(1)
                
                # Process each mask
                stuff_memory_list = {}
                
                for k in range(cur_classes.shape[0]):
                    pred_class = cur_classes[k].item()
                    isthing = pred_class in self.things_ids
                    
                    # Get mask
                    mask = (cur_mask_ids == k) & (cur_masks[:, k] >= 0.5)
                    mask_area = mask.sum().item()
                    
                    if mask_area > 0:
                        if isthing:
                            # Thing class - new instance
                            current_segment_id += 1
                            instance_seg[mask] = current_segment_id
                            semantic_seg[mask] = pred_class
                            panoptic_seg[mask] = current_segment_id
                        else:
                            # Stuff class - merge with existing
                            if pred_class not in stuff_memory_list:
                                current_segment_id += 1
                                stuff_memory_list[pred_class] = current_segment_id
                            
                            semantic_seg[mask] = pred_class
                            panoptic_seg[mask] = stuff_memory_list[pred_class]
            
            # Convert to numpy
            sem_pred.append(semantic_seg.cpu().numpy())
            ins_pred.append(instance_seg.cpu().numpy())
        
        return sem_pred, ins_pred
    
    def subsample_ground_truth_for_eval(self, batch, sem_pred):
        """Subsample ground truth labels to match prediction dimensions for evaluation"""
        subsampled_batch = {}
        subsampled_batch['fname'] = batch['fname']
        
        subsampled_sem_labels = []
        subsampled_ins_labels = []
        
        for i in range(len(sem_pred)):
            pred_size = len(sem_pred[i])
            
            # Get original labels
            sem_label = batch['sem_label'][i]
            ins_label = batch['ins_label'][i]
            
            # Convert to tensors if needed
            if isinstance(sem_label, np.ndarray):
                sem_label = torch.from_numpy(sem_label)
            if isinstance(ins_label, np.ndarray):
                ins_label = torch.from_numpy(ins_label)
            
            # Flatten if needed
            if sem_label.dim() > 1:
                sem_label = sem_label.squeeze(-1)
            if ins_label.dim() > 1:
                ins_label = ins_label.squeeze(-1)
            
            original_size = len(sem_label)
            
            if pred_size < original_size:
                # Subsample ground truth to match predictions using same deterministic approach
                indices = torch.linspace(0, original_size-1, pred_size).long()
                subsampled_sem = sem_label[indices]
                subsampled_ins = ins_label[indices]
            else:
                # No subsampling needed or pred is larger (shouldn't happen)
                subsampled_sem = sem_label[:pred_size]
                subsampled_ins = ins_label[:pred_size]
            
            subsampled_sem_labels.append(subsampled_sem.cpu().numpy())
            subsampled_ins_labels.append(subsampled_ins.cpu().numpy())
        
        subsampled_batch['sem_label'] = subsampled_sem_labels
        subsampled_batch['ins_label'] = subsampled_ins_labels
        
        return subsampled_batch