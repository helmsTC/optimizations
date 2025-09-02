# mask_pls/models/dgcnn/maskpls_dgcnn.py
"""
MaskPLS with DGCNN backbone for ONNX-compatible panoptic segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Fix the import - use the correct module
try:
    from pytorch_lightning import LightningModule
except ImportError:
    from lightning.pytorch import LightningModule

from mask_pls.models.dgcnn.dgcnn_backbone import DGCNNPretrainedBackbone
from mask_pls.models.decoder import MaskedTransformerDecoder
from mask_pls.models.loss import MaskLoss, SemLoss
from mask_pls.utils.evaluate_panoptic import PanopticEvaluator


class MaskPLSDGCNN(LightningModule):  # Make sure it inherits from LightningModule
    """
    MaskPLS model with DGCNN backbone
    """
    def __init__(self, cfg):
        super(MaskPLSDGCNN, self).__init__()  # Explicit super call
        self.save_hyperparameters(dict(cfg))
        self.cfg = cfg
        
        dataset = cfg.MODEL.DATASET
        self.num_classes = cfg[dataset].NUM_CLASSES
        self.things_ids = []  # Will be set by datamodule
        
        # DGCNN backbone with pretrained weights
        pretrained_path = cfg.get('PRETRAINED_PATH', None)
        self.backbone = DGCNNPretrainedBackbone(cfg.BACKBONE, pretrained_path)
        
        # Transformer decoder
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
        self.warmup_steps = cfg.TRAIN.get('WARMUP_STEPS', 1000)
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
        
        # Semantic losses with proper shape handling
        sem_labels = []
        batch_size = len(x['sem_label'])
        
        for i in range(batch_size):
            labels = x['sem_label'][i]
            if isinstance(labels, np.ndarray):
                labels = torch.from_numpy(labels).long().cuda()
            
            # CRITICAL FIX: Ensure labels are 1D
            while labels.dim() > 1:
                if labels.shape[-1] == 1:
                    labels = labels.squeeze(-1)
                else:
                    labels = labels.reshape(-1)
            
            # Get valid mask for this sample
            valid_mask = ~padding[i]
            
            # Apply valid mask
            if valid_mask.sum() > 0:
                min_len = min(labels.shape[0], valid_mask.shape[0])
                valid_labels = labels[:min_len][valid_mask[:min_len]]
                sem_labels.append(valid_labels)
        
        if sem_labels:
            sem_labels = torch.cat(sem_labels)
            
            # Ensure sem_labels is 1D
            if sem_labels.dim() != 1:
                sem_labels = sem_labels.reshape(-1)
            
            # Get valid semantic logits
            sem_logits_valid = []
            for i in range(batch_size):
                valid_mask = ~padding[i]
                if valid_mask.sum() > 0:
                    sem_logits_valid.append(sem_logits[i][valid_mask])
            
            if sem_logits_valid:
                sem_logits_valid = torch.cat(sem_logits_valid, dim=0)
                
                assert sem_labels.dim() == 1, f"sem_labels must be 1D, got shape {sem_labels.shape}"
                assert sem_logits_valid.dim() == 2, f"sem_logits_valid must be 2D, got shape {sem_logits_valid.shape}"
                
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
        """Training step - required by Lightning"""
        outputs, padding, sem_logits = self(batch)
        losses = self.get_losses(batch, outputs, padding, sem_logits)
        
        # Log individual losses
        for k, v in losses.items():
            self.log(f'train/{k}', v, batch_size=self.cfg.TRAIN.BATCH_SIZE)
        
        # Total loss
        total_loss = sum(losses.values())
        self.log('train_loss', total_loss, batch_size=self.cfg.TRAIN.BATCH_SIZE)
        
        # Update step counter
        self.current_step += 1
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        outputs, padding, sem_logits = self(batch)
        losses = self.get_losses(batch, outputs, padding, sem_logits)
        
        # Log losses
        for k, v in losses.items():
            self.log(f'val/{k}', v, batch_size=self.cfg.TRAIN.BATCH_SIZE)
        
        total_loss = sum(losses.values())
        self.log('val_loss', total_loss, batch_size=self.cfg.TRAIN.BATCH_SIZE)
        
        # Panoptic inference
        sem_pred, ins_pred = self.panoptic_inference(outputs, padding)
        self.evaluator.update(sem_pred, ins_pred, batch)
        
        self.validation_step_outputs.append(total_loss)
        return total_loss
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch"""
        # Compute metrics
        pq = self.evaluator.get_mean_pq()
        iou = self.evaluator.get_mean_iou()
        rq = self.evaluator.get_mean_rq()
        
        self.log('metrics/pq', pq)
        self.log('metrics/iou', iou)
        self.log('metrics/rq', rq)
        
        print(f"\nValidation Metrics - PQ: {pq:.4f}, IoU: {iou:.4f}, RQ: {rq:.4f}")
        
        # Reset
        self.evaluator.reset()
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizers - required by Lightning"""
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
        
        # Cosine annealing with warmup
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.cfg.TRAIN.MAX_EPOCH
            ),
            'interval': 'epoch'
        }
        
        return [optimizer], [scheduler]
    
    def optimizer_step(self, *args, **kwargs):
        """Custom optimizer step for warmup"""
        # Learning rate warmup
        if self.current_step < self.warmup_steps:
            lr_scale = min(1.0, float(self.current_step) / float(self.warmup_steps))
            for pg in self.optimizers().param_groups:
                if 'lr' in pg:
                    pg['lr'] = lr_scale * self.cfg.TRAIN.LR
        
        # Call parent optimizer_step
        super().optimizer_step(*args, **kwargs)
    
    def panoptic_inference(self, outputs, padding):
        """Panoptic segmentation inference"""
        mask_cls = outputs["pred_logits"]
        mask_pred = outputs["pred_masks"]
        
        sem_pred = []
        ins_pred = []
        
        for b in range(mask_cls.shape[0]):
            scores, labels = mask_cls[b].max(-1)
            mask_pred_b = mask_pred[b][~padding[b]].sigmoid()
            
            # Filter valid predictions
            keep = labels.ne(self.num_classes)
            
            if keep.sum() == 0:
                sem_pred.append(np.zeros(mask_pred_b.shape[0], dtype=np.int32))
                ins_pred.append(np.zeros(mask_pred_b.shape[0], dtype=np.int32))
                continue
            
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred_b[:, keep]
            
            # Get instance masks
            cur_prob_masks = cur_scores.unsqueeze(0) * cur_masks
            panoptic_seg = torch.zeros(cur_masks.shape[0], dtype=torch.int32, 
                                     device=cur_masks.device)
            segments_info = []
            
            current_segment_id = 0
            cur_mask_ids = cur_prob_masks.argmax(1)
            
            # Process each mask
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.things_ids
                mask = (cur_mask_ids == k) & (cur_masks[:, k] >= 0.5)
                mask_area = mask.sum().item()
                
                if mask_area > 0:
                    if isthing:
                        current_segment_id += 1
                        panoptic_seg[mask] = current_segment_id
                    else:
                        if pred_class not in stuff_memory_list:
                            current_segment_id += 1
                            stuff_memory_list[pred_class] = current_segment_id
                        panoptic_seg[mask] = stuff_memory_list[pred_class]
                    
                    segments_info.append({
                        "id": current_segment_id,
                        "isthing": isthing,
                        "category_id": pred_class
                    })
            
            # Convert to numpy
            sem = torch.zeros_like(panoptic_seg)
            ins = torch.zeros_like(panoptic_seg)
            
            for seg_info in segments_info:
                mask = panoptic_seg == seg_info["id"]
                sem[mask] = seg_info["category_id"]
                if seg_info["isthing"]:
                    ins[mask] = seg_info["id"]
            
            sem_pred.append(sem.cpu().numpy())
            ins_pred.append(ins.cpu().numpy())
        
        return sem_pred, ins_pred