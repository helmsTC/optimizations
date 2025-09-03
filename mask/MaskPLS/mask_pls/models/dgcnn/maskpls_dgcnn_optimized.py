# mask_pls/models/dgcnn/maskpls_dgcnn_fixed.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_lightning.core.lightning import LightningModule

from mask_pls.models.dgcnn.dgcnn_backbone_efficient_fixed import FixedDGCNNBackbone
from mask_pls.models.decoder import MaskedTransformerDecoder
from mask_pls.models.loss import MaskLoss, SemLoss
from mask_pls.utils.evaluate_panoptic import PanopticEvaluator

class MaskPLSDGCNNFixed(LightningModule):
    """Fixed MaskPLS model with DGCNN backbone"""
    
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
        self.backbone = FixedDGCNNBackbone(cfg.BACKBONE, pretrained_path)
        self.backbone.set_num_classes(self.num_classes)
        
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
        
        # Training config
        self.warmup_steps = cfg.TRAIN.get('WARMUP_STEPS', 500)
        self.current_step = 0
    
    def forward(self, x):
        # DGCNN feature extraction with subsample tracking
        feats, coords, pad_masks, sem_logits = self.backbone(x)
        
        # Transformer decoder
        outputs, padding = self.decoder(feats, coords, pad_masks)
        
        return outputs, padding, sem_logits
    
    def get_losses(self, x, outputs, padding, sem_logits):
        """Compute losses with proper subsampling handling"""
        # Get subsample indices from backbone
        subsample_indices = self.backbone.subsample_indices
        
        # Prepare targets for mask loss
        targets = {
            'classes': x['masks_cls'],
            'masks': []
        }
        
        # Adjust masks for subsampling
        for b, masks in enumerate(x['masks']):
            if b in subsample_indices:
                indices = subsample_indices[b]
                # Subsample masks
                if masks.shape[0] > 0:
                    subsampled_masks = masks[:, indices]
                    targets['masks'].append(subsampled_masks)
                else:
                    targets['masks'].append(masks)
            else:
                targets['masks'].append(masks)
        
        # Compute mask loss
        loss_mask = self.mask_loss(outputs, targets, x['masks_ids'], x['pt_coord'])
        
        # Compute semantic loss
        sem_labels = []
        valid_sem_logits = []
        
        batch_size = len(x['sem_label'])
        
        for b in range(batch_size):
            labels = x['sem_label'][b]
            
            # Convert to tensor
            if isinstance(labels, np.ndarray):
                labels = torch.from_numpy(labels).long().cuda()
            
            # Flatten
            if labels.dim() > 1:
                labels = labels.squeeze(-1)
            
            # Apply subsampling if needed
            if b in subsample_indices:
                indices = subsample_indices[b]
                if indices.shape[0] < labels.shape[0]:
                    labels = labels[indices]
            
            # Apply valid mask
            valid_mask = ~padding[b]
            num_valid = valid_mask.sum()
            
            if num_valid > 0:
                min_size = min(labels.shape[0], valid_mask.shape[0])
                valid_labels = labels[:min_size][valid_mask[:min_size]]
                valid_logits = sem_logits[b][valid_mask][:min_size]
                
                if valid_labels.numel() > 0:
                    sem_labels.append(valid_labels)
                    valid_sem_logits.append(valid_logits)
        
        # Compute semantic loss
        if sem_labels:
            sem_labels = torch.cat(sem_labels, dim=0)
            valid_sem_logits = torch.cat(valid_sem_logits, dim=0)
            
            # Ensure correct dimensions
            if sem_labels.dim() != 1:
                sem_labels = sem_labels.reshape(-1)
            
            loss_sem = self.sem_loss(valid_sem_logits, sem_labels)
        else:
            loss_sem = {
                'sem_ce': torch.tensor(0.0, device=self.device),
                'sem_lov': torch.tensor(0.0, device=self.device)
            }
        
        loss_mask.update(loss_sem)
        return loss_mask
    
    def training_step(self, batch, batch_idx):
        outputs, padding, sem_logits = self(batch)
        losses = self.get_losses(batch, outputs, padding, sem_logits)
        
        # Log losses
        for k, v in losses.items():
            self.log(f'train/{k}', v, batch_size=self.cfg.TRAIN.BATCH_SIZE, prog_bar=True)
        
        total_loss = sum(losses.values())
        self.log('train_loss', total_loss, batch_size=self.cfg.TRAIN.BATCH_SIZE, prog_bar=True)
        
        self.current_step += 1
        
        # Clear cache periodically
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            outputs, padding, sem_logits = self(batch)
            losses = self.get_losses(batch, outputs, padding, sem_logits)
        
        # Log losses
        for k, v in losses.items():
            self.log(f'val/{k}', v, batch_size=self.cfg.TRAIN.BATCH_SIZE)
        
        total_loss = sum(losses.values())
        self.log('val_loss', total_loss, batch_size=self.cfg.TRAIN.BATCH_SIZE)
        
        # Panoptic inference
        sem_pred, ins_pred = self.panoptic_inference(outputs, padding)
        
        # Prepare batch for evaluation (handle subsampling)
        eval_batch = self.prepare_batch_for_eval(batch, sem_pred)
        self.evaluator.update(sem_pred, ins_pred, eval_batch)
        
        self.validation_step_outputs.append(total_loss.item())
        
        # Clear cache
        torch.cuda.empty_cache()
        
        return total_loss
    
    def on_validation_epoch_end(self):
        # Compute metrics
        pq = self.evaluator.get_mean_pq()
        iou = self.evaluator.get_mean_iou()
        rq = self.evaluator.get_mean_rq()
        
        self.log('metrics/pq', pq, prog_bar=True)
        self.log('metrics/iou', iou, prog_bar=True)
        self.log('metrics/rq', rq, prog_bar=True)
        
        print(f"\nValidation Metrics - PQ: {pq:.4f}, IoU: {iou:.4f}, RQ: {rq:.4f}")
        
        # Reset
        self.evaluator.reset()
        self.validation_step_outputs.clear()
        torch.cuda.empty_cache()
    
    def configure_optimizers(self):
        # Separate parameters
        backbone_params = []
        decoder_params = []
        
        for name, param in self.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                decoder_params.append(param)
        
        # Different learning rates for backbone and decoder
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': self.cfg.TRAIN.LR * 0.5},
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
                pg['lr'] = pg['initial_lr'] * lr_scale if 'initial_lr' in pg else pg['lr'] * lr_scale
        
        super().optimizer_step(*args, **kwargs)
    
    def prepare_batch_for_eval(self, batch, sem_pred):
        """Prepare batch for evaluation with subsampling handling"""
        subsample_indices = self.backbone.subsample_indices
        
        eval_batch = {'fname': batch['fname']}
        eval_sem_labels = []
        eval_ins_labels = []
        
        for b in range(len(sem_pred)):
            sem_label = batch['sem_label'][b]
            ins_label = batch['ins_label'][b]
            
            # Convert to numpy if needed
            if isinstance(sem_label, torch.Tensor):
                sem_label = sem_label.cpu().numpy()
            if isinstance(ins_label, torch.Tensor):
                ins_label = ins_label.cpu().numpy()
            
            # Apply subsampling if needed
            if b in subsample_indices:
                indices = subsample_indices[b].cpu().numpy()
                if indices.shape[0] < sem_label.shape[0]:
                    sem_label = sem_label[indices]
                    ins_label = ins_label[indices]
            
            # Match prediction size
            pred_size = len(sem_pred[b])
            if pred_size < len(sem_label):
                sem_label = sem_label[:pred_size]
                ins_label = ins_label[:pred_size]
            
            eval_sem_labels.append(sem_label)
            eval_ins_labels.append(ins_label)
        
        eval_batch['sem_label'] = eval_sem_labels
        eval_batch['ins_label'] = eval_ins_labels
        
        return eval_batch
    
    def panoptic_inference(self, outputs, padding):
        """Panoptic segmentation inference"""
        mask_cls = outputs["pred_logits"]
        mask_pred = outputs["pred_masks"]
        
        batch_size = mask_cls.shape[0]
        sem_pred = []
        ins_pred = []
        
        for b in range(batch_size):
            valid_mask = ~padding[b]
            num_valid = valid_mask.sum().item()
            
            if num_valid == 0:
                sem_pred.append(np.zeros(0, dtype=np.int32))
                ins_pred.append(np.zeros(0, dtype=np.int32))
                continue
            
            scores, labels = mask_cls[b].max(-1)
            mask_pred_b = mask_pred[b][valid_mask].sigmoid()
            
            # Filter valid predictions
            keep = labels.ne(self.num_classes)
            
            if keep.sum() == 0:
                sem_pred.append(np.zeros(num_valid, dtype=np.int32))
                ins_pred.append(np.zeros(num_valid, dtype=np.int32))
                continue
            
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred_b[:, keep]
            
            # Instance assignment
            cur_prob_masks = cur_scores.unsqueeze(0) * cur_masks
            
            panoptic_seg = torch.zeros(num_valid, dtype=torch.int32, device=cur_masks.device)
            semantic_seg = torch.zeros(num_valid, dtype=torch.int32, device=cur_masks.device)
            instance_seg = torch.zeros(num_valid, dtype=torch.int32, device=cur_masks.device)
            
            current_segment_id = 0
            
            if cur_masks.shape[1] > 0:
                cur_mask_ids = cur_prob_masks.argmax(1)
                stuff_memory_list = {}
                
                for k in range(cur_classes.shape[0]):
                    pred_class = cur_classes[k].item()
                    isthing = pred_class in self.things_ids
                    
                    mask = (cur_mask_ids == k) & (cur_masks[:, k] >= 0.5)
                    mask_area = mask.sum().item()
                    
                    if mask_area > 0:
                        if isthing:
                            current_segment_id += 1
                            instance_seg[mask] = current_segment_id
                            semantic_seg[mask] = pred_class
                        else:
                            if pred_class not in stuff_memory_list:
                                current_segment_id += 1
                                stuff_memory_list[pred_class] = current_segment_id
                            semantic_seg[mask] = pred_class
            
            sem_pred.append(semantic_seg.cpu().numpy())
            ins_pred.append(instance_seg.cpu().numpy())
        
        return sem_pred, ins_pred