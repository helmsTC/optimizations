"""
Simplified loss for the ONNX model that avoids CUDA errors
"""
import torch
import torch.nn.functional as F
from torch import nn
from mask_pls.models.matcher import HungarianMatcher
from mask_pls.utils.misc import (get_world_size, is_dist_avail_and_initialized,
                                 pad_stack, sample_points)


class SimplifiedMaskLoss(nn.Module):
    """Simplified version of MaskLoss with better bounds checking"""
    
    def __init__(self, cfg, data_cfg):
        super().__init__()
        self.num_classes = data_cfg.NUM_CLASSES  # 20 for KITTI
        self.ignore_label = data_cfg.IGNORE_LABEL  # 0 for KITTI
        self.matcher = HungarianMatcher(cfg.WEIGHTS, cfg.P_RATIO)
        
        self.weight_dict = {
            cfg.WEIGHTS_KEYS[i]: cfg.WEIGHTS[i] for i in range(len(cfg.WEIGHTS))
        }
        
        self.eos_coef = cfg.EOS_COEF
        
        # Create weights for all classes including no-object
        # Important: we have num_classes + 1 outputs (including no-object class)
        weights = torch.ones(self.num_classes + 1)
        if self.ignore_label >= 0 and self.ignore_label < self.num_classes:
            weights[self.ignore_label] = 0.0  # Down-weight ignore class
        weights[-1] = self.eos_coef  # Lower weight for no-object class
        self.weights = weights
        
        # pointwise mask loss parameters
        self.num_points = cfg.NUM_POINTS
        self.n_mask_pts = cfg.NUM_MASK_PTS
    
    def forward(self, outputs, targets, masks_ids, coors):
        """Compute losses with careful bounds checking"""
        losses = {}
        
        # Validate inputs
        pred_logits = outputs["pred_logits"]
        assert pred_logits.shape[-1] == self.num_classes + 1, \
            f"Expected {self.num_classes + 1} classes, got {pred_logits.shape[-1]}"
        
        # Check target classes are in valid range
        for i, cls_list in enumerate(targets["classes"]):
            if len(cls_list) > 0:
                max_cls = cls_list.max().item() if isinstance(cls_list, torch.Tensor) else max(cls_list)
                assert max_cls < self.num_classes, \
                    f"Target class {max_cls} >= num_classes {self.num_classes}"
        
        # Compute the average number of target masks for normalization
        num_masks = sum(len(t) for t in targets["classes"])
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=pred_logits.device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()
        
        # Get outputs without auxiliary
        outputs_no_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        
        # Retrieve the matching between outputs and targets
        indices = self.matcher(outputs_no_aux, targets)
        
        # Compute losses
        losses.update(
            self.get_losses(outputs, targets, indices, num_masks, masks_ids, coors)
        )
        
        # Handle auxiliary outputs if present
        if "aux_outputs" in outputs and len(outputs["aux_outputs"]) > 0:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                l_dict = self.get_losses(
                    aux_outputs, targets, indices, num_masks, masks_ids, coors
                )
                l_dict = {f"{i}_" + k: v for k, v in l_dict.items()}
                losses.update(l_dict)
        
        # Apply loss weights
        weighted_losses = {}
        for loss_name, loss_value in losses.items():
            # Find the corresponding weight key
            for weight_key in self.weight_dict:
                if weight_key in loss_name:
                    weighted_losses[loss_name] = loss_value * self.weight_dict[weight_key]
                    break
            else:
                # If no weight found, use the loss as is
                weighted_losses[loss_name] = loss_value
        
        return weighted_losses
    
    def get_losses(self, outputs, targets, indices, num_masks, masks_ids, coors, do_cls=True):
        losses = {}
        if do_cls:
            losses.update(self.loss_classes(outputs, targets, indices))
        losses.update(self.loss_masks(outputs, targets, indices, num_masks, masks_ids))
        return losses
    
    def loss_classes(self, outputs, targets, indices):
        """Classification loss with careful bounds checking"""
        assert "pred_logits" in outputs
        pred_logits = outputs["pred_logits"].float()
        
        device = pred_logits.device
        batch_size, num_queries, num_classes_plus_one = pred_logits.shape
        
        # Verify dimensions
        assert num_classes_plus_one == self.num_classes + 1, \
            f"Expected {self.num_classes + 1} classes, got {num_classes_plus_one}"
        
        # Get indices for matched predictions
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        
        # Get target classes for matched predictions
        target_classes_o = torch.cat(
            [t[J] for t, (_, J) in zip(targets["classes"], indices)]
        ).to(device)
        
        # Verify target classes are in valid range
        if len(target_classes_o) > 0:
            assert target_classes_o.max() < self.num_classes, \
                f"Target class {target_classes_o.max()} >= {self.num_classes}"
        
        # Initialize all queries to "no object" class (last class)
        # IMPORTANT: For cross_entropy with C classes, targets must be in [0, C-1]
        # So for 21 classes (0-20), no-object class is index 20
        no_object_class = self.num_classes  # This is 20 for KITTI
        target_classes = torch.full(
            (batch_size, num_queries),
            no_object_class,
            dtype=torch.int64,
            device=device
        )
        
        # Assign matched target classes
        if len(batch_idx) > 0:
            target_classes[batch_idx, src_idx] = target_classes_o
        
        # Verify all targets are in valid range [0, num_classes]
        assert target_classes.min() >= 0, f"Negative target class: {target_classes.min()}"
        assert target_classes.max() <= self.num_classes, \
            f"Target class {target_classes.max()} > {self.num_classes}"
        
        # Compute cross entropy loss
        # pred_logits: [B, Q, C+1] -> [B, C+1, Q]
        # target_classes: [B, Q]
        loss_ce = F.cross_entropy(
            pred_logits.transpose(1, 2),
            target_classes,
            weight=self.weights.to(device),
            reduction='mean'
        )
        
        losses = {"loss_ce": loss_ce}
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_masks, masks_ids):
        """Compute mask losses (dice and BCE)"""
        assert "pred_masks" in outputs
        
        # Get masks from targets
        masks = [t for t in targets["masks"]]
        if len(masks) == 0 or sum(len(m) for m in masks) == 0:
            # No masks in this batch
            device = outputs["pred_masks"].device
            dummy_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return {"loss_mask": dummy_loss, "loss_dice": dummy_loss}
        
        n_masks = [m.shape[0] for m in masks]
        target_masks = pad_stack(masks)
        
        # Get indices for predictions and targets
        pred_idx = self._get_pred_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices, n_masks)
        
        # Get predicted masks for matched queries
        pred_masks = outputs["pred_masks"]
        pred_masks = pred_masks[pred_idx[0], :, pred_idx[1]]
        target_masks = target_masks.to(pred_masks)
        target_masks = target_masks[tgt_idx]
        
        if pred_masks.shape[0] == 0:
            # No matched masks
            device = pred_masks.device
            dummy_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return {"loss_mask": dummy_loss, "loss_dice": dummy_loss}
        
        # Sample points for efficient computation
        with torch.no_grad():
            # Sample points within masks and random points
            point_idx = sample_points(masks, masks_ids, self.n_mask_pts, self.num_points)
            
            # Get point labels and logits
            n_masks.insert(0, 0)
            nm = torch.cumsum(torch.tensor(n_masks), 0)
            
            point_labels = []
            point_logits = []
            
            for i, p_idx in enumerate(point_idx):
                if nm[i+1] > nm[i]:  # Check if there are masks for this sample
                    # Ensure indices are within bounds
                    max_idx = pred_masks.shape[1] - 1
                    p_idx = p_idx[p_idx <= max_idx]
                    
                    if len(p_idx) > 0:
                        batch_labels = target_masks[nm[i]:nm[i+1]][:, p_idx]
                        batch_logits = pred_masks[nm[i]:nm[i+1]][:, p_idx]
                        point_labels.append(batch_labels)
                        point_logits.append(batch_logits)
        
        if len(point_labels) == 0:
            # No valid points sampled
            device = pred_masks.device
            dummy_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return {"loss_mask": dummy_loss, "loss_dice": dummy_loss}
        
        point_labels = torch.cat(point_labels)
        point_logits = torch.cat(point_logits)
        
        # Compute losses
        losses = {
            "loss_mask": self.sigmoid_ce_loss(point_logits, point_labels, num_masks),
            "loss_dice": self.dice_loss(point_logits, point_labels, num_masks),
        }
        
        return losses
    
    def sigmoid_ce_loss(self, inputs, targets, num_masks):
        """Binary cross entropy loss"""
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        return loss.mean(1).sum() / max(num_masks, 1)
    
    def dice_loss(self, inputs, targets, num_masks):
        """Dice loss"""
        inputs = inputs.sigmoid()
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / max(num_masks, 1)
    
    def _get_pred_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def _get_tgt_permutation_idx(self, indices, n_masks):
        # permute targets following indices
        batch_idx = []
        tgt_idx = []
        offset = 0
        
        for b, (_, tgt) in enumerate(indices):
            batch_idx.extend([b] * len(tgt))
            tgt_idx.extend((tgt + offset).tolist())
            offset += n_masks[b]
        
        return tgt_idx
