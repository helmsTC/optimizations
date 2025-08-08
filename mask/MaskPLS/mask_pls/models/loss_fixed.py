"""
Fixed loss functions for the simplified model
"""
import torch
import torch.nn.functional as F
from torch import nn
from mask_pls.models.matcher import HungarianMatcher
from mask_pls.utils.misc import (get_world_size, is_dist_avail_and_initialized,
                                 pad_stack, sample_points)


class MaskLossFixed(nn.Module):
    """Fixed version of MaskLoss that handles class indices correctly"""
    
    def __init__(self, cfg, data_cfg):
        super().__init__()
        self.num_classes = data_cfg.NUM_CLASSES
        self.ignore = data_cfg.IGNORE_LABEL
        self.matcher = HungarianMatcher(cfg.WEIGHTS, cfg.P_RATIO)
        
        self.weight_dict = {
            cfg.WEIGHTS_KEYS[i]: cfg.WEIGHTS[i] for i in range(len(cfg.WEIGHTS))
        }
        
        self.eos_coef = cfg.EOS_COEF
        
        # FIXED: Create weights for all classes including no-object
        # The no-object class is at index num_classes
        weights = torch.ones(self.num_classes + 1)
        weights[0] = 0.0  # Ignore unlabeled (if it appears in masks)
        weights[-1] = self.eos_coef  # Lower weight for no-object class
        self.weights = weights
        
        # pointwise mask loss parameters
        self.num_points = cfg.NUM_POINTS
        self.n_mask_pts = cfg.NUM_MASK_PTS
    
    def forward(self, outputs, targets, masks_ids, coors):
        losses = {}
        
        # Compute the average number of target masks for normalization
        num_masks = sum(len(t) for t in targets["classes"])
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()
        
        outputs_no_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        
        # Retrieve the matching between outputs and targets
        indices = self.matcher(outputs_no_aux, targets)
        
        losses.update(
            self.get_losses(outputs, targets, indices, num_masks, masks_ids, coors)
        )
        
        # Handle auxiliary outputs if present
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                l_dict = self.get_losses(
                    aux_outputs, targets, indices, num_masks, masks_ids, coors
                )
                l_dict = {f"{i}_" + k: v for k, v in l_dict.items()}
                losses.update(l_dict)
        
        losses = {
            l: losses[l] * self.weight_dict[k]
            for l in losses
            for k in self.weight_dict
            if k in l
        }
        
        return losses
    
    def get_losses(self, outputs, targets, indices, num_masks, masks_ids, coors, do_cls=True):
        if do_cls:
            classes = self.loss_classes(outputs, targets, indices)
        else:
            classes = {}
        masks = self.loss_masks(outputs, targets, indices, num_masks, masks_ids)
        classes.update(masks)
        return classes
    
    def loss_classes(self, outputs, targets, indices):
        """Classification loss (NLL) - FIXED version"""
        assert "pred_logits" in outputs
        pred_logits = outputs["pred_logits"].float()
        
        idx = self._get_pred_permutation_idx(indices)
        
        # Get target classes for matched predictions
        target_classes_o = torch.cat(
            [t[J] for t, (_, J) in zip(targets["classes"], indices)]
        ).to(pred_logits.device)
        
        # FIXED: Initialize with no-object class (last class)
        target_classes = torch.full(
            pred_logits.shape[:2],
            self.num_classes,  # This is the no-object class index
            dtype=torch.int64,
            device=pred_logits.device,
        )
        
        # Assign matched target classes
        target_classes[idx] = target_classes_o
        
        # FIXED: Use -100 as ignore index (PyTorch default) instead of 0
        # This avoids conflicts with class 0 (unlabeled)
        loss_ce = F.cross_entropy(
            pred_logits.transpose(1, 2),
            target_classes,
            self.weights.to(pred_logits),
            ignore_index=-100,  # Use PyTorch default ignore index
        )
        
        losses = {"loss_ce": loss_ce}
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_masks, masks_ids):
        """Compute the losses related to the masks: the focal loss and the dice loss."""
        assert "pred_masks" in outputs
        
        masks = [t for t in targets["masks"]]
        n_masks = [m.shape[0] for m in masks]
        target_masks = pad_stack(masks)
        
        pred_idx = self._get_pred_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices, n_masks)
        pred_masks = outputs["pred_masks"]
        pred_masks = pred_masks[pred_idx[0], :, pred_idx[1]]
        target_masks = target_masks.to(pred_masks)
        target_masks = target_masks[tgt_idx]
        
        with torch.no_grad():
            idx = sample_points(masks, masks_ids, self.n_mask_pts, self.num_points)
            n_masks.insert(0, 0)
            nm = torch.cumsum(torch.tensor(n_masks), 0)
            point_labels = torch.cat(
                [target_masks[nm[i] : nm[i + 1]][:, p] for i, p in enumerate(idx)]
            )
        point_logits = torch.cat(
            [pred_masks[nm[i] : nm[i + 1]][:, p] for i, p in enumerate(idx)]
        )
        
        del pred_masks
        del target_masks
        
        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }
        
        return losses
    
    def _get_pred_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def _get_tgt_permutation_idx(self, indices, n_masks):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        # From [B,id] to [id] of stacked masks
        cont_id = torch.cat([torch.arange(n) for n in n_masks])
        b_id = torch.stack((batch_idx, cont_id), axis=1)
        map_m = torch.zeros((torch.max(batch_idx) + 1, max(n_masks)))
        for i in range(len(b_id)):
            map_m[b_id[i, 0], b_id[i, 1]] = i
        stack_ids = [
            int(map_m[batch_idx[i], tgt_idx[i]]) for i in range(len(batch_idx))
        ]
        return stack_ids


# Import the dice and sigmoid losses from original
def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_masks


dice_loss_jit = torch.jit.script(dice_loss)
sigmoid_ce_loss_jit = torch.jit.script(sigmoid_ce_loss)
