"""
Simplified training script that's easier to use
Save as: mask/MaskPLS/mask_pls/scripts/train_simplified.py
"""

import os
from os.path import join
import click
import torch
import torch.nn.functional as F
import yaml
from easydict import EasyDict as edict

from mask_pls.datasets.semantic_dataset import SemanticDatasetModule
from mask_pls.models.mask_model import MaskPS
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


# Import our simplified components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.onnx.simplified_model import create_onnx_model


class SimplifiedMaskPSAdapter(MaskPS):
    """
    Adapter that makes the simplified model work with the existing training infrastructure
    """
    def __init__(self, cfg):
        # Initialize parent without creating the original model
        super(MaskPS, self).__init__()  # Skip MaskPS.__init__
        
        self.save_hyperparameters(dict(cfg))
        self.cfg = cfg
        
        # Create simplified model instead of original
        print("Creating Simplified ONNX-compatible model for training...")
        self.simplified_model = create_onnx_model(cfg)
        
        # Create components from parent class
        from mask_pls.models.loss import MaskLoss, SemLoss
        from mask_pls.utils.evaluate_panoptic import PanopticEvaluator
        
        self.mask_loss = MaskLoss(cfg.LOSS, cfg[cfg.MODEL.DATASET])
        self.sem_loss = SemLoss(cfg.LOSS.SEM.WEIGHTS)
        self.evaluator = PanopticEvaluator(cfg[cfg.MODEL.DATASET], cfg.MODEL.DATASET)
        
        # Store dataset info
        self.dataset = cfg.MODEL.DATASET
        self.num_classes = cfg[cfg.MODEL.DATASET].NUM_CLASSES
        self.coordinate_bounds = cfg[cfg.MODEL.DATASET].SPACE
        
    def forward(self, x):
        """Override forward to use simplified model"""
        # Pre-voxelize batch
        voxel_batch = []
        coord_batch = []
        valid_indices = []
        
        for i in range(len(x['pt_coord'])):
            pts = torch.from_numpy(x['pt_coord'][i]).float().cuda()
            feat = torch.from_numpy(x['feats'][i]).float().cuda()
            
            # Filter valid points
            bounds = self.coordinate_bounds
            valid_mask = torch.ones(pts.shape[0], dtype=torch.bool, device=pts.device)
            for dim in range(3):
                valid_mask &= (pts[:, dim] >= bounds[dim][0]) & (pts[:, dim] < bounds[dim][1])
            
            valid_pts = pts[valid_mask]
            valid_feat = feat[valid_mask]
            valid_idx = torch.where(valid_mask)[0]
            
            # Subsample if training
            if self.training and hasattr(self.cfg, 'TRAIN') and hasattr(self.cfg.TRAIN, 'SUBSAMPLE'):
                max_pts = self.cfg[self.dataset].SUB_NUM_POINTS
                if len(valid_pts) > max_pts:
                    perm = torch.randperm(len(valid_pts))[:max_pts]
                    valid_pts = valid_pts[perm]
                    valid_feat = valid_feat[perm]
                    valid_idx = valid_idx[perm]
            
            # Normalize coordinates
            norm_coords = torch.zeros_like(valid_pts)
            for dim in range(3):
                norm_coords[:, dim] = (valid_pts[:, dim] - bounds[dim][0]) / (bounds[dim][1] - bounds[dim][0])
            
            # Voxelize
            voxel = self.simplified_model.voxelize_points(
                valid_pts.unsqueeze(0),
                valid_feat.unsqueeze(0)
            )[0]
            
            voxel_batch.append(voxel)
            coord_batch.append(norm_coords)
            valid_indices.append(valid_idx)
        
        # Pad coordinates
        max_pts = max(c.shape[0] for c in coord_batch)
        padded_coords = []
        padding_masks = []
        
        for coords in coord_batch:
            n_pts = coords.shape[0]
            if n_pts < max_pts:
                coords = F.pad(coords, (0, 0, 0, max_pts - n_pts))
            padded_coords.append(coords)
            
            mask = torch.zeros(max_pts, dtype=torch.bool, device=coords.device)
            mask[n_pts:] = True
            padding_masks.append(mask)
        
        # Stack batch
        voxel_features = torch.stack(voxel_batch)
        point_coords = torch.stack(padded_coords)
        padding = torch.stack(padding_masks)
        
        # Forward through simplified model
        pred_logits, pred_masks, sem_logits = self.simplified_model(voxel_features, point_coords)
        
        # Format outputs to match original model
        outputs = {
            'pred_logits': pred_logits,
            'pred_masks': pred_masks,
            'aux_outputs': []  # No auxiliary outputs in simplified model
        }
        
        return outputs, padding, sem_logits
    
    def getLoss(self, x, outputs, padding, sem_logits):
        """Override to handle simplified model outputs"""
        targets = {"classes": x["masks_cls"], "masks": x["masks"]}
        loss_mask = self.mask_loss(outputs, targets, x["masks_ids"], x["pt_coord"])
        
        # Semantic loss on valid points
        sem_labels = [torch.from_numpy(l).long().cuda() for l in x["sem_label"]]
        sem_labels = torch.cat([s.squeeze(1) for s in sem_labels], dim=0)
        sem_logits_valid = sem_logits[~padding]
        
        # Match dimensions
        min_len = min(sem_labels.shape[0], sem_logits_valid.shape[0])
        if min_len > 0:
            loss_sem_bb = self.sem_loss(sem_logits_valid[:min_len], sem_labels[:min_len])
        else:
            loss_sem_bb = {'sem_ce': torch.tensor(0.0).cuda(), 'sem_lov': torch.tensor(0.0).cuda()}
        
        loss_mask.update(loss_sem_bb)
        return loss_mask
    
    def configure_optimizers(self):
        """Use the same optimizer setup as original"""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.TRAIN.LR)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.cfg.TRAIN.STEP, gamma=self.cfg.TRAIN.DECAY
        )
        return [optimizer], [scheduler]
    
    # The rest of the methods (training_step, validation_step, etc.) 
    # are inherited from the parent class and should work as-is


def getDir(obj):
    return os.path.dirname(os.path.abspath(obj))


@click.command()
@click.option("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
@click.option("--nuscenes", is_flag=True, help="Use NuScenes dataset instead of KITTI")
@click.option("--export_every", type=int, default=10, help="Export ONNX model every N epochs")
def main(checkpoint, nuscenes, export_every):
    """
    Train the simplified ONNX-compatible MaskPLS model
    """
    print("=" * 60)
    print("Training Simplified MaskPLS (ONNX-Compatible)")
    print("=" * 60)
    
    # Load configurations
    model_cfg = edict(yaml.safe_load(open(join(getDir(__file__), "../config/model.yaml"))))
    backbone_cfg = edict(yaml.safe_load(open(join(getDir(__file__), "../config/backbone.yaml"))))
    decoder_cfg = edict(yaml.safe_load(open(join(getDir(__file__), "../config/decoder.yaml"))))
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    
    if nuscenes:
        cfg.MODEL.DATASET = "NUSCENES"
    
    # Update experiment ID to distinguish from original
    cfg.EXPERIMENT.ID = cfg.EXPERIMENT.ID + "_simplified_onnx"
    
    print(f"Dataset: {cfg.MODEL.DATASET}")
    print(f"Experiment ID: {cfg.EXPERIMENT.ID}")
    print(f"Batch size: {cfg.TRAIN.BATCH_SIZE}")
    print(f"Learning rate: {cfg.TRAIN.LR}")
    print(f"Max epochs: {cfg.TRAIN.MAX_EPOCH}")
    
    # Create data module (uses original dataset)
    data = SemanticDatasetModule(cfg)
    
    # Create our adapted model
    model = SimplifiedMaskPSAdapter(cfg)
    
    if checkpoint:
        print(f"\nLoading checkpoint: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location="cpu")
        if 'state_dict' in ckpt:
            # Try to load what we can
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in ckpt['state_dict'].items() if k in model_dict and v.shape == model_dict[k].shape}
            model.load_state_dict(pretrained_dict, strict=False)
            print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} parameters from checkpoint")
        else:
            print("Warning: Checkpoint doesn't contain 'state_dict'")
    
    # Logger
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
    
    # Custom callback to export ONNX periodically
    class ONNXExportCallback(pl.callbacks.Callback):
        def on_epoch_end(self, trainer, pl_module):
            if (trainer.current_epoch + 1) % export_every == 0:
                print(f"\nExporting ONNX model at epoch {trainer.current_epoch + 1}...")
                try:
                    from models.onnx.simplified_model import export_model_to_onnx
                    export_path = f"experiments/{cfg.EXPERIMENT.ID}/model_epoch{trainer.current_epoch+1}.onnx"
                    export_model_to_onnx(pl_module.simplified_model, export_path)
                    print(f"✓ Exported to {export_path}")
                except Exception as e:
                    print(f"✗ ONNX export failed: {e}")
    
    onnx_callback = ONNXExportCallback()
    
    # Trainer
    trainer = Trainer(
        gpus=cfg.TRAIN.N_GPUS,
        accelerator="ddp" if cfg.TRAIN.N_GPUS > 1 else None,
        logger=tb_logger,
        max_epochs=cfg.TRAIN.MAX_EPOCH,
        callbacks=[lr_monitor, pq_ckpt, iou_ckpt, onnx_callback],
        log_every_n_steps=1,
        gradient_clip_val=0.5,
        accumulate_grad_batches=cfg.TRAIN.BATCH_ACC,
        resume_from_checkpoint=checkpoint if checkpoint and os.path.exists(checkpoint) else None,
    )
    
    # Train!
    print("\nStarting training...")
    trainer.fit(model, data)
    
    # Export final ONNX model
    print("\nExporting final ONNX model...")
    try:
        from models.onnx.simplified_model import export_model_to_onnx
        export_path = f"experiments/{cfg.EXPERIMENT.ID}/model_final.onnx"
        export_model_to_onnx(model.simplified_model, export_path)
        print(f"✓ Final model exported to {export_path}")
    except Exception as e:
        print(f"✗ Final ONNX export failed: {e}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Checkpoints saved in: experiments/{cfg.EXPERIMENT.ID}")
    print("=" * 60)


if __name__ == "__main__":
    main()
