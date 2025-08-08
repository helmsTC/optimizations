"""
Even simpler training script that avoids CUDA/numpy issues
Save as: mask/MaskPLS/mask_pls/scripts/train_simple_alt.py
"""

import os
import torch
import yaml
from easydict import EasyDict as edict
from os.path import join

# Set device handling to avoid multi-GPU issues
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.onnx.simplified_model import create_onnx_model, export_model_to_onnx
from datasets.semantic_dataset import SemanticDatasetModule
from models.loss import MaskLoss, SemLoss
from utils.evaluate_panoptic import PanopticEvaluator


def train_epoch(model, dataloader, mask_loss, sem_loss, optimizer, device='cuda'):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Process batch
        try:
            # Voxelize and prepare data
            voxel_batch = []
            coord_batch = []
            
            for i in range(len(batch['pt_coord'])):
                pts = torch.from_numpy(batch['pt_coord'][i]).float().to(device)
                feat = torch.from_numpy(batch['feats'][i]).float().to(device)
                
                # Simple bounds check
                bounds = [[-48.0, 48.0], [-48.0, 48.0], [-4.0, 1.5]]
                valid_mask = torch.ones(len(pts), dtype=torch.bool, device=device)
                for d in range(3):
                    valid_mask &= (pts[:, d] >= bounds[d][0]) & (pts[:, d] < bounds[d][1])
                
                if valid_mask.sum() == 0:
                    continue
                    
                valid_pts = pts[valid_mask]
                valid_feat = feat[valid_mask]
                
                # Normalize coords
                norm_coords = torch.zeros_like(valid_pts)
                for d in range(3):
                    norm_coords[:, d] = (valid_pts[:, d] - bounds[d][0]) / (bounds[d][1] - bounds[d][0])
                
                # Voxelize
                voxel = model.voxelize_points(valid_pts.unsqueeze(0), valid_feat.unsqueeze(0))[0]
                
                voxel_batch.append(voxel)
                coord_batch.append(norm_coords)
            
            if len(voxel_batch) == 0:
                continue
                
            # Pad coordinates
            max_pts = max(c.shape[0] for c in coord_batch)
            padded_coords = []
            for coords in coord_batch:
                if coords.shape[0] < max_pts:
                    coords = torch.nn.functional.pad(coords, (0, 0, 0, max_pts - coords.shape[0]))
                padded_coords.append(coords)
            
            # Stack
            voxels = torch.stack(voxel_batch)
            coords = torch.stack(padded_coords)
            
            # Forward pass
            pred_logits, pred_masks, sem_logits = model(voxels, coords)
            
            # Compute losses (simplified)
            outputs = {'pred_logits': pred_logits, 'pred_masks': pred_masks, 'aux_outputs': []}
            targets = {'classes': batch['masks_cls'], 'masks': batch['masks']}
            
            # Mask loss
            loss_dict = mask_loss(outputs, targets, batch['masks_ids'], batch['pt_coord'])
            
            # Total loss
            loss = sum(loss_dict.values())
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}: loss = {loss.item():.4f}")
                
        except Exception as e:
            print(f"  Error in batch {batch_idx}: {e}")
            continue
    
    return total_loss / max(num_batches, 1)


def main():
    """Simple training loop"""
    print("=" * 60)
    print("Simple MaskPLS ONNX Training")
    print("=" * 60)
    
    # Load config
    model_cfg = edict(yaml.safe_load(open(join(os.path.dirname(__file__), "../config/model.yaml"))))
    backbone_cfg = edict(yaml.safe_load(open(join(os.path.dirname(__file__), "../config/backbone.yaml"))))
    decoder_cfg = edict(yaml.safe_load(open(join(os.path.dirname(__file__), "../config/decoder.yaml"))))
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    
    # Settings
    cfg.TRAIN.BATCH_SIZE = 1
    cfg.TRAIN.NUM_WORKERS = 0  # Avoid multiprocessing issues
    num_epochs = 10  # Start small
    
    print(f"Dataset: {cfg.MODEL.DATASET}")
    print(f"Batch size: {cfg.TRAIN.BATCH_SIZE}")
    print(f"Epochs: {num_epochs}")
    
    # Create model
    print("\nCreating model...")
    model = create_onnx_model(cfg).cuda()
    
    # Create data
    print("Loading data...")
    data = SemanticDatasetModule(cfg)
    data.setup()
    train_loader = data.train_dataloader()
    
    # Loss functions
    mask_loss = MaskLoss(cfg.LOSS, cfg[cfg.MODEL.DATASET])
    sem_loss = SemLoss(cfg.LOSS.SEM.WEIGHTS)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.TRAIN.LR)
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        avg_loss = train_epoch(model, train_loader, mask_loss, sem_loss, optimizer)
        print(f"Average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            ckpt_path = f"simple_maskpls_epoch{epoch+1}.ckpt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'cfg': cfg
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")
            
            # Export ONNX
            try:
                onnx_path = f"simple_maskpls_epoch{epoch+1}.onnx"
                export_model_to_onnx(model, onnx_path)
                print(f"Exported ONNX: {onnx_path}")
            except Exception as e:
                print(f"ONNX export failed: {e}")
    
    print("\nTraining complete!")
    
    # Final save
    final_ckpt = "simple_maskpls_final.ckpt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'cfg': cfg
    }, final_ckpt)
    print(f"Saved final model: {final_ckpt}")
    
    # Final ONNX export
    try:
        export_model_to_onnx(model, "simple_maskpls_final.onnx")
        print("Exported final ONNX model")
    except:
        pass


if __name__ == "__main__":
    main()
