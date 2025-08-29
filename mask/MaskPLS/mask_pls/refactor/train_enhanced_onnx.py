# mask/MaskPLS/mask_pls/scripts/train_enhanced_onnx.py
"""
Enhanced training script with proper loss computation and high resolution
"""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

from os.path import join
import click
import torch
import yaml
import numpy as np
import time
from pathlib import Path
from easydict import EasyDict as edict
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
import torch.nn.functional as F

# Import components
from mask_pls.models.onnx.enhanced_model import EnhancedMaskPLSONNX
from mask_pls.models.onnx.voxelizer import HighResVoxelizer
from mask_pls.datasets.semantic_dataset import SemanticDatasetModule
from mask_pls.models.loss import MaskLoss, SemLoss
from mask_pls.models.matcher import HungarianMatcher
from mask_pls.utils.evaluate_panoptic import PanopticEvaluator
from mask_pls.utils.misc import sample_points, pad_stack


class EnhancedMaskLoss(torch.nn.Module):
    """Fixed mask loss with proper index handling"""
    def __init__(self, cfg, data_cfg):
        super().__init__()
        
        self.num_classes = data_cfg.NUM_CLASSES
        self.ignore_label = data_cfg.IGNORE_LABEL
        self.matcher = HungarianMatcher(cfg.WEIGHTS, cfg.P_RATIO)
        
        self.weight_dict = {
            cfg.WEIGHTS_KEYS[i]: cfg.WEIGHTS[i] 
            for i in range(len(cfg.WEIGHTS))
        }
        
        self.eos_coef = cfg.EOS_COEF
        
        # Class weights
        weights = torch.ones(self.num_classes + 1)
        weights[0] = 0.0
        weights[-1] = self.eos_coef
        self.register_buffer('weights', weights)
        
        # Use original sampling parameters
        self.num_points = cfg.NUM_POINTS
        self.n_mask_pts = cfg.NUM_MASK_PTS
        
    def forward(self, outputs, targets, mask_indices):
        """Forward with fixed index handling"""
        losses = {}
        
        num_masks = sum(len(t) for t in targets["classes"])

