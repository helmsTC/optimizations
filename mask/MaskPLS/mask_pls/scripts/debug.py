#!/usr/bin/env python3
"""
Debug script to find exact CUDA error
Save as: mask/MaskPLS/mask_pls/scripts/debug_cuda.py
"""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

import torch
import numpy as np
import yaml
from easydict import EasyDict as edict
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

print("="*60)
print("CUDA Error Debugging")
print("="*60)

# Load all configs
config_dir = Path(__file__).parent.parent / "config"
model_cfg = edict(yaml.safe_load(open(config_dir / "model.yaml")))
backbone_cfg = edict(yaml.safe_load(open(config_dir / "backbone.yaml")))
decoder_cfg = edict(yaml.safe_load(open(config_dir / "decoder.yaml")))
cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})

dataset = cfg.MODEL.DATASET
print(f"\n1. Configuration:")
print(f"   Dataset: {dataset}")
print(f"   NUM_CLASSES in config: {cfg[dataset].NUM_CLASSES}")
print(f"   IGNORE_LABEL in config: {cfg[dataset].IGNORE_LABEL}")

# Check the model
print(f"\n2. Checking model configuration:")
from models.onnx.simplified_model import MaskPLSSimplifiedONNX

model = MaskPLSSimplifiedONNX(cfg).cuda()
print(f"   Model num_classes: {model.num_classes}")
print(f"   Sem head out_features: {model.sem_head.out_features}")

if model.sem_head.out_features != model.num_classes:
    print(f"   ❌ ERROR: Mismatch! sem_head outputs {model.sem_head.out_features} but num_classes is {model.num_classes}")
else:
    print(f"   ✓ OK: sem_head matches num_classes")

# Check the dataset
print(f"\n3. Checking dataset labels:")
from datasets.semantic_dataset import SemanticDatasetModule

data_module = SemanticDatasetModule(cfg)
data_module.setup()

# Get the learning map
yaml_path = Path(cfg[dataset].CONFIG)
with open(yaml_path, 'r') as f:
    sem_yaml = yaml.safe_load(f)

print(f"\n   Learning map (first 10 entries):")
learning_map = sem_yaml['learning_map']
for k, v in list(learning_map.items())[:10]:
    print(f"     {k} -> {v}")

print(f"\n   Max mapped value: {max(learning_map.values())}")
print(f"   Expected max: {cfg[dataset].NUM_CLASSES - 1}")

# Check actual data
print(f"\n4. Checking actual batch data:")
train_loader = data_module.train_dataloader()
batch = next(iter(train_loader))

for i, labels in enumerate(batch['sem_label']):
    unique = np.unique(labels)
    print(f"\n   Sample {i}:")
    print(f"     Shape: {labels.shape}")
    print(f"     Unique values: {unique}")
    print(f"     Min: {unique.min()}, Max: {unique.max()}")
    
    if unique.max() >= cfg[dataset].NUM_CLASSES:
        print(f"     ❌ ERROR: Max label {unique.max()} >= NUM_CLASSES {cfg[dataset].NUM_CLASSES}")
        print(f"     Labels > {cfg[dataset].NUM_CLASSES - 1}: {unique[unique >= cfg[dataset].NUM_CLASSES]}")

# Test the loss function
print(f"\n5. Testing loss functions:")
from models.loss import SemLoss

sem_loss = SemLoss(cfg.LOSS.SEM.WEIGHTS)
print(f"   CrossEntropyLoss ignore_index: {sem_loss.cross_entropy.ignore_index}")

# Test with dummy data
logits = torch.randn(100, cfg[dataset].NUM_CLASSES).cuda()
labels = torch.randint(0, cfg[dataset].NUM_CLASSES, (100,)).cuda()

try:
    loss = sem_loss(logits, labels)
    print(f"   ✓ Loss computation successful with valid labels")
except Exception as e:
    print(f"   ❌ Error with valid labels: {e}")

# Test with out-of-bounds label
bad_labels = torch.tensor([0, 5, 10, 19, 20]).cuda()  # 20 is out of bounds for KITTI
logits_bad = torch.randn(5, cfg[dataset].NUM_CLASSES).cuda()

print(f"\n   Testing with out-of-bounds labels [0, 5, 10, 19, 20]:")
try:
    loss = sem_loss.cross_entropy(logits_bad, bad_labels)
    print(f"   ❌ No error - this is the problem!")
except Exception as e:
    print(f"   ✓ Got expected error: {type(e).__name__}")
    print(f"   Error message: {str(e)}")

print("\n" + "="*60)
print("Diagnosis Summary:")
print("="*60)

# Summary
issues = []
if model.sem_head.out_features != model.num_classes:
    issues.append("Semantic head output mismatch")
if max(learning_map.values()) >= cfg[dataset].NUM_CLASSES:
    issues.append("Learning map has invalid values")
if sem_loss.cross_entropy.ignore_index != cfg[dataset].IGNORE_LABEL:
    issues.append("Loss function ignore_index mismatch")

if issues:
    print("Found issues:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("No obvious configuration issues found.")
    print("The error might be in the data processing or forward pass.")

print("\nNext steps:")
print("1. Check if the learning_map is being applied correctly in the dataset")
print("2. Add bounds checking in the model forward pass")
print("3. Check if masks/instance labels are causing the issue")
