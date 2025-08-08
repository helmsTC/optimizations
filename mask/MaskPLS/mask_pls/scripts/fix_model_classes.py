#!/usr/bin/env python3
"""
Simple script to fix the semantic head class mismatch issue
Save as: mask/MaskPLS/mask_pls/scripts/fix_model_classes.py
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Fixing MaskPLS model class configuration...")

# 1. Check the simplified_model.py file
simplified_model_path = "../models/onnx/simplified_model.py"

# Read the file
with open(simplified_model_path, 'r') as f:
    content = f.read()

# Check if sem_head is using num_classes
if "self.sem_head = nn.Linear(cfg.DECODER.HIDDEN_DIM, self.num_classes)" not in content:
    print("✗ Found issue: sem_head not using self.num_classes")
    
    # Replace the sem_head line
    old_line = "self.sem_head = nn.Linear(cfg.DECODER.HIDDEN_DIM, self.num_classes)"
    if "self.sem_head = nn.Linear(cfg.BACKBONE.CHANNELS[-1], 20)" in content:
        # Original model had this
        content = content.replace(
            "self.sem_head = nn.Linear(cfg.BACKBONE.CHANNELS[-1], 20)",
            "self.sem_head = nn.Linear(cfg.DECODER.HIDDEN_DIM, self.num_classes)"
        )
    elif "self.sem_head = nn.Linear" in content:
        # Find and replace any sem_head initialization
        import re
        pattern = r'self\.sem_head = nn\.Linear\([^,]+,\s*\d+\)'
        replacement = 'self.sem_head = nn.Linear(cfg.DECODER.HIDDEN_DIM, self.num_classes)'
        content = re.sub(pattern, replacement, content)
    
    # Write back
    with open(simplified_model_path, 'w') as f:
        f.write(content)
    print("✓ Fixed sem_head initialization")
else:
    print("✓ sem_head already correctly configured")

# 2. Check the loss.py file
loss_path = "../models/loss.py"

with open(loss_path, 'r') as f:
    loss_content = f.read()

# Check if CrossEntropyLoss has ignore_index=0
if "CrossEntropyLoss(ignore_index=0)" not in loss_content:
    print("✗ Found issue: CrossEntropyLoss not ignoring index 0")
    
    # Replace the line
    loss_content = loss_content.replace(
        "self.cross_entropy = torch.nn.CrossEntropyLoss()",
        "self.cross_entropy = torch.nn.CrossEntropyLoss(ignore_index=0)"
    )
    
    # Also make sure it exists in the SemLoss class
    if "class SemLoss" in loss_content and "ignore_index=" not in loss_content:
        # Find the __init__ method and fix it
        import re
        pattern = r'(self\.cross_entropy = torch\.nn\.CrossEntropyLoss)\(\)'
        replacement = r'\1(ignore_index=0)'
        loss_content = re.sub(pattern, replacement, loss_content)
    
    with open(loss_path, 'w') as f:
        f.write(loss_content)
    print("✓ Fixed CrossEntropyLoss ignore_index")
else:
    print("✓ CrossEntropyLoss already has ignore_index=0")

print("\nAll fixes applied!")
print("\nNow run training with these commands:")
print("cd ..")
print("python scripts/train_simplified_model.py --batch_size 1 --lr 0.00001 --epochs 50 --gpus 1 --debug --num_workers 0")
