# mask_pls/DGCNN_IMPLEMENTATION.md

# MaskPLS with DGCNN Backbone - Implementation Guide

## Overview

This implementation replaces the MinkowskiEngine sparse convolutions with a DGCNN (Dynamic Graph CNN) backbone, enabling ONNX export while maintaining competitive performance. DGCNN is particularly suitable for point cloud processing as it operates directly on points without voxelization, preserving fine-grained details.

## Key Changes from Original MaskPLS

### 1. Backbone Architecture
- **Original**: MinkowskiEngine sparse 3D convolutions with voxelization at 0.05m resolution
- **New**: DGCNN backbone operating directly on point coordinates and features
- **Benefits**: 
  - No information loss from voxelization
  - ONNX compatible
  - Can leverage pre-trained weights from classification/segmentation tasks

### 2. Feature Extraction
- **Original**: KNN interpolation using PyKeOps for voxel-to-point mapping
- **New**: Direct point feature extraction through dynamic graph convolution
- **Benefits**:
  - More accurate feature representation
  - No interpolation errors
  - Better gradient flow

### 3. Data Augmentation
- Added comprehensive augmentation pipeline:
  - Random rotation (z-axis)
  - Random scaling (0.95-1.05)
  - Random flipping (x, y axes)
  - Random jittering (Gaussian noise)
  - Random dropout (10% points)

### 4. Training Strategy
- Learning rate warmup (1000 steps)
- Different learning rates for backbone (0.1x) and decoder
- Mixed precision training (FP16)
- Gradient clipping (1.0)
- Early stopping with patience of 20 epochs

## Performance Improvements

### Expected Metrics (after full training):
- **PQ (Panoptic Quality)**: 50-55% (vs. original ~60%)
- **IoU (Intersection over Union)**: 55-60% (vs. original ~65%)
- **RQ (Recognition Quality)**: 60-65% (vs. original ~70%)

The slight performance gap is expected due to:
1. Lack of sparse convolution's efficiency
2. Different feature extraction mechanism
3. Need for more training epochs to converge

## Installation

```bash
# Install additional dependencies
pip install torch-geometric  # For graph operations
pip install onnx onnxruntime onnx-simplifier  # For ONNX export

# Download pre-trained DGCNN weights (optional)
wget https://github.com/WangYueFt/dgcnn/raw/master/pretrained/model.cls.1024.t7 -O dgcnn_modelnet40.pth