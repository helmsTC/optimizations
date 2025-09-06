#!/usr/bin/env python3
"""
Simple example to test your exported model
"""

import torch
import numpy as np
import sys
from pathlib import Path


def load_bin_file(bin_path):
    """Load a .bin point cloud file"""
    # KITTI format: N x 4 (x, y, z, intensity)
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    print(f"Loaded {len(points)} points from {bin_path}")
    return points


def simple_inference(model_path, bin_path):
    """Minimal inference example"""
    
    # Load model
    print(f"Loading model from {model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    print(f"Model loaded on {device}")
    
    # Load point cloud
    point_cloud = load_bin_file(bin_path)
    
    # Extract points and features
    points = point_cloud[:, :3]  # xyz
    features = point_cloud  # xyz + intensity
    
    # Filter to reasonable bounds (KITTI)
    mask = (
        (points[:, 0] > -48) & (points[:, 0] < 48) &
        (points[:, 1] > -48) & (points[:, 1] < 48) &
        (points[:, 2] > -4) & (points[:, 2] < 1.5)
    )
    points = points[mask]
    features = features[mask]
    print(f"After filtering: {len(points)} points")
    
    # Limit number of points
    max_points = 50000
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
        features = features[idx]
        print(f"Subsampled to {max_points} points")
    
    # Convert to tensors and add batch dimension
    points_tensor = torch.from_numpy(points).float().unsqueeze(0).to(device)  # [1, N, 3]
    features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)  # [1, N, 4]
    
    print(f"Input shapes: points={points_tensor.shape}, features={features_tensor.shape}")
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        outputs = model(points_tensor, features_tensor)
    
    # Process outputs
    pred_logits = outputs[0]  # [B, Q, C+1] - class predictions for queries
    pred_masks = outputs[1]   # [B, Q, N] or [B, N, Q] - mask predictions
    sem_logits = outputs[2]   # [B, N, C] - semantic predictions per point
    
    print(f"\nOutput shapes:")
    print(f"  Query logits: {pred_logits.shape}")
    print(f"  Mask predictions: {pred_masks.shape}")
    print(f"  Semantic logits: {sem_logits.shape}")
    
    # Get semantic predictions
    sem_pred = torch.argmax(sem_logits[0], dim=-1).cpu().numpy()
    
    # Get instance predictions (simplified)
    query_classes = torch.argmax(pred_logits[0], dim=-1).cpu().numpy()
    
    # Count predictions
    unique_classes, counts = np.unique(sem_pred, return_counts=True)
    
    print(f"\nSemantic predictions:")
    class_names = ['unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 
                   'other-vehicle', 'person', 'bicyclist', 'motorcyclist',
                   'road', 'parking', 'sidewalk', 'other-ground', 'building',
                   'fence', 'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign']
    
    for cls_id, count in zip(unique_classes, counts):
        if cls_id < len(class_names):
            print(f"  Class {cls_id} ({class_names[cls_id]}): {count} points")
        else:
            print(f"  Class {cls_id}: {count} points")
    
    # Count valid queries (not background)
    num_queries = len(query_classes)
    valid_queries = query_classes < 20  # Assuming 20 classes for KITTI
    print(f"\nQuery predictions: {valid_queries.sum()}/{num_queries} valid")
    
    return {
        'points': points,
        'semantic': sem_pred,
        'query_classes': query_classes
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python simple_inference.py <model.pt> <pointcloud.bin>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    bin_path = sys.argv[2]
    
    if not Path(model_path).exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)
    
    if not Path(bin_path).exists():
        print(f"Error: Point cloud not found: {bin_path}")
        sys.exit(1)
    
    results = simple_inference(model_path, bin_path)
    
    print("\n✓ Inference complete!")
    
    # Optional: save results
    output_path = "predictions.npz"
    np.savez_compressed(
        output_path,
        points=results['points'],
        semantic=results['semantic'],
        query_classes=results['query_classes']
    )
    print(f"Results saved to {output_path}")