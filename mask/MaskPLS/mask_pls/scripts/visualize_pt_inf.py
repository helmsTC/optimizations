#!/usr/bin/env python3
"""
Visualize inference results using Open3D
"""

import numpy as np
import torch
import argparse
from pathlib import Path


def visualize_with_open3d(model_path, bin_path, save_path=None):
    """Visualize segmentation results with Open3D"""
    try:
        import open3d as o3d
    except ImportError:
        print("Please install Open3D: pip install open3d")
        return
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    
    # Load point cloud
    points_raw = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    points = points_raw[:, :3]
    intensity = points_raw[:, 3]
    
    # Filter points
    mask = (
        (points[:, 0] > -48) & (points[:, 0] < 48) &
        (points[:, 1] > -48) & (points[:, 1] < 48) &
        (points[:, 2] > -4) & (points[:, 2] < 1.5)
    )
    points = points[mask]
    intensity = intensity[mask]
    
    # Subsample if needed
    if len(points) > 50000:
        idx = np.random.choice(len(points), 50000, replace=False)
        points = points[idx]
        intensity = intensity[idx]
    
    # Prepare tensors
    points_tensor = torch.from_numpy(points).float().unsqueeze(0).to(device)
    features_tensor = torch.from_numpy(points_raw[mask][:len(points)]).float().unsqueeze(0).to(device)
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        outputs = model(points_tensor, features_tensor)
    
    # Get semantic predictions
    sem_logits = outputs[2][0]  # [N, C]
    sem_pred = torch.argmax(sem_logits, dim=-1).cpu().numpy()
    
    # Create color map for semantic classes
    # Using a nice color palette
    color_map = np.array([
        [0, 0, 0],        # 0: unlabeled - black
        [245, 150, 100],  # 1: car - orange
        [245, 230, 100],  # 2: bicycle - yellow
        [150, 60, 30],    # 3: motorcycle - brown
        [180, 30, 80],    # 4: truck - purple
        [255, 0, 0],      # 5: other-vehicle - red
        [30, 30, 255],    # 6: person - blue
        [200, 40, 255],   # 7: bicyclist - violet
        [90, 30, 150],    # 8: motorcyclist - dark purple
        [255, 0, 255],    # 9: road - magenta
        [255, 150, 255],  # 10: parking - light magenta
        [75, 0, 75],      # 11: sidewalk - dark purple
        [75, 0, 175],     # 12: other-ground - purple
        [0, 200, 255],    # 13: building - cyan
        [50, 120, 255],   # 14: fence - light blue
        [0, 175, 0],      # 15: vegetation - green
        [0, 60, 135],     # 16: trunk - dark blue
        [80, 240, 150],   # 17: terrain - light green
        [150, 240, 255],  # 18: pole - light cyan
        [0, 0, 255],      # 19: traffic-sign - blue
    ]) / 255.0
    
    # Extend color map if needed
    if sem_pred.max() >= len(color_map):
        extra_colors = np.random.rand(sem_pred.max() - len(color_map) + 1, 3)
        color_map = np.vstack([color_map, extra_colors])
    
    # Assign colors
    colors = color_map[sem_pred]
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    
    # Visualize
    print("\nVisualization controls:")
    print("  - Mouse: Rotate view")
    print("  - Scroll: Zoom")
    print("  - Ctrl+Mouse: Pan")
    print("  - H: Show help")
    print("  - Q/Esc: Close")
    
    if save_path:
        # Save point cloud
        o3d.io.write_point_cloud(save_path, pcd)
        print(f"Point cloud saved to {save_path}")
        
        # Also save as image
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(pcd)
        vis.add_geometry(coord_frame)
        
        # Set view
        ctr = vis.get_view_control()
        ctr.set_zoom(0.4)
        ctr.set_front([0.5, -0.5, -0.5])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 0, 1])
        
        vis.poll_events()
        vis.update_renderer()
        
        image_path = save_path.replace('.ply', '.png')
        vis.capture_screen_image(image_path)
        print(f"Screenshot saved to {image_path}")
        
        vis.destroy_window()
    else:
        # Interactive visualization
        o3d.visualization.draw_geometries(
            [pcd, coord_frame],
            window_name="MaskPLS Semantic Segmentation",
            width=1024,
            height=768,
            left=50,
            top=50
        )
    
    # Print statistics
    unique_classes, counts = np.unique(sem_pred, return_counts=True)
    print(f"\nDetected {len(unique_classes)} classes:")
    
    class_names = ['unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 
                   'other-vehicle', 'person', 'bicyclist', 'motorcyclist',
                   'road', 'parking', 'sidewalk', 'other-ground', 'building',
                   'fence', 'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign']
    
    for cls_id, count in zip(unique_classes, counts):
        percentage = count / len(sem_pred) * 100
        if cls_id < len(class_names):
            print(f"  {class_names[cls_id]:15} (class {cls_id:2}): {count:6} points ({percentage:5.1f}%)")
        else:
            print(f"  Class {cls_id:2}: {count:6} points ({percentage:5.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize segmentation results')
    parser.add_argument('model', help='Path to .pt model')
    parser.add_argument('pointcloud', help='Path to .bin file')
    parser.add_argument('--save', help='Save visualization to file (.ply)')
    
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        exit(1)
    
    if not Path(args.pointcloud).exists():
        print(f"Error: Point cloud not found: {args.pointcloud}")
        exit(1)
    
    visualize_with_open3d(args.model, args.pointcloud, args.save)