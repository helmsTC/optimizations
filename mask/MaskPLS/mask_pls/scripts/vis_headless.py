#!/usr/bin/env python3
"""
Headless script to run inference and save colored PLY files
No display required - runs on servers/clusters
"""

import numpy as np
import torch
import argparse
from pathlib import Path
import struct


class HeadlessInference:
    """Run inference and save PLY without display"""
    
    def __init__(self, model_path, device='cuda'):
        print(f"Loading model from {model_path}")
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")
        
        # KITTI color map (BGR -> RGB for PLY)
        self.color_map = np.array([
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
        ], dtype=np.uint8)
        
        self.class_names = ['unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 
                           'other-vehicle', 'person', 'bicyclist', 'motorcyclist',
                           'road', 'parking', 'sidewalk', 'other-ground', 'building',
                           'fence', 'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign']
    
    def process_bin_file(self, bin_path):
        """Process a .bin point cloud file"""
        # Load point cloud
        print(f"\nProcessing: {bin_path}")
        points_raw = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        points = points_raw[:, :3]
        intensity = points_raw[:, 3]
        
        print(f"  Loaded {len(points)} points")
        
        # Filter points (KITTI bounds)
        mask = (
            (points[:, 0] > -48) & (points[:, 0] < 48) &
            (points[:, 1] > -48) & (points[:, 1] < 48) &
            (points[:, 2] > -4) & (points[:, 2] < 1.5)
        )
        points = points[mask]
        intensity = intensity[mask]
        features = points_raw[mask]
        
        print(f"  After filtering: {len(points)} points")
        
        # Subsample if needed
        max_points = 50000
        if len(points) > max_points:
            idx = np.random.choice(len(points), max_points, replace=False)
            points = points[idx]
            intensity = intensity[idx]
            features = features[idx]
            print(f"  Subsampled to {max_points} points")
        
        # Prepare tensors
        points_tensor = torch.from_numpy(points).float().unsqueeze(0).to(self.device)
        features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        
        # Run inference
        print("  Running inference...")
        with torch.no_grad():
            outputs = self.model(points_tensor, features_tensor)
        
        # Get predictions
        pred_logits = outputs[0][0]  # [Q, C+1]
        pred_masks = outputs[1][0]   # [Q, N] or [N, Q]
        sem_logits = outputs[2][0]   # [N, C]
        
        # Semantic predictions
        sem_pred = torch.argmax(sem_logits, dim=-1).cpu().numpy()
        sem_scores = torch.softmax(sem_logits, dim=-1).max(dim=-1)[0].cpu().numpy()
        
        # Instance predictions (simplified)
        query_classes = torch.argmax(pred_logits, dim=-1).cpu().numpy()
        query_scores = torch.softmax(pred_logits, dim=-1).max(dim=-1)[0].cpu().numpy()
        
        # Get instance masks
        if pred_masks.shape[0] != len(points):
            pred_masks = pred_masks.transpose(0, 1)
        
        mask_probs = torch.sigmoid(pred_masks).cpu().numpy()
        
        # Simple instance assignment
        instance_pred = np.zeros(len(points), dtype=np.int32)
        valid_queries = query_classes < 20  # Not background
        
        if valid_queries.any():
            valid_masks = mask_probs[:, valid_queries]
            valid_scores = query_scores[valid_queries]
            
            # Weight by query scores
            weighted_masks = valid_masks * valid_scores[None, :]
            
            # Assign instances
            max_scores = weighted_masks.max(axis=1)
            instance_pred = weighted_masks.argmax(axis=1) + 1
            instance_pred[max_scores < 0.5] = 0
        
        # Statistics
        unique_classes, counts = np.unique(sem_pred, return_counts=True)
        print(f"  Detected {len(unique_classes)} semantic classes")
        for cls_id, count in zip(unique_classes, counts):
            if cls_id < len(self.class_names):
                print(f"    {self.class_names[cls_id]:15}: {count:6} points")
        
        num_instances = len(np.unique(instance_pred)) - 1
        print(f"  Detected {num_instances} instances")
        
        return {
            'points': points,
            'semantic': sem_pred,
            'instance': instance_pred,
            'intensity': intensity,
            'semantic_scores': sem_scores
        }
    
    def save_ply(self, data, output_path, color_mode='semantic'):
        """
        Save point cloud as PLY file with colors
        
        Args:
            data: Dictionary with points, semantic, instance, etc.
            output_path: Output PLY file path
            color_mode: 'semantic', 'instance', or 'intensity'
        """
        points = data['points']
        semantic = data['semantic']
        instance = data['instance']
        intensity = data['intensity']
        
        # Generate colors based on mode
        if color_mode == 'semantic':
            # Extend color map if needed
            if semantic.max() >= len(self.color_map):
                extra_colors = np.random.randint(0, 255, (semantic.max() - len(self.color_map) + 1, 3), dtype=np.uint8)
                color_map = np.vstack([self.color_map, extra_colors])
            else:
                color_map = self.color_map
            colors = color_map[semantic]
            
        elif color_mode == 'instance':
            # Random colors for instances
            num_instances = instance.max() + 1
            instance_colors = np.random.randint(0, 255, (num_instances, 3), dtype=np.uint8)
            instance_colors[0] = [100, 100, 100]  # Gray for no instance
            colors = instance_colors[instance]
            
        else:  # intensity
            # Map intensity to grayscale
            intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-8)
            intensity_uint8 = (intensity_norm * 255).astype(np.uint8)
            colors = np.stack([intensity_uint8, intensity_uint8, intensity_uint8], axis=1)
        
        # Write PLY file
        self.write_ply(output_path, points, colors, semantic, instance)
        print(f"  Saved PLY to {output_path}")
    
    def write_ply(self, filename, points, colors, semantic=None, instance=None):
        """Write PLY file with points and colors"""
        num_points = len(points)
        
        # Prepare vertex data
        vertex_data = []
        for i in range(num_points):
            vertex_data.append((
                points[i, 0], points[i, 1], points[i, 2],
                colors[i, 0], colors[i, 1], colors[i, 2],
                semantic[i] if semantic is not None else 0,
                instance[i] if instance is not None else 0
            ))
        
        # Write PLY
        with open(filename, 'wb') as f:
            # Header
            header = f"""ply
format binary_little_endian 1.0
comment MaskPLS segmentation results
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property int semantic_class
property int instance_id
end_header
"""
            f.write(header.encode('ascii'))
            
            # Write binary data
            for v in vertex_data:
                # x, y, z as float32
                f.write(struct.pack('<fff', v[0], v[1], v[2]))
                # r, g, b as uint8
                f.write(struct.pack('<BBB', v[3], v[4], v[5]))
                # semantic and instance as int32
                f.write(struct.pack('<ii', v[6], v[7]))
    
    def save_npz(self, data, output_path):
        """Save results as compressed NumPy archive"""
        np.savez_compressed(
            output_path,
            points=data['points'],
            semantic=data['semantic'],
            instance=data['instance'],
            intensity=data['intensity'],
            semantic_scores=data['semantic_scores']
        )
        print(f"  Saved NPZ to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Headless inference and PLY export')
    parser.add_argument('model', help='Path to .pt model')
    parser.add_argument('input', help='Path to .bin file or directory')
    parser.add_argument('--output', default='./output', help='Output directory')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--color', choices=['semantic', 'instance', 'intensity'], 
                       default='semantic', help='Coloring mode for PLY')
    parser.add_argument('--save-npz', action='store_true', help='Also save as NPZ')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize inference
    inferencer = HeadlessInference(args.model, args.device)
    
    # Get input files
    input_path = Path(args.input)
    if input_path.is_file():
        bin_files = [input_path]
    else:
        bin_files = list(input_path.glob('**/*.bin'))
        print(f"Found {len(bin_files)} .bin files")
    
    # Process each file
    for bin_file in bin_files:
        # Run inference
        results = inferencer.process_bin_file(str(bin_file))
        
        # Prepare output paths
        if input_path.is_dir():
            # For directory input, preserve relative structure
            rel_path = bin_file.relative_to(input_path)
            output_name = str(rel_path.with_suffix('')).replace('/', '_').replace('\\', '_')
        else:
            # For single file input
            output_name = bin_file.stem
        
        # Save PLY with semantic colors
        ply_path_sem = output_dir / f"{output_name}_semantic.ply"
        inferencer.save_ply(results, str(ply_path_sem), color_mode='semantic')
        
        # Save PLY with instance colors
        ply_path_ins = output_dir / f"{output_name}_instance.ply"
        inferencer.save_ply(results, str(ply_path_ins), color_mode='instance')
        
        # Save NPZ if requested
        if args.save_npz:
            npz_path = output_dir / f"{output_name}_results.npz"
            inferencer.save_npz(results, str(npz_path))
    
    print(f"\nProcessed {len(bin_files)} files")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()