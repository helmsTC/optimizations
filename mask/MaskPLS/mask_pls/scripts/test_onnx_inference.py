"""
Test script for ONNX model inference on .bin point cloud files
Processes point clouds using the exported ONNX model
"""

import os
import sys
import numpy as np
import onnxruntime as ort
import torch
import click
from pathlib import Path
import time
from collections import OrderedDict


class ONNXPointCloudProcessor:
    """Process point clouds using ONNX model"""
    
    def __init__(self, onnx_model_path, max_points=70000, voxel_shape=(64, 64, 32), 
                 coordinate_bounds=None, device='cpu'):
        """
        Initialize the ONNX processor
        
        Args:
            onnx_model_path: Path to the ONNX model file
            max_points: Maximum number of points to process
            voxel_shape: Voxel grid dimensions (D, H, W)
            coordinate_bounds: Point cloud bounds [[-x,+x], [-y,+y], [-z,+z]]
            device: 'cpu' or 'cuda'
        """
        self.max_points = max_points
        self.voxel_shape = voxel_shape
        self.device = device
        
        # Default KITTI bounds if not provided
        if coordinate_bounds is None:
            self.coordinate_bounds = [[-48.0, 48.0], [-48.0, 48.0], [-4.0, 1.5]]
        else:
            self.coordinate_bounds = coordinate_bounds
        
        # Precompute bounds for efficiency
        self.bounds_min = np.array([bounds[0] for bounds in self.coordinate_bounds])
        self.bounds_max = np.array([bounds[1] for bounds in self.coordinate_bounds])
        self.bounds_range = self.bounds_max - self.bounds_min
        
        print(f"Loading ONNX model: {onnx_model_path}")
        
        # Setup ONNX Runtime
        providers = ['CPUExecutionProvider']
        if device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print("Using CUDA for inference")
        else:
            print("Using CPU for inference")
        
        self.session = ort.InferenceSession(onnx_model_path, providers=providers)
        
        # Print model info
        print("\nModel Information:")
        print(f"  Input shapes:")
        for input_info in self.session.get_inputs():
            print(f"    {input_info.name}: {input_info.shape}")
        print(f"  Output shapes:")
        for output_info in self.session.get_outputs():
            print(f"    {output_info.name}: {output_info.shape}")
        print(f"  Processing parameters:")
        print(f"    Max points: {self.max_points}")
        print(f"    Voxel shape: {self.voxel_shape}")
        print(f"    Coordinate bounds: {self.coordinate_bounds}")
        print("")
        
    def load_bin_pointcloud(self, bin_path):
        """
        Load point cloud from .bin file (KITTI format)
        
        Args:
            bin_path: Path to .bin file
            
        Returns:
            points: numpy array of shape [N, 4] (x, y, z, intensity)
        """
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        print(f"Loaded point cloud: {points.shape[0]} points from {bin_path}")
        return points
    
    def preprocess_pointcloud(self, points):
        """
        Preprocess point cloud for ONNX model
        
        Args:
            points: numpy array of shape [N, 4] (x, y, z, intensity)
            
        Returns:
            voxel_features: numpy array [1, 4, D, H, W]
            point_coords: numpy array [1, max_points, 3]
            valid_mask: boolean mask for valid points
        """
        print("Preprocessing point cloud...")
        
        # Step 1: Filter points within bounds
        xyz = points[:, :3]
        intensity = points[:, 3:4]
        
        valid_mask = np.all((xyz >= self.bounds_min) & (xyz < self.bounds_max), axis=1)
        valid_points = xyz[valid_mask]
        valid_intensity = intensity[valid_mask]
        
        print(f"  Points within bounds: {valid_points.shape[0]}/{points.shape[0]}")
        
        if len(valid_points) == 0:
            print("  Warning: No points within bounds!")
            valid_points = xyz[:1]  # Use first point as fallback
            valid_intensity = intensity[:1]
        
        # Step 2: Subsample if too many points
        if len(valid_points) > self.max_points:
            indices = np.random.choice(len(valid_points), self.max_points, replace=False)
            valid_points = valid_points[indices]
            valid_intensity = valid_intensity[indices]
            print(f"  Subsampled to: {len(valid_points)} points")
        
        # Step 3: Normalize coordinates to [0, 1]
        normalized_coords = (valid_points - self.bounds_min) / self.bounds_range
        normalized_coords = np.clip(normalized_coords, 0, 0.999)
        
        # Step 4: Create voxel grid
        D, H, W = self.voxel_shape
        voxel_features = np.zeros((4, D, H, W), dtype=np.float32)
        voxel_counts = np.zeros((D, H, W), dtype=np.float32)
        
        # Convert to voxel indices
        voxel_indices = (normalized_coords * np.array([D, H, W])).astype(np.int32)
        voxel_indices = np.clip(voxel_indices, [0, 0, 0], [D-1, H-1, W-1])
        
        # Accumulate features in voxels
        for i, (d, h, w) in enumerate(voxel_indices):
            voxel_features[0, d, h, w] += valid_points[i, 0]  # x
            voxel_features[1, d, h, w] += valid_points[i, 1]  # y
            voxel_features[2, d, h, w] += valid_points[i, 2]  # z
            voxel_features[3, d, h, w] += valid_intensity[i, 0]  # intensity
            voxel_counts[d, h, w] += 1
        
        # Average the accumulated features
        non_empty = voxel_counts > 0
        voxel_features[:, non_empty] /= voxel_counts[non_empty]
        
        print(f"  Non-empty voxels: {np.sum(non_empty)}/{D*H*W}")
        
        # Step 5: Prepare point coordinates (pad to max_points)
        if len(normalized_coords) < self.max_points:
            # Pad with zeros
            padded_coords = np.zeros((self.max_points, 3), dtype=np.float32)
            padded_coords[:len(normalized_coords)] = normalized_coords
            point_mask = np.zeros(self.max_points, dtype=bool)
            point_mask[:len(normalized_coords)] = True
        else:
            padded_coords = normalized_coords[:self.max_points]
            point_mask = np.ones(self.max_points, dtype=bool)
        
        # Add batch dimension
        voxel_features = voxel_features[np.newaxis, ...]  # [1, 4, D, H, W]
        padded_coords = padded_coords[np.newaxis, ...]    # [1, max_points, 3]
        
        print("  Preprocessing complete")
        return voxel_features, padded_coords, point_mask
    
    def run_inference(self, voxel_features, point_coords):
        """
        Run inference using ONNX model
        
        Args:
            voxel_features: numpy array [1, 4, D, H, W]
            point_coords: numpy array [1, max_points, 3]
            
        Returns:
            pred_logits: Class predictions [1, num_queries, num_classes+1]
            pred_masks: Mask predictions [1, max_points, num_queries]
            sem_logits: Semantic predictions [1, max_points, num_classes]
        """
        print("Running ONNX inference...")
        
        # Prepare inputs
        inputs = {
            'voxel_features': voxel_features.astype(np.float32),
            'point_coords': point_coords.astype(np.float32)
        }
        
        # Run inference
        start_time = time.time()
        outputs = self.session.run(None, inputs)
        inference_time = time.time() - start_time
        
        pred_logits, pred_masks, sem_logits = outputs
        
        print(f"  Inference completed in {inference_time:.3f}s")
        print(f"  Output shapes:")
        print(f"    pred_logits: {pred_logits.shape}")
        print(f"    pred_masks: {pred_masks.shape}")
        print(f"    sem_logits: {sem_logits.shape}")
        
        return pred_logits, pred_masks, sem_logits
    
    def postprocess_results(self, pred_logits, pred_masks, sem_logits, point_mask, 
                           confidence_threshold=0.5, num_classes=20):
        """
        Postprocess inference results
        
        Args:
            pred_logits: Class predictions [1, num_queries, num_classes+1]
            pred_masks: Mask predictions [1, max_points, num_queries]
            sem_logits: Semantic predictions [1, max_points, num_classes]
            point_mask: Valid point mask
            confidence_threshold: Confidence threshold for masks
            num_classes: Number of semantic classes
            
        Returns:
            semantic_labels: Semantic class for each point
            instance_labels: Instance ID for each point
            detected_objects: List of detected objects with metadata
        """
        print("Postprocessing results...")
        
        # Remove batch dimension
        pred_logits = pred_logits[0]  # [num_queries, num_classes+1]
        pred_masks = pred_masks[0]    # [max_points, num_queries]
        sem_logits = sem_logits[0]    # [max_points, num_classes]
        
        # Only process valid points
        valid_points = np.sum(point_mask)
        pred_masks_valid = pred_masks[:valid_points]
        sem_logits_valid = sem_logits[:valid_points]
        
        # Semantic segmentation
        semantic_probs = torch.softmax(torch.from_numpy(sem_logits_valid), dim=1).numpy()
        semantic_labels = np.argmax(semantic_probs, axis=1)
        
        # Instance segmentation
        # Get class predictions and scores
        class_probs = torch.softmax(torch.from_numpy(pred_logits), dim=1).numpy()
        scores, predicted_classes = np.max(class_probs[:, :-1], axis=1), np.argmax(class_probs[:, :-1], axis=1)
        
        # Keep predictions with good confidence
        keep_mask = scores > confidence_threshold
        valid_queries = np.where(keep_mask)[0]
        
        instance_labels = np.zeros(valid_points, dtype=np.int32)
        detected_objects = []
        
        instance_id = 1
        for query_idx in valid_queries:
            # Get mask for this query
            mask_logits = pred_masks_valid[:, query_idx]
            mask_probs = 1 / (1 + np.exp(-mask_logits))  # sigmoid
            mask = mask_probs > confidence_threshold
            
            if np.sum(mask) > 10:  # Minimum points for valid instance
                instance_labels[mask] = instance_id
                
                detected_objects.append({
                    'instance_id': instance_id,
                    'class_id': predicted_classes[query_idx],
                    'confidence': scores[query_idx],
                    'num_points': np.sum(mask),
                    'mask': mask
                })
                
                instance_id += 1
        
        print(f"  Valid points processed: {valid_points}")
        print(f"  Detected instances: {len(detected_objects)}")
        print(f"  Semantic classes found: {len(np.unique(semantic_labels))}")
        
        # Extend to full point cloud size (pad with zeros)
        full_semantic = np.zeros(self.max_points, dtype=np.int32)
        full_instance = np.zeros(self.max_points, dtype=np.int32)
        full_semantic[:valid_points] = semantic_labels
        full_instance[:valid_points] = instance_labels
        
        return full_semantic, full_instance, detected_objects
    
    def process_pointcloud(self, bin_path, confidence_threshold=0.5, num_classes=20):
        """
        Complete pipeline to process a point cloud
        
        Args:
            bin_path: Path to .bin file
            confidence_threshold: Confidence threshold for detections
            num_classes: Number of semantic classes
            
        Returns:
            results: Dictionary with all results
        """
        print(f"\n{'='*60}")
        print(f"Processing: {bin_path}")
        print(f"{'='*60}")
        
        # Load point cloud
        points = self.load_bin_pointcloud(bin_path)
        
        # Preprocess
        voxel_features, point_coords, point_mask = self.preprocess_pointcloud(points)
        
        # Run inference
        pred_logits, pred_masks, sem_logits = self.run_inference(voxel_features, point_coords)
        
        # Postprocess
        semantic_labels, instance_labels, detected_objects = self.postprocess_results(
            pred_logits, pred_masks, sem_logits, point_mask, confidence_threshold, num_classes
        )
        
        results = {
            'original_points': points,
            'semantic_labels': semantic_labels,
            'instance_labels': instance_labels,
            'detected_objects': detected_objects,
            'point_mask': point_mask,
            'inference_outputs': {
                'pred_logits': pred_logits,
                'pred_masks': pred_masks,
                'sem_logits': sem_logits
            }
        }
        
        print(f"\nProcessing completed!")
        return results


def save_results(results, output_dir, filename_base):
    """Save results to files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save semantic labels
    semantic_path = output_dir / f"{filename_base}_semantic.npy"
    np.save(semantic_path, results['semantic_labels'])
    
    # Save instance labels
    instance_path = output_dir / f"{filename_base}_instance.npy"
    np.save(instance_path, results['instance_labels'])
    
    # Save detected objects info
    import json
    objects_info = []
    for obj in results['detected_objects']:
        info = {k: v for k, v in obj.items() if k != 'mask'}  # Don't save mask array
        info['confidence'] = float(info['confidence'])  # Convert to JSON serializable
        objects_info.append(info)
    
    objects_path = output_dir / f"{filename_base}_objects.json"
    with open(objects_path, 'w') as f:
        json.dump(objects_info, f, indent=2)
    
    # Save colored point cloud (if requested)
    colored_points = np.column_stack([
        results['original_points'][:len(results['semantic_labels'])],
        results['semantic_labels'][:len(results['original_points'])],
        results['instance_labels'][:len(results['original_points'])]
    ])
    colored_path = output_dir / f"{filename_base}_colored.npy"
    np.save(colored_path, colored_points)
    
    print(f"\nResults saved to {output_dir}:")
    print(f"  - Semantic labels: {semantic_path}")
    print(f"  - Instance labels: {instance_path}")
    print(f"  - Object detections: {objects_path}")
    print(f"  - Colored point cloud: {colored_path}")


@click.command()
@click.argument('onnx_model', type=click.Path(exists=True))
@click.argument('pointcloud_path', type=click.Path())
@click.option('--max_points', default=70000, help='Maximum points to process')
@click.option('--confidence', default=0.5, help='Confidence threshold for detections')
@click.option('--num_classes', default=20, help='Number of semantic classes (20 for KITTI, 17 for NuScenes)')
@click.option('--output_dir', default='results', help='Output directory for results')
@click.option('--device', default='cpu', help='Device to use: cpu or cuda')
@click.option('--dataset', default='kitti', help='Dataset type: kitti or nuscenes')
@click.option('--batch_process', is_flag=True, help='Process all .bin files in directory')
def main(onnx_model, pointcloud_path, max_points, confidence, num_classes, output_dir, device, dataset, batch_process):
    """
    Test ONNX model inference on point cloud files
    
    Examples:
        # Process single file
        python test_onnx_inference.py model.onnx pointcloud.bin
        
        # Process with custom settings
        python test_onnx_inference.py model.onnx pointcloud.bin --max_points 70000 --confidence 0.6 --device cuda
        
        # Process all .bin files in directory
        python test_onnx_inference.py model.onnx /path/to/bin/files/ --batch_process
    """
    
    # Set coordinate bounds based on dataset
    if dataset.lower() == 'nuscenes':
        coordinate_bounds = [[-50.0, 50.0], [-50.0, 50.0], [-5.0, 3.0]]
        num_classes = 17 if num_classes == 20 else num_classes  # Auto-adjust if default
    else:  # KITTI
        coordinate_bounds = [[-48.0, 48.0], [-48.0, 48.0], [-4.0, 1.5]]
    
    # Initialize processor
    processor = ONNXPointCloudProcessor(
        onnx_model_path=onnx_model,
        max_points=max_points,
        coordinate_bounds=coordinate_bounds,
        device=device
    )
    
    pointcloud_path = Path(pointcloud_path)
    
    if batch_process or pointcloud_path.is_dir():
        # Process all .bin files in directory
        if pointcloud_path.is_dir():
            bin_files = list(pointcloud_path.glob("*.bin"))
            if not bin_files:
                print(f"No .bin files found in {pointcloud_path}")
                return
            
            print(f"Found {len(bin_files)} .bin files to process")
            
            for bin_file in bin_files:
                try:
                    results = processor.process_pointcloud(
                        bin_file, confidence, num_classes
                    )
                    save_results(results, output_dir, bin_file.stem)
                except Exception as e:
                    print(f"Error processing {bin_file}: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            print("Error: Path is not a directory for batch processing")
    else:
        # Process single file
        if not pointcloud_path.suffix == '.bin':
            print("Error: File must have .bin extension")
            return
        
        results = processor.process_pointcloud(
            pointcloud_path, confidence, num_classes
        )
        save_results(results, output_dir, pointcloud_path.stem)


if __name__ == "__main__":
    main()