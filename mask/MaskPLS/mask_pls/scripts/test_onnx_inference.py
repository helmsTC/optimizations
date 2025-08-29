"""
Test script for ONNX model inference on .bin point cloud files
Processes point clouds using the exported ONNX model
Saves results in SemanticKITTI format for API viewer
"""

import os
import sys
import numpy as np
import onnxruntime as ort
import torch
import click
import yaml
from pathlib import Path
import time
from collections import OrderedDict


class ONNXPointCloudProcessor:
    """Process point clouds using ONNX model"""
    
    def __init__(self, onnx_model_path, max_points=70000, voxel_shape=(64, 64, 32), 
                 coordinate_bounds=None, device='cpu', dataset='kitti', config_path=None):
        """
        Initialize the ONNX processor
        
        Args:
            onnx_model_path: Path to the ONNX model file
            max_points: Maximum number of points to process
            voxel_shape: Voxel grid dimensions (D, H, W)
            coordinate_bounds: Point cloud bounds [[-x,+x], [-y,+y], [-z,+z]]
            device: 'cpu' or 'cuda'
            dataset: Dataset type ('kitti' or 'nuscenes')
            config_path: Path to semantic-kitti.yaml or nuscenes.yaml config file
        """
        self.max_points = max_points
        self.voxel_shape = voxel_shape
        self.device = device
        self.dataset = dataset
        
        # Load class mappings from config file
        if config_path is None:
            # Try to find the config file relative to script location
            script_dir = Path(__file__).parent
            if dataset.lower() == 'nuscenes':
                config_path = script_dir.parent / 'datasets' / 'nuscenes.yaml'
            else:
                config_path = script_dir.parent / 'datasets' / 'semantic-kitti.yaml'
        
        if Path(config_path).exists():
            print(f"Loading class mappings from: {config_path}")
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Get learning_map_inv from config
            self.learning_map_inv = np.zeros(20, dtype=np.uint32)
            if 'learning_map_inv' in config:
                for k, v in config['learning_map_inv'].items():
                    if k < len(self.learning_map_inv):
                        self.learning_map_inv[k] = v
            else:
                print("Warning: learning_map_inv not found in config, using identity mapping")
                self.learning_map_inv = np.arange(20, dtype=np.uint32)
            
            # Store other useful mappings
            self.class_strings = config.get('labels', {})
            self.learning_map = config.get('learning_map', {})
            
            print(f"  Loaded {len(self.learning_map_inv)} class mappings")
        else:
            print(f"Warning: Config file not found at {config_path}, using default mappings")
            # Fallback to hardcoded mappings for KITTI
            self.learning_map_inv = np.array([
                0, 10, 11, 15, 18, 20, 30, 31, 32, 40,
                44, 48, 49, 50, 51, 70, 71, 72, 80, 81
            ], dtype=np.uint32)
            self.class_strings = {}
            self.learning_map = {}
        
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
            original_indices: mapping from processed points to original points
        """
        print("Preprocessing point cloud...")
        
        # Step 1: Filter points within bounds
        xyz = points[:, :3]
        intensity = points[:, 3:4]
        
        valid_mask = np.all((xyz >= self.bounds_min) & (xyz < self.bounds_max), axis=1)
        valid_points = xyz[valid_mask]
        valid_intensity = intensity[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        
        print(f"  Points within bounds: {valid_points.shape[0]}/{points.shape[0]}")
        
        if len(valid_points) == 0:
            print("  Warning: No points within bounds!")
            valid_points = xyz[:1]  # Use first point as fallback
            valid_intensity = intensity[:1]
            valid_indices = np.array([0])
        
        # Step 2: Subsample if too many points
        if len(valid_points) > self.max_points:
            indices = np.random.choice(len(valid_points), self.max_points, replace=False)
            valid_points = valid_points[indices]
            valid_intensity = valid_intensity[indices]
            valid_indices = valid_indices[indices]
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
        return voxel_features, padded_coords, point_mask, valid_indices
    
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
        
        return semantic_labels[:valid_points], instance_labels[:valid_points], detected_objects
    
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
        voxel_features, point_coords, point_mask, valid_indices = self.preprocess_pointcloud(points)
        
        # Run inference
        pred_logits, pred_masks, sem_logits = self.run_inference(voxel_features, point_coords)
        
        # Postprocess
        semantic_labels, instance_labels, detected_objects = self.postprocess_results(
            pred_logits, pred_masks, sem_logits, point_mask, confidence_threshold, num_classes
        )
        
        # Create full labels array (same size as original point cloud)
        full_semantic_labels = np.zeros(len(points), dtype=np.uint16)
        full_instance_labels = np.zeros(len(points), dtype=np.uint16)
        
        # Map predictions back to original indices
        num_valid = len(semantic_labels)
        if num_valid > 0 and len(valid_indices) >= num_valid:
            full_semantic_labels[valid_indices[:num_valid]] = semantic_labels
            full_instance_labels[valid_indices[:num_valid]] = instance_labels
        
        results = {
            'original_points': points,
            'semantic_labels': full_semantic_labels,
            'instance_labels': full_instance_labels,
            'detected_objects': detected_objects,
            'point_mask': point_mask,
            'valid_indices': valid_indices,
            'inference_outputs': {
                'pred_logits': pred_logits,
                'pred_masks': pred_masks,
                'sem_logits': sem_logits
            }
        }
        
        print(f"\nProcessing completed!")
        return results


def encode_semantickitti_label(semantic, instance):
    """
    Encode semantic and instance labels into SemanticKITTI format
    
    SemanticKITTI uses 32-bit integers where:
    - Lower 16 bits: semantic label (original SemanticKITTI label, not predicted class)
    - Upper 16 bits: instance label
    
    Args:
        semantic: Semantic class label (original SemanticKITTI label 0-255)
        instance: Instance ID (0-65535)
    
    Returns:
        label: 32-bit encoded label
    """
    return ((instance & 0xFFFF) << 16) | (semantic & 0xFFFF)


def save_results_semantickitti(results, output_dir, filename_base, save_pointcloud=True, 
                              learning_map_inv=None, class_strings=None):
    """Save results in SemanticKITTI format"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get labels (these are predicted classes 0-19)
    semantic_labels = results['semantic_labels']
    instance_labels = results['instance_labels']
    
    # Map predicted classes to original SemanticKITTI labels
    if learning_map_inv is not None:
        # Ensure semantic_labels are within valid range
        semantic_labels = np.clip(semantic_labels, 0, len(learning_map_inv) - 1)
        # Map to original labels
        semantic_labels_mapped = learning_map_inv[semantic_labels].astype(np.uint32)
    else:
        semantic_labels_mapped = semantic_labels.astype(np.uint32)
    
    # Encode labels in SemanticKITTI format
    encoded_labels = np.zeros(len(semantic_labels), dtype=np.uint32)
    for i in range(len(semantic_labels)):
        encoded_labels[i] = encode_semantickitti_label(
            semantic_labels_mapped[i], 
            instance_labels[i]
        )
    
    # Save label file
    label_path = output_dir / f"{filename_base}.label"
    encoded_labels.astype(np.uint32).tofile(label_path)
    print(f"\nSaved label file: {label_path}")
    print(f"  Shape: {encoded_labels.shape}")
    print(f"  Dtype: {encoded_labels.dtype}")
    print(f"  Unique predicted classes: {len(np.unique(semantic_labels))}")
    if learning_map_inv is not None:
        print(f"  Unique mapped labels: {len(np.unique(semantic_labels_mapped))}")
    print(f"  Unique instances: {len(np.unique(instance_labels[instance_labels > 0]))}")
    
    # Save point cloud (optional - usually you'd use the original)
    if save_pointcloud:
        pointcloud_path = output_dir / f"{filename_base}.bin"
        results['original_points'].astype(np.float32).tofile(pointcloud_path)
        print(f"Saved point cloud: {pointcloud_path}")
    
    # Save detected objects info as text file for reference
    import json
    objects_info = []
    for obj in results['detected_objects']:
        info = {k: v for k, v in obj.items() if k != 'mask'}  # Don't save mask array
        # Convert numpy types to Python native types
        info['confidence'] = float(info['confidence'])
        info['instance_id'] = int(info['instance_id'])
        info['class_id'] = int(info['class_id'])
        info['num_points'] = int(info['num_points'])
        objects_info.append(info)
    
    objects_path = output_dir / f"{filename_base}_objects.json"
    with open(objects_path, 'w') as f:
        json.dump(objects_info, f, indent=2)
    print(f"Saved object detections: {objects_path}")
    
    # Create a simple statistics file
    stats_path = output_dir / f"{filename_base}_stats.txt"
    with open(stats_path, 'w') as f:
        f.write(f"Point cloud statistics for {filename_base}\n")
        f.write(f"{'='*50}\n")
        f.write(f"Total points: {len(results['original_points'])}\n")
        f.write(f"Labeled points: {np.sum(semantic_labels > 0)}\n")
        f.write(f"Unlabeled points: {np.sum(semantic_labels == 0)}\n")
        f.write(f"\nPredicted classes (0-19):\n")
        unique_classes, counts = np.unique(semantic_labels, return_counts=True)
        for cls, count in zip(unique_classes, counts):
            if learning_map_inv is not None and cls < len(learning_map_inv):
                mapped_label = learning_map_inv[cls]
                class_name = class_strings.get(int(mapped_label), "unknown") if class_strings else "unknown"
                f.write(f"  Class {cls} -> Label {mapped_label} ({class_name}): {count} points\n")
            else:
                f.write(f"  Class {cls}: {count} points\n")
        f.write(f"\nInstances detected: {len(results['detected_objects'])}\n")
        for obj in results['detected_objects']:
            f.write(f"  Instance {obj['instance_id']}: Class {obj['class_id']}, "
                   f"{obj['num_points']} points, confidence {obj['confidence']:.3f}\n")
    print(f"Saved statistics: {stats_path}")


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
@click.option('--save_pointcloud', is_flag=True, help='Save point cloud .bin file (default: False)')
@click.option('--config', type=click.Path(exists=True), help='Path to semantic-kitti.yaml or nuscenes.yaml')
def main(onnx_model, pointcloud_path, max_points, confidence, num_classes, output_dir, device, 
         dataset, batch_process, save_pointcloud, config):
    """
    Test ONNX model inference on point cloud files and save in SemanticKITTI format
    
    Examples:
        # Process single file
        python test_onnx_inference.py model.onnx pointcloud.bin
        
        # Process with custom settings
        python test_onnx_inference.py model.onnx pointcloud.bin --max_points 70000 --confidence 0.6 --device cuda
        
        # Process all .bin files in directory
        python test_onnx_inference.py model.onnx /path/to/bin/files/ --batch_process
        
        # Save point cloud along with labels
        python test_onnx_inference.py model.onnx pointcloud.bin --save_pointcloud
    
    Output format:
        - *.label: SemanticKITTI format labels (32-bit: upper 16 bits = instance, lower 16 bits = semantic)
        - *.bin: Original point cloud (if --save_pointcloud is set)
        - *_objects.json: Detected objects metadata
        - *_stats.txt: Processing statistics
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
        device=device,
        dataset=dataset,
        config_path=config
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
                    save_results_semantickitti(results, output_dir, bin_file.stem, save_pointcloud, 
                                              processor.learning_map_inv if dataset.lower() == 'kitti' else None,
                                              processor.class_strings if hasattr(processor, 'class_strings') else None)
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
        save_results_semantickitti(results, output_dir, pointcloud_path.stem, save_pointcloud,
                                  processor.learning_map_inv if dataset.lower() == 'kitti' else None,
                                  processor.class_strings if hasattr(processor, 'class_strings') else None)
    
    print(f"\n{'='*60}")
    print("Processing complete! Files saved in SemanticKITTI format.")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
