"""
Complete example of using the MaskPLS ONNX model for inference
"""

import numpy as np
import onnxruntime as ort
import time
from pathlib import Path
from typing import Dict, List, Tuple


class MaskPLSONNXInference:
    """
    Complete inference pipeline for MaskPLS ONNX model
    """
    def __init__(self, 
                 onnx_path: str,
                 dataset: str = 'KITTI',
                 providers: List[str] = None):
        """
        Args:
            onnx_path: Path to ONNX model
            dataset: 'KITTI' or 'NUSCENES'
            providers: ONNX Runtime providers (default: auto-detect)
        """
        self.onnx_path = onnx_path
        self.dataset = dataset
        
        # Dataset-specific configuration
        if dataset == 'KITTI':
            self.num_classes = 20
            self.spatial_shape = (32, 32, 16)
            self.coordinate_bounds = [[-48.0, 48.0], [-48.0, 48.0], [-4.0, 1.5]]
            self.things_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        else:  # NuScenes
            self.num_classes = 17
            self.spatial_shape = (32, 32, 16)
            self.coordinate_bounds = [[-50.0, 50.0], [-50.0, 50.0], [-5.0, 3.0]]
            self.things_ids = [2, 3, 4, 5, 6, 7, 9, 10]
        
        # Create preprocessor
        from preprocess_for_onnx import SemanticKITTIPreprocessor
        self.preprocessor = SemanticKITTIPreprocessor(dataset)
        
        # Initialize ONNX Runtime
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        print(f"Loading ONNX model from {onnx_path}")
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        print(f"Using providers: {self.session.get_providers()}")
        
        # Get input/output names
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
    def preprocess(self, points: np.ndarray, features: np.ndarray) -> Dict:
        """Preprocess point cloud"""
        return self.preprocessor.preprocess_single(points, features)
    
    def inference(self, voxel_features: np.ndarray, point_coords: np.ndarray) -> Dict:
        """
        Run ONNX model inference
        
        Args:
            voxel_features: [B, C, D, H, W] voxelized features
            point_coords: [B, N, 3] normalized point coordinates
        
        Returns:
            Model outputs dictionary
        """
        # Ensure correct dtypes
        voxel_features = voxel_features.astype(np.float32)
        point_coords = point_coords.astype(np.float32)
        
        # Run inference
        outputs = self.session.run(self.output_names, {
            self.input_names[0]: voxel_features,
            self.input_names[1]: point_coords
        })
        
        return {
            'pred_logits': outputs[0],
            'pred_masks': outputs[1],
            'sem_logits': outputs[2]
        }
    
    def postprocess(self, outputs: Dict, num_points: int) -> Dict:
        """
        Postprocess model outputs to get final predictions
        
        Args:
            outputs: Model outputs
            num_points: Number of valid points
        
        Returns:
            Dictionary with semantic and instance predictions
        """
        pred_logits = outputs['pred_logits'][0]  # [Q, C+1]
        pred_masks = outputs['pred_masks'][0]    # [N, Q]
        sem_logits = outputs['sem_logits'][0]    # [N, C]
        
        # Only use valid points
        pred_masks = pred_masks[:num_points]
        sem_logits = sem_logits[:num_points]
        
        # Semantic segmentation
        sem_pred = np.argmax(sem_logits, axis=-1)
        
        # Panoptic segmentation
        scores, labels = np.max(pred_logits, axis=-1), np.argmax(pred_logits, axis=-1)
        
        # Filter out no-object predictions
        keep = labels < self.num_classes
        
        if np.any(keep):
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = pred_masks[:, keep]
            
            # Apply sigmoid to masks
            cur_masks = 1 / (1 + np.exp(-cur_masks))
            
            # Get instance predictions
            cur_prob_masks = cur_scores * cur_masks
            mask_ids = np.argmax(cur_prob_masks, axis=1)
            
            # Create instance segmentation
            ins_pred = np.zeros_like(sem_pred)
            instance_id = 1
            
            for k in range(len(cur_classes)):
                mask = (mask_ids == k) & (cur_masks[:, k] >= 0.5)
                if np.sum(mask) > 0:
                    # Check if it's a thing class
                    if cur_classes[k] in self.things_ids:
                        ins_pred[mask] = instance_id
                        instance_id += 1
                    # Update semantic prediction
                    sem_pred[mask] = cur_classes[k]
        else:
            ins_pred = np.zeros_like(sem_pred)
        
        return {
            'semantic': sem_pred,
            'instance': ins_pred
        }
    
    def process_pointcloud(self, points: np.ndarray, features: np.ndarray) -> Dict:
        """
        Complete pipeline: preprocess -> inference -> postprocess
        
        Args:
            points: [N, 3] point coordinates
            features: [N, C] point features
        
        Returns:
            Segmentation results
        """
        # Preprocess
        start_time = time.time()
        preprocessed = self.preprocess(points, features)
        preprocess_time = time.time() - start_time
        
        # Prepare for inference (add batch dimension)
        voxel_features = preprocessed['voxel_features'][np.newaxis, ...]
        point_coords = preprocessed['point_coords'][np.newaxis, ...]
        
        # Inference
        start_time = time.time()
        outputs = self.inference(voxel_features, point_coords)
        inference_time = time.time() - start_time
        
        # Postprocess
        start_time = time.time()
        results = self.postprocess(outputs, preprocessed['num_points'])
        postprocess_time = time.time() - start_time
        
        # Map back to original indices
        semantic_full = np.zeros(len(points), dtype=np.int32)
        instance_full = np.zeros(len(points), dtype=np.int32)
        
        valid_indices = preprocessed['point_indices'][:preprocessed['num_points']]
        semantic_full[valid_indices] = results['semantic']
        instance_full[valid_indices] = results['instance']
        
        return {
            'semantic': semantic_full,
            'instance': instance_full,
            'timing': {
                'preprocess': preprocess_time * 1000,
                'inference': inference_time * 1000,
                'postprocess': postprocess_time * 1000,
                'total': (preprocess_time + inference_time + postprocess_time) * 1000
            }
        }
    
    def process_batch(self, batch: List[Tuple[np.ndarray, np.ndarray]]) -> List[Dict]:
        """Process a batch of point clouds"""
        results = []
        for points, features in batch:
            results.append(self.process_pointcloud(points, features))
        return results


def visualize_results(points: np.ndarray, 
                     semantic: np.ndarray, 
                     instance: np.ndarray,
                     save_path: str = None):
    """
    Visualize segmentation results using Open3D
    """
    try:
        import open3d as o3d
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Color by semantic labels
        num_classes = 20
        colors = np.random.rand(num_classes + 1, 3)
        colors[0] = [0.5, 0.5, 0.5]  # Unknown/void class
        
        point_colors = colors[semantic]
        pcd.colors = o3d.utility.Vector3dVector(point_colors)
        
        # Visualize
        if save_path:
            o3d.io.write_point_cloud(save_path, pcd)
            print(f"Saved visualization to {save_path}")
        else:
            o3d.visualization.draw_geometries([pcd], window_name="Semantic Segmentation")
            
    except ImportError:
        print("Open3D not installed. Skipping visualization.")


# Example usage
def main():
    """Example of using the ONNX model"""
    
    # Configuration
    onnx_model_path = "maskpls_simplified.onnx"
    dataset = "KITTI"
    
    # Check if model exists
    if not Path(onnx_model_path).exists():
        print(f"Error: ONNX model not found at {onnx_model_path}")
        print("Please run the conversion script first.")
        return
    
    # Initialize inference pipeline
    print("Initializing MaskPLS ONNX inference pipeline...")
    pipeline = MaskPLSONNXInference(onnx_model_path, dataset)
    
    # Generate dummy point cloud for testing
    print("\nGenerating test point cloud...")
    num_points = 100000
    points = np.random.randn(num_points, 3) * 20
    intensity = np.random.rand(num_points, 1)
    features = np.concatenate([points, intensity], axis=1)
    
    # Process point cloud
    print("\nProcessing point cloud...")
    results = pipeline.process_pointcloud(points, features)
    
    # Print results
    print(f"\nResults:")
    print(f"  Semantic classes: {np.unique(results['semantic'])}")
    print(f"  Number of instances: {np.max(results['instance'])}")
    print(f"\nTiming:")
    for key, value in results['timing'].items():
        print(f"  {key}: {value:.2f} ms")
    
    # Visualize results (optional)
    visualize = input("\nVisualize results? (y/n): ").strip().lower() == 'y'
    if visualize:
        visualize_results(points, results['semantic'], results['instance'])
    
    # Benchmark
    benchmark = input("\nRun benchmark? (y/n): ").strip().lower() == 'y'
    if benchmark:
        print("\nRunning benchmark (100 iterations)...")
        times = []
        for i in range(100):
            # Generate random point cloud
            n_pts = np.random.randint(50000, 150000)
            pts = np.random.randn(n_pts, 3) * 20
            feat = np.concatenate([pts, np.random.rand(n_pts, 1)], axis=1)
            
            # Process
            res = pipeline.process_pointcloud(pts, feat)
            times.append(res['timing']['total'])
            
            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/100")
        
        times = np.array(times)
        print(f"\nBenchmark results:")
        print(f"  Mean: {np.mean(times):.2f} ms")
        print(f"  Std: {np.std(times):.2f} ms")
        print(f"  Min: {np.min(times):.2f} ms")
        print(f"  Max: {np.max(times):.2f} ms")
        print(f"  FPS: {1000 / np.mean(times):.2f}")


if __name__ == "__main__":
    main()
