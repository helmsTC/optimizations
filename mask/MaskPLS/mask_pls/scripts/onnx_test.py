# mask_pls/scripts/test_onnx_model.py
"""
Test script for exported MaskPLS-DGCNN ONNX model
Tests inference, performance, and optionally real data
"""

import os
import sys
import time
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path
import click
import yaml
from typing import Dict, List, Tuple, Optional

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


class ONNXModelTester:
    """Test suite for ONNX model"""
    
    def __init__(self, model_path: str, providers: List[str] = None):
        """
        Initialize tester
        
        Args:
            model_path: Path to ONNX model
            providers: List of execution providers (default: auto-detect)
        """
        self.model_path = model_path
        
        # Set providers
        if providers is None:
            available_providers = ort.get_available_providers()
            self.providers = []
            if 'CUDAExecutionProvider' in available_providers:
                self.providers.append('CUDAExecutionProvider')
            if 'CPUExecutionProvider' in available_providers:
                self.providers.append('CPUExecutionProvider')
        else:
            self.providers = providers
        
        print(f"Available providers: {ort.get_available_providers()}")
        print(f"Using providers: {self.providers}")
        
        # Load model
        self.session = ort.InferenceSession(model_path, providers=self.providers)
        
        # Get model info
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.input_shapes = [inp.shape for inp in self.session.get_inputs()]
        self.output_shapes = [out.shape for out in self.session.get_outputs()]
        
    def print_model_info(self):
        """Print model information"""
        print("\n" + "="*60)
        print("MODEL INFORMATION")
        print("="*60)
        
        print(f"Model: {self.model_path}")
        print(f"Provider being used: {self.session.get_providers()[0]}")
        
        print("\nInputs:")
        for i, (name, shape) in enumerate(zip(self.input_names, self.input_shapes)):
            print(f"  {i}: {name} - Shape: {shape}")
        
        print("\nOutputs:")
        for i, (name, shape) in enumerate(zip(self.output_names, self.output_shapes)):
            print(f"  {i}: {name} - Shape: {shape}")
        
        # Model size
        model_size = Path(self.model_path).stat().st_size / (1024 * 1024)
        print(f"\nModel size: {model_size:.2f} MB")
    
    def test_basic_inference(self, batch_size=1, num_points=10000):
        """Test basic inference with random data"""
        print("\n" + "="*60)
        print("BASIC INFERENCE TEST")
        print("="*60)
        
        print(f"Testing with batch_size={batch_size}, num_points={num_points}")
        
        # Create random input
        point_coords = np.random.randn(batch_size, num_points, 3).astype(np.float32)
        point_features = np.random.randn(batch_size, num_points, 4).astype(np.float32)
        
        inputs = {
            self.input_names[0]: point_coords,
            self.input_names[1]: point_features
        }
        
        # Run inference
        try:
            start = time.time()
            outputs = self.session.run(None, inputs)
            inference_time = (time.time() - start) * 1000  # ms
            
            print(f"✓ Inference successful in {inference_time:.2f} ms")
            
            # Analyze outputs
            pred_logits, pred_masks, sem_logits = outputs
            
            print(f"\nOutput shapes:")
            print(f"  pred_logits: {pred_logits.shape}")
            print(f"  pred_masks: {pred_masks.shape}")
            print(f"  sem_logits: {sem_logits.shape}")
            
            # Check for valid outputs
            if np.any(np.isnan(pred_logits)) or np.any(np.isinf(pred_logits)):
                print("⚠ Warning: pred_logits contains NaN or Inf")
            if np.any(np.isnan(pred_masks)) or np.any(np.isinf(pred_masks)):
                print("⚠ Warning: pred_masks contains NaN or Inf")
            if np.any(np.isnan(sem_logits)) or np.any(np.isinf(sem_logits)):
                print("⚠ Warning: sem_logits contains NaN or Inf")
            
            # Analyze predictions
            self.analyze_predictions(pred_logits, pred_masks, sem_logits)
            
            return True
            
        except Exception as e:
            print(f"✗ Inference failed: {e}")
            return False
    
    def analyze_predictions(self, pred_logits, pred_masks, sem_logits):
        """Analyze model predictions"""
        print("\nPrediction Analysis:")
        
        # Analyze semantic predictions
        sem_preds = np.argmax(sem_logits, axis=-1)
        unique_classes = np.unique(sem_preds)
        print(f"  Semantic classes predicted: {unique_classes}")
        
        # Analyze instance predictions
        pred_scores = np.max(pred_logits, axis=-1)
        pred_classes = np.argmax(pred_logits, axis=-1)
        
        # Count valid queries (not background)
        num_classes = sem_logits.shape[-1]
        valid_queries = pred_classes < num_classes  # Assuming last class is "no object"
        num_valid = np.sum(valid_queries)
        
        print(f"  Valid instance queries: {num_valid}/{pred_classes.shape[1]}")
        print(f"  Mean confidence: {np.mean(pred_scores):.3f}")
        print(f"  Max confidence: {np.max(pred_scores):.3f}")
        
        # Analyze masks
        mask_probs = 1 / (1 + np.exp(-pred_masks))  # Sigmoid
        print(f"  Mask probability range: [{np.min(mask_probs):.3f}, {np.max(mask_probs):.3f}]")
    
    def benchmark_performance(self, batch_sizes=[1], point_counts=[1000, 5000, 10000, 20000], num_runs=10):
        """Benchmark model performance with different input sizes"""
        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARK")
        print("="*60)
        
        results = []
        
        for batch_size in batch_sizes:
            for num_points in point_counts:
                print(f"\nTesting batch_size={batch_size}, num_points={num_points}")
                
                # Create input
                point_coords = np.random.randn(batch_size, num_points, 3).astype(np.float32)
                point_features = np.random.randn(batch_size, num_points, 4).astype(np.float32)
                
                inputs = {
                    self.input_names[0]: point_coords,
                    self.input_names[1]: point_features
                }
                
                # Warmup
                for _ in range(3):
                    _ = self.session.run(None, inputs)
                
                # Benchmark
                times = []
                for _ in range(num_runs):
                    start = time.time()
                    outputs = self.session.run(None, inputs)
                    times.append((time.time() - start) * 1000)  # ms
                
                mean_time = np.mean(times)
                std_time = np.std(times)
                fps = 1000 / mean_time
                
                print(f"  Mean: {mean_time:.2f} ± {std_time:.2f} ms")
                print(f"  FPS: {fps:.2f}")
                
                results.append({
                    'batch_size': batch_size,
                    'num_points': num_points,
                    'mean_time': mean_time,
                    'std_time': std_time,
                    'fps': fps
                })
        
        # Summary
        print("\n" + "-"*60)
        print("BENCHMARK SUMMARY")
        print("-"*60)
        print(f"{'Batch':<6} {'Points':<8} {'Time (ms)':<12} {'FPS':<8}")
        print("-"*60)
        for r in results:
            print(f"{r['batch_size']:<6} {r['num_points']:<8} "
                  f"{r['mean_time']:.2f} ± {r['std_time']:.2f}  {r['fps']:.2f}")
        
        return results
    
    def test_dynamic_shapes(self, num_tests=10):
        """Test model with various dynamic input shapes"""
        print("\n" + "="*60)
        print("DYNAMIC SHAPE TEST")
        print("="*60)
        
        success_count = 0
        
        for i in range(num_tests):
            # Random sizes
            batch_size = np.random.choice([1, 2, 4])
            num_points = np.random.randint(1000, 50000)
            
            print(f"Test {i+1}/{num_tests}: batch={batch_size}, points={num_points}", end=" ")
            
            # Create input
            point_coords = np.random.randn(batch_size, num_points, 3).astype(np.float32)
            point_features = np.random.randn(batch_size, num_points, 4).astype(np.float32)
            
            inputs = {
                self.input_names[0]: point_coords,
                self.input_names[1]: point_features
            }
            
            try:
                outputs = self.session.run(None, inputs)
                print("✓")
                success_count += 1
            except Exception as e:
                print(f"✗ ({str(e)[:50]}...)")
        
        print(f"\nSuccess rate: {success_count}/{num_tests} ({100*success_count/num_tests:.1f}%)")
        return success_count == num_tests
    
    def test_with_real_data(self, data_path: str, dataset: str = 'KITTI'):
        """Test with real point cloud data"""
        print("\n" + "="*60)
        print(f"TESTING WITH REAL {dataset} DATA")
        print("="*60)
        
        if not Path(data_path).exists():
            print(f"✗ Data path not found: {data_path}")
            return False
        
        # Load point cloud
        if data_path.endswith('.bin'):
            # KITTI format
            points = np.fromfile(data_path, dtype=np.float32).reshape(-1, 4)
            coords = points[:, :3]
            intensity = points[:, 3:4]
            features = points  # xyz + intensity
        elif data_path.endswith('.npy'):
            # NumPy format
            data = np.load(data_path)
            coords = data[:, :3]
            features = data[:, :4] if data.shape[1] >= 4 else np.concatenate([coords, np.ones((len(coords), 1))], axis=1)
        else:
            print(f"✗ Unsupported file format: {data_path}")
            return False
        
        print(f"Loaded {len(coords)} points")
        
        # Add batch dimension
        coords_batch = coords[np.newaxis, :, :].astype(np.float32)
        features_batch = features[np.newaxis, :, :].astype(np.float32)
        
        inputs = {
            self.input_names[0]: coords_batch,
            self.input_names[1]: features_batch
        }
        
        # Run inference
        try:
            start = time.time()
            outputs = self.session.run(None, inputs)
            inference_time = (time.time() - start) * 1000
            
            print(f"✓ Inference successful in {inference_time:.2f} ms")
            
            pred_logits, pred_masks, sem_logits = outputs
            
            # Get predictions
            sem_pred = np.argmax(sem_logits[0], axis=-1)
            
            # Visualize if possible
            if HAS_OPEN3D:
                self.visualize_predictions(coords, sem_pred)
            
            return True
            
        except Exception as e:
            print(f"✗ Inference failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def visualize_predictions(self, points, labels):
        """Visualize predictions using Open3D"""
        if not HAS_OPEN3D:
            print("Open3D not available for visualization")
            return
        
        print("\nVisualizing predictions...")
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Color by labels
        colors = self.labels_to_colors(labels)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Visualize
        o3d.visualization.draw_geometries([pcd], window_name="Semantic Predictions")
    
    def labels_to_colors(self, labels):
        """Convert labels to colors"""
        # Simple color map
        num_classes = labels.max() + 1
        colormap = plt.cm.get_cmap('tab20', num_classes) if HAS_MATPLOTLIB else None
        
        if colormap:
            colors = colormap(labels)[:, :3]
        else:
            # Manual color map
            colors = np.zeros((len(labels), 3))
            for i in range(num_classes):
                mask = labels == i
                colors[mask] = np.random.rand(3)
        
        return colors


@click.command()
@click.option('--model', '-m', required=True, help='Path to ONNX model')
@click.option('--batch-size', '-b', default=1, help='Batch size for testing')
@click.option('--num-points', '-n', default=10000, help='Number of points')
@click.option('--benchmark', is_flag=True, help='Run performance benchmark')
@click.option('--dynamic', is_flag=True, help='Test dynamic shapes')
@click.option('--data', '-d', default=None, help='Path to real point cloud data (.bin or .npy)')
@click.option('--dataset', default='KITTI', help='Dataset type (KITTI or NUSCENES)')
@click.option('--provider', '-p', default=None, help='Execution provider (CUDA or CPU)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def main(model, batch_size, num_points, benchmark, dynamic, data, dataset, provider, verbose):
    """Test ONNX model for MaskPLS-DGCNN"""
    
    print("="*60)
    print("MaskPLS-DGCNN ONNX Model Tester")
    print("="*60)
    
    # Set providers
    providers = None
    if provider:
        if provider.upper() == 'CUDA':
            providers = ['CUDAExecutionProvider']
        elif provider.upper() == 'CPU':
            providers = ['CPUExecutionProvider']
    
    # Create tester
    tester = ONNXModelTester(model, providers)
    
    # Print model info
    tester.print_model_info()
    
    # Run basic test
    success = tester.test_basic_inference(batch_size, num_points)
    if not success:
        print("\n✗ Basic inference test failed!")
        return
    
    # Run benchmark if requested
    if benchmark:
        batch_sizes = [1, 2, 4] if batch_size == 1 else [batch_size]
        point_counts = [1000, 5000, 10000, 20000, 30000]
        tester.benchmark_performance(batch_sizes, point_counts)
    
    # Test dynamic shapes if requested
    if dynamic:
        success = tester.test_dynamic_shapes()
        if not success:
            print("\n⚠ Some dynamic shape tests failed")
    
    # Test with real data if provided
    if data:
        tester.test_with_real_data(data, dataset)
    
    print("\n" + "="*60)
    print("Testing completed!")
    print("="*60)


if __name__ == "__main__":
    main()