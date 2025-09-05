# mask_pls/scripts/test_onnx_model_fixed.py
"""
Fixed test script for exported MaskPLS-DGCNN ONNX model
Handles the actual input/output shapes from the export
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

# Optional imports
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
    """Test suite for ONNX model - Fixed for actual export format"""
    
    def __init__(self, model_path: str, providers: List[str] = None):
        """Initialize tester"""
        self.model_path = model_path
        
        # Set providers
        if providers is None:
            available_providers = ort.get_available_providers()
            self.providers = []
            if 'CUDAExecutionProvider' in available_providers:
                self.providers.append('CUDAExecutionProvider')
            self.providers.append('CPUExecutionProvider')
        else:
            self.providers = providers
        
        print(f"Available providers: {ort.get_available_providers()}")
        print(f"Using providers: {self.providers}")
        
        # Load model
        try:
            self.session = ort.InferenceSession(model_path, providers=self.providers)
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise
        
        # Get model info
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.input_shapes = [inp.shape for inp in self.session.get_inputs()]
        self.output_shapes = [out.shape for out in self.session.get_outputs()]
        self.input_types = [inp.type for inp in self.session.get_inputs()]
        
        # Detect model type based on inputs
        self.detect_model_type()
    
    def detect_model_type(self):
        """Detect if this is semantic-only or full model"""
        if len(self.input_names) == 2:
            if 'intensity' in self.input_names[1]:
                self.model_type = 'semantic'
                print("Detected model type: Semantic segmentation only")
            else:
                self.model_type = 'full'
                print("Detected model type: Full (with instance segmentation)")
        else:
            self.model_type = 'unknown'
            print("Warning: Unknown model type")
    
    def print_model_info(self):
        """Print model information"""
        print("\n" + "="*60)
        print("MODEL INFORMATION")
        print("="*60)
        
        print(f"Model: {self.model_path}")
        print(f"Model type: {self.model_type}")
        print(f"Provider being used: {self.session.get_providers()[0]}")
        
        print("\nInputs:")
        for i, (name, shape, dtype) in enumerate(zip(self.input_names, self.input_shapes, self.input_types)):
            print(f"  {i}: {name}")
            print(f"     Shape: {shape}")
            print(f"     Type: {dtype}")
        
        print("\nOutputs:")
        for i, (name, shape) in enumerate(zip(self.output_names, self.output_shapes)):
            print(f"  {i}: {name} - Shape: {shape}")
        
        # Model size
        model_size = Path(self.model_path).stat().st_size / (1024 * 1024)
        print(f"\nModel size: {model_size:.2f} MB")
    
    def prepare_inputs(self, num_points=10000, batch_size=None):
        """Prepare inputs based on model type"""
        
        if self.model_type == 'semantic':
            # Semantic model expects: points [N, 3], intensity [N, 1]
            points = np.random.randn(num_points, 3).astype(np.float32)
            intensity = np.random.randn(num_points, 1).astype(np.float32)
            
            inputs = {
                self.input_names[0]: points,
                self.input_names[1]: intensity
            }
            
        elif self.model_type == 'full':
            # Full model expects: point_coords [N, 3], point_features [N, 4]
            # Check if model expects batch dimension
            first_shape = self.input_shapes[0]
            
            if len(first_shape) == 3 and batch_size:
                # Model expects [B, N, 3]
                points = np.random.randn(batch_size, num_points, 3).astype(np.float32)
                features = np.random.randn(batch_size, num_points, 4).astype(np.float32)
            else:
                # Model expects [N, 3]
                points = np.random.randn(num_points, 3).astype(np.float32)
                features = np.random.randn(num_points, 4).astype(np.float32)
            
            inputs = {
                self.input_names[0]: points,
                self.input_names[1]: features
            }
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return inputs
    
    def test_basic_inference(self, num_points=10000):
        """Test basic inference with random data"""
        print("\n" + "="*60)
        print("BASIC INFERENCE TEST")
        print("="*60)
        
        print(f"Testing with num_points={num_points}")
        
        # Prepare inputs
        try:
            inputs = self.prepare_inputs(num_points)
            
            print("\nInput shapes:")
            for name, arr in inputs.items():
                print(f"  {name}: {arr.shape} ({arr.dtype})")
            
        except Exception as e:
            print(f"✗ Failed to prepare inputs: {e}")
            return False
        
        # Run inference
        try:
            start = time.time()
            outputs = self.session.run(None, inputs)
            inference_time = (time.time() - start) * 1000  # ms
            
            print(f"\n✓ Inference successful in {inference_time:.2f} ms")
            
            print(f"\nOutput shapes:")
            for i, (name, output) in enumerate(zip(self.output_names, outputs)):
                print(f"  {name}: {output.shape}")
                
                # Check for NaN/Inf
                if np.any(np.isnan(output)):
                    print(f"    ⚠ Warning: {name} contains NaN")
                if np.any(np.isinf(output)):
                    print(f"    ⚠ Warning: {name} contains Inf")
            
            # Analyze based on model type
            if self.model_type == 'semantic':
                self.analyze_semantic_predictions(outputs[0])
            elif self.model_type == 'full':
                self.analyze_full_predictions(outputs)
            
            return True
            
        except Exception as e:
            print(f"✗ Inference failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def analyze_semantic_predictions(self, sem_logits):
        """Analyze semantic segmentation predictions"""
        print("\nSemantic Prediction Analysis:")
        
        sem_preds = np.argmax(sem_logits, axis=-1)
        unique_classes = np.unique(sem_preds)
        
        print(f"  Shape: {sem_logits.shape}")
        print(f"  Predicted classes: {unique_classes}")
        print(f"  Confidence range: [{np.min(sem_logits):.3f}, {np.max(sem_logits):.3f}]")
        
        # Softmax probabilities
        probs = np.exp(sem_logits) / np.sum(np.exp(sem_logits), axis=-1, keepdims=True)
        print(f"  Mean max probability: {np.mean(np.max(probs, axis=-1)):.3f}")
    
    def analyze_full_predictions(self, outputs):
        """Analyze full model predictions"""
        print("\nFull Model Prediction Analysis:")
        
        pred_logits, pred_masks, sem_logits = outputs
        
        # Semantic analysis
        print("\nSemantic branch:")
        self.analyze_semantic_predictions(sem_logits)
        
        # Instance analysis
        print("\nInstance branch:")
        print(f"  Query logits shape: {pred_logits.shape}")
        print(f"  Mask predictions shape: {pred_masks.shape}")
        
        # Analyze queries
        pred_scores = np.max(pred_logits, axis=-1)
        pred_classes = np.argmax(pred_logits, axis=-1)
        
        num_classes = sem_logits.shape[-1]
        valid_queries = pred_classes < num_classes
        
        print(f"  Valid queries: {np.sum(valid_queries)}/{len(pred_classes)}")
        print(f"  Mean confidence: {np.mean(pred_scores):.3f}")
        
        # Analyze masks
        mask_probs = 1 / (1 + np.exp(-pred_masks))
        print(f"  Mask probability range: [{np.min(mask_probs):.3f}, {np.max(mask_probs):.3f}]")
    
    def benchmark_performance(self, point_counts=[1000, 5000, 10000, 20000], num_runs=10):
        """Benchmark model performance"""
        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARK")
        print("="*60)
        
        results = []
        
        for num_points in point_counts:
            print(f"\nTesting with {num_points} points...")
            
            try:
                # Prepare input
                inputs = self.prepare_inputs(num_points)
                
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
                fps = 1000 / mean_time if mean_time > 0 else 0
                
                print(f"  Mean: {mean_time:.2f} ± {std_time:.2f} ms")
                print(f"  FPS: {fps:.2f}")
                
                results.append({
                    'num_points': num_points,
                    'mean_time': mean_time,
                    'std_time': std_time,
                    'fps': fps
                })
                
            except Exception as e:
                print(f"  Failed: {e}")
        
        # Summary
        if results:
            print("\n" + "-"*60)
            print("BENCHMARK SUMMARY")
            print("-"*60)
            print(f"{'Points':<10} {'Time (ms)':<15} {'FPS':<10}")
            print("-"*60)
            for r in results:
                print(f"{r['num_points']:<10} {r['mean_time']:.2f} ± {r['std_time']:.2f}  {r['fps']:.2f}")
        
        return results
    
    def test_with_real_data(self, data_path: str):
        """Test with real point cloud data"""
        print("\n" + "="*60)
        print("TESTING WITH REAL DATA")
        print("="*60)
        
        if not Path(data_path).exists():
            print(f"✗ Data path not found: {data_path}")
            return False
        
        # Load point cloud
        if data_path.endswith('.bin'):
            # KITTI format
            data = np.fromfile(data_path, dtype=np.float32).reshape(-1, 4)
            points = data[:, :3]
            intensity = data[:, 3:4]
        elif data_path.endswith('.npy'):
            data = np.load(data_path)
            points = data[:, :3]
            intensity = data[:, 3:4] if data.shape[1] >= 4 else np.ones((len(points), 1))
        else:
            print(f"✗ Unsupported format: {data_path}")
            return False
        
        print(f"Loaded {len(points)} points")
        
        # Prepare inputs based on model type
        if self.model_type == 'semantic':
            inputs = {
                self.input_names[0]: points.astype(np.float32),
                self.input_names[1]: intensity.astype(np.float32)
            }
        else:  # full
            features = np.concatenate([points, intensity], axis=1)
            inputs = {
                self.input_names[0]: points.astype(np.float32),
                self.input_names[1]: features.astype(np.float32)
            }
        
        # Run inference
        try:
            start = time.time()
            outputs = self.session.run(None, inputs)
            inference_time = (time.time() - start) * 1000
            
            print(f"✓ Inference successful in {inference_time:.2f} ms")
            
            # Get semantic predictions
            if self.model_type == 'semantic':
                sem_logits = outputs[0]
            else:
                sem_logits = outputs[2]
            
            sem_pred = np.argmax(sem_logits, axis=-1)
            
            # Visualize if possible
            if HAS_OPEN3D:
                self.visualize_predictions(points, sem_pred)
            
            return True
            
        except Exception as e:
            print(f"✗ Inference failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def visualize_predictions(self, points, labels):
        """Visualize predictions using Open3D"""
        if not HAS_OPEN3D:
            return
        
        print("\nVisualizing predictions...")
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Color by labels
        num_classes = labels.max() + 1
        colors = np.zeros((len(labels), 3))
        for i in range(num_classes):
            mask = labels == i
            colors[mask] = np.random.rand(3)
        
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd], window_name="Predictions")


@click.command()
@click.option('--model', '-m', required=True, help='Path to ONNX model')
@click.option('--num-points', '-n', default=10000, help='Number of points')
@click.option('--benchmark', is_flag=True, help='Run benchmark')
@click.option('--data', '-d', default=None, help='Test with real data')
@click.option('--provider', '-p', default=None, help='Force provider (CUDA/CPU)')
def main(model, num_points, benchmark, data, provider):
    """Test exported ONNX model"""
    
    print("="*60)
    print("MaskPLS-DGCNN ONNX Model Tester (Fixed)")
    print("="*60)
    
    # Set providers
    providers = None
    if provider:
        if provider.upper() == 'CUDA':
            providers = ['CUDAExecutionProvider']
        elif provider.upper() == 'CPU':
            providers = ['CPUExecutionProvider']
    
    # Create tester
    try:
        tester = ONNXModelTester(model, providers)
    except Exception as e:
        print(f"Failed to initialize tester: {e}")
        return
    
    # Print model info
    tester.print_model_info()
    
    # Run basic test
    success = tester.test_basic_inference(num_points)
    if not success:
        print("\n✗ Basic inference failed!")
        return
    
    # Benchmark
    if benchmark:
        tester.benchmark_performance()
    
    # Test with real data
    if data:
        tester.test_with_real_data(data)
    
    print("\n" + "="*60)
    print("Testing completed!")
    print("="*60)


if __name__ == "__main__":
    main()