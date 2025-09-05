# inference_onnx_gpu.py
"""
Example of running the exported ONNX model on GPU
Shows that even though export was on CPU, inference runs on GPU
"""

import numpy as np
import onnxruntime as ort
import time

def run_inference_gpu(onnx_path, num_points=50000, num_runs=10):
    """
    Run ONNX model inference on GPU
    """
    print("="*60)
    print("ONNX GPU Inference")
    print("="*60)
    
    # Create session with CUDA provider (GPU)
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    print("\nCreating ONNX Runtime session...")
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    # Check which provider is being used
    actual_provider = session.get_providers()[0]
    print(f"✓ Using provider: {actual_provider}")
    
    if 'CUDA' in actual_provider:
        print("✓ Model is running on GPU!")
    else:
        print("⚠ Model is running on CPU (CUDA not available)")
    
    # Get input/output names
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    
    print(f"\nModel Info:")
    print(f"  Inputs: {input_names}")
    print(f"  Outputs: {output_names}")
    
    # Create dummy input data
    print(f"\nPreparing input data ({num_points} points)...")
    point_coords = np.random.randn(num_points, 3).astype(np.float32)
    point_features = np.random.randn(num_points, 4).astype(np.float32)
    
    inputs = {
        'point_coords': point_coords,
        'point_features': point_features
    }
    
    # Warmup
    print("\nWarming up...")
    for _ in range(3):
        _ = session.run(output_names, inputs)
    
    # Benchmark
    print(f"\nRunning {num_runs} inference iterations...")
    times = []
    
    for i in range(num_runs):
        start = time.time()
        outputs = session.run(output_names, inputs)
        elapsed = time.time() - start
        times.append(elapsed)
        
        if i == 0:
            # Print first output shapes
            print(f"\nOutput shapes:")
            for name, output in zip(output_names, outputs):
                print(f"  {name}: {output.shape}")
    
    # Statistics
    mean_time = np.mean(times) * 1000  # Convert to ms
    std_time = np.std(times) * 1000
    min_time = np.min(times) * 1000
    max_time = np.max(times) * 1000
    
    print(f"\nPerformance Statistics:")
    print(f"  Mean: {mean_time:.2f} ms")
    print(f"  Std:  {std_time:.2f} ms")
    print(f"  Min:  {min_time:.2f} ms")
    print(f"  Max:  {max_time:.2f} ms")
    print(f"  FPS:  {1000/mean_time:.2f}")
    
    # Memory usage (if CUDA)
    if 'CUDA' in actual_provider:
        try:
            import torch
            if torch.cuda.is_available():
                print(f"\nGPU Memory:")
                print(f"  Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
                print(f"  Reserved:  {torch.cuda.memory_reserved()/1024**2:.2f} MB")
        except:
            pass
    
    return outputs

def compare_providers(onnx_path, num_points=10000):
    """
    Compare CPU vs GPU performance
    """
    print("\n" + "="*60)
    print("Comparing CPU vs GPU Performance")
    print("="*60)
    
    # Prepare data
    point_coords = np.random.randn(num_points, 3).astype(np.float32)
    point_features = np.random.randn(num_points, 4).astype(np.float32)
    inputs = {
        'point_coords': point_coords,
        'point_features': point_features
    }
    
    # Test CPU
    print("\n--- CPU Performance ---")
    session_cpu = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    times_cpu = []
    for _ in range(5):
        start = time.time()
        _ = session_cpu.run(None, inputs)
        times_cpu.append(time.time() - start)
    
    cpu_mean = np.mean(times_cpu) * 1000
    print(f"CPU Mean time: {cpu_mean:.2f} ms")
    
    # Test GPU (if available)
    try:
        print("\n--- GPU Performance ---")
        session_gpu = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
        
        times_gpu = []
        for _ in range(5):
            start = time.time()
            _ = session_gpu.run(None, inputs)
            times_gpu.append(time.time() - start)
        
        gpu_mean = np.mean(times_gpu) * 1000
        print(f"GPU Mean time: {gpu_mean:.2f} ms")
        
        speedup = cpu_mean / gpu_mean
        print(f"\n✓ GPU Speedup: {speedup:.2f}x faster than CPU")
        
    except Exception as e:
        print(f"⚠ GPU not available: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ONNX model on GPU")
    parser.add_argument('model', help='Path to ONNX model')
    parser.add_argument('--num_points', type=int, default=50000, help='Number of points')
    parser.add_argument('--num_runs', type=int, default=10, help='Number of inference runs')
    parser.add_argument('--compare', action='store_true', help='Compare CPU vs GPU')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_providers(args.model, args.num_points)
    else:
        run_inference_gpu(args.model, args.num_points, args.num_runs)