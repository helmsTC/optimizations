"""
ONNX model optimization utilities
"""

import onnx
import onnxruntime as ort
import numpy as np
from typing import List, Optional, Dict
from pathlib import Path


def optimize_onnx_model(input_path: str, output_path: str, 
                        optimization_level: str = 'all') -> bool:
    """
    Optimize ONNX model for inference
    
    Args:
        input_path: Path to input ONNX model
        output_path: Path to save optimized model
        optimization_level: 'basic', 'extended', or 'all'
    
    Returns:
        Success status
    """
    try:
        import onnxoptimizer
        
        # Load model
        model = onnx.load(input_path)
        
        # Define optimization passes based on level
        if optimization_level == 'basic':
            passes = [
                'eliminate_identity',
                'eliminate_nop_dropout',
                'eliminate_nop_pad',
                'eliminate_unused_initializer',
                'eliminate_deadend',
            ]
        elif optimization_level == 'extended':
            passes = [
                'eliminate_identity',
                'eliminate_nop_dropout',
                'eliminate_nop_pad',
                'eliminate_unused_initializer',
                'eliminate_deadend',
                'fuse_bn_into_conv',
                'fuse_consecutive_concats',
                'fuse_consecutive_reduce_unsqueeze',
                'fuse_consecutive_squeezes',
                'fuse_consecutive_transposes',
            ]
        else:  # 'all'
            passes = onnxoptimizer.get_available_passes()
            # Remove passes that might cause issues
            unsafe_passes = [
                'extract_constant_to_initializer',
                'fuse_qkv',
                'fuse_matmul_add_bias_into_gemm'
            ]
            passes = [p for p in passes if p not in unsafe_passes]
        
        print(f"Applying {len(passes)} optimization passes...")
        
        # Optimize model
        optimized_model = onnxoptimizer.optimize(model, passes)
        
        # Save optimized model
        onnx.save(optimized_model, output_path)
        
        # Compare file sizes
        original_size = Path(input_path).stat().st_size / 1024 / 1024
        optimized_size = Path(output_path).stat().st_size / 1024 / 1024
        reduction = (1 - optimized_size / original_size) * 100
        
        print(f"  Original size: {original_size:.2f} MB")
        print(f"  Optimized size: {optimized_size:.2f} MB")
        print(f"  Size reduction: {reduction:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        return False


def simplify_onnx_model(input_path: str, output_path: str) -> bool:
    """
    Simplify ONNX model using onnx-simplifier
    
    Args:
        input_path: Path to input ONNX model
        output_path: Path to save simplified model
    
    Returns:
        Success status
    """
    try:
        import onnxsim
        
        print("Simplifying ONNX model...")
        
        # Load model
        model = onnx.load(input_path)
        
        # Simplify
        model_simplified, check = onnxsim.simplify(
            model,
            dynamic_input_shape=True,
            input_shapes={'points': [1, 50000, 3], 'features': [1, 50000, 4]}
        )
        
        if check:
            # Save simplified model
            onnx.save(model_simplified, output_path)
            print("  ✓ Model simplified successfully")
            return True
        else:
            print("  ✗ Simplification check failed")
            return False
            
    except ImportError:
        print("  ⚠ onnx-simplifier not installed")
        print("  Install with: pip install onnx-simplifier")
        return False
    except Exception as e:
        print(f"  ✗ Simplification failed: {e}")
        return False


def quantize_onnx_model(input_path: str, output_path: str, 
                       quantization_mode: str = 'dynamic') -> bool:
    """
    Quantize ONNX model for faster inference
    
    Args:
        input_path: Path to input ONNX model
        output_path: Path to save quantized model
        quantization_mode: 'dynamic' or 'static'
    
    Returns:
        Success status
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType
        
        print(f"Quantizing model ({quantization_mode})...")
        
        if quantization_mode == 'dynamic':
            # Dynamic quantization (no calibration data needed)
            quantize_dynamic(
                input_path,
                output_path,
                weight_type=QuantType.QInt8,
                optimize_model=True
            )
        else:
            # Static quantization (requires calibration data)
            print("  Static quantization requires calibration data")
            return False
        
        # Compare file sizes
        original_size = Path(input_path).stat().st_size / 1024 / 1024
        quantized_size = Path(output_path).stat().st_size / 1024 / 1024
        reduction = (1 - quantized_size / original_size) * 100
        
        print(f"  Original size: {original_size:.2f} MB")
        print(f"  Quantized size: {quantized_size:.2f} MB")
        print(f"  Size reduction: {reduction:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Quantization failed: {e}")
        return False


def profile_onnx_model(model_path: str, num_runs: int = 100) -> Dict:
    """
    Profile ONNX model performance
    
    Args:
        model_path: Path to ONNX model
        num_runs: Number of inference runs
    
    Returns:
        Performance statistics dictionary
    """
    import time
    
    print(f"Profiling model: {model_path}")
    
    # Create session with profiling enabled
    options = ort.SessionOptions()
    options.enable_profiling = True
    
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(model_path, options, providers=providers)
    
    # Get model info
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    
    print(f"  Providers: {session.get_providers()}")
    print(f"  Inputs: {input_names}")
    print(f"  Outputs: {output_names}")
    
    # Create dummy input
    test_points = np.random.randn(1, 10000, 3).astype(np.float32)
    test_features = np.random.randn(1, 10000, 4).astype(np.float32)
    inputs = {
        input_names[0]: test_points,
        input_names[1]: test_features
    }
    
    # Warmup
    print(f"  Warming up...")
    for _ in range(10):
        _ = session.run(None, inputs)
    
    # Profile
    print(f"  Running {num_runs} iterations...")
    times = []
    memory_usage = []
    
    for i in range(num_runs):
        start = time.time()
        outputs = session.run(None, inputs)
        times.append(time.time() - start)
        
        # Estimate memory usage from output sizes
        mem = sum(out.nbytes for out in outputs) / 1024 / 1024  # MB
        memory_usage.append(mem)
    
    # Calculate statistics
    stats = {
        'mean_time_ms': np.mean(times) * 1000,
        'std_time_ms': np.std(times) * 1000,
        'min_time_ms': np.min(times) * 1000,
        'max_time_ms': np.max(times) * 1000,
        'p50_time_ms': np.percentile(times, 50) * 1000,
        'p95_time_ms': np.percentile(times, 95) * 1000,
        'p99_time_ms': np.percentile(times, 99) * 1000,
        'fps': 1.0 / np.mean(times),
        'mean_memory_mb': np.mean(memory_usage),
        'model_size_mb': Path(model_path).stat().st_size / 1024 / 1024
    }
    
    # Print results
    print("\nPerformance Statistics:")
    print(f"  Mean: {stats['mean_time_ms']:.2f} ms")
    print(f"  Std: {stats['std_time_ms']:.2f} ms")
    print(f"  Min: {stats['min_time_ms']:.2f} ms")
    print(f"  Max: {stats['max_time_ms']:.2f} ms")
    print(f"  P50: {stats['p50_time_ms']:.2f} ms")
    print(f"  P95: {stats['p95_time_ms']:.2f} ms")
    print(f"  P99: {stats['p99_time_ms']:.2f} ms")
    print(f"  FPS: {stats['fps']:.2f}")
    print(f"  Memory: {stats['mean_memory_mb']:.2f} MB")
    print(f"  Model size: {stats['model_size_mb']:.2f} MB")
    
    # End profiling
    prof_file = session.end_profiling()
    print(f"\nProfiling data saved to: {prof_file}")
    
    return stats


def optimize_for_mobile(input_path: str, output_path: str) -> bool:
    """
    Optimize ONNX model for mobile deployment
    
    Args:
        input_path: Path to input ONNX model
        output_path: Path to save mobile-optimized model
    
    Returns:
        Success status
    """
    print("Optimizing for mobile deployment...")
    
    try:
        # First optimize
        temp_path = output_path.replace('.onnx', '_temp.onnx')
        optimize_onnx_model(input_path, temp_path, 'extended')
        
        # Then quantize
        quantize_onnx_model(temp_path, output_path, 'dynamic')
        
        # Clean up temp file
        Path(temp_path).unlink()
        
        print("  ✓ Mobile optimization complete")
        return True
        
    except Exception as e:
        print(f"  ✗ Mobile optimization failed: {e}")
        return False


def convert_to_fp16(input_path: str, output_path: str) -> bool:
    """
    Convert ONNX model to FP16 precision
    
    Args:
        input_path: Path to input ONNX model
        output_path: Path to save FP16 model
    
    Returns:
        Success status
    """
    try:
        from onnxconverter_common import float16
        
        print("Converting to FP16 precision...")
        
        # Load model
        model = onnx.load(input_path)
        
        # Convert to FP16
        model_fp16 = float16.convert_float_to_float16(model)
        
        # Save FP16 model
        onnx.save(model_fp16, output_path)
        
        # Compare file sizes
        original_size = Path(input_path).stat().st_size / 1024 / 1024
        fp16_size = Path(output_path).stat().st_size / 1024 / 1024
        reduction = (1 - fp16_size / original_size) * 100
        
        print(f"  Original size: {original_size:.2f} MB")
        print(f"  FP16 size: {fp16_size:.2f} MB")
        print(f"  Size reduction: {reduction:.1f}%")
        
        return True
        
    except ImportError:
        print("  ⚠ onnxconverter-common not installed")
        print("  Install with: pip install onnxconverter-common")
        return False
    except Exception as e:
        print(f"  ✗ FP16 conversion failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ONNX Model Optimization")
    parser.add_argument('input', help='Input ONNX model path')
    parser.add_argument('--output', help='Output path (default: input_optimized.onnx)')
    parser.add_argument('--optimize', action='store_true', help='Apply optimizations')
    parser.add_argument('--simplify', action='store_true', help='Simplify model')
    parser.add_argument('--quantize', action='store_true', help='Quantize model')
    parser.add_argument('--fp16', action='store_true', help='Convert to FP16')
    parser.add_argument('--mobile', action='store_true', help='Optimize for mobile')
    parser.add_argument('--profile', action='store_true', help='Profile model')
    
    args = parser.parse_args()
    
    if not args.output:
        base = Path(args.input).stem
        args.output = f"{base}_optimized.onnx"
    
    # Apply requested optimizations
    if args.optimize:
        optimize_onnx_model(args.input, args.output)
        args.input = args.output
    
    if args.simplify:
        output = args.output.replace('.onnx', '_simplified.onnx')
        simplify_onnx_model(args.input, output)
        args.input = output
    
    if args.quantize:
        output = args.output.replace('.onnx', '_quantized.onnx')
        quantize_onnx_model(args.input, output)
        args.input = output
    
    if args.fp16:
        output = args.output.replace('.onnx', '_fp16.onnx')
        convert_to_fp16(args.input, output)
        args.input = output
    
    if args.mobile:
        output = args.output.replace('.onnx', '_mobile.onnx')
        optimize_for_mobile(args.input, output)
        args.input = output
    
    if args.profile:
        profile_onnx_model(args.input)