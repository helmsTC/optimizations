#!/usr/bin/env python3
"""
Convert MaskPLS PyTorch checkpoint to ONNX format
Usage: python convert_to_onnx.py --checkpoint model.ckpt --output model.onnx
"""

import os
import sys
import click
import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
import yaml
from pathlib import Path
from easydict import EasyDict as edict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.onnx.onnx_model import MaskPLSONNX, MaskPLSExportWrapper, load_checkpoint_weights
from utils.onnx.optimization import optimize_onnx_model
from utils.onnx.validation import validate_onnx_model


def load_configs(config_dir=None):
    """Load all configuration files"""
    if config_dir is None:
        config_dir = Path(__file__).parent.parent / 'config'
    
    configs = {}
    config_files = ['model.yaml', 'backbone.yaml', 'decoder.yaml']
    
    for config_file in config_files:
        config_path = config_dir / config_file
        if config_path.exists():
            with open(config_path, 'r') as f:
                configs.update(yaml.safe_load(f))
    
    # Check for ONNX-specific config
    onnx_config_path = config_dir / 'onnx_config.yaml'
    if onnx_config_path.exists():
        with open(onnx_config_path, 'r') as f:
            configs['ONNX'] = yaml.safe_load(f)['ONNX']
    
    return edict(configs)


def export_to_onnx(model, output_path, batch_size=1, num_points=50000, opset_version=13):
    """
    Export PyTorch model to ONNX format
    """
    print(f"Exporting model to ONNX format...")
    
    # Set model to evaluation mode
    model.eval()
    
    # Create wrapper for clean export
    export_model = MaskPLSExportWrapper(model)
    
    # Create dummy input
    dummy_points = torch.randn(batch_size, num_points, 3)
    dummy_features = torch.randn(batch_size, num_points, 4)  # xyz + intensity
    
    # Export to ONNX
    with torch.no_grad():
        torch.onnx.export(
            export_model,
            (dummy_points, dummy_features),
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['points', 'features'],
            output_names=['pred_logits', 'pred_masks', 'sem_logits'],
            dynamic_axes={
                'points': {0: 'batch_size', 1: 'num_points'},
                'features': {0: 'batch_size', 1: 'num_points'},
                'pred_logits': {0: 'batch_size'},
                'pred_masks': {0: 'batch_size', 1: 'num_points'},
                'sem_logits': {0: 'batch_size', 1: 'num_points'}
            },
            verbose=False
        )
    
    print(f"✓ Model exported to {output_path}")
    
    # Verify the exported model
    try:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model validation passed")
    except Exception as e:
        print(f"✗ ONNX model validation failed: {e}")
        return False
    
    return True


def test_onnx_inference(onnx_path, batch_size=1, num_points=10000):
    """
    Test ONNX model inference
    """
    print("\nTesting ONNX inference...")
    
    # Create ONNX Runtime session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    # Get input/output info
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    
    print(f"Input names: {input_names}")
    print(f"Output names: {output_names}")
    
    # Create test input
    test_points = np.random.randn(batch_size, num_points, 3).astype(np.float32)
    test_features = np.random.randn(batch_size, num_points, 4).astype(np.float32)
    
    # Run inference
    try:
        outputs = session.run(
            output_names,
            {
                input_names[0]: test_points,
                input_names[1]: test_features
            }
        )
        
        print("\nOutput shapes:")
        for name, output in zip(output_names, outputs):
            print(f"  {name}: {output.shape}")
        
        print("✓ ONNX inference test passed")
        return True
        
    except Exception as e:
        print(f"✗ ONNX inference test failed: {e}")
        return False


def benchmark_performance(onnx_path, num_runs=100):
    """
    Benchmark ONNX model performance
    """
    import time
    
    print("\nBenchmarking performance...")
    
    # Create session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    # Prepare inputs
    input_names = [inp.name for inp in session.get_inputs()]
    test_points = np.random.randn(1, 10000, 3).astype(np.float32)
    test_features = np.random.randn(1, 10000, 4).astype(np.float32)
    inputs = {
        input_names[0]: test_points,
        input_names[1]: test_features
    }
    
    # Warmup
    print("Warming up...")
    for _ in range(10):
        _ = session.run(None, inputs)
    
    # Benchmark
    print(f"Running {num_runs} iterations...")
    times = []
    for _ in range(num_runs):
        start = time.time()
        _ = session.run(None, inputs)
        times.append(time.time() - start)
    
    # Calculate statistics
    avg_time = np.mean(times) * 1000  # Convert to ms
    std_time = np.std(times) * 1000
    min_time = np.min(times) * 1000
    max_time = np.max(times) * 1000
    fps = 1000 / avg_time
    
    print("\nPerformance Results:")
    print(f"  Average: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"  Min: {min_time:.2f} ms")
    print(f"  Max: {max_time:.2f} ms")
    print(f"  FPS: {fps:.2f}")
    
    return avg_time


@click.command()
@click.option('--checkpoint', '-c', required=True, type=click.Path(exists=True),
              help='Path to PyTorch checkpoint (.ckpt or .pth)')
@click.option('--output', '-o', type=click.Path(),
              help='Output ONNX file path')
@click.option('--config-dir', type=click.Path(exists=True),
              help='Directory containing config files')
@click.option('--dataset', type=click.Choice(['KITTI', 'NUSCENES']),
              default='KITTI', help='Dataset type')
@click.option('--batch-size', default=1, type=int,
              help='Batch size for export')
@click.option('--num-points', default=50000, type=int,
              help='Number of points for export')
@click.option('--opset-version', default=13, type=int,
              help='ONNX opset version')
@click.option('--optimize', is_flag=True,
              help='Optimize ONNX model')
@click.option('--validate', is_flag=True,
              help='Validate converted model')
@click.option('--benchmark', is_flag=True,
              help='Benchmark model performance')
@click.option('--tensorrt', is_flag=True,
              help='Convert to TensorRT')
def main(checkpoint, output, config_dir, dataset, batch_size, num_points,
         opset_version, optimize, validate, benchmark, tensorrt):
    """
    Convert MaskPLS checkpoint to ONNX format
    """
    print("=" * 50)
    print("MaskPLS to ONNX Converter")
    print("=" * 50)
    
    # Determine output path
    if output is None:
        checkpoint_name = Path(checkpoint).stem
        output = f"{checkpoint_name}.onnx"
    
    # Load configurations
    print("\n1. Loading configurations...")
    cfg = load_configs(config_dir)
    cfg.MODEL.DATASET = dataset
    
    # Add ONNX-specific settings if not present
    if 'ONNX' not in cfg:
        cfg.ONNX = edict({
            'EXPORT': {
                'OPSET_VERSION': opset_version,
                'DYNAMIC_AXES': True
            },
            'VOXEL': {
                'SIZE': 0.05,
                'SPATIAL_SHAPE': [96, 96, 8]
            }
        })
    
    print(f"  Dataset: {dataset}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of points: {num_points}")
    
    # Create ONNX model
    print("\n2. Creating ONNX-compatible model...")
    model = MaskPLSONNX(cfg)
    
    # Load checkpoint weights
    print(f"\n3. Loading weights from {checkpoint}...")
    try:
        model = load_checkpoint_weights(model, checkpoint, strict=False)
        print("  ✓ Weights loaded successfully")
    except Exception as e:
        print(f"  ⚠ Warning: Could not load all weights: {e}")
        print("  Continuing with partially loaded or random weights...")
    
    # Export to ONNX
    print(f"\n4. Exporting to ONNX...")
    success = export_to_onnx(
        model, output, batch_size, num_points, opset_version
    )
    
    if not success:
        print("\n✗ Export failed!")
        sys.exit(1)
    
    # Optimize if requested
    if optimize:
        print(f"\n5. Optimizing ONNX model...")
        try:
            import onnxoptimizer
            
            optimized_path = output.replace('.onnx', '_optimized.onnx')
            optimize_onnx_model(output, optimized_path)
            print(f"  ✓ Optimized model saved to {optimized_path}")
            output = optimized_path
            
        except ImportError:
            print("  ⚠ onnxoptimizer not installed. Skipping optimization.")
            print("  Install with: pip install onnxoptimizer")
    
    # Test inference
    print(f"\n6. Testing ONNX inference...")
    test_success = test_onnx_inference(output, batch_size, num_points)
    
    if not test_success:
        print("\n⚠ Warning: Inference test failed!")
    
    # Validate if requested
    if validate and checkpoint:
        print(f"\n7. Validating against original model...")
        try:
            from utils.onnx.validation import validate_conversion
            
            is_valid = validate_conversion(
                checkpoint, output, cfg, num_samples=10
            )
            
            if is_valid:
                print("  ✓ Validation passed")
            else:
                print("  ⚠ Validation failed - outputs differ")
                
        except Exception as e:
            print(f"  ⚠ Could not validate: {e}")
    
    # Benchmark if requested
    if benchmark:
        print(f"\n8. Benchmarking performance...")
        benchmark_performance(output)
    
    # Convert to TensorRT if requested
    if tensorrt:
        print(f"\n9. Converting to TensorRT...")
        try:
            from deployment.tensorrt.convert_trt import convert_to_tensorrt
            
            trt_path = output.replace('.onnx', '.trt')
            trt_success = convert_to_tensorrt(output, trt_path)
            
            if trt_success:
                print(f"  ✓ TensorRT engine saved to {trt_path}")
            else:
                print("  ✗ TensorRT conversion failed")
                
        except ImportError:
            print("  ⚠ TensorRT not installed. Skipping TensorRT conversion.")
            print("  Install with: pip install tensorrt")
    
    # Summary
    print("\n" + "=" * 50)
    print("Conversion Complete!")
    print("=" * 50)
    print(f"Output file: {output}")
    print(f"File size: {os.path.getsize(output) / 1024 / 1024:.2f} MB")
    
    print("\nNext steps:")
    print("1. Test with your data:")
    print(f"   python -m mask_pls.scripts.evaluate_onnx --model {output}")
    print("2. Deploy for inference:")
    print(f"   python -m mask_pls.deployment.serve --model {output}")
    print("3. Integrate with ROS:")
    print(f"   rosrun mask_pls maskpls_onnx_node.py --model {output}")


if __name__ == '__main__':
    main()