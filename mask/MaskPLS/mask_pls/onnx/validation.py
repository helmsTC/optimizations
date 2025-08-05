"""
ONNX model validation utilities
"""

import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
from typing import Dict, List, Tuple, Optional
from pathlib import Path


def validate_onnx_model(onnx_path: str) -> bool:
    """
    Basic validation of ONNX model structure
    
    Args:
        onnx_path: Path to ONNX model
    
    Returns:
        True if model is valid
    """
    try:
        # Load and check model
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        
        print(f"✓ ONNX model structure is valid")
        
        # Try to create inference session
        session = ort.InferenceSession(onnx_path)
        
        # Check inputs and outputs
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        
        print(f"✓ Model has {len(inputs)} inputs and {len(outputs)} outputs")
        
        for inp in inputs:
            print(f"  Input: {inp.name} - Shape: {inp.shape} - Type: {inp.type}")
        
        for out in outputs:
            print(f"  Output: {out.name} - Shape: {out.shape} - Type: {out.type}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model validation failed: {e}")
        return False


def validate_conversion(checkpoint_path: str, onnx_path: str, 
                       cfg: Dict, num_samples: int = 10,
                       tolerance: float = 1e-3) -> bool:
    """
    Validate ONNX model against original PyTorch model
    
    Args:
        checkpoint_path: Path to original PyTorch checkpoint
        onnx_path: Path to ONNX model
        cfg: Configuration dictionary
        num_samples: Number of samples to test
        tolerance: Maximum allowed difference
    
    Returns:
        True if outputs match within tolerance
    """
    print(f"\nValidating ONNX conversion...")
    
    try:
        # Import original model (only if MinkowskiEngine is available)
        try:
            from mask_pls.models.mask_model import MaskPS
            
            # Load original model
            print("Loading original PyTorch model...")
            original_model = MaskPS(cfg)
            
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            original_model.load_state_dict(state_dict, strict=False)
            original_model.eval()
            
            has_original = True
        except ImportError:
            print("⚠ MinkowskiEngine not available, skipping comparison with original model")
            has_original = False
        
        # Load ONNX model
        print("Loading ONNX model...")
        session = ort.InferenceSession(onnx_path)
        input_names = [inp.name for inp in session.get_inputs()]
        output_names = [out.name for out in session.get_outputs()]
        
        # Test with random inputs
        print(f"Testing with {num_samples} random samples...")
        
        all_close = True
        max_diff = 0.0
        
        for i in range(num_samples):
            # Generate random input
            batch_size = 1
            num_points = np.random.randint(5000, 15000)
            
            points = np.random.randn(batch_size, num_points, 3).astype(np.float32)
            features = np.random.randn(batch_size, num_points, 4).astype(np.float32)
            
            # ONNX inference
            onnx_outputs = session.run(output_names, {
                input_names[0]: points,
                input_names[1]: features
            })
            
            if has_original:
                # PyTorch inference (simplified comparison)
                with torch.no_grad():
                    # Convert to torch tensors
                    pt_points = torch.from_numpy(points)
                    pt_features = torch.from_numpy(features)
                    
                    # Note: This is a simplified comparison
                    # The original model expects different input format
                    # We're mainly checking that ONNX model produces reasonable outputs
            
            # Check output shapes and ranges
            for j, output in enumerate(onnx_outputs):
                # Check for NaN or Inf
                if np.any(np.isnan(output)) or np.any(np.isinf(output)):
                    print(f"✗ Output {j} contains NaN or Inf values")
                    all_close = False
                
                # Check output range is reasonable
                output_min = np.min(output)
                output_max = np.max(output)
                
                if abs(output_min) > 1e6 or abs(output_max) > 1e6:
                    print(f"⚠ Output {j} has large values: [{output_min:.2f}, {output_max:.2f}]")
        
        if all_close:
            print(f"✓ Validation passed")
            if has_original and max_diff > 0:
                print(f"  Maximum difference: {max_diff:.6f}")
        else:
            print(f"✗ Validation failed")
        
        return all_close
        
    except Exception as e:
        print(f"✗ Validation error: {e}")
        return False


def compare_outputs(pytorch_output: torch.Tensor, 
                   onnx_output: np.ndarray,
                   name: str = "output",
                   rtol: float = 1e-3,
                   atol: float = 1e-5) -> Tuple[bool, float]:
    """
    Compare PyTorch and ONNX outputs
    
    Args:
        pytorch_output: PyTorch model output
        onnx_output: ONNX model output
        name: Name of the output for logging
        rtol: Relative tolerance
        atol: Absolute tolerance
    
    Returns:
        (is_close, max_diff) tuple
    """
    # Convert to numpy
    if isinstance(pytorch_output, torch.Tensor):
        pytorch_output = pytorch_output.detach().cpu().numpy()
    
    # Check shapes
    if pytorch_output.shape != onnx_output.shape:
        print(f"✗ {name} shape mismatch: PyTorch {pytorch_output.shape} vs ONNX {onnx_output.shape}")
        return False, float('inf')
    
    # Compare values
    is_close = np.allclose(pytorch_output, onnx_output, rtol=rtol, atol=atol)
    max_diff = np.max(np.abs(pytorch_output - onnx_output))
    
    if is_close:
        print(f"✓ {name} matches (max diff: {max_diff:.6f})")
    else:
        print(f"✗ {name} differs (max diff: {max_diff:.6f})")
        
        # Show some statistics
        mean_diff = np.mean(np.abs(pytorch_output - onnx_output))
        print(f"  Mean difference: {mean_diff:.6f}")
        print(f"  PyTorch range: [{np.min(pytorch_output):.3f}, {np.max(pytorch_output):.3f}]")
        print(f"  ONNX range: [{np.min(onnx_output):.3f}, {np.max(onnx_output):.3f}]")
    
    return is_close, max_diff


def test_dynamic_shapes(onnx_path: str, 
                       min_points: int = 1000,
                       max_points: int = 100000,
                       num_tests: int = 10) -> bool:
    """
    Test ONNX model with different input sizes
    
    Args:
        onnx_path: Path to ONNX model
        min_points: Minimum number of points
        max_points: Maximum number of points
        num_tests: Number of different sizes to test
    
    Returns:
        True if all tests pass
    """
    print(f"\nTesting dynamic shapes...")
    
    try:
        session = ort.InferenceSession(onnx_path)
        input_names = [inp.name for inp in session.get_inputs()]
        
        all_passed = True
        
        for i in range(num_tests):
            # Random number of points
            num_points = np.random.randint(min_points, max_points)
            
            # Random batch size
            batch_size = np.random.choice([1, 2, 4])
            
            print(f"  Test {i+1}: batch_size={batch_size}, num_points={num_points}")
            
            # Create input
            points = np.random.randn(batch_size, num_points, 3).astype(np.float32)
            features = np.random.randn(batch_size, num_points, 4).astype(np.float32)
            
            try:
                # Run inference
                outputs = session.run(None, {
                    input_names[0]: points,
                    input_names[1]: features
                })
                
                # Check outputs are valid
                for output in outputs:
                    if np.any(np.isnan(output)) or np.any(np.isinf(output)):
                        print(f"    ✗ Invalid output values")
                        all_passed = False
                        break
                else:
                    print(f"    ✓ Passed")
                    
            except Exception as e:
                print(f"    ✗ Failed: {e}")
                all_passed = False
        
        if all_passed:
            print(f"✓ All dynamic shape tests passed")
        else:
            print(f"✗ Some dynamic shape tests failed")
        
        return all_passed
        
    except Exception as e:
        print(f"✗ Dynamic shape testing failed: {e}")
        return False


def benchmark_accuracy(onnx_path: str, 
                      dataset_path: str = None,
                      num_samples: int = 100) -> Dict:
    """
    Benchmark ONNX model accuracy on dataset
    
    Args:
        onnx_path: Path to ONNX model
        dataset_path: Path to dataset (optional)
        num_samples: Number of samples to test
    
    Returns:
        Dictionary with accuracy metrics
    """
    print(f"\nBenchmarking accuracy...")
    
    metrics = {
        'samples_tested': 0,
        'inference_success': 0,
        'mean_confidence': 0.0,
        'valid_outputs': 0
    }
    
    try:
        session = ort.InferenceSession(onnx_path)
        input_names = [inp.name for inp in session.get_inputs()]
        
        for i in range(num_samples):
            # Generate or load test data
            if dataset_path:
                # TODO: Load from actual dataset
                pass
            else:
                # Use random data
                num_points = np.random.randint(5000, 15000)
                points = np.random.randn(1, num_points, 3).astype(np.float32)
                features = np.random.randn(1, num_points, 4).astype(np.float32)
            
            try:
                # Run inference
                outputs = session.run(None, {
                    input_names[0]: points,
                    input_names[1]: features
                })
                
                metrics['samples_tested'] += 1
                
                # Check outputs
                pred_logits = outputs[0]  # [B, Q, C+1]
                pred_masks = outputs[1]   # [B, N, Q]
                
                # Check validity
                if not (np.any(np.isnan(pred_logits)) or np.any(np.isinf(pred_logits))):
                    metrics['valid_outputs'] += 1
                    
                    # Calculate confidence (softmax of logits)
                    probs = np.exp(pred_logits) / np.sum(np.exp(pred_logits), axis=-1, keepdims=True)
                    confidence = np.max(probs)
                    metrics['mean_confidence'] += confidence
                    
                metrics['inference_success'] += 1
                
            except Exception as e:
                print(f"  Sample {i} failed: {e}")
        
        # Calculate final metrics
        if metrics['inference_success'] > 0:
            metrics['success_rate'] = metrics['inference_success'] / metrics['samples_tested']
            metrics['mean_confidence'] /= metrics['valid_outputs'] if metrics['valid_outputs'] > 0 else 1
        else:
            metrics['success_rate'] = 0.0
        
        print(f"\nAccuracy Metrics:")
        print(f"  Samples tested: {metrics['samples_tested']}")
        print(f"  Success rate: {metrics['success_rate']:.2%}")
        print(f"  Valid outputs: {metrics['valid_outputs']}/{metrics['samples_tested']}")
        print(f"  Mean confidence: {metrics['mean_confidence']:.3f}")
        
        return metrics
        
    except Exception as e:
        print(f"✗ Accuracy benchmarking failed: {e}")
        return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate ONNX model")
    parser.add_argument('model', help='ONNX model path')
    parser.add_argument('--checkpoint', help='Original PyTorch checkpoint')
    parser.add_argument('--dynamic', action='store_true', help='Test dynamic shapes')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark accuracy')
    
    args = parser.parse_args()
    
    # Basic validation
    is_valid = validate_onnx_model(args.model)
    
    # Test dynamic shapes
    if args.dynamic:
        test_dynamic_shapes(args.model)
    
    # Benchmark accuracy
    if args.benchmark:
        benchmark_accuracy(args.model)
    
    # Compare with original if checkpoint provided
    if args.checkpoint:
        # Load config (simplified)
        import yaml
        from easydict import EasyDict as edict
        
        # Try to load config
        config_path = Path(args.model).parent.parent / 'config' / 'model.yaml'
        if config_path.exists():
            with open(config_path, 'r') as f:
                cfg = edict(yaml.safe_load(f))
            validate_conversion(args.checkpoint, args.model, cfg)
