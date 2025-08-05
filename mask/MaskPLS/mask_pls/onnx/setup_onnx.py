#!/usr/bin/env python3
"""
Setup script for ONNX migration
Run this from the mask/MaskPLS directory
"""

import os
import sys
from pathlib import Path
import subprocess


def create_directories():
    """Create necessary directories for ONNX modules"""
    directories = [
        "mask_pls/models/onnx",
        "mask_pls/utils/onnx",
        "mask_pls/deployment/docker",
        "mask_pls/deployment/tensorrt",
        "mask_pls/deployment/ros",
        "onnx_models",
        "tests"
    ]
    
    print("Creating directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}")
    
    return True


def create_init_files():
    """Create __init__.py files"""
    
    init_files = {
        "mask_pls/models/onnx/__init__.py": '''"""
ONNX-compatible models for MaskPLS
"""

try:
    from .dense_backbone import DenseConv3DBackbone, ResidualBlock3D, KNNInterpolationONNX
    from .onnx_decoder import ONNXCompatibleDecoder, TransformerDecoderLayerONNX, PositionalEncodingONNX
    from .onnx_model import MaskPLSONNX, MaskPLSExportWrapper, create_onnx_model, load_checkpoint_weights
    
    __all__ = [
        'DenseConv3DBackbone',
        'ResidualBlock3D', 
        'KNNInterpolationONNX',
        'ONNXCompatibleDecoder',
        'TransformerDecoderLayerONNX',
        'PositionalEncodingONNX',
        'MaskPLSONNX',
        'MaskPLSExportWrapper',
        'create_onnx_model',
        'load_checkpoint_weights'
    ]
except ImportError as e:
    print(f"Warning: Some ONNX modules not yet available: {e}")
    __all__ = []
''',
        
        "mask_pls/utils/onnx/__init__.py": '''"""
ONNX utilities for model conversion and optimization
"""

try:
    from .optimization import (
        optimize_onnx_model,
        simplify_onnx_model,
        quantize_onnx_model,
        profile_onnx_model,
        optimize_for_mobile,
        convert_to_fp16
    )
    
    from .validation import (
        validate_onnx_model,
        validate_conversion,
        compare_outputs,
        test_dynamic_shapes,
        benchmark_accuracy
    )
    
    __all__ = [
        'optimize_onnx_model',
        'simplify_onnx_model',
        'quantize_onnx_model',
        'profile_onnx_model',
        'optimize_for_mobile',
        'convert_to_fp16',
        'validate_onnx_model',
        'validate_conversion',
        'compare_outputs',
        'test_dynamic_shapes',
        'benchmark_accuracy'
    ]
except ImportError as e:
    print(f"Warning: Some utilities not yet available: {e}")
    __all__ = []
''',
        
        "mask_pls/deployment/__init__.py": '''"""
Deployment utilities for MaskPLS
"""
''',
        
        "tests/__init__.py": '''"""
Test modules for MaskPLS ONNX conversion
"""
'''
    }
    
    print("\nCreating __init__.py files...")
    for filepath, content in init_files.items():
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  ✓ {filepath}")
    
    return True


def check_dependencies():
    """Check and install required dependencies"""
    
    required_packages = {
        'onnx': 'onnx>=1.13.0',
        'onnxruntime': 'onnxruntime-gpu>=1.14.0',
        'easydict': 'easydict>=1.9',
        'numpy': 'numpy>=1.20.0',
        'torch': 'torch>=1.10.0',
        'yaml': 'PyYAML>=6.0'
    }
    
    optional_packages = {
        'onnxoptimizer': 'onnxoptimizer>=0.3.0',
        'onnxsim': 'onnx-simplifier>=0.4.0'
    }
    
    print("\nChecking required dependencies...")
    missing_required = []
    missing_optional = []
    
    for module_name, package_name in required_packages.items():
        try:
            if module_name == 'yaml':
                __import__('yaml')
            else:
                __import__(module_name)
            print(f"  ✓ {module_name} installed")
        except ImportError:
            print(f"  ✗ {module_name} not installed")
            missing_required.append(package_name)
    
    print("\nChecking optional dependencies...")
    for module_name, package_name in optional_packages.items():
        try:
            if module_name == 'onnxsim':
                __import__('onnxsim')
            else:
                __import__(module_name)
            print(f"  ✓ {module_name} installed")
        except ImportError:
            print(f"  ⚠ {module_name} not installed (optional)")
            missing_optional.append(package_name)
    
    if missing_required:
        print("\n" + "="*50)
        print("Installing missing required packages...")
        print("="*50)
        
        install_cmd = [sys.executable, "-m", "pip", "install"] + missing_required
        
        try:
            subprocess.run(install_cmd, check=True)
            print("✓ Required packages installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install packages: {e}")
            print("\nPlease install manually:")
            print(f"  pip install {' '.join(missing_required)}")
    
    if missing_optional:
        print("\nOptional packages not installed:")
        for pkg in missing_optional:
            print(f"  pip install {pkg}")
        print("\nThese are optional but recommended for optimization.")
    
    return len(missing_required) == 0


def create_onnx_config():
    """Create ONNX configuration file"""
    
    config_content = '''# ONNX-specific configuration
ONNX:
  EXPORT:
    OPSET_VERSION: 13
    DYNAMIC_AXES: true
    SIMPLIFY: true
  
  VOXEL:
    SIZE: 0.05
    SPATIAL_SHAPE: [96, 96, 8]
    MAX_POINTS: 100000
  
  OPTIMIZATION:
    USE_FP16: false
    USE_INT8: false
    TENSORRT: false
  
  INFERENCE:
    BATCH_SIZE: 1
    NUM_THREADS: 4
    PROVIDERS: ['CUDAExecutionProvider', 'CPUExecutionProvider']
'''
    
    config_path = Path("mask_pls/config/onnx_config.yaml")
    
    if not config_path.exists():
        print("\nCreating ONNX configuration file...")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            f.write(config_content)
        print(f"  ✓ {config_path}")
    else:
        print(f"\n  ℹ ONNX config already exists: {config_path}")
    
    return True


def main():
    """Main setup function"""
    
    print("="*50)
    print("MaskPLS ONNX Setup Script")
    print("="*50)
    
    # Check if we're in the right directory
    if not Path("setup.py").exists() or not Path("mask_pls").exists():
        print("\n✗ Error: Please run this script from the mask/MaskPLS directory")
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        print("\n✗ Failed to create directories")
        sys.exit(1)
    
    # Create __init__.py files
    if not create_init_files():
        print("\n✗ Failed to create __init__.py files")
        sys.exit(1)
    
    # Create ONNX config
    create_onnx_config()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    print("\n" + "="*50)
    print("Setup Complete!")
    print("="*50)
    
    print("\nNext steps:")
    print("\n1. The ONNX model files have been provided. Save them to:")
    print("   - mask_pls/models/onnx/dense_backbone.py")
    print("   - mask_pls/models/onnx/onnx_decoder.py")
    print("   - mask_pls/models/onnx/onnx_model.py")
    
    print("\n2. The utility files have been provided. Save them to:")
    print("   - mask_pls/utils/onnx/optimization.py")
    print("   - mask_pls/utils/onnx/validation.py")
    
    print("\n3. The conversion scripts have been provided. Save them to:")
    print("   - mask_pls/scripts/convert_to_onnx.py")
    print("   - mask_pls/scripts/convert_to_onnx_simple.py")
    
    print("\n4. Run the conversion:")
    print("   python mask_pls/scripts/convert_to_onnx_simple.py")
    print("\n   Or with full features:")
    print("   python mask_pls/scripts/convert_to_onnx.py \\")
    print("       --checkpoint your_model.ckpt \\")
    print("       --output onnx_models/mask_pls.onnx \\")
    print("       --optimize --benchmark")
    
    if not deps_ok:
        print("\n⚠ Warning: Some required dependencies are missing.")
        print("  Please install them before running the conversion.")
    
    print("\n✓ Setup script completed successfully!")


if __name__ == "__main__":
    main()
