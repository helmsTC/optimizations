#!/usr/bin/env python3
"""
Simple ONNX model inspector
"""
import sys
import glob
import os

def inspect_onnx_model(model_path):
    """Inspect ONNX model using torch.onnx tools"""
    try:
        import torch
        print(f"Inspecting: {model_path}")
        
        # Try to load with torch first
        try:
            # Read the file as binary and check if it starts with ONNX magic
            with open(model_path, 'rb') as f:
                header = f.read(8)
                print(f"File header: {header}")
            
            # Try importing onnx directly
            try:
                import onnx
                model = onnx.load(model_path)
                
                print('\nInputs:')
                for i, inp in enumerate(model.graph.input):
                    dims = [d.dim_value if d.dim_value > 0 else 'dynamic' for d in inp.type.tensor_type.shape.dim]
                    print(f'  {i}: "{inp.name}" - {dims}')
                
                print('\nOutputs:')
                for i, out in enumerate(model.graph.output):
                    dims = [d.dim_value if d.dim_value > 0 else 'dynamic' for d in out.type.tensor_type.shape.dim]
                    print(f'  {i}: "{out.name}" - {dims}')
                    
                return True
                
            except ImportError:
                print("onnx not available, trying onnxruntime...")
                
                try:
                    import onnxruntime as ort
                    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                    
                    print('\nInputs:')
                    for i, inp in enumerate(session.get_inputs()):
                        print(f'  {i}: "{inp.name}" - {inp.shape} ({inp.type})')
                    
                    print('\nOutputs:')
                    for i, out in enumerate(session.get_outputs()):
                        print(f'  {i}: "{out.name}" - {out.shape} ({out.type})')
                        
                    return True
                    
                except ImportError:
                    print("Neither onnx nor onnxruntime available")
                    return False
                    
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
            
    except ImportError:
        print("PyTorch not available")
        return False

def main():
    # Find ONNX files
    onnx_files = []
    for pattern in ['*.onnx', '**/*.onnx']:
        onnx_files.extend(glob.glob(pattern, recursive=True))
    
    if not onnx_files:
        print('No ONNX files found in current directory')
        return
    
    print('Found ONNX files:')
    for i, f in enumerate(onnx_files):
        size = os.path.getsize(f) / (1024*1024)
        mtime = os.path.getmtime(f)
        print(f'  {i}: {f} ({size:.1f} MB, modified: {mtime})')
    
    # Check the most recent one
    latest_file = max(onnx_files, key=os.path.getmtime)
    print(f'\nInspecting most recent: {latest_file}')
    print('='*60)
    
    success = inspect_onnx_model(latest_file)
    
    if not success:
        print("\nFailed to inspect model. Trying simple file info...")
        try:
            size = os.path.getsize(latest_file)
            print(f"File size: {size} bytes ({size/(1024*1024):.2f} MB)")
            
            with open(latest_file, 'rb') as f:
                header = f.read(32)
                print(f"Header bytes: {header.hex()}")
                
        except Exception as e:
            print(f"Even file reading failed: {e}")

if __name__ == '__main__':
    main()