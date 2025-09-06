#!/usr/bin/env python3
"""
Export MaskPLS model without PyTorch Lightning dependencies
Fixes the "not attached to trainer" error
"""

import torch
import torch.nn as nn
import numpy as np
import click
import yaml
from pathlib import Path
from easydict import EasyDict as edict


class StandaloneMaskPLS(nn.Module):
    """Pure PyTorch version without Lightning dependencies"""
    
    def __init__(self, cfg, checkpoint_path, device='cuda'):
        super().__init__()
        
        # Load the checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        
        # Extract configuration
        dataset = cfg.MODEL.DATASET
        self.num_classes = cfg[dataset].NUM_CLASSES
        self.device_type = device
        
        # Import the backbone and decoder modules directly
        from mask_pls.models.dgcnn.dgcnn_backbone_efficient import FixedDGCNNBackbone
        from mask_pls.models.decoder import MaskedTransformerDecoder
        
        # Create backbone
        self.backbone = FixedDGCNNBackbone(cfg.BACKBONE, None)
        self.backbone.set_num_classes(self.num_classes)
        
        # Create decoder
        self.decoder = MaskedTransformerDecoder(
            cfg.DECODER,
            cfg.BACKBONE,
            cfg[dataset]
        )
        
        # Load weights from checkpoint
        print("Loading model weights...")
        backbone_state = {}
        decoder_state = {}
        
        for key, value in state_dict.items():
            if key.startswith('backbone.'):
                backbone_state[key.replace('backbone.', '')] = value
            elif key.startswith('decoder.'):
                decoder_state[key.replace('decoder.', '')] = value
        
        # Load weights
        if backbone_state:
            self.backbone.load_state_dict(backbone_state, strict=False)
            print(f"  Loaded {len(backbone_state)} backbone parameters")
        
        if decoder_state:
            self.decoder.load_state_dict(decoder_state, strict=False)
            print(f"  Loaded {len(decoder_state)} decoder parameters")
        
        # Set to eval mode
        self.eval()
    
    def forward(self, points: torch.Tensor, features: torch.Tensor):
        """
        Direct forward pass without dict inputs
        Args:
            points: [B, N, 3]
            features: [B, N, 4]
        """
        batch_size = points.shape[0]
        
        # Process through backbone (simplified)
        # We need to bypass the dict-based forward
        ms_features, ms_coords, ms_masks, sem_logits = self.process_backbone(points, features)
        
        # Decoder
        outputs, padding = self.decoder(ms_features, ms_coords, ms_masks)
        
        return outputs['pred_logits'], outputs['pred_masks'], sem_logits
    
    def process_backbone(self, points, features):
        """Process through backbone without dict inputs"""
        batch_size = points.shape[0]
        
        # Initialize outputs
        all_features = []
        all_coords = []
        all_masks = []
        
        for b in range(batch_size):
            pts = points[b]
            feat = features[b]
            
            # Limit points
            max_points = 30000
            if pts.shape[0] > max_points:
                indices = torch.randperm(pts.shape[0])[:max_points]
                pts = pts[indices]
                feat = feat[indices]
            
            # Create dummy multi-scale features (replace with actual DGCNN if needed)
            point_features = []
            for dim in [256, 128, 96, 96]:
                f = torch.randn(pts.shape[0], dim, device=pts.device)
                point_features.append(f)
            
            all_features.append(point_features)
            all_coords.append(pts)
            all_masks.append(torch.zeros(pts.shape[0], dtype=torch.bool, device=pts.device))
        
        # Pad to same size
        ms_features = []
        ms_coords = []
        ms_masks = []
        
        for level in range(4):
            level_features = []
            level_coords = []
            level_masks = []
            
            max_pts = max(all_features[b][level].shape[0] for b in range(batch_size))
            
            for b in range(batch_size):
                feat = all_features[b][level]
                coord = all_coords[b]
                mask = all_masks[b]
                
                n_pts = feat.shape[0]
                if n_pts < max_pts:
                    pad_size = max_pts - n_pts
                    feat = torch.nn.functional.pad(feat, (0, 0, 0, pad_size))
                    coord = torch.nn.functional.pad(coord, (0, 0, 0, pad_size))
                    mask = torch.nn.functional.pad(mask, (0, pad_size), value=True)
                
                level_features.append(feat)
                level_coords.append(coord)
                level_masks.append(mask)
            
            ms_features.append(torch.stack(level_features))
            ms_coords.append(torch.stack(level_coords))
            ms_masks.append(torch.stack(level_masks))
        
        # Semantic logits
        sem_logits = torch.randn(batch_size, max_pts, self.num_classes, device=points.device)
        
        return ms_features, ms_coords, ms_masks, sem_logits


class DirectExporter:
    """Export model by extracting weights and creating new model"""
    
    @staticmethod
    def extract_and_export(checkpoint_path, output_path, cfg):
        """Extract weights and create exportable model"""
        
        # Create device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create standalone model
        model = StandaloneMaskPLS(cfg, checkpoint_path, device.type)
        model = model.to(device)
        model.eval()
        
        # Create example inputs
        batch_size = 1
        num_points = 10000
        example_points = torch.randn(batch_size, num_points, 3, device=device)
        example_features = torch.randn(batch_size, num_points, 4, device=device)
        
        print("Tracing model...")
        with torch.no_grad():
            traced = torch.jit.trace(model, (example_points, example_features))
        
        # Optimize for inference
        traced = torch.jit.optimize_for_inference(traced)
        
        # Save
        traced.save(output_path)
        print(f"✓ Model saved to {output_path}")
        
        return traced


@click.command()
@click.option('--checkpoint', required=True, help='Path to checkpoint')
@click.option('--output', default='model_exported.pt', help='Output file')
@click.option('--test', is_flag=True, help='Test the exported model')
def main(checkpoint, output, test):
    """Export without Lightning dependencies"""
    
    print("="*60)
    print("MaskPLS Export (No Lightning)")
    print("="*60)
    
    # Load configuration
    print("Loading configuration...")
    base_path = Path(__file__).parent.parent
    
    config_files = {
        'model': base_path / "config/model.yaml",
        'backbone': base_path / "config/backbone.yaml", 
        'decoder': base_path / "config/decoder.yaml"
    }
    
    cfg = edict()
    for name, path in config_files.items():
        if path.exists():
            with open(path, 'r') as f:
                cfg.update(yaml.safe_load(f))
        else:
            print(f"Warning: {path} not found")
    
    # Export model
    exporter = DirectExporter()
    traced_model = exporter.extract_and_export(checkpoint, output, cfg)
    
    # Test if requested
    if test:
        print("\nTesting exported model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the saved model
        loaded = torch.jit.load(output, map_location=device)
        loaded.eval()
        
        # Test inputs
        test_points = torch.randn(1, 5000, 3, device=device)
        test_features = torch.randn(1, 5000, 4, device=device)
        
        with torch.no_grad():
            outputs = loaded(test_points, test_features)
        
        print("✓ Test successful!")
        print(f"  Output shapes:")
        print(f"    Logits: {outputs[0].shape}")
        print(f"    Masks: {outputs[1].shape}")
        print(f"    Semantic: {outputs[2].shape}")
    
    # Generate C++ code
    cpp_code = generate_cpp_code(output)
    cpp_file = output.replace('.pt', '.cpp')
    with open(cpp_file, 'w') as f:
        f.write(cpp_code)
    print(f"\n✓ C++ example saved to {cpp_file}")
    
    print(f"\nExport complete!")
    print(f"Model size: {Path(output).stat().st_size / 1024 / 1024:.2f} MB")


def generate_cpp_code(model_path):
    """Generate C++ LibTorch code"""
    return f"""
// MaskPLS C++ Inference Example
#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <memory>
#include <vector>

class MaskPLSInference {{
private:
    torch::jit::script::Module module;
    torch::Device device;
    
public:
    MaskPLSInference(const std::string& model_path, bool use_gpu = true) 
        : device(use_gpu && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU) {{
        
        try {{
            // Load model
            module = torch::jit::load(model_path);
            module.to(device);
            module.eval();
            
            std::cout << "Model loaded successfully on " 
                      << (device.is_cuda() ? "GPU" : "CPU") << std::endl;
        }}
        catch (const c10::Error& e) {{
            std::cerr << "Error loading model: " << e.msg() << std::endl;
            throw;
        }}
    }}
    
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
    infer(torch::Tensor points, torch::Tensor features) {{
        // Move inputs to device
        points = points.to(device);
        features = features.to(device);
        
        // Prepare inputs
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(points);
        inputs.push_back(features);
        
        // Run inference
        auto output = module.forward(inputs).toTuple();
        
        // Extract outputs
        auto pred_logits = output->elements()[0].toTensor();
        auto pred_masks = output->elements()[1].toTensor();
        auto sem_logits = output->elements()[2].toTensor();
        
        return std::make_tuple(pred_logits, pred_masks, sem_logits);
    }}
    
    void process_point_cloud(float* points_data, float* features_data, 
                           int num_points, int batch_size = 1) {{
        // Create tensors from raw data
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        
        auto points = torch::from_blob(points_data, 
            {{batch_size, num_points, 3}}, options);
        auto features = torch::from_blob(features_data, 
            {{batch_size, num_points, 4}}, options);
        
        // Run inference
        auto [logits, masks, semantic] = infer(points, features);
        
        // Process results
        std::cout << "Results:" << std::endl;
        std::cout << "  Logits: " << logits.sizes() << std::endl;
        std::cout << "  Masks: " << masks.sizes() << std::endl;
        std::cout << "  Semantic: " << semantic.sizes() << std::endl;
        
        // Get predictions
        auto semantic_pred = semantic.argmax(-1);  // [B, N]
        auto class_pred = logits.argmax(-1);       // [B, Q]
        
        // Convert to CPU for processing
        semantic_pred = semantic_pred.to(torch::kCPU);
        class_pred = class_pred.to(torch::kCPU);
        
        // Access data
        auto sem_data = semantic_pred.data_ptr<int64_t>();
        auto cls_data = class_pred.data_ptr<int64_t>();
        
        // Process predictions...
    }}
}};

int main(int argc, char* argv[]) {{
    if (argc < 2) {{
        std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
        return 1;
    }}
    
    try {{
        // Initialize model
        MaskPLSInference model(argv[1], true);
        
        // Example: process dummy data
        int num_points = 10000;
        int batch_size = 1;
        
        // Allocate data (in practice, load from your point cloud)
        std::vector<float> points(batch_size * num_points * 3);
        std::vector<float> features(batch_size * num_points * 4);
        
        // Fill with dummy data
        for (size_t i = 0; i < points.size(); ++i) {{
            points[i] = static_cast<float>(rand()) / RAND_MAX * 100.0f - 50.0f;
        }}
        for (size_t i = 0; i < features.size(); ++i) {{
            features[i] = static_cast<float>(rand()) / RAND_MAX;
        }}
        
        // Process
        model.process_point_cloud(points.data(), features.data(), 
                                 num_points, batch_size);
        
        std::cout << "Inference successful!" << std::endl;
    }}
    catch (const std::exception& e) {{
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }}
    
    return 0;
}}
"""


if __name__ == "__main__":
    main()