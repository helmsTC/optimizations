# ============================================================================
# FILE: mask_pls/utils/benchmarking.py (NEW FILE)  
# ============================================================================

"""
NEW FILE: Benchmarking utilities for optimization
"""

import time
import torch
import numpy as np

class ModelBenchmark:
    """
    Comprehensive benchmarking for MaskPLS optimization
    """
    
    def __init__(self, num_runs=30, warmup_runs=5):
        self.num_runs = num_runs
        self.warmup_runs = warmup_runs
    
    def benchmark_model(self, model, sample_input, name="Model"):
        """Benchmark a single model"""
        model.eval()
        times = []
        memory_usage = []
        
        # Warmup
        for _ in range(self.warmup_runs):
            with torch.no_grad():
                _ = model(sample_input)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Actual benchmark
        for i in range(self.num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            
            start_time = time.time()
            with torch.no_grad():
                outputs = model(sample_input)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_usage.append(torch.cuda.max_memory_allocated() / 1024**2)  # MB
            
            times.append(time.time() - start_time)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return {
            'name': name,
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'mean_memory': np.mean(memory_usage) if memory_usage else 0,
            'times': times
        }
    
    def compare_models(self, original_model, optimized_model, sample_input):
        """Compare original vs optimized model"""
        print("🏁 Benchmarking models...")
        
        orig_stats = self.benchmark_model(original_model, sample_input, "Original")
        opt_stats = self.benchmark_model(optimized_model, sample_input, "Optimized")
        
        speedup = orig_stats['mean_time'] / opt_stats['mean_time']
        memory_ratio = orig_stats['mean_memory'] / opt_stats['mean_memory'] if opt_stats['mean_memory'] > 0 else 1.0
        
        print(f"📊 Benchmark Results:")
        print(f"   Original:  {orig_stats['mean_time']:.4f}s ± {orig_stats['std_time']:.4f}s")
        print(f"   Optimized: {opt_stats['mean_time']:.4f}s ± {opt_stats['std_time']:.4f}s")
        print(f"   🚀 Speedup: {speedup:.2f}x")
        print(f"   💾 Memory ratio: {memory_ratio:.2f}x")
        
        return speedup, memory_ratio
