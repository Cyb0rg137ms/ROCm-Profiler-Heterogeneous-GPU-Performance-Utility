"""
diagnostics.py
==============
Executes performance diagnostics on AMD GPUs, measuring Host-to-Device memory 
bandwidth and raw matrix multiplication throughput (TFLOPs).
"""

import time
from typing import Dict, Any

class GPUDiagnosticsEngine:
    """Benchmarks GPU memory transfers and matrix compute capabilities."""
    
    def __init__(self):
        self.torch_available = False
        try:
            import torch
            self.torch_available = torch.cuda.is_available()
        except ImportError:
            self.torch_available = False

    def run_bandwidth_test(self, size_mb: float = 50.0) -> Dict[str, Any]:
        """
        Measures Host-to-Device (H2D) and Device-to-Host (D2H) memory bandwidth.
        
        Returns:
            Dictionary with transfer times and GB/s bandwidth speeds.
        """
        # Calculate size in bytes
        bytes_count = int(size_mb * 1024 * 1024)
        
        if not self.torch_available:
            # Mock fallback metrics
            return {
                "h2d_latency_ms": 3.2,
                "h2d_bandwidth_gbs": 15.6,
                "d2h_latency_ms": 3.4,
                "d2h_bandwidth_gbs": 14.7,
                "size_processed_mb": size_mb,
                "engine": "mock_fallback"
            }
            
        import torch
        
        # 1. Host-to-Device (H2D) copy
        host_tensor = torch.randn(bytes_count // 4, dtype=torch.float32)  # float32 = 4 bytes
        
        t0 = time.time()
        gpu_tensor = host_tensor.to("cuda")
        torch.cuda.synchronize()
        h2d_time = time.time() - t0
        
        # 2. Device-to-Host (D2H) copy
        t0 = time.time()
        back_tensor = gpu_tensor.to("cpu")
        torch.cuda.synchronize()
        d2h_time = time.time() - t0
        
        # Calculate speed (GB/s)
        gb_size = size_mb / 1024.0
        h2d_gbs = gb_size / h2d_time if h2d_time > 0 else 0.0
        d2h_gbs = gb_size / d2h_time if d2h_time > 0 else 0.0
        
        return {
            "h2d_latency_ms": h2d_time * 1000.0,
            "h2d_bandwidth_gbs": h2d_gbs,
            "d2h_latency_ms": d2h_time * 1000.0,
            "d2h_bandwidth_gbs": d2h_gbs,
            "size_processed_mb": size_mb,
            "engine": "rocm_cuda_h2d"
        }

    def run_compute_throughput(self, matrix_size: int = 1024) -> Dict[str, Any]:
        """
        Measures matrix multiplication (GEMM) performance in Floating Point Operations (TFLOPs).
        
        Returns:
            Dictionary containing operation counts and TFLOPs throughput.
        """
        if not self.torch_available:
            return {
                "flops_tflops": 4.5,
                "matrix_size": matrix_size,
                "execution_time_ms": 0.5,
                "engine": "mock_fallback"
            }
            
        import torch
        
        # GEMM allocation (N x N)
        n = matrix_size
        a = torch.randn(n, n, dtype=torch.float32, device="cuda")
        b = torch.randn(n, n, dtype=torch.float32, device="cuda")
        
        # Warmup loop
        for _ in range(5):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        
        # Timing loop
        iterations = 50
        t0 = time.time()
        for _ in range(iterations):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed = (time.time() - t0) / iterations
        
        # Calculate operations count: 2 * N^3 operations per GEMM
        ops = 2.0 * (n ** 3)
        tflops = (ops / elapsed) / 1e12
        
        return {
            "flops_tflops": tflops,
            "matrix_size": n,
            "execution_time_ms": elapsed * 1000.0,
            "engine": "rocm_cuda_gemm"
        }
