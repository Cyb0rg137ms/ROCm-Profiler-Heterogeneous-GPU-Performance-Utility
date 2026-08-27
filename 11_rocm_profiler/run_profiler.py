"""
run_profiler.py
===============
Command Line Interface to poll hardware statistics and run AMD GPU performance diagnostics.
"""

import argparse
from profiler.monitor import ROCmHardwareMonitor
from profiler.diagnostics import GPUDiagnosticsEngine

def main():
    parser = argparse.ArgumentParser(description="ROCm-Profiler: AMD GPU Telemetry & Benchmark CLI")
    parser.add_argument("--card", type=int, default=0, help="Target GPU Card index")
    parser.add_argument("--size_mb", type=float, default=25.0, help="Data transfer size for bandwidth tests (MB)")
    parser.add_argument("--matrix_n", type=int, default=1024, help="Matrix dimension size for compute tests")
    
    args = parser.parse_args()
    
    print("==================================================")
    print("      ROCM-PROFILER: AMD GPU TELEMETRY ENGINE     ")
    print("==================================================")
    
    # 1. Hardware Monitor Telemetry
    print(f"Polling GPU card index {args.card} status...")
    monitor = ROCmHardwareMonitor(card_index=args.card)
    metrics = monitor.poll_metrics()
    
    print("\n[TELEMETRY METRICS]")
    print(f"  GPU Operational Status: {metrics['status'].upper()}")
    print(f"  Core Temperature:       {metrics['temperature_c']:.1f} °C")
    print(f"  VRAM Memory Allocated:  {metrics['vram_used_gb']:.2f} GB / {metrics['vram_total_gb']:.2f} GB")
    print(f"  Active Power Draw:      {metrics['power_draw_watts']:.1f} Watts")
    
    # 2. Performance Diagnostics
    print("\n[DIAGNOSTICS BENCHMARKS]")
    engine = GPUDiagnosticsEngine()
    
    # Bandwidth
    print(f"Running Host-to-Device memory copies ({args.size_mb} MB)...")
    bw = engine.run_bandwidth_test(size_mb=args.size_mb)
    print(f"  Host-to-Device Speed: {bw['h2d_bandwidth_gbs']:.2f} GB/s (Latency: {bw['h2d_latency_ms']:.2f} ms)")
    print(f"  Device-to-Host Speed: {bw['d2h_bandwidth_gbs']:.2f} GB/s (Latency: {bw['d2h_latency_ms']:.2f} ms)")
    
    # Compute throughput
    print(f"Running float32 GEMM matrix multiplications ({args.matrix_n}x{args.matrix_n})...")
    comp = engine.run_compute_throughput(matrix_size=args.matrix_n)
    print(f"  Compute Throughput:   {comp['flops_tflops']:.2f} TFLOPs (Step: {comp['execution_time_ms']:.3f} ms)")
    
    print("\n[REPORT SUCCESS] Hardware profiling checks completed.")
    print("==================================================")

if __name__ == "__main__":
    main()
