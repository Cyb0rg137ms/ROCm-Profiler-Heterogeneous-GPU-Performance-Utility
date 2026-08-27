"""
analyser.py
===========
Bottleneck analysis and roofline modeling for ROCm GPU kernel traces.
"""

from typing import List, Dict, Tuple, Any
from collections import defaultdict
from profiler.records import ProfileRecord, EventType

class BottleneckAnalyser:
    """Analyzes profile records to identify bottlenecks, hotspots, and compute roofline models."""

    def __init__(self, records: List[ProfileRecord]):
        self.records = records

    def top_kernels(self, n: int = 5) -> List[Tuple[str, float]]:
        """
        Groups kernel records by name and returns the top n kernels by total duration (in ns).
        """
        if not self.records:
            return []
            
        kernel_times = defaultdict(float)
        for r in self.records:
            if r.event_type == EventType.KERNEL:
                kernel_times[r.name] += r.duration_ns
                
        sorted_kernels = sorted(kernel_times.items(), key=lambda item: item[1], reverse=True)
        return sorted_kernels[:n]

    def find_hotspots(self) -> List[Dict[str, Any]]:
        """
        Detects performance hotspots:
        - Kernels that run abnormally slowly.
        - High ratio of memory copies relative to kernel execution.
        """
        hotspots = []
        if not self.records:
            return hotspots

        # 1. Check total time distribution
        total_kernel_time = sum(r.duration_ns for r in self.records if r.event_type == EventType.KERNEL)
        total_memcpy_time = sum(r.duration_ns for r in self.records if r.event_type == EventType.MEMCPY)
        total_time = total_kernel_time + total_memcpy_time

        if total_time > 0 and (total_memcpy_time / total_time) > 0.5:
            hotspots.append({
                "category": "High Memory Overhead",
                "message": f"Memory transfers consume {(total_memcpy_time / total_time) * 100:.1f}% of total elapsed time. Consider pinning memory or using kernel fusion.",
                "severity": "HIGH"
            })

        # 2. Find kernels with unusually long individual execution times
        kernel_durations = [r.duration_ns for r in self.records if r.event_type == EventType.KERNEL]
        if kernel_durations:
            avg_duration = sum(kernel_durations) / len(kernel_durations)
            for r in self.records:
                if r.event_type == EventType.KERNEL and r.duration_ns > 5 * avg_duration and r.duration_ns > 1_000_000:
                    hotspots.append({
                        "category": "Long Kernel Execution",
                        "message": f"Kernel '{r.name}' duration ({r.duration_ms:.2f} ms) is 5x larger than average kernel duration ({avg_duration / 1e6:.2f} ms). Check grid sizes and register pressure.",
                        "severity": "MEDIUM"
                    })
                    
        return hotspots

    def roofline_model(self, peak_gflops: float, peak_bandwidth_gbps: float) -> List[Dict[str, Any]]:
        """
        Performs Roofline analysis on kernels.
        Estimates arithmetic intensity (FLOPs/byte) and categorizes performance limits.
        """
        roofline_results = []
        for r in self.records:
            if r.event_type != EventType.KERNEL:
                continue
                
            # Retrieve or mock FLOP counts and memory footprints from metadata
            flops = r.metadata.get("flops", 0.0)
            bytes_accessed = r.metadata.get("bytes_accessed", 0.0)
            
            if bytes_accessed <= 0:
                # Mock fallback estimation if metadata is absent
                if "matmul" in r.name.lower() or "gemm" in r.name.lower():
                    # Matmul: highly compute bound
                    flops = 2.0 * (1024 ** 3)
                    bytes_accessed = 3.0 * (1024 ** 2) * 4  # reading A, B, writing C
                else:
                    # Pointwise operations: memory bound
                    bytes_accessed = 1024 * 1024 * 4
                    flops = 1024 * 1024
                    
            intensity = flops / bytes_accessed if bytes_accessed > 0 else 0.0
            
            # GFLOPS achieved
            achieved_gflops = (flops / r.duration_ns) if r.duration_ns > 0 else 0.0
            
            # Roofline ceiling: min(Peak GFLOPs, Arithmetic Intensity * Peak Bandwidth)
            ceiling_gflops = min(peak_gflops, intensity * peak_bandwidth_gbps)
            
            # Efficiency
            efficiency = (achieved_gflops / ceiling_gflops) if ceiling_gflops > 0 else 0.0
            
            # Bound classification
            is_compute_bound = intensity >= (peak_gflops / peak_bandwidth_gbps)
            bound_type = "Compute Bound" if is_compute_bound else "Memory Bound"
            
            roofline_results.append({
                "kernel_name": r.name,
                "duration_ms": r.duration_ms,
                "arithmetic_intensity": intensity,
                "achieved_gflops": achieved_gflops,
                "ceiling_gflops": ceiling_gflops,
                "efficiency_pct": efficiency * 100.0,
                "bound_type": bound_type
            })
            
        return roofline_results
