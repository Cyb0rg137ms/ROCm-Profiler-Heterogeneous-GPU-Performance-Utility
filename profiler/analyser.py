"""
analyser.py
===========
Bottleneck analysis and roofline modeling for ROCm GPU kernel traces.
"""

from collections import defaultdict
from typing import Any, Dict, List, Tuple

from profiler.records import EventType, ProfileRecord


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
            
        kernel_times: Dict[str, float] = defaultdict(float)
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
        hotspots: List[Dict[str, Any]] = []
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
        """Roofline placement of every kernel that carries ``flops`` and ``bytes_accessed`` metadata.

        ``peak_bandwidth_gbps`` is in GB/s (bytes). v2: kernels without counters are
        reported with ``bound_type="unknown"`` instead of v1's invented FLOP/byte
        counts, and ``exceeds_ceiling`` flags metadata that implies a throughput above
        the roofline (a sign of wrong counters or peaks).
        """
        if peak_gflops <= 0 or peak_bandwidth_gbps <= 0:
            raise ValueError("peaks must be positive")
        ridge = peak_gflops / peak_bandwidth_gbps
        out: List[Dict[str, Any]] = []
        for r in self.records:
            if r.event_type != EventType.KERNEL:
                continue
            flops = float(r.metadata.get("flops", 0.0) or 0.0)
            nbytes = float(r.metadata.get("bytes_accessed", 0.0) or 0.0)
            if flops <= 0 or nbytes <= 0 or r.duration_ns <= 0:
                out.append({"kernel_name": r.name, "duration_ms": r.duration_ms, "arithmetic_intensity": None,
                            "achieved_gflops": None, "ceiling_gflops": None, "efficiency_pct": None,
                            "bound_type": "unknown", "exceeds_ceiling": False})
                continue
            ai = flops / nbytes
            achieved = flops / r.duration_ns
            ceiling = min(peak_gflops, ai * peak_bandwidth_gbps)
            out.append({
                "kernel_name": r.name,
                "duration_ms": r.duration_ms,
                "arithmetic_intensity": ai,
                "achieved_gflops": achieved,
                "ceiling_gflops": ceiling,
                "efficiency_pct": 100.0 * achieved / ceiling,
                "bound_type": "Compute Bound" if ai >= ridge else "Memory Bound",
                "exceeds_ceiling": achieved > ceiling * 1.0001,
            })
        return out
