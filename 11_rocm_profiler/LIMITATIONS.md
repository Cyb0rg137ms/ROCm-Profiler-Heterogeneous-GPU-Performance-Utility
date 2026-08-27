"""
LIMITATIONS.md — ROCm Profiler
==============================

# Known Limitations and Future Work

## 1. Hardware Metrics Sysfs Support
The hardware monitor reads power and temperature metrics from the Linux AMDGPU driver sysfs interface (`/sys/class/drm/card*/device/hwmon/hwmon*`):
- **Windows Support**: Sysfs is a Linux-only filesystem. On Windows, the profiler falls back to PyTorch-estimated load scaling or mock values.
- **Sysfs Paths**: Sysfs layout can vary slightly between different Linux kernel and ROCm driver versions, which may cause metric parsing failures in rare configurations.

## 2. In-Process Python Overhead
The context-manager-based `KernelTracer` and `MemoryTracker` operate within the Python runtime:
- **Instrumentation Overhead**: Intercepting and timing every operation in Python introduces a performance penalty. It is suitable for application-level profiling, but not for micro-benchmarking tiny kernels (durations < 1 microsecond).
- **Stack Unwinding Performance**: Capturing python call stacks via `traceback.format_stack()` is computationally expensive and should only be enabled during diagnostic/debugging runs.

## 3. Mock Fallback Dependency
When ROCm and PyTorch are unavailable:
- The diagnostics engine uses simulated execution times and throughput.
- This allows testing the parser, writer, and analyser in CI environments, but does not represent actual hardware limits.
"""
