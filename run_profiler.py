"""v1 entry point kept for compatibility (``python run_profiler.py``); see ``rocm-profiler --help``."""

import json

from profiler.diagnostics import GPUDiagnosticsEngine
from profiler.monitor import ROCmHardwareMonitor

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="ROCm-Profiler telemetry + diagnostics")
    ap.add_argument("--card", type=int, default=0)
    ap.add_argument("--size_mb", type=float, default=25.0)
    ap.add_argument("--matrix_n", type=int, default=1024)
    a = ap.parse_args()
    e = GPUDiagnosticsEngine()
    print(json.dumps({"telemetry": ROCmHardwareMonitor(a.card).poll(), "bandwidth": e.run_bandwidth_test(a.size_mb),
                      "gemm": e.run_compute_throughput(a.matrix_n)}, indent=2))
