"""Measured bandwidth and GEMM throughput on the best available device.

v2: every number is measured. v1 returned hard-coded "mock_fallback" values
(15.6 GB/s, 4.5 TFLOPs) whenever no GPU was present and timed single,
un-warmed copies with ``time.time``. Without a GPU the engine now measures the
host (``engine`` = ``"cpu_*"``); with a ROCm/CUDA device it times pinned-memory
H2D/D2H copies and on-device GEMMs with warm-up and synchronisation.
"""

from __future__ import annotations

from typing import Any

from profiler.timing import time_fn


def _torch() -> Any:
    try:
        import torch

        return torch
    except ImportError:  # pragma: no cover - torch is a hard dependency of the diagnostics
        return None


class GPUDiagnosticsEngine:
    def __init__(self, device: str | None = None, repeats: int = 10) -> None:
        torch = _torch()
        self.torch_available = bool(torch is not None and torch.cuda.is_available())
        self.device = device or ("cuda" if self.torch_available else "cpu")
        self.repeats = repeats
        self.backend = "none"
        if torch is not None:
            if self.device.startswith("cuda"):
                self.backend = "rocm" if getattr(torch.version, "hip", None) else "cuda"
            else:
                self.backend = "cpu"

    def _sync(self) -> Any:
        torch = _torch()
        return torch.cuda.synchronize if self.device.startswith("cuda") else None

    def run_bandwidth_test(self, size_mb: float = 50.0) -> dict[str, Any]:
        """H2D / D2H bandwidth (GPU) or host memcpy bandwidth (CPU). GB/s = 1e9 bytes/s."""
        torch = _torch()
        n = int(size_mb * 1024 * 1024) // 4
        nbytes = n * 4
        if self.device.startswith("cuda"):
            host = torch.empty(n, dtype=torch.float32).pin_memory()
            dev = torch.empty(n, dtype=torch.float32, device=self.device)
            h2d = time_fn(lambda: dev.copy_(host, non_blocking=True), repeats=self.repeats, sync=self._sync())
            d2h = time_fn(lambda: host.copy_(dev, non_blocking=True), repeats=self.repeats, sync=self._sync())
            engine = f"{self.backend}_pinned_copy"
        else:
            src = torch.ones(n, dtype=torch.float32)
            dst = torch.empty_like(src)
            h2d = d2h = time_fn(lambda: dst.copy_(src), repeats=self.repeats)
            engine = "cpu_memcpy"
        return {
            "h2d_latency_ms": h2d.median_s * 1e3,
            "h2d_bandwidth_gbs": nbytes / h2d.median_s / 1e9,
            "d2h_latency_ms": d2h.median_s * 1e3,
            "d2h_bandwidth_gbs": nbytes / d2h.median_s / 1e9,
            "size_processed_mb": size_mb,
            "repeats": self.repeats,
            "engine": engine,
        }

    def run_compute_throughput(self, matrix_size: int = 1024, dtype: str = "float32") -> dict[str, Any]:
        """Square GEMM throughput; FLOPs = 2 N^3 per multiply."""
        torch = _torch()
        n = matrix_size
        dt = getattr(torch, dtype)
        a = torch.randn(n, n, device=self.device).to(dt)
        b = torch.randn(n, n, device=self.device).to(dt)
        t = time_fn(lambda: a @ b, repeats=self.repeats, warmup=3, sync=self._sync())
        return {
            "flops_tflops": 2.0 * n**3 / t.median_s / 1e12,
            "matrix_size": n,
            "dtype": dtype,
            "execution_time_ms": t.median_s * 1e3,
            "iqr_ms": t.iqr_s * 1e3,
            "engine": f"{self.backend}_gemm",
        }

    def stream_triad(self, size_mb: float = 64.0) -> dict[str, Any]:
        """STREAM-style triad a = b + s*c; counts 3 arrays x 4 bytes of traffic per element."""
        torch = _torch()
        n = int(size_mb * 1024 * 1024) // 4
        b = torch.ones(n, device=self.device)
        c = torch.ones(n, device=self.device)
        a = torch.empty(n, device=self.device)
        t = time_fn(lambda: torch.add(b, c, alpha=2.0, out=a), repeats=self.repeats, sync=self._sync())
        return {"bandwidth_gbs": 3 * 4 * n / t.median_s / 1e9, "size_mb": size_mb, "engine": f"{self.backend}_triad"}
