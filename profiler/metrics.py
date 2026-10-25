"""Unit-explicit performance metrics."""

from __future__ import annotations


class MetricsAggregator:
    @staticmethod
    def compute_gflops(flop_count: float, elapsed_ns: float) -> float:
        """GFLOP/s = flops / ns. Returns +inf for zero elapsed time."""
        return float("inf") if elapsed_ns == 0 else flop_count / elapsed_ns

    @staticmethod
    def compute_bandwidth_gbps(bytes_transferred: float, elapsed_ns: float) -> float:
        """GigaBITS per second (network-style units). For GB/s use :meth:`compute_bandwidth_gb_s`."""
        return float("inf") if elapsed_ns == 0 else bytes_transferred * 8.0 / elapsed_ns

    @staticmethod
    def compute_bandwidth_gb_s(bytes_transferred: float, elapsed_ns: float) -> float:
        """GigaBYTES per second (1e9 B/s), the unit GPU datasheets use."""
        return float("inf") if elapsed_ns == 0 else bytes_transferred / elapsed_ns

    @staticmethod
    def arithmetic_intensity(flops: float, bytes_accessed: float) -> float:
        if bytes_accessed <= 0:
            raise ValueError("bytes_accessed must be positive")
        return flops / bytes_accessed
