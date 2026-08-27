"""
metrics.py
==========
Metrics aggregator for calculating GFLOPS, memory bandwidth,
and other GPU-specific performance indicators.
"""

class MetricsAggregator:
    """Aggregates and computes advanced profiling metrics."""

    @staticmethod
    def compute_gflops(flop_count: float, elapsed_ns: float) -> float:
        """
        Computes GFLOPS (Giga Floating-Point Operations Per Second).
        Handles division by zero by returning infinity.
        """
        if elapsed_ns == 0:
            return float("inf")
        # flop_count / (elapsed_ns / 1e9) -> FLOPS
        # FLOPS / 1e9 -> GFLOPS
        # Net result: flop_count / elapsed_ns
        return flop_count / elapsed_ns

    @staticmethod
    def compute_bandwidth_gbps(bytes_transferred: float, elapsed_ns: float) -> float:
        """
        Computes network/VRAM bandwidth in Gigabits per second (Gbps).
        Handles division by zero by returning infinity.
        """
        if elapsed_ns == 0:
            return float("inf")
        # (bytes_transferred * 8) / (elapsed_ns / 1e9) -> bps
        # bps / 1e9 -> Gbps
        # Net result: (bytes_transferred * 8) / elapsed_ns
        return (bytes_transferred * 8.0) / elapsed_ns
