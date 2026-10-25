"""Robust micro-benchmark timing: warm-up, repeated samples, median/IQR, device sync."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Timing:
    samples_s: tuple[float, ...]

    @property
    def median_s(self) -> float:
        return statistics.median(self.samples_s)

    @property
    def min_s(self) -> float:
        return min(self.samples_s)

    @property
    def iqr_s(self) -> float:
        if len(self.samples_s) < 4:
            return 0.0
        q = statistics.quantiles(self.samples_s, n=4)
        return q[2] - q[0]


def time_fn(
    fn: Callable[[], object], *, repeats: int = 10, warmup: int = 2, sync: Callable[[], None] | None = None
) -> Timing:
    """Time ``fn`` ``repeats`` times after ``warmup`` calls; ``sync`` (e.g. ``torch.cuda.synchronize``)
    is called before starting and before stopping every sample so asynchronous devices are measured correctly."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    for _ in range(warmup):
        fn()
    s = sync or (lambda: None)
    out = []
    for _ in range(repeats):
        s()
        t0 = time.perf_counter()
        fn()
        s()
        out.append(time.perf_counter() - t0)
    return Timing(tuple(out))
