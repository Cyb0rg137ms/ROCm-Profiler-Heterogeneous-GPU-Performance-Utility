"""Timeline analytics: busy time (interval union), idle gaps, stream concurrency, overlap of copy and compute."""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass

from profiler.records import EventType, ProfileRecord


def interval_union(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def _length(iv: list[tuple[int, int]]) -> int:
    return sum(e - s for s, e in iv)


def intersect_length(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    """Total overlap of two *sorted, disjoint* interval lists (two-pointer sweep, O(len a + len b))."""
    i = j = tot = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        tot += max(0, hi - lo)
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


@dataclass(frozen=True)
class TimelineSummary:
    span_ns: int
    busy_ns: int
    kernel_busy_ns: int
    copy_busy_ns: int
    copy_compute_overlap_ns: int
    max_concurrency: int
    largest_gap_ns: int
    per_stream_busy_ns: dict[int, int]

    @property
    def utilisation(self) -> float:
        return self.busy_ns / self.span_ns if self.span_ns else 0.0

    @property
    def overlap_fraction_of_copies(self) -> float:
        return self.copy_compute_overlap_ns / self.copy_busy_ns if self.copy_busy_ns else 0.0


def summarise(records: list[ProfileRecord]) -> TimelineSummary:
    timed = [r for r in records if r.event_type != EventType.MARKER and r.duration_ns > 0]
    if not timed:
        return TimelineSummary(0, 0, 0, 0, 0, 0, 0, {})
    span = max(r.end_ns for r in timed) - min(r.start_ns for r in timed)
    allu = interval_union([(r.start_ns, r.end_ns) for r in timed])
    ku = interval_union([(r.start_ns, r.end_ns) for r in timed if r.event_type == EventType.KERNEL])
    cu = interval_union([(r.start_ns, r.end_ns) for r in timed if r.event_type in (EventType.MEMCPY, EventType.MEMSET)])
    by_stream: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r in timed:
        by_stream[r.stream_id].append((r.start_ns, r.end_ns))
    edges = sorted([(r.start_ns, 1) for r in timed] + [(r.end_ns, -1) for r in timed])  # ends sort before starts
    cur = peak = 0
    for _, d in edges:
        cur += d
        peak = max(peak, cur)
    gaps = [b[0] - a[1] for a, b in itertools.pairwise(allu)]
    return TimelineSummary(
        span,
        _length(allu),
        _length(ku),
        _length(cu),
        intersect_length(ku, cu),
        peak,
        max(gaps, default=0),
        {s: _length(interval_union(v)) for s, v in by_stream.items()},
    )
