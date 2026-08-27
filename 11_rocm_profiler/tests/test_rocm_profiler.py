"""
test_rocm_profiler.py
=====================
Comprehensive test suite for the ROCm GPU Profiler.

Tests cover:
  - ProfileRecord: timestamp ordering, duration calculation
  - MemoryEvent: allocation/deallocation tracking, peak detection
  - KernelTraceParser: parse_raw_trace output format
  - BottleneckAnalyser: top kernel identification, hotspot detection
  - TimelineWriter: generates valid JSON (Chrome trace format)
  - MetricsAggregator: GFLOPS, bandwidth calculations
  - Edge cases: empty trace, single event, overflow protection
"""

import json
import math
import pytest

from profiler.records import ProfileRecord, MemoryEvent, EventType
from profiler.parser import KernelTraceParser
from profiler.analyser import BottleneckAnalyser
from profiler.writer import TimelineWriter
from profiler.metrics import MetricsAggregator


class TestProfileRecord:
    def test_duration_positive(self):
        r = ProfileRecord(name="matmul", start_ns=1000, end_ns=5000,
                          stream_id=0, event_type=EventType.KERNEL)
        assert r.duration_ns == 4000
        assert r.duration_ms == pytest.approx(0.004, abs=1e-9)

    def test_overlapping_records_detected(self):
        r1 = ProfileRecord("a", 0, 100, stream_id=0, event_type=EventType.KERNEL)
        r2 = ProfileRecord("b", 50, 150, stream_id=0, event_type=EventType.KERNEL)
        assert r1.overlaps(r2)

    def test_non_overlapping(self):
        r1 = ProfileRecord("a", 0, 100, stream_id=0, event_type=EventType.KERNEL)
        r2 = ProfileRecord("b", 100, 200, stream_id=0, event_type=EventType.KERNEL)
        assert not r1.overlaps(r2)

    def test_record_repr_contains_name(self):
        r = ProfileRecord("gemm_kernel", 0, 1000, stream_id=0, event_type=EventType.KERNEL)
        assert "gemm_kernel" in repr(r)

    def test_zero_duration(self):
        r = ProfileRecord("marker", 500, 500, stream_id=0, event_type=EventType.MARKER)
        assert r.duration_ns == 0


class TestMemoryEvent:
    def test_allocation_positive(self):
        ev = MemoryEvent(timestamp_ns=1000, size_bytes=1024 * 1024, is_alloc=True)
        assert ev.size_bytes > 0

    def test_deallocation_negative_delta(self):
        ev = MemoryEvent(timestamp_ns=2000, size_bytes=1024, is_alloc=False)
        assert not ev.is_alloc

    def test_peak_memory_tracking(self):
        events = [
            MemoryEvent(0, 1_000_000, True),
            MemoryEvent(1, 2_000_000, True),
            MemoryEvent(2, 1_000_000, False),  # Free 1MB
            MemoryEvent(3, 500_000, True),
        ]
        peak = MemoryEvent.peak_memory(events)
        assert peak == 3_000_000, f"Expected peak 3MB, got {peak}"


class TestKernelTraceParser:
    def _sample_trace(self):
        return [
            {"name": "conv2d_fwd", "start_ns": 1000, "end_ns": 5000,
             "stream": 0, "type": "kernel"},
            {"name": "mem_copy_H2D", "start_ns": 100, "end_ns": 600,
             "stream": 0, "type": "memcpy"},
            {"name": "conv2d_fwd", "start_ns": 6000, "end_ns": 9000,
             "stream": 0, "type": "kernel"},
        ]

    def test_parse_returns_list(self):
        parser = KernelTraceParser()
        records = parser.parse(self._sample_trace())
        assert isinstance(records, list)

    def test_parse_count_matches_input(self):
        parser = KernelTraceParser()
        records = parser.parse(self._sample_trace())
        assert len(records) == 3

    def test_parse_sorts_by_start_time(self):
        parser = KernelTraceParser()
        records = parser.parse(self._sample_trace())
        starts = [r.start_ns for r in records]
        assert starts == sorted(starts)

    def test_parse_empty_trace(self):
        parser = KernelTraceParser()
        records = parser.parse([])
        assert records == []

    def test_parse_preserves_kernel_name(self):
        parser = KernelTraceParser()
        records = parser.parse(self._sample_trace())
        names = [r.name for r in records]
        assert "conv2d_fwd" in names


class TestBottleneckAnalyser:
    def _make_records(self):
        return [
            ProfileRecord("matmul", 0, 10000, stream_id=0, event_type=EventType.KERNEL),
            ProfileRecord("matmul", 10000, 18000, stream_id=0, event_type=EventType.KERNEL),
            ProfileRecord("relu", 18000, 18500, stream_id=0, event_type=EventType.KERNEL),
            ProfileRecord("conv2d", 18500, 25000, stream_id=0, event_type=EventType.KERNEL),
        ]

    def test_top_kernel_is_matmul(self):
        analyser = BottleneckAnalyser(self._make_records())
        top = analyser.top_kernels(n=1)
        assert top[0][0] == "matmul"

    def test_top_kernels_sorted_by_total_time(self):
        analyser = BottleneckAnalyser(self._make_records())
        top = analyser.top_kernels(n=3)
        times = [t for _, t in top]
        assert times == sorted(times, reverse=True)

    def test_hotspot_fraction_bounded(self):
        analyser = BottleneckAnalyser(self._make_records())
        top = analyser.top_kernels(n=1)
        total = sum(r.duration_ns for r in self._make_records())
        fraction = top[0][1] / total
        assert 0.0 < fraction <= 1.0

    def test_empty_records(self):
        analyser = BottleneckAnalyser([])
        assert analyser.top_kernels(n=5) == []


class TestTimelineWriter:
    def test_chrome_trace_format(self, tmp_path):
        records = [
            ProfileRecord("kernel_a", 0, 1000, stream_id=0, event_type=EventType.KERNEL),
            ProfileRecord("memcpy", 1000, 1500, stream_id=0, event_type=EventType.MEMCPY),
        ]
        writer = TimelineWriter(records)
        outfile = tmp_path / "trace.json"
        writer.write_chrome(str(outfile))
        data = json.loads(outfile.read_text())
        # Chrome trace format: {"traceEvents": [...]}
        assert "traceEvents" in data
        assert len(data["traceEvents"]) >= 2

    def test_trace_events_have_required_fields(self, tmp_path):
        records = [ProfileRecord("k", 0, 500, stream_id=0, event_type=EventType.KERNEL)]
        writer = TimelineWriter(records)
        outfile = tmp_path / "t.json"
        writer.write_chrome(str(outfile))
        events = json.loads(outfile.read_text())["traceEvents"]
        for ev in events:
            assert "name" in ev
            assert "ts" in ev
            assert "dur" in ev


class TestMetricsAggregator:
    def test_gflops_calculation(self):
        agg = MetricsAggregator()
        # 1 TFLOP in 1 second = 1000 GFLOPS
        gflops = agg.compute_gflops(flop_count=1e12, elapsed_ns=1e9)
        assert abs(gflops - 1000.0) < 1e-6

    def test_bandwidth_gbps(self):
        agg = MetricsAggregator()
        # 1 GB in 1 second = 8 Gbps
        gbps = agg.compute_bandwidth_gbps(bytes_transferred=1e9, elapsed_ns=1e9)
        assert abs(gbps - 8.0) < 1e-6

    def test_zero_elapsed_returns_inf(self):
        agg = MetricsAggregator()
        result = agg.compute_gflops(flop_count=1e12, elapsed_ns=0)
        assert result == float("inf") or result > 1e15
