import json

import pytest

from profiler.analyser import BottleneckAnalyser
from profiler.diagnostics import GPUDiagnosticsEngine
from profiler.metrics import MetricsAggregator
from profiler.monitor import ROCmHardwareMonitor
from profiler.parser import KernelTraceParser, TraceFormatError
from profiler.records import EventType, ProfileRecord
from profiler.session import KernelTracer, MemoryTracker, ProfilingSession
from profiler.timeline import intersect_length, interval_union, summarise
from profiler.timing import time_fn
from profiler.writer import TimelineWriter

K, M = EventType.KERNEL, EventType.MEMCPY


def test_time_fn_counts_and_sync():
    calls = []
    t = time_fn(lambda: calls.append(1), repeats=4, warmup=2, sync=lambda: calls.append(0))
    assert len(t.samples_s) == 4 and calls.count(1) == 6 and calls.count(0) == 8
    with pytest.raises(ValueError):
        time_fn(lambda: None, repeats=0)


def test_diagnostics_are_measured_not_mocked():
    e = GPUDiagnosticsEngine(repeats=3)
    bw = e.run_bandwidth_test(4)
    g = e.run_compute_throughput(128)
    assert "mock" not in bw["engine"] and "mock" not in g["engine"]
    assert g["flops_tflops"] == pytest.approx(2 * 128**3 / (g["execution_time_ms"] / 1e3) / 1e12)
    assert e.stream_triad(4)["bandwidth_gbs"] > 0


def test_monitor_reads_fake_sysfs(tmp_path):
    hw = tmp_path / "card0" / "device" / "hwmon" / "hwmon3"
    hw.mkdir(parents=True)
    (hw / "temp1_input").write_text("61000\n")
    (hw / "power1_average").write_text("87500000\n")
    dev = tmp_path / "card0" / "device"
    (dev / "mem_info_vram_used").write_text(str(2 * 2**30))
    (dev / "mem_info_vram_total").write_text(str(16 * 2**30))
    (dev / "gpu_busy_percent").write_text("42")
    m = ROCmHardwareMonitor(0, sysfs=tmp_path)
    raw = m.poll()
    assert raw["temperature_c"] == 61.0 and raw["power_draw_watts"] == 87.5
    assert raw["vram_used_gb"] == 2.0 and raw["vram_total_gb"] == 16.0 and raw["gpu_busy_pct"] == 42.0
    assert m.poll_metrics()["status"] == "active"


def test_monitor_without_sensors_reports_unavailable(tmp_path):
    m = ROCmHardwareMonitor(0, sysfs=tmp_path)
    assert m.poll()["temperature_c"] is None
    pm = m.poll_metrics()
    assert pm["status"] == "unavailable" and pm["temperature_c"] == 0.0


def test_parser_preserves_metadata_and_validates():
    p = KernelTraceParser()
    r = p.parse([{"name": "g", "start_ns": 0, "end_ns": 10, "flops": 5, "device": 1}])[0]
    assert r.metadata["flops"] == 5 and r.device_id == 1
    with pytest.raises(TraceFormatError):
        p.parse([{"name": "g", "start_ns": 10, "end_ns": 0}])
    with pytest.raises(TraceFormatError):
        p.parse([{"name": "g", "start_ns": 0}])


def test_rocprof_csv(tmp_path):
    f = tmp_path / "results.csv"
    f.write_text("Index,KernelName,gpu-id,queue-id,BeginNs,EndNs\n0,gemm,0,1,100,600\n1,relu,0,1,700,750\n")
    recs = KernelTraceParser().parse_rocprof_csv(f)
    assert [r.name for r in recs] == ["gemm", "relu"] and recs[0].stream_id == 1 and recs[0].duration_ns == 500
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    with pytest.raises(TraceFormatError):
        KernelTraceParser().parse_rocprof_csv(bad)


def test_chrome_roundtrip(tmp_path):
    recs = [ProfileRecord("a", 1000, 3000, 0, K, metadata={"flops": 1}), ProfileRecord("b", 3000, 4000, 1, M)]
    f = tmp_path / "t.json"
    TimelineWriter(recs).write_chrome(str(f))
    back = KernelTraceParser().parse_chrome_trace(f)
    assert [(r.name, r.start_ns, r.end_ns, r.event_type) for r in back] == [
        ("a", 1000, 3000, K), ("b", 3000, 4000, M)]


def test_torch_profiler_ingest(tmp_path):
    import torch
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU]) as prof:
        torch.randn(64, 64) @ torch.randn(64, 64)
    f = tmp_path / "p.json"
    prof.export_chrome_trace(str(f))
    names = {r.name for r in KernelTraceParser().parse_chrome_trace(f)}
    assert any("mm" in n for n in names)


def test_interval_algebra():
    assert interval_union([(5, 7), (0, 2), (1, 3), (7, 8)]) == [(0, 3), (5, 8)]
    assert intersect_length([(0, 10)], [(2, 3), (5, 12)]) == 1 + 5


def test_timeline_summary():
    recs = [ProfileRecord("k1", 0, 100, 0, K), ProfileRecord("k2", 50, 150, 1, K),
            ProfileRecord("c", 120, 220, 2, M), ProfileRecord("k3", 300, 400, 0, K)]
    s = summarise(recs)
    assert s.span_ns == 400 and s.busy_ns == 320 and s.kernel_busy_ns == 250
    assert s.copy_compute_overlap_ns == 30 and s.max_concurrency == 2 and s.largest_gap_ns == 80
    assert s.per_stream_busy_ns == {0: 200, 1: 100, 2: 100}
    assert summarise([]).utilisation == 0.0


def test_roofline_unknown_without_counters_and_exceeds_flag():
    recs = [ProfileRecord("matmul", 0, 1000, 0, K),
            ProfileRecord("fast", 0, 1, 0, K, metadata={"flops": 1e9, "bytes_accessed": 1e6})]
    rl = BottleneckAnalyser(recs).roofline_model(1000.0, 100.0)
    assert rl[0]["bound_type"] == "unknown" and rl[0]["arithmetic_intensity"] is None
    assert rl[1]["exceeds_ceiling"]
    with pytest.raises(ValueError):
        BottleneckAnalyser(recs).roofline_model(0, 1)


def test_roofline_memory_bound_math():
    r = ProfileRecord("axpy", 0, 1000, 0, K, metadata={"flops": 2e3, "bytes_accessed": 12e3})
    x = BottleneckAnalyser([r]).roofline_model(1000.0, 100.0)[0]
    assert x["bound_type"] == "Memory Bound"
    assert x["ceiling_gflops"] == pytest.approx(100 * 2 / 12)
    assert x["achieved_gflops"] == pytest.approx(2.0)


def test_memory_tracker_unique_handles_leaks_and_errors():
    s = ProfilingSession()
    m = MemoryTracker(s, capture_stacks=False)
    with s:
        hs = [m.allocate(8) for _ in range(1000)]
        assert len(set(hs)) == 1000
        for h in hs[:-1]:
            m.deallocate(h, 8)
        assert m.leaks() == {hs[-1]: 8}
        with pytest.raises(KeyError):
            m.deallocate(hs[0], 8)
        with pytest.raises(ValueError):
            m.deallocate(hs[-1], 4)
        with pytest.raises(ValueError):
            m.allocate(-1)


def test_tracer_sync_and_stream():
    calls = []
    s = ProfilingSession()
    t = KernelTracer(s, sync=lambda: calls.append(1))
    with s, t.trace("k", stream_id=3):
        pass
    assert calls == [1, 1] and s.records[0].stream_id == 3 and s.records[0].duration_ns >= 0


def test_html_escapes_names(tmp_path):
    recs = [ProfileRecord("<script>x</script>", 0, 10, 0, K)]
    f = tmp_path / "r.html"
    TimelineWriter(recs).generate_html_report(str(f))
    txt = f.read_text()
    assert "<script>x</script>" not in txt and "&lt;script&gt;" in txt


def test_bandwidth_units():
    assert MetricsAggregator.compute_bandwidth_gb_s(1e9, 1e9) == 1.0
    assert MetricsAggregator.arithmetic_intensity(10, 5) == 2
    with pytest.raises(ValueError):
        MetricsAggregator.arithmetic_intensity(1, 0)


def test_cli_analyze(tmp_path, capsys):
    from profiler.cli import main

    f = tmp_path / "t.json"
    TimelineWriter([ProfileRecord("a", 0, 1000, 0, K, metadata={"flops": 10, "bytes_accessed": 10})]).write_chrome(str(f))
    html = tmp_path / "r.html"
    assert main(["analyze", str(f), "--peak-gflops", "100", "--peak-gbs", "10", "--html", str(html)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["events"] == 1 and out["roofline"][0]["kernel_name"] == "a" and html.exists()


def test_bench_quick_smoke():
    from profiler.benchmarks import run, to_markdown

    r = run(quick=True)
    assert r["empirical_roofline"]["peak_gflops"] > 0 and "GEMM" in to_markdown(r)
