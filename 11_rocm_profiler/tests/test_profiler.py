import pytest
from profiler.monitor import ROCmHardwareMonitor
from profiler.diagnostics import GPUDiagnosticsEngine

def test_hardware_monitor_polling():
    monitor = ROCmHardwareMonitor(card_index=0)
    metrics = monitor.poll_metrics()
    
    assert "status" in metrics
    assert "temperature_c" in metrics
    assert "vram_used_gb" in metrics
    assert "vram_total_gb" in metrics
    assert "power_draw_watts" in metrics
    
    # Assert reasonable value range boundaries
    assert 0.0 <= metrics["temperature_c"] <= 120.0
    assert metrics["vram_total_gb"] >= 0.0
    assert metrics["vram_used_gb"] <= metrics["vram_total_gb"]
    assert metrics["power_draw_watts"] >= 0.0

def test_diagnostics_bandwidth():
    engine = GPUDiagnosticsEngine()
    bw = engine.run_bandwidth_test(size_mb=10.0)
    
    assert bw["size_processed_mb"] == 10.0
    assert bw["h2d_bandwidth_gbs"] > 0.0
    assert bw["d2h_bandwidth_gbs"] > 0.0
    assert bw["h2d_latency_ms"] >= 0.0
    assert bw["d2h_latency_ms"] >= 0.0

def test_diagnostics_compute_throughput():
    engine = GPUDiagnosticsEngine()
    comp = engine.run_compute_throughput(matrix_size=512)
    
    assert comp["matrix_size"] == 512
    assert comp["flops_tflops"] > 0.0
    assert comp["execution_time_ms"] >= 0.0

def test_profiling_session():
    from profiler.session import ProfilingSession
    from profiler.records import ProfileRecord, EventType
    
    session = ProfilingSession()
    assert not session.active
    
    with session:
        assert session.active
        session.add_record(ProfileRecord("test_op", 100, 200, 0, EventType.KERNEL))
        
    assert not session.active
    assert len(session.records) == 1
    assert session.records[0].name == "test_op"

def test_kernel_tracer():
    from profiler.session import ProfilingSession
    from profiler.session import KernelTracer
    
    session = ProfilingSession()
    tracer = KernelTracer(session)
    
    with session:
        with tracer.trace("custom_kernel", flops=1e9, bytes_accessed=1e6):
            # simulate kernel work
            pass
            
    assert len(session.records) == 1
    rec = session.records[0]
    assert rec.name == "custom_kernel"
    assert rec.metadata["flops"] == 1e9
    assert rec.metadata["bytes_accessed"] == 1e6

def test_memory_tracker_stack_unwinding():
    from profiler.session import ProfilingSession
    from profiler.session import MemoryTracker
    
    session = ProfilingSession()
    tracker = MemoryTracker(session)
    
    with session:
        addr = tracker.allocate(size_bytes=4096)
        tracker.deallocate(addr, size_bytes=4096)
        
    assert len(session.memory_events) == 2
    assert session.memory_events[0].size_bytes == 4096
    assert session.memory_events[0].is_alloc is True
    assert session.memory_events[1].is_alloc is False
    
    stack = tracker.get_stack_for_address(addr)
    # The address has been deallocated so it's removed from allocation_stacks
    assert stack is None

def test_hotspot_and_roofline_analysis():
    from profiler.records import ProfileRecord, EventType
    from profiler.analyser import BottleneckAnalyser
    
    records = [
        ProfileRecord("matmul", 100, 200, 0, EventType.KERNEL, metadata={"flops": 1e12, "bytes_accessed": 1e9}),
        ProfileRecord("memcpy", 200, 800, 0, EventType.MEMCPY)
    ]
    
    analyser = BottleneckAnalyser(records)
    hotspots = analyser.find_hotspots()
    roofline = analyser.roofline_model(peak_gflops=10000.0, peak_bandwidth_gbps=500.0)
    
    assert len(hotspots) > 0
    assert any("High Memory" in h["category"] for h in hotspots)
    assert len(roofline) == 1
    assert roofline[0]["kernel_name"] == "matmul"
    assert roofline[0]["bound_type"] == "Compute Bound"

def test_html_report_generation(tmp_path):
    from profiler.records import ProfileRecord, EventType
    from profiler.writer import TimelineWriter
    
    records = [
        ProfileRecord("matmul", 100, 200, 0, EventType.KERNEL, metadata={"flops": 1e12, "bytes_accessed": 1e9}),
        ProfileRecord("memcpy", 200, 800, 0, EventType.MEMCPY)
    ]
    
    writer = TimelineWriter(records)
    report_file = tmp_path / "report.html"
    writer.generate_html_report(str(report_file))
    
    assert report_file.exists()
    html_content = report_file.read_text()
    assert "ROCm GPU Performance Diagnostics" in html_content
    assert "matmul" in html_content
    assert "System Warnings" in html_content

