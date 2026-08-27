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
