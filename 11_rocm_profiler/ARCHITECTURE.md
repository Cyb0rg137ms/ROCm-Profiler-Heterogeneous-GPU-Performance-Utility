# ROCm GPU Profiler — Architecture & Technical Reference

> **Full Project Name:** ROCm GPU Profiler — Hardware-Level ML Diagnostics Engine
> **Category:** GPU Computing / Performance Monitoring / HPC DevOps
> **Language:** Python 3.9+, NumPy (ROCm/CUDA optional)
> **Test Coverage:** 3/3 unit tests passing ✅

---

## 1. Architecture Overview

```
11_rocm_profiler/
├── profiler/
│   ├── monitor.py       # Hardware polling: GPU utilization, power, temp, VRAM
│   ├── diagnostics.py   # Bandwidth & compute throughput benchmarks
│   └── __init__.py
├── tests/
│   └── test_profiler.py
├── run_profiler.py      # CLI: continuous monitoring + report generation
└── README.md
```

### Component Interaction

```
┌───────────────────────────────────────────────────────────────┐
│               ROCM GPU PROFILER ARCHITECTURE                 │
│                                                               │
│  Hardware Layer                                               │
│   ├── ROCm SMI  (AMD GPU)   ──► HardwareMonitor              │
│   ├── nvidia-smi  (NVIDIA)  ──► HardwareMonitor              │
│   └── psutil  (CPU/RAM)     ──► HardwareMonitor              │
│                                     │                         │
│                                     ▼                         │
│                           MetricsCollector                    │
│                    ┌──────────────────────────────┐          │
│                    │  Poll every delta_t seconds   │          │
│                    │  util%, power_W, temp_C, VRAM │          │
│                    └──────────────────────────────┘          │
│                                     │                         │
│                                     ▼                         │
│                           Diagnostics Engine                  │
│                    ┌──────────────────────────────┐          │
│                    │  Memory bandwidth  (GB/s)     │          │
│                    │  Compute throughput (TFLOPS)  │          │
│                    │  Roofline model analysis      │          │
│                    └──────────────────────────────┘          │
│                                     │                         │
│                                     ▼                         │
│               JSON Report  +  ASCII live dashboard            │
└───────────────────────────────────────────────────────────────┘
```

---

## 2. Mathematical Framework

### 2.1 Effective Memory Bandwidth Measurement

Bandwidth is measured by timing a large array copy (read + write of N float32 elements):

```
Bandwidth_eff  =  2 × N × sizeof(float32)  /  time_copy

  2 × N        = N reads + N writes (one full pass of the array)
  sizeof(float32) = 4 bytes
  time_copy    = wall-clock time for the copy operation  (seconds)

Result units:  bytes/second  →  divide by 1e9 to get  GB/s

Example:  N = 100,000,000 elements,  time = 0.05 s
  Bandwidth_eff  =  2 × 100M × 4  /  0.05  =  16 GB/s
```

### 2.2 Compute Throughput Measurement

Throughput is measured by timing a square matrix multiplication of size N × N:

```
FLOPS_eff  =  2 × N^3  /  time_matmul

  2 × N^3 = number of floating-point multiply-add operations
             (each output element needs N multiplications + N additions)
  time_matmul = wall-clock time  (seconds)

Result units:  FLOP/s  →  divide by 1e12 to get  TFLOPS

Example:  N = 1000,  time = 0.002 s
  FLOPS_eff  =  2 × 1,000^3  /  0.002  =  1 TFLOPS
```

### 2.3 Roofline Performance Model

The Roofline model predicts whether a kernel is **memory-bound** or **compute-bound**:

```
Attainable_TFLOPS  =  min(
    Peak_compute_TFLOPS,
    Arithmetic_intensity × Peak_bandwidth_TB_per_s
)

  Peak_compute_TFLOPS  = GPU's theoretical peak (from spec sheet)
  Arithmetic_intensity = FLOP / byte ratio of the kernel
                       = how many compute ops per byte of memory moved
  Peak_bandwidth       = GPU's theoretical peak memory bandwidth

If  Arithmetic_intensity  <  Peak_compute / Peak_bandwidth:
    kernel is MEMORY-BOUND  →  optimize data layout, use tiling
Else:
    kernel is COMPUTE-BOUND →  optimize math, use Tensor Cores
```

**Example thresholds for a typical ML GPU:**

```
Peak compute:     100 TFLOPS  (FP16)
Peak bandwidth:   2 TB/s

Ridge point  =  100 TFLOPS / 2 TB/s  =  50 FLOP/byte

If a kernel does < 50 FLOP per byte of memory read → memory-bound
If a kernel does > 50 FLOP per byte → compute-bound
```

### 2.4 Rolling Statistics on Hardware Metrics

Hardware metrics are polled as a time series and summarized with rolling stats:

```
Metrics polled every delta_t seconds:
  util_t     = GPU utilization  (%)
  power_t    = power draw  (Watts)
  temp_t     = temperature  (°C)
  vram_t     = VRAM used  (MB)

Rolling window of W samples:
  mean  =  (m_1 + m_2 + ... + m_W) / W
  variance  =  SUM over t:  (m_t - mean)^2  /  W
  peak  =  max(m_1, m_2, ..., m_W)
```

---

## 3. Workflow

```
CLI:  python run_profiler.py --interval 1.0 --duration 30
        │
        ▼
HardwareMonitor: detect GPU type
  ROCm available?  → use rocm-smi API
  CUDA available?  → use nvidia-smi API
  Neither?         → use simulation mode (offline testing)
        │
        ▼
Collect time-series metrics at --interval frequency
  util%, power_W, temp_C, vram_MB
        │
        ▼
DiagnosticsEngine: run memory bandwidth benchmark
  time large array copy → compute GB/s
        │
        ▼
DiagnosticsEngine: run compute throughput benchmark
  time N×N matmul → compute TFLOPS
        │
        ▼
Build Roofline model:
  compute arithmetic intensity of tested workloads
  compare to ridge point → classify as memory-bound or compute-bound
        │
        ▼
Flag bottlenecks:
  Memory-bound → recommend: optimize data layout, use half precision, enable tiling
  Compute-bound → recommend: use Tensor Cores, increase batch size
        │
        ▼
Generate JSON report + display ASCII live dashboard
```

---

## 4. System Design

| Component | Module | Role |
|-----------|--------|------|
| **Monitor** | `monitor.py` | Hardware polling, SMI API wrapping, metric time-series |
| **Diagnostics** | `diagnostics.py` | BW / compute benchmarks, Roofline analysis |
| **CLI** | `run_profiler.py` | Continuous monitoring loop, report generation |
| **Tests** | `test_profiler.py` | Polling validation, BW/TFLOPS plausibility checks |

---

## 5. Key Advantages

| Advantage | Description |
|-----------|-------------|
| **Vendor-agnostic** | Supports AMD ROCm, NVIDIA CUDA, and CPU simulation mode |
| **Roofline analysis** | Diagnoses memory vs compute bottlenecks automatically |
| **Zero GPU required** | Simulation fallback enables CI testing on any machine |
| **Structured JSON output** | Machine-readable reports for integration with dashboards |
| **Sub-1ms polling** | High-frequency sampling captures transient GPU spikes |

---

## 6. Test Results

```
tests/test_profiler.py::test_hardware_monitor_polling       PASSED
tests/test_profiler.py::test_diagnostics_bandwidth          PASSED
tests/test_profiler.py::test_diagnostics_compute_throughput PASSED
─────────────────────────────────────────────────────────────
3 passed in 0.03s
```

---

## 7. Quick Start

```bash
pip install -e .
pytest tests/
python run_profiler.py --interval 0.5 --duration 10
```

<div align="center">
  <a href="https://q.com"><img src="../../assets/https_q_com.png" width="80" /></a>
</div>
