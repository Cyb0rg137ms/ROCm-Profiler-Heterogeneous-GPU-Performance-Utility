"""Measured benchmark suite (`rocm-profiler bench`). Runs on whatever device is present (CPU in CI)."""

from __future__ import annotations

import json
import platform
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from profiler.analyser import BottleneckAnalyser
from profiler.diagnostics import GPUDiagnosticsEngine
from profiler.parser import KernelTraceParser
from profiler.records import EventType, ProfileRecord
from profiler.session import KernelTracer, MemoryTracker, ProfilingSession
from profiler.timeline import summarise
from profiler.timing import time_fn


def _overhead(n: int) -> dict[str, float]:
    s = ProfilingSession()
    tr = KernelTracer(s)
    t_plain = time_fn(lambda: [None for _ in range(n)], repeats=5).median_s

    def traced() -> None:
        for _ in range(n):
            with tr.trace("k"):
                pass

    with s:
        t_tr = time_fn(traced, repeats=5).median_s
    out = {"tracer_ns_per_block": (t_tr - t_plain) / n * 1e9}
    for stacks in (False, True):
        m = MemoryTracker(s, capture_stacks=stacks)

        def alloc_free(m: MemoryTracker = m) -> None:
            for _ in range(n // 10):
                m.deallocate(m.allocate(64), 64)

        with s:
            out[f"alloc_free_us_{'stacks' if stacks else 'nostacks'}"] = (
                time_fn(alloc_free, repeats=3).median_s / (n // 10) * 1e6
            )
    return out


def _timeline_scale(n: int, seed: int = 0) -> dict[str, float]:
    rng = random.Random(seed)
    recs, t = [], 0
    for i in range(n):
        t += rng.randint(0, 2000)
        typ = EventType.MEMCPY if i % 4 == 0 else EventType.KERNEL
        recs.append(ProfileRecord(f"k{i % 50}", t, t + rng.randint(100, 5000), i % 4, typ))
    t0 = time.perf_counter()
    summ = summarise(recs)
    dt = time.perf_counter() - t0
    return {
        "events": n,
        "seconds": dt,
        "events_per_s": n / dt,
        "utilisation": summ.utilisation,
        "max_concurrency": summ.max_concurrency,
        "copy_overlap": summ.overlap_fraction_of_copies,
    }


def _torch_profiler_roundtrip() -> dict[str, Any]:
    from torch.profiler import ProfilerActivity, profile

    model = torch.nn.Sequential(torch.nn.Linear(256, 512), torch.nn.ReLU(), torch.nn.Linear(512, 10))
    x = torch.randn(64, 256)
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(5):
            model(x)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "trace.json"
        prof.export_chrome_trace(str(p))
        recs = KernelTraceParser().parse_chrome_trace(p)
    top = BottleneckAnalyser(recs).top_kernels(3)
    return {"events": len(recs), "top3": [(n, round(t / 1e3, 1)) for n, t in top]}


def run(quick: bool = False) -> dict[str, Any]:
    t_all = time.perf_counter()
    eng = GPUDiagnosticsEngine(repeats=5 if quick else 15)
    res: dict[str, Any] = {
        "host": {
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
            "device": eng.device,
            "backend": eng.backend,
        },
    }
    sizes = [128, 256, 512] if quick else [64, 128, 256, 512, 1024, 2048]
    res["gemm"] = [eng.run_compute_throughput(n) for n in sizes]
    res["triad"] = [eng.stream_triad(mb) for mb in ([16] if quick else [4, 16, 64, 256])]
    res["copy"] = eng.run_bandwidth_test(16 if quick else 128)
    peak_gflops = max(g["flops_tflops"] for g in res["gemm"]) * 1e3
    peak_bw = max(t["bandwidth_gbs"] for t in res["triad"])
    res["empirical_roofline"] = {
        "peak_gflops": peak_gflops,
        "peak_bw_gbs": peak_bw,
        "ridge_flop_per_byte": peak_gflops / peak_bw,
    }
    # place measured kernels on the measured roofline
    recs = []
    for g in res["gemm"]:
        n = g["matrix_size"]
        recs.append(
            ProfileRecord(
                f"gemm{n}",
                0,
                int(g["execution_time_ms"] * 1e6),
                0,
                EventType.KERNEL,
                metadata={"flops": 2 * n**3, "bytes_accessed": 3 * 4 * n * n},
            )
        )
    for t in res["triad"]:
        n = int(t["size_mb"] * 2**20) // 4
        dur = 3 * 4 * n / t["bandwidth_gbs"]
        recs.append(
            ProfileRecord(
                f"triad{t['size_mb']:g}MB",
                0,
                int(dur),
                0,
                EventType.KERNEL,
                metadata={"flops": 2 * n, "bytes_accessed": 3 * 4 * n},
            )
        )
    res["roofline"] = BottleneckAnalyser(recs).roofline_model(peak_gflops, peak_bw)
    res["overhead"] = _overhead(2_000 if quick else 20_000)
    res["timeline"] = _timeline_scale(10_000 if quick else 200_000)
    res["torch_profiler"] = _torch_profiler_roundtrip()
    res["total_s"] = time.perf_counter() - t_all
    return res


def to_markdown(r: dict[str, Any]) -> str:
    h = r["host"]
    o = [
        f"host: {h['machine']}, torch {h['torch']}, {h['threads']} threads, device={h['device']} ({h['backend']})",
        "",
        "| GEMM N | ms (median) | GFLOP/s |",
        "|---|---|---|",
    ]
    o += [f"| {g['matrix_size']} | {g['execution_time_ms']:.3f} | {g['flops_tflops'] * 1e3:.1f} |" for g in r["gemm"]]
    o += ["", "| triad MB | GB/s |", "|---|---|"] + [
        f"| {t['size_mb']:g} | {t['bandwidth_gbs']:.1f} |" for t in r["triad"]
    ]
    c = r["copy"]
    o += ["", f"copy ({c['engine']}, {c['size_processed_mb']:g} MB): {c['h2d_bandwidth_gbs']:.1f} GB/s"]
    e = r["empirical_roofline"]
    o += [
        f"empirical roofline: {e['peak_gflops']:.1f} GFLOP/s, {e['peak_bw_gbs']:.1f} GB/s, ridge {e['ridge_flop_per_byte']:.2f} FLOP/B",
        "",
        "| kernel | FLOP/B | GFLOP/s | % of roof | bound |",
        "|---|---|---|---|---|",
    ]
    o += [
        f"| {x['kernel_name']} | {x['arithmetic_intensity']:.2f} | {x['achieved_gflops']:.1f} | {x['efficiency_pct']:.0f} | {x['bound_type']} |"
        for x in r["roofline"]
    ]
    v = r["overhead"]
    o += [
        "",
        f"tracer overhead: {v['tracer_ns_per_block']:.0f} ns/block; alloc+free: {v['alloc_free_us_nostacks']:.2f} us "
        f"(no stacks) vs {v['alloc_free_us_stacks']:.2f} us (stacks)",
    ]
    t = r["timeline"]
    o += [f"timeline summarise: {t['events']:,} events in {t['seconds'] * 1e3:.0f} ms ({t['events_per_s']:,.0f} ev/s)"]
    p = r["torch_profiler"]
    o += [f"torch.profiler round-trip: {p['events']} events; top3 {p['top3']}", f"\ntotal {r['total_s']:.1f} s"]
    return "\n".join(o)


def plot(r: dict[str, Any], path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    e = r["empirical_roofline"]
    ai = np.logspace(-2, 3, 200)
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=130)
    ax.loglog(ai, np.minimum(e["peak_gflops"], ai * e["peak_bw_gbs"]), "k-", lw=2, label="measured roofline")
    for x in r["roofline"]:
        m = "o" if x["kernel_name"].startswith("gemm") else "s"
        ax.loglog(x["arithmetic_intensity"], x["achieved_gflops"], m, ms=7, alpha=0.85, label=x["kernel_name"])
    ax.set_xlabel("arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("GFLOP/s")
    h = r["host"]
    ax.set_title(
        f"Empirical roofline, {h['device']} ({h['threads']} threads), measured by `rocm-profiler bench`", fontsize=9
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path)


def save(r: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(r, indent=2, default=str))
