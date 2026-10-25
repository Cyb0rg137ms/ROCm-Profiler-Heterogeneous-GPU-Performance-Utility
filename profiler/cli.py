"""``rocm-profiler`` command line: telemetry, diagnostics, trace analysis, benchmark."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rocm-profiler")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("telemetry").add_argument("--card", type=int, default=0)
    d = sub.add_parser("diag")
    d.add_argument("--size_mb", type=float, default=25.0)
    d.add_argument("--matrix_n", type=int, default=1024)
    a = sub.add_parser("analyze", help="analyse a trace (.json chrome/torch.profiler, or rocprof .csv)")
    a.add_argument("trace")
    a.add_argument("--peak-gflops", type=float)
    a.add_argument("--peak-gbs", type=float)
    a.add_argument("--html")
    a.add_argument("--chrome")
    b = sub.add_parser("bench")
    b.add_argument("--quick", action="store_true")
    b.add_argument("--json")
    b.add_argument("--plot")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    if args.cmd == "telemetry":
        from profiler.monitor import ROCmHardwareMonitor

        print(json.dumps(ROCmHardwareMonitor(args.card).poll(), indent=2))
    elif args.cmd == "diag":
        from profiler.diagnostics import GPUDiagnosticsEngine

        e = GPUDiagnosticsEngine()
        print(
            json.dumps(
                {"bandwidth": e.run_bandwidth_test(args.size_mb), "gemm": e.run_compute_throughput(args.matrix_n)},
                indent=2,
            )
        )
    elif args.cmd == "analyze":
        from profiler.analyser import BottleneckAnalyser
        from profiler.parser import KernelTraceParser
        from profiler.timeline import summarise
        from profiler.writer import TimelineWriter

        p = KernelTraceParser()
        recs = p.parse_rocprof_csv(args.trace) if args.trace.endswith(".csv") else p.parse_chrome_trace(args.trace)
        an = BottleneckAnalyser(recs)
        out = {
            "events": len(recs),
            "timeline": asdict(summarise(recs)),
            "top_kernels": an.top_kernels(10),
            "hotspots": an.find_hotspots(),
        }
        if args.peak_gflops and args.peak_gbs:
            out["roofline"] = an.roofline_model(args.peak_gflops, args.peak_gbs)
        w = TimelineWriter(recs)
        if args.html:
            w.generate_html_report(args.html, args.peak_gflops or 15000.0, args.peak_gbs or 800.0)
        if args.chrome:
            w.write_chrome(args.chrome)
        print(json.dumps(out, indent=2, default=str))
    else:
        from profiler.benchmarks import plot, run, save, to_markdown

        r = run(quick=args.quick)
        print(to_markdown(r))
        if args.json:
            save(r, args.json)
        if args.plot:
            Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
            plot(r, args.plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
