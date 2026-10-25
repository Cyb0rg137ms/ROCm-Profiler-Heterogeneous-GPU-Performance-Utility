"""
writer.py
=========
Timeline writing utilities. Generates Chrome Trace Event Format JSON
and premium, interactive HTML diagnostic reports.
"""

import html
import json
from typing import List

from profiler.analyser import BottleneckAnalyser
from profiler.records import EventType, ProfileRecord


class TimelineWriter:
    """Writes profile recordings to file formats including Chrome tracing json and HTML reports."""

    def __init__(self, records: List[ProfileRecord]):
        self.records = records

    def write_chrome(self, file_path: str):
        """
        Exports the records in Chrome Trace Event Format (JSON).
        This can be loaded directly into chrome://tracing or speedscope.app.
        """
        events = []
        for r in self.records:
            # Determine category string
            cat = r.event_type.name
            
            # Chrome tracing expects timestamps in microseconds
            ts_us = r.start_ns / 1000.0
            dur_us = r.duration_ns / 1000.0
            
            # Format event dict
            events.append({
                "name": r.name,
                "cat": cat,
                "ph": "X",  # Complete event type
                "ts": ts_us,
                "dur": dur_us,
                "pid": 1,
                "tid": r.stream_id,
                "args": r.metadata
            })
            
        with open(file_path, "w") as f:
            json.dump({"traceEvents": events}, f, indent=2)

    def generate_html_report(self, file_path: str, peak_gflops: float = 15000.0, peak_bandwidth_gbps: float = 800.0):
        """
        Generates a premium glassmorphic/vibrant dark mode HTML report.
        Summarizes kernel timings, bottleneck findings, and roofline analysis.
        """
        analyser = BottleneckAnalyser(self.records)
        top_k = analyser.top_kernels(n=5)
        hotspots = analyser.find_hotspots()
        roofline = analyser.roofline_model(peak_gflops=peak_gflops, peak_bandwidth_gbps=peak_bandwidth_gbps)

        # Basic counts & stats
        num_kernels = sum(1 for r in self.records if r.event_type == EventType.KERNEL)
        num_memcpys = sum(1 for r in self.records if r.event_type == EventType.MEMCPY)
        total_time_ms = sum(r.duration_ms for r in self.records)

        # Construct top kernels list HTML
        top_kernels_html = ""
        for name, duration_ns in top_k:
            duration_ms = duration_ns / 1e6
            pct = (duration_ms / total_time_ms * 100) if total_time_ms > 0 else 0.0
            top_kernels_html += f"""
            <div class="kernel-item">
                <span class="kernel-name">{html.escape(name)}</span>
                <span class="kernel-duration">{duration_ms:.3f} ms ({pct:.1f}%)</span>
            </div>
            """

        # Construct hotspots list HTML
        hotspots_html = ""
        if not hotspots:
            hotspots_html = "<div class='no-hotspots'>No critical performance hotspots detected!</div>"
        for h in hotspots:
            sev_class = f"sev-{h['category'].lower().replace(' ', '-')}"
            hotspots_html += f"""
            <div class="hotspot-card {sev_class}">
                <strong>{html.escape(h['category'])}</strong>
                <p>{html.escape(h['message'])}</p>
            </div>
            """

        # Construct roofline list HTML
        roofline_html = ""
        for r in roofline:
            def _fmt(v, spec):
                return "n/a" if v is None else format(v, spec)

            roofline_html += f"""
            <tr>
                <td>{html.escape(r['kernel_name'])}</td>
                <td>{_fmt(r['arithmetic_intensity'], '.3f')} FLOP/B</td>
                <td>{_fmt(r['achieved_gflops'], '.1f')} GFLOPS</td>
                <td>{_fmt(r['ceiling_gflops'], '.1f')} GFLOPS</td>
                <td>{_fmt(r['efficiency_pct'], '.1f')}%</td>
                <td><span class="bound-tag {r['bound_type'].lower().replace(' ', '-')}">{r['bound_type']}</span></td>
            </tr>
            """

        # Complete HTML template
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ROCm GPU Performance Diagnostics Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0b0c10;
            --card-bg: rgba(31, 40, 51, 0.45);
            --border-color: rgba(102, 252, 241, 0.15);
            --primary: #66fcf1;
            --primary-hover: #45a29e;
            --text-main: #c5c6c7;
            --text-header: #ffffff;
            --accent-red: #ff4a5a;
            --accent-green: #2ecc71;
            --accent-yellow: #f1c40f;
        }}

        body {{
            background: radial-gradient(circle at top left, #1f2833 0%, #0b0c10 100%);
            color: var(--text-main);
            font-family: 'Outfit', sans-serif;
            margin: 0;
            padding: 40px 20px;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        header {{
            text-align: center;
            margin-bottom: 50px;
        }}

        header h1 {{
            color: var(--text-header);
            font-size: 3.5rem;
            font-weight: 800;
            margin: 0 0 10px 0;
            letter-spacing: -1px;
            background: linear-gradient(45deg, #66fcf1, #45a29e);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        header p {{
            font-size: 1.2rem;
            opacity: 0.8;
            margin: 0;
        }}

        .grid-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}

        .stat-card {{
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 25px;
            text-align: center;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}

        .stat-card:hover {{
            transform: translateY(-5px);
            border-color: var(--primary);
            box-shadow: 0 10px 20px rgba(102, 252, 241, 0.1);
        }}

        .stat-card h3 {{
            margin: 0 0 10px 0;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: var(--primary);
        }}

        .stat-card .val {{
            font-size: 2.5rem;
            font-weight: 800;
            color: var(--text-header);
        }}

        .main-section {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 30px;
            margin-bottom: 40px;
        }}

        @media (max-width: 900px) {{
            .main-section {{
                grid-template-columns: 1fr;
            }}
        }}

        .section-card {{
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 30px;
        }}

        .section-card h2 {{
            margin-top: 0;
            margin-bottom: 25px;
            font-size: 1.8rem;
            color: var(--text-header);
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 10px;
        }}

        .kernel-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}

        .kernel-name {{
            font-family: 'JetBrains Mono', monospace;
            color: var(--primary);
            font-weight: 600;
        }}

        .kernel-duration {{
            font-size: 0.95rem;
        }}

        .hotspot-card {{
            background: rgba(255, 74, 90, 0.1);
            border-left: 4px solid var(--accent-red);
            border-radius: 8px;
            padding: 15px 20px;
            margin-bottom: 15px;
        }}

        .hotspot-card p {{
            margin: 5px 0 0 0;
            font-size: 0.95rem;
            line-height: 1.4;
        }}

        .no-hotspots {{
            color: var(--accent-green);
            font-weight: 600;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}

        th, td {{
            padding: 15px 20px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}

        th {{
            color: var(--primary);
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        td {{
            font-size: 0.95rem;
        }}

        .bound-tag {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
        }}

        .bound-tag.compute-bound {{
            background: rgba(46, 204, 113, 0.2);
            color: var(--accent-green);
        }}

        .bound-tag.memory-bound {{
            background: rgba(241, 196, 110, 0.2);
            color: var(--accent-yellow);
        }}

        footer {{
            text-align: center;
            padding: 40px 0;
            font-size: 0.85rem;
            opacity: 0.5;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>ROCm GPU Performance Diagnostics</h1>
            <p>AMD Instinct™ & Radeon™ Telemetry Profiling Report</p>
        </header>

        <section class="grid-stats">
            <div class="stat-card">
                <h3>Total Trace Events</h3>
                <div class="val">{num_kernels + num_memcpys}</div>
            </div>
            <div class="stat-card">
                <h3>Kernel Launches</h3>
                <div class="val">{num_kernels}</div>
            </div>
            <div class="stat-card">
                <h3>Memory Transfers</h3>
                <div class="val">{num_memcpys}</div>
            </div>
            <div class="stat-card">
                <h3>Active Profiling Time</h3>
                <div class="val">{total_time_ms:.2f} ms</div>
            </div>
        </section>

        <section class="main-section">
            <div class="section-card">
                <h2>Top Kernel Execution Hotspots</h2>
                {top_kernels_html}
            </div>

            <div class="section-card">
                <h2>System Warnings</h2>
                {hotspots_html}
            </div>
        </section>

        <section class="section-card" style="margin-bottom: 40px;">
            <h2>Roofline Model & Bound Analysis</h2>
            <table>
                <thead>
                    <tr>
                        <th>Kernel Name</th>
                        <th>Arithmetic Intensity</th>
                        <th>Achieved Performance</th>
                        <th>Roofline Ceiling</th>
                        <th>Compute Efficiency</th>
                        <th>Limiting Bound</th>
                    </tr>
                </thead>
                <tbody>
                    {roofline_html}
                </tbody>
            </table>
        </section>

        <footer>
            Report generated by Antigravity ROCm Profiler • Ecosystem mapping courtesy of q.com
        </footer>
    </div>
</body>
</html>
"""
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_content)
