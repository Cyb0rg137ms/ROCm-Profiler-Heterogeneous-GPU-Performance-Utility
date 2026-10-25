"""Trace ingestion: raw event dicts, rocprof CSV, and Chrome-trace / torch.profiler JSON.

v2: metadata, device and any extra fields are preserved (v1 dropped them, so the
roofline could never see FLOP counts from parsed traces); events with
``end < start`` raise instead of silently getting duration 0.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, ClassVar

from profiler.records import EventType, ProfileRecord

_CORE = {"name", "start_ns", "end_ns", "stream", "type", "device"}


class TraceFormatError(ValueError):
    pass


class KernelTraceParser:
    TYPE_MAP: ClassVar[dict[str, EventType]] = {
        "kernel": EventType.KERNEL,
        "memcpy": EventType.MEMCPY,
        "memset": EventType.MEMSET,
        "marker": EventType.MARKER,
        "barrier": EventType.BARRIER,
    }

    def _event_type(self, t: str) -> EventType:
        return self.TYPE_MAP.get(t.lower(), EventType.KERNEL)

    def parse(self, raw_events: list[dict[str, Any]]) -> list[ProfileRecord]:
        records = []
        for i, ev in enumerate(raw_events):
            try:
                start, end = int(ev["start_ns"]), int(ev["end_ns"])
                name = str(ev["name"])
            except KeyError as e:
                raise TraceFormatError(f"event {i} missing field {e}") from None
            if end < start:
                raise TraceFormatError(f"event {i} ({name}) ends before it starts")
            md = dict(ev.get("metadata", {}))
            md.update({k: v for k, v in ev.items() if k not in _CORE and k != "metadata"})
            records.append(
                ProfileRecord(
                    name,
                    start,
                    end,
                    int(ev.get("stream", 0)),
                    self._event_type(str(ev.get("type", "kernel"))),
                    int(ev.get("device", 0)),
                    md,
                )
            )
        return sorted(records, key=lambda r: r.start_ns)

    def parse_rocprof_csv(self, path: str | Path) -> list[ProfileRecord]:
        """rocprof ``--stats``/``results.csv`` style: KernelName, BeginNs, EndNs[, gpu-id, queue-id, grd, wgr, ...]."""
        rows = []
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                name = row.get("KernelName") or row.get("Name") or row.get("kernel-name")
                b = row.get("BeginNs") or row.get("BeginNS") or row.get("Start_Timestamp")
                e = row.get("EndNs") or row.get("EndNS") or row.get("End_Timestamp")
                if name is None or b is None or e is None:
                    raise TraceFormatError(f"unrecognised rocprof CSV header: {list(row)}")
                extra = {k: v for k, v in row.items() if k not in {"KernelName", "BeginNs", "EndNs"}}
                rows.append(
                    {
                        "name": name,
                        "start_ns": int(b),
                        "end_ns": int(e),
                        "type": "kernel",
                        "stream": int(row.get("queue-id") or row.get("Queue_Id") or 0),
                        "device": int(row.get("gpu-id") or row.get("GPU_ID") or 0),
                        "metadata": extra,
                    }
                )
        return self.parse(rows)

    def parse_chrome_trace(self, path: str | Path, categories: set[str] | None = None) -> list[ProfileRecord]:
        """Chrome trace (``traceEvents`` with ``ph == "X"``, µs timestamps), as written by
        ``torch.profiler``'s ``export_chrome_trace`` or by :class:`profiler.writer.TimelineWriter`."""
        data = json.loads(Path(path).read_text())
        events = data["traceEvents"] if isinstance(data, dict) else data
        rows = []
        for ev in events:
            if ev.get("ph") != "X" or "dur" not in ev:
                continue
            cat = str(ev.get("cat", "kernel"))
            if categories and cat not in categories:
                continue
            start = round(float(ev["ts"]) * 1000)
            low = cat.lower()
            typ = "memcpy" if "memcpy" in low else "memset" if "memset" in low else "kernel"
            tid = ev.get("tid", 0)
            rows.append(
                {
                    "name": ev.get("name", "?"),
                    "start_ns": start,
                    "end_ns": start + round(float(ev["dur"]) * 1000),
                    "type": typ,
                    "stream": tid if isinstance(tid, int) else hash(tid) % 10_000,
                    "metadata": {"cat": cat, **(ev.get("args") or {})},
                }
            )
        return self.parse(rows)
