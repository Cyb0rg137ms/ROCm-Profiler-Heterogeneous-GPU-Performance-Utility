"""parser.py — Kernel trace parser for ROCm profiler."""
from __future__ import annotations
from typing import Any, Dict, List
from profiler.records import ProfileRecord, EventType


class KernelTraceParser:
    """Parses raw HIP/ROCm trace event dicts into ProfileRecord objects."""

    TYPE_MAP = {
        "kernel": EventType.KERNEL,
        "memcpy": EventType.MEMCPY,
        "memset": EventType.MEMSET,
        "marker": EventType.MARKER,
    }

    def parse(self, raw_events: List[Dict[str, Any]]) -> List[ProfileRecord]:
        records = []
        for ev in raw_events:
            event_type = self.TYPE_MAP.get(ev.get("type", "kernel"), EventType.KERNEL)
            records.append(ProfileRecord(
                name=ev["name"],
                start_ns=ev["start_ns"],
                end_ns=ev["end_ns"],
                stream_id=ev.get("stream", 0),
                event_type=event_type,
            ))
        return sorted(records, key=lambda r: r.start_ns)
