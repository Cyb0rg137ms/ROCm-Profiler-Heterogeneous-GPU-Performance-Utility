"""
records.py
==========
Core data structures for the ROCm GPU profiler.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List


class EventType(Enum):
    KERNEL = auto()
    MEMCPY = auto()
    MEMSET = auto()
    MARKER = auto()
    BARRIER = auto()


@dataclass
class ProfileRecord:
    name: str
    start_ns: int
    end_ns: int
    stream_id: int
    event_type: EventType
    device_id: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def duration_ns(self) -> int:
        return max(0, self.end_ns - self.start_ns)

    @property
    def duration_ms(self) -> float:
        return self.duration_ns / 1e6

    def overlaps(self, other: "ProfileRecord") -> bool:
        return (self.start_ns < other.end_ns and self.end_ns > other.start_ns
                and self.stream_id == other.stream_id)

    def __repr__(self) -> str:
        return (f"ProfileRecord({self.name!r}, "
                f"{self.duration_ns}ns, {self.event_type.name})")


@dataclass
class MemoryEvent:
    timestamp_ns: int
    size_bytes: int
    is_alloc: bool
    device_id: int = 0

    @staticmethod
    def peak_memory(events: List["MemoryEvent"]) -> int:
        current = 0
        peak = 0
        for ev in sorted(events, key=lambda e: e.timestamp_ns):
            current += ev.size_bytes if ev.is_alloc else -ev.size_bytes
            peak = max(peak, current)
        return peak
