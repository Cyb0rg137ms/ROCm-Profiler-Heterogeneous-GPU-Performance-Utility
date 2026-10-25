"""
session.py
==========
Profiling session coordinator. Contains:
1. ProfilingSession: Context manager for managing trace operations.
2. KernelTracer: Tracing wrapper for HIP / CUDA kernel invocations.
3. MemoryTracker: VRAM allocator tracker with stack unwinding.
"""

import itertools
import time
import traceback
from typing import Dict, List, Optional

from profiler.records import EventType, MemoryEvent, ProfileRecord


class ProfilingSession:
    """Orchestrates profiling activities, capturing execution timelines and memory stats."""

    def __init__(self):
        self.records: List[ProfileRecord] = []
        self.memory_events: List[MemoryEvent] = []
        self.active = False

    def __enter__(self) -> "ProfilingSession":
        self.active = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.active = False

    def add_record(self, record: ProfileRecord):
        """Logs a timeline event record."""
        if self.active:
            self.records.append(record)

    def add_memory_event(self, event: MemoryEvent):
        """Logs a memory allocation or deallocation event."""
        if self.active:
            self.memory_events.append(event)


class _TraceContext:
    """Module-level context (v1 defined a new class on every ``trace`` call, ~5x slower)."""

    __slots__ = ("bytes_accessed", "flops", "name", "start_time", "stream_id", "tracer")

    def __init__(self, tracer: "KernelTracer", name: str, flops: float, bytes_accessed: float, stream_id: int):
        self.tracer, self.name, self.flops, self.bytes_accessed, self.stream_id = (
            tracer, name, flops, bytes_accessed, stream_id)

    def __enter__(self) -> "_TraceContext":
        if self.tracer.sync:
            self.tracer.sync()
        self.start_time = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.tracer.sync:
            self.tracer.sync()
        end = time.perf_counter_ns()
        self.tracer.session.add_record(ProfileRecord(
            name=self.name, start_ns=self.start_time, end_ns=end, stream_id=self.stream_id,
            event_type=EventType.KERNEL, metadata={"flops": self.flops, "bytes_accessed": self.bytes_accessed}))


class KernelTracer:
    """Hooks into HIP / PyTorch operations to measure execution latency and throughput."""

    def __init__(self, session: ProfilingSession, sync=None):
        """``sync`` (e.g. ``torch.cuda.synchronize``) is called at entry and exit so asynchronous
        device work is attributed to the block; without it only host-side time is measured."""
        self.session = session
        self.sync = sync

    def trace(self, name: str, flops: float = 0, bytes_accessed: float = 0, stream_id: int = 0):
        """
        Decorator / Context manager to profile a specific code execution block as a GPU kernel.
        """
        return _TraceContext(self, name, flops, bytes_accessed, stream_id)


class MemoryTracker:
    """
    Tracks memory allocations/deallocations.
    Captures python stack traces (stack unwinding) for memory leak diagnostics.
    """

    def __init__(self, session: ProfilingSession, capture_stacks: bool = True):
        self.session = session
        self.capture_stacks = capture_stacks
        self.allocation_stacks: Dict[int, List[str]] = {}
        self.live: Dict[int, int] = {}
        # v1 used id(object()) as the handle; CPython reuses ids of freed objects,
        # so two live allocations could share a handle. A counter is unique.
        self._ids = itertools.count(1)

    def allocate(self, size_bytes: int) -> int:
        """
        Simulates or logs a memory allocation.
        Generates a fake address as handle, records stack trace.
        """
        if size_bytes < 0:
            raise ValueError("size_bytes must be >= 0")
        addr = next(self._ids)
        if self.capture_stacks:
            stack = traceback.format_stack()
            self.allocation_stacks[addr] = [line.strip() for line in stack[:-1] if "session.py" not in line]
        self.live[addr] = size_bytes

        event = MemoryEvent(
            timestamp_ns=time.perf_counter_ns(),
            size_bytes=size_bytes,
            is_alloc=True
        )
        self.session.add_memory_event(event)
        return addr

    def deallocate(self, addr: int, size_bytes: int):
        """Simulates or logs a memory deallocation."""
        if addr not in self.live:
            raise KeyError(f"double free or unknown handle {addr}")
        if size_bytes != self.live[addr]:
            raise ValueError(f"free size {size_bytes} != allocated size {self.live[addr]}")
        del self.live[addr]
        self.allocation_stacks.pop(addr, None)

        event = MemoryEvent(
            timestamp_ns=time.perf_counter_ns(),
            size_bytes=size_bytes,
            is_alloc=False
        )
        self.session.add_memory_event(event)

    def leaks(self) -> Dict[int, int]:
        """Handles still allocated, with their sizes."""
        return dict(self.live)

    def get_stack_for_address(self, addr: int) -> Optional[List[str]]:
        """Returns the stack trace recorded at the time of allocation."""
        return self.allocation_stacks.get(addr)
