"""
session.py
==========
Profiling session coordinator. Contains:
1. ProfilingSession: Context manager for managing trace operations.
2. KernelTracer: Tracing wrapper for HIP / CUDA kernel invocations.
3. MemoryTracker: VRAM allocator tracker with stack unwinding.
"""

import time
import traceback
from typing import List, Dict, Any, Optional
from profiler.records import ProfileRecord, MemoryEvent, EventType

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


class KernelTracer:
    """Hooks into HIP / PyTorch operations to measure execution latency and throughput."""

    def __init__(self, session: ProfilingSession):
        self.session = session

    def trace(self, name: str, flops: float = 0, bytes_accessed: float = 0):
        """
        Decorator / Context manager to profile a specific code execution block as a GPU kernel.
        """
        outer_self = self
        
        class TraceContext:
            def __enter__(self):
                self.start_time = time.time_ns()
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                end_time = time.time_ns()
                metadata = {
                    "flops": flops,
                    "bytes_accessed": bytes_accessed
                }
                record = ProfileRecord(
                    name=name,
                    start_ns=self.start_time,
                    end_ns=end_time,
                    stream_id=0,
                    event_type=EventType.KERNEL,
                    metadata=metadata
                )
                outer_self.session.add_record(record)
                
        return TraceContext()


class MemoryTracker:
    """
    Tracks memory allocations/deallocations.
    Captures python stack traces (stack unwinding) for memory leak diagnostics.
    """

    def __init__(self, session: ProfilingSession):
        self.session = session
        self.allocation_stacks: Dict[int, List[str]] = {}

    def allocate(self, size_bytes: int) -> int:
        """
        Simulates or logs a memory allocation.
        Generates a fake address as handle, records stack trace.
        """
        addr = id(object())  # generate unique handle
        stack = traceback.format_stack()
        # Clean up the stack output to keep only relevant lines
        clean_stack = [line.strip() for line in stack[:-1] if "session.py" not in line]
        
        self.allocation_stacks[addr] = clean_stack
        
        event = MemoryEvent(
            timestamp_ns=time.time_ns(),
            size_bytes=size_bytes,
            is_alloc=True
        )
        self.session.add_memory_event(event)
        return addr

    def deallocate(self, addr: int, size_bytes: int):
        """Simulates or logs a memory deallocation."""
        if addr in self.allocation_stacks:
            del self.allocation_stacks[addr]
            
        event = MemoryEvent(
            timestamp_ns=time.time_ns(),
            size_bytes=size_bytes,
            is_alloc=False
        )
        self.session.add_memory_event(event)

    def get_stack_for_address(self, addr: int) -> Optional[List[str]]:
        """Returns the stack trace recorded at the time of allocation."""
        return self.allocation_stacks.get(addr)
