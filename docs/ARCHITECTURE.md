# Architecture

```mermaid
flowchart LR
    subgraph Capture
      KT[session.KernelTracer<br/>perf_counter_ns + optional device sync]
      MT[session.MemoryTracker<br/>unique handles, leak report, optional stacks]
      TP[torch.profiler / rocprof]
    end
    subgraph Ingest
      P[parser.KernelTraceParser<br/>dicts, rocprof CSV, Chrome/torch JSON]
    end
    subgraph Analyse
      TL[timeline.summarise<br/>interval union, gaps, concurrency, copy/compute overlap]
      BA[analyser.BottleneckAnalyser<br/>top kernels, hotspots, roofline]
    end
    subgraph Measure
      DG[diagnostics.GPUDiagnosticsEngine<br/>GEMM, triad, H2D/D2H]
      MO[monitor.ROCmHardwareMonitor<br/>amdgpu sysfs, torch mem_get_info]
    end
    KT --> R[(ProfileRecord list)]
    TP --> P --> R
    R --> TL & BA
    DG -->|measured peaks| BA
    R --> W[writer.TimelineWriter<br/>Chrome trace, escaped HTML]
    BA --> W
```

- **Timing** (`profiler.timing.time_fn`) does warm-up, repeated samples, median and IQR. It calls `sync` before
  starting and before stopping every sample, so asynchronous GPU work is attributed to the right sample.
- **Timeline algorithms**:
  - `summarise` sorts once (O(n log n)) and computes the interval union for busy time.
  - A sweep over start/end edges gives the maximum concurrency.
  - A two-pointer intersection of the kernel and copy unions gives the copy/compute overlap.
- **Roofline**:
  - The peaks are inputs. `bench` measures them empirically: the best GEMM and the best triad run.
  - Kernels without FLOP/byte counters are labelled `unknown` rather than guessed.
