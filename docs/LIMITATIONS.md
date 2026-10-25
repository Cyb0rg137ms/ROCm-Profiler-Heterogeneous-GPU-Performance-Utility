# Limitations

- **Untested on GPUs.** The ROCm/CUDA code paths (pinned H2D/D2H copies, on-device GEMM, `torch.version.hip`
  detection, sysfs telemetry) are implemented, but CI and the published numbers are CPU-only, and no AMD GPU was
  available while this was written. The sysfs reader is tested against a fake sysfs tree using the amdgpu file names
  (`temp1_input`, `power1_average`, `mem_info_vram_*`, `gpu_busy_percent`).
- **No hardware counters.** The tool does not use rocprof/omniperf PMC collection. FLOP and byte counts must come from
  the trace metadata or from the caller.
- **CPU numbers are noisy.** The box is a shared VM, so expect ±10–20% run to run. The 4–64 MB triad results mostly
  measure the cache hierarchy, not DRAM. The 256 MB result (59 GB/s) is the closest to a DRAM figure.
- **Host-side tracing only.** `KernelTracer` measures host wall time. Without `sync=torch.cuda.synchronize` it does not
  capture GPU execution, and with sync it serialises the device.
- **Partial stream mapping for Chrome traces.** Non-integer `tid`s are hashed to a stream id.
