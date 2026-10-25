"""Telemetry for AMD GPUs from the amdgpu sysfs interface, plus torch memory counters.

v2: no invented values. v1 fabricated temperature (``45 + allocated % 30``) and
power, and reported a "mock" 16 GB card when no GPU was present; it also
hard-coded ``hwmon0``. Missing sensors are now reported as ``None`` in
``poll()``; the v1-compatible ``poll_metrics()`` reports 0.0 with
``status="unavailable"`` and ``source`` naming where each value came from.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _read_int(p: Path) -> int | None:
    try:
        return int(p.read_text().strip())
    except (OSError, ValueError):
        return None


class ROCmHardwareMonitor:
    def __init__(self, card_index: int = 0, sysfs: str | Path = "/sys/class/drm") -> None:
        self.card_index = card_index
        self.device_dir = Path(sysfs) / f"card{card_index}" / "device"

    @property
    def sysfs_root(self) -> Path | None:
        hw = sorted((self.device_dir / "hwmon").glob("hwmon*"))
        return hw[0] if hw else None

    def is_rocm_environment(self) -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available() and getattr(torch.version, "hip", None))
        except ImportError:
            return False

    def poll(self) -> dict[str, Any]:
        """Raw telemetry; ``None`` where a sensor is not available."""
        m: dict[str, Any] = {
            "temperature_c": None,
            "power_draw_watts": None,
            "vram_used_gb": None,
            "vram_total_gb": None,
            "gpu_busy_pct": None,
            "sources": [],
        }
        hw = self.sysfs_root
        if hw is not None:
            t = _read_int(hw / "temp1_input")
            if t is not None:
                m["temperature_c"] = t / 1000.0
            p = _read_int(hw / "power1_average")
            if p is None:
                p = _read_int(hw / "power1_input")
            if p is not None:
                m["power_draw_watts"] = p / 1e6
            m["sources"].append(str(hw))
        used = _read_int(self.device_dir / "mem_info_vram_used")
        total = _read_int(self.device_dir / "mem_info_vram_total")
        busy = _read_int(self.device_dir / "gpu_busy_percent")
        if total:
            m["vram_used_gb"], m["vram_total_gb"] = (used or 0) / 2**30, total / 2**30
            m["sources"].append("sysfs mem_info_vram_*")
        if busy is not None:
            m["gpu_busy_pct"] = float(busy)
        if m["vram_total_gb"] is None:
            try:
                import torch

                if torch.cuda.is_available():
                    i = min(self.card_index, torch.cuda.device_count() - 1)
                    free, total_b = torch.cuda.mem_get_info(i)
                    m["vram_total_gb"], m["vram_used_gb"] = total_b / 2**30, (total_b - free) / 2**30
                    m["sources"].append("torch.cuda.mem_get_info")
            except Exception as e:  # driver errors must not crash telemetry
                log.debug("torch telemetry failed: %s", e)
        return m

    def poll_metrics(self) -> dict[str, Any]:
        """v1-compatible view: floats (0.0 when unavailable) plus ``status`` and ``source``."""
        raw = self.poll()
        out: dict[str, Any] = {k: float(raw[k] or 0.0) for k in ("temperature_c", "vram_used_gb", "vram_total_gb", "power_draw_watts")}
        out["status"] = "active" if raw["sources"] else "unavailable"
        out["source"] = ", ".join(raw["sources"]) or "none"
        return out
