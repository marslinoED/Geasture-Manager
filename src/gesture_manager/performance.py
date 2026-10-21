"""Low overhead process CPU and working-set sampling for local diagnostics."""

from __future__ import annotations

from ctypes import Structure, byref, sizeof, c_size_t, c_uint32, c_void_p
from dataclasses import dataclass
import os
from time import monotonic, process_time


@dataclass(frozen=True)
class ProcessMetrics:
    cpu_percent: float | None
    working_set_mb: float | None


class ProcessMetricsSampler:
    """Measure process CPU time and Windows working set without a new dependency."""

    def __init__(self) -> None:
        self._last_wall = monotonic()
        self._last_cpu = process_time()

    def sample(self) -> ProcessMetrics:
        now_wall, now_cpu = monotonic(), process_time()
        elapsed = now_wall - self._last_wall
        if elapsed >= 0.5:
            cpu = max(0.0, (now_cpu - self._last_cpu) * 100.0 / elapsed)
            self._last_wall, self._last_cpu = now_wall, now_cpu
        else:
            cpu = None
        memory = self._working_set_mb()
        return ProcessMetrics(cpu, memory)

    @staticmethod
    def _working_set_mb() -> float | None:
        if os.name != "nt":
            return None

        class MemoryCounters(Structure):
            _fields_ = [
                ("cb", c_uint32), ("PageFaultCount", c_uint32),
                ("PeakWorkingSetSize", c_size_t), ("WorkingSetSize", c_size_t),
                ("QuotaPeakPagedPoolUsage", c_size_t), ("QuotaPagedPoolUsage", c_size_t),
                ("QuotaPeakNonPagedPoolUsage", c_size_t), ("QuotaNonPagedPoolUsage", c_size_t),
                ("PagefileUsage", c_size_t), ("PeakPagefileUsage", c_size_t),
            ]

        try:
            import ctypes
            process = ctypes.windll.kernel32.GetCurrentProcess()  # type: ignore[attr-defined]
            counters = MemoryCounters()
            counters.cb = sizeof(counters)
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(  # type: ignore[attr-defined]
                c_void_p(process), byref(counters), counters.cb)
            return counters.WorkingSetSize / (1024 * 1024) if ok else None
        except (AttributeError, OSError, ValueError):
            return None
