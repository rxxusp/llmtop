"""GPU sampler: wraps NVML to collect per-GPU live state.

Handles the GB10/Jetson unified-memory case where nvmlDeviceGetMemoryInfo is
NOT_SUPPORTED but per-process compute memory and nearly all other calls work fine.

Public API:
    GpuSampler   — create once, call sample() each poll, close() on teardown.
"""

from __future__ import annotations

from typing import Optional

import psutil

from .models import GpuSample, ProcessInfo

# NVML sentinel value returned instead of None for "not available" memory.
_NVML_MEM_SENTINEL: int = 2**64 - 1

# Device-name substrings that signal unified (CPU+GPU shared) memory architectures.
_UNIFIED_NAMES: tuple[str, ...] = ("GB10", "Thor", "Orin", "Xavier", "Jetson", "Tegra")


def _is_sentinel(value: Optional[int]) -> bool:
    """Return True when *value* is the NVML 'not available' memory sentinel or None."""
    return value is None or value == _NVML_MEM_SENTINEL


class GpuSampler:
    """Collects live GpuSample objects for every GPU via NVML.

    Instantiation calls nvmlInit(); failures are caught and reflected as
    ``available=False``.  All per-field getters are individually wrapped so that
    a NOT_SUPPORTED call on one field (e.g. power cap, mem clock, fan on GB10)
    never prevents the other fields from being populated.
    """

    def __init__(self) -> None:
        self._available: bool = False
        self._device_count: int = 0
        self._nvml_ok: bool = False
        try:
            import pynvml  # type: ignore[import]
            self._pynvml = pynvml
            pynvml.nvmlInit()
            self._device_count = pynvml.nvmlDeviceGetCount()
            self._available = True
            self._nvml_ok = True
        except Exception:
            self._available = False

    @property
    def available(self) -> bool:
        """True when NVML was successfully initialised and at least one GPU exists."""
        return self._available and self._device_count > 0

    def sample(self) -> list[GpuSample]:
        """Return one :class:`GpuSample` per GPU; empty list when NVML is unavailable."""
        if not self._nvml_ok:
            return []
        results: list[GpuSample] = []
        pynvml = self._pynvml
        for index in range(self._device_count):
            sample = self._sample_one(pynvml, index)
            results.append(sample)
        return results

    def _sample_one(self, pynvml, index: int) -> GpuSample:  # noqa: ANN001
        """Build a GpuSample for device *index*, tolerating NOT_SUPPORTED on any field."""
        # Device handle — if this fails we return a minimal error sample.
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        except Exception as exc:
            return GpuSample(index=index, name="<unknown>", error=str(exc))

        # --- device name ---
        name: str = "<unknown>"
        try:
            name = pynvml.nvmlDeviceGetName(handle)
        except Exception:
            pass

        # Detect unified-memory architectures by name before we try meminfo.
        unified_by_name = any(token in name for token in _UNIFIED_NAMES)

        # --- utilization ---
        util_pct: Optional[float] = None
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            util_pct = float(util.gpu)
        except Exception:
            pass

        # --- memory ---
        mem_used_bytes: Optional[int] = None
        mem_total_bytes: Optional[int] = None
        unified_memory: bool = unified_by_name
        note: Optional[str] = None

        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used_bytes = mem_info.used
            mem_total_bytes = mem_info.total
        except Exception:
            # NOT_SUPPORTED on GB10 — switch to unified-memory path.
            unified_memory = True
            note = "unified memory (shared with system RAM)"
            vm = psutil.virtual_memory()
            mem_total_bytes = vm.total
            # mem_used_bytes will be filled from per-process sum below; fall back
            # to psutil.used if no process data is available.

        # --- per-process GPU memory ---
        procs: list[ProcessInfo] = self._collect_procs(pynvml, handle)

        # Fill mem_used_bytes from process sum when unified and not from meminfo.
        if unified_memory and mem_used_bytes is None:
            valid_proc_mems = [
                p.gpu_mem_bytes for p in procs if p.gpu_mem_bytes is not None
            ]
            if valid_proc_mems:
                mem_used_bytes = sum(valid_proc_mems)
            else:
                mem_used_bytes = psutil.virtual_memory().used

        # A device flagged unified by name (e.g. Jetson/Orin) where meminfo
        # happened to succeed still deserves the explanatory note for --json
        # consumers and the UI label.
        if unified_memory and note is None:
            note = "unified memory (shared with system RAM)"

        # --- temperature ---
        temp_c: Optional[float] = None
        try:
            temp_c = float(pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            ))
        except Exception:
            pass

        # --- power (usage + cap) ---
        power_w: Optional[float] = None
        try:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            power_w = power_mw / 1000.0
        except Exception:
            pass

        power_cap_w: Optional[float] = None
        try:
            cap_mw = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle)
            power_cap_w = cap_mw / 1000.0
        except Exception:
            # NOT_SUPPORTED on GB10
            pass

        # --- clocks ---
        clock_sm_mhz: Optional[int] = None
        try:
            clock_sm_mhz = int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
        except Exception:
            pass

        clock_mem_mhz: Optional[int] = None
        try:
            clock_mem_mhz = int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM))
        except Exception:
            # NOT_SUPPORTED on GB10
            pass

        # --- fan ---
        fan_pct: Optional[float] = None
        try:
            fan_pct = float(pynvml.nvmlDeviceGetFanSpeed(handle))
        except Exception:
            # NOT_SUPPORTED on GB10
            pass

        # --- throttle ---
        throttled: bool = False
        try:
            reasons = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
            idle = getattr(pynvml, "nvmlClocksThrottleReasonGpuIdle", 1)
            none_reason = getattr(pynvml, "nvmlClocksThrottleReasonNone", 0)
            throttled = reasons not in {0, idle, none_reason}
        except Exception:
            pass

        return GpuSample(
            index=index,
            name=name,
            util_pct=util_pct,
            mem_used_bytes=mem_used_bytes,
            mem_total_bytes=mem_total_bytes,
            temp_c=temp_c,
            power_w=power_w,
            power_cap_w=power_cap_w,
            clock_sm_mhz=clock_sm_mhz,
            clock_mem_mhz=clock_mem_mhz,
            fan_pct=fan_pct,
            throttled=throttled,
            unified_memory=unified_memory,
            procs=procs,
            note=note,
        )

    def _collect_procs(self, pynvml, handle) -> list[ProcessInfo]:  # noqa: ANN001
        """Collect per-process GPU memory from compute + graphics running processes.

        Compute entries win when a pid appears in both lists (dedup by pid).
        The NVML 'not available' sentinel (2**64-1) is treated as None.
        """
        seen: dict[int, ProcessInfo] = {}

        def _add(raw_procs, overwrite: bool) -> None:
            for p in raw_procs:
                raw_mem = getattr(p, "usedGpuMemory", None)
                mem: Optional[int] = None if _is_sentinel(raw_mem) else raw_mem

                pid = p.pid
                proc_name: Optional[str] = None
                try:
                    ps = psutil.Process(pid)
                    proc_name = ps.name()
                except Exception:
                    pass

                info = ProcessInfo(pid=pid, name=proc_name, gpu_mem_bytes=mem)
                if pid not in seen or overwrite:
                    seen[pid] = info

        # Graphics first (lower priority), then compute (overwrites on collision).
        try:
            gfx = pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
            _add(gfx, overwrite=False)
        except Exception:
            pass

        try:
            comp = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            _add(comp, overwrite=True)
        except Exception:
            pass

        return list(seen.values())

    def close(self) -> None:
        """Shut down NVML. Idempotent; swallows all errors."""
        if not self._nvml_ok:
            return
        try:
            self._pynvml.nvmlShutdown()
        except Exception:
            pass
        self._nvml_ok = False
        self._available = False
