"""GPU sampler: the unified-memory (GB10) branch and the no-NVML degrade path.

NVML is faked by injecting a stand-in ``pynvml`` module, so these tests run on
any host (with or without a GPU).
"""

from __future__ import annotations

import sys
import types

import pytest

from llmtop.gpu import GpuSampler

GiB = 1024 ** 3


class _NVMLError(Exception):
    pass


class _Proc:
    def __init__(self, pid, mem):
        self.pid = pid
        self.usedGpuMemory = mem


def _make_fake_pynvml(*, init_raises=False, name="NVIDIA GB10"):
    m = types.ModuleType("pynvml")
    m.NVMLError = _NVMLError
    m.NVML_TEMPERATURE_GPU = 0
    m.NVML_CLOCK_SM = 1
    m.NVML_CLOCK_MEM = 2
    m.nvmlClocksThrottleReasonGpuIdle = 1
    m.nvmlClocksThrottleReasonNone = 0

    def nvmlInit():
        if init_raises:
            raise _NVMLError("driver not loaded")

    m.nvmlInit = nvmlInit
    m.nvmlShutdown = lambda: None
    m.nvmlDeviceGetCount = lambda: 1
    m.nvmlDeviceGetHandleByIndex = lambda i: object()
    m.nvmlDeviceGetName = lambda h: name

    util = types.SimpleNamespace(gpu=7, memory=0)
    m.nvmlDeviceGetUtilizationRates = lambda h: util

    def meminfo(h):
        raise _NVMLError("Not Supported")          # GB10 behaviour
    m.nvmlDeviceGetMemoryInfo = meminfo

    m.nvmlDeviceGetTemperature = lambda h, kind: 40
    m.nvmlDeviceGetPowerUsage = lambda h: 10000     # mW -> 10 W

    def cap(h):
        raise _NVMLError("Not Supported")           # GB10: power cap unsupported
    m.nvmlDeviceGetEnforcedPowerLimit = cap

    def clock(h, kind):
        if kind == m.NVML_CLOCK_MEM:
            raise _NVMLError("Not Supported")       # GB10: mem clock unsupported
        return 2400
    m.nvmlDeviceGetClockInfo = clock

    def fan(h):
        raise _NVMLError("Not Supported")           # GB10: fan unsupported
    m.nvmlDeviceGetFanSpeed = fan

    m.nvmlDeviceGetCurrentClocksThrottleReasons = lambda h: 0
    m.nvmlDeviceGetComputeRunningProcesses = lambda h: [_Proc(99991, 100 * GiB)]
    m.nvmlDeviceGetGraphicsRunningProcesses = lambda h: [_Proc(99992, 160 * 1024 ** 2)]
    return m


def test_unified_memory_branch(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", _make_fake_pynvml())
    sampler = GpuSampler()
    assert sampler.available
    samples = sampler.sample()
    sampler.close()

    assert len(samples) == 1
    g = samples[0]
    assert g.unified_memory is True
    assert g.note and "unified" in g.note
    # Working fields populate.
    assert g.util_pct == 7.0
    assert g.temp_c == 40.0
    assert g.power_w == 10.0
    assert g.clock_sm_mhz == 2400
    # Unsupported-on-GB10 fields degrade to None (not 0).
    assert g.power_cap_w is None
    assert g.clock_mem_mhz is None
    assert g.fan_pct is None
    # Total comes from system RAM; used is the sum of per-process GPU memory.
    assert g.mem_total_bytes is not None and g.mem_total_bytes > 0
    assert g.mem_used_bytes == 100 * GiB + 160 * 1024 ** 2
    # Per-process attribution works even though total meminfo did not.
    pids = {p.pid for p in g.procs}
    assert 99991 in pids and 99992 in pids


def test_no_nvml_degrades(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", _make_fake_pynvml(init_raises=True))
    sampler = GpuSampler()
    assert sampler.available is False
    assert sampler.sample() == []
    sampler.close()  # must be safe even when never initialised


def test_name_based_unified_detection(monkeypatch):
    # Even if meminfo worked, a Jetson/Orin name marks unified memory.
    fake = _make_fake_pynvml(name="NVIDIA Jetson Orin")
    fake.nvmlDeviceGetMemoryInfo = lambda h: types.SimpleNamespace(used=1 * GiB, total=8 * GiB)
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    sampler = GpuSampler()
    g = sampler.sample()[0]
    sampler.close()
    assert g.unified_memory is True
