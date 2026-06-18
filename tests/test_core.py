"""Monitor: VRAM attribution and appeared/disappeared event diffing."""

from __future__ import annotations

import os

from llmtop import core
from llmtop.core import Monitor, _EngineState
from llmtop.models import (
    EngineInfo,
    EngineMetrics,
    EngineType,
    GpuSample,
    ProcessInfo,
)


def _engine(port, etype=EngineType.VLLM, name="vLLM", pid=None):
    return EngineInfo(
        engine_type=etype, name=name,
        base_url=f"http://127.0.0.1:{port}", host="127.0.0.1", port=port, pid=pid,
    )


def test_vram_attribution_by_pid():
    mon = Monitor(enable_gpu=False)
    pid = os.getpid()
    eng = _engine(8000, pid=pid)
    mon._states[eng.key] = _EngineState(eng)
    gpu = GpuSample(index=0, name="g", procs=[ProcessInfo(pid=pid, gpu_mem_bytes=12345)])
    mon._attribute_vram([gpu])
    mon.close()
    assert eng.vram_bytes == 12345


def test_appeared_and_disappeared_events(monkeypatch):
    eng_a = _engine(8088, EngineType.VLLM, "vLLM")
    eng_b = _engine(11434, EngineType.OLLAMA, "Ollama")
    sequence = [[eng_a], [eng_a, eng_b], [eng_b]]
    state = {"i": 0}

    def fake_discover(client=None, extra_ports=()):
        idx = min(state["i"], len(sequence) - 1)
        state["i"] += 1
        return list(sequence[idx])

    class FakeAdapter:
        def metrics(self, engine, client, previous=None, dt=None):
            return EngineMetrics()

    monkeypatch.setattr(core, "discover", fake_discover)
    monkeypatch.setattr(core, "adapter_for", lambda et: FakeAdapter())

    mon = Monitor(enable_gpu=False, rediscover_every=0.0)
    s1 = mon.poll()
    s2 = mon.poll()
    s3 = mon.poll()
    mon.close()

    assert any("appeared" in e and "vLLM" in e for e in s1.events)
    assert any("appeared" in e and "Ollama" in e for e in s2.events)
    assert any("disappeared" in e and "vLLM" in e for e in s3.events)


def test_poll_never_raises_even_if_gpu_blows_up(monkeypatch):
    mon = Monitor(enable_gpu=False)

    def boom():
        raise RuntimeError("nvml exploded")

    monkeypatch.setattr(mon._gpu, "sample", boom)
    snap = mon.poll()          # must not raise
    mon.close()
    assert any("GPU sample error" in e for e in snap.errors)
