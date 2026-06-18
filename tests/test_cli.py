"""CLI: JSON serialisation, --version, and headless --json path."""

from __future__ import annotations

import json

import pytest

from llmtop import cli
from llmtop.cli import main, snapshot_to_dict
from llmtop.models import EngineInfo, EngineType, GpuSample, Snapshot, SystemSample


def test_snapshot_to_dict_serialises_enums_and_tuples():
    snap = Snapshot(
        timestamp=1.0,
        gpus=[GpuSample(index=0, name="GB10", unified_memory=True)],
        system=SystemSample(load_avg=(1.0, 2.0, 3.0)),
        engines=[EngineInfo(engine_type=EngineType.VLLM, name="vLLM",
                            base_url="http://x:1", host="x", port=1)],
    )
    d = snapshot_to_dict(snap)
    json.dumps(d)  # must be JSON-serialisable without a custom encoder
    assert d["engines"][0]["engine_type"] == "vllm"
    assert isinstance(d["system"]["load_avg"], list)
    assert d["gpus"][0]["unified_memory"] is True


def test_version_exits_zero():
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


def test_json_mode_prints_snapshot(monkeypatch, capsys):
    snap = Snapshot(timestamp=1.0, engines=[])

    class FakeMonitor:
        def __init__(self, **kwargs):
            pass

        def poll(self):
            return snap

        def close(self):
            pass

    # _run_json does `from .core import Monitor`, so patch it on the core module.
    monkeypatch.setattr("llmtop.core.Monitor", FakeMonitor)
    rc = main(["--json", "--no-gpu"])
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert data["engines"] == []
    assert "timestamp" in data
