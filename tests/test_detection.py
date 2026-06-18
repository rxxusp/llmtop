"""End-to-end fingerprint tests: the regression cases found against live engines.

These lock in the fixes for the systemic false-positive bug where every specific
adapter treated HTTP 401 as a positive signal (so an auth-gated router got
mis-claimed), and where Ollama claimed Open WebUI via a lone /api/version.
"""

from __future__ import annotations

from llmtop.discovery.fingerprint import fingerprint
from llmtop.models import Candidate, EngineType
from tests.conftest import make_client


def test_auth_router_is_openai_not_a_specific_engine(auth_router_routes):
    client = make_client(auth_router_routes)
    engine = fingerprint(Candidate(host="test", port=8077, signals=["port-scan"]), client)
    assert engine.engine_type == EngineType.OPENAI
    assert engine.last_error and "API key" in engine.last_error


def test_open_webui_is_unknown_not_ollama(open_webui_routes):
    client = make_client(open_webui_routes)
    engine = fingerprint(Candidate(host="test", port=8080, signals=["port-scan"]), client)
    assert engine.engine_type == EngineType.UNKNOWN


def test_vllm_wins_over_generic(vllm_routes):
    client = make_client(vllm_routes)
    engine = fingerprint(Candidate(host="test", port=8088, signals=["port-scan"]), client)
    assert engine.engine_type == EngineType.VLLM
    assert engine.primary_model == "qwen36-coder"


def test_ollama_detected_end_to_end(ollama_routes):
    client = make_client(ollama_routes)
    engine = fingerprint(Candidate(host="test", port=11434, signals=["port-scan"]), client)
    assert engine.engine_type == EngineType.OLLAMA
    assert len(engine.models) == 4


def test_generic_openai_list(monkeypatch):
    client = make_client({"/v1/models": {"object": "list",
                                         "data": [{"id": "a"}, {"id": "b"}]}})
    engine = fingerprint(Candidate(host="test", port=9000, signals=["port-scan"]), client)
    assert engine.engine_type == EngineType.OPENAI
    assert {m.id for m in engine.models} == {"a", "b"}


def test_silent_port_is_unknown():
    # Everything errors -> still returns a record, never raises.
    client = make_client({}, default_status=500)
    engine = fingerprint(Candidate(host="test", port=4321, signals=["port-scan"]), client)
    assert engine.engine_type == EngineType.UNKNOWN
    assert engine.port == 4321
