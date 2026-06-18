"""Ollama adapter: detection strictness + loaded/idle model handling."""

from __future__ import annotations

from llmtop.adapters.ollama import OllamaAdapter
from llmtop.models import Candidate, EngineType
from tests.conftest import make_client


def test_detect_on_real_tags(ollama_routes):
    client = make_client(ollama_routes)
    engine = OllamaAdapter.detect(Candidate(host="test", port=11434), client)
    assert engine is not None
    assert engine.engine_type == EngineType.OLLAMA


def test_describe_marks_models_idle_when_ps_empty(ollama_routes):
    client = make_client(ollama_routes)
    cand = Candidate(host="test", port=11434)
    engine = OllamaAdapter.detect(cand, client)
    OllamaAdapter().describe(engine, client)
    assert engine.version == "0.20.0"
    assert len(engine.models) == 4
    # /api/ps is empty in the fixture -> nothing is loaded in VRAM.
    assert all(m.loaded is False for m in engine.models)
    first = next(m for m in engine.models if m.id == "llama3.1:8b")
    assert first.quantization == "Q4_K_M"
    assert first.family == "llama"


def test_does_not_claim_open_webui(open_webui_routes):
    # /api/tags is HTML; /api/version is JSON but the port is not Ollama's and
    # there is no process hint -> must NOT be claimed as Ollama.
    client = make_client(open_webui_routes)
    engine = OllamaAdapter.detect(Candidate(host="test", port=8080), client)
    assert engine is None


def test_does_not_claim_auth_router_without_hint(auth_router_routes):
    client = make_client(auth_router_routes)
    engine = OllamaAdapter.detect(Candidate(host="test", port=8077), client)
    assert engine is None


def test_claims_auth_router_when_process_hints_ollama(auth_router_routes):
    client = make_client(auth_router_routes)
    cand = Candidate(host="test", port=11434, hint=EngineType.OLLAMA)
    engine = OllamaAdapter.detect(cand, client)
    assert engine is not None
    assert engine.engine_type == EngineType.OLLAMA
    assert "API key" in (engine.last_error or "")
