"""vLLM adapter: detection, describe, metrics, and rate derivation."""

from __future__ import annotations

from llmtop.adapters.vllm import VLLMAdapter
from llmtop.models import Candidate, EngineMetrics, EngineType
from tests.conftest import make_client


def test_detect_requires_vllm_metrics(vllm_routes):
    client = make_client(vllm_routes)
    cand = Candidate(host="test", port=8088)
    engine = VLLMAdapter.detect(cand, client)
    assert engine is not None
    assert engine.engine_type == EngineType.VLLM


def test_detect_rejects_plain_openai_server():
    # /v1/models present but no vllm: metrics -> not vLLM.
    client = make_client({"/v1/models": {"object": "list", "data": [{"id": "x"}]},
                          "/metrics": (404, "no metrics")})
    engine = VLLMAdapter.detect(Candidate(host="test", port=8000), client)
    assert engine is None


def test_describe_populates_model_and_version(vllm_routes):
    client = make_client(vllm_routes)
    cand = Candidate(host="test", port=8088)
    engine = VLLMAdapter.detect(cand, client)
    VLLMAdapter().describe(engine, client)
    assert engine.version == "0.20.1+test"
    assert engine.primary_model == "qwen36-coder"
    assert engine.models[0].context_length == 262144


def test_metrics_parse_and_derive():
    body = (
        'vllm:generation_tokens_total{model_name="m"} 1000.0\n'
        'vllm:prompt_tokens_total{model_name="m"} 500.0\n'
        'vllm:num_requests_running{model_name="m"} 1.0\n'
        'vllm:num_requests_waiting{model_name="m"} 0.0\n'
        'vllm:kv_cache_usage_perc{model_name="m"} 0.25\n'
    )
    routes = {
        "/v1/models": {"object": "list", "data": [{"id": "m", "max_model_len": 4096}]},
        "/metrics": (200, body),
        "/version": {"version": "0.20.1+test"},
    }
    client = make_client(routes)
    cand = Candidate(host="test", port=8088)
    engine = VLLMAdapter.detect(cand, client)
    adapter = VLLMAdapter()

    m1 = adapter.metrics(engine, client, previous=None, dt=None)
    assert m1.tokens_total == 1000
    assert m1.prompt_tokens_total == 500
    assert m1.requests_running == 1
    assert m1.requests_waiting == 0
    assert m1.kv_cache_pct == 25.0          # 0.25 -> percent
    assert m1.decode_tps is None            # no previous sample yet

    prev = EngineMetrics(tokens_total=900, prompt_tokens_total=460)
    m2 = adapter.metrics(engine, client, previous=prev, dt=2.0)
    assert m2.decode_tps == 50.0            # (1000-900)/2
    assert m2.prefill_tps == 20.0           # (500-460)/2


def test_metrics_never_raises_on_dead_endpoint():
    client = make_client({}, default_status=500)
    engine = VLLMAdapter.detect(Candidate(host="test", port=8088), client)
    # detect returns None for a 500 server; metrics on a hand-made engine must
    # still degrade gracefully rather than raise.
    from llmtop.models import EngineInfo
    e = EngineInfo(engine_type=EngineType.VLLM, name="vLLM",
                   base_url="http://test:8088", host="test", port=8088)
    m = VLLMAdapter().metrics(e, client, previous=None, dt=None)
    assert isinstance(m, EngineMetrics)
    assert m.error is not None
