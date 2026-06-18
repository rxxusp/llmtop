"""Regression tests for the 'degrade, never raise' contract.

These lock in the lead-review findings: adapters must not raise out of
describe()/metrics() on hostile-but-legal JSON shapes (null sub-objects, arrays
where dicts are expected, non-dict list entries) or non-finite Prometheus values
(+Inf / NaN, which would otherwise raise OverflowError on int()).
"""

from __future__ import annotations

import math

from llmtop.adapters.prom import as_float, as_int
from llmtop.adapters.llamacpp import LlamaCppAdapter
from llmtop.adapters.tgi import TGIAdapter
from llmtop.adapters.sglang import SGLangAdapter
from llmtop.adapters.ollama import OllamaAdapter
from llmtop.adapters.vllm import VLLMAdapter
from llmtop.models import EngineInfo, EngineType
from tests.conftest import make_client


def _engine(etype, port):
    return EngineInfo(engine_type=etype, name=etype.value,
                      base_url=f"http://test:{port}", host="test", port=port)


# ---- finite-safe coercion helpers -----------------------------------------

def test_as_int_and_as_float_are_finite_safe():
    assert as_int(float("inf")) is None
    assert as_int(float("-inf")) is None
    assert as_int(float("nan")) is None
    assert as_int("5") == 5
    assert as_int(None) is None
    assert as_float(float("inf")) is None
    assert as_float("nope") is None
    assert as_float("1.5") == 1.5


# ---- B1: llama.cpp describe on null/non-dict props -------------------------

def test_llamacpp_describe_null_generation_settings_does_not_raise():
    client = make_client({"/props": {"default_generation_settings": None,
                                     "model": "/models/foo.Q4_K_M.gguf", "n_ctx": 4096}})
    eng = _engine(EngineType.LLAMACPP, 8080)
    LlamaCppAdapter().describe(eng, client)        # must not raise
    assert eng.models and eng.models[0].id == "foo.Q4_K_M"
    assert eng.models[0].quantization == "Q4_K_M"


def test_llamacpp_describe_props_is_a_list_does_not_raise():
    client = make_client({"/props": [1, 2, 3]})
    eng = _engine(EngineType.LLAMACPP, 8080)
    LlamaCppAdapter().describe(eng, client)        # must not raise


# ---- B2: non-dict describe bodies / non-dict list entries ------------------

def test_tgi_describe_array_body_does_not_raise():
    client = make_client({"/info": [1, 2, 3]})
    eng = _engine(EngineType.TGI, 8080)
    TGIAdapter().describe(eng, client)             # must not raise


def test_sglang_describe_array_body_does_not_raise():
    client = make_client({"/get_model_info": [1, 2, 3]})
    eng = _engine(EngineType.SGLANG, 30000)
    SGLangAdapter().describe(eng, client)          # must not raise


def test_ollama_describe_skips_non_dict_model_entries():
    client = make_client({
        "/api/version": {"version": "9.9"},
        "/api/tags": {"models": ["a-bare-string", {"name": "real:model"}]},
        "/api/ps": {"models": ["also-a-string"]},
    })
    eng = _engine(EngineType.OLLAMA, 11434)
    OllamaAdapter().describe(eng, client)          # must not raise
    assert [m.id for m in eng.models] == ["real:model"]


# ---- B3: non-finite Prometheus counters -----------------------------------

def test_vllm_metrics_inf_counter_does_not_raise():
    routes = {"/metrics": (200, 'vllm:generation_tokens_total{m="x"} +Inf\n'
                                'vllm:num_requests_running{m="x"} 1.0\n')}
    eng = _engine(EngineType.VLLM, 8088)
    m = VLLMAdapter().metrics(eng, make_client(routes), previous=None, dt=None)
    assert m.error is None
    assert m.tokens_total is None          # +Inf rejected, not an OverflowError
    assert m.requests_running == 1


def test_llamacpp_metrics_inf_counter_does_not_raise():
    routes = {"/metrics": (200, "llamacpp:tokens_predicted_total +Inf\n"
                                "llamacpp:requests_processing 2\n")}
    eng = _engine(EngineType.LLAMACPP, 8080)
    m = LlamaCppAdapter().metrics(eng, make_client(routes), previous=None, dt=None)
    assert m.error is None
    assert m.tokens_total is None
    assert m.requests_running == 2


def test_tgi_and_sglang_metrics_inf_counter_do_not_raise():
    tgi_routes = {"/metrics": (200, "tgi_request_generated_tokens_sum +Inf\n"
                                    "tgi_queue_size 3\n")}
    mt = TGIAdapter().metrics(_engine(EngineType.TGI, 8080),
                              make_client(tgi_routes), previous=None, dt=None)
    assert mt.error is None and mt.tokens_total is None and mt.requests_waiting == 3

    sg_routes = {"/metrics": (200, "sglang:gen_throughput +Inf\n"
                                   "sglang:num_running_reqs 4\n")}
    ms = SGLangAdapter().metrics(_engine(EngineType.SGLANG, 30000),
                                 make_client(sg_routes), previous=None, dt=None)
    assert ms.error is None and ms.decode_tps is None and ms.requests_running == 4
