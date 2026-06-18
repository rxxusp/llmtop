"""vLLM engine adapter.

Detects vLLM instances by:
1. ``GET /v1/models`` returning a valid OpenAI-style model list **and**
2. ``GET /metrics`` containing at least one ``vllm:`` series.

A ``GET /version`` is attempted for the version string but is not required for
detection.

Metrics are read from the Prometheus ``/metrics`` endpoint using the shared
:mod:`~llmtop.adapters.prom` parser.  Token-rate derivation uses
:func:`~llmtop.adapters.base.derive_rate` with cumulative counter totals
handed in from the previous poll.

KV-cache percentage: vLLM exposes ``vllm:kv_cache_usage_perc`` (value 0..1)
in newer releases; older releases used ``vllm:gpu_cache_usage_perc``.  We try
the new name first and fall back to the old one.  The result is multiplied by
100 to produce a 0–100 percentage.
"""

from __future__ import annotations

import re
from typing import Optional

import httpx

from ..models import Candidate, EngineInfo, EngineMetrics, EngineType, ModelInfo
from .base import Adapter, derive_rate
from .prom import as_float, as_int, parse_prometheus, prom_value

# Heuristics for quantization / dtype from model-id strings.  Ordered so the
# most-specific match wins (checked in order, first match used).
_QUANT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bNVFP4\b", re.IGNORECASE), "NVFP4"),
    (re.compile(r"\bFP8\b", re.IGNORECASE), "FP8"),
    (re.compile(r"\bAWQ\b", re.IGNORECASE), "awq"),
    (re.compile(r"\bGPTQ\b", re.IGNORECASE), "gptq"),
    (re.compile(r"\bSqueezeLLM\b", re.IGNORECASE), "squeezellm"),
    (re.compile(r"\b(Q\d[_A-Z0-9]*)\b"), None),  # llama.cpp-style in model id
]

_DTYPE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bbfloat16\b", re.IGNORECASE), "bfloat16"),
    (re.compile(r"\bbf16\b", re.IGNORECASE), "bfloat16"),
    (re.compile(r"\bfloat16\b", re.IGNORECASE), "float16"),
    (re.compile(r"\bfp16\b", re.IGNORECASE), "float16"),
    (re.compile(r"\bfloat32\b", re.IGNORECASE), "float32"),
]


def _infer_quant(model_id: str) -> Optional[str]:
    """Best-effort quantization label from model id string."""
    for pat, label in _QUANT_PATTERNS:
        m = pat.search(model_id)
        if m:
            return label if label is not None else m.group(1)
    return None


def _infer_dtype(model_id: str) -> Optional[str]:
    """Best-effort dtype label from model id string."""
    for pat, label in _DTYPE_PATTERNS:
        if pat.search(model_id):
            return label
    return None


class VLLMAdapter(Adapter):
    """Adapter for vLLM inference servers.

    Detection requires both a valid ``GET /v1/models`` response AND a
    ``GET /metrics`` body that contains at least one ``vllm:`` series.
    This two-signal test avoids false positives against other OpenAI-compatible
    servers that happen to serve a ``/v1/models`` endpoint.
    """

    engine_type = EngineType.VLLM
    default_ports = (8000, 8001, 8088)
    priority = 10

    @classmethod
    def detect(cls, candidate: Candidate, client: httpx.Client) -> Optional[EngineInfo]:
        """Return an :class:`EngineInfo` if the candidate is a vLLM server.

        Probes ``/v1/models`` first (cheap); only if that succeeds probes
        ``/metrics`` to confirm the ``vllm:`` signature. Returns ``None`` on
        any network error or if the vLLM signature is absent.
        """
        base = candidate.base_url
        try:
            r = client.get(f"{base}/v1/models")
        except Exception:
            return None

        # 401/403 → engine is present but auth-blocked.  We cannot confirm it
        # is specifically vLLM without metrics, so fall through to generic.
        if r.status_code not in (200, 401, 403):
            return None

        if r.status_code == 200:
            try:
                data = r.json()
            except Exception:
                return None
            if not isinstance(data, dict) or data.get("object") != "list":
                return None
        # else: 401/403 path — we have no JSON body to verify, skip vLLM-specific check

        # Probe /metrics to confirm vllm: signature.
        try:
            mr = client.get(f"{base}/metrics")
        except Exception:
            # Can't confirm vLLM without metrics signal; let generic pick it up.
            return None

        if mr.status_code != 200:
            # metrics unreachable; can't confirm vLLM.
            return None

        metrics_text = mr.text
        if "vllm:" not in metrics_text:
            return None

        engine = EngineInfo(
            engine_type=EngineType.VLLM,
            name="vLLM",
            base_url=base,
            host=candidate.host,
            port=candidate.port,
            pid=candidate.pid,
            process=candidate.process,
            signals=list(candidate.signals) + ["vllm:metrics"],
        )
        return engine

    def describe(self, engine: EngineInfo, client: httpx.Client) -> None:
        """Populate models, version, and context length from ``/v1/models`` and ``/version``."""
        base = engine.base_url

        # --- /version ---
        try:
            vr = client.get(f"{base}/version")
            if vr.status_code == 200:
                vdata = vr.json()
                engine.version = str(vdata.get("version", "")).strip() or None
        except Exception:
            pass

        # --- /v1/models ---
        try:
            mr = client.get(f"{base}/v1/models")
            if mr.status_code in (401, 403):
                engine.last_error = "requires API key (set LLMTOP_API_KEY to introspect)"
                return
            if mr.status_code != 200:
                return
            mdata = mr.json()
        except Exception:
            return

        items = mdata.get("data", []) if isinstance(mdata, dict) else []
        models: list[ModelInfo] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            model_id: str = str(item.get("id", "")).strip()
            if not model_id:
                continue
            max_len = item.get("max_model_len")
            ctx = int(max_len) if isinstance(max_len, (int, float)) else None
            models.append(ModelInfo(
                id=model_id,
                context_length=ctx,
                quantization=_infer_quant(model_id),
                dtype=_infer_dtype(model_id),
            ))

        engine.models = models

    def metrics(
        self,
        engine: EngineInfo,
        client: httpx.Client,
        previous: Optional[EngineMetrics] = None,
        dt: Optional[float] = None,
    ) -> EngineMetrics:
        """Scrape ``/metrics`` and return structured :class:`EngineMetrics`.

        Token-rate fields (``decode_tps``, ``prefill_tps``) are derived from
        the delta of cumulative counters against the previous poll. All fields
        are ``None`` when the endpoint fails or a counter is absent.
        """
        base = engine.base_url
        try:
            r = client.get(f"{base}/metrics")
            if r.status_code != 200:
                return EngineMetrics(error=f"GET /metrics → HTTP {r.status_code}")
            text = r.text
        except Exception as exc:
            return EngineMetrics(error=f"GET /metrics failed: {exc}")

        try:
            parsed = parse_prometheus(text)
        except Exception as exc:
            return EngineMetrics(error=f"metrics parse failed: {exc}")

        # --- cumulative counters (as_int is finite-safe: rejects NaN/±Inf) ---
        gen_total = as_int(prom_value(parsed, "vllm:generation_tokens_total"))
        prompt_total = as_int(prom_value(parsed, "vllm:prompt_tokens_total"))

        # --- rates from delta ---
        prev_gen = previous.tokens_total if previous else None
        prev_prompt = previous.prompt_tokens_total if previous else None

        decode_tps = derive_rate(gen_total, prev_gen, dt)
        prefill_tps = derive_rate(prompt_total, prev_prompt, dt)

        # --- request counts ---
        running = as_int(prom_value(parsed, "vllm:num_requests_running"))
        waiting = as_int(prom_value(parsed, "vllm:num_requests_waiting"))

        # --- KV cache (0..1 → *100) ---
        kv_f = as_float(prom_value(parsed, "vllm:kv_cache_usage_perc"))
        if kv_f is None:
            kv_f = as_float(prom_value(parsed, "vllm:gpu_cache_usage_perc"))
        kv_pct = kv_f * 100.0 if kv_f is not None else None

        # --- request_success totals (sum across finished_reason labels) ---
        req_total = as_int(prom_value(parsed, "vllm:request_success_total"))

        return EngineMetrics(
            decode_tps=decode_tps,
            prefill_tps=prefill_tps,
            requests_running=running,
            requests_waiting=waiting,
            kv_cache_pct=kv_pct,
            tokens_total=gen_total,
            prompt_tokens_total=prompt_total,
            requests_total=req_total,
            raw={k: v for k, v in parsed.items() if k.startswith("vllm:")},
        )
