"""llama.cpp server adapter.

Probes the llama.cpp REST server (default ports 8080, 8081) via ``GET /health`` and
``GET /props``. Prometheus metrics are fetched from ``GET /metrics`` when enabled;
if that endpoint returns 404, an ``EngineMetrics(error=...)`` is returned rather than
crashing.
"""

from __future__ import annotations

import re
from typing import Optional

import httpx

from ..models import Candidate, EngineInfo, EngineMetrics, EngineType, ModelInfo
from .base import Adapter, derive_rate
from .prom import as_float, as_int, parse_prometheus, prom_value

# Regex to extract a standard GGUF quantization tag from a filename, e.g.
# "Q4_K_M", "Q5_K_S", "IQ3_XS", "F16", "Q8_0", etc.
_QUANT_RE = re.compile(
    r"\b(IQ[1-4]_[A-Z_]+|Q[2-9]_[KF]_[A-Z]+|Q[2-9]_[0-9]|F(?:16|32)|BF16)\b",
    re.IGNORECASE,
)


def _parse_quant_from_path(path: str) -> Optional[str]:
    """Extract a GGUF quantization tag from a model file path or name."""
    if not path:
        return None
    basename = path.rsplit("/", 1)[-1]
    m = _QUANT_RE.search(basename)
    if m:
        return m.group(0).upper()
    return None


def _model_id_from_path(path: str) -> str:
    """Derive a short model id from a file path."""
    if not path:
        return "unknown"
    basename = path.rsplit("/", 1)[-1]
    # Strip .gguf suffix if present
    if basename.lower().endswith(".gguf"):
        basename = basename[:-5]
    return basename


class LlamaCppAdapter(Adapter):
    """Adapter for the llama.cpp HTTP server (llama-server)."""

    engine_type = EngineType.LLAMACPP
    default_ports = (8080, 8081)
    priority = 30

    @classmethod
    def detect(cls, candidate: Candidate, client: httpx.Client) -> Optional[EngineInfo]:
        """Return an EngineInfo if the candidate looks like a llama.cpp server.

        Detection tries ``GET /health`` (returns ``{"status":"ok"}``) first, then
        ``GET /props``. A 401/403 is treated as a positive signal.
        """
        base = candidate.base_url

        # Try /health first
        try:
            h_resp = client.get(f"{base}/health")
        except Exception:
            h_resp = None

        if h_resp is not None:
            if h_resp.status_code in (401, 403) and candidate.hint == cls.engine_type:
                return EngineInfo(
                    engine_type=EngineType.LLAMACPP,
                    name="llama.cpp",
                    base_url=base,
                    host=candidate.host,
                    port=candidate.port,
                    pid=candidate.pid,
                    process=candidate.process,
                    last_error="requires API key (set LLMTOP_API_KEY to introspect)",
                )
            if h_resp.status_code == 200:
                # `{"status":"ok"}` from /health is NOT distinctive — many servers
                # (including OpenAI-style routers) expose exactly that. Only trust it
                # as llama.cpp when the port or the process hint corroborates; for any
                # other port the authoritative signal is /props below.
                corroborated = (
                    candidate.port in cls.default_ports
                    or candidate.hint == cls.engine_type
                )
                try:
                    data = h_resp.json()
                    if isinstance(data, dict) and data.get("status") == "ok" and corroborated:
                        return EngineInfo(
                            engine_type=EngineType.LLAMACPP,
                            name="llama.cpp",
                            base_url=base,
                            host=candidate.host,
                            port=candidate.port,
                            pid=candidate.pid,
                            process=candidate.process,
                        )
                except Exception:
                    pass

        # Try /props as fallback detection signal
        try:
            p_resp = client.get(f"{base}/props")
        except Exception:
            return None

        if p_resp.status_code in (401, 403) and candidate.hint == cls.engine_type:
            return EngineInfo(
                engine_type=EngineType.LLAMACPP,
                name="llama.cpp",
                base_url=base,
                host=candidate.host,
                port=candidate.port,
                pid=candidate.pid,
                process=candidate.process,
                last_error="requires API key (set LLMTOP_API_KEY to introspect)",
            )

        if p_resp.status_code == 200:
            try:
                pdata = p_resp.json()
                # /props typically has default_generation_settings
                if isinstance(pdata, dict) and "default_generation_settings" in pdata:
                    return EngineInfo(
                        engine_type=EngineType.LLAMACPP,
                        name="llama.cpp",
                        base_url=base,
                        host=candidate.host,
                        port=candidate.port,
                        pid=candidate.pid,
                        process=candidate.process,
                    )
            except Exception:
                pass

        return None

    def describe(self, engine: EngineInfo, client: httpx.Client) -> None:
        """Enrich engine with model id, quantization, and context size from ``/props``."""
        base = engine.base_url
        try:
            p_resp = client.get(f"{base}/props")
            if p_resp.status_code != 200:
                return
            pdata = p_resp.json()
        except Exception:
            return

        if not isinstance(pdata, dict):
            return

        # Extract model path and n_ctx. ``default_generation_settings`` may be
        # null or absent on some builds, so coerce to an empty dict before use.
        dgs = pdata.get("default_generation_settings") or {}
        if not isinstance(dgs, dict):
            dgs = {}
        model_path: str = (dgs.get("model") or "") if dgs else ""
        # Some versions surface the path at the top level
        if not model_path:
            model_path = pdata.get("model_path") or pdata.get("model") or ""

        n_ctx: Optional[int] = None
        raw_ctx = dgs.get("n_ctx") or pdata.get("n_ctx")
        if raw_ctx is not None:
            try:
                n_ctx = int(raw_ctx)
            except (TypeError, ValueError):
                pass

        model_id = _model_id_from_path(model_path) if model_path else "unknown"
        quant = _parse_quant_from_path(model_path)

        engine.models = [
            ModelInfo(
                id=model_id,
                quantization=quant,
                context_length=n_ctx,
                loaded=True,
            )
        ]
        if model_path:
            engine.flags["model_path"] = model_path
        if n_ctx is not None:
            engine.flags["n_ctx"] = n_ctx

    def metrics(
        self,
        engine: EngineInfo,
        client: httpx.Client,
        previous: Optional[EngineMetrics] = None,
        dt: Optional[float] = None,
    ) -> EngineMetrics:
        """Return serving metrics from ``GET /metrics`` (Prometheus format).

        If the endpoint returns 404, the metrics endpoint is not enabled on this
        llama.cpp server; return an ``EngineMetrics(error=...)`` instead of crashing.
        """
        base = engine.base_url
        try:
            m_resp = client.get(f"{base}/metrics")
        except Exception as exc:
            return EngineMetrics(error=str(exc))

        if m_resp.status_code == 404:
            return EngineMetrics(error="metrics endpoint not enabled")

        if m_resp.status_code != 200:
            return EngineMetrics(error=f"GET /metrics returned HTTP {m_resp.status_code}")

        parsed = parse_prometheus(m_resp.text)
        prompt_total = as_int(prom_value(parsed, "llamacpp:prompt_tokens_total"))
        tokens_predicted = as_int(prom_value(parsed, "llamacpp:tokens_predicted_total"))
        requests_processing = as_int(prom_value(parsed, "llamacpp:requests_processing"))
        requests_deferred = as_int(prom_value(parsed, "llamacpp:requests_deferred"))
        kv_ratio = as_float(prom_value(parsed, "llamacpp:kv_cache_usage_ratio"))

        return EngineMetrics(
            decode_tps=derive_rate(tokens_predicted, previous.tokens_total if previous else None, dt),
            prefill_tps=derive_rate(prompt_total, previous.prompt_tokens_total if previous else None, dt),
            tokens_total=tokens_predicted,
            prompt_tokens_total=prompt_total,
            requests_running=requests_processing,
            requests_waiting=requests_deferred,
            kv_cache_pct=(kv_ratio * 100.0) if kv_ratio is not None else None,
            raw=dict(parsed),
        )
