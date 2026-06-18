"""Generic OpenAI-compatible adapter.

This adapter is the penultimate fallback — it catches any server that speaks
the OpenAI ``GET /v1/models`` API but was not claimed by a more specific
adapter (vLLM, Ollama, llama.cpp, TGI, SGLang).  It is also the adapter used
for **ROUTER** engines (routers that front multiple backends and present a
merged model list via the OpenAI protocol).

Authentication handling (per ARCHITECTURE.md):
  An HTTP 401 or 403 response to ``GET /v1/models`` is treated as a positive
  detection signal: the endpoint exists and the engine is running, but
  introspection is blocked by an API key.  In this case we still return a
  valid ``EngineInfo`` with ``last_error`` set and an empty model list.

Metrics:
  We attempt ``GET /metrics`` (Prometheus text format).  If the endpoint
  returns a non-200 status or the body is not Prometheus text, we return an
  empty ``EngineMetrics`` without an error, since most generic OpenAI servers
  do not expose Prometheus metrics.
"""

from __future__ import annotations

from typing import Optional

import httpx

from ..models import Candidate, EngineInfo, EngineMetrics, EngineType, ModelInfo
from .base import Adapter
from .prom import parse_prometheus


class OpenAIGenericAdapter(Adapter):
    """Adapter for generic OpenAI-compatible inference servers and routers.

    Detection: ``GET /v1/models`` returns ``{"object": "list", "data": [...]}``
    (HTTP 200, 401, or 403).  A 401/403 is itself a detection signal.
    """

    engine_type = EngineType.OPENAI
    default_ports = (8000, 8001, 5000, 3000, 1234, 4000, 4891)
    priority = 90

    @classmethod
    def detect(cls, candidate: Candidate, client: httpx.Client) -> Optional[EngineInfo]:
        """Probe ``GET /v1/models`` and return an :class:`EngineInfo` on success.

        Returns ``None`` if the port does not respond or returns an unexpected
        response (e.g. 404 or a non-JSON body on HTTP 200).  HTTP 401/403
        counts as "detected but auth-blocked" and returns a partial
        ``EngineInfo`` with ``last_error`` set.
        """
        base = candidate.base_url
        try:
            r = client.get(f"{base}/v1/models")
        except Exception:
            return None

        engine_type = EngineType.OPENAI

        if r.status_code in (401, 403):
            engine = EngineInfo(
                engine_type=engine_type,
                name="OpenAI-compatible",
                base_url=base,
                host=candidate.host,
                port=candidate.port,
                pid=candidate.pid,
                process=candidate.process,
                signals=list(candidate.signals) + [f"v1/models→{r.status_code}"],
                last_error="requires API key (set LLMTOP_API_KEY to introspect)",
            )
            return engine

        if r.status_code != 200:
            return None

        try:
            data = r.json()
        except Exception:
            return None

        if not isinstance(data, dict) or data.get("object") != "list":
            return None

        engine = EngineInfo(
            engine_type=engine_type,
            name="OpenAI-compatible",
            base_url=base,
            host=candidate.host,
            port=candidate.port,
            pid=candidate.pid,
            process=candidate.process,
            signals=list(candidate.signals) + ["v1/models→200"],
        )
        return engine

    def describe(self, engine: EngineInfo, client: httpx.Client) -> None:
        """Populate model list from ``GET /v1/models``.

        Silently degrades: if the request fails or auth is blocked we leave
        ``engine.models`` empty and update ``last_error`` where appropriate.
        """
        base = engine.base_url
        try:
            r = client.get(f"{base}/v1/models")
        except Exception:
            return

        if r.status_code in (401, 403):
            engine.last_error = "requires API key (set LLMTOP_API_KEY to introspect)"
            return

        if r.status_code != 200:
            return

        try:
            data = r.json()
        except Exception:
            return

        items = data.get("data", []) if isinstance(data, dict) else []
        models: list[ModelInfo] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            model_id: str = str(item.get("id", "")).strip()
            if not model_id:
                continue
            ctx_raw = item.get("max_model_len") or item.get("context_length")
            ctx = int(ctx_raw) if isinstance(ctx_raw, (int, float)) else None
            models.append(ModelInfo(id=model_id, context_length=ctx))

        engine.models = models

        # Attempt to identify router engines: a large number of served models
        # often indicates a routing proxy.  The discovery layer does the
        # authoritative router correlation; here we just note it in flags.
        if len(models) > 3:
            engine.flags.setdefault("served_model_count", len(models))

    def metrics(
        self,
        engine: EngineInfo,
        client: httpx.Client,
        previous: Optional[EngineMetrics] = None,
        dt: Optional[float] = None,
    ) -> EngineMetrics:
        """Try ``GET /metrics`` for Prometheus data; return empty on failure.

        Most generic OpenAI-compatible servers do not expose a Prometheus
        ``/metrics`` endpoint.  A missing or non-200 endpoint is treated as
        "metrics not available" rather than an error.
        """
        base = engine.base_url
        try:
            r = client.get(f"{base}/metrics")
        except Exception:
            return EngineMetrics()

        if r.status_code != 200:
            return EngineMetrics()

        # Check that the body looks like Prometheus text (not JSON or HTML).
        text = r.text
        if not text or text.lstrip().startswith(("{", "[")):
            return EngineMetrics()

        try:
            parsed = parse_prometheus(text)
        except Exception:
            return EngineMetrics()

        if not parsed:
            return EngineMetrics()

        # Return whatever we can parse; accumulate raw for the detail pane.
        return EngineMetrics(raw=dict(parsed))
