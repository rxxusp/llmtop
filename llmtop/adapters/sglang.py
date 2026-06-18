"""SGLang engine adapter.

Probes SGLang servers (default port 30000) via ``GET /get_model_info`` (which returns
a ``model_path`` key) and/or ``GET /health``. Prometheus metrics are fetched from
``GET /metrics``.
"""

from __future__ import annotations

from typing import Optional

import httpx

from ..models import Candidate, EngineInfo, EngineMetrics, EngineType, ModelInfo
from .base import Adapter
from .prom import as_float, as_int, parse_prometheus, prom_value


def _model_id_from_path(path: str) -> str:
    """Derive a short model id from a HF-style path or local filesystem path."""
    if not path:
        return "unknown"
    # HF model paths look like "Qwen/Qwen3-8B"; take the last component
    return path.rstrip("/").rsplit("/", 1)[-1] or path


class SGLangAdapter(Adapter):
    """Adapter for SGLang (https://github.com/sgl-project/sglang)."""

    engine_type = EngineType.SGLANG
    default_ports = (30000,)
    priority = 15

    @classmethod
    def detect(cls, candidate: Candidate, client: httpx.Client) -> Optional[EngineInfo]:
        """Return an EngineInfo if the candidate looks like an SGLang server.

        Detection: ``GET /get_model_info`` must return a JSON body containing a
        ``model_path`` key. Falls back to ``GET /health`` for a simple liveness
        check if model_info is not available. HTTP 401/403 is a positive signal.
        """
        base = candidate.base_url

        # Primary probe: /get_model_info
        try:
            mi_resp = client.get(f"{base}/get_model_info")
        except Exception:
            mi_resp = None

        if mi_resp is not None:
            if mi_resp.status_code in (401, 403) and candidate.hint == cls.engine_type:
                return EngineInfo(
                    engine_type=EngineType.SGLANG,
                    name="SGLang",
                    base_url=base,
                    host=candidate.host,
                    port=candidate.port,
                    pid=candidate.pid,
                    process=candidate.process,
                    last_error="requires API key (set LLMTOP_API_KEY to introspect)",
                )
            if mi_resp.status_code == 200:
                try:
                    data = mi_resp.json()
                    if isinstance(data, dict) and "model_path" in data:
                        return EngineInfo(
                            engine_type=EngineType.SGLANG,
                            name="SGLang",
                            base_url=base,
                            host=candidate.host,
                            port=candidate.port,
                            pid=candidate.pid,
                            process=candidate.process,
                        )
                except Exception:
                    pass

        # Fallback: /health
        try:
            h_resp = client.get(f"{base}/health")
        except Exception:
            return None

        if h_resp.status_code in (401, 403) and candidate.hint == cls.engine_type:
            return EngineInfo(
                engine_type=EngineType.SGLANG,
                name="SGLang",
                base_url=base,
                host=candidate.host,
                port=candidate.port,
                pid=candidate.pid,
                process=candidate.process,
                last_error="requires API key (set LLMTOP_API_KEY to introspect)",
            )

        # SGLang's /health returns 200 with an empty body or "{'status': 'ok'}"
        if h_resp.status_code == 200:
            # Only claim this as SGLang if we're on the default SGLang port or hint matches
            if candidate.port == 30000 or candidate.hint == EngineType.SGLANG:
                return EngineInfo(
                    engine_type=EngineType.SGLANG,
                    name="SGLang",
                    base_url=base,
                    host=candidate.host,
                    port=candidate.port,
                    pid=candidate.pid,
                    process=candidate.process,
                )

        return None

    def describe(self, engine: EngineInfo, client: httpx.Client) -> None:
        """Enrich engine with model info from ``GET /get_model_info``."""
        base = engine.base_url
        try:
            resp = client.get(f"{base}/get_model_info")
            if resp.status_code != 200:
                return
            data = resp.json()
        except Exception:
            return

        if not isinstance(data, dict):
            return

        model_path: str = data.get("model_path") or ""
        model_id = _model_id_from_path(model_path) if model_path else "unknown"

        # SGLang may also expose context_len, dtype, etc.
        context_len: Optional[int] = as_int(
            data.get("context_length")
            or data.get("max_total_num_tokens")
            or data.get("context_len")
        )

        dtype: Optional[str] = data.get("dtype")

        engine.models = [
            ModelInfo(
                id=model_id,
                context_length=context_len,
                dtype=dtype,
                loaded=True,
                extra={"model_path": model_path} if model_path else {},
            )
        ]

        if model_path:
            engine.flags["model_path"] = model_path
        for key in ("is_generation", "preferred_sampling_params", "max_total_num_tokens"):
            if key in data:
                engine.flags[key] = data[key]

    def metrics(
        self,
        engine: EngineInfo,
        client: httpx.Client,
        previous: Optional[EngineMetrics] = None,
        dt: Optional[float] = None,
    ) -> EngineMetrics:
        """Return serving metrics from ``GET /metrics`` (Prometheus text format).

        Key SGLang series:
        - ``sglang:num_running_reqs``   -> requests_running
        - ``sglang:num_queue_reqs``     -> requests_waiting
        - ``sglang:gen_throughput``     -> decode_tps (instantaneous rate from server)
        - ``sglang:token_usage``        -> kv_cache_pct (0-1 -> *100)
        - ``sglang:cache_hit_rate``     -> stored in raw for detail pane
        """
        base = engine.base_url
        try:
            m_resp = client.get(f"{base}/metrics")
        except Exception as exc:
            return EngineMetrics(error=str(exc))

        if m_resp.status_code == 404:
            return EngineMetrics(error="metrics endpoint not available")

        if m_resp.status_code != 200:
            return EngineMetrics(error=f"GET /metrics returned HTTP {m_resp.status_code}")

        parsed = parse_prometheus(m_resp.text)

        running = as_int(prom_value(parsed, "sglang:num_running_reqs"))
        waiting = as_int(prom_value(parsed, "sglang:num_queue_reqs"))

        # sglang:gen_throughput is an instantaneous tok/s reported by the server;
        # use it directly rather than deriving from cumulative counters.
        gen_throughput = as_float(prom_value(parsed, "sglang:gen_throughput"))

        token_usage = as_float(prom_value(parsed, "sglang:token_usage"))
        kv_pct = (token_usage * 100.0) if token_usage is not None else None

        return EngineMetrics(
            decode_tps=gen_throughput,
            requests_running=running,
            requests_waiting=waiting,
            kv_cache_pct=kv_pct,
            raw=dict(parsed),
        )
