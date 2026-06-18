"""Text Generation Inference (TGI) adapter.

Probes Hugging Face TGI servers (default ports 8080, 80, 3000) via ``GET /info``
which returns the model_id, max_total_tokens, and version. Prometheus metrics are
fetched from ``GET /metrics``.
"""

from __future__ import annotations

from typing import Optional

import httpx

from ..models import Candidate, EngineInfo, EngineMetrics, EngineType, ModelInfo
from .base import Adapter, derive_rate
from .prom import as_int, parse_prometheus, prom_value


class TGIAdapter(Adapter):
    """Adapter for Hugging Face Text Generation Inference (TGI)."""

    engine_type = EngineType.TGI
    default_ports = (8080, 80, 3000)
    priority = 20

    @classmethod
    def detect(cls, candidate: Candidate, client: httpx.Client) -> Optional[EngineInfo]:
        """Return an EngineInfo if the candidate looks like a TGI server.

        Detection: ``GET /info`` must return a JSON body with both ``model_id``
        and ``max_total_tokens`` keys. HTTP 401/403 is a positive signal.
        """
        base = candidate.base_url
        try:
            resp = client.get(f"{base}/info")
        except Exception:
            return None

        if resp.status_code in (401, 403) and candidate.hint == cls.engine_type:
            return EngineInfo(
                engine_type=EngineType.TGI,
                name="TGI",
                base_url=base,
                host=candidate.host,
                port=candidate.port,
                pid=candidate.pid,
                process=candidate.process,
                last_error="requires API key (set LLMTOP_API_KEY to introspect)",
            )

        if resp.status_code == 200:
            try:
                data = resp.json()
                if isinstance(data, dict) and "model_id" in data and "max_total_tokens" in data:
                    return EngineInfo(
                        engine_type=EngineType.TGI,
                        name="TGI",
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
        """Enrich engine with model info, context length, and version from ``/info``."""
        base = engine.base_url
        try:
            resp = client.get(f"{base}/info")
            if resp.status_code != 200:
                return
            data = resp.json()
        except Exception:
            return

        if not isinstance(data, dict):
            return

        model_id: str = data.get("model_id") or "unknown"
        version: Optional[str] = data.get("version")
        max_total_tokens: Optional[int] = as_int(data.get("max_total_tokens"))

        dtype: Optional[str] = data.get("dtype")
        quant_method: Optional[str] = data.get("quantize")

        engine.version = version
        engine.models = [
            ModelInfo(
                id=model_id,
                context_length=max_total_tokens,
                dtype=dtype,
                quantization=quant_method,
                loaded=True,
            )
        ]

        # Store notable flags from /info
        for key in ("max_batch_size", "max_best_of", "max_stop_sequences",
                    "waiting_served_ratio", "max_input_tokens", "max_total_tokens"):
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

        Key TGI series:
        - ``tgi_queue_size``                  -> requests_waiting
        - ``tgi_batch_current_size``          -> requests_running
        - ``tgi_request_generated_tokens``    -> tokens_total (counter)
        - ``tgi_request_input_length``        -> prompt_tokens_total (counter)
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

        def _get(name: str) -> Optional[float]:
            return prom_value(parsed, name)

        # Queue and batch sizes
        queue_size = as_int(_get("tgi_queue_size"))
        batch_size = as_int(_get("tgi_batch_current_size"))

        # Cumulative counters. TGI exposes a _sum suffix for histogram counters;
        # tgi_request_generated_tokens_sum is total tokens generated across requests.
        gen_tokens_total = as_int(
            _get("tgi_request_generated_tokens_sum")
            if _get("tgi_request_generated_tokens_sum") is not None
            else _get("tgi_request_generated_tokens")
        )
        # tgi_request_input_length_sum is total prompt tokens.
        prompt_tokens_total = as_int(
            _get("tgi_request_input_length_sum")
            if _get("tgi_request_input_length_sum") is not None
            else _get("tgi_request_input_length")
        )
        requests_total = as_int(
            _get("tgi_request_count")
            if _get("tgi_request_count") is not None
            else _get("tgi_batch_inference_count")
        )

        prev_gen = previous.tokens_total if previous else None
        prev_prompt = previous.prompt_tokens_total if previous else None

        return EngineMetrics(
            decode_tps=derive_rate(gen_tokens_total, prev_gen, dt),
            prefill_tps=derive_rate(prompt_tokens_total, prev_prompt, dt),
            requests_running=batch_size,
            requests_waiting=queue_size,
            tokens_total=gen_tokens_total,
            prompt_tokens_total=prompt_tokens_total,
            requests_total=requests_total,
            raw=dict(parsed),
        )
