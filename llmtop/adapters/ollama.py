"""Ollama engine adapter.

Probes the Ollama REST API (default port 11434) to discover installed and currently
loaded models. Token-rate metrics are not exposed by Ollama; those fields are left
as None. ``requests_running`` is derived from the count of models loaded in VRAM via
``GET /api/ps``.
"""

from __future__ import annotations

from typing import Optional

import httpx

from ..models import Candidate, EngineInfo, EngineMetrics, EngineType, ModelInfo
from .base import Adapter


class OllamaAdapter(Adapter):
    """Adapter for Ollama (https://ollama.ai)."""

    engine_type = EngineType.OLLAMA
    default_ports = (11434,)
    priority = 5

    @classmethod
    def _engine(cls, candidate: Candidate) -> EngineInfo:
        return EngineInfo(
            engine_type=EngineType.OLLAMA,
            name="Ollama",
            base_url=candidate.base_url,
            host=candidate.host,
            port=candidate.port,
            pid=candidate.pid,
            process=candidate.process,
        )

    @classmethod
    def detect(cls, candidate: Candidate, client: httpx.Client) -> Optional[EngineInfo]:
        """Return an EngineInfo if the candidate is genuinely an Ollama server.

        The distinctive, low-false-positive signal is ``GET /api/tags`` returning
        a JSON object whose ``models`` field is a list. We deliberately do NOT
        claim on:
          - a bare HTTP 401/403 (every auth-gated server, e.g. an OpenAI router,
            returns that on every path) unless the process scan already told us
            this PID is Ollama (``candidate.hint``); and
          - a lone ``/api/version`` response, because other apps proxy it
            (Open WebUI returns ``{"version": ...}`` from ``/api/version`` while
            serving HTML everywhere else).
        """
        base = candidate.base_url
        hinted = candidate.hint == cls.engine_type

        try:
            resp = client.get(f"{base}/api/tags")
        except Exception:
            resp = None

        if resp is not None:
            if resp.status_code == 200:
                try:
                    data = resp.json()
                except Exception:
                    data = None
                if isinstance(data, dict) and isinstance(data.get("models"), list):
                    return cls._engine(candidate)
            # Auth-gated: only claim Ollama when the process scan agrees.
            if resp.status_code in (401, 403) and hinted:
                eng = cls._engine(candidate)
                eng.last_error = "requires API key (set LLMTOP_API_KEY to introspect)"
                return eng

        # Secondary signal: /api/version, trusted only when the process hint or the
        # canonical Ollama port agrees (keeps Open WebUI and similar proxies out).
        if hinted or candidate.port in cls.default_ports:
            try:
                v_resp = client.get(f"{base}/api/version")
                if v_resp.status_code == 200:
                    vdata = v_resp.json()
                    if isinstance(vdata, dict) and "version" in vdata:
                        return cls._engine(candidate)
            except Exception:
                pass

        return None

    def describe(self, engine: EngineInfo, client: httpx.Client) -> None:
        """Enrich engine with installed + loaded model information and version.

        ``GET /api/tags`` lists all installed models. ``GET /api/ps`` lists models
        currently loaded in VRAM. Models present only in tags are marked
        ``loaded=False``; models present in ps are ``loaded=True`` with vram/size
        populated.
        """
        base = engine.base_url

        # Fetch version
        try:
            v_resp = client.get(f"{base}/api/version")
            if v_resp.status_code == 200:
                vdata = v_resp.json()
                engine.version = vdata.get("version")
        except Exception:
            pass

        # Fetch tags (all installed models)
        tags_models: list[dict] = []
        try:
            t_resp = client.get(f"{base}/api/tags")
            if t_resp.status_code == 200:
                tdata = t_resp.json()
                tags_models = tdata.get("models", []) or []
        except Exception:
            pass

        # Fetch currently loaded models (/api/ps)
        ps_by_name: dict[str, dict] = {}
        try:
            ps_resp = client.get(f"{base}/api/ps")
            if ps_resp.status_code == 200:
                psdata = ps_resp.json()
                models = psdata.get("models") if isinstance(psdata, dict) else None
                for m in (models or []):
                    if not isinstance(m, dict):
                        continue
                    name = m.get("name") or m.get("model") or ""
                    if name:
                        ps_by_name[name] = m
        except Exception:
            pass

        model_list: list[ModelInfo] = []

        # Build ModelInfo for each installed model
        for tag in tags_models:
            if not isinstance(tag, dict):
                continue
            model_name: str = tag.get("name") or tag.get("model") or ""
            if not model_name:
                continue

            details = tag.get("details") or {}
            quant = details.get("quantization_level") or None
            family = details.get("family") or None
            size_bytes = tag.get("size")

            # Check if this model is in the ps (loaded in VRAM) response
            if model_name in ps_by_name:
                ps_entry = ps_by_name[model_name]
                ps_details = ps_entry.get("details") or {}
                ctx = None
                # ps entries may have context_length in details
                if "context_length" in ps_details:
                    try:
                        ctx = int(ps_details["context_length"])
                    except (TypeError, ValueError):
                        pass
                vram = ps_entry.get("size_vram")
                ps_size = ps_entry.get("size")
                model_list.append(
                    ModelInfo(
                        id=model_name,
                        quantization=quant,
                        family=family,
                        loaded=True,
                        size_bytes=ps_size if ps_size is not None else size_bytes,
                        vram_bytes=vram if vram is not None else None,
                        context_length=ctx,
                    )
                )
            else:
                model_list.append(
                    ModelInfo(
                        id=model_name,
                        quantization=quant,
                        family=family,
                        loaded=False,
                        size_bytes=size_bytes,
                    )
                )

        # Also add any ps models that were NOT in tags (edge case: recently loaded)
        tags_names = {
            (t.get("name") or t.get("model") or "")
            for t in tags_models
            if isinstance(t, dict)
        }
        for ps_name, ps_entry in ps_by_name.items():
            if ps_name not in tags_names:
                ps_details = ps_entry.get("details") or {}
                quant = ps_details.get("quantization_level") or None
                family = ps_details.get("family") or None
                vram = ps_entry.get("size_vram")
                ps_size = ps_entry.get("size")
                ctx = None
                if "context_length" in ps_details:
                    try:
                        ctx = int(ps_details["context_length"])
                    except (TypeError, ValueError):
                        pass
                model_list.append(
                    ModelInfo(
                        id=ps_name,
                        quantization=quant,
                        family=family,
                        loaded=True,
                        size_bytes=ps_size,
                        vram_bytes=vram,
                        context_length=ctx,
                    )
                )

        engine.models = model_list

    def metrics(
        self,
        engine: EngineInfo,
        client: httpx.Client,
        previous: Optional[EngineMetrics] = None,
        dt: Optional[float] = None,
    ) -> EngineMetrics:
        """Return serving metrics for Ollama.

        Ollama does not expose token-rate Prometheus metrics; decode_tps, prefill_tps,
        and kv_cache_pct are always None. ``requests_running`` is derived from the
        count of models currently loaded in ``/api/ps``.
        """
        base = engine.base_url
        try:
            ps_resp = client.get(f"{base}/api/ps")
            if ps_resp.status_code == 200:
                psdata = ps_resp.json()
                loaded = psdata.get("models") or []
                return EngineMetrics(
                    requests_running=len(loaded),
                    # Ollama has no token-rate or KV-cache metrics
                    decode_tps=None,
                    prefill_tps=None,
                    kv_cache_pct=None,
                )
            return EngineMetrics(error=f"GET /api/ps returned HTTP {ps_resp.status_code}")
        except Exception as exc:
            return EngineMetrics(error=str(exc))
