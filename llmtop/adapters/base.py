"""Engine adapter interface.

An adapter teaches llmtop how to recognize one family of inference server and how
to read its models, config, and live metrics. Adding support for a new engine is
exactly: subclass :class:`Adapter`, implement ``detect`` / ``describe`` /
``metrics``, and register it in ``llmtop/adapters/__init__.py``.

Contract / safety rules every adapter MUST obey:

- READ-ONLY. Only issue cheap, idempotent introspection/metrics requests
  (``GET /v1/models``, ``GET /metrics``, ``GET /api/tags`` ...). NEVER call a
  completion/generation endpoint or anything that consumes tokens or mutates
  server state.
- TIME-BOUNDED. Use the provided ``client`` (it carries a short timeout). Never
  block indefinitely. Catch your own network/parse errors and degrade to None /
  ``EngineMetrics(error=...)`` rather than raising out of ``metrics``.
- STATELESS. Adapters keep no cross-poll state. The monitor owns history and
  hands the previous :class:`EngineMetrics` to ``metrics`` so rates can be
  derived from counter deltas.
"""

from __future__ import annotations

import abc
from typing import Optional

import httpx

from ..models import Candidate, EngineInfo, EngineMetrics, EngineType


class Adapter(abc.ABC):
    """Base class for all engine adapters."""

    #: The engine family this adapter handles.
    engine_type: EngineType = EngineType.UNKNOWN

    #: Ports commonly used by this engine, fed into the port-scan hint list.
    default_ports: tuple[int, ...] = ()

    #: Lower runs earlier during fingerprinting. Specific engines (vLLM, Ollama,
    #: llama.cpp, TGI, SGLang) should be < generic OpenAI (which is < unknown).
    priority: int = 100

    @classmethod
    @abc.abstractmethod
    def detect(cls, candidate: Candidate, client: httpx.Client) -> Optional[EngineInfo]:
        """Probe ``candidate`` read-only. Return a populated :class:`EngineInfo`
        if this adapter recognizes the server, else ``None``.

        Implementations should set at minimum ``engine_type``, ``name``,
        ``base_url``, ``host``, ``port`` and copy ``pid``/``process`` from the
        candidate. ``describe`` will enrich further. Must not raise on a normal
        miss or network error: return ``None`` instead.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def describe(self, engine: EngineInfo, client: httpx.Client) -> None:
        """Enrich ``engine`` in place with models, version, context length, dtype/
        quantization, and notable flags. Called once after detection (and may be
        called again on refresh). Tolerate partial data; never raise."""
        raise NotImplementedError

    @abc.abstractmethod
    def metrics(
        self,
        engine: EngineInfo,
        client: httpx.Client,
        previous: Optional[EngineMetrics] = None,
        dt: Optional[float] = None,
    ) -> EngineMetrics:
        """Return current serving metrics for ``engine``.

        ``previous`` is the metrics object from the prior poll for this same
        engine (or None on the first poll); ``dt`` is the elapsed seconds between
        them. Use them to convert cumulative token counters into ``decode_tps`` /
        ``prefill_tps``. Always return an :class:`EngineMetrics`; on failure
        return ``EngineMetrics(error=str(exc))`` rather than raising.
        """
        raise NotImplementedError


def derive_rate(
    current_total: Optional[int],
    previous_total: Optional[int],
    dt: Optional[float],
) -> Optional[float]:
    """Helper: per-second rate from two cumulative counter readings.

    Returns None when inputs are missing, dt is non-positive, or the counter went
    backwards (engine restart / counter reset) so callers never show a negative
    or bogus spike.
    """
    if current_total is None or previous_total is None or not dt or dt <= 0:
        return None
    delta = current_total - previous_total
    if delta < 0:
        return None
    return delta / dt
