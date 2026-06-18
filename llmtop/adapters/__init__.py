"""Adapter registry.

Detection order matters: more specific engines are probed before generic
OpenAI-compatible, which is probed before the catch-all unknown adapter. Order is
controlled by each adapter's ``priority`` (ascending).

To add an engine: create ``llmtop/adapters/<name>.py`` with an ``Adapter``
subclass, import it here, and add it to ``_DETECTORS``. That is the entire
extension path.
"""

from __future__ import annotations

from ..models import EngineType
from .base import Adapter, derive_rate
from .vllm import VLLMAdapter
from .ollama import OllamaAdapter
from .llamacpp import LlamaCppAdapter
from .tgi import TGIAdapter
from .sglang import SGLangAdapter
from .openai_generic import OpenAIGenericAdapter
from .unknown import UnknownAdapter

# Specific detectors, tried in ascending priority. UnknownAdapter is NOT here; it
# is the explicit fallback used by the fingerprinter only when all of these miss.
_DETECTORS: list[type[Adapter]] = sorted(
    [
        VLLMAdapter,
        OllamaAdapter,
        LlamaCppAdapter,
        TGIAdapter,
        SGLangAdapter,
        OpenAIGenericAdapter,
    ],
    key=lambda a: a.priority,
)

UNKNOWN_ADAPTER = UnknownAdapter

# One stateless instance per engine type, for describe()/metrics() dispatch.
_INSTANCES: dict[EngineType, Adapter] = {a.engine_type: a() for a in _DETECTORS}
_INSTANCES[EngineType.UNKNOWN] = UnknownAdapter()
# A router is OpenAI-compatible on the wire; reuse the generic adapter for it.
_INSTANCES.setdefault(EngineType.ROUTER, OpenAIGenericAdapter())


def iter_detectors() -> list[type[Adapter]]:
    """Adapter classes in detection order (excludes the unknown fallback)."""
    return list(_DETECTORS)


def adapter_for(engine_type: EngineType) -> Adapter:
    """Return the stateless adapter instance that handles ``engine_type``."""
    return _INSTANCES.get(engine_type, _INSTANCES[EngineType.UNKNOWN])


__all__ = [
    "Adapter",
    "derive_rate",
    "VLLMAdapter",
    "OllamaAdapter",
    "LlamaCppAdapter",
    "TGIAdapter",
    "SGLangAdapter",
    "OpenAIGenericAdapter",
    "UnknownAdapter",
    "UNKNOWN_ADAPTER",
    "iter_detectors",
    "adapter_for",
]
