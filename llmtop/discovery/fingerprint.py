"""Fingerprint a candidate host:port into an EngineInfo.

Tries each registered adapter in priority order (see ``iter_detectors``).
The first adapter whose ``detect()`` returns a non-None result wins.  If none
match but the port responded to *any* HTTP request, falls back to the
``UnknownAdapter``.  If the port is TCP-open but HTTP-silent (connection
refused to HTTP, or all requests timed out), still returns an UNKNOWN
``EngineInfo`` so the caller has a record of the port.
"""

from __future__ import annotations

import httpx

from ..models import Candidate, EngineInfo, EngineType
from ..adapters import adapter_for, iter_detectors, UNKNOWN_ADAPTER


def fingerprint(candidate: Candidate, client: httpx.Client) -> EngineInfo:
    """Probe *candidate* with every registered adapter in priority order.

    Algorithm
    ---------
    1. Walk ``iter_detectors()`` (ascending ``priority``).  For each adapter
       class call ``cls.detect(candidate, client)``.  The first non-None result
       claims the candidate.
    2. If the winning adapter returned an ``EngineInfo``, call
       ``adapter_for(engine.engine_type).describe(engine, client)`` to enrich
       it with models, version, etc.
    3. If ALL adapters return ``None``, try the ``UNKNOWN_ADAPTER`` (which
       probes for any HTTP response).
    4. If even that returns ``None`` (port is TCP-open but HTTP-silent or
       timed out), construct a bare ``UNKNOWN`` ``EngineInfo`` and record
       the signal in ``last_error``.

    Parameters
    ----------
    candidate:
        The host:port (and any process context) to probe.
    client:
        The shared ``httpx.Client`` with pre-configured timeout and auth
        headers.  Adapters MUST use this client and MUST NOT block.

    Returns
    -------
    EngineInfo
        Always returns something; never raises.
    """
    engine: EngineInfo | None = None

    for adapter_cls in iter_detectors():
        try:
            result = adapter_cls.detect(candidate, client)
        except Exception as exc:
            # An adapter that raises is a bug, but we must not crash.
            result = None
            _ = exc  # available in debugger

        if result is not None:
            engine = result
            # Copy discovery signals from candidate if the adapter didn't.
            for sig in candidate.signals:
                if sig not in engine.signals:
                    engine.signals.append(sig)
            break

    if engine is None:
        # Try the unknown/catch-all adapter as an explicit fallback.
        try:
            engine = UNKNOWN_ADAPTER.detect(candidate, client)
        except Exception:
            engine = None

    if engine is None:
        # Port is TCP-open but we got nothing from HTTP.  Synthesise a record.
        engine = EngineInfo(
            engine_type=EngineType.UNKNOWN,
            name=f"unknown@{candidate.port}",
            base_url=candidate.base_url,
            host=candidate.host,
            port=candidate.port,
            pid=candidate.pid,
            process=candidate.process,
            signals=list(candidate.signals),
            last_error="TCP open but no HTTP response recognised",
        )
        return engine

    # Enrich with describe() using the correct adapter for the detected type.
    try:
        adapter_for(engine.engine_type).describe(engine, client)
    except Exception as exc:
        if engine.last_error is None:
            engine.last_error = f"describe failed: {exc}"

    return engine
