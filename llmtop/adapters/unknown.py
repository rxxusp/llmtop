"""Unknown / catch-all adapter.

Used by the fingerprinter as an explicit last resort when no other adapter
claims a candidate.  It always "detects" successfully — returning an
:class:`EngineInfo` with ``engine_type=UNKNOWN`` — so that even unrecognised
HTTP servers appear in the TUI rather than silently vanishing.

Information collected:
- The HTTP ``Server`` response header (if present).
- The HTTP status code of the root (``/``) path probe.
- Optionally the ``Content-Type`` of the root response.

Metrics: always empty (``EngineMetrics()``).
Describe: records whatever the ``/`` root probe returned in ``flags``.
"""

from __future__ import annotations

from typing import Optional

import httpx

from ..models import Candidate, EngineInfo, EngineMetrics, EngineType
from .base import Adapter


class UnknownAdapter(Adapter):
    """Catch-all adapter that always succeeds detection.

    This adapter MUST be probed last (``priority=100``).  The fingerprinter
    uses it explicitly as the fallback rather than including it in the normal
    detection loop, so that more specific adapters always get their chance
    first.
    """

    engine_type = EngineType.UNKNOWN
    default_ports: tuple[int, ...] = ()
    priority = 100

    @classmethod
    def detect(cls, candidate: Candidate, client: httpx.Client) -> Optional[EngineInfo]:
        """Always succeeds: return an UNKNOWN :class:`EngineInfo` for the candidate.

        Probes the root path ``/`` to capture the server header and status.
        If even the root probe raises (e.g. connection refused), we still
        return a minimal ``EngineInfo`` so the port is recorded.
        """
        base = candidate.base_url
        server_header: Optional[str] = None
        status_note: Optional[str] = None

        try:
            r = client.get(f"{base}/")
            server_header = r.headers.get("server") or r.headers.get("Server")
            status_note = f"HTTP {r.status_code}"
        except httpx.ConnectError:
            status_note = "connection refused"
        except httpx.TimeoutException:
            status_note = "timeout"
        except Exception as exc:
            status_note = f"probe error: {type(exc).__name__}"

        name = "Unknown"
        if server_header:
            # Use the first token of the Server header as the display name.
            name = server_header.split("/")[0].strip() or "Unknown"

        engine = EngineInfo(
            engine_type=EngineType.UNKNOWN,
            name=name,
            base_url=base,
            host=candidate.host,
            port=candidate.port,
            pid=candidate.pid,
            process=candidate.process,
            signals=list(candidate.signals) + ["unknown-fallback"],
            last_error=status_note,
        )

        if server_header:
            engine.flags["server_header"] = server_header
        if status_note:
            engine.flags["root_status"] = status_note

        return engine

    def describe(self, engine: EngineInfo, client: httpx.Client) -> None:
        """Attempt to refresh the server header / root status in ``engine.flags``.

        Any failure is silently swallowed; this adapter must never raise.
        """
        base = engine.base_url
        try:
            r = client.get(f"{base}/")
            server = r.headers.get("server") or r.headers.get("Server")
            if server:
                engine.flags["server_header"] = server
            engine.flags["root_status"] = f"HTTP {r.status_code}"
        except Exception:
            pass

    def metrics(
        self,
        engine: EngineInfo,
        client: httpx.Client,
        previous: Optional[EngineMetrics] = None,
        dt: Optional[float] = None,
    ) -> EngineMetrics:
        """Return empty metrics; unknown engines have no parseable metrics endpoint."""
        return EngineMetrics()
