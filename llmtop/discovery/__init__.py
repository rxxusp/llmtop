"""Discovery package for llmtop.

Public API re-exported from sub-modules:

- :func:`discover` — full autodiscovery pipeline (processes + ports + fingerprint)
- :func:`fingerprint` — fingerprint a single :class:`~llmtop.models.Candidate`
- :func:`scan_ports` — TCP-connect probe a list of ports
- :func:`scan_processes` — walk the process table for engine processes
- :func:`correlate_router` — post-process engines to identify routers
- :data:`DEFAULT_PORTS` — well-known inference-server port list
"""

from __future__ import annotations

from .discover import discover, correlate_router
from .fingerprint import fingerprint
from .port_scan import scan_ports, DEFAULT_PORTS
from .process_scan import scan_processes

__all__ = [
    "discover",
    "fingerprint",
    "scan_ports",
    "scan_processes",
    "correlate_router",
    "DEFAULT_PORTS",
]
