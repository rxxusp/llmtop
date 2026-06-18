"""TCP port scanner: determine which ports on localhost are accepting connections.

Uses non-blocking ``socket.connect_ex`` with an explicit timeout, so the
scanner is completely safe to call from the UI thread (though ``discover.py``
runs it in a thread anyway).  No raw sockets, no root required.
"""

from __future__ import annotations

import socket
from typing import Iterable


#: Well-known default ports tried on every scan.  Covers vLLM (8000, 8001,
#: 8088), llama.cpp (8080, 8081), Ollama (11434), TGI (80, 3000, 8080),
#: SGLang (30000), common routers (8077, 8086, 8099), LM Studio (1234),
#: GPT4All (4891), Open WebUI (8080 also covered above), and generic extras.
DEFAULT_PORTS: tuple[int, ...] = (
    8000,   # vLLM default
    8001,   # vLLM secondary / test
    8077,   # llm-router
    8080,   # llama.cpp / TGI / Open WebUI
    8081,   # llama.cpp secondary
    8086,   # diffusiongemma / secondary router
    8088,   # vLLM qwen36-coder
    8099,   # ds4 / antirez llama.cpp
    5000,   # generic Flask / older servers
    3000,   # TGI / generic
    11434,  # Ollama
    30000,  # SGLang
    1234,   # LM Studio
    4000,   # generic
    4891,   # GPT4All
    7860,   # Gradio
    8888,   # Jupyter / generic
    9090,   # Prometheus / generic
    80,     # TGI production / nginx proxy
    443,    # HTTPS (useful to detect if port is up, adapter handles TLS miss)
)


def scan_ports(
    ports: Iterable[int],
    host: str = "127.0.0.1",
    timeout: float = 0.15,
) -> list[int]:
    """Return the subset of *ports* on *host* that accept a TCP connection.

    Parameters
    ----------
    ports:
        Iterable of port numbers to probe.  Duplicates are automatically
        de-duplicated; order of the returned list is the order that open
        ports were encountered.
    host:
        The IP address to probe.  Defaults to ``"127.0.0.1"`` (loopback).
        ``"0.0.0.0"`` and ``"::"`` are silently rewritten to ``"127.0.0.1"``.
    timeout:
        Per-port connect timeout in seconds.  Keep this short (0.15 s default)
        so scanning 20+ ports finishes in well under a second on loopback.

    Returns
    -------
    list[int]
        Open ports in the order they were found.
    """
    if host in ("0.0.0.0", "::", "*"):
        host = "127.0.0.1"

    seen: set[int] = set()
    open_ports: list[int] = []

    for port in ports:
        if port in seen:
            continue
        seen.add(port)

        try:
            # AF_INET works for IPv4 loopback; AF_INET6 would be needed for
            # pure IPv6 listeners, but in practice LLM servers bind 0.0.0.0.
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(timeout)
                rc = sock.connect_ex((host, port))
                if rc == 0:
                    open_ports.append(port)
        except OSError:
            # Port invalid, address unreachable, etc. — skip silently.
            continue

    return open_ports
