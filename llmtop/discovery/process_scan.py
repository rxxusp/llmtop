"""Process scanner: find inference engine processes on the host.

Walks the process table via psutil, classifies command lines against known
patterns for each engine family, and collects the TCP ports each matching
process is listening on.
"""

from __future__ import annotations

import re
from typing import Optional

import psutil

from ..models import EngineType, ProcessInfo


# ---------------------------------------------------------------------------
# Command-line patterns per engine type.
# Each list entry is a substring OR a compiled regex to search against the
# space-joined command line of a process.  First match wins.
# ---------------------------------------------------------------------------

#: Patterns used to classify a process command line to a known EngineType.
CMDLINE_PATTERNS: dict[EngineType, list[str | re.Pattern[str]]] = {
    EngineType.VLLM: [
        "vllm serve",
        "vllm.entrypoints",
        "-m vllm",
        "python -m vllm",
        re.compile(r"\bvllm\b"),
    ],
    EngineType.LLAMACPP: [
        "llama-server",
        "llama.cpp",
        "llama_server",
        re.compile(r"\bllama[-_]?server\b"),
        re.compile(r"llama\.cpp"),
    ],
    EngineType.OLLAMA: [
        "ollama serve",
        "ollama runner",
        re.compile(r"\bollama\b"),
    ],
    EngineType.TGI: [
        "text-generation-launcher",
        "text_generation_server",
        "text_generation_launcher",
        "text-generation-server",
    ],
    EngineType.SGLANG: [
        "sglang.launch_server",
        "-m sglang",
        re.compile(r"\bsglang\b"),
    ],
    # Generic: uvicorn/python serving a model via --port and --model/--model-path.
    # Checked last among the specific types (lower priority wins via ordering).
    EngineType.OPENAI: [
        re.compile(r"\buvicorn\b.*--port\b"),
        re.compile(r"\bpython\b.*--port\b.*--model\b"),
        re.compile(r"\bpython\b.*--model-path\b.*--port\b"),
        re.compile(r"\bpython\b.*--port\b.*--model-path\b"),
    ],
}


def classify_cmdline(cmdline: list[str]) -> Optional[EngineType]:
    """Classify a process command line to the best-matching :class:`EngineType`.

    Matches are attempted in the order defined in ``CMDLINE_PATTERNS``.  The
    more-specific engines (VLLM, LLAMACPP, OLLAMA, TGI, SGLANG) are tried
    before the generic OPENAI pattern so that a vLLM process that also uses
    ``--port`` is not misclassified as generic.

    Parameters
    ----------
    cmdline:
        The raw argument list from ``psutil.Process.cmdline()``.

    Returns
    -------
    Optional[EngineType]
        The matched engine family, or ``None`` when nothing matched.
    """
    if not cmdline:
        return None

    joined = " ".join(cmdline)
    joined_lower = joined.lower()

    for engine_type, patterns in CMDLINE_PATTERNS.items():
        for pat in patterns:
            if isinstance(pat, re.Pattern):
                if pat.search(joined) or pat.search(joined_lower):
                    return engine_type
            else:
                if pat in joined or pat in joined_lower:
                    return engine_type

    return None


def _listening_ports_for_pid(pid: int) -> list[int]:
    """Return all TCP ports that *pid* is currently listening on.

    Uses ``psutil.net_connections`` with ``kind='inet'`` and filters to
    connections whose ``laddr.port`` is in LISTEN state and whose ``pid``
    matches.  Falls back to an empty list on any error (``AccessDenied``,
    ``NoSuchProcess``, permission issues on Linux without privilege, …).
    """
    ports: list[int] = []
    try:
        # net_connections(kind="inet") is preferred to Process.net_connections()
        # because the latter can raise NoSuchProcess mid-iteration.
        for conn in psutil.net_connections(kind="inet"):
            if conn.pid == pid and conn.status == psutil.CONN_LISTEN and conn.laddr:
                port = conn.laddr.port
                if port not in ports:
                    ports.append(port)
    except (psutil.AccessDenied, psutil.NoSuchProcess, PermissionError, OSError):
        pass
    return ports


def scan_processes() -> list[ProcessInfo]:
    """Scan the process table for inference engine processes.

    Returns one :class:`ProcessInfo` per process that looks like an inference
    engine (matched by :func:`classify_cmdline`).  Each result has:

    - ``pid``, ``name``, ``cmdline``, ``create_time`` — from psutil
    - ``cpu_percent`` — point-in-time sample (may be 0.0 on first call)
    - ``rss_bytes`` — resident system RAM
    - ``username`` — process owner
    - ``ports`` — TCP listening ports owned by this pid
    - ``hint`` — the :class:`EngineType` guessed from the command line

    Processes for which psutil raises ``AccessDenied`` or ``NoSuchProcess``
    are silently skipped.
    """
    results: list[ProcessInfo] = []

    try:
        procs = list(psutil.process_iter(
            attrs=["pid", "name", "cmdline", "create_time",
                   "cpu_percent", "memory_info", "username"],
        ))
    except Exception:  # psutil failure should not crash
        return results

    for proc in procs:
        try:
            info = proc.info
            cmdline: list[str] = info.get("cmdline") or []
            hint = classify_cmdline(cmdline)
            if hint is None:
                continue

            pid: int = info["pid"]
            name: Optional[str] = info.get("name")
            create_time: Optional[float] = info.get("create_time")
            cpu_pct: Optional[float] = info.get("cpu_percent")
            mem_info = info.get("memory_info")
            rss: Optional[int] = mem_info.rss if mem_info is not None else None
            username: Optional[str] = info.get("username")

            ports = _listening_ports_for_pid(pid)

            proc_info = ProcessInfo(
                pid=pid,
                name=name,
                cmdline=cmdline,
                create_time=create_time,
                cpu_percent=cpu_pct,
                rss_bytes=rss,
                gpu_mem_bytes=None,  # filled later by the monitor via NVML
                username=username,
                ports=ports,
                hint=hint,
            )
            results.append(proc_info)
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        except Exception:
            # Any unexpected failure: skip this process, never crash.
            continue

    return results
