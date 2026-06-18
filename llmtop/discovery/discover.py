"""Top-level discovery orchestrator.

Ties together process scanning, port scanning, and HTTP fingerprinting into
a single ``discover()`` call that returns a list of ``EngineInfo`` objects
representing all inference engines found on localhost.

Also exports ``correlate_router``, which post-processes the list to identify
router/proxy engines and wire up their ``.backends`` relationships.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

import httpx

from ..models import Candidate, EngineInfo, EngineType
from .process_scan import scan_processes
from .port_scan import scan_ports, DEFAULT_PORTS
from .fingerprint import fingerprint


# ---------------------------------------------------------------------------
# Router correlation
# ---------------------------------------------------------------------------

def _model_ids(engine: EngineInfo) -> set[str]:
    """Return the set of model ids advertised by *engine*."""
    return {m.id for m in engine.models}


def _ports_from_urls(engines: list[EngineInfo]) -> dict[int, EngineInfo]:
    """Build a port → engine map for fast lookup."""
    return {e.port: e for e in engines}


def _is_router_by_process(engine: EngineInfo) -> bool:
    """Heuristic: does the process name/cmdline suggest a proxy/router?"""
    if engine.process is None:
        return False
    cmdline = " ".join(engine.process.cmdline).lower()
    name = (engine.process.name or "").lower()
    proxy_keywords = (
        "router", "proxy", "gateway", "litellm", "lm-router",
        "llm-router", "nginx", "caddy", "haproxy",
    )
    return any(kw in cmdline or kw in name for kw in proxy_keywords)


def _model_overlap_ratio(router_models: set[str], backend_models: set[str]) -> float:
    """Fraction of backend models that appear in the router model list."""
    if not backend_models:
        return 0.0
    overlap = router_models & backend_models
    return len(overlap) / len(backend_models)


def correlate_router(engines: list[EngineInfo]) -> list[EngineInfo]:
    """Identify router engines and wire up backend relationships.

    A router is an engine that either:

    - Has a process cmdline/name matching router/proxy keywords (highest
      confidence), **and** serves at least 2 models, OR
    - Is ``EngineType.OPENAI`` (generic) and its model list has significant
      overlap with models served by OTHER discovered engines (suggests it is
      re-exporting them), OR
    - Serves a large number of models (≥ 3) while other engines serve a
      subset of exactly those models.

    When an engine is identified as a router:
    - ``engine.is_router`` is set to ``True``
    - ``engine.engine_type`` is set to ``EngineType.ROUTER``
    - ``engine.backends`` is populated with the backend ``EngineInfo`` objects
    - Each backend's ``routed_by`` is set to the router's ``base_url``

    Parameters
    ----------
    engines:
        The raw list from fingerprinting.  Modified in-place; the same list
        (with updated fields) is returned.

    Returns
    -------
    list[EngineInfo]
        The input list with router relationships filled in.  Backends remain
        in the list (``routed_by`` is set but they are not removed), so
        callers that want only top-level engines can filter on ``routed_by``.
    """
    if len(engines) <= 1:
        return engines

    # Build model-id → [engine] index for all non-router engines.
    # We work in two passes: first classify routers, then assign backends.

    # --- Pass 1: identify candidates ---
    router_candidates: list[EngineInfo] = []
    backend_candidates: list[EngineInfo] = []

    for engine in engines:
        if engine.is_router or engine.engine_type == EngineType.ROUTER:
            # Already classified (e.g. by adapter).
            router_candidates.append(engine)
        else:
            backend_candidates.append(engine)

    # Build model-set for each non-router engine (may be empty if introspection
    # was blocked by auth, etc.).
    backend_model_sets: dict[str, set[str]] = {
        e.base_url: _model_ids(e) for e in backend_candidates
    }
    all_backend_models: set[str] = set()
    for s in backend_model_sets.values():
        all_backend_models.update(s)

    for engine in backend_candidates:
        my_models = _model_ids(engine)
        num_models = len(my_models)

        # Skip engines that have no models at all (auth-blocked, etc.)
        # unless they are identified by process heuristics.
        process_is_router = _is_router_by_process(engine)

        # Overlap with backend models from OTHER engines.
        others_union: set[str] = set()
        for url, s in backend_model_sets.items():
            if url != engine.base_url:
                others_union.update(s)

        overlap = len(my_models & others_union)
        if others_union:
            overlap_ratio = overlap / len(others_union)
        else:
            overlap_ratio = 0.0

        # Heuristics:
        # 1. Process cmdline says "router/proxy" and engine has ≥ 2 models.
        # 2. Engine has ≥ 3 models and a meaningful fraction (>0.4) overlap
        #    with other engines' model names — it re-exports them.
        # 3. Engine is generic OpenAI type, has ≥ 3 models, and all other
        #    non-generic engines' models appear inside its list.
        is_router = False

        if process_is_router and num_models >= 1:
            is_router = True
        elif num_models >= 3 and overlap_ratio >= 0.4 and len(others_union) >= 1:
            is_router = True
        elif (engine.engine_type == EngineType.OPENAI
              and num_models >= 3
              and overlap_ratio >= 0.5):
            is_router = True

        if is_router:
            router_candidates.append(engine)
            backend_candidates = [e for e in backend_candidates
                                   if e.base_url != engine.base_url]
            # Rebuild model sets without the new router.
            backend_model_sets = {
                e.base_url: _model_ids(e)
                for e in backend_candidates
            }
            all_backend_models = set()
            for s in backend_model_sets.values():
                all_backend_models.update(s)

    # --- Pass 2: assign backends to each router ---
    for router in router_candidates:
        router.is_router = True
        router.engine_type = EngineType.ROUTER
        router.backends = []

        router_models = _model_ids(router)

        for backend in backend_candidates:
            # A backend is linked to this router if:
            # (a) the backend's models appear in the router's list (router
            #     re-exports the backend), OR
            # (b) the process-is-router heuristic fired and there are no
            #     model overlaps to compare (auth-blocked router, etc.), OR
            # (c) there is any single model ID match.
            b_models = _model_ids(backend)
            if b_models:
                shared = router_models & b_models
                if shared or _model_overlap_ratio(router_models, b_models) >= 0.5:
                    router.backends.append(backend)
                    if backend.routed_by is None:
                        backend.routed_by = router.base_url
            elif _is_router_by_process(router):
                # Router with no model overlap info — link all backends
                # tentatively (router is the single proxy for everything).
                router.backends.append(backend)
                if backend.routed_by is None:
                    backend.routed_by = router.base_url

    return engines


# ---------------------------------------------------------------------------
# discover()
# ---------------------------------------------------------------------------

def discover(
    client: Optional[httpx.Client] = None,
    *,
    extra_ports: Iterable[int] = (),
    include_system_ports: bool = True,
    timeout: float = 1.0,
) -> list[EngineInfo]:
    """Discover all inference engines currently running on localhost.

    Steps
    -----
    1. Scan the process table (``scan_processes``) for inference engine
       processes, collecting their listening ports and engine-type hints.
    2. Build the candidate port set: ``DEFAULT_PORTS`` ∪ process-owned ports
       ∪ *extra_ports*.  Probe each with a fast TCP connect (``scan_ports``).
    3. For each open port, build a :class:`Candidate` with process context
       (if a matching process owns that port).
    4. Fingerprint each candidate (``fingerprint``), deduplicated by
       ``host:port``.
    5. Post-process with ``correlate_router`` to identify proxy/router engines
       and wire up their ``.backends`` relationships.

    Authentication
    --------------
    When ``LLMTOP_API_KEY`` or ``OPENAI_API_KEY`` is set the ``Authorization:
    Bearer <key>`` header is attached to the shared client so that auth-
    protected endpoints (like the llm-router on :8077) are introspectable.
    ``LLMTOP_API_KEY`` takes precedence.

    Parameters
    ----------
    client:
        An existing ``httpx.Client`` to reuse.  If ``None`` a new client is
        created with *timeout* and ``trust_env=False``.
    extra_ports:
        Additional ports to include in the scan (e.g. from ``--port`` CLI
        flags).
    include_system_ports:
        If ``True`` (default) the ``DEFAULT_PORTS`` list is included.  Set to
        ``False`` to scan *only* process-discovered and extra ports.
    timeout:
        Per-request HTTP timeout in seconds when creating a new client.

    Returns
    -------
    list[EngineInfo]
        All discovered engines, routers first (with ``is_router=True`` and
        populated ``.backends``), followed by standalone engines.  Backends
        remain in the list with ``routed_by`` set.
    """
    _own_client = False
    if client is None:
        headers: dict[str, str] = {}
        api_key = os.environ.get("LLMTOP_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        client = httpx.Client(
            timeout=timeout,
            trust_env=False,
            headers=headers,
        )
        _own_client = True

    try:
        return _discover_impl(client, extra_ports, include_system_ports)
    finally:
        if _own_client:
            try:
                client.close()
            except Exception:
                pass


def _discover_impl(
    client: httpx.Client,
    extra_ports: Iterable[int],
    include_system_ports: bool,
) -> list[EngineInfo]:
    """Internal implementation; assumes *client* is already configured."""

    # --- Step 1: process scan ---
    processes = scan_processes()

    # Build a port → ProcessInfo map (one process per port; if two processes
    # share a port we keep the first match found).
    port_to_process: dict[int, tuple] = {}  # port -> (ProcessInfo, hint)
    for proc in processes:
        hint = proc.hint
        for port in proc.ports:
            if port not in port_to_process:
                port_to_process[port] = (proc, hint)

    # --- Step 2: build candidate port set and scan ---
    candidate_ports: set[int] = set()
    if include_system_ports:
        candidate_ports.update(DEFAULT_PORTS)
    candidate_ports.update(port_to_process.keys())
    for p in extra_ports:
        candidate_ports.add(p)

    open_ports = scan_ports(candidate_ports)

    # --- Step 3: build Candidate objects ---
    seen_keys: set[str] = set()
    candidates: list[Candidate] = []
    host = "127.0.0.1"

    for port in open_ports:
        key = f"{host}:{port}"
        if key in seen_keys:
            continue
        seen_keys.add(key)

        proc_info, hint = port_to_process.get(port, (None, None))

        signals: list[str] = ["port-scan"]
        if proc_info is not None:
            signals.append("process-scan")

        candidates.append(
            Candidate(
                host=host,
                port=port,
                pid=proc_info.pid if proc_info is not None else None,
                process=proc_info,
                hint=hint,
                signals=signals,
            )
        )

    # --- Step 4: fingerprint each candidate ---
    engines: list[EngineInfo] = []
    for cand in candidates:
        try:
            eng = fingerprint(cand, client)
        except Exception as exc:
            # Should never happen (fingerprint catches internally), but belt+suspenders.
            eng = EngineInfo(
                engine_type=EngineType.UNKNOWN,
                name=f"unknown@{cand.port}",
                base_url=cand.base_url,
                host=cand.host,
                port=cand.port,
                pid=cand.pid,
                process=cand.process,
                signals=list(cand.signals),
                last_error=f"fingerprint raised: {exc}",
            )
        engines.append(eng)

    # --- Step 5: router correlation ---
    engines = correlate_router(engines)

    # Sort: routers first, then by port ascending.
    engines.sort(key=lambda e: (0 if e.is_router else 1, e.port))

    return engines
