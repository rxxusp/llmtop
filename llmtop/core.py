"""Monitor: the central poll-and-deliver loop for llmtop.

Owns hardware samplers, the httpx client, the engine discovery cache, per-engine
metrics history, VRAM attribution, and snapshot diffing to produce event strings.

Usage::

    monitor = Monitor(interval=2.0)
    try:
        await monitor.run(callback=my_async_fn)
    finally:
        monitor.close()

Or for a single synchronous snapshot (e.g. ``--json`` mode)::

    snap = monitor.poll()
    monitor.close()
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from typing import Any, Awaitable, Callable, Iterable, Optional

import httpx

from .adapters import adapter_for
from .discovery import discover
from .discovery.process_scan import classify_cmdline
from .gpu import GpuSampler
from .models import (
    EngineInfo,
    EngineMetrics,
    EngineType,
    GpuSample,
    ProcessInfo,
    Snapshot,
)
from .system import SystemSampler

log = logging.getLogger(__name__)

# Maximum events kept in the internal ring before the TUI drains them.
_MAX_EVENT_RING = 200


def _build_client(timeout: float) -> httpx.Client:
    """Create the shared httpx.Client with an optional API key header."""
    headers: dict[str, str] = {}
    api_key = os.environ.get("LLMTOP_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return httpx.Client(
        timeout=timeout,
        trust_env=False,
        headers=headers,
    )


class _EngineState:
    """Per-engine mutable bookkeeping held inside Monitor."""

    __slots__ = ("engine", "prev_metrics", "prev_time", "seen_models")

    def __init__(self, engine: EngineInfo) -> None:
        self.engine: EngineInfo = engine
        self.prev_metrics: Optional[EngineMetrics] = None
        self.prev_time: Optional[float] = None
        self.seen_models: set[str] = {m.id for m in engine.models}


class Monitor:
    """Central monitor: discovers engines, polls GPU/system/metrics, emits Snapshots.

    Parameters
    ----------
    interval:
        Seconds between polls (passed through to :py:meth:`run`).
    timeout:
        HTTP timeout in seconds for all engine/metrics requests.
    rediscover_every:
        Full re-discovery interval in seconds.  Between re-discoveries the known
        engines have their metrics refreshed but no new port/process scan runs.
    extra_ports:
        Additional ports to include in every discovery scan.
    enable_gpu:
        Set False to skip NVML initialisation (useful in CI or GPU-less hosts).
    """

    def __init__(
        self,
        *,
        interval: float = 2.0,
        timeout: float = 1.0,
        rediscover_every: float = 8.0,
        extra_ports: Iterable[int] = (),
        enable_gpu: bool = True,
    ) -> None:
        self._interval = interval
        self._timeout = timeout
        self._rediscover_every = rediscover_every
        self._extra_ports: tuple[int, ...] = tuple(extra_ports)

        # Hardware samplers.
        self._gpu: GpuSampler = GpuSampler() if enable_gpu else _NullGpuSampler()  # type: ignore[assignment]
        self._sys = SystemSampler()

        # Shared HTTP client (short-timeout, no env proxy injection).
        self._client = _build_client(timeout)

        # Discovery cache: keyed by EngineInfo.key (base_url).
        self._states: dict[str, _EngineState] = {}

        # When we last ran a full discovery.
        self._last_discover: Optional[float] = None

        # Previous snapshot (for diffing).
        self._prev_snapshot: Optional[Snapshot] = None

        # Event ring: new events are appended here; each call to poll() drains
        # them into the returned Snapshot.events so nothing is lost across polls.
        self._event_ring: deque[str] = deque(maxlen=_MAX_EVENT_RING)

        # Closed flag.
        self._closed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def poll(self) -> Snapshot:
        """Build and return one synchronous :class:`~llmtop.models.Snapshot`.

        Never raises.  All per-source failures are collected into
        ``Snapshot.errors``.
        """
        ts = time.time()
        errors: list[str] = []

        # 1. Hardware.
        gpus = self._sample_gpus(errors)
        system = self._sample_system(errors)

        # 2. Discovery / metrics.
        need_discover = (
            self._last_discover is None
            or (ts - self._last_discover) >= self._rediscover_every
        )
        if need_discover:
            self._run_discovery(ts, errors)
        else:
            self._refresh_metrics(ts, errors)

        # 3. VRAM attribution.
        self._attribute_vram(gpus)

        # 4. Collect engine list (flat, same order as discovery order).
        engines = [st.engine for st in self._states.values()]

        # 5. Diff vs previous snapshot → events.
        events = self._drain_events()
        if self._prev_snapshot is not None:
            new_events = self._diff(self._prev_snapshot, engines)
            events.extend(new_events)
            for e in new_events:
                log.debug("event: %s", e)

        # 6. Build snapshot.
        snap = Snapshot(
            timestamp=ts,
            gpus=gpus,
            system=system,
            engines=engines,
            events=events,
            errors=errors,
        )
        self._prev_snapshot = snap
        return snap

    async def run(
        self,
        callback: Callable[[Snapshot], Awaitable[None] | None],
    ) -> None:
        """Poll on a fixed interval, calling *callback* with each Snapshot.

        Runs until cancelled.  Blocking work (poll) runs in a thread so the
        asyncio event loop remains responsive.
        """
        while True:
            start = asyncio.get_running_loop().time()

            try:
                snap = await asyncio.to_thread(self.poll)
            except Exception as exc:  # pragma: no cover
                log.exception("Unexpected error in poll thread: %s", exc)
                # Build a minimal error snapshot so the TUI/callback always gets
                # something rather than silently stalling.
                snap = Snapshot(
                    timestamp=time.time(),
                    errors=[f"poll thread raised: {exc}"],
                )

            try:
                result = callback(snap)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:  # pragma: no cover
                log.exception("Callback raised: %s", exc)

            elapsed = asyncio.get_running_loop().time() - start
            sleep_for = max(0.0, self._interval - elapsed)
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

    def force_rediscover(self) -> None:
        """Force the next :meth:`poll` to run a full process+port rescan instead
        of only refreshing metrics on known engines."""
        self._last_discover = None

    def close(self) -> None:
        """Release resources (httpx client, NVML).  Idempotent."""
        if self._closed:
            return
        self._closed = True
        try:
            self._client.close()
        except Exception:
            pass
        try:
            self._gpu.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_gpus(self, errors: list[str]) -> list[GpuSample]:
        try:
            return self._gpu.sample()
        except Exception as exc:
            errors.append(f"GPU sample error: {exc}")
            return []

    def _sample_system(self, errors: list[str]) -> Any:
        try:
            return self._sys.sample()
        except Exception as exc:
            from .models import SystemSample  # local import to avoid cycles at parse time
            errors.append(f"System sample error: {exc}")
            return SystemSample()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _run_discovery(self, ts: float, errors: list[str]) -> None:
        """Full process+port scan, fingerprint, update states."""
        try:
            found: list[EngineInfo] = discover(
                client=self._client,
                extra_ports=self._extra_ports,
            )
        except Exception as exc:
            errors.append(f"Discovery error: {exc}")
            found = []

        self._last_discover = ts

        found_keys: set[str] = set()
        for engine in found:
            k = engine.key
            found_keys.add(k)
            if k not in self._states:
                # New engine: emit appeared event immediately into the ring.
                self._event_ring.append(
                    f"Engine appeared: {engine.name} ({engine.engine_type.value}) at {engine.base_url}"
                )
                state = _EngineState(engine)
                self._states[k] = state
            else:
                # Known engine: update process/model info but preserve history.
                existing = self._states[k]
                # Track model changes.
                new_model_ids = {m.id for m in engine.models}
                for mid in new_model_ids - existing.seen_models:
                    self._event_ring.append(
                        f"Model loaded: {mid} on {engine.name}"
                    )
                for mid in existing.seen_models - new_model_ids:
                    self._event_ring.append(
                        f"Model unloaded: {mid} on {engine.name}"
                    )
                existing.seen_models = new_model_ids
                # Replace engine object (fresh describe) but keep history.
                existing.engine = engine

        # Detect disappeared engines.
        disappeared = [k for k in list(self._states.keys()) if k not in found_keys]
        for k in disappeared:
            st = self._states.pop(k)
            self._event_ring.append(
                f"Engine disappeared: {st.engine.name} ({st.engine.engine_type.value}) at {k}"
            )

        # Refresh metrics for all surviving/new engines.
        self._refresh_metrics(ts, errors)

    # ------------------------------------------------------------------
    # Metric refresh (between discoveries)
    # ------------------------------------------------------------------

    def _refresh_metrics(self, ts: float, errors: list[str]) -> None:
        """Call adapter.metrics() for every known engine; update state."""
        for k, state in list(self._states.items()):
            engine = state.engine
            try:
                adapter = adapter_for(engine.engine_type)
                dt: Optional[float] = None
                if state.prev_time is not None:
                    dt = ts - state.prev_time
                new_metrics = adapter.metrics(
                    engine,
                    self._client,
                    previous=state.prev_metrics,
                    dt=dt,
                )
            except Exception as exc:
                err_msg = f"Metrics error for {engine.name}: {exc}"
                errors.append(err_msg)
                new_metrics = EngineMetrics(error=str(exc))

            # Emit event on new metrics error (rising-edge only).
            had_error = bool(state.prev_metrics and state.prev_metrics.error)
            has_error = bool(new_metrics.error)
            if has_error and not had_error:
                self._event_ring.append(
                    f"Metrics error: {engine.name}: {new_metrics.error}"
                )

            # Update history.
            state.prev_metrics = new_metrics
            state.prev_time = ts
            engine.metrics = new_metrics

    # ------------------------------------------------------------------
    # VRAM attribution
    # ------------------------------------------------------------------

    def _attribute_vram(self, gpus: list[GpuSample]) -> None:
        """Attribute GPU memory to engines and set ``vram_bytes``.

        Two strategies, in order:

        1. **PID match** — when discovery linked the engine to a listening PID,
           sum the GPU memory of that PID and its child PIDs.
        2. **Process-tree classification** — when the engine's PID is unknown
           (common: the engine runs as root or inside a container, so the
           socket→PID map is unreadable), classify each *remaining* GPU process
           by walking its ``cmdline``/ancestry to an :class:`EngineType` and
           attribute it to the unique discovered engine of that type. This also
           recovers the engine's launcher PID and uptime as a side effect.
        """
        if not gpus:
            return

        # Build pid→gpu_mem_bytes map across all GPUs (sum if multi-GPU).
        pid_vram: dict[int, int] = {}
        for gpu in gpus:
            for proc in gpu.procs:
                if proc.gpu_mem_bytes is not None:
                    pid_vram[proc.pid] = (
                        pid_vram.get(proc.pid, 0) + proc.gpu_mem_bytes
                    )

        if not pid_vram:
            return

        # Build a child→engine pid map from psutil (best-effort).
        child_to_engine: dict[int, int] = {}
        try:
            import psutil  # already a dependency

            engine_pids: set[int] = {
                st.engine.pid for st in self._states.values() if st.engine.pid is not None
            }
            for epid in engine_pids:
                try:
                    proc = psutil.Process(epid)
                    for child in proc.children(recursive=True):
                        child_to_engine[child.pid] = epid
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception:
            pass

        # --- Strategy 1: explicit PID match ---
        consumed: set[int] = set()
        for st in self._states.values():
            engine = st.engine
            if engine.pid is None:
                continue
            total_vram = 0
            counted = False
            if engine.pid in pid_vram:
                total_vram += pid_vram[engine.pid]
                consumed.add(engine.pid)
                counted = True
            for cpid, epid in child_to_engine.items():
                if epid == engine.pid and cpid in pid_vram:
                    total_vram += pid_vram[cpid]
                    consumed.add(cpid)
                    counted = True
            if counted:
                engine.vram_bytes = total_vram
                if engine.models:
                    engine.models[0].vram_bytes = total_vram

        # --- Strategy 2: classify the remaining GPU processes by type ---
        remaining = {pid: mem for pid, mem in pid_vram.items() if pid not in consumed}
        if not remaining:
            return

        type_vram: dict[EngineType, int] = {}
        type_launcher: dict[EngineType, ProcessInfo] = {}
        for pid, mem in remaining.items():
            etype, proc_info = self._classify_proc_tree(pid)
            if etype is None:
                continue
            type_vram[etype] = type_vram.get(etype, 0) + mem
            # Keep the earliest-started launcher as the representative process.
            prev = type_launcher.get(etype)
            if proc_info is not None:
                ct = proc_info.create_time
                prev_ct = prev.create_time if prev is not None else None
                if prev is None or (ct is not None and (prev_ct is None or ct < prev_ct)):
                    type_launcher[etype] = proc_info

        if not type_vram:
            return

        # Engines that still lack a VRAM figure, grouped by type.
        by_type: dict[EngineType, list[EngineInfo]] = {}
        for st in self._states.values():
            if st.engine.vram_bytes is None:
                by_type.setdefault(st.engine.engine_type, []).append(st.engine)

        for etype, vram in type_vram.items():
            candidates = by_type.get(etype, [])
            if len(candidates) != 1 or vram <= 0:
                # Ambiguous (multiple engines of this type) or nothing to assign;
                # leave as n/a rather than guess wrong.
                continue
            engine = candidates[0]
            engine.vram_bytes = vram
            if engine.models:
                engine.models[0].vram_bytes = vram
            proc_info = type_launcher.get(etype)
            if proc_info is not None:
                if engine.pid is None:
                    engine.pid = proc_info.pid
                    if engine.process is None:
                        engine.process = proc_info
                if engine.uptime_s is None and proc_info.create_time is not None:
                    engine.uptime_s = max(0.0, time.time() - proc_info.create_time)

    def _classify_proc_tree(
        self, pid: int
    ) -> tuple[Optional[EngineType], Optional[ProcessInfo]]:
        """Classify a GPU process by walking it and its ancestors.

        Returns ``(engine_type, launcher_process)``. The engine type is taken
        from the leaf process (the actual GPU user) when it is recognizable,
        else the nearest classifiable ancestor. The launcher is the
        earliest-started process in the chain of that same type (e.g. the
        ``vllm serve`` parent rather than the ``VLLM::EngineCore`` worker), with
        its name/cmdline/create_time/username populated for display.
        """
        try:
            import psutil

            proc = psutil.Process(pid)
        except Exception:
            return None, None

        def classify(p: Any) -> Optional[EngineType]:
            try:
                cmd = p.cmdline()
            except Exception:
                cmd = []
            etype = classify_cmdline(cmd) if cmd else None
            if etype is None:
                try:
                    name = p.name()
                except Exception:
                    name = ""
                if name:
                    etype = classify_cmdline([name])
            return etype

        try:
            chain = [proc] + list(proc.parents())
        except Exception:
            chain = [proc]

        etype = classify(proc)
        if etype is None:
            for ancestor in chain[1:]:
                etype = classify(ancestor)
                if etype is not None:
                    break
        if etype is None:
            return None, None

        best = proc
        best_ct: Optional[float] = None
        for p in chain:
            if classify(p) == etype:
                try:
                    ct = p.create_time()
                except Exception:
                    ct = None
                if ct is not None and (best_ct is None or ct < best_ct):
                    best_ct = ct
                    best = p

        def _safe(fn: Callable[[], Any], default: Any = None) -> Any:
            try:
                return fn()
            except Exception:
                return default

        info = ProcessInfo(
            pid=best.pid,
            name=_safe(best.name),
            cmdline=_safe(best.cmdline, []) or [],
            create_time=best_ct if best_ct is not None else _safe(best.create_time),
            username=_safe(best.username),
            hint=etype,
        )
        return etype, info

    # ------------------------------------------------------------------
    # Snapshot diffing
    # ------------------------------------------------------------------

    def _diff(
        self,
        prev: Snapshot,
        engines: list[EngineInfo],
    ) -> list[str]:
        """Compare previous and current engine lists; return new event strings.

        Engine appeared/disappeared events are emitted eagerly in the discovery
        path; here we detect model-level transitions that aren't caught there,
        such as a model being swapped while the engine is still up.
        """
        events: list[str] = []

        prev_by_key = {e.key: e for e in prev.engines}
        curr_by_key = {e.key: e for e in engines}

        for key, curr_engine in curr_by_key.items():
            prev_engine = prev_by_key.get(key)
            if prev_engine is None:
                # Already captured in _run_discovery; skip duplicate.
                continue

            prev_primary = prev_engine.primary_model
            curr_primary = curr_engine.primary_model
            if prev_primary != curr_primary:
                if curr_primary is None:
                    events.append(f"Model unloaded: {prev_primary} on {curr_engine.name}")
                elif prev_primary is None:
                    events.append(f"Model loaded: {curr_primary} on {curr_engine.name}")
                else:
                    events.append(
                        f"Model swapped: {prev_primary} → {curr_primary} on {curr_engine.name}"
                    )

        return events

    # ------------------------------------------------------------------
    # Event ring drain
    # ------------------------------------------------------------------

    def _drain_events(self) -> list[str]:
        """Drain and return all pending events from the ring."""
        evts = list(self._event_ring)
        self._event_ring.clear()
        return evts

    # ------------------------------------------------------------------
    # Context-manager support (optional convenience)
    # ------------------------------------------------------------------

    def __enter__(self) -> Monitor:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class _NullGpuSampler:
    """Stand-in GpuSampler used when enable_gpu=False."""

    @property
    def available(self) -> bool:
        return False

    def sample(self) -> list[GpuSample]:
        return []

    def close(self) -> None:
        pass
