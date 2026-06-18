"""Core data models for llmtop.

These dataclasses are the shared contract between the discovery layer, the engine
adapters, the GPU/system samplers, the monitor loop, and the TUI. Everything that
crosses a module boundary is one of these types.

Design rules:
- Every metric field is Optional and defaults to None. A value of None means
  "not available / not applicable" and must render as ``n/a`` in the UI, never
  as 0 and never crash. Producers fill what they can; consumers tolerate gaps.
- Cumulative counters (``*_total``) hold the raw monotonically-increasing value
  scraped from an engine. Rates (``*_tps``) are derived by the monitor from the
  delta between two snapshots. Adapters may compute rates themselves when given a
  previous sample, otherwise leave them None.
- ``raw`` dicts carry adapter-specific values for the drill-down detail pane;
  they are free-form and never required by core logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class EngineType(str, Enum):
    """Known inference engine families. ``str`` mix-in so values JSON-serialize."""

    VLLM = "vllm"
    LLAMACPP = "llama.cpp"
    OLLAMA = "ollama"
    TGI = "tgi"
    SGLANG = "sglang"
    OPENAI = "openai-compatible"
    ROUTER = "router"
    UNKNOWN = "unknown"


@dataclass
class ModelInfo:
    """A single model served by an engine."""

    id: str
    quantization: Optional[str] = None       # e.g. "Q4_K_M", "NVFP4", "awq"
    dtype: Optional[str] = None              # e.g. "bfloat16", "float16"
    context_length: Optional[int] = None     # max context / n_ctx / max_model_len
    loaded: bool = True                      # loaded in VRAM vs idle/swapped-out
    size_bytes: Optional[int] = None         # on-disk / declared model size
    vram_bytes: Optional[int] = None         # resident VRAM for this model if known
    family: Optional[str] = None             # arch family if exposed (e.g. "qwen3")
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineMetrics:
    """Live serving metrics for one engine.

    Counters are cumulative-since-start (raw scrape). Rates are per-second and are
    either derived by the monitor from consecutive samples or computed by the
    adapter when it is handed the previous metrics object.
    """

    decode_tps: Optional[float] = None          # generation/decode tokens per sec
    prefill_tps: Optional[float] = None         # prompt/prefill tokens per sec
    requests_running: Optional[int] = None      # active requests
    requests_waiting: Optional[int] = None      # queued requests (queue depth)
    kv_cache_pct: Optional[float] = None        # KV cache utilization 0-100
    tokens_total: Optional[int] = None          # cumulative generated tokens
    prompt_tokens_total: Optional[int] = None   # cumulative prompt tokens
    requests_total: Optional[int] = None        # cumulative finished requests
    raw: dict[str, Any] = field(default_factory=dict)  # raw values for detail pane
    error: Optional[str] = None                 # why metrics are missing, if so


@dataclass
class ProcessInfo:
    """A process associated with an engine and/or holding GPU memory."""

    pid: int
    name: Optional[str] = None
    cmdline: list[str] = field(default_factory=list)
    create_time: Optional[float] = None     # epoch seconds (psutil.create_time)
    cpu_percent: Optional[float] = None
    rss_bytes: Optional[int] = None         # resident system memory
    gpu_mem_bytes: Optional[int] = None     # GPU/compute memory (NVML), if known
    username: Optional[str] = None
    ports: list[int] = field(default_factory=list)  # listening ports owned by pid
    hint: Optional["EngineType"] = None     # engine family guessed from the cmdline


@dataclass
class Candidate:
    """A discovery candidate: a reachable host:port plus any process context.

    Produced by the discovery layer (process scan + port scan) and handed to
    adapters' ``detect`` so they can probe and claim it.
    """

    host: str
    port: int
    pid: Optional[int] = None
    process: Optional[ProcessInfo] = None
    hint: Optional[EngineType] = None       # guessed type from process cmdline
    signals: list[str] = field(default_factory=list)  # how it was discovered

    @property
    def base_url(self) -> str:
        host = self.host if self.host not in ("0.0.0.0", "::", "*") else "127.0.0.1"
        return f"http://{host}:{self.port}"


@dataclass
class EngineInfo:
    """A discovered inference engine and everything known about it."""

    engine_type: EngineType
    name: str                                # human display name
    base_url: str                            # http://host:port
    host: str
    port: int
    version: Optional[str] = None
    pid: Optional[int] = None
    process: Optional[ProcessInfo] = None
    models: list[ModelInfo] = field(default_factory=list)
    flags: dict[str, Any] = field(default_factory=dict)  # notable backend config
    uptime_s: Optional[float] = None
    vram_bytes: Optional[int] = None         # VRAM attributed to this engine's pid
    metrics: EngineMetrics = field(default_factory=EngineMetrics)

    # Router topology
    is_router: bool = False
    backends: list["EngineInfo"] = field(default_factory=list)  # for routers
    routed_by: Optional[str] = None          # base_url of a router fronting this

    signals: list[str] = field(default_factory=list)  # discovery signals
    last_error: Optional[str] = None

    @property
    def key(self) -> str:
        """Stable identity for diffing snapshots / tracking history."""
        return self.base_url

    @property
    def primary_model(self) -> Optional[str]:
        return self.models[0].id if self.models else None


@dataclass
class GpuSample:
    """One GPU's live state for a single snapshot."""

    index: int
    name: str
    util_pct: Optional[float] = None
    mem_used_bytes: Optional[int] = None
    mem_total_bytes: Optional[int] = None
    temp_c: Optional[float] = None
    power_w: Optional[float] = None
    power_cap_w: Optional[float] = None
    clock_sm_mhz: Optional[int] = None
    clock_mem_mhz: Optional[int] = None
    fan_pct: Optional[float] = None
    throttled: bool = False
    unified_memory: bool = False            # GB10/Jetson: VRAM shared with sys RAM
    procs: list[ProcessInfo] = field(default_factory=list)  # GPU compute/graphics
    note: Optional[str] = None              # e.g. "unified memory (shared w/ RAM)"
    error: Optional[str] = None

    @property
    def mem_pct(self) -> Optional[float]:
        if self.mem_used_bytes is None or not self.mem_total_bytes:
            return None
        return 100.0 * self.mem_used_bytes / self.mem_total_bytes


@dataclass
class SystemSample:
    """Host CPU/RAM for a single snapshot."""

    cpu_pct: Optional[float] = None
    cpu_count: Optional[int] = None
    ram_used_bytes: Optional[int] = None
    ram_total_bytes: Optional[int] = None
    load_avg: Optional[tuple[float, float, float]] = None

    @property
    def ram_pct(self) -> Optional[float]:
        if self.ram_used_bytes is None or not self.ram_total_bytes:
            return None
        return 100.0 * self.ram_used_bytes / self.ram_total_bytes


@dataclass
class Snapshot:
    """One full poll: hardware + all engines at a point in time."""

    timestamp: float
    gpus: list[GpuSample] = field(default_factory=list)
    system: SystemSample = field(default_factory=SystemSample)
    engines: list[EngineInfo] = field(default_factory=list)
    events: list[str] = field(default_factory=list)   # transitions since last poll
    errors: list[str] = field(default_factory=list)   # non-fatal sampling errors

    @property
    def gpu(self) -> Optional[GpuSample]:
        """Convenience accessor for the first GPU, if any."""
        return self.gpus[0] if self.gpus else None
