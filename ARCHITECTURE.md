# llmtop internal architecture & contract

This is the binding contract for all modules. Implement to these exact names and
signatures. Everything crossing a module boundary is a dataclass from
`llmtop/models.py`. Read `llmtop/models.py` and `llmtop/adapters/base.py` first.

## Authentication (zero-config, but key-aware)
Some local endpoints require an API key (verified: the llm-router on :8077 returns
HTTP 401 `{"error":{"message":"missing or invalid API key",...}}` to an unauthenticated
`GET /v1/models`). Handling:
- The monitor builds the shared `httpx.Client` with an `Authorization: Bearer <key>`
  header when **either** `LLMTOP_API_KEY` or `OPENAI_API_KEY` is set in the
  environment (check LLMTOP_API_KEY first). Default: no header (truly zero-config).
- Adapters treat HTTP 401/403 as "engine present but introspection blocked": still
  return a valid `EngineInfo` (best-guess `engine_type`, e.g. OPENAI), set
  `last_error="requires API key (set LLMTOP_API_KEY to introspect)"`, and leave
  models/metrics empty rather than discarding the engine. A responding 401 is a
  positive detection signal, not a miss.

## Golden rules
- **Read-only & safe.** Never call a generation/completion endpoint or anything
  that consumes tokens or mutates a server. Only GET metrics/introspection.
- **Degrade, never crash.** Missing GPU, no engines, non-NVIDIA, partial metrics,
  timeouts: catch and return partial data with `None`/`error` fields. No
  exception should escape `sample()`, `poll()`, `detect()`, `describe()`,
  `metrics()`.
- **Every metric is Optional.** `None` renders as `n/a`. Never fake a 0.
- **Time-bounded.** All HTTP via the passed `httpx.Client` (short timeout). All
  socket ops have timeouts. Nothing blocks the UI thread.
- **No root required** for anything in v1.

## Module layout & public API (implement exactly)

### `llmtop/gpu.py`  — owner: GPU/system slice
```python
class GpuSampler:
    def __init__(self) -> None: ...          # nvmlInit(); set self.available=False on failure, never raise
    @property
    def available(self) -> bool: ...
    def sample(self) -> list[GpuSample]: ...  # one per GPU; [] if NVML unavailable/no GPU
    def close(self) -> None: ...              # nvmlShutdown(), idempotent, swallow errors
```
**Unified-memory handling (CRITICAL — verified on NVIDIA GB10 / driver 580):**
- `nvmlDeviceGetMemoryInfo` raises `NVMLError` code 3 (`NVML_ERROR_NOT_SUPPORTED`)
  on GB10 and Jetson-class. On that error: set `unified_memory=True`, set
  `note="unified memory (shared with system RAM)"`, and fill memory from psutil:
  `mem_total_bytes = psutil.virtual_memory().total`. For `mem_used_bytes`, prefer
  the sum of per-process compute GPU memory (see below); if unavailable, fall back
  to `psutil.virtual_memory().used`.
- Also treat as unified by name match: any of `GB10`, `GH200`(no — leave),
  `Thor`, `Orin`, `Xavier`, `Jetson`, `Tegra` in the device name.
- **Per-process GPU memory works even when total does not.** Populate `procs`
  from `nvmlDeviceGetComputeRunningProcesses` + `nvmlDeviceGetGraphicsRunningProcesses`,
  each as a `ProcessInfo(pid=p.pid, gpu_mem_bytes=p.usedGpuMemory)` (usedGpuMemory
  may be `None`/a large sentinel — treat the NVML "not available" sentinel
  `2**64-1`/`None` as None). Dedup by pid (compute wins).
- Wrap EVERY nvml getter in its own try/except so one unsupported field
  (power cap, mem clock, fan = NOT_SUPPORTED on GB10) does not lose the others.
- Convert: power mW→W (`/1000`), enforced power limit mW→W via
  `nvmlDeviceGetEnforcedPowerLimit` (NOT_SUPPORTED on GB10 → None). Clocks:
  `NVML_CLOCK_SM`, `NVML_CLOCK_MEM` (mem NOT_SUPPORTED on GB10 → None).
- `throttled`: `nvmlDeviceGetCurrentClocksThrottleReasons` present and not in
  {0, nvmlClocksThrottleReasonGpuIdle, nvmlClocksThrottleReasonNone}. Tolerate
  missing symbol.
- Known-good on GB10: util, temp (`NVML_TEMPERATURE_GPU`), power, SM clock,
  compute/graphics procs. Known NOT_SUPPORTED: total mem, power cap, mem clock,
  fan. Build and test against exactly this.

### `llmtop/system.py` — owner: GPU/system slice
```python
class SystemSampler:
    def __init__(self) -> None: ...          # prime psutil.cpu_percent (call once, interval=None)
    def sample(self) -> SystemSample: ...     # non-blocking cpu_percent(interval=None)
```
Use psutil for cpu_percent(interval=None), virtual_memory (used/total), cpu_count,
os.getloadavg() (guard on platforms lacking it).

### `llmtop/discovery/` — owner: discovery slice
- `process_scan.py`:
  ```python
  CMDLINE_PATTERNS: dict[EngineType, list[str]]   # substrings/regex to match
  def classify_cmdline(cmdline: list[str]) -> Optional[EngineType]: ...
  def scan_processes() -> list[ProcessInfo]: ...   # all procs that look like an engine,
      # with .hint set via classify, .ports filled from psutil net_connections (kind="inet",
      # status LISTEN) where accessible; tolerate AccessDenied/NoSuchProcess.
  ```
  Match launchers: `vllm serve`, `python -m vllm`, `vllm.entrypoints`,
  `llama-server`, `llama.cpp`, `ollama`/`ollama serve`/`ollama runner`,
  `text-generation-launcher`/`text_generation_server` (TGI), `sglang.launch_server`/
  `sglang`, `-m sglang`, generic uvicorn/python with `--port` and (`--model`/`--model-path`).
- `port_scan.py`:
  ```python
  DEFAULT_PORTS: tuple[int, ...]  # 8000,8001,8077,8080,8086,8088,8099,5000,3000,11434,
                                  # 30000,1234,4000,4891,7860,8888,9090,80, ...
  def scan_ports(ports: Iterable[int], host: str = "127.0.0.1",
                 timeout: float = 0.15) -> list[int]: ...  # TCP-connect open check
  ```
- `fingerprint.py`:
  ```python
  def fingerprint(candidate: Candidate, client: httpx.Client) -> EngineInfo: ...
  # try iter_detectors() in order: first cls.detect(candidate, client) that returns
  # non-None wins; then call adapter_for(engine.engine_type).describe(engine, client).
  # If all miss but the port responded to HTTP at all, use UnknownAdapter; if no HTTP,
  # still return an UNKNOWN EngineInfo describing the raw port+process (mark last_error).
  ```
- `discover.py`:
  ```python
  def discover(client: httpx.Client | None = None, *, extra_ports: Iterable[int] = (),
               include_system_ports: bool = True, timeout: float = 1.0) -> list[EngineInfo]: ...
  # 1) scan_processes(); collect their listening ports + hints.
  # 2) candidate ports = DEFAULT_PORTS ∪ process ports ∪ extra_ports; scan_ports() to filter open.
  # 3) build Candidate per open (host,port), attaching pid/process/hint when a process owns it.
  # 4) fingerprint each (dedup by host:port); skip ports already claimed as a router backend.
  # 5) correlate_router(engines): if an engine advertises many models whose names map to
  #    OTHER discovered engines, OR a known router signature (served model list >> 1 and a
  #    process cmdline / header indicating a proxy), set is_router=True, engine_type=ROUTER,
  #    move matched engines into .backends and set their routed_by. Return top-level engines
  #    (routers contain their backends; backends still listed too is OK but mark routed_by).
  def correlate_router(engines: list[EngineInfo]) -> list[EngineInfo]: ...
  ```
  Create the httpx.Client if not passed: `httpx.Client(timeout=timeout, trust_env=False)`.

### `llmtop/adapters/*.py` — owner: adapter slices
Each adapter subclasses `Adapter` (see base.py docstring), sets `engine_type`,
`default_ports`, `priority`, and implements `detect`/`describe`/`metrics`.
Per-engine probe specifics:
- **vLLM** (`priority=10`, ports 8000,8001,8088): detect via `GET /v1/models` AND a
  Prometheus `GET /metrics` containing `vllm:` series (or a `/version`). Metrics from
  `/metrics`: `vllm:generation_tokens_total` (→tokens_total, derive decode_tps),
  `vllm:prompt_tokens_total` (→prompt_tokens_total, derive prefill_tps),
  `vllm:num_requests_running`, `vllm:num_requests_waiting`,
  `vllm:kv_cache_usage_perc` (→kv_cache_pct, value 0..1 → *100; fall back to the
  older name `vllm:gpu_cache_usage_perc` if absent). NOTE: every vLLM series
  carries a label set, e.g.
  `vllm:generation_tokens_total{engine="0",model_name="qwen36-coder"} 0.0` —
  the Prometheus parser MUST strip `{...}` labels and SUM across all label sets
  for a given metric name. Parse Prometheus text by hand (no extra dep). model id
  + max_model_len from
  `/v1/models` (`data[].id`, `data[].max_model_len`). dtype/quant from model id
  heuristics if present.
- **Ollama** (`priority=5`, port 11434): detect via `GET /api/tags` (has `.models[]`)
  or `GET /api/version`. describe: `/api/tags` for installed, `GET /api/ps` for
  LOADED models (those in VRAM now) → `ModelInfo(loaded=True, vram_bytes=size_vram,
  size_bytes=size, context_length=... if in details)`; tags-only models are
  `loaded=False`. quant from `details.quantization_level`, family from `details.family`.
  Ollama exposes no token-rate metrics → leave decode_tps/kv None (set requests via
  /api/ps count of loaded). version from `/api/version`.
- **llama.cpp server** (`priority=30`, port 8080,8081): detect via `GET /health`
  (`{"status":"ok"}`) AND/OR `GET /props`. describe: `/props` → `default_generation_settings`/
  `n_ctx`, model path → id+quant (parse GGUF filename for quant like `Q4_K_M`).
  metrics: `GET /metrics` IF enabled (Prometheus: `llamacpp:prompt_tokens_total`,
  `llamacpp:tokens_predicted_total`, `llamacpp:requests_processing`,
  `llamacpp:requests_deferred`, `llamacpp:kv_cache_usage_ratio`). If /metrics 404s,
  return EngineMetrics() with error="metrics endpoint not enabled".
- **TGI** (`priority=20`, port 8080,80,3000): detect via `GET /info` (has
  `model_id`, `max_total_tokens`). metrics via `GET /metrics` (Prometheus:
  `tgi_request_*`, `tgi_batch_current_size`, `tgi_queue_size`,
  `tgi_request_generated_tokens`...). version from `/info.version`.
- **SGLang** (`priority=15`, port 30000): detect via `GET /get_model_info`
  (`model_path`) and/or `GET /health`. metrics via `GET /metrics` (Prometheus:
  `sglang:num_running_reqs`, `sglang:num_queue_reqs`, `sglang:gen_throughput`,
  `sglang:token_usage`, `sglang:cache_hit_rate`).
- **OpenAI-generic** (`priority=90`): detect via `GET /v1/models` returning
  `{object:"list", data:[...]}` when nothing more specific matched. describe: list
  model ids. metrics: try `GET /metrics`; otherwise EngineMetrics() empty. This is
  also the adapter used for ROUTER engines.
- **Unknown** (`priority=100`, `engine_type=UNKNOWN`): always "detects" as a last
  resort if asked. describe: record server header / status. metrics: empty.

Use `derive_rate(current_total, previous_total, dt)` from base for tok/s.
Prometheus parsing helper: each adapter can implement a small local parser, OR
import a shared one from `llmtop/adapters/prom.py` (you may create that file):
```python
def parse_prometheus(text: str) -> dict[str, float]: ...  # name (ignoring labels) -> summed value
def prom_value(parsed, name, default=None): ...
```
If you create `prom.py`, keep it dependency-free.

### `llmtop/core.py` — owner: core slice
```python
class Monitor:
    def __init__(self, *, interval: float = 2.0, timeout: float = 1.0,
                 rediscover_every: float = 8.0, extra_ports: Iterable[int] = (),
                 enable_gpu: bool = True) -> None: ...
    def poll(self) -> Snapshot: ...       # ONE synchronous snapshot (used by --json & tests)
    async def run(self, callback: Callable[[Snapshot], Awaitable[None] | None]) -> None: ...
        # async loop: every `interval`s build a Snapshot (run the blocking poll in a thread:
        # await asyncio.to_thread(self.poll)) and await/call callback(snapshot). Re-run full
        # discovery only every `rediscover_every`s; between, refresh metrics on known engines.
    def close(self) -> None: ...
```
Monitor responsibilities:
- Owns one `GpuSampler`, one `SystemSampler`, one `httpx.Client(timeout=timeout, trust_env=False)`.
- Caches discovered engines; refreshes their metrics each poll via
  `adapter_for(e.engine_type).metrics(e, client, previous, dt)`, keeping per-engine
  previous metrics + timestamp for rate derivation.
- Attributes VRAM: match `engine.pid` (and child pids) to `GpuSample.procs[].gpu_mem_bytes`,
  set `engine.vram_bytes` and the primary model's `vram_bytes`.
- Diffs vs previous snapshot to produce `events`: engine appeared/disappeared,
  model loaded/swapped, metrics error appeared. Keep an internal event ring if handy.
- Never raises out of `poll()`; collect per-source errors into `Snapshot.errors`.

### `llmtop/cli.py` + `llmtop/__main__.py` — owner: cli slice
```python
def main(argv: list[str] | None = None) -> int: ...   # console entrypoint
```
argparse flags: `--json` (print ONE snapshot as JSON to stdout, exit 0),
`--once` (same as --json but human table, optional), `--interval FLOAT` (default 2.0),
`--port INT` (repeatable → extra_ports), `--no-gpu`, `--timeout FLOAT`,
`--version`, `--debug` (tracebacks). Default (no --json): construct `Monitor` and
launch the TUI (`from .tui.app import LlmtopApp; LlmtopApp(monitor, interval).run()`).
`__main__.py`: `from .cli import main; raise SystemExit(main())`.
JSON serialization: a `snapshot_to_dict(snap) -> dict` (put in cli.py) using
`dataclasses.asdict` with enum→`.value` and tuple→list normalization; `json.dumps(..., default=str)`.
`--json` must work headless with no TTY and exit non-zero only on hard failure.

### `llmtop/tui/app.py` (+ optional `tui/widgets.py`) — owner: TUI slice
```python
class LlmtopApp(App):
    def __init__(self, monitor: Monitor, interval: float = 2.0): ...
```
Textual app. Depends ONLY on `Monitor` + models (Snapshot). Layout:
- **Header band:** per-GPU gauges — util %, mem used/total (label "unified" when
  `gpu.unified_memory`), temp, power (show `n/a` when None), + a braille/sparkline
  of recent util and tok/s history.
- **Main table:** sortable `DataTable` of engines: columns Engine, Model, Port,
  PID, tok/s (decode), reqs (run/wait), KV%, VRAM, Uptime. Routers shown with their
  backends indented beneath. Color pressure: KV% >85 red, queue>0 yellow, throttled red.
- **Detail pane:** toggled by Enter/select on a row → full flags/config, all models,
  raw metrics, per-engine throughput sparkline.
- **Event log:** bottom scrolling log fed from `snapshot.events`.
- **Keybinds:** `q` quit, `p` pause polling, `s` cycle sort column, `f` filter
  prompt, `enter` toggle detail, `r` force rediscover, `?` help.
- Drive updates with a Textual worker / `set_interval` that calls
  `await asyncio.to_thread(monitor.poll)` (do NOT block the event loop) and updates
  widgets. Tolerate a poll returning errors; surface them in the event log.
- Must run on a normal 80x24 terminal and resize gracefully.

## Testing contract — owner: tests slice (later wave)
- Pure-unit, no live servers: drive adapters with `httpx.MockTransport` / a fake
  client returning RECORDED fixture bodies (Prometheus text + JSON) under
  `tests/fixtures/`. Assert detect()/describe()/metrics() parse correctly and that
  `derive_rate` handles resets. Test the unified-memory GPU branch by
  monkeypatching the nvml calls. Test `--json` headless smoke. Test "no engines"
  and "no GPU" degrade paths.

## Style
- Python 3.11+, `from __future__ import annotations` in every module.
- Stdlib + the four deps only (textual, httpx, psutil, nvidia-ml-py). No new runtime deps.
- Type hints throughout; small focused functions; docstrings on public API.
- Match the tone/structure already in `models.py` and `adapters/base.py`.
