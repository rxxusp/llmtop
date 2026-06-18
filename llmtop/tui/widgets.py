"""Custom Textual widgets for llmtop.

Provides:
- :class:`GpuHeaderWidget`  — per-GPU gauge + braille sparkline band.
- :class:`EngineTable`      — sortable DataTable of discovered engines.
- :class:`DetailPane`       — expandable detail view for a selected engine.
- :class:`EventLogWidget`   — scrolling event log fed from snapshot.events.

All widgets accept a :class:`~llmtop.models.Snapshot` and re-render it
in-place; they never poll externally. Callers call the ``update_snapshot``
method whenever a new snapshot arrives.
"""

from __future__ import annotations

import collections
import time
from typing import Optional

from rich.text import Text
from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import DataTable, RichLog, Sparkline, Static

from ..models import EngineInfo, GpuSample, Snapshot

# ---------------------------------------------------------------------------
# Braille sparkline history helpers
# ---------------------------------------------------------------------------

_HISTORY_LEN = 60  # number of data-points to keep per series

_BRAILLE_ROWS = 4  # 4 vertical levels per character (Braille block)


def _fmt_bytes(b: Optional[int], *, binary: bool = True) -> str:
    """Format byte count as a human-readable string, or 'n/a' if None."""
    if b is None:
        return "n/a"
    if binary:
        units = ["B", "KiB", "MiB", "GiB", "TiB"]
        div = 1024
    else:
        units = ["B", "KB", "MB", "GB", "TB"]
        div = 1000
    v: float = float(b)
    for u in units[:-1]:
        if v < div:
            return f"{v:.1f} {u}"
        v /= div
    return f"{v:.1f} {units[-1]}"


def _fmt_float(v: Optional[float], decimals: int = 1, suffix: str = "") -> str:
    """Format an optional float or return 'n/a'."""
    if v is None:
        return "n/a"
    return f"{v:.{decimals}f}{suffix}"


def _fmt_int(v: Optional[int], suffix: str = "") -> str:
    """Format an optional int or return 'n/a'."""
    if v is None:
        return "n/a"
    return f"{v}{suffix}"


def _uptime_str(uptime_s: Optional[float]) -> str:
    """Format uptime in seconds as a human string."""
    if uptime_s is None:
        return "n/a"
    s = int(uptime_s)
    h, remainder = divmod(s, 3600)
    m, sec = divmod(remainder, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def _engine_label(e: EngineInfo) -> str:
    """Short display name for an engine."""
    return e.name or e.engine_type.value


# ---------------------------------------------------------------------------
# GPU Header Widget
# ---------------------------------------------------------------------------


class GpuHeaderWidget(Widget):
    """Horizontal band showing one gauge + sparkline per GPU.

    Displays: GPU name, utilisation %, memory used/total (labels "unified"
    for unified-memory devices), temperature, power, SM clock, and a
    braille sparkline of recent util history.
    """

    DEFAULT_CSS = """
    GpuHeaderWidget {
        height: auto;
        width: 1fr;
        padding: 0 1;
    }
    GpuHeaderWidget .gpu-row {
        height: 3;
        width: 1fr;
    }
    GpuHeaderWidget .gpu-label {
        width: auto;
        padding: 0 1 0 0;
        color: $text;
    }
    GpuHeaderWidget .gpu-gauge-area {
        width: 1fr;
        height: 3;
    }
    GpuHeaderWidget .gpu-spark {
        height: 3;
        width: 20;
        min-width: 10;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Per-GPU util history (deque for sparkline)
        self._util_history: dict[int, collections.deque[float]] = {}
        self._toks_history: dict[str, collections.deque[float]] = {}
        self._last_snapshot: Optional[Snapshot] = None

    def compose(self) -> ComposeResult:
        yield Static("No GPU data yet", id="gpu-placeholder")

    def _get_util_history(self, gpu_idx: int) -> collections.deque[float]:
        if gpu_idx not in self._util_history:
            self._util_history[gpu_idx] = collections.deque(
                [0.0] * _HISTORY_LEN, maxlen=_HISTORY_LEN
            )
        return self._util_history[gpu_idx]

    def update_snapshot(self, snap: Snapshot) -> None:
        """Refresh the GPU header from a new snapshot."""
        self._last_snapshot = snap

        for gpu in snap.gpus:
            hist = self._get_util_history(gpu.index)
            hist.append(gpu.util_pct if gpu.util_pct is not None else 0.0)

        self._rebuild_gpu_display(snap)

    def _gpu_line(self, gpu: GpuSample) -> str:
        """Build a one-line summary string for a single GPU."""
        name = gpu.name or f"GPU {gpu.index}"
        # Truncate name if too long
        if len(name) > 24:
            name = name[:22] + ".."

        mem_label = "unified" if gpu.unified_memory else "VRAM"
        if gpu.mem_used_bytes is not None and gpu.mem_total_bytes:
            mem_str = f"{_fmt_bytes(gpu.mem_used_bytes)}/{_fmt_bytes(gpu.mem_total_bytes)} [{mem_label}]"
        elif gpu.mem_used_bytes is not None:
            mem_str = f"{_fmt_bytes(gpu.mem_used_bytes)} [{mem_label}]"
        else:
            mem_str = f"n/a [{mem_label}]"

        util_str = _fmt_float(gpu.util_pct, suffix="%")
        temp_str = _fmt_float(gpu.temp_c, decimals=0, suffix="°C")
        power_str = _fmt_float(gpu.power_w, decimals=0, suffix="W")
        if gpu.power_cap_w is not None:
            power_str += f"/{_fmt_float(gpu.power_cap_w, decimals=0, suffix='W')}"
        clock_str = _fmt_int(gpu.clock_sm_mhz, suffix=" MHz")

        throttle_flag = " [THROTTLED]" if gpu.throttled else ""

        return (
            f"GPU{gpu.index} {name}  util={util_str}  mem={mem_str}"
            f"  temp={temp_str}  pwr={power_str}  sm={clock_str}{throttle_flag}"
        )

    def _rebuild_gpu_display(self, snap: Snapshot) -> None:
        """Recompose internal static text for each GPU."""
        if not snap.gpus:
            try:
                placeholder = self.query_one("#gpu-placeholder", Static)
                placeholder.update("No GPU detected")
            except Exception:
                pass
            return

        # Hide placeholder if GPUs present
        try:
            placeholder = self.query_one("#gpu-placeholder", Static)
            placeholder.display = False
        except Exception:
            pass

        for gpu in snap.gpus:
            widget_id = f"gpu-line-{gpu.index}"
            line_text = self._gpu_line(gpu)

            rich_line = Text(line_text)
            # Color the util% portion red/yellow/green
            try:
                existing = self.query_one(f"#{widget_id}", Static)
                existing.update(rich_line)
            except Exception:
                # Widget doesn't exist yet — mount it
                new_static = Static(rich_line, id=widget_id)
                self.mount(new_static)

            # Update sparkline if it exists
            spark_id = f"gpu-spark-{gpu.index}"
            hist = self._get_util_history(gpu.index)
            try:
                spark = self.query_one(f"#{spark_id}", Sparkline)
                spark.data = list(hist)
            except Exception:
                # Mount a new sparkline
                spark = Sparkline(
                    list(hist),
                    id=spark_id,
                    min_color="green",
                    max_color="red",
                )
                self.mount(spark)


# ---------------------------------------------------------------------------
# Engine Table
# ---------------------------------------------------------------------------

# Sort column cycle order
_SORT_COLUMNS = ["Engine", "Port", "tok/s", "KV%", "VRAM", "Reqs"]

# Column widths (approximate characters)
_COL_WIDTHS = {
    "Engine": 20,
    "Model": 22,
    "Port": 6,
    "PID": 7,
    "tok/s": 7,
    "Reqs": 8,
    "KV%": 6,
    "VRAM": 10,
    "Uptime": 8,
}


class EngineTable(Widget):
    """Sortable DataTable showing all discovered engines.

    Routers are shown with their backends indented beneath them.
    Color-coded:
    - KV% > 85 → red
    - queue > 0 → yellow
    - throttled → row tinted red (via engine info from GPU data)
    """

    DEFAULT_CSS = """
    EngineTable {
        height: 1fr;
        width: 1fr;
    }
    EngineTable DataTable {
        height: 1fr;
        width: 1fr;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._sort_col_idx: int = 0  # index into _SORT_COLUMNS
        self._sort_reverse: bool = False
        self._filter: str = ""
        self._row_keys: list[str] = []  # ordered engine keys matching table rows
        self._last_snap: Optional[Snapshot] = None
        self._selected_key: Optional[str] = None
        self._throttled_pids: set[int] = set()  # cached so filtering keeps highlights

    def compose(self) -> ComposeResult:
        table: DataTable[Text] = DataTable(id="engine-datatable", zebra_stripes=True)
        yield table

    def on_mount(self) -> None:
        """Set up columns after mount."""
        table = self.query_one("#engine-datatable", DataTable)
        for col, width in _COL_WIDTHS.items():
            table.add_column(col, width=width, key=col)
        table.cursor_type = "row"

    def update_snapshot(self, snap: Snapshot, *, throttled_pids: set[int]) -> None:
        """Refresh the engine table from a new snapshot.

        ``throttled_pids`` is the set of pids whose GPU is currently throttled.
        """
        self._last_snap = snap
        self._throttled_pids = throttled_pids
        table = self.query_one("#engine-datatable", DataTable)

        # Flatten engine list into display rows (routers first, then their backends
        # indented, then standalone engines)
        rows = self._build_rows(snap, throttled_pids=throttled_pids)

        # Track current cursor row so we can restore it
        cursor_row = table.cursor_row

        # Clear and repopulate (simplest for correctness; table is typically small)
        table.clear()
        self._row_keys = []

        for row_data in rows:
            key, cells, style = row_data
            styled_cells = [Text(str(c), style=style) if style else Text(str(c)) for c in cells]
            # Apply per-cell coloring
            self._apply_cell_colors(styled_cells, cells)
            table.add_row(*styled_cells, key=key)
            self._row_keys.append(key)

        # Restore cursor position (clamp to valid range)
        if self._row_keys:
            new_row = min(cursor_row, len(self._row_keys) - 1)
            if new_row >= 0:
                table.move_cursor(row=new_row)

    def _apply_cell_colors(self, styled: list[Text], raw: list[str]) -> None:
        """Apply color coding to KV%, tok/s, Reqs cells in place."""
        col_names = list(_COL_WIDTHS.keys())
        for i, (text_obj, raw_val) in enumerate(zip(styled, raw)):
            col = col_names[i] if i < len(col_names) else ""
            if col == "KV%":
                try:
                    val = float(raw_val.rstrip("%"))
                    if val > 85:
                        text_obj.stylize("bold red")
                    elif val > 70:
                        text_obj.stylize("yellow")
                except (ValueError, AttributeError):
                    pass
            elif col == "Reqs":
                # Format is "run/wait" — highlight if wait > 0
                if "/" in raw_val:
                    parts = raw_val.split("/")
                    try:
                        wait = int(parts[1])
                        if wait > 0:
                            text_obj.stylize("yellow")
                    except (ValueError, IndexError):
                        pass

    def _build_rows(
        self, snap: Snapshot, *, throttled_pids: set[int]
    ) -> list[tuple[str, list[str], str]]:
        """Build a flat list of (key, [cells], rich_style) tuples."""
        rows: list[tuple[str, list[str], str]] = []

        # Partition: routers, backends of routers, standalone
        router_backend_keys: set[str] = set()
        for e in snap.engines:
            if e.is_router:
                for b in e.backends:
                    router_backend_keys.add(b.key)

        def _engine_row(
            e: EngineInfo, indent: bool = False
        ) -> tuple[str, list[str], str]:
            prefix = "  " if indent else ""
            engine_label = prefix + _engine_label(e)

            model_str = e.primary_model or "n/a"
            if len(model_str) > 20:
                model_str = model_str[:18] + ".."

            port_str = str(e.port)
            pid_str = str(e.pid) if e.pid is not None else "n/a"

            m = e.metrics
            toks_str = _fmt_float(m.decode_tps, suffix="")
            reqs_run = _fmt_int(m.requests_running)
            reqs_wait = _fmt_int(m.requests_waiting)
            reqs_str = f"{reqs_run}/{reqs_wait}"
            kv_str = _fmt_float(m.kv_cache_pct, suffix="%") if m.kv_cache_pct is not None else "n/a"
            vram_str = _fmt_bytes(e.vram_bytes)
            uptime_str = _uptime_str(e.uptime_s)

            # Row style for throttled GPU engines
            style = ""
            if e.pid is not None and e.pid in throttled_pids:
                style = "bold red"

            cells = [
                engine_label,
                model_str,
                port_str,
                pid_str,
                toks_str,
                reqs_str,
                kv_str,
                vram_str,
                uptime_str,
            ]
            return (e.key, cells, style)

        # Apply filter
        filt = self._filter.lower()

        def _matches(e: EngineInfo) -> bool:
            if not filt:
                return True
            haystack = (
                _engine_label(e)
                + (e.primary_model or "")
                + str(e.port)
                + e.engine_type.value
            ).lower()
            return filt in haystack

        # Top-level engines = routers + standalone (backends render nested under
        # their router, never at the top level). Sort that list by the active
        # column; backends stay attached to their router.
        top_level = [e for e in snap.engines if e.key not in router_backend_keys]
        col = _SORT_COLUMNS[self._sort_col_idx % len(_SORT_COLUMNS)]
        top_level.sort(key=lambda e: self._sort_key(e, col), reverse=self._sort_reverse)

        for e in top_level:
            backend_match = e.is_router and any(_matches(b) for b in e.backends)
            if not _matches(e) and not backend_match:
                continue
            rows.append(_engine_row(e, indent=False))
            if e.is_router:
                for b in e.backends:
                    if _matches(b):
                        rows.append(_engine_row(b, indent=True))

        return rows

    @staticmethod
    def _sort_key(e: EngineInfo, col: str):
        """Sort key for an engine row under the active column. None-valued
        metrics sort below real values (ascending)."""
        m = e.metrics
        if col == "Port":
            return e.port
        if col == "tok/s":
            return m.decode_tps if m.decode_tps is not None else -1.0
        if col == "KV%":
            return m.kv_cache_pct if m.kv_cache_pct is not None else -1.0
        if col == "VRAM":
            return e.vram_bytes if e.vram_bytes is not None else -1
        if col == "Reqs":
            return (m.requests_running or 0) + (m.requests_waiting or 0)
        # Default / "Engine": case-insensitive name.
        return _engine_label(e).lower()

    def cycle_sort(self) -> None:
        """Advance to next sort column."""
        self._sort_col_idx = (self._sort_col_idx + 1) % len(_SORT_COLUMNS)
        col = _SORT_COLUMNS[self._sort_col_idx]
        self.notify(f"Sort: {col}")

    def set_filter(self, text: str) -> None:
        """Set filter text and re-render, preserving the throttle highlights."""
        self._filter = text
        if self._last_snap:
            self.update_snapshot(self._last_snap, throttled_pids=self._throttled_pids)

    def get_selected_engine(self, snap: Optional[Snapshot]) -> Optional[EngineInfo]:
        """Return the EngineInfo for the currently highlighted row."""
        if snap is None:
            return None
        table = self.query_one("#engine-datatable", DataTable)
        row = table.cursor_row
        if row < 0 or row >= len(self._row_keys):
            return None
        key = self._row_keys[row]
        # Search snap engines (including backends)
        for e in snap.engines:
            if e.key == key:
                return e
            for b in e.backends:
                if b.key == key:
                    return b
        return None


# ---------------------------------------------------------------------------
# Detail Pane
# ---------------------------------------------------------------------------


class DetailPane(Widget):
    """Detail pane shown when the user presses Enter on an engine row.

    Shows: full engine flags/config, all models, raw metrics dict, plus a
    per-engine tok/s sparkline.
    """

    DEFAULT_CSS = """
    DetailPane {
        height: 10;
        width: 1fr;
        border: solid $accent;
        padding: 0 1;
        display: none;
    }
    DetailPane.visible {
        display: block;
    }
    DetailPane #detail-spark {
        height: 3;
        width: 1fr;
    }
    DetailPane #detail-content {
        height: 1fr;
        width: 1fr;
    }
    """

    visible_state: reactive[bool] = reactive(False)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._engine: Optional[EngineInfo] = None
        self._toks_history: collections.deque[float] = collections.deque(
            [0.0] * _HISTORY_LEN, maxlen=_HISTORY_LEN
        )

    def compose(self) -> ComposeResult:
        yield Static("", id="detail-content")
        yield Sparkline(
            list(self._toks_history),
            id="detail-spark",
            min_color="blue",
            max_color="cyan",
        )

    def show_engine(self, engine: Optional[EngineInfo]) -> None:
        """Update detail pane with a new engine selection."""
        if engine is None:
            self._engine = None
            self.remove_class("visible")
            return

        self._engine = engine
        self.add_class("visible")

        # Update tok/s sparkline
        tps = engine.metrics.decode_tps
        self._toks_history.append(tps if tps is not None else 0.0)
        try:
            spark = self.query_one("#detail-spark", Sparkline)
            spark.data = list(self._toks_history)
        except Exception:
            pass

        # Build detail text
        lines: list[str] = []
        lines.append(f"[bold]Engine:[/] {_engine_label(engine)}  type={engine.engine_type.value}")
        lines.append(f"[bold]URL:[/] {engine.base_url}  version={engine.version or 'n/a'}")
        lines.append(f"[bold]PID:[/] {engine.pid or 'n/a'}  uptime={_uptime_str(engine.uptime_s)}")
        if engine.last_error:
            lines.append(f"[bold red]Error:[/] {engine.last_error}")

        # Models
        if engine.models:
            lines.append("[bold]Models:[/]")
            for m in engine.models[:8]:
                loaded_flag = " [loaded]" if m.loaded else ""
                ctx = f"  ctx={m.context_length}" if m.context_length else ""
                quant = f"  quant={m.quantization}" if m.quantization else ""
                dtype = f"  dtype={m.dtype}" if m.dtype else ""
                lines.append(f"  {m.id}{loaded_flag}{ctx}{quant}{dtype}")
            if len(engine.models) > 8:
                lines.append(f"  ... and {len(engine.models) - 8} more")

        # Metrics
        me = engine.metrics
        lines.append("[bold]Metrics:[/]")
        lines.append(
            f"  decode_tps={_fmt_float(me.decode_tps, 2)}"
            f"  prefill_tps={_fmt_float(me.prefill_tps, 2)}"
        )
        lines.append(
            f"  running={_fmt_int(me.requests_running)}"
            f"  waiting={_fmt_int(me.requests_waiting)}"
            f"  kv={_fmt_float(me.kv_cache_pct, 1, '%')}"
        )
        lines.append(
            f"  tokens_total={_fmt_int(me.tokens_total)}"
            f"  prompt_total={_fmt_int(me.prompt_tokens_total)}"
        )
        if me.error:
            lines.append(f"  [red]metrics error:[/] {me.error}")

        # Flags / config
        if engine.flags:
            lines.append("[bold]Config flags:[/]")
            for k, v in list(engine.flags.items())[:10]:
                lines.append(f"  {k}={v}")

        # Raw metrics (first 6 entries)
        if me.raw:
            lines.append("[bold]Raw metrics (sample):[/]")
            for k, v in list(me.raw.items())[:6]:
                lines.append(f"  {k}={v}")

        # Backends
        if engine.is_router and engine.backends:
            lines.append(f"[bold]Backends ({len(engine.backends)}):[/]")
            for b in engine.backends:
                lines.append(f"  {_engine_label(b)}  {b.base_url}")

        content_text = "\n".join(lines)
        try:
            content = self.query_one("#detail-content", Static)
            content.update(content_text)
        except Exception:
            pass

    def toggle(self, engine: Optional[EngineInfo] = None) -> None:
        """Toggle detail pane visibility."""
        if self.has_class("visible") and engine is None:
            self.remove_class("visible")
            self._engine = None
        else:
            self.show_engine(engine)

    def refresh_engine(self, engine: Optional[EngineInfo]) -> None:
        """Refresh with updated engine data if the pane is currently visible."""
        if self.has_class("visible") and engine is not None:
            self.show_engine(engine)


# ---------------------------------------------------------------------------
# Event Log Widget
# ---------------------------------------------------------------------------


class EventLogWidget(Widget):
    """Scrolling event log fed from Snapshot.events and Snapshot.errors.

    Shows the last N events in reverse-chronological order (newest at bottom).
    """

    DEFAULT_CSS = """
    EventLogWidget {
        height: 4;
        width: 1fr;
        border: solid $panel;
    }
    EventLogWidget RichLog {
        height: 1fr;
        width: 1fr;
    }
    """

    def __init__(self, max_lines: int = 100, **kwargs) -> None:
        super().__init__(**kwargs)
        self._max_lines = max_lines

    def compose(self) -> ComposeResult:
        yield RichLog(max_lines=self._max_lines, markup=True, id="event-richlog")

    def append_events(self, snap: Snapshot) -> None:
        """Append new events and errors from the snapshot to the log."""
        try:
            log = self.query_one("#event-richlog", RichLog)
        except Exception:
            return

        ts = time.strftime("%H:%M:%S", time.localtime(snap.timestamp))

        for ev in snap.events:
            log.write(f"[dim]{ts}[/dim] [bold cyan]EVENT[/] {ev}")

        for err in snap.errors:
            log.write(f"[dim]{ts}[/dim] [bold red]ERROR[/] {err}")

    def write_message(self, msg: str) -> None:
        """Write an arbitrary message to the log (e.g. keybind confirmations)."""
        try:
            log = self.query_one("#event-richlog", RichLog)
            log.write(msg)
        except Exception:
            pass
