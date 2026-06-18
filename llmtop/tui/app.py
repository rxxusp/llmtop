"""llmtop Textual TUI application.

The :class:`LlmtopApp` is the root of the Textual widget tree. It:

1. Accepts a ``Monitor`` instance (from :mod:`llmtop.core`) and an interval.
2. Runs a periodic background worker that calls
   ``await asyncio.to_thread(monitor.poll)`` so the event loop is never blocked.
3. Distributes each new :class:`~llmtop.models.Snapshot` to the child widgets
   :class:`~llmtop.tui.widgets.GpuHeaderWidget`,
   :class:`~llmtop.tui.widgets.EngineTable`,
   :class:`~llmtop.tui.widgets.DetailPane`, and
   :class:`~llmtop.tui.widgets.EventLogWidget`.

Keybinds:
  q       quit
  p       pause/resume polling
  s       cycle sort column
  f       open filter input
  enter   toggle detail pane for selected row
  r       force rediscovery on next poll
  ?       show help screen
"""

from __future__ import annotations

import time
from typing import Optional

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import DataTable, Footer, Header, Input, Static

from ..models import EngineInfo, Snapshot
from .widgets import DetailPane, EngineTable, EventLogWidget, GpuHeaderWidget

# ---------------------------------------------------------------------------
# Help screen
# ---------------------------------------------------------------------------

_HELP_TEXT = """\
[bold cyan]llmtop keybindings[/bold cyan]

  [bold]q[/]        Quit
  [bold]p[/]        Pause / resume polling
  [bold]s[/]        Cycle sort column
  [bold]f[/]        Open filter prompt
  [bold]Enter[/]    Toggle detail pane for selected engine
  [bold]r[/]        Force re-discovery on next poll
  [bold]?[/]        This help screen
  [bold]Escape[/]   Close filter / detail pane / this screen

  Colour coding:
    [red]Red[/]     KV cache > 85 % or GPU throttled
    [yellow]Yellow[/]  Queue depth > 0 (requests waiting)
    [green]Green[/]   Nominal

  'n/a' indicates metric not available or not supported.

  Press any key to close.
"""


class HelpScreen(ModalScreen[None]):
    """A modal overlay showing keybind help."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("question_mark", "dismiss", "Close"),
        Binding("q", "dismiss", "Close"),
    ]

    def compose(self) -> ComposeResult:
        yield Static(_HELP_TEXT, id="help-text", markup=True)

    def on_key(self) -> None:
        """Any key closes the help screen."""
        self.dismiss()


# ---------------------------------------------------------------------------
# Filter bar
# ---------------------------------------------------------------------------


class FilterBar(Widget):
    """A one-line input bar for filtering the engine table."""

    DEFAULT_CSS = """
    FilterBar {
        height: 1;
        width: 1fr;
        display: none;
    }
    FilterBar.active {
        display: block;
    }
    FilterBar Input {
        width: 1fr;
        height: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Input(placeholder="filter engines…", id="filter-input")

    def activate(self) -> None:
        """Show the filter bar and focus the input."""
        self.add_class("active")
        try:
            self.query_one("#filter-input", Input).focus()
        except Exception:
            pass

    def deactivate(self) -> None:
        """Hide the filter bar and clear it."""
        self.remove_class("active")
        try:
            inp = self.query_one("#filter-input", Input)
            inp.value = ""
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Status bar
# ---------------------------------------------------------------------------


class StatusBar(Static):
    """One-line status indicator: paused/running + last-poll age."""

    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        width: 1fr;
        color: $text-muted;
        text-align: right;
        padding: 0 1;
    }
    """

    def update_status(
        self,
        paused: bool,
        last_poll: Optional[float],
        engines: int,
        errors: int,
    ) -> None:
        """Refresh status bar text."""
        state = "[bold yellow]PAUSED[/]" if paused else "[bold green]LIVE[/]"
        age = "never" if last_poll is None else f"{time.time() - last_poll:.1f}s ago"
        err_part = f"  [red]{errors} error(s)[/]" if errors else ""
        self.update(
            f"{state}  engines={engines}  last poll={age}{err_part}"
        )


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------


class LlmtopApp(App[None]):
    """Textual TUI for llmtop.

    Parameters
    ----------
    monitor:
        A :class:`~llmtop.core.Monitor` instance (or any object with a
        ``poll() -> Snapshot`` method for testing).
    interval:
        Polling interval in seconds (default 2.0).
    """

    TITLE = "llmtop"
    SUB_TITLE = "nvtop for local LLM inference"

    CSS = """
    Screen {
        layout: vertical;
    }
    #gpu-header-container {
        height: auto;
        max-height: 6;
        width: 1fr;
        border-bottom: solid $panel;
    }
    #main-split {
        height: 1fr;
        width: 1fr;
    }
    #engine-table-container {
        height: 1fr;
        width: 1fr;
    }
    #detail-pane {
        height: 12;
        width: 1fr;
    }
    #event-log-container {
        height: 5;
        width: 1fr;
        border-top: solid $panel;
    }
    #status-bar {
        height: 1;
    }
    #filter-bar {
        height: 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("p", "toggle_pause", "Pause/Resume"),
        Binding("s", "cycle_sort", "Sort"),
        Binding("f", "open_filter", "Filter"),
        Binding("enter", "toggle_detail", "Detail", show=False),
        Binding("r", "force_rediscover", "Rediscover"),
        Binding("question_mark", "show_help", "Help"),
        Binding("escape", "handle_escape", "Escape", show=False),
    ]

    # Reactive: whether polling is paused
    _paused: reactive[bool] = reactive(False)

    def __init__(self, monitor: object, interval: float = 2.0) -> None:
        super().__init__()
        self._monitor = monitor
        self._interval = max(0.1, interval)
        self._last_snap: Optional[Snapshot] = None
        self._last_poll_time: Optional[float] = None
        self._error_count: int = 0
        self._filter_active: bool = False
        self._detail_engine: Optional[EngineInfo] = None
        self._force_rediscover: bool = False
        self._poll_timer = None

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        # GPU header band
        with Vertical(id="gpu-header-container"):
            yield GpuHeaderWidget(id="gpu-header")

        # Main content area: engine table + optional detail pane
        with Vertical(id="main-split"):
            with Vertical(id="engine-table-container"):
                yield EngineTable(id="engine-table")
            yield DetailPane(id="detail-pane")

        # Filter bar (hidden unless active)
        yield FilterBar(id="filter-bar")

        # Event log
        with Vertical(id="event-log-container"):
            yield EventLogWidget(id="event-log")

        # Status bar
        yield StatusBar(id="status-bar")

        yield Footer()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        """Start the polling loop after the widget tree is mounted."""
        self._start_polling()

    def _start_polling(self) -> None:
        """Launch the periodic poll worker."""
        self._poll_timer = self.set_interval(self._interval, self._tick)

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    async def _tick(self) -> None:
        """Called by set_interval every ``_interval`` seconds."""
        if self._paused:
            return
        self._do_poll()

    @work(thread=True, group="poll", exclusive=True)
    def _do_poll(self) -> None:
        """Run monitor.poll() in a worker thread to avoid blocking the event loop."""
        try:
            # Support both monitor objects and simple poll() interfaces
            monitor = self._monitor
            if self._force_rediscover:
                # Ask the monitor to run a full rescan on this poll, if supported.
                force = getattr(monitor, "force_rediscover", None)
                if callable(force):
                    force()
                self._force_rediscover = False

            snap: Snapshot = monitor.poll()  # type: ignore[union-attr]
            # Post the snapshot back to the main thread
            self.call_from_thread(self._apply_snapshot, snap)
        except Exception as exc:
            self.call_from_thread(
                self._apply_error, f"Poll error: {exc}"
            )

    def _apply_snapshot(self, snap: Snapshot) -> None:
        """Apply a fresh snapshot to all widgets (runs on the main thread)."""
        self._last_snap = snap
        self._last_poll_time = time.time()
        self._error_count = len(snap.errors)

        # Determine throttled pids
        throttled_pids: set[int] = set()
        for gpu in snap.gpus:
            if gpu.throttled:
                for proc in gpu.procs:
                    throttled_pids.add(proc.pid)

        # Update GPU header
        try:
            gpu_header = self.query_one("#gpu-header", GpuHeaderWidget)
            gpu_header.update_snapshot(snap)
        except Exception:
            pass

        # Update engine table
        try:
            table = self.query_one("#engine-table", EngineTable)
            table.update_snapshot(snap, throttled_pids=throttled_pids)
        except Exception:
            pass

        # Refresh detail pane if visible
        try:
            detail = self.query_one("#detail-pane", DetailPane)
            if detail.has_class("visible") and self._detail_engine is not None:
                # Find updated engine
                updated = self._find_engine(snap, self._detail_engine.key)
                if updated:
                    self._detail_engine = updated
                    detail.refresh_engine(updated)
        except Exception:
            pass

        # Append events to log
        try:
            event_log = self.query_one("#event-log", EventLogWidget)
            event_log.append_events(snap)
        except Exception:
            pass

        # Update status bar
        try:
            status = self.query_one("#status-bar", StatusBar)
            status.update_status(
                paused=self._paused,
                last_poll=self._last_poll_time,
                engines=len(snap.engines),
                errors=self._error_count,
            )
        except Exception:
            pass

    def _apply_error(self, message: str) -> None:
        """Display a poll error in the event log."""
        try:
            event_log = self.query_one("#event-log", EventLogWidget)
            event_log.write_message(f"[bold red]ERROR[/] {message}")
        except Exception:
            pass
        try:
            status = self.query_one("#status-bar", StatusBar)
            status.update_status(
                paused=self._paused,
                last_poll=self._last_poll_time,
                engines=0,
                errors=1,
            )
        except Exception:
            pass

    def _find_engine(self, snap: Snapshot, key: str) -> Optional[EngineInfo]:
        """Find an EngineInfo by key in a snapshot (including router backends)."""
        for e in snap.engines:
            if e.key == key:
                return e
            for b in e.backends:
                if b.key == key:
                    return b
        return None

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()

    def action_toggle_pause(self) -> None:
        """Toggle polling pause state."""
        self._paused = not self._paused
        try:
            event_log = self.query_one("#event-log", EventLogWidget)
            msg = "Polling [bold yellow]paused[/]" if self._paused else "Polling [bold green]resumed[/]"
            event_log.write_message(msg)
        except Exception:
            pass
        # Refresh status bar
        try:
            status = self.query_one("#status-bar", StatusBar)
            status.update_status(
                paused=self._paused,
                last_poll=self._last_poll_time,
                engines=len(self._last_snap.engines) if self._last_snap else 0,
                errors=self._error_count,
            )
        except Exception:
            pass

    def action_cycle_sort(self) -> None:
        """Cycle sort column in the engine table."""
        try:
            table = self.query_one("#engine-table", EngineTable)
            table.cycle_sort()
        except Exception:
            pass

    def action_open_filter(self) -> None:
        """Show the filter input bar."""
        self._filter_active = True
        try:
            fb = self.query_one("#filter-bar", FilterBar)
            fb.activate()
        except Exception:
            pass

    def action_toggle_detail(self) -> None:
        """Toggle the detail pane for the currently selected engine."""
        try:
            detail = self.query_one("#detail-pane", DetailPane)
            if detail.has_class("visible"):
                # Hide
                detail.toggle(None)
                self._detail_engine = None
            else:
                # Show for selected row
                engine = self._get_selected_engine()
                if engine:
                    self._detail_engine = engine
                    detail.show_engine(engine)
        except Exception:
            pass

    def action_force_rediscover(self) -> None:
        """Flag that the next poll should force rediscovery."""
        self._force_rediscover = True
        try:
            event_log = self.query_one("#event-log", EventLogWidget)
            event_log.write_message("[bold cyan]INFO[/] Force rediscovery scheduled")
        except Exception:
            pass

    def action_show_help(self) -> None:
        """Push the help overlay screen."""
        self.push_screen(HelpScreen())

    def action_handle_escape(self) -> None:
        """Escape: close filter bar if active, else close detail pane."""
        if self._filter_active:
            self._filter_active = False
            try:
                fb = self.query_one("#filter-bar", FilterBar)
                fb.deactivate()
                # Clear filter in table
                table = self.query_one("#engine-table", EngineTable)
                table.set_filter("")
                # Re-focus the engine table
                dt = table.query_one("#engine-datatable", DataTable)
                dt.focus()
            except Exception:
                pass
        else:
            try:
                detail = self.query_one("#detail-pane", DetailPane)
                if detail.has_class("visible"):
                    detail.toggle(None)
                    self._detail_engine = None
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Input events
    # ------------------------------------------------------------------

    def on_input_changed(self, event: Input.Changed) -> None:
        """Live-filter engine table as the user types in the filter bar."""
        if event.input.id == "filter-input":
            try:
                table = self.query_one("#engine-table", EngineTable)
                table.set_filter(event.value)
            except Exception:
                pass

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Submit filter (Enter in filter bar) → return focus to table."""
        if event.input.id == "filter-input":
            self._filter_active = False
            try:
                fb = self.query_one("#filter-bar", FilterBar)
                fb.deactivate()
                dt = self.query_one("#engine-datatable", DataTable)
                dt.focus()
            except Exception:
                pass

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """If detail pane is open, refresh it for the newly highlighted row."""
        try:
            detail = self.query_one("#detail-pane", DetailPane)
            if detail.has_class("visible"):
                engine = self._get_selected_engine()
                if engine:
                    self._detail_engine = engine
                    detail.refresh_engine(engine)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_selected_engine(self) -> Optional[EngineInfo]:
        """Return the EngineInfo for the currently highlighted table row."""
        try:
            table = self.query_one("#engine-table", EngineTable)
            return table.get_selected_engine(self._last_snap)
        except Exception:
            return None
