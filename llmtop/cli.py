"""CLI entry-point for llmtop.

Provides the ``main`` function (wired as the ``llmtop`` console script) and the
``snapshot_to_dict`` helper used by ``--json`` mode.

Usage modes:
- Default: launch the Textual TUI (requires a TTY).
- ``--json``: print one :class:`~llmtop.models.Snapshot` as indented JSON to
  stdout and exit 0. Works headless with no TTY.
- (future) ``--once``: print a human-readable table and exit.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import traceback
from typing import Any

from . import __version__


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------

def snapshot_to_dict(snap: Any) -> dict[str, Any]:
    """Convert a :class:`~llmtop.models.Snapshot` to a plain dict.

    Uses ``dataclasses.asdict`` so nested dataclasses are handled recursively.
    Post-processes the result to:

    - Convert :class:`~llmtop.models.EngineType` (and any other ``Enum``)
      values to their ``.value`` string.
    - Convert ``tuple`` values to ``list`` (e.g. ``load_avg``).
    """
    raw = dataclasses.asdict(snap)
    return _normalise(raw)


def _normalise(obj: Any) -> Any:
    """Recursively normalise Enum→value and tuple→list in an asdict output."""
    import enum  # stdlib, always available

    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _normalise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalise(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmtop",
        description=(
            "nvtop for local LLM inference. Autodiscovers vLLM, llama.cpp, "
            "Ollama, TGI, SGLang, and generic OpenAI-compatible servers and "
            "displays live GPU + serving metrics."
        ),
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"llmtop {__version__}",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help=(
            "Print one snapshot as JSON to stdout and exit. "
            "Works headless with no TTY."
        ),
    )

    parser.add_argument(
        "--once",
        action="store_true",
        default=False,
        help=(
            "Print a one-shot human-readable summary table and exit "
            "(same data as --json, different format)."
        ),
    )

    parser.add_argument(
        "--interval",
        metavar="FLOAT",
        type=float,
        default=2.0,
        help="Poll interval in seconds for TUI refresh (default: %(default)s).",
    )

    parser.add_argument(
        "--port",
        metavar="INT",
        type=int,
        action="append",
        dest="extra_ports",
        default=[],
        help=(
            "Additional port to include in discovery (repeatable). "
            "These are added on top of the default port list."
        ),
    )

    parser.add_argument(
        "--no-gpu",
        action="store_true",
        default=False,
        help="Disable GPU sampling (useful on non-NVIDIA hosts).",
    )

    parser.add_argument(
        "--timeout",
        metavar="FLOAT",
        type=float,
        default=1.0,
        help="HTTP request timeout in seconds (default: %(default)s).",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Show full tracebacks instead of abbreviated error messages.",
    )

    return parser


# ---------------------------------------------------------------------------
# --json headless path
# ---------------------------------------------------------------------------

def _run_json(args: argparse.Namespace) -> int:
    """Collect one snapshot and print it as JSON. Returns an exit code."""
    try:
        from .core import Monitor  # type: ignore[import]
    except Exception as exc:  # noqa: BLE001
        msg = f"Failed to import Monitor: {exc}"
        if args.debug:
            traceback.print_exc(file=sys.stderr)
        else:
            print(msg, file=sys.stderr)
        return 1

    monitor: Any = None
    try:
        monitor = Monitor(
            interval=args.interval,
            timeout=args.timeout,
            extra_ports=args.extra_ports,
            enable_gpu=not args.no_gpu,
        )
        snap = monitor.poll()
        data = snapshot_to_dict(snap)
        print(json.dumps(data, indent=2, default=str))
        return 0
    except Exception as exc:  # noqa: BLE001
        msg = f"Error collecting snapshot: {exc}"
        if args.debug:
            traceback.print_exc(file=sys.stderr)
        else:
            print(msg, file=sys.stderr)
        return 1
    finally:
        if monitor is not None:
            try:
                monitor.close()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# --once human-table path
# ---------------------------------------------------------------------------

def _run_once(args: argparse.Namespace) -> int:
    """Print a one-shot human-readable table. Returns an exit code."""
    try:
        from .core import Monitor  # type: ignore[import]
    except Exception as exc:  # noqa: BLE001
        msg = f"Failed to import Monitor: {exc}"
        if args.debug:
            traceback.print_exc(file=sys.stderr)
        else:
            print(msg, file=sys.stderr)
        return 1

    monitor: Any = None
    try:
        monitor = Monitor(
            interval=args.interval,
            timeout=args.timeout,
            extra_ports=args.extra_ports,
            enable_gpu=not args.no_gpu,
        )
        snap = monitor.poll()
        _print_table(snap)
        return 0
    except Exception as exc:  # noqa: BLE001
        msg = f"Error collecting snapshot: {exc}"
        if args.debug:
            traceback.print_exc(file=sys.stderr)
        else:
            print(msg, file=sys.stderr)
        return 1
    finally:
        if monitor is not None:
            try:
                monitor.close()
            except Exception:  # noqa: BLE001
                pass


def _fmt(value: Any, unit: str = "", precision: int = 1) -> str:
    """Format a potentially-None value, appending unit when not None."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{precision}f}{unit}"
    return f"{value}{unit}"


def _print_table(snap: Any) -> None:
    """Print a minimal ASCII table of the snapshot to stdout."""
    import time

    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(snap.timestamp))
    print(f"llmtop snapshot: {ts}")
    print()

    # GPU section
    if snap.gpus:
        print("GPUs")
        hdr = f"  {'#':>2}  {'Name':<30}  {'Util':>6}  {'Mem':>18}  {'Temp':>6}  {'Power':>8}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for g in snap.gpus:
            mem_str: str
            if g.mem_used_bytes is not None and g.mem_total_bytes:
                used_gb = g.mem_used_bytes / 1024 ** 3
                total_gb = g.mem_total_bytes / 1024 ** 3
                mem_str = f"{used_gb:.1f}/{total_gb:.1f} GB"
                if g.unified_memory:
                    mem_str += " (unified)"
            else:
                mem_str = "n/a"
            print(
                f"  {g.index:>2}  {g.name:<30}  "
                f"{_fmt(g.util_pct, '%'):>6}  "
                f"{mem_str:>18}  "
                f"{_fmt(g.temp_c, '°C'):>6}  "
                f"{_fmt(g.power_w, 'W'):>8}"
            )
        print()
    else:
        print("GPUs: none detected")
        print()

    # System section
    sys_s = snap.system
    cpu_str = _fmt(sys_s.cpu_pct, "%") if sys_s.cpu_pct is not None else "n/a"
    ram_str: str
    if sys_s.ram_used_bytes is not None and sys_s.ram_total_bytes:
        ru = sys_s.ram_used_bytes / 1024 ** 3
        rt = sys_s.ram_total_bytes / 1024 ** 3
        ram_str = f"{ru:.1f}/{rt:.1f} GB"
    else:
        ram_str = "n/a"
    print(f"System  CPU: {cpu_str}  RAM: {ram_str}")
    if sys_s.load_avg is not None:
        la = sys_s.load_avg
        print(f"        Load avg: {la[0]:.2f} {la[1]:.2f} {la[2]:.2f}")
    print()

    # Engines section
    if snap.engines:
        print("Engines")
        col_w = [24, 36, 6, 7, 8, 12, 6, 10]
        cols = ["Engine", "Model", "Port", "PID", "tok/s", "reqs(r/w)", "KV%", "Uptime"]
        hdr_line = "  " + "  ".join(f"{c:<{w}}" for c, w in zip(cols, col_w))
        print(hdr_line)
        print("  " + "-" * (len(hdr_line) - 2))
        for eng in snap.engines:
            prefix = "  "
            _print_engine_row(eng, col_w, prefix)
            for backend in eng.backends:
                _print_engine_row(backend, col_w, "    ")
    else:
        print("Engines: none discovered")

    if snap.errors:
        print()
        print("Errors:")
        for err in snap.errors:
            print(f"  {err}")

    if snap.events:
        print()
        print("Events:")
        for ev in snap.events:
            print(f"  {ev}")


def _print_engine_row(eng: Any, col_w: list[int], prefix: str) -> None:
    """Print one engine as a table row."""
    name = eng.engine_type.value if hasattr(eng.engine_type, "value") else str(eng.engine_type)
    model = eng.primary_model or "n/a"
    port = str(eng.port)
    pid = str(eng.pid) if eng.pid is not None else "n/a"
    m = eng.metrics
    tok_s = _fmt(m.decode_tps, "", precision=1) if m.decode_tps is not None else "n/a"
    if m.requests_running is not None or m.requests_waiting is not None:
        rr = m.requests_running if m.requests_running is not None else "?"
        rw = m.requests_waiting if m.requests_waiting is not None else "?"
        reqs = f"{rr}/{rw}"
    else:
        reqs = "n/a"
    kv = _fmt(m.kv_cache_pct, "%", precision=0) if m.kv_cache_pct is not None else "n/a"
    if eng.uptime_s is not None:
        up_s = int(eng.uptime_s)
        if up_s >= 3600:
            uptime = f"{up_s // 3600}h{(up_s % 3600) // 60}m"
        elif up_s >= 60:
            uptime = f"{up_s // 60}m{up_s % 60}s"
        else:
            uptime = f"{up_s}s"
    else:
        uptime = "n/a"

    vals = [name, model, port, pid, tok_s, reqs, kv, uptime]
    row = prefix + "  ".join(f"{v:<{w}}" for v, w in zip(vals, col_w))
    print(row)


# ---------------------------------------------------------------------------
# TUI launch path
# ---------------------------------------------------------------------------

def _run_tui(args: argparse.Namespace) -> int:
    """Construct Monitor and launch the Textual TUI. Returns an exit code."""
    try:
        from .core import Monitor  # type: ignore[import]
    except Exception as exc:  # noqa: BLE001
        msg = f"Failed to import Monitor: {exc}"
        if args.debug:
            traceback.print_exc(file=sys.stderr)
        else:
            print(msg, file=sys.stderr)
        return 1

    try:
        from .tui.app import LlmtopApp  # type: ignore[import]
    except Exception as exc:  # noqa: BLE001
        msg = f"Failed to import LlmtopApp (TUI): {exc}"
        if args.debug:
            traceback.print_exc(file=sys.stderr)
        else:
            print(msg, file=sys.stderr)
        return 1

    monitor: Any = None
    try:
        monitor = Monitor(
            interval=args.interval,
            timeout=args.timeout,
            extra_ports=args.extra_ports,
            enable_gpu=not args.no_gpu,
        )
        app = LlmtopApp(monitor, args.interval)
        app.run()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:  # noqa: BLE001
        msg = f"TUI error: {exc}"
        if args.debug:
            traceback.print_exc(file=sys.stderr)
        else:
            print(msg, file=sys.stderr)
        return 1
    finally:
        if monitor is not None:
            try:
                monitor.close()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Console script entry-point for ``llmtop``.

    Parameters
    ----------
    argv:
        Argument list (excluding the program name). Defaults to ``sys.argv[1:]``
        when ``None``.

    Returns
    -------
    int
        Exit code: 0 on success, non-zero on failure.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.json:
        return _run_json(args)

    if args.once:
        return _run_once(args)

    return _run_tui(args)
