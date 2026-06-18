"""Dependency-free Prometheus text-format parser.

Parses the standard Prometheus text exposition format (as served by vLLM,
llama.cpp, TGI, etc.) without any third-party libraries.

Key behaviour:
- Every metric name is stripped of its ``{...}`` label set before being
  stored in the result dict.
- When multiple label-sets exist for the same base name (very common in
  vLLM where every series carries ``{engine="0",model_name="..."}``), their
  values are **summed** together. This is correct for counters and gauges
  whose label sets fan out a single logical quantity (e.g.
  ``vllm:generation_tokens_total`` split by ``model_name``).
- Histogram ``_bucket`` / ``_count`` / ``_sum`` suffixes are stored under
  their full mangled names (e.g. ``foo_bucket``, ``foo_count``, ``foo_sum``),
  NOT under the base name ``foo``. Callers that only want the bare totals can
  ignore them; callers that need percentile calculation can reconstruct the
  histogram from the suffixed entries.
- ``# HELP`` and ``# TYPE`` comment lines are silently skipped.
- Malformed lines are silently skipped (never raise).
"""

from __future__ import annotations

import math
import re
from typing import Optional

# Matches a bare metric line: name (with optional {labels}), whitespace, value,
# optional timestamp. We capture name-with-labels and the value string.
_LINE_RE = re.compile(
    r"^([A-Za-z_:][A-Za-z0-9_:]*(?:\{[^}]*\})?)\s+([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|NaN|[+-]?Inf)"
)

# Captures the label block ``{...}`` so we can strip it to get the base name.
_LABEL_RE = re.compile(r"\{[^}]*\}")


def _strip_labels(name_with_labels: str) -> str:
    """Return *name_with_labels* with any ``{...}`` label block removed."""
    return _LABEL_RE.sub("", name_with_labels)


def parse_prometheus(text: str) -> dict[str, float]:
    """Parse Prometheus text exposition format into a flat ``{name: value}`` dict.

    The returned dict maps each **base metric name** (labels stripped) to the
    **sum of all label-set values** observed for that name.  ``NaN`` values
    are treated as ``0.0`` for summing purposes; ``+Inf`` / ``-Inf`` map to
    the corresponding Python float.

    Args:
        text: Raw Prometheus text body (UTF-8 decoded).

    Returns:
        ``dict[str, float]`` — never raises, skips unparseable lines.
    """
    result: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE_RE.match(line)
        if m is None:
            continue
        raw_name, raw_value = m.group(1), m.group(2)
        base_name = _strip_labels(raw_name)
        try:
            value = float(raw_value)
        except ValueError:
            continue
        # NaN → 0 for accumulation (avoids poisoning sums)
        if value != value:  # NaN check without math import
            value = 0.0
        result[base_name] = result.get(base_name, 0.0) + value
    return result


def prom_value(
    parsed: dict[str, float],
    name: str,
    default: Optional[float] = None,
) -> Optional[float]:
    """Look up a single metric from a ``parse_prometheus`` result.

    Args:
        parsed: Dict produced by :func:`parse_prometheus`.
        name:   Exact base metric name (no labels).
        default: Value to return when the name is absent (default ``None``).

    Returns:
        The summed float value or *default*.
    """
    return parsed.get(name, default)


def as_float(val: object) -> Optional[float]:
    """Coerce a metric value to a finite float, or ``None``.

    Returns ``None`` for ``None``, non-numeric input, ``NaN``, and ``±Inf`` so
    that callers never propagate a non-finite value into the UI or crash on an
    ``int(±Inf)`` :class:`OverflowError`.
    """
    if val is None:
        return None
    try:
        f = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def as_int(val: object) -> Optional[int]:
    """Coerce a metric value to an int, or ``None`` (finite-safe; see :func:`as_float`)."""
    f = as_float(val)
    return int(f) if f is not None else None
