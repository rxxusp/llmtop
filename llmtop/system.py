"""System-level sampler: CPU, RAM, and load average via psutil.

Public API:
    SystemSampler  — create once, call sample() each poll.
"""

from __future__ import annotations

import os
from typing import Optional

import psutil

from .models import SystemSample


class SystemSampler:
    """Collects host CPU and RAM metrics via psutil.

    The constructor primes ``psutil.cpu_percent`` with a non-blocking
    ``interval=None`` call so that subsequent calls return a meaningful
    rolling value rather than 0.0.
    """

    def __init__(self) -> None:
        # Prime the CPU measurement baseline; first call always returns 0.0
        # so we discard the result.
        try:
            psutil.cpu_percent(interval=None)
        except Exception:
            pass

    def sample(self) -> SystemSample:
        """Return a :class:`SystemSample` with current CPU and RAM state.

        Non-blocking: uses ``cpu_percent(interval=None)`` for a rolling
        measurement since the previous call.  All fields are individually
        guarded so a partial psutil failure still returns what it can.
        """
        cpu_pct: Optional[float] = None
        try:
            cpu_pct = psutil.cpu_percent(interval=None)
        except Exception:
            pass

        cpu_count: Optional[int] = None
        try:
            cpu_count = psutil.cpu_count()
        except Exception:
            pass

        ram_used_bytes: Optional[int] = None
        ram_total_bytes: Optional[int] = None
        try:
            vm = psutil.virtual_memory()
            ram_used_bytes = vm.used
            ram_total_bytes = vm.total
        except Exception:
            pass

        load_avg: Optional[tuple[float, float, float]] = None
        try:
            la = os.getloadavg()
            load_avg = (la[0], la[1], la[2])
        except (AttributeError, OSError):
            # os.getloadavg() is not available on Windows
            pass

        return SystemSample(
            cpu_pct=cpu_pct,
            cpu_count=cpu_count,
            ram_used_bytes=ram_used_bytes,
            ram_total_bytes=ram_total_bytes,
            load_avg=load_avg,
        )
