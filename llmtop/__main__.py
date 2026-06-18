"""Package entry-point: ``python -m llmtop``.

Delegates immediately to :func:`llmtop.cli.main` so that the package can be
invoked as ``python -m llmtop`` in addition to the ``llmtop`` console script.
"""

from __future__ import annotations

from .cli import main

raise SystemExit(main())
