"""Internal progress-callback plumbing for long-running selectors."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias


ProgressCallback: TypeAlias = Callable[[int, int | None, dict[str, Any]], None]


def report_progress(
    callback: ProgressCallback | None,
    step: int,
    total: int | None,
    /,
    **info: Any,
) -> None:
    """Report one completed unit using a fresh snapshot dictionary."""
    if callback is None:
        return
    callback(
        int(step),
        None if total is None else int(total),
        dict(info),
    )
