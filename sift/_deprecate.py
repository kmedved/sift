"""Internal helpers for staged SIFT API deprecations."""

import warnings
from collections.abc import Callable
from functools import wraps
from inspect import currentframe
from pathlib import Path
from typing import ParamSpec, TypeVar


_P = ParamSpec("_P")
_R = TypeVar("_R")
_THIS_FILE = Path(__file__).resolve()
_PACKAGE_ROOT = _THIS_FILE.parent


def warn_deprecated(
    name: str,
    since: str = "0.9",
    removal: str = "1.0",
) -> None:
    """Warn that a SIFT API is deprecated.

    This helper is intended to be called from the deprecated API itself.  The
    warning therefore points at that API's caller rather than SIFT internals.
    """
    warnings.warn(
        f"{name} is deprecated since SIFT {since} and will be removed in "
        f"SIFT {removal}.",
        FutureWarning,
        stacklevel=3,
    )


def deprecated_alias(
    name: str,
    target: Callable[_P, _R],
    *,
    since: str = "0.9",
    removal: str = "1.0",
) -> Callable[_P, _R]:
    """Return a warning alias that forwards every call to ``target``."""

    @wraps(target)
    def forward(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        warn_deprecated(name, since=since, removal=removal)
        return target(*args, **kwargs)

    return forward


def _external_warning_stacklevel() -> int:
    """Return the first caller frame outside the installed SIFT package."""

    current = currentframe()
    frame = None if current is None else current.f_back
    stacklevel = 1
    while frame is not None:
        filename = Path(frame.f_code.co_filename).resolve()
        if filename != _THIS_FILE and _PACKAGE_ROOT not in filename.parents:
            return stacklevel
        frame = frame.f_back
        stacklevel += 1
    return 3


def warn_external(message: str, category: type[Warning]) -> None:
    """Emit a warning at the first caller frame outside the SIFT package."""

    warnings.warn(
        message,
        category,
        stacklevel=_external_warning_stacklevel(),
    )


def warn_random_state_none(
    entry_point: str,
    *,
    stacklevel: int | None = None,
) -> None:
    """Warn about a nondeterministic default scheduled to change in 1.0."""

    message = (
        f"{entry_point} currently uses nondeterministic entropy when "
        "random_state=None; SIFT 1.0 will default to random_state=0. Pass an "
        "integer seed to make this call reproducible and silence this warning."
    )
    if stacklevel is None:
        warn_external(message, FutureWarning)
        return
    warnings.warn(message, FutureWarning, stacklevel=stacklevel)
