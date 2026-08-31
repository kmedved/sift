"""Internal helpers for staged SIFT API deprecations."""

import warnings
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar


_P = ParamSpec("_P")
_R = TypeVar("_R")


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
