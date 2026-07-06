"""Numba helpers used by SIFT kernels."""

from __future__ import annotations

from collections.abc import Callable
import inspect
import logging
import re
from typing import Any

from numba import njit


logger = logging.getLogger(__name__)

_NUMBA_CACHE_LOCATOR_ERROR = re.compile(
    r"^cannot cache function .+: no locator available for file .+$"
)


def _is_numba_cache_locator_error(exc: RuntimeError) -> bool:
    return bool(_NUMBA_CACHE_LOCATOR_ERROR.search(str(exc)))


def _compile_njit_optional_cache(
    fn: Callable[..., Any],
    jit_args: tuple[Any, ...],
    jit_kwargs: dict[str, Any],
) -> Any:
    kwargs = dict(jit_kwargs)
    cache_requested = kwargs.pop("cache", True)
    if not cache_requested:
        return njit(*jit_args, cache=False, **kwargs)(fn)

    try:
        return njit(*jit_args, cache=True, **kwargs)(fn)
    except RuntimeError as exc:
        if not _is_numba_cache_locator_error(exc):
            raise
        logger.warning(
            "Numba persistent cache unavailable for %s; retrying with cache=False",
            getattr(fn, "__qualname__", getattr(fn, "__name__", repr(fn))),
        )
        return njit(*jit_args, cache=False, **kwargs)(fn)


def njit_optional_cache(*jit_args: Any, **jit_kwargs: Any) -> Any:
    """Like ``numba.njit``, but falls back when persistent caching is unavailable."""
    if len(jit_args) == 1 and inspect.isfunction(jit_args[0]):
        return _compile_njit_optional_cache(jit_args[0], (), jit_kwargs)

    def decorator(fn: Callable[..., Any]) -> Any:
        return _compile_njit_optional_cache(fn, jit_args, jit_kwargs)

    return decorator
