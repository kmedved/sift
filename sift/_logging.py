"""Package logging configuration for SIFT progress messages."""

from __future__ import annotations

import logging
import sys
from typing import Literal


_LOGGER_NAME = "sift"
_HANDLER_MARKER = "_sift_default_handler"
_DISABLED_LEVEL = logging.CRITICAL + 1

logger = logging.getLogger(_LOGGER_NAME)


class _FallbackStderrHandler(logging.Handler):
    """Show progress by default, but defer to application logging handlers."""

    def emit(self, record: logging.LogRecord) -> None:
        if _has_external_handler(record):
            return
        try:
            message = self.format(record)
            sys.stderr.write(f"{message}\n")
            sys.stderr.flush()
        except Exception:
            self.handleError(record)


def _has_external_handler(record: logging.LogRecord) -> bool:
    current: logging.Logger | None = logger
    while current is not None:
        for handler in current.handlers:
            if (
                not getattr(handler, _HANDLER_MARKER, False)
                and record.levelno >= handler.level
            ):
                return True
        if not current.propagate:
            break
        current = current.parent
    return False


def _ensure_default_handler() -> logging.Handler:
    for handler in logger.handlers:
        if getattr(handler, _HANDLER_MARKER, False):
            return handler

    handler = _FallbackStderrHandler()
    setattr(handler, _HANDLER_MARKER, True)
    handler.set_name("sift.default")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return handler


_ensure_default_handler()
logger.propagate = True
if logger.level == logging.NOTSET:
    logger.setLevel(logging.INFO)


def set_verbosity(level: Literal["info", "debug"] | None) -> None:
    """Set package logging to INFO, DEBUG, or fully silent.

    Every progress message a SIFT selector emits under ``verbose=True`` is a
    record on the ``"sift"`` logger.  This helper sets that logger's level and
    the level of SIFT's own fallback stderr handler in a single call, so
    progress can be turned up to DEBUG or switched off entirely without any
    :mod:`logging` configuration of your own.  Call it once near import time;
    it mutates process-wide logging state and returns ``None``.  With no call
    at all the package starts at INFO, which is the level ``verbose=True`` emits at.

    Parameters
    ----------
    level : {"info", "debug"} or None
        Target level.  ``"info"`` is the package default and shows selector
        progress (path steps, knockoff thresholds, cache notices).  ``"debug"``
        additionally admits DEBUG records.  ``None`` silences SIFT completely
        by raising both the logger and the fallback handler one step above
        ``logging.CRITICAL``, so nothing is emitted no matter which
        ``verbose=True`` a selector was given.

    Returns
    -------
    None
        The function is called for its effect on the ``"sift"`` logger.

    Raises
    ------
    ValueError
        If ``level`` is not ``"info"``, ``"debug"``, or ``None``.

    See Also
    --------
    logging.Logger.setLevel : The underlying level control this wraps.
    sift.select_cefsplus : Filter selector whose ``verbose`` flag logs here.
    sift.select_fdr : Knockoff selector whose ``verbose`` flag logs here.

    Notes
    -----
    SIFT installs exactly one fallback stderr handler on the ``"sift"`` logger
    so ``verbose=True`` output is visible in a bare interpreter.  That handler
    stands aside for any record another handler -- on the ``"sift"`` logger or
    on an ancestor still reachable by propagation -- would emit at its own
    level, so an application that configures logging never sees doubled
    messages.  ``set_verbosity`` only changes levels: it never removes the
    fallback handler and never adds a second one, and repeated calls are
    idempotent.  Per-call silencing is still available through each selector's
    ``verbose=False`` argument; use this function for a process-wide default.

    Examples
    --------
    >>> import logging
    >>> from sift import set_verbosity
    >>> set_verbosity("debug")
    >>> logging.getLogger("sift").level == logging.DEBUG
    True
    >>> set_verbosity(None)  # silence every SIFT progress message
    >>> logging.getLogger("sift").isEnabledFor(logging.CRITICAL)
    False
    >>> set_verbosity("info")  # back to the package default
    >>> logging.getLogger("sift").level == logging.INFO
    True
    """

    handler = _ensure_default_handler()
    if level is None:
        handler.setLevel(_DISABLED_LEVEL)
        logger.setLevel(_DISABLED_LEVEL)
        return
    if level == "info":
        handler.setLevel(logging.INFO)
        logger.setLevel(logging.INFO)
        return
    if level == "debug":
        handler.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        return
    raise ValueError("level must be 'info', 'debug', or None")


__all__ = ["set_verbosity"]
