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

    SIFT keeps one fallback stderr handler so ``verbose=True`` progress remains
    visible without application logging configuration. If an application has
    installed its own handler, records propagate normally and the fallback
    steps aside to avoid duplicate messages.
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
