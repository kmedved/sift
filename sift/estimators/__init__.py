"""Estimator utilities for relevance and mutual information."""

from importlib import import_module
from typing import Any

__all__ = ["copula", "joint_mi", "relevance"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
