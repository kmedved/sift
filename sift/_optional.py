"""Centralized optional dependency handling."""

from importlib import import_module
from importlib.util import find_spec

HAS_NUMBA = False
HAS_CATBOOST = False
HAS_CATEGORY_ENCODERS = False
HAS_POLARS = False


if find_spec("numba") is not None:
    njit = import_module("numba").njit
    HAS_NUMBA = True
else:
    def njit(*args, **kwargs):
        def wrapper(fn):
            return fn
        return wrapper


if find_spec("catboost") is not None:
    catboost = import_module("catboost")
    HAS_CATBOOST = True
else:
    catboost = None


if find_spec("category_encoders") is not None:
    ce = import_module("category_encoders")
    HAS_CATEGORY_ENCODERS = True
else:
    ce = None


if find_spec("polars") is not None:
    polars = import_module("polars")
    HAS_POLARS = True
else:
    polars = None
