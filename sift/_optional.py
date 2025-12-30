"""Centralized optional dependency handling."""

from importlib.util import find_spec

HAS_NUMBA = find_spec("numba") is not None
HAS_CATBOOST = find_spec("catboost") is not None
HAS_CATEGORY_ENCODERS = find_spec("category_encoders") is not None
HAS_POLARS = find_spec("polars") is not None

# numba special case: need decorator at module level
if HAS_NUMBA:
    from numba import njit
else:
    def njit(*args, **kwargs):
        def wrapper(fn):
            return fn
        return wrapper
