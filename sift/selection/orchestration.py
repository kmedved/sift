"""Shared internal F6 selection orchestration.

Public routes are ``ModelSelector.fit`` (generic sklearn ranking) and
``catboost_select`` (native CatBoost SHAP/Pool preset). Both run the same
prepare → evaluate → choose → finalize contract. Backend-specific ranking,
fold voting, and count padding stay on the backend; this module does not
unify numerical algorithms.
"""

from __future__ import annotations

from typing import Any


class SelectionBackend:
    """Internal selection backend. Not a public plugin/factory API."""

    def prepare(self, X, y, **context: Any) -> Any:
        raise NotImplementedError

    def evaluate(self, prepared: Any) -> Any:
        raise NotImplementedError

    def choose(self, prepared: Any, evaluated: Any) -> Any:
        raise NotImplementedError

    def finalize(self, prepared: Any, evaluated: Any, chosen: Any) -> Any:
        raise NotImplementedError


def run_selection(backend: SelectionBackend, X, y, **context: Any):
    """Run one selection through a backend's four-stage contract."""
    prepared = backend.prepare(X, y, **context)
    evaluated = backend.evaluate(prepared)
    chosen = backend.choose(prepared, evaluated)
    return backend.finalize(prepared, evaluated, chosen)
