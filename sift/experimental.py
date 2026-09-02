"""Staged access to SIFT's research-oriented automatic-k helpers."""

from __future__ import annotations

from typing import Any

from sift._deprecate import warn_external
from sift.selection.auto_k import (
    compute_objective_for_path as _compute_objective_for_path,
    select_k_posterior as _select_k_posterior,
)
from sift.selection.auto_k_knockoff import (
    select_k_knockoff_path as _select_k_knockoff_path,
)
from sift.selection.auto_k_resample import (
    bootstrap_paths as _bootstrap_paths,
    null_objective_paths as _null_objective_paths,
    select_k_perm_gap as _select_k_perm_gap,
    select_k_stability as _select_k_stability,
)
from sift.selection.auto_k_stop import (
    path_gain_pvalues as _path_gain_pvalues,
    select_k_changepoint as _select_k_changepoint,
    select_k_chi2_stop as _select_k_chi2_stop,
    select_k_forward_stop as _select_k_forward_stop,
)
from sift.selection.auto_k_xfit import (
    gaussian_cv_curves as _gaussian_cv_curves,
    select_k_gaussian_cv as _select_k_gaussian_cv,
    select_k_xfit_objective as _select_k_xfit_objective,
    xfit_objective_curves as _xfit_objective_curves,
)
from sift.selection.path_eval import (
    FeaturePathEvaluationResult as _FeaturePathEvaluationResult,
)


_EXPERIMENTAL_OBJECTS = {
    "FeaturePathEvaluationResult": _FeaturePathEvaluationResult,
    "select_k_posterior": _select_k_posterior,
    "path_gain_pvalues": _path_gain_pvalues,
    "select_k_changepoint": _select_k_changepoint,
    "select_k_chi2_stop": _select_k_chi2_stop,
    "select_k_forward_stop": _select_k_forward_stop,
    "bootstrap_paths": _bootstrap_paths,
    "null_objective_paths": _null_objective_paths,
    "select_k_perm_gap": _select_k_perm_gap,
    "select_k_stability": _select_k_stability,
    "gaussian_cv_curves": _gaussian_cv_curves,
    "select_k_gaussian_cv": _select_k_gaussian_cv,
    "select_k_xfit_objective": _select_k_xfit_objective,
    "select_k_knockoff_path": _select_k_knockoff_path,
    "xfit_objective_curves": _xfit_objective_curves,
    "compute_objective_for_path": _compute_objective_for_path,
}

__all__ = list(_EXPERIMENTAL_OBJECTS)


def __getattr__(name: str) -> Any:
    try:
        value = _EXPERIMENTAL_OBJECTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    warn_external(
        f"sift.experimental.{name} is an experimental SIFT 0.9 API and may "
        "change before 1.0. The existing top-level import remains available "
        "through 0.9 for compatibility.",
        FutureWarning,
    )
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
