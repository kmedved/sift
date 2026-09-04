"""Normalized auto-k curve payload helpers."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

#: ``diagnostics_`` key holding the normalized auto-k curve payload.  The
#: payload is producer-side data with a fixed shape so that result adapters
#: never have to guess which route-specific diagnostic column is the criterion.
AUTO_K_CURVE_KEY = "auto_k_curve"

#: Columns of the normalized curve frame, in order.
AUTO_K_CURVE_COLUMNS = ("k", "criterion", "criterion_se", "selected")

# Route -> (criterion column, standard-error column or None, direction).
# The criterion is the quantity whose value across ``k`` drives that route's
# stopping/selection decision, so plotting it always explains the chosen k.
_AUTO_K_CURVE_CRITERIA: dict[str, tuple[str, Optional[str], str]] = {
    "evaluate": ("score", "score_se", "higher_is_better"),
    "gaussian_cv": ("score", "score_se", "higher_is_better"),
    "xfit_objective": ("score", "score_se", "higher_is_better"),
    "penalized_objective": ("penalized_score", None, "higher_is_better"),
    "elbow": ("objective", None, "higher_is_better"),
    "k_posterior": ("post", None, "higher_is_better"),
    "perm_gap": ("gap", "gap_se", "higher_is_better"),
    "stability": ("phi", "phi_se", "higher_is_better"),
    "changepoint": ("log_scaled_gain", None, "higher_is_better"),
    "chi2_stop": ("p_max", None, "lower_is_better"),
    "forward_stop": ("Y_running_mean", None, "lower_is_better"),
}

# Routes whose diagnostics are deliberately not a k-indexed criterion path.
# They report an explicit reason rather than a fabricated curve.
_AUTO_K_CURVE_UNAVAILABLE: dict[str, str] = {
    "knockoff_path": (
        "knockoff_path diagnostics carry one row per candidate feature and knockoff "
        "draw rather than one row per k, so the route has no k-indexed criterion curve"
    ),
    "consensus": (
        "consensus diagnostics carry one row per member method's k vote rather than "
        "one row per k, so the route has no k-indexed criterion curve"
    ),
}


def _auto_k_curve_unavailable(route: str, reason: str) -> dict:
    return {
        "available": False,
        "route": route,
        "criterion": None,
        "criterion_direction": None,
        "unavailable_reason": reason,
        "curve": None,
    }


def build_auto_k_curve_payload(
    *,
    k_method: str,
    diagnostics: Optional[pd.DataFrame],
    summary: Optional[dict],
) -> dict:
    """Normalize one auto-k route's diagnostics into the standard curve payload.

    Every auto-k route reports its search under a different column name
    (``score``, ``penalized_score``, ``objective``, ``phi``, ``p_max``, ...).
    This function resolves the route actually run — following the ``auto``
    router's ``routed_method`` when present — and returns a fixed-shape payload
    so result adapters never inspect route-specific diagnostics themselves.

    Returns
    -------
    dict
        ``{"available", "route", "criterion", "criterion_direction",
        "unavailable_reason", "curve"}``.  When ``available`` is true, ``curve``
        is a DataFrame with exactly the columns ``k``, ``criterion``,
        ``criterion_se``, and ``selected``; ``criterion`` names the source
        diagnostic column and ``criterion_direction`` is ``"higher_is_better"``
        or ``"lower_is_better"``.  When it is false, ``curve`` is ``None`` and
        ``unavailable_reason`` says why the route has no k-indexed curve.
    """
    summary = summary or {}
    route = str(summary.get("routed_method") or summary.get("method") or k_method)

    if route in _AUTO_K_CURVE_UNAVAILABLE:
        return _auto_k_curve_unavailable(route, _AUTO_K_CURVE_UNAVAILABLE[route])
    spec = _AUTO_K_CURVE_CRITERIA.get(route)
    if spec is None:
        return _auto_k_curve_unavailable(
            route,
            f"no normalized criterion is defined for auto-k route {route!r}",
        )
    if not isinstance(diagnostics, pd.DataFrame) or diagnostics.empty:
        return _auto_k_curve_unavailable(
            route,
            f"auto-k route {route!r} produced no diagnostics rows",
        )

    column, se_column, direction = spec
    missing = [name for name in ("k", column) if name not in diagnostics.columns]
    if missing:
        return _auto_k_curve_unavailable(
            route,
            f"auto-k route {route!r} diagnostics are missing the {missing} "
            "column(s) required for a normalized curve",
        )

    k_values = pd.to_numeric(diagnostics["k"], errors="coerce")
    if k_values.isna().any():
        return _auto_k_curve_unavailable(
            route,
            f"auto-k route {route!r} diagnostics contain a non-numeric k",
        )
    k_int = k_values.to_numpy(dtype=np.int64)
    criterion = pd.to_numeric(diagnostics[column], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if se_column is not None and se_column in diagnostics.columns:
        criterion_se = pd.to_numeric(diagnostics[se_column], errors="coerce").to_numpy(
            dtype=np.float64
        )
    else:
        criterion_se = np.full(k_int.shape[0], np.nan, dtype=np.float64)

    # ``selected`` marks the k the route actually returned.  Deriving it from the
    # summary rather than a per-route ``selected`` column keeps stop rules that
    # were floored by ``min_k`` (and routes with no such column) truthful.
    selected_k = summary.get("selected_k")
    if selected_k is None:
        selected = np.zeros(k_int.shape[0], dtype=bool)
    else:
        selected = k_int == int(selected_k)

    curve = pd.DataFrame(
        {
            "k": k_int,
            "criterion": criterion,
            "criterion_se": criterion_se,
            "selected": selected,
        }
    ).sort_values("k", kind="mergesort").reset_index(drop=True)
    return {
        "available": True,
        "route": route,
        "criterion": column,
        "criterion_direction": direction,
        "unavailable_reason": None,
        "curve": curve,
    }
