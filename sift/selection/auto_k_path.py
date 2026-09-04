"""In-sample objective evaluation for an ordered feature path."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:
    from sift.estimators.copula import FeatureCache


def compute_objective_for_path(
    cache: "FeatureCache",
    y: np.ndarray,
    feature_path: List[str],
    *,
    shrink: float = 1e-6,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute objective path for an arbitrary ordered feature_path.

    Objective at step t:
        obj[t] = log|Σ_S| - log|Σ_{y,S}|
               = 2 * I(y; S)   (Gaussian MI proxy)

    This is the shared objective primitive, not a k rule: no
    ``AutoKConfig.k_method`` routes to it. The path-only rules
    (``elbow``, ``penalized_objective``, ``k_posterior``, ``chi2_stop``,
    ``forward_stop``, ``changepoint``, ``perm_gap``) consume exactly this
    curve as their ``objective_path`` argument, and the orchestrators build it
    while running the greedy. Call it directly to re-score an ordering you
    already have -- a hand-picked feature list, a path from another selector,
    or a saved path checked against a new target -- against one full cache.
    It is a discovery-flavored quantity (conditional information carried by
    the prefix), not a predictive score, and it is computed in sample on every
    cache row: it is not the cross-fitting primitive. Fold and bootstrap
    methods must build fold-local correlations and call
    ``objective_from_corr_path`` instead.

    Parameters
    ----------
    cache : FeatureCache
        Prebuilt Gaussian-copula cache from ``build_cache``. Supplies the
        rank-Gauss matrix ``Z``, the row subsample, the sample weights, and
        (when ``compute_Rxx=True``) the cached feature correlation matrix,
        which is sliced instead of recomputed. Duplicate non-synthetic feature
        names are rejected.
    y : ndarray of shape (n_rows_original,)
        Target aligned to the *original* rows the cache was built from, not to
        the cached subsample. It is raveled, sliced by ``cache.row_idx``, and
        rank-Gauss transformed under the cache weights.
    feature_path : list of str or int
        Ordered features. Strings resolve through ``cache.feature_names``,
        integers are original column indices. Entries that are unknown or that
        fell out of ``cache.valid_cols`` are skipped silently, so the result
        can be shorter than the input.
    shrink : float, default 1e-6
        Shrinkage of the correlation matrix toward the identity for numerical
        stability, forwarded to ``objective_from_corr_path``.
    eps : float, default 1e-12
        Floor on the Schur-complement determinants, forwarded to
        ``objective_from_corr_path``.

    Returns
    -------
    objective : ndarray of shape (n_resolved,)
        Cumulative, monotonically non-decreasing objective after each resolved
        step, indexed from ``k=1``. Empty when ``feature_path`` is empty or
        nothing resolves to a valid cache column.

    Raises
    ------
    ValueError
        If ``y`` does not have ``cache.n_rows_original`` rows, if the cache
        fails its structural contract (missing provenance marker, non-finite
        ``Z``, inconsistent ``valid_cols``/``row_idx``/weights), or if the
        cache carries duplicate feature names.

    See Also
    --------
    select_k_elbow : Consumes this curve.
    select_k_penalized_objective : Consumes this curve.
    select_k_chi2_stop : Consumes this curve.
    build_cache : Builds the ``FeatureCache`` this expects.

    Notes
    -----
    ``obj[t] = -log(1 - R^2_t)`` where ``R^2_t = r_S' R_S^-1 r_S`` is the
    squared multiple correlation of the copula-space target on the first ``t``
    path features, so the per-step gain is ``-log(1 - rho^2_t)`` for the
    sample partial correlation of the entering feature. Target correlations
    are clipped to ``+/-0.999999`` before the Schur-complement recursion,
    which costs ``O(k^2)`` in total; extracting ``R_path`` is an ``O(k^2)``
    slice with a cached ``Rxx`` and one weighted correlation otherwise.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from sift import build_cache, compute_objective_for_path
    >>> rng = np.random.default_rng(0)
    >>> X = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
    >>> y = X["a"] + 0.7 * X["b"] + 0.2 * rng.normal(size=200)
    >>> cache = build_cache(X, compute_Rxx=True)
    >>> objective = compute_objective_for_path(cache, y.to_numpy(), ["a", "b", "c"])
    >>> objective.shape
    (3,)
    >>> bool(np.all(np.diff(objective) >= 0.0))
    True
    >>> compute_objective_for_path(cache, y.to_numpy(), ["missing"]).size
    0
    """
    from sift.estimators.copula import (
        weighted_corr_with_vector,
        weighted_correlation_matrix,
        weighted_rank_gauss_1d,
    )
    from sift.selection.objective import objective_from_corr_path
    from sift.selection.knockoff_filter import (
        _reject_duplicate_feature_names,
        _validate_prebuilt_cache_structure,
    )

    _validate_prebuilt_cache_structure(cache)
    _reject_duplicate_feature_names(cache)

    if not feature_path:
        return np.empty(0, dtype=np.float64)

    valid_cols = np.asarray(cache.valid_cols)
    orig_to_valid = {int(orig): int(pos) for pos, orig in enumerate(valid_cols)}

    name_to_orig = {}
    if cache.feature_names:
        name_to_orig = {name: i for i, name in enumerate(cache.feature_names)}

    path_valid_pos = []
    for f in feature_path:
        if isinstance(f, str):
            orig_idx = name_to_orig.get(f, None)
            if orig_idx is None:
                continue
        else:
            orig_idx = int(f)

        vpos = orig_to_valid.get(int(orig_idx), None)
        if vpos is None:
            continue
        path_valid_pos.append(vpos)

    if not path_valid_pos:
        return np.empty(0, dtype=np.float64)

    path_valid_pos = np.asarray(path_valid_pos, dtype=np.int64)

    y_arr = np.asarray(y).ravel()
    if y_arr.shape[0] != cache.n_rows_original:
        raise ValueError(
            f"y has {y_arr.shape[0]} rows but cache was built from "
            f"{cache.n_rows_original} rows"
        )
    ys = y_arr[np.asarray(cache.row_idx)]
    zy = weighted_rank_gauss_1d(ys, cache.sample_weight)
    r_y_full = weighted_corr_with_vector(cache.Z, zy, cache.sample_weight).astype(np.float64)

    r_path = r_y_full[path_valid_pos].copy()
    np.clip(r_path, -0.999999, 0.999999, out=r_path)

    if cache.Rxx is not None:
        R_full = np.asarray(cache.Rxx, dtype=np.float64)
        R_path = np.ascontiguousarray(R_full[np.ix_(path_valid_pos, path_valid_pos)], dtype=np.float64)
    else:
        Z_path = np.ascontiguousarray(cache.Z[:, path_valid_pos], dtype=np.float64)
        R_path = weighted_correlation_matrix(
            Z_path,
            np.asarray(cache.sample_weight, dtype=np.float64),
            backend="blas",
        )

    return objective_from_corr_path(R_path, r_path, shrink=shrink, eps=eps)


compute_objective_for_path.__module__ = "sift.selection.auto_k"
