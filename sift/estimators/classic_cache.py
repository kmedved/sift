"""Numeric feature-side cache for classic mRMR/JMI/JMIM reuse."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ClassicFeatureCache:
    """Cached numeric feature matrix for classic filter reuse.

    Holds mean-imputed float64 features on the retained rows, together with
    the row index, both normalized selection weights and raw MI weights, and
    name provenance. Build one with ``build_classic_cache`` and pass it as
    ``cache=`` to ``select_mrmr`` (``estimator="classic"``) or to
    ``select_jmi`` / ``select_jmim`` on the non-Gaussian estimators. Nothing
    about the target is stored, so one cache is valid for every 1-D ``y``
    aligned to the same original rows. This is not a Gaussian copula
    ``FeatureCache`` and is not accepted by ``select_cached``.

    Parameters
    ----------
    X : ndarray of shape (n_cached_rows, n_features), float64
        Mean-imputed numeric features on the retained rows, in original
        column order. Constants are not dropped.
    row_idx : ndarray of shape (n_cached_rows,), int
        Positions in the original matrix of the retained positive-weight
        rows, after optional subsampling without replacement.
    sample_weight : ndarray of shape (n_cached_rows,), float64
        Normalized weights of the retained rows (mean exactly 1), used for
        selection and relevance.
    mi_w : ndarray of shape (n_cached_rows,), float64
        Raw selected MI weights: unnormalized ``sample_weight[row_idx]`` when
        weights were supplied, otherwise ones. Binned JMI must see this, not
        the normalized vector.
    n_rows_original : int
        Row count of the matrix the cache was built from. Consumers require
        a target of exactly this length before slicing with ``row_idx``.
    feature_names : list of str
        Labels of every original column (length ``n_features``).
    feature_names_are_synthetic : bool
        ``True`` when names are generated ``"x0", "x1", ...`` from an
        unlabelled array.
    subsample : int or None
        Builder ``subsample`` argument that produced this cache.
    random_state : int
        Builder seed. It affected the cache only when ``subsample_applied``
        is True.
    weights_supplied : bool
        ``True`` when the builder received an explicit ``sample_weight``.
        KSG rejects that case even if every weight is one.
    subsample_applied : bool
        ``True`` when a without-replacement draw actually ran.

    See Also
    --------
    build_classic_cache : Build this cache from a numeric feature matrix.
    sift.select_mrmr : Classic mRMR route that accepts this cache.
    sift.select_jmi : Non-Gaussian JMI route that accepts this cache.
    FeatureCache : Gaussian copula cache; a different type.

    Notes
    -----
    The cache is positional: ``row_idx`` is the only bridge back to the
    caller's rows, and columns must match names and order. Call-time
    ``sample_weight``, ``subsample``, and ``random_state`` are rejected.
    Categorical encoding, ``within``, nested auto-k, and Gaussian methods
    are rejected rather than guessed.

    Examples
    --------
    >>> import numpy as np
    >>> from sift import build_classic_cache, select_mrmr
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(80, 4))
    >>> y1 = X[:, 0] + 0.1 * rng.normal(size=80)
    >>> cache = build_classic_cache(X, subsample=None)
    >>> cache.X.shape, bool(cache.weights_supplied)
    ((80, 4), False)
    >>> select_mrmr(X, y1, k=1, task="regression", cache=cache, verbose=False)
    ['x0']
    """

    X: np.ndarray
    row_idx: np.ndarray
    sample_weight: np.ndarray
    mi_w: np.ndarray
    n_rows_original: int
    feature_names: list[str]
    feature_names_are_synthetic: bool
    subsample: int | None
    random_state: int
    weights_supplied: bool
    subsample_applied: bool


def is_classic_cache(cache: object) -> bool:
    return isinstance(cache, ClassicFeatureCache)


def _reject_duplicate_feature_names(names: list[str]) -> None:
    index = pd.Index(names, dtype=object, tupleize_cols=False)
    duplicate_mask = index.duplicated(keep="first")
    if duplicate_mask.any():
        duplicates = index[duplicate_mask].tolist()
        sample = duplicates[:5]
        suffix = "..." if len(duplicates) > 5 else ""
        raise ValueError(f"Duplicate feature names are not supported: {sample}{suffix}")


def build_classic_cache(
    X,
    sample_weight: np.ndarray | None = None,
    subsample: int | None = 50_000,
    random_state: int = 0,
) -> ClassicFeatureCache:
    """Build a numeric feature-side cache for classic filters.

    Mean-imputes the full numeric matrix, then keeps positive-weight rows
    (optionally subsampled) with both normalized selection weights and raw
    MI weights. Reach for this when several 1-D targets share one numeric
    ``X`` on classic mRMR or non-Gaussian JMI/JMIM. The target is not used.
    Categorical columns are rejected; unsupervised ordinal/frequency
    encoding is a later stage.

    Parameters
    ----------
    X : DataFrame or ndarray of shape (n_samples, n_features)
        Numeric feature matrix. DataFrame labels are recorded;
        an unlabelled array gets ``"x0", "x1", ...`` and
        ``feature_names_are_synthetic=True``. Object, category, string, and
        datetime-like columns are rejected. Non-finite entries are
        mean-imputed over the full matrix before row selection.
    sample_weight : ndarray of shape (n_samples,) or None, default None
        Non-negative finite row weights, ``None`` meaning unweighted.
        Zero-weight rows are excluded. Stored both normalized (mean 1) and
        as raw selected MI weights.
    subsample : int or None, default 50000
        Maximum number of positive-weight rows to retain. ``None`` keeps
        every positive-weight row. When more rows qualify, exactly
        ``subsample`` of them are drawn without replacement using
        ``random_state``.
    random_state : int, default 0
        Seed for the subsampling draw only. Unused when no draw runs.

    Returns
    -------
    ClassicFeatureCache
        Imputed float64 ``X`` on retained rows, ``row_idx``, normalized
        ``sample_weight``, raw ``mi_w``, and provenance flags.

    Raises
    ------
    ValueError
        If ``X`` is not 2-D numeric, holds object/category/string or
        datetime-like columns, if column labels are duplicated (including
        repeated NaNs), if ``sample_weight`` is invalid, if ``subsample``
        is not a positive integer or ``None``, if ``random_state`` is not
        an integer, or if no positive-weight rows remain.

    See Also
    --------
    ClassicFeatureCache : Returned container.
    sift.select_mrmr : Pass the cache as ``cache=`` with ``estimator="classic"``.
    build_cache : Gaussian copula cache builder.

    Examples
    --------
    >>> import numpy as np
    >>> from sift import build_classic_cache, select_jmi
    >>> rng = np.random.default_rng(1)
    >>> X = rng.normal(size=(60, 3))
    >>> y = X[:, 1] + 0.05 * rng.normal(size=60)
    >>> cache = build_classic_cache(X, subsample=None)
    >>> select_jmi(X, y, k=1, task="regression", cache=cache, verbose=False)
    ['x1']
    """
    from sift._impute import mean_impute
    from sift._preprocess import (
        ensure_weights,
        extract_feature_names,
        reject_datetime_like_features,
        to_numpy,
    )

    if subsample is not None and (
        isinstance(subsample, (bool, np.bool_))
        or not isinstance(subsample, (int, np.integer))
        or int(subsample) < 1
    ):
        raise ValueError("subsample must be a positive integer or None")
    if isinstance(random_state, (bool, np.bool_)) or not isinstance(
        random_state, (int, np.integer)
    ):
        raise ValueError("random_state must be an integer")
    random_state = int(random_state)
    if subsample is not None:
        subsample = int(subsample)

    feature_names = extract_feature_names(X)
    feature_names_are_synthetic = feature_names is None
    reject_datetime_like_features(X)
    if hasattr(X, "select_dtypes"):
        non_numeric = X.select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()
        if non_numeric:
            sample = non_numeric[:5]
            suffix = "..." if len(non_numeric) > 5 else ""
            raise ValueError(
                f"Non-numeric columns found: {sample}{suffix}. "
                "Encode categorical columns before building a ClassicFeatureCache."
            )
    if not hasattr(X, "shape") or len(getattr(X, "shape", ())) != 2:
        X_probe = np.asarray(X)
        if X_probe.ndim != 2:
            raise ValueError("X must be a 2D feature matrix")
    X_arr = to_numpy(X, dtype=np.float64)
    n, p = X_arr.shape
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(p)]
    elif len(feature_names) != p:
        raise ValueError(
            f"X has {p} columns but feature names have length {len(feature_names)}"
        )
    _reject_duplicate_feature_names(feature_names)

    X_arr = mean_impute(X_arr, copy=True)
    weights_supplied = sample_weight is not None
    w_norm = ensure_weights(sample_weight, n, normalize=True)
    raw_full = (
        None
        if not weights_supplied
        else ensure_weights(sample_weight, n, normalize=False)
    )
    positive = np.flatnonzero(w_norm > 0.0)
    if positive.size == 0:
        raise ValueError("Subsample has zero total weight; check sample_weight/subsample.")
    subsample_applied = bool(subsample is not None and positive.size > int(subsample))
    if subsample_applied:
        rng = np.random.default_rng(random_state)
        row_idx = rng.choice(positive, size=int(subsample), replace=False)
    else:
        row_idx = positive
    X_sub = np.array(X_arr[row_idx], dtype=np.float64, copy=True)
    w_sub = w_norm[row_idx]
    weight_mean = float(w_sub.mean())
    if not np.isfinite(weight_mean) or weight_mean <= 0.0:
        raise ValueError("Subsample has zero total weight; check sample_weight/subsample.")
    w_sub = w_sub / weight_mean
    if not weights_supplied:
        mi_w = np.ones(row_idx.size, dtype=np.float64)
    else:
        assert raw_full is not None
        mi_w = np.asarray(raw_full[row_idx], dtype=np.float64)
    return ClassicFeatureCache(
        X=X_sub,
        row_idx=np.asarray(row_idx, dtype=np.int64),
        sample_weight=np.asarray(w_sub, dtype=np.float64),
        mi_w=mi_w,
        n_rows_original=int(n),
        feature_names=list(feature_names),
        feature_names_are_synthetic=bool(feature_names_are_synthetic),
        subsample=None if subsample is None else int(subsample),
        random_state=random_state,
        weights_supplied=bool(weights_supplied),
        subsample_applied=bool(subsample_applied),
    )


def validate_classic_cache_structure(
    cache: ClassicFeatureCache,
    *,
    original_n_features: int | None = None,
    n_rows: int | None = None,
) -> None:
    """Validate the structural contract of a prebuilt classic cache."""
    try:
        cache_vars = vars(cache)
    except TypeError as exc:
        raise ValueError("prebuilt classic cache is missing required structural fields") from exc
    required = (
        "X",
        "row_idx",
        "sample_weight",
        "mi_w",
        "n_rows_original",
        "feature_names",
        "feature_names_are_synthetic",
        "subsample",
        "random_state",
        "weights_supplied",
        "subsample_applied",
    )
    missing = [name for name in required if name not in cache_vars]
    if missing:
        raise ValueError(
            "prebuilt classic cache is missing required structural fields: "
            + ", ".join(missing)
        )
    X = np.asarray(cache.X)
    row_idx = np.asarray(cache.row_idx)
    sample_weight = np.asarray(cache.sample_weight)
    mi_w = np.asarray(cache.mi_w)
    if X.ndim != 2:
        raise ValueError("cache.X must be a 2-D array")
    if X.dtype != np.float64:
        raise ValueError("cache.X must be float64")
    if not np.isfinite(X).all():
        raise ValueError("cache.X must contain only finite values")
    n_cached, n_features = int(X.shape[0]), int(X.shape[1])
    if row_idx.ndim != 1 or not np.issubdtype(row_idx.dtype, np.integer):
        raise ValueError("cache.row_idx must be a 1-D integer array")
    if int(row_idx.size) != n_cached:
        raise ValueError("cache.row_idx length must match cache.X rows")
    if np.unique(row_idx).size != row_idx.size:
        raise ValueError("cache.row_idx must not contain duplicates")
    n_rows_original = int(cache.n_rows_original)
    if n_rows_original < n_cached:
        raise ValueError("cache.n_rows_original is smaller than the cached row count")
    if np.any(row_idx < 0) or np.any(row_idx >= n_rows_original):
        raise ValueError(
            f"cache.row_idx values are outside [0, {n_rows_original})"
        )
    if sample_weight.shape != (n_cached,) or mi_w.shape != (n_cached,):
        raise ValueError(
            "cache.sample_weight and cache.mi_w must be 1-D arrays matching cache.X rows"
        )
    if not np.isfinite(sample_weight).all() or np.any(sample_weight < 0.0):
        raise ValueError("cache.sample_weight must be finite and non-negative")
    if not np.isfinite(mi_w).all() or np.any(mi_w < 0.0):
        raise ValueError("cache.mi_w must be finite and non-negative")
    if not isinstance(cache.feature_names_are_synthetic, (bool, np.bool_)):
        raise ValueError("cache.feature_names_are_synthetic must be boolean")
    if not isinstance(cache.weights_supplied, (bool, np.bool_)):
        raise ValueError("cache.weights_supplied must be boolean")
    if not isinstance(cache.subsample_applied, (bool, np.bool_)):
        raise ValueError("cache.subsample_applied must be boolean")
    names = cache.feature_names
    if not isinstance(names, list) or len(names) != n_features:
        raise ValueError("cache.feature_names must be a list matching cache.X columns")
    _reject_duplicate_feature_names(names)
    if original_n_features is not None and int(original_n_features) != n_features:
        raise ValueError(
            f"X has {int(original_n_features)} columns but the classic cache "
            f"was built from {n_features}"
        )
    if n_rows is not None and int(n_rows) != n_rows_original:
        raise ValueError(
            f"X has {int(n_rows)} rows but the classic cache was built from "
            f"{n_rows_original} rows"
        )


def validate_classic_cache_compatibility(X, cache: ClassicFeatureCache) -> None:
    """Bind a classic cache to the caller's matrix names, order, and shape."""
    x_shape = X.shape if hasattr(X, "shape") else np.asarray(X).shape
    if len(x_shape) != 2:
        raise ValueError("X must be a 2D feature matrix")
    n_rows, n_features = int(x_shape[0]), int(x_shape[1])
    validate_classic_cache_structure(
        cache,
        original_n_features=n_features,
        n_rows=n_rows,
    )
    synthetic = bool(cache.feature_names_are_synthetic)
    names = list(cache.feature_names)
    if synthetic:
        if isinstance(X, pd.DataFrame):
            raise ValueError(
                "A cache built from unnamed/positional features requires X to be "
                "the compatible positional ndarray; rebuild the cache from this "
                "DataFrame to establish column names and order"
            )
        if len(names) != n_features:
            raise ValueError(
                f"X has {n_features} columns but the positional cache was built from "
                f"{len(names)}"
            )
        return
    if not isinstance(X, pd.DataFrame):
        raise ValueError(
            "A named classic cache requires X to be a DataFrame with the "
            "same column names and order used to build the cache"
        )
    if list(X.columns) != names:
        raise ValueError(
            "X columns do not match cache.feature_names (names and order must "
            "be identical); fit the cache from the same matrix"
        )
