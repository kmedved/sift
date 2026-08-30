import numpy as np
import pandas as pd
import numbers
from typing import Optional, List, Dict, Tuple, Callable
from dataclasses import dataclass, fields, replace
import warnings

from sklearn.utils.extmath import randomized_svd
from sklearn.ensemble import HistGradientBoostingRegressor


def leverage_scores_multi_alpha(
    Xs: np.ndarray,
    V: Optional[np.ndarray],
    S: Optional[np.ndarray],
    s2: Optional[np.ndarray],
    leverage_batch_size: int,
) -> np.ndarray:
    """Compute leverage scores across multiple ridge parameters."""
    n = Xs.shape[0]
    if V is None or S is None or s2 is None:
        return np.ones(n, dtype=np.float32)

    s2_pos = s2[s2 > 1e-8]
    if s2_pos.size:
        qs = np.percentile(s2_pos, [5, 25, 50, 75, 95]).astype(np.float32)
        alphas = np.unique(np.clip(np.array([1e-6, *qs], dtype=np.float32), 1e-8, None))
    else:
        alphas = np.array([1e-6], dtype=np.float32)

    invS = (1.0 / (S + 1e-12)).astype(np.float32)
    W = (s2[:, None] / (s2[:, None] + alphas[None, :])).astype(np.float32)

    lev = np.empty(n, dtype=np.float32)
    B = leverage_batch_size
    for start in range(0, n, B):
        stop = min(n, start + B)
        XV = Xs[start:stop] @ V
        U_chunk = XV * invS
        U2 = U_chunk * U_chunk
        lev_multi = U2 @ W
        lev[start:stop] = lev_multi.mean(axis=1)

    lev = np.maximum(lev, 1e-12)
    lev /= lev.mean()
    return lev


def _is_int_like(value: object) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _is_real_like(value: object) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _validate_smart_sampler_config(config: "SmartSamplerConfig") -> None:
    if not _is_real_like(config.sample_frac):
        raise TypeError("sample_frac must be a real number.")
    if not np.isfinite(config.sample_frac):
        raise ValueError("sample_frac must be finite.")
    if not (0 < config.sample_frac <= 1):
        raise ValueError("sample_frac must be in (0, 1].")
    if not _is_int_like(config.min_per_group) or config.min_per_group < 1:
        raise ValueError("min_per_group must be an integer >= 1.")
    if not _is_int_like(config.pilot_sample_size) or config.pilot_sample_size < 1:
        raise ValueError("pilot_sample_size must be an integer >= 1.")
    if not _is_int_like(config.leverage_batch_size) or config.leverage_batch_size < 1:
        raise ValueError("leverage_batch_size must be an integer >= 1.")
    if config.svd_sample_size is not None and (
        not _is_int_like(config.svd_sample_size) or config.svd_sample_size < 1
    ):
        raise ValueError("svd_sample_size must be an integer >= 1 or None.")
    for name in ("weight_clip_quantile", "residual_weight_cap", "uniform_floor", "anchor_max_share"):
        value = getattr(config, name)
        if not _is_real_like(value):
            raise TypeError(f"{name} must be a real number.")
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite.")
    if not (0 <= config.weight_clip_quantile <= 1):
        raise ValueError("weight_clip_quantile must be in [0, 1].")
    if config.residual_weight_cap < 0:
        raise ValueError("residual_weight_cap must be >= 0.")
    if not (0 <= config.uniform_floor <= 1):
        raise ValueError("uniform_floor must be in [0, 1].")
    if not (0 <= config.anchor_max_share <= 1):
        raise ValueError("anchor_max_share must be in [0, 1].")
    if config.group_col is not None and not isinstance(config.group_col, str):
        raise TypeError("group_col must be a string or None.")
    if config.time_col is not None and not isinstance(config.time_col, str):
        raise TypeError("time_col must be a string or None.")
    if config.anchor_fn is not None and not callable(config.anchor_fn):
        raise TypeError("anchor_fn must be callable or None.")
    if config.random_state is not None and not _is_int_like(config.random_state):
        raise TypeError("random_state must be an integer or None.")
    if not isinstance(config.verbose, (bool, np.bool_)):
        raise TypeError("verbose must be a bool.")


# =============================================================================
# Smart Sampler Configuration
# =============================================================================

@dataclass
class SmartSamplerConfig:
    """
    Configuration for smart sampling behavior.

    Parameters
    ----------
    sample_frac : float
        Target fraction of rows to sample.
    group_col : str, optional
        Column defining groups/entities (e.g., user_id, patient_id, ticker).
        If None, treats all rows as one group.
    time_col : str, optional
        Column defining time ordering within groups.
    min_per_group : int
        Minimum rows to keep per group.
    pilot_sample_size : int
        Size of pilot sample for residual estimation.
    leverage_batch_size : int
        Batch size for leverage score computation (memory vs speed).
    svd_sample_size : int, optional
        Row subsample size for randomized SVD (speed on huge n). If set below n, leverage
        scores become approximate. If None, uses all rows.
    weight_clip_quantile : float
        Quantile for clipping extreme weights.
    residual_weight_cap : float
        Maximum weight for residual-based scores (vs leverage).
    uniform_floor : float
        Minimum base probability (ensures coverage).
    anchor_fn : callable, optional
        Function(df, group_col, time_col) -> boolean mask identifying anchor rows.
        Retained anchors are included with probability 1. At least one anchor is
        retained per non-empty group when ``anchor_max_share`` is positive.
    anchor_max_share : float
        Maximum share of per-group quota for anchors.
    random_state : int, optional
        Random seed.
    verbose : bool
        Print progress.
    """
    sample_frac: float = 0.10
    group_col: Optional[str] = None
    time_col: Optional[str] = None
    min_per_group: int = 2
    pilot_sample_size: int = 50_000
    leverage_batch_size: int = 200_000
    svd_sample_size: Optional[int] = None
    weight_clip_quantile: float = 0.99
    residual_weight_cap: float = 0.4
    uniform_floor: float = 0.05
    anchor_fn: Optional[Callable] = None
    anchor_max_share: float = 0.4
    random_state: Optional[int] = 42
    verbose: bool = True

    def __post_init__(self) -> None:
        _validate_smart_sampler_config(self)


# =============================================================================
# Smart Sampler
# =============================================================================

@dataclass(frozen=True)
class SmartSampleArrays:
    df: pd.DataFrame
    Xs: np.ndarray
    y: np.ndarray | None
    group_indices: Dict[object, np.ndarray]


@dataclass(frozen=True)
class SmartSampleScores:
    leverage: np.ndarray
    residual: np.ndarray
    base: np.ndarray
    beta: float


def _resolve_smart_config(
    config: Optional[SmartSamplerConfig],
    overrides: dict,
) -> SmartSamplerConfig:
    if config is None:
        return SmartSamplerConfig(**overrides)
    if overrides:
        valid_fields = {field.name for field in fields(SmartSamplerConfig)}
        unknown = sorted(set(overrides) - valid_fields)
        if unknown:
            raise TypeError(f"Unknown SmartSamplerConfig override(s): {unknown}")
        return replace(config, **overrides)
    _validate_smart_sampler_config(config)
    return config


def _validate_smart_columns(
    df: pd.DataFrame,
    feature_cols: List[str],
    y_col: str,
    config: SmartSamplerConfig,
) -> None:
    required = set(feature_cols + [y_col])
    if config.group_col:
        required.add(config.group_col)
    if config.time_col:
        required.add(config.time_col)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if "sample_weight" in df.columns:
        raise ValueError(
            "smart_sample writes its inverse-probability weights to a "
            "'sample_weight' column, but the input already has one. Rename or "
            "drop that column first so a feature is not silently overwritten."
        )


def _prepare_scaled_matrix(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    from sift._impute import mean_impute

    X = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    X = mean_impute(X, copy=False)

    mu = X.mean(axis=0, dtype=np.float64)
    sigma = X.std(axis=0, dtype=np.float64)
    sigma[sigma < 1e-12] = 1.0
    X -= mu.astype(np.float32)
    X /= sigma.astype(np.float32)
    if not np.isfinite(X).all():
        raise ValueError(
            "X contains non-finite values after imputation/scaling; check for extreme magnitudes."
        )
    return X


def _prepare_smart_arrays(
    df: pd.DataFrame,
    feature_cols: List[str],
    y_col: str,
    config: SmartSamplerConfig,
) -> SmartSampleArrays:
    _validate_smart_columns(df, feature_cols, y_col, config)
    df_reset = df.reset_index(drop=True)
    Xs = _prepare_scaled_matrix(df_reset, feature_cols)
    y = None
    if config.residual_weight_cap > 0:
        y = df_reset[y_col].to_numpy(dtype=np.float64, copy=True)
        if not np.isfinite(y).all():
            raise ValueError("y must be finite (no NaN or inf) when residual_weight_cap > 0")
    return SmartSampleArrays(
        df=df_reset,
        Xs=Xs,
        y=y,
        group_indices=_build_group_indices(df_reset, config),
    )


def _build_group_indices(
    df: pd.DataFrame,
    config: SmartSamplerConfig,
) -> Dict[object, np.ndarray]:
    if not config.group_col:
        return {"_all": np.arange(len(df))}

    group_vals = df[config.group_col]
    group_key = group_vals.astype("object")
    if group_vals.isna().any():
        sentinel = "__SIFT_MISSING_GROUP__"
        existing = set(group_key[~group_vals.isna()].unique().tolist())
        while sentinel in existing:
            sentinel += "_"
        group_key = group_key.where(~group_vals.isna(), sentinel)
    return {g: idx for g, idx in df.groupby(group_key, sort=False).indices.items()}


def _compute_svd_factors(
    Xs: np.ndarray,
    config: SmartSamplerConfig,
    rng: np.random.Generator,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    n, d = Xs.shape
    k = int(min(128, d, max(16, np.ceil(np.log2(d + 1)) * 8)))
    k = min(k, max(1, min(n, d) - 1))
    try:
        X_svd = Xs
        if config.svd_sample_size is not None:
            svd_rows = min(n, int(config.svd_sample_size))
            if svd_rows < n:
                if config.verbose:
                    print(
                        f"SVD computed on {svd_rows:,}/{n:,} rows; leverage scores are approximate."
                    )
                svd_idx = rng.choice(n, size=svd_rows, replace=False)
                X_svd = Xs[svd_idx]
        n_svd = X_svd.shape[0]
        k_svd = min(k, max(1, min(n_svd, d) - 1))
        _, S, Vt = randomized_svd(
            X_svd,
            n_components=k_svd,
            n_iter=4,
            random_state=config.random_state,
        )
        V = Vt.T.astype(np.float32)
        S = S.astype(np.float32)
        return V, S, S * S
    except Exception as e:
        warnings.warn(f"SVD failed ({e}); using uniform geometry scores.", RuntimeWarning)
        return None, None, None


def _compute_residual_scores(
    Xs: np.ndarray,
    y: np.ndarray | None,
    config: SmartSamplerConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    n = Xs.shape[0]
    ps = min(n, config.pilot_sample_size)
    pilot_all = rng.choice(n, size=ps, replace=False)

    min_val_size = min(100, ps // 4)
    half = max(min_val_size, ps // 2)
    half = min(half, ps - min_val_size)

    pilot_train = pilot_all[:half]
    pilot_val = pilot_all[half:]

    beta = 0.0
    res_scores = np.ones(n, dtype=np.float32)
    if config.residual_weight_cap <= 0:
        return res_scores, beta

    assert y is not None
    if len(pilot_train) >= 50 and len(pilot_val) >= 20:
        try:
            y_float = np.asarray(y, dtype=np.float64)
            target_origin = float(np.median(y_float[pilot_all]))
            y_centered = y_float - target_origin
            var_y = float(np.var(y_centered[pilot_all]))
            if not np.isfinite(var_y) or var_y <= 0.0:
                return res_scores, beta

            pilot_train_model = HistGradientBoostingRegressor(
                max_iter=50,
                max_depth=4,
                learning_rate=0.1,
                random_state=config.random_state,
            )
            pilot_val_model = HistGradientBoostingRegressor(
                max_iter=50,
                max_depth=4,
                learning_rate=0.1,
                random_state=config.random_state,
            )
            pilot_train_model.fit(Xs[pilot_train], y_centered[pilot_train])
            pilot_val_model.fit(Xs[pilot_val], y_centered[pilot_val])

            # Two-fold cross-fitting: neither pilot fold is scored by a model
            # trained on that fold.
            preds = np.empty(n, dtype=np.float64)
            preds[pilot_val] = pilot_train_model.predict(Xs[pilot_val])
            preds[pilot_train] = pilot_val_model.predict(Xs[pilot_train])
            nonpilot_mask = np.ones(n, dtype=bool)
            nonpilot_mask[pilot_all] = False
            nonpilot = np.flatnonzero(nonpilot_mask)
            if nonpilot.size:
                # Assign each non-pilot row to exactly one unseen model. An
                # ensemble prediction would have lower residual variance than
                # the one-model OOF predictions used for pilot rows, making
                # pilot membership itself affect sampling scores.
                assignment_rng = np.random.default_rng(config.random_state)
                use_train_model = assignment_rng.integers(
                    0,
                    2,
                    size=nonpilot.size,
                    dtype=np.int8,
                ).astype(bool)
                train_model_rows = nonpilot[use_train_model]
                val_model_rows = nonpilot[~use_train_model]
                if train_model_rows.size:
                    preds[train_model_rows] = pilot_train_model.predict(
                        Xs[train_model_rows]
                    )
                if val_model_rows.size:
                    preds[val_model_rows] = pilot_val_model.predict(Xs[val_model_rows])
            if not np.isfinite(preds).all():
                raise ValueError("pilot model produced non-finite predictions")

            pilot_resid = y_centered[pilot_all] - preds[pilot_all]
            pilot_mse = float(np.mean(pilot_resid ** 2))
            r2 = max(0.0, min(1.0, 1.0 - pilot_mse / var_y))

            resid_all = np.abs(y_centered - preds)
            mean_resid = float(np.mean(resid_all))
            if not np.isfinite(mean_resid) or mean_resid <= 0.0:
                return res_scores, beta
            res_scores = resid_all / mean_resid
            res_scores = np.maximum(res_scores, np.finfo(np.float64).tiny)
            res_scores /= res_scores.mean()

            beta = min(config.residual_weight_cap, r2)
            if config.verbose:
                print(f"Pilot R² = {r2:.3f} → residual weight β = {beta:.3f}")
        except Exception as e:
            warnings.warn(f"Pilot model failed ({e}); using geometry only.", RuntimeWarning)
    elif config.verbose:
        warnings.warn(f"Dataset too small for pilot model (n={n}); using geometry only.", RuntimeWarning)
    return res_scores, beta


def _compute_sampling_scores(
    arrays: SmartSampleArrays,
    config: SmartSamplerConfig,
    rng: np.random.Generator,
) -> SmartSampleScores:
    V, S, s2 = _compute_svd_factors(arrays.Xs, config, rng)
    lev_scores = leverage_scores_multi_alpha(arrays.Xs, V, S, s2, config.leverage_batch_size)
    res_scores, beta = _compute_residual_scores(arrays.Xs, arrays.y, config, rng)
    base_scores = (1.0 - beta) * lev_scores + beta * res_scores
    base_scores = (
        (1 - config.uniform_floor)
        * (base_scores / (base_scores.mean() + 1e-12))
        + config.uniform_floor
    )
    return SmartSampleScores(
        leverage=lev_scores,
        residual=res_scores,
        base=base_scores,
        beta=beta,
    )


def _build_anchor_mask(df: pd.DataFrame, config: SmartSamplerConfig) -> np.ndarray:
    if config.anchor_fn is None:
        return np.zeros(len(df), dtype=bool)
    anchor_mask = np.asarray(
        config.anchor_fn(df, config.group_col, config.time_col),
        dtype=bool,
    ).reshape(-1)
    if len(anchor_mask) != len(df):
        raise ValueError(
            f"anchor_fn returned {len(anchor_mask)} rows but df has {len(df)} rows"
        )
    if config.verbose and anchor_mask.any():
        print(f"Anchors: {anchor_mask.sum():,} rows")
    return anchor_mask


def _poisson_calibrated(
    local_scores: np.ndarray,
    budget: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    m = local_scores.size
    if budget <= 0 or m == 0:
        return np.array([], dtype=int), np.zeros(0, dtype=np.float32)

    s = np.maximum(local_scores.astype(np.float64), 0.0)
    s_sum = s.sum()
    if s_sum == 0:
        p = np.full(m, 1.0 / m, dtype=np.float64)
    else:
        p = s / s_sum

    p_max = p.max()
    if p_max == 0:
        return np.array([], dtype=int), np.zeros(0, dtype=np.float32)

    def expected(tau: float) -> float:
        return float(np.minimum(1.0, tau * p).sum())

    lo, hi = 0.0, max(1.0, budget / p_max)
    while expected(hi) < budget and hi < 1e12:
        hi *= 2.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if expected(mid) < budget:
            lo = mid
        else:
            hi = mid
    pi = np.minimum(1.0, hi * p)

    chosen = rng.random(m) < pi
    return np.nonzero(chosen)[0].astype(int), pi.astype(np.float32)


def _add_rows(pi: np.ndarray, indices: np.ndarray, pis: np.ndarray) -> None:
    if indices.size == 0:
        return
    idx = np.asarray(indices, dtype=np.intp)
    vals = np.asarray(pis, dtype=np.float32)
    np.add.at(pi, idx, vals)
    # Only the touched entries can exceed 1; clipping the whole vector here
    # made every per-group call O(n).
    pi[idx] = np.minimum(pi[idx], 1.0)


def _cap_group_anchors(
    g_idx: np.ndarray,
    anchor_mask: np.ndarray,
    base_scores: np.ndarray,
    target_g: int,
    config: SmartSamplerConfig,
) -> np.ndarray:
    anchor_pos = np.flatnonzero(anchor_mask[g_idx])
    if not anchor_pos.size:
        return anchor_pos
    if config.anchor_max_share <= 0 or target_g <= 0:
        return np.array([], dtype=int)

    max_anchor_keep = max(1, int(np.floor(config.anchor_max_share * target_g)))
    max_anchor_keep = min(max_anchor_keep, target_g)
    if anchor_pos.size <= max_anchor_keep:
        return anchor_pos

    anchor_scores = base_scores[g_idx[anchor_pos]]
    top_local = np.argpartition(-anchor_scores, max_anchor_keep - 1)[:max_anchor_keep]
    return anchor_pos[top_local]


def _sample_group_into_pi(
    pi: np.ndarray,
    g_idx: np.ndarray,
    base_scores: np.ndarray,
    anchor_mask: np.ndarray,
    config: SmartSamplerConfig,
    rng: np.random.Generator,
) -> None:
    n_g = g_idx.size
    target_g = max(config.min_per_group, int(np.floor(config.sample_frac * n_g)))
    if target_g >= n_g:
        _add_rows(pi, g_idx, np.ones(n_g, dtype=np.float32))
        return

    anchor_pos = _cap_group_anchors(g_idx, anchor_mask, base_scores, target_g, config)
    g_anchor = g_idx[anchor_pos]
    if g_anchor.size:
        _add_rows(pi, g_anchor, np.ones(g_anchor.size, dtype=np.float32))

    pool_mask = np.ones(n_g, dtype=bool)
    if anchor_pos.size:
        pool_mask[anchor_pos] = False
    pool = g_idx[pool_mask]
    remaining = max(0, target_g - g_anchor.size)

    if remaining <= 0 or pool.size <= 0:
        return

    chosen_local, pi_local = _poisson_calibrated(base_scores[pool], remaining, rng)
    if chosen_local.size:
        _add_rows(pi, pool[chosen_local], pi_local[chosen_local])

    short = remaining - chosen_local.size
    if short <= 0:
        return

    leftover_mask = np.ones(pool.size, dtype=bool)
    if chosen_local.size:
        leftover_mask[chosen_local] = False
    leftover = pool[leftover_mask]
    if leftover.size == 0:
        return

    n_need = min(short, leftover.size)
    if n_need > 0:
        top_local = np.argpartition(-base_scores[leftover], n_need - 1)[:n_need]
        need = leftover[top_local]
        _add_rows(pi, need, np.ones(need.size, dtype=np.float32))


def _sample_inclusion_probabilities(
    arrays: SmartSampleArrays,
    scores: SmartSampleScores,
    anchor_mask: np.ndarray,
    config: SmartSamplerConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    pi = np.zeros(len(arrays.df), dtype=np.float32)
    for g_idx in arrays.group_indices.values():
        _sample_group_into_pi(pi, g_idx, scores.base, anchor_mask, config, rng)
    return pi


def _assemble_sample(
    arrays: SmartSampleArrays,
    pi: np.ndarray,
    config: SmartSamplerConfig,
) -> pd.DataFrame:
    final_idx = np.flatnonzero(pi > 0)
    final_pi = pi[final_idx].astype(np.float32, copy=True)
    final_pi = np.clip(final_pi, 1e-12, 1.0)
    final_w = 1.0 / final_pi

    if 0.5 < config.weight_clip_quantile < 1.0 and final_w.size > 1:
        cap = np.quantile(final_w, config.weight_clip_quantile)
        final_w = np.minimum(final_w, cap)
    final_w /= final_w.mean() + 1e-12

    out = arrays.df.iloc[final_idx].copy()
    out["sample_weight"] = final_w

    if config.verbose:
        n_groups = out[config.group_col].nunique() if config.group_col else 1
        total_groups = len(arrays.group_indices)
        print(
            f"Sampled {len(out):,} rows ({len(out)/len(arrays.df):.1%}), "
            f"{n_groups:,}/{total_groups:,} groups"
        )

    return out.reset_index(drop=True)

def smart_sample(
    df: pd.DataFrame,
    feature_cols: List[str],
    y_col: str,
    config: Optional[SmartSamplerConfig] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Informative subsampler for large datasets.

    Combines leverage-based geometric sampling with residual-based hard case
    detection. Returns a sample with approximate inverse-probability weights
    for bias reduction in downstream estimation.

    Note: The weights are approximate due to deterministic top-up for minimum
    group coverage. They reduce bias but are not exact Horvitz-Thompson weights.

    Parameters
    ----------
    df : DataFrame
        Input data with features, target, and optionally grouping columns.
    feature_cols : list of str
        Feature column names.
    y_col : str
        Target column name.
    config : SmartSamplerConfig, optional
        Configuration object. If None, uses defaults with any kwargs overrides.
    **kwargs
        Override any SmartSamplerConfig parameters.

    Returns
    -------
    DataFrame
        Sampled data with 'sample_weight' column (approximate inverse
        inclusion probability, mean-normalized).
    """
    config = _resolve_smart_config(config, kwargs)
    rng = np.random.default_rng(config.random_state)
    arrays = _prepare_smart_arrays(df, feature_cols, y_col, config)

    if config.verbose:
        n, d = arrays.Xs.shape
        target_total = int(np.floor(config.sample_frac * n))
        print(
            f"Smart sampler: {n:,} rows × {d} features → target "
            f"{target_total:,} ({config.sample_frac:.1%})"
        )

    scores = _compute_sampling_scores(arrays, config, rng)
    anchor_mask = _build_anchor_mask(arrays.df, config)
    pi = _sample_inclusion_probabilities(arrays, scores, anchor_mask, config, rng)
    return _assemble_sample(arrays, pi, config)


__all__ = [
    "SmartSamplerConfig",
    "leverage_scores_multi_alpha",
    "smart_sample",
]
