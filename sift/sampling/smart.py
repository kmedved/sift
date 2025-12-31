import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Union, Callable
from dataclasses import dataclass
import warnings

from sklearn.preprocessing import StandardScaler
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
    weight_clip_quantile : float
        Quantile for clipping extreme weights.
    residual_weight_cap : float
        Maximum weight for residual-based scores (vs leverage).
    uniform_floor : float
        Minimum base probability (ensures coverage).
    anchor_fn : callable, optional
        Function(df, group_col, time_col) -> boolean mask identifying anchor rows.
        Anchors are always included with probability 1.
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
    weight_clip_quantile: float = 0.99
    residual_weight_cap: float = 0.4
    uniform_floor: float = 0.05
    anchor_fn: Optional[Callable] = None
    anchor_max_share: float = 0.4
    random_state: Optional[int] = 42
    verbose: bool = True


# =============================================================================
# Smart Sampler
# =============================================================================

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
    # Build config
    if config is None:
        config = SmartSamplerConfig(**kwargs)
    else:
        # Apply any overrides
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)

    rng = np.random.default_rng(config.random_state)

    # Validate columns
    required = set(feature_cols + [y_col])
    if config.group_col:
        required.add(config.group_col)
    if config.time_col:
        required.add(config.time_col)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if not (0 < config.sample_frac <= 1):
        raise ValueError("sample_frac must be in (0, 1].")

    df = df.reset_index(drop=True)

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[y_col].to_numpy(dtype=np.float32)

    # Treat inf as missing before imputation
    X = np.where(np.isfinite(X), X, np.nan)

    # Check y finiteness (only needed when residual weighting is enabled)
    if config.residual_weight_cap > 0 and not np.isfinite(y).all():
        raise ValueError("y must be finite (no NaN or inf) when residual_weight_cap > 0")

    # Mean-impute missing values (replace all-NaN columns with 0)
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)  # Handle all-NaN columns
    nan_mask = np.isnan(X)
    if nan_mask.any():
        X[nan_mask] = col_means[np.where(nan_mask)[1]]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype(np.float32)

    # Verify finiteness after scaling
    if not np.isfinite(Xs).all():
        raise ValueError(
            "X contains non-finite values after imputation/scaling; check for extreme magnitudes."
        )

    n, d = Xs.shape
    target_total = int(np.floor(config.sample_frac * n))

    if config.verbose:
        print(f"Smart sampler: {n:,} rows × {d} features → target {target_total:,} ({config.sample_frac:.1%})")

    # Build group indices
    if config.group_col:
        group_indices: Dict[Union[int, str], np.ndarray] = {
            g: idx for g, idx in df.groupby(config.group_col, sort=False).indices.items()
        }
    else:
        group_indices = {'_all': np.arange(n)}

    # -------------------------------------------------------------------------
    # Randomized SVD for leverage scores
    # -------------------------------------------------------------------------
    k = int(min(128, d, max(16, np.ceil(np.log2(d + 1)) * 8)))
    k = min(k, max(1, min(n, d) - 1))  # Cap by matrix dimensions
    try:
        _, S, Vt = randomized_svd(Xs, n_components=k, n_iter=4, random_state=config.random_state)
        V = Vt.T.astype(np.float32)
        S = S.astype(np.float32)
        s2 = S * S
    except Exception as e:
        warnings.warn(f"SVD failed ({e}); using uniform geometry scores.", RuntimeWarning)
        V, S, s2 = None, None, None

    # -------------------------------------------------------------------------
    # Multi-alpha leverage scores
    # -------------------------------------------------------------------------
    lev_scores = leverage_scores_multi_alpha(Xs, V, S, s2, config.leverage_batch_size)

    # -------------------------------------------------------------------------
    # Pilot model for residual-based scores
    # -------------------------------------------------------------------------
    ps = min(n, config.pilot_sample_size)
    pilot_all = rng.choice(n, size=ps, replace=False)

    # Ensure we have both train and val sets (handle small n)
    min_val_size = min(100, ps // 4)  # At least 100 or 25% of pilot
    half = max(min_val_size, ps // 2)
    half = min(half, ps - min_val_size)  # Ensure val has at least min_val_size

    pilot_train = pilot_all[:half]
    pilot_val = pilot_all[half:]

    beta = 0.0
    res_scores = np.ones(n, dtype=np.float32)

    # Skip pilot if residual weighting is disabled or we don't have enough data
    if config.residual_weight_cap > 0:
        if len(pilot_train) >= 50 and len(pilot_val) >= 20:
            try:
                pilot = HistGradientBoostingRegressor(
                    max_iter=50, max_depth=4, learning_rate=0.1,
                    random_state=config.random_state
                )
                pilot.fit(Xs[pilot_train], y[pilot_train])

                val_pred = pilot.predict(Xs[pilot_val])
                val_resid = y[pilot_val] - val_pred
                val_mse = float(np.mean(val_resid ** 2))
                var_y = float(np.var(y[pilot_val])) + 1e-12
                r2 = max(0.0, min(1.0, 1.0 - val_mse / var_y))

                preds = pilot.predict(Xs)
                resid_all = np.abs(y - preds).astype(np.float32)
                res_scores = np.maximum(resid_all, 1e-12)
                res_scores /= res_scores.mean()

                beta = min(config.residual_weight_cap, r2)
                if config.verbose:
                    print(f"Pilot R² = {r2:.3f} → residual weight β = {beta:.3f}")
            except Exception as e:
                warnings.warn(f"Pilot model failed ({e}); using geometry only.", RuntimeWarning)
        elif config.verbose:
            warnings.warn(f"Dataset too small for pilot model (n={n}); using geometry only.", RuntimeWarning)

    base_scores = (1.0 - beta) * lev_scores + beta * res_scores
    base_scores = (1 - config.uniform_floor) * (base_scores / (base_scores.mean() + 1e-12)) + config.uniform_floor

    # -------------------------------------------------------------------------
    # Anchor points
    # -------------------------------------------------------------------------
    if config.anchor_fn is not None:
        anchor_mask = config.anchor_fn(df, config.group_col, config.time_col)
        if config.verbose and anchor_mask.any():
            print(f"Anchors: {anchor_mask.sum():,} rows")
    else:
        anchor_mask = np.zeros(n, dtype=bool)

    # -------------------------------------------------------------------------
    # Poisson sampling with calibration
    # -------------------------------------------------------------------------
    def poisson_calibrated(local_scores: np.ndarray, budget: int) -> Tuple[np.ndarray, np.ndarray]:
        m = local_scores.size
        if budget <= 0 or m == 0:
            return np.array([], dtype=int), np.zeros(0, dtype=np.float32)

        s = np.maximum(local_scores.astype(np.float64), 0.0)
        s_sum = s.sum()
        if s_sum == 0:
            # All scores zero - fall back to uniform
            p = np.full(m, 1.0 / m, dtype=np.float64)
        else:
            p = s / s_sum

        # Guard against p.max() == 0
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
        tau = hi
        pi = np.minimum(1.0, tau * p)

        chosen = (rng.random(m) < pi)
        chosen_idx = np.nonzero(chosen)[0]
        return chosen_idx.astype(int), pi.astype(np.float32)

    pi_dict: Dict[int, float] = {}

    def add_rows(indices: np.ndarray, pis: np.ndarray):
        for j, ridx in enumerate(indices):
            pi_dict[ridx] = min(1.0, pi_dict.get(ridx, 0.0) + float(pis[j]))

    for g, g_idx in group_indices.items():
        n_g = g_idx.size
        target_g = max(config.min_per_group, int(np.floor(config.sample_frac * n_g)))
        if target_g >= n_g:
            add_rows(g_idx, np.ones(n_g, dtype=np.float32))
            continue

        # Anchors for this group (cap share)
        g_anchor = g_idx[anchor_mask[g_idx]]
        if g_anchor.size:
            if config.anchor_max_share <= 0:
                g_anchor = np.array([], dtype=int)
            else:
                max_anchor_keep = max(1, int(np.floor(config.anchor_max_share * target_g)))
                max_anchor_keep = min(max_anchor_keep, target_g)
                if g_anchor.size > max_anchor_keep:
                    top_local = np.argpartition(-base_scores[g_anchor], max_anchor_keep - 1)[:max_anchor_keep]
                    g_anchor = g_anchor[top_local]
        if g_anchor.size:
            add_rows(g_anchor, np.ones(g_anchor.size, dtype=np.float32))

        pool = g_idx[~np.isin(g_idx, g_anchor)]
        remaining = max(0, target_g - g_anchor.size)

        if remaining > 0 and pool.size > 0:
            chosen_local, pi_local = poisson_calibrated(base_scores[pool], budget=remaining)

            if chosen_local.size:
                add_rows(pool[chosen_local], pi_local[chosen_local])

            short = remaining - chosen_local.size
            if short > 0:
                leftover_mask = np.ones(pool.size, dtype=bool)
                if chosen_local.size:
                    leftover_mask[chosen_local] = False
                leftover = pool[leftover_mask]
                if leftover.size > 0:
                    n_need = min(short, leftover.size)
                    if n_need > 0:
                        top_local = np.argpartition(-base_scores[leftover], n_need - 1)[:n_need]
                        need = leftover[top_local]
                        add_rows(need, np.ones(need.size, dtype=np.float32))

    # -------------------------------------------------------------------------
    # Assemble output
    # -------------------------------------------------------------------------
    final_idx = np.fromiter(pi_dict.keys(), dtype=int, count=len(pi_dict))
    final_pi = np.fromiter(pi_dict.values(), dtype=np.float32, count=len(pi_dict))
    final_pi = np.clip(final_pi, 1e-12, 1.0)
    final_w = 1.0 / final_pi

    if 0.5 < config.weight_clip_quantile < 1.0 and final_w.size > 1:
        cap = np.quantile(final_w, config.weight_clip_quantile)
        final_w = np.minimum(final_w, cap)
    final_w /= (final_w.mean() + 1e-12)

    out = df.iloc[final_idx].copy()
    out['sample_weight'] = final_w

    if config.verbose:
        n_groups = out[config.group_col].nunique() if config.group_col else 1
        total_groups = len(group_indices)
        print(f"Sampled {len(out):,} rows ({len(out)/n:.1%}), {n_groups:,}/{total_groups:,} groups")

    return out.reset_index(drop=True)


__all__ = [
    "SmartSamplerConfig",
    "leverage_scores_multi_alpha",
    "smart_sample",
]
