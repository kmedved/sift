"""
Stability Selection with Optional Smart Sampling

A robust feature selection module for linear models with large datasets.
Combines stability selection (Meinshausen & Bühlmann, 2010) with optional
leverage-based smart sampling for computational efficiency.

Domain-agnostic: works for sports analytics, finance, biomedical, etc.

Usage:
    from sift import StabilitySelector

    # Basic usage
    selector = StabilitySelector(threshold=0.6)
    selector.fit(X, y)
    X_selected = selector.transform(X)

    # With smart sampling for large grouped/panel data
    selector = StabilitySelector(
        threshold=0.6,
        use_smart_sampler=True,
        sampler_config=panel_config('user_id', 'timestamp', sample_frac=0.15)
    )
    selector.fit(df, y)
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Union, Callable
from dataclasses import dataclass
import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import (
    Lasso, LassoCV, ElasticNet, ElasticNetCV,
    LogisticRegression, LogisticRegressionCV
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.extmath import randomized_svd
from sklearn.ensemble import HistGradientBoostingRegressor
from joblib import Parallel, delayed


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
# Built-in Anchor Functions
# =============================================================================

def no_anchors(df: pd.DataFrame, group_col: str, time_col: str) -> np.ndarray:
    """No anchor points."""
    return np.zeros(len(df), dtype=bool)


def first_per_group(df: pd.DataFrame, group_col: str, time_col: str) -> np.ndarray:
    """Anchor the first observation per group (by time if available)."""
    if group_col is None:
        return np.zeros(len(df), dtype=bool)

    if time_col and time_col in df.columns:
        df_sorted = df.sort_values([group_col, time_col], kind="mergesort")
        mask = ~df_sorted.duplicated(subset=[group_col], keep='first')
        return mask.reindex(df.index).values
    else:
        return (~df.duplicated(subset=[group_col], keep='first')).values


def first_and_last_per_group(df: pd.DataFrame, group_col: str, time_col: str) -> np.ndarray:
    """Anchor first and last observation per group (by time if available)."""
    if group_col is None:
        return np.zeros(len(df), dtype=bool)

    if time_col and time_col in df.columns:
        df_sorted = df.sort_values([group_col, time_col], kind="mergesort")
        first = ~df_sorted.duplicated(subset=[group_col], keep='first')
        last = ~df_sorted.duplicated(subset=[group_col], keep='last')
        mask = first | last
        return mask.reindex(df.index).values
    else:
        first = ~df.duplicated(subset=[group_col], keep='first')
        last = ~df.duplicated(subset=[group_col], keep='last')
        return (first | last).values


def periodic_anchors(period_cols: Union[str, List[str]], keep_first: bool = True):
    """
    Factory for anchoring first observation per (group, period).

    Parameters
    ----------
    period_cols : str or list of str
        Column(s) defining the period. Can be a single column like 'month'
        or multiple columns like ['month', 'year'].
    keep_first : bool
        If True, keep first observation per period. If False, keep last.

    Examples
    --------
    periodic_anchors('month')  # First row per group per month
    periodic_anchors(['season', 'month'])  # First row per group per season-month
    """
    # Normalize to list
    if isinstance(period_cols, str):
        period_cols = [period_cols]

    def anchor_fn(df: pd.DataFrame, group_col: str, time_col: str) -> np.ndarray:
        if group_col is None:
            return np.zeros(len(df), dtype=bool)

        # Check all period columns exist
        missing = [c for c in period_cols if c not in df.columns]
        if missing:
            return np.zeros(len(df), dtype=bool)

        subset = [group_col] + period_cols
        if time_col and time_col in df.columns:
            sort_cols = subset + [time_col]
            df_sorted = df.sort_values(sort_cols, kind="mergesort")
            mask = ~df_sorted.duplicated(subset=subset, keep='first' if keep_first else 'last')
            return mask.reindex(df.index).values

        return (~df.duplicated(subset=subset, keep='first' if keep_first else 'last')).values

    return anchor_fn


def quantile_anchors(
    value_col: str,
    quantile: float = 0.95,
    per_group: bool = True,
    subgroup_cols: Optional[List[str]] = None
):
    """
    Factory for anchoring rows above a quantile threshold.

    Parameters
    ----------
    value_col : str
        Column containing values to compute quantile on.
    quantile : float
        Quantile threshold (0-1). Rows at or above this quantile are anchored.
    per_group : bool
        If True, compute quantile within each group. If False, global quantile.
    subgroup_cols : list of str, optional
        Additional columns to group by when computing quantiles.
        E.g., ['season'] computes quantile per (group, season).

    Examples
    --------
    quantile_anchors('usage', 0.95)  # Top 5% usage per group
    quantile_anchors('poss_pct', 0.90, subgroup_cols=['season'])  # Top 10% per group-season
    """
    def anchor_fn(df: pd.DataFrame, group_col: str, time_col: str) -> np.ndarray:
        if value_col not in df.columns:
            return np.zeros(len(df), dtype=bool)

        if per_group and group_col:
            # Build groupby columns
            groupby_cols = [group_col]
            if subgroup_cols:
                # Filter to columns that exist
                valid_subgroups = [c for c in subgroup_cols if c in df.columns]
                groupby_cols.extend(valid_subgroups)

            threshold = df.groupby(groupby_cols)[value_col].transform(
                lambda s: s.quantile(quantile)
            )
        else:
            threshold = df[value_col].quantile(quantile)

        return (df[value_col] >= threshold).values

    return anchor_fn


def event_window_anchors(event_col: str, window: int = 3):
    """
    Factory for anchoring rows within a window of an event.

    Example: event_window_anchors('team_change', 3) anchors ±3 rows around changes.
    """
    def anchor_fn(df: pd.DataFrame, group_col: str, time_col: str) -> np.ndarray:
        if event_col not in df.columns:
            return np.zeros(len(df), dtype=bool)

        mask = np.zeros(len(df), dtype=bool)

        if group_col is None:
            event_idx = np.where(df[event_col].values == 1)[0]
            for idx in event_idx:
                lo = max(0, idx - window)
                hi = min(len(df), idx + window + 1)
                mask[lo:hi] = True
        else:
            for _, g_idx in df.groupby(group_col, sort=False).indices.items():
                if len(g_idx) == 0:
                    continue

                # Sort by time within group if available
                if time_col and time_col in df.columns:
                    order = np.argsort(df[time_col].iloc[g_idx].values)
                    g_sorted = g_idx[order]
                else:
                    g_sorted = g_idx

                event_local = np.where(df[event_col].iloc[g_sorted].values == 1)[0]
                for loc in event_local:
                    lo = max(0, loc - window)
                    hi = min(len(g_sorted), loc + window + 1)
                    mask[g_sorted[lo:hi]] = True

        return mask

    return anchor_fn


def combine_anchors(*anchor_fns):
    """Combine multiple anchor functions with OR logic."""
    def combined(df: pd.DataFrame, group_col: str, time_col: str) -> np.ndarray:
        mask = np.zeros(len(df), dtype=bool)
        for fn in anchor_fns:
            mask |= fn(df, group_col, time_col)
        return mask
    return combined


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
    def leverage_scores_multi_alpha() -> np.ndarray:
        if V is None:
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
        B = config.leverage_batch_size
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

    lev_scores = leverage_scores_multi_alpha()

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


# =============================================================================
# Stability Selector
# =============================================================================

class StabilitySelector(BaseEstimator, TransformerMixin):
    """
    Stability selection for linear models with optional smart sampling.

    Fits Lasso/ElasticNet (regression) or LogisticRegression (classification)
    on bootstrap subsamples and keeps features selected consistently across runs.
    Handles correlated features by revealing which ones are robustly predictive
    vs. interchangeable proxies.

    Note: This is a practical stability selection implementation inspired by
    Meinshausen & Bühlmann (2010), but does not provide formal false-positive
    control. Use it as a robust heuristic for pre-filtering features.

    Parameters
    ----------
    n_bootstrap : int, default=50
        Number of bootstrap iterations.
    sample_frac : float, default=0.5
        Fraction of data to use in each bootstrap sample.
    threshold : float, default=0.6
        Minimum selection frequency to keep a feature.
    alpha : float, optional
        Regularization strength. If None, estimated via CV.
    l1_ratio : float, default=1.0
        ElasticNet mixing (1.0 = Lasso, <1.0 = ElasticNet). Only for regression.
    task : str, default='regression'
        Either 'regression' or 'classification'.
    max_features : int, optional
        Hard cap on number of selected features.
    use_smart_sampler : bool, default=False
        Whether to apply smart sampling before stability selection.
    sampler_config : SmartSamplerConfig, optional
        Configuration for smart sampler.
    store_coefs : bool, default=True
        Whether to store full coefficient matrix from all bootstraps.
        Set False to save memory (disables get_coef_stability and plot_coef_distributions).
    coef_threshold : float, default=1e-8
        Threshold for considering a coefficient as non-zero.
    n_jobs : int, default=-1
        Number of parallel jobs (-1 = all cores).
    parallel_backend : str, default='threads'
        Joblib backend preference. 'threads' has lower memory overhead,
        'processes' is more isolated. Set to None for joblib default.
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, default=True
        Print progress information.

    Attributes
    ----------
    selection_frequencies_ : ndarray of shape (n_features,)
        Fraction of bootstrap runs in which each feature was selected.
    selected_features_ : ndarray
        Indices of selected features.
    selected_feature_names_ : list of str
        Names of selected features.
    n_features_selected_ : int
        Number of selected features.
    alpha_ : float
        Regularization alpha used.
    coef_bootstrap_ : ndarray of shape (n_bootstrap, n_features), optional
        Coefficients from each bootstrap run. Only available if store_coefs=True.
    """

    def __init__(
        self,
        n_bootstrap: int = 50,
        sample_frac: float = 0.5,
        threshold: float = 0.6,
        alpha: Optional[float] = None,
        l1_ratio: float = 1.0,
        task: str = 'regression',
        max_features: Optional[int] = None,
        use_smart_sampler: bool = False,
        sampler_config: Optional[SmartSamplerConfig] = None,
        store_coefs: bool = True,
        coef_threshold: float = 1e-8,
        n_jobs: int = -1,
        parallel_backend: str = 'threads',
        random_state: Optional[int] = None,
        verbose: bool = True
    ):
        self.n_bootstrap = n_bootstrap
        self.sample_frac = sample_frac
        self.threshold = threshold
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.task = task
        self.max_features = max_features
        self.use_smart_sampler = use_smart_sampler
        self.sampler_config = sampler_config
        self.store_coefs = store_coefs
        self.coef_threshold = coef_threshold
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend
        self.random_state = random_state
        self.verbose = verbose

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        sample_weight: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> 'StabilitySelector':
        """
        Run stability selection.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.
        feature_names : list of str, optional
            Feature names.

        Returns
        -------
        self
        """
        # Input validation
        if self.task not in ('regression', 'classification'):
            raise ValueError(f"task must be 'regression' or 'classification', got '{self.task}'")
        if self.use_smart_sampler:
            X, y, sample_weight, feature_names = self._apply_smart_sampler(X, y, sample_weight)
        else:
            X, y, sample_weight, feature_names = self._prep_arrays(
                X, y, sample_weight, feature_names
            )

        n, p = X.shape
        self.feature_names_in_ = feature_names
        self.n_features_in_ = p

        # Impute NaNs (smart_sample may return original rows with NaNs)
        if np.isnan(X).any():
            col_means = np.nanmean(X, axis=0)
            col_means = np.where(np.isnan(col_means), 0.0, col_means)
            nan_mask = np.isnan(X)
            X[nan_mask] = col_means[np.where(nan_mask)[1]]

        # Standardize
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Get alpha
        if self.alpha is None:
            self.alpha_ = self._find_alpha(X_scaled, y, sample_weight)
        else:
            self.alpha_ = self.alpha

        if self.verbose:
            task_str = 'classification' if self.task == 'classification' else 'regression'
            print(f"Stability selection ({task_str}): {self.n_bootstrap} bootstraps, "
                  f"α={self.alpha_:.4f}, threshold={self.threshold}")

        # Bootstrap
        rng = np.random.default_rng(self.random_state)
        seeds = rng.integers(0, 2**31, size=self.n_bootstrap)
        subsample_size = max(2, int(n * self.sample_frac))  # At least 2 samples
        subsample_size = min(subsample_size, n)  # Can't exceed n

        alpha = self.alpha_
        l1_ratio = self.l1_ratio
        task = self.task
        coef_threshold = self.coef_threshold

        # For classification, we need stratified sampling to avoid single-class subsamples
        is_classification = task == 'classification'
        if is_classification:
            classes = np.unique(y)
            n_classes = len(classes)
            # Pre-compute indices per class for stratified sampling
            class_indices = {c: np.where(y == c)[0] for c in classes}
            class_counts = np.array([len(class_indices[c]) for c in classes])

            # Ensure subsample_size >= n_classes
            if subsample_size < n_classes:
                subsample_size = n_classes

        def _stratified_indices(rng_local, subsample_size):
            """Get stratified sample indices that sum to exactly subsample_size."""
            # Proportional allocation
            props = class_counts / class_counts.sum()
            raw = props * subsample_size
            counts = np.floor(raw).astype(int)

            # Ensure at least 1 per class
            counts = np.maximum(counts, 1)
            # Cap by availability
            counts = np.minimum(counts, class_counts)

            # Adjust to hit exact subsample_size
            total = counts.sum()
            frac = raw - np.floor(raw)

            if total < subsample_size:
                need = subsample_size - total
                room = class_counts - counts
                order = np.argsort(-frac)  # Prioritize largest fractional parts
                for j in order:
                    if need == 0:
                        break
                    if room[j] > 0:
                        add = min(room[j], need)
                        counts[j] += add
                        need -= add
            elif total > subsample_size:
                extra = total - subsample_size
                order = np.argsort(-counts)  # Remove from largest first
                for j in order:
                    if extra == 0:
                        break
                    can_drop = counts[j] - 1  # Keep at least 1
                    if can_drop > 0:
                        drop = min(can_drop, extra)
                        counts[j] -= drop
                        extra -= drop

            idx_list = [
                rng_local.choice(class_indices[c], size=counts[i], replace=False)
                for i, c in enumerate(classes)
                if counts[i] > 0
            ]
            return np.concatenate(idx_list)

        def single_run(seed):
            rng_local = np.random.default_rng(seed)

            if is_classification:
                idx = _stratified_indices(rng_local, subsample_size)
            else:
                idx = rng_local.choice(n, size=subsample_size, replace=False)

            if task == 'classification':
                # C = 1/alpha for LogisticRegression
                model = LogisticRegression(
                    penalty='l1',
                    solver='saga',
                    C=1.0 / alpha,
                    max_iter=3000,
                    random_state=seed,  # Reproducibility
                    n_jobs=1  # Avoid nested parallelism
                )
            elif l1_ratio >= 1.0:
                model = Lasso(alpha=alpha, max_iter=3000)
            else:
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=3000)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_scaled[idx], y[idx], sample_weight=sample_weight[idx])

            coef = model.coef_

            # Handle coefficient shapes
            if coef.ndim == 2:
                if coef.shape[0] == 1:
                    # Binary classification: preserve sign
                    coef_flat = coef[0]
                    selected = np.abs(coef_flat) > coef_threshold
                    coef_summary = coef_flat  # Signed coefficients
                else:
                    # True multiclass: aggregate across classes
                    selected = np.any(np.abs(coef) > coef_threshold, axis=0)
                    # For multiclass, take max abs (sign is ambiguous)
                    coef_summary = np.max(np.abs(coef), axis=0)
            else:
                # Regression
                selected = np.abs(coef) > coef_threshold
                coef_summary = coef.ravel()

            return selected.astype(np.int8), coef_summary.astype(np.float32)

        # Chunked execution to reduce peak memory
        # (avoids holding all n_bootstrap results in memory at once)
        chunk_size = min(20, self.n_bootstrap)  # Process 20 at a time

        sel_count = np.zeros(p, dtype=np.int32)
        sum_abs_coef = np.zeros(p, dtype=np.float64)

        if self.store_coefs:
            self.coef_bootstrap_ = np.empty((self.n_bootstrap, p), dtype=np.float32)

        bootstrap_idx = 0
        for chunk_start in range(0, self.n_bootstrap, chunk_size):
            chunk_end = min(chunk_start + chunk_size, self.n_bootstrap)
            chunk_seeds = seeds[chunk_start:chunk_end]

            chunk_results = Parallel(n_jobs=self.n_jobs, prefer=self.parallel_backend)(
                delayed(single_run)(seed) for seed in chunk_seeds
            )

            # Aggregate this chunk immediately, then discard
            for selected, coef_summary in chunk_results:
                sel_count += selected.astype(np.int32)
                sum_abs_coef += np.abs(coef_summary)

                if self.store_coefs:
                    self.coef_bootstrap_[bootstrap_idx] = coef_summary
                bootstrap_idx += 1

            # chunk_results goes out of scope here, memory freed

        self.selection_frequencies_ = (sel_count / self.n_bootstrap).astype(np.float32)
        self.mean_abs_coef_ = (sum_abs_coef / self.n_bootstrap).astype(np.float32)

        # Select features
        mask = self.selection_frequencies_ >= self.threshold

        if self.max_features is not None and mask.sum() > self.max_features:
            top_idx = np.argsort(-self.selection_frequencies_, kind="mergesort")[:self.max_features]
            mask = np.zeros(p, dtype=bool)
            mask[top_idx] = True

        selected = np.where(mask)[0]
        order = np.argsort(-self.selection_frequencies_[selected], kind="mergesort")
        self.selected_features_ = selected[order]
        self.selected_feature_names_ = [feature_names[i] for i in self.selected_features_]
        self.n_features_selected_ = len(self.selected_features_)

        if self.verbose:
            print(f"Selected {self.n_features_selected_} / {p} features")

        return self

    def _prep_arrays(
        self,
        X,
        y,
        sample_weight,
        feature_names
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Convert inputs to arrays, extract feature names."""
        exclude = set()
        if self.use_smart_sampler and self.sampler_config:
            if self.sampler_config.group_col:
                exclude.add(self.sampler_config.group_col)
            if self.sampler_config.time_col:
                exclude.add(self.sampler_config.time_col)

        if isinstance(X, pd.DataFrame):
            feature_names = feature_names or [c for c in X.columns if c not in exclude]
            X = X[feature_names].values
        else:
            feature_names = feature_names or [f"x{i}" for i in range(X.shape[1])]

        if isinstance(y, pd.Series):
            y = y.values

        X = np.asarray(X, dtype=np.float32)
        X = np.where(np.isfinite(X), X, np.nan)

        # Handle labels properly for classification
        if self.task == 'classification':
            # Use LabelEncoder to handle string/categorical labels
            self._label_encoder = LabelEncoder()
            y = self._label_encoder.fit_transform(y).astype(np.int32)
            self.classes_ = self._label_encoder.classes_
        else:
            y = np.asarray(y, dtype=np.float32)

        if sample_weight is None:
            sample_weight = np.ones(len(y), dtype=np.float32)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float32)

        return X, y, sample_weight, feature_names

    def _apply_smart_sampler(
        self,
        X,
        y,
        sample_weight=None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Apply smart sampler to reduce data size."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("use_smart_sampler=True requires X to be a DataFrame")

        # Don't allow user sample_weight with smart sampler - they conflict
        if sample_weight is not None:
            raise ValueError(
                "Cannot use both sample_weight and use_smart_sampler=True. "
                "The smart sampler generates its own weights. Either pass sample_weight "
                "with use_smart_sampler=False, or let the smart sampler generate weights."
            )

        config = self.sampler_config or SmartSamplerConfig()
        # Fix: use `is None` check to handle random_state=0
        if config.random_state is None:
            config.random_state = self.random_state if self.random_state is not None else 42
        config.verbose = self.verbose

        # For classification, disable residual-based sampling (regression on class IDs is meaningless)
        # Use geometry-only sampling (leverage + uniform floor + anchors)
        if self.task == 'classification':
            config.residual_weight_cap = 0.0

        exclude = set()
        if config.group_col:
            exclude.add(config.group_col)
        if config.time_col:
            exclude.add(config.time_col)
        feature_names = [c for c in X.columns if c not in exclude]

        # Build df with encoded y BEFORE sampling
        df = X.copy()

        if isinstance(y, pd.Series):
            y_raw = y.values
        else:
            y_raw = np.asarray(y)

        if self.task == 'classification':
            # Encode labels BEFORE sampling so string labels work
            self._label_encoder = LabelEncoder()
            y_enc = self._label_encoder.fit_transform(y_raw).astype(np.int32)
            self.classes_ = self._label_encoder.classes_
            y_col = '_y_enc'
            df[y_col] = y_enc
        else:
            y_col = '_y'
            df[y_col] = y_raw.astype(np.float32)

        sampled = smart_sample(
            df=df,
            feature_cols=feature_names,
            y_col=y_col,
            config=config
        )

        X_out = sampled[feature_names].values.astype(np.float32)
        weights_out = sampled['sample_weight'].values.astype(np.float32)

        if self.task == 'classification':
            y_out = sampled[y_col].values.astype(np.int32)

            # Check that all classes survived sampling
            present_classes = np.unique(y_out)
            if len(present_classes) != len(self.classes_):
                missing = set(range(len(self.classes_))) - set(present_classes)
                missing_labels = [self.classes_[i] for i in missing]
                raise ValueError(
                    f"Smart sampler dropped class(es): {missing_labels}. "
                    f"Increase sample_frac or disable use_smart_sampler for classification."
                )
        else:
            y_out = sampled[y_col].values.astype(np.float32)

        self.sampled_n_ = len(sampled)

        return X_out, y_out, weights_out, feature_names

    def _find_alpha(
        self,
        X_scaled: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray
    ) -> float:
        """Estimate alpha via CV on subsample."""
        n = X_scaled.shape[0]
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(n, size=min(30_000, n), replace=False)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if self.task == 'classification':
                # LogisticRegressionCV uses C (inverse of alpha)
                # We search over C values and convert back
                cv_model = LogisticRegressionCV(
                    penalty='l1',
                    solver='saga',
                    cv=3,
                    Cs=20,
                    max_iter=2000,
                    random_state=self.random_state,
                    n_jobs=1
                )
                cv_model.fit(X_scaled[idx], y[idx], sample_weight=sample_weight[idx])

                # C_ can be scalar or per-class array for multiclass
                C = cv_model.C_
                if np.ndim(C) > 0 and len(C) > 1:
                    C_best = float(np.mean(C))  # Average across classes
                else:
                    C_best = float(C[0]) if np.ndim(C) > 0 else float(C)
                return 1.0 / C_best
            elif self.l1_ratio >= 1.0:
                cv_model = LassoCV(cv=3, n_alphas=30, max_iter=2000)
            else:
                cv_model = ElasticNetCV(
                    l1_ratio=self.l1_ratio, cv=3, n_alphas=30, max_iter=2000
                )

            cv_model.fit(X_scaled[idx], y[idx], sample_weight=sample_weight[idx])

        return cv_model.alpha_

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Reduce X to selected features."""
        if isinstance(X, pd.DataFrame):
            return X[self.selected_feature_names_].values
        return np.asarray(X)[:, self.selected_features_]

    def fit_transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        **fit_params
    ) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y, **fit_params).transform(X)

    def get_feature_info(self) -> pd.DataFrame:
        """
        Get DataFrame with feature selection details.

        Returns
        -------
        DataFrame with columns:
            feature: name
            frequency: selection frequency
            mean_abs_coef: mean absolute coefficient across bootstraps
            selected: whether it passed threshold
        """
        return pd.DataFrame({
            'feature': self.feature_names_in_,
            'frequency': self.selection_frequencies_,
            'mean_abs_coef': self.mean_abs_coef_,
            'selected': self.selection_frequencies_ >= self.threshold
        }).sort_values('frequency', ascending=False).reset_index(drop=True)

    def get_support(self, indices: bool = False) -> np.ndarray:
        """Get mask or indices of selected features."""
        if indices:
            return self.selected_features_
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.selected_features_] = True
        return mask

    def get_coef_stability(self) -> pd.DataFrame:
        """
        Get coefficient stability analysis.

        Returns DataFrame with mean, std, and CV of coefficients
        across bootstrap runs for each feature.

        Note on coefficient semantics:
        - Regression: signed coefficients (can be positive or negative)
        - Binary classification: signed coefficients (positive = class 1)
        - Multiclass classification: absolute coefficients (max abs across classes)

        Requires store_coefs=True (default).
        """
        if not hasattr(self, 'coef_bootstrap_'):
            raise ValueError(
                "Coefficient matrix not available. "
                "Set store_coefs=True when creating the selector."
            )

        coef_mean = self.coef_bootstrap_.mean(axis=0)
        coef_std = self.coef_bootstrap_.std(axis=0)
        coef_cv = np.where(
            np.abs(coef_mean) > 1e-10,
            coef_std / np.abs(coef_mean),
            np.inf
        )

        return pd.DataFrame({
            'feature': self.feature_names_in_,
            'coef_mean': coef_mean,
            'coef_std': coef_std,
            'coef_cv': coef_cv,
            'frequency': self.selection_frequencies_,
            'selected': self.selection_frequencies_ >= self.threshold
        }).sort_values('frequency', ascending=False).reset_index(drop=True)

    def tune_threshold(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        thresholds: List[float] = [0.4, 0.5, 0.6, 0.7, 0.8],
        cv: int = 3,
        scoring: Optional[str] = None
    ) -> Tuple[float, pd.DataFrame]:
        """
        Find optimal threshold by cross-validating downstream model performance.

        Must be called after fit(). Tests each threshold and evaluates
        ElasticNet (regression) or LogisticRegression (classification)
        performance on the selected feature subset.

        Parameters
        ----------
        X : array-like or DataFrame
            Training data (same as used in fit, or held-out).
        y : array-like
            Target values.
        thresholds : list of float
            Threshold values to test.
        cv : int
            Number of cross-validation folds.
        scoring : str, optional
            Scoring metric. Default: 'r2' for regression, 'accuracy' for classification.

        Returns
        -------
        best_threshold : float
            Threshold with highest CV score.
        results : DataFrame
            Threshold, n_features, mean_score, std_score for each threshold tested.
        """
        from sklearn.model_selection import cross_val_score

        if not hasattr(self, 'selection_frequencies_'):
            raise ValueError("Must call fit() before tune_threshold()")

        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names_in_].values
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        # Scale X
        X_scaled = self._scaler.transform(X)

        # Default scoring
        if scoring is None:
            scoring = 'accuracy' if self.task == 'classification' else 'r2'

        results = []
        for thresh in thresholds:
            mask = self.selection_frequencies_ >= thresh
            n_selected = mask.sum()

            if n_selected == 0:
                results.append({
                    'threshold': thresh,
                    'n_features': 0,
                    'mean_score': np.nan,
                    'std_score': np.nan
                })
                continue

            X_subset = X_scaled[:, mask]

            # Fit downstream model
            if self.task == 'classification':
                model = LogisticRegression(
                    penalty='l2',
                    solver='lbfgs',
                    max_iter=1000
                )
            else:
                model = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=2000)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(model, X_subset, y, cv=cv, scoring=scoring)

            results.append({
                'threshold': thresh,
                'n_features': n_selected,
                'mean_score': scores.mean(),
                'std_score': scores.std()
            })

        results_df = pd.DataFrame(results)

        # Find best (highest mean score, ignoring NaN)
        valid = results_df.dropna(subset=['mean_score'])
        if len(valid) == 0:
            best_thresh = thresholds[0]
        else:
            best_idx = valid['mean_score'].idxmax()
            best_thresh = valid.loc[best_idx, 'threshold']

        if self.verbose:
            print(f"Threshold tuning results (scoring={scoring}):")
            print(results_df.to_string(index=False))
            print(f"Best threshold: {best_thresh}")

        return best_thresh, results_df

    def set_threshold(self, threshold: float) -> 'StabilitySelector':
        """
        Update threshold and recompute selected features.

        Useful after tune_threshold() to apply the optimal threshold.

        Parameters
        ----------
        threshold : float
            New threshold value.

        Returns
        -------
        self
        """
        if not hasattr(self, 'selection_frequencies_'):
            raise ValueError("Must call fit() before set_threshold()")

        self.threshold = threshold
        mask = self.selection_frequencies_ >= threshold

        if self.max_features is not None and mask.sum() > self.max_features:
            top_idx = np.argsort(-self.selection_frequencies_, kind="mergesort")[:self.max_features]
            mask = np.zeros(self.n_features_in_, dtype=bool)
            mask[top_idx] = True

        selected = np.where(mask)[0]
        order = np.argsort(-self.selection_frequencies_[selected], kind="mergesort")
        self.selected_features_ = selected[order]
        self.selected_feature_names_ = [self.feature_names_in_[i] for i in self.selected_features_]
        self.n_features_selected_ = len(self.selected_features_)

        if self.verbose:
            print(f"Updated threshold to {threshold}: {self.n_features_selected_} features selected")

        return self

    def plot_frequencies(
        self,
        top_n: int = 50,
        figsize: Optional[Tuple[float, float]] = None,
        show_coef: bool = False
    ):
        """
        Bar plot of selection frequencies.

        Parameters
        ----------
        top_n : int
            Number of top features to show.
        figsize : tuple, optional
            Figure size.
        show_coef : bool
            If True, show mean coefficient as bar color intensity.
        """
        import matplotlib.pyplot as plt

        info = self.get_feature_info().head(top_n)

        if figsize is None:
            figsize = (10, max(6, top_n * 0.25))

        fig, ax = plt.subplots(figsize=figsize)

        if show_coef:
            coef_norm = info['mean_abs_coef'] / (info['mean_abs_coef'].max() + 1e-10)
            colors = plt.cm.Blues(0.3 + 0.7 * coef_norm)
        else:
            colors = ['steelblue' if s else 'lightgray' for s in info['selected']]

        ax.barh(range(len(info)), info['frequency'], color=colors)
        ax.set_yticks(range(len(info)))
        ax.set_yticklabels(info['feature'])
        ax.axvline(
            self.threshold,
            color='red',
            linestyle='--',
            label=f'threshold={self.threshold}'
        )
        ax.set_xlabel('Selection Frequency')
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.legend(loc='lower right')
        ax.set_title(f'Stability Selection ({self.n_features_selected_} features selected)')
        plt.tight_layout()

        return fig, ax

    def plot_coef_distributions(self, features: Optional[List[str]] = None, top_n: int = 12):
        """
        Plot coefficient distributions across bootstrap runs.

        Parameters
        ----------
        features : list of str, optional
            Specific features to plot. If None, uses top_n by frequency.
        top_n : int
            Number of top features if features not specified.

        Requires store_coefs=True (default).
        """
        import matplotlib.pyplot as plt

        if not hasattr(self, 'coef_bootstrap_'):
            raise ValueError(
                "Coefficient matrix not available. "
                "Set store_coefs=True when creating the selector."
            )

        if features is None:
            info = self.get_feature_info()
            features = info['feature'].head(top_n).tolist()

        n_features = len(features)
        ncols = min(4, n_features)
        nrows = int(np.ceil(n_features / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.5 * nrows))
        axes = np.atleast_2d(axes).flatten()

        for i, feat in enumerate(features):
            ax = axes[i]
            idx = self.feature_names_in_.index(feat)
            coefs = self.coef_bootstrap_[:, idx]

            ax.hist(coefs, bins=20, edgecolor='white', alpha=0.7)
            ax.axvline(0, color='red', linestyle='--', alpha=0.5)
            ax.set_title(f'{feat}\nfreq={self.selection_frequencies_[idx]:.2f}', fontsize=9)
            ax.set_xlabel('Coefficient')

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        return fig, axes


# =============================================================================
# Convenience Functions
# =============================================================================

def stability_select(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    threshold: float = 0.6,
    n_bootstrap: int = 50,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quick stability selection.

    Returns
    -------
    selected_indices : ndarray
    frequencies : ndarray
    """
    selector = StabilitySelector(
        threshold=threshold,
        n_bootstrap=n_bootstrap,
        **kwargs
    )
    selector.fit(X, y)
    return selector.selected_features_, selector.selection_frequencies_


def stability_regression(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    K: int,
    threshold: float = 0.6,
    n_bootstrap: int = 50,
    alpha: Optional[float] = None,
    l1_ratio: float = 1.0,
    sample_frac: float = 0.5,
    use_smart_sampler: bool = False,
    sampler_config: Optional[SmartSamplerConfig] = None,
    random_state: Optional[int] = None,
    n_jobs: int = -1,
    verbose: bool = True,
) -> List[str]:
    """
    Stability selection for regression.

    Fits Lasso/ElasticNet on bootstrap subsamples and returns features
    selected consistently across runs.

    Parameters
    ----------
    X : array-like or DataFrame of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Continuous target variable.
    K : int
        Maximum number of features to select.
    threshold : float, default=0.6
        Minimum selection frequency to keep a feature.
    n_bootstrap : int, default=50
        Number of bootstrap iterations.
    alpha : float, optional
        Regularization strength. If None, estimated via CV.
    l1_ratio : float, default=1.0
        ElasticNet mixing (1.0 = Lasso, <1.0 = ElasticNet).
    sample_frac : float, default=0.5
        Fraction of data per bootstrap sample.
    use_smart_sampler : bool, default=False
        Whether to apply leverage-based smart sampling.
    sampler_config : SmartSamplerConfig, optional
        Configuration for smart sampler.
    random_state : int, optional
        Random seed for reproducibility.
    n_jobs : int, default=-1
        Number of parallel jobs.
    verbose : bool, default=True
        Print progress information.

    Returns
    -------
    selected_features : list of str
        Names of selected features.
    """
    selector = StabilitySelector(
        task='regression',
        threshold=threshold,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        l1_ratio=l1_ratio,
        sample_frac=sample_frac,
        max_features=K,
        use_smart_sampler=use_smart_sampler,
        sampler_config=sampler_config,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    selector.fit(X, y)
    return selector.selected_feature_names_


def stability_classif(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    K: int,
    threshold: float = 0.6,
    n_bootstrap: int = 50,
    alpha: Optional[float] = None,
    sample_frac: float = 0.5,
    use_smart_sampler: bool = False,
    sampler_config: Optional[SmartSamplerConfig] = None,
    random_state: Optional[int] = None,
    n_jobs: int = -1,
    verbose: bool = True,
) -> List[str]:
    """
    Stability selection for classification.

    Fits L1-regularized LogisticRegression on bootstrap subsamples and
    returns features selected consistently across runs.

    Parameters
    ----------
    X : array-like or DataFrame of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Categorical target variable.
    K : int
        Maximum number of features to select.
    threshold : float, default=0.6
        Minimum selection frequency to keep a feature.
    n_bootstrap : int, default=50
        Number of bootstrap iterations.
    alpha : float, optional
        Regularization strength. If None, estimated via CV.
    sample_frac : float, default=0.5
        Fraction of data per bootstrap sample.
    use_smart_sampler : bool, default=False
        Whether to apply leverage-based smart sampling.
    sampler_config : SmartSamplerConfig, optional
        Configuration for smart sampler.
    random_state : int, optional
        Random seed for reproducibility.
    n_jobs : int, default=-1
        Number of parallel jobs.
    verbose : bool, default=True
        Print progress information.

    Returns
    -------
    selected_features : list of str
        Names of selected features.
    """
    selector = StabilitySelector(
        task='classification',
        threshold=threshold,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        sample_frac=sample_frac,
        max_features=K,
        use_smart_sampler=use_smart_sampler,
        sampler_config=sampler_config,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    selector.fit(X, y)
    return selector.selected_feature_names_


# =============================================================================
# Domain-Specific Presets
# =============================================================================


def panel_config(
    group_col: str,
    time_col: Optional[str] = None,
    sample_frac: float = 0.15
) -> SmartSamplerConfig:
    """Generic preset for panel/longitudinal data."""
    return SmartSamplerConfig(
        sample_frac=sample_frac,
        group_col=group_col,
        time_col=time_col,
        anchor_fn=first_and_last_per_group,
        min_per_group=2,
    )


def cross_section_config(sample_frac: float = 0.15) -> SmartSamplerConfig:
    """Preset for cross-sectional (non-grouped) data."""
    return SmartSamplerConfig(
        sample_frac=sample_frac,
        group_col=None,
        time_col=None,
        anchor_fn=None,
        min_per_group=1,
    )


def financial_config(
    ticker_col: str = 'ticker',
    date_col: str = 'date',
    sample_frac: float = 0.15
) -> SmartSamplerConfig:
    """Preset for financial panel data (stock returns, etc.)."""
    return SmartSamplerConfig(
        sample_frac=sample_frac,
        group_col=ticker_col,
        time_col=date_col,
        anchor_fn=combine_anchors(
            first_and_last_per_group,
            periodic_anchors('year'),
        ),
        min_per_group=5,
    )


def medical_config(
    patient_col: str = 'patient_id',
    time_col: Optional[str] = 'visit_date',
    sample_frac: float = 0.20
) -> SmartSamplerConfig:
    """Preset for medical/clinical longitudinal data."""
    return SmartSamplerConfig(
        sample_frac=sample_frac,
        group_col=patient_col,
        time_col=time_col,
        anchor_fn=first_and_last_per_group,
        min_per_group=2,
        # More conservative for medical data
        residual_weight_cap=0.3,
        weight_clip_quantile=0.95,
    )
