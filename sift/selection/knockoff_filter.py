"""FDR-calibrated Gaussian-copula knockoff selection."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Hashable, Sequence
from typing import Any, Callable, Optional
import warnings

import numpy as np
import pandas as pd

from sift._preprocess import to_numpy
from sift.estimators.copula import (
    FeatureCache,
    build_cache,
    gaussian_mi_from_corr,
    weighted_corr_with_vector,
    weighted_correlation_matrix,
    weighted_rank_gauss_1d,
)
from sift.estimators.knockoffs import (
    GaussianKnockoffModel,
    fit_gaussian_knockoffs,
    gaussian_knockoff_mean,
    sample_gaussian_knockoffs,
)


_STATISTIC_NOT_ENABLED = (
    "is reserved for a future tie-safe knockoff statistic and is not yet enabled"
)
_CEFSPLUS_DEFAULT_PATH_DEPTH = 10
_INTEGER_TARGET_WARNING_EMITTED = False


class _SubsampleDefaultType:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "<subsample default: 50,000 rows when X is given>"

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return (_SubsampleDefaultType, ())


_SUBSAMPLE_DEFAULT = _SubsampleDefaultType()


@dataclass(frozen=True)
class KnockoffSelectionResult:
    """Result object for q-calibrated knockoff selection."""

    selected_features: list[Any]
    selected_indices: Optional[list[int]]
    selector_metadata: dict[str, Any]
    W: pd.DataFrame
    threshold: Optional[float]
    selection_frequency: Optional[pd.Series]
    diagnostics_: Optional[dict[str, Any]] = None

    def get_feature_ranking(self) -> pd.DataFrame:
        ranking = self.W.copy()
        ranking["_feature_order"] = np.arange(len(ranking), dtype=np.int64)
        ranking = ranking.sort_values(
            ["W", "_feature_order"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        ranking = ranking.drop(columns=["_feature_order"])
        ranking.insert(2, "rank", np.arange(1, len(ranking) + 1, dtype=np.int64))
        columns = ["feature"]
        if "feature_group" in ranking.columns:
            columns.append("feature_group")
        columns.extend(
            [
                "W",
                "rank",
                "selected",
                "selection_frequency",
                "selected_index",
                "relevance",
                "selector",
            ]
        )
        return ranking[columns]


@dataclass(frozen=True)
class KnockoffStatContext:
    """Shared precomputed inputs for knockoff feature statistics."""

    Z: np.ndarray
    Zt: np.ndarray
    zy: np.ndarray
    w: np.ndarray
    model: GaussianKnockoffModel
    r: np.ndarray
    rt: np.ndarray
    kept: np.ndarray
    G: np.ndarray
    r_aug: np.ndarray
    options: dict[str, Any]
    n_jobs: int
    rng: np.random.Generator
    statistic_name: str = ""


@dataclass(frozen=True)
class KnockoffStatSpec:
    name: str
    fn: Callable[[KnockoffStatContext], np.ndarray]
    enabled: bool = True
    needs_screening: bool = True
    allowed_options: frozenset[str] = frozenset()


def _validate_probability(value: float, name: str, *, upper_inclusive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite float in (0, 1)")
    value_float = float(value)
    upper_ok = value_float <= 1.0 if upper_inclusive else value_float < 1.0
    if not np.isfinite(value_float) or value_float <= 0.0 or not upper_ok:
        interval = "(0, 1]" if upper_inclusive else "(0, 1)"
        raise ValueError(f"{name} must be a finite float in {interval}")
    return value_float


def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    if isinstance(value, (float, np.floating)) and not float(value).is_integer():
        raise ValueError(f"{name} must be a positive integer")
    value_int = int(value)
    if value_int != value:
        raise ValueError(f"{name} must be a positive integer")
    if value_int < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value_int


def _validate_offset(offset: int) -> int:
    if isinstance(offset, (bool, np.bool_)):
        raise ValueError("offset must be 0 or 1")
    if isinstance(offset, (float, np.floating)) and not float(offset).is_integer():
        raise ValueError("offset must be 0 or 1")
    offset_int = int(offset)
    if offset_int != offset:
        raise ValueError("offset must be 0 or 1")
    if offset_int not in (0, 1):
        raise ValueError("offset must be 0 or 1")
    return offset_int


def _validate_screen_pairs(screen_pairs: int | None) -> int | None:
    if screen_pairs is None:
        return None
    return _validate_positive_int(screen_pairs, "screen_pairs")


def _validate_nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite non-negative float")
    value_float = float(value)
    if not np.isfinite(value_float) or value_float < 0.0:
        raise ValueError(f"{name} must be a finite non-negative float")
    return value_float


def _warn_if_integer_multiclass_target(y: Any) -> None:
    global _INTEGER_TARGET_WARNING_EMITTED
    if _INTEGER_TARGET_WARNING_EMITTED:
        return
    y_raw = np.asarray(y)
    if y_raw.size == 0 or not np.issubdtype(y_raw.dtype, np.integer):
        return
    n_unique = np.unique(y_raw.ravel()).shape[0]
    if 3 <= n_unique <= 20:
        _INTEGER_TARGET_WARNING_EMITTED = True
        warnings.warn(
            "select_fdr treats y as a continuous target; integer labels with "
            "3-20 unique values look multiclass. For multiclass discovery, run "
            "one-vs-rest targets and combine the selected features.",
            UserWarning,
            stacklevel=3,
        )


def _validate_cache_rxx(Rxx: np.ndarray, p: int) -> np.ndarray:
    R = np.asarray(Rxx, dtype=np.float64)
    if R.shape != (p, p):
        raise ValueError(f"cache.Rxx must have shape ({p}, {p})")
    if not np.isfinite(R).all():
        raise ValueError("cache.Rxx must contain only finite values")
    if not np.allclose(R, R.T, atol=1e-6, rtol=1e-6):
        raise ValueError("cache.Rxx must be symmetric")
    if not np.allclose(np.diag(R), 1.0, atol=1e-5, rtol=0.0):
        raise ValueError("cache.Rxx must have a unit diagonal")
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)
    return R


def _reject_duplicate_feature_names(cache: FeatureCache) -> None:
    if cache.feature_names is None or cache.feature_names_are_synthetic:
        return
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in cache.feature_names:
        if name in seen:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        sample = duplicates[:5]
        suffix = "..." if len(duplicates) > 5 else ""
        raise ValueError(f"Duplicate feature names are not supported: {sample}{suffix}")


def _feature_names_for_valid_cols(cache: FeatureCache) -> list[Any]:
    if cache.feature_names is None:
        return [f"x{int(i)}" for i in cache.valid_cols]
    return [cache.feature_names[int(i)] for i in cache.valid_cols]


def _stable_group_codes(groups: Sequence[Any]) -> tuple[list[Any], np.ndarray]:
    labels: list[Any] = []
    mapping: dict[Any, int] = {}
    codes = np.empty(len(groups), dtype=np.int64)
    for i, group in enumerate(groups):
        missing = pd.isna(group)
        is_missing = bool(np.any(missing)) if isinstance(missing, np.ndarray) else bool(missing)
        if is_missing:
            raise ValueError("feature_groups must not contain missing values")
        if not isinstance(group, Hashable):
            raise ValueError("feature_groups values must be hashable")
        if group not in mapping:
            mapping[group] = len(labels)
            labels.append(group)
        codes[i] = mapping[group]
    return labels, codes


def _resolve_feature_groups(cache: FeatureCache, feature_groups: Sequence[Any] | None) -> tuple[list[Any], np.ndarray] | None:
    if feature_groups is None:
        return None
    groups_list = list(feature_groups)
    p_valid = cache.Z.shape[1]
    n_original = len(cache.feature_names) if cache.feature_names is not None else None
    if len(groups_list) == p_valid:
        valid_groups = groups_list
    elif n_original is not None and len(groups_list) == n_original:
        valid_groups = [groups_list[int(i)] for i in cache.valid_cols]
    else:
        expected = f"{p_valid}" if n_original is None else f"{p_valid} or {n_original}"
        raise ValueError(
            f"feature_groups has length {len(groups_list)}; expected exactly {expected} "
            "(valid cache columns, or the original input columns)"
        )
    return _stable_group_codes(valid_groups)


def _weighted_variance(Z: np.ndarray, w: np.ndarray, *, batch_size: int = 50_000) -> np.ndarray:
    Z_arr = np.asarray(Z)
    was_1d = Z_arr.ndim == 1
    if was_1d:
        Z_arr = Z_arr[:, None]
    if Z_arr.ndim != 2:
        raise ValueError("Z must be 1D or 2D")
    w64 = np.asarray(w, dtype=np.float64).ravel()
    if Z_arr.shape[0] != w64.shape[0]:
        raise ValueError("w length must match Z rows")
    if not np.isfinite(w64).all() or np.any(w64 < 0.0):
        raise ValueError("cache.sample_weight must be finite and non-negative")
    w_sum = float(w64.sum())
    if w_sum <= 0.0:
        raise ValueError("cache.sample_weight must sum to > 0")
    sums = np.zeros(Z_arr.shape[1], dtype=np.float64)
    sq_sums = np.zeros(Z_arr.shape[1], dtype=np.float64)
    batch_size = max(1, int(batch_size))
    for start in range(0, Z_arr.shape[0], batch_size):
        stop = min(Z_arr.shape[0], start + batch_size)
        Zb = np.asarray(Z_arr[start:stop], dtype=np.float64)
        wb = w64[start:stop]
        sums += wb @ Zb
        sq_sums += wb @ (Zb * Zb)
    mean = sums / w_sum
    var = sq_sums / w_sum - mean * mean
    np.maximum(var, 0.0, out=var)
    return var[0] if was_1d else var


def _resolve_cache(
    X,
    *,
    cache: FeatureCache | None,
    sample_weight,
    subsample: Any,
    random_state: int,
    n_jobs: int,
) -> FeatureCache:
    if (X is None) == (cache is None):
        raise ValueError("Exactly one of X or cache must be provided")
    if cache is not None:
        if sample_weight is not None:
            raise ValueError("sample_weight cannot be passed with a prebuilt cache")
        if subsample is not _SUBSAMPLE_DEFAULT:
            raise ValueError("subsample cannot be passed with a prebuilt cache")
        return cache
    resolved_subsample = 50_000 if subsample is _SUBSAMPLE_DEFAULT else subsample
    return build_cache(
        X,
        sample_weight=sample_weight,
        subsample=resolved_subsample,
        random_state=random_state,
        compute_Rxx=True,
        n_jobs=n_jobs,
        rank_backend="processes" if n_jobs != 1 else "serial",
    )


def _build_active_rxx(cache: FeatureCache, active: np.ndarray, *, verbose: bool) -> np.ndarray:
    p = cache.Z.shape[1]
    if cache.Rxx is not None:
        R_full = np.asarray(cache.Rxx, dtype=np.float64)
        if R_full.shape != (p, p):
            raise ValueError(f"cache.Rxx must have shape ({p}, {p})")
        active_count = int(active.sum())
        R_active = R_full[np.ix_(active, active)]
        return np.ascontiguousarray(_validate_cache_rxx(R_active, active_count), dtype=np.float64)

    if verbose:
        print("cache.Rxx is None; computing a local weighted correlation matrix.")
    Z_active = (
        np.asarray(cache.Z)
        if bool(active.all())
        else np.ascontiguousarray(cache.Z[:, active])
    )
    return weighted_correlation_matrix(
        Z_active,
        np.asarray(cache.sample_weight, dtype=np.float64),
        backend="blas",
    )


def _pair_screen(
    r: np.ndarray,
    rt: np.ndarray,
    screen_pairs: int | None,
) -> np.ndarray:
    p = r.shape[0]
    if screen_pairs is None or screen_pairs >= p:
        return np.arange(p, dtype=np.int64)
    m = min(p, int(screen_pairs))
    pair_score = np.maximum(np.abs(r), np.abs(rt))
    order = np.lexsort((np.arange(p, dtype=np.int64), -pair_score))
    return np.asarray(order[:m], dtype=np.int64)


def _build_augmented_correlation(
    model: GaussianKnockoffModel,
    kept: np.ndarray,
) -> np.ndarray:
    Sigma_m = np.asarray(model.Sigma_g[np.ix_(kept, kept)], dtype=np.float64)
    D_m = np.diag(np.asarray(model.s[kept], dtype=np.float64))
    cross = Sigma_m - D_m
    return np.block([[Sigma_m, cross], [cross, Sigma_m]])


def _build_context(
    Z: np.ndarray,
    Zt: np.ndarray,
    zy: np.ndarray,
    w: np.ndarray,
    model: GaussianKnockoffModel,
    *,
    screen_pairs: int | None,
    options: dict[str, Any],
    n_jobs: int,
    rng: np.random.Generator,
    build_augmented: bool = True,
    statistic_name: str = "",
    r: np.ndarray | None = None,
) -> KnockoffStatContext:
    r = np.asarray(weighted_corr_with_vector(Z, zy, w) if r is None else r, dtype=np.float64).ravel()
    if r.shape[0] != Z.shape[1]:
        raise ValueError("precomputed r length must match Z columns")
    rt = np.asarray(weighted_corr_with_vector(Zt, zy, w), dtype=np.float64).ravel()
    kept = _pair_screen(r, rt, screen_pairs)
    if build_augmented:
        G = _build_augmented_correlation(model, kept)
        r_aug = np.concatenate([r[kept], rt[kept]]).astype(np.float64, copy=False)
    else:
        G = np.empty((0, 0), dtype=np.float64)
        r_aug = np.empty(0, dtype=np.float64)
    return KnockoffStatContext(
        Z=Z,
        Zt=Zt,
        zy=zy,
        w=w,
        model=model,
        r=r,
        rt=rt,
        kept=kept,
        G=G,
        r_aug=r_aug,
        options=options,
        n_jobs=n_jobs,
        rng=rng,
        statistic_name=statistic_name,
    )


def _stat_relevance(context: KnockoffStatContext) -> np.ndarray:
    return (
        np.asarray(gaussian_mi_from_corr(context.r), dtype=np.float64)
        - np.asarray(gaussian_mi_from_corr(context.rt), dtype=np.float64)
    )


def _center_weighted(A: np.ndarray, w: np.ndarray) -> np.ndarray:
    w64 = np.asarray(w, dtype=np.float64)
    mean = (w64 @ A) / float(w64.sum())
    return A - mean


def _fit_lasso_coefficients(
    Z_aug: np.ndarray,
    zy: np.ndarray,
    w: np.ndarray,
    *,
    options: dict[str, Any],
    n_jobs: int,
    random_state: int,
    alphas: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.linear_model import LassoCV

    Z_centered = _center_weighted(np.asarray(Z_aug, dtype=np.float64), w)
    y_centered = _center_weighted(np.asarray(zy, dtype=np.float64)[:, None], w).ravel()
    sqrt_w = np.sqrt(np.asarray(w, dtype=np.float64))
    X_fit = Z_centered * sqrt_w[:, None]
    y_fit = y_centered * sqrt_w

    params: dict[str, Any] = {
        "cv": int(options.get("cv", 5)),
        "fit_intercept": False,
        "random_state": random_state,
        "n_jobs": n_jobs,
        "max_iter": int(options.get("max_iter", 5000)),
        "tol": float(options.get("tol", 1e-4)),
    }
    if "selection" in options:
        params["selection"] = options["selection"]
    if alphas is not None:
        params["alphas"] = alphas
    elif "alphas" in options:
        params["alphas"] = options["alphas"]
    else:
        params["eps"] = float(options.get("eps", 1e-3))
        params["n_alphas"] = int(options.get("n_alphas", 100))

    model = LassoCV(**params)
    model.fit(X_fit, y_fit)
    return np.asarray(model.coef_, dtype=np.float64), np.asarray(model.alphas_, dtype=np.float64)


def _stat_lcd(context: KnockoffStatContext) -> np.ndarray:
    kept = context.kept
    m = kept.shape[0]
    out = np.zeros(context.Z.shape[1], dtype=np.float64)
    if m == 0:
        return out

    Z_m = np.asarray(context.Z[:, kept], dtype=np.float64)
    Zt_m = np.asarray(context.Zt[:, kept], dtype=np.float64)
    seed = int(context.rng.integers(0, np.iinfo(np.int32).max))
    inner_n_jobs = context.n_jobs
    beta1, alphas = _fit_lasso_coefficients(
        np.column_stack([Z_m, Zt_m]),
        context.zy,
        context.w,
        options=context.options,
        n_jobs=inner_n_jobs,
        random_state=seed,
    )
    beta2, _ = _fit_lasso_coefficients(
        np.column_stack([Zt_m, Z_m]),
        context.zy,
        context.w,
        options=context.options,
        n_jobs=inner_n_jobs,
        random_state=seed,
        alphas=alphas,
    )
    W_kept = 0.5 * (np.abs(beta1[:m]) - np.abs(beta1[m:]))
    W_kept += 0.5 * (np.abs(beta2[m:]) - np.abs(beta2[:m]))
    out[kept] = W_kept
    return out


def _validate_path_depth(value: Any, m: int, *, default: int | None = None) -> int:
    if value is None:
        return m if default is None else min(m, default)
    depth = _validate_positive_int(value, "path_depth")
    return min(depth, 2 * m)


def _cefsplus_incremental_scores(
    G: np.ndarray,
    r: np.ndarray,
    *,
    path_depth: int,
    tie_break: np.ndarray,
    min_gain_ratio: float = 0.0,
    shrink: float = 1e-6,
    eps: float = 1e-10,
    tie_tol: float = 1e-12,
) -> np.ndarray:
    G_arr = np.asarray(G, dtype=np.float64)
    r_arr = np.asarray(r, dtype=np.float64).ravel()
    if G_arr.ndim != 2 or G_arr.shape[0] != G_arr.shape[1]:
        raise ValueError("G must be square")
    if G_arr.shape[0] != r_arr.shape[0] or G_arr.shape[0] % 2:
        raise ValueError("G/r dimensions must describe original-knockoff pairs")
    if not np.isfinite(G_arr).all() or not np.isfinite(r_arr).all():
        raise ValueError("G and r must contain only finite values")

    n_aug = r_arr.shape[0]
    n_pairs = n_aug // 2
    if n_aug == 0 or path_depth <= 0 or np.all(np.abs(r_arr) <= tie_tol):
        return np.zeros(n_aug, dtype=np.float64)

    Gs = (1.0 - shrink) * G_arr.copy()
    np.fill_diagonal(Gs, 1.0)
    rs = (1.0 - shrink) * r_arr
    tie_break_arr = np.asarray(tie_break, dtype=np.float64).ravel()
    if tie_break_arr.shape[0] != n_aug:
        tie_break_arr = np.asarray(gaussian_mi_from_corr(rs), dtype=np.float64)

    h = np.zeros(n_aug, dtype=np.float64)
    remaining = np.ones(n_aug, dtype=bool)
    selected = np.empty(0, dtype=np.int64)
    inv_S = np.empty((0, 0), dtype=np.float64)
    inv_yS = np.array([[1.0]], dtype=np.float64)
    logdet_S = 0.0
    logdet_yS = 0.0
    count = 0
    min_gain_abs = 0.0

    while count < path_depth and bool(remaining.any()):
        rem = np.flatnonzero(remaining)
        s = selected.shape[0]
        if s == 0:
            s1 = np.ones(rem.shape[0], dtype=np.float64)
            lf = np.zeros(rem.shape[0], dtype=np.float64)
            B = np.empty((0, rem.shape[0]), dtype=np.float64)
        else:
            B = Gs[np.ix_(selected, rem)]
            tmp = inv_S @ B
            s1 = np.maximum(1.0 - np.einsum("ij,ij->j", B, tmp), eps)
            lf = logdet_S + np.log(s1)

        B2 = np.vstack([rs[rem], B])
        tmp2 = inv_yS @ B2
        s2 = np.maximum(1.0 - np.einsum("ij,ij->j", B2, tmp2), eps)
        lc = logdet_yS + np.log(s2)
        scores = lf - lc
        best_score = float(np.max(scores))
        if not np.isfinite(best_score):
            break
        gain_best = best_score - (logdet_S - logdet_yS)
        if count > 0 and min_gain_ratio > 0.0 and gain_best < min_gain_abs:
            break
        if count == 0:
            first_gain = max(gain_best, eps)
            min_gain_abs = min_gain_ratio * first_gain
        tied = rem[np.abs(scores - best_score) <= tie_tol]

        pair_ids = tied % n_pairs
        sides = tied >= n_pairs
        neutralized = False
        for pair_id in np.unique(pair_ids):
            pair_sides = sides[pair_ids == pair_id]
            if pair_sides.size > 1 and np.any(pair_sides) and np.any(~pair_sides):
                remaining[int(pair_id)] = False
                remaining[int(pair_id) + n_pairs] = False
                neutralized = True
        if neutralized:
            continue

        best_tie_break = float(np.max(tie_break_arr[tied]))
        tied = tied[np.abs(tie_break_arr[tied] - best_tie_break) <= tie_tol]
        pair_ids = tied % n_pairs
        sides = tied >= n_pairs
        neutralized = False
        for pair_id in np.unique(pair_ids):
            pair_sides = sides[pair_ids == pair_id]
            if pair_sides.size > 1 and np.any(pair_sides) and np.any(~pair_sides):
                remaining[int(pair_id)] = False
                remaining[int(pair_id) + n_pairs] = False
                neutralized = True
        if neutralized:
            continue

        pair_order = tied % n_pairs
        j = int(tied[np.argmin(pair_order)])
        rem_pos = int(np.where(rem == j)[0][0])
        s1_best = s1[rem_pos]

        if s == 0:
            inv_S = np.array([[1.0 / s1_best]], dtype=np.float64)
        else:
            b = B[:, rem_pos].reshape(-1, 1)
            v = inv_S @ b
            inv_S_new = np.empty((s + 1, s + 1), dtype=np.float64)
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                inv_S_new[:s, :s] = inv_S + (v @ v.T) / s1_best
                inv_S_new[:s, s] = (-v[:, 0]) / s1_best
                inv_S_new[s, :s] = (-v[:, 0]) / s1_best
                inv_S_new[s, s] = 1.0 / s1_best
            if not np.isfinite(inv_S_new).all():
                break
            inv_S = inv_S_new
        logdet_S += float(np.log(s1[rem_pos]))

        b2 = B2[:, rem_pos].reshape(-1, 1)
        v2 = inv_yS @ b2
        s2_best = s2[rem_pos]
        inv_yS_new = np.empty((s + 2, s + 2), dtype=np.float64)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            inv_yS_new[: s + 1, : s + 1] = inv_yS + (v2 @ v2.T) / s2_best
            inv_yS_new[: s + 1, s + 1] = (-v2[:, 0]) / s2_best
            inv_yS_new[s + 1, : s + 1] = (-v2[:, 0]) / s2_best
            inv_yS_new[s + 1, s + 1] = 1.0 / s2_best
        if not np.isfinite(inv_yS_new).all():
            break
        inv_yS = inv_yS_new
        logdet_yS += float(np.log(s2_best))

        gain = float(np.log(s1_best) - np.log(s2_best))
        h[j] = max(gain, 0.0)
        selected = np.append(selected, j)
        remaining[j] = False
        count += 1

    return h


def _stat_cefsplus(context: KnockoffStatContext) -> np.ndarray:
    kept = context.kept
    m = kept.shape[0]
    out = np.zeros(context.Z.shape[1], dtype=np.float64)
    if m == 0:
        return out
    path_depth = _validate_path_depth(
        context.options.get("path_depth"),
        m,
        default=_CEFSPLUS_DEFAULT_PATH_DEPTH,
    )
    r_aug = np.asarray(context.r_aug, dtype=np.float64)
    if np.all(np.abs(r_aug) <= 1e-12):
        return out
    tie_break = np.asarray(gaussian_mi_from_corr(r_aug), dtype=np.float64)
    min_gain_ratio = _validate_nonnegative_float(context.options.get("min_gain_ratio", 0.0), "min_gain_ratio")
    h = _cefsplus_incremental_scores(
        context.G,
        r_aug,
        path_depth=path_depth,
        tie_break=tie_break,
        min_gain_ratio=min_gain_ratio,
    )
    out[kept] = h[:m] - h[m:]
    return out


def _reserved_statistic(context: KnockoffStatContext) -> np.ndarray:
    name = context.statistic_name or "statistic"
    raise ValueError(f"{name} {_STATISTIC_NOT_ENABLED}")


_KNOCKOFF_STAT_REGISTRY: dict[str, KnockoffStatSpec] = {
    "relevance": KnockoffStatSpec(
        "relevance",
        _stat_relevance,
        enabled=True,
        needs_screening=False,
        allowed_options=frozenset(),
    ),
    "lcd": KnockoffStatSpec(
        "lcd",
        _stat_lcd,
        enabled=False,
        needs_screening=True,
        allowed_options=frozenset({"cv", "max_iter", "tol", "selection", "alphas", "eps", "n_alphas"}),
    ),
    "cefsplus": KnockoffStatSpec(
        "cefsplus",
        _stat_cefsplus,
        enabled=True,
        needs_screening=True,
        allowed_options=frozenset({"path_depth", "min_gain_ratio"}),
    ),
    "mrmr_diff": KnockoffStatSpec("mrmr_diff", _reserved_statistic, enabled=False),
    "mrmr_quot": KnockoffStatSpec("mrmr_quot", _reserved_statistic, enabled=False),
    "jmi": KnockoffStatSpec("jmi", _reserved_statistic, enabled=False),
    "jmim": KnockoffStatSpec("jmim", _reserved_statistic, enabled=False),
}
VALID_KNOCKOFF_STATISTICS = tuple(_KNOCKOFF_STAT_REGISTRY)


def _get_statistic(statistic: str) -> KnockoffStatSpec:
    key = str(statistic).lower()
    if key not in _KNOCKOFF_STAT_REGISTRY:
        valid = ", ".join(VALID_KNOCKOFF_STATISTICS)
        raise ValueError(f"Unknown knockoff statistic {statistic!r}; expected one of: {valid}")
    spec = _KNOCKOFF_STAT_REGISTRY[key]
    if not spec.enabled:
        raise ValueError(f"Knockoff statistic {key!r} {_STATISTIC_NOT_ENABLED}")
    return spec


def knockoff_threshold(W: np.ndarray, q: float, *, offset: int = 1) -> float:
    """Return the knockoff/knockoff+ threshold, or ``inf`` if none exists."""

    q_float = _validate_probability(q, "q")
    offset_int = _validate_offset(offset)
    W_arr = np.asarray(W, dtype=np.float64).ravel()
    if not np.isfinite(W_arr).all():
        raise ValueError("W must contain only finite values")
    ts = np.unique(np.abs(W_arr[W_arr != 0.0]))
    for t in ts:
        fdp = (offset_int + np.sum(W_arr <= -t)) / max(1, np.sum(W_arr >= t))
        if fdp <= q_float:
            return float(t)
    return float(np.inf)


def _group_knockoff_statistics(
    W: np.ndarray,
    group_codes: np.ndarray,
    n_groups: int,
    *,
    tol: float = 1e-12,
) -> np.ndarray:
    W_arr = np.asarray(W, dtype=np.float64).ravel()
    codes = np.asarray(group_codes, dtype=np.int64).ravel()
    if W_arr.shape[0] != codes.shape[0]:
        raise ValueError("group_codes length must match W")
    out = np.zeros(n_groups, dtype=np.float64)
    for group_idx in range(n_groups):
        values = W_arr[codes == group_idx]
        if values.size == 0:
            continue
        max_abs = float(np.max(np.abs(values)))
        if max_abs <= tol:
            continue
        tied = values[np.abs(np.abs(values) - max_abs) <= tol]
        has_pos = bool(np.any(tied > tol))
        has_neg = bool(np.any(tied < -tol))
        if has_pos and has_neg:
            continue
        out[group_idx] = max_abs if has_pos else -max_abs
    return out


def sample_knockoffs(
    cache: FeatureCache,
    *,
    s_method: str = "equi",
    min_eig: float = 1e-3,
    random_state: int = 0,
) -> np.ndarray:
    """Fit and sample one Gaussian-copula knockoff draw for a cache."""

    _reject_duplicate_feature_names(cache)
    w = np.asarray(cache.sample_weight, dtype=np.float64)
    if not np.isfinite(w).all() or np.any(w < 0.0) or float(w.sum()) <= 0.0:
        raise ValueError("cache.sample_weight must be finite, non-negative, and sum to > 0")
    variances = _weighted_variance(cache.Z, w)
    active = variances > 1e-12
    if not bool(active.any()):
        raise ValueError("No active non-constant features remain for knockoffs")
    R_active = _build_active_rxx(cache, active, verbose=False)
    model = fit_gaussian_knockoffs(R_active, s_method=s_method, min_eig=min_eig)
    rng = np.random.default_rng(random_state)
    Z_active = (
        np.asarray(cache.Z, dtype=np.float32)
        if bool(active.all())
        else np.ascontiguousarray(cache.Z[:, active], dtype=np.float32)
    )
    Zt_active = sample_gaussian_knockoffs(Z_active, model, rng)
    Zt = np.zeros_like(cache.Z, dtype=np.float32)
    Zt[:, active] = Zt_active
    return Zt


def _all_zero_result(
    *,
    cache: FeatureCache,
    feature_names: list[Any],
    group_labels: list[Any] | None = None,
    group_codes: np.ndarray | None = None,
    relevance: np.ndarray,
    metadata: dict[str, Any],
    diagnostic_reason: str,
) -> KnockoffSelectionResult:
    n_draws = int(metadata.get("n_draws", 1))
    if n_draws == 1:
        threshold: float | None = float(np.inf)
        selection_frequency_arr = np.full(len(feature_names), np.nan)
        selection_frequency = None
    else:
        threshold = None
        selection_frequency_arr = np.zeros(len(feature_names), dtype=np.float64)
        selection_frequency = pd.Series(
            selection_frequency_arr,
            index=feature_names,
            name="selection_frequency",
        )
    W_table = pd.DataFrame(
        {
            "feature": feature_names,
            "selected_index": cache.valid_cols.astype(np.int64),
            "W": np.zeros(len(feature_names), dtype=np.float64),
            "selected": np.zeros(len(feature_names), dtype=bool),
            "selection_frequency": selection_frequency_arr,
            "relevance": relevance,
            "selector": "knockoff_fdr",
        }
    )
    if group_labels is not None and group_codes is not None:
        W_table["feature_group"] = [group_labels[int(code)] for code in group_codes]
    for draw_idx in range(n_draws):
        W_table[f"W_draw_{draw_idx}"] = np.zeros(len(feature_names), dtype=np.float64)
    diagnostics = {
        "thresholds": [float(np.inf)] * n_draws,
        "selection_sets": [[] for _ in range(n_draws)],
        "reason": diagnostic_reason,
    }
    if group_labels is not None and group_codes is not None:
        diagnostics["feature_groups"] = group_labels
        diagnostics["group_W_draws"] = [
            [0.0] * len(group_labels)
            for _ in range(n_draws)
        ]
        diagnostics["group_thresholds"] = [float(np.inf)] * n_draws
    return KnockoffSelectionResult(
        selected_features=[],
        selected_indices=[],
        selector_metadata=metadata,
        W=W_table,
        threshold=threshold,
        selection_frequency=selection_frequency,
        diagnostics_=diagnostics,
    )


def select_fdr(
    X=None,
    y=None,
    *,
    q: float = 0.1,
    statistic: str = "relevance",
    n_draws: int = 1,
    eta: float = 0.5,
    offset: int = 1,
    s_method: str = "equi",
    min_eig: float = 1e-3,
    screen_pairs: int | None = 2000,
    statistic_options: dict | None = None,
    feature_groups: Sequence[Any] | None = None,
    sample_weight=None,
    subsample: Any = _SUBSAMPLE_DEFAULT,
    cache: FeatureCache | None = None,
    random_state: int = 0,
    n_jobs: int = 1,
    verbose: bool = True,
) -> KnockoffSelectionResult:
    """Select features by a q-calibrated Gaussian-copula knockoff filter."""

    q_float = _validate_probability(q, "q")
    n_draws_int = _validate_positive_int(n_draws, "n_draws")
    eta_float = _validate_probability(eta, "eta", upper_inclusive=True)
    offset_int = _validate_offset(offset)
    screen_pairs_int = _validate_screen_pairs(screen_pairs)
    stat_spec = _get_statistic(statistic)
    options = dict(statistic_options or {})
    unknown_options = set(options) - stat_spec.allowed_options
    if unknown_options:
        allowed = sorted(stat_spec.allowed_options) or ["<none>"]
        raise ValueError(
            f"Unknown statistic_options for {stat_spec.name!r}: {sorted(unknown_options)}; "
            f"allowed: {allowed}"
        )

    resolved_cache = _resolve_cache(
        X,
        cache=cache,
        sample_weight=sample_weight,
        subsample=subsample,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    _reject_duplicate_feature_names(resolved_cache)
    p_valid = resolved_cache.Z.shape[1]
    if resolved_cache.valid_cols.shape[0] != p_valid:
        raise ValueError("cache.valid_cols length must match cache.Z columns")
    feature_names = _feature_names_for_valid_cols(resolved_cache)
    group_info = _resolve_feature_groups(resolved_cache, feature_groups)
    if group_info is None:
        group_labels: list[Any] | None = None
        group_codes = None
    else:
        group_labels, group_codes = group_info

    if y is None:
        raise ValueError("y is required")
    y_arr = to_numpy(y, dtype=np.float32).ravel()
    if y_arr.shape[0] != resolved_cache.n_rows_original:
        raise ValueError(
            f"y has {y_arr.shape[0]} rows but cache was built from "
            f"{resolved_cache.n_rows_original} rows"
        )
    if not np.isfinite(y_arr).all():
        raise ValueError("y must contain only finite values")
    _warn_if_integer_multiclass_target(y)

    w = np.asarray(resolved_cache.sample_weight, dtype=np.float64).ravel()
    if w.shape[0] != resolved_cache.Z.shape[0]:
        raise ValueError("cache.sample_weight length must match cache.Z rows")
    if not np.isfinite(w).all() or np.any(w < 0.0) or float(w.sum()) <= 0.0:
        raise ValueError("cache.sample_weight must be finite, non-negative, and sum to > 0")

    variances = _weighted_variance(resolved_cache.Z, w)
    active = variances > 1e-12
    n_zero_variance = int((~active).sum())
    if not bool(active.any()):
        raise ValueError("No active non-constant features remain for knockoffs")

    R_active = _build_active_rxx(resolved_cache, active, verbose=verbose)
    model = fit_gaussian_knockoffs(R_active, s_method=s_method, min_eig=min_eig)
    active_positions = np.flatnonzero(active).astype(np.int64)
    path_depth_requested = options.get("path_depth")
    if stat_spec.name == "cefsplus":
        m_pairs = active_positions.shape[0] if screen_pairs_int is None else min(active_positions.shape[0], screen_pairs_int)
        path_depth_effective = _validate_path_depth(
            path_depth_requested,
            int(m_pairs),
            default=_CEFSPLUS_DEFAULT_PATH_DEPTH,
        )
        options["path_depth"] = path_depth_effective
    else:
        path_depth_effective = None
    Z_active = (
        np.asarray(resolved_cache.Z, dtype=np.float32)
        if bool(active.all())
        else np.ascontiguousarray(resolved_cache.Z[:, active], dtype=np.float32)
    )

    ys = y_arr[resolved_cache.row_idx]
    zy = np.asarray(weighted_rank_gauss_1d(ys, w), dtype=np.float64)
    zy_var = float(_weighted_variance(zy[:, None], w)[0])

    relevance = np.zeros(p_valid, dtype=np.float64)
    r_orig_active = None
    if zy_var > 1e-12:
        r_orig = np.asarray(weighted_corr_with_vector(Z_active, zy, w), dtype=np.float64)
        r_orig_active = r_orig
        relevance[active_positions] = np.asarray(gaussian_mi_from_corr(r_orig), dtype=np.float64)

    metadata: dict[str, Any] = {
        "selector": "knockoff_fdr",
        "n_features": int(p_valid),
        "q": q_float,
        "offset": offset_int,
        "statistic": stat_spec.name,
        "s_method": s_method,
        "n_draws": n_draws_int,
        "eta": eta_float,
        "screen_pairs": screen_pairs_int,
        "path_depth_requested": path_depth_requested,
        "path_depth": path_depth_effective,
        "gamma": float(model.gamma),
        "lambda_min": float(model.lambda_min),
        "s_mean": float(np.mean(model.s)),
        "random_state": int(random_state),
        "n_rows_used": int(resolved_cache.Z.shape[0]),
        "fdr_control": "approximate_plugin",
        "validity_model": "gaussian_copula_plugin",
        "weighted_model": bool(np.ptp(w) > 1e-9),
        "n_zero_weight_variance_features": n_zero_variance,
        "feature_groups": group_labels is not None,
        "n_feature_groups": None if group_labels is None else len(group_labels),
    }

    if zy_var <= 1e-12:
        return _all_zero_result(
            cache=resolved_cache,
            feature_names=feature_names,
            group_labels=group_labels,
            group_codes=group_codes,
            relevance=relevance,
            metadata=metadata,
            diagnostic_reason="zero_target_variance",
        )

    seed_sequence = np.random.SeedSequence(random_state)
    child_sequences = seed_sequence.spawn(n_draws_int)
    W_draws = np.zeros((n_draws_int, p_valid), dtype=np.float64)
    thresholds: list[float] = []
    group_W_draws: list[list[float]] = []
    group_thresholds: list[float] = []
    selection_sets_valid: list[list[int]] = []
    mean_active = gaussian_knockoff_mean(Z_active, model) if n_draws_int > 1 else None
    active_group_codes = None if group_codes is None else group_codes[active_positions]

    for draw_idx, child in enumerate(child_sequences):
        rng = np.random.default_rng(child)
        Zt_active = sample_gaussian_knockoffs(Z_active, model, rng, mean=mean_active)
        context = _build_context(
            Z_active,
            Zt_active,
            zy,
            w,
            model,
            screen_pairs=screen_pairs_int if stat_spec.needs_screening else None,
            options=options,
            n_jobs=n_jobs,
            rng=rng,
            build_augmented=stat_spec.needs_screening,
            statistic_name=stat_spec.name,
            r=r_orig_active,
        )
        W_active = np.asarray(stat_spec.fn(context), dtype=np.float64).ravel()
        if W_active.shape[0] != active_positions.shape[0]:
            raise RuntimeError("Knockoff statistic returned the wrong number of W values")
        if not np.isfinite(W_active).all():
            raise RuntimeError("Knockoff statistic returned non-finite W values")
        W_draws[draw_idx, active_positions] = W_active
        if active_group_codes is None or group_labels is None:
            threshold = knockoff_threshold(W_active, q_float, offset=offset_int)
            if np.isfinite(threshold):
                selected_active = np.where(W_active >= threshold)[0]
            else:
                selected_active = np.empty(0, dtype=np.int64)
        else:
            group_W = _group_knockoff_statistics(W_active, active_group_codes, len(group_labels))
            threshold = knockoff_threshold(group_W, q_float, offset=offset_int)
            group_W_draws.append(group_W.astype(float).tolist())
            group_thresholds.append(threshold)
            if np.isfinite(threshold):
                selected_group_codes = np.flatnonzero(group_W >= threshold)
                selected_active = np.where(np.isin(active_group_codes, selected_group_codes))[0]
            else:
                selected_active = np.empty(0, dtype=np.int64)
        thresholds.append(threshold)
        selected_valid = active_positions[selected_active]
        selection_sets_valid.append(selected_valid.astype(int).tolist())

    mean_W = W_draws.mean(axis=0)
    if n_draws_int == 1:
        selection_frequency_arr = np.full(p_valid, np.nan, dtype=np.float64)
        threshold_out: float | None = thresholds[0]
        selected_mask = np.zeros(p_valid, dtype=bool)
        if selection_sets_valid:
            selected_mask[np.asarray(selection_sets_valid[0], dtype=np.int64)] = True
        selection_frequency = None
    else:
        selected_by_draw = np.zeros((n_draws_int, p_valid), dtype=np.float64)
        for draw_idx, selected_valid in enumerate(selection_sets_valid):
            selected_by_draw[draw_idx, np.asarray(selected_valid, dtype=np.int64)] = 1.0
        selection_frequency_arr = selected_by_draw.mean(axis=0)
        threshold_out = None
        selected_mask = selection_frequency_arr >= eta_float
        selection_frequency = pd.Series(selection_frequency_arr, index=feature_names, name="selection_frequency")

    selected_valid_positions = np.where(selected_mask)[0]
    selected_order = selected_valid_positions[
        np.lexsort((selected_valid_positions, -mean_W[selected_valid_positions]))
    ]
    selected_features = [feature_names[int(i)] for i in selected_order]
    selected_indices = [int(resolved_cache.valid_cols[int(i)]) for i in selected_order]

    W_table = pd.DataFrame(
        {
            "feature": feature_names,
            "selected_index": resolved_cache.valid_cols.astype(np.int64),
            "W": mean_W,
            "selected": selected_mask,
            "selection_frequency": selection_frequency_arr,
            "relevance": relevance,
            "selector": "knockoff_fdr",
        }
    )
    if group_codes is not None and group_labels is not None:
        W_table["feature_group"] = [group_labels[int(code)] for code in group_codes]
    for draw_idx in range(n_draws_int):
        W_table[f"W_draw_{draw_idx}"] = W_draws[draw_idx]

    diagnostics = {
        "thresholds": thresholds,
        "selection_sets": [
            [int(resolved_cache.valid_cols[int(i)]) for i in selected_valid]
            for selected_valid in selection_sets_valid
        ],
        "active_valid_positions": active_positions.astype(int).tolist(),
    }
    if group_codes is not None and group_labels is not None:
        diagnostics["feature_groups"] = group_labels
        diagnostics["group_W_draws"] = group_W_draws
        diagnostics["group_thresholds"] = group_thresholds

    if verbose:
        threshold_text = "derandomized" if threshold_out is None else f"threshold={threshold_out:.6g}"
        threshold_name = "knockoff+" if offset_int == 1 else "knockoff"
        print(
            f"{threshold_name} q={q_float:.3g}: selected {len(selected_features)} features "
            f"({threshold_text}, s_mean={metadata['s_mean']:.3g})"
        )

    return KnockoffSelectionResult(
        selected_features=selected_features,
        selected_indices=selected_indices,
        selector_metadata=metadata,
        W=W_table,
        threshold=threshold_out,
        selection_frequency=selection_frequency,
        diagnostics_=diagnostics,
    )


__all__ = [
    "KnockoffSelectionResult",
    "KnockoffStatContext",
    "KnockoffStatSpec",
    "VALID_KNOCKOFF_STATISTICS",
    "knockoff_threshold",
    "sample_knockoffs",
    "select_fdr",
]
