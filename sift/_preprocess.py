"""Input preprocessing: validation, conversion, subsampling, encoding."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timedelta
from typing import Any, List, Literal, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

EstimatorJMI = Literal["auto", "binned", "r2", "ksg", "gaussian"]
EstimatorMRMR = Literal["classic", "gaussian"]
RelevanceMethod = Literal["f", "ks", "rf"]
CatEncoding = Literal[
    "none",
    "target_cv",
    "target",
    "loo",
    "james_stein",
    "loo_logit",
]
Formula = Literal["quotient", "difference"]
Task = Literal["regression", "classification"]

HIGHER_IS_BETTER = frozenset(
    {
        "AUC",
        "ACCURACY",
        "R2",
        "F1",
        "F1_WEIGHTED",
        "F1_MACRO",
        "ROC_AUC",
        "PRECISION",
        "RECALL",
        "NDCG",
        "MAP",
        "PRAUC",
        "MCC",
        "BALANCEDACCURACY",
    }
)
LOWER_IS_BETTER = frozenset(
    {
        "RMSE",
        "MAE",
        "LOGLOSS",
        "MSE",
        "MAPE",
        "SMAPE",
        "MULTICLASS",
        "MULTICLASSONEVSALL",
        "CROSSENTROPY",
    }
)


def infer_higher_is_better(metric: str) -> bool:
    """Infer whether higher metric values are better."""
    metric_upper = metric.upper().split(":")[0]
    if metric_upper in HIGHER_IS_BETTER:
        return True
    if metric_upper in LOWER_IS_BETTER:
        return False
    raise ValueError(
        f"Cannot infer score direction for metric {metric!r}. "
        "Pass higher_is_better=True or False explicitly."
    )


def best_score_from_dict(scores: dict, higher_is_better: bool) -> Tuple[Any, float]:
    """Return best (index, score), preferring the smallest numeric index on ties."""
    if not scores:
        return 0, float("nan")

    valid = {k: v for k, v in scores.items() if np.isfinite(v)}
    if not valid:
        return 0, float("nan")

    def tie_key(key, position: int) -> tuple[int, float | int]:
        if isinstance(key, (int, float, np.integer, np.floating)) and not isinstance(key, bool):
            key_value = float(key)
            if np.isfinite(key_value):
                return 0, key_value
        return 1, position

    valid_items = list(valid.items())

    def ranking_key(item):
        position, (score_key, score) = item
        order_group, order_value = tie_key(score_key, position)
        if higher_is_better:
            return score, -order_group, -order_value
        return score, order_group, order_value

    if higher_is_better:
        best = max(enumerate(valid_items), key=ranking_key)[1]
    else:
        best = min(enumerate(valid_items), key=ranking_key)[1]
    return best[0], best[1]


# --- Input conversion ---


def validate_task(task: str) -> None:
    """Validate the public selector task before dispatch."""
    if task not in {"regression", "classification"}:
        raise ValueError(
            "task must be 'regression' or 'classification', "
            f"got {task!r}"
        )


def validate_classification_target(y) -> None:
    """Reject continuous or multi-output targets passed as class labels."""
    from sklearn.utils.multiclass import type_of_target

    y_raw = y.values if hasattr(y, "values") else np.asarray(y)
    if pd.isna(y_raw).any():
        raise ValueError("Missing values in y are not allowed for classification.")
    if pd.api.types.is_numeric_dtype(y_raw):
        y_num = np.asarray(y_raw, dtype=np.float64)
        if not np.isfinite(y_num).all():
            raise ValueError("Non-finite values in y are not allowed for classification.")
    target_type = type_of_target(y_raw)
    if target_type not in {"binary", "multiclass"}:
        raise ValueError(
            "Classification y must contain discrete binary or multiclass labels; "
            f"got target type {target_type!r}."
        )


def to_numpy(data, dtype=np.float32) -> np.ndarray:
    """Convert Pandas/Polars/list to numpy array."""
    if hasattr(data, "to_pandas"):
        data = data.to_pandas()
    if isinstance(data, (pd.DataFrame, pd.Series)):
        try:
            return data.to_numpy(dtype=dtype, na_value=np.nan)
        except TypeError:
            arr = data.to_numpy()
            if arr.dtype == object:
                arr = np.where(pd.isna(arr), np.nan, arr)
            return arr.astype(dtype)
    if hasattr(data, "values"):
        return np.asarray(data.values, dtype=dtype)
    return np.asarray(data, dtype=dtype)


def extract_feature_names(X) -> Optional[List[str]]:
    """Extract column names from DataFrame, or None for ndarray."""
    if hasattr(X, "columns"):
        return list(X.columns)
    return None


def _is_temporal_feature_dtype(dtype) -> bool:
    """Recognize NumPy, pandas, and Arrow-backed datetime/duration dtypes."""
    arrow_dtype = getattr(dtype, "pyarrow_dtype", None)
    arrow_kind = (
        str(arrow_dtype).partition("[")[0]
        if arrow_dtype is not None
        else None
    )
    return (
        pd.api.types.is_datetime64_any_dtype(dtype)
        or pd.api.types.is_timedelta64_dtype(dtype)
        # pandas 2.2 does not classify Arrow duration dtypes as timedelta,
        # but they retain NumPy's temporal dtype kind.
        or getattr(dtype, "kind", None) in {"M", "m"}
        # Arrow time-of-day dtypes instead report object kind.
        or arrow_kind in {"date32", "date64", "duration", "time32", "time64", "timestamp"}
    )


def reject_datetime_like_features(X) -> None:
    """Reject NumPy, pandas, and Arrow temporal features before coercion."""
    temporal_object_types = (
        date,
        datetime,
        timedelta,
        np.datetime64,
        np.timedelta64,
        pd.Timestamp,
        pd.Timedelta,
    )

    def contains_temporal_objects(values) -> bool:
        try:
            flat = np.asarray(values, dtype=object).ravel()
        except (TypeError, ValueError):
            return False
        return any(
            value is pd.NaT or isinstance(value, temporal_object_types)
            for value in flat
        )

    if isinstance(X, pd.DataFrame):
        datetime_like = [
            name
            for i, (name, column_dtype) in enumerate(X.dtypes.items())
            if _is_temporal_feature_dtype(column_dtype)
            or (
                pd.api.types.is_object_dtype(column_dtype)
                and contains_temporal_objects(X.iloc[:, i])
            )
        ]
    else:
        try:
            array = np.asarray(X)
        except (TypeError, ValueError):
            array = None
        dtype = None if array is None else array.dtype
        datetime_like = (
            ["<array>"]
            if dtype is not None
            and (
                _is_temporal_feature_dtype(dtype)
                or (
                    pd.api.types.is_object_dtype(dtype)
                    and contains_temporal_objects(array)
                )
            )
            else []
        )
    if datetime_like:
        sample = datetime_like[:5]
        suffix = "..." if len(datetime_like) > 5 else ""
        raise ValueError(
            "Datetime or timedelta feature columns (including time-of-day values) "
            "are not supported: "
            f"{sample}{suffix}. Convert them to numeric features explicitly."
        )


# --- Validation ---


def ensure_weights(
    sample_weight: np.ndarray | None,
    n: int,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Validate and normalize sample weights.

    Parameters
    ----------
    sample_weight : array-like or None
        Sample weights. If None, returns uniform weights.
    n : int
        Expected number of samples.
    normalize : bool
        If True, normalize weights to mean=1.

    Returns
    -------
    w : ndarray of shape (n,)
        Validated, non-negative, finite weights.
    """
    if sample_weight is None:
        return np.ones(n, dtype=np.float64)

    w = np.asarray(sample_weight, dtype=np.float64).ravel()

    if w.shape[0] != n:
        raise ValueError(f"sample_weight has {w.shape[0]} elements but expected {n}")
    if not np.isfinite(w).all():
        raise ValueError("sample_weight contains non-finite values")
    if np.any(w < 0):
        raise ValueError("sample_weight contains negative values")
    if not np.any(w > 0):
        raise ValueError("sample_weight must contain at least one positive value")

    if normalize:
        scale = float(np.max(w))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("sample_weight max must be finite and > 0 to normalize")
        w = w / scale
        mean = float(w.mean())
        if not np.isfinite(mean) or mean <= 0.0:
            raise ValueError("sample_weight mean must be finite and > 0 to normalize")
        w = w / mean
        # Quantize normalized weights to float32 precision, then restore their
        # mean in float64. This greatly reduces rescaling-induced ulp changes
        # that can alter tree tie-breaking without claiming exact invariance
        # for every representable input and scale.
        w = w.astype(np.float32).astype(np.float64)
        w /= float(w.mean())

    return w


def validate_k(k, *, allow_auto: bool = True) -> int | Literal["auto"]:
    """Validate a public feature-count argument."""
    if k == "auto":
        if allow_auto:
            return "auto"
        raise ValueError("k='auto' is not supported here")

    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise ValueError("k must be a positive integer")

    k_int = int(k)
    if k_int < 1:
        raise ValueError("k must be >= 1")
    return k_int


def validate_inputs(
    X, y, task: str, impute: bool = True, *, dtype=np.float64
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Validate and convert inputs.

    ``dtype`` controls the returned feature dtype. The default is float64 so
    classic filters do not lose distinctions between large-offset values (for
    example, values near ``1e8``). Callers that have a deliberate lower
    precision contract can opt into ``np.float32`` explicitly.
    """
    from sift._impute import mean_impute

    validate_task(task)

    feature_names = extract_feature_names(X)
    reject_datetime_like_features(X)
    if hasattr(X, "select_dtypes"):
        non_numeric = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        if non_numeric:
            sample = non_numeric[:5]
            suffix = "..." if len(non_numeric) > 5 else ""
            raise ValueError(
                f"Non-numeric columns found: {sample}{suffix}. "
                "Either encode them first or set cat_encoding to 'target_cv', "
                "'loo', 'target', 'james_stein', or binary-only 'loo_logit'."
            )
    X_arr = to_numpy(X, dtype=np.float64)

    if impute:
        X_arr = mean_impute(X_arr, copy=True)

    X_arr = X_arr.astype(dtype, copy=False)

    if task == "classification":
        if hasattr(y, "values"):
            y_raw = y.values
        else:
            y_raw = np.asarray(y)

        validate_classification_target(y_raw)

        _, y_arr = np.unique(y_raw, return_inverse=True)
        y_arr = y_arr.astype(np.int32)
    else:
        y_arr = to_numpy(y, dtype=np.float64)
        if not np.isfinite(y_arr).all():
            raise ValueError("Non-finite values in y are not allowed for regression.")

    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(f"X has {X_arr.shape[0]} rows but y has {y_arr.shape[0]}")

    if feature_names is None:
        feature_names = [f"x{i}" for i in range(X_arr.shape[1])]

    return X_arr, y_arr.ravel(), feature_names


def check_regression_only(task: str, estimator: str) -> None:
    """Raise if using regression-only estimator for classification."""
    regression_only = {"gaussian", "r2", "ksg"}
    if task == "classification" and estimator in regression_only:
        raise ValueError(
            f"estimator='{estimator}' is regression-only. "
            "Use estimator='binned' for classification."
        )


def resolve_jmi_estimator(estimator: str, task: str) -> str:
    """Resolve 'auto' to concrete estimator."""
    if estimator == "auto":
        return "binned" if task == "classification" else "r2"
    return estimator


# --- Subsampling ---


def subsample_xy(
    X: np.ndarray,
    y: np.ndarray,
    subsample: Optional[int],
    random_state: int,
    *,
    sample_weight: Optional[np.ndarray] = None,
    return_idx: bool = False,
) -> tuple:
    """Subsample X, y, and optionally sample_weight."""
    n = X.shape[0]
    w = ensure_weights(sample_weight, n, normalize=True)

    positive = np.flatnonzero(w > 0.0)
    if subsample is not None and positive.size > subsample:
        rng = np.random.default_rng(random_state)
        row_idx = rng.choice(positive, size=subsample, replace=False)
    else:
        row_idx = positive
    X_sub, y_sub, w_sub = X[row_idx], y[row_idx], w[row_idx]

    mean = float(w_sub.mean())
    if not np.isfinite(mean) or mean <= 0.0:
        raise ValueError("Subsample weight mean must be finite and > 0")
    w_sub = w_sub / mean

    if return_idx:
        return X_sub, y_sub, w_sub, row_idx
    return X_sub, y_sub, w_sub


# --- Categorical encoding ---


@contextmanager
def suppress_category_encoder_pandas_warnings():
    """Hide narrow pandas 3.0 deprecation warnings emitted by category_encoders."""
    try:
        from pandas.errors import Pandas4Warning
    except (ImportError, AttributeError):
        Pandas4Warning = Warning

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*future\.no_silent_downcasting.*",
            category=Pandas4Warning,
        )
        yield


class LeaveOneOutLogitEncoder:
    """Smoothed leave-one-out logit encoder for binary targets."""

    def __init__(
        self,
        cols: List[str],
        *,
        smoothing: float = 20.0,
        clip_min: float = 1e-4,
        clip_max: float = 1.0 - 1e-4,
    ):
        try:
            smoothing_float = float(smoothing)
            clip_min_float = float(clip_min)
            clip_max_float = float(clip_max)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "loo_smoothing and LOO-logit clip bounds must be finite numeric values"
            ) from exc
        if smoothing_float <= 0.0 or not np.isfinite(smoothing_float):
            raise ValueError("loo_smoothing must be positive and finite")
        if (
            not np.isfinite(clip_min_float)
            or not np.isfinite(clip_max_float)
            or not 0.0 < clip_min_float < clip_max_float < 1.0
        ):
            raise ValueError("loo_clip_min and loo_clip_max must satisfy 0 < min < max < 1")
        self.cols = list(cols)
        self.smoothing = smoothing_float
        self.clip_min = clip_min_float
        self.clip_max = clip_max_float

    @staticmethod
    def _series_with_missing_sentinel(series: pd.Series) -> pd.Series:
        sentinel = "__SIFT_MISSING_CATEGORY__"
        values = set(series.dropna().astype(object).tolist())
        while sentinel in values:
            sentinel += "_"
        return series.astype(object).where(~series.isna(), sentinel)

    @staticmethod
    def _get_column_series(X: pd.DataFrame, col: str) -> pd.Series:
        series = X.loc[:, col]
        if isinstance(series, pd.DataFrame):
            raise ValueError(
                "loo_logit encoding requires unique DataFrame column names "
                "for encoded categorical columns"
            )
        return series

    def _validate_binary_y(self, y) -> np.ndarray:
        y_arr = np.asarray(y).ravel()
        if pd.isna(y_arr).any():
            raise ValueError("loo_logit encoding requires a binary target without missing values")
        try:
            y_num = y_arr.astype(np.float64)
        except (TypeError, ValueError):
            y_num = None
        if y_num is not None:
            if not np.isfinite(y_num).all():
                raise ValueError("loo_logit encoding requires finite binary target values")
            unique_num = np.unique(y_num)
            if len(unique_num) != 2:
                raise ValueError("loo_logit encoding requires exactly two target classes")
            if set(unique_num.tolist()) == {0.0, 1.0}:
                return y_num

        unique = pd.unique(y_arr)
        if len(unique) != 2:
            raise ValueError("loo_logit encoding requires exactly two target classes")
        mapping = {unique[0]: 0.0, unique[1]: 1.0}
        return np.array([mapping[value] for value in y_arr], dtype=np.float64)

    def _logit(self, p: np.ndarray | float) -> np.ndarray | float:
        p_clipped = np.clip(p, self.clip_min, self.clip_max)
        return np.log(p_clipped / (1.0 - p_clipped))

    def _validate_sample_weight(self, sample_weight, n: int) -> np.ndarray:
        return ensure_weights(sample_weight, n, normalize=True)

    def fit(self, X: pd.DataFrame, y, sample_weight=None):
        y_arr = self._validate_binary_y(y)
        w = self._validate_sample_weight(sample_weight, len(y_arr))
        self.global_prior_ = float(np.sum(w * y_arr) / np.sum(w))
        self.global_logit_ = float(self._logit(self.global_prior_))
        self.category_maps_: dict[str, dict[object, float]] = {}

        for col in self.cols:
            series = self._series_with_missing_sentinel(self._get_column_series(X, col))
            codes, uniques = pd.factorize(series, sort=False)
            counts = np.bincount(codes, weights=w)
            sums = np.bincount(codes, weights=w * y_arr)
            p = (sums + self.smoothing * self.global_prior_) / (counts + self.smoothing)
            enc = self._logit(p)
            self.category_maps_[col] = dict(zip(uniques.tolist(), enc.tolist()))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for col in self.cols:
            mapping = self.category_maps_[col]
            series = self._series_with_missing_sentinel(self._get_column_series(X_out, col))
            X_out[col] = series.map(mapping).fillna(self.global_logit_).astype(float)
        return X_out

    def fit_transform(self, X: pd.DataFrame, y, sample_weight=None) -> pd.DataFrame:
        y_arr = self._validate_binary_y(y)
        w = self._validate_sample_weight(sample_weight, len(y_arr))
        self.fit(X, y_arr, sample_weight=w)
        X_out = X.copy()

        for col in self.cols:
            series = self._series_with_missing_sentinel(self._get_column_series(X_out, col))
            codes, _ = pd.factorize(series, sort=False)
            counts = np.bincount(codes, weights=w)
            sums = np.bincount(codes, weights=w * y_arr)
            p = (
                sums[codes]
                - w * y_arr
                + self.smoothing * self.global_prior_
            ) / (counts[codes] - w + self.smoothing)
            X_out[col] = self._logit(p).astype(float)
        return X_out


def validate_target_cv_encoding_flags(
    cat_encoding, allow_full_data_target_encoding
) -> None:
    """Reject ``target_cv`` combined with the full-data escape hatch.

    ``allow_full_data_target_encoding=True`` opts into fitting a supervised
    encoder on every row, which directly contradicts the cross-fitted
    ``target_cv`` contract.  Silently ignoring the flag would leave callers
    believing they had opted out of cross-fitting.
    """
    if cat_encoding == "target_cv" and bool(allow_full_data_target_encoding):
        raise ValueError(
            "cat_encoding='target_cv' cannot be combined with "
            "allow_full_data_target_encoding=True: target_cv is cross-fitted by "
            "construction, so the full-data escape hatch contradicts it. Drop "
            "allow_full_data_target_encoding, or choose a legacy supervised "
            "encoding ('target', 'loo', 'james_stein', 'loo_logit') if you "
            "really want full-data fitting."
        )


def target_cv_n_splits(
    y,
    *,
    target_type: Literal["auto", "continuous", "binary"] = "auto",
    cv: int = 5,
) -> int:
    """Return the usable deterministic fold count for native target encoding."""
    from sklearn.utils.multiclass import type_of_target

    if isinstance(cv, (bool, np.bool_)) or not isinstance(cv, (int, np.integer)):
        raise ValueError("target_cv cv must be an integer >= 2")
    requested = int(cv)
    if requested < 2:
        raise ValueError("target_cv cv must be an integer >= 2")
    if target_type not in {"auto", "continuous", "binary"}:
        raise ValueError("target_cv target_type must be 'auto', 'continuous', or 'binary'")

    y_arr = np.asarray(y).reshape(-1)
    if y_arr.size < 2:
        raise ValueError("target_cv requires at least two rows")
    inferred_kind = type_of_target(y_arr)
    target_kind = inferred_kind if target_type == "auto" else target_type
    if inferred_kind == "multiclass" and target_type != "continuous":
        raise ValueError(
            "cat_encoding='target_cv' does not yet support multiclass targets: "
            "sklearn expands each categorical feature to one column per class, "
            "which requires block-aware selection"
        )
    if target_kind == "binary":
        _, counts = np.unique(y_arr, return_counts=True)
        if counts.size != 2:
            raise ValueError("target_cv target_type='binary' requires exactly two classes")
        effective = min(requested, int(counts.min()))
    else:
        effective = min(requested, int(y_arr.size))
    if effective < 2:
        raise ValueError(
            "target_cv requires at least two rows per class for binary targets"
        )
    return effective


class TargetCVEncoder(TransformerMixin, BaseEstimator):
    """Cross-fitted, prior-centered target encoder with DataFrame output.

    Every emitted value is a *centered category effect*: the fold-local (or
    full-fit) category estimate minus the training prior that produced it.
    Out-of-fold training rows emit ``fold_encoding - fold_training_prior`` and
    inference rows emit ``full_fit_encoding - full_training_prior``.  A category
    that the fitting rows never saw therefore emits a zero centered effect (the
    raw global-mean estimate before centering) instead of a fold-identifying
    prior, so unique-ID, group-proxy, and timestamp-proxy columns cannot become
    fold markers.

    **What centering does and does not guarantee.**  Centering neutralizes only
    *unseen-in-fold* emissions: a level absent from a fold's training rows emits
    exactly zero instead of a prior that identifies the complement folds.  It is
    not a defence against high cardinality as such.  A level that appears two or
    more times in a fold's training rows still transmits those sibling rows'
    targets, which is ordinary target-encoding behavior, so a near-unique
    identifier whose rows share a latent target remains a selectable feature.
    Drop ID-like columns, or pass ``groups=`` so all of an identifier's rows land
    in one fold, if that cross-row information must not reach selection.

    All fold kinds (``fixed_k``, ``group``, ``time``) run through this one
    engine.  sklearn's ``TargetEncoder`` does not expose the per-fold priors the
    centering contract needs, so it is not used as a backend; the fixed-k folds
    reproduce its ``KFold``/``StratifiedKFold(shuffle=True, random_state=...)``
    split construction and its ``smooth="auto"`` empirical-Bayes shrinkage,
    generalized to weighted rows.  ``smooth="auto"`` is available on every fold
    kind, weighted or not, because that generalization replaces integer counts
    with weighted row mass (see ``_auto_smoothed_encoding``).

    The encoder intentionally preserves one output column per raw categorical
    feature.  sklearn's multiclass target encoding expands every input feature
    to one column per class, which cannot be represented by SIFT's current
    feature-selection contract until block-aware selection lands.
    """

    def __init__(
        self,
        cols: List[str],
        *,
        target_type: Literal["auto", "continuous", "binary"] = "auto",
        smooth: Literal["auto"] | float = "auto",
        cv: int = 5,
        random_state: int = 0,
        target_prior: float | None = None,
        warmup_policy: Literal["exclude", "zero_weight"] = "zero_weight",
    ):
        self.cols = cols
        self.target_type = target_type
        self.smooth = smooth
        self.cv = cv
        self.random_state = random_state
        self.target_prior = target_prior
        self.warmup_policy = warmup_policy

    def _validate_frame(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("target_cv encoding requires X to be a pandas DataFrame")
        if not X.columns.is_unique:
            raise ValueError("target_cv encoding requires unique DataFrame column names")
        missing = [col for col in self.cols if col not in X.columns]
        if missing:
            raise ValueError(f"target_cv columns are missing from X: {missing[:5]}")

    def _requested_cv(self) -> int:
        if isinstance(self.cv, (bool, np.bool_)) or not isinstance(
            self.cv, (int, np.integer)
        ):
            raise ValueError("target_cv cv must be an integer >= 2")
        requested = int(self.cv)
        if requested < 2:
            raise ValueError("target_cv cv must be an integer >= 2")
        return requested

    def _validate_smoothing(self) -> Literal["auto"] | float:
        """Resolve ``target_cv_smoothing`` for every fold kind.

        ``"auto"`` is accepted on all of them.  The empirical-Bayes prior it
        needs is defined by *weighted row mass* rather than integer counts (see
        ``_auto_smoothed_encoding``), and every quantity that definition
        requires -- the weighted prior ``sum(w*y)/sum(w)``, the weighted target
        variance ``sum(w*(y-prior)^2)/sum(w)``, each category's weighted mass and
        its weighted sum of squared deviations -- exists for any fitting slice
        with positive total weight.  ``ensure_weights`` already rejects negative,
        non-finite, and all-zero weights, so there is no weighted, grouped, or
        time-aware case in which ``"auto"`` is undefined but an explicit float
        would not be.
        """
        if self.smooth == "auto":
            resolved: Literal["auto"] | float = "auto"
        else:
            if isinstance(self.smooth, (bool, np.bool_)) or not isinstance(
                self.smooth, (int, float, np.integer, np.floating)
            ):
                raise ValueError(
                    "target_cv_smoothing must be 'auto' or a non-negative float"
                )
            resolved = float(self.smooth)
            if not np.isfinite(resolved) or resolved < 0.0:
                raise ValueError("target_cv_smoothing must be finite and >= 0")
        if self.warmup_policy not in {"exclude", "zero_weight"}:
            raise ValueError("warmup_policy must be 'exclude' or 'zero_weight'")
        return resolved

    def _target_values(self, y) -> tuple[np.ndarray, str]:
        from sklearn.preprocessing import LabelEncoder
        from sklearn.utils.multiclass import type_of_target

        y_arr = np.asarray(y).reshape(-1)
        inferred = type_of_target(y_arr)
        target_kind = inferred if self.target_type == "auto" else self.target_type
        if inferred == "multiclass" and self.target_type != "continuous":
            raise ValueError(
                "cat_encoding='target_cv' does not yet support multiclass targets: "
                "sklearn expands each categorical feature to one column per class, "
                "which requires block-aware selection"
            )
        if target_kind == "binary":
            label_encoder = LabelEncoder().fit(y_arr)
            if label_encoder.classes_.size != 2:
                raise ValueError("target_cv target_type='binary' requires exactly two classes")
            self.classes_ = label_encoder.classes_
            values = label_encoder.transform(y_arr).astype(np.float64)
            if self.target_prior is not None and not 0.0 <= float(self.target_prior) <= 1.0:
                raise ValueError("target_prior must be between 0 and 1 for binary targets")
            return values, "binary"
        if target_kind != "continuous":
            raise ValueError(
                "target_cv supports only continuous regression and binary classification targets"
            )
        try:
            values = np.asarray(y_arr, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("continuous target_cv targets must be numeric") from exc
        if not np.isfinite(values).all():
            raise ValueError("target_cv y contains non-finite values")
        return values, "continuous"

    @staticmethod
    def _normalized_series(series: pd.Series) -> pd.Series:
        return series.astype(object).where(~series.isna(), np.nan)

    @staticmethod
    def _centered_targets(
        y_active: np.ndarray,
        weights: np.ndarray,
        total_weight: float,
    ) -> tuple[np.ndarray, float]:
        """Return ``(y - prior, prior)`` with a refined weighted prior.

        A single ``sum(w*y)/sum(w)`` pass is only accurate to about
        ``eps * |y|``, which is ~1e-8 for a target offset by 1e8 -- the same
        order as the category effects being estimated.  Re-running the weighted
        mean on the residuals gives a correction that is itself accurate to
        ``eps * |y - prior|``, and applying it to the *residuals* rather than to
        ``prior`` keeps it off the coarse ulp grid at ``|y|``'s magnitude, where
        it would simply be rounded away.  ``y_active - prior`` is exact whenever
        the two are within a factor of two of each other, so the returned
        centered targets carry no offset-induced error.
        """
        prior = float(np.dot(weights, y_active) / total_weight)
        residual = y_active - prior
        correction = float(np.dot(weights, residual) / total_weight)
        return residual - correction, prior + correction

    @staticmethod
    def _weighted_group_ssd(
        codes: np.ndarray,
        weights: np.ndarray,
        centered_y: np.ndarray,
        counts: np.ndarray,
        sums: np.ndarray,
    ) -> np.ndarray:
        """Two-pass weighted within-category sum of squared deviations.

        ``sums``/``counts`` are the first pass, so ``centered_y - means[codes]``
        is exactly ``y - mean_i`` whatever offset ``centered_y`` was centered by.
        The deviations are formed *before* they are squared and accumulated, so
        no ``E[y^2] - E[y]^2`` cancellation can occur: an offset target such as
        ``y + 1e8`` makes ``sum(w*y^2)`` and ``count * mean^2`` agree to ~16
        digits while their difference is the small quantity being sought, which
        is what used to leave ``lambda_i`` -- and therefore the encoding --
        dominated by rounding error.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            means = sums / counts
        deviations = centered_y - means[codes]
        return np.bincount(
            codes,
            weights=weights * np.square(deviations),
            minlength=counts.size,
        )

    @staticmethod
    def _auto_smoothed_encoding(
        counts: np.ndarray,
        sums: np.ndarray,
        ssd: np.ndarray,
        y_variance: float,
    ) -> np.ndarray:
        """Empirical-Bayes shrinkage matching sklearn's ``smooth='auto'``.

        ``lambda_i = w_i * s2y / (w_i * s2y + ssd_i / w_i)`` shrinks each
        category mean toward the training prior, where ``w_i`` is the category's
        (possibly weighted) row mass and ``ssd_i`` its within-category weighted
        sum of squared deviations.  With unit weights this reduces exactly to
        sklearn's integer-count formula.

        The weighted definition is the integer formula with every count replaced
        by weighted row mass: ``prior = sum(w*y)/sum(w)``,
        ``s2y = sum(w*(y-prior)^2)/sum(w)``, ``w_i = sum_{rows in i} w``,
        ``ssd_i = sum_{rows in i} w*(y - mean_i)^2``.  Duplicating a row ``m``
        times and giving it weight ``m`` therefore produce identical encodings,
        which is what makes ``smooth="auto"`` well defined for the weighted,
        grouped, and time-aware paths and not only for unweighted fixed-k folds.

        ``sums`` carries the *prior-centered* weighted target sum, so ``means``
        is ``mean_i - prior`` and the returned value is the centered category
        effect ``lambda_i * (mean_i - prior)`` that the encoder emits, not the
        uncentered estimate.  Shrinking in centered space also removes the final
        ``estimate - prior`` cancellation, which is why the emitted effects are
        invariant to an additive shift of ``y``.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            means = sums / counts
            lam = (y_variance * counts) / (y_variance * counts + ssd / counts)
        encoded = lam * means
        # A NaN lambda means either an empty category or a degenerate
        # zero-variance target; sklearn falls back to the prior in both cases,
        # which is the zero effect in centered space.
        return np.where(np.isfinite(encoded), encoded, 0.0)

    def _fit_custom_maps(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        sample_weight: np.ndarray,
        smooth: Literal["auto"] | float,
    ) -> tuple[dict[str, dict[object, float]], float]:
        """Fit centered category effects and return them with their prior.

        Every mapped value is ``category_estimate - prior`` so that a category
        the fitting rows never saw maps to zero rather than to the prior itself.

        Because the emitted quantity is that difference, every moment below is
        accumulated on ``y - prior`` rather than on ``y``: the centered sums,
        the global target variance, and the within-category sums of squared
        deviations.  Nothing is ever reconstructed as ``E[y^2] - E[y]^2``, so an
        offset target (``y + 1e8``) produces the same effects as ``y`` instead of
        losing the leading digits of every moment to cancellation.  This is the
        single engine behind the fixed-k, grouped, and time-aware fold kinds and
        behind both the out-of-fold and full-fit inference maps, so all of them
        inherit the invariance.
        """
        active = sample_weight > 0.0
        if not bool(np.any(active)):
            raise ValueError("target_cv fit rows have zero total sample_weight")
        weights = sample_weight[active]
        y_active = y[active]
        # Individual weights can all be finite while their aggregate overflows
        # (for example, several values near ``float64.max``). Frequency-weight
        # semantics make arbitrary rescaling incorrect for ``smooth="auto"``,
        # so fail before an infinite mass can turn the prior into NaN and the
        # centered category maps into silent zeros.
        with np.errstate(over="ignore", invalid="ignore"):
            total_weight = float(weights.sum())
        if not np.isfinite(total_weight):
            raise ValueError(
                "target_cv fit rows must have finite total sample_weight"
            )
        centered_y, prior = self._centered_targets(y_active, weights, total_weight)
        auto = smooth == "auto"
        if auto:
            y_variance = float(
                np.dot(weights, np.square(centered_y)) / total_weight
            )
        mappings: dict[str, dict[object, float]] = {}
        for col in self.cols:
            series = self._normalized_series(X.loc[active, col])
            codes, categories = pd.factorize(
                series,
                sort=False,
                use_na_sentinel=False,
            )
            counts = np.bincount(codes, weights=weights, minlength=len(categories))
            sums = np.bincount(
                codes,
                weights=weights * centered_y,
                minlength=len(categories),
            )
            if auto:
                ssd = self._weighted_group_ssd(
                    codes, weights, centered_y, counts, sums
                )
                effects = self._auto_smoothed_encoding(counts, sums, ssd, y_variance)
            else:
                # ``(sums_raw + smooth*prior) / (counts + smooth) - prior``
                # rewritten on the centered sums; algebraically identical and
                # free of the same cancellation.
                effects = sums / (counts + smooth)
            mappings[col] = {
                category: float(value)
                for category, value in zip(categories.tolist(), effects.tolist())
            }
        return mappings, prior

    def _apply_custom_maps(
        self,
        X: pd.DataFrame,
        mappings: dict[str, dict[object, float]],
    ) -> pd.DataFrame:
        """Emit centered effects; unseen categories map to a zero effect."""
        X_out = X.copy()
        for col in self.cols:
            series = self._normalized_series(X_out[col])
            X_out[col] = series.map(mappings[col]).fillna(0.0).astype(np.float64)
        return X_out

    def _active_split_count(
        self,
        y: np.ndarray,
        target_kind: str,
        active: np.ndarray,
    ) -> int:
        """Effective fixed-k fold count over the active (positive-weight) rows.

        ``fit`` and ``fit_transform`` share this so the split count ``fit``
        advertises is the one ``fit_transform`` would actually cross-fit with.
        """
        return target_cv_n_splits(
            y[active],
            target_type="binary" if target_kind == "binary" else "continuous",
            cv=self.cv,
        )

    def _fixed_splits(
        self,
        y: np.ndarray,
        target_kind: str,
        active: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        from sklearn.model_selection import KFold, StratifiedKFold

        active_idx = np.flatnonzero(active)
        requested = self._active_split_count(y, target_kind, active)
        if target_kind == "binary":
            splitter = StratifiedKFold(
                n_splits=requested,
                shuffle=True,
                random_state=self.random_state,
            )
            split_iter = splitter.split(active_idx, y[active])
        else:
            splitter = KFold(
                n_splits=requested,
                shuffle=True,
                random_state=self.random_state,
            )
            split_iter = splitter.split(active_idx)
        return [
            (active_idx[train_local], active_idx[valid_local])
            for train_local, valid_local in split_iter
        ]

    def _group_splits(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        groups,
        active: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        from sklearn.model_selection import GroupKFold

        group_arr = np.asarray(groups).reshape(-1)
        if group_arr.size != len(X):
            raise ValueError(f"groups has {group_arr.size} rows but X has {len(X)}")
        if bool(np.asarray(pd.isna(group_arr), dtype=bool).any()):
            raise ValueError("groups must not contain missing values")
        active_idx = np.flatnonzero(active)
        n_groups = int(pd.unique(group_arr[active]).size)
        n_splits = min(self._requested_cv(), n_groups)
        if n_splits < 2:
            raise ValueError(f"target_cv group folds require at least 2 groups, got {n_groups}")
        splitter = GroupKFold(n_splits=n_splits)
        return [
            (active_idx[train_local], active_idx[valid_local])
            for train_local, valid_local in splitter.split(
                X.iloc[active_idx],
                y[active_idx],
                group_arr[active_idx],
            )
        ]

    def _time_splits(
        self,
        time,
        active: np.ndarray,
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
        values = np.asarray(time).reshape(-1)
        if values.size != active.size:
            raise ValueError(f"time has {values.size} rows but X has {active.size}")
        if bool(np.asarray(pd.isna(values), dtype=bool).any()):
            raise ValueError("time values must not contain missing values")
        active_idx = np.flatnonzero(active)
        active_values = values[active_idx]
        try:
            order_local = np.argsort(active_values, kind="mergesort")
            ordered = active_values[order_local]
            if ordered.dtype.kind == "O":
                for previous, current in zip(ordered[:-1], ordered[1:]):
                    if bool(current < previous):
                        raise TypeError
            elif np.asarray(ordered[1:] < ordered[:-1], dtype=bool).any():
                raise TypeError
            distinct = np.asarray(ordered[1:] != ordered[:-1], dtype=bool)
        except (TypeError, ValueError) as exc:
            raise TypeError("time values must be orderable") from exc
        boundaries = np.flatnonzero(distinct) + 1
        timestamp_groups = np.split(active_idx[order_local], boundaries)
        n_timestamps = len(timestamp_groups)
        n_splits = min(self._requested_cv(), n_timestamps)
        if n_splits < 2:
            raise ValueError(
                "target_cv time folds require at least 2 distinct timestamps"
            )
        block_groups = np.array_split(np.arange(n_timestamps), n_splits)
        blocks = [
            np.concatenate([timestamp_groups[int(i)] for i in block]).astype(
                np.int64,
                copy=False,
            )
            for block in block_groups
        ]
        warmup = blocks[0]
        splits: list[tuple[np.ndarray, np.ndarray]] = []
        history: list[np.ndarray] = [blocks[0]]
        for block in blocks[1:]:
            splits.append((np.concatenate(history), block))
            history.append(block)
        return splits, warmup

    def _replace_columns(self, X: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
        encoded = np.asarray(values, dtype=np.float64)
        if encoded.ndim != 2 or encoded.shape[1] != len(self.cols):
            raise ValueError(
                "cat_encoding='target_cv' must preserve one encoded column per "
                "categorical feature; multiclass expansion is not supported"
            )
        X_out = X.copy()
        for index, col in enumerate(self.cols):
            X_out[col] = encoded[:, index]
        return X_out

    def fit(self, X: pd.DataFrame, y, sample_weight=None):
        self._validate_frame(X)
        y_arr = np.asarray(y).reshape(-1)
        if len(X) != y_arr.size:
            raise ValueError(f"X has {len(X)} rows but y has {y_arr.size}")
        if self.target_prior is not None or self.warmup_policy != "zero_weight":
            raise ValueError(
                "target_prior and warmup_policy are only meaningful for time-aware "
                "target_cv fit_transform"
            )
        smooth = self._validate_smoothing()
        y_values, target_kind = self._target_values(y_arr)
        weights = ensure_weights(sample_weight, len(X), normalize=False)
        self.category_maps_, self.global_prior_ = self._fit_custom_maps(
            X,
            y_values,
            weights,
            smooth,
        )
        self.target_kind_ = target_kind
        self.fit_mode_ = "sift"
        # ``fit`` alone builds only the target-blind inference maps; the split
        # count it reports is the one ``fit_transform`` would cross-fit with, so
        # it is computed over the same active (positive-weight) rows rather than
        # over all rows.  Zero-weight rows never enter a fold.
        self.n_splits_ = self._active_split_count(y_values, target_kind, weights > 0.0)
        self.encoding_cv_ = {"kind": "fixed_k", "n_splits": self.n_splits_}
        return self

    def fit_transform(
        self,
        X: pd.DataFrame,
        y,
        sample_weight=None,
        groups=None,
        time=None,
    ) -> pd.DataFrame:
        self._validate_frame(X)
        y_arr = np.asarray(y).reshape(-1)
        if len(X) != y_arr.size:
            raise ValueError(f"X has {len(X)} rows but y has {y_arr.size}")
        if groups is not None and time is not None:
            raise ValueError(
                "cat_encoding='target_cv' does not support groups and time together"
            )
        if time is None and self.target_prior is not None:
            raise ValueError("target_prior is only meaningful for time-aware target_cv")
        if time is None and self.warmup_policy != "zero_weight":
            raise ValueError("warmup_policy is only meaningful for time-aware target_cv")
        if (
            time is not None
            and self.target_prior is not None
            and self.warmup_policy == "exclude"
        ):
            raise ValueError(
                "target_prior and warmup_policy='exclude' are mutually exclusive"
            )
        smooth = self._validate_smoothing()
        y_values, target_kind = self._target_values(y_arr)
        raw_weights = ensure_weights(sample_weight, len(X), normalize=False)
        active = raw_weights > 0.0
        # Centered effects, so a row no fold could encode stays at the neutral
        # zero effect instead of carrying a fold-identifying prior.
        values = np.zeros((len(X), len(self.cols)), dtype=np.float64)
        if groups is not None:
            splits = self._group_splits(X, y_values, groups, active)
            kind = "group"
            warmup = np.empty(0, dtype=np.int64)
        elif time is not None:
            splits, warmup = self._time_splits(time, active)
            kind = "time"
            if self.target_prior is not None:
                prior = float(self.target_prior)
                if not np.isfinite(prior):
                    raise ValueError("target_prior must be finite")
                if target_kind == "binary" and not 0.0 <= prior <= 1.0:
                    raise ValueError(
                        "target_prior must be between 0 and 1 for binary targets"
                    )
                # The warmup rows are encoded against this explicit
                # target-independent prior, so their centered effect is the zero
                # already sitting in ``values``; only their selection weight
                # differs from the no-prior case below.
        else:
            splits = self._fixed_splits(y_values, target_kind, active)
            kind = "fixed_k"
            warmup = np.empty(0, dtype=np.int64)

        for train_idx, valid_idx in splits:
            mappings, _ = self._fit_custom_maps(
                X.iloc[train_idx],
                y_values[train_idx],
                raw_weights[train_idx],
                smooth,
            )
            fold_values = self._apply_custom_maps(X.iloc[valid_idx], mappings)
            values[valid_idx, :] = fold_values.loc[:, self.cols].to_numpy(
                dtype=np.float64,
            )

        effective_weights = raw_weights.copy()
        if time is not None and self.target_prior is None:
            effective_weights[warmup] = 0.0
        if not bool(np.any(effective_weights > 0.0)):
            raise ValueError(
                "target_cv warmup handling leaves no positive selection weight"
            )
        self.effective_sample_weight_ = (
            effective_weights
            if sample_weight is not None or warmup.size > 0
            else None
        )
        self.warmup_mask_ = np.ones(len(X), dtype=bool)
        self.warmup_mask_[warmup] = False
        self.category_maps_, self.global_prior_ = self._fit_custom_maps(
            X,
            y_values,
            raw_weights,
            smooth,
        )
        self.target_kind_ = target_kind
        self.fit_mode_ = "sift"
        self.n_splits_ = len(splits) + (1 if time is not None else 0)
        self.encoding_cv_ = {"kind": kind, "n_splits": self.n_splits_}
        return self._replace_columns(X, values)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, ["fit_mode_", "encoding_cv_"])
        self._validate_frame(X)
        return self._apply_custom_maps(X, self.category_maps_)


def encode_categoricals(
    X: pd.DataFrame,
    y: pd.Series,
    cat_features: List[str],
    method: CatEncoding,
    *,
    loo_smoothing: float = 20.0,
    loo_clip_min: float = 1e-4,
    loo_clip_max: float = 1.0 - 1e-4,
    sample_weight=None,
    target_type: Literal["auto", "continuous", "binary"] = "auto",
    target_cv_n_splits: int = 5,
    target_cv_smoothing: Literal["auto"] | float = "auto",
    target_prior: float | None = None,
    warmup_policy: Literal["exclude", "zero_weight"] = "zero_weight",
    groups=None,
    time=None,
) -> pd.DataFrame:
    """Apply target encoding to categorical features."""
    if method == "none":
        return X
    if method == "target_cv":
        encoder = TargetCVEncoder(
            cat_features,
            target_type=target_type,
            smooth=target_cv_smoothing,
            cv=target_cv_n_splits,
            target_prior=target_prior,
            warmup_policy=warmup_policy,
        )
        return encoder.fit_transform(
            X,
            y,
            sample_weight=sample_weight,
            groups=groups,
            time=time,
        )
    if method == "loo_logit":
        encoder = LeaveOneOutLogitEncoder(
            cat_features,
            smoothing=loo_smoothing,
            clip_min=loo_clip_min,
            clip_max=loo_clip_max,
        )
        return encoder.fit_transform(X, y, sample_weight=sample_weight)
    if sample_weight is not None:
        raise ValueError(
            "sample_weight with supervised categorical encoding is only supported "
            "for cat_encoding='loo_logit'. category_encoders-backed methods "
            "('loo', 'target', 'james_stein') do not consume sample weights."
        )
    try:
        import category_encoders as ce
    except ImportError as exc:
        raise ImportError(
            "category_encoders required for categorical encoding. "
            "Install with: pip install category_encoders"
        ) from exc

    encoders = {
        "loo": ce.LeaveOneOutEncoder,
        "target": ce.TargetEncoder,
        "james_stein": ce.JamesSteinEncoder,
    }
    encoder = encoders[method](cols=cat_features, handle_missing="return_nan")
    with suppress_category_encoder_pandas_warnings():
        return encoder.fit_transform(X, y)
