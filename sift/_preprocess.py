"""Input preprocessing: validation, conversion, subsampling, encoding."""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

EstimatorJMI = Literal["auto", "binned", "r2", "ksg", "gaussian"]
EstimatorMRMR = Literal["classic", "gaussian"]
RelevanceMethod = Literal["f", "ks", "rf"]
CatEncoding = Literal["none", "target", "loo", "james_stein"]
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
    return False


def best_score_from_dict(scores: dict, higher_is_better: bool) -> Tuple[int, float]:
    """Return best (index, score) given a dict of scores."""
    if not scores:
        return 0, float("nan")

    valid = {k: v for k, v in scores.items() if np.isfinite(v)}
    if not valid:
        return 0, float("nan")

    if higher_is_better:
        best = max(valid.items(), key=lambda x: x[1])
    else:
        best = min(valid.items(), key=lambda x: x[1])
    return best[0], best[1]


# --- Input conversion ---


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


# --- Validation ---


def validate_inputs(
    X, y, task: str, impute: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Validate and convert inputs."""
    feature_names = extract_feature_names(X)
    if hasattr(X, "select_dtypes"):
        non_numeric = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        if non_numeric:
            sample = non_numeric[:5]
            suffix = "..." if len(non_numeric) > 5 else ""
            raise ValueError(
                f"Non-numeric columns found: {sample}{suffix}. "
                "Either encode them first or set cat_encoding to 'loo', "
                "'target', or 'james_stein'."
            )
    X_arr = to_numpy(X, dtype=np.float64)

    if impute:
        X_arr = np.where(np.isfinite(X_arr), X_arr, np.nan)
        col_means = np.nanmean(X_arr, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        nan_mask = np.isnan(X_arr)
        if nan_mask.any():
            X_arr[nan_mask] = col_means[np.where(nan_mask)[1]]

    X_arr = X_arr.astype(np.float32)

    if task == "classification":
        if hasattr(y, "values"):
            y_raw = y.values
        else:
            y_raw = np.asarray(y)

        if pd.api.types.is_numeric_dtype(y_raw):
            try:
                y_num = np.asarray(y_raw, dtype=np.float64)
            except (TypeError, ValueError):
                y_num = None
            if y_num is not None and not np.isfinite(y_num).all():
                raise ValueError("Non-finite values in y are not allowed for classification.")

        if pd.isna(y_raw).any():
            raise ValueError("Missing values in y are not allowed for classification.")

        _, y_arr = np.unique(y_raw, return_inverse=True)
        y_arr = y_arr.astype(np.int32)
    else:
        y_arr = to_numpy(y, dtype=np.float32)
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
) -> Tuple[np.ndarray, np.ndarray]:
    """Subsample to at most `subsample` rows."""
    if subsample is None or len(X) <= subsample:
        return X, y
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X), size=subsample, replace=False)
    return X[idx], y[idx]


# --- Categorical encoding ---


def encode_categoricals(
    X: pd.DataFrame,
    y: pd.Series,
    cat_features: List[str],
    method: CatEncoding,
) -> pd.DataFrame:
    """Apply target encoding to categorical features."""
    if method == "none":
        return X
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
    return encoder.fit_transform(X, y)
