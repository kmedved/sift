"""Shared sklearn contracts for SIFT selector classes."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse


OUTPUT_ORDERS = frozenset({"legacy", "original"})


def validate_output_order(value: object) -> str:
    """Validate and return a selector output-order option."""
    if not isinstance(value, str) or value not in OUTPUT_ORDERS:
        raise ValueError("output_order must be 'legacy' or 'original'")
    return value


def ordered_indices(indices: Any, output_order: object) -> np.ndarray:
    """Return selected positions in the configured transform order."""
    order = validate_output_order(output_order)
    selected = np.asarray(indices, dtype=np.int64).reshape(-1)
    if order == "original":
        return np.sort(selected, kind="stable")
    return selected.copy()


def reject_sparse(X: Any, *, operation: str) -> None:
    """Reject sparse input with one stable, actionable message."""
    if sparse.issparse(X):
        raise TypeError(
            f"Sparse matrices are not supported by SIFT selectors during {operation}; "
            "pass a dense NumPy array or pandas DataFrame."
        )


def _has_complex_dtype(X: Any, materialized: np.ndarray | None) -> bool:
    """Report complex input without dispatching through ``__array_function__``.

    ``np.iscomplexobj`` is a NumPy array function, so duck-typed array-likes
    that implement ``__array_function__`` defensively -- sklearn's
    ``_NotAnArray`` estimator-check helper is the canonical example -- raise
    ``TypeError`` instead of answering.  Inspect declared dtypes first and fall
    back to the already-materialized ``np.asarray`` view, which only relies on
    the ``__array__`` protocol.
    """
    if isinstance(X, pd.DataFrame):
        # Reading column dtypes avoids materializing a wide frame twice.
        return any(getattr(dtype, "kind", None) == "c" for dtype in X.dtypes)
    if isinstance(X, pd.Series):
        return getattr(X.dtype, "kind", None) == "c"
    kind = getattr(getattr(X, "dtype", None), "kind", None)
    if kind is not None:
        return kind == "c"
    if materialized is None:
        try:
            materialized = np.asarray(X)
        except Exception:  # pragma: no cover - defer to the caller's own error
            return False
    return getattr(materialized.dtype, "kind", None) == "c"


def validate_fit_matrix(X: Any) -> None:
    """Apply the shared dense/shape/complex fit-input contract."""
    reject_sparse(X, operation="fit")
    materialized: np.ndarray | None = None
    shape = getattr(X, "shape", None)
    if shape is None:
        # Array-likes without a shape (lists, duck arrays) are materialized once
        # through ``__array__`` and reused for the complex-dtype check below.
        materialized = np.asarray(X)
        shape = materialized.shape
    if len(shape) != 2:
        raise ValueError(
            "X must be a 2D feature matrix (2-dimensional). Reshape your data with "
            "X.reshape(-1, 1) for a single feature."
        )
    if int(shape[0]) == 0:
        raise ValueError(
            f"Found array with 0 sample(s) (shape={shape}) while a minimum of 1 is required."
        )
    if int(shape[1]) == 0:
        raise ValueError(
            f"Found array with 0 feature(s) (shape={shape}) while a minimum of 1 is required."
        )
    if _has_complex_dtype(X, materialized):
        # sklearn's ``check_complex_data`` matches on "Complex data not supported".
        raise ValueError("Complex data not supported by SIFT selectors")


def inverse_selected_matrix(
    X_selected: Any,
    *,
    n_features: int,
    selected_indices: Any,
) -> np.ndarray:
    """Insert zero columns around a dense selected-feature matrix."""
    reject_sparse(X_selected, operation="inverse_transform")
    values = np.asarray(X_selected)
    if values.ndim != 2:
        raise ValueError(
            "X must be a 2-dimensional (2D) selected-feature matrix. Reshape your data with "
            "X.reshape(-1, 1) for a single feature."
        )
    indices = np.asarray(selected_indices, dtype=np.int64).reshape(-1)
    if values.shape[1] != indices.size:
        raise ValueError(
            "X has a different number of selected features than this selector's "
            "transform output"
        )
    restored = np.zeros((values.shape[0], int(n_features)), dtype=values.dtype)
    restored[:, indices] = values
    return restored


def selector_tags(
    tags: Any,
    *,
    non_deterministic: bool = False,
    binary_only: bool = False,
) -> Any:
    """Set selector tags across sklearn's legacy and object tag APIs."""
    if isinstance(tags, dict):
        updated = dict(tags)
        updated["allow_nan"] = True
        updated["requires_y"] = True
        if non_deterministic:
            updated["non_deterministic"] = True
        if binary_only:
            updated["binary_only"] = True
        return updated

    tags.input_tags.allow_nan = True
    tags.target_tags.required = True
    if non_deterministic:
        tags.non_deterministic = True
    # ``binary_only`` deliberately has no object-tag counterpart here.  sklearn
    # 1.6 replaced the flat ``binary_only`` key with
    # ``Tags.classifier_tags.multi_class = False``, and ``classifier_tags`` is
    # only populated for estimators whose ``estimator_type`` is "classifier".
    # SIFT's binary selectors are transformers, so the honest representation is
    # ``classifier_tags is None`` plus the runtime two-class validation error;
    # fabricating a ``ClassifierTags`` block on a transformer would misdeclare
    # the estimator type to sklearn's dispatch helpers.  See
    # ``tests/contracts/test_sklearn_selector_integration.py`` for the pinned
    # cross-version representation.
    return tags


def feature_names_array(feature_names: Any) -> np.ndarray:
    """Return fitted feature names as sklearn's 1-D object ndarray.

    sklearn requires ``feature_names_in_`` to be a one-dimensional NumPy object
    array.  Filling an empty object array element-wise keeps tuple and other
    non-scalar labels intact, which ``np.asarray(names, dtype=object)`` would
    otherwise expand into a two-dimensional array.
    """
    names = list(feature_names)
    result = np.empty(len(names), dtype=object)
    result[:] = names
    return result


def _feature_name_mismatch_message(fitted: list, observed: list) -> str:
    """Reproduce sklearn's standard feature-name mismatch message.

    sklearn builds this text in the private ``_check_feature_names`` helper,
    which lives on ``BaseEstimator`` up to sklearn 1.6 and moved to a module
    function in 1.7.  ``check_dataframe_column_names_consistency`` matches the
    exact wording, so build it here instead of reaching into either location.
    """

    def add_names(names: list) -> str:
        output = ""
        for index, name in enumerate(names):
            if index >= 5:
                output += "- ...\n"
                break
            output += f"- {name}\n"
        return output

    unexpected = sorted(set(observed) - set(fitted))
    missing = sorted(set(fitted) - set(observed))
    message = "The feature names should match those that were passed during fit.\n"
    if unexpected:
        message += "Feature names unseen at fit time:\n" + add_names(unexpected)
    if missing:
        message += "Feature names seen at fit time, yet now missing:\n" + add_names(
            missing
        )
    if not unexpected and not missing:
        message += "Feature names must be in the same order as they were in fit.\n"
    return message


def check_fitted_column_identity(X: Any, fitted_names: Any) -> None:
    """Enforce the fitted DataFrame column contract during ``transform``.

    All-string columns get sklearn's standard mismatch wording, which names the
    unexpected and missing labels; any other label type keeps SIFT's own strict
    order/identity message.
    """
    observed = list(getattr(X, "columns", ()))
    fitted = list(fitted_names)
    if observed == fitted:
        return
    if (
        # An empty column set is not "feature names" to sklearn, so it keeps the
        # SIFT-specific message below.
        observed
        and all(isinstance(label, str) for label in fitted)
        and all(isinstance(label, str) for label in observed)
    ):
        raise ValueError(_feature_name_mismatch_message(fitted, observed))
    raise ValueError("DataFrame columns must match fitted columns and order")


__all__: list[str] = []
