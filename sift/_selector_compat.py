"""Shared sklearn contracts for SIFT selector classes."""

from __future__ import annotations

from typing import Any

import numpy as np
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


def validate_fit_matrix(X: Any) -> None:
    """Apply the shared dense/shape/complex fit-input contract."""
    reject_sparse(X, operation="fit")
    shape = X.shape if hasattr(X, "shape") else np.asarray(X).shape
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
    if np.iscomplexobj(X):
        raise ValueError("Complex data is not supported by SIFT selectors")


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


def selector_tags(tags: Any, *, non_deterministic: bool = False) -> Any:
    """Set selector tags across sklearn's legacy and object tag APIs."""
    if isinstance(tags, dict):
        updated = dict(tags)
        updated["allow_nan"] = True
        updated["requires_y"] = True
        if non_deterministic:
            updated["non_deterministic"] = True
        return updated

    tags.input_tags.allow_nan = True
    tags.target_tags.required = True
    if non_deterministic:
        tags.non_deterministic = True
    return tags


__all__: list[str] = []
