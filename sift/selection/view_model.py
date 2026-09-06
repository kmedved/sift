"""Fitted ModelSelector adapter for additive SelectionView construction."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd

from sift._selector_compat import ordered_indices, validate_output_order
from sift.selection.reproducibility import describe_estimator, snapshot_selector_kwargs
from sift.selection.view import (
    CURVE_COLUMNS,
    SelectionView,
    _coerce_feature_names,
    _coerce_indices,
    _labels_equal,
    _numeric_vector,
    _strict_integer,
    _validate_selected_identity,
)


def _as_model_selector(selector: Any, input_features: Any) -> SelectionView:
    from sklearn.utils.validation import check_is_fitted

    required = [
        "feature_names_in_",
        "n_features_in_",
        "selected_features_",
        "selected_indices_",
        "n_features_selected_",
    ]
    check_is_fitted(selector, required)

    n_features = _strict_integer(
        selector.n_features_in_,
        label="ModelSelector n_features_in_",
        minimum=1,
    )
    raw_features = _coerce_feature_names(selector.feature_names_in_)
    if raw_features is None or len(raw_features) != n_features:
        raise ValueError("ModelSelector feature_names_in_ must match n_features_in_")
    raw_values = np.empty(n_features, dtype=object)
    raw_values[:] = raw_features
    raw_index = pd.Index(raw_values, dtype=object, tupleize_cols=False)
    if raw_index.duplicated().any():
        raise ValueError("ModelSelector feature_names_in_ must be unique")

    supplied_features = _coerce_feature_names(input_features)
    if supplied_features is not None:
        if len(supplied_features) != n_features or any(
            not _labels_equal(expected, observed)
            for expected, observed in zip(raw_features, supplied_features)
        ):
            raise ValueError(
                "input_features must match ModelSelector feature_names_in_ in exact order"
            )

    selected_indices = _coerce_indices(
        selector.selected_indices_,
        label="ModelSelector selected_indices_",
    )
    assert selected_indices is not None
    selected = _coerce_feature_names(selector.selected_features_)
    if selected is None:
        raise ValueError("ModelSelector selected_features_ must be an ordered iterable")
    selected_indices = _validate_selected_identity(
        selected,
        selected_indices,
        raw_features,
    )
    if any(index >= n_features for index in selected_indices):
        raise ValueError("ModelSelector selected_indices_ contains an out-of-bounds position")
    selected_count = _strict_integer(
        selector.n_features_selected_,
        label="ModelSelector n_features_selected_",
        minimum=0,
    )
    if selected_count != len(selected_indices):
        raise ValueError("ModelSelector n_features_selected_ must match selected_indices_")

    output_order = validate_output_order(selector.output_order)
    view_indices = [int(index) for index in ordered_indices(selected_indices, output_order)]
    view_selected = [raw_features[index] for index in view_indices]
    selected_mask = np.zeros(n_features, dtype=bool)
    selected_mask[selected_indices] = True

    path_rank = pd.Series(pd.array([pd.NA] * n_features, dtype="Int64"))
    for rank, index in enumerate(view_indices, start=1):
        path_rank.iloc[index] = rank

    frequencies = getattr(selector, "selection_frequencies_", None)
    freq_values = None
    if frequencies is not None:
        freq_values = _numeric_vector(
            frequencies,
            label="ModelSelector selection_frequencies_",
            length=n_features,
        )
        if not np.isfinite(freq_values).all() or (
            (freq_values < 0.0) | (freq_values > 1.0)
        ).any():
            raise ValueError(
                "ModelSelector selection_frequencies_ must be finite values in [0, 1]"
            )

    table = pd.DataFrame(
        {
            "feature": raw_features,
            "selected_index": pd.array(range(n_features), dtype="Int64"),
            "path_rank": path_rank,
            "selected": selected_mask,
        }
    )
    if freq_values is not None:
        table["selection_frequency"] = freq_values.copy()
        table["gain"] = freq_values.copy()
        gain_source = "selection_frequency"
    else:
        gain_source = "none"

    scores_by_k = getattr(selector, "scores_by_k_", None)
    curve = None
    if scores_by_k:
        se = getattr(selector, "scores_by_k_se_", None) or {}
        chosen = int(getattr(selector, "n_features_to_select_", selected_count))
        rows = []
        for k in sorted(int(key) for key in scores_by_k):
            se_val = se.get(k, np.nan) if isinstance(se, dict) else np.nan
            rows.append(
                {
                    "k": int(k),
                    "criterion": float(scores_by_k[k]),
                    "criterion_se": float(se_val) if se_val is not None else np.nan,
                    "selected": int(k) == chosen,
                }
            )
        curve = pd.DataFrame(rows, columns=list(CURVE_COLUMNS))

    generated_names = bool(getattr(selector, "_fit_feature_names_generated_", False))
    input_kind = getattr(selector, "_fit_input_kind_", "unknown")
    if input_kind not in {"dataframe", "positional", "unknown"}:
        raise ValueError("ModelSelector _fit_input_kind_ is invalid")

    transform_selector = type(selector)(estimator=selector.estimator)
    transform_selector.feature_names_in_ = raw_values.copy()
    transform_selector.n_features_in_ = n_features
    transform_selector.selected_features_ = list(selected)
    transform_selector.selected_indices_ = np.asarray(selected_indices, dtype=np.int64)
    transform_selector.output_order = output_order
    transform_selector._fit_feature_names_generated_ = generated_names
    if hasattr(selector, "_row_metadata_columns_"):
        transform_selector._row_metadata_columns_ = tuple(selector._row_metadata_columns_)
    if hasattr(selector, "_sklearn_output_config"):
        transform_selector._sklearn_output_config = copy.deepcopy(
            selector._sklearn_output_config
        )

    fit_configured = getattr(selector, "_fit_configured_options_", None)
    if isinstance(fit_configured, dict):
        configured = copy.deepcopy(fit_configured)
    else:
        configured = snapshot_selector_kwargs(
            {
                "estimator": describe_estimator(selector.estimator),
                "method": selector.method,
                "n_features_to_select": selector.n_features_to_select,
                "min_features_to_select": int(selector.min_features_to_select),
                "step": int(selector.step),
                "nested": bool(selector.nested),
                "importance": selector.importance
                if not callable(selector.importance)
                else "callable",
                "threshold": float(selector.threshold),
                "n_resamples": int(selector.n_resamples),
                "random_state": int(selector.random_state),
                "parsimony_tolerance": float(selector.parsimony_tolerance),
                "selection_patience": int(selector.selection_patience),
                "output_order": output_order,
            }
        )
    nested_scores = getattr(selector, "nested_scores_", None)
    metadata = {
        "adapter": "ModelSelector",
        "selector": "model_selector",
        "curve_available": curve is not None,
        "table_complete": True,
        "input_kind": input_kind,
        "raw_namespace": "fitted_candidate_features",
        "output_order": output_order,
        "selected_feature_count": selected_count,
        "gain_source": gain_source,
        "method": configured.get("method", selector.method),
        "nested": bool(configured.get("nested", selector.nested)),
        "n_features": n_features,
        "n_rows_original": int(getattr(selector, "_n_rows_original_", 0)) or None,
        "configured_options": configured,
        "selection_curve_is_nested_score": False,
    }
    if freq_values is not None:
        metadata["selection_frequency_source"] = "stability_resample_fraction"
        metadata["threshold"] = float(configured.get("threshold", selector.threshold))
        metadata["n_resamples"] = int(configured.get("n_resamples", selector.n_resamples))
    diagnostics = {
        "fit_context": {
            "sample_weight": bool(getattr(selector, "_fit_used_sample_weight_", False)),
            "groups": bool(getattr(selector, "_fit_used_groups_", False)),
            "time": bool(getattr(selector, "_fit_used_time_", False)),
        }
    }
    if nested_scores is not None:
        diagnostics["nested_scores"] = np.asarray(nested_scores, dtype=np.float64).tolist()
    fold_notes = getattr(selector, "nested_fold_diagnostics_", None)
    if fold_notes is not None:
        diagnostics["nested_fold_diagnostics"] = copy.deepcopy(list(fold_notes))

    return SelectionView(
        features=view_selected,
        indices=view_indices,
        raw_features=raw_features,
        n_raw_features=n_features,
        raw_table=table,
        curve=curve,
        metadata=metadata,
        diagnostics=diagnostics,
        transformer=transform_selector.transform,
        inverse_transformer=transform_selector.inverse_transform,
    )
