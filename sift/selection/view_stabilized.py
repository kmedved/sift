"""Fitted Stabilized adapter for additive SelectionView construction."""

from __future__ import annotations

import copy
import math
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from sift._selector_compat import ordered_indices, validate_output_order
from sift.selection.reproducibility import describe_estimator, snapshot_selector_kwargs
from sift.selection.view import (
    SelectionView,
    _coerce_feature_names,
    _coerce_indices,
    _labels_equal,
    _numeric_vector,
    _strict_integer,
    _validate_selected_identity,
)


def _as_stabilized_selector(selector: Any, input_features: Any) -> SelectionView:
    from sklearn.utils.validation import check_is_fitted

    required = [
        "feature_names_in_",
        "n_features_in_",
        "selected_features_",
        "selected_indices_",
        "n_features_selected_",
        "selection_frequencies_",
    ]
    check_is_fitted(selector, required)

    n_features = _strict_integer(
        selector.n_features_in_,
        label="Stabilized n_features_in_",
        minimum=1,
    )
    raw_features = _coerce_feature_names(selector.feature_names_in_)
    if raw_features is None or len(raw_features) != n_features:
        raise ValueError("Stabilized feature_names_in_ must match n_features_in_")
    raw_values = np.empty(n_features, dtype=object)
    raw_values[:] = raw_features
    raw_index = pd.Index(raw_values, dtype=object, tupleize_cols=False)
    if raw_index.duplicated().any():
        raise ValueError("Stabilized feature_names_in_ must be unique")

    supplied_features = _coerce_feature_names(input_features)
    if supplied_features is not None:
        if len(supplied_features) != n_features or any(
            not _labels_equal(expected, observed)
            for expected, observed in zip(raw_features, supplied_features)
        ):
            raise ValueError(
                "input_features must match Stabilized feature_names_in_ in exact order"
            )

    selected_indices = _coerce_indices(
        selector.selected_indices_,
        label="Stabilized selected_indices_",
    )
    assert selected_indices is not None
    selected = _coerce_feature_names(selector.selected_features_)
    if selected is None:
        raise ValueError("Stabilized selected_features_ must be an ordered iterable")
    selected_indices = _validate_selected_identity(
        selected,
        selected_indices,
        raw_features,
    )
    if any(index >= n_features for index in selected_indices):
        raise ValueError("Stabilized selected_indices_ contains an out-of-bounds position")
    selected_count = _strict_integer(
        selector.n_features_selected_,
        label="Stabilized n_features_selected_",
        minimum=0,
    )
    if selected_count != len(selected_indices):
        raise ValueError("Stabilized n_features_selected_ must match selected_indices_")

    frequencies = _numeric_vector(
        selector.selection_frequencies_,
        label="Stabilized selection_frequencies_",
        length=n_features,
    )
    if not np.isfinite(frequencies).all() or ((frequencies < 0.0) | (frequencies > 1.0)).any():
        raise ValueError(
            "Stabilized selection_frequencies_ must be finite values in [0, 1]"
        )

    fit_configured = getattr(selector, "_fit_configured_options_", None)
    if fit_configured is not None and not isinstance(fit_configured, dict):
        raise ValueError("Stabilized _fit_configured_options_ must be a mapping")
    if not isinstance(selector.threshold, Real):
        raise ValueError("Stabilized threshold must be a real value in [0, 1]")
    if isinstance(fit_configured, dict) and "threshold" in fit_configured:
        threshold = float(fit_configured["threshold"])
    else:
        threshold = float(selector.threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Stabilized threshold must be a finite value in [0, 1]")
    if isinstance(fit_configured, dict) and "n_resamples" in fit_configured:
        n_resamples_requested = _strict_integer(
            fit_configured["n_resamples"],
            label="Stabilized n_resamples",
            minimum=1,
        )
    else:
        n_resamples_requested = _strict_integer(
            selector.n_resamples,
            label="Stabilized n_resamples",
            minimum=1,
        )
    mode = getattr(selector, "_aggregation_mode_", "frequency")
    if mode == "frequency":
        expected_mask = frequencies >= threshold
        expected_indices = np.flatnonzero(expected_mask)
        expected_indices = expected_indices[
            np.argsort(-frequencies[expected_indices], kind="mergesort")
        ].tolist()
        if selected_indices != expected_indices:
            raise ValueError(
                "Stabilized selected_indices_ is inconsistent with its threshold "
                "and selection_frequencies_"
            )

    selected_mask = np.zeros(n_features, dtype=bool)
    selected_mask[selected_indices] = True
    output_order = validate_output_order(selector.output_order)
    view_indices = [int(index) for index in ordered_indices(selected_indices, output_order)]
    view_selected = [raw_features[index] for index in view_indices]

    path_rank = pd.Series(pd.array([pd.NA] * n_features, dtype="Int64"))
    for rank, index in enumerate(view_indices, start=1):
        path_rank.iloc[index] = rank
    table = pd.DataFrame(
        {
            "feature": raw_features,
            "selected_index": pd.array(range(n_features), dtype="Int64"),
            "path_rank": path_rank,
            "selected": selected_mask,
            "selection_frequency": frequencies.copy(),
            "gain": frequencies.copy(),
        }
    )

    has_generated_marker = hasattr(selector, "_fit_feature_names_generated_")
    has_input_kind_marker = hasattr(selector, "_fit_input_kind_")
    if not (has_generated_marker and has_input_kind_marker):
        generated_names = False
        input_kind = "unknown"
    else:
        generated_names = selector._fit_feature_names_generated_
        fit_input_kind = selector._fit_input_kind_
        if not isinstance(generated_names, (bool, np.bool_)):
            raise ValueError("Stabilized _fit_feature_names_generated_ must be boolean")
        if fit_input_kind not in {"dataframe", "positional"}:
            raise ValueError(
                "Stabilized _fit_input_kind_ must be 'dataframe' or 'positional'"
            )
        generated_names = bool(generated_names)
        input_kind = fit_input_kind

    transform_selector = type(selector)(selector=selector.selector)
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

    proxy_correlations = getattr(selector, "_proxy_correlations", None)
    proxy_usable = False
    if proxy_correlations is not None:
        stored_columns = [int(column) for column in proxy_correlations.columns.tolist()]
        if stored_columns == view_indices:
            proxy_correlations = proxy_correlations.copy()
            proxy_usable = True
        elif set(view_indices).issubset(stored_columns):
            proxy_correlations = proxy_correlations.loc[:, view_indices].copy()
            proxy_usable = True
        else:
            proxy_correlations = None
    resample_selections = None
    if proxy_usable:
        stored_resamples = getattr(selector, "_resample_selections_", None)
        if stored_resamples is not None:
            resample_selections = np.asarray(stored_resamples, dtype=bool).copy()

    extra = dict(getattr(selector, "_extra_result_metadata_", {}) or {})
    n_rows = getattr(selector, "_n_rows_original_", None)
    n_rows_used = getattr(selector, "_n_rows_used_", None)
    if isinstance(fit_configured, dict):
        configured = copy.deepcopy(fit_configured)
    else:
        configured = snapshot_selector_kwargs(
            {
                "base_selector": describe_estimator(selector.selector),
                "n_resamples": int(selector.n_resamples),
                "resample": selector.resample,
                "threshold": float(selector.threshold),
                "sample_frac": selector.sample_frac,
                "aggregation": selector.aggregation,
                "random_state": int(selector.random_state),
                "store_proxies": bool(selector.store_proxies),
                "output_order": selector.output_order,
                "n_jobs": selector.n_jobs,
                "block_size": selector.block_size,
                "block_method": selector.block_method,
            }
        )
    actual_seed = getattr(selector, "_actual_random_state_", None)
    metadata = {
        "adapter": "Stabilized",
        "selector": "stabilized",
        "curve_available": False,
        "table_complete": True,
        "input_kind": input_kind,
        "raw_namespace": "fitted_candidate_features",
        "output_order": output_order,
        "selected_feature_count": selected_count,
        "gain_source": "selection_frequency",
        "selection_frequency_source": (
            "knockoff_evalues" if mode == "evalues" else "completed_resample_fraction"
        ),
        "n_resamples": n_resamples_requested,
        "n_resamples_requested": n_resamples_requested,
        "n_resamples_completed": int(
            getattr(selector, "_n_completed_resamples_", n_resamples_requested)
        ),
        "aggregation": "evalues" if mode == "evalues" else "frequency",
        "store_proxies": bool(getattr(selector, "store_proxies", False)),
        "n_features": n_features,
        "n_rows_original": None if n_rows is None else int(n_rows),
        "fdr_control": extra.get("fdr_control", "none"),
        "configured_options": configured,
    }
    if mode != "evalues":
        metadata["threshold"] = threshold
        metadata["resample"] = configured.get("resample", selector.resample)
        metadata["sample_frac"] = configured.get("sample_frac", selector.sample_frac)
    if n_rows_used is not None:
        metadata["n_rows_used"] = int(n_rows_used)
    if actual_seed is not None:
        metadata["random_state"] = (
            int(actual_seed)
            if isinstance(actual_seed, (int, np.integer))
            else actual_seed
        )
    if extra:
        metadata["knockoff_metadata"] = extra
        if extra.get("n_rows_used") is not None and n_rows_used is None:
            metadata["n_rows_used"] = int(extra["n_rows_used"])
        if extra.get("n_rows_original") is not None and n_rows is not None:
            metadata["n_rows_original"] = int(n_rows)
    row_counts = getattr(selector, "_resample_row_counts_", None)
    unique_counts = getattr(selector, "_resample_unique_counts_", None)
    diagnostics = {
        "fit_context": {
            "sample_weight": bool(getattr(selector, "_fit_used_sample_weight_", False)),
            "groups": bool(getattr(selector, "_fit_used_groups_", False)),
            "time": bool(getattr(selector, "_fit_used_time_", False)),
        },
        "aggregation_mode": mode,
        "n_completed_resamples": int(
            getattr(selector, "_n_completed_resamples_", n_resamples_requested)
        ),
        "rng": getattr(
            selector,
            "_rng_mechanism_",
            "numpy.random.SeedSequence.spawn",
        ),
    }
    if row_counts is not None:
        diagnostics["resample_n_rows"] = [int(value) for value in np.asarray(row_counts)]
    if unique_counts is not None:
        diagnostics["resample_n_unique"] = [
            int(value) for value in np.asarray(unique_counts)
        ]
    return SelectionView(
        features=view_selected,
        indices=view_indices,
        raw_features=raw_features,
        n_raw_features=n_features,
        raw_table=table,
        metadata=metadata,
        diagnostics=diagnostics,
        transformer=transform_selector.transform,
        inverse_transformer=transform_selector.inverse_transform,
        proxy_correlations=proxy_correlations,
        resample_selections=resample_selections,
    )
