"""Fitted StabilitySelector adapter for additive SelectionView construction."""

from __future__ import annotations

import copy
import math
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from sift._selector_compat import ordered_indices, validate_output_order
from sift.selection.view import (
    SelectionView,
    _coerce_feature_names,
    _coerce_indices,
    _labels_equal,
    _numeric_vector,
    _strict_integer,
    _validate_selected_identity,
)


def _as_stability_selector(selector: Any, input_features: Any) -> SelectionView:
    from sklearn.utils.validation import check_is_fitted

    required = [
        "feature_names_in_",
        "n_features_in_",
        "selected_features_",
        "selected_feature_names_",
        "n_features_selected_",
        "selection_frequencies_",
        "mean_abs_coef_",
        "alpha_",
        "alpha_rule_effective_",
    ]
    check_is_fitted(selector, required)

    n_features = _strict_integer(
        selector.n_features_in_,
        label="StabilitySelector n_features_in_",
        minimum=1,
    )
    raw_features = _coerce_feature_names(selector.feature_names_in_)
    if raw_features is None or len(raw_features) != n_features:
        raise ValueError(
            "StabilitySelector feature_names_in_ must match n_features_in_"
        )
    raw_values = np.empty(n_features, dtype=object)
    raw_values[:] = raw_features
    raw_index = pd.Index(raw_values, dtype=object, tupleize_cols=False)
    if raw_index.duplicated().any():
        raise ValueError("StabilitySelector feature_names_in_ must be unique")

    supplied_features = _coerce_feature_names(input_features)
    if supplied_features is not None:
        if len(supplied_features) != n_features or any(
            not _labels_equal(expected, observed)
            for expected, observed in zip(raw_features, supplied_features)
        ):
            raise ValueError(
                "input_features must match StabilitySelector feature_names_in_ "
                "in exact order"
            )

    selected_indices = _coerce_indices(
        selector.selected_features_,
        label="StabilitySelector selected_features_",
    )
    assert selected_indices is not None
    selected = _coerce_feature_names(selector.selected_feature_names_)
    if selected is None:
        raise ValueError(
            "StabilitySelector selected_feature_names_ must be an ordered iterable"
        )
    selected_indices = _validate_selected_identity(
        selected,
        selected_indices,
        raw_features,
    )
    if any(index >= n_features for index in selected_indices):
        raise ValueError(
            "StabilitySelector selected_features_ contains an out-of-bounds position"
        )
    selected_count = _strict_integer(
        selector.n_features_selected_,
        label="StabilitySelector n_features_selected_",
        minimum=0,
    )
    if selected_count != len(selected_indices):
        raise ValueError(
            "StabilitySelector n_features_selected_ must match selected_features_"
        )

    frequencies = _numeric_vector(
        selector.selection_frequencies_,
        label="StabilitySelector selection_frequencies_",
        length=n_features,
    )
    if not np.isfinite(frequencies).all() or (
        (frequencies < 0.0) | (frequencies > 1.0)
    ).any():
        raise ValueError(
            "StabilitySelector selection_frequencies_ must be finite values in [0, 1]"
        )
    mean_abs_coef = _numeric_vector(
        selector.mean_abs_coef_,
        label="StabilitySelector mean_abs_coef_",
        length=n_features,
    )
    if not np.isfinite(mean_abs_coef).all() or (mean_abs_coef < 0.0).any():
        raise ValueError(
            "StabilitySelector mean_abs_coef_ must contain finite non-negative values"
        )
    mean_abs_coef_values = np.asarray(selector.mean_abs_coef_).copy()

    if selector.task not in {"regression", "classification"}:
        raise ValueError(
            "StabilitySelector task must be 'regression' or 'classification'"
        )
    if not isinstance(selector.threshold, Real):
        raise ValueError("StabilitySelector threshold must be a real value in [0, 1]")
    threshold = float(selector.threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("StabilitySelector threshold must be a finite value in [0, 1]")
    if selector.max_features is None:
        max_features = None
    else:
        max_features = _strict_integer(
            selector.max_features,
            label="StabilitySelector max_features",
            minimum=1,
        )
    if not isinstance(selector.alpha_, Real):
        raise ValueError("StabilitySelector alpha_ must be a finite positive value")
    alpha = float(selector.alpha_)
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("StabilitySelector alpha_ must be a finite positive value")
    if selector.alpha_rule_effective_ not in {"fixed", "one_se", "best"}:
        raise ValueError(
            "StabilitySelector alpha_rule_effective_ must be 'fixed', 'one_se', or 'best'"
        )
    if not isinstance(selector.coef_threshold, Real):
        raise ValueError(
            "StabilitySelector coef_threshold must be a finite non-negative value"
        )
    coef_threshold = float(selector.coef_threshold)
    if not math.isfinite(coef_threshold) or coef_threshold < 0.0:
        raise ValueError(
            "StabilitySelector coef_threshold must be a finite non-negative value"
        )
    n_bootstrap_requested = _strict_integer(
        selector.n_bootstrap,
        label="StabilitySelector n_bootstrap",
        minimum=1,
    )

    expected_mask = frequencies >= threshold
    if max_features is not None and expected_mask.sum() > max_features:
        top_indices = np.argsort(-frequencies, kind="mergesort")[:max_features]
        expected_mask = np.zeros(n_features, dtype=bool)
        expected_mask[top_indices] = True
    expected_indices = np.flatnonzero(expected_mask)
    expected_indices = expected_indices[
        np.argsort(-frequencies[expected_indices], kind="mergesort")
    ].tolist()
    if selected_indices != expected_indices:
        raise ValueError(
            "StabilitySelector selected_features_ is inconsistent with its "
            "threshold, max_features, and selection_frequencies_"
        )
    selected_mask = np.zeros(n_features, dtype=bool)
    selected_mask[selected_indices] = True

    # The fitted selector's ``output_order`` drives transform, get_support,
    # and get_feature_names_out.  The view must not silently disagree with it,
    # so features, indices, and path_rank all follow the same order.
    output_order = validate_output_order(selector.output_order)
    view_indices = [int(index) for index in ordered_indices(selected_indices, output_order)]
    view_selected = [raw_features[index] for index in view_indices]

    path_rank = pd.Series(
        pd.array([pd.NA] * n_features, dtype="Int64"),
    )
    for rank, index in enumerate(view_indices, start=1):
        path_rank.iloc[index] = rank
    table = pd.DataFrame(
        {
            "feature": raw_features,
            "selected_index": pd.array(range(n_features), dtype="Int64"),
            "path_rank": path_rank,
            "selected": selected_mask,
            "selection_frequency": frequencies.copy(),
            "mean_abs_coef": mean_abs_coef_values.copy(),
            "gain": mean_abs_coef_values.copy(),
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
            raise ValueError(
                "StabilitySelector _fit_feature_names_generated_ must be boolean"
            )
        if fit_input_kind not in {"dataframe", "positional"}:
            raise ValueError(
                "StabilitySelector _fit_input_kind_ must be 'dataframe' or 'positional'"
            )
        generated_names = bool(generated_names)
        input_kind = fit_input_kind
    generated_names = bool(generated_names)

    # Freeze only the fitted state used by transform.  This preserves sklearn's
    # set_output wrapping without retaining X, coefficient matrices, callbacks,
    # or a live reference whose behavior could change after refit.
    transform_selector = type(selector)()
    # Match the fitted selector's public dtype contract: sklearn requires
    # ``feature_names_in_`` to be a one-dimensional object ndarray.
    transform_selector.feature_names_in_ = raw_values.copy()
    transform_selector.n_features_in_ = n_features
    transform_selector.selected_features_ = np.asarray(
        selected_indices, dtype=np.int64
    )
    transform_selector.selected_feature_names_ = list(selected)
    transform_selector.output_order = output_order
    transform_selector._fit_feature_names_generated_ = generated_names
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

    coefs_available = hasattr(selector, "coef_bootstrap_")
    completed_bootstraps = None
    coef_shape = None
    if coefs_available:
        coef_matrix = np.asarray(selector.coef_bootstrap_)
        if (
            coef_matrix.ndim != 2
            or coef_matrix.shape[0] < 1
            or coef_matrix.shape[1] != n_features
            or not np.issubdtype(coef_matrix.dtype, np.number)
            or not np.isfinite(coef_matrix).all()
        ):
            raise ValueError(
                "StabilitySelector coef_bootstrap_ must be a finite numeric matrix "
                "with one column per fitted feature"
            )
        completed_bootstraps = int(coef_matrix.shape[0])
        coef_shape = tuple(int(value) for value in coef_matrix.shape)

    metadata = {
        "adapter": "StabilitySelector",
        "selector": "stability",
        "curve_available": False,
        "table_complete": True,
        "input_kind": input_kind,
        "raw_namespace": "fitted_candidate_features",
        "output_order": output_order,
        "task": selector.task,
        "threshold": threshold,
        "max_features": max_features,
        "selected_feature_count": selected_count,
        "alpha": alpha,
        "alpha_rule_effective": selector.alpha_rule_effective_,
        "coef_threshold": coef_threshold,
        "gain_source": "mean_abs_coef",
        "selection_frequency_source": "completed_bootstrap_fraction",
        "n_bootstrap_requested": n_bootstrap_requested,
        "coefs_available": coefs_available,
        "store_proxies": bool(getattr(selector, "store_proxies", False)),
    }
    diagnostics = {
        "fit_context": {
            "sample_weight": bool(
                getattr(selector, "_fit_used_sample_weight_", False)
            ),
            "groups": bool(getattr(selector, "_fit_used_groups_", False)),
            "time": bool(getattr(selector, "_fit_used_time_", False)),
            "smart_sampler": bool(selector.use_smart_sampler),
        },
        "sampled_n": getattr(selector, "sampled_n_", None),
        "coefs_available": coefs_available,
        "coef_bootstrap_shape": coef_shape,
        "completed_bootstraps": completed_bootstraps,
    }
    return SelectionView(
        features=view_selected,
        indices=view_indices,
        raw_features=raw_features,
        n_raw_features=n_features,
        raw_table=table,
        metadata=metadata,
        diagnostics=diagnostics,
        transformer=transform_selector.transform,
        proxy_correlations=proxy_correlations,
        resample_selections=resample_selections,
    )
