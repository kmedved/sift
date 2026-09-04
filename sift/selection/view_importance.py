"""Permutation-importance adapter for additive SelectionView construction."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from sift.selection.view import (
    SelectionView,
    _coerce_feature_names,
    _coerce_indices,
    _labels_equal,
    _numeric_vector,
    _strict_integer,
)


def _as_importance_result(result: Any, input_features: Any) -> SelectionView:
    snapshot = result._adapter_snapshot()
    ranking = snapshot["ranking"]
    if not isinstance(ranking, pd.DataFrame):
        raise ValueError("ImportanceResult ranking must be a pandas DataFrame")
    expected_columns = [
        "feature",
        "importance_mean",
        "importance_std",
        "baseline_score",
    ]
    if list(ranking.columns) != expected_columns:
        raise ValueError(
            "ImportanceResult ranking must have exactly the legacy columns "
            f"{expected_columns}"
        )

    feature_names = _coerce_feature_names(snapshot["feature_names"])
    if feature_names is None:
        raise ValueError("ImportanceResult feature_names must be an ordered iterable")
    n_features = len(feature_names)
    if len(ranking) != n_features:
        raise ValueError(
            "ImportanceResult ranking rows must match the original feature width"
        )
    ranking_indices = _coerce_indices(
        snapshot["ranking_indices"],
        label="ImportanceResult ranking_indices",
    )
    if ranking_indices is None or set(ranking_indices) != set(range(n_features)):
        raise ValueError(
            "ImportanceResult ranking_indices must contain every raw feature position"
        )
    for row, position in enumerate(ranking_indices):
        if not _labels_equal(ranking["feature"].iloc[row], feature_names[position]):
            raise ValueError(
                "ImportanceResult ranking feature identities do not match ranking_indices"
            )

    metadata = snapshot["selector_metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError("ImportanceResult selector_metadata must be a mapping")
    metadata = copy.deepcopy(dict(metadata))
    if metadata.get("selector") != "permutation_importance":
        raise ValueError(
            "ImportanceResult selector_metadata must identify permutation_importance"
        )
    metadata_n_features = _strict_integer(
        metadata.get("n_features"),
        label="ImportanceResult selector_metadata n_features",
        minimum=0,
    )
    if metadata_n_features != n_features:
        raise ValueError(
            "ImportanceResult selector_metadata n_features must match feature_names"
        )
    n_repeats = _strict_integer(
        metadata.get("n_repeats"),
        label="ImportanceResult selector_metadata n_repeats",
        minimum=1,
    )
    input_kind = metadata.get("input_kind")
    if input_kind not in {"dataframe", "positional"}:
        raise ValueError(
            "ImportanceResult selector_metadata input_kind must be 'dataframe' or "
            "'positional'"
        )
    if metadata.get("selection_semantics") != "ranking_only":
        raise ValueError(
            "ImportanceResult selector_metadata selection_semantics must be "
            "'ranking_only'"
        )

    try:
        raw_importances = np.asarray(snapshot["importances"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "ImportanceResult importances_ must be a numeric matrix"
        ) from exc
    if raw_importances.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(
            "ImportanceResult importances_ must contain real non-boolean numeric values"
        )
    importances = raw_importances.astype(np.float64, copy=True)
    if importances.ndim != 2 or importances.shape != (n_features, n_repeats):
        raise ValueError(
            "ImportanceResult importances_ shape must be "
            "(n_features, n_repeats)"
        )

    ranking_means = _numeric_vector(
        ranking["importance_mean"],
        label="ImportanceResult importance_mean",
        length=n_features,
    )
    ranking_stds = _numeric_vector(
        ranking["importance_std"],
        label="ImportanceResult importance_std",
        length=n_features,
    )
    ranking_baselines = _numeric_vector(
        ranking["baseline_score"],
        label="ImportanceResult baseline_score",
        length=n_features,
    )
    baseline_value = snapshot["baseline_score"]
    if isinstance(baseline_value, (bool, np.bool_)) or not isinstance(
        baseline_value,
        (Real, np.integer, np.floating),
    ):
        raise ValueError("ImportanceResult baseline_score must be a real number")
    baseline_score = float(baseline_value)
    if not np.allclose(
        ranking_baselines,
        baseline_score,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    ):
        raise ValueError(
            "ImportanceResult ranking baseline_score must match baseline_score"
        )
    raw_means = np.mean(importances, axis=1)
    raw_stds = np.std(importances, axis=1)
    if not np.allclose(
        ranking_means,
        raw_means[ranking_indices],
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    ) or not np.allclose(
        ranking_stds,
        raw_stds[ranking_indices],
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    ):
        raise ValueError(
            "ImportanceResult ranking mean/std must match importances_ using ddof=0"
        )

    supplied_features = _coerce_feature_names(input_features)
    if input_kind == "dataframe":
        if input_features is not None and not result._matches_original_features(
            input_features
        ):
            raise ValueError(
                "input_features must match ImportanceResult DataFrame columns in exact order"
            )
        raw_features = feature_names
    else:
        if any(
            not _labels_equal(feature, position)
            for position, feature in enumerate(feature_names)
        ):
            raise ValueError(
                "positional ImportanceResult feature_names must equal raw positions"
            )
        if supplied_features is not None and len(supplied_features) != n_features:
            raise ValueError(
                "input_features length must match the positional ImportanceResult width"
            )
        raw_features = feature_names if supplied_features is None else supplied_features

    selected = [raw_features[position] for position in ranking_indices]
    rank_by_position = np.empty(n_features, dtype=np.int64)
    for rank, position in enumerate(ranking_indices, start=1):
        rank_by_position[position] = rank
    table = pd.DataFrame(
        {
            "feature": raw_features,
            "selected_index": pd.array(range(n_features), dtype="Int64"),
            "path_rank": pd.array(rank_by_position, dtype="Int64"),
            "selected": np.ones(n_features, dtype=bool),
            "gain": raw_means.copy(),
            "importance_mean": raw_means.copy(),
            "importance_std": raw_stds.copy(),
            "baseline_score": baseline_score,
        }
    )
    result_diagnostics = snapshot["diagnostics"]
    if not isinstance(result_diagnostics, Mapping):
        raise ValueError("ImportanceResult diagnostics_ must be a mapping")
    diagnostics = copy.deepcopy(dict(result_diagnostics))
    diagnostics["permutation_importance_repeats"] = importances.copy()
    metadata.update(
        {
            "adapter": "ImportanceResult",
            "curve_available": False,
            "gain_source": "permutation_importance_mean",
            "table_complete": True,
            "input_kind": input_kind,
        }
    )
    return SelectionView(
        features=selected,
        indices=ranking_indices,
        raw_features=raw_features,
        n_raw_features=n_features,
        raw_table=table,
        metadata=metadata,
        diagnostics=diagnostics,
    )
