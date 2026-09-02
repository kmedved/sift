"""Compatibility cells for public auto-k calls with split metadata."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
import pytest

import sift


@dataclass(frozen=True)
class AutoKRoute:
    name: str
    selector: object
    kwargs: dict
    target_name: str
    metric: str


ROUTES = (
    AutoKRoute(
        "mrmr",
        sift.select_mrmr,
        {"task": "regression", "estimator": "classic", "mrmr_backend": "serial"},
        "y",
        "rmse",
    ),
    AutoKRoute(
        "jmi",
        sift.select_jmi,
        {"task": "regression", "estimator": "r2"},
        "y",
        "rmse",
    ),
    AutoKRoute(
        "jmim",
        sift.select_jmim,
        {"task": "regression", "estimator": "r2"},
        "y",
        "rmse",
    ),
    AutoKRoute("cefsplus", sift.select_cefsplus, {}, "y", "rmse"),
    AutoKRoute(
        "cefsplus_binary",
        sift.select_cefsplus_binary,
        {},
        "y_binary",
        "logloss",
    ),
)


EXPECTED_INDICES = {
    ("time_holdout", "mrmr", False): [0],
    ("time_holdout", "mrmr", True): [0],
    ("time_holdout", "jmi", False): [0, 2, 1],
    ("time_holdout", "jmi", True): [0, 2, 1],
    ("time_holdout", "jmim", False): [0, 2, 1],
    ("time_holdout", "jmim", True): [0, 2, 1],
    ("time_holdout", "cefsplus", False): [0, 2, 1],
    ("time_holdout", "cefsplus", True): [1, 2, 0],
    ("time_holdout", "cefsplus_binary", False): [0, 2, 1],
    ("time_holdout", "cefsplus_binary", True): [0, 2, 1],
    ("group_cv", "mrmr", False): [0, 1, 3],
    ("group_cv", "mrmr", True): [0, 1, 3],
    ("group_cv", "jmi", False): [0, 2, 1],
    ("group_cv", "jmi", True): [0, 2, 1],
    ("group_cv", "jmim", False): [0, 2, 1],
    ("group_cv", "jmim", True): [0, 2, 1],
    ("group_cv", "cefsplus", False): [0, 2, 1],
    ("group_cv", "cefsplus", True): [1, 2, 0],
    ("group_cv", "cefsplus_binary", False): [0, 2, 1],
    ("group_cv", "cefsplus_binary", True): [0, 2, 1],
}


COMMON_AUTO_METADATA = {
    "selector",
    "k_requested",
    "k",
    "top_m",
    "n_features",
    "auto_k",
    "auto_k_mode",
    "k_method",
    "auto_k_strategy",
    "selection_rule",
}
EXPECTED_METADATA_KEYS = {
    "mrmr": COMMON_AUTO_METADATA
    | {"task", "estimator", "formula", "relevance"},
    "jmi": COMMON_AUTO_METADATA
    | {"task", "estimator", "relevance", "aggregation"},
    "jmim": COMMON_AUTO_METADATA
    | {"task", "estimator", "relevance", "aggregation"},
    "cefsplus": COMMON_AUTO_METADATA,
    "cefsplus_binary": COMMON_AUTO_METADATA
    | {
        "loss",
        "weighted",
        "class_weight",
        "class_weight_scope",
        "ridge",
        "refit_every",
        "corr_prune",
        "subsample",
        "random_state",
        "cat_encoding",
        "loo_smoothing",
        "loo_clip_min",
        "loo_clip_max",
        "target_mapping",
        "objective_penalty",
        "binary_objective_mode",
    },
}


BINARY_DIAGNOSTIC_KEYS = {
    "path_scores",
    "candidate_indices",
    "valid_indices",
    "univariate_scores",
    "dropped_features",
    "numerical_failures",
    "invalid_conditional_information",
    "n_valid_features",
    "n_screened_features",
    "n_gram_blocks",
    "n_logistic_refits",
    "n_constant_or_nonfinite",
    "n_corr_pruned",
    "n_outside_top_m",
    "subsample_row_idx",
    "cat_features_requested",
    "cat_features_used",
    "auto_k",
    "auto_k_diagnostics",
    "auto_k_curve",
}


AUTO_DIAGNOSTIC_COLUMNS = [
    "k",
    "score",
    "score_mean",
    "score_std",
    "score_se",
    "n_splits",
    "n_finite",
    "split_scores",
    "best_k",
    "best_score",
    "within_tolerance",
    "in_selected_plateau",
    "selection_rule",
    "selection_rule_effective",
    "one_se_unavailable",
    "selected",
]


def _context(contract_data, strategy):
    if strategy == "time_holdout":
        return {"time": np.arange(len(contract_data.y), dtype=np.int64)}
    return {
        "groups": np.repeat(
            np.arange(len(contract_data.y) // 12, dtype=np.int64),
            12,
        )
    }


def _call(route, contract_data, strategy, input_kind, weighted, *, return_result):
    X = (
        contract_data.X.copy()
        if input_kind == "dataframe"
        else contract_data.X_array.copy()
    )
    target = np.array(getattr(contract_data, route.target_name), copy=True)
    config = sift.AutoKConfig(
        k_method="evaluate",
        strategy=strategy,
        metric=route.metric,
        min_k=1,
        max_k=3,
        val_frac=0.25,
        n_splits=3,
        random_state=42,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = route.selector(
            X,
            target,
            k="auto",
            auto_k_config=config,
            sample_weight=(
                contract_data.sample_weight.copy() if weighted else None
            ),
            subsample=None,
            verbose=False,
            return_result=return_result,
            **_context(contract_data, strategy),
            **route.kwargs,
        )
    return result, [(item.category, str(item.message)) for item in caught]


@pytest.mark.parametrize("route", ROUTES, ids=lambda route: route.name)
@pytest.mark.parametrize("strategy", ("time_holdout", "group_cv"))
@pytest.mark.parametrize("input_kind", ("dataframe", "ndarray"))
@pytest.mark.parametrize("weighted", (False, True), ids=("unweighted", "weighted"))
def test_auto_k_split_context_matrix(
    contract_data,
    route,
    strategy,
    input_kind,
    weighted,
):
    """Pin ordered outputs, result shape, warnings, and auto-k metadata."""
    result, caught = _call(
        route,
        contract_data,
        strategy,
        input_kind,
        weighted,
        return_result=True,
    )
    expected_indices = EXPECTED_INDICES[(strategy, route.name, weighted)]
    names = (
        contract_data.X.columns.tolist()
        if input_kind == "dataframe"
        else [f"x{i}" for i in range(contract_data.X.shape[1])]
    )
    expected_features = [names[index] for index in expected_indices]

    assert caught == []
    assert type(result) is sift.FilterSelectionResult
    assert result.selected_indices == expected_indices
    assert result.selected_features == expected_features
    assert set(result.selector_metadata) == EXPECTED_METADATA_KEYS[route.name]
    assert result.selector_metadata["selector"] == route.name
    assert result.selector_metadata["k_requested"] == "auto"
    assert result.selector_metadata["k"] == len(expected_indices)
    assert result.selector_metadata["n_features"] == 4
    assert result.selector_metadata["auto_k"] is True
    assert result.selector_metadata["auto_k_mode"] == "prefix_only"
    assert result.selector_metadata["k_method"] == "evaluate"
    assert result.selector_metadata["auto_k_strategy"] == strategy
    assert result.selector_metadata["selection_rule"] == "best"

    ranking = result.get_feature_ranking()
    assert ranking["feature"].iloc[: len(expected_features)].tolist() == expected_features
    assert ranking.loc[ranking["selected"], "selected_index"].tolist() == expected_indices

    # Stage 1.4/R2: auto-k producers retain the complete ranking and the
    # normalized curve payload they already computed.
    if route.name in {"mrmr", "jmi", "jmim"}:
        assert list(result.ranking_.columns) == [
            "feature",
            "rank",
            "selected",
            "selected_index",
            "relevance",
            "selector",
        ]
        assert set(result.diagnostics_) == {
            "path_relevance",
            "auto_k",
            "auto_k_diagnostics",
            "auto_k_curve",
        }
    elif route.name == "cefsplus":
        assert list(result.ranking_.columns) == [
            "feature",
            "rank",
            "selected",
            "selected_index",
            "relevance",
            "selector",
        ]
        assert set(result.diagnostics_) == {
            "auto_k",
            "auto_k_diagnostics",
            "auto_k_curve",
        }
    else:
        assert result.selector_metadata["weighted"] is weighted
        assert list(result.ranking_.columns) == [
            "feature",
            "rank",
            "selected",
            "selected_index",
            "relevance",
            "score",
            "selector",
        ]
        assert set(result.diagnostics_) == BINARY_DIAGNOSTIC_KEYS

    if result.diagnostics_ is not None:
        summary = result.diagnostics_["auto_k"]
        diagnostics = result.diagnostics_["auto_k_diagnostics"]
        assert summary["method"] == "evaluate"
        assert summary["selection_rule"] == "best"
        assert summary["selected_k"] == len(expected_indices)
        assert summary["effective_min_k"] == 1
        assert summary["effective_max_k"] == 3
        assert list(diagnostics.columns) == AUTO_DIAGNOSTIC_COLUMNS
        assert diagnostics.loc[diagnostics["selected"], "k"].tolist() == [
            len(expected_indices)
        ]

        curve_payload = result.diagnostics_["auto_k_curve"]
        assert curve_payload["available"] is True
        assert curve_payload["route"] == "evaluate"
        assert curve_payload["criterion"] == "score"
        assert curve_payload["criterion_direction"] == "higher_is_better"
        curve = curve_payload["curve"]
        assert list(curve.columns) == ["k", "criterion", "criterion_se", "selected"]
        assert curve.loc[curve["selected"], "k"].tolist() == [len(expected_indices)]
        assert curve["k"].tolist() == diagnostics["k"].tolist()
        assert curve["criterion"].tolist() == diagnostics["score"].tolist()


@pytest.mark.parametrize("route", ROUTES, ids=lambda route: route.name)
def test_auto_k_legacy_list_matches_result(contract_data, route):
    legacy, legacy_warnings = _call(
        route,
        contract_data,
        "time_holdout",
        "dataframe",
        False,
        return_result=False,
    )
    result, result_warnings = _call(
        route,
        contract_data,
        "time_holdout",
        "dataframe",
        False,
        return_result=True,
    )

    assert legacy_warnings == result_warnings == []
    assert type(legacy) is list
    assert legacy == result.selected_features


def _assert_results_equivalent(left, right):
    assert left.selected_features == right.selected_features
    assert left.selected_indices == right.selected_indices
    assert left.selector_metadata == right.selector_metadata
    if left.ranking_ is None or right.ranking_ is None:
        assert left.ranking_ is right.ranking_ is None
    else:
        pd.testing.assert_frame_equal(left.ranking_, right.ranking_)


@pytest.mark.parametrize("route", ROUTES, ids=lambda route: route.name)
@pytest.mark.parametrize("strategy", ("time_holdout", "group_cv"))
def test_auto_k_omitted_config_matches_explicit_effective_default(
    contract_data,
    route,
    strategy,
):
    """Omission inference and explicit current defaults stay equivalent."""
    context = _context(contract_data, strategy)
    explicit = (
        sift.AutoKConfig(k_method="auto")
        if route.name in {"cefsplus", "cefsplus_binary"}
        else sift.AutoKConfig(strategy=strategy)
    )

    results = []
    warning_signatures = []
    for config in (None, explicit):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = route.selector(
                contract_data.X.copy(),
                np.array(getattr(contract_data, route.target_name), copy=True),
                k="auto",
                auto_k_config=config,
                subsample=None,
                verbose=False,
                return_result=True,
                **context,
                **route.kwargs,
            )
        results.append(result)
        warning_signatures.append(
            [(item.category, str(item.message)) for item in caught]
        )

    assert warning_signatures == [[], []]
    _assert_results_equivalent(*results)


@pytest.mark.parametrize("route", ROUTES, ids=lambda route: route.name)
def test_fixed_k_rejects_auto_k_split_metadata(contract_data, route):
    with pytest.raises(
        ValueError,
        match="groups and time are only meaningful for auto-k evaluation",
    ):
        route.selector(
            contract_data.X.copy(),
            np.array(getattr(contract_data, route.target_name), copy=True),
            k=2,
            groups=np.zeros(len(contract_data.y), dtype=np.int64),
            time=np.arange(len(contract_data.y), dtype=np.int64),
            verbose=False,
            **route.kwargs,
        )


@pytest.mark.parametrize("route", ROUTES, ids=lambda route: route.name)
@pytest.mark.parametrize(
    ("strategy", "message"),
    (
        ("time_holdout", "requires time parameter"),
        ("group_cv", "requires groups parameter"),
    ),
)
def test_evaluate_auto_k_rejects_missing_required_context(
    contract_data,
    route,
    strategy,
    message,
):
    config = sift.AutoKConfig(
        k_method="evaluate",
        strategy=strategy,
        min_k=1,
        max_k=3,
    )
    with pytest.raises(ValueError, match=message):
        route.selector(
            contract_data.X.copy(),
            np.array(getattr(contract_data, route.target_name), copy=True),
            k="auto",
            auto_k_config=config,
            verbose=False,
            **route.kwargs,
        )


@pytest.mark.parametrize(
    ("field", "message"),
    (("groups", "groups has 95 elements"), ("time", "time has 95 elements")),
)
def test_auto_k_rejects_misaligned_split_metadata(contract_data, field, message):
    with pytest.raises(ValueError, match=message):
        sift.select_cefsplus(
            contract_data.X.copy(),
            contract_data.y.copy(),
            k="auto",
            auto_k_config=sift.AutoKConfig(k_method="elbow", min_k=1, max_k=3),
            verbose=False,
            **{field: np.arange(len(contract_data.y) - 1)},
        )
