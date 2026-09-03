"""Fast, exact checks for the legacy public API spine."""

import inspect
from dataclasses import fields

import numpy as np
import pytest

import sift


LEGACY_EXPECTED_ALL = [
    "__version__",
    "FeatureCache",
    "build_cache",
    "KnockoffSelectionResult",
    "sample_knockoffs",
    "select_cached",
    "select_cefsplus",
    "select_cefsplus_binary",
    "select_fdr",
    "select_jmi",
    "select_jmim",
    "select_mrmr",
    "AutoKConfig",
    "FeaturePathEvaluationResult",
    "FilterSelectionResult",
    "evaluate_feature_path",
    "select_k_auto",
    "select_k_elbow",
    "select_k_penalized_objective",
    "select_k_posterior",
    "path_gain_pvalues",
    "select_k_changepoint",
    "select_k_chi2_stop",
    "select_k_forward_stop",
    "bootstrap_paths",
    "null_objective_paths",
    "select_k_perm_gap",
    "select_k_stability",
    "gaussian_cv_curves",
    "select_k_gaussian_cv",
    "select_k_xfit_objective",
    "select_k_knockoff_path",
    "xfit_objective_curves",
    "compute_objective_for_path",
    "BorutaSelector",
    "BorutaResult",
    "select_boruta",
    "select_boruta_shap",
    "MRMRSelector",
    "JMISelector",
    "JMIMSelector",
    "CEFSPlusSelector",
    "CEFSPlusBinarySelector",
    "KnockoffSelector",
    "permutation_importance",
    "SmartSamplerConfig",
    "smart_sample",
    "panel_config",
    "cross_section_config",
    "StabilitySelector",
    "stability_regression",
    "stability_classif",
    "catboost_select",
    "catboost_regression",
    "catboost_classif",
]
EXPECTED_ALL = [*LEGACY_EXPECTED_ALL, "set_verbosity", "SelectionView", "as_result"]


def _default(callable_, name):
    return inspect.signature(callable_).parameters[name].default


def test_version_and_ordered_public_exports():
    assert sift.__version__ == "0.9.1.dev0"
    assert sift.__all__ == EXPECTED_ALL
    assert sift.__all__[:55] == LEGACY_EXPECTED_ALL
    assert sift.__all__[55:] == ["set_verbosity", "SelectionView", "as_result"]
    assert len(sift.__all__) == 58


@pytest.mark.parametrize("name", EXPECTED_ALL)
def test_every_public_export_resolves(name):
    assert getattr(sift, name) is not None


def test_high_risk_function_defaults():
    assert _default(sift.select_cefsplus, "k") == 75
    assert _default(sift.permutation_importance, "n_repeats") == 10
    assert _default(sift.permutation_importance, "n_jobs") == -1
    assert _default(sift.permutation_importance, "random_state") is None
    assert _default(sift.permutation_importance, "return_result") is False
    assert _default(sift.select_cached, "method") == "cefsplus"
    assert _default(sift.select_cached, "corr_prune") == "auto"
    assert _default(sift.select_cached, "return_objective") is False
    assert _default(sift.select_cached, "return_indices") is False
    assert _default(sift.select_cached, "return_result") is False

    stability_defaults = inspect.signature(sift.StabilitySelector).parameters
    assert stability_defaults["n_bootstrap"].default == 50
    assert stability_defaults["threshold"].default == 0.6
    assert stability_defaults["n_jobs"].default == -1
    assert stability_defaults["random_state"].default is None
    assert stability_defaults["verbose"].default is True
    assert stability_defaults["penalty"].default is None

    defaults = {field.name: field.default for field in fields(sift.AutoKConfig)}
    assert {
        name: defaults[name]
        for name in ("k_method", "strategy", "metric", "min_k", "max_k", "random_state")
    } == {
        "k_method": "evaluate",
        "strategy": "time_holdout",
        "metric": "auto",
        "min_k": 5,
        "max_k": 100,
        "random_state": 42,
    }


METHODS = (
    ("mrmr", sift.select_mrmr, {"task": "regression", "estimator": "classic", "mrmr_backend": "serial"}, "y"),
    # The public classic JMI/JMIM implementation is the r2 estimator path.
    ("jmi", sift.select_jmi, {"task": "regression", "estimator": "r2"}, "y"),
    ("jmim", sift.select_jmim, {"task": "regression", "estimator": "r2"}, "y"),
    ("cefsplus", sift.select_cefsplus, {}, "y"),
    ("cefsplus_binary", sift.select_cefsplus_binary, {}, "y_binary"),
)


GOLDEN = {
    "mrmr": [0, 1],
    "jmi": [0, 2],
    "jmim": [0, 2],
    "cefsplus": [0, 2],
    "cefsplus_binary": [0, 2],
}

WEIGHTED_GOLDEN = {
    **GOLDEN,
    "cefsplus": [1, 2],
}


def _warnings_signature(records):
    return sorted((record.category.__name__, str(record.message)) for record in records)


@pytest.mark.parametrize("name,selector,kwargs,target_name", METHODS, ids=[m[0] for m in METHODS])
@pytest.mark.parametrize("input_kind", ("dataframe", "ndarray"))
@pytest.mark.parametrize("weighted", (False, True), ids=("uniform", "weighted"))
@pytest.mark.parametrize("return_result", (False, True), ids=("legacy", "result"))
def test_fixed_k_matrix_contract(
    contract_data, name, selector, kwargs, target_name, input_kind, weighted, return_result
):
    expected_indices = WEIGHTED_GOLDEN[name] if weighted else GOLDEN[name]
    expected_names = [
        (contract_data.X.columns if input_kind == "dataframe" else [f"x{i}" for i in range(4)])[i]
        for i in expected_indices
    ]

    def run_once():
        import warnings

        # Fresh copies make repeated calls independent even if a future route
        # normalizes inputs in place.
        X = contract_data.X.copy() if input_kind == "dataframe" else contract_data.X_array.copy()
        y = np.array(getattr(contract_data, target_name), copy=True)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            result = selector(
                X,
                y,
                k=2,
                verbose=False,
                return_result=return_result,
                sample_weight=(contract_data.sample_weight.copy() if weighted else None),
                **kwargs,
            )
        return result, _warnings_signature(records)

    first, first_warnings = run_once()
    second, second_warnings = run_once()
    assert first_warnings == second_warnings == []

    if not return_result:
        assert type(first) is list
        assert first == expected_names
        assert second == first
        return

    assert type(first) is sift.FilterSelectionResult
    assert first.selected_features == expected_names
    assert first.selected_indices == expected_indices
    assert second.selected_features == first.selected_features
    assert second.selected_indices == first.selected_indices
    ranking = first.ranking_
    assert ranking is not None
    expected_columns = [
        "feature",
        "rank",
        "selected",
        "selected_index",
        "relevance",
        "selector",
    ]
    if name == "cefsplus_binary":
        expected_columns.insert(-1, "score")
    assert list(ranking.columns) == expected_columns
    assert ranking["rank"].tolist() == [1, 2, 3, 4]
    assert ranking["feature"].iloc[:2].tolist() == expected_names
    assert ranking["selected"].tolist() == [True, True, False, False]
    assert ranking.loc[ranking["selected"], "selected_index"].tolist() == expected_indices
    assert first.selector_metadata["k_requested"] == 2
    assert first.selector_metadata["k"] == 2
    assert first.selector_metadata["auto_k"] is False


def test_exact_result_metadata_spine(contract_data):
    mrmr = sift.select_mrmr(
        contract_data.X,
        contract_data.y,
        k=2,
        task="regression",
        estimator="classic",
        mrmr_backend="serial",
        verbose=False,
        return_result=True,
    )
    cefsplus = sift.select_cefsplus(
        contract_data.X, contract_data.y, k=2, verbose=False, return_result=True
    )
    assert set(mrmr.selector_metadata) == {
        "selector", "k_requested", "k", "top_m", "n_features", "auto_k",
        "task", "estimator", "formula", "relevance",
    }
    assert set(cefsplus.selector_metadata) == {
        "selector", "k_requested", "k", "top_m", "n_features", "auto_k",
    }

    jmi = sift.select_jmi(
        contract_data.X.copy(), contract_data.y.copy(), k=2, task="regression",
        estimator="r2", verbose=False, return_result=True,
    )
    jmim = sift.select_jmim(
        contract_data.X.copy(), contract_data.y.copy(), k=2, task="regression",
        estimator="r2", verbose=False, return_result=True,
    )
    binary = sift.select_cefsplus_binary(
        contract_data.X.copy(), contract_data.y_binary.copy(), k=2,
        verbose=False, return_result=True,
    )
    expected_jmi = {
        "selector", "k_requested", "k", "top_m", "n_features", "auto_k",
        "task", "estimator", "relevance", "aggregation",
    }
    assert set(jmi.selector_metadata) == expected_jmi
    assert set(jmim.selector_metadata) == expected_jmi
    assert set(binary.selector_metadata) == {
        "selector", "k_requested", "k", "top_m", "n_features", "auto_k",
        "loss", "weighted", "class_weight", "class_weight_scope", "ridge",
        "refit_every", "corr_prune", "subsample", "random_state", "cat_encoding",
        "loo_smoothing", "loo_clip_min", "loo_clip_max", "target_mapping",
    }


def test_filter_omitted_defaults_equal_effective_current_defaults(contract_data):
    cases = (
        (
            sift.select_mrmr,
            {"task": "regression"},
            {"task": "regression", "estimator": "classic", "mrmr_backend": "auto", "subsample": 50_000, "random_state": 0},
            contract_data.y,
        ),
        (
            sift.select_jmi,
            {"task": "regression"},
            {"task": "regression", "estimator": "r2", "subsample": 50_000, "random_state": 0},
            contract_data.y,
        ),
        (
            sift.select_jmim,
            {"task": "regression"},
            {"task": "regression", "estimator": "r2", "subsample": 50_000, "random_state": 0},
            contract_data.y,
        ),
        (
            sift.select_cefsplus,
            {},
            {"subsample": 50_000, "random_state": 0},
            contract_data.y,
        ),
        (
            sift.select_cefsplus_binary,
            {},
            {"loss": "logloss", "ridge": 1e-4, "refit_every": 1, "subsample": None, "random_state": 0},
            contract_data.y_binary,
        ),
    )
    for selector, omitted, explicit, target in cases:
        left = selector(
            contract_data.X.copy(), np.array(target, copy=True), k=2, verbose=False, **omitted
        )
        right = selector(
            contract_data.X.copy(), np.array(target, copy=True), k=2, verbose=False, **explicit
        )
        assert type(left) is list and type(right) is list
        assert left == right


GAUSSIAN_CACHE_METHODS = (
    ("mrmr", sift.select_mrmr, {"task": "regression", "estimator": "gaussian"}),
    ("jmi", sift.select_jmi, {"task": "regression", "estimator": "gaussian"}),
    ("jmim", sift.select_jmim, {"task": "regression", "estimator": "gaussian"}),
    ("cefsplus", sift.select_cefsplus, {}),
)


@pytest.mark.parametrize(
    "name,selector,kwargs",
    GAUSSIAN_CACHE_METHODS,
    ids=[method[0] for method in GAUSSIAN_CACHE_METHODS],
)
def test_named_cache_gaussian_cells_are_stable(contract_data, name, selector, kwargs):
    import warnings

    def run_once():
        cache = sift.build_cache(contract_data.X.copy(), subsample=None)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            result = selector(
                contract_data.X.copy(),
                contract_data.y.copy(),
                k=2,
                cache=cache,
                verbose=False,
                return_result=True,
                **kwargs,
            )
        return result, _warnings_signature(records)

    first, first_warnings = run_once()
    second, second_warnings = run_once()
    assert type(first) is sift.FilterSelectionResult
    assert type(second) is sift.FilterSelectionResult
    assert first.selected_features == second.selected_features == ["signal", "weak"]
    assert first.selected_indices == second.selected_indices == [0, 2]
    assert list(first.ranking_.columns) == [
        "feature",
        "rank",
        "selected",
        "selected_index",
        "relevance",
        "selector",
    ]
    assert first.selector_metadata["k_requested"] == 2
    assert first.selector_metadata["k"] == 2
    assert first.selector_metadata["auto_k"] is False
    assert first_warnings == second_warnings
    # This small n=96 mRMR case provides a deterministic warning-category/count
    # contract; broader warning-message coverage remains a later B0 expansion.
    if name == "mrmr":
        assert [category for category, _ in first_warnings] == ["UserWarning"]
    else:
        assert first_warnings == []
