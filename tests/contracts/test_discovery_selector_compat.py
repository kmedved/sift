"""Compatibility contracts for Boruta and knockoff selector estimators."""

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

import sift


@dataclass(frozen=True)
class DiscoveryContractData:
    X: pd.DataFrame
    y: np.ndarray
    sample_weight: np.ndarray


class WeightedCorrelationRegressor(BaseEstimator):
    """Tiny deterministic native-importance estimator used by Boruta."""

    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        weights = (
            np.ones(len(y_arr), dtype=float)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float)
        )
        weights = weights / weights.sum()
        X_centered = X_arr - (weights[:, None] * X_arr).sum(axis=0)
        y_centered = y_arr - (weights * y_arr).sum()
        numerator = (weights[:, None] * X_centered * y_centered[:, None]).sum(axis=0)
        denominator = np.sqrt(
            (weights[:, None] * X_centered**2).sum(axis=0)
            * (weights * y_centered**2).sum()
        )
        self.feature_importances_ = np.divide(
            np.abs(numerator),
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        )
        return self


@pytest.fixture(scope="module")
def boruta_contract_data() -> DiscoveryContractData:
    n_major, n_minor = 16, 8
    major = np.concatenate(
        (np.linspace(-3.0, 3.0, n_major), np.linspace(0.2, 0.8, n_minor))
    )
    minor = np.concatenate(
        (np.linspace(-0.5, 0.5, n_major), np.linspace(-4.0, 4.0, n_minor))
    )
    regime = np.concatenate((np.zeros(n_major), np.ones(n_minor)))
    X = pd.DataFrame(
        {
            "major": major,
            "minor": minor,
            "regime": regime,
            "noise": np.sin(np.arange(n_major + n_minor)),
        }
    )
    y = np.where(regime < 0.5, major, minor)
    sample_weight = np.concatenate((np.ones(n_major), np.full(n_minor, 8.0)))
    return DiscoveryContractData(X=X, y=y, sample_weight=sample_weight)


@pytest.fixture(scope="module")
def knockoff_contract_data() -> DiscoveryContractData:
    rng = np.random.default_rng(10)
    n_samples, n_features = 55, 6
    X_arr = rng.normal(size=(n_samples, n_features))
    X_arr[:, 1] = 0.45 * X_arr[:, 0] + rng.normal(scale=0.9, size=n_samples)
    y = 1.8 * X_arr[:, 0] - 1.2 * X_arr[:, 2] + rng.normal(
        scale=0.6, size=n_samples
    )
    X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(n_features)])
    return DiscoveryContractData(
        X=X,
        y=y,
        sample_weight=np.linspace(0.5, 1.5, n_samples),
    )


def _fit_without_warnings(selector, X, y, sample_weight):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        returned = selector.fit(X, y, sample_weight=sample_weight)
    assert returned is selector
    assert [(item.category, str(item.message)) for item in caught] == []
    return selector


BORUTA_EXPLICIT_DEFAULTS = {
    "n_estimators": "auto",
    "task": "regression",
    "importance": "native",
    "alpha": 0.05,
    "perc": 100,
    "resolve_tentative": True,
    "max_features": None,
    "shadow_method": "global",
    "shadow_mode": "columns",
    "block_size": "auto",
    "cat_features": None,
    "cat_encoding": "none",
    "allow_full_data_target_encoding": False,
    "importance_data": "train",
    "test_size": 0.3,
    "shap_sample_size": 2000,
    "early_stop_rounds": 5,
    "output_order": "legacy",
}


@pytest.mark.parametrize("input_kind", ("dataframe", "ndarray"))
@pytest.mark.parametrize("weighted", (False, True), ids=("unweighted", "weighted"))
def test_boruta_default_explicit_result_method_and_transform_contract(
    boruta_contract_data,
    input_kind,
    weighted,
):
    data = boruta_contract_data
    X = data.X if input_kind == "dataframe" else data.X.to_numpy()
    sample_weight = data.sample_weight if weighted else None
    common = {
        "estimator": WeightedCorrelationRegressor(),
        "max_iter": 4,
        "random_state": 7,
        "verbose": False,
    }

    implicit = _fit_without_warnings(
        sift.BorutaSelector(**common),
        X.copy(),
        data.y.copy(),
        None if sample_weight is None else sample_weight.copy(),
    )
    explicit = _fit_without_warnings(
        sift.BorutaSelector(**common, **BORUTA_EXPLICIT_DEFAULTS),
        X.copy(),
        data.y.copy(),
        None if sample_weight is None else sample_weight.copy(),
    )

    all_names = (
        list(data.X.columns)
        if input_kind == "dataframe"
        else [f"x{i}" for i in range(data.X.shape[1])]
    )
    expected_indices = [1] if weighted else [0, 1]
    expected_names = [all_names[index] for index in expected_indices]
    expected_status = (
        np.array([-1, 1, -1, -1], dtype=np.int8)
        if weighted
        else np.array([1, 1, -1, -1], dtype=np.int8)
    )
    expected_hits = (
        np.array([2, 4, 0, 0], dtype=np.int32)
        if weighted
        else np.array([4, 4, 0, 0], dtype=np.int32)
    )

    for fitted in (implicit, explicit):
        assert isinstance(fitted.feature_names_in_, np.ndarray)
        assert fitted.feature_names_in_.dtype == object
        assert list(fitted.feature_names_in_) == all_names
        assert fitted.n_features_in_ == 4
        assert fitted.selected_features_ == expected_names
        assert fitted.n_iter_ == 4
        np.testing.assert_array_equal(fitted.status_, expected_status)
        np.testing.assert_array_equal(fitted.hits_, expected_hits)
        assert fitted.status_.dtype == np.int8
        assert fitted.hits_.dtype == np.int32
        assert fitted.shadow_thresholds_.shape == (4,)
        assert fitted.mean_importance_.shape == (4,)
        np.testing.assert_array_equal(
            fitted.get_support(), expected_status == 1
        )
        np.testing.assert_array_equal(
            fitted.get_support(indices=True),
            np.array(expected_indices, dtype=np.int64),
        )
        assert fitted.get_feature_names_out().tolist() == expected_names

        transformed = fitted.transform(X.copy())
        if input_kind == "dataframe":
            assert type(transformed) is pd.DataFrame
            assert transformed.columns.tolist() == expected_names
            np.testing.assert_array_equal(
                transformed.to_numpy(), data.X.iloc[:, expected_indices].to_numpy()
            )
        else:
            assert type(transformed) is np.ndarray
            assert transformed.shape == (len(data.y), len(expected_indices))
            np.testing.assert_array_equal(
                transformed, data.X.to_numpy()[:, expected_indices]
            )

        # This is intentionally a method, unlike KnockoffSelector.result_.
        assert callable(fitted.result_)
        result = fitted.result_()
        assert type(result) is sift.BorutaResult
        assert result.feature_names == all_names
        assert result.selected_features() == expected_names
        np.testing.assert_array_equal(result.status, expected_status)
        np.testing.assert_array_equal(result.hits, expected_hits)
        assert result.n_iter == 4
        assert list(result.get_feature_ranking().columns) == [
            "feature",
            "mean_importance",
            "hits",
            "status",
        ]

    np.testing.assert_array_equal(implicit.status_, explicit.status_)
    np.testing.assert_array_equal(implicit.hits_, explicit.hits_)
    np.testing.assert_array_equal(
        implicit.shadow_thresholds_, explicit.shadow_thresholds_
    )
    np.testing.assert_array_equal(implicit.mean_importance_, explicit.mean_importance_)


KNOCKOFF_EXPLICIT_DEFAULTS = {
    "statistic": "relevance",
    "n_draws": 1,
    "eta": 0.5,
    "s_method": "equi",
    "min_eig": 1e-3,
    "screen_pairs": 2000,
    "statistic_options": None,
    "feature_groups": None,
    "group_corr_threshold": 0.7,
    "cat_features": None,
    "cat_encoding": "none",
    "allow_full_data_target_encoding": False,
    "loo_smoothing": 20.0,
    "loo_clip_min": 1e-4,
    "loo_clip_max": 1.0 - 1e-4,
    "subsample": 50_000,
    "n_jobs": 1,
    "cache": None,
    "output_order": "legacy",
}

KNOCKOFF_METADATA_KEYS = {
    "selector",
    "n_features",
    "q",
    "offset",
    "statistic",
    "s_method",
    "n_draws",
    "eta",
    "screen_pairs",
    "path_depth_requested",
    "path_depth",
    "path_depth_initial",
    "path_depth_adaptive",
    "gamma",
    "lambda_min",
    "s_mean",
    "s_median",
    "n_low_power_features",
    "random_state",
    "n_rows_used",
    "fdr_control",
    "per_draw_fdr_control",
    "q_scope",
    "aggregation",
    "aggregation_threshold",
    "aggregation_fdr_control",
    "aggregation_preserves_per_draw_fdr",
    "validity_model",
    "weighted_model",
    "n_zero_weight_variance_features",
    "feature_groups",
    "n_feature_groups",
    "group_mode",
    "group_fdr_control",
}


@pytest.mark.parametrize("input_kind", ("dataframe", "ndarray"))
@pytest.mark.parametrize("weighted", (False, True), ids=("unweighted", "weighted"))
def test_knockoff_default_explicit_result_metadata_and_transform_contract(
    knockoff_contract_data,
    input_kind,
    weighted,
):
    data = knockoff_contract_data
    X = data.X if input_kind == "dataframe" else data.X.to_numpy()
    sample_weight = data.sample_weight if weighted else None
    common = {"q": 0.4, "offset": 0, "random_state": 12, "verbose": False}

    implicit = _fit_without_warnings(
        sift.KnockoffSelector(**common),
        X.copy(),
        data.y.copy(),
        None if sample_weight is None else sample_weight.copy(),
    )
    explicit = _fit_without_warnings(
        sift.KnockoffSelector(**common, **KNOCKOFF_EXPLICIT_DEFAULTS),
        X.copy(),
        data.y.copy(),
        None if sample_weight is None else sample_weight.copy(),
    )

    all_names = (
        list(data.X.columns)
        if input_kind == "dataframe"
        else [f"x{i}" for i in range(data.X.shape[1])]
    )
    expected_indices = [0, 2, 1, 3] if weighted else [0, 2, 1, 3, 5]
    expected_names = [all_names[index] for index in expected_indices]
    expected_ranking = [all_names[index] for index in [0, 2, 1, 3, 5, 4]]

    for fitted in (implicit, explicit):
        assert isinstance(fitted.feature_names_in_, np.ndarray)
        assert fitted.feature_names_in_.dtype == object
        assert list(fitted.feature_names_in_) == all_names
        assert fitted.n_features_in_ == 6
        assert fitted.selected_features_ == expected_names
        np.testing.assert_array_equal(
            fitted.selected_indices_, np.array(expected_indices, dtype=np.int64)
        )
        np.testing.assert_array_equal(
            fitted.get_support(),
            np.isin(np.arange(6), expected_indices),
        )
        np.testing.assert_array_equal(
            fitted.get_support(indices=True),
            np.array(expected_indices, dtype=np.int64),
        )
        assert fitted.get_feature_names_out().tolist() == expected_names

        transformed = fitted.transform(X.copy())
        if input_kind == "dataframe":
            assert type(transformed) is pd.DataFrame
            assert transformed.columns.tolist() == expected_names
            np.testing.assert_array_equal(
                transformed.to_numpy(), data.X.iloc[:, expected_indices].to_numpy()
            )
        else:
            assert type(transformed) is np.ndarray
            assert transformed.shape == (len(data.y), len(expected_indices))
            np.testing.assert_array_equal(
                transformed, data.X.to_numpy()[:, expected_indices]
            )

        assert type(fitted.result_) is sift.KnockoffSelectionResult
        result = fitted.result_
        assert result.selected_features == expected_names
        assert result.selected_indices == expected_indices
        assert set(result.selector_metadata) == KNOCKOFF_METADATA_KEYS
        assert result.selector_metadata["weighted_model"] is weighted
        assert result.selector_metadata["n_rows_used"] == len(data.y)
        assert result.selector_metadata["aggregation"] == "single_draw"
        assert list(result.W.columns) == [
            "feature",
            "selected_index",
            "W",
            "selected",
            "selection_frequency",
            "relevance",
            "selector",
            "W_draw_0",
        ]
        assert result.W["feature"].tolist() == all_names
        assert result.W["selected"].sum() == len(expected_indices)
        ranking = result.get_feature_ranking()
        assert list(ranking.columns) == [
            "feature",
            "W",
            "rank",
            "selected",
            "selection_frequency",
            "selected_index",
            "relevance",
            "selector",
        ]
        assert ranking["feature"].tolist() == expected_ranking
        assert ranking.loc[ranking["selected"], "feature"].tolist() == expected_names

    np.testing.assert_array_equal(implicit.selected_indices_, explicit.selected_indices_)
    pd.testing.assert_frame_equal(implicit.result_.W, explicit.result_.W)
    assert implicit.result_.selector_metadata == explicit.result_.selector_metadata
