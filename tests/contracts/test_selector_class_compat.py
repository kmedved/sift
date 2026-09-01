"""Compatibility contracts for sklearn-style selector classes."""

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
import pytest

import sift


@dataclass(frozen=True)
class SelectorContractData:
    X: pd.DataFrame
    y: np.ndarray
    sample_weight: np.ndarray


@pytest.fixture(scope="module")
def selector_regression_data() -> SelectorContractData:
    """Favor ``major`` by row count and ``minor`` by total sample weight."""
    n_major, n_minor = 12, 6
    major_target = np.linspace(-2.0, 2.0, n_major)
    minor_target = np.linspace(-2.0, 2.0, n_minor)
    y = np.concatenate((major_target, minor_target))
    X = pd.DataFrame(
        {
            "major": np.concatenate(
                (
                    major_target + 0.03 * np.sin(np.arange(n_major)),
                    np.sin(2.3 * np.arange(n_minor)),
                )
            ),
            "minor": np.concatenate(
                (
                    np.cos(1.7 * np.arange(n_major)),
                    minor_target + 0.03 * np.cos(np.arange(n_minor)),
                )
            ),
            "noise": np.sin(0.91 * np.arange(n_major + n_minor)),
        }
    )
    sample_weight = np.concatenate((np.ones(n_major), np.full(n_minor, 8.0)))
    return SelectorContractData(X=X, y=y, sample_weight=sample_weight)


@pytest.fixture(scope="module")
def selector_binary_data() -> SelectorContractData:
    """Binary counterpart whose weighted and unweighted winners differ."""
    n_major, n_minor = 12, 6
    n_samples = n_major + n_minor
    y = np.resize(np.array([0, 1], dtype=np.int64), n_samples)
    X = pd.DataFrame(
        {
            "major": np.concatenate(
                (
                    y[:n_major] + 0.08 * np.sin(1.7 * np.arange(n_major)),
                    0.5 + 0.7 * np.sin(2.1 * np.arange(n_minor)),
                )
            ),
            "minor": np.concatenate(
                (
                    0.5 + 0.7 * np.cos(1.3 * np.arange(n_major)),
                    y[n_major:] + 0.08 * np.cos(1.7 * np.arange(n_minor)),
                )
            ),
            "noise": np.sin(0.91 * np.arange(n_samples)),
        }
    )
    sample_weight = np.concatenate((np.ones(n_major), np.full(n_minor, 12.0)))
    return SelectorContractData(X=X, y=y, sample_weight=sample_weight)


FILTER_SELECTOR_ROUTES = (
    (
        "mrmr",
        sift.MRMRSelector,
        {
            "task": "regression",
            "relevance": "f",
            "estimator": "classic",
            "formula": "quotient",
            "top_m": None,
            "cat_features": None,
            "cat_encoding": "none",
            "allow_full_data_target_encoding": False,
            "subsample": 50_000,
            "random_state": 0,
            "n_jobs": 1,
            "mrmr_backend": "auto",
            "cache": None,
            "auto_k_config": None,
            "output_order": "legacy",
        },
        "regression",
    ),
    (
        "jmi",
        sift.JMISelector,
        {
            "task": "regression",
            "estimator": "r2",
            "relevance": "f",
            "top_m": None,
            "cat_features": None,
            "cat_encoding": "none",
            "allow_full_data_target_encoding": False,
            "subsample": 50_000,
            "random_state": 0,
            "cache": None,
            "auto_k_config": None,
            "output_order": "legacy",
        },
        "regression",
    ),
    (
        "jmim",
        sift.JMIMSelector,
        {
            "task": "regression",
            "estimator": "r2",
            "relevance": "f",
            "top_m": None,
            "cat_features": None,
            "cat_encoding": "none",
            "allow_full_data_target_encoding": False,
            "subsample": 50_000,
            "random_state": 0,
            "cache": None,
            "auto_k_config": None,
            "output_order": "legacy",
        },
        "regression",
    ),
    (
        "cefsplus",
        sift.CEFSPlusSelector,
        {
            "top_m": None,
            "corr_prune": None,
            "cat_features": None,
            "cat_encoding": "none",
            "allow_full_data_target_encoding": False,
            "subsample": 50_000,
            "random_state": 0,
            "cache": None,
            "auto_k_config": None,
            "output_order": "legacy",
        },
        "regression",
    ),
    (
        "cefsplus_binary",
        sift.CEFSPlusBinarySelector,
        {
            "loss": "logloss",
            "top_m": None,
            "corr_prune": None,
            "class_weight": None,
            "ridge": 1e-4,
            "refit_every": 1,
            "cat_features": None,
            "cat_encoding": "none",
            "loo_smoothing": 20.0,
            "loo_clip_min": 1e-4,
            "loo_clip_max": 1.0 - 1e-4,
            "allow_full_data_target_encoding": False,
            "subsample": None,
            "random_state": 0,
            "auto_k_config": None,
            "output_order": "legacy",
        },
        "binary",
    ),
)


def _fit_without_warnings(selector, X, y, sample_weight):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        returned = selector.fit(X, y, sample_weight=sample_weight)
    assert returned is selector
    assert [(item.category, str(item.message)) for item in caught] == []
    return selector


@pytest.mark.parametrize(
    "name,selector_cls,explicit_defaults,target_kind",
    FILTER_SELECTOR_ROUTES,
    ids=[route[0] for route in FILTER_SELECTOR_ROUTES],
)
@pytest.mark.parametrize("input_kind", ("dataframe", "ndarray"))
@pytest.mark.parametrize("weighted", (False, True), ids=("unweighted", "weighted"))
def test_filter_selector_default_explicit_and_fitted_contract(
    selector_regression_data,
    selector_binary_data,
    name,
    selector_cls,
    explicit_defaults,
    target_kind,
    input_kind,
    weighted,
):
    """Pin exact names, order, shapes, warnings, and weight propagation."""
    del name
    data = selector_binary_data if target_kind == "binary" else selector_regression_data
    X_source = data.X if input_kind == "dataframe" else data.X.to_numpy()
    weights = data.sample_weight if weighted else None

    implicit = _fit_without_warnings(
        selector_cls(k=1, verbose=False),
        X_source.copy(),
        data.y.copy(),
        None if weights is None else weights.copy(),
    )
    explicit = _fit_without_warnings(
        selector_cls(k=1, verbose=False, **explicit_defaults),
        X_source.copy(),
        data.y.copy(),
        None if weights is None else weights.copy(),
    )

    expected_index = 1 if weighted else 0
    all_names = (
        list(data.X.columns)
        if input_kind == "dataframe"
        else [f"x{i}" for i in range(data.X.shape[1])]
    )
    expected_names = [all_names[expected_index]]

    for fitted in (implicit, explicit):
        assert fitted.feature_names_in_ == all_names
        assert fitted.n_features_in_ == 3
        assert fitted.selected_features_ == expected_names
        assert type(fitted.selected_indices_) is np.ndarray
        assert fitted.selected_indices_.dtype == np.int64
        np.testing.assert_array_equal(
            fitted.selected_indices_, np.array([expected_index], dtype=np.int64)
        )
        np.testing.assert_array_equal(
            fitted.get_support(),
            np.arange(3) == expected_index,
        )
        np.testing.assert_array_equal(
            fitted.get_support(indices=True),
            np.array([expected_index], dtype=np.int64),
        )
        output_names = fitted.get_feature_names_out()
        assert type(output_names) is np.ndarray
        assert output_names.dtype == object
        assert output_names.tolist() == expected_names

        transformed = fitted.transform(X_source.copy())
        if input_kind == "dataframe":
            assert type(transformed) is pd.DataFrame
            assert transformed.index.equals(data.X.index)
            assert transformed.columns.tolist() == expected_names
            np.testing.assert_array_equal(
                transformed.to_numpy(), data.X.iloc[:, [expected_index]].to_numpy()
            )
        else:
            assert type(transformed) is np.ndarray
            assert transformed.shape == (len(data.y), 1)
            np.testing.assert_array_equal(
                transformed, data.X.to_numpy()[:, [expected_index]]
            )

    assert implicit.selected_features_ == explicit.selected_features_
    np.testing.assert_array_equal(implicit.selected_indices_, explicit.selected_indices_)
