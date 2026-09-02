"""Exact compatibility contracts for :class:`sift.StabilitySelector`."""

import warnings

import numpy as np
import pandas as pd
import pytest

import sift


@pytest.fixture(scope="module")
def stability_contract_data():
    """Return a compact design where weights reverse the stable winner."""
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
    return X, y, sample_weight


STABILITY_FIXED_OPTIONS = {
    "n_bootstrap": 6,
    "alpha": 0.5,
    "n_jobs": 1,
    "random_state": 0,
    "verbose": False,
}

STABILITY_EXPLICIT_DEFAULTS = {
    "sample_frac": 0.5,
    "threshold": 0.6,
    "alpha_rule": "one_se",
    "l1_ratio": 1.0,
    "task": "regression",
    "max_features": None,
    "block_size": "auto",
    "block_method": "moving",
    "use_smart_sampler": False,
    "sampler_config": None,
    "store_coefs": True,
    "coef_threshold": 1e-8,
    "parallel_backend": "threads",
    "output_order": "legacy",
}


def _fit_stability(selector, X, y, sample_weight):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        returned = selector.fit(X, y, sample_weight=sample_weight)
    assert returned is selector
    assert [(item.category, str(item.message)) for item in caught] == []
    return selector


@pytest.mark.parametrize("input_kind", ("dataframe", "ndarray"))
@pytest.mark.parametrize("weighted", (False, True), ids=("unweighted", "weighted"))
def test_stability_default_explicit_weight_and_metadata_contract(
    stability_contract_data,
    input_kind,
    weighted,
):
    X_frame, y, discriminating_weight = stability_contract_data
    X = X_frame if input_kind == "dataframe" else X_frame.to_numpy()
    sample_weight = discriminating_weight if weighted else None

    implicit = _fit_stability(
        sift.StabilitySelector(**STABILITY_FIXED_OPTIONS),
        X.copy(),
        y.copy(),
        None if sample_weight is None else sample_weight.copy(),
    )
    explicit = _fit_stability(
        sift.StabilitySelector(
            **STABILITY_FIXED_OPTIONS,
            **STABILITY_EXPLICIT_DEFAULTS,
        ),
        X.copy(),
        y.copy(),
        None if sample_weight is None else sample_weight.copy(),
    )

    expected_index = 1 if weighted else 0
    all_names = (
        list(X_frame.columns)
        if input_kind == "dataframe"
        else [f"x{i}" for i in range(X_frame.shape[1])]
    )
    expected_names = [all_names[expected_index]]
    expected_frequencies = (
        np.array([0.0, 1.0, 1.0 / 6.0])
        if weighted
        else np.array([5.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0])
    )
    expected_info_order = (
        [all_names[1], all_names[2], all_names[0]]
        if weighted
        else all_names
    )

    for fitted in (implicit, explicit):
        assert isinstance(fitted.feature_names_in_, np.ndarray)
        assert fitted.feature_names_in_.dtype == object
        assert list(fitted.feature_names_in_) == all_names
        assert fitted.n_features_in_ == 3
        assert fitted.selected_feature_names_ == expected_names
        assert fitted.n_features_selected_ == 1
        assert fitted.alpha_ == 0.5
        assert fitted.alpha_rule_effective_ == "fixed"
        assert fitted._fit_used_sample_weight_ is weighted
        assert fitted._fit_used_groups_ is False
        assert fitted._fit_used_time_ is False
        assert fitted._fit_feature_names_generated_ is (input_kind == "ndarray")

        assert type(fitted.selected_features_) is np.ndarray
        assert fitted.selected_features_.dtype == np.int64
        np.testing.assert_array_equal(
            fitted.selected_features_, np.array([expected_index], dtype=np.int64)
        )
        assert type(fitted.selection_frequencies_) is np.ndarray
        assert fitted.selection_frequencies_.dtype == np.float64
        np.testing.assert_allclose(
            fitted.selection_frequencies_, expected_frequencies, rtol=0.0, atol=0.0
        )
        assert fitted.mean_abs_coef_.shape == (3,)
        assert fitted.mean_abs_coef_.dtype == np.float32
        assert fitted.coef_bootstrap_.shape == (6, 3)
        assert fitted.coef_bootstrap_.dtype == np.float32

        np.testing.assert_array_equal(
            fitted.get_support(), np.arange(3) == expected_index
        )
        np.testing.assert_array_equal(
            fitted.get_support(indices=True),
            np.array([expected_index], dtype=np.int64),
        )
        output_names = fitted.get_feature_names_out()
        assert type(output_names) is np.ndarray
        assert output_names.dtype == object
        assert output_names.tolist() == expected_names
        assert fitted.get_feature_names_out(all_names).tolist() == expected_names

        transformed = fitted.transform(X.copy())
        assert type(transformed) is np.ndarray
        assert transformed.shape == (len(y), 1)
        np.testing.assert_array_equal(
            transformed, X_frame.to_numpy()[:, [expected_index]]
        )

        feature_info = fitted.get_feature_info()
        assert type(feature_info) is pd.DataFrame
        assert list(feature_info.columns) == [
            "feature",
            "frequency",
            "mean_abs_coef",
            "selected",
        ]
        assert feature_info["feature"].tolist() == expected_info_order
        assert feature_info["selected"].sum() == 1
        assert feature_info.loc[feature_info["selected"], "feature"].tolist() == expected_names

    np.testing.assert_array_equal(
        implicit.selected_features_, explicit.selected_features_
    )
    np.testing.assert_array_equal(
        implicit.selection_frequencies_, explicit.selection_frequencies_
    )
    np.testing.assert_array_equal(implicit.coef_bootstrap_, explicit.coef_bootstrap_)
    np.testing.assert_array_equal(implicit.mean_abs_coef_, explicit.mean_abs_coef_)
