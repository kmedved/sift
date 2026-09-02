"""Weighted contracts for five representative fixed-k filter cells.

The broader contract suite supplies the shared core fixture; these tests add two
deterministic regime-shift fixtures and exercise the five representative
function/backend cells across DataFrame/ndarray and legacy/result forms.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

import sift


@dataclass(frozen=True)
class WeightedContractData:
    """A regime-shift fixture whose preferred feature is weight-dependent."""

    X: pd.DataFrame
    y: np.ndarray
    sample_weight: np.ndarray


@pytest.fixture(scope="module")
def weighted_regression_data() -> WeightedContractData:
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
    return WeightedContractData(X=X, y=y, sample_weight=sample_weight)


@pytest.fixture(scope="module")
def weighted_binary_data() -> WeightedContractData:
    """Binary counterpart of the deterministic regime-shift fixture."""
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
    return WeightedContractData(X=X, y=y, sample_weight=sample_weight)


ROUTES = (
    (
        "mrmr",
        sift.select_mrmr,
        {
            "task": "regression",
            "estimator": "classic",
            "mrmr_backend": "serial",
        },
        "regression",
    ),
    (
        "jmi",
        sift.select_jmi,
        {"task": "regression", "estimator": "r2"},
        "regression",
    ),
    (
        "jmim",
        sift.select_jmim,
        {"task": "regression", "estimator": "r2"},
        "regression",
    ),
    ("cefsplus", sift.select_cefsplus, {}, "regression"),
    ("cefsplus_binary", sift.select_cefsplus_binary, {}, "binary"),
)


@pytest.mark.parametrize(
    "name,selector,kwargs,target_kind", ROUTES, ids=[r[0] for r in ROUTES]
)
@pytest.mark.parametrize("input_kind", ("dataframe", "ndarray"))
@pytest.mark.parametrize("return_result", (False, True), ids=("legacy", "result"))
def test_fixed_k_sample_weight_changes_selected_feature(
    weighted_regression_data,
    weighted_binary_data,
    name,
    selector,
    kwargs,
    target_kind,
    input_kind,
    return_result,
):
    """Weights must change the winner, not merely be accepted by the API."""
    del name
    data = weighted_binary_data if target_kind == "binary" else weighted_regression_data
    feature_names = ["major", "minor"] if input_kind == "dataframe" else ["x0", "x1"]

    def select(sample_weight):
        X = data.X.copy() if input_kind == "dataframe" else data.X.to_numpy(copy=True)
        result = selector(
            X,
            data.y.copy(),
            k=1,
            sample_weight=(
                None if sample_weight is None else np.array(sample_weight, copy=True)
            ),
            subsample=None,
            random_state=0,
            verbose=False,
            return_result=return_result,
            **kwargs,
        )
        if not return_result:
            assert type(result) is list
            return result, None
        assert type(result) is sift.FilterSelectionResult
        return result.selected_features, result.selected_indices

    unweighted, unweighted_indices = select(None)
    uniform, uniform_indices = select(np.ones_like(data.sample_weight))
    weighted, weighted_indices = select(data.sample_weight)

    assert unweighted == uniform == [feature_names[0]]
    assert weighted == [feature_names[1]]
    if return_result:
        assert unweighted_indices == uniform_indices == [0]
        assert weighted_indices == [1]
    else:
        assert unweighted_indices is uniform_indices is weighted_indices is None
