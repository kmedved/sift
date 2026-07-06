import numpy as np
import pandas as pd
import pytest

from sift import (
    build_cache,
    select_cached,
    select_cefsplus,
    select_jmi,
    select_jmim,
    select_mrmr,
)
from sift.selection.auto_k import compute_objective_for_path
from sift.selection.cefsplus import cefsplus_loop, cefsplus_loop_with_objective
from sift.selection.objective import objective_from_corr_path


def test_select_cefsplus_regression():
    rng = np.random.default_rng(42)
    n, p = 500, 20
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = X["f0"] + 0.5 * X["f1"] + rng.normal(size=n) * 0.3

    selected = select_cefsplus(X, y, k=5, verbose=False)
    assert len(selected) == 5
    assert "f0" in selected


def test_select_cached_with_cache():
    rng = np.random.default_rng(42)
    n, p = 500, 20
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])

    cache = build_cache(X, subsample=400)

    y1 = X["f0"] + rng.normal(size=n) * 0.3
    y2 = X["f5"] + rng.normal(size=n) * 0.3

    sel1 = select_cached(cache, y1, k=5)
    sel2 = select_cached(cache, y2, k=5)

    assert len(sel1) == 5
    assert len(sel2) == 5
    assert "f0" in sel1
    assert "f5" in sel2


def test_build_cache_handles_nonfinite():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(200, 10)))
    X.iloc[0, 0] = np.nan
    X.iloc[1, 1] = np.inf
    X.iloc[2, 2] = -np.inf

    cache = build_cache(X, subsample=None)
    assert np.isfinite(cache.Z).all()


def test_build_cache_rejects_zero_weight_subsample():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(20, 4)))
    sample_weight = np.zeros(20)
    sample_weight[0] = 1.0

    with pytest.raises(ValueError, match="zero total weight"):
        build_cache(X, sample_weight=sample_weight, subsample=5, random_state=0)


def test_select_cached_rejects_y_length_mismatch():
    rng = np.random.default_rng(42)
    n, p = 50, 6
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = X["f0"].to_numpy()
    cache = build_cache(X, subsample=None)

    with pytest.raises(ValueError, match="cache was built from 50 rows"):
        select_cached(cache, y[:-1], k=3)

    with pytest.raises(ValueError, match="cache was built from 50 rows"):
        select_cached(cache, np.r_[y, 0.0], k=3)


def test_compute_objective_for_path_rejects_y_length_mismatch():
    rng = np.random.default_rng(42)
    n, p = 50, 6
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = X["f0"].to_numpy()
    cache = build_cache(X, subsample=None)

    with pytest.raises(ValueError, match="cache was built from 50 rows"):
        compute_objective_for_path(cache, y[:-1], ["f0", "f1"])


def _duplicate_signal_cache():
    rng = np.random.default_rng(123)
    n = 80
    signal = rng.normal(size=n)
    X = pd.DataFrame({f"s{i}": signal for i in range(6)})
    y = signal + rng.normal(size=n) * 0.01
    return X, y, build_cache(X, subsample=None, compute_Rxx=True)


@pytest.mark.parametrize("method", ["jmi", "jmim", "mrmr_quot", "mrmr_diff"])
def test_cached_gaussian_methods_do_not_corr_prune_by_default(method):
    _, y, cache = _duplicate_signal_cache()

    selected_default = select_cached(cache, y, k=4, method=method, top_m=6)
    selected_none = select_cached(cache, y, k=4, method=method, top_m=6, corr_prune=None)

    assert len(selected_default) == 4
    assert len(selected_none) == 4


def test_cached_gaussian_methods_can_opt_into_corr_prune():
    _, y, cache = _duplicate_signal_cache()

    selected = select_cached(cache, y, k=4, method="jmi", top_m=6, corr_prune=0.95)

    assert len(selected) == 1


def test_select_cached_cefsplus_auto_corr_prune_remains_default():
    _, y, cache = _duplicate_signal_cache()

    selected = select_cached(cache, y, k=4, method="cefsplus", top_m=6)

    assert len(selected) == 1


@pytest.mark.parametrize(
    ("selector", "kwargs"),
    [
        (select_mrmr, {"estimator": "gaussian"}),
        (select_jmi, {"estimator": "gaussian"}),
        (select_jmim, {"estimator": "gaussian"}),
    ],
)
def test_public_gaussian_selectors_with_cache_return_k_without_corr_prune(selector, kwargs):
    X, y, cache = _duplicate_signal_cache()

    selected = selector(
        X,
        y,
        k=4,
        task="regression",
        cache=cache,
        top_m=6,
        verbose=False,
        **kwargs,
    )

    assert len(selected) == 4


@pytest.mark.parametrize("k", [0, 1, 4, 10])
def test_cefsplus_loop_and_objective_loop_select_same_path(k):
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(80, 5))
    y = 1.5 * X[:, 0] - 0.5 * X[:, 2] + rng.normal(scale=0.2, size=80)
    R = np.corrcoef(X, rowvar=False)
    r = np.corrcoef(np.column_stack([X, y]), rowvar=False)[:-1, -1]
    tie_break_rel = np.abs(r)

    selected = cefsplus_loop(R, r, k, tie_break_rel)
    selected_with_objective, objective = cefsplus_loop_with_objective(
        R,
        r,
        k,
        tie_break_rel,
    )

    np.testing.assert_array_equal(selected, selected_with_objective)
    expected_objective = objective_from_corr_path(
        R[np.ix_(selected, selected)],
        r[selected],
    )
    np.testing.assert_allclose(objective, expected_objective)
