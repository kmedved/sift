import inspect

import numpy as np
import pandas as pd
import pytest

import sift.selection.filter_api as filter_api_module
import sift.selection.knockoff_filter as knockoff_filter_module
import sift.selection.loops as loops_module
import sift.stability as stability_module
from sift import (
    BorutaSelector,
    build_cache,
    select_boruta,
    select_boruta_shap,
    select_cefsplus,
    select_cefsplus_binary,
    select_fdr,
    select_jmi,
    select_jmim,
    select_mrmr,
)
from sift.stability import StabilitySelector
from sift.selection.loops import jmi_select


@pytest.mark.parametrize(
    ("selector", "kwargs"),
    [
        (select_mrmr, {"estimator": "classic"}),
        (select_mrmr, {"estimator": "gaussian"}),
        (select_jmi, {"estimator": "r2"}),
        (select_jmim, {"estimator": "r2"}),
    ],
)
def test_filter_selectors_reject_invalid_task(selector, kwargs):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 5))
    y = X[:, 0] + 0.1 * rng.normal(size=len(X))

    with pytest.raises(ValueError, match="task must be"):
        selector(X, y, k=2, task="regresion", verbose=False, **kwargs)


def test_classification_rejects_continuous_target():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 4))
    y = np.linspace(0.0, 1.0, len(X))

    with pytest.raises(ValueError, match="discrete binary or multiclass"):
        select_mrmr(X, y, k=2, task="classification", verbose=False)


@pytest.mark.parametrize(
    "selector",
    [
        lambda X, y, cache, w: select_cefsplus(
            X, y, k=2, cache=cache, sample_weight=w, verbose=False
        ),
        lambda X, y, cache, w: select_mrmr(
            X,
            y,
            k=2,
            task="regression",
            estimator="gaussian",
            cache=cache,
            sample_weight=w,
            verbose=False,
        ),
    ],
)
def test_gaussian_cache_rejects_call_time_sample_weight(selector):
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(100, 5)))
    y = X[0].to_numpy() + 0.1 * rng.normal(size=len(X))
    cache = build_cache(X, subsample=None)
    weights = np.linspace(0.5, 1.5, len(X))

    with pytest.raises(ValueError, match="fixed by the supplied cache"):
        selector(X, y, cache, weights)


@pytest.mark.parametrize(
    ("selector", "kwargs"),
    [
        (select_mrmr, {"estimator": "classic"}),
        (select_jmi, {"estimator": "r2"}),
        (select_jmim, {"estimator": "r2"}),
    ],
)
def test_classic_regression_selection_is_target_offset_invariant(selector, kwargs):
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(300, 5)), columns=[f"f{i}" for i in range(5)])
    y = X["f0"].to_numpy() + 0.05 * rng.normal(size=len(X))

    baseline = selector(X, y, k=2, task="regression", verbose=False, **kwargs)
    shifted = selector(X, y + 1e8, k=2, task="regression", verbose=False, **kwargs)

    assert baseline == shifted
    assert baseline[0] == "f0"


def test_stability_regression_is_target_offset_invariant():
    rng = np.random.default_rng(4)
    X = pd.DataFrame(rng.normal(size=(240, 5)), columns=[f"f{i}" for i in range(5)])
    y = X["f0"].to_numpy() + 0.05 * rng.normal(size=len(X))
    options = dict(
        n_bootstrap=8,
        sample_frac=0.7,
        threshold=0.6,
        alpha=0.1,
        n_jobs=1,
        random_state=5,
        verbose=False,
        store_coefs=False,
    )

    baseline = StabilitySelector(**options).fit(X, y).selected_feature_names_
    shifted = StabilitySelector(**options).fit(X, y + 1e8).selected_feature_names_

    assert baseline == shifted
    assert "f0" in baseline


def test_selectors_are_invariant_to_tiny_feature_units():
    rng = np.random.default_rng(6)
    n = 400
    signal = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "signal": signal,
            "noise1": rng.normal(size=n),
            "noise2": rng.normal(size=n),
        }
    )
    X_tiny = X.copy()
    X_tiny["signal"] *= 1e-13
    y = signal + 0.2 * rng.normal(size=n)
    y_binary = (signal + 0.2 * rng.normal(size=n) > 0.0).astype(np.int32)

    classic_selectors = (
        (select_mrmr, {"estimator": "classic"}),
        (select_jmi, {"estimator": "r2"}),
        (select_jmim, {"estimator": "r2"}),
    )
    for selector, kwargs in classic_selectors:
        baseline = selector(
            X, y, k=1, task="regression", subsample=None, verbose=False, **kwargs
        )
        tiny = selector(
            X_tiny,
            y,
            k=1,
            task="regression",
            subsample=None,
            verbose=False,
            **kwargs,
        )
        assert baseline == tiny == ["signal"]

    assert select_cefsplus(X, y, k=1, subsample=None, verbose=False) == ["signal"]
    assert select_cefsplus(X_tiny, y, k=1, subsample=None, verbose=False) == [
        "signal"
    ]
    assert select_cefsplus_binary(
        X, y_binary, k=1, subsample=None, verbose=False
    ) == ["signal"]
    assert select_cefsplus_binary(
        X_tiny, y_binary, k=1, subsample=None, verbose=False
    ) == ["signal"]


def test_perfect_class_separator_does_not_truncate_mrmr_path():
    rng = np.random.default_rng(8)
    y = np.repeat([0, 1], 80)
    X = pd.DataFrame(
        {
            "perfect": y.astype(float),
            "signal": y + 0.2 * rng.normal(size=len(y)),
            "noise": rng.normal(size=len(y)),
        }
    )

    selected = select_mrmr(
        X,
        y,
        k=2,
        task="classification",
        estimator="classic",
        subsample=None,
        verbose=False,
    )

    assert selected[0] == "perfect"
    assert len(selected) == 2


def test_zero_weight_rows_do_not_change_seeded_knockoffs():
    rng = np.random.default_rng(7)
    n, p = 160, 8
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = 1.5 * X["f0"].to_numpy() - X["f1"].to_numpy() + rng.normal(size=n)

    zero_X = pd.DataFrame(
        rng.normal(size=(19, p)),
        columns=X.columns,
    )
    X_augmented = pd.concat([zero_X.iloc[:8], X, zero_X.iloc[8:]], ignore_index=True)
    y_augmented = np.concatenate([rng.normal(size=8), y, rng.normal(size=11)])
    weights = np.concatenate([np.zeros(8), np.ones(n), np.zeros(11)])

    base = select_fdr(
        X,
        y,
        q=0.5,
        offset=0,
        statistic="ridge",
        sample_weight=np.ones(n),
        subsample=None,
        random_state=17,
        verbose=False,
    )
    augmented = select_fdr(
        X_augmented,
        y_augmented,
        q=0.5,
        offset=0,
        statistic="ridge",
        sample_weight=weights,
        subsample=None,
        random_state=17,
        verbose=False,
    )

    assert base.selected_features == augmented.selected_features
    np.testing.assert_array_equal(base.W["W"], augmented.W["W"])
    assert augmented.selector_metadata["n_rows_used"] == n


def test_function_categorical_defaults_are_safe_and_truthful():
    for selector in (
        select_mrmr,
        select_jmi,
        select_jmim,
        select_cefsplus,
        select_cefsplus_binary,
        select_boruta,
        select_boruta_shap,
    ):
        assert inspect.signature(selector).parameters["cat_encoding"].default == "none"
    assert inspect.signature(BorutaSelector).parameters["cat_encoding"].default == "none"

    X = pd.DataFrame(
        {
            "category": pd.Series(["a", "b"] * 40, dtype="category"),
            "numeric": np.arange(80, dtype=float),
        }
    )
    y = np.arange(80, dtype=float)
    with pytest.raises(ValueError, match="Non-numeric columns") as exc_info:
        select_mrmr(X, y, k=1, task="regression", verbose=False)
    assert "leakage-prone" not in str(exc_info.value)


def test_routed_auto_k_metadata_omits_unused_strategy_and_selection_rule():
    rng = np.random.default_rng(9)
    X = pd.DataFrame(rng.normal(size=(140, 8)), columns=[f"f{i}" for i in range(8)])
    y = X["f0"].to_numpy() + 0.3 * rng.normal(size=len(X))

    result = select_cefsplus(X, y, k="auto", return_result=True, verbose=False)

    assert result.selector_metadata["k_method"] == "auto"
    assert "auto_k_strategy" not in result.selector_metadata
    assert "selection_rule" not in result.selector_metadata
    assert result.diagnostics_["auto_k"]["routed_method"] == "penalized_objective"


def test_stability_threshold_tuning_is_target_scale_invariant():
    rng = np.random.default_rng(10)
    X = pd.DataFrame(rng.normal(size=(120, 5)), columns=[f"f{i}" for i in range(5)])
    y = (
        1.2 * X["f0"].to_numpy()
        - 0.5 * X["f1"].to_numpy()
        + rng.normal(scale=0.5, size=len(X))
    )
    options = dict(
        n_bootstrap=6,
        sample_frac=0.7,
        threshold=0.5,
        alpha=None,
        n_jobs=1,
        random_state=11,
        verbose=False,
        store_coefs=False,
    )
    baseline = StabilitySelector(**options).fit(X, y)
    scaled = StabilitySelector(**options).fit(X, y * 100.0)

    best, results = baseline.tune_threshold(
        X, y, thresholds=(0.4, 0.6, 0.8), cv=3
    )
    best_scaled, results_scaled = scaled.tune_threshold(
        X, y * 100.0, thresholds=(0.4, 0.6, 0.8), cv=3
    )

    assert best == best_scaled
    np.testing.assert_allclose(
        results[["mean_score", "n_features"]],
        results_scaled[["mean_score", "n_features"]],
        rtol=1e-10,
        atol=1e-12,
    )


class _RecordingThreadLimit:
    def __init__(self, calls, *, limits):
        self.calls = calls
        self.limits = limits

    def __enter__(self):
        self.calls.append(self.limits)

    def __exit__(self, exc_type, exc, traceback):
        return False


def _record_thread_limits(monkeypatch, module):
    calls = []

    def factory(*, limits):
        return _RecordingThreadLimit(calls, limits=limits)

    monkeypatch.setattr(module, "threadpool_limits", factory)
    return calls


def test_binary_cefsplus_limits_native_thread_pools(monkeypatch):
    calls = _record_thread_limits(monkeypatch, filter_api_module)
    rng = np.random.default_rng(12)
    X = rng.normal(size=(80, 5))
    y = (X[:, 0] + 0.2 * rng.normal(size=len(X)) > 0.0).astype(int)

    select_cefsplus_binary(X, y, k=2, verbose=False)

    assert calls == [1]


def test_ridge_knockoffs_limit_native_thread_pools(monkeypatch):
    calls = _record_thread_limits(monkeypatch, knockoff_filter_module)
    rng = np.random.default_rng(13)
    X = rng.normal(size=(90, 6))
    y = X[:, 0] + 0.2 * rng.normal(size=len(X))

    select_fdr(
        X,
        y,
        q=0.5,
        offset=0,
        statistic="RIDGE",
        screen_pairs=None,
        subsample=None,
        verbose=False,
    )

    assert calls == [1]


def test_stability_selection_limits_native_thread_pools(monkeypatch):
    calls = _record_thread_limits(monkeypatch, stability_module)
    rng = np.random.default_rng(14)
    X = rng.normal(size=(90, 6))
    y = X[:, 0] + 0.2 * rng.normal(size=len(X))

    StabilitySelector(
        n_bootstrap=4,
        alpha=0.1,
        n_jobs=1,
        random_state=15,
        verbose=False,
        store_coefs=False,
    ).fit(X, y)

    assert calls == [1]


def test_r2_jmi_limits_native_thread_pools(monkeypatch):
    calls = _record_thread_limits(monkeypatch, loops_module)
    rng = np.random.default_rng(16)
    X = rng.normal(size=(100, 8))
    y = X[:, 0] + 0.2 * rng.normal(size=len(X))
    relevance = np.linspace(1.0, 0.1, X.shape[1])

    jmi_select(X, y, 3, relevance, mi_estimator="r2")

    assert calls == [1]


def test_cefsplus_default_keeps_suppressor_pair_eligible():
    rng = np.random.default_rng(200)
    n, p = 1500, 5
    rho = 0.99
    x0 = rng.normal(size=n)
    x1 = rho * x0 + np.sqrt(1.0 - rho**2) * rng.normal(size=n)
    X = pd.DataFrame(
        rng.normal(size=(n, p)),
        columns=[f"f{i}" for i in range(p)],
    )
    X["f0"] = x0
    X["f1"] = x1
    y = (x1 - x0) / np.sqrt(2.0 * (1.0 - rho))
    y += 0.1 * rng.normal(size=n)

    default = select_cefsplus(X, y, k=2, subsample=None, verbose=False)
    pruned = select_cefsplus(
        X,
        y,
        k=2,
        corr_prune=0.95,
        subsample=None,
        verbose=False,
    )

    assert set(default) == {"f0", "f1"}
    assert len({"f0", "f1"} & set(pruned)) == 1


def test_cefsplus_knockoff_default_adapts_beyond_ten_discoveries():
    rng = np.random.default_rng(100)
    n, p = 2200, 120
    rho = 0.4
    innovations = rng.normal(size=(n, p))
    X = np.empty_like(innovations)
    X[:, 0] = innovations[:, 0]
    for j in range(1, p):
        X[:, j] = rho * X[:, j - 1] + np.sqrt(1.0 - rho**2) * innovations[:, j]
    support = rng.choice(p, size=24, replace=False)
    beta = np.zeros(p)
    beta[support] = 7.5 * rng.choice([-1.0, 1.0], size=len(support)) / np.sqrt(n)
    y = X @ beta + rng.normal(size=n)

    result = select_fdr(
        X,
        y,
        q=0.1,
        statistic="cefsplus",
        screen_pairs=None,
        subsample=None,
        random_state=100,
        verbose=False,
    )

    assert len(result.selected_features) > 10
    assert result.selector_metadata["path_depth_initial"] == 20
    assert result.selector_metadata["path_depth"] > 20
    assert result.selector_metadata["path_depth_adaptive"]
    assert not result.selector_metadata["path_depth_saturated"]
