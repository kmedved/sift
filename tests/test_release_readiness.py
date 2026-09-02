import inspect
import warnings

import numpy as np
import pandas as pd
import pytest

import sift.selection.filter_api as filter_api_module
import sift.selection.knockoff_filter as knockoff_filter_module
import sift.selection.loops as loops_module
import sift.stability as stability_module
from sift import (
    BorutaSelector,
    KnockoffSelector,
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


def _fold_marker_frame(seed: int, n: int = 600):
    """Unique-ID fixture reproducing the pre-0.9 target_cv fold-marker leak."""
    rng = np.random.default_rng(seed)
    cities = np.array([f"city_{i}" for i in range(8)], dtype=object)
    city = rng.choice(cities, size=n)
    effects = dict(zip(cities.tolist(), rng.normal(size=8).tolist()))
    x1 = rng.normal(size=n)
    y = x1 + np.array([effects[value] for value in city]) + rng.normal(
        scale=0.3, size=n
    )
    X = pd.DataFrame(
        {
            "id": [f"id_{i}" for i in range(n)],
            "city": pd.Series(city, dtype=object),
            "x1": x1,
            "x_noise": rng.normal(size=n),
        }
    )
    return X, y


def test_target_cv_is_leakage_safe_for_unique_identifier_columns():
    """Release gate for the §1.1 centering claim across the whole surface.

    Before centering, the unique ``id`` column carried each row's complement
    folds' prior and entered mRMR's top three in every seed of this design.
    """
    for seed in range(8):
        X, y = _fold_marker_frame(seed)
        for selector, kwargs in (
            (select_mrmr, {"task": "regression", "estimator": "classic"}),
            (select_jmi, {"task": "regression", "estimator": "r2"}),
            (select_jmim, {"task": "regression", "estimator": "r2"}),
        ):
            selected = selector(
                X,
                y,
                3,
                cat_encoding="target_cv",
                subsample=None,
                verbose=False,
                **kwargs,
            )
            assert "id" not in selected, f"{selector.__name__} seed={seed}"

    X, y = _fold_marker_frame(0)
    selector = BorutaSelector(
        n_estimators=20,
        max_iter=2,
        cat_encoding="target_cv",
        verbose=False,
    ).fit(X, y)
    assert "id" not in selector.selected_features_


def test_target_cv_rejects_the_full_data_escape_hatch_across_entry_points():
    """Release gate for C3: the contradictory combination is never ignored."""
    X, y = _fold_marker_frame(1, n=120)
    y_binary = (y > np.median(y)).astype(np.int64)
    message = "cannot be combined with allow_full_data_target_encoding=True"

    with pytest.raises(ValueError, match=message):
        select_mrmr(
            X,
            y,
            1,
            task="regression",
            cat_encoding="target_cv",
            allow_full_data_target_encoding=True,
            verbose=False,
        )
    with pytest.raises(ValueError, match=message):
        select_cefsplus_binary(
            X,
            y_binary,
            1,
            cat_encoding="target_cv",
            allow_full_data_target_encoding=True,
            verbose=False,
        )
    with pytest.raises(ValueError, match=message):
        select_boruta(
            X,
            y,
            n_estimators=10,
            max_iter=2,
            cat_encoding="target_cv",
            allow_full_data_target_encoding=True,
            verbose=False,
        )


def test_knockoff_fdr_claims_stay_honest_under_categorical_encoding():
    """Release gate for C4: no silent Model-X claim on target-derived inputs."""
    rng = np.random.default_rng(4242)
    n = 240
    signal = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "team": pd.Series(
                np.resize(np.array(["a", "b", "c", "d"], dtype=object), n),
                dtype=object,
            ),
            "signal": signal,
            "noise": rng.normal(size=n),
        }
    )
    y = (signal + 0.4 * rng.normal(size=n) > 0).astype(np.int64)

    with pytest.raises(ValueError, match="does not support cat_encoding='target_cv'"):
        KnockoffSelector(q=0.2, cat_encoding="target_cv", verbose=False).fit(X, y)

    # select_fdr deliberately gains no cat_encoding parameter: function parity
    # is not a valid fix for a claim that does not survive the encoding.
    assert "cat_encoding" not in inspect.signature(select_fdr).parameters

    legacy = KnockoffSelector(q=0.2, cat_encoding="loo_logit", verbose=False)
    with pytest.warns(UserWarning, match="no FDR claim applies"):
        legacy.fit(X, y)
    assert legacy.result_.selector_metadata["fdr_control"] == "none"
    assert "validity_note" in legacy.result_.selector_metadata


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


@pytest.mark.parametrize(
    "make_selector",
    [
        lambda: __import__("sift").MRMRSelector(k=2, task="regression", verbose=False),
        lambda: __import__("sift").CEFSPlusSelector(k=2, verbose=False),
        lambda: __import__("sift").KnockoffSelector(q=0.5, verbose=False),
    ],
)
def test_selector_classes_reject_one_dimensional_x(make_selector):
    rng = np.random.default_rng(21)
    x = rng.normal(size=60)

    with pytest.raises(ValueError, match="2D feature matrix"):
        make_selector().fit(x, x)


def test_routed_auto_k_accepts_auto_dense_fields_without_unused_warnings():
    from sift import AutoKConfig

    rng = np.random.default_rng(22)
    X = pd.DataFrame(rng.normal(size=(300, 30)), columns=[f"f{i}" for i in range(30)])
    y = X["f0"].to_numpy() + 0.1 * rng.normal(size=len(X))
    config = AutoKConfig(
        k_method="auto",
        auto_dense_check=True,
        auto_dense_min_k=2,
        auto_dense_min_frac=0.05,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        select_cefsplus(X, y, k="auto", auto_k_config=config, verbose=False)

    unused = [w for w in caught if "does not use it" in str(w.message)]
    assert unused == []


def test_auto_router_time_context_routes_to_best_without_one_se_fallback_warning():
    from sift import AutoKConfig

    rng = np.random.default_rng(23)
    X = pd.DataFrame(rng.normal(size=(240, 20)), columns=[f"f{i}" for i in range(20)])
    y = X["f0"].to_numpy() + 0.1 * rng.normal(size=len(X))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = select_mrmr(
            X,
            y,
            k="auto",
            task="regression",
            estimator="gaussian",
            auto_k_config=AutoKConfig(k_method="auto"),
            time=np.arange(len(X)),
            verbose=False,
            return_result=True,
        )

    fallbacks = [w for w in caught if "falling back" in str(w.message)]
    assert fallbacks == []
    assert result.diagnostics_["auto_k"]["selection_rule"] == "best"


def test_auto_router_warns_when_zero_features_are_selected():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(2000, 100)))
    y = rng.normal(size=len(X))

    with pytest.warns(UserWarning, match="selected 0 features"):
        selected = select_cefsplus(X, y, k="auto", verbose=False)
    assert selected == []


def test_cefsplus_warns_on_integer_multiclass_looking_target():
    rng = np.random.default_rng(25)
    X = pd.DataFrame(rng.normal(size=(200, 8)), columns=[f"f{i}" for i in range(8)])
    y_labels = rng.integers(0, 6, size=len(X)).astype(float)

    with pytest.warns(UserWarning, match="looks\\s+like multiclass labels"):
        select_cefsplus(X, y_labels, k=2, verbose=False)

    y_binary = (X["f0"] > 0).to_numpy().astype(float)
    y_continuous = X["f0"].to_numpy() + 0.1 * rng.normal(size=len(X))
    for safe_target in (y_binary, y_continuous):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            select_cefsplus(X, safe_target, k=2, verbose=False)
        assert [w for w in caught if "multiclass labels" in str(w.message)] == []


def test_stability_selection_frequencies_are_float64():
    from sift import StabilitySelector

    rng = np.random.default_rng(26)
    X = rng.normal(size=(120, 5))
    y = X[:, 0] + 0.1 * rng.normal(size=len(X))
    selector = StabilitySelector(
        n_bootstrap=4,
        alpha=0.5,
        n_jobs=1,
        random_state=0,
        verbose=False,
        store_coefs=False,
    ).fit(X, y)

    assert np.asarray(selector.selection_frequencies_).dtype == np.float64


def test_binary_auto_router_rejects_auto_dense_options():
    from sift import AutoKConfig

    rng = np.random.default_rng(27)
    X = pd.DataFrame(rng.normal(size=(200, 12)), columns=[f"f{i}" for i in range(12)])
    y = (X["f0"].to_numpy() + 0.2 * rng.normal(size=len(X)) > 0).astype(int)
    config = AutoKConfig(
        k_method="auto",
        auto_dense_check=True,
        auto_dense_min_k=2,
        auto_dense_min_frac=0.05,
    )

    # Binary CEFS+ has no dense-regime diagnostic; silently ignoring the opt-in
    # (or warning about fields the router "consumed") would both be dishonest.
    with pytest.raises(ValueError, match="not supported for binary log-loss CEFS"):
        select_cefsplus_binary(X, y, k="auto", auto_k_config=config, verbose=False)

    # The default (unset) fields keep routing cleanly with no unused-field noise.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        select_cefsplus_binary(
            X, y, k="auto", auto_k_config=AutoKConfig(k_method="auto"), verbose=False
        )
    assert [w for w in caught if "does not use it" in str(w.message)] == []

    # Brier mode is intentionally different: it delegates to Gaussian CEFS+
    # and therefore retains the Gaussian dense-check behavior. Pin the routing
    # diagnostics rather than requiring data-dependent disagreement to warn.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = select_cefsplus_binary(
            X,
            y,
            k="auto",
            loss="brier",
            auto_k_config=config,
            verbose=False,
            return_result=True,
        )

    assert "f0" in result.selected_features
    dense_check = result.diagnostics_["auto_k"]["auto_routing"]["dense_check"]
    assert dense_check["enabled"] is True
    assert dense_check["ran"] is True
    assert [w for w in caught if "does not use it" in str(w.message)] == []


def test_gaussian_router_fallback_dispatch_strips_router_only_fields(monkeypatch):
    # Exercises the real fallback call site in select_gaussian_auto_path: the
    # primary routed run degenerates, and the SECOND dispatched config (built
    # from the caller's original config) must not carry the auto_dense_* fields
    # the router already consumed.
    from sift import AutoKConfig
    import sift.selection.filter_auto_k as filter_auto_k

    class DummyCache:
        sample_weight = np.ones(20, dtype=np.float64)
        valid_cols = np.arange(5, dtype=np.int64)

    dispatched = []
    dense_check_calls = []

    def fake_runner(routed_config, **_kwargs):
        dispatched.append(routed_config)
        if len(dispatched) == 1:
            return [], [], pd.DataFrame(), {
                "selected_k": 0,
                "stopped_by": "degenerate_folds",
            }
        return ["x0"], [0], pd.DataFrame(), {"selected_k": 1}

    def capture_dense_check(**kwargs):
        dense_check_calls.append(kwargs)

    monkeypatch.setattr(filter_auto_k, "_run_gaussian_routed_path", fake_runner)
    monkeypatch.setattr(filter_auto_k, "_run_auto_dense_check", capture_dense_check)

    requested_config = AutoKConfig(
        k_method="auto",
        min_k=1,
        max_k=4,
        auto_dense_check=True,
        auto_dense_min_k=2,
        auto_dense_min_frac=0.05,
        auto_dense_disagreement_ratio=9.0,
    )

    _selected, _indices, _diag, summary = filter_auto_k.select_gaussian_auto_path(
        cache=DummyCache(),
        y=np.zeros(20),
        method="mrmr_quot",
        max_k=4,
        top_m=10,
        auto_k_config=requested_config,
        verbose=False,
    )

    assert len(dispatched) == 2
    assert len(dense_check_calls) == 1
    dense_check_config = dense_check_calls[0]["config"]
    assert dense_check_config is requested_config
    assert dense_check_config.auto_dense_check is True
    assert dense_check_config.auto_dense_min_k == 2
    assert dense_check_config.auto_dense_min_frac == 0.05
    assert dense_check_config.auto_dense_disagreement_ratio == 9.0
    assert summary["auto_routing"]["fallback"]["chosen"] == "penalized_objective"
    defaults = AutoKConfig()
    for config in dispatched:
        assert config.auto_dense_check == defaults.auto_dense_check
        assert config.auto_dense_min_k == defaults.auto_dense_min_k
        assert config.auto_dense_min_frac == defaults.auto_dense_min_frac
        assert (
            config.auto_dense_disagreement_ratio
            == defaults.auto_dense_disagreement_ratio
        )


def test_stability_transform_dataframe_contract():
    from sift import StabilitySelector

    rng = np.random.default_rng(28)
    X = pd.DataFrame(rng.normal(size=(150, 6)), columns=list("abcdef"))
    y = X["a"].to_numpy() + 0.1 * rng.normal(size=len(X))
    options = dict(n_bootstrap=5, alpha=0.3, n_jobs=1, verbose=False, random_state=0)

    fitted = StabilitySelector(**options).fit(X, y)
    assert fitted.selected_feature_names_  # sanity: something was selected

    # Name-based selection tolerates extra and reordered columns.
    extra = fitted.transform(X.assign(zz=1.0))
    reordered = fitted.transform(X[list(X.columns[::-1])])
    np.testing.assert_array_equal(extra, fitted.transform(X))
    np.testing.assert_array_equal(reordered, fitted.transform(X))

    # Missing or renamed selected columns raise a clear ValueError, not KeyError.
    with pytest.raises(ValueError, match="missing selected feature column"):
        fitted.transform(X.drop(columns=[fitted.selected_feature_names_[0]]))
    with pytest.raises(ValueError, match="missing selected feature column"):
        fitted.transform(X.rename(columns=lambda c: c + "_x"))

    # A selector fitted on a positional array rejects DataFrame transform input.
    positional = StabilitySelector(**options).fit(X.to_numpy(), y)
    with pytest.raises(ValueError, match="positional array"):
        positional.transform(X)
    generated_names = pd.DataFrame(
        X.to_numpy(), columns=positional.feature_names_in_
    )
    with pytest.raises(ValueError, match="positional array"):
        positional.tune_threshold(
            generated_names,
            y,
            thresholds=[0.5],
            cv=2,
        )
    np.testing.assert_array_equal(
        positional.transform(X.to_numpy()),
        X.to_numpy()[:, positional.selected_features_],
    )

    # Explicit feature_names with ndarray input keep the named DataFrame path.
    named = StabilitySelector(**options).fit(
        X.to_numpy(), y, feature_names=list(X.columns)
    )
    np.testing.assert_array_equal(named.transform(X), named.transform(X.to_numpy()))

    duplicate_extra = pd.concat([X, X[["f"]]], axis=1)
    with pytest.raises(ValueError, match="Duplicate DataFrame column labels"):
        fitted.tune_threshold(
            duplicate_extra,
            y,
            thresholds=[0.5],
            cv=2,
        )


def test_stability_smart_sampler_honors_explicit_feature_subset():
    from sift.sampling.smart import SmartSamplerConfig

    rng = np.random.default_rng(31)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=["a", "b", "c"])
    y = X["a"].to_numpy() + 0.05 * rng.normal(size=len(X))
    selector = StabilitySelector(
        n_bootstrap=4,
        sample_frac=0.7,
        threshold=0.0,
        alpha=0.1,
        use_smart_sampler=True,
        sampler_config=SmartSamplerConfig(
            sample_frac=0.8,
            residual_weight_cap=0.0,
            random_state=0,
            verbose=False,
        ),
        n_jobs=1,
        random_state=0,
        verbose=False,
    ).fit(X, y, feature_names=["a"])

    assert list(selector.feature_names_in_) == ["a"]
    assert set(selector.selected_feature_names_) <= {"a"}
    transformed = selector.transform(X)
    assert transformed.shape == (len(X), len(selector.selected_feature_names_))

    ordered = StabilitySelector(
        n_bootstrap=2,
        sample_frac=0.7,
        threshold=0.0,
        alpha=0.1,
        use_smart_sampler=True,
        sampler_config=SmartSamplerConfig(
            sample_frac=0.8,
            residual_weight_cap=0.0,
            random_state=0,
            verbose=False,
        ),
        n_jobs=1,
        random_state=0,
        verbose=False,
    ).fit(X, y, feature_names=("c", "a"))
    assert list(ordered.feature_names_in_) == ["c", "a"]
    assert set(ordered.selected_feature_names_) <= {"c", "a"}


def test_stability_smart_sampler_rejects_explicit_nonnumeric_feature_names():
    rng = np.random.default_rng(32)
    X = pd.DataFrame(
        {
            "a": rng.normal(size=120),
            "category": np.resize(["left", "right"], 120),
        }
    )
    y = X["a"].to_numpy() + 0.05 * rng.normal(size=len(X))

    selector = StabilitySelector(
        n_bootstrap=2,
        alpha=0.1,
        use_smart_sampler=True,
        n_jobs=1,
        random_state=0,
        verbose=False,
    )
    with pytest.raises(ValueError, match="must reference numeric.*non-numeric"):
        selector.fit(X, y, feature_names=["a", "category"])


def test_stability_smart_sampler_excludes_explicit_metadata_names():
    from sift.sampling.smart import SmartSamplerConfig

    rng = np.random.default_rng(33)
    n = 120
    X = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "group": np.repeat(np.arange(12), 10),
            "time": np.tile(np.arange(10), 12),
            "c": rng.normal(size=n),
        }
    )
    y = X["a"].to_numpy() + 0.05 * rng.normal(size=n)

    selector = StabilitySelector(
        n_bootstrap=2,
        alpha=0.1,
        use_smart_sampler=True,
        sampler_config=SmartSamplerConfig(
            group_col="group",
            time_col="time",
            sample_frac=0.8,
            residual_weight_cap=0.0,
            random_state=0,
            verbose=False,
        ),
        n_jobs=1,
        random_state=0,
        verbose=False,
    ).fit(X, y, feature_names=["a", "group", "time", "c"])

    assert list(selector.feature_names_in_) == ["a", "c"]
    assert set(selector.selected_feature_names_) <= {"a", "c"}


def test_stability_smart_sampler_tune_threshold_retains_metadata_columns():
    from sift.sampling.smart import SmartSamplerConfig

    rng = np.random.default_rng(36)
    n = 120
    X = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "group": np.repeat(np.arange(12), 10),
            "time": pd.date_range("2020-01-01", periods=n, freq="h"),
            "c": rng.normal(size=n),
            "unused_numeric": rng.normal(size=n),
        }
    )
    y = X["a"].to_numpy() + 0.05 * rng.normal(size=n)
    selector = StabilitySelector(
        n_bootstrap=2,
        sample_frac=0.7,
        threshold=0.0,
        alpha=0.1,
        use_smart_sampler=True,
        sampler_config=SmartSamplerConfig(
            group_col="group",
            time_col="time",
            sample_frac=0.8,
            residual_weight_cap=0.0,
            random_state=0,
            verbose=False,
        ),
        n_jobs=1,
        random_state=0,
        verbose=False,
    ).fit(X, y, feature_names=["a", "group", "time"])

    best, results = selector.tune_threshold(X, y, thresholds=[0.0], cv=2)

    assert best == 0.0
    assert list(selector.feature_names_in_) == ["a"]
    assert results.loc[0, "n_features"] == 1.0
    assert results.loc[0, "n_finite"] == 2


@pytest.mark.parametrize(
    ("use_smart_sampler", "feature_names"),
    [
        (False, None),
        (True, None),
        (True, ["a", "elapsed"]),
    ],
)
def test_stability_rejects_timedelta_feature_columns(
    use_smart_sampler,
    feature_names,
):
    rng = np.random.default_rng(37)
    X = pd.DataFrame(
        {
            "a": rng.normal(size=100),
            "elapsed": pd.to_timedelta(np.arange(100), unit="h"),
        }
    )
    y = X["a"].to_numpy() + 0.05 * rng.normal(size=len(X))
    selector = StabilitySelector(
        n_bootstrap=2,
        alpha=0.1,
        use_smart_sampler=use_smart_sampler,
        n_jobs=1,
        random_state=0,
        verbose=False,
    )

    with pytest.raises(ValueError, match="Datetime or timedelta feature columns"):
        selector.fit(X, y, feature_names=feature_names)


@pytest.mark.parametrize(
    "feature_names",
    [
        "ab",
        b"ab",
        bytearray(b"ab"),
        memoryview(b"ab"),
        {"a", "b"},
        {"a": 1, "b": 2},
        np.array("ab"),
        np.array([["a", "b"]]),
        pd.DataFrame([[1, 2]], columns=["a", "b"]),
    ],
    ids=[
        "str",
        "bytes",
        "bytearray",
        "memoryview",
        "set",
        "mapping",
        "scalar-array",
        "matrix-array",
        "dataframe",
    ],
)
def test_stability_rejects_scalar_or_unordered_feature_name_containers(feature_names):
    rng = np.random.default_rng(34)
    X = pd.DataFrame(rng.normal(size=(100, 4)), columns=list("abcd"))
    y = X["a"].to_numpy() + 0.05 * rng.normal(size=len(X))

    with pytest.raises(ValueError, match="ordered, one-dimensional iterable"):
        StabilitySelector(
            n_bootstrap=2,
            alpha=0.1,
            n_jobs=1,
            random_state=0,
            verbose=False,
        ).fit(X, y, feature_names=feature_names)


@pytest.mark.parametrize(
    ("feature_names", "expected"),
    [
        (pd.Index(["a", "c"]), ["a", "c"]),
        (np.array(["c", "a"]), ["c", "a"]),
    ],
    ids=["pandas-index", "one-dimensional-array"],
)
def test_stability_accepts_ordered_one_dimensional_feature_name_containers(
    feature_names,
    expected,
):
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(100, 3)), columns=["a", "b", "c"])
    y = X["a"].to_numpy() + 0.05 * rng.normal(size=len(X))

    selector = StabilitySelector(
        n_bootstrap=2,
        threshold=0.0,
        alpha=0.1,
        n_jobs=1,
        random_state=0,
        verbose=False,
    ).fit(X, y, feature_names=feature_names)

    assert list(selector.feature_names_in_) == expected
    assert selector.get_feature_names_out(feature_names).shape == (2,)


def test_stability_rejects_unhashable_feature_name_entry():
    rng = np.random.default_rng(38)
    X = rng.normal(size=(100, 2))
    y = X[:, 0] + 0.05 * rng.normal(size=len(X))

    with pytest.raises(ValueError, match="entries must be hashable"):
        StabilitySelector(
            n_bootstrap=2,
            alpha=0.1,
            n_jobs=1,
            random_state=0,
            verbose=False,
        ).fit(
            X,
            y,
            feature_names=[memoryview(bytearray(b"a")), "b"],
        )


def test_stability_feature_names_out_preserves_tuple_labels():
    rng = np.random.default_rng(35)
    columns = pd.MultiIndex.from_tuples(
        [("left", "a"), ("left", "b"), ("right", "c")]
    )
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=columns)
    y = X[("left", "a")].to_numpy() + 0.05 * rng.normal(size=len(X))

    selector = StabilitySelector(
        n_bootstrap=2,
        threshold=0.0,
        alpha=0.1,
        n_jobs=1,
        random_state=0,
        verbose=False,
    ).fit(X, y)

    output_names = selector.get_feature_names_out()
    supplied_names = selector.get_feature_names_out(list(columns))
    assert output_names.ndim == 1
    assert supplied_names.ndim == 1
    assert output_names.tolist() == selector.selected_feature_names_
    assert supplied_names.tolist() == selector.selected_feature_names_
    assert selector.transform(X).shape == (len(X), len(output_names))


@pytest.mark.parametrize(
    "missing_label",
    [float("nan"), pd.NaT, pd.NA],
    ids=["nan", "nat", "pd-na"],
)
def test_stability_feature_names_out_accepts_matching_missing_labels(missing_label):
    rng = np.random.default_rng(39)
    columns = pd.Index([missing_label, "b", "c"], dtype=object)
    X = pd.DataFrame(rng.normal(size=(100, 3)), columns=columns)
    y = X.iloc[:, 0].to_numpy() + 0.05 * rng.normal(size=len(X))
    selector = StabilitySelector(
        n_bootstrap=2,
        threshold=0.0,
        alpha=0.1,
        n_jobs=1,
        random_state=0,
        verbose=False,
    ).fit(X, y)

    output_names = selector.get_feature_names_out(list(columns))

    assert output_names.ndim == 1
    assert output_names.shape == (3,)
    assert selector.transform(X).shape == (len(X), 3)


def test_stability_requires_exact_multiindex_feature_labels():
    rng = np.random.default_rng(40)
    X_fit = rng.normal(size=(100, 1))
    y = X_fit[:, 0] + 0.05 * rng.normal(size=len(X_fit))
    selector = StabilitySelector(
        n_bootstrap=2,
        threshold=0.0,
        alpha=0.1,
        n_jobs=1,
        random_state=0,
        verbose=False,
    ).fit(X_fit, y, feature_names=[("left",)])
    ambiguous = pd.DataFrame(
        rng.normal(size=(100, 2)),
        columns=pd.MultiIndex.from_tuples(
            [("left", "a"), ("left", "b")]
        ),
    )

    with pytest.raises(ValueError, match="missing selected feature column"):
        selector.transform(ambiguous)
    with pytest.raises(ValueError, match="missing fitted feature column"):
        selector.tune_threshold(ambiguous, y, thresholds=[0.5], cv=2)


def test_stability_distinguishes_distinct_tuple_feature_labels():
    rng = np.random.default_rng(41)
    X = rng.normal(size=(100, 2))
    y = X[:, 0] + 0.05 * rng.normal(size=len(X))
    feature_names = [("c",), ("c", float("nan"))]

    selector = StabilitySelector(
        n_bootstrap=2,
        threshold=0.0,
        alpha=0.1,
        n_jobs=1,
        random_state=0,
        verbose=False,
    ).fit(X, y, feature_names=feature_names)

    assert selector.get_feature_names_out(feature_names).shape == (2,)


def test_stability_rejects_duplicate_and_empty_feature_names():
    from sift import StabilitySelector

    rng = np.random.default_rng(29)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=["a", "b", "c"])
    y = X["a"].to_numpy() + 0.1 * rng.normal(size=len(X))
    options = dict(n_bootstrap=4, alpha=0.3, n_jobs=1, verbose=False, random_state=0)

    # Duplicate DataFrame columns previously widened transform output silently
    # (selecting 'a' from columns [a, a, b] returned both copies).
    X_dup = pd.concat([X[["a"]], X], axis=1)
    with pytest.raises(ValueError, match="Duplicate DataFrame column labels"):
        StabilitySelector(**options).fit(X_dup, y)

    fitted = StabilitySelector(**options).fit(X, y)
    with pytest.raises(ValueError, match="Duplicate DataFrame column labels"):
        fitted.transform(X_dup)

    with pytest.raises(ValueError, match="feature_names must be unique"):
        StabilitySelector(**options).fit(X.to_numpy(), y, feature_names=["a", "a", "b"])

    # feature_names=[] is explicit-but-wrong, not "absent": reject it up front
    # for arrays and DataFrames alike instead of failing deep inside sklearn.
    with pytest.raises(ValueError, match="non-empty"):
        StabilitySelector(**options).fit(X.to_numpy(), y, feature_names=[])
    with pytest.raises(ValueError, match="non-empty"):
        StabilitySelector(**options).fit(X, y, feature_names=[])

    # Repeated NaN labels are duplicates even though NaN != NaN defeats set().
    with pytest.raises(ValueError, match="feature_names must be unique"):
        StabilitySelector(**options).fit(
            X.to_numpy(), y, feature_names=[float("nan"), float("nan"), "b"]
        )

    # Explicit names must reference existing DataFrame columns.
    with pytest.raises(ValueError, match="missing"):
        StabilitySelector(**options).fit(X, y, feature_names=["a", "zzz"])


def test_stability_failed_fit_leaves_selector_unfitted():
    from sklearn.utils.validation import check_is_fitted
    from sklearn.exceptions import NotFittedError
    from sift import StabilitySelector

    rng = np.random.default_rng(30)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=["a", "b", "c"])
    y = X["a"].to_numpy() + 0.1 * rng.normal(size=len(X))
    X_dup = pd.concat([X[["a"]], X], axis=1)
    options = dict(n_bootstrap=4, alpha=0.3, n_jobs=1, verbose=False, random_state=0)

    # Failed initial fit: no attribute may make the generic fitted-check pass.
    selector = StabilitySelector(**options)
    with pytest.raises(ValueError, match="Duplicate DataFrame column labels"):
        selector.fit(X_dup, y)
    with pytest.raises(NotFittedError):
        check_is_fitted(selector)

    # Failed refit clears the previous fitted state (matching the existing
    # cleanup contract for failures later in fit).
    selector.fit(X, y)
    check_is_fitted(selector)
    with pytest.raises(ValueError, match="Duplicate DataFrame column labels"):
        selector.fit(X_dup, y)
    with pytest.raises(NotFittedError):
        check_is_fitted(selector)
