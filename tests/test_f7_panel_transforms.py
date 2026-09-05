"""Public-contract tests for F7 panel within/between transforms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

import sift
from sift.selection.auto_k import AutoKConfig, select_k_auto
from sift.selection.within import TWO_WAY_ITERATIONS, fit_within_transform


def _panel(n_groups=8, n_time=6, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), n_time)
    time = np.tile(np.arange(n_time), n_groups)
    n = groups.shape[0]
    within_noise = rng.normal(size=n)
    entity = 3.0 * rng.normal(size=n_groups)[groups]
    X = pd.DataFrame(
        {
            "between_only": entity,
            "within_signal": within_noise,
            "noise": rng.normal(size=n),
        }
    )
    y = entity + 0.4 * within_noise + 0.05 * rng.normal(size=n)
    return X, y, groups, time


def test_within_none_matches_omitted_argument():
    X, y, groups, _time = _panel()
    omitted = sift.select_mrmr(X, y, k=2, task="regression", verbose=False)
    explicit = sift.select_mrmr(
        X, y, k=2, task="regression", within=None, verbose=False
    )
    assert omitted == explicit


def test_fixed_k_still_rejects_unused_groups():
    X, y, groups, time = _panel()
    with pytest.raises(ValueError, match="only meaningful for auto-k evaluation"):
        sift.select_mrmr(
            X, y, k=2, task="regression", groups=groups, verbose=False
        )
    with pytest.raises(ValueError, match="omit time for a fixed-k within='groups'"):
        sift.select_mrmr(
            X,
            y,
            k=2,
            task="regression",
            groups=groups,
            time=time,
            within="groups",
            verbose=False,
        )


def test_groups_within_prefers_within_signal():
    X, y, groups, _time = _panel()
    raw = sift.select_mrmr(X, y, k=1, task="regression", verbose=False)
    within = sift.select_mrmr(
        X, y, k=1, task="regression", groups=groups, within="groups", verbose=False
    )
    assert raw == ["between_only"]
    assert within == ["within_signal"]


@pytest.mark.parametrize(
    "selector",
    (
        lambda **kw: sift.select_mrmr(task="regression", estimator="classic", **kw),
        lambda **kw: sift.select_jmi(task="regression", estimator="r2", **kw),
        lambda **kw: sift.select_jmim(task="regression", estimator="r2", **kw),
        lambda **kw: sift.select_cefsplus(**kw),
        lambda **kw: sift.select_mrmr(task="regression", estimator="gaussian", **kw),
    ),
)
def test_within_groups_public_selectors_recover_within_signal(selector):
    X, y, groups, _time = _panel()
    selected = selector(X=X, y=y, k=1, groups=groups, within="groups", verbose=False)
    assert selected == ["within_signal"]


def test_ranking_exposes_within_and_between_columns():
    X, y, groups, _time = _panel()
    result = sift.select_mrmr(
        X,
        y,
        k=1,
        task="regression",
        groups=groups,
        within="groups",
        verbose=False,
        return_result=True,
    )
    ranking = result.ranking_
    assert list(ranking.columns) == [
        "feature",
        "rank",
        "selected",
        "selected_index",
        "relevance",
        "within_relevance",
        "between_relevance",
        "selector",
    ]
    assert result.selector_metadata["within"] == "groups"
    within_row = ranking.set_index("feature").loc["within_signal"]
    between_row = ranking.set_index("feature").loc["between_only"]
    assert within_row["within_relevance"] == pytest.approx(within_row["relevance"])
    assert between_row["between_relevance"] > within_row["between_relevance"]
    assert within_row["within_relevance"] > between_row["within_relevance"]
    view = sift.as_result(result, input_features=list(X.columns))
    assert "within_relevance" in view.table.columns
    assert "between_relevance" in view.table.columns


def test_sample_weight_changes_group_means():
    X, y, groups, _time = _panel(n_groups=4, n_time=4, seed=1)
    w = np.ones(len(y), dtype=np.float64)
    w[0] = 25.0
    result_u = sift.select_mrmr(
        X,
        y,
        k=2,
        task="regression",
        groups=groups,
        within="groups",
        verbose=False,
        return_result=True,
    )
    result_w = sift.select_mrmr(
        X,
        y,
        k=2,
        task="regression",
        groups=groups,
        within="groups",
        sample_weight=w,
        verbose=False,
        return_result=True,
    )
    y_arr = np.asarray(y, dtype=np.float64)
    X_arr = X.to_numpy(dtype=np.float64)
    fitted_u = fit_within_transform(
        "groups", X_arr, y_arr, groups, None, np.ones(len(y))
    )
    fitted_w = fit_within_transform("groups", X_arr, y_arr, groups, None, w)
    assert not np.allclose(fitted_u.group_effects_X, fitted_w.group_effects_X)
    assert not np.allclose(
        result_u.ranking_["within_relevance"].to_numpy(dtype=float),
        result_w.ranking_["within_relevance"].to_numpy(dtype=float),
    )


def test_unseen_group_falls_back_to_training_grand_mean():
    groups = np.array([0, 0, 1, 1, 2, 2])
    X = np.asarray([[0.0], [2.0], [10.0], [12.0], [100.0], [102.0]])
    y = np.asarray([0.0, 2.0, 10.0, 12.0, 100.0, 102.0])
    w = np.ones(6)
    train = np.array([0, 1, 2, 3])
    fitted = fit_within_transform("groups", X[train], y[train], groups[train], None, w[train])
    X_va, y_va = fitted.transform(X[4:], y[4:], groups[4:], None)
    grand_x = 6.0
    grand_y = 6.0
    assert X_va[:, 0] == pytest.approx([100.0 - grand_x, 102.0 - grand_x])
    assert y_va == pytest.approx([100.0 - grand_y, 102.0 - grand_y])


def test_two_way_uses_documented_iteration_count():
    assert TWO_WAY_ITERATIONS == 5
    X, y, groups, time = _panel(n_groups=6, n_time=5, seed=3)
    result = sift.select_cefsplus(
        X,
        y,
        k=1,
        groups=groups,
        time=time,
        within="two_way",
        verbose=False,
        return_result=True,
    )
    assert result.selector_metadata["within"] == "two_way"
    assert result.selector_metadata["within_two_way_iterations"] == 5
    groups_only = sift.select_cefsplus(
        X, y, k=1, groups=groups, within="groups", verbose=False
    )
    two_way = result.selected_features
    assert isinstance(two_way, list)
    assert groups_only or two_way


def test_evaluate_is_fold_local_not_global_demean():
    rng = np.random.default_rng(4)
    n_groups, n_time = 6, 8
    groups = np.repeat(np.arange(n_groups), n_time)
    n = groups.shape[0]
    entity = np.linspace(-3.0, 3.0, n_groups)[groups]
    X = pd.DataFrame(
        {
            "between_only": entity,
            "noise": rng.normal(size=n),
        }
    )
    y = entity + 0.05 * rng.normal(size=n)
    config = AutoKConfig(
        k_method="evaluate",
        strategy="group_cv",
        n_splits=3,
        min_k=1,
        max_k=2,
        selection_rule="best",
    )
    _best_k, _feats, diag = select_k_auto(
        X,
        np.asarray(y, dtype=np.float64),
        ["between_only", "noise"],
        config,
        groups=groups,
        within="groups",
    )
    X_arr = X.to_numpy(dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    w = np.ones(n)
    global_fit = fit_within_transform("groups", X_arr, y_arr, groups, None, w)
    X_global, y_global = global_fit.transform(X_arr, y_arr, groups, None)
    # Held-out groups keep between variation after train-only fallback,
    # so fold scores are not the near-zero residuals of a global demean.
    assert float(np.var(y_global)) < 0.05
    assert diag["score_mean"].notna().any()
    assert float(diag["score_mean"].min()) > 0.05


def test_gaussian_cv_with_within_does_not_require_cache():
    X, y, groups, _time = _panel(n_groups=6, n_time=8, seed=5)
    config = AutoKConfig(
        k_method="gaussian_cv",
        strategy="group_cv",
        xfit_folds=3,
        min_k=1,
        max_k=2,
        selection_rule="best",
    )
    selected = sift.select_cefsplus(
        X,
        y,
        k="auto",
        groups=groups,
        within="groups",
        auto_k_config=config,
        verbose=False,
        subsample=None,
    )
    assert selected
    assert "within_signal" in selected


def test_unsupported_combinations_raise():
    X, y, groups, time = _panel()
    cache = sift.build_cache(X, subsample=None)
    with pytest.raises(ValueError, match="prebuilt cache"):
        sift.select_cefsplus(
            X, y, k=1, cache=cache, groups=groups, within="groups", verbose=False
        )
    with pytest.raises(ValueError, match="task='regression'"):
        sift.select_mrmr(
            X,
            (np.asarray(y) > np.median(y)).astype(int),
            k=1,
            task="classification",
            groups=groups,
            within="groups",
            verbose=False,
        )
    with pytest.raises(ValueError, match="requires groups"):
        sift.select_mrmr(X, y, k=1, task="regression", within="groups", verbose=False)
    with pytest.raises(ValueError, match="requires groups and time"):
        sift.select_mrmr(
            X,
            y,
            k=1,
            task="regression",
            groups=groups,
            within="two_way",
            verbose=False,
        )
    with pytest.raises(ValueError, match="evaluate"):
        sift.select_cefsplus(
            X,
            y,
            k="auto",
            groups=groups,
            within="groups",
            auto_k_config=AutoKConfig(k_method="elbow", min_k=1, max_k=2),
            verbose=False,
        )
    with pytest.raises(ValueError, match="must be None, 'groups', or 'two_way'"):
        sift.select_mrmr(
            X, y, k=1, task="regression", groups=groups, within="entity", verbose=False
        )


def test_sklearn_wrapper_transform_returns_raw_columns():
    X, y, groups, _time = _panel()
    selector = sift.MRMRSelector(
        k=1, task="regression", within="groups", verbose=False
    )
    selector.fit(X, y, groups=groups)
    assert selector.selected_features_ == ["within_signal"]
    transformed = selector.transform(X)
    assert list(transformed.columns) == ["within_signal"]
    pd.testing.assert_series_equal(transformed["within_signal"], X["within_signal"])
    cloned = clone(selector)
    cloned.fit(X, y, groups=groups)
    assert cloned.selected_features_ == ["within_signal"]
    with pytest.raises(ValueError, match="nested"):
        sift.MRMRSelector(
            k="auto",
            task="regression",
            within="groups",
            auto_k_config=AutoKConfig(
                auto_k_mode="nested",
                k_method="evaluate",
                strategy="group_cv",
                min_k=1,
                max_k=2,
            ),
            verbose=False,
        ).fit(X, y, groups=groups)


def test_group_constant_large_offset_is_not_a_within_signal():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(20), 7)
    weights = rng.uniform(0.2, 2.0, len(groups))
    x = (1e8 + rng.normal(size=20))[groups]
    X = pd.DataFrame({"entity_only": x})
    y = 2.0 * x
    time = np.tile(np.arange(7), 20)
    for estimator, extra in (
        ("classic", {"within": "groups"}),
        ("gaussian", {"within": "groups"}),
        ("classic", {"within": "two_way", "time": time}),
        ("gaussian", {"within": "two_way", "time": time}),
    ):
        with pytest.raises(ValueError, match="no within-entity signal remains"):
            sift.select_mrmr(
                X,
                y,
                k=1,
                task="regression",
                estimator=estimator,
                groups=groups,
                sample_weight=weights,
                subsample=None,
                verbose=False,
                return_result=True,
                **extra,
            )


@pytest.mark.parametrize("kind", ("datetime", "timedelta"))
def test_within_rejects_datetime_like_columns(kind):
    n = 60
    groups = np.repeat(np.arange(6), 10)
    x = np.arange(n, dtype=np.float64)
    temporal = (
        pd.to_datetime(np.arange(n), unit="D")
        if kind == "datetime"
        else pd.to_timedelta(np.arange(n), unit="D")
    )
    X = pd.DataFrame({"date": temporal, "x": x})
    y = x + 0.01 * np.arange(n)
    with pytest.raises(ValueError, match="Datetime or timedelta"):
        sift.select_cefsplus(X, y, k=1, verbose=False)
    with pytest.raises(ValueError, match="Datetime or timedelta"):
        sift.select_cefsplus(
            X, y, k=1, groups=groups, within="groups", verbose=False
        )
    config = AutoKConfig(
        k_method="evaluate",
        strategy="group_cv",
        n_splits=3,
        min_k=1,
        max_k=2,
        selection_rule="best",
    )
    with pytest.raises(ValueError, match="Datetime or timedelta"):
        select_k_auto(
            X,
            y,
            ["date", "x"],
            config,
            groups=groups,
            within="groups",
        )


def test_within_proxies_use_demeaned_target():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(20), 10)
    n = len(groups)
    signal = rng.normal(size=n)
    skew = np.where(np.arange(n) % 10 == 0, 9.0, -1.0) * np.where(groups < 10, 1.0, -1.0)
    X = pd.DataFrame({"within": signal, "skew": skew})
    y = 100.0 * groups + signal
    result = sift.select_cefsplus(
        X,
        y,
        k=1,
        within="groups",
        groups=groups,
        top_m=1,
        return_result=True,
        store_proxies=True,
        subsample=None,
        verbose=False,
    )
    assert result.selected_features == ["within"]
    view = sift.as_result(result, input_features=list(X.columns))
    assert view.metadata.get("proxy_correlations_stored") is True


def test_select_k_auto_within_rejects_classification():
    rng = np.random.default_rng(0)
    n = 60
    groups = np.repeat(np.arange(6), 10)
    X = pd.DataFrame(
        {"x": rng.normal(size=n), "z": rng.normal(size=n)}
    )
    y = (X["x"].to_numpy() > 0).astype(int)
    config = AutoKConfig(
        k_method="evaluate",
        strategy="group_cv",
        n_splits=3,
        min_k=1,
        max_k=2,
        selection_rule="best",
    )
    with pytest.raises(ValueError, match="task='regression'"):
        select_k_auto(
            X,
            y,
            ["x", "z"],
            config,
            groups=groups,
            within="groups",
            task="classification",
        )
