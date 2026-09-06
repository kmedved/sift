"""Joint multi-target CEFS+ (F5): parity, oracle, evaluate, auto-k, rejections."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from sift import (
    AutoKConfig,
    CEFSPlusSelector,
    build_cache,
    evaluate_feature_path,
    select_cached,
    select_cefsplus,
    select_cefsplus_binary,
    select_fdr,
    select_jmi,
    select_k_auto,
)
from sift.selection.auto_k_core import weighted_regression_score
from sift.selection.auto_k_objective import _log_comb, _penalty_array
from sift.selection.cefsplus_multi import (
    TARGET_CONDITION_CAP,
    _spd_inverse_middle,
    joint_logdet_oracle,
    multivariate_ic_df,
)
from sift.selection.panel import build_candidate_panel


def _shared_signal_frame(n=240, p=12, q=3, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    noise = 0.15 * rng.normal(size=(n, q))
    loads = [
        (1.0, 0.8),
        (1.0, 0.5),
        (0.7, 1.0),
        (0.6, 0.6),
    ]
    cols = [
        loads[j % len(loads)][0] * X["f0"]
        + loads[j % len(loads)][1] * X["f1"]
        + noise[:, j]
        for j in range(q)
    ]
    return X, np.column_stack(cols)


def test_one_column_target_matches_1d_cached_and_uncached():
    rng = np.random.default_rng(1)
    n, p = 160, 8
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = (X["f2"] - 1.5 * X["f4"] + 0.2 * rng.normal(size=n)).to_numpy()
    y_col = y.reshape(-1, 1)
    cache = build_cache(X, subsample=None, compute_Rxx=True)

    names_1d = select_cefsplus(X, y, k=3, verbose=False)
    names_col = select_cefsplus(X, y_col, k=3, verbose=False)
    assert names_col == names_1d

    cached_1d, obj_1d = select_cached(cache, y, k=3, return_objective=True)
    cached_col, obj_col = select_cached(cache, y_col, k=3, return_objective=True)
    assert cached_col == cached_1d == names_1d
    np.testing.assert_allclose(obj_col, obj_1d, rtol=0.0, atol=0.0)

    wrapper = CEFSPlusSelector(k=3, verbose=False).fit(X, y_col)
    assert list(wrapper.selected_features_) == names_1d


def test_joint_gain_matches_direct_logdet_oracle():
    X, Y = _shared_signal_frame()
    cache = build_cache(X, subsample=None, compute_Rxx=True)
    names, indices, objective = select_cached(
        cache, Y, k=3, return_indices=True, return_objective=True
    )
    assert len(names) == 3
    assert "f0" in names[:2] and "f1" in names[:2]
    panel = build_candidate_panel(cache, Y, k=3, top_m=20, method="cefsplus")
    local = [
        int(np.flatnonzero(panel.original == idx)[0]) for idx in indices
    ]
    oracle = joint_logdet_oracle(panel.R, panel.C, panel.Ryy, local)
    np.testing.assert_allclose(float(objective[-1]), float(oracle), rtol=1e-8, atol=1e-8)


def test_spd_inverse_middle_is_c_sigma_inv_ct():
    rng = np.random.default_rng(2)
    sigma = rng.normal(size=(4, 4))
    sigma = sigma @ sigma.T + np.eye(4)
    C = rng.normal(size=(3, 4))
    got = _spd_inverse_middle(sigma, C, shrink=1e-6, eps=1e-12)
    expected = C @ np.linalg.inv(sigma) @ C.T
    np.testing.assert_allclose(got, expected, rtol=1e-8, atol=1e-8)


def test_public_cached_uncached_weighted_and_result_metadata():
    X, Y = _shared_signal_frame()
    w = np.linspace(0.5, 1.5, len(X))
    uncached = select_cefsplus(X, Y, k=4, verbose=False)
    cache = build_cache(X, subsample=None, compute_Rxx=True)
    cached = select_cached(cache, Y, k=4)
    assert cached == uncached
    w_cache = build_cache(X, sample_weight=w, subsample=None, compute_Rxx=True)
    weighted = select_cached(w_cache, Y, k=4)
    assert len(weighted) == 4
    assert "f0" in weighted and "f1" in weighted
    view = select_cached(cache, Y, k=3, return_result=True)
    extra = view.metadata
    assert extra["n_targets"] == 3
    assert extra["ic_df_rule"] == "q_k"
    assert extra["target_condition_cap"] == TARGET_CONDITION_CAP
    assert np.isfinite(extra["target_condition_number"])
    result = select_cefsplus(X, Y, k=3, verbose=False, return_result=True)
    assert result.selector_metadata["n_targets"] == 3
    steps: list[int] = []

    def callback(step, total, info):
        steps.append(int(step))
        assert info.get("selector") == "cefsplus"

    select_cefsplus(X, Y, k=3, verbose=False, callback=callback)
    assert steps == [1, 2, 3]


def test_include_conditions_path_keeps_k_discoveries_and_conditional_objective():
    X, Y = _shared_signal_frame(p=8)
    base = select_cefsplus(X, Y, k=2, verbose=False)
    k1 = select_cefsplus(X, Y, k=1, include=["f3"], verbose=False)
    assert k1[0] == "f3"
    assert len(k1) == 2
    assert k1[1] in {"f0", "f1"}
    got = select_cefsplus(X, Y, k=2, include=["f3"], verbose=False)
    assert got[0] == "f3"
    assert len(got) == 3
    assert got != ["f3"] + base[1:]
    assert "f0" in got
    one_d = select_cefsplus(X, Y[:, 0], k=2, include=["f3"], verbose=False)
    assert one_d[0] == "f3" and len(one_d) == 3
    singleton = select_cefsplus(
        X,
        Y,
        k=2,
        include=["f3"],
        feature_blocks={name: [name] for name in X.columns},
        verbose=False,
    )
    assert singleton == got
    cache = build_cache(X, subsample=None, compute_Rxx=True)
    names, idx, obj = select_cached(
        cache,
        Y,
        k=2,
        include=["f3"],
        return_indices=True,
        return_objective=True,
    )
    assert names == got
    assert len(obj) == 2
    panel = build_candidate_panel(cache, Y, k=3, top_m=20, method="cefsplus")
    orig_to_local = {int(o): i for i, o in enumerate(panel.original)}
    include_local = [orig_to_local[int(X.columns.get_loc("f3"))]]
    disc_local = [orig_to_local[int(i)] for i in idx if int(i) != int(X.columns.get_loc("f3"))]
    oracle_full = joint_logdet_oracle(
        panel.R, panel.C, panel.Ryy, include_local + disc_local
    )
    oracle_inc = joint_logdet_oracle(panel.R, panel.C, panel.Ryy, include_local)
    np.testing.assert_allclose(
        float(obj[-1]), float(oracle_full - oracle_inc), rtol=1e-8, atol=1e-8
    )
    blocked = select_cefsplus(
        X,
        Y,
        k=1,
        feature_blocks={"signal": ["f0", "f1"], "noise": ["f2", "f3"]},
        verbose=False,
    )
    assert set(blocked[:2]) == {"f0", "f1"}


def test_collinear_targets_rejected_with_drop_or_combine_guidance():
    X, Y = _shared_signal_frame(q=2)
    Y[:, 1] = Y[:, 0]
    with pytest.raises(ValueError, match="Drop or combine"):
        select_cefsplus(X, Y, k=2, verbose=False)


def test_constant_multi_target_columns_rejected_on_retained_rows():
    rng = np.random.default_rng(5)
    n, p = 80, 6
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    signal = X["f0"].to_numpy() + 0.2 * rng.normal(size=n)
    ones = np.ones(n)
    with pytest.raises(ValueError, match="Drop or combine"):
        select_cefsplus(X, np.column_stack([signal, ones]), k=2, verbose=False)
    with pytest.raises(ValueError, match="Drop or combine"):
        select_cefsplus(X, np.ones((n, 2)), k=2, verbose=False)
    select_cefsplus(X, ones, k=2, verbose=False)
    weights = np.ones(n)
    weights[:20] = 0.0
    y_hidden = np.ones(n)
    y_hidden[:20] = rng.normal(size=20)
    cache = build_cache(X, sample_weight=weights, subsample=None)
    with pytest.raises(ValueError, match="Drop or combine"):
        select_cached(cache, np.column_stack([signal, y_hidden]), k=2)
    sub = build_cache(X, subsample=40, random_state=0)
    y_sub = np.ones(n)
    keep = np.asarray(sub.row_idx)
    y_sub[keep] = 1.0
    outside = np.setdiff1d(np.arange(n), keep)
    if outside.size:
        y_sub[outside] = rng.normal(size=outside.size)
    with pytest.raises(ValueError, match="Drop or combine"):
        select_cached(sub, np.column_stack([signal, y_sub]), k=2)


def test_unsupported_2d_combinations_are_rejected():
    X, Y = _shared_signal_frame()
    groups = np.repeat(np.arange(4), len(X) // 4)
    with pytest.raises(ValueError, match="method='cefsplus'"):
        select_cached(build_cache(X, subsample=None), Y, k=2, method="jmi")
    with pytest.raises(ValueError, match="select_cefsplus"):
        select_jmi(X, Y, k=2, task="regression", verbose=False)
    with pytest.raises(ValueError, match="binary CEFS"):
        select_cefsplus_binary(X, (Y > 0).astype(int), k=2, verbose=False)
    with pytest.raises(ValueError, match="within"):
        select_cefsplus(X, Y, k=2, groups=groups, within="groups", verbose=False)
    with pytest.raises(ValueError, match="supervised cat_encoding"):
        labeled = X.copy()
        labeled["city"] = np.array(["a", "b", "c", "d"])[np.arange(len(X)) % 4]
        select_cefsplus(
            labeled,
            Y,
            k=2,
            cat_encoding="target_cv",
            verbose=False,
        )
    with pytest.raises(ValueError, match="auto-k method"):
        select_cefsplus(
            X,
            Y,
            k="auto",
            auto_k_config=AutoKConfig(
                k_method="gaussian_cv", strategy="kfold", min_k=1, max_k=4
            ),
            verbose=False,
        )
    with pytest.raises(ValueError, match="auto-k method"):
        select_cefsplus(
            X,
            Y,
            k="auto",
            auto_k_config=AutoKConfig(k_method="perm_gap", max_k=4, min_k=0),
            verbose=False,
        )
    with pytest.raises(ValueError, match="2-D y is only supported"):
        select_fdr(X, Y, q=0.2)


def test_select_k_auto_keeps_string_classification_labels():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 6)), columns=[f"f{i}" for i in range(6)])
    labels = np.where(X["f0"] > 0, "yes", "no")
    cfg = AutoKConfig(
        k_method="evaluate",
        strategy="time_holdout",
        min_k=1,
        max_k=3,
        selection_rule="best",
    )
    best_k, features, _diag = select_k_auto(
        X,
        labels,
        ["f0", "f1", "f2"],
        cfg,
        task="classification",
        time=np.arange(200),
    )
    assert best_k == 1
    assert features == ["f0"]
    Y = np.column_stack([X["f0"].to_numpy(), X["f1"].to_numpy()])
    with pytest.raises(ValueError, match="classification"):
        select_k_auto(
            X,
            Y,
            ["f0", "f1", "f2"],
            cfg,
            task="classification",
            time=np.arange(200),
        )


def test_evaluate_multioutput_ridge_row_weights_and_public_path():
    rng = np.random.default_rng(3)
    n = 80
    X = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.normal(size=n),
        }
    )
    Y = np.column_stack([2.0 * X["a"], -X["b"]]) + 0.05 * rng.normal(size=(n, 2))
    w = np.linspace(0.4, 1.6, n)
    result = evaluate_feature_path(
        X,
        Y,
        feature_path=["a", "b", "c"],
        k_grid=[1, 2, 3],
        estimator=LinearRegression(),
        scoring="rmse",
        sample_weight=w,
        splitter=(np.arange(0, 50), np.arange(50, 80)),
    )
    assert result.best_k in {1, 2, 3}
    y_va = Y[50:]
    w_va = w[50:] / w[50:].mean()
    pred = LinearRegression().fit(X.iloc[:50, :2], Y[:50], sample_weight=w[:50]).predict(
        X.iloc[50:, :2]
    )
    expected = weighted_regression_score(y_va, pred, "rmse", sample_weight=w_va)
    np.testing.assert_allclose(result.scores[2], expected, rtol=1e-6, atol=1e-6)
    one_d = evaluate_feature_path(
        X,
        Y[:, 0],
        feature_path=["a", "b", "c"],
        k_grid=[1, 2],
        estimator=LinearRegression(),
        scoring="rmse",
        splitter=(np.arange(0, 50), np.arange(50, 80)),
        random_state=0,
    )
    col = evaluate_feature_path(
        X,
        Y[:, :1],
        feature_path=["a", "b", "c"],
        k_grid=[1, 2],
        estimator=LinearRegression(),
        scoring="rmse",
        splitter=(np.arange(0, 50), np.arange(50, 80)),
        random_state=0,
    )
    assert one_d.best_k == col.best_k
    np.testing.assert_allclose(one_d.scores[1], col.scores[1])


def test_ic_df_is_qk_likelihood_not_search_index_or_independent_sum():
    cfg = AutoKConfig(
        k_method="penalized_objective",
        objective_penalty="ebic",
        min_k=0,
        max_k=10,
    )
    ks = np.arange(0, 6, dtype=np.int64)
    n_eff = 400.0
    p = 20
    q = 3
    pen_k, _, gamma, _ = _penalty_array(
        cfg, ks, n_eff=n_eff, n_candidates=p, dimension=None
    )
    pen_qk, _, _, _ = _penalty_array(
        cfg, ks, n_eff=n_eff, n_candidates=p, dimension=q * ks.astype(np.float64)
    )
    logn = np.log(n_eff)
    comb = 2.0 * gamma * _log_comb(p, ks)
    np.testing.assert_allclose(pen_k, ks * logn + comb)
    np.testing.assert_allclose(pen_qk, q * ks * logn + comb)
    independent_sum = q * pen_k
    np.testing.assert_allclose(pen_qk, independent_sum - (q - 1) * comb)
    np.testing.assert_array_equal(multivariate_ic_df(ks[1:], q), q * ks[1:])


def test_auto_k_evaluate_penalized_and_measured_default_on_shared_design():
    X, Y = _shared_signal_frame(n=320, p=16, q=3, seed=4)
    time = np.arange(len(X))
    eval_cfg = AutoKConfig(
        k_method="evaluate",
        strategy="time_holdout",
        min_k=1,
        max_k=6,
        val_frac=0.25,
    )
    evaluated = select_cefsplus(
        X, Y, k="auto", time=time, auto_k_config=eval_cfg, verbose=False
    )
    assert 1 <= len(evaluated) <= 6
    assert "f0" in evaluated and "f1" in evaluated
    pen_cfg = AutoKConfig(
        k_method="penalized_objective",
        objective_penalty="ebic",
        min_k=0,
        max_k=8,
    )
    penalized = select_cefsplus(
        X, Y, k="auto", auto_k_config=pen_cfg, verbose=False, return_result=True
    )
    assert penalized.selector_metadata["n_targets"] == 3
    assert penalized.diagnostics_["auto_k"]["ic_df_rule"] == "q_k"
    measured = select_cefsplus(X, Y, k="auto", verbose=False)
    assert "f0" in measured and "f1" in measured
    elbowed = select_cefsplus(
        X,
        Y,
        k="auto",
        auto_k_config=AutoKConfig(k_method="elbow", min_k=1, max_k=6),
        verbose=False,
    )
    assert "f0" in elbowed and "f1" in elbowed
    k_df = select_cefsplus(
        X,
        Y,
        k="auto",
        auto_k_config=AutoKConfig(
            k_method="penalized_objective",
            objective_penalty="ebic",
            min_k=0,
            max_k=8,
        ),
        verbose=False,
    )
    assert set(k_df[:2]) <= set(X.columns)
    from sift.selection.cefsplus_multi import multivariate_ic_df
    from sift.selection.filter_auto_k_cache import _cached_filter_path
    from sift.selection.filter_auto_k_common import _select_penalized_count

    cache = build_cache(X, subsample=None, compute_Rxx=True)
    _path, _indices, objective = _cached_filter_path(
        cache,
        Y,
        8,
        method="cefsplus",
        top_m=40,
        corr_prune="auto",
        want_indices=True,
        return_objective=True,
    )
    n_steps = len(np.asarray(objective).ravel())
    count_qk, _ = _select_penalized_count(
        objective,
        pen_cfg,
        objective_scale="n_eff",
        n_samples=len(cache.sample_weight),
        sample_weight=cache.sample_weight,
        n_candidates=len(cache.valid_cols),
        path_length=n_steps,
        df_path=multivariate_ic_df(np.arange(1, n_steps + 1), 3),
        ic_dimension="df",
    )
    count_k, _ = _select_penalized_count(
        objective,
        pen_cfg,
        objective_scale="n_eff",
        n_samples=len(cache.sample_weight),
        sample_weight=cache.sample_weight,
        n_candidates=len(cache.valid_cols),
        path_length=n_steps,
        ic_dimension="k",
    )
    assert count_qk <= count_k
    assert count_qk >= 2
