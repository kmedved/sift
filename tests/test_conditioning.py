"""Public-contract tests for F1 include/exclude/candidates conditioning."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from sift import (
    CEFSPlusBinarySelector,
    CEFSPlusSelector,
    JMISelector,
    JMIMSelector,
    KnockoffSelector,
    MRMRSelector,
    as_result,
    build_cache,
    select_cached,
    select_cefsplus,
    select_cefsplus_binary,
    select_fdr,
    select_jmi,
    select_jmim,
    select_mrmr,
)
from sift.selection.auto_k import AutoKConfig
from sift.selection.cefsplus_binary_common import binary_refit_loglik_gains
from sift.selection.loops import _mrmr_loop_blas, _mrmr_loop_processes, mrmr_select


def _small_regression(n=80, p=8, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = X["f0"] + 0.7 * X["f1"] + 0.4 * X["f2"] + 0.1 * rng.normal(size=n)
    return X, y


def _schur_objective(R, r, idx, shrink=1e-6):
    """Independent log-det I(y; S) using a joint covariance, not the production loop."""
    if len(idx) == 0:
        return 0.0
    scale = 1.0 - shrink
    S = np.asarray(idx, dtype=np.int64)
    Rs = scale * np.asarray(R, dtype=np.float64)[np.ix_(S, S)]
    np.fill_diagonal(Rs, 1.0)
    ry = scale * np.asarray(r, dtype=np.float64)[S]
    n = len(S)
    joint = np.eye(n + 1, dtype=np.float64)
    joint[1:, 1:] = Rs
    joint[0, 1:] = ry
    joint[1:, 0] = ry
    sign_s, logdet_s = np.linalg.slogdet(Rs)
    sign_j, logdet_j = np.linalg.slogdet(joint)
    if sign_s <= 0 or sign_j <= 0:
        raise AssertionError("oracle covariance is not PD")
    return float(logdet_s - logdet_j)


def test_cefsplus_conditional_path_matches_schur_oracle():
    rng = np.random.default_rng(1)
    n, p = 120, 6
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=list("abcdef"))
    y = X["a"] + 0.8 * X["c"] + 0.5 * X["e"] + 0.05 * rng.normal(size=n)
    cache = build_cache(X, compute_Rxx=True, subsample=None)
    from sift.selection.panel import build_candidate_panel

    panel = build_candidate_panel(cache, y.to_numpy(), k=3, top_m=p, corr_prune=None)
    include_orig = 1
    include_local = int(np.flatnonzero(panel.original == include_orig)[0])
    remaining = [i for i in range(len(panel.rel)) if i != include_local]
    gains = [
        _schur_objective(panel.R, panel.r, [include_local, j])
        - _schur_objective(panel.R, panel.r, [include_local])
        for j in remaining
    ]
    expected_local = remaining[int(np.argmax(gains))]
    expected = cache.feature_names[int(panel.original[expected_local])]
    view = select_cefsplus(
        X,
        y,
        k=1,
        include=["b"],
        top_m=p,
        corr_prune=None,
        verbose=False,
        return_result=True,
        subsample=None,
    )
    assert view.selected_features[0] == "b"
    assert view.selected_features[1] == expected
    path = np.asarray(view.diagnostics_["objective_path"], dtype=float)
    assert path.shape == (1,)
    assert path[0] == pytest.approx(max(gains), rel=1e-4, abs=1e-6)
    gain = np.asarray(view.diagnostics_["objective_gain"], dtype=float)
    assert gain.shape == (1,)
    assert gain[0] == pytest.approx(path[0], rel=0, abs=1e-12)
    uncond = select_cefsplus(X, y, k=2, top_m=p, corr_prune=None, verbose=False)
    assert view.selected_features != uncond
    assert "b" not in uncond


def test_validation_rejects_duplicates_unknown_overlap_empty_and_unordered():
    X, y = _small_regression()
    with pytest.raises(ValueError, match="duplicate"):
        select_cefsplus(X, y, k=1, include=["f0", "f0"], verbose=False)
    with pytest.raises(ValueError, match="unknown"):
        select_cefsplus(X, y, k=1, include=["nope"], verbose=False)
    with pytest.raises(ValueError, match="overlap"):
        select_cefsplus(X, y, k=1, include=["f0"], exclude=["f0"], verbose=False)
    with pytest.raises(ValueError, match="empty candidate pool"):
        select_cefsplus(X, y, k=1, include=["f0"], candidates=["f0"], verbose=False)
    with pytest.raises(ValueError, match="unordered"):
        select_cefsplus(X, y, k=1, include={"f0"}, verbose=False)
    with pytest.raises(ValueError, match="unordered"):
        select_cefsplus(X, y, k=1, exclude={"f1"}, verbose=False)


def test_named_integer_labels_are_not_string_coerced():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 4)))
    X.columns = [0, 1, 2, 3]
    y = X[0] + 0.2 * rng.normal(size=40)
    selected = select_mrmr(
        X, y, k=1, task="regression", include=[0], estimator="classic", verbose=False
    )
    assert selected[0] == 0
    with pytest.raises(ValueError, match="unknown"):
        select_mrmr(
            X, y, k=1, task="regression", include=["0"], estimator="classic", verbose=False
        )


def test_evaluate_k_counts_additional_discoveries():
    rng = np.random.default_rng(4)
    X = pd.DataFrame(rng.normal(size=(80, 5)), columns=list("abcde"))
    y = X["b"] + 0.8 * X["c"] + 0.1 * rng.normal(size=80)
    cfg = AutoKConfig(
        k_method="evaluate",
        strategy="time_holdout",
        min_k=1,
        max_k=1,
        val_frac=0.3,
    )
    t = np.arange(len(X))
    classic = select_mrmr(
        X,
        y,
        k="auto",
        task="regression",
        include=["a"],
        auto_k_config=cfg,
        time=t,
        verbose=False,
        estimator="classic",
    )
    assert classic[0] == "a"
    assert len(classic) == 2
    jmi = select_jmi(
        X,
        y,
        k="auto",
        task="regression",
        include=["a"],
        auto_k_config=cfg,
        time=t,
        verbose=False,
        estimator="r2",
    )
    assert jmi[0] == "a"
    assert len(jmi) == 2
    gauss = select_cefsplus(
        X,
        y,
        k="auto",
        include=["a"],
        auto_k_config=cfg,
        time=t,
        verbose=False,
        return_result=True,
    )
    assert gauss.selected_features[0] == "a"
    assert len(gauss.selected_features) == 2
    assert gauss.diagnostics_["auto_k"]["selected_k"] == 1
    zero = select_cefsplus(
        X,
        y,
        k="auto",
        include=["a", "e"],
        auto_k_config=AutoKConfig(
            k_method="evaluate",
            strategy="time_holdout",
            min_k=0,
            max_k=1,
            val_frac=0.3,
        ),
        time=t,
        verbose=False,
        return_result=True,
    )
    assert zero.selected_features[:2] == ["a", "e"]
    assert zero.diagnostics_["auto_k"]["selected_k"] in {0, 1}


def test_mrmr_backend_honored_on_constrained_pool(monkeypatch):
    X, y = _small_regression()
    calls = {"blas": 0, "processes": 0}

    def wrap(fn, key):
        def inner(*args, **kwargs):
            calls[key] += 1
            return fn(*args, **kwargs)

        return inner

    monkeypatch.setattr(
        "sift.selection.loops._mrmr_loop_blas", wrap(_mrmr_loop_blas, "blas")
    )
    monkeypatch.setattr(
        "sift.selection.loops._mrmr_loop_processes",
        wrap(_mrmr_loop_processes, "processes"),
    )
    select_mrmr(
        X,
        y,
        k=2,
        task="regression",
        candidates=["f0", "f1", "f2", "f3"],
        mrmr_backend="processes",
        n_jobs=2,
        verbose=False,
        estimator="classic",
    )
    assert calls["processes"] == 1
    assert calls["blas"] == 0
    select_mrmr(
        X,
        y,
        k=1,
        task="regression",
        include=["f3"],
        mrmr_backend="serial",
        verbose=False,
        estimator="classic",
    )
    assert calls["blas"] == 0


def test_binary_callback_scores_and_failed_include():
    rng = np.random.default_rng(5)
    X = pd.DataFrame(rng.normal(size=(120, 6)), columns=list("abcdef"))
    y = (X["a"] + 0.8 * X["b"] > 0).astype(int)
    steps = []

    def cb(step, total, info=None):
        steps.append((step, total))

    result = select_cefsplus_binary(
        X, y, k=2, include=["c"], verbose=False, subsample=None, callback=cb,
        return_result=True,
    )
    assert steps == [(1, 2), (2, 2)]
    assert result.selected_features[0] == "c"
    ranking = result.get_feature_ranking()
    cond_row = ranking.loc[ranking["feature"] == "c"].iloc[0]
    assert pd.isna(cond_row["score"])
    disc = ranking.loc[ranking["selected"] & (ranking["feature"] != "c"), "score"]
    assert disc.notna().all()
    X_bad = X.copy()
    X_bad["c"] = 1.0
    with pytest.raises(ValueError, match="usable variation|could not be fit|not a valid"):
        select_cefsplus_binary(
            X_bad, y, k=1, include=["c"], verbose=False, subsample=None
        )


def test_binary_evaluate_and_refit_use_include_base():
    rng = np.random.default_rng(6)
    X = pd.DataFrame(rng.normal(size=(90, 5)), columns=list("abcde"))
    y = (X["b"] + 0.7 * X["c"] > 0).astype(int)
    t = np.arange(len(X))
    cfg = AutoKConfig(
        k_method="evaluate",
        strategy="time_holdout",
        min_k=1,
        max_k=1,
        val_frac=0.3,
    )
    selected = select_cefsplus_binary(
        X,
        y,
        k="auto",
        include=["a"],
        auto_k_config=cfg,
        time=t,
        verbose=False,
        subsample=None,
    )
    assert selected[0] == "a"
    assert len(selected) == 2
    cfg_refit = AutoKConfig(
        k_method="penalized_objective",
        max_k=3,
        min_k=1,
        binary_objective_mode="refit",
        objective_penalty="ebic",
    )
    view = select_cefsplus_binary(
        X,
        y,
        k="auto",
        include=["a"],
        auto_k_config=cfg_refit,
        verbose=False,
        subsample=None,
        return_result=True,
    )
    assert view.selected_features[0] == "a"
    Xn = X.to_numpy()
    yn = y.to_numpy().astype(float)
    w = np.ones(len(X))
    disc = [X.columns.get_loc(name) for name in view.selected_features[1:]]
    gains, _ = binary_refit_loglik_gains(
        Xn, yn, w, disc, ridge=1e-4, include_original=[0]
    )
    assert gains.shape[0] == len(disc)


def test_unsupported_auto_k_and_xfit_rejected():
    X, y = _small_regression()
    with pytest.raises(ValueError, match="cannot honor exact"):
        select_cefsplus(
            X,
            y,
            k="auto",
            include=["f0"],
            auto_k_config=AutoKConfig(k_method="xfit_objective", max_k=10, strategy="kfold"),
            verbose=False,
        )
    with pytest.raises(ValueError, match="cannot honor exact"):
        select_cefsplus(
            X,
            y,
            k="auto",
            include=["f0"],
            auto_k_config=AutoKConfig(k_method="stability", max_k=10),
            verbose=False,
        )


def test_select_fdr_conditions_on_include_and_as_result():
    rng = np.random.default_rng(7)
    n, p = 250, 8
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    signal = X["f0"].to_numpy()
    noise = rng.normal(size=n)
    y = 2.0 * signal + 0.2 * rng.normal(size=n)
    X["inc_signal"] = signal + 1e-3 * rng.normal(size=n)
    X["inc_noise"] = noise
    a = select_fdr(
        X,
        y,
        include=["inc_signal"],
        include_provenance="prespecified",
        verbose=False,
        q=0.2,
        candidates=["f0", "f1", "f2", "f3", "f4", "f5"],
    )
    b = select_fdr(
        X,
        y,
        include=["inc_noise"],
        include_provenance="prespecified",
        verbose=False,
        q=0.2,
        candidates=["f0", "f1", "f2", "f3", "f4", "f5"],
    )
    assert not np.allclose(a.W["W"].to_numpy(), b.W["W"].to_numpy())
    assert a.selected_features[0] == "inc_signal"
    assert bool(a.W.loc[a.W["feature"] == "inc_signal", "selected"].iloc[0])
    view = as_result(a, input_features=list(X.columns))
    assert "inc_signal" in view.features
    multi = select_fdr(
        X,
        y,
        include=["inc_noise"],
        include_provenance="prespecified",
        verbose=False,
        q=0.2,
        n_draws=2,
        candidates=["f0", "f1", "f2", "f3"],
    )
    as_result(multi, input_features=list(X.columns))
    freq = float(multi.W.loc[multi.W["feature"] == "inc_noise", "selection_frequency"].iloc[0])
    assert freq == 1.0


def test_select_fdr_constant_target_and_data_derived_metadata():
    rng = np.random.default_rng(8)
    X = pd.DataFrame(rng.normal(size=(80, 5)), columns=list("abcde"))
    y = np.ones(80)
    result = select_fdr(
        X,
        y,
        include=["a"],
        include_provenance="prespecified",
        verbose=False,
    )
    assert result.selected_features == ["a"]
    assert "conditioning" in result.diagnostics_
    assert result.W.loc[result.W["feature"] == "a", "selected"].iloc[0]
    derived = select_fdr(
        X,
        rng.normal(size=80),
        include=["a"],
        include_provenance="data_derived",
        verbose=False,
        q=0.2,
        n_draws=2,
    )
    assert derived.selector_metadata["fdr_control"] == "none"
    assert derived.selector_metadata["exploratory"] is True
    assert derived.selector_metadata["aggregation_preserves_per_draw_fdr"] is False
    cand_only = select_fdr(
        X,
        rng.normal(size=80) + X["b"],
        candidates=["b", "c"],
        include_provenance="prespecified",
        verbose=False,
        q=0.2,
    )
    assert set(cand_only.selected_features) <= {"b", "c"}
    excl = select_fdr(
        X,
        rng.normal(size=80) + X["b"],
        exclude=["b"],
        include_provenance="prespecified",
        verbose=False,
        q=0.2,
    )
    assert "b" not in excl.selected_features


def test_proxy_rebuild_keeps_include():
    rng = np.random.default_rng(9)
    X = pd.DataFrame(rng.normal(size=(60, 20)), columns=[f"f{i}" for i in range(20)])
    y = X["f0"] + 0.01 * X["f19"] + 0.05 * rng.normal(size=60)
    view = select_cefsplus(
        X,
        y,
        k=1,
        include=["f19"],
        top_m=2,
        verbose=False,
        return_result=True,
        store_proxies=True,
        subsample=None,
    )
    assert view.selected_features[0] == "f19"
    frame = as_result(view, input_features=list(X.columns)).proxies("f19")
    assert frame is not None


def test_wrappers_clone_transform_and_ksg():
    X, y = _small_regression()
    for cls in (CEFSPlusSelector, MRMRSelector, JMISelector, JMIMSelector):
        kwargs = {"k": 2, "include": ["f0"], "verbose": False}
        if cls is not CEFSPlusSelector:
            kwargs["task"] = "regression"
        est = cls(**kwargs)
        cloned = clone(est)
        assert cloned.include == ["f0"]
        fitted = est.fit(X, y)
        assert fitted.selected_features_[0] == "f0"
        Xt = fitted.transform(X)
        assert Xt.shape[1] == len(fitted.selected_features_)
        assert list(fitted.get_feature_names_out())[0] == "f0"
    yb = (y > y.median()).astype(int)
    bin_est = CEFSPlusBinarySelector(k=1, include=["f0"], verbose=False).fit(X, yb)
    assert bin_est.selected_features_[0] == "f0"
    ksg = select_jmi(
        X.head(40),
        y.head(40),
        k=1,
        task="regression",
        include=["f0"],
        estimator="ksg",
        verbose=False,
        subsample=None,
    )
    assert ksg[0] == "f0"
    brier = select_cefsplus_binary(
        X, yb, k=1, include=["f0"], loss="brier", verbose=False, subsample=None
    )
    assert brier[0] == "f0"
    knock = KnockoffSelector(
        include=["f7"] if "f7" in X.columns else ["f3"],
        include_provenance="sample_split",
        verbose=False,
        q=0.2,
    )
    knock.fit(X, y)
    assert knock.selected_features_[0] in {"f7", "f3"}


def test_constant_include_rejected_consistently():
    X, y = _small_regression()
    X = X.copy()
    X["f0"] = 1.0
    with pytest.raises(ValueError, match="usable variation|not present|not a valid"):
        select_mrmr(X, y, k=1, task="regression", include=["f0"], verbose=False)
    with pytest.raises(ValueError, match="usable variation|not present|not a valid"):
        select_cefsplus(X, y, k=1, include=["f0"], verbose=False, subsample=None)
    y_bin = (y > np.median(y)).astype(int)
    with pytest.raises(ValueError, match="usable variation|not present|not a valid"):
        select_cefsplus_binary(X, y_bin, k=1, include=["f0"], verbose=False, subsample=None)


def test_cache_named_positional_and_select_cached_tail():
    X, y = _small_regression()
    cache = build_cache(X, compute_Rxx=True, subsample=None)
    named = select_cached(cache, y, k=1, include=["f0"], top_m=10)
    assert named[0] == "f0"
    Xn = X.to_numpy()
    cache_p = build_cache(Xn, compute_Rxx=True, subsample=None)
    pos = select_cached(cache_p, y.to_numpy(), k=1, include=[0], top_m=10)
    assert pos[0] in {0, "x0"}
    params = list(inspect.signature(select_cached).parameters)
    assert params[-3:] == ["include", "exclude", "candidates"]
    assert "compose_include" not in params
    assert params.index("callback") == 9


def test_omitted_conditioning_is_noop_parity():
    X, y = _small_regression()
    baseline = select_cefsplus(X, y, k=3, verbose=False, return_result=True)
    explicit = select_cefsplus(
        X,
        y,
        k=3,
        include=None,
        exclude=None,
        candidates=None,
        verbose=False,
        return_result=True,
    )
    assert baseline.selected_features == explicit.selected_features
    assert baseline.selected_indices == explicit.selected_indices
    assert "conditioning" not in (baseline.diagnostics_ or {})
    assert "conditioning" not in baseline.selector_metadata
    cache = build_cache(X, compute_Rxx=True, subsample=None)
    assert select_cached(cache, y, k=3) == select_cached(
        cache, y, k=3, include=None, exclude=None, candidates=None
    )
    assert select_mrmr(X, y, k=3, task="regression", verbose=False) == select_mrmr(
        X,
        y,
        k=3,
        task="regression",
        include=None,
        exclude=None,
        candidates=None,
        verbose=False,
    )
    for n_draws in (1, 3):
        fdr_a = select_fdr(X, y, verbose=False, q=0.2, n_draws=n_draws, random_state=0)
        fdr_b = select_fdr(
            X,
            y,
            verbose=False,
            q=0.2,
            n_draws=n_draws,
            random_state=0,
            include=None,
            exclude=None,
            candidates=None,
        )
        assert fdr_a.selected_features == fdr_b.selected_features
        assert fdr_a.selected_indices == fdr_b.selected_indices
        assert fdr_a.threshold == fdr_b.threshold
        assert fdr_a.W.equals(fdr_b.W)
        pd.testing.assert_frame_equal(fdr_a.W, fdr_b.W, check_exact=True)
        for column in ("W", "relevance"):
            np.testing.assert_array_equal(
                fdr_a.W[column].to_numpy(),
                fdr_b.W[column].to_numpy(),
            )
        draw_cols = [c for c in fdr_a.W.columns if str(c).startswith("W_draw_")]
        for column in draw_cols:
            np.testing.assert_array_equal(
                fdr_a.W[column].to_numpy(),
                fdr_b.W[column].to_numpy(),
            )
        assert fdr_a.selector_metadata == fdr_b.selector_metadata
        if fdr_a.selection_frequency is None:
            assert fdr_b.selection_frequency is None
        else:
            assert fdr_a.selection_frequency.equals(fdr_b.selection_frequency)
            np.testing.assert_array_equal(
                fdr_a.selection_frequency.to_numpy(),
                fdr_b.selection_frequency.to_numpy(),
            )
        assert "conditioning" not in fdr_a.selector_metadata
        assert "exploratory" not in fdr_a.selector_metadata


def test_fdr_dropped_provenance_not_false_zero_variance():
    rng = np.random.default_rng(11)
    X = pd.DataFrame(rng.normal(size=(120, 6)), columns=list("abcdef"))
    y = 1.5 * X["b"] + 0.2 * rng.normal(size=120)
    result = select_fdr(
        X,
        y,
        include=["a"],
        exclude=["e"],
        candidates=["b", "c", "d"],
        include_provenance="prespecified",
        verbose=False,
        q=0.2,
    )
    meta = result.selector_metadata
    assert meta["n_zero_weight_variance_features"] == 0
    reasons = meta.get("dropped_feature_reasons", [])
    assert "zero_weight_variance" not in reasons
    positions = meta.get("dropped_feature_positions", [])
    assert len(positions) == len(reasons)
    healthy = {list(X.columns).index(name) for name in list("abcdef")}
    assert healthy.isdisjoint(positions)


def test_fdr_residual_zero_uses_truthful_reason():
    rng = np.random.default_rng(11)
    X = pd.DataFrame(rng.normal(size=(120, 4)), columns=list("abcd"))
    X["copy_a"] = X["a"]
    y = 1.5 * X["b"] + 0.2 * rng.normal(size=120)
    result = select_fdr(
        X,
        y,
        include=["a"],
        candidates=["b", "c", "d", "copy_a"],
        include_provenance="prespecified",
        verbose=False,
        q=0.2,
        subsample=None,
    )
    meta = result.selector_metadata
    assert meta["n_zero_weight_variance_features"] == 0
    reasons = meta.get("dropped_feature_reasons", [])
    positions = meta.get("dropped_feature_positions", [])
    copy_pos = list(X.columns).index("copy_a")
    assert copy_pos in positions
    assert reasons[positions.index(copy_pos)] == "zero_residual_variance"
    assert "zero_weight_variance" not in reasons


def test_fdr_constant_target_discovery_roles():
    rng = np.random.default_rng(12)
    X = pd.DataFrame(rng.normal(size=(80, 5)), columns=list("abcde"))
    result = select_fdr(
        X,
        np.ones(80),
        include=["a"],
        candidates=["b", "c"],
        include_provenance="prespecified",
        verbose=False,
    )
    roles = dict(zip(result.W["feature"].tolist(), result.W["role"].tolist()))
    assert roles["a"] == "include"
    assert roles["b"] == "discovery"
    assert roles["c"] == "discovery"
    assert roles["d"] == "ineligible"
    assert roles["e"] == "ineligible"


def test_fdr_constant_target_residual_zero_not_discovery():
    rng = np.random.default_rng(12)
    X = pd.DataFrame(rng.normal(size=(80, 3)), columns=list("abc"))
    X["copy_a"] = X["a"]
    y = np.ones(80)
    for n_draws in (1, 3):
        result = select_fdr(
            X,
            y,
            include=["a"],
            candidates=["b", "copy_a"],
            include_provenance="prespecified",
            verbose=False,
            n_draws=n_draws,
            subsample=None,
        )
        meta = result.selector_metadata
        copy_pos = list(X.columns).index("copy_a")
        positions = meta.get("dropped_feature_positions", [])
        reasons = meta.get("dropped_feature_reasons", [])
        assert copy_pos in positions
        assert reasons[positions.index(copy_pos)] == "zero_residual_variance"
        roles = dict(zip(result.W["feature"].tolist(), result.W["role"].tolist()))
        assert roles["a"] == "include"
        assert roles["b"] == "discovery"
        assert roles["copy_a"] == "ineligible"
        dropped = {
            pos: reason for pos, reason in zip(positions, reasons)
        }
        for feature, role in roles.items():
            pos = list(X.columns).index(feature)
            if dropped.get(pos) == "zero_residual_variance":
                assert role != "discovery"


def test_fdr_singular_include_rejected_ordinary_correlation_kept():
    rng = np.random.default_rng(13)
    X = pd.DataFrame(rng.normal(size=(150, 4)), columns=list("abcd"))
    y = X["b"] + 0.2 * rng.normal(size=150)
    X["copy_a"] = X["a"]
    with pytest.raises(ValueError, match="singular"):
        select_fdr(
            X,
            y,
            include=["a", "copy_a"],
            candidates=["b", "c", "d"],
            include_provenance="prespecified",
            verbose=False,
            q=0.2,
            subsample=None,
        )
    X2 = pd.DataFrame(rng.normal(size=(150, 4)), columns=list("abcd"))
    y2 = X2["b"] + 0.2 * rng.normal(size=150)
    X2["corr"] = 0.55 * X2["a"] + 0.45 * rng.normal(size=150)
    kept = select_fdr(
        X2,
        y2,
        include=["a", "corr"],
        candidates=["b", "c", "d"],
        include_provenance="prespecified",
        verbose=False,
        q=0.2,
        subsample=None,
    )
    assert kept.selected_features[0] in {"a", "corr"}
    X3 = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
    y3 = X3["b"] + 0.2 * rng.normal(size=200)
    X3["near"] = 2.0 * X3["a"] + 3.0
    with pytest.raises(ValueError, match="singular"):
        select_fdr(
            X3,
            y3,
            include=["a", "near"],
            candidates=["b", "c", "d"],
            include_provenance="prespecified",
            verbose=False,
            q=0.2,
            subsample=None,
        )


def test_evaluate_encodes_included_categorical():
    rng = np.random.default_rng(14)
    n = 60
    X = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "cat": rng.choice(["x", "y", "z"], size=n),
        }
    )
    y = X["a"] + 0.3 * rng.normal(size=n)
    cfg = AutoKConfig(
        k_method="evaluate",
        strategy="time_holdout",
        min_k=1,
        max_k=1,
        val_frac=0.3,
    )
    t = np.arange(n)
    selected = select_mrmr(
        X,
        y,
        k="auto",
        task="regression",
        estimator="classic",
        cat_encoding="target_cv",
        include=["cat"],
        auto_k_config=cfg,
        time=t,
        verbose=False,
    )
    assert selected[0] == "cat"
    explicit = select_mrmr(
        X,
        y,
        k="auto",
        task="regression",
        estimator="classic",
        cat_encoding="target_cv",
        cat_features=["cat"],
        include=["cat"],
        auto_k_config=cfg,
        time=t,
        verbose=False,
    )
    assert explicit[0] == "cat"
    gaussian = select_mrmr(
        X,
        y,
        k="auto",
        task="regression",
        estimator="gaussian",
        cat_encoding="target_cv",
        include=["cat"],
        auto_k_config=cfg,
        time=t,
        verbose=False,
        subsample=None,
    )
    assert gaussian[0] == "cat"
    y_bin = (X["a"] > 0).astype(int)
    binary = select_cefsplus_binary(
        X,
        y_bin,
        k="auto",
        cat_encoding="target_cv",
        include=["cat"],
        auto_k_config=cfg,
        time=t,
        verbose=False,
        subsample=None,
    )
    assert binary[0] == "cat"
    binary_explicit = select_cefsplus_binary(
        X,
        y_bin,
        k="auto",
        cat_encoding="target_cv",
        cat_features=["cat"],
        include=["cat"],
        auto_k_config=cfg,
        time=t,
        verbose=False,
        subsample=None,
    )
    assert binary_explicit[0] == "cat"


def test_gain_test_counts_discovery_universe_only():
    from sift.selection.filter_auto_k_common import _gain_test_candidate_inputs

    rng = np.random.default_rng(15)
    X = pd.DataFrame(rng.normal(size=(80, 6)), columns=list("abcdef"))
    y = X["c"] + 0.2 * rng.normal(size=80)
    cache = build_cache(X, compute_Rxx=True, subsample=None)
    unused = {"include": ["a"], "exclude": None, "candidates": ["c", "d", "e"]}
    counts = {}
    for mode in ("all", "panel", "li_ji"):
        cfg = AutoKConfig(k_method="chi2_stop", max_k=3, min_k=1, m_mode=mode)
        p, eigs = _gain_test_candidate_inputs(
            cache, y.to_numpy(), 3, 10, None, "cefsplus", cfg, unused=unused
        )
        counts[mode] = (p, None if eigs is None else int(eigs.shape[0]))
    assert counts["all"] == (3, None)
    assert counts["panel"] == (3, None)
    assert counts["li_ji"][0] == 3
    assert counts["li_ji"][1] == 3
    view = select_cefsplus(
        X,
        y,
        k="auto",
        include=["a"],
        candidates=["c", "d", "e"],
        auto_k_config=AutoKConfig(k_method="chi2_stop", max_k=3, min_k=1, m_mode="panel"),
        verbose=False,
        return_result=True,
        subsample=None,
    )
    assert view.selected_features[0] == "a"
    assert view.diagnostics_["auto_k"]["m_mode"] == "panel"
    assert view.diagnostics_["auto_k"]["p_candidates"] == 3
    chi2_diag = view.diagnostics_["auto_k_diagnostics"]
    assert int(chi2_diag["m_eff"].iloc[0]) == 3
    forward = select_cefsplus(
        X,
        y,
        k="auto",
        include=["a"],
        candidates=["c", "d", "e"],
        auto_k_config=AutoKConfig(k_method="forward_stop", max_k=3, min_k=1, m_mode="li_ji"),
        verbose=False,
        return_result=True,
        subsample=None,
    )
    assert forward.diagnostics_["auto_k"]["p_candidates"] == 3
    assert forward.diagnostics_["auto_k_diagnostics"]["m_eff"].iloc[0] == pytest.approx(3.0, abs=1e-6)


def test_conditioned_non_cefs_objective_matches_schur():
    rng = np.random.default_rng(16)
    X = pd.DataFrame(rng.normal(size=(100, 6)), columns=list("abcdef"))
    y = X["c"] + 0.7 * X["d"] + 0.05 * rng.normal(size=100)
    cache = build_cache(X, compute_Rxx=True, subsample=None)
    from sift.selection.panel import build_candidate_panel

    for method in ("mrmr_diff", "mrmr_quot", "jmi", "jmim"):
        names, obj = select_cached(
            cache, y, k=2, method=method, include=["b"], top_m=6, corr_prune=None,
            return_objective=True,
        )
        assert names[0] == "b"
        assert obj.shape == (2,)
        panel = build_candidate_panel(
            cache, y.to_numpy(), 2, top_m=6, corr_prune=None, method=method,
            protect_valid=np.array([list(cache.feature_names).index("b")]),
            pool_valid=np.arange(len(cache.valid_cols)),
        )
        include_local = int(np.flatnonzero(panel.original == list(X.columns).index("b"))[0])
        disc_orig = [list(X.columns).index(name) for name in names[1:]]
        disc_local = [
            int(np.flatnonzero(panel.original == orig)[0]) for orig in disc_orig
        ]
        expected = []
        base = _schur_objective(panel.R, panel.r, [include_local])
        acc = [include_local]
        for loc in disc_local:
            acc.append(loc)
            expected.append(_schur_objective(panel.R, panel.r, acc) - base)
        assert np.allclose(obj, expected, rtol=1e-4, atol=1e-6)
    elbow = select_mrmr(
        X,
        y,
        k="auto",
        task="regression",
        estimator="gaussian",
        include=["b"],
        auto_k_config=AutoKConfig(k_method="elbow", max_k=3, min_k=1),
        verbose=False,
        return_result=True,
        subsample=None,
    )
    assert elbow.selected_features[0] == "b"
    assert len(elbow.selected_features) >= 2


def _oracle_schur_corr_eigs(R, n_protect, shrink=1e-6):
    """Independent Schur-conditional correlation eigenvalues; not the production helper."""
    G = (1.0 - shrink) * np.asarray(R, dtype=np.float64)
    np.fill_diagonal(G, 1.0)
    rss = G[:n_protect, :n_protect]
    rsd = G[:n_protect, n_protect:]
    rdd = G[n_protect:, n_protect:]
    schur = rdd - rsd.T @ np.linalg.solve(rss, rsd)
    schur = 0.5 * (schur + schur.T)
    scale = np.sqrt(np.clip(np.diag(schur), 0.0, np.inf))
    inv = np.divide(1.0, scale, out=np.zeros_like(scale), where=scale > 1e-12)
    corr = schur * inv[:, None] * inv[None, :]
    np.fill_diagonal(corr, np.where(scale > 1e-12, 1.0, 0.0))
    return np.linalg.eigvalsh(corr)


def _li_ji_m(eigs):
    eigs = np.asarray(eigs, dtype=np.float64)
    parts = np.where(eigs >= 1.0, 1.0 + (eigs - np.floor(eigs)), eigs)
    return float(max(1.0, np.sum(parts)))


def test_li_ji_uses_schur_conditional_discovery_correlation():
    from sift.selection.filter_auto_k_common import (
        _conditioning_valid_sets,
        _gain_test_candidate_inputs,
    )
    from sift.selection.panel import build_candidate_panel

    rng = np.random.default_rng(21)
    n = 200
    s = rng.normal(size=n)
    noise = rng.normal(size=(n, 8))
    X = pd.DataFrame({"s": s})
    for i in range(8):
        X[f"d{i}"] = s + 0.2 * noise[:, i]
    X["z"] = rng.normal(size=n)
    y = X["z"] + 0.05 * rng.normal(size=n)
    cache = build_cache(X, compute_Rxx=True, subsample=None)
    unused = {
        "include": ["s"],
        "exclude": None,
        "candidates": [f"d{i}" for i in range(8)],
    }
    cfg = AutoKConfig(k_method="chi2_stop", max_k=3, min_k=1, m_mode="li_ji")
    p, eigs = _gain_test_candidate_inputs(
        cache, y.to_numpy(), 3, 12, None, "cefsplus", cfg, unused=unused
    )
    assert p == 8
    protect, pool = _conditioning_valid_sets(cache, unused)
    panel = build_candidate_panel(
        cache, y.to_numpy(), 3, top_m=12, corr_prune=None, method="cefsplus",
        protect_valid=protect, pool_valid=pool,
    )
    n_protect = int(protect.size)
    marginal = np.linalg.eigvalsh(panel.R[n_protect:, n_protect:])
    expected = _oracle_schur_corr_eigs(panel.R, n_protect)
    np.testing.assert_allclose(np.sort(eigs), np.sort(expected), rtol=1e-8, atol=1e-10)
    m_cond = _li_ji_m(expected)
    m_marg = _li_ji_m(marginal)
    assert abs(m_cond - m_marg) > 1.0
    omitted = {"include": None, "exclude": None, "candidates": None}
    _p0, eigs0 = _gain_test_candidate_inputs(
        cache, y.to_numpy(), 3, 12, None, "cefsplus", cfg, unused=omitted
    )
    panel0 = build_candidate_panel(
        cache, y.to_numpy(), 3, top_m=12, corr_prune=None, method="cefsplus"
    )
    np.testing.assert_array_equal(eigs0, np.linalg.eigvalsh(panel0.R))
    view = select_cefsplus(
        X,
        y,
        k="auto",
        include=["s"],
        candidates=[f"d{i}" for i in range(8)],
        auto_k_config=AutoKConfig(k_method="chi2_stop", max_k=3, min_k=1, m_mode="li_ji"),
        verbose=False,
        return_result=True,
        subsample=None,
        top_m=12,
    )
    m_eff0 = float(view.diagnostics_["auto_k_diagnostics"]["m_eff"].iloc[0])
    assert abs(m_eff0 - m_cond) < abs(m_eff0 - m_marg)
    assert m_eff0 == pytest.approx(m_cond, rel=1e-8, abs=1e-8)


def test_binary_constant_include_and_unusable_candidates():
    rng = np.random.default_rng(22)
    n = 60
    const_X = pd.DataFrame({"a": np.ones(n), "b": np.ones(n)})
    y_bin = np.array([0, 1] * (n // 2))
    omitted = select_cefsplus_binary(const_X, y_bin, k=1, verbose=False, subsample=None)
    explicit_none = select_cefsplus_binary(
        const_X, y_bin, k=1, include=None, exclude=None, candidates=None,
        verbose=False, subsample=None,
    )
    assert omitted == explicit_none
    with pytest.raises(ValueError, match="not a valid|usable variation"):
        select_cefsplus_binary(
            const_X, y_bin, k=1, include=["a"], verbose=False, subsample=None
        )
    X = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "const": np.ones(n),
            "b": rng.normal(size=n),
        }
    )
    y = (X["a"] > 0).astype(int)
    with pytest.raises(ValueError, match="no usable discovery|no valid cache"):
        select_cefsplus_binary(
            X, y, k=1, include=["a"], candidates=["const"], verbose=False, subsample=None
        )
    with pytest.raises(ValueError, match="no usable discovery|no valid cache"):
        select_mrmr(
            X, y.astype(float), k=1, task="regression", include=["a"],
            candidates=["const"], verbose=False,
        )
    with pytest.raises(ValueError, match="no usable discovery|no valid cache"):
        select_cefsplus(
            X, X["a"], k=1, include=["a"], candidates=["const"],
            verbose=False, subsample=None,
        )
    include_only = select_cefsplus_binary(
        X, y, k=1, include=["a"], verbose=False, subsample=None
    )
    assert include_only[0] == "a"


def test_binary_candidate_diagnostics_exclude_include():
    rng = np.random.default_rng(17)
    X = pd.DataFrame(rng.normal(size=(80, 5)), columns=list("abcde"))
    y = (X["b"] > 0).astype(int)
    result = select_cefsplus_binary(
        X, y, k=2, include=["a"], verbose=False, subsample=None, return_result=True
    )
    assert result.selected_features[0] == "a"
    cand = result.diagnostics_["candidate_indices"]
    include_pos = list(X.columns).index("a")
    assert include_pos not in cand
    assert result.diagnostics_["n_screened_features"] == len(cand)
    assert "a" in result.diagnostics_["conditioning"]["include"]


def test_candidate_list_permutation_does_not_change_path():
    rng = np.random.default_rng(18)
    X = pd.DataFrame(rng.normal(size=(90, 5)), columns=list("abcde"))
    tied = rng.normal(size=90)
    X["c"] = tied
    X["d"] = tied
    X["e"] = tied
    y = tied + 0.2 * rng.normal(size=90)
    a = select_cefsplus(
        X, y, k=2, candidates=["e", "c", "d"], verbose=False, subsample=None, top_m=10
    )
    b = select_cefsplus(
        X, y, k=2, candidates=["d", "e", "c"], verbose=False, subsample=None, top_m=10
    )
    assert a == b

