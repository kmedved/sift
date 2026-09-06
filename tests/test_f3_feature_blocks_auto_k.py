"""Block-aware auto-k: prefix boundaries, penalties, folds, singleton parity."""

from __future__ import annotations

from math import comb, log
import warnings

import numpy as np
import pandas as pd
import pytest

from sift import (
    AutoKConfig,
    CEFSPlusSelector,
    as_result,
    build_cache,
    gaussian_cv_curves,
    select_cefsplus,
    select_k_auto,
    select_mrmr,
    xfit_objective_curves,
)
from sift.selection.auto_k import select_k_penalized_objective
from sift.selection.blocks import (
    SUPPORTED_BLOCK_AUTO_K,
    UNSUPPORTED_BLOCK_AUTO_K,
    gaussian_copula_prefix_df,
    require_block_auto_k,
)


def _block_frame(n=160, seed=3):
    rng = np.random.default_rng(seed)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "ab__0": z1 + 0.05 * rng.normal(size=n),
            "ab__1": z1 + 0.2 * rng.normal(size=n),
            "c": z2,
            "n0": rng.normal(size=n),
            "n1": rng.normal(size=n),
        }
    )
    y = z1 + 0.8 * z2 + 0.15 * rng.normal(size=n)
    blocks = {"ab": ["ab__0", "ab__1"]}
    return X, y, blocks


def test_evaluate_selects_complete_unequal_block_prefixes():
    X, y, blocks = _block_frame()
    time = np.arange(len(X))
    cfg = AutoKConfig(
        k_method="evaluate",
        strategy="time_holdout",
        min_k=1,
        max_k=3,
        val_frac=0.3,
        selection_rule="best",
    )
    result = select_cefsplus(
        X,
        y,
        k="auto",
        auto_k_config=cfg,
        time=time,
        feature_blocks=blocks,
        subsample=None,
        verbose=False,
        return_result=True,
    )
    names = result.selected_features
    if "ab__0" in names or "ab__1" in names:
        assert {"ab__0", "ab__1"} <= set(names)
    md = result.selector_metadata
    assert md["n_columns_selected"] == len(names)
    assert md["n_blocks_selected"] == md["k"]
    diag = result.diagnostics_["auto_k_diagnostics"]
    assert int(diag["k"].max()) <= 3
    assert not (diag["k"] > 3).any()
    view = as_result(result, input_features=list(X.columns))
    assert view.k == len(names)


def test_penalized_ebic_uses_block_multiplicity_not_raw_width():
    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=list("abcdef"))
    y = 0.2 * X.a + 0.2 * X.b + 0.1 * X.c + rng.normal(size=n)
    blocks = {"ab": ["a", "b"], "cde": ["c", "d", "e"]}
    cfg = AutoKConfig(
        k_method="penalized_objective",
        objective_penalty="ebic",
        ebic_gamma=0.5,
        min_k=0,
        max_k=3,
    )
    blocked = select_cefsplus(
        X,
        y,
        k="auto",
        auto_k_config=cfg,
        feature_blocks=blocks,
        subsample=None,
        verbose=False,
        return_result=True,
    )
    diag = blocked.diagnostics_["auto_k_diagnostics"]
    b = int(diag["n_candidates"].iloc[0])
    gamma = float(diag["ebic_gamma"].iloc[0])
    n_eff = float(diag["n_eff"].iloc[0])
    assert b == 3
    for _, row in diag[diag["k"] > 0].iterrows():
        k = int(row["k"])
        df = float(row["df"])
        expected = df * log(n_eff) + 2.0 * gamma * log(comb(b, k))
        assert float(row["penalty"]) == pytest.approx(expected, rel=1e-12, abs=1e-9)


def test_gaussian_copula_df_excludes_constant_padding():
    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "const": np.ones(n),
        }
    )
    y = X["a"] + 0.1 * rng.normal(size=n)
    from sift import build_cache

    cache = build_cache(X, compute_Rxx=True, subsample=None)
    widths = (2, 3)
    path = [0, 1, 2]
    dfs = gaussian_copula_prefix_df(cache, path, widths)
    assert dfs[0] == pytest.approx(2.0, abs=1e-9)
    assert dfs[1] == pytest.approx(2.0, abs=1e-9)


def test_unsupported_calibrated_block_auto_k_errors():
    X, y, blocks = _block_frame()
    with pytest.raises(ValueError, match="scalar column steps"):
        select_cefsplus(
            X,
            y,
            k="auto",
            auto_k_config=AutoKConfig(k_method="perm_gap", max_k=3),
            feature_blocks=blocks,
            subsample=None,
            verbose=False,
        )
    require_block_auto_k("evaluate")
    with pytest.raises(ValueError, match="k_method='stability'"):
        require_block_auto_k("stability")


def test_singleton_auto_k_matches_no_block():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(120, 5)), columns=list("abcde"))
    y = X.a + 0.5 * X.b + 0.1 * rng.normal(size=len(X))
    time = np.arange(len(X))
    cfg = AutoKConfig(
        k_method="evaluate",
        strategy="time_holdout",
        min_k=1,
        max_k=3,
        val_frac=0.25,
        selection_rule="best",
    )
    kwargs = dict(
        k="auto",
        auto_k_config=cfg,
        time=time,
        subsample=None,
        verbose=False,
    )
    plain = select_cefsplus(X, y, **kwargs)
    blocked = select_cefsplus(
        X, y, feature_blocks={c: [c] for c in reversed(X.columns)}, **kwargs
    )
    assert blocked == plain


def test_auto_k_weights_and_include_keep_complete_blocks():
    X, y, blocks = _block_frame()
    w = np.linspace(0.4, 1.6, len(X))
    time = np.arange(len(X))
    cfg = AutoKConfig(
        k_method="evaluate",
        strategy="time_holdout",
        min_k=1,
        max_k=2,
        val_frac=0.3,
        selection_rule="best",
    )
    result = select_cefsplus(
        X,
        y,
        k="auto",
        auto_k_config=cfg,
        time=time,
        sample_weight=w,
        include=["ab__0", "ab__1"],
        feature_blocks=blocks,
        subsample=None,
        verbose=False,
        return_result=True,
    )
    assert {"ab__0", "ab__1"} <= set(result.selected_features)
    md = result.selector_metadata
    assert md["k"] == md["n_blocks_selected"]
    assert md["k"] == result.diagnostics_["auto_k"]["selected_k"]
    assert md["n_blocks_selected_total"] == md["n_blocks_selected"] + 1
    assert md["n_columns_selected"] == len(result.selected_features)
    assert "ab" not in md["selected_blocks"]


def test_nested_refit_uses_block_k():
    X, y, blocks = _block_frame()
    groups = np.repeat(np.arange(8), len(X) // 8 + 1)[: len(X)]
    cfg = AutoKConfig(
        k_method="evaluate",
        auto_k_mode="nested",
        strategy="group_cv",
        n_splits=4,
        min_k=1,
        max_k=2,
        selection_rule="best",
    )
    selector = CEFSPlusSelector(
        k="auto",
        auto_k_config=cfg,
        feature_blocks=blocks,
        subsample=None,
        verbose=False,
    )
    selector.fit(X, y, groups=groups)
    assert selector.k_ in {1, 2}
    names = list(selector.selected_features_)
    if "ab__0" in names or "ab__1" in names:
        assert {"ab__0", "ab__1"} <= set(names)
    Xt = selector.transform(X)
    assert Xt.shape[1] == len(names)


def test_penalized_no_block_math_untouched():
    rng = np.random.default_rng(0)
    obj = np.cumsum(rng.uniform(0.05, 0.2, size=6))
    cfg = AutoKConfig(
        k_method="penalized_objective",
        objective_penalty="ebic",
        min_k=0,
        max_k=6,
    )
    k1, d1 = select_k_penalized_objective(
        obj, cfg, objective_scale=50.0, n_samples=50, n_candidates=10
    )
    k2, d2 = select_k_penalized_objective(
        obj, cfg, objective_scale=50.0, n_samples=50, n_candidates=10, df_path=None
    )
    assert k1 == k2
    assert d1["penalized_score"].tolist() == d2["penalized_score"].tolist()


def test_select_k_auto_prefix_sizes_stay_in_block_units():
    X, y, _blocks = _block_frame()
    cfg = AutoKConfig(
        k_method="evaluate",
        strategy="time_holdout",
        min_k=1,
        max_k=3,
        val_frac=0.3,
        selection_rule="best",
    )
    best_k, features, diag = select_k_auto(
        X,
        np.asarray(y),
        ["ab__0", "ab__1", "c", "n0"],
        cfg,
        time=np.arange(len(X)),
        prefix_sizes=(2, 3, 4),
    )
    assert best_k in {1, 2, 3}
    assert set(diag["k"].astype(int)) <= {1, 2, 3}
    assert int(diag["k"].max()) <= 3
    assert len(features) in {2, 3, 4}
    if "ab__0" in features or "ab__1" in features:
        assert features[:2] == ["ab__0", "ab__1"]


def test_elbow_and_gaussian_cv_keep_complete_blocks():
    X, y, blocks = _block_frame()
    time = np.arange(len(X))
    elbow = select_cefsplus(
        X,
        y,
        k="auto",
        auto_k_config=AutoKConfig(k_method="elbow", min_k=1, max_k=3),
        feature_blocks=blocks,
        subsample=None,
        verbose=False,
        return_result=True,
    )
    names = elbow.selected_features
    if "ab__0" in names or "ab__1" in names:
        assert {"ab__0", "ab__1"} <= set(names)
    cv = select_cefsplus(
        X,
        y,
        k="auto",
        auto_k_config=AutoKConfig(
            k_method="gaussian_cv",
            strategy="time_holdout",
            min_k=1,
            max_k=3,
            val_frac=0.3,
            selection_rule="best",
        ),
        time=time,
        feature_blocks=blocks,
        subsample=None,
        verbose=False,
        return_result=True,
    )
    cv_names = cv.selected_features
    if "ab__0" in cv_names or "ab__1" in cv_names:
        assert {"ab__0", "ab__1"} <= set(cv_names)
    xfit = select_cefsplus(
        X,
        y,
        k="auto",
        auto_k_config=AutoKConfig(
            k_method="xfit_objective",
            strategy="kfold",
            xfit_folds=4,
            min_k=1,
            max_k=3,
            selection_rule="one_se",
        ),
        feature_blocks=blocks,
        subsample=None,
        verbose=False,
        return_result=True,
    )
    xf_names = xfit.selected_features
    if "ab__0" in xf_names or "ab__1" in xf_names:
        assert {"ab__0", "ab__1"} <= set(xf_names)


def test_default_auto_routing_and_wrapper_cache_alignment():
    X, y, blocks = _block_frame()
    time = np.arange(len(X))
    routed = select_cefsplus(
        X,
        y,
        k="auto",
        feature_blocks=blocks,
        subsample=None,
        verbose=False,
        return_result=True,
    )
    summary = routed.diagnostics_["auto_k"]
    assert summary["method"] in SUPPORTED_BLOCK_AUTO_K
    assert summary["method"] != "perm_gap"
    names = routed.selected_features
    if "ab__0" in names or "ab__1" in names:
        assert {"ab__0", "ab__1"} <= set(names)
    cfg = AutoKConfig(
        k_method="evaluate",
        strategy="time_holdout",
        min_k=1,
        max_k=2,
        val_frac=0.3,
        selection_rule="best",
    )
    fn = select_cefsplus(
        X,
        y,
        k="auto",
        auto_k_config=cfg,
        time=time,
        feature_blocks=blocks,
        subsample=None,
        verbose=False,
        return_result=True,
    )
    wrapper = CEFSPlusSelector(
        k="auto",
        auto_k_config=cfg,
        feature_blocks=blocks,
        subsample=None,
        verbose=False,
    )
    wrapper.fit(X, y, time=time)
    assert list(wrapper.selected_features_) == list(fn.selected_features)
    cache = build_cache(X, compute_Rxx=True, subsample=None)
    cached = select_cefsplus(
        X,
        y,
        k="auto",
        auto_k_config=cfg,
        time=time,
        cache=cache,
        feature_blocks=blocks,
        verbose=False,
        return_result=True,
    )
    assert list(cached.selected_features) == list(fn.selected_features)
    classic = select_mrmr(
        X,
        y,
        k="auto",
        auto_k_config=cfg,
        time=time,
        task="regression",
        feature_blocks=blocks,
        subsample=None,
        verbose=False,
        return_result=True,
    )
    cnames = classic.selected_features
    if "ab__0" in cnames or "ab__1" in cnames:
        assert {"ab__0", "ab__1"} <= set(cnames)


def test_bic_df_is_model_dimension_ebic_uses_block_k():
    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "const": np.ones(n),
        }
    )
    y = X["a"] + 0.1 * rng.normal(size=n)
    blocks = {"ab": ["a", "b"], "z": ["const"]}
    bic = select_cefsplus(
        X,
        y,
        k="auto",
        auto_k_config=AutoKConfig(
            k_method="penalized_objective",
            objective_penalty="bic",
            min_k=1,
            max_k=2,
        ),
        feature_blocks=blocks,
        subsample=None,
        verbose=False,
        return_result=True,
    )
    bdiag = bic.diagnostics_["auto_k_diagnostics"]
    row1 = bdiag[bdiag["k"] == 1].iloc[0]
    assert int(row1["k"]) == 1
    assert float(row1["df"]) == pytest.approx(2.0, abs=1e-8)
    ebic = select_cefsplus(
        X,
        y,
        k="auto",
        auto_k_config=AutoKConfig(
            k_method="penalized_objective",
            objective_penalty="ebic",
            min_k=1,
            max_k=2,
        ),
        feature_blocks=blocks,
        subsample=None,
        verbose=False,
        return_result=True,
    )
    ediag = ebic.diagnostics_["auto_k_diagnostics"]
    # The constant block has no cache-valid member, so it is not in B.
    assert int(ediag["n_candidates"].iloc[0]) == 1
    assert int(ediag["k"].max()) <= 2


def test_nested_include_and_group_cv_keep_blocks():
    X, y, blocks = _block_frame()
    groups = np.repeat(np.arange(8), len(X) // 8 + 1)[: len(X)]
    cfg = AutoKConfig(
        k_method="evaluate",
        auto_k_mode="nested",
        strategy="group_cv",
        n_splits=4,
        min_k=1,
        max_k=2,
        selection_rule="best",
    )
    selector = CEFSPlusSelector(
        k="auto",
        auto_k_config=cfg,
        feature_blocks=blocks,
        include=["ab__0", "ab__1"],
        subsample=None,
        verbose=False,
    )
    selector.fit(X, y, groups=groups)
    assert {"ab__0", "ab__1"} <= set(selector.selected_features_)
    assert selector.k_ in {1, 2}
    Xt = selector.transform(X)
    assert Xt.shape[1] == len(selector.selected_features_)
    assert set(UNSUPPORTED_BLOCK_AUTO_K) & SUPPORTED_BLOCK_AUTO_K == set()


def test_copula_df_is_weighted_rank_not_shrinkage():
    rng = np.random.default_rng(14)
    n = 80
    a = rng.normal(size=n)
    X = pd.DataFrame({"a": a, "b": a, "c": rng.normal(size=n), "constant": np.ones(n)})
    cache = build_cache(X, compute_Rxx=True, subsample=None)
    dfs = gaussian_copula_prefix_df(cache, [0, 1, 2, 3], (2, 3, 4))
    assert dfs.tolist() == pytest.approx([1.0, 2.0, 2.0], abs=1e-9)
    cache_no_r = build_cache(X, compute_Rxx=False, subsample=None)
    dfs_no_r = gaussian_copula_prefix_df(cache_no_r, [0, 1, 2, 3], (2, 3, 4))
    assert dfs_no_r.tolist() == pytest.approx([1.0, 2.0, 2.0], abs=1e-9)
    included = gaussian_copula_prefix_df(
        cache, [0, 1, 2], (2, 3), include_indices=(1,)
    )
    assert included.tolist() == pytest.approx([0.0, 1.0], abs=1e-9)


def test_ebic_exclude_does_not_inflate_search_universe():
    rng = np.random.default_rng(14)
    n = 180
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=list("abcdef"))
    y = 0.22 * X.a + 0.20 * X.b + 0.12 * X.c + rng.normal(size=n)
    blocks = {"ab": ["a", "b"], "cde": ["c", "d", "e"]}
    xx = X.copy()
    xx["z0"] = y + rng.normal(size=n) * 0.03
    xx["z1"] = y + rng.normal(size=n) * 0.04
    cfg = AutoKConfig(
        k_method="penalized_objective",
        objective_penalty="ebic",
        ebic_gamma=0.5,
        min_k=0,
        max_k=3,
    )
    common = dict(
        k="auto",
        auto_k_config=cfg,
        subsample=None,
        verbose=False,
        return_result=True,
    )
    base = select_cefsplus(X, y, feature_blocks=blocks, **common)
    excl = select_cefsplus(
        xx,
        y,
        feature_blocks={**blocks, "z": ["z0", "z1"]},
        exclude=["z0", "z1"],
        **common,
    )
    b1 = int(base.diagnostics_["auto_k_diagnostics"]["n_candidates"].iloc[0])
    b2 = int(excl.diagnostics_["auto_k_diagnostics"]["n_candidates"].iloc[0])
    assert b1 == b2 == 3


def test_ric_block_penalty_uses_model_df_and_block_universe():
    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=list("abcdef"))
    y = 0.2 * X.a + 0.2 * X.b + rng.normal(size=n)
    blocks = {"ab": ["a", "b"], "cde": ["c", "d", "e"]}
    res = select_cefsplus(
        X,
        y,
        k="auto",
        auto_k_config=AutoKConfig(
            k_method="penalized_objective",
            objective_penalty="ric",
            min_k=0,
            max_k=3,
        ),
        feature_blocks=blocks,
        subsample=None,
        verbose=False,
        return_result=True,
    )
    diag = res.diagnostics_["auto_k_diagnostics"]
    b = int(diag["n_candidates"].iloc[0])
    assert b == 3
    for _, row in diag[diag["k"] > 0].iterrows():
        expected = 2.0 * float(row["df"]) * log(b)
        assert float(row["penalty"]) == pytest.approx(expected, rel=1e-12, abs=1e-9)


def test_nested_default_bounds_use_block_units():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(120, 6)), columns=list("abcdef"))
    y = X.a + 0.1 * rng.normal(size=len(X))
    blocks = {"abc": list("abc"), "def": list("def")}
    cfg = AutoKConfig(
        k_method="evaluate",
        auto_k_mode="nested",
        strategy="group_cv",
        n_splits=3,
    )
    selector = CEFSPlusSelector(
        k="auto",
        auto_k_config=cfg,
        feature_blocks=blocks,
        subsample=None,
        verbose=False,
    )
    selector.fit(X, y, groups=np.repeat(np.arange(6), 20))
    assert selector.k_ in {1, 2}
    scores = selector.nested_auto_k_diagnostics_["scores"]
    assert int(scores["k"].max()) <= 2
    assert not scores.empty


def test_nested_fit_time_blocks_and_integer_labels():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(120, 6)), columns=list("abcdef"))
    y = X.a + 0.1 * rng.normal(size=len(X))
    groups = np.repeat(np.arange(6), 20)
    cfg = AutoKConfig(
        k_method="evaluate",
        auto_k_mode="nested",
        strategy="group_cv",
        n_splits=3,
        min_k=1,
        max_k=2,
        selection_rule="best",
    )
    blocks = {"abc": list("abc"), "def": list("def")}
    selector = CEFSPlusSelector(
        k="auto", auto_k_config=cfg, subsample=None, verbose=False
    )
    selector.fit(X, y, groups=groups, feature_blocks=blocks)
    names = list(selector.selected_features_)
    if set("abc") & set(names):
        assert set("abc") <= set(names)
    folds = selector.nested_auto_k_diagnostics_["folds"]
    for path in folds["path"]:
        if not path:
            continue
        if set("abc") & set(path):
            assert set("abc") <= set(path)
        if set("def") & set(path):
            assert set("def") <= set(path)

    X_int = X.copy()
    X_int.columns = list(range(6))
    int_blocks = {"abc": [0, 1, 2], "def": [3, 4, 5]}
    int_sel = CEFSPlusSelector(
        k="auto",
        auto_k_config=cfg,
        feature_blocks=int_blocks,
        subsample=None,
        verbose=False,
    )
    int_sel.fit(X_int, y, groups=groups)
    assert int_sel.k_ in {1, 2}
    selected = list(int_sel.selected_features_)
    if 0 in selected or 1 in selected or 2 in selected:
        assert {0, 1, 2} <= set(selected)


def test_singleton_mapping_preserves_calibrated_and_default_routes():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(120, 5)), columns=list("abcde"))
    y = X.a + 0.5 * X.b + 0.1 * rng.normal(size=len(X))
    identity = {c: [c] for c in X.columns}
    cfg = AutoKConfig(k_method="chi2_stop", min_k=0, max_k=3)
    plain = select_cefsplus(
        X, y, k="auto", auto_k_config=cfg, subsample=None, verbose=False
    )
    blocked = select_cefsplus(
        X,
        y,
        k="auto",
        auto_k_config=cfg,
        feature_blocks=identity,
        subsample=None,
        verbose=False,
    )
    assert blocked == plain
    wide = pd.DataFrame(rng.normal(size=(28, 36)))
    y_wide = wide[0] + rng.normal(size=28)
    id_wide = {c: [c] for c in wide.columns}
    routed = select_cefsplus(
        wide, y_wide, k="auto", subsample=None, verbose=False, return_result=True
    )
    routed_id = select_cefsplus(
        wide,
        y_wide,
        k="auto",
        feature_blocks=id_wide,
        subsample=None,
        verbose=False,
        return_result=True,
    )
    assert routed.diagnostics_["auto_k"]["method"] == routed_id.diagnostics_["auto_k"]["method"]


def test_public_curve_helpers_reject_unused_conditioning():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(80, 4)), columns=list("abcd"))
    y = X.a.to_numpy() + rng.normal(size=80)
    cache = build_cache(X, compute_Rxx=True, subsample=None)
    cfg = AutoKConfig(
        k_method="gaussian_cv",
        strategy="kfold",
        xfit_folds=3,
        min_k=1,
        max_k=2,
    )
    with pytest.raises(ValueError, match="include/exclude/candidates"):
        gaussian_cv_curves(
            cache,
            y,
            config=cfg,
            top_m=4,
            corr_prune="auto",
            method="cefsplus",
            feature_blocks={"ab": ["a", "b"]},
            exclude=["a", "b"],
        )
    xcfg = AutoKConfig(
        k_method="xfit_objective",
        strategy="kfold",
        xfit_folds=3,
        min_k=1,
        max_k=2,
    )
    with pytest.raises(ValueError, match="include/exclude/candidates"):
        xfit_objective_curves(
            cache,
            y,
            config=xcfg,
            top_m=4,
            corr_prune="auto",
            method="cefsplus",
            include=["a"],
        )
    with pytest.raises(ValueError, match="include"):
        select_cefsplus(
            X,
            y,
            k="auto",
            auto_k_config=cfg,
            feature_blocks={"ab": ["a", "b"]},
            include=["a", "b"],
            subsample=None,
            verbose=False,
        )


def test_auto_dense_check_compares_block_k_and_forwards_blocks(monkeypatch):
    rng = np.random.default_rng(0)
    n, p = 80, 12
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"x{i}" for i in range(p)])
    y = X["x0"] + X["x1"] + rng.normal(size=n)
    blocks = {f"b{i}": [f"x{2 * i}", f"x{2 * i + 1}"] for i in range(6)}
    captured = {}

    import sift.selection.filter_auto_k as filter_auto_k
    import sift.selection.auto_k_xfit as auto_k_xfit

    real_curves = auto_k_xfit.gaussian_cv_curves

    def spy_curves(*args, **kwargs):
        captured["feature_blocks"] = kwargs.get("feature_blocks")
        return real_curves(*args, **kwargs)

    monkeypatch.setattr(filter_auto_k, "gaussian_cv_curves", spy_curves)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        select_cefsplus(
            X,
            y,
            k="auto",
            auto_k_config=AutoKConfig(
                k_method="auto",
                auto_dense_check=True,
                auto_dense_min_k=1,
                auto_dense_min_frac=0.0,
                min_k=0,
                max_k=6,
            ),
            feature_blocks=blocks,
            subsample=None,
            verbose=False,
            return_result=True,
        )
    assert captured.get("feature_blocks") == blocks
    messages = [str(w.message) for w in caught if "dense-signal" in str(w.message)]
    for message in messages:
        assert "k=" in message


def test_xfit_no_block_and_identity_keep_column_step_df(monkeypatch):
    rng = np.random.default_rng(13)
    n = 120
    a = rng.normal(size=n)
    X = pd.DataFrame(
        {"a": a, "dup": a, "b": rng.normal(size=n), "c": rng.normal(size=n)}
    )
    y = a + 0.5 * rng.normal(size=n)
    cache = build_cache(X, compute_Rxx=True, subsample=None)
    cfg = AutoKConfig(
        k_method="xfit_objective",
        strategy="kfold",
        xfit_folds=3,
        min_k=1,
        max_k=4,
    )
    kw = dict(config=cfg, top_m=4, corr_prune=None, method="cefsplus")
    rank_calls: list[int] = []
    import sift.selection.blocks as blocks_module

    real_rank = blocks_module.weighted_copula_design_rank

    def spy_rank(*args, **kwargs):
        rank_calls.append(1)
        return real_rank(*args, **kwargs)

    monkeypatch.setattr(blocks_module, "weighted_copula_design_rank", spy_rank)
    none_curve = xfit_objective_curves(cache, y, **kw)
    assert rank_calls == []
    identity_curve = xfit_objective_curves(
        cache, y, feature_blocks={c: [c] for c in X.columns}, **kw
    )
    assert rank_calls == []
    assert list(none_curve["k"]) == list(identity_curve["k"])
    xfit_objective_curves(
        cache, y, feature_blocks={"ad": ["a", "dup"], "b": ["b"]}, **kw
    )
    assert rank_calls


def test_nested_explicit_none_overrides_constructor():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(120, 6)), columns=list("abcdef"))
    y = X.a + X.b + 0.1 * rng.normal(size=len(X))
    groups = np.repeat(np.arange(6), 20)
    cfg = AutoKConfig(
        k_method="evaluate",
        auto_k_mode="nested",
        strategy="group_cv",
        n_splits=3,
        min_k=1,
        max_k=2,
        selection_rule="best",
    )
    common = dict(k="auto", auto_k_config=cfg, subsample=None, verbose=False)
    expected_none = CEFSPlusSelector(**common).fit(X, y, groups=groups)
    actual_none = CEFSPlusSelector(
        **common, feature_blocks={"abc": list("abc"), "def": list("def")}
    ).fit(X, y, groups=groups, feature_blocks=None)
    assert actual_none.k_ == expected_none.k_
    exp_folds = expected_none.nested_auto_k_diagnostics_["folds"]
    act_folds = actual_none.nested_auto_k_diagnostics_["folds"]
    assert [len(p) for p in exp_folds["path"]] == [len(p) for p in act_folds["path"]]

    expected_inc = CEFSPlusSelector(
        **common, feature_blocks={"ab": ["a", "b"]}
    ).fit(X, y, groups=groups)
    actual_inc = CEFSPlusSelector(
        **common, feature_blocks={"ab": ["a", "b"]}, include=["a", "b"]
    ).fit(X, y, groups=groups, include=None)
    assert actual_inc.k_ == expected_inc.k_
    for row in actual_inc.nested_auto_k_diagnostics_["folds"].itertuples():
        if int(row.k) == 1:
            assert len(row.path) == 2


def test_nested_constant_only_blocks_are_not_available_units():
    rng = np.random.default_rng(13)
    X = pd.DataFrame(rng.normal(size=(120, 6)), columns=list("abcdef"))
    X["g"] = 1.0
    X["h"] = 1.0
    X["i"] = 1.0
    y = X.a + 0.1 * rng.normal(size=120)
    groups = np.repeat(np.arange(6), 20)
    cfg = AutoKConfig(
        k_method="evaluate",
        auto_k_mode="nested",
        strategy="group_cv",
        n_splits=3,
    )
    selector = CEFSPlusSelector(
        k="auto",
        auto_k_config=cfg,
        feature_blocks={"abc": list("abc"), "def": list("def"), "const": list("ghi")},
        subsample=None,
        verbose=False,
    )
    selector.fit(X, y, groups=groups)
    assert selector.k_ in {1, 2}
    scores = selector.nested_auto_k_diagnostics_["scores"]
    assert not scores.empty
    assert int(scores["k"].max()) <= 2
    assert int(scores["n_finite"].max()) > 0
