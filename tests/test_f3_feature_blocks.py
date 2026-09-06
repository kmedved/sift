"""Fixed-k F3 feature_blocks: atomic groups, joint gain, alias plumbing."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from sift import (
    AutoKConfig,
    CEFSPlusBinarySelector,
    CEFSPlusSelector,
    JMISelector,
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
from sift.selection.blocks import ONEHOT_PREFIX_SEP, resolve_feature_blocks
from sift.selection.cefsplus import _chol_logdet


@contextmanager
def _ignore_gaussian_mrmr_floor():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="Gaussian mRMR")
        yield


def _regression_frame(n=120, p=6, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = X["f0"] + 0.8 * X["f1"] + 0.1 * rng.normal(size=n)
    return X, y


def _joint_gain_oracle(R, r, members, shrink=1e-6, eps=1e-12):
    scale = 1.0 - shrink
    idx = np.asarray(members, dtype=np.int64)
    g = scale * np.asarray(R, dtype=np.float64)[np.ix_(idx, idx)]
    np.fill_diagonal(g, 1.0)
    c = scale * np.asarray(r, dtype=np.float64)[idx]
    g_y = g - np.outer(c, c)
    return _chol_logdet(g, shrink=shrink, eps=eps) - _chol_logdet(
        g_y, shrink=shrink, eps=eps
    )


def test_auto_prefix_does_not_split_ordinary_underscores():
    names = ["user_id", "city_code", "city__NY", "city__LA", "x"]
    resolved = resolve_feature_blocks("auto", feature_names=names, named=True)
    assert ONEHOT_PREFIX_SEP == "__"
    assert resolved.block_ids[0] == "city"
    assert set(resolved.members[0]) == {2, 3}
    assert "user_id" in resolved.block_ids
    assert "city_code" in resolved.block_ids


def test_explicit_blocks_reject_overlap_and_unknown():
    names = ["a", "b", "c"]
    with pytest.raises(ValueError, match="overlap"):
        resolve_feature_blocks(
            {"g1": ["a", "b"], "g2": ["b"]},
            feature_names=names,
            named=True,
        )
    with pytest.raises(ValueError, match="unknown"):
        resolve_feature_blocks(
            {"g": ["a", "missing"]},
            feature_names=names,
            named=True,
        )


def test_cefsplus_joint_block_gain_beats_representative():
    rng = np.random.default_rng(4)
    n = 400
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    noise = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "sig__a": z1 + 0.05 * rng.normal(size=n),
            "sig__b": z2 + 0.05 * rng.normal(size=n),
            "decoy": z1 + 0.4 * rng.normal(size=n),
            "noise": rng.normal(size=n),
        }
    )
    y = z1 + z2 + 0.05 * noise
    cache = build_cache(X, compute_Rxx=True, subsample=None)
    panel_r = cache.Rxx
    from sift.estimators.copula import weighted_corr_with_vector, weighted_rank_gauss_1d

    zy = weighted_rank_gauss_1d(np.asarray(y), cache.sample_weight)
    r = weighted_corr_with_vector(cache.Z, zy, cache.sample_weight)
    joint = _joint_gain_oracle(panel_r, r, [0, 1])
    one_a = _joint_gain_oracle(panel_r, r, [0])
    one_b = _joint_gain_oracle(panel_r, r, [1])
    assert joint > max(one_a, one_b) + 1e-8

    selected = select_cefsplus(
        X, y, k=1, verbose=False, feature_blocks="auto", subsample=None
    )
    assert set(selected) == {"sig__a", "sig__b"}
    column_only = select_cefsplus(X, y, k=1, verbose=False, subsample=None)
    assert set(column_only) != {"sig__a", "sig__b"}


def test_unequal_blocks_k_metadata_and_transform():
    rng = np.random.default_rng(1)
    n = 150
    X = pd.DataFrame(
        {
            "b__1": rng.normal(size=n) + 1.5,
            "b__2": rng.normal(size=n) + 1.5,
            "b__3": rng.normal(size=n) + 1.5,
            "s": rng.normal(size=n),
        }
    )
    y = X["b__1"] + X["b__2"] + X["b__3"] + 0.05 * rng.normal(size=n)
    result = select_cefsplus(
        X,
        y,
        k=1,
        verbose=False,
        return_result=True,
        feature_blocks="auto",
        subsample=None,
    )
    assert set(result.selected_features) == {"b__1", "b__2", "b__3"}
    md = result.selector_metadata
    assert md["n_columns_selected"] == 3
    assert md["n_blocks_selected"] == 1
    assert md["k"] == 1
    assert md["k_requested"] == 1
    view = as_result(result, input_features=list(X.columns))
    assert view.k == 3
    assert "block_id" in view.raw_table.columns
    selector = CEFSPlusSelector(
        k=1, verbose=False, feature_blocks="auto", subsample=None
    )
    Xt = selector.fit_transform(X, y)
    assert Xt.shape == (n, 3)
    assert selector.get_support().sum() == 3


def test_singleton_blocks_match_omitted_feature_blocks():
    X, y = _regression_frame()
    blocks = {name: [name] for name in X.columns}
    for fn in (select_mrmr, select_jmi, select_cefsplus):
        kwargs = {"verbose": False, "k": 2, "subsample": None}
        if fn is select_mrmr:
            kwargs["task"] = "regression"
        if fn is select_jmi:
            kwargs["task"] = "regression"
        plain = fn(X, y, **kwargs)
        blocked = fn(X, y, feature_blocks=blocks, **kwargs)
        assert blocked == plain


def test_weights_and_cache_mapping_expand_raw_columns():
    rng = np.random.default_rng(2)
    n = 80
    X = pd.DataFrame(
        {
            "blk__u": rng.normal(size=n),
            "blk__v": rng.normal(size=n),
            "const": np.ones(n),
            "noise": rng.normal(size=n),
        }
    )
    y = X["blk__u"] + X["blk__v"] + 0.05 * rng.normal(size=n)
    w = np.linspace(0.2, 1.8, n)
    cache = build_cache(X, sample_weight=w, compute_Rxx=True, subsample=None)
    valid_names = {X.columns[int(i)] for i in cache.valid_cols}
    assert "const" not in valid_names
    view = select_cached(
        cache,
        y,
        k=1,
        feature_blocks={"sig": ["blk__u", "blk__v", "const"]},
        return_result=True,
    )
    assert "blk__u" in view.features and "blk__v" in view.features
    assert "const" in view.features
    assert view.metadata["n_columns_selected"] == len(view.features)
    weighted = select_cefsplus(
        X,
        y,
        k=1,
        verbose=False,
        sample_weight=w,
        feature_blocks={"sig": ["blk__u", "blk__v"]},
        subsample=None,
        return_result=True,
    )
    assert set(weighted.selected_features) == {"blk__u", "blk__v"}


def test_classic_estimator_identity_and_split_restriction():
    X, y = _regression_frame()
    blocks = {"ab": ["f0", "f1"], "c": ["f2"]}
    classic = select_mrmr(
        X,
        y,
        k=1,
        task="regression",
        verbose=False,
        feature_blocks=blocks,
        subsample=None,
        estimator="classic",
        relevance="f",
    )
    assert set(classic) <= {"f0", "f1", "f2"}
    if set(classic) & {"f0", "f1"}:
        assert set(classic) >= {"f0", "f1"}
    with pytest.raises(ValueError, match="split"):
        select_mrmr(
            X,
            y,
            k=1,
            task="regression",
            verbose=False,
            feature_blocks=blocks,
            include=["f0"],
            subsample=None,
        )
    r2 = select_jmi(
        X,
        y,
        k=1,
        task="regression",
        estimator="r2",
        verbose=False,
        feature_blocks=blocks,
        subsample=None,
    )
    assert isinstance(r2, list)


def test_k_auto_and_binary_logloss_rejected():
    X, y = _regression_frame()
    blocks = {"g": ["f0", "f1"]}
    auto = select_cefsplus(
        X, y, k="auto", verbose=False, feature_blocks=blocks, subsample=None
    )
    assert isinstance(auto, list)
    if "f0" in auto or "f1" in auto:
        assert {"f0", "f1"} <= set(auto)
    with pytest.raises(ValueError, match="scalar column steps"):
        select_cefsplus(
            X,
            y,
            k="auto",
            auto_k_config=AutoKConfig(k_method="perm_gap", max_k=3),
            verbose=False,
            feature_blocks=blocks,
            subsample=None,
        )
    y_bin = (y > y.median()).astype(int)
    selected = select_cefsplus_binary(
        X,
        y_bin,
        k=1,
        loss="logloss",
        verbose=False,
        feature_blocks=blocks,
        subsample=None,
    )
    assert isinstance(selected, list)
    if "f0" in selected or "f1" in selected:
        assert {"f0", "f1"} <= set(selected)
    with pytest.raises(ValueError, match="binary log-loss"):
        select_cefsplus_binary(
            X,
            y_bin,
            k="auto",
            auto_k_config=AutoKConfig(k_method="changepoint", max_k=3),
            verbose=False,
            feature_blocks=blocks,
            subsample=None,
        )
    clone(CEFSPlusBinarySelector(k=1, verbose=False))


def test_knockoff_alias_parity_and_conflict():
    rng = np.random.default_rng(0)
    n, p = 60, 6
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=list("abcdef"))
    y = X["a"] + X["b"] + 0.1 * rng.normal(size=n)
    mapping = {"ab": ["a", "b"], "cd": ["c", "d"]}
    labels = ["ab", "ab", "cd", "cd", "e", "f"]
    via_blocks = select_fdr(
        X, y, q=0.5, feature_blocks=mapping, verbose=False, subsample=None
    )
    via_groups = select_fdr(
        X, y, q=0.5, feature_groups=labels, verbose=False, subsample=None
    )
    assert via_blocks.selected_features == via_groups.selected_features
    with pytest.raises(ValueError, match="feature_blocks and feature_groups"):
        select_fdr(
            X,
            y,
            q=0.5,
            feature_groups=labels,
            feature_blocks={"other": ["a", "c"]},
            verbose=False,
            subsample=None,
        )
    both = select_fdr(
        X,
        y,
        q=0.5,
        feature_groups=labels,
        feature_blocks=mapping,
        verbose=False,
        subsample=None,
    )
    assert both.selected_features == via_groups.selected_features


def test_no_block_defaults_unchanged():
    X, y = _regression_frame()
    assert select_cefsplus(X, y, k=2, verbose=False, subsample=None) == select_cefsplus(
        X, y, k=2, verbose=False, subsample=None, feature_blocks=None
    )
    selector = clone(MRMRSelector(k=2, task="regression", verbose=False, subsample=None))
    selector.fit(X, y)
    assert len(selector.selected_features_) == 2
    jmi = clone(JMISelector(k=2, task="regression", verbose=False, subsample=None))
    jmi.fit(X, y)
    assert jmi.n_features_in_ == X.shape[1]


def test_included_blocks_do_not_consume_discovery_top_m():
    rng = np.random.default_rng(80)
    X = pd.DataFrame(rng.normal(size=(180, 6)), columns=list("abcdef"))
    y = 2 * X.a + X.b + 0.4 * X.c + rng.normal(size=len(X))
    blocks = {"ab": ["a", "b"], "cd": ["c", "d"]}
    include = ["a", "b"]
    for fn, extra in (
        (select_cefsplus, {}),
        (select_mrmr, {"task": "regression", "estimator": "gaussian"}),
        (select_jmi, {"task": "regression", "estimator": "gaussian"}),
        (select_jmim, {"task": "regression", "estimator": "gaussian"}),
        (select_mrmr, {"task": "regression", "estimator": "classic"}),
    ):
        warn_ctx = (
            _ignore_gaussian_mrmr_floor()
            if extra.get("estimator") == "gaussian" and fn is select_mrmr
            else nullcontext()
        )
        with warn_ctx:
            result = fn(
                X,
                y,
                k=1,
                feature_blocks=blocks,
                include=include,
                top_m=1,
                subsample=None,
                verbose=False,
                return_result=True,
                **extra,
            )
        assert set(include) <= set(result.selected_features), (fn, extra)
        assert len(result.selected_features) > 2, (fn, extra, result.selected_features)
        md = result.selector_metadata
        assert md["k"] == 1, (fn, extra, md)
        assert md["n_blocks_selected"] == 1, (fn, extra, md)
        assert md["k_requested"] == 1
        assert md["n_blocks_selected_total"] == 2, (fn, extra, md)
        assert md["n_columns_selected"] == len(result.selected_features)
        assert "ab" not in md["selected_blocks"]


def test_duplicate_block_labels_are_rejected_including_knockoff_alias():
    names = ["a", "b", "c"]
    with pytest.raises(ValueError, match="duplicate block labels"):
        resolve_feature_blocks({"b": ["a"]}, feature_names=names, named=True)
    with pytest.raises(ValueError, match="duplicate block labels"):
        resolve_feature_blocks(
            "auto",
            feature_names=["a__x", "a__y", "a"],
            named=True,
        )
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 3)), columns=names)
    y = X["a"] + 0.1 * rng.normal(size=len(X))
    with pytest.raises(ValueError, match="duplicate block labels"):
        select_fdr(
            X, y, q=0.5, feature_blocks={"b": ["a"]}, verbose=False, subsample=None
        )


def test_singleton_mapping_preserves_ties_and_conditioning():
    rng = np.random.default_rng(80)
    X = pd.DataFrame(rng.normal(size=(180, 6)), columns=list("abcdef"))
    y = 2 * X.a + X.b + 0.4 * X.c + rng.normal(size=len(X))
    Xt = X.copy()
    Xt["b"] = Xt["a"]
    reversed_blocks = {c: [c] for c in reversed(Xt.columns)}
    cases = (
        (select_cefsplus, {}),
        (select_mrmr, {"task": "regression", "estimator": "gaussian"}),
        (select_jmi, {"task": "regression", "estimator": "gaussian"}),
        (select_jmim, {"task": "regression", "estimator": "gaussian"}),
        (select_mrmr, {"task": "regression", "estimator": "classic"}),
        (select_jmi, {"task": "regression", "estimator": "r2"}),
    )
    for fn, extra in cases:
        args = dict(k=2, subsample=None, verbose=False, **extra)

        def _warn():
            if extra.get("estimator") == "gaussian" and fn is select_mrmr:
                return _ignore_gaussian_mrmr_floor()
            return nullcontext()

        with _warn():
            plain = fn(Xt, y, **args)
        with _warn():
            blocked = fn(Xt, y, feature_blocks=reversed_blocks, **args)
        assert blocked == plain, (fn.__name__, extra, plain, blocked)
        cond_args = dict(args, k=1, include=["a"], top_m=1)
        with _warn():
            plain_c = fn(Xt, y, **cond_args)
        with _warn():
            blocked_c = fn(Xt, y, feature_blocks=reversed_blocks, **cond_args)
        assert blocked_c == plain_c, (fn.__name__, extra, plain_c, blocked_c)


def test_atomic_proxy_panel_and_constant_member_error():
    rng = np.random.default_rng(14)
    X = pd.DataFrame(rng.normal(size=(120, 5)), columns=list("abcde"))
    y = X.a + X.b + 0.1 * rng.normal(size=len(X))
    blocks = {"pair": ["a", "b"]}
    result = select_cefsplus(
        X,
        y,
        k=1,
        feature_blocks=blocks,
        top_m=1,
        subsample=None,
        store_proxies=True,
        return_result=True,
        verbose=False,
    )
    assert set(result.selected_features) == {"a", "b"}
    view = as_result(result, input_features=list(X.columns))
    proxies = view.proxies("a", r_min=0.0)
    assert list(proxies.columns) == ["feature", "selected_index", "correlation"]
    Xc = X.copy()
    Xc["b"] = 1.0
    cache = build_cache(Xc, subsample=None, compute_Rxx=True)
    expanded = select_cached(
        cache, y, k=1, feature_blocks=blocks, return_result=True
    )
    assert "b" in expanded.features
    with pytest.raises(ValueError, match="unavailable block members"):
        select_cached(
            cache,
            y,
            k=1,
            feature_blocks=blocks,
            store_proxies=True,
            return_result=True,
        )
