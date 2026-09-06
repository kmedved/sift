"""Binary log-loss CEFS+ feature_blocks: joint score, auto-k, singleton parity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sift import (
    AutoKConfig,
    CEFSPlusBinarySelector,
    as_result,
    select_cefsplus_binary,
)
from sift.selection.cefsplus_binary import (
    intercept_only_prob,
    logistic_joint_score_test,
    logistic_score_test_scores,
)


def _binary_block_frame(n=160, seed=4):
    rng = np.random.default_rng(seed)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "ab__0": z1 + 0.05 * rng.normal(size=n),
            "ab__1": z1 + 0.2 * rng.normal(size=n),
            "c": z2,
            "n0": rng.normal(size=n),
            "const": np.ones(n),
        }
    )
    logits = 2.2 * z1 + 0.7 * z2
    y = (logits + 0.4 * rng.normal(size=n) > 0.0).astype(int)
    blocks = {"ab": ["ab__0", "ab__1"], "pad": ["const"]}
    return X, y, blocks


def test_joint_score_is_not_max_or_sum_of_scalars():
    rng = np.random.default_rng(0)
    n = 120
    z = rng.normal(size=n)
    Z = np.column_stack([z + 0.05 * rng.normal(size=n), z + 0.08 * rng.normal(size=n)])
    w = np.ones(n)
    y = (1.5 * z + rng.normal(size=n) > 0.0).astype(float)
    p = intercept_only_prob(y, w)
    joint, _, _ = logistic_joint_score_test(Z, y, w, p, ridge=1e-4)
    scalars, _, _ = logistic_score_test_scores(Z, y, w, p, ridge=1e-4)
    assert np.isfinite(joint)
    assert joint != pytest.approx(float(np.max(scalars)), rel=1e-6, abs=1e-8)
    assert joint != pytest.approx(float(np.sum(scalars)), rel=1e-6, abs=1e-8)


def test_singleton_block_path_matches_no_block():
    X, y, _ = _binary_block_frame()
    X = X[["ab__0", "c", "n0"]]
    identity = {c: [c] for c in X.columns}
    for refit_every in (1, 2):
        plain = select_cefsplus_binary(
            X, y, k=2, subsample=None, verbose=False, refit_every=refit_every
        )
        blocked = select_cefsplus_binary(
            X,
            y,
            k=2,
            subsample=None,
            verbose=False,
            refit_every=refit_every,
            feature_blocks=identity,
        )
        assert blocked == plain


def test_unequal_blocks_expand_constants_and_count_blocks():
    X, y, blocks = _binary_block_frame()
    result = select_cefsplus_binary(
        X,
        y,
        k=1,
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
    assert md["k"] == md["n_blocks_selected"]
    view = as_result(result, input_features=list(X.columns))
    assert view.k == len(names)
    scores = result.ranking_["score"]
    selected = result.ranking_["selected"]
    block_id = result.ranking_["block_id"]
    ab_scores = scores[(selected) & (block_id == "ab")].dropna()
    if len(ab_scores):
        assert ab_scores.nunique() == 1


def test_binary_f1_and_weights_keep_complete_blocks():
    X, y, blocks = _binary_block_frame()
    w = np.linspace(0.4, 1.6, len(X))
    with pytest.raises(ValueError, match="include"):
        select_cefsplus_binary(
            X,
            y,
            k=1,
            include=["ab__0"],
            feature_blocks=blocks,
            subsample=None,
            verbose=False,
        )
    result = select_cefsplus_binary(
        X,
        y,
        k=1,
        include=["ab__0", "ab__1"],
        sample_weight=w,
        feature_blocks=blocks,
        subsample=None,
        verbose=False,
        return_result=True,
    )
    assert {"ab__0", "ab__1"} <= set(result.selected_features)
    md = result.selector_metadata
    assert md["k"] == 1
    assert md["n_blocks_selected"] == 1
    assert md["k_requested"] == 1
    assert md["n_blocks_selected_total"] >= 2
    assert md["n_columns_selected"] == len(result.selected_features)
    assert "ab" not in md["selected_blocks"]


def test_binary_ebic_uses_model_df_not_block_count():
    X, y, blocks = _binary_block_frame()
    result = select_cefsplus_binary(
        X,
        y,
        k="auto",
        auto_k_config=AutoKConfig(
            k_method="penalized_objective",
            objective_penalty="ebic",
            ebic_gamma=0.5,
            min_k=0,
            max_k=3,
        ),
        feature_blocks=blocks,
        subsample=None,
        verbose=False,
        return_result=True,
    )
    diag = result.diagnostics_["auto_k_diagnostics"]
    row = diag[diag["k"] > 0].iloc[0]
    assert float(row["df"]) != float(row["k"]) or int(row["k"]) == 0
    b = int(row["n_candidates"])
    from math import comb, log

    expected = float(row["df"]) * log(float(row["n_eff"])) + 2.0 * float(
        row["ebic_gamma"]
    ) * log(comb(b, int(row["k"])))
    assert float(row["penalty"]) == pytest.approx(expected, rel=1e-10, abs=1e-8)
    score_test = select_cefsplus_binary(
        X,
        y,
        k="auto",
        auto_k_config=AutoKConfig(
            k_method="penalized_objective",
            objective_penalty="ebic",
            binary_objective_mode="score_test",
            ebic_gamma=0.5,
            min_k=0,
            max_k=3,
        ),
        feature_blocks=blocks,
        subsample=None,
        verbose=False,
        return_result=True,
    )
    sdiag = score_test.diagnostics_["auto_k_diagnostics"]
    assert bool(sdiag["score_test_ic_approximation"].iloc[0])
    assert int(sdiag["k"].max()) <= 3


def test_binary_evaluate_and_nested_use_block_k():
    X, y, blocks = _binary_block_frame()
    time = np.arange(len(X))
    cfg = AutoKConfig(
        k_method="evaluate",
        strategy="time_holdout",
        min_k=1,
        max_k=2,
        val_frac=0.3,
        selection_rule="best",
    )
    fn = select_cefsplus_binary(
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
    names = fn.selected_features
    if "ab__0" in names or "ab__1" in names:
        assert {"ab__0", "ab__1"} <= set(names)
    assert fn.selector_metadata["k"] == fn.selector_metadata["n_blocks_selected"]
    groups = np.repeat(np.arange(8), len(X) // 8 + 1)[: len(X)]
    nested_cfg = AutoKConfig(
        k_method="evaluate",
        auto_k_mode="nested",
        strategy="group_cv",
        n_splits=4,
        min_k=1,
        max_k=2,
        selection_rule="best",
    )
    selector = CEFSPlusBinarySelector(
        k="auto",
        auto_k_config=nested_cfg,
        feature_blocks=blocks,
        subsample=None,
        verbose=False,
    )
    selector.fit(X, y, groups=groups)
    assert selector.k_ in {1, 2}
    Xt = selector.transform(X)
    assert Xt.shape[1] == len(selector.selected_features_)
    if "ab__0" in selector.selected_features_ or "ab__1" in selector.selected_features_:
        assert {"ab__0", "ab__1"} <= set(selector.selected_features_)


def test_binary_auto_and_brier_delegate():
    X, y, blocks = _binary_block_frame()
    auto = select_cefsplus_binary(
        X, y, k="auto", feature_blocks=blocks, subsample=None, verbose=False
    )
    assert isinstance(auto, list)
    brier = select_cefsplus_binary(
        X,
        y,
        k=1,
        loss="brier",
        feature_blocks=blocks,
        subsample=None,
        verbose=False,
    )
    assert isinstance(brier, list)
    if "ab__0" in brier or "ab__1" in brier:
        assert {"ab__0", "ab__1"} <= set(brier)


def test_constant_include_on_all_invalid_block_path_raises():
    X = pd.DataFrame(np.ones((40, 3)), columns=list("abc"))
    y = np.arange(40) % 2
    blocks = {"pair": ["a", "b"]}
    with pytest.raises(
        ValueError, match="include feature 'a' is not a valid non-constant"
    ):
        select_cefsplus_binary(
            X,
            y,
            k=1,
            include=["a", "b"],
            feature_blocks=blocks,
            subsample=None,
            verbose=False,
        )
    with pytest.raises(
        ValueError, match="include feature 'a' is not a valid non-constant"
    ):
        select_cefsplus_binary(
            X, y, k=1, include=["a", "b"], subsample=None, verbose=False
        )
    w = np.random.default_rng(0).uniform(0.2, 2.0, 40)
    with pytest.raises(
        ValueError, match="include feature 'a' is not a valid non-constant"
    ):
        select_cefsplus_binary(
            X,
            y,
            k=1,
            include=["a", "b"],
            feature_blocks=blocks,
            sample_weight=w,
            subsample=None,
            verbose=False,
        )
    assert (
        select_cefsplus_binary(
            X, y, k=1, feature_blocks=blocks, subsample=None, verbose=False
        )
        == []
    )


def test_identity_include_metadata_uses_discovery_k():
    rng = np.random.default_rng(2)
    n = 80
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    y = ((X["a"] + X["c"]) > 0).astype(int)
    kwargs = dict(
        k=1,
        include=["a", "b"],
        subsample=None,
        verbose=False,
        return_result=True,
    )
    plain = select_cefsplus_binary(X, y, **kwargs)
    ident = select_cefsplus_binary(
        X, y, feature_blocks={c: [c] for c in X.columns}, **kwargs
    )
    assert ident.selected_features == plain.selected_features
    np.testing.assert_array_equal(ident.ranking_["score"], plain.ranking_["score"])
    pmd = plain.selector_metadata
    imd = ident.selector_metadata
    assert pmd["k"] == len(plain.selected_features)
    assert imd["k"] == 1
    assert imd["n_blocks_selected"] == 1
    assert imd["n_blocks_selected_total"] == pmd["k"]
    assert imd["n_columns_selected"] == pmd["k"]
    view_plain = as_result(plain, input_features=list(X.columns))
    view_ident = as_result(ident, input_features=list(X.columns))
    assert view_plain.k == view_ident.k == len(plain.selected_features)


def test_weighted_constant_only_block_is_not_selected():
    for seed in range(4):
        rng = np.random.default_rng(seed)
        n = 80
        X = pd.DataFrame(
            {
                "a": np.ones(n),
                "b": np.full(n, 2.0),
                "signal": rng.normal(size=n),
            }
        )
        y = (X["signal"] > 0).astype(int)
        w = rng.uniform(0.2, 2.0, n)
        result = select_cefsplus_binary(
            X,
            y,
            k=2,
            feature_blocks={"const": ["a", "b"]},
            sample_weight=w,
            subsample=None,
            verbose=False,
            return_result=True,
        )
        assert result.selected_features == ["signal"], seed
        assert result.selector_metadata["k"] == 1
        assert result.selector_metadata["n_blocks_selected"] == 1


def test_tiny_nonconstant_block_is_kept():
    rng = np.random.default_rng(1)
    n = 120
    z = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "a": 1e-13 * z,
            "b": 1e-13 * z + 1e-15 * rng.normal(size=n),
            "noise": rng.normal(size=n),
        }
    )
    y = (z > 0).astype(int)
    w = rng.uniform(0.2, 2.0, n)
    selected = select_cefsplus_binary(
        X,
        y,
        k=1,
        feature_blocks={"sig": ["a", "b"]},
        sample_weight=w,
        subsample=None,
        verbose=False,
    )
    assert set(selected) == {"a", "b"}


def test_constant_padded_block_expands_without_refit_failure():
    rng = np.random.default_rng(66)
    n = 160
    raw = rng.normal(size=(n, 5))
    y = (raw[:, 0] - raw[:, 1] + 0.5 * raw[:, 2] > 0).astype(int)
    X = pd.DataFrame(raw, columns=list("abcde"))
    X["pad"] = 1.0
    cfg = AutoKConfig(k_method="penalized_objective", min_k=1, max_k=2)
    result = select_cefsplus_binary(
        X,
        y,
        k="auto",
        auto_k_config=cfg,
        feature_blocks={"pair": ["a", "b", "pad"]},
        subsample=None,
        verbose=False,
        return_result=True,
    )
    names = result.selected_features
    if {"a", "b", "pad"} & set(names):
        assert {"a", "b", "pad"} <= set(names)
    assert int(result.diagnostics_["auto_k"]["binary_refit_failures"]) == 0
    assert result.selector_metadata["k"] == result.diagnostics_["auto_k"]["selected_k"]
    assert result.selector_metadata["k"] == result.selector_metadata["n_blocks_selected"]


def test_include_blocks_are_not_additional_k():
    rng = np.random.default_rng(2)
    n = 120
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=list("abcdef"))
    y = ((X["a"] + X["c"] - X["d"]) > 0).astype(int)
    blocks = {"inc": ["a", "b"], "pair": ["c", "d"]}
    result = select_cefsplus_binary(
        X,
        y,
        k=1,
        include=["a", "b"],
        feature_blocks=blocks,
        subsample=None,
        verbose=False,
        return_result=True,
    )
    assert {"a", "b", "c", "d"} <= set(result.selected_features)
    md = result.selector_metadata
    assert md["k_requested"] == 1
    assert md["k"] == 1
    assert md["n_blocks_selected"] == 1
    assert md["n_blocks_selected_total"] == 2
    assert md["selected_blocks"] == ["pair"]
    assert md["n_columns_selected"] == len(result.selected_features)
    view = as_result(result, input_features=list(X.columns))
    assert view.k == len(result.selected_features)

    cfg = AutoKConfig(
        k_method="evaluate",
        strategy="time_holdout",
        min_k=1,
        max_k=1,
        val_frac=0.3,
        selection_rule="best",
    )
    auto = select_cefsplus_binary(
        X,
        y,
        k="auto",
        auto_k_config=cfg,
        time=np.arange(n),
        include=["a", "b"],
        feature_blocks=blocks,
        subsample=None,
        verbose=False,
        return_result=True,
    )
    amd = auto.selector_metadata
    assert amd["k"] == amd["n_blocks_selected"]
    assert amd["k"] == auto.diagnostics_["auto_k"]["selected_k"]
    assert amd["n_blocks_selected_total"] == amd["n_blocks_selected"] + 1
    assert "inc" not in amd["selected_blocks"]
    assert {"a", "b"} <= set(auto.selected_features)

    nested_cfg = AutoKConfig(
        k_method="evaluate",
        auto_k_mode="nested",
        strategy="group_cv",
        n_splits=4,
        min_k=1,
        max_k=1,
        selection_rule="best",
    )
    groups = np.repeat(np.arange(8), n // 8 + 1)[:n]
    selector = CEFSPlusBinarySelector(
        k="auto",
        auto_k_config=nested_cfg,
        include=["a", "b"],
        feature_blocks=blocks,
        subsample=None,
        verbose=False,
    )
    selector.fit(X, y, groups=groups)
    assert selector.k_ == 1
    assert {"a", "b"} <= set(selector.selected_features_)
    assert selector.transform(X).shape[1] == len(selector.selected_features_)
