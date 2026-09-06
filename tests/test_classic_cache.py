"""Public contracts for ClassicFeatureCache reuse."""

from __future__ import annotations

from dataclasses import replace
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.validation import check_is_fitted

from sift import (
    AutoKConfig,
    ClassicFeatureCache,
    FeatureCache,
    MRMRSelector,
    build_cache,
    build_classic_cache,
    select_cached,
    select_jmi,
    select_jmim,
    select_mrmr,
)
from sift.estimators.classic_cache import is_classic_cache


def _regression_frame(n=80, p=5, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y1 = X["f0"] + 0.2 * X["f1"] + 0.05 * rng.normal(size=n)
    y2 = X["f3"] + 0.15 * X["f2"] + 0.05 * rng.normal(size=n)
    return X, y1, y2


def test_classic_cache_public_parity_two_targets_and_counts():
    X, y1, y2 = _regression_frame()
    cache = build_classic_cache(X, subsample=None)
    assert is_classic_cache(cache)
    assert cache.X.dtype == np.float64
    assert cache.X.shape == (len(X), X.shape[1])
    assert not hasattr(cache, "Rxx")
    for y in (y1, y2):
        for k in (1, 2):
            cached = select_mrmr(
                X, y, k=k, task="regression", cache=cache, verbose=False, return_result=True
            )
            plain = select_mrmr(
                X, y, k=k, task="regression", verbose=False, return_result=True, subsample=None
            )
            assert cached.selected_features == plain.selected_features
            assert cached.selected_indices == plain.selected_indices
            cached_rank = cached.get_feature_ranking()
            plain_rank = plain.get_feature_ranking()
            assert list(cached_rank["feature"]) == list(plain_rank["feature"])
            np.testing.assert_allclose(
                cached_rank["relevance"], plain_rank["relevance"]
            )


def test_classic_cache_jmi_jmim_and_wrapper_parity():
    X, y1, _ = _regression_frame()
    cache = build_classic_cache(X, subsample=None)
    for fn in (select_jmi, select_jmim):
        cached = fn(X, y1, k=2, task="regression", cache=cache, verbose=False)
        plain = fn(X, y1, k=2, task="regression", verbose=False, subsample=None)
        assert cached == plain
    fitted = MRMRSelector(k=2, task="regression", verbose=False).fit(X, y1, cache=cache)
    check_is_fitted(fitted)
    assert fitted.selected_features_ == select_mrmr(
        X, y1, k=2, task="regression", cache=cache, verbose=False
    )


def test_classic_cache_skips_feature_impute_on_reuse(monkeypatch):
    import sift._impute as impute_mod
    import sift.selection.filter_payloads as payloads

    X, y1, y2 = _regression_frame()
    orig_impute = impute_mod.mean_impute
    orig_validate = payloads.validate_inputs
    counts = {"impute": 0, "validate_inputs": 0}

    def spy_impute(*args, **kwargs):
        counts["impute"] += 1
        return orig_impute(*args, **kwargs)

    def spy_validate(*args, **kwargs):
        counts["validate_inputs"] += 1
        return orig_validate(*args, **kwargs)

    monkeypatch.setattr(impute_mod, "mean_impute", spy_impute)
    monkeypatch.setattr(payloads, "validate_inputs", spy_validate)
    cache = build_classic_cache(X, subsample=None)
    impute_at_build = counts["impute"]
    assert impute_at_build >= 1
    assert counts["validate_inputs"] == 0
    select_mrmr(X, y1, k=2, task="regression", cache=cache, verbose=False)
    select_mrmr(X, y2, k=1, task="regression", cache=cache, verbose=False)
    assert counts["impute"] == impute_at_build
    assert counts["validate_inputs"] == 0


def test_classic_cache_preserves_binned_mi_w_and_unweighted_eval():
    rng = np.random.default_rng(4)
    n = 40
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    y = pd.Series((X["a"] + rng.normal(size=n) > 0).astype(int))
    weights = np.array([0.0, 2.0, 3.0, 4.0] * 10, dtype=np.float64)
    cache = build_classic_cache(X, sample_weight=weights, subsample=None)
    assert cache.weights_supplied is True
    np.testing.assert_allclose(cache.mi_w, weights[cache.row_idx])
    assert not np.allclose(cache.mi_w, cache.sample_weight)
    cached = select_jmi(
        X,
        y,
        k=2,
        task="classification",
        estimator="binned",
        cache=cache,
        verbose=False,
        return_result=True,
    )
    plain = select_jmi(
        X,
        y,
        k=2,
        task="classification",
        estimator="binned",
        sample_weight=weights,
        subsample=None,
        verbose=False,
        return_result=True,
    )
    assert cached.selected_features == plain.selected_features
    unweighted_cache = build_classic_cache(X, subsample=None)
    assert unweighted_cache.weights_supplied is False
    result = select_mrmr(
        X,
        X["a"],
        k=1,
        task="regression",
        cache=unweighted_cache,
        verbose=False,
        return_result=True,
        auto_k_config=None,
    )
    assert result.selected_features


def test_classic_cache_type_and_override_walls():
    X, y1, _ = _regression_frame()
    classic = build_classic_cache(X, subsample=None)
    gaussian = build_cache(X, subsample=None)
    with pytest.raises(TypeError, match="select_mrmr"):
        select_cached(classic, y1, k=1)
    with pytest.raises(ValueError, match="estimator='gaussian'"):
        select_mrmr(
            X, y1, k=1, task="regression", estimator="gaussian", cache=classic, verbose=False
        )
    with pytest.raises(ValueError, match="estimator='gaussian'"):
        select_mrmr(
            X, y1, k=1, task="regression", estimator="classic", cache=gaussian, verbose=False
        )
    with pytest.raises(ValueError, match="sample_weight"):
        select_mrmr(
            X,
            y1,
            k=1,
            task="regression",
            cache=classic,
            sample_weight=np.ones(len(X)),
            verbose=False,
        )
    with pytest.raises(ValueError, match="subsample"):
        select_mrmr(
            X, y1, k=1, task="regression", cache=classic, subsample=20, verbose=False
        )
    with pytest.raises(ValueError, match="random_state"):
        select_mrmr(
            X, y1, k=1, task="regression", cache=classic, random_state=3, verbose=False
        )
    with pytest.raises(ValueError, match="within"):
        select_mrmr(
            X,
            y1,
            k=1,
            task="regression",
            cache=classic,
            within="groups",
            groups=np.repeat(np.arange(8), 10),
            verbose=False,
        )
    with pytest.raises(ValueError, match="encoding"):
        select_mrmr(
            X,
            y1,
            k=1,
            task="regression",
            cache=classic,
            cat_encoding="target_cv",
            verbose=False,
        )
    weighted = build_classic_cache(X, sample_weight=np.ones(len(X)), subsample=None)
    with pytest.raises(ValueError, match="ksg"):
        select_jmi(
            X,
            y1,
            k=1,
            task="regression",
            estimator="ksg",
            cache=weighted,
            verbose=False,
        )
    positional = build_classic_cache(X.to_numpy(), subsample=None)
    with pytest.raises(ValueError, match="positional ndarray"):
        select_mrmr(X, y1, k=1, task="regression", cache=positional, verbose=False)
    named = build_classic_cache(X, subsample=None)
    with pytest.raises(ValueError, match="DataFrame"):
        select_mrmr(
            X.to_numpy(), y1, k=1, task="regression", cache=named, verbose=False
        )


def test_classic_cache_conditioning_blocks_and_auto_evaluate():
    X, y1, _ = _regression_frame(n=48)
    cache = build_classic_cache(X, subsample=None)
    cached = select_mrmr(
        X,
        y1,
        k=1,
        task="regression",
        cache=cache,
        include=["f0"],
        candidates=["f1", "f2", "f3"],
        verbose=False,
    )
    plain = select_mrmr(
        X,
        y1,
        k=1,
        task="regression",
        include=["f0"],
        candidates=["f1", "f2", "f3"],
        subsample=None,
        verbose=False,
    )
    assert cached == plain
    groups = np.repeat(np.arange(8), 6)
    cfg = AutoKConfig(k_method="evaluate", strategy="group_cv", min_k=1, max_k=2, n_splits=2)
    cached_auto = select_mrmr(
        X,
        y1,
        k="auto",
        task="regression",
        cache=cache,
        groups=groups,
        auto_k_config=cfg,
        verbose=False,
    )
    plain_auto = select_mrmr(
        X,
        y1,
        k="auto",
        task="regression",
        groups=groups,
        auto_k_config=cfg,
        subsample=None,
        verbose=False,
    )
    assert cached_auto == plain_auto


def test_classic_cache_pickle_and_provenance():
    X, y1, y2 = _regression_frame()
    cache = build_classic_cache(X, subsample=None, random_state=0)
    restored = pickle.loads(pickle.dumps(cache))
    assert type(restored) is ClassicFeatureCache
    assert restored.subsample_applied is False
    first = select_mrmr(X, y1, k=1, task="regression", cache=cache, verbose=False)
    second = select_mrmr(X, y2, k=1, task="regression", cache=cache, verbose=False)
    assert first != second
    result = select_mrmr(
        X, y1, k=1, task="regression", cache=cache, verbose=False, return_result=True
    )
    meta = result.selector_metadata
    assert meta["cache_kind"] == "classic"
    assert meta["cache_backed"] is True
    assert "random_state" not in meta
    drawn = build_classic_cache(X, subsample=20, random_state=7)
    assert drawn.subsample_applied is True
    drawn_result = select_mrmr(
        X, y1, k=1, task="regression", cache=drawn, verbose=False, return_result=True
    )
    assert drawn_result.selector_metadata["random_state"] == 7
    assert type(build_cache(X, subsample=None)) is FeatureCache


def test_classic_cache_rejects_duplicate_and_nan_labels():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(40, 6))
    y = pd.Series(values[:, 0])
    duplicated = pd.DataFrame(values, columns=["a", "a", "c", "d", "e", "f"])
    with pytest.raises(ValueError, match="Duplicate feature names"):
        build_classic_cache(duplicated, subsample=None)
    nan_cols = pd.DataFrame(values, columns=[np.nan, np.nan, "c", "d", "e", "f"])
    with pytest.raises(ValueError, match="Duplicate feature names"):
        build_classic_cache(nan_cols, subsample=None)

    unique = pd.DataFrame(values, columns=list("abcdef"))
    cache = build_classic_cache(unique, subsample=None)
    duplicated_names = replace(cache, feature_names=["a", "a", "c", "d", "e", "f"])
    with pytest.raises(ValueError, match="Duplicate feature names"):
        select_mrmr(
            unique, y, k=1, task="regression", cache=duplicated_names, verbose=False
        )
    nan_names = replace(cache, feature_names=[np.nan, np.nan, "c", "d", "e", "f"])
    with pytest.raises(ValueError, match="Duplicate feature names"):
        select_mrmr(
            unique, y, k=1, task="regression", cache=nan_names, verbose=False
        )
