import numpy as np
import pandas as pd
import pytest

from sift import build_cache, select_cefsplus, select_jmi, select_jmim, select_mrmr
import sift.selection.filter_payloads as filter_payloads
from sift.selection.auto_k import AutoKConfig


def _high_cardinality_data(n=80):
    rng = np.random.default_rng(20260420)
    signal = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "id": [f"id_{i}" for i in range(n)],
            "signal": signal,
            "noise_a": rng.normal(size=n),
            "noise_b": rng.normal(size=n),
        }
    )
    y = signal + 0.1 * rng.normal(size=n)
    return X, y


def _patch_encoder(monkeypatch):
    def fake_encode_categoricals(X, y, cat_features, method, **_kwargs):
        X_enc = X.copy()
        for col in cat_features:
            X_enc[col] = pd.factorize(X_enc[col])[0].astype(float)
        return X_enc

    monkeypatch.setattr(filter_payloads, "encode_categoricals", fake_encode_categoricals)


FUNCTION_SELECTORS = [
    pytest.param(
        select_mrmr,
        {"task": "regression", "estimator": "classic"},
        id="mrmr-classic",
    ),
    pytest.param(
        select_jmi,
        {"task": "regression", "estimator": "r2"},
        id="jmi-r2",
    ),
    pytest.param(
        select_jmim,
        {"task": "regression", "estimator": "r2"},
        id="jmim-r2",
    ),
    pytest.param(select_cefsplus, {}, id="cefsplus"),
]


@pytest.mark.parametrize("selector, kwargs", FUNCTION_SELECTORS)
@pytest.mark.parametrize("cat_encoding", ["target", "loo", "james_stein"])
def test_function_selectors_reject_supervised_cat_encoding_by_default(
    selector,
    kwargs,
    cat_encoding,
):
    X, y = _high_cardinality_data()

    with pytest.raises(ValueError, match="allow_full_data_target_encoding=True"):
        selector(
            X,
            y,
            k=2,
            cat_features=["id"],
            cat_encoding=cat_encoding,
            subsample=None,
            verbose=False,
            **kwargs,
        )


@pytest.mark.parametrize("selector, kwargs", FUNCTION_SELECTORS)
@pytest.mark.parametrize("cat_encoding", ["target", "loo"])
def test_function_selectors_allow_explicit_full_data_target_encoding_opt_in(
    selector,
    kwargs,
    cat_encoding,
    monkeypatch,
):
    _patch_encoder(monkeypatch)
    X, y = _high_cardinality_data()

    selected = selector(
        X,
        y,
        k=2,
        cat_features=["id"],
        cat_encoding=cat_encoding,
        allow_full_data_target_encoding=True,
        subsample=None,
        verbose=False,
        **kwargs,
    )

    assert 0 < len(selected) <= 2


def test_function_selector_auto_k_rejects_supervised_cat_encoding_by_default():
    X, y = _high_cardinality_data()
    cfg = AutoKConfig(k_method="elbow", min_k=1, max_k=2)

    with pytest.raises(ValueError, match="allow_full_data_target_encoding=True"):
        select_mrmr(
            X,
            y,
            k="auto",
            task="regression",
            estimator="gaussian",
            auto_k_config=cfg,
            cat_features=["id"],
            cat_encoding="target",
            subsample=None,
            verbose=False,
        )


def test_function_selector_auto_k_allows_explicit_target_encoding_opt_in(monkeypatch):
    _patch_encoder(monkeypatch)
    X, y = _high_cardinality_data()
    cfg = AutoKConfig(k_method="elbow", min_k=1, max_k=2)

    selected = select_mrmr(
        X,
        y,
        k="auto",
        task="regression",
        estimator="gaussian",
        auto_k_config=cfg,
        cat_features=["id"],
        cat_encoding="target",
        allow_full_data_target_encoding=True,
        subsample=None,
        verbose=False,
    )

    assert 0 < len(selected) <= 2


def test_category_encoder_backed_target_encoding_rejects_sample_weight():
    X, y = _high_cardinality_data()

    with pytest.raises(ValueError, match="sample_weight.*loo_logit"):
        select_cefsplus(
            X,
            y,
            k=2,
            cat_features=["id"],
            cat_encoding="target",
            allow_full_data_target_encoding=True,
            sample_weight=np.ones(len(y)),
            subsample=None,
            verbose=False,
        )


def test_gaussian_auto_k_with_prebuilt_cache_does_not_require_encoding_opt_in():
    X, y = _high_cardinality_data()
    X_cached = X[["signal", "noise_a", "noise_b"]]
    cache = build_cache(X_cached, subsample=None)
    cfg = AutoKConfig(k_method="elbow", min_k=1, max_k=2)

    with pytest.raises(ValueError, match="names and order"):
        select_mrmr(
            X,
            y,
            k="auto",
            task="regression",
            estimator="gaussian",
            cache=cache,
            auto_k_config=cfg,
            verbose=False,
        )

    selected = select_mrmr(
        X_cached,
        y,
        k="auto",
        task="regression",
        estimator="gaussian",
        cache=cache,
        auto_k_config=cfg,
        cat_features=["id"],
        cat_encoding="target",
        verbose=False,
    )

    assert 0 < len(selected) <= 2


def test_cefsplus_with_prebuilt_cache_does_not_require_encoding_opt_in():
    X, y = _high_cardinality_data()
    X_cached = X[["signal", "noise_a", "noise_b"]]
    cache = build_cache(X_cached, subsample=None)

    with pytest.raises(ValueError, match="names and order"):
        select_cefsplus(X, y, k=2, cache=cache, verbose=False)

    selected = select_cefsplus(
        X_cached,
        y,
        k=2,
        cache=cache,
        cat_features=["id"],
        cat_encoding="target",
        verbose=False,
    )

    assert 0 < len(selected) <= 2


def test_function_selector_cat_encoding_none_keeps_non_numeric_validation():
    X, y = _high_cardinality_data()

    with pytest.raises(ValueError, match="Non-numeric columns"):
        select_mrmr(
            X,
            y,
            k=2,
            task="regression",
            cat_features=["id"],
            cat_encoding="none",
            subsample=None,
            verbose=False,
        )


@pytest.mark.parametrize("selector, kwargs", FUNCTION_SELECTORS)
def test_function_selectors_do_not_require_opt_in_without_categorical_columns(
    selector,
    kwargs,
):
    rng = np.random.default_rng(20260421)
    X = pd.DataFrame(rng.normal(size=(60, 4)), columns=[f"x{i}" for i in range(4)])
    y = X["x0"].to_numpy() + 0.1 * rng.normal(size=len(X))

    selected = selector(
        X,
        y,
        k=2,
        cat_encoding="target",
        subsample=None,
        verbose=False,
        **kwargs,
    )

    assert 0 < len(selected) <= 2
