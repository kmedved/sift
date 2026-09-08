from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from sift import MRMRSelector, select_mrmr
from sift.selection import filter_auto_k
from sift.selection.auto_k import AutoKConfig


def test_onehot_numeric_ndarray_is_a_noop_for_function_and_wrapper():
    rng = np.random.default_rng(17)
    X = rng.normal(size=(80, 5))
    y = X[:, 0] - 0.25 * X[:, 1]

    selected = select_mrmr(
        X, y, 2, task="regression", cat_encoding="onehot", verbose=False
    )
    selector = MRMRSelector(
        k=2, task="regression", cat_encoding="onehot", verbose=False
    ).fit(X, y)

    assert selected == selector.selected_features_
    assert selected == select_mrmr(
        X, y, 2, task="regression", cat_encoding="none", verbose=False
    )


def test_onehot_ndarray_with_categorical_mapping_fails_clearly():
    X = np.arange(40.0).reshape(20, 2)
    y = X[:, 0]

    with pytest.raises(TypeError, match="requires a pandas DataFrame"):
        select_mrmr(
            X,
            y,
            1,
            task="regression",
            cat_features=["x0"],
            cat_encoding="onehot",
            verbose=False,
        )

    with pytest.raises(TypeError, match="requires a pandas DataFrame"):
        MRMRSelector(
            k=1,
            task="regression",
            cat_features=["x0"],
            cat_encoding="onehot",
            verbose=False,
        ).fit(X, y)


def test_auto_conditioning_rejects_unsupported_router_method_before_work(monkeypatch):
    class DummyCache:
        sample_weight = np.ones(20, dtype=np.float64)
        valid_cols = np.arange(5, dtype=np.int64)

    calls = []

    def fake_route(config, _facts):
        return replace(config, k_method="perm_gap"), "heavy_weight_skew"

    def should_not_run(*_args, **_kwargs):
        calls.append(True)
        raise AssertionError("routed path should be preflighted first")

    monkeypatch.setattr(filter_auto_k, "_auto_route_config", fake_route)
    monkeypatch.setattr(filter_auto_k, "_run_gaussian_routed_path", should_not_run)

    with pytest.raises(
        ValueError,
        match=r"k_method='auto'.*unsupported k_method='perm_gap'",
    ):
        filter_auto_k.select_gaussian_auto_path(
            cache=DummyCache(),
            y=np.zeros(20),
            method="cefsplus",
            max_k=4,
            top_m=5,
            auto_k_config=AutoKConfig(k_method="auto"),
            include=["x0"],
            verbose=False,
        )

    assert calls == []
