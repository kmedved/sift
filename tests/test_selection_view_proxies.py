"""Advanced SelectionView proxy and degradation contracts."""

from __future__ import annotations

import inspect
import pickle

import numpy as np
import pandas as pd
import pytest

import sift
from sift.selection.auto_k import AutoKConfig
import sift.selection.proxies as proxy_module


def _proxy_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=240)
    X = pd.DataFrame(
        {
            "signal": signal,
            "proxy": signal + 0.01 * rng.normal(size=len(signal)),
            "noise": rng.normal(size=len(signal)),
            "constant": np.ones(len(signal)),
        }
    )
    y = signal + 0.05 * rng.normal(size=len(signal))
    return X, y


@pytest.mark.parametrize(
    "method",
    ("cefsplus", "jmi", "jmim", "mrmr_quot", "mrmr_diff"),
)
def test_cached_proxy_storage_is_bounded_positional_and_excludes_selected(method):
    X, y = _proxy_data()
    cache = sift.build_cache(X, subsample=None, compute_Rxx=True)
    view = sift.select_cached(
        cache,
        y,
        k=1,
        method=method,
        top_m=3,
        corr_prune=None,
        warn_noise_floor=False,
        return_result=True,
        store_proxies=True,
    )

    assert view.metadata["proxy_correlations_stored"] is True
    assert view.metadata["proxy_candidate_count"] == 3
    assert view.metadata["proxy_storage_bytes"] == 3 * np.dtype(np.float32).itemsize
    assert view._proxy_correlations.to_numpy().dtype == np.float32

    stored = view._proxy_correlations[selected_index := view.indices[0]]
    expected_by_raw_position = {
        int(raw_position): cache.Rxx[local_position, np.flatnonzero(cache.valid_cols == selected_index)[0]]
        for local_position, raw_position in enumerate(cache.valid_cols)
    }
    for candidate_index, correlation in stored.items():
        assert correlation == expected_by_raw_position[int(candidate_index)]

    by_name = view.proxies(view.features[0], r_min=0.8)
    by_position = view.proxies_at(selected_index, r_min=0.8)
    pd.testing.assert_frame_equal(by_name, by_position)
    assert selected_index not in by_name["selected_index"].tolist()
    assert set(by_name["feature"]) == {"signal", "proxy"}.difference(view.features)
    assert (by_name["correlation"].abs() >= 0.8).all()

    before = by_name.copy(deep=True)
    X.iloc[:, :] = 0.0
    pd.testing.assert_frame_equal(view.proxies_at(selected_index, r_min=0.8), before)


@pytest.mark.parametrize(
    ("selector", "kwargs"),
    [
        (sift.select_cefsplus, {}),
        (sift.select_mrmr, {"task": "regression", "estimator": "gaussian"}),
        (sift.select_jmi, {"task": "regression", "estimator": "gaussian"}),
        (sift.select_jmim, {"task": "regression", "estimator": "gaussian"}),
    ],
)
def test_gaussian_filter_results_carry_opt_in_proxy_block(selector, kwargs):
    X, y = _proxy_data()
    result = selector(
        X,
        y,
        k=1,
        top_m=3,
        verbose=False,
        return_result=True,
        store_proxies=True,
        **kwargs,
    )

    assert type(result) is sift.FilterSelectionResult
    view = result.result_view(input_features=X.columns)
    assert view.metadata["proxy_correlations_stored"] is True
    assert not view.proxies(result.selected_features[0], r_min=0.8).empty

    restored = pickle.loads(pickle.dumps(result))
    restored_view = restored.result_view(input_features=X.columns)
    pd.testing.assert_frame_equal(
        restored_view.proxies_at(restored_view.indices[0], r_min=0.0),
        view.proxies_at(view.indices[0], r_min=0.0),
    )


def test_auto_k_and_brier_delegate_preserve_proxy_payload():
    X, y = _proxy_data()
    auto = sift.select_cefsplus(
        X,
        y,
        k="auto",
        auto_k_config=AutoKConfig(k_method="elbow", min_k=1, max_k=1),
        top_m=3,
        verbose=False,
        return_result=True,
        store_proxies=True,
    )
    auto_view = sift.as_result(auto, input_features=X.columns)
    assert auto_view.metadata["proxy_correlations_stored"] is True
    assert not auto_view.proxies_at(auto_view.indices[0], r_min=0.8).empty

    binary = sift.select_cefsplus_binary(
        X,
        y > np.median(y),
        k=1,
        loss="brier",
        top_m=3,
        verbose=False,
        return_result=True,
        store_proxies=True,
    )
    binary_view = sift.as_result(binary, input_features=X.columns)
    assert binary_view.metadata["proxy_correlations_stored"] is True
    assert not binary_view.proxies_at(binary_view.indices[0], r_min=0.8).empty


def test_proxy_opt_in_rejects_meaningless_and_unsupported_calls():
    X, y = _proxy_data()
    cache = sift.build_cache(X, subsample=None)

    with pytest.raises(ValueError, match="requires return_result=True"):
        sift.select_cached(cache, y, k=1, store_proxies=True)
    with pytest.raises(ValueError, match="requires return_result=True"):
        sift.select_cefsplus(X, y, k=1, store_proxies=True, verbose=False)
    with pytest.raises(ValueError, match="Gaussian/cached"):
        sift.select_mrmr(
            X,
            y,
            k=1,
            task="regression",
            estimator="classic",
            return_result=True,
            store_proxies=True,
            verbose=False,
        )
    with pytest.raises(ValueError, match="Gaussian/cached"):
        sift.select_cefsplus_binary(
            X,
            y > np.median(y),
            k=1,
            loss="logloss",
            return_result=True,
            store_proxies=True,
            verbose=False,
        )


def test_proxy_name_ambiguity_has_real_positional_escape_hatch():
    raw_features = ["dup", "dup", "selected", "noise"]
    raw_table = pd.DataFrame(
        {
            "feature": raw_features,
            "selected_index": pd.array(range(4), dtype="Int64"),
            "path_rank": pd.array([pd.NA, 1, 2, pd.NA], dtype="Int64"),
            "selected": [False, True, True, False],
            "relevance": [0.9, 1.0, 0.8, 0.1],
        }
    )
    correlations = pd.DataFrame(
        np.asarray(
            [
                [0.95, 0.10],
                [1.00, 0.20],
                [0.20, 1.00],
                [0.90, 0.10],
            ],
            dtype=np.float32,
        ),
        index=[0, 1, 2, 3],
        columns=[1, 2],
    )
    view = sift.SelectionView(
        features=["dup", "selected"],
        indices=[1, 2],
        raw_features=raw_features,
        n_raw_features=4,
        raw_table=raw_table,
        metadata={"table_complete": True},
        proxy_correlations=correlations,
    )

    with pytest.raises(ValueError, match=r"proxies_at\(selected_index"):
        view.proxies("dup")
    positional = view.proxies_at(1, r_min=0.8)
    assert positional["selected_index"].tolist() == [0, 3]
    assert positional["feature"].tolist() == ["dup", "noise"]
    assert not set(positional["selected_index"]).intersection(view.indices)


def test_proxy_storage_cap_is_enforced_before_retention(monkeypatch):
    monkeypatch.setattr(proxy_module, "MAX_PROXY_CORRELATION_BYTES", 4)
    raw_table = pd.DataFrame(
        {
            "feature": ["a", "b"],
            "selected_index": pd.array([0, 1], dtype="Int64"),
            "path_rank": pd.array([1, pd.NA], dtype="Int64"),
            "selected": [True, False],
        }
    )
    with pytest.raises(ValueError, match="exceeding the.*MiB limit"):
        sift.SelectionView(
            features=["a"],
            indices=[0],
            raw_features=["a", "b"],
            n_raw_features=2,
            raw_table=raw_table,
            metadata={"table_complete": True},
            proxy_correlations=pd.DataFrame([[1.0], [0.5]], index=[0, 1], columns=[0]),
        )


def test_partial_table_plot_fails_instead_of_silently_plotting_subset():
    partial = sift.FilterSelectionResult(
        selected_features=["a"],
        selected_indices=[0],
        selector_metadata={"selector": "fixture", "k": 1, "n_features": 2},
    )
    view = sift.as_result(partial)

    with pytest.raises(NotImplementedError, match="plot data is incomplete"):
        view.plot(ax=object())


def test_store_proxies_defaults_are_additive():
    for function in (
        sift.select_cached,
        sift.select_mrmr,
        sift.select_jmi,
        sift.select_jmim,
        sift.select_cefsplus,
        sift.select_cefsplus_binary,
    ):
        assert inspect.signature(function).parameters["store_proxies"].default is False
