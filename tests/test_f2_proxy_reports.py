"""Public-contract tests for F2 redundancy reports and proxy clusters."""

from __future__ import annotations

import inspect
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

import sift
from sift.selection.proxies import MAX_RESAMPLE_SELECTION_BYTES
import sift.selection.proxies as proxy_module


def _report_block():
    raw_features = ["dup", "dup", "anchor", "bridge", "other", "noise"]
    raw_table = pd.DataFrame(
        {
            "feature": raw_features,
            "selected_index": pd.array(range(6), dtype="Int64"),
            "path_rank": pd.array([1, pd.NA, 2, pd.NA, 3, pd.NA], dtype="Int64"),
            "selected": [True, False, True, False, True, False],
        }
    )
    correlations = pd.DataFrame(
        np.asarray(
            [
                [1.00, 0.10, 0.05],
                [0.96, 0.12, 0.04],
                [0.10, 1.00, 0.08],
                [0.91, 0.88, 0.05],
                [0.05, 0.08, 1.00],
                [0.20, 0.10, 0.15],
            ],
            dtype=np.float32,
        ),
        index=[0, 1, 2, 3, 4, 5],
        columns=[0, 2, 4],
    )
    view = sift.SelectionView(
        features=["dup", "anchor", "other"],
        indices=[0, 2, 4],
        raw_features=raw_features,
        n_raw_features=6,
        raw_table=raw_table,
        metadata={"table_complete": True},
        proxy_correlations=correlations,
    )
    return view, correlations


def test_redundancy_report_lists_unselected_edges_in_path_order():
    view, _ = _report_block()
    report = view.redundancy_report(r_min=0.8)
    assert list(report.columns) == [
        "selected_feature",
        "selected_index",
        "feature",
        "candidate_index",
        "correlation",
    ]
    assert report["candidate_index"].tolist() == [1, 3, 3]
    assert report["selected_index"].tolist() == [0, 0, 2]
    assert 0 not in report["candidate_index"].tolist()
    assert 2 not in report["candidate_index"].tolist()
    assert 4 not in report["candidate_index"].tolist()
    assert report["correlation"].tolist() == pytest.approx([0.96, 0.91, 0.88])


def test_proxy_clusters_join_multi_anchor_components():
    view, _ = _report_block()
    clusters = view.proxy_clusters(r_min=0.8)
    assert list(clusters.columns) == [
        "cluster_id",
        "feature",
        "selected_index",
        "selected",
        "cluster_frequency",
    ]
    by_cluster = {
        int(cluster_id): group["selected_index"].tolist()
        for cluster_id, group in clusters.groupby("cluster_id", sort=True)
    }
    assert by_cluster[0] == [0, 2, 1, 3]
    assert by_cluster[1] == [4]
    assert clusters["cluster_frequency"].isna().all()
    assert view.metadata["cluster_frequencies_available"] is False


def test_duplicate_labels_use_raw_positions():
    view, _ = _report_block()
    report = view.redundancy_report(r_min=0.8)
    first = report.iloc[0]
    assert first["selected_feature"] == "dup"
    assert int(first["selected_index"]) == 0
    assert first["feature"] == "dup"
    assert int(first["candidate_index"]) == 1
    clusters = view.proxy_clusters(r_min=0.8)
    selected_dups = clusters.loc[clusters["selected_index"].isin([0, 1])]
    assert set(selected_dups["selected"]) == {True, False}


def test_missing_proxy_storage_raises_store_proxies_guidance():
    X, y = _proxy_xy()
    view = sift.select_cefsplus(
        X, y, k=1, verbose=False, return_result=True, store_proxies=False
    )
    view = sift.as_result(view, input_features=X.columns)
    with pytest.raises(NotImplementedError, match="store_proxies=True"):
        view.redundancy_report()
    with pytest.raises(NotImplementedError, match="store_proxies=True"):
        view.proxy_clusters()


@pytest.mark.parametrize("bad", [True, False, float("nan"), float("inf"), -0.1, 1.1, "0.8"])
def test_r_min_rejected_once_for_reports(bad):
    view, _ = _report_block()
    with pytest.raises(ValueError, match="r_min"):
        view.redundancy_report(r_min=bad)
    with pytest.raises(ValueError, match="r_min"):
        view.proxy_clusters(r_min=bad)
    with pytest.raises(ValueError, match="r_min"):
        view.proxies_at(0, r_min=bad)


def test_threshold_boundaries_and_zero_selection():
    view, _ = _report_block()
    at_one = view.redundancy_report(r_min=1.0)
    assert at_one.empty
    at_zero = view.redundancy_report(r_min=0.0)
    assert set(at_zero["candidate_index"]) == {1, 3, 5}
    empty_table = pd.DataFrame(
        {
            "feature": ["a", "b"],
            "selected_index": pd.array([0, 1], dtype="Int64"),
            "path_rank": pd.array([pd.NA, pd.NA], dtype="Int64"),
            "selected": [False, False],
        }
    )
    empty = sift.SelectionView(
        features=[],
        indices=[],
        raw_features=["a", "b"],
        n_raw_features=2,
        raw_table=empty_table,
        metadata={"table_complete": True},
        proxy_correlations=pd.DataFrame(
            np.zeros((2, 0), dtype=np.float32),
            index=[0, 1],
            columns=[],
        ),
    )
    assert empty.redundancy_report().empty
    assert empty.proxy_clusters().empty


def test_copy_pickle_json_omit_hidden_matrices():
    view, _ = _report_block()
    payload = view.to_dict()
    dumped = str(payload)
    assert "0.96" not in dumped
    assert "resample_selections" not in payload
    restored = pickle.loads(pickle.dumps(view))
    pd.testing.assert_frame_equal(
        restored.redundancy_report(r_min=0.8),
        view.redundancy_report(r_min=0.8),
    )
    pd.testing.assert_frame_equal(
        restored.proxy_clusters(r_min=0.8),
        view.proxy_clusters(r_min=0.8),
    )


def test_gaussian_producers_support_aggregate_reports():
    X, y = _proxy_xy()
    view = sift.select_cefsplus(
        X,
        y,
        k=1,
        top_m=3,
        verbose=False,
        return_result=True,
        store_proxies=True,
        subsample=None,
    )
    view = sift.as_result(view, input_features=X.columns)
    report = view.redundancy_report(r_min=0.8)
    assert view.features[0] not in set(report["feature"])
    clusters = view.proxy_clusters(r_min=0.8)
    assert view.indices[0] in set(clusters["selected_index"])
    assert clusters["cluster_frequency"].isna().all()


def test_rank_gaussian_invariance_under_monotone_transforms():
    rng = np.random.default_rng(4)
    signal = rng.normal(size=180)
    base = pd.DataFrame(
        {
            "signal": signal,
            "proxy": signal + 0.01 * rng.normal(size=len(signal)),
            "noise": rng.normal(size=len(signal)),
        }
    )
    y = signal + 0.05 * rng.normal(size=len(signal))
    increasing = 2.0 * base + 3.0
    decreasing = -base
    views = []
    for frame in (base, increasing, decreasing):
        cache = sift.build_cache(frame, subsample=None, compute_Rxx=True)
        views.append(
            sift.select_cached(
                cache,
                y,
                k=1,
                method="cefsplus",
                top_m=3,
                corr_prune=None,
                return_result=True,
                store_proxies=True,
            )
        )
    base_report = views[0].redundancy_report(r_min=0.0).sort_values(
        ["selected_index", "candidate_index"]
    )
    up_report = views[1].redundancy_report(r_min=0.0).sort_values(
        ["selected_index", "candidate_index"]
    )
    down_report = views[2].redundancy_report(r_min=0.0).sort_values(
        ["selected_index", "candidate_index"]
    )
    pd.testing.assert_frame_equal(
        base_report.reset_index(drop=True),
        up_report.reset_index(drop=True),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        base_report["correlation"].to_numpy(),
        down_report["correlation"].to_numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
    pd.testing.assert_frame_equal(
        views[0].proxy_clusters(r_min=0.8)[["cluster_id", "selected_index", "selected"]],
        views[2].proxy_clusters(r_min=0.8)[["cluster_id", "selected_index", "selected"]],
    )
    one_flipped = base.copy()
    one_flipped["proxy"] = -one_flipped["proxy"]
    cache_flip = sift.build_cache(one_flipped, subsample=None, compute_Rxx=True)
    flipped = sift.select_cached(
        cache_flip,
        y,
        k=1,
        method="cefsplus",
        top_m=3,
        corr_prune=None,
        return_result=True,
        store_proxies=True,
    )
    base_edge = (
        views[0]
        .redundancy_report(r_min=0.0)
        .set_index(["selected_index", "candidate_index"])["correlation"]
    )
    flip_edge = (
        flipped.redundancy_report(r_min=0.0)
        .set_index(["selected_index", "candidate_index"])["correlation"]
    )
    proxy_pos = list(base.columns).index("proxy")
    selected_pos = views[0].indices[0]
    key = (selected_pos, proxy_pos)
    if selected_pos != proxy_pos and key in base_edge.index and key in flip_edge.index:
        assert flip_edge.loc[key] == pytest.approx(-float(base_edge.loc[key]), rel=1e-5, abs=1e-5)
    pd.testing.assert_frame_equal(
        views[0].proxy_clusters(r_min=0.8)[["cluster_id", "selected_index", "selected"]],
        flipped.proxy_clusters(r_min=0.8)[["cluster_id", "selected_index", "selected"]],
    )


def test_stability_cluster_frequency_on_controlled_resamples():
    view, _ = _report_block()
    resamples = np.zeros((4, 6), dtype=bool)
    resamples[0, 0] = True
    resamples[1, 1] = True
    resamples[2, 2] = True
    resamples[3, 3] = True
    with_freq = sift.SelectionView(
        features=view.features,
        indices=view.indices,
        raw_features=view.raw_features,
        n_raw_features=6,
        raw_table=view.raw_table,
        metadata={"table_complete": True},
        proxy_correlations=view._proxy_correlations,
        resample_selections=resamples,
    )
    clusters = with_freq.proxy_clusters(r_min=0.8)
    freq = {
        int(cluster_id): float(group["cluster_frequency"].iloc[0])
        for cluster_id, group in clusters.groupby("cluster_id")
    }
    assert freq[0] == pytest.approx(1.0)
    assert freq[1] == pytest.approx(0.0)
    assert with_freq.metadata["cluster_frequencies_available"] is True
    assert with_freq.metadata["n_resamples_stored"] == 4
    payload = with_freq.to_dict()
    assert "resample_selections" not in payload
    assert payload["metadata"]["cluster_frequencies_available"] is True


def test_stability_selector_store_proxies_and_cluster_frequency(monkeypatch):
    rng = np.random.default_rng(7)
    n = 80
    signal = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "a": signal,
            "b": signal + 1e-3 * rng.normal(size=n),
            "noise": rng.normal(size=n),
        }
    )
    y = signal + 0.05 * rng.normal(size=n)
    state = {"i": 0}

    def fake_fit(self, X_scaled, y_arr, sample_weight, train_idx, seed):
        del y_arr, sample_weight, train_idx, seed
        p = X_scaled.shape[1]
        selected = np.zeros(p, dtype=np.int8)
        coef = np.zeros(p, dtype=np.float32)
        chosen = state["i"] % 2
        selected[chosen] = 1
        coef[chosen] = 1.0
        state["i"] += 1
        return selected, coef

    monkeypatch.setattr(
        sift.StabilitySelector,
        "_fit_single_stability_run",
        fake_fit,
    )
    selector = sift.StabilitySelector(
        n_bootstrap=10,
        threshold=0.4,
        max_features=1,
        store_coefs=False,
        store_proxies=True,
        random_state=0,
        verbose=False,
        n_jobs=1,
    )
    selector.fit(X, y)
    assert selector.selected_feature_names_ == ["a"]
    assert selector.selection_frequencies_[0] == pytest.approx(0.5)
    assert selector.selection_frequencies_[1] == pytest.approx(0.5)
    view = selector.result_view_
    clusters = view.proxy_clusters(r_min=0.8)
    selected_cluster = clusters.loc[clusters["selected_index"] == 0, "cluster_id"].iloc[0]
    members = clusters.loc[clusters["cluster_id"] == selected_cluster]
    assert set(members["selected_index"]) >= {0, 1}
    assert float(members["cluster_frequency"].iloc[0]) == pytest.approx(1.0)
    assert not hasattr(selector, "coef_bootstrap_")


def test_stability_store_proxies_zero_selection_and_ndarray_names():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(40, 4))
    y = rng.normal(size=40)
    selector = sift.StabilitySelector(
        n_bootstrap=5,
        threshold=1.0,
        store_proxies=True,
        store_coefs=False,
        random_state=0,
        verbose=False,
        n_jobs=1,
    )
    selector.fit(X, y)
    view = selector.result_view_
    assert view.k == 0
    assert view.redundancy_report(r_min=0.8).empty
    assert view.proxy_clusters(r_min=0.8).empty
    assert view.metadata["proxy_correlations_stored"] is True
    assert view.metadata["cluster_frequencies_available"] is True


def test_stability_clone_default_and_failed_refit_cleanup(monkeypatch):
    X, y = _proxy_xy()
    selector = sift.StabilitySelector(random_state=0, verbose=False, n_jobs=1)
    assert inspect.signature(sift.StabilitySelector).parameters["store_proxies"].default is False
    cloned = clone(selector)
    assert cloned.store_proxies is False
    selector.fit(X, y)
    assert not hasattr(selector, "_proxy_correlations")

    def boom(*args, **kwargs):
        raise RuntimeError("forced fit failure")

    monkeypatch.setattr(sift.StabilitySelector, "_run_stability_chunks", boom)
    with pytest.raises(RuntimeError, match="forced fit failure"):
        selector.fit(X, y)
    assert not hasattr(selector, "selected_features_")
    assert not hasattr(selector, "_proxy_correlations")
    assert not hasattr(selector, "_resample_selections_")


def test_stability_sample_weights_change_copula_block(monkeypatch):
    rng = np.random.default_rng(8)
    n = 60
    signal = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "signal": signal,
            "proxy": np.concatenate([signal[: n // 2], rng.normal(size=n - n // 2)]),
            "noise": rng.normal(size=n),
        }
    )
    y = signal + 0.05 * rng.normal(size=n)
    weights = np.ones(n)
    weights[: n // 2] = 25.0

    def finalize(self, sel_count, sum_abs_coef, n_runs, feature_names):
        del sel_count, sum_abs_coef, n_runs
        self.selection_frequencies_ = np.zeros(len(feature_names), dtype=np.float64)
        self.selection_frequencies_[0] = 1.0
        self.mean_abs_coef_ = np.ones(len(feature_names), dtype=np.float32)
        self.selected_features_ = np.array([0], dtype=np.int64)
        self.selected_feature_names_ = [feature_names[0]]
        self.n_features_selected_ = 1

    monkeypatch.setattr(sift.StabilitySelector, "_finalize_stability_selection", finalize)
    common = dict(
        n_bootstrap=4,
        threshold=0.2,
        store_proxies=True,
        store_coefs=False,
        random_state=1,
        verbose=False,
        n_jobs=1,
    )
    unweighted = sift.StabilitySelector(**common).fit(X, y)
    weighted = sift.StabilitySelector(**common).fit(X, y, sample_weight=weights)
    left = unweighted.result_view_.redundancy_report(r_min=0.0)
    right = weighted.result_view_.redundancy_report(r_min=0.0)
    merged = left.merge(
        right,
        on=["selected_index", "candidate_index"],
        suffixes=("_u", "_w"),
    )
    assert not merged.empty
    assert not np.allclose(merged["correlation_u"], merged["correlation_w"])


def _tiny_proxy_table(n_features=2):
    selected = [True] + [False] * (n_features - 1)
    ranks = [1] + [pd.NA] * (n_features - 1)
    names = [chr(ord("a") + i) for i in range(n_features)]
    return pd.DataFrame(
        {
            "feature": names,
            "selected_index": pd.array(range(n_features), dtype="Int64"),
            "path_rank": pd.array(ranks, dtype="Int64"),
            "selected": selected,
        }
    )


def test_resample_selection_cap_is_enforced(monkeypatch):
    monkeypatch.setattr(proxy_module, "MAX_RESAMPLE_SELECTION_BYTES", 1)
    raw_table = _tiny_proxy_table(8)
    correlations = pd.DataFrame(
        np.zeros((8, 1), dtype=np.float32),
        index=range(8),
        columns=[0],
    )
    correlations.iloc[0, 0] = 1.0
    with pytest.raises(ValueError, match="MiB limit"):
        sift.SelectionView(
            features=["a"],
            indices=[0],
            raw_features=list(raw_table["feature"]),
            n_raw_features=8,
            raw_table=raw_table,
            metadata={"table_complete": True},
            proxy_correlations=correlations,
            resample_selections=np.ones((1, 8), dtype=bool),
        )
    monkeypatch.setattr(proxy_module, "MAX_RESAMPLE_SELECTION_BYTES", 8)
    view = sift.SelectionView(
        features=["a"],
        indices=[0],
        raw_features=list(raw_table["feature"]),
        n_raw_features=8,
        raw_table=raw_table,
        metadata={"table_complete": True},
        proxy_correlations=correlations,
        resample_selections=np.ones((1, 8), dtype=bool),
    )
    assert view._resample_selections.nbytes == 8
    assert view.metadata["resample_selection_storage_bytes"] == 8
    assert MAX_RESAMPLE_SELECTION_BYTES == 16 * 1024**2


def test_proxy_clusters_union_direct_selected_selected_edges():
    raw_table = pd.DataFrame(
        {
            "feature": ["a", "b"],
            "selected_index": pd.array([0, 1], dtype="Int64"),
            "path_rank": pd.array([1, 2], dtype="Int64"),
            "selected": [True, True],
        }
    )
    correlations = pd.DataFrame(
        np.asarray([[1.0, 0.95], [0.95, 1.0]], dtype=np.float32),
        index=[0, 1],
        columns=[0, 1],
    )
    resamples = np.asarray([[True, False], [False, True]], dtype=bool)
    view = sift.SelectionView(
        features=["a", "b"],
        indices=[0, 1],
        raw_features=["a", "b"],
        n_raw_features=2,
        raw_table=raw_table,
        metadata={"table_complete": True},
        proxy_correlations=correlations,
        resample_selections=resamples,
    )
    assert view.redundancy_report(r_min=0.8).empty
    clusters = view.proxy_clusters(r_min=0.8)
    assert set(clusters["cluster_id"]) == {0}
    assert set(clusters["selected_index"]) == {0, 1}
    assert float(clusters["cluster_frequency"].iloc[0]) == pytest.approx(1.0)


def _oracle_cluster_members(block, selected, r_min):
    parent = {int(pos): int(pos) for pos in selected}

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left, right):
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    selected_list = [int(i) for i in selected]
    for selected_pos in selected_list:
        values = block[selected_pos]
        for candidate_pos, correlation in zip(
            np.asarray(block.index, dtype=np.int64),
            values.to_numpy(dtype=np.float64),
        ):
            candidate_pos = int(candidate_pos)
            if candidate_pos == selected_pos:
                continue
            if abs(float(correlation)) >= r_min:
                union(selected_pos, candidate_pos)
    groups = {}
    for selected_pos in selected_list:
        groups.setdefault(find(selected_pos), set()).add(selected_pos)
    for node in parent:
        if node in set(selected_list):
            continue
        root = find(node)
        if root in groups:
            groups[root].add(node)
    return {frozenset(members) for members in groups.values()}


def test_proxy_clusters_match_independent_graph_oracle():
    rng = np.random.default_rng(11)
    selected = [0, 2, 5]
    names = [f"f{i}" for i in range(7)]
    for _ in range(200):
        seed = rng.normal(size=(7, 7))
        gram = seed.T @ seed
        scale = np.sqrt(np.diag(gram))
        corr = gram / np.outer(scale, scale)
        np.fill_diagonal(corr, 1.0)
        block = pd.DataFrame(
            np.asarray(corr[:, selected], dtype=np.float32),
            index=range(7),
            columns=selected,
        )
        path_rank = pd.array(
            [selected.index(i) + 1 if i in selected else pd.NA for i in range(7)],
            dtype="Int64",
        )
        raw_table = pd.DataFrame(
            {
                "feature": names,
                "selected_index": pd.array(range(7), dtype="Int64"),
                "path_rank": path_rank,
                "selected": [i in selected for i in range(7)],
            }
        )
        view = sift.SelectionView(
            features=[names[i] for i in selected],
            indices=list(selected),
            raw_features=names,
            n_raw_features=7,
            raw_table=raw_table,
            metadata={"table_complete": True},
            proxy_correlations=block,
        )
        impl = {
            frozenset(group["selected_index"].tolist())
            for _, group in view.proxy_clusters(r_min=0.6).groupby("cluster_id")
        }
        assert impl == _oracle_cluster_members(block, selected, 0.6)


def test_resample_without_proxy_block_is_rejected():
    raw_table = _tiny_proxy_table(2)
    with pytest.raises(ValueError, match="proxy_correlations"):
        sift.SelectionView(
            features=["a"],
            indices=[0],
            raw_features=["a", "b"],
            n_raw_features=2,
            raw_table=raw_table,
            metadata={"table_complete": True},
            resample_selections=np.ones((2, 2), dtype=bool),
        )


def test_set_threshold_hides_unusable_proxy_and_resample_payload(monkeypatch):
    rng = np.random.default_rng(12)
    n = 50
    signal = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "a": signal,
            "b": signal + 0.02 * rng.normal(size=n),
            "noise": rng.normal(size=n),
        }
    )
    y = signal + 0.05 * rng.normal(size=n)
    state = {"i": 0}

    def fake_fit(self, X_scaled, y_arr, sample_weight, train_idx, seed):
        del y_arr, sample_weight, train_idx, seed
        p = X_scaled.shape[1]
        selected = np.zeros(p, dtype=np.int8)
        coef = np.zeros(p, dtype=np.float32)
        selected[0] = 1
        coef[0] = 1.0
        if state["i"] < 6:
            selected[1] = 1
            coef[1] = 1.0
        state["i"] += 1
        return selected, coef

    monkeypatch.setattr(sift.StabilitySelector, "_fit_single_stability_run", fake_fit)
    selector = sift.StabilitySelector(
        n_bootstrap=10,
        threshold=0.75,
        store_coefs=False,
        store_proxies=True,
        random_state=0,
        verbose=False,
        n_jobs=1,
    )
    selector.fit(X, y)
    assert list(selector.selected_features_) == [0]
    assert hasattr(selector, "_proxy_correlations")
    assert hasattr(selector, "_resample_selections_")
    expanded = selector.set_threshold(0.5).result_view_
    assert expanded.metadata["proxy_correlations_stored"] is False
    assert expanded.metadata["cluster_frequencies_available"] is False
    with pytest.raises(NotImplementedError, match="store_proxies=True"):
        expanded.proxy_clusters()
    restored = selector.set_threshold(0.75).result_view_
    assert restored.metadata["proxy_correlations_stored"] is True
    assert restored.metadata["cluster_frequencies_available"] is True
    assert not restored.proxy_clusters(r_min=0.8).empty


def test_zero_weight_extreme_rows_match_dropped_rows(monkeypatch):
    n = 20
    x = np.linspace(0.0, 1.0, n)
    noise = np.linspace(-0.2, 0.2, n)
    X_core = pd.DataFrame({"signal": x, "noise": noise})
    y_core = x.copy()
    extreme = pd.DataFrame(
        {"signal": np.full(5, 1e15), "noise": np.full(5, 1e15)}
    )
    X = pd.concat([X_core, extreme], ignore_index=True)
    y = np.concatenate([y_core, np.zeros(5)])
    weights = np.concatenate([np.ones(n), np.zeros(5)])

    def finalize(self, sel_count, sum_abs_coef, n_runs, feature_names):
        del sel_count, sum_abs_coef, n_runs
        self.selection_frequencies_ = np.array([1.0, 0.0])
        self.mean_abs_coef_ = np.array([1.0, 0.0], dtype=np.float32)
        self.selected_features_ = np.array([0], dtype=np.int64)
        self.selected_feature_names_ = [feature_names[0]]
        self.n_features_selected_ = 1

    monkeypatch.setattr(sift.StabilitySelector, "_finalize_stability_selection", finalize)
    common = dict(
        n_bootstrap=3,
        threshold=0.2,
        store_proxies=True,
        store_coefs=False,
        random_state=0,
        verbose=False,
        n_jobs=1,
    )
    with_zeros = sift.StabilitySelector(**common).fit(X, y, sample_weight=weights)
    dropped = sift.StabilitySelector(**common).fit(X_core, y_core)
    left = with_zeros.result_view_.redundancy_report(r_min=0.0)
    right = dropped.result_view_.redundancy_report(r_min=0.0)
    pd.testing.assert_frame_equal(
        left.reset_index(drop=True),
        right.reset_index(drop=True),
        rtol=1e-6,
        atol=1e-6,
    )


def test_missing_values_ignore_zero_weight_rows(monkeypatch):
    a = np.array([0.0, 1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    b = np.arange(10, dtype=np.float64)
    y = np.arange(10, dtype=np.float64)
    X = pd.DataFrame({"a": a, "b": b})
    extra = pd.DataFrame({"a": np.full(5, 10000.0), "b": np.full(5, -10000.0)})
    X_ext = pd.concat([X, extra], ignore_index=True)
    y_ext = np.concatenate([y, np.zeros(5)])
    weights = np.concatenate([np.ones(10), np.zeros(5)])

    def finalize(self, sel_count, sum_abs_coef, n_runs, feature_names):
        del sel_count, sum_abs_coef, n_runs
        self.selection_frequencies_ = np.array([1.0, 0.0])
        self.mean_abs_coef_ = np.array([1.0, 0.0], dtype=np.float32)
        self.selected_features_ = np.array([0], dtype=np.int64)
        self.selected_feature_names_ = [feature_names[0]]
        self.n_features_selected_ = 1

    monkeypatch.setattr(sift.StabilitySelector, "_finalize_stability_selection", finalize)
    common = dict(
        n_bootstrap=3,
        threshold=0.2,
        store_proxies=True,
        store_coefs=False,
        random_state=0,
        verbose=False,
        n_jobs=1,
    )
    baseline = sift.StabilitySelector(**common).fit(X, y)
    contaminated = sift.StabilitySelector(**common).fit(
        X_ext, y_ext, sample_weight=weights
    )
    left = float(baseline._proxy_correlations.loc[1, 0])
    right = float(contaminated._proxy_correlations.loc[1, 0])
    assert right == pytest.approx(left, rel=0, abs=0)
    assert left == pytest.approx(0.97429746, rel=1e-5, abs=1e-5)


def test_threshold_zero_selected_constant_is_singleton_cluster():
    rng = np.random.default_rng(13)
    n = 40
    signal = rng.normal(size=n)
    X = pd.DataFrame({"signal": signal, "constant": np.ones(n)})
    y = signal + 0.05 * rng.normal(size=n)
    plain = sift.StabilitySelector(
        n_bootstrap=4,
        threshold=0.0,
        store_proxies=False,
        store_coefs=False,
        random_state=0,
        verbose=False,
        n_jobs=1,
    ).fit(X, y)
    with_proxies = sift.StabilitySelector(
        n_bootstrap=4,
        threshold=0.0,
        store_proxies=True,
        store_coefs=False,
        random_state=0,
        verbose=False,
        n_jobs=1,
    ).fit(X, y)
    assert set(plain.selected_feature_names_) == set(with_proxies.selected_feature_names_)
    view = with_proxies.result_view_
    clusters = view.proxy_clusters(r_min=0.8)
    const_rows = clusters.loc[clusters["feature"] == "constant"]
    assert not const_rows.empty
    const_id = int(const_rows["cluster_id"].iloc[0])
    assert set(clusters.loc[clusters["cluster_id"] == const_id, "feature"]) == {"constant"}


def test_stability_proxy_uses_direct_column_block(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("full correlation matrix must not be computed")

    monkeypatch.setattr(
        "sift.estimators.copula.weighted_correlation_matrix",
        boom,
    )
    X, y = _proxy_xy(3)
    selector = sift.StabilitySelector(
        n_bootstrap=4,
        threshold=0.2,
        store_proxies=True,
        store_coefs=False,
        random_state=0,
        verbose=False,
        n_jobs=1,
    )
    selector.fit(X, y)
    assert selector.result_view_.metadata["proxy_correlations_stored"] is True


def test_proxy_block_cap_is_checked_before_rank(monkeypatch):
    monkeypatch.setattr(proxy_module, "MAX_PROXY_CORRELATION_BYTES", 4)

    def boom(*args, **kwargs):
        raise AssertionError("rank/correlation must not run after cap failure")

    monkeypatch.setattr("sift.estimators.copula.weighted_rank_gauss_2d", boom)
    monkeypatch.setattr(proxy_module, "weighted_correlation_columns", boom)
    rng = np.random.default_rng(14)
    X = pd.DataFrame(rng.normal(size=(30, 5)), columns=list("abcde"))
    y = X["a"] + 0.1 * rng.normal(size=30)
    with pytest.raises(ValueError, match="MiB limit"):
        sift.StabilitySelector(
            n_bootstrap=3,
            threshold=0.0,
            store_proxies=True,
            store_coefs=False,
            random_state=0,
            verbose=False,
            n_jobs=1,
        ).fit(X, y)


def test_direct_weighted_block_matches_full_correlation_oracle():
    rng = np.random.default_rng(15)
    n, p = 40, 6
    Z = rng.normal(size=(n, p))
    w = rng.random(n) + 0.1
    sqrt_w = np.sqrt(w)
    full = (Z * sqrt_w[:, None]).T @ (Z * sqrt_w[:, None]) / w.sum()
    np.clip(full, -0.999999, 0.999999, out=full)
    np.fill_diagonal(full, 1.0)
    selected = [0, 3, 5]
    block = proxy_module.weighted_correlation_columns(Z, w, selected)
    np.testing.assert_allclose(block, full[:, selected], atol=1e-15, rtol=0)
    stored = np.asarray(block, dtype=np.float32)
    expected = np.asarray(full[:, selected], dtype=np.float32)
    np.testing.assert_array_equal(stored, expected)


def _proxy_xy(seed: int = 0):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=120)
    X = pd.DataFrame(
        {
            "signal": signal,
            "proxy": signal + 0.01 * rng.normal(size=len(signal)),
            "noise": rng.normal(size=len(signal)),
        }
    )
    y = signal + 0.05 * rng.normal(size=len(signal))
    return X, y
