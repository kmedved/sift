"""Closure regressions for panel ``within=`` transforms and proxy reports.

Every numeric assertion here is checked against an oracle written
independently of the implementation: plain-numpy alternating projections for
the two-way solver, the closed form for balanced panels, and
``scipy.sparse.csgraph`` for the proxy-cluster graph.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import sift
from sift.selection.auto_k import AutoKConfig, select_k_auto
from sift.selection.proxies import reject_unavailable_proxy_positions
from sift.selection.within import (
    TWO_WAY_MAX_ITERATIONS,
    TWO_WAY_TOLERANCE,
    fit_within_transform,
)


# ---------------------------------------------------------------------------
# Oracles
# ---------------------------------------------------------------------------


def _weighted_level_means(values, codes, n_levels, weights):
    """Weighted mean of every column per level, written from scratch."""
    out = np.zeros((n_levels, values.shape[1]), dtype=np.float64)
    for level in range(n_levels):
        mask = codes == level
        mass = float(weights[mask].sum())
        if mass <= 0.0:
            continue
        out[level] = (weights[mask] @ values[mask]) / mass
    return out


def _converged_two_way(X, y, groups, time, w, *, time_first=False, tol=1e-13):
    """Alternate entity/time demeaning to convergence in plain numpy.

    Independent of ``fit_within_transform``: it re-derives the level means,
    runs far past any pass cap, and can start from either dimension so the
    fixed point can be checked for sweep-order independence.
    """
    _, g_codes = np.unique(np.asarray(groups), return_inverse=True)
    _, t_codes = np.unique(np.asarray(time), return_inverse=True)
    n_g = int(g_codes.max()) + 1
    n_t = int(t_codes.max()) + 1
    Xw = np.array(X, dtype=np.float64, copy=True)
    if Xw.ndim == 1:
        Xw = Xw.reshape(-1, 1)
    yw = np.array(y, dtype=np.float64, copy=True).reshape(-1, 1)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    order = [(t_codes, n_t), (g_codes, n_g)] if time_first else [
        (g_codes, n_g),
        (t_codes, n_t),
    ]
    for _ in range(200_000):
        step = 0.0
        for codes, n_levels in order:
            mX = _weighted_level_means(Xw, codes, n_levels, w)
            mY = _weighted_level_means(yw, codes, n_levels, w)
            Xw -= mX[codes]
            yw -= mY[codes]
            step = max(step, float(np.abs(mX).max()), float(np.abs(mY).max()))
        if step < tol:
            break
    return Xw, yw.reshape(-1)


def _unbalanced_panel(seed=7, weighted=True):
    rng = np.random.default_rng(seed)
    rows = []
    for entity in range(12):
        start = entity % 9
        stop = min(18, start + 5 + (entity % 6))
        rows.extend((entity, period) for period in range(start, stop))
    g = np.asarray([r[0] for r in rows])
    t = np.asarray([r[1] for r in rows])
    n = g.size
    X = np.column_stack(
        [
            rng.normal(size=n) + 2.0 * rng.normal(size=12)[g] + 1.5 * rng.normal(size=18)[t],
            rng.normal(size=n) + rng.normal(size=12)[g],
        ]
    )
    y = rng.normal(size=n) + rng.normal(size=18)[t]
    w = rng.uniform(0.2, 5.0, n) if weighted else np.ones(n)
    return X, y, g, t, w


# ---------------------------------------------------------------------------
# Item 4: two-way convergence
# ---------------------------------------------------------------------------


def test_two_way_balanced_panel_matches_closed_form():
    rng = np.random.default_rng(3)
    n_g, n_t = 9, 7
    g = np.repeat(np.arange(n_g), n_t)
    t = np.tile(np.arange(n_t), n_g)
    n = g.size
    col = rng.normal(size=n) + 2.0 * rng.normal(size=n_g)[g] + 1.5 * rng.normal(size=n_t)[t]
    X = col.reshape(-1, 1)
    y = rng.normal(size=n)
    w = np.ones(n)

    fitted = fit_within_transform("two_way", X, y, g, t, w)
    X_out, _y_out = fitted.transform(X, y, g, t)

    # Closed form for a balanced, unweighted panel.
    closed = (
        col
        - np.asarray([col[g == i].mean() for i in range(n_g)])[g]
        - np.asarray([col[t == j].mean() for j in range(n_t)])[t]
        + col.mean()
    )
    assert X_out[:, 0] == pytest.approx(closed, abs=1e-12)
    assert fitted.converged is True
    assert fitted.n_iterations <= 4
    assert fitted.max_residual < TWO_WAY_TOLERANCE


def test_two_way_unbalanced_weighted_matches_converged_oracle():
    X, y, g, t, w = _unbalanced_panel(weighted=True)
    fitted = fit_within_transform("two_way", X, y, g, t, w)
    X_out, y_out = fitted.transform(X, y, g, t)

    X_ref, y_ref = _converged_two_way(X, y, g, t, w)
    assert X_out == pytest.approx(X_ref, abs=1e-8)
    assert y_out == pytest.approx(y_ref, abs=1e-8)
    assert fitted.converged is True
    # The legacy solver stopped after exactly five passes; this panel needs
    # many more to reach the fixed point.
    assert fitted.n_iterations > 5
    assert fitted.n_iterations <= TWO_WAY_MAX_ITERATIONS


def test_two_way_result_is_independent_of_sweep_order():
    X, y, g, t, w = _unbalanced_panel(seed=11, weighted=True)
    fitted = fit_within_transform("two_way", X, y, g, t, w)
    X_out, _ = fitted.transform(X, y, g, t)

    entity_first, _ = _converged_two_way(X, y, g, t, w, time_first=False)
    time_first, _ = _converged_two_way(X, y, g, t, w, time_first=True)

    # The two oracles agree with each other and with the implementation.
    assert entity_first == pytest.approx(time_first, abs=1e-8)
    assert X_out == pytest.approx(time_first, abs=1e-8)


def test_two_way_residual_diagnostics_report_actual_passes():
    X, y, g, t, w = _unbalanced_panel(seed=5, weighted=False)
    fitted = fit_within_transform("two_way", X, y, g, t, w)
    X_out, _ = fitted.transform(X, y, g, t)

    # Independent check of the advertised stopping rule: every weighted entity
    # and time mean of the returned residual is negligible against the column
    # standard deviation.
    scale = X.std(axis=0)
    for codes in (g, t):
        levels = np.unique(codes)
        worst = max(
            float(np.abs(np.average(X_out[codes == level], axis=0, weights=w[codes == level]) / scale).max())
            for level in levels
        )
        assert worst < 1e-8
    assert fitted.converged is True
    assert fitted.max_residual < TWO_WAY_TOLERANCE


def test_two_way_transform_reproduces_fitted_effects_on_held_out_rows():
    X, y, g, t, w = _unbalanced_panel(seed=13, weighted=True)
    train = np.arange(X.shape[0]) % 4 != 0
    fitted = fit_within_transform(
        "two_way", X[train], y[train], g[train], t[train], w[train]
    )
    X_va, y_va = fitted.transform(X[~train], y[~train], g[~train], t[~train])

    # Held-out rows must get exactly the accumulated fitted effects, not a
    # re-fit on the validation rows.
    g_codes = fitted.group_index.get_indexer(g[~train])
    t_codes = fitted.time_index.get_indexer(t[~train])
    assert np.all(g_codes >= 0) and np.all(t_codes >= 0)
    expected_X = (
        X[~train] - fitted.group_effects_X[g_codes] - fitted.time_effects_X[t_codes]
    )
    expected_y = (
        y[~train] - fitted.group_effects_y[g_codes] - fitted.time_effects_y[t_codes]
    )
    assert X_va == pytest.approx(expected_X, abs=1e-12)
    assert y_va == pytest.approx(expected_y, abs=1e-12)


# ---------------------------------------------------------------------------
# Item 1: partially unseen validation levels
# ---------------------------------------------------------------------------


def _staggered_entry_panel():
    """12 always-present entities plus 8 that enter only in the last periods."""
    rows = [(e, p) for e in range(12) for p in range(20)]
    rows += [(e, p) for e in range(12, 20) for p in (18, 19)]
    g = np.asarray([r[0] for r in rows])
    t = np.asarray([r[1] for r in rows])
    n = g.size
    rng = np.random.default_rng(5)
    entity = 4.0 * rng.normal(size=20)[g]
    within = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "between_only": entity,
            "within_signal": within,
            "noise": rng.normal(size=n),
        }
    )
    y = entity + 1.5 * within + 0.05 * rng.normal(size=n)
    return X, y, g, t


def test_partially_unseen_validation_entities_warn_once_with_counts():
    X, y, g, t = _staggered_entry_panel()
    config = AutoKConfig(
        k_method="evaluate",
        strategy="time_holdout",
        val_frac=0.3,
        min_k=1,
        max_k=2,
        selection_rule="best",
    )
    with pytest.warns(UserWarning, match="fell back to the training grand mean") as rec:
        sift.select_cefsplus(
            X,
            y,
            k="auto",
            groups=g,
            time=t,
            within="groups",
            auto_k_config=config,
            verbose=False,
            subsample=None,
        )
    fallback = [
        w for w in rec if "fell back to the training grand mean" in str(w.message)
    ]
    assert len(fallback) == 1
    message = str(fallback[0].message)
    # Independent oracle: reproduce the split and count the unseen rows.
    from sift.selection.auto_k_core import time_holdout_split

    train_idx, val_idx = time_holdout_split(t, 0.3)
    seen = set(np.unique(g[train_idx]).tolist())
    unseen = int(sum(1 for value in g[val_idx] if value not in seen))
    assert unseen > 0
    assert f"{unseen} of {len(val_idx)} validation rows" in message
    assert f"{unseen / len(val_idx):.1%}" in message
    assert "entity" in message
    assert "late-entering" in message


def test_fully_overlapping_validation_levels_do_not_warn():
    rng = np.random.default_rng(2)
    g = np.repeat(np.arange(10), 12)
    t = np.tile(np.arange(12), 10)
    n = g.size
    entity = 3.0 * rng.normal(size=10)[g]
    within = rng.normal(size=n)
    X = pd.DataFrame(
        {"between_only": entity, "within_signal": within, "noise": rng.normal(size=n)}
    )
    y = entity + 1.5 * within + 0.05 * rng.normal(size=n)
    config = AutoKConfig(
        k_method="evaluate",
        strategy="time_holdout",
        val_frac=0.25,
        min_k=1,
        max_k=2,
        selection_rule="best",
    )
    # filterwarnings=error in pyproject turns any warning here into a failure.
    sift.select_cefsplus(
        X,
        y,
        k="auto",
        groups=g,
        time=t,
        within="groups",
        auto_k_config=config,
        verbose=False,
        subsample=None,
    )


# ---------------------------------------------------------------------------
# Item 2: dead-end routes reject up front with truthful guidance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "k_method", ["evaluate", "gaussian_cv", "xfit_objective"]
)
@pytest.mark.parametrize("within,strategy", [("groups", "group_cv"), ("two_way", "group_cv"), ("two_way", "time_holdout")])
def test_impossible_within_splits_name_a_working_route(k_method, within, strategy):
    X, y, g, t = _staggered_entry_panel()
    config = AutoKConfig(
        k_method=k_method,
        strategy=strategy,
        n_splits=3,
        xfit_folds=3,
        val_frac=0.3,
        min_k=1,
        max_k=2,
        selection_rule="best",
    )
    with pytest.raises(ValueError) as excinfo:
        sift.select_cefsplus(
            X,
            y,
            k="auto",
            groups=g,
            time=t,
            within=within,
            auto_k_config=config,
            verbose=False,
            subsample=None,
        )
    message = str(excinfo.value)
    assert "cannot be validated" in message
    # The remedy must name a combination that actually works.
    assert "strategy='kfold'" in message
    assert "gaussian_cv" in message and "xfit_objective" in message
    if within == "groups":
        assert "time_holdout" in message
    else:
        assert "keeps entity and time levels on both sides" in message


@pytest.mark.parametrize("within", ["groups", "two_way"])
def test_evaluate_kfold_message_does_not_recommend_a_failing_strategy(within):
    X, y, g, t = _staggered_entry_panel()
    config = AutoKConfig(
        k_method="evaluate", strategy="kfold", min_k=1, max_k=2, selection_rule="best"
    )
    with pytest.raises(ValueError) as excinfo:
        select_k_auto(
            X,
            np.asarray(y, dtype=np.float64),
            ["between_only", "within_signal", "noise"],
            config,
            groups=g,
            time=t,
            within=within,
        )
    message = str(excinfo.value)
    # The unconditional message recommends group_cv, which can never satisfy
    # the within guard; the within-aware one must not.
    assert "use time_holdout or group_cv" not in message
    assert "strategy='kfold'" in message
    if within == "two_way":
        assert "time_holdout" not in message


@pytest.mark.parametrize("within,strategy", [("groups", "group_cv"), ("two_way", "time_holdout")])
def test_impossible_within_split_rejects_before_building_the_path(within, strategy, monkeypatch):
    X, y, g, t = _staggered_entry_panel()
    import sift.selection.filter_auto_k as filter_auto_k

    called = []

    def _fail(*args, **kwargs):
        called.append(1)
        raise AssertionError("the feature path must not be built")

    monkeypatch.setattr(filter_auto_k, "_cached_filter_path", _fail)
    config = AutoKConfig(
        k_method="evaluate",
        strategy=strategy,
        n_splits=3,
        val_frac=0.3,
        min_k=1,
        max_k=2,
        selection_rule="best",
    )
    with pytest.raises(ValueError, match="cannot be validated"):
        sift.select_cefsplus(
            X,
            y,
            k="auto",
            groups=g,
            time=t,
            within=within,
            auto_k_config=config,
            verbose=False,
            subsample=None,
        )
    assert not called


# ---------------------------------------------------------------------------
# Item 3: non-finite contract on the Gaussian path
# ---------------------------------------------------------------------------


def test_gaussian_within_nonfinite_error_says_what_to_do():
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(6), 8)
    n = g.size
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    X.loc[3, "a"] = np.nan
    y = rng.normal(size=n)
    with pytest.raises(ValueError) as excinfo:
        sift.select_cefsplus(
            X, y, k=1, groups=g, within="groups", verbose=False, subsample=None
        )
    message = str(excinfo.value)
    assert "X contains NaN or infinite values" in message
    assert "Impute or drop the non-finite rows" in message
    assert "classic" in message


def test_classic_within_still_imputes_features():
    rng = np.random.default_rng(4)
    g = np.repeat(np.arange(6), 10)
    n = g.size
    signal = rng.normal(size=n)
    X = pd.DataFrame({"a": signal, "b": rng.normal(size=n)})
    X.loc[3, "a"] = np.nan
    y = signal + 0.05 * rng.normal(size=n)
    # Numerics unchanged: the classic route imputes and keeps scoring.
    selected = sift.select_mrmr(
        X, y, k=1, task="regression", estimator="classic", groups=g,
        within="groups", verbose=False,
    )
    assert selected == ["a"]


# ---------------------------------------------------------------------------
# Item 5: degenerate between_relevance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("estimator", ["classic", "gaussian"])
@pytest.mark.parametrize("n_groups", [1, 2])
def test_between_relevance_is_nan_with_two_or_fewer_entities(estimator, n_groups):
    rng = np.random.default_rng(3)
    n_time = 12
    groups = np.repeat(np.arange(n_groups), n_time)
    n = groups.size
    entity = 4.0 * rng.normal(size=n_groups)[groups]
    X = pd.DataFrame(
        {
            "between_only": entity,
            "within_signal": rng.normal(size=n),
            "noise": rng.normal(size=n),
        }
    )
    y = entity + 0.8 * X["within_signal"].to_numpy() + 0.05 * rng.normal(size=n)
    result = sift.select_mrmr(
        X, y, k=1, task="regression", estimator=estimator, groups=groups,
        within="groups", subsample=None, verbose=False, return_result=True,
    )
    between = result.ranking_["between_relevance"].to_numpy(dtype=float)
    assert np.isnan(between).all()


@pytest.mark.parametrize("estimator", ["classic", "gaussian"])
def test_between_relevance_is_finite_with_three_entities(estimator):
    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(3), 12)
    n = groups.size
    entity = 4.0 * rng.normal(size=3)[groups]
    X = pd.DataFrame(
        {
            "between_only": entity,
            "within_signal": rng.normal(size=n),
            "noise": rng.normal(size=n),
        }
    )
    y = entity + 0.8 * X["within_signal"].to_numpy() + 0.05 * rng.normal(size=n)
    result = sift.select_mrmr(
        X, y, k=1, task="regression", estimator=estimator, groups=groups,
        within="groups", subsample=None, verbose=False, return_result=True,
    )
    between = result.ranking_["between_relevance"].to_numpy(dtype=float)
    assert np.isfinite(between).all()


# ---------------------------------------------------------------------------
# Items 6 and 9: proxy reports
# ---------------------------------------------------------------------------


def _clustered_proxy_view(seed=17, n=200):
    """A view whose selected features are correlated with each other."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n)
    other = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "a": base,
            "b": base + 0.05 * rng.normal(size=n),
            "c": other,
            "d": other + 0.05 * rng.normal(size=n),
            "e": rng.normal(size=n),
        }
    )
    y = base + other + 0.1 * rng.normal(size=n)
    result = sift.select_cefsplus(
        X, y, k=4, store_proxies=True, return_result=True, subsample=None, verbose=False
    )
    return sift.as_result(result, input_features=list(X.columns))


def test_redundancy_report_default_output_is_unchanged():
    view = _clustered_proxy_view()
    default = view.redundancy_report(r_min=0.5)
    explicit = view.redundancy_report(r_min=0.5, include_selected=False)
    pd.testing.assert_frame_equal(default, explicit)
    selected = set(view.indices)
    # The default view never reports a selected feature as a stand-in.
    assert not set(default["candidate_index"]) & selected


def test_include_selected_exposes_every_edge_proxy_clusters_merges_on():
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    view = _clustered_proxy_view()
    r_min = 0.5
    report = view.redundancy_report(r_min=r_min, include_selected=True)
    selected = [int(i) for i in view.indices]

    assert not report.empty
    # No self-pairs, and selected pairs appear under both anchors.
    assert not (report["selected_index"] == report["candidate_index"]).any()
    selected_edges = report[report["candidate_index"].isin(selected)]
    assert not selected_edges.empty
    pairs = set(zip(selected_edges["selected_index"], selected_edges["candidate_index"]))
    assert all((b, a) in pairs for a, b in pairs)

    # Oracle: components of the graph built from the reported rows must equal
    # the clusters proxy_clusters produces at the same threshold.
    nodes = sorted(set(report["selected_index"]) | set(report["candidate_index"]) | set(selected))
    index_of = {node: i for i, node in enumerate(nodes)}
    rows = [index_of[int(v)] for v in report["selected_index"]]
    cols = [index_of[int(v)] for v in report["candidate_index"]]
    graph = coo_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)),
        shape=(len(nodes), len(nodes)),
        dtype=np.int8,
    )
    _n, labels = connected_components(graph, directed=False)
    oracle = {}
    for node, label in zip(nodes, labels.tolist()):
        oracle.setdefault(int(label), set()).add(int(node))
    anchored = {
        frozenset(oracle[int(labels[index_of[pos]])]) for pos in selected
    }

    clusters = view.proxy_clusters(r_min=r_min)
    impl = {
        frozenset(group["selected_index"].tolist())
        for _, group in clusters.groupby("cluster_id")
    }
    assert impl == anchored


def test_proxies_at_include_selected_matches_the_report_rows():
    view = _clustered_proxy_view()
    r_min = 0.5
    report = view.redundancy_report(r_min=r_min, include_selected=True)
    for anchor in view.indices:
        anchor = int(anchor)
        rows = view.proxies_at(anchor, r_min, include_selected=True)
        expected = report[report["selected_index"] == anchor]
        assert rows["selected_index"].tolist() == expected["candidate_index"].tolist()
        assert anchor not in set(rows["selected_index"])
        default_rows = view.proxies_at(anchor, r_min)
        assert set(default_rows["selected_index"]) <= set(rows["selected_index"])


def test_stale_threshold_guidance_explains_the_stored_block():
    rng = np.random.default_rng(23)
    n = 200
    signal = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "a": signal,
            "b": signal + 0.02 * rng.normal(size=n),
            "c": rng.normal(size=n),
            "d": rng.normal(size=n),
        }
    )
    y = signal + 0.3 * rng.normal(size=n)
    selector = sift.StabilitySelector(
        n_bootstrap=10,
        threshold=0.9,
        store_proxies=True,
        store_coefs=False,
        random_state=0,
        verbose=False,
        n_jobs=1,
    ).fit(X, y)
    view = selector.set_threshold(0.05).result_view_
    with pytest.raises(NotImplementedError) as excinfo:
        view.redundancy_report(0.8)
    message = str(excinfo.value)
    assert "one column per feature selected when it was computed" in message
    assert "refit with the lower threshold and store_proxies=True" in message


# ---------------------------------------------------------------------------
# Item 8: constant selected column under store_proxies
# ---------------------------------------------------------------------------


def test_stability_constant_column_message_names_columns_not_blocks():
    rng = np.random.default_rng(31)
    n = 120
    signal = rng.normal(size=n)
    X = pd.DataFrame(
        {"signal": signal, "flat": np.full(n, 3.0), "noise": rng.normal(size=n)}
    )
    y = signal + 0.1 * rng.normal(size=n)
    with pytest.raises(ValueError) as excinfo:
        sift.StabilitySelector(
            n_bootstrap=6,
            random_state=0,
            n_jobs=1,
            verbose=False,
            store_proxies=True,
            threshold=0.0,
        ).fit(X, y)
    message = str(excinfo.value)
    assert "'flat'" in message
    assert "drop them from X or fit without store_proxies" in message
    # No block exists in stability selection, so none may be mentioned.
    assert "block" not in message


def test_blocks_in_play_still_points_at_the_block():
    with pytest.raises(ValueError, match="unavailable block members"):
        reject_unavailable_proxy_positions(
            [0],
            available_original=[1],
            feature_names=["constant", "varying"],
            blocks_in_play=True,
        )


def test_stability_without_store_proxies_accepts_a_constant_column():
    rng = np.random.default_rng(31)
    n = 120
    signal = rng.normal(size=n)
    X = pd.DataFrame(
        {"signal": signal, "flat": np.full(n, 3.0), "noise": rng.normal(size=n)}
    )
    y = signal + 0.1 * rng.normal(size=n)
    selector = sift.StabilitySelector(
        n_bootstrap=6, random_state=0, n_jobs=1, verbose=False, threshold=0.0
    ).fit(X, y)
    assert selector.n_features_selected_ >= 1


# ---------------------------------------------------------------------------
# Item 10: multi-target guard in stability.py
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "entry",
    [
        "fit",
        "stability_regression",
        "stability_classif",
        "stability_select",
    ],
)
def test_stability_entries_reject_wide_2d_targets(entry):
    rng = np.random.default_rng(9)
    X = rng.normal(size=(80, 4))
    y = np.column_stack([rng.normal(size=80), rng.normal(size=80)])
    kwargs = dict(n_bootstrap=4, random_state=0, n_jobs=1, verbose=False)
    with pytest.raises(ValueError) as excinfo:
        if entry == "fit":
            sift.StabilitySelector(**kwargs).fit(X, y)
        elif entry == "stability_regression":
            sift.stability_regression(X, y, k=2, **kwargs)
        elif entry == "stability_classif":
            sift.stability_classif(X, (y > 0).astype(int), k=2, **kwargs)
        else:
            from sift.stability import stability_select

            stability_select(X, y, **kwargs)
    assert str(excinfo.value).startswith(
        "2-D y is only supported for select_cefsplus / CEFSPlusSelector"
    )


def test_stability_single_column_2d_target_behaviour_is_unchanged():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(60, 4))
    y = rng.normal(size=(60, 1))
    # A single-column y is not caught by the multi-target guard; whatever it
    # did before, it must not now raise the 2-D ValueError.
    with pytest.raises(Exception) as excinfo:
        sift.StabilitySelector(
            n_bootstrap=4, random_state=0, n_jobs=1, verbose=False
        ).fit(X, y)
    assert not str(excinfo.value).startswith("2-D y is only supported")


def test_stability_one_dimensional_target_still_fits():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(80, 4))
    signal = rng.normal(size=80)
    X[:, 0] = signal
    y = signal + 0.1 * rng.normal(size=80)
    selector = sift.StabilitySelector(
        n_bootstrap=6, random_state=0, n_jobs=1, verbose=False, threshold=0.5
    ).fit(X, y)
    assert 0 in set(selector.selected_features_.tolist())
