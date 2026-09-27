"""Closure regressions for panel ``within=`` transforms and proxy reports.

Every numeric assertion here is checked against an oracle written
independently of the implementation: plain-numpy alternating projections for
the two-way solver, the closed form for balanced panels, and
``scipy.sparse.csgraph`` for the proxy-cluster graph.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin

import sift
from sift.selection.auto_k import AutoKConfig, select_k_auto
from sift.selection.proxies import reject_unavailable_proxy_positions
from sift.selection.within import (
    TWO_WAY_MAX_ITERATIONS,
    TWO_WAY_TOLERANCE,
    UnseenWithinLevelTally,
    fit_within_transform,
    warn_unseen_within_validation_levels,
    within_split_guidance,
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
    # The remedy must not send the user back to the holdout that warned.
    assert "strategy='time_holdout'" not in message
    assert message.endswith(
        "Drop or otherwise handle late-entering or early-exiting entities, or "
        "switch to k_method='gaussian_cv' or 'xfit_objective' with "
        "strategy='kfold' on the Gaussian path (select_cefsplus / "
        "CEFSPlusSelector, or estimator='gaussian' for mRMR, JMI and JMIM), "
        "which holds out rows instead of whole periods"
    )


def test_unseen_time_warning_preserves_known_entity_effect():
    tally = UnseenWithinLevelTally()
    tally.add(mode="two_way", n_rows=4, entity_unseen=0, time_unseen=2)
    with pytest.warns(UserWarning) as caught:
        warn_unseen_within_validation_levels(tally)
    message = str(caught[0].message)
    assert "2 of 4 validation rows (50.0%) had an unseen time level" in message
    assert "time effect was omitted" in message
    assert "other dimension still applies" in message
    assert "grand mean" not in message


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
# 1.0.1: within guidance is truthful on every public auto-k route
# ---------------------------------------------------------------------------


def _balanced_panel():
    """10 entities x 12 periods: every level has many rows, so nothing warns."""
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
    return X, y, g, t


#: Public auto-k routes that accept ``within``.  A ``+gaussian`` route passes
#: ``estimator="gaussian"``, the only mRMR/JMI/JMIM estimator that offers
#: ``gaussian_cv`` and ``xfit_objective``; the others score ``evaluate`` only.
_CLASSIC_WITHIN_ROUTES = [
    "select_mrmr",
    "select_jmi",
    "select_jmim",
    "MRMRSelector",
    "JMISelector",
    "JMIMSelector",
    "select_k_auto",
]
_GAUSSIAN_WITHIN_ROUTES = [
    "select_mrmr+gaussian",
    "select_jmi+gaussian",
    "select_jmim+gaussian",
    "select_cefsplus",
    "MRMRSelector+gaussian",
    "JMISelector+gaussian",
    "JMIMSelector+gaussian",
    "CEFSPlusSelector",
]

#: The (k_method, strategy) pairs that can validate each within mode.
_WORKING_WITHIN_ROUTES = {
    "two_way": {("gaussian_cv", "kfold"), ("xfit_objective", "kfold")},
    "groups": {
        ("gaussian_cv", "kfold"),
        ("xfit_objective", "kfold"),
        ("evaluate", "time_holdout"),
    },
}


def _run_within_route(route, X, y, g, t, *, within, config):
    name, _, variant = route.partition("+")
    if name == "select_k_auto":
        # select_k_auto scores prefixes of a given path; hand it the path the
        # selectors learn on this panel.
        _best_k, selected, _diag = select_k_auto(
            X,
            np.asarray(y, dtype=np.float64),
            ["within_signal", "noise", "between_only"],
            config,
            groups=g,
            time=t,
            within=within,
        )
        return list(selected)
    options = {"verbose": False}
    if name not in {"select_cefsplus", "CEFSPlusSelector"}:
        options["task"] = "regression"
    if variant == "gaussian":
        options["estimator"] = "gaussian"
    if name.startswith("select_"):
        return list(
            getattr(sift, name)(
                X, y, k="auto", groups=g, time=t, within=within,
                auto_k_config=config, **options,
            )
        )
    selector = getattr(sift, name)(
        k="auto", within=within, auto_k_config=config, **options
    )
    return list(selector.fit(X, y, groups=g, time=t).selected_features_)


def _dead_end_prefix(within, k_method, strategy):
    if strategy == "kfold":
        return (
            "AutoKConfig.strategy='kfold' is only supported by gaussian_cv and "
            f"xfit_objective, and with within={within!r} the remaining evaluate "
            "strategies cannot all be validated. "
        )
    return (
        f"within={within!r} cannot be validated with k_method={k_method!r} and "
        f"strategy={strategy!r}: "
    )


@pytest.mark.parametrize("within", ["groups", "two_way"])
def test_within_guidance_names_exactly_the_routes_that_work(within):
    guidance = within_split_guidance(within)
    quoted = set(re.findall(r"'([a-z_]+)'", guidance))
    working = _WORKING_WITHIN_ROUTES[within]
    assert quoted & {"evaluate", "gaussian_cv", "xfit_objective"} == {
        method for method, _ in working
    }
    assert quoted & {"kfold", "time_holdout", "group_cv"} == {
        strategy for _, strategy in working
    }
    # gaussian_cv / xfit_objective exist only on the Gaussian path, so the
    # guidance must say how mRMR/JMI/JMIM users get there.
    assert (
        "the Gaussian path (select_cefsplus / CEFSPlusSelector, or "
        "estimator='gaussian' for mRMR, JMI and JMIM)"
    ) in guidance
    if within == "two_way":
        assert guidance.endswith(
            "so select_k_auto and the classic estimators cannot validate "
            "within='two_way'"
        )


@pytest.mark.parametrize("route", _CLASSIC_WITHIN_ROUTES + _GAUSSIAN_WITHIN_ROUTES)
@pytest.mark.parametrize(
    "within,strategy",
    [
        ("groups", "kfold"),
        ("two_way", "kfold"),
        ("groups", "group_cv"),
        ("two_way", "group_cv"),
        ("two_way", "time_holdout"),
    ],
)
def test_evaluate_dead_ends_name_a_working_route_on_every_public_route(
    route, within, strategy
):
    X, y, g, t = _balanced_panel()
    config = AutoKConfig(k_method="evaluate", strategy=strategy, min_k=1, max_k=2)
    with pytest.raises(ValueError) as excinfo:
        _run_within_route(route, X, y, g, t, within=within, config=config)
    message = str(excinfo.value)
    assert message.startswith(_dead_end_prefix(within, "evaluate", strategy))
    assert message.endswith(within_split_guidance(within))
    assert "use time_holdout or group_cv" not in message


@pytest.mark.parametrize("route", _GAUSSIAN_WITHIN_ROUTES)
@pytest.mark.parametrize("k_method", ["gaussian_cv", "xfit_objective"])
@pytest.mark.parametrize(
    "within,strategy",
    [("groups", "group_cv"), ("two_way", "group_cv"), ("two_way", "time_holdout")],
)
def test_fold_method_dead_ends_name_a_working_route_on_every_public_route(
    route, k_method, within, strategy
):
    X, y, g, t = _balanced_panel()
    config = AutoKConfig(
        k_method=k_method, strategy=strategy, min_k=1, max_k=2, xfit_folds=3
    )
    with pytest.raises(ValueError) as excinfo:
        _run_within_route(route, X, y, g, t, within=within, config=config)
    message = str(excinfo.value)
    assert message.startswith(_dead_end_prefix(within, k_method, strategy))
    assert message.endswith(within_split_guidance(within))


def _recommended_route_cases():
    cases = []
    for within, working in sorted(_WORKING_WITHIN_ROUTES.items()):
        for k_method, strategy in sorted(working):
            routes = list(_GAUSSIAN_WITHIN_ROUTES)
            if k_method == "evaluate":
                routes = _CLASSIC_WITHIN_ROUTES + routes
            cases.extend((within, k_method, strategy, route) for route in routes)
    return cases


@pytest.mark.parametrize("within,k_method,strategy,route", _recommended_route_cases())
def test_following_the_within_guidance_succeeds_on_every_public_route(
    within, k_method, strategy, route
):
    X, y, g, t = _balanced_panel()
    options = {"xfit_folds": 3} if strategy == "kfold" else {}
    config = AutoKConfig(
        k_method=k_method, strategy=strategy, min_k=1, max_k=2, **options
    )
    # filterwarnings=error: the recommended route must not warn on a panel
    # whose levels all have many rows.
    selected = _run_within_route(route, X, y, g, t, within=within, config=config)
    assert selected == ["within_signal"]


@pytest.mark.parametrize("route", _GAUSSIAN_WITHIN_ROUTES)
@pytest.mark.parametrize("k_method", ["auto", "elbow"])
@pytest.mark.parametrize("within", ["groups", "two_way"])
def test_non_fold_auto_k_methods_with_within_name_a_working_route(
    route, k_method, within
):
    X, y, g, t = _balanced_panel()
    if k_method == "auto" and route not in {"select_cefsplus", "CEFSPlusSelector"}:
        # mRMR/JMI/JMIM have no zero-config router; request it explicitly.
        config = AutoKConfig(k_method="auto")
    elif k_method == "auto":
        config = None  # CEFS+ routes k="auto" without a config to the router
    else:
        config = AutoKConfig(k_method="elbow")
    with pytest.raises(ValueError) as excinfo:
        _run_within_route(route, X, y, g, t, within=within, config=config)
    router = " (the zero-config k='auto' router)" if k_method == "auto" else ""
    assert str(excinfo.value) == (
        f"within={within!r} cannot score auto-k k_method={k_method!r}{router}; "
        "choose an auto_k_config it can validate. "
        f"{within_split_guidance(within)}"
    )


@pytest.mark.parametrize(
    "route", [r for r in _CLASSIC_WITHIN_ROUTES if r != "select_k_auto"]
)
def test_zero_config_two_way_on_classic_routes_names_a_working_route(route):
    X, y, g, t = _balanced_panel()
    # No config: time is present, so k="auto" infers evaluate/time_holdout.
    with pytest.raises(ValueError) as excinfo:
        _run_within_route(route, X, y, g, t, within="two_way", config=None)
    message = str(excinfo.value)
    assert message.startswith(_dead_end_prefix("two_way", "evaluate", "time_holdout"))
    assert message.endswith(within_split_guidance("two_way"))


def test_nested_auto_k_with_within_names_a_working_route():
    X, y, g, t = _balanced_panel()
    config = AutoKConfig(auto_k_mode="nested", k_method="evaluate", strategy="time_holdout")
    selector = sift.CEFSPlusSelector(
        k="auto", within="two_way", auto_k_config=config, verbose=False
    )
    with pytest.raises(ValueError) as excinfo:
        selector.fit(X, y, groups=g, time=t)
    assert str(excinfo.value) == (
        "within is not supported with auto_k_mode='nested'; use "
        "auto_k_mode='prefix_only' so demeaning stays fold-local. "
        f"{within_split_guidance('two_way')}"
    )


def _sparse_level_panel():
    """A balanced core plus one-row entities and one-row periods."""
    rows = [(e, p) for e in range(10) for p in range(12)]
    rows += [(10, 0), (11, 3), (12, 7)]
    rows += [(0, 12), (1, 13)]
    g = np.asarray([r[0] for r in rows])
    t = np.asarray([r[1] for r in rows])
    n = g.size
    rng = np.random.default_rng(3)
    entity = 3.0 * rng.normal(size=13)[g]
    within = rng.normal(size=n)
    X = pd.DataFrame(
        {"between_only": entity, "within_signal": within, "noise": rng.normal(size=n)}
    )
    y = entity + 1.5 * within + 0.05 * rng.normal(size=n)
    return X, y, g, t


@pytest.mark.parametrize("k_method", ["gaussian_cv", "xfit_objective"])
@pytest.mark.parametrize("within", ["groups", "two_way"])
def test_kfold_unseen_level_warning_does_not_recommend_its_own_route(k_method, within):
    from sklearn.model_selection import KFold

    X, y, g, t = _sparse_level_panel()
    config = AutoKConfig(
        k_method=k_method, strategy="kfold", xfit_folds=3, min_k=1, max_k=2
    )
    with pytest.warns(UserWarning) as rec:
        selected = sift.select_cefsplus(
            X, y, k="auto", groups=g, time=t, within=within,
            auto_k_config=config, verbose=False, subsample=None,
        )
    assert selected == ["within_signal"]
    unseen = [w for w in rec if "auto-k scoring:" in str(w.message)]
    assert len(unseen) == 1
    message = str(unseen[0].message)

    # Independent oracle: rebuild the folds and count validation rows whose
    # level never appears in that fold's training rows.
    n = g.size
    entity_unseen = time_unseen = 0
    folds = KFold(n_splits=3, shuffle=True, random_state=config.random_state)
    for train_idx, val_idx in folds.split(np.arange(n)):
        entity_unseen += int(np.isin(g[val_idx], g[train_idx], invert=True).sum())
        time_unseen += int(np.isin(t[val_idx], t[train_idx], invert=True).sum())
    assert entity_unseen > 0 and time_unseen > 0
    assert (
        f"{entity_unseen} of {n} validation rows ({entity_unseen / n:.1%}) had an "
        "unseen entity level"
    ) in message
    if within == "two_way":
        assert (
            f"{time_unseen} of {n} validation rows ({time_unseen / n:.1%}) had an "
            "unseen time level"
        ) in message
    # The route that produced the warning is the recommended one, so the
    # remedy must name the cause instead of recommending the route again.
    assert within_split_guidance(within) not in message
    assert "choose a split" not in message
    assert message.endswith(
        "Under strategy='kfold' a level is unseen only when all of its rows "
        "fall in the same validation fold, so the affected levels have very "
        "few rows (a single-row level is always unseen): drop or pool them, or "
        "raise AutoKConfig.xfit_folds so fewer of their rows are held out "
        "together"
    )


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


@pytest.mark.parametrize("estimator", ["classic", "gaussian"])
@pytest.mark.parametrize("n_groups", [1, 2, 3])
def test_view_table_keeps_panel_columns_whenever_the_ranking_has_them(
    estimator, n_groups
):
    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(n_groups), 12)
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
    table = sift.as_result(result).table
    # The column set must not depend on whether the entity-level score was
    # degenerate (NaN with two or fewer entities) on this particular data.
    assert list(table.columns) == [
        "feature",
        "selected_index",
        "path_rank",
        "selected",
        "relevance",
        "within_relevance",
        "between_relevance",
    ]
    ranking = result.ranking_.set_index("feature")
    for column in ("within_relevance", "between_relevance"):
        np.testing.assert_array_equal(
            table[column].to_numpy(dtype=float),
            ranking.loc[table["feature"], column].to_numpy(dtype=float),
        )
    between = table["between_relevance"].to_numpy(dtype=float)
    if n_groups <= 2:
        assert np.isnan(between).all()
    else:
        assert np.isfinite(between).all()


def test_view_table_without_within_has_no_panel_columns():
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(60, 3)), columns=["a", "b", "c"])
    y = X["a"].to_numpy() + 0.1 * rng.normal(size=60)
    result = sift.select_mrmr(
        X, y, k=1, task="regression", verbose=False, return_result=True
    )
    columns = list(sift.as_result(result).table.columns)
    assert "within_relevance" not in columns
    assert "between_relevance" not in columns


# ---------------------------------------------------------------------------
# 1.0.1: two-way convergence diagnostics in result metadata
# ---------------------------------------------------------------------------


_TWO_WAY_SELECTORS = [
    (sift.select_cefsplus, {}),
    (sift.select_mrmr, {"task": "regression", "estimator": "classic"}),
]


@pytest.mark.parametrize("selector,options", _TWO_WAY_SELECTORS)
def test_two_way_metadata_reports_convergence_of_the_path_fit(selector, options):
    X, y, g, t, w = _unbalanced_panel(seed=7, weighted=False)
    result = selector(
        X, y, k=1, within="two_way", groups=g, time=t,
        return_result=True, subsample=None, verbose=False, **options,
    )
    fitted = fit_within_transform("two_way", X, y, g, t, w)
    meta = result.selector_metadata
    assert meta["within_two_way_iterations"] == fitted.n_iterations
    assert meta["within_two_way_converged"] is True
    assert meta["within_two_way_max_residual"] == fitted.max_residual
    assert meta["within_two_way_max_residual"] < TWO_WAY_TOLERANCE
    view_meta = sift.as_result(result).metadata
    assert view_meta["within_two_way_converged"] is True
    assert view_meta["within_two_way_max_residual"] == fitted.max_residual


@pytest.mark.parametrize("selector,options", _TWO_WAY_SELECTORS)
def test_within_groups_metadata_has_no_two_way_diagnostics(selector, options):
    X, y, g, _t, _w = _unbalanced_panel(seed=7, weighted=False)
    meta = selector(
        X, y, k=1, within="groups", groups=g,
        return_result=True, subsample=None, verbose=False, **options,
    ).selector_metadata
    assert meta["within"] == "groups"
    assert not {
        "within_two_way_iterations",
        "within_two_way_converged",
        "within_two_way_max_residual",
    } & set(meta)


def _staircase_panel(n_entities=12, width=3):
    """Entity i is observed only in periods i .. i + width - 1.

    The entity/time graph is a long chain, which the alternating projection
    crosses one link per pass, so 200 passes cannot reach the tolerance.
    """
    g = np.repeat(np.arange(n_entities), width)
    t = (np.arange(n_entities)[:, None] + np.arange(width)[None, :]).reshape(-1)
    rng = np.random.default_rng(0)
    n = g.size
    signal = rng.normal(size=n)
    X = pd.DataFrame({"signal": signal, "noise": rng.normal(size=n)})
    y = signal + 0.1 * rng.normal(size=n)
    return X, y, g, t


def _two_way_least_squares_residual(values, groups, time):
    """Exact unweighted two-way residual: OLS on entity and time dummies."""
    entity_dummies = pd.get_dummies(groups).to_numpy(dtype=float)
    time_dummies = pd.get_dummies(time).to_numpy(dtype=float)
    design = np.column_stack([entity_dummies, time_dummies])
    coef, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coef


@pytest.mark.parametrize("selector,options", _TWO_WAY_SELECTORS)
def test_two_way_iteration_cap_warns_and_reports_non_convergence(selector, options):
    X, y, g, t = _staircase_panel()
    with pytest.warns(UserWarning) as rec:
        result = selector(
            X, y, k=1, within="two_way", groups=g, time=t,
            return_result=True, subsample=None, verbose=False, **options,
        )
    capped = [w for w in rec if "pass cap" in str(w.message)]
    assert len(capped) == 1
    meta = result.selector_metadata
    assert meta["within_two_way_iterations"] == TWO_WAY_MAX_ITERATIONS
    assert meta["within_two_way_converged"] is False
    residual = meta["within_two_way_max_residual"]
    assert residual > TWO_WAY_TOLERANCE
    assert str(capped[0].message) == (
        f"within='two_way' demeaning stopped at the {TWO_WAY_MAX_ITERATIONS}-pass "
        f"cap with a scaled level-mean residual of {residual:.3e}, above the "
        f"{TWO_WAY_TOLERANCE:.0e} tolerance; the entity and time effects are not "
        "fully separated. This usually means the panel splits into weakly "
        "connected entity/time components -- check for entities or periods that "
        "barely overlap the rest of the panel, or use within='groups'"
    )
    # Independent check that the cap really cut the projection short: the
    # capped transform is measurably away from the exact least-squares
    # two-way residual.
    with pytest.warns(UserWarning, match="pass cap"):
        fitted = fit_within_transform(
            "two_way", X.to_numpy(), y, g, t, np.ones(g.size)
        )
    X_capped, _ = fitted.transform(X.to_numpy(), y, g, t)
    exact = _two_way_least_squares_residual(X.to_numpy(), g, t)
    assert np.abs(X_capped - exact).max() > 1e-6
    assert fitted.max_residual == residual


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
    assert view.metadata["proxy_correlations_stale"] is True
    with pytest.raises(NotImplementedError) as excinfo:
        view.redundancy_report(0.8)
    assert str(excinfo.value) == _STALE_PROXY_MESSAGE


_NEVER_STORED_PROXY_MESSAGE = (
    "proxy correlations were not stored for this selection; rerun or refit "
    "selection with store_proxies=True"
)
_STALE_PROXY_MESSAGE = (
    "proxy correlations are unavailable for this selected set: the stored "
    "proxy block holds one column per feature selected when it was computed, "
    "and a threshold change added features it cannot describe; refit with the "
    "lower threshold and store_proxies=True"
)


def _proxy_message(call):
    with pytest.raises(NotImplementedError) as excinfo:
        call()
    return str(excinfo.value)


def test_never_stored_proxy_guidance_does_not_blame_a_threshold_change():
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
    function_view = sift.as_result(
        sift.select_cefsplus(X, y, k=2, verbose=False, return_result=True),
        input_features=list(X.columns),
    )
    selector = sift.StabilitySelector(
        n_bootstrap=10,
        threshold=0.9,
        store_coefs=False,
        random_state=0,
        verbose=False,
        n_jobs=1,
    ).fit(X, y)
    stability_view = selector.result_view_
    lowered_view = selector.set_threshold(0.05).result_view_
    assert function_view.metadata.get("proxy_correlations_stale") is None
    assert stability_view.metadata["proxy_correlations_stale"] is False
    assert lowered_view.metadata["proxy_correlations_stale"] is False
    # store_proxies was never set, so even after the threshold change the
    # message must point at store_proxies rather than at the threshold.
    for view in (function_view, stability_view, lowered_view):
        assert _proxy_message(view.redundancy_report) == _NEVER_STORED_PROXY_MESSAGE
        assert _proxy_message(view.proxy_clusters) == _NEVER_STORED_PROXY_MESSAGE
        assert (
            _proxy_message(lambda: view.proxies_at(view.indices[0]))
            == _NEVER_STORED_PROXY_MESSAGE
        )


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


class _SelectEverything(SelectorMixin, BaseEstimator):
    """Generic sklearn selector with no block concept: keeps every column."""

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def _get_support_mask(self):
        return np.ones(self.n_features_in_, dtype=bool)


def _stabilized_constant_frame():
    rng = np.random.default_rng(31)
    n = 120
    signal = rng.normal(size=n)
    X = pd.DataFrame(
        {"signal": signal, "flat": np.full(n, 3.0), "noise": rng.normal(size=n)}
    )
    y = signal + 0.1 * rng.normal(size=n)
    return X, y


def test_stabilized_constant_column_message_names_columns_without_blocks():
    X, y = _stabilized_constant_frame()
    with pytest.raises(ValueError) as excinfo:
        sift.Stabilized(
            _SelectEverything(),
            n_resamples=4,
            store_proxies=True,
            random_state=0,
            verbose=False,
        ).fit(X, y)
    assert str(excinfo.value) == (
        "store_proxies=True cannot retain finite copula correlations for "
        "selected constant features: ['flat'] (positions [1]). Those columns "
        "stay in the selection but have no finite copula correlation to "
        "store; drop them from X or fit without store_proxies"
    )


def test_stabilized_constant_block_member_message_points_at_the_block():
    X, y = _stabilized_constant_frame()
    base = sift.CEFSPlusSelector(
        k=1, feature_blocks={"sf": ["signal", "flat"]}, verbose=False
    )
    # The constant column reaches the selection only through the atomic block.
    fitted = sift.Stabilized(base, n_resamples=4, random_state=0, verbose=False).fit(
        X, y
    )
    assert fitted.selected_features_ == ["signal", "flat"]
    with pytest.raises(ValueError) as excinfo:
        sift.Stabilized(
            base, n_resamples=4, store_proxies=True, random_state=0, verbose=False
        ).fit(X, y)
    assert str(excinfo.value) == (
        "store_proxies=True cannot retain finite copula correlations for "
        "selected constant features or cache-dropped constant or otherwise "
        "unavailable block members: ['flat'] (positions [1]). Atomic selection "
        "still expands those raw columns; omit store_proxies or drop "
        "unavailable members from the block"
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


def _single_column_problem(task):
    rng = np.random.default_rng(9)
    X = pd.DataFrame(rng.normal(size=(80, 4)), columns=["a", "b", "c", "d"])
    signal = X["a"].to_numpy() + 0.5 * X["b"].to_numpy() + 0.3 * rng.normal(size=80)
    if task == "classification":
        y = np.where(signal > 0.0, "up", "down")
    else:
        y = signal
    return X, y


def _as_single_column(y, shape):
    if shape == "frame":
        return pd.DataFrame({"target": y})
    return np.asarray(y).reshape(-1, 1)


@pytest.mark.parametrize("use_smart_sampler", [False, True], ids=["plain", "smart"])
@pytest.mark.parametrize("shape", ["column", "frame"])
@pytest.mark.parametrize("task", ["regression", "classification"])
def test_stability_single_column_2d_target_fits_exactly_like_1d(
    task, shape, use_smart_sampler
):
    X, y = _single_column_problem(task)
    kwargs = dict(
        task=task,
        n_bootstrap=6,
        random_state=0,
        n_jobs=1,
        verbose=False,
        threshold=0.5,
        use_smart_sampler=use_smart_sampler,
        sampler_config=(
            sift.SmartSamplerConfig(sample_frac=0.8) if use_smart_sampler else None
        ),
    )
    flat = sift.StabilitySelector(**kwargs).fit(X, y)
    column = sift.StabilitySelector(**kwargs).fit(X, _as_single_column(y, shape))

    np.testing.assert_array_equal(
        column.selection_frequencies_, flat.selection_frequencies_
    )
    np.testing.assert_array_equal(column.mean_abs_coef_, flat.mean_abs_coef_)
    np.testing.assert_array_equal(column.selected_features_, flat.selected_features_)
    assert column.alpha_ == flat.alpha_
    assert column.selected_feature_names_ == flat.selected_feature_names_
    assert flat.selected_feature_names_[0] == "a"
    if task == "classification":
        np.testing.assert_array_equal(column.classes_, ["down", "up"])


@pytest.mark.parametrize("shape", ["column", "frame"])
@pytest.mark.parametrize(
    "entry", ["stability_regression", "stability_classif", "stability_select"]
)
def test_stability_functions_accept_a_single_column_target(entry, shape):
    task = "classification" if entry == "stability_classif" else "regression"
    X, y = _single_column_problem(task)
    kwargs = dict(n_bootstrap=6, random_state=0, n_jobs=1, verbose=False)
    if entry == "stability_select":
        from sift.stability import stability_select

        flat = stability_select(X, y, threshold=0.5, **kwargs)
        column = stability_select(
            X, _as_single_column(y, shape), threshold=0.5, **kwargs
        )
        np.testing.assert_array_equal(column[0], flat[0])
        np.testing.assert_array_equal(column[1], flat[1])
        return
    function = getattr(sift, entry)
    flat = function(X, y, k=2, threshold=0.5, **kwargs)
    column = function(X, _as_single_column(y, shape), k=2, threshold=0.5, **kwargs)
    assert column == flat
    assert flat[0] == "a"


def test_stability_tune_threshold_rejects_a_wide_target_clearly():
    X, y = _single_column_problem("regression")
    selector = sift.StabilitySelector(
        n_bootstrap=4, random_state=0, n_jobs=1, verbose=False
    ).fit(X, y)
    with pytest.raises(ValueError) as excinfo:
        selector.tune_threshold(X, np.column_stack([y, y]), cv=2)
    assert str(excinfo.value).startswith(
        "2-D y is only supported for select_cefsplus / CEFSPlusSelector"
    )
    assert "y has shape (80, 2)" in str(excinfo.value)


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
