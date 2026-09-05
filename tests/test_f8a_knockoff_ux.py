"""Public-contract tests for F8a knockoff feasibility diagnostics."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sift import KnockoffSelector, select_fdr
from sift.estimators.copula import build_cache
from sift.selection.knockoff_filter import knockoff_threshold


def _feasibility_warnings(caught):
    return [item for item in caught if "m*q < 1" in str(item.message)]


def _frame(n=160, p=12, seed=0, n_signal=4):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = X.iloc[:, :n_signal].sum(axis=1).to_numpy() + 0.25 * rng.normal(size=n)
    return X, y


def test_min_feasible_q_uses_effective_m_not_raw_width():
    X, y = _frame(p=12)
    X["const"] = 1.0
    result = select_fdr(X, y, q=0.2, offset=1, random_state=0, verbose=False)
    meta = result.selector_metadata
    assert meta["n_features_input"] == 13
    assert meta["n_tested"] == 12
    assert meta["n_tested_unit"] == "feature"
    assert meta["n_eligible"] == 12
    assert meta["tested_state"] == "post_screening"
    assert meta["n_infeasible_draws"] == 0
    assert meta["min_feasible_q"] == pytest.approx(1.0 / 12.0)
    assert meta["n_tested_per_draw"] == [12]
    assert meta["tested_sets_vary"] is False
    assert meta["fdr_control"] == "approximate_plugin"
    assert meta["n_discoveries_offset_0"] >= len(result.selected_features)
    assert "const" not in result.selected_features


def test_knockoff_plus_warns_only_when_m_q_lt_one():
    X, y = _frame(p=8, n_signal=3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        infeasible = select_fdr(X, y, q=0.1, offset=1, random_state=1, verbose=False)
    matching = _feasibility_warnings(caught)
    assert len(matching) == 1
    assert "cannot select any tested unit" in str(matching[0].message)
    assert Path(matching[0].filename) == Path(__file__)
    assert infeasible.selector_metadata["n_tested"] == 8
    assert infeasible.selector_metadata["n_eligible"] == 8
    assert infeasible.selector_metadata["tested_state"] == "post_screening"
    assert infeasible.selector_metadata["n_infeasible_draws"] == 1
    assert infeasible.selector_metadata["min_feasible_q"] == pytest.approx(0.125)
    assert infeasible.selector_metadata["fdr_control"] == "approximate_plugin"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        feasible = select_fdr(X, y, q=0.2, offset=1, random_state=1, verbose=False)
    assert _feasibility_warnings(caught) == []
    assert feasible.selector_metadata["n_tested"] == 8
    assert feasible.selector_metadata["n_infeasible_draws"] == 0

    with warnings.catch_warnings(record=True) as caught_off0:
        warnings.simplefilter("always")
        off0 = select_fdr(X, y, q=0.1, offset=0, random_state=1, verbose=False)
    assert not any("m*q < 1" in str(item.message) for item in caught_off0)
    assert off0.selector_metadata["n_discoveries_offset_0"] == len(off0.selected_features)


def test_offset_zero_count_reuses_w_and_excludes_include():
    X, y = _frame(p=10, n_signal=3)
    result = select_fdr(
        X,
        y,
        q=0.2,
        offset=1,
        include=["f9"],
        include_provenance="prespecified",
        random_state=2,
        verbose=False,
    )
    meta = result.selector_metadata
    assert meta["n_tested"] == 9
    assert "f9" in result.selected_features
    assert all(
        9 not in draw for draw in result.diagnostics_["offset_zero_selection_sets"]
    )
    W = result.W.set_index("feature")
    w_disc = W.loc[W["role"] == "discovery", "W"].to_numpy()
    threshold0 = knockoff_threshold(w_disc, 0.2, offset=0)
    expected = int(np.sum(w_disc >= threshold0)) if np.isfinite(threshold0) else 0
    assert meta["n_discoveries_offset_0"] == expected
    assert meta["n_discoveries_offset_0_per_draw"] == [expected]


def test_grouped_m_is_group_count_not_raw_p():
    X, y = _frame(p=9, n_signal=3)
    groups = ["a", "a", "a", "b", "b", "b", "c", "c", "c"]
    with pytest.warns(UserWarning, match=r"m\*q < 1"):
        result = select_fdr(
            X,
            y,
            q=0.2,
            offset=1,
            feature_groups=groups,
            random_state=3,
            verbose=False,
        )
    meta = result.selector_metadata
    assert meta["n_tested"] == 3
    assert meta["n_tested_unit"] == "group"
    assert meta["n_eligible"] == 3
    assert meta["tested_state"] == "post_screening"
    assert meta["n_infeasible_draws"] == 1
    assert meta["min_feasible_q"] == pytest.approx(1.0 / 3.0)
    assert meta["fdr_control"] == "none"
    assert meta["group_fdr_control"] == "none"


def test_cluster_representatives_set_tested_unit_and_expand_offset_zero():
    rng = np.random.default_rng(4)
    n = 180
    z = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "a": z + 0.01 * rng.normal(size=n),
            "a_dup": z + 0.01 * rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.normal(size=n),
        }
    )
    y = 2.0 * z + rng.normal(scale=0.3, size=n)
    result = select_fdr(
        X,
        y,
        q=0.2,
        offset=0,
        feature_groups="auto",
        group_corr_threshold=0.9,
        random_state=4,
        verbose=False,
    )
    meta = result.selector_metadata
    assert meta["n_tested_unit"] == "cluster_representative"
    assert meta["n_tested"] == meta["n_representatives"]
    assert meta["n_tested"] < meta["n_features_input"]
    assert meta["fdr_control"] == "none"
    assert meta["n_discoveries_offset_0"] >= 0


def test_screen_pairs_is_the_tested_count_and_sets_may_vary():
    X, y = _frame(p=12, n_signal=4)
    result = select_fdr(
        X,
        y,
        q=0.2,
        offset=1,
        statistic="ridge",
        screen_pairs=5,
        n_draws=3,
        eta=0.5,
        random_state=5,
        verbose=False,
    )
    meta = result.selector_metadata
    assert meta["n_tested"] == 5
    assert meta["n_tested_per_draw"] == [5, 5, 5]
    assert meta["n_eligible"] == 12
    assert meta["tested_state"] == "post_screening"
    assert meta["n_infeasible_draws"] == 0
    assert meta["min_feasible_q"] == pytest.approx(0.2)
    assert meta["aggregation"] == "selection_frequency"
    assert meta["fdr_control"] == "none"
    assert len(meta["n_discoveries_offset_0_per_draw"]) == 3


def test_zero_target_records_degenerate_feasibility():
    X, _y = _frame(p=6)
    cache = build_cache(X, subsample=None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = select_fdr(
            cache=cache,
            y=np.ones(len(X)),
            q=0.2,
            offset=1,
            verbose=False,
        )
    assert _feasibility_warnings(caught) == []
    meta = result.selector_metadata
    assert meta["tested_state"] == "not_run"
    assert meta["n_tested"] == 0
    assert meta["n_tested_per_draw"] == []
    assert meta["n_eligible"] == 6
    assert meta["n_infeasible_draws"] == 0
    assert meta["min_feasible_q"] == float("inf")
    assert meta["n_discoveries_offset_0"] == 0
    assert meta["n_discoveries_offset_0_per_draw"] == []
    assert result.diagnostics_["offset_zero_selection_sets"] == []
    assert result.diagnostics_["tested_state"] == "not_run"


def test_sklearn_wrapper_exposes_feasibility_and_raw_transform():
    X, y = _frame(p=12, n_signal=4)
    selector = KnockoffSelector(q=0.2, offset=1, random_state=6, verbose=False)
    selector.fit(X, y)
    meta = selector.result_.selector_metadata
    assert meta["n_tested"] == 12
    assert meta["min_feasible_q"] == pytest.approx(1.0 / 12.0)
    transformed = selector.transform(X)
    assert list(transformed.columns) == selector.selected_features_
    pd.testing.assert_frame_equal(transformed, X[selector.selected_features_])


def test_grouped_screened_draws_warn_per_draw_not_aggregate():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, 12)), columns=[f"f{i}" for i in range(12)])
    y = rng.normal(size=60)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = select_fdr(
            X,
            y,
            q=0.5,
            statistic="ridge",
            screen_pairs=2,
            feature_groups=[0] * 6 + [1] * 6,
            n_draws=10,
            eta=0.1,
            random_state=0,
            verbose=False,
        )
    matching = _feasibility_warnings(caught)
    assert len(matching) == 1
    message = str(matching[0].message)
    assert "cannot select on" in message
    assert "does not imply the aggregated selection is empty" in message
    assert "cannot select any tested unit" not in message
    assert Path(matching[0].filename) == Path(__file__)
    meta = result.selector_metadata
    assert len(result.selected_features) == 12
    assert meta["n_tested_unit"] == "group"
    assert meta["n_eligible"] == 2
    assert meta["tested_state"] == "post_screening"
    assert meta["n_tested_per_draw"] == [1, 2, 2, 2, 2, 1, 2, 1, 2, 2]
    assert meta["n_tested"] == 1
    assert meta["min_feasible_q"] == pytest.approx(1.0)
    assert 0 < meta["n_infeasible_draws"] < meta["n_draws"]


def test_screened_constant_target_does_not_invent_tested_counts():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 18)), columns=[f"f{i}" for i in range(18)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = select_fdr(
            X,
            y=np.ones(len(X)),
            q=0.2,
            statistic="ridge",
            screen_pairs=2,
            n_draws=3,
            verbose=False,
        )
    assert _feasibility_warnings(caught) == []
    meta = result.selector_metadata
    assert meta["tested_state"] == "not_run"
    assert meta["n_tested"] == 0
    assert meta["n_tested_per_draw"] == []
    assert meta["n_eligible"] == 18
    assert meta["n_infeasible_draws"] == 0
    assert meta["min_feasible_q"] == float("inf")
    assert meta["n_discoveries_offset_0_per_draw"] == []
    assert result.diagnostics_["offset_zero_selection_sets"] == []


def test_sklearn_wrapper_feasibility_warning_points_to_caller():
    X, y = _frame(p=8, n_signal=3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        KnockoffSelector(q=0.1, offset=1, random_state=6, verbose=False).fit(X, y)
    matching = _feasibility_warnings(caught)
    assert len(matching) == 1
    assert "cannot select any tested unit" in str(matching[0].message)
    assert Path(matching[0].filename) == Path(__file__)
