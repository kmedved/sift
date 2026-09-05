"""Public-contract tests for opt-in knockoff e-value aggregation."""

from __future__ import annotations

from types import SimpleNamespace
import copy
import warnings

import numpy as np
import pandas as pd
import pytest

from sift import KnockoffSelector, as_result, select_fdr
from sift.selection.knockoff_filter import (
    _knockoff_draw_evalues,
    _pair_screen,
    _stat_cefsplus,
    _stat_lsm,
    e_bh_reject,
    e_bh_threshold,
    knockoff_threshold,
)


def _frame(n=160, p=12, seed=0, n_signal=4):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = X.iloc[:, :n_signal].sum(axis=1).to_numpy() + 0.25 * rng.normal(size=n)
    return X, y


def test_omitted_aggregation_preserves_frequency_vote_and_draws():
    X, y = _frame()
    omitted = select_fdr(X, y, q=0.2, n_draws=5, eta=0.5, random_state=0, verbose=False)
    explicit = select_fdr(
        X,
        y,
        q=0.2,
        n_draws=5,
        eta=0.5,
        aggregation="selection_frequency",
        random_state=0,
        verbose=False,
    )
    assert omitted.selected_features == explicit.selected_features
    assert omitted.selector_metadata["aggregation"] == "selection_frequency"
    assert omitted.selector_metadata["fdr_control"] == "none"
    pd.testing.assert_frame_equal(omitted.W, explicit.W)


def test_evalues_reuse_the_same_draws_as_frequency_vote():
    X, y = _frame()
    freq = select_fdr(X, y, q=0.2, n_draws=5, random_state=0, verbose=False)
    ev = select_fdr(
        X, y, q=0.2, n_draws=5, aggregation="evalues", random_state=0, verbose=False
    )
    pd.testing.assert_series_equal(freq.W["W"], ev.W["W"])
    for draw_idx in range(5):
        pd.testing.assert_series_equal(
            freq.W[f"W_draw_{draw_idx}"], ev.W[f"W_draw_{draw_idx}"]
        )
    assert ev.selector_metadata["aggregation"] == "evalues"
    assert ev.selector_metadata["evalue_bound"] == "aggregate_null_expectation"
    assert ev.selector_metadata["evalue_validated"] is True
    assert ev.selector_metadata["fdr_control"] == "approximate_plugin"
    assert ev.selector_metadata["aggregation_fdr_control"] == "approximate_plugin"
    assert ev.selector_metadata["q_scope"] == "aggregated"
    assert ev.selector_metadata["evalue_m"] == ev.selector_metadata["n_tested"]
    assert ev.selector_metadata["evalue_m"] == 12
    assert ev.selector_metadata["evalue_zero_padded"] is False
    assert "evalue" in ev.W.columns
    assert ev.threshold is None


def test_knockoff_evalue_and_ebh_literal_arithmetic():
    W = np.array([4.0, 3.0, 2.0, -1.0])
    threshold = knockoff_threshold(W, 0.5, offset=1)
    assert threshold == pytest.approx(2.0)
    tested = np.ones(4, dtype=bool)
    e = _knockoff_draw_evalues(W, threshold=threshold, tested_mask=tested, m=4)
    n_neg = int(np.sum(W <= -threshold))
    expected = np.where(W >= threshold, 4.0 / (1.0 + n_neg), 0.0)
    np.testing.assert_allclose(e, expected)
    screened = np.array([True, True, False, True])
    e_pad = _knockoff_draw_evalues(W, threshold=threshold, tested_mask=screened, m=5)
    assert e_pad[2] == 0.0
    assert e_pad[0] == pytest.approx(5.0 / (1.0 + int(np.sum(screened & (W <= -threshold)))))

    e_avg = np.array([20.0, 15.0, 14.0, 0.0, 0.0])
    assert e_bh_threshold(e_avg, 0.2, m=5) == pytest.approx(25.0 / 3.0)
    np.testing.assert_array_equal(
        e_bh_reject(e_avg, 0.2, m=5),
        np.array([True, True, True, False, False]),
    )


def test_zero_padding_uses_common_universe_m_not_nonzero_count():
    W = np.array([3.0, 0.0, -0.5, 2.0])
    tested = np.array([True, False, True, True])
    threshold = knockoff_threshold(W[tested], 0.5, offset=1)
    e = _knockoff_draw_evalues(W, threshold=threshold, tested_mask=tested, m=4)
    assert e[1] == 0.0
    assert np.count_nonzero(e) < 4
    selected_weight = float(np.max(e))
    n_neg = int(np.sum(tested & (W <= -threshold)))
    assert selected_weight == pytest.approx(4.0 / (1.0 + n_neg))


def test_pair_screen_is_swap_invariant():
    rng = np.random.default_rng(4)
    r = rng.normal(size=8)
    rt = rng.normal(size=8)
    kept = _pair_screen(r, rt, 3)
    swapped = _pair_screen(rt, r, 3)
    np.testing.assert_array_equal(np.sort(kept), np.sort(swapped))


def test_adaptive_cefsplus_sign_flip_fails_on_saturating_path():
    p = 12
    r = np.linspace(0.25, 0.10, p)
    ctx = SimpleNamespace(
        Z=np.zeros((100, p)),
        kept=np.arange(p),
        G=np.eye(2 * p),
        r_aug=np.r_[r, np.zeros(p)],
        options={
            "path_depth": 10,
            "_adaptive_path_depth": True,
            "_q": 0.2,
            "_offset": 1,
        },
    )
    original = _stat_cefsplus(ctx)
    swapped = copy.deepcopy(ctx)
    swapped.options = {
        "path_depth": 10,
        "_adaptive_path_depth": True,
        "_q": 0.2,
        "_offset": 1,
    }
    swapped.r_aug[[0, p]] = swapped.r_aug[[p, 0]]
    flipped = _stat_cefsplus(swapped)
    expect = original.copy()
    expect[0] *= -1.0
    assert ctx.options["_path_depth_used"] != swapped.options["_path_depth_used"]
    assert not np.allclose(flipped, expect)


def test_truncated_lsm_equal_pair_is_not_antisymmetric():
    ctx = SimpleNamespace(
        Z=np.zeros((100, 2)),
        kept=np.arange(2),
        G=np.eye(4),
        r_aug=np.array([0.5, 0.3, 0.5, 0.1]),
        options={"max_steps": 1},
        rng=np.random.default_rng(0),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        W = _stat_lsm(ctx)
    assert W[0] != 0.0


def test_path_statistics_are_exploratory_in_evalue_mode():
    X, y = _frame(p=8, n_signal=3)
    cefs = select_fdr(
        X,
        y,
        q=0.5,
        n_draws=2,
        aggregation="evalues",
        statistic="cefsplus",
        random_state=0,
        verbose=False,
    )
    assert cefs.selector_metadata["evalue_validated"] is False
    assert (
        "cefsplus_path_not_sign_flip_guaranteed"
        in cefs.selector_metadata["evalue_exploratory_reasons"]
    )
    assert cefs.selector_metadata["fdr_control"] == "none"
    assert cefs.selector_metadata["aggregation_fdr_control"] == "none"
    assert cefs.selector_metadata["per_draw_fdr_control"] == "none"
    lsm = select_fdr(
        X,
        y,
        q=0.5,
        n_draws=2,
        aggregation="evalues",
        statistic="lsm",
        random_state=0,
        verbose=False,
    )
    assert lsm.selector_metadata["evalue_validated"] is False
    assert (
        "lsm_truncated_path_not_sign_flip_guaranteed"
        in lsm.selector_metadata["evalue_exploratory_reasons"]
    )
    assert lsm.selector_metadata["fdr_control"] == "none"
    assert lsm.selector_metadata["per_draw_fdr_control"] == "none"
    assert lsm.selector_metadata["aggregation_fdr_control"] == "none"


def test_varying_pair_screen_is_exploratory_and_records_padding():
    X, y = _frame(p=12, n_signal=4)
    result = select_fdr(
        X,
        y,
        q=0.5,
        statistic="ridge",
        screen_pairs=4,
        n_draws=6,
        aggregation="evalues",
        random_state=1,
        verbose=False,
    )
    meta = result.selector_metadata
    assert meta["aggregation"] == "evalues"
    assert meta["evalue_validated"] is False
    assert "screening_universe_not_fixed_before_statistics" in meta["evalue_exploratory_reasons"]
    assert meta["fdr_control"] == "none"
    assert meta["aggregation_fdr_control"] == "none"
    assert meta["per_draw_fdr_control"] == "approximate_plugin"
    assert meta["evalue_m"] == len(meta["evalue_universe"])
    assert meta["evalue_m"] >= meta["n_tested"]
    assert result.diagnostics_["screening_sets"]
    assert len(result.diagnostics_["screening_sets"]) == 6
    assert result.diagnostics_["evalue_universe"] == meta["evalue_universe"]


def test_grouped_evalues_are_exploratory():
    X, y = _frame(p=9, n_signal=3)
    result = select_fdr(
        X,
        y,
        q=0.5,
        n_draws=4,
        aggregation="evalues",
        feature_groups=["a"] * 3 + ["b"] * 3 + ["c"] * 3,
        random_state=3,
        verbose=False,
    )
    meta = result.selector_metadata
    assert meta["evalue_validated"] is False
    assert "grouped_or_representative_heuristic" in meta["evalue_exploratory_reasons"]
    assert meta["fdr_control"] == "none"
    assert meta["n_tested_unit"] == "group"


def test_constant_target_evalues_are_not_run():
    X, _y = _frame(p=6)
    result = select_fdr(
        X,
        y=np.ones(len(X)),
        n_draws=3,
        aggregation="evalues",
        verbose=False,
    )
    meta = result.selector_metadata
    assert meta["tested_state"] == "not_run"
    assert meta["evalue_m"] == 0
    assert meta["evalue_universe"] == []
    assert meta["evalue_validated"] is False
    assert meta["n_tested_per_draw"] == []
    assert meta["aggregation_threshold"] is None
    assert result.W["evalue"].eq(0.0).all()


def test_incompatible_evalue_options_raise():
    X, y = _frame(p=8)
    with pytest.raises(ValueError, match="n_draws > 1"):
        select_fdr(X, y, aggregation="evalues", n_draws=1, verbose=False)
    with pytest.raises(ValueError, match="offset=1"):
        select_fdr(
            X, y, aggregation="evalues", n_draws=3, offset=0, verbose=False
        )
    with pytest.raises(ValueError, match="selection_frequency"):
        select_fdr(
            X, y, aggregation="selection_frequency", n_draws=1, verbose=False
        )
    with pytest.raises(ValueError, match="aggregation must be"):
        select_fdr(X, y, aggregation="mean", n_draws=3, verbose=False)


def test_cluster_evalues_expand_and_remain_exploratory():
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
        q=0.5,
        n_draws=3,
        aggregation="evalues",
        feature_groups="auto",
        group_corr_threshold=0.9,
        random_state=4,
        verbose=False,
    )
    meta = result.selector_metadata
    assert meta["aggregation"] == "evalues_then_cluster_expansion"
    assert meta["fdr_control"] == "none"
    assert meta["evalue_validated"] is False
    assert "grouped_or_representative_heuristic" in meta["evalue_exploratory_reasons"]
    assert "evalue" in result.W.columns
    assert "is_representative" in result.W.columns


def test_evalues_keep_include_out_of_the_tested_universe():
    X, y = _frame(p=10, n_signal=3)
    result = select_fdr(
        X,
        y,
        q=0.2,
        n_draws=3,
        aggregation="evalues",
        include=["f9"],
        include_provenance="prespecified",
        random_state=2,
        verbose=False,
    )
    assert "f9" in result.selected_features
    assert 9 not in result.selector_metadata["evalue_universe"]
    assert result.selector_metadata["evalue_validated"] is True
    assert result.selector_metadata["fdr_control"] == "approximate_plugin"


def test_sklearn_wrapper_exposes_evalues():
    X, y = _frame()
    selector = KnockoffSelector(
        q=0.2,
        n_draws=4,
        aggregation="evalues",
        random_state=6,
        verbose=False,
    )
    selector.fit(X, y)
    meta = selector.result_.selector_metadata
    assert meta["aggregation"] == "evalues"
    assert meta["evalue_validated"] is True
    assert meta["fdr_control"] == "approximate_plugin"
    transformed = selector.transform(X)
    assert list(transformed.columns) == selector.selected_features_


def test_evalue_survives_ranking_and_normalized_view():
    X, y = _frame()
    result = select_fdr(
        X, y, q=0.5, n_draws=5, aggregation="evalues", random_state=0, verbose=False
    )
    ranking = result.get_feature_ranking()
    assert "evalue" in ranking.columns
    assert float(ranking["evalue"].max()) > 0.0
    view = as_result(result, input_features=list(X.columns))
    assert "evalue" in view.table.columns
    assert float(view.table["evalue"].max()) > 0.0


def test_grouped_evalues_respect_exclude_mask():
    rng = np.random.default_rng(4)
    X = pd.DataFrame(rng.normal(size=(1200, 12)), columns=[f"f{i}" for i in range(12)])
    y = X.iloc[:, :8].sum(axis=1).to_numpy() + 0.1 * rng.normal(size=1200)
    freq = select_fdr(
        X,
        y,
        q=0.5,
        n_draws=3,
        feature_groups=np.repeat(np.arange(6), 2),
        exclude=["f1"],
        include_provenance="prespecified",
        verbose=False,
    )
    ev = select_fdr(
        X,
        y,
        q=0.5,
        n_draws=3,
        aggregation="evalues",
        feature_groups=np.repeat(np.arange(6), 2),
        exclude=["f1"],
        include_provenance="prespecified",
        verbose=False,
    )
    f1_freq = freq.W.loc[freq.W["feature"] == "f1"].iloc[0]
    f1_ev = ev.W.loc[ev.W["feature"] == "f1"].iloc[0]
    assert bool(f1_freq["selected"]) is False
    assert bool(f1_ev["selected"]) is False
    assert f1_ev["role"] == "ineligible"
    assert float(f1_ev["evalue"]) == 0.0
    assert "f1" not in ev.selected_features


def test_supervised_encoding_downgrades_evalue_validity_fields():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(300, 12)), columns=[f"f{i}" for i in range(12)])
    X["team"] = np.resize(np.array(["a", "b", "c"], dtype=object), 300)
    y = (X.iloc[:, :6].sum(axis=1) + rng.normal(size=300) > 0).astype(np.int64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        selector = KnockoffSelector(
            q=0.2,
            n_draws=3,
            aggregation="evalues",
            cat_encoding="loo_logit",
            verbose=False,
        ).fit(X, y)
    assert all(issubclass(item.category, UserWarning) for item in caught)
    meta = selector.result_.selector_metadata
    assert meta["fdr_control"] == "none"
    assert meta["aggregation_fdr_control"] == "none"
    assert meta["evalue_validated"] is False
    assert "supervised_categorical_encoding" in meta["evalue_exploratory_reasons"]
    assert meta["per_draw_fdr_control"] == "none"


def test_supervised_encoding_downgrades_nested_representative_result():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(300, 12)), columns=[f"f{i}" for i in range(12)])
    X["team"] = np.resize(np.array(["a", "b", "c"], dtype=object), 300)
    y = (X.iloc[:, :6].sum(axis=1) + rng.normal(size=300) > 0).astype(np.int64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = KnockoffSelector(
            q=0.5,
            n_draws=3,
            aggregation="evalues",
            cat_encoding="loo_logit",
            feature_groups="auto",
            verbose=False,
        ).fit(X, y).result_
    assert all(issubclass(item.category, UserWarning) for item in caught)
    nested = result.diagnostics_["representative_result"]
    for meta in (result.selector_metadata, nested.selector_metadata):
        assert meta["fdr_control"] == "none"
        assert meta["aggregation_fdr_control"] == "none"
        assert meta["evalue_validated"] is False
        assert meta["per_draw_fdr_control"] == "none"


def test_legacy_encoded_auto_groups_leave_nested_representative_metadata():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(300, 12)), columns=[f"f{i}" for i in range(12)])
    X["team"] = np.resize(np.array(["a", "b", "c"], dtype=object), 300)
    y = (X.iloc[:, :6].sum(axis=1) + rng.normal(size=300) > 0).astype(np.int64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = KnockoffSelector(
            q=0.5,
            n_draws=3,
            cat_encoding="loo_logit",
            feature_groups="auto",
            verbose=False,
        ).fit(X, y).result_
    assert all(issubclass(item.category, UserWarning) for item in caught)
    assert result.selector_metadata["fdr_control"] == "none"
    assert result.selector_metadata["per_draw_fdr_control"] == "none"
    assert result.selector_metadata["cat_encoding"] == "loo_logit"
    nested = result.diagnostics_["representative_result"].selector_metadata
    assert nested["fdr_control"] == "none"
    assert nested["per_draw_fdr_control"] == "approximate_plugin"
    assert "cat_encoding" not in nested
