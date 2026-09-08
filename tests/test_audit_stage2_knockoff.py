"""Focused regression tests for the accepted Stage 2 knockoff corrections."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from sift import KnockoffSelector, select_fdr
from sift.selection.knockoff_filter import (
    _draw_knockoff_plus_infeasible,
    e_bh_reject,
    e_bh_threshold,
)


def test_e_bh_rejects_nan_and_negative_but_accepts_positive_infinity():
    with pytest.raises(ValueError, match="non-negative.*NaN"):
        e_bh_threshold(np.array([np.nan, 8.0]), 0.2)
    with pytest.raises(ValueError, match="non-negative.*NaN"):
        e_bh_reject(np.array([-1.0, 8.0]), 0.2)

    assert e_bh_threshold(np.array([np.inf]), 0.2) == pytest.approx(5.0)
    np.testing.assert_array_equal(e_bh_reject(np.array([np.inf]), 0.2), [True])


def test_knockoff_plus_boundary_q_one_over_m_is_feasible():
    assert _draw_knockoff_plus_infeasible(0, 0.1) is True
    assert _draw_knockoff_plus_infeasible(49, 1.0 / 49.0) is False
    assert _draw_knockoff_plus_infeasible(49, np.nextafter(1.0 / 49.0, 0.0)) is True


def test_evalue_aggregation_does_not_use_eta_for_e_bh():
    rng = np.random.default_rng(8)
    X = pd.DataFrame(rng.normal(size=(120, 6)), columns=[f"f{i}" for i in range(6)])
    y = X["f0"].to_numpy() + 0.2 * rng.normal(size=len(X))
    low = select_fdr(
        X, y, q=0.5, n_draws=3, eta=0.1, aggregation="evalues", random_state=2, verbose=False
    )
    high = select_fdr(
        X, y, q=0.5, n_draws=3, eta=0.9, aggregation="evalues", random_state=2, verbose=False
    )
    assert low.selected_features == high.selected_features
    assert low.selector_metadata["aggregation_threshold"] is None


def test_cluster_expansion_does_not_copy_representative_evalues_to_members():
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
    W = result.W
    non_representatives = ~W["is_representative"].to_numpy(dtype=bool)
    assert W.loc[non_representatives, "evalue"].isna().all()
    np.testing.assert_allclose(
        W["representative_evalue"],
        W["evalue"].where(~non_representatives, W["representative_evalue"]),
        equal_nan=True,
    )
    assert set(result.selector_metadata["evalue_universe"]).issubset(
        set(W.loc[W["is_representative"], "selected_index"])
    )
    assert result.selector_metadata["q_calibration_unit"] == "cluster_representative"


def test_supervised_encoding_downgrade_reaches_ordinary_nested_result():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(160, 8)), columns=[f"f{i}" for i in range(8)])
    X["team"] = np.resize(np.array(["a", "b", "c"], dtype=object), len(X))
    y = (X.iloc[:, :4].sum(axis=1) + rng.normal(size=len(X)) > 0).astype(np.int64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = KnockoffSelector(
            q=0.5,
            n_draws=3,
            cat_encoding="loo_logit",
            feature_groups="auto",
            verbose=False,
        ).fit(X, y).result_
    assert caught
    assert result.selector_metadata["representative_fdr_control"] == "none"
    assert result.selector_metadata["representative_per_draw_fdr_control"] == "none"
    nested = result.diagnostics_["representative_result"].selector_metadata
    assert nested["fdr_control"] == "none"
    assert nested["per_draw_fdr_control"] == "none"
    assert nested["cat_encoding"] == "loo_logit"
    assert "Model-X exchangeability" in nested["validity_note"]
