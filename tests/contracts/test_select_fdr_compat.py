"""Compatibility contracts for the public knockoff-FDR entry point."""

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
import pytest

import sift
from sift.selection import select_fdr as selection_select_fdr


@dataclass(frozen=True)
class FDRData:
    X: pd.DataFrame
    y: np.ndarray
    sample_weight: np.ndarray


@pytest.fixture(scope="module")
def fdr_data() -> FDRData:
    rng = np.random.default_rng(811)
    n = 160
    values = rng.normal(size=(n, 8))
    y = 3.0 * values[:, 0] - 2.0 * values[:, 2] + 0.2 * rng.normal(size=n)
    X = pd.DataFrame(
        values,
        columns=[
            "strong_a",
            "noise_a",
            "strong_b",
            "noise_b",
            "noise_c",
            "noise_d",
            "noise_e",
            "noise_f",
        ],
    )
    return FDRData(
        X=X,
        y=y,
        sample_weight=np.linspace(0.3, 1.7, n),
    )


METADATA_KEYS = {
    "selector",
    "n_features",
    "q",
    "offset",
    "statistic",
    "s_method",
    "n_draws",
    "eta",
    "screen_pairs",
    "path_depth_requested",
    "path_depth",
    "path_depth_initial",
    "path_depth_adaptive",
    "gamma",
    "lambda_min",
    "s_mean",
    "s_median",
    "n_low_power_features",
    "random_state",
    "n_rows_used",
    "fdr_control",
    "per_draw_fdr_control",
    "q_scope",
    "aggregation",
    "aggregation_threshold",
    "aggregation_fdr_control",
    "aggregation_preserves_per_draw_fdr",
    "validity_model",
    "weighted_model",
    "n_zero_weight_variance_features",
    "n_features_input",
    "dropped_feature_positions",
    "dropped_feature_reasons",
    "feature_groups",
    "n_feature_groups",
    "group_mode",
    "group_fdr_control",
    "min_feasible_q",
    "n_tested",
    "n_tested_unit",
    "n_tested_per_draw",
    "n_eligible",
    "tested_state",
    "n_infeasible_draws",
    "tested_sets_vary",
    "n_discoveries_offset_0",
    "n_discoveries_offset_0_per_draw",
}


@pytest.mark.parametrize("input_kind", ("dataframe", "ndarray"))
@pytest.mark.parametrize("weighted", (False, True), ids=("unweighted", "weighted"))
def test_public_select_fdr_matrix(fdr_data, input_kind, weighted):
    X = fdr_data.X.copy() if input_kind == "dataframe" else fdr_data.X.to_numpy(copy=True)
    expected_features = ["strong_a", "strong_b"] if input_kind == "dataframe" else ["x0", "x2"]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sift.select_fdr(
            X,
            fdr_data.y.copy(),
            q=0.5,
            offset=0,
            statistic="relevance",
            sample_weight=fdr_data.sample_weight.copy() if weighted else None,
            subsample=None,
            random_state=17,
            verbose=False,
        )

    # NumPy 2.x must not leak spurious matmul warnings from finite weighted
    # variance calculations (the regression fixed in this compatibility batch).
    assert [(item.category, str(item.message)) for item in caught] == []
    assert type(result) is sift.KnockoffSelectionResult
    assert result.selected_features == expected_features
    assert result.selected_indices == [0, 2]
    assert set(result.selector_metadata) == METADATA_KEYS
    assert result.selector_metadata["selector"] == "knockoff_fdr"
    assert result.selector_metadata["n_features"] == 8
    assert result.selector_metadata["q"] == 0.5
    assert result.selector_metadata["offset"] == 0
    assert result.selector_metadata["statistic"] == "relevance"
    assert result.selector_metadata["weighted_model"] is weighted
    assert result.selector_metadata["fdr_control"] == "approximate_plugin"
    assert result.selector_metadata["aggregation"] == "single_draw"
    assert result.threshold is not None and np.isfinite(result.threshold)
    assert result.selection_frequency is None
    assert list(result.W.columns) == [
        "feature",
        "selected_index",
        "W",
        "selected",
        "selection_frequency",
        "relevance",
        "selector",
        "W_draw_0",
    ]
    assert result.W["feature"].tolist() == (
        fdr_data.X.columns.tolist()
        if input_kind == "dataframe"
        else [f"x{i}" for i in range(8)]
    )
    assert result.W.loc[result.W["selected"], "feature"].tolist() == expected_features
    assert np.isfinite(result.W["W"]).all()
    assert set(result.diagnostics_) == {
        "thresholds",
        "selection_sets",
        "offset_zero_selection_sets",
        "active_valid_positions",
    }
    assert result.diagnostics_["selection_sets"] == [[0, 2]]
    assert result.diagnostics_["active_valid_positions"] == list(range(8))

    ranking = result.get_feature_ranking()
    assert list(ranking.columns) == [
        "feature",
        "W",
        "rank",
        "selected",
        "selection_frequency",
        "selected_index",
        "relevance",
        "selector",
    ]
    assert ranking["feature"].iloc[:2].tolist() == expected_features
    assert ranking["rank"].tolist() == list(range(1, 9))


def test_public_select_fdr_weight_changes_statistics(fdr_data):
    common = {
        "q": 0.5,
        "offset": 0,
        "statistic": "relevance",
        "subsample": None,
        "random_state": 17,
        "verbose": False,
    }
    unweighted = sift.select_fdr(fdr_data.X.copy(), fdr_data.y.copy(), **common)
    weighted = sift.select_fdr(
        fdr_data.X.copy(),
        fdr_data.y.copy(),
        sample_weight=fdr_data.sample_weight.copy(),
        **common,
    )

    assert unweighted.selected_indices == weighted.selected_indices == [0, 2]
    assert not np.allclose(
        unweighted.W["W"].to_numpy(),
        weighted.W["W"].to_numpy(),
    )
    assert unweighted.threshold != weighted.threshold


def test_select_fdr_omitted_defaults_match_explicit_current_defaults(fdr_data):
    with warnings.catch_warnings(record=True) as omitted_warnings:
        warnings.simplefilter("always")
        omitted = sift.select_fdr(
            fdr_data.X.copy(),
            fdr_data.y.copy(),
            verbose=False,
        )
    with warnings.catch_warnings(record=True) as explicit_warnings:
        warnings.simplefilter("always")
        explicit = sift.select_fdr(
            fdr_data.X.copy(),
            fdr_data.y.copy(),
            q=0.1,
            statistic="relevance",
            n_draws=1,
            eta=0.5,
            offset=1,
            s_method="equi",
            min_eig=1e-3,
            screen_pairs=2000,
            statistic_options=None,
            feature_groups=None,
            group_corr_threshold=0.7,
            sample_weight=None,
            subsample=50_000,
            cache=None,
            random_state=0,
            n_jobs=1,
            verbose=False,
        )

    omitted_sig = [(item.category, str(item.message)) for item in omitted_warnings]
    explicit_sig = [(item.category, str(item.message)) for item in explicit_warnings]
    assert omitted_sig == explicit_sig
    assert omitted_sig and "m*q < 1" in omitted_sig[0][1]
    assert omitted.selected_features == explicit.selected_features
    assert omitted.selected_indices == explicit.selected_indices
    assert omitted.selector_metadata == explicit.selector_metadata
    assert omitted.threshold == explicit.threshold
    assert omitted.selection_frequency is explicit.selection_frequency is None
    pd.testing.assert_frame_equal(omitted.W, explicit.W)
    assert omitted.diagnostics_ == explicit.diagnostics_


def test_select_fdr_is_the_public_selection_export():
    assert sift.select_fdr is selection_select_fdr


def test_select_fdr_prebuilt_cache_rejects_construction_overrides(fdr_data):
    cache = sift.build_cache(
        fdr_data.X.copy(),
        subsample=None,
        compute_Rxx=True,
    )
    with pytest.raises(ValueError, match="sample_weight cannot be passed"):
        sift.select_fdr(
            cache=cache,
            y=fdr_data.y,
            sample_weight=fdr_data.sample_weight,
            verbose=False,
        )
    with pytest.raises(ValueError, match="subsample cannot be passed"):
        sift.select_fdr(
            cache=cache,
            y=fdr_data.y,
            subsample=None,
            verbose=False,
        )
    with pytest.raises(ValueError, match="Exactly one of X or cache"):
        sift.select_fdr(
            fdr_data.X,
            fdr_data.y,
            cache=cache,
            verbose=False,
        )


def test_select_fdr_rejects_misaligned_feature_groups(fdr_data):
    with pytest.raises(ValueError, match="expected exactly 8"):
        sift.select_fdr(
            fdr_data.X,
            fdr_data.y,
            feature_groups=["a", "b"],
            verbose=False,
        )
