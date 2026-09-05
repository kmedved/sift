from dataclasses import replace
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

import sift.selection.knockoff_filter as knockoff_filter_module
from sift.estimators.copula import FeatureCache, build_cache
from sift import select_cached, select_cefsplus
from sift.estimators.knockoffs import fit_gaussian_knockoffs, sample_gaussian_knockoffs
from sift.selection.knockoff_filter import (
    _KNOCKOFF_STAT_REGISTRY,
    _build_active_rxx,
    _build_context,
    _cefsplus_incremental_scores,
    _group_knockoff_statistics,
    _lasso_entry_penalties,
    _SUBSAMPLE_DEFAULT,
    knockoff_threshold,
    sample_knockoffs,
    select_fdr,
)
from sift.selectors import KnockoffSelector


def _expect_infeasible_knockoff_plus():
    return pytest.warns(UserWarning, match=r"knockoff\+ \(offset=1\).*m\*q < 1")


def _signal_frame(n: int = 90, p: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    X[:, 1] = 0.45 * X[:, 0] + rng.normal(scale=0.9, size=n)
    y = 1.8 * X[:, 0] - 1.2 * X[:, 2] + rng.normal(scale=0.6, size=n)
    columns = [f"f{i}" for i in range(p)]
    return pd.DataFrame(X, columns=columns), y


def _context_for_cache(cache, y, *, seed: int = 0, screen_pairs=None):
    model = fit_gaussian_knockoffs(cache.Rxx)
    rng = np.random.default_rng(seed)
    Zt = sample_gaussian_knockoffs(cache.Z, model, rng)
    from sift.estimators.copula import weighted_rank_gauss_1d

    zy = weighted_rank_gauss_1d(np.asarray(y, dtype=np.float32), cache.sample_weight)
    return _build_context(
        cache.Z,
        Zt,
        zy,
        cache.sample_weight,
        model,
        screen_pairs=screen_pairs,
        options={},
        n_jobs=1,
        rng=np.random.default_rng(seed + 1),
    )


def test_knockoff_threshold_arithmetic_and_validation():
    W = np.array([3.0, 2.0, 1.0, -1.0])

    assert knockoff_threshold(W, 0.5, offset=1) == pytest.approx(2.0)
    assert knockoff_threshold(W, 0.5, offset=0) == pytest.approx(1.0)
    assert np.isinf(knockoff_threshold(np.array([1.0, -1.0]), 0.1, offset=1))
    assert np.isinf(knockoff_threshold(np.zeros(3), 0.5, offset=1))

    with pytest.raises(ValueError, match="q"):
        knockoff_threshold(W, True)
    with pytest.raises(ValueError, match="offset"):
        knockoff_threshold(W, 0.5, offset=2)
    with pytest.raises(ValueError, match="finite"):
        knockoff_threshold(np.array([np.nan]), 0.5)


def test_knockoff_threshold_matches_literal_scan_with_duplicates_and_zeros():
    rng = np.random.default_rng(1901)
    W = rng.choice(np.array([-3.0, -1.0, 0.0, 1.0, 2.0, 2.0]), size=500)

    for q in (0.1, 0.25, 0.7):
        for offset in (0, 1):
            expected = np.inf
            for threshold in np.unique(np.abs(W[W != 0.0])):
                ratio = (offset + np.sum(W <= -threshold)) / max(
                    1, np.sum(W >= threshold)
                )
                if ratio <= q:
                    expected = float(threshold)
                    break
            assert knockoff_threshold(W, q, offset=offset) == expected


def test_enabled_statistics_are_swap_antisymmetric():
    X, y = _signal_frame(n=80, p=6, seed=2)
    cache = build_cache(X, compute_Rxx=True, random_state=0)
    context = _context_for_cache(cache, y, seed=3)

    swapped = cache.Z.copy()
    swapped_t = context.Zt.copy()
    j = 2
    swapped[:, j], swapped_t[:, j] = swapped_t[:, j].copy(), swapped[:, j].copy()
    swapped_context = _build_context(
        swapped,
        swapped_t,
        context.zy,
        context.w,
        context.model,
        screen_pairs=None,
        options={},
        n_jobs=1,
        rng=np.random.default_rng(4),
    )
    g_perm = np.arange(context.G.shape[0])
    m = context.kept.shape[0]
    if m:
        g_perm[j] = j + m
        g_perm[j + m] = j
        swapped_context = replace(
            swapped_context,
            G=swapped_context.G[np.ix_(g_perm, g_perm)],
        )

    for spec in _KNOCKOFF_STAT_REGISTRY.values():
        if not spec.enabled:
            continue
        W = spec.fn(context)
        W_swapped = spec.fn(swapped_context)
        expected = W.copy()
        expected[j] *= -1.0
        np.testing.assert_allclose(W_swapped, expected, atol=1e-12)


def test_enabled_statistics_are_antisymmetric_under_multiple_pair_swaps():
    X, y = _signal_frame(n=120, p=12, seed=31)
    cache = build_cache(X, compute_Rxx=True, random_state=0)
    context = _context_for_cache(cache, y, seed=7, screen_pairs=None)
    p = cache.Z.shape[1]
    rng = np.random.default_rng(22)

    for _ in range(10):
        size = int(rng.integers(1, 4))
        swap_pairs = np.sort(rng.choice(p, size=size, replace=False))
        swapped = cache.Z.copy()
        swapped_t = context.Zt.copy()
        for j in swap_pairs:
            swapped[:, j], swapped_t[:, j] = swapped_t[:, j].copy(), swapped[:, j].copy()
        swapped_context = _build_context(
            swapped,
            swapped_t,
            context.zy,
            context.w,
            context.model,
            screen_pairs=None,
            options={},
            n_jobs=1,
            rng=np.random.default_rng(8),
        )
        g_perm = np.arange(context.G.shape[0])
        for j in swap_pairs:
            g_perm[j] = j + p
            g_perm[j + p] = j
        swapped_context = replace(
            swapped_context,
            G=swapped_context.G[np.ix_(g_perm, g_perm)],
        )

        for spec in _KNOCKOFF_STAT_REGISTRY.values():
            if not spec.enabled:
                continue
            W = spec.fn(context)
            W_swapped = spec.fn(swapped_context)
            expected = W.copy()
            expected[swap_pairs] *= -1.0
            np.testing.assert_allclose(W_swapped, expected, atol=1e-10)


def test_lsm_entry_penalties_follow_coefficient_path_after_lasso_drops():
    from sklearn.linear_model import lars_path_gram

    rng = np.random.default_rng(67)
    n, p = 300, 12
    X = rng.normal(size=(n, p))
    X[:, 1] = 0.95 * X[:, 0] + 0.3122 * rng.normal(size=n)
    X[:, 5] = 0.7 * X[:, 0] + 0.714 * rng.normal(size=n)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = X[:, 0] - 0.8 * X[:, 1] + 0.3 * X[:, 5] + rng.normal(scale=0.8, size=n)
    y -= y.mean()

    alphas, active, coefs = lars_path_gram(
        Xy=X.T @ y,
        Gram=X.T @ X,
        n_samples=n,
        method="lasso",
        max_iter=2 * p,
        eps=np.finfo(np.float64).eps,
    )
    assert len(alphas) - 1 > len(active)  # This path contains lasso drops.

    actual = _lasso_entry_penalties(alphas, coefs, active)
    expected = np.zeros(p)
    for col in range(p):
        nonzero = np.flatnonzero(coefs[col] != 0.0)
        if nonzero.size:
            expected[col] = alphas[max(int(nonzero[0]) - 1, 0)]
    for col in active:
        if not np.any(coefs[col] != 0.0):
            expected[col] = alphas[-1]

    old_final_active_mapping = np.zeros(p)
    for step, col in enumerate(active):
        old_final_active_mapping[col] = alphas[step]
    assert not np.allclose(old_final_active_mapping, expected)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


def test_cefsplus_tie_safe_wrapper_neutralizes_exact_pair_ties():
    G = np.eye(4, dtype=np.float64)
    r = np.array([0.5, 0.2, 0.5, 0.1], dtype=np.float64)
    tie_break = np.abs(r)

    h = _cefsplus_incremental_scores(G, r, path_depth=2, tie_break=tie_break)

    assert h[0] == pytest.approx(0.0)
    assert h[2] == pytest.approx(0.0)


def test_cefsplus_scores_use_objective_gain_not_rank_position():
    G = np.eye(4, dtype=np.float64)
    r = np.array([0.5, 0.2, 0.1, 0.05], dtype=np.float64)

    h = _cefsplus_incremental_scores(G, r, path_depth=2, tie_break=np.abs(r))

    shrink = 1.0 - 1e-6
    assert h[0] == pytest.approx(-np.log(1.0 - (0.5 * shrink) ** 2))
    assert h[0] < 1.0
    assert 0.0 < h[1] < h[0]


def test_cefsplus_scores_are_not_capped_at_first_gain():
    G = np.array(
        [
            [1.0, -0.35533709, -0.13037301, 0.02853066, -0.03676764, 0.24226190],
            [-0.35533709, 1.0, 0.15282338, -0.31051784, 0.53295197, -0.06022371],
            [-0.13037301, 0.15282338, 1.0, -0.47128653, 0.00133019, -0.35018935],
            [0.02853066, -0.31051784, -0.47128653, 1.0, -0.07580618, 0.46630033],
            [-0.03676764, 0.53295197, 0.00133019, -0.07580618, 1.0, -0.05207734],
            [0.24226190, -0.06022371, -0.35018935, 0.46630033, -0.05207734, 1.0],
        ],
        dtype=np.float64,
    )
    r = np.array([0.18113113, -0.17153982, -0.08009479, 0.45687760, 0.48604647, 0.07018047])

    h = _cefsplus_incremental_scores(G, r, path_depth=3, tie_break=np.abs(r))

    assert h[4] > 0.0
    assert h[1] > h[4]


def test_select_fdr_cefsplus_min_gain_zero_matches_omitted_option():
    X, y = _signal_frame(n=80, p=8, seed=30)

    omitted = select_fdr(
        X,
        y,
        statistic="cefsplus",
        statistic_options={"path_depth": 6},
        q=0.5,
        offset=0,
        random_state=12,
        verbose=False,
    )
    explicit_zero = select_fdr(
        X,
        y,
        statistic="cefsplus",
        statistic_options={"path_depth": 6, "min_gain_ratio": 0.0},
        q=0.5,
        offset=0,
        random_state=12,
        verbose=False,
    )

    np.testing.assert_array_equal(omitted.W["W"].to_numpy(), explicit_zero.W["W"].to_numpy())
    assert omitted.selected_features == explicit_zero.selected_features


def test_select_fdr_cefsplus_smoke_and_path_depth_metadata():
    X, y = _signal_frame(n=70, p=6, seed=13)

    result = select_fdr(
        X,
        y,
        statistic="cefsplus",
        statistic_options={"path_depth": 4},
        q=0.5,
        offset=0,
        random_state=3,
        verbose=False,
    )

    assert result.selector_metadata["statistic"] == "cefsplus"
    assert result.selector_metadata["path_depth"] == 4
    assert result.selector_metadata["path_depth_requested"] == 4
    assert np.isfinite(result.W["W"]).all()


def test_select_fdr_reports_effective_path_depth_metadata():
    X, y = _signal_frame(n=80, p=20, seed=32)

    defaulted = select_fdr(
        X,
        y,
        statistic="cefsplus",
        screen_pairs=None,
        q=0.5,
        offset=0,
        random_state=3,
        verbose=False,
    )
    capped = select_fdr(
        X.iloc[:, :6],
        y,
        statistic="cefsplus",
        statistic_options={"path_depth": 99},
        q=0.5,
        offset=0,
        random_state=3,
        verbose=False,
    )
    relevance = select_fdr(X, y, statistic="relevance", random_state=3, verbose=False)

    assert defaulted.selector_metadata["path_depth_requested"] is None
    assert defaulted.selector_metadata["path_depth"] == 10
    assert capped.selector_metadata["path_depth_requested"] == 99
    assert capped.selector_metadata["path_depth"] == 12
    assert relevance.selector_metadata["path_depth_requested"] is None
    assert relevance.selector_metadata["path_depth"] is None


def test_select_fdr_validates_statistic_options_by_statistic():
    X, y = _signal_frame(n=55, p=6, seed=33)

    with pytest.raises(ValueError, match="path_dept.*path_depth"):
        select_fdr(X, y, statistic="cefsplus", statistic_options={"path_dept": 3}, verbose=False)
    with pytest.raises(ValueError, match="Unknown statistic_options.*<none>") as exc:
        select_fdr(X, y, statistic="relevance", statistic_options={"min_gain_ratio": 0.0}, verbose=False)
    assert "_statistic_name" not in str(exc.value)

    with _expect_infeasible_knockoff_plus():
        result = select_fdr(
            X,
            y,
            statistic="cefsplus",
            statistic_options={"path_depth": 3, "min_gain_ratio": 0.0},
            verbose=False,
        )
    assert result.selector_metadata["path_depth"] == 3


def test_group_knockoff_statistics_are_sign_flip_safe():
    W = np.array([2.0, -1.0, 0.5, -3.0, 4.0, -4.0])
    codes = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)

    group_W = _group_knockoff_statistics(W, codes, 3)

    np.testing.assert_allclose(group_W, np.array([2.0, -3.0, 0.0]))
    np.testing.assert_allclose(_group_knockoff_statistics(-W, codes, 3), -group_W)


def test_select_fdr_feature_groups_thresholds_and_expands_selected_groups():
    X, y = _signal_frame(n=80, p=6, seed=14)
    groups = ["ab", "ab", "cd", "cd", "ef", "ef"]

    result = select_fdr(
        X,
        y,
        q=0.95,
        offset=0,
        feature_groups=groups,
        random_state=4,
        verbose=False,
    )

    assert result.selector_metadata["feature_groups"] is True
    assert result.selector_metadata["n_feature_groups"] == 3
    assert result.selector_metadata["group_mode"] == "signed_max_heuristic"
    assert result.selector_metadata["group_fdr_control"] == "none"
    assert result.selector_metadata["per_draw_fdr_control"] == "none"
    assert result.selector_metadata["fdr_control"] == "none"
    assert result.selector_metadata["aggregation_preserves_per_draw_fdr"] is False
    assert result.W["feature_group"].tolist() == groups
    assert "feature_group" in result.get_feature_ranking().columns
    assert result.diagnostics_["feature_groups"] == ["ab", "cd", "ef"]
    assert len(result.diagnostics_["group_W_draws"]) == 1
    assert len(result.diagnostics_["group_W_draws"][0]) == 3
    assert result.diagnostics_["group_thresholds"] == result.diagnostics_["thresholds"]
    selected_groups = result.W.loc[result.W["selected"], "feature_group"].unique()
    for group in selected_groups:
        group_rows = result.W["feature_group"] == group
        assert result.W.loc[group_rows, "selected"].all()


def test_select_fdr_feature_groups_zero_target_keeps_group_column():
    X, _ = _signal_frame(n=50, p=4, seed=15)
    groups = ["a", "a", "b", "b"]

    result = select_fdr(
        X,
        np.ones(len(X)),
        n_draws=2,
        feature_groups=groups,
        verbose=False,
    )

    assert result.W["feature_group"].tolist() == groups
    assert result.selector_metadata["tested_state"] == "not_run"
    assert result.selector_metadata["n_tested"] == 0
    assert result.selector_metadata["n_eligible"] == 2
    assert result.selector_metadata["feature_groups"] is True
    assert result.selector_metadata["group_fdr_control"] == "none"
    assert result.selector_metadata["per_draw_fdr_control"] == "none"
    assert result.diagnostics_["feature_groups"] == ["a", "b"]
    assert result.diagnostics_["group_W_draws"] == [[0.0, 0.0], [0.0, 0.0]]
    assert result.diagnostics_["group_thresholds"] == [float(np.inf), float(np.inf)]


def test_select_fdr_feature_groups_preserve_tuple_labels():
    X, y = _signal_frame(n=70, p=4, seed=39)
    groups = [("lag", 0), ("lag", 0), ("lag", 1), ("lag", 1)]

    result = select_fdr(
        X,
        y,
        q=0.8,
        offset=0,
        feature_groups=groups,
        random_state=3,
        verbose=False,
    )

    assert result.W["feature_group"].tolist() == groups
    assert result.diagnostics_["feature_groups"] == [("lag", 0), ("lag", 1)]


def test_select_fdr_feature_groups_accept_original_or_valid_lengths_only():
    X, y = _signal_frame(n=60, p=5, seed=34)
    X = X.copy()
    X["f3"] = 1.0
    X["f4"] = 2.0
    cache = build_cache(X, compute_Rxx=True, random_state=0)
    assert cache.valid_cols.tolist() == [0, 1, 2]

    original_groups = ["a", "b", "c", "dropped1", "dropped2"]
    with _expect_infeasible_knockoff_plus():
        original = select_fdr(cache=cache, y=y, feature_groups=original_groups, verbose=False)
    assert original.W["feature_group"].tolist() == ["a", "b", "c"]

    valid_groups = ["valid_a", "valid_b", "valid_c"]
    with _expect_infeasible_knockoff_plus():
        valid = select_fdr(cache=cache, y=y, feature_groups=valid_groups, verbose=False)
    assert valid.W["feature_group"].tolist() == valid_groups

    for bad in (["a", "b", "c", "d"], ["a", "b", "c", "d", "e", "f"], ["a", "b"]):
        with pytest.raises(ValueError, match="expected exactly 3 or 5"):
            select_fdr(cache=cache, y=y, feature_groups=bad, verbose=False)


def test_reserved_statistics_raise_clear_error():
    X, y = _signal_frame(n=50, p=5, seed=4)

    for statistic, spec in _KNOCKOFF_STAT_REGISTRY.items():
        if spec.enabled:
            continue
        with pytest.raises(ValueError, match="not yet enabled"):
            select_fdr(X, y, statistic=statistic, verbose=False)


def test_select_fdr_positional_pandas_y_and_cache_default_subsample():
    X, y = _signal_frame(n=70, p=6, seed=5)
    cache = build_cache(X, compute_Rxx=True, random_state=0)
    y_series = pd.Series(y, index=np.arange(10_000, 10_000 + len(y)))

    with _expect_infeasible_knockoff_plus():
        series_result = select_fdr(cache=cache, y=y_series, random_state=11, verbose=False)
    with _expect_infeasible_knockoff_plus():
        array_result = select_fdr(cache=cache, y=y, random_state=11, verbose=False)

    pd.testing.assert_frame_equal(series_result.W, array_result.W)
    assert series_result.selected_features == array_result.selected_features


def test_select_fdr_rejects_explicit_subsample_with_cache_even_when_default_value():
    X, y = _signal_frame(n=70, p=6, seed=35)
    cache = build_cache(X, compute_Rxx=True, random_state=0)

    with _expect_infeasible_knockoff_plus():
        select_fdr(cache=cache, y=y, verbose=False)
    with pytest.raises(ValueError, match="subsample"):
        select_fdr(cache=cache, y=y, subsample=50_000, verbose=False)
    with pytest.raises(ValueError, match="subsample"):
        select_fdr(cache=cache, y=y, subsample=None, verbose=False)


def test_select_fdr_preserves_non_string_feature_labels_and_ranks_mixed_labels():
    rng = np.random.default_rng(12)
    X = pd.DataFrame(
        rng.normal(size=(80, 4)),
        columns=[10, "1", 1, "x"],
    )
    y = X[10].to_numpy() + rng.normal(scale=0.2, size=len(X))

    result = select_fdr(X, y, q=0.5, offset=0, random_state=0, verbose=False)
    ranking = result.get_feature_ranking()

    assert result.W["feature"].tolist() == [10, "1", 1, "x"]
    assert set(result.selected_features).issubset({10, "1", 1, "x"})
    assert len(ranking) == 4


def test_select_fdr_weighted_inactive_column_and_local_rxx():
    Z = np.array(
        [
            [-1.0, -0.5, 10.0],
            [1.0, 0.5, 10.0],
            [0.0, 1.0, -3.0],
            [0.0, -1.0, 3.0],
        ],
        dtype=np.float32,
    )
    cache = FeatureCache(
        Z=Z,
        Rxx=None,
        valid_cols=np.array([0, 1, 2]),
        row_idx=np.arange(4),
        sample_weight=np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32),
        n_rows_original=4,
        feature_names=["a", "b", "zero_weight_only"],
        feature_names_are_synthetic=False,
    )

    with _expect_infeasible_knockoff_plus():
        result = select_fdr(cache=cache, y=np.array([0.0, 1.0, 0.0, 1.0]), verbose=False)

    row = result.W.loc[result.W["feature"] == "zero_weight_only"].iloc[0]
    assert row["W"] == pytest.approx(0.0)
    assert not bool(row["selected"])
    assert result.selector_metadata["n_zero_weight_variance_features"] == 1
    assert result.selector_metadata["weighted_model"] is True
    assert cache.Rxx is None


def test_select_fdr_validates_only_active_rxx_submatrix():
    Z = np.array(
        [
            [-1.0, -0.5, 10.0],
            [1.0, 0.5, 10.0],
            [0.0, 1.0, -3.0],
            [0.0, -1.0, 3.0],
        ],
        dtype=np.float32,
    )
    Rxx = np.array(
        [
            [1.0, 0.0, np.nan],
            [0.0, 1.0, np.nan],
            [np.nan, np.nan, np.nan],
        ],
        dtype=np.float32,
    )
    cache = FeatureCache(
        Z=Z,
        Rxx=Rxx,
        valid_cols=np.array([0, 1, 2]),
        row_idx=np.arange(4),
        sample_weight=np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32),
        n_rows_original=4,
        feature_names=["a", "b", "zero_weight_only"],
        feature_names_are_synthetic=False,
    )

    with _expect_infeasible_knockoff_plus():
        result = select_fdr(cache=cache, y=np.array([0.0, 1.0, 0.0, 1.0]), verbose=False)

    assert result.selector_metadata["n_zero_weight_variance_features"] == 1
    assert result.W.loc[result.W["feature"] == "zero_weight_only", "W"].iloc[0] == pytest.approx(0.0)


def test_build_active_rxx_skips_copy_when_every_column_is_active(monkeypatch):
    Z = np.array(
        [
            [-1.0, 0.5, 1.0],
            [0.0, -0.5, 0.0],
            [1.0, 1.0, -1.0],
        ],
        dtype=np.float32,
    )
    cache = FeatureCache(
        Z=Z,
        Rxx=None,
        valid_cols=np.arange(3),
        row_idx=np.arange(3),
        sample_weight=np.ones(3, dtype=np.float32),
        n_rows_original=3,
        feature_names=["a", "b", "c"],
        feature_names_are_synthetic=False,
    )
    seen = {}

    def fake_weighted_correlation_matrix(Z_arg, w, *, backend):
        seen["shares_memory"] = np.shares_memory(Z_arg, Z)
        seen["dtype"] = Z_arg.dtype
        return np.eye(Z_arg.shape[1], dtype=np.float64)

    monkeypatch.setattr(
        "sift.selection.knockoff_filter.weighted_correlation_matrix",
        fake_weighted_correlation_matrix,
    )

    R = _build_active_rxx(cache, np.ones(3, dtype=bool), verbose=False)

    assert seen == {"shares_memory": True, "dtype": np.dtype("float32")}
    np.testing.assert_array_equal(R, np.eye(3))


def test_select_fdr_result_metadata_ranking_and_determinism():
    X, y = _signal_frame(n=45, p=70, seed=6)

    with pytest.warns(UserWarning) as first_record:
        first = select_fdr(X, y, q=0.4, offset=0, random_state=22, verbose=False)
    assert any("approximate plug-in" in str(w.message) for w in first_record)
    with pytest.warns(UserWarning) as second_record:
        second = select_fdr(X, y, q=0.4, offset=0, random_state=22, verbose=False)
    assert any("approximate plug-in" in str(w.message) for w in second_record)

    pd.testing.assert_frame_equal(first.W, second.W)
    assert first.selected_features == second.selected_features
    assert first.selector_metadata["selector"] == "knockoff_fdr"
    assert first.selector_metadata["fdr_control"] == "approximate_plugin"
    assert first.selector_metadata["validity_model"] == "gaussian_copula_plugin"
    assert first.selector_metadata["gamma"] > 0.0
    assert first.selector_metadata["weighted_model"] is False
    assert first.threshold is None or np.isfinite(first.threshold) or np.isinf(first.threshold)

    ranking = first.get_feature_ranking()
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
    assert ranking["rank"].tolist() == list(range(1, len(ranking) + 1))


def test_select_fdr_derandomized_frequencies_and_zero_target():
    X, _ = _signal_frame(n=60, p=6, seed=7)
    y = np.ones(len(X))

    result = select_fdr(X, y, n_draws=3, eta=0.75, random_state=0, verbose=False)

    assert result.selector_metadata["tested_state"] == "not_run"
    assert result.selector_metadata["n_tested"] == 0
    assert result.selector_metadata["n_tested_per_draw"] == []
    assert result.threshold is None
    assert result.selection_frequency is not None
    assert result.selection_frequency.eq(0.0).all()
    assert result.selected_features == []
    assert result.diagnostics_["reason"] == "zero_target_variance"
    assert result.selector_metadata["q_scope"] == "per_draw"
    assert result.selector_metadata["per_draw_fdr_control"] == "approximate_plugin"
    assert result.selector_metadata["fdr_control"] == "none"
    assert result.selector_metadata["aggregation_fdr_control"] == "none"
    assert not result.selector_metadata["aggregation_preserves_per_draw_fdr"]
    assert ["W_draw_0", "W_draw_1", "W_draw_2"] == [
        col for col in result.W.columns if col.startswith("W_draw_")
    ]
    assert result.W[["W_draw_0", "W_draw_1", "W_draw_2"]].eq(0.0).all().all()


def test_multi_draw_reuses_invariant_augmented_correlation(monkeypatch):
    import sift.selection.knockoff_filter as knockoff_module

    X, y = _signal_frame(n=100, p=8, seed=107)
    calls = 0
    original = knockoff_module._build_augmented_correlation

    def wrapped(model, kept):
        nonlocal calls
        calls += 1
        return original(model, kept)

    monkeypatch.setattr(knockoff_module, "_build_augmented_correlation", wrapped)
    select_fdr(
        X,
        y,
        q=0.4,
        statistic="ridge",
        screen_pairs=None,
        n_draws=3,
        random_state=7,
        verbose=False,
    )

    assert calls == 1


def test_select_fdr_validation_errors():
    X, y = _signal_frame(n=40, p=5, seed=8)
    cache = build_cache(X, compute_Rxx=True, random_state=0)

    with pytest.raises(ValueError, match="Exactly one"):
        select_fdr(y=y)
    with pytest.raises(ValueError, match="Exactly one"):
        select_fdr(X, y, cache=cache)
    with pytest.raises(ValueError, match="sample_weight"):
        select_fdr(cache=cache, y=y, sample_weight=np.ones(len(y)))
    with pytest.raises(ValueError, match="subsample"):
        select_fdr(cache=cache, y=y, subsample=10)
    with pytest.raises(ValueError, match="subsample"):
        select_fdr(cache=cache, y=y, subsample=50_000)
    with pytest.raises(ValueError, match="rows"):
        select_fdr(X, y[:-1])
    bad_y = y.copy()
    bad_y[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        select_fdr(X, bad_y)
    with pytest.raises(ValueError, match="offset"):
        select_fdr(X, y, offset=0.5)
    with pytest.raises(ValueError, match="n_draws"):
        select_fdr(X, y, n_draws=1.9)
    with pytest.raises(ValueError, match="screen_pairs"):
        select_fdr(X, y, screen_pairs=1.2)
    with pytest.raises(ValueError, match="Duplicate"):
        select_fdr(pd.DataFrame(X.to_numpy(), columns=["a", "a", "b", "c", "d"]), y)

    bad_shape = FeatureCache(
        Z=cache.Z,
        Rxx=np.eye(2, dtype=np.float32),
        valid_cols=cache.valid_cols,
        row_idx=cache.row_idx,
        sample_weight=cache.sample_weight,
        n_rows_original=cache.n_rows_original,
        feature_names=cache.feature_names,
        feature_names_are_synthetic=cache.feature_names_are_synthetic,
    )
    with pytest.raises(ValueError, match="shape"):
        select_fdr(cache=bad_shape, y=y)

    bad_sym = FeatureCache(
        Z=cache.Z,
        Rxx=cache.Rxx.copy(),
        valid_cols=cache.valid_cols,
        row_idx=cache.row_idx,
        sample_weight=cache.sample_weight,
        n_rows_original=cache.n_rows_original,
        feature_names=cache.feature_names,
        feature_names_are_synthetic=cache.feature_names_are_synthetic,
    )
    bad_sym.Rxx[0, 1] = 0.25
    bad_sym.Rxx[1, 0] = -0.25
    with pytest.raises(ValueError, match="symmetric"):
        select_fdr(cache=bad_sym, y=y)


def test_sample_knockoffs_convenience_shape_and_determinism():
    X, _ = _signal_frame(n=45, p=5, seed=9)
    cache = build_cache(X, compute_Rxx=True, random_state=0)

    Zt1 = sample_knockoffs(cache, random_state=123)
    Zt2 = sample_knockoffs(cache, random_state=123)

    assert Zt1.shape == cache.Z.shape
    assert Zt1.dtype == np.float32
    # Same seed, same interpreter: bit-identical. This is the determinism the
    # library actually promises, so it stays an exact comparison.
    np.testing.assert_array_equal(Zt1, Zt2)
    # The pinned draw is only reproducible to float32 precision across builds.
    # ``mean_op``/``noise_chol`` come out of LAPACK (``eigh``/``cho_factor``)
    # and are then applied as float32 BLAS GEMMs, neither of which is bit-stable
    # across NumPy/SciPy versions: numpy 2.5.2 + scipy 1.18.1 reproduces this
    # block to a max relative deviation of 6.4e-8, under one float32 ulp
    # (eps = 1.19e-7). rtol=1e-6 is ~8 ulp -- loose enough for that rounding,
    # far tighter than any real change to the sampler or its RNG stream.
    np.testing.assert_allclose(
        Zt1[:2, :4],
        np.array(
            [
                [-0.38233775, 0.27777314, 0.23124033, 1.3764987],
                [-0.19207978, 0.07742535, -0.44679824, 0.5254327],
            ],
            dtype=np.float32,
        ),
        rtol=1e-6,
        atol=0.0,
    )


def test_select_fdr_warns_once_for_integer_multiclass_target(monkeypatch):
    monkeypatch.setattr(knockoff_filter_module, "_INTEGER_TARGET_WARNING_EMITTED", False)
    rng = np.random.default_rng(151)
    X = rng.normal(size=(80, 5))
    y = np.tile(np.arange(4), 20)

    with pytest.warns(UserWarning, match="multiclass"):
        select_fdr(X, y, q=0.4, offset=0, subsample=None, random_state=0, verbose=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        select_fdr(X, y, q=0.4, offset=0, subsample=None, random_state=1, verbose=False)

    assert not any("multiclass" in str(item.message) for item in caught)


def test_knockoff_selector_fit_transform_support_clone_and_failed_refit_clears_result():
    X, y = _signal_frame(n=55, p=6, seed=10)
    selector = KnockoffSelector(q=0.4, offset=0, random_state=12, verbose=False)
    cloned = clone(selector)

    assert isinstance(cloned, KnockoffSelector)
    selector.fit(X, y)
    transformed = selector.transform(X)
    support = selector.get_support()

    assert hasattr(selector, "result_")
    assert transformed.shape[0] == X.shape[0]
    assert transformed.shape[1] == len(selector.selected_indices_)
    assert support.dtype == bool
    assert support.shape == (X.shape[1],)

    bad = pd.DataFrame(X.to_numpy(), columns=["dup", "dup", "c", "d", "e", "f"])
    with pytest.raises(ValueError, match="Duplicate"):
        selector.fit(bad, y)
    assert not hasattr(selector, "result_")


def test_knockoff_selector_rejects_sample_weight_with_cache():
    X, y = _signal_frame(n=50, p=5, seed=11)
    cache = build_cache(X, compute_Rxx=True, random_state=0)
    weights = np.ones(len(y))

    with pytest.raises(ValueError, match="sample_weight"):
        KnockoffSelector(q=0.4, verbose=False, cache=cache).fit(X, y, sample_weight=weights)
    with pytest.raises(ValueError, match="sample_weight"):
        KnockoffSelector(q=0.4, verbose=False).fit(X, y, sample_weight=weights, cache=cache)


def test_knockoff_selector_cache_subsample_auto_survives_clone_and_pickle():
    X, y = _signal_frame(n=50, p=5, seed=36)
    cache = build_cache(X, compute_Rxx=True, random_state=0)
    selector = KnockoffSelector(q=0.4, offset=0, verbose=False, cache=cache)

    cloned = clone(selector)
    restored = pickle.loads(pickle.dumps(selector))

    assert selector.subsample == "auto"
    assert cloned.subsample == "auto"
    assert restored.subsample == "auto"
    cloned.fit(X, y)
    restored.fit(X, y)
    KnockoffSelector(q=0.4, offset=0, verbose=False, cache=cache).fit(
        X, y, subsample="auto"
    )

    with pytest.raises(ValueError, match="subsample"):
        KnockoffSelector(q=0.4, verbose=False, cache=cache, subsample=50_000).fit(X, y)
    with pytest.raises(ValueError, match="subsample"):
        KnockoffSelector(q=0.4, verbose=False, cache=cache, subsample=None).fit(X, y)


def test_knockoff_selector_cache_path_validates_x_matches_cache_columns():
    X, y = _signal_frame(n=50, p=5, seed=37)
    cache = build_cache(X, compute_Rxx=True, random_state=0)

    KnockoffSelector(q=0.4, offset=0, verbose=False, cache=cache).fit(X, y)
    with pytest.raises(ValueError, match="X columns do not match"):
        KnockoffSelector(q=0.4, verbose=False, cache=cache).fit(X.rename(columns={"f0": "renamed"}), y)
    with pytest.raises(ValueError, match="X columns do not match"):
        KnockoffSelector(q=0.4, verbose=False, cache=cache).fit(X[list(reversed(X.columns))], y)

    X_arr = X.to_numpy()
    synthetic_cache = build_cache(X_arr, compute_Rxx=True, random_state=0)
    KnockoffSelector(q=0.4, offset=0, verbose=False, cache=synthetic_cache).fit(X_arr, y)
    with pytest.raises(ValueError, match="X has 4 columns"):
        KnockoffSelector(q=0.4, verbose=False, cache=synthetic_cache).fit(X_arr[:, :4], y)


def test_knockoff_selector_cache_path_validates_x_and_y_rows():
    X, y = _signal_frame(n=50, p=5, seed=38)
    cache = build_cache(X, compute_Rxx=True, random_state=0)

    with pytest.raises(ValueError, match="X has 51 rows but y has 50 rows"):
        KnockoffSelector(q=0.4, verbose=False, cache=cache).fit(
            pd.concat([X, X.iloc[:1]], ignore_index=True), y
        )

    with pytest.raises(ValueError, match="cache was built with 50 rows but X has 51 rows"):
        KnockoffSelector(q=0.4, verbose=False, cache=cache).fit(
            pd.concat([X, X.iloc[:1]], ignore_index=True), np.r_[y, y[:1]]
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda cache: setattr(cache, "valid_cols", np.array([-1, 0, 1, 2, 3])),
        lambda cache: setattr(cache, "valid_cols", np.array([0, 0, 1, 2, 3])),
        lambda cache: setattr(cache, "valid_cols", np.array([0.0, 1.0, 2.0, 3.0, 4.0])),
        lambda cache: setattr(cache, "valid_cols", np.array([[0, 1, 2, 3, 4]])),
        lambda cache: setattr(cache, "row_idx", np.array([0, 1, 2])),
        lambda cache: setattr(cache, "row_idx", np.arange(50, dtype=np.float64)),
        lambda cache: setattr(cache, "row_idx", np.r_[0, 0, np.arange(2, 50)]),
        lambda cache: setattr(cache, "sample_weight", np.ones((50, 1))),
        lambda cache: setattr(cache, "Rxx", np.eye(4, dtype=np.float32)),
        lambda cache: setattr(cache, "Z", cache.Z.astype(object)),
        lambda cache: setattr(cache, "feature_names_are_synthetic", "false"),
        lambda cache: setattr(cache, "feature_names", "x0x1x2x3x4"),
        lambda cache: setattr(cache, "feature_names", None),
        lambda cache: (
            setattr(cache, "feature_names_are_synthetic", True),
            setattr(cache, "feature_names", ["a", "b", "c", "d", "e"]),
        ),
    ],
)
def test_select_fdr_rejects_malformed_prebuilt_cache(mutate):
    X, y = _signal_frame(n=50, p=5, seed=39)
    cache = build_cache(X, compute_Rxx=True, random_state=0)
    mutate(cache)

    with pytest.raises(ValueError):
        select_fdr(cache=cache, y=y, verbose=False)


@pytest.mark.parametrize("selector", [select_fdr, select_cefsplus])
def test_prebuilt_cache_without_provenance_marker_requires_rebuild(selector):
    X, y = _signal_frame(n=50, p=5, seed=40)
    cache = build_cache(X, compute_Rxx=True, random_state=0)
    delattr(cache, "feature_names_are_synthetic")

    with pytest.raises(ValueError, match="feature_names_are_synthetic|rebuild"):
        if selector is select_fdr:
            selector(cache=cache, y=y, verbose=False)
        else:
            selector(X, y, k=1, cache=cache, verbose=False)


def test_select_cached_without_provenance_marker_requires_rebuild():
    X, y = _signal_frame(n=50, p=5, seed=41)
    cache = build_cache(X, compute_Rxx=True, random_state=0)
    delattr(cache, "feature_names_are_synthetic")

    with pytest.raises(ValueError, match="feature_names_are_synthetic|rebuild"):
        select_cached(cache, y, k=1)


def test_array_cache_without_provenance_marker_requires_rebuild_before_name_checks():
    X, y = _signal_frame(n=50, p=5, seed=42)
    X_arr = X.to_numpy()
    cache = build_cache(X_arr, compute_Rxx=True, random_state=0)
    delattr(cache, "feature_names_are_synthetic")

    with pytest.raises(ValueError, match="feature_names_are_synthetic|rebuild"):
        select_cefsplus(X_arr, y, k=1, cache=cache, verbose=False)
    with pytest.raises(ValueError, match="feature_names_are_synthetic|rebuild"):
        KnockoffSelector(q=0.4, verbose=False, cache=cache).fit(X_arr, y)


@pytest.mark.parametrize(
    "fit_kwargs, message",
    [
        ({"groups": np.arange(40)}, "row groups"),
        ({"time": np.arange(40)}, "time-aware"),
        ({"auto_k_config": object()}, "auto_k_config"),
    ],
)
def test_knockoff_selector_rejects_unsupported_fit_time_arguments(fit_kwargs, message):
    X, y = _signal_frame(n=40, p=5, seed=16)

    with pytest.raises(ValueError, match=message):
        KnockoffSelector(verbose=False).fit(X, y, **fit_kwargs)


def _ar1_design(n: int, p: int, rho: float, k_true: int, amp: float, seed: int):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, p))
    X = np.empty_like(Z)
    X[:, 0] = Z[:, 0]
    for j in range(1, p):
        X[:, j] = rho * X[:, j - 1] + np.sqrt(1.0 - rho**2) * Z[:, j]
    beta = np.zeros(p)
    idx = rng.choice(p, size=k_true, replace=False)
    beta[idx] = amp * rng.choice([-1.0, 1.0], size=k_true) / np.sqrt(n)
    y = X @ beta + rng.normal(size=n)
    truth = np.zeros(p, dtype=bool)
    truth[idx] = True
    return X, y, truth


@pytest.mark.parametrize("statistic", ["lsm", "ridge"])
def test_lsm_and_ridge_statistics_control_fdr_and_have_power(statistic):
    fdps = []
    powers = []
    for seed in range(4):
        X, y, truth = _ar1_design(n=1500, p=120, rho=0.5, k_true=12, amp=6.0, seed=seed)
        result = select_fdr(X, y, q=0.2, statistic=statistic, random_state=seed, verbose=False)
        selected = np.asarray(result.selected_indices, dtype=int)
        fdps.append((~truth[selected]).sum() / max(1, selected.size))
        powers.append(truth[selected].sum() / truth.sum())
        assert result.selector_metadata["statistic"] == statistic
        assert np.isfinite(result.W["W"]).all()
    assert np.mean(fdps) <= 0.3
    assert np.mean(powers) >= 0.4


def test_lsm_beats_marginal_relevance_on_correlated_design():
    X, y, truth = _ar1_design(n=3000, p=200, rho=0.6, k_true=20, amp=7.0, seed=11)
    base = select_fdr(X, y, q=0.1, statistic="relevance", random_state=11, verbose=False)
    lsm = select_fdr(X, y, q=0.1, statistic="lsm", random_state=11, verbose=False)
    base_power = truth[np.asarray(base.selected_indices, dtype=int)].sum()
    lsm_power = truth[np.asarray(lsm.selected_indices, dtype=int)].sum()
    assert lsm_power >= base_power


def test_lsm_and_ridge_options_are_validated():
    X, y = _signal_frame(n=80, p=6, seed=5)
    with pytest.raises(ValueError, match="Unknown statistic_options"):
        select_fdr(X, y, statistic="lsm", statistic_options={"ridge_lambda": 0.1}, verbose=False)
    with pytest.raises(ValueError, match="max_steps"):
        select_fdr(X, y, statistic="lsm", statistic_options={"max_steps": 0}, verbose=False)
    with pytest.raises(ValueError, match="ridge_lambda"):
        select_fdr(X, y, statistic="ridge", statistic_options={"ridge_lambda": 0.0}, verbose=False)
    with _expect_infeasible_knockoff_plus():
        out = select_fdr(X, y, statistic="ridge", statistic_options={"ridge_lambda": 0.25}, verbose=False)
    assert out.selector_metadata["statistic"] == "ridge"
    with _expect_infeasible_knockoff_plus():
        out = select_fdr(X, y, statistic="lsm", statistic_options={"max_steps": 4}, verbose=False)
    assert out.selector_metadata["statistic"] == "lsm"


def test_select_fdr_warns_when_knockoffs_have_no_power_and_reports_s_diagnostics():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(300, 4))
    # Near-duplicate columns force the equicorrelated s towards zero.
    X = np.column_stack([base, base + 1e-3 * rng.normal(size=base.shape)])
    y = base[:, 0] + rng.normal(size=300)
    with pytest.warns(UserWarning) as record:
        result = select_fdr(X, y, q=0.2, verbose=False)
    assert any("very little power" in str(w.message) for w in record)
    assert result.selector_metadata["s_median"] < 0.05
    assert result.selector_metadata["n_low_power_features"] == 8

    X_ok, y_ok = _signal_frame(n=200, p=6, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ok = select_fdr(X_ok, y_ok, q=0.2, verbose=False)
    assert ok.selector_metadata["s_median"] > 0.05
    assert ok.selector_metadata["n_low_power_features"] == 0


def test_cluster_feature_groups_medoids_and_labels():
    from sift.selection.knockoff_filter import cluster_feature_groups

    R = np.array(
        [
            [1.0, 0.95, 0.1, 0.0],
            [0.95, 1.0, 0.1, 0.0],
            [0.1, 0.1, 1.0, 0.05],
            [0.0, 0.0, 0.05, 1.0],
        ]
    )
    labels, reps = cluster_feature_groups(R, corr_threshold=0.7)
    assert labels[0] == labels[1]
    assert len({labels[0], labels[2], labels[3]}) == 3
    assert reps.shape[0] == 3
    assert set(reps.tolist()) >= {2, 3}
    with pytest.raises(ValueError, match="group_corr_threshold"):
        cluster_feature_groups(R, corr_threshold=1.0)


def test_select_fdr_auto_feature_groups_expands_selected_clusters():
    rng = np.random.default_rng(3)
    n = 1500
    n_signal = 8
    signal = rng.normal(size=(n, n_signal))
    # Each signal has two near-copies; plus independent noise columns.
    X = np.column_stack(
        [signal, signal + 0.05 * rng.normal(size=signal.shape), signal + 0.05 * rng.normal(size=signal.shape), rng.normal(size=(n, 12))]
    )
    y = signal @ np.linspace(3.0, 1.5, n_signal) + rng.normal(size=n)
    columns = [f"f{i}" for i in range(X.shape[1])]
    frame = pd.DataFrame(X, columns=columns)

    with pytest.warns(UserWarning) as record:
        plain = select_fdr(frame, y, q=0.2, statistic="lsm", verbose=False)
    assert any("very little power" in str(w.message) for w in record)
    auto = select_fdr(frame, y, q=0.2, statistic="lsm", feature_groups="auto", group_corr_threshold=0.7, verbose=False)

    md = auto.selector_metadata
    assert md["feature_groups"] is True
    assert md["group_mode"] == "cluster_representative"
    assert md["n_feature_groups"] == n_signal + 12
    assert md["discovery_unit"] == "cluster"
    assert md["q_calibration_unit"] == "cluster_representative"
    assert md["representative_fdr_control"] == "approximate_plugin"
    assert md["representative_per_draw_fdr_control"] == "approximate_plugin"
    assert md["group_fdr_control"] == "none"
    assert md["group_per_draw_fdr_control"] == "none"
    assert md["feature_level_fdr_control"] == "none"
    assert md["fdr_control"] == "none"
    assert md["per_draw_fdr_control"] == "none"
    assert md["aggregation_fdr_control"] == "none"
    assert md["aggregation_preserves_per_draw_fdr"] is False
    assert set(auto.W.columns) >= {"feature_group", "is_representative"}
    assert auto.W["is_representative"].sum() == md["n_feature_groups"]
    # Members of a cluster share W and selection status.
    for group_id, block in auto.W.groupby("feature_group"):
        assert block["W"].nunique() == 1
        assert block["selected"].nunique() == 1
    # Selected clusters expand to all members; the near-collinear plain run has no power.
    selected = set(auto.selected_features)
    for j in range(n_signal):
        cluster = {f"f{j}", f"f{j + n_signal}", f"f{j + 2 * n_signal}"}
        assert cluster <= selected or not (cluster & selected)
    assert len(auto.selected_features) >= 3 * 5
    assert not any(int(name[1:]) >= 3 * n_signal for name in auto.selected_features)
    assert len(plain.selected_features) < len(auto.selected_features)
    ranking = auto.get_feature_ranking()
    assert "feature_group" in ranking.columns
    assert auto.diagnostics_["representative_result"].selector_metadata["n_features"] == md["n_feature_groups"]

    with pytest.raises(ValueError, match="feature_groups must be"):
        select_fdr(frame, y, feature_groups="clusters", verbose=False)

    selector = KnockoffSelector(q=0.2, statistic="lsm", feature_groups="auto", group_corr_threshold=0.7, verbose=False).fit(frame, y)
    assert list(selector.selected_features_) == auto.selected_features


def test_select_fdr_preserves_large_offset_float64_target_ordering():
    X, y = _signal_frame(n=200, p=8, seed=205)
    base = select_fdr(X, y, q=0.5, offset=0, random_state=3, verbose=False)
    shifted = select_fdr(
        X,
        y + 1e10,
        q=0.5,
        offset=0,
        random_state=3,
        verbose=False,
    )

    assert np.unique((y + 1e10).astype(np.float32)).size == 1
    assert shifted.diagnostics_.get("reason") != "zero_target_variance"
    assert shifted.selected_features == base.selected_features
    np.testing.assert_allclose(shifted.W["W"], base.W["W"], rtol=0, atol=0)
