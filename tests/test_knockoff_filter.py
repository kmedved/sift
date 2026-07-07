from dataclasses import replace
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

import sift.selection.knockoff_filter as knockoff_filter_module
from sift.estimators.copula import FeatureCache, build_cache
from sift.estimators.knockoffs import fit_gaussian_knockoffs, sample_gaussian_knockoffs
from sift.selection.knockoff_filter import (
    _KNOCKOFF_STAT_REGISTRY,
    _build_active_rxx,
    _build_context,
    _cefsplus_incremental_scores,
    _group_knockoff_statistics,
    _SUBSAMPLE_DEFAULT,
    knockoff_threshold,
    sample_knockoffs,
    select_fdr,
)
from sift.selectors import KnockoffSelector


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
    assert result.selector_metadata["feature_groups"] is True
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
    original = select_fdr(cache=cache, y=y, feature_groups=original_groups, verbose=False)
    assert original.W["feature_group"].tolist() == ["a", "b", "c"]

    valid_groups = ["valid_a", "valid_b", "valid_c"]
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

    series_result = select_fdr(cache=cache, y=y_series, random_state=11, verbose=False)
    array_result = select_fdr(cache=cache, y=y, random_state=11, verbose=False)

    pd.testing.assert_frame_equal(series_result.W, array_result.W)
    assert series_result.selected_features == array_result.selected_features


def test_select_fdr_rejects_explicit_subsample_with_cache_even_when_default_value():
    X, y = _signal_frame(n=70, p=6, seed=35)
    cache = build_cache(X, compute_Rxx=True, random_state=0)

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

    with pytest.warns(UserWarning, match="approximate plug-in"):
        first = select_fdr(X, y, q=0.4, offset=0, random_state=22, verbose=False)
    with pytest.warns(UserWarning, match="approximate plug-in"):
        second = select_fdr(X, y, q=0.4, offset=0, random_state=22, verbose=False)

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

    assert result.threshold is None
    assert result.selection_frequency is not None
    assert result.selection_frequency.eq(0.0).all()
    assert result.selected_features == []
    assert result.diagnostics_["reason"] == "zero_target_variance"
    assert ["W_draw_0", "W_draw_1", "W_draw_2"] == [
        col for col in result.W.columns if col.startswith("W_draw_")
    ]
    assert result.W[["W_draw_0", "W_draw_1", "W_draw_2"]].eq(0.0).all().all()


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
    np.testing.assert_array_equal(Zt1, Zt2)
    np.testing.assert_array_equal(
        Zt1[:2, :4],
        np.array(
            [
                [-0.38233775, 0.27777314, 0.23124033, 1.3764987],
                [-0.19207978, 0.07742535, -0.44679824, 0.5254327],
            ],
            dtype=np.float32,
        ),
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


def test_knockoff_selector_cache_subsample_sentinel_survives_clone_and_pickle():
    X, y = _signal_frame(n=50, p=5, seed=36)
    cache = build_cache(X, compute_Rxx=True, random_state=0)
    selector = KnockoffSelector(q=0.4, offset=0, verbose=False, cache=cache)

    cloned = clone(selector)
    restored = pickle.loads(pickle.dumps(selector))

    assert selector.subsample is _SUBSAMPLE_DEFAULT
    assert cloned.subsample is _SUBSAMPLE_DEFAULT
    assert restored.subsample is _SUBSAMPLE_DEFAULT
    cloned.fit(X, y)
    restored.fit(X, y)

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
