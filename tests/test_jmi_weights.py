import numpy as np
import pytest
from sift import select_jmi, select_jmim
from sift.estimators.joint_mi import (
    binned_joint_mi,
    binned_joint_mi_indexed,
    binned_joint_mi_indexed_prebinned,
    quantile_bin_matrix,
    _quantile_bin,
    _weighted_percentile,
)


def test_jmi_classic_accepts_weights():
    rng = np.random.default_rng(42)
    n, p = 100, 8
    X = rng.normal(size=(n, p))
    y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(size=n) * 0.1
    w = rng.uniform(0.5, 2.0, size=n)

    # r2 estimator
    selected = select_jmi(
        X,
        y,
        k=3,
        task="regression",
        sample_weight=w,
        estimator="r2",
        verbose=False,
    )
    assert len(selected) == 3
    assert "x0" in selected  # should pick the most predictive feature

    # binned estimator
    selected_binned = select_jmi(
        X,
        y,
        k=3,
        task="regression",
        sample_weight=w,
        estimator="binned",
        verbose=False,
    )
    assert len(selected_binned) == 3


def test_jmim_classic_accepts_weights():
    rng = np.random.default_rng(42)
    n, p = 100, 8
    X = rng.normal(size=(n, p))
    y = X[:, 0] + rng.normal(size=n) * 0.1
    w = rng.uniform(0.5, 2.0, size=n)

    selected = select_jmim(
        X,
        y,
        k=3,
        task="regression",
        sample_weight=w,
        estimator="r2",
        verbose=False,
    )
    assert len(selected) == 3


def test_weight_scaling_invariance_jmi():
    """Weights scaled by constant should give same results."""
    rng = np.random.default_rng(123)
    n, p = 80, 6
    X = rng.normal(size=(n, p))
    y = X[:, 0] * 2 + rng.normal(size=n) * 0.1
    w = rng.uniform(0.5, 2.0, size=n)

    sel1 = select_jmi(X, y, k=3, task="regression", sample_weight=w, estimator="r2", verbose=False)
    sel2 = select_jmi(
        X,
        y,
        k=3,
        task="regression",
        sample_weight=w * 10,
        estimator="r2",
        verbose=False,
    )
    sel3 = select_jmi(
        X,
        y,
        k=3,
        task="regression",
        sample_weight=w / w.sum(),
        estimator="r2",
        verbose=False,
    )

    assert sel1 == sel2 == sel3


def test_ksg_jmi_allows_unweighted_selection():
    rng = np.random.default_rng(321)
    n, p = 35, 5
    X = rng.normal(size=(n, p))
    y = X[:, 0] + rng.normal(size=n) * 0.1

    selected = select_jmi(
        X,
        y,
        k=2,
        task="regression",
        estimator="ksg",
        top_m=4,
        verbose=False,
    )

    assert len(selected) <= 2


@pytest.mark.parametrize("selector", [select_jmi, select_jmim])
def test_ksg_public_selectors_reject_sample_weight(selector):
    rng = np.random.default_rng(654)
    n, p = 30, 5
    X = rng.normal(size=(n, p))
    y = X[:, 0] + rng.normal(size=n) * 0.1

    with pytest.raises(ValueError, match="ksg.*sample_weight"):
        selector(
            X,
            y,
            k=2,
            task="regression",
            estimator="ksg",
            sample_weight=np.ones(n),
            verbose=False,
        )


def test_ksg_low_level_rejects_sample_weight():
    from sift.selection.loops import jmi_select

    rng = np.random.default_rng(654)
    n, p = 30, 5
    X = rng.normal(size=(n, p)).astype(np.float32)
    y = X[:, 0] + rng.normal(size=n).astype(np.float32) * 0.1
    relevance = np.linspace(1.0, 0.2, p)

    with pytest.raises(ValueError, match="ksg.*sample_weight"):
        jmi_select(
            X,
            y,
            k=2,
            relevance=relevance,
            mi_estimator="ksg",
            sample_weight=np.ones(n),
        )


def test_binned_jmi_integer_weights_match_literal_replication_with_ties():
    """Weighted edges and entropy agree with expanding integer row counts."""
    selected = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 3.0])
    candidates = np.column_stack(
        [selected, [1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0], np.ones(7)]
    )
    y = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 3.0])
    weights = np.array([1, 2, 0, 3, 1, 2, 1], dtype=np.int64)
    row_idx = np.repeat(np.arange(weights.size), weights)

    weighted = binned_joint_mi(selected, candidates, y, weights, n_bins=3)
    replicated = binned_joint_mi(
        selected[row_idx], candidates[row_idx], y[row_idx],
        np.ones(row_idx.size), n_bins=3,
    )
    np.testing.assert_allclose(weighted, replicated, atol=1e-12, rtol=1e-12)


def test_weighted_quantile_edges_ignore_zero_weight_rows_and_degenerate_values():
    values = np.array([0.0, 1.0, 2.0, 1e12])
    weights = np.array([1.0, 1.0, 1.0, 0.0])
    percentiles = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
    expected = np.percentile(values[:3], percentiles)
    np.testing.assert_allclose(_weighted_percentile(values, percentiles, weights), expected)
    np.testing.assert_array_equal(
        _quantile_bin(values, 3, weights)[weights > 0],
        _quantile_bin(values[:3], 3),
    )
    np.testing.assert_array_equal(
        _quantile_bin(np.array([1.0, 1.0, 999.0]), 4, np.array([2.0, 0.0, 0.0])),
        np.zeros(3, dtype=np.int32),
    )


def test_weighted_percentile_is_scale_invariant_and_keeps_tiny_positive_support():
    values = np.array([0.0, 10.0, 20.0, 30.0])
    percentiles = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
    weights = np.array([2.0, 3.0, 5.0, 9.0])
    np.testing.assert_allclose(
        _weighted_percentile(values, percentiles, weights),
        _weighted_percentile(values, percentiles, 0.5 * weights),
        atol=1e-12,
        rtol=1e-12,
    )
    tiny = _weighted_percentile(
        np.array([0.0, 10.0, 20.0]),
        percentiles,
        np.array([1e-13, 1.0, 1.0]),
    )
    assert tiny[0] == 0.0
    assert tiny[-1] == 20.0


def test_weighted_percentile_is_scale_invariant_with_extreme_dynamic_range():
    values = np.array(
        [
            0.64028182,
            -0.61251719,
            -0.21743198,
            -1.09199879,
            -0.53796814,
            2.22423771,
            0.84660054,
            0.44437179,
        ]
    )
    weights = np.array(
        [
            1.23839336e5,
            4.50646384e2,
            2.87389989e10,
            1.02230114e6,
            6.95627844,
            3.33858850e-6,
            4.42515896,
            3.55075782e5,
        ]
    )
    percentiles = np.linspace(0.0, 100.0, 21)

    expected = _weighted_percentile(values, percentiles, weights)
    actual = _weighted_percentile(values, percentiles, 1e-9 * weights)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    assert actual[0] == np.min(values)
    assert actual[-1] == np.max(values)


def test_weighted_percentile_avoids_integer_rank_overflow():
    values = np.array([0.0, 1.0])
    percentiles = np.array([0.0, 50.0, 100.0])
    weights = np.array([float(2**63), 1.0])

    result = _weighted_percentile(values, percentiles, weights)

    assert np.isfinite(result).all()
    np.testing.assert_array_equal(result, np.array([0.0, 0.0, 1.0]))


def test_weighted_percentile_normalizes_common_integer_frequency_factor():
    values = np.array([-2.0, -4.0])
    percentiles = np.array([0.0, 25.0, 50.0, 75.0, 76.0, 90.0, 100.0])

    primitive = _weighted_percentile(values, percentiles, np.array([1.0, 4.0]))
    scaled = _weighted_percentile(values, percentiles, np.array([3.0, 12.0]))
    np.testing.assert_allclose(scaled, primitive, atol=1e-12, rtol=1e-12)


def test_weighted_percentile_is_tie_safe_and_preserves_query_order():
    values = np.array([2.0, 2.0, -2.0])
    weights = np.array([1.0, 6.0, 6.0])
    percentiles = np.array([75.0, 0.0, 50.0, 100.0, 25.0])
    result = _weighted_percentile(values, percentiles, weights)
    expected = np.percentile(
        np.repeat(values, weights.astype(np.int64)), percentiles,
    )
    np.testing.assert_allclose(result, expected, atol=1e-12, rtol=1e-12)
    ordered = _weighted_percentile(values, np.sort(percentiles), weights)
    assert np.all(np.diff(ordered) >= 0.0)


def test_low_level_binned_jmi_scores_are_scale_invariant():
    rng = np.random.default_rng(91)
    X = rng.normal(size=(40, 6))
    selected = X[:, 0]
    y = rng.normal(size=40)
    weights = 10.0 ** rng.uniform(-10.0, 10.0, size=40)
    cand_idx = np.array([1, 3, 5])

    direct = binned_joint_mi(selected, X[:, cand_idx], y, weights)
    direct_scaled = binned_joint_mi(
        selected, X[:, cand_idx], y, 1e-9 * weights,
    )
    indexed = binned_joint_mi_indexed(
        X, cand_idx, selected, y, weights,
    )
    indexed_scaled = binned_joint_mi_indexed(
        X, cand_idx, selected, y, 1e-9 * weights,
    )
    X_binned = quantile_bin_matrix(X, 10, weights=weights)
    s_binned = _quantile_bin(selected, 10, weights=weights)
    y_binned = _quantile_bin(y, 10, weights=weights)
    prebinned = binned_joint_mi_indexed_prebinned(
        X_binned, cand_idx, s_binned, y_binned, weights,
        n_bins=10, n_y_bins=10,
    )
    prebinned_scaled = binned_joint_mi_indexed_prebinned(
        X_binned, cand_idx, s_binned, y_binned, 1e-9 * weights,
        n_bins=10, n_y_bins=10,
    )

    for expected, actual in (
        (direct, direct_scaled),
        (indexed, indexed_scaled),
        (prebinned, prebinned_scaled),
    ):
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0.0)


@pytest.mark.parametrize("selector", [select_jmi, select_jmim])
def test_public_binned_jmi_integer_weights_match_row_replication(selector):
    """The optimized public pre-binning path must consume weighted edges too."""
    rng = np.random.default_rng(12)
    X = rng.integers(0, 8, size=(60, 6)).astype(np.float64)
    y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(size=len(X))
    weights = rng.integers(0, 4, size=len(X))
    weights[0] = 3
    row_idx = np.repeat(np.arange(len(X)), weights)

    weighted = selector(
        X,
        y,
        k=4,
        task="regression",
        estimator="binned",
        sample_weight=weights,
        subsample=None,
        verbose=False,
    )
    replicated = selector(
        X[row_idx],
        y[row_idx],
        k=4,
        task="regression",
        estimator="binned",
        subsample=None,
        verbose=False,
    )
    assert weighted == replicated


@pytest.mark.parametrize("selector", [select_jmi, select_jmim])
def test_public_binned_jmi_row_replication_adversarial_tie(selector):
    X = np.array(
        [[2, 4, 3], [5, 5, 2], [5, 3, 1], [0, 2, 4],
         [6, 7, 6], [3, 6, 0], [6, 1, 3], [2, 1, 4]],
        dtype=np.float64,
    )
    y = np.array([6, 7, 4, 6, 6, 7, 2, 1], dtype=np.float64)
    weights = np.array([1, 2, 1, 0, 3, 3, 2, 0])
    row_idx = np.repeat(np.arange(len(X)), weights)

    weighted = selector(
        X, y, k=3, task="regression", estimator="binned",
        sample_weight=weights, subsample=None, verbose=False,
    )
    replicated = selector(
        X[row_idx], y[row_idx], k=3, task="regression", estimator="binned",
        subsample=None, verbose=False,
    )
    assert weighted == replicated


@pytest.mark.parametrize("selector", [select_jmi, select_jmim])
def test_public_binned_jmi_global_rescaling_is_selection_invariant(selector):
    """Entropy accumulation must not let a global weight scale change ties."""
    rng = np.random.default_rng(55)
    for _ in range(11):
        X = rng.normal(size=(10, 7))
        y = rng.normal(size=10)
        weights = 10.0 ** rng.uniform(-8.0, 8.0, size=10)

    unscaled = selector(
        X, y, k=5, task="regression", estimator="binned",
        sample_weight=weights, subsample=None, verbose=False,
    )
    rescaled = selector(
        X, y, k=5, task="regression", estimator="binned",
        sample_weight=1e-9 * weights, subsample=None, verbose=False,
    )
    assert unscaled == rescaled
