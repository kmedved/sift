import numpy as np

from sift.estimators import joint_mi as jmi


def test_binned_jmi_prebin_equivalence():
    rng = np.random.default_rng(0)
    n, p = 200, 20
    X_full = rng.normal(size=(n, p)).astype(np.float64)
    selected = X_full[:, 0]
    y = rng.normal(size=n).astype(np.float64)
    w = np.ones(n, dtype=np.float64)

    cand_idx = rng.choice(np.arange(p), size=8, replace=False)

    scores_old = jmi.binned_joint_mi_indexed(
        X_full,
        cand_idx,
        selected,
        y,
        w,
        n_bins=10,
        y_kind="continuous",
    )

    X_binned = jmi.quantile_bin_matrix(X_full, 10)
    s_binned = jmi._quantile_bin(selected, 10)
    y_binned = jmi._quantile_bin(y, 10)

    scores_new = jmi.binned_joint_mi_indexed_prebinned(
        X_binned,
        cand_idx,
        s_binned,
        y_binned,
        w,
        n_bins=10,
        n_y_bins=10,
    )

    np.testing.assert_allclose(scores_old, scores_new, atol=1e-8, rtol=1e-6)


def test_quantile_bin_matrix_indexed_matches_full():
    rng = np.random.default_rng(1)
    n, p = 128, 12
    X_full = rng.normal(size=(n, p)).astype(np.float64)
    cand_idx = np.array([0, 3, 5, 9])

    full = jmi.quantile_bin_matrix(X_full, 7)[:, cand_idx]
    indexed = jmi.quantile_bin_matrix_indexed(X_full, cand_idx, 7)

    np.testing.assert_array_equal(indexed, full)


def test_binned_jmi_discrete_sparse_integer_labels_are_compacted():
    rng = np.random.default_rng(2)
    n = 120
    selected = rng.normal(size=n)
    candidates = rng.normal(size=(n, 3))
    y_sparse = np.where(np.arange(n) % 3 == 0, 100, np.where(np.arange(n) % 3 == 1, 5, 0))
    y_compact = np.unique(y_sparse, return_inverse=True)[1]
    w = np.ones(n, dtype=np.float64)

    sparse_scores = jmi.binned_joint_mi(
        selected,
        candidates,
        y_sparse,
        w,
        n_bins=5,
        y_kind="discrete",
    )
    compact_scores = jmi.binned_joint_mi(
        selected,
        candidates,
        y_compact,
        w,
        n_bins=5,
        y_kind="discrete",
    )

    np.testing.assert_allclose(sparse_scores, compact_scores)
