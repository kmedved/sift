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
