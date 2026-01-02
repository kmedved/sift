"""Tests for optimization patches."""

import numpy as np
import pandas as pd
import pytest
import sift
from sift.selection.auto_k import AutoKConfig


@pytest.fixture
def sample_data():
    """Generate sample regression data with time and groups."""
    np.random.seed(42)
    n, p = 1000, 50
    X = pd.DataFrame(np.random.randn(n, p), columns=[f"x{i}" for i in range(p)])
    y = X["x0"] + 0.5 * X["x1"] + 0.3 * X["x2"] + 0.1 * np.random.randn(n)
    groups = np.repeat(np.arange(100), 10)
    time = np.arange(n)
    return X, y.values, groups, time


class TestBLASCorrelation:
    def test_produces_valid_correlation_matrix(self, sample_data):
        from sift.estimators.copula import weighted_correlation_matrix, weighted_rank_gauss_2d

        X, _, _, _ = sample_data
        Z = weighted_rank_gauss_2d(X.values.astype(np.float64), np.ones(len(X)))
        w = np.ones(len(X), dtype=np.float64)

        R = weighted_correlation_matrix(Z, w)

        assert R.shape == (50, 50)
        np.testing.assert_allclose(np.diag(R), 1.0)
        np.testing.assert_allclose(R, R.T)
        assert np.all(np.abs(R) <= 1.0)


class TestJMIIndexed:
    def test_indexed_matches_original(self, sample_data):
        from sift.estimators.joint_mi import r2_joint_mi, r2_joint_mi_indexed

        X, y, _, _ = sample_data
        X_arr = X.values.astype(np.float64)
        y_arr = y.astype(np.float64)
        w = np.ones(len(X), dtype=np.float64)

        selected = X_arr[:, 0]
        cand_indices = np.array([1, 2, 3, 4, 5], dtype=np.int64)

        candidates = X_arr[:, cand_indices]
        scores_orig = r2_joint_mi(selected, candidates, y_arr, w)

        scores_indexed = r2_joint_mi_indexed(X_arr, cand_indices, selected, y_arr, w)

        np.testing.assert_allclose(scores_orig, scores_indexed, rtol=1e-5)


class TestBinnedIndexed:
    def test_binned_indexed_matches_original(self, sample_data):
        from sift.estimators.joint_mi import binned_joint_mi, binned_joint_mi_indexed

        X, y, _, _ = sample_data
        X_arr = X.values.astype(np.float64)
        y_arr = y.astype(np.float64)
        w = np.ones(len(X), dtype=np.float64)

        selected = X_arr[:, 0]
        cand_indices = np.array([1, 2, 3, 4, 5], dtype=np.int64)

        candidates = X_arr[:, cand_indices]
        scores_orig = binned_joint_mi(
            selected,
            candidates,
            y_arr,
            w,
            n_bins=10,
            y_kind="continuous",
        )

        scores_indexed = binned_joint_mi_indexed(
            X_arr,
            cand_indices,
            selected,
            y_arr,
            w,
            n_bins=10,
            y_kind="continuous",
        )

        np.testing.assert_allclose(scores_orig, scores_indexed, rtol=1e-9, atol=1e-12)


class TestAutoK:
    def test_time_holdout(self, sample_data):
        X, y, _, time = sample_data

        features = sift.select_cefsplus(
            X, y, k="auto", time=time, verbose=False
        )

        assert len(features) > 0
        assert len(features) <= 100
        assert any(f in ["x0", "x1", "x2"] for f in features[:10])

    def test_group_cv(self, sample_data):
        X, y, groups, _ = sample_data

        config = AutoKConfig(strategy="group_cv", max_k=50)
        features = sift.select_cefsplus(
            X, y, k="auto", groups=groups, auto_k_config=config, verbose=False
        )

        assert len(features) > 0
        assert len(features) <= 50

    def test_fixed_k_unchanged(self, sample_data):
        X, y, _, _ = sample_data

        features = sift.select_cefsplus(X, y, k=10, verbose=False)

        assert len(features) == 10


class TestCEFSPlusObjective:
    def test_returns_objective_path(self, sample_data):
        X, y, _, _ = sample_data

        cache = sift.build_cache(X, subsample=500)
        features, objective = sift.select_cached(
            cache, y, k=20, method="cefsplus", return_objective=True
        )

        assert len(features) == 20
        assert len(objective) == 20
        assert np.all(np.diff(objective) >= -1e-8)


class TestObjectivePathHelper:
    def test_compute_objective_for_path(self, sample_data):
        from sift.selection.auto_k import compute_objective_for_path

        X, y, _, _ = sample_data
        cache = sift.build_cache(X, subsample=300)
        feature_path = [f"x{i}" for i in range(10)]

        objective = compute_objective_for_path(cache, y, feature_path)

        assert len(objective) == 10
        assert np.isfinite(objective).all()
        assert np.all(np.diff(objective) >= -1e-8)


class TestAutoKElbowGaussian:
    def test_mrmr_gaussian_elbow_no_time_groups(self, sample_data):
        X, y, _, _ = sample_data
        cfg = AutoKConfig(k_method="elbow", max_k=50, min_k=5)
        feats = sift.select_mrmr(
            X,
            y,
            k="auto",
            task="regression",
            estimator="gaussian",
            auto_k_config=cfg,
            verbose=False,
        )
        assert 0 < len(feats) <= 50
        assert any(f in ["x0", "x1", "x2"] for f in feats[:10])

    def test_jmi_gaussian_elbow_no_time_groups(self, sample_data):
        X, y, _, _ = sample_data
        cfg = AutoKConfig(k_method="elbow", max_k=50, min_k=5)
        feats = sift.select_jmi(
            X,
            y,
            k="auto",
            task="regression",
            estimator="gaussian",
            auto_k_config=cfg,
            verbose=False,
        )
        assert 0 < len(feats) <= 50

    def test_jmim_gaussian_elbow_no_time_groups(self, sample_data):
        X, y, _, _ = sample_data
        cfg = AutoKConfig(k_method="elbow", max_k=50, min_k=5)
        feats = sift.select_jmim(
            X,
            y,
            k="auto",
            task="regression",
            estimator="gaussian",
            auto_k_config=cfg,
            verbose=False,
        )
        assert 0 < len(feats) <= 50

    def test_cefsplus_elbow_no_time_groups(self, sample_data):
        X, y, _, _ = sample_data
        cfg = AutoKConfig(k_method="elbow", max_k=50, min_k=5)
        feats = sift.select_cefsplus(
            X,
            y,
            k="auto",
            auto_k_config=cfg,
            verbose=False,
        )
        assert 0 < len(feats) <= 50

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
