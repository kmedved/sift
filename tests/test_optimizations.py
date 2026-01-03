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


@pytest.fixture
def classification_data():
    """Generate sample classification data."""
    np.random.seed(42)
    n, p = 1000, 50
    X = pd.DataFrame(np.random.randn(n, p), columns=[f"x{i}" for i in range(p)])
    logits = X["x0"] + 0.5 * X["x1"] + 0.3 * X["x2"]
    y = (logits > 0).astype(int).values
    groups = np.repeat(np.arange(100), 10)
    time = np.arange(n)
    return X, y, groups, time


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


class TestClassificationAutoK:
    def test_time_holdout_logloss(self, classification_data):
        from sift.selection.auto_k import AutoKConfig, select_k_auto

        X, y, _, time = classification_data
        config = AutoKConfig(strategy="time_holdout", max_k=30, metric="logloss")
        feature_path = [f"x{i}" for i in range(30)]

        best_k, selected, diag = select_k_auto(
            X, y, feature_path, config, time=time, task="classification"
        )

        assert 5 <= best_k <= 30
        assert len(selected) == best_k
        assert (diag["score"][diag["score"] < np.inf] >= 0).all()

    def test_group_cv_error(self, classification_data):
        from sift.selection.auto_k import AutoKConfig, select_k_auto

        X, y, groups, _ = classification_data
        config = AutoKConfig(strategy="group_cv", max_k=20, metric="error")
        feature_path = [f"x{i}" for i in range(20)]

        best_k, selected, diag = select_k_auto(
            X, y, feature_path, config, groups=groups, task="classification"
        )

        assert best_k > 0
        valid_scores = diag["score"][diag["score"] < np.inf]
        assert (valid_scores >= 0).all() and (valid_scores <= 1).all()

    def test_auto_metric_defaults_to_logloss(self, classification_data):
        from sift.selection.auto_k import AutoKConfig, select_k_auto

        X, y, _, time = classification_data
        config = AutoKConfig(strategy="time_holdout", max_k=20, metric="auto")
        feature_path = [f"x{i}" for i in range(20)]

        best_k, _, _ = select_k_auto(
            X, y, feature_path, config, time=time, task="classification"
        )

        assert best_k > 0

    def test_invalid_metric_raises(self, classification_data):
        from sift.selection.auto_k import AutoKConfig, select_k_auto

        X, y, _, time = classification_data
        config = AutoKConfig(strategy="time_holdout", metric="rmse")
        feature_path = [f"x{i}" for i in range(20)]

        with pytest.raises(ValueError, match="invalid for.*classification"):
            select_k_auto(X, y, feature_path, config, time=time, task="classification")

    def test_logloss_handles_single_class_val_fold(self):
        """Ensure logloss doesn't fail when validation fold has one class."""
        from sift.selection.auto_k import AutoKConfig, select_k_auto

        np.random.seed(42)
        n, p = 100, 20
        X = pd.DataFrame(np.random.randn(n, p), columns=[f"x{i}" for i in range(p)])

        y = np.array([0] * 40 + [1] * 40 + [1] * 20)
        time = np.arange(n)

        config = AutoKConfig(strategy="time_holdout", max_k=15, metric="logloss", val_frac=0.2)
        feature_path = [f"x{i}" for i in range(15)]

        best_k, selected, diag = select_k_auto(
            X, y, feature_path, config, time=time, task="classification"
        )

        assert best_k > 0
        assert len(selected) == best_k
        assert (diag["score"] < np.inf).any()

    def test_mean_impute_preserves_float32(self):
        from sift._impute import mean_impute

        X = np.array([[1.0, np.nan]], dtype=np.float32)
        X_imp = mean_impute(X, copy=True)
        assert X_imp.dtype == np.float32

    def test_validate_inputs_does_not_mutate(self):
        from sift._preprocess import validate_inputs

        X = np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float64)
        X_orig = X.copy()
        validate_inputs(X, np.array([0, 1]), task="regression", impute=True)
        np.testing.assert_array_equal(X, X_orig)


class TestInputValidation:
    def test_groups_length_mismatch(self, sample_data):
        X, y, _, _ = sample_data
        wrong_groups = np.arange(500)

        with pytest.raises(ValueError, match="groups has .* elements but X has .* rows"):
            sift.select_cefsplus(X, y, k="auto", groups=wrong_groups)

    def test_time_length_mismatch(self, sample_data):
        X, y, _, _ = sample_data
        wrong_time = np.arange(500)

        with pytest.raises(ValueError, match="time has .* elements but X has .* rows"):
            sift.select_cefsplus(X, y, k="auto", time=wrong_time)


class TestObjectiveModule:
    def test_objective_from_corr_path(self, sample_data):
        from sift.selection.objective import objective_from_corr_path

        rng = np.random.default_rng(0)
        n, k = 200, 10
        X = rng.standard_normal((n, k))
        y = rng.standard_normal(n)

        R = np.corrcoef(X, rowvar=False)
        r = np.corrcoef(X, y, rowvar=False)[-1, :-1]

        obj = objective_from_corr_path(R, r)

        assert len(obj) == k
        assert np.isfinite(obj).all()
        assert np.all(np.diff(obj) >= -1e-8)


class TestCentralizedUtilities:
    def test_ensure_weights_validation(self):
        from sift._preprocess import ensure_weights

        w = ensure_weights(None, 100)
        assert len(w) == 100
        np.testing.assert_allclose(w.mean(), 1.0)

        with pytest.raises(ValueError, match="negative"):
            ensure_weights(np.array([-1, 1, 1]), 3)

        with pytest.raises(ValueError, match="expected"):
            ensure_weights(np.ones(5), 10)

    def test_mean_impute(self):
        from sift._impute import mean_impute

        X = np.array([[1, 2], [np.nan, 4], [3, np.inf]])
        X_imp = mean_impute(X)

        assert np.isfinite(X_imp).all()
        assert X_imp[1, 0] == 2.0
        assert X_imp[2, 1] == 3.0

    def test_build_cache_does_not_mutate_input(self):
        from sift.estimators.copula import build_cache

        X = np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float64)
        X_orig = X.copy()
        _ = build_cache(X, subsample=None)
        np.testing.assert_array_equal(X, X_orig)

    def test_ensure_weights_overflow_safe(self):
        from sift._preprocess import ensure_weights

        w = np.array([1e308, 1e308], dtype=np.float64)
        wn = ensure_weights(w, 2, normalize=True)
        assert np.isfinite(wn).all()
        np.testing.assert_allclose(wn.mean(), 1.0, rtol=0, atol=1e-12)

    def test_ensure_weights_underflow_safe(self):
        from sift._preprocess import ensure_weights

        w = np.array([np.nextafter(0, 1), 0.0], dtype=np.float64)
        wn = ensure_weights(w, 2, normalize=True)
        np.testing.assert_allclose(wn.mean(), 1.0, rtol=0, atol=1e-12)


class TestSelectKAutoOptimized:
    def test_time_holdout_returns_valid_k(self, sample_data):
        from sift.selection.auto_k import AutoKConfig, select_k_auto

        X, y, _, time = sample_data
        config = AutoKConfig(strategy="time_holdout", max_k=30, min_k=5)
        feature_path = [f"x{i}" for i in range(30)]

        best_k, selected, diag = select_k_auto(X, y, feature_path, config, time=time)

        assert 5 <= best_k <= 30
        assert len(selected) == best_k
        assert len(diag) > 0
        assert diag["score"].notna().any()

    def test_group_cv_caps_splits(self, sample_data):
        from sift.selection.auto_k import AutoKConfig, select_k_auto

        X, y, _, _ = sample_data
        groups = np.repeat([0, 1, 2], len(X) // 3 + 1)[: len(X)]
        config = AutoKConfig(strategy="group_cv", n_splits=5, max_k=20)
        feature_path = [f"x{i}" for i in range(20)]

        best_k, selected, diag = select_k_auto(X, y, feature_path, config, groups=groups)

        assert best_k > 0
        assert len(selected) == best_k

    def test_group_cv_fails_with_one_group(self, sample_data):
        from sift.selection.auto_k import AutoKConfig, select_k_auto

        X, y, _, _ = sample_data
        groups = np.zeros(len(X), dtype=int)
        config = AutoKConfig(strategy="group_cv", max_k=20)
        feature_path = [f"x{i}" for i in range(20)]

        with pytest.raises(ValueError, match="at least 2 groups"):
            select_k_auto(X, y, feature_path, config, groups=groups)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
