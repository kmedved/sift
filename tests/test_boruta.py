"""Tests for Boruta feature selection."""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from sift.boruta import (
    BorutaLoopResult,
    BorutaResult,
    BorutaSelector,
    _compute_auto_n_estimators,
    select_boruta,
    select_boruta_shap,
)
from sift.boruta_helpers import _group_time_holdout_split, _weighted_mean_abs


class TestBorutaBasic:
    """Basic functionality tests."""

    def test_selects_informative_feature(self):
        """Boruta should select the most informative feature."""
        rng = np.random.default_rng(42)
        n, p = 500, 8
        X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
        y = 5.0 * X["f0"] + rng.normal(size=n) * 0.3

        selected = select_boruta(
            X,
            y,
            task="regression",
            max_iter=30,
            alpha=0.1,
            verbose=False,
            random_state=42,
        )

        assert "f0" in selected

    def test_returns_list_of_strings(self):
        """Should return list of feature names."""
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(100, 5)), columns=list("abcde"))
        y = X["a"] + rng.normal(size=100) * 0.1

        selected = select_boruta(X, y, max_iter=10, verbose=False)

        assert isinstance(selected, list)
        assert all(isinstance(f, str) for f in selected)

    def test_numpy_input(self):
        """Should work with numpy arrays."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(100, 5))
        y = X[:, 0] + rng.normal(size=100) * 0.1

        selected = select_boruta(X, y, max_iter=10, verbose=False)

        assert isinstance(selected, list)
        assert all(f.startswith("x") for f in selected)

    def test_numpy_input_rejects_dataframe_only_options(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 4))
        y = X[:, 0] + rng.normal(size=40) * 0.1

        selected = select_boruta(X, y, cat_features=[], max_iter=1, verbose=False)
        assert isinstance(selected, list)

        with pytest.raises(ValueError, match="group_col requires X"):
            select_boruta(X, y, group_col="group", max_iter=1, verbose=False)
        with pytest.raises(ValueError, match="time_col requires X"):
            select_boruta(X, y, time_col="time", max_iter=1, verbose=False)
        with pytest.raises(ValueError, match="cat_features requires X"):
            select_boruta(X, y, cat_features=["cat"], max_iter=1, verbose=False)

    def test_result_ranking_preserves_tied_importance_order(self):
        result = BorutaResult(
            feature_names=["b", "a", "c"],
            status=np.array([1, 1, -1], dtype=np.int8),
            hits=np.array([1, 1, 0], dtype=np.int32),
            n_iter=1,
            shadow_thresholds=np.array([0.0]),
            mean_importance=np.array([1.0, 1.0, 0.0]),
        )

        ranking = result.get_feature_ranking()

        assert ranking["feature"].tolist()[:2] == ["b", "a"]

    def test_classification(self):
        """Should work for classification."""
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(200, 6)), columns=[f"f{i}" for i in range(6)])
        y = (X["f0"] + X["f1"] > 0).astype(int)

        selected = select_boruta(
            X,
            y,
            task="classification",
            max_iter=15,
            verbose=False,
        )

        assert isinstance(selected, list)


class TestBorutaWeights:
    """Sample weight tests."""

    def test_accepts_weights(self):
        """Should accept sample_weight parameter."""
        rng = np.random.default_rng(42)
        n, p = 200, 6
        X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
        y = X["f0"] + rng.normal(size=n) * 0.3
        w = rng.uniform(0.5, 2.0, size=n)

        selected = select_boruta(
            X,
            y,
            sample_weight=w,
            max_iter=10,
            verbose=False,
        )

        assert isinstance(selected, list)

    def test_weight_scaling_invariance(self):
        """Weights scaled by constant should give same results."""
        rng = np.random.default_rng(123)
        n, p = 150, 5
        X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
        y = 2.0 * X["f0"] + rng.normal(size=n) * 0.3
        w = rng.uniform(0.5, 2.0, size=n)

        sel1 = select_boruta(
            X, y, sample_weight=w, max_iter=10, verbose=False, random_state=0
        )
        sel2 = select_boruta(
            X,
            y,
            sample_weight=w * 10,
            max_iter=10,
            verbose=False,
            random_state=0,
        )

        assert sel1 == sel2


class TestBorutaTimeSeries:
    """Time-series shadow permutation tests."""

    def test_auto_selects_circular_shift_with_groups_and_time(self):
        """shadow_method='auto' should use circular_shift with groups+time."""
        rng = np.random.default_rng(1)
        n, p = 200, 5
        X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
        y = X["f0"] + rng.normal(size=n) * 0.3

        groups = np.repeat(np.arange(10), 20)
        time = np.tile(np.arange(20), 10)

        selected = select_boruta(
            X,
            y,
            groups=groups,
            time=time,
            shadow_method="auto",
            max_iter=10,
            verbose=False,
        )

        assert isinstance(selected, list)

    def test_group_col_convenience(self):
        """group_col parameter should extract groups from DataFrame."""
        rng = np.random.default_rng(2)
        n, p = 150, 4
        X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
        X["player_id"] = np.repeat(np.arange(15), 10)
        y = X["f0"] + rng.normal(size=n) * 0.3

        selected = select_boruta(
            X,
            y,
            group_col="player_id",
            shadow_method="within_group",
            max_iter=10,
            verbose=False,
        )

        assert "player_id" not in selected

    def test_within_group_requires_groups(self):
        """shadow_method='within_group' should require groups."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(100, 5))
        y = X[:, 0]

        with pytest.raises(ValueError, match="requires groups"):
            select_boruta(X, y, shadow_method="within_group", max_iter=5, verbose=False)


class TestBorutaResult:
    """BorutaResult and return_result tests."""

    def test_return_result(self):
        """return_result=True should return BorutaResult."""
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(100, 5)), columns=list("abcde"))
        y = X["a"] + rng.normal(size=100) * 0.1

        result = select_boruta(X, y, max_iter=10, verbose=False, return_result=True)

        assert isinstance(result, BorutaResult)
        assert len(result.feature_names) == 5
        assert result.status.shape == (5,)
        assert result.n_iter > 0

    def test_result_selected_features(self):
        """BorutaResult.selected_features() should match accepted."""
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(100, 5)), columns=list("abcde"))
        y = X["a"] + rng.normal(size=100) * 0.1

        result = select_boruta(X, y, max_iter=10, verbose=False, return_result=True)

        selected = result.selected_features()
        accepted_idx = np.where(result.status == 1)[0]
        expected = [result.feature_names[i] for i in accepted_idx]

        assert selected == expected


class TestBorutaSelector:
    """Sklearn-style BorutaSelector tests."""

    def test_fit_transform(self):
        """fit_transform should return reduced feature matrix."""
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(100, 5)), columns=list("abcde"))
        y = X["a"] + rng.normal(size=100) * 0.1

        selector = BorutaSelector(max_iter=10, verbose=False)
        X_transformed = selector.fit_transform(X, y)

        assert X_transformed.shape[1] <= X.shape[1]
        assert len(selector.selected_features_) == X_transformed.shape[1]

    def test_get_support(self):
        """get_support should return boolean mask."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(100, 5))
        y = X[:, 0]

        selector = BorutaSelector(max_iter=10, verbose=False)
        selector.fit(X, y)

        mask = selector.get_support()
        assert mask.shape == (5,)
        assert mask.dtype == bool

        indices = selector.get_support(indices=True)
        assert np.array_equal(np.where(mask)[0], indices)

    def test_transform_reapplies_fitted_categorical_encoder(self, monkeypatch):
        X = pd.DataFrame(
            {
                "cat": pd.Series(["a", "b", "a", "b", "a", "b"], dtype="category"),
                "num": [0.0, 1.0, 0.2, 1.2, 0.1, 1.1],
            }
        )
        y = np.array([0, 1, 0, 1, 0, 1])

        def fake_run(self, fit_data):
            return BorutaLoopResult(
                status=np.array([1, -1]),
                hits=np.array([1, 0]),
                n_trials=1,
                shadow_thresholds=np.array([0.0]),
                mean_importance=np.array([1.0, 0.0]),
            )

        monkeypatch.setattr(BorutaSelector, "_run_boruta_iterations", fake_run)

        selector = BorutaSelector(
            task="classification",
            cat_encoding="loo_logit",
            max_iter=1,
            verbose=False,
        ).fit(X, y)
        transformed = selector.transform(X)

        assert transformed.columns.tolist() == ["cat"]
        assert pd.api.types.is_numeric_dtype(transformed["cat"])
        with pytest.raises(ValueError, match="requires a DataFrame"):
            selector.transform(X.to_numpy())

    def test_failed_refit_clears_previous_fit_state(self, monkeypatch):
        X = pd.DataFrame(np.random.default_rng(0).normal(size=(40, 3)), columns=list("abc"))
        y = X["a"].to_numpy()
        selector = BorutaSelector(max_iter=2, verbose=False).fit(X, y)
        assert selector.selected_features_ is not None

        def fail_run(self, fit_data):
            raise RuntimeError("boom")

        monkeypatch.setattr(BorutaSelector, "_run_boruta_iterations", fail_run)
        with pytest.raises(RuntimeError, match="boom"):
            selector.fit(X, y)
        for attr in (
            "categorical_encoder_",
            "categorical_features_",
            "_categorical_encoding_applied_",
            "selected_features_",
            "status_",
            "feature_names_in_",
        ):
            assert not hasattr(selector, attr)
        with pytest.raises(NotFittedError):
            selector.transform(X)


class TestBorutaOptions:
    """Configuration option tests."""

    def test_auto_n_estimators(self):
        """Auto n_estimators should be bounded + scale sensibly (fast-by-default)."""
        a = _compute_auto_n_estimators(10, 10)
        b = _compute_auto_n_estimators(100, 10)
        c = _compute_auto_n_estimators(100, 5)
        d = _compute_auto_n_estimators(50_000, 5)
        e = _compute_auto_n_estimators(1, 10_000)

        assert 50 <= a <= 500
        assert 50 <= b <= 500
        assert 50 <= c <= 500
        assert 50 <= d <= 500
        assert 50 <= e <= 500

        assert b >= a
        assert c >= b

        assert d == 500
        assert e == 50

    def test_max_features_cap(self):
        """max_features should limit output size."""
        rng = np.random.default_rng(42)
        n, p = 200, 10
        X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
        y = X["f0"] + 0.5 * X["f1"] + 0.3 * X["f2"] + rng.normal(size=n) * 0.1

        selected = select_boruta(
            X,
            y,
            max_features=2,
            max_iter=20,
            alpha=0.3,
            verbose=False,
        )

        assert len(selected) <= 2

    def test_max_features_cap_ties_keep_lowest_feature_index(self):
        selector = BorutaSelector(max_features=2, verbose=False)
        status = np.array([1, 1, 1, -1], dtype=np.int8)
        mean_importance = np.array([0.5, 0.5, 0.5, 0.0], dtype=np.float64)

        resolved = selector._resolve_boruta_final_status(
            status,
            mean_importance,
            np.array([0.0], dtype=np.float64),
        )

        np.testing.assert_array_equal(resolved, np.array([1, 1, -1, -1], dtype=np.int8))

    def test_native_importance_data_test_raises(self):
        """Native importances cannot honestly score held-out data."""
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(200, 5)), columns=list("abcde"))
        y = X["a"] + rng.normal(size=200) * 0.1

        selector = BorutaSelector(
            importance="native",
            importance_data="test",
            test_size=0.3,
            max_iter=5,
            verbose=False,
        )

        with pytest.raises(
            ValueError,
            match="importance_data='test'.*importance='native'.*importance='shap'",
        ):
            selector.fit(X, y)

    def test_native_importance_data_train_still_runs(self):
        """Native importances remain supported on fit data."""
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(200, 5)), columns=list("abcde"))
        y = X["a"] + rng.normal(size=200) * 0.1

        selected = select_boruta(
            X,
            y,
            importance="native",
            importance_data="train",
            max_iter=10,
            verbose=False,
        )

        assert isinstance(selected, list)

    def test_early_stopping(self):
        """Should stop early when no progress."""
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(100, 10)), columns=[f"f{i}" for i in range(10)])
        y = rng.normal(size=100)

        selector = BorutaSelector(
            max_iter=100,
            early_stop_rounds=1,
            alpha=1e-12,
            verbose=False,
        )
        selector.fit(X, y)

        assert selector.n_iter_ < 50

    def test_binomial_rejection_can_reject_zero_hit_features(self, monkeypatch):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(120, 4)), columns=list("abcd"))
        y = rng.normal(size=120)

        def zero_importance(self, est, X, y, w_score, **kwargs):
            del kwargs
            return np.zeros(X.shape[1] * 2, dtype=np.float64)

        monkeypatch.setattr(BorutaSelector, "_compute_importance", zero_importance)
        selector = BorutaSelector(
            max_iter=8,
            alpha=0.5,
            resolve_tentative=False,
            early_stop_rounds=20,
            verbose=False,
        )
        selector.fit(X, y)

        assert np.all(selector.status_ == -1)

    def test_early_stopping_waits_until_binomial_decision_horizon(self, monkeypatch):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(120, 10)), columns=[f"f{i}" for i in range(10)])
        y = rng.normal(size=120)

        def strong_signal_importance(self, est, X, y, w_score, **kwargs):
            del kwargs
            n_active = X.shape[1]
            real = np.zeros(n_active, dtype=np.float64)
            real[: min(2, n_active)] = 2.0
            shadow = np.ones(n_active, dtype=np.float64)
            return np.concatenate([real, shadow])

        monkeypatch.setattr(BorutaSelector, "_compute_importance", strong_signal_importance)
        selector = BorutaSelector(
            max_iter=20,
            resolve_tentative=False,
            verbose=False,
        )
        selector.fit(X, y)

        assert selector.n_iter_ >= 8
        assert selector.selected_features_ == ["f0", "f1"]

    def test_compute_importance_precomputes_shadow_group_info(self, monkeypatch):
        import sift._permute as permute_module
        import sift.boruta as boruta_module

        rng = np.random.default_rng(0)
        X = rng.normal(size=(12, 3))
        y = rng.normal(size=12)
        groups = np.repeat([0, 1, 2], 4)
        time = np.tile(np.arange(4), 3)
        w = np.ones(12, dtype=np.float64)

        real_build_group_info = boruta_module.build_group_info
        calls = {"n": 0}

        def counting_build_group_info(*args, **kwargs):
            calls["n"] += 1
            return real_build_group_info(*args, **kwargs)

        def fail_fallback_build_group_info(*args, **kwargs):
            raise AssertionError("permute_matrix should receive precomputed group_info")

        monkeypatch.setattr(boruta_module, "build_group_info", counting_build_group_info)
        monkeypatch.setattr(permute_module, "build_group_info", fail_fallback_build_group_info)
        monkeypatch.setattr(boruta_module, "_fit_estimator", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            boruta_module,
            "_get_native_importance",
            lambda est: np.ones(X.shape[1] * 2, dtype=np.float64),
        )

        selector = BorutaSelector(importance="native", importance_data="train", verbose=False)
        importance = selector._compute_importance(
            object(),
            X,
            y,
            w,
            w_fit=None,
            groups=groups,
            time=time,
            seed=0,
            shadow_method="circular_shift",
            shadow_mode="columns",
            block_size="auto",
        )

        assert calls["n"] == 1
        assert importance.shape == (X.shape[1] * 2,)

    def test_train_mode_reuses_shadow_group_info_across_iterations(self, monkeypatch):
        import sift._permute as permute_module
        import sift.boruta as boruta_module

        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(12, 3)), columns=list("abc"))
        y = rng.normal(size=12)
        groups = np.repeat([0, 1, 2], 4)
        time = np.tile(np.arange(4), 3)

        real_build_group_info = boruta_module.build_group_info
        calls = {"n": 0}

        def counting_build_group_info(*args, **kwargs):
            calls["n"] += 1
            return real_build_group_info(*args, **kwargs)

        def fail_fallback_build_group_info(*args, **kwargs):
            raise AssertionError("permute_matrix should receive precomputed group_info")

        def fake_fit(est, X_ext, y, w_fit, **kwargs):
            est.n_features_seen_ = X_ext.shape[1]

        monkeypatch.setattr(boruta_module, "build_group_info", counting_build_group_info)
        monkeypatch.setattr(permute_module, "build_group_info", fail_fallback_build_group_info)
        monkeypatch.setattr(boruta_module, "_fit_estimator", fake_fit)
        monkeypatch.setattr(
            boruta_module,
            "_get_native_importance",
            lambda est: np.zeros(est.n_features_seen_, dtype=np.float64),
        )

        selector = BorutaSelector(
            importance="native",
            shadow_method="circular_shift",
            max_iter=3,
            alpha=1e-12,
            early_stop_rounds=10,
            resolve_tentative=False,
            verbose=False,
        )
        selector.fit(X, y, groups=groups, time=time)

        assert selector.n_iter_ == 3
        assert calls["n"] == 1


class TestBorutaShap:
    """Boruta-Shap tests (requires catboost)."""

    def test_weighted_mean_abs_handles_modern_shap_ndarray_shape(self):
        values = np.array(
            [
                [[1.0, -3.0], [2.0, -4.0], [5.0, -7.0]],
                [[2.0, -4.0], [4.0, -6.0], [6.0, -8.0]],
            ]
        )
        weights = np.array([1.0, 3.0])

        out = _weighted_mean_abs(values, weights, n_features=3)

        expected_per_row = np.array([[2.0, 3.0, 6.0], [3.0, 5.0, 7.0]])
        expected = (expected_per_row * weights[:, None]).sum(axis=0) / weights.sum()
        np.testing.assert_allclose(out, expected)

    def test_weighted_mean_abs_handles_catboost_native_shape(self):
        values = np.array(
            [
                [[1.0, 2.0, 5.0], [-3.0, -4.0, -7.0]],
                [[2.0, 4.0, 6.0], [-4.0, -6.0, -8.0]],
            ]
        )
        weights = np.array([1.0, 3.0])

        out = _weighted_mean_abs(values, weights, n_features=3, feature_axis=2)

        expected_per_row = np.array([[2.0, 3.0, 6.0], [3.0, 5.0, 7.0]])
        expected = (expected_per_row * weights[:, None]).sum(axis=0) / weights.sum()
        np.testing.assert_allclose(out, expected)

    def test_shap_backend(self):
        """select_boruta_shap should use SHAP importance."""
        pytest.importorskip("catboost")
        rng = np.random.default_rng(42)
        n, p = 200, 6
        X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
        y = X["f0"] + rng.normal(size=n) * 0.3

        selected = select_boruta_shap(
            X,
            y,
            max_iter=10,
            verbose=False,
        )

        assert isinstance(selected, list)

    def test_shap_importance_data_test(self):
        """SHAP importances can score the held-out split."""
        pytest.importorskip("catboost")
        rng = np.random.default_rng(0)
        n, p = 160, 5
        X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
        y = X["f0"] + rng.normal(size=n) * 0.2

        selected = select_boruta_shap(
            X,
            y,
            importance_data="test",
            test_size=0.25,
            max_iter=5,
            verbose=False,
        )

        assert isinstance(selected, list)

    @pytest.mark.parametrize(
        "y, expected_loss",
        [
            (np.array([0, 1] * 20), "Logloss"),
            (np.array([0, 1, 2] * 14 + [0, 1]), "MultiClass"),
        ],
    )
    def test_default_classification_loss_matches_class_count(
        self, monkeypatch, y, expected_loss
    ):
        """Default SHAP classification estimator should pick the right loss."""
        pytest.importorskip("catboost")
        rng = np.random.default_rng(123)
        X = pd.DataFrame(
            rng.normal(size=(y.shape[0], 4)), columns=[f"f{i}" for i in range(4)]
        )
        captured = {}

        def fake_compute_importance(self, est, X, y, w_score, **kwargs):
            del kwargs
            captured["estimator"] = est
            return np.zeros(X.shape[1] * 2, dtype=np.float64)

        monkeypatch.setattr(BorutaSelector, "_compute_importance", fake_compute_importance)

        selector = BorutaSelector(
            task="classification",
            importance="shap",
            max_iter=1,
            early_stop_rounds=1,
            verbose=False,
        )
        selector.fit(X, y)

        est = captured["estimator"]
        loss = None
        allow_writing_files = None
        if hasattr(est, "get_params"):
            params = est.get_params(deep=False)
            loss = params.get("loss_function")
            allow_writing_files = params.get("allow_writing_files")
        if loss is None and hasattr(est, "get_all_params"):
            params = est.get_all_params()
            loss = params.get("loss_function")
            allow_writing_files = params.get("allow_writing_files")

        assert loss == expected_loss
        assert allow_writing_files is False


class TestBorutaValidation:
    """Runtime validation for enum-like options."""

    @pytest.mark.parametrize(
        "kwargs, pattern",
        [
            ({"task": "bogus"}, r"task must be one of .*'regression'.*'classification'"),
            (
                {"importance": "bogus"},
                r"importance must be one of .*'native'.*'shap'",
            ),
            (
                {"importance_data": "bogus"},
                r"importance_data must be one of .*'train'.*'test'",
            ),
            (
                {"shadow_method": "bogus"},
                r"shadow_method must be one of .*'auto'.*'global'.*'within_group'.*'block'.*'circular_shift'",
            ),
            (
                {"shadow_mode": "bogus"},
                r"shadow_mode must be one of .*'columns'.*'rows'",
            ),
            (
                {"block_size": "bogus"},
                r"block_size must be a positive integer or 'auto'",
            ),
            (
                {"block_size": 0},
                r"block_size must be a positive integer or 'auto'",
            ),
        ],
    )
    def test_invalid_options_raise_clear_value_error(self, kwargs, pattern):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(40, 4)), columns=list("abcd"))
        y = X["a"] + rng.normal(size=40) * 0.1

        selector = BorutaSelector(max_iter=1, verbose=False, **kwargs)

        with pytest.raises(ValueError, match=pattern):
            selector.fit(X, y)


class TestBorutaNanHandling:
    """NaN handling tests."""

    def test_group_time_holdout_keeps_nan_group_rows(self):
        groups = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, np.nan, np.nan, np.nan])
        time = np.arange(len(groups))

        train_idx, test_idx = _group_time_holdout_split(groups, time, test_size=0.34)

        covered = set(train_idx.tolist()) | set(test_idx.tolist())
        assert covered == set(range(len(groups)))
        assert set(train_idx).isdisjoint(set(test_idx))

    def test_handles_nan_values(self):
        """Should impute NaN values without error."""
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(100, 5)), columns=list("abcde"))
        X.iloc[0, 0] = np.nan
        X.iloc[5, 2] = np.nan
        y = X["a"].fillna(0) + rng.normal(size=100) * 0.1

        selected = select_boruta(X, y, max_iter=10, verbose=False)

        assert isinstance(selected, list)


class TestBorutaCategoricals:
    """Categorical encoding tests."""

    @pytest.mark.parametrize("cat_encoding", ["target", "loo", "james_stein", "loo_logit"])
    def test_importance_data_test_rejects_supervised_cat_encoding(self, cat_encoding):
        rng = np.random.default_rng(0)
        n = 60
        cat = pd.Series(rng.choice(["a", "b", "c"], size=n), dtype="category")
        X = pd.DataFrame({"cat": cat, "num": rng.normal(size=n)})
        y = (X["num"] > 0).astype(int).to_numpy()

        with pytest.raises(ValueError, match="importance_data='test'.*cat_encoding"):
            select_boruta_shap(
                X,
                y,
                task="classification",
                importance_data="test",
                cat_features=["cat"],
                cat_encoding=cat_encoding,
                max_iter=1,
                verbose=False,
            )

    def test_cat_encoding_runs(self):
        """Categorical columns should be encodable for Boruta."""
        pytest.importorskip("category_encoders")
        rng = np.random.default_rng(0)
        n = 200
        X = pd.DataFrame(
            {
                "num": rng.normal(size=n),
                "cat": pd.Series(rng.choice(["a", "b", "c"], size=n), dtype="category"),
            }
        )
        y = X["num"] + rng.normal(size=n) * 0.1

        selected = select_boruta(
            X,
            y,
            cat_encoding="loo",
            max_iter=10,
            verbose=False,
        )

        assert isinstance(selected, list)


class TestPermutationCorrectness:
    """Tests for permutation utility correctness."""

    def test_block_permute_is_valid_permutation(self):
        """Block permute should produce a permutation of original values."""
        from sift._permute import build_group_info, permute_array

        rng = np.random.default_rng(42)
        groups = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        time = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        x = np.arange(10, dtype=np.float64)

        group_info = build_group_info(groups, time)
        permuted = permute_array(
            x, method="block", group_info=group_info, block_size=2, rng=rng
        )

        assert sorted(permuted[:5].tolist()) == sorted(x[:5].tolist())
        assert sorted(permuted[5:].tolist()) == sorted(x[5:].tolist())


class TestBorutaTransform:
    """Tests for transform correctness."""

    def test_transform_selects_by_name_not_position(self):
        """transform() should select columns by name, not position."""
        rng = np.random.default_rng(42)
        n = 200

        X_fit = pd.DataFrame(
            {
                "a": rng.normal(size=n),
                "b": rng.normal(size=n),
                "c": rng.normal(size=n),
            }
        )
        y = 3.0 * X_fit["a"] + rng.normal(size=n) * 0.1

        selector = BorutaSelector(max_iter=15, verbose=False, random_state=42)
        selector.fit(X_fit, y)

        X_transform = pd.DataFrame(
            {
                "extra": rng.normal(size=n),
                "a": X_fit["a"],
                "b": X_fit["b"],
                "c": X_fit["c"],
            }
        )

        result = selector.transform(X_transform)

        assert isinstance(result, pd.DataFrame)
        for col in result.columns:
            assert col in selector.feature_names_in_
            assert col != "extra"


class TestShadowModeRows:
    """Tests for row-wise shadow permutation."""

    def test_shadow_mode_rows_runs(self):
        """shadow_mode='rows' should run without error."""
        rng = np.random.default_rng(0)
        n, p = 200, 6
        X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
        y = X["f0"] + rng.normal(size=n) * 0.2
        groups = np.repeat(np.arange(10), 20)
        time = np.tile(np.arange(20), 10)

        selected = select_boruta(
            X,
            y,
            groups=groups,
            time=time,
            shadow_method="circular_shift",
            shadow_mode="rows",
            max_iter=10,
            verbose=False,
        )
        assert isinstance(selected, list)
