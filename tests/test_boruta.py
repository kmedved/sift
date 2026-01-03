"""Tests for Boruta feature selection."""

import numpy as np
import pandas as pd
import pytest

from sift.boruta import (
    BorutaResult,
    BorutaSelector,
    _compute_auto_n_estimators,
    select_boruta,
    select_boruta_shap,
)


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

    def test_importance_data_test(self):
        """importance_data='test' should use held-out data."""
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(200, 5)), columns=list("abcde"))
        y = X["a"] + rng.normal(size=200) * 0.1

        selected = select_boruta(
            X,
            y,
            importance_data="test",
            test_size=0.3,
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


class TestBorutaShap:
    """Boruta-Shap tests (requires catboost)."""

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


class TestBorutaNanHandling:
    """NaN handling tests."""

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
