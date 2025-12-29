"""Tests for CatBoost feature selection."""

import pytest
import numpy as np
import pandas as pd

catboost = pytest.importorskip("catboost")

from sift.catboost import (
    catboost_select,
    catboost_regression,
    catboost_classif,
    CatBoostSelectionResult,
    _resolve_higher_is_better,
    _resolve_metric_and_direction,
    _resolve_loss_function,
    _best_score_from_dict,
    _generate_feature_counts,
    _get_feature_types,
    _aggregate_feature_lists,
)


class TestScoreDirection:
    """Tests for score direction handling."""

    def test_resolve_rmse(self):
        metric, hib = _resolve_higher_is_better('RMSE', None, 'regression')
        assert metric == 'RMSE'
        assert hib is False

    def test_resolve_auc(self):
        metric, hib = _resolve_higher_is_better('AUC', None, 'classification')
        assert metric == 'AUC'
        assert hib is True

    def test_resolve_explicit_override(self):
        metric, hib = _resolve_higher_is_better('RMSE', True, 'regression')
        assert hib is True  # Explicit override

    def test_resolve_default_regression(self):
        metric, hib = _resolve_higher_is_better(None, None, 'regression')
        assert metric == 'RMSE'
        assert hib is False

    def test_resolve_default_classification(self):
        metric, hib = _resolve_higher_is_better(None, None, 'classification')
        assert metric == 'Logloss'
        assert hib is False

    def test_multiclass_detection(self):
        """Test that multiclass targets use MultiClass metric."""
        y = pd.Series([0, 1, 2, 0, 1, 2])  # 3 classes
        metric, hib = _resolve_metric_and_direction(
            task='classification', y=y, eval_metric=None, higher_is_better=None
        )
        assert metric == 'MultiClass'
        assert hib is False

    def test_binary_detection(self):
        """Test that binary targets use Logloss metric."""
        y = pd.Series([0, 1, 0, 1])  # 2 classes
        metric, hib = _resolve_metric_and_direction(
            task='classification', y=y, eval_metric=None, higher_is_better=None
        )
        assert metric == 'Logloss'
        assert hib is False

    def test_multiclass_loss_function(self):
        """Test that multiclass targets use MultiClass loss."""
        y = pd.Series([0, 1, 2, 0, 1, 2])
        loss = _resolve_loss_function(task='classification', y=y, loss_function=None)
        assert loss == 'MultiClass'

    def test_best_score_lower_is_better(self):
        scores = {10: 0.5, 5: 0.3, 3: 0.4}
        best_k, best_score = _best_score_from_dict(scores, higher_is_better=False)
        assert best_k == 5
        assert best_score == 0.3

    def test_best_score_higher_is_better(self):
        scores = {10: 0.5, 5: 0.9, 3: 0.7}
        best_k, best_score = _best_score_from_dict(scores, higher_is_better=True)
        assert best_k == 5
        assert best_score == 0.9


class TestFeatureTypes:
    """Tests for feature type detection."""

    def test_object_is_categorical_by_default(self):
        X = pd.DataFrame({
            'num': [1.0, 2.0, 3.0],
            'cat': pd.Categorical(['a', 'b', 'c']),
            'obj': ['x', 'y', 'z'],
        })
        cat_features, text_features = _get_feature_types(X, list(X.columns), None)
        assert 'cat' in cat_features
        assert 'obj' in cat_features  # object → categorical by default
        assert text_features == []

    def test_explicit_text_features(self):
        X = pd.DataFrame({
            'num': [1.0, 2.0, 3.0],
            'text_col': ['hello world', 'foo bar', 'test text'],
        })
        cat_features, text_features = _get_feature_types(
            X, list(X.columns), text_features=['text_col']
        )
        assert 'text_col' in text_features
        assert 'text_col' not in cat_features


class TestFeatureCounts:
    """Tests for feature count generation."""

    def test_includes_baseline(self):
        counts = _generate_feature_counts(100, min_features=5, step_function=0.5)
        assert 100 in counts  # Baseline included

    def test_includes_min(self):
        counts = _generate_feature_counts(100, min_features=5, step_function=0.5)
        assert 5 in counts

    def test_descending_order(self):
        counts = _generate_feature_counts(100, min_features=5, step_function=0.67)
        assert counts == sorted(counts, reverse=True)


class TestFeatureAggregation:
    """Tests for feature list aggregation."""

    def test_frequency_ordering(self):
        """Features selected more often should rank higher."""
        lists = [
            ['f0', 'f1', 'f2'],
            ['f0', 'f1', 'f3'],
            ['f0', 'f2', 'f3'],
        ]
        ordered, stability = _aggregate_feature_lists(lists)
        # f0 appears 3 times, should be first
        assert ordered[0] == 'f0'
        assert stability['f0'] == 1.0

    def test_rank_tiebreak(self):
        """Among equal frequency, earlier average position wins."""
        lists = [
            ['f0', 'f1', 'f2'],
            ['f0', 'f2', 'f1'],
        ]
        ordered, stability = _aggregate_feature_lists(lists)
        # f0 always first (position 0)
        # f1: positions [1, 2] → mean 1.5
        # f2: positions [2, 1] → mean 1.5
        # Both f1/f2 have same frequency and mean rank, alphabetical breaks tie
        assert ordered[0] == 'f0'
        assert stability['f0'] == 1.0
        assert stability['f1'] == 1.0
        assert stability['f2'] == 1.0

    def test_k_limit(self):
        """Should limit to k features when specified."""
        lists = [['f0', 'f1', 'f2', 'f3', 'f4']]
        ordered, _ = _aggregate_feature_lists(lists, k=3)
        assert len(ordered) == 3

    def test_empty_input(self):
        """Handle empty input gracefully."""
        ordered, stability = _aggregate_feature_lists([])
        assert ordered == []
        assert len(stability) == 0


class TestCatBoostRegression:
    """Integration tests for regression."""

    def test_basic(self):
        np.random.seed(42)
        n, p = 200, 20
        X = pd.DataFrame(np.random.randn(n, p), columns=[f'f{i}' for i in range(p)])
        y = X['f0'] + 0.5 * X['f1'] + np.random.randn(n) * 0.3

        selected = catboost_regression(
            X, y, K=5,
            n_splits=2,
            prefilter_k=None,
            n_estimators=50,
            verbose=False,
            random_state=42,
        )

        assert len(selected) == 5
        assert 'f0' in selected

    def test_with_prefilter(self):
        np.random.seed(42)
        n, p = 200, 30
        X = pd.DataFrame(np.random.randn(n, p), columns=[f'f{i}' for i in range(p)])
        y = X['f0'] + np.random.randn(n) * 0.3

        selected = catboost_regression(
            X, y, K=5,
            n_splits=2,
            prefilter_k=15,
            prefilter_method='catboost',
            n_estimators=50,
            verbose=False,
            random_state=42,
        )

        assert len(selected) == 5

    def test_prediction_algorithm(self):
        """Test fastest algorithm option."""
        np.random.seed(42)
        n, p = 200, 15
        X = pd.DataFrame(np.random.randn(n, p), columns=[f'f{i}' for i in range(p)])
        y = X['f0'] + np.random.randn(n) * 0.3

        selected = catboost_regression(
            X, y, K=5,
            n_splits=2,
            prefilter_k=None,
            n_estimators=50,
            algorithm='prediction',  # Fastest
            verbose=False,
            random_state=42,
        )

        assert len(selected) == 5

    def test_forward_selection(self):
        """Test forward selection algorithm."""
        np.random.seed(42)
        n, p = 200, 15
        X = pd.DataFrame(np.random.randn(n, p), columns=[f'f{i}' for i in range(p)])
        y = X['f0'] + 0.5 * X['f1'] + np.random.randn(n) * 0.3

        selected = catboost_regression(
            X, y, K=5,
            n_splits=2,
            prefilter_k=None,
            n_estimators=50,
            algorithm='forward',  # Forward selection
            verbose=False,
            random_state=42,
        )

        assert len(selected) == 5
        # Forward selection should identify informative features
        assert 'f0' in selected or 'f1' in selected


class TestCustomSplitters:
    """Tests for custom CV splitter support."""

    def test_time_series_split(self):
        """Test with TimeSeriesSplit for time series data."""
        from sklearn.model_selection import TimeSeriesSplit

        np.random.seed(42)
        n, p = 300, 15
        X = pd.DataFrame(np.random.randn(n, p), columns=[f'f{i}' for i in range(p)])
        # Time series: target depends on recent values
        y = X['f0'] + 0.3 * X['f1'] + np.random.randn(n) * 0.3

        result = catboost_select(
            X, y, K=5,
            task='regression',
            cv=TimeSeriesSplit(n_splits=3),
            prefilter_k=None,
            n_estimators=50,
            verbose=False,
            random_state=42,
        )

        assert len(result.selected_features) == 5
        assert len(result.scores_by_k) > 0

    def test_group_kfold(self):
        """Test with GroupKFold for grouped data."""
        from sklearn.model_selection import GroupKFold

        np.random.seed(42)
        n_groups = 20
        samples_per_group = 15
        n = n_groups * samples_per_group
        p = 15

        # Create grouped data (like NBA players)
        X = pd.DataFrame(np.random.randn(n, p), columns=[f'f{i}' for i in range(p)])
        X['player_id'] = np.repeat(np.arange(n_groups), samples_per_group)
        y = X['f0'] + np.random.randn(n) * 0.3

        result = catboost_select(
            X, y, K=5,
            task='regression',
            cv=GroupKFold(n_splits=3),
            group_col='player_id',
            prefilter_k=None,
            n_estimators=50,
            verbose=False,
            random_state=42,
        )

        assert len(result.selected_features) == 5
        assert 'player_id' not in result.selected_features  # Group col should be excluded


class TestCatBoostClassification:
    """Integration tests for classification."""

    def test_basic(self):
        np.random.seed(42)
        n, p = 200, 20
        X = pd.DataFrame(np.random.randn(n, p), columns=[f'f{i}' for i in range(p)])
        y = pd.Series((X['f0'] + X['f1'] > 0).astype(int))

        selected = catboost_classif(
            X, y, K=5,
            n_splits=2,
            prefilter_k=None,
            n_estimators=50,
            verbose=False,
            random_state=42,
        )

        assert len(selected) == 5

    def test_multiclass(self):
        """Test multiclass classification uses correct metric/loss."""
        np.random.seed(42)
        n, p = 200, 15
        X = pd.DataFrame(np.random.randn(n, p), columns=[f'f{i}' for i in range(p)])
        # 3 classes
        y = pd.Series(np.random.choice([0, 1, 2], n))

        result = catboost_select(
            X, y, K=5,
            task='classification',
            n_splits=2,
            prefilter_k=None,
            n_estimators=50,
            verbose=False,
            random_state=42,
        )

        assert len(result.selected_features) == 5
        assert result.metric == 'MultiClass'


class TestKGuarantee:
    """Tests for K guarantee - always return exactly K features when specified."""

    def test_exact_k_returned(self):
        """When K is specified, exactly K features should be returned."""
        np.random.seed(42)
        n, p = 200, 20
        X = pd.DataFrame(np.random.randn(n, p), columns=[f'f{i}' for i in range(p)])
        y = X['f0'] + np.random.randn(n) * 0.3

        for k in [3, 5, 10, 15]:
            result = catboost_select(
                X, y, K=k,
                task='regression',
                n_splits=2,
                prefilter_k=None,
                n_estimators=50,
                verbose=False,
                random_state=42,
            )
            assert len(result.selected_features) == k, f"Expected {k} features, got {len(result.selected_features)}"

    def test_exact_k_with_stability(self):
        """K guarantee should hold even with stability selection."""
        np.random.seed(42)
        n, p = 200, 15
        X = pd.DataFrame(np.random.randn(n, p), columns=[f'f{i}' for i in range(p)])
        y = X['f0'] + np.random.randn(n) * 0.3

        result = catboost_select(
            X, y, K=7,
            task='regression',
            use_stability=True,
            n_bootstrap=10,
            stability_threshold=0.8,  # High threshold - may not have 7 stable features
            prefilter_k=None,
            n_estimators=50,
            verbose=False,
            random_state=42,
        )

        # Should still return exactly 7 features even if fewer pass threshold
        assert len(result.selected_features) == 7


class TestCatBoostSelect:
    """Tests for full catboost_select API."""

    def test_result_dataclass(self):
        np.random.seed(42)
        n, p = 150, 15
        X = pd.DataFrame(np.random.randn(n, p), columns=[f'f{i}' for i in range(p)])
        y = X['f0'] + np.random.randn(n) * 0.3

        result = catboost_select(
            X, y, K=None,
            task='regression',
            min_features=3,
            n_splits=2,
            prefilter_k=None,
            n_estimators=50,
            verbose=False,
            random_state=42,
        )

        assert isinstance(result, CatBoostSelectionResult)
        assert len(result.selected_features) > 0
        assert result.best_k in result.scores_by_k
        assert isinstance(result.feature_importances, pd.Series)
        assert result.higher_is_better is False  # RMSE

        mean, std = result.score_at_k(result.best_k)
        assert np.isfinite(mean)

    def test_with_categorical(self):
        np.random.seed(42)
        n = 200
        X = pd.DataFrame({
            'num1': np.random.randn(n),
            'num2': np.random.randn(n),
            'cat1': pd.Categorical(np.random.choice(['A', 'B', 'C'], n)),
        })
        y = X['num1'] + (X['cat1'] == 'A').astype(float) + np.random.randn(n) * 0.3

        selected = catboost_regression(
            X, y, K=3,
            n_splits=2,
            prefilter_k=None,
            n_estimators=50,
            verbose=False,
            random_state=42,
        )

        assert len(selected) == 3

    def test_with_groups(self):
        np.random.seed(42)
        n, p = 200, 10
        X = pd.DataFrame(np.random.randn(n, p), columns=[f'f{i}' for i in range(p)])
        X['group'] = np.repeat(np.arange(20), 10)
        y = X['f0'] + np.random.randn(n) * 0.3

        selected = catboost_regression(
            X, y, K=5,
            group_col='group',
            n_splits=2,
            prefilter_k=None,
            n_estimators=50,
            verbose=False,
            random_state=42,
        )

        assert len(selected) == 5
        assert 'group' not in selected

    def test_with_weights(self):
        np.random.seed(42)
        n, p = 200, 10
        X = pd.DataFrame(np.random.randn(n, p), columns=[f'f{i}' for i in range(p)])
        X['weight'] = np.random.uniform(0.5, 2.0, n)
        y = X['f0'] + np.random.randn(n) * 0.3

        selected = catboost_regression(
            X, y, K=5,
            sample_weight_col='weight',
            n_splits=2,
            prefilter_k=None,
            n_estimators=50,
            verbose=False,
            random_state=42,
        )

        assert len(selected) == 5
        assert 'weight' not in selected

    def test_stability_selection(self):
        np.random.seed(42)
        n, p = 200, 15
        X = pd.DataFrame(np.random.randn(n, p), columns=[f'f{i}' for i in range(p)])
        y = X['f0'] + 0.5 * X['f1'] + np.random.randn(n) * 0.3

        result = catboost_select(
            X, y, K=5,
            task='regression',
            use_stability=True,
            n_bootstrap=5,
            stability_threshold=0.4,
            prefilter_k=None,
            n_estimators=50,
            verbose=False,
            random_state=42,
        )

        assert result.stability_scores is not None
        assert len(result.stability_scores) > 0


class TestResultMethods:
    """Tests for CatBoostSelectionResult methods."""

    def test_features_within_tolerance_with_features_by_k(self):
        result = CatBoostSelectionResult(
            selected_features=['f0', 'f1', 'f2', 'f3', 'f4'],
            best_k=5,
            scores_by_k={10: 0.50, 5: 0.48, 3: 0.51},
            scores_std_by_k={10: 0.02, 5: 0.02, 3: 0.02},
            feature_importances=pd.Series({'f0': 1.0, 'f1': 0.8, 'f2': 0.5, 'f3': 0.3, 'f4': 0.1}),
            features_by_k={10: ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9'],
                           5: ['f0', 'f1', 'f2', 'f3', 'f4'],
                           3: ['f0', 'f1', 'f2']},
            metric='RMSE',
            higher_is_better=False,
        )

        # Best is 0.48 at k=5. With 5% tolerance, threshold is 0.504
        # k=3 (0.51) is within tolerance
        parsimonious = result.features_within_tolerance(tolerance=0.05)
        assert len(parsimonious) == 3
        assert parsimonious == ['f0', 'f1', 'f2']

    def test_score_at_k(self):
        result = CatBoostSelectionResult(
            selected_features=['f0'],
            best_k=5,
            scores_by_k={5: 0.48, 10: 0.50},
            scores_std_by_k={5: 0.02, 10: 0.03},
            feature_importances=pd.Series(),
            metric='RMSE',
            higher_is_better=False,
        )

        mean, std = result.score_at_k(5)
        assert mean == 0.48
        assert std == 0.02

        mean, std = result.score_at_k(99)  # Not present
        assert np.isnan(mean)


class TestCatFeaturesParameter:
    """Tests for explicit cat_features parameter."""

    def test_explicit_cat_features(self):
        """Test that explicit cat_features are used."""
        np.random.seed(42)
        n = 200
        X = pd.DataFrame({
            'num1': np.random.randn(n),
            'num2': np.random.randn(n),
            'int_cat': np.random.choice([1, 2, 3], n),  # Integer-encoded categorical
        })
        y = X['num1'] + (X['int_cat'] == 1).astype(float) * 2 + np.random.randn(n) * 0.3

        result = catboost_select(
            X, y, K=3,
            task='regression',
            cat_features=['int_cat'],  # Explicit
            n_splits=2,
            prefilter_k=None,
            n_estimators=50,
            verbose=False,
            random_state=42,
        )

        assert len(result.selected_features) == 3

    def test_cat_features_merged_with_detected(self):
        """Test that explicit cat_features are merged with auto-detected."""
        np.random.seed(42)
        n = 200
        X = pd.DataFrame({
            'num1': np.random.randn(n),
            'int_cat': np.random.choice([1, 2, 3], n),
            'str_cat': pd.Categorical(np.random.choice(['A', 'B'], n)),
        })
        y = X['num1'] + np.random.randn(n) * 0.3

        # str_cat should be auto-detected, int_cat is explicit
        result = catboost_select(
            X, y, K=3,
            task='regression',
            cat_features=['int_cat'],
            n_splits=2,
            prefilter_k=None,
            n_estimators=50,
            verbose=False,
            random_state=42,
        )

        assert len(result.selected_features) == 3

    def test_treat_object_as_categorical_false_warning(self):
        """Test warning when treat_object_as_categorical=False with orphan object cols."""
        np.random.seed(42)
        n = 100
        X = pd.DataFrame({
            'num1': np.random.randn(n),
            'obj_col': np.random.choice(['A', 'B', 'C'], n),
        })
        y = pd.Series(np.random.randn(n))

        import warnings as w
        with w.catch_warnings(record=True) as caught:
            w.simplefilter("always")
            result = catboost_select(
                X, y, K=2,
                task='regression',
                treat_object_as_categorical=False,
                n_splits=2,
                prefilter_k=None,
                n_estimators=30,
                verbose=False,
                random_state=42,
            )

            # Should have warned about orphan object column
            orphan_warnings = [x for x in caught if 'object column' in str(x.message)]
            assert len(orphan_warnings) >= 1


class TestKPrefilterInteraction:
    """Tests for K + prefilter_k edge cases."""

    def test_k_larger_than_prefilter_k(self):
        """Test K > prefilter_k warns and uses available features."""
        np.random.seed(42)
        n = 200
        X = pd.DataFrame({
            f'f{i}': np.random.randn(n) for i in range(50)
        })
        y = X['f0'] + X['f1'] * 2 + np.random.randn(n) * 0.3

        import warnings as w
        with w.catch_warnings(record=True) as caught:
            w.simplefilter("always")
            result = catboost_select(
                X, y, K=30,  # Request 30 features
                prefilter_k=15,  # But only prefilter to 15
                task='regression',
                n_splits=2,
                n_estimators=50,
                algorithm='prediction',
                verbose=False,
                random_state=42,
            )

            # Should get 15 (capped by prefilter), not 30
            assert len(result.selected_features) == 15

            # Should have warned about K exceeding evaluated count
            k_warnings = [x for x in caught if 'exceeds' in str(x.message)]
            assert len(k_warnings) >= 1


class TestForwardGreedyGuard:
    """Tests for forward_greedy safety limits."""

    def test_forward_greedy_too_many_features_raises(self):
        """Test forward_greedy raises error for too many features."""
        np.random.seed(42)
        n = 100
        # Create 250 features (exceeds MAX_FORWARD_GREEDY_FEATURES=200)
        X = pd.DataFrame({
            f'f{i}': np.random.randn(n) for i in range(250)
        })
        y = pd.Series(np.random.randn(n))

        import pytest
        with pytest.raises(ValueError, match="forward_greedy is O"):
            catboost_select(
                X, y, K=10,
                algorithm='forward_greedy',
                prefilter_k=None,  # Don't prefilter
                n_splits=2,
                verbose=False,
            )

    def test_forward_greedy_k_too_large_raises(self):
        """Test forward_greedy raises error for K > limit."""
        np.random.seed(42)
        n = 100
        X = pd.DataFrame({
            f'f{i}': np.random.randn(n) for i in range(50)
        })
        y = pd.Series(np.random.randn(n))

        import pytest
        with pytest.raises(ValueError, match="forward_greedy is O"):
            catboost_select(
                X, y, K=35,  # Exceeds MAX_FORWARD_GREEDY_K=30
                algorithm='forward_greedy',
                n_splits=2,
                verbose=False,
            )
