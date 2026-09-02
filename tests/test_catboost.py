"""Tests for CatBoost feature selection."""

import pytest
import numpy as np
import pandas as pd

catboost = pytest.importorskip("catboost")

pytestmark = pytest.mark.catboost

from sift.catboost import (  # noqa: E402
    catboost_select,
    catboost_classif,
    catboost_regression,
    CatBoostSelectionResult,
    _resolve_metric_and_direction,
    _resolve_loss_function,
    _generate_feature_counts,
    _get_feature_types,
    _aggregate_feature_lists,
    _choose_catboost_target_k,
    _select_final_catboost_features,
)
import sift.catboost as catboost_module  # noqa: E402
import sift.catboost_common as catboost_common_module  # noqa: E402
from sift._preprocess import best_score_from_dict as _best_score_from_dict  # noqa: E402


class TestScoreDirection:
    """Tests for score direction handling."""

    def test_resolve_rmse(self):
        metric, hib = _resolve_metric_and_direction(
            task='regression',
            y=pd.Series([0.0, 1.0]),
            eval_metric='RMSE',
            higher_is_better=None,
        )
        assert metric == 'RMSE'
        assert hib is False

    def test_resolve_auc(self):
        metric, hib = _resolve_metric_and_direction(
            task='classification',
            y=pd.Series([0, 1]),
            eval_metric='AUC',
            higher_is_better=None,
        )
        assert metric == 'AUC'
        assert hib is True

    def test_resolve_explicit_override(self):
        metric, hib = _resolve_metric_and_direction(
            task='regression',
            y=pd.Series([0.0, 1.0]),
            eval_metric='RMSE',
            higher_is_better=True,
        )
        assert metric == 'RMSE'
        assert hib is True  # Explicit override

    def test_resolve_default_regression(self):
        metric, hib = _resolve_metric_and_direction(
            task='regression',
            y=pd.Series([0.0, 1.0]),
            eval_metric=None,
            higher_is_better=None,
        )
        assert metric == 'RMSE'
        assert hib is False

    def test_resolve_default_classification(self):
        metric, hib = _resolve_metric_and_direction(
            task='classification',
            y=pd.Series([0, 1, 0, 1]),
            eval_metric=None,
            higher_is_better=None,
        )
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

    def test_unknown_metric_requires_explicit_direction(self):
        with pytest.raises(ValueError, match="higher_is_better"):
            _resolve_metric_and_direction(
                task="classification",
                y=pd.Series([0, 1, 0, 1]),
                eval_metric="AUC_MACRO_BOGUS",
                higher_is_better=None,
            )

        metric, hib = _resolve_metric_and_direction(
            task="classification",
            y=pd.Series([0, 1, 0, 1]),
            eval_metric="AUC_MACRO_BOGUS",
            higher_is_better=True,
        )
        assert metric == "AUC_MACRO_BOGUS"
        assert hib is True

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

    def test_choose_target_k_ignores_nan_scores(self):
        target_k, best_k, best_score, scores_mean, _ = _choose_catboost_target_k(
            {5: [float("nan")], 3: [1.0]},
            k_req=None,
            resolved_hib=False,
            tolerance=0.0,
            selection_patience=3,
            verbose=False,
        )

        assert target_k == 3
        assert best_k == 3
        assert best_score == 1.0
        assert scores_mean == {3: 1.0}

    def test_final_selection_requires_recorded_feature_list(self):
        with pytest.raises(RuntimeError, match="No feature list was recorded"):
            _select_final_catboost_features(
                target_k=2,
                k_req=None,
                all_features_by_k={},
                all_features=["b", "a"],
                prefilter_features_first=None,
                use_stability=False,
                stability_threshold=0.6,
            )

    def test_final_selection_does_not_pad_unrecorded_features(self):
        selected, _ = _select_final_catboost_features(
            target_k=3,
            k_req=3,
            all_features_by_k={3: [["a"]]},
            all_features=["a", "b", "c"],
            prefilter_features_first=None,
            use_stability=False,
            stability_threshold=0.6,
        )

        assert selected == ["a"]

    def test_final_selection_without_stability_does_not_return_stability_scores(self):
        selected, stability_scores = _select_final_catboost_features(
            target_k=2,
            k_req=None,
            all_features_by_k={2: [["a", "b"], ["a", "c"]]},
            all_features=["a", "b", "c"],
            prefilter_features_first=None,
            use_stability=False,
            stability_threshold=0.6,
        )

        assert selected == ["a", "b"]
        assert stability_scores is None

    def test_stability_selection_never_exceeds_target_k(self):
        selected, stability_scores = _select_final_catboost_features(
            target_k=2,
            k_req=None,
            all_features_by_k={
                2: [["a", "b"], ["a", "c"], ["b", "c"]],
            },
            all_features=["a", "b", "c"],
            prefilter_features_first=None,
            use_stability=True,
            stability_threshold=0.5,
        )

        assert len(selected) == 2
        assert stability_scores is not None


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


class TestCatBoostInputValidation:
    """Tests for typo-sensitive input columns."""

    def test_catboost_import_does_not_break_numba_relevance(self):
        """Importing CatBoost should not make classic filter relevance crash."""
        from sift import select_mrmr

        rng = np.random.default_rng(20260420)
        n = 80
        signal = rng.normal(size=n)
        X = pd.DataFrame(
            {
                "signal": signal,
                "noise_a": rng.normal(size=n),
                "noise_b": rng.normal(size=n),
            }
        )
        y = signal + 0.1 * rng.normal(size=n)

        selected = select_mrmr(
            X,
            y,
            k=2,
            task="regression",
            estimator="classic",
            subsample=None,
            verbose=False,
        )

        assert selected[0] == "signal"

    def test_missing_group_col_raises(self):
        X = pd.DataFrame({"f0": [0.0, 1.0, 2.0, 3.0], "f1": [1.0, 0.0, 1.0, 0.0]})
        y = pd.Series([0.0, 1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="group_col"):
            catboost_select(X, y, k=1, group_col="missing", verbose=False)

    def test_missing_sample_weight_col_raises(self):
        X = pd.DataFrame({"f0": [0.0, 1.0, 2.0, 3.0], "f1": [1.0, 0.0, 1.0, 0.0]})
        y = pd.Series([0.0, 1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="sample_weight_col"):
            catboost_select(X, y, k=1, sample_weight_col="missing", verbose=False)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"task": "bogus"}, "task=.*invalid"),
            ({"algorithm": "bogus"}, "algorithm=.*invalid"),
            ({"prefilter_method": "bogus"}, "prefilter_method=.*invalid"),
            ({"step_function": 1.0}, "step_function must be a finite float"),
            ({"use_stability": True, "n_bootstrap": 0}, "n_bootstrap must be a positive integer"),
        ],
    )
    def test_invalid_public_options_raise(self, kwargs, match):
        X = pd.DataFrame({"f0": [0.0, 1.0, 2.0, 3.0], "f1": [1.0, 0.0, 1.0, 0.0]})
        y = pd.Series([0.0, 1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match=match):
            catboost_select(
                X,
                y,
                k=1,
                prefilter_k=None,
                n_estimators=10,
                n_splits=2,
                verbose=False,
                **kwargs,
            )

    def test_group_kfold_without_groups_raises(self):
        from sklearn.model_selection import GroupKFold

        X = pd.DataFrame({"f0": [0.0, 1.0, 2.0, 3.0], "f1": [1.0, 0.0, 1.0, 0.0]})
        y = pd.Series([0.0, 1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="GroupKFold requires group_col"):
            catboost_select(
                X,
                y,
                k=1,
                cv=GroupKFold(n_splits=2),
                prefilter_k=None,
                n_estimators=10,
                verbose=False,
                random_state=0,
            )

    def test_cv_and_stability_are_mutually_exclusive(self):
        from sklearn.model_selection import KFold

        X = pd.DataFrame({"f0": [0.0, 1.0, 2.0, 3.0], "f1": [1.0, 0.0, 1.0, 0.0]})
        y = pd.Series([0.0, 1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="mutually exclusive"):
            catboost_select(
                X,
                y,
                k=1,
                cv=KFold(n_splits=2),
                use_stability=True,
                prefilter_k=None,
                n_estimators=10,
                verbose=False,
                random_state=0,
            )

    def test_custom_splitter_without_groups_argument_rejects_group_col(self):
        class UngroupedSplitter:
            def split(self, X, y):
                del X, y
                yield np.array([0, 1]), np.array([2, 3])

        X = pd.DataFrame(
            {
                "f0": [0.0, 1.0, 2.0, 3.0],
                "group": [0, 0, 1, 1],
            }
        )
        y = pd.Series([0.0, 1.0, 2.0, 3.0])

        with pytest.raises(TypeError, match="does not accept groups"):
            catboost_select(
                X,
                y,
                k=1,
                cv=UngroupedSplitter(),
                group_col="group",
                prefilter_k=None,
                n_estimators=10,
                verbose=False,
                random_state=0,
            )

    def test_internal_custom_splitter_type_error_propagates(self):
        class BrokenSplitter:
            def split(self, X, y):
                del X, y
                raise TypeError("internal splitter failure")

        X = pd.DataFrame({"f0": [0.0, 1.0, 2.0, 3.0]})
        y = pd.Series([0.0, 1.0, 2.0, 3.0])

        with pytest.raises(TypeError, match="internal splitter failure"):
            catboost_select(
                X,
                y,
                k=1,
                cv=BrokenSplitter(),
                prefilter_k=None,
                n_estimators=10,
                verbose=False,
                random_state=0,
            )


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
            X, y, k=5,
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
            X, y, k=5,
            n_splits=2,
            prefilter_k=15,
            prefilter_method='catboost',
            n_estimators=50,
            verbose=False,
            random_state=42,
        )

        assert len(selected) == 5

    def test_catboost_prefilter_protects_categorical_and_text_features(self, monkeypatch):
        class FakePool:
            def __init__(self, X, **kwargs):
                self.X = X
                self.kwargs = kwargs

        class FakeRegressor:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def fit(self, pool):
                self.pool = pool

            def get_feature_importance(self, pool, type):
                del pool, type
                return np.array([10.0, 9.0, 0.1, 0.0], dtype=np.float64)

        monkeypatch.setattr(catboost_common_module, "Pool", FakePool)
        monkeypatch.setattr(catboost_common_module, "CatBoostRegressor", FakeRegressor)
        X = pd.DataFrame(
            {
                "n0": [0.0, 1.0, 2.0, 3.0],
                "n1": [1.0, 2.0, 3.0, 4.0],
                "cat": pd.Series(["a", "b", "a", "c"], dtype="category"),
                "txt": ["alpha", "beta", "gamma", "delta"],
            }
        )
        y = pd.Series([0.0, 1.0, 0.0, 1.0])

        selected = catboost_common_module._catboost_importance_prefilter(
            X,
            y,
            k=1,
            task="regression",
            cat_features=["cat"],
            text_features=["txt"],
            random_state=0,
            n_jobs=1,
        )

        assert selected == ["n0", "cat", "txt"]

    def test_prefilter_receives_fold_sample_weight(self, monkeypatch):
        X = pd.DataFrame(np.arange(20, dtype=float).reshape(5, 4), columns=list("abcd"))
        y = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0])
        weights = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        train_idx = np.array([0, 1, 3])
        val_idx = np.array([2, 4])
        captured = {}

        def fake_prefilter(*args, sample_weight=None, **kwargs):
            captured["sample_weight"] = sample_weight.copy()
            return ["a", "b"]

        def fake_select(*args, **kwargs):
            return {1: 0.1}, {1: ["a"]}

        monkeypatch.setattr(catboost_module, "_prefilter_features", fake_prefilter)
        monkeypatch.setattr(catboost_module, "_select_features_single_split", fake_select)

        catboost_module._run_catboost_split_evaluation(
            X_work=X,
            y=y,
            sample_weights=weights,
            splits=[(train_idx, val_idx)],
            all_features=list(X.columns),
            counts=[1],
            task="regression",
            model_params={},
            cat_features_final=[],
            text_feat=[],
            prefilter_k=2,
            prefilter_method="mrmr",
            random_state=0,
            n_jobs=1,
            algorithm="prediction",
            resolved_metric="RMSE",
            resolved_hib=False,
            train_early_stopping_rounds=3,
            steps=1,
            k_req=1,
            verbose=False,
        )

        pd.testing.assert_series_equal(
            captured["sample_weight"],
            weights.iloc[train_idx],
        )

    def test_user_overfitting_detector_params_are_preserved(self, monkeypatch):
        X = pd.DataFrame(np.arange(20, dtype=float).reshape(5, 4), columns=list("abcd"))
        y = pd.Series(np.arange(5, dtype=float))
        captured = {}

        def fake_select(*args, model_params, train_early_stopping_rounds, **kwargs):
            del args, kwargs
            captured["model_params"] = dict(model_params)
            captured["fit_early_stopping"] = train_early_stopping_rounds
            return {1: 0.1}, {1: ["a"]}

        monkeypatch.setattr(catboost_module, "_select_features_single_split", fake_select)

        catboost_module._run_catboost_split_evaluation(
            X_work=X,
            y=y,
            sample_weights=None,
            splits=[(np.array([0, 1, 2]), np.array([3, 4]))],
            all_features=list(X.columns),
            counts=[1],
            task="regression",
            model_params={"od_type": "IncToDec", "od_pval": 0.01, "od_wait": 7},
            cat_features_final=[],
            text_feat=[],
            prefilter_k=None,
            prefilter_method="none",
            random_state=0,
            n_jobs=1,
            algorithm="prediction",
            resolved_metric="RMSE",
            resolved_hib=False,
            train_early_stopping_rounds=20,
            steps=1,
            k_req=1,
            verbose=False,
        )

        assert captured["model_params"]["od_type"] == "IncToDec"
        assert captured["model_params"]["od_pval"] == 0.01
        assert captured["model_params"]["od_wait"] == 7
        assert captured["fit_early_stopping"] is None

    def test_prediction_algorithm(self):
        """Test fastest algorithm option."""
        np.random.seed(42)
        n, p = 200, 15
        X = pd.DataFrame(np.random.randn(n, p), columns=[f'f{i}' for i in range(p)])
        y = X['f0'] + np.random.randn(n) * 0.3

        selected = catboost_regression(
            X, y, k=5,
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
            X, y, k=5,
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
            X, y, k=5,
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
            X, y, k=5,
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
            X, y, k=5,
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
            X, y, k=5,
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
                X, y, k=k,
                task='regression',
                n_splits=2,
                prefilter_k=None,
                n_estimators=50,
                verbose=False,
                random_state=42,
            )
            assert len(result.selected_features) == k, f"Expected {k} features, got {len(result.selected_features)}"

    def test_k_larger_than_feature_count_caps(self):
        """When K exceeds available features, the selector caps to n_features."""
        np.random.seed(42)
        n, p = 120, 4
        X = pd.DataFrame(np.random.randn(n, p), columns=[f'f{i}' for i in range(p)])
        y = X['f0'] + np.random.randn(n) * 0.3

        import warnings as w
        with w.catch_warnings(record=True) as caught:
            w.simplefilter("always")
            result = catboost_select(
                X, y, k=10,
                task='regression',
                n_splits=2,
                prefilter_k=None,
                n_estimators=30,
                verbose=False,
                random_state=42,
            )

        assert len(result.selected_features) == p
        assert result.best_k == p
        assert any("exceeds max evaluated feature count" in str(item.message) for item in caught)

    def test_exact_k_with_stability(self):
        """K guarantee should hold even with stability selection."""
        np.random.seed(42)
        n, p = 200, 15
        X = pd.DataFrame(np.random.randn(n, p), columns=[f'f{i}' for i in range(p)])
        y = X['f0'] + np.random.randn(n) * 0.3

        result = catboost_select(
            X, y, k=7,
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
            X, y, k=None,
            task='regression',
            min_features=3,
            selection_patience=2,
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
        assert result.selection_patience == 2

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
            X, y, k=3,
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
            X, y, k=5,
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
            X, y, k=5,
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
            X, y, k=5,
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
            scores_by_k={10: 0.52, 5: 0.48, 3: 0.50},
            scores_std_by_k={10: 0.02, 5: 0.02, 3: 0.02},
            feature_importances=pd.Series({'f0': 1.0, 'f1': 0.8, 'f2': 0.5, 'f3': 0.3, 'f4': 0.1}),
            features_by_k={10: ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9'],
                           5: ['f0', 'f1', 'f2', 'f3', 'f4'],
                           3: ['f0', 'f1', 'f2']},
            metric='RMSE',
            higher_is_better=False,
        )

        # Best is 0.48 at k=5. With 5% tolerance, threshold is 0.504
        # k=3 (0.50) is within tolerance, k=10 (0.52) is not
        parsimonious = result.features_within_tolerance(tolerance=0.05)
        assert len(parsimonious) == 3
        assert parsimonious == ['f0', 'f1', 'f2']

    def test_features_within_tolerance_handles_negative_lower_is_better_scores(self):
        result = CatBoostSelectionResult(
            selected_features=["f0", "f1", "f2", "f3", "f4"],
            best_k=5,
            scores_by_k={5: -1.0, 3: -0.95, 2: -0.8},
            scores_std_by_k={},
            feature_importances=pd.Series(dtype=float),
            features_by_k={5: ["f0", "f1", "f2", "f3", "f4"], 3: ["f0", "f1", "f2"]},
            metric="NEG_LOSS",
            higher_is_better=False,
        )

        assert result.features_within_tolerance(tolerance=0.1) == ["f0", "f1", "f2"]

    def test_features_within_tolerance_handles_negative_higher_is_better_scores(self):
        result = CatBoostSelectionResult(
            selected_features=["f0", "f1", "f2", "f3", "f4"],
            best_k=5,
            scores_by_k={5: -0.5, 3: -0.52, 2: -0.7},
            scores_std_by_k={},
            feature_importances=pd.Series(dtype=float),
            features_by_k={5: ["f0", "f1", "f2", "f3", "f4"], 3: ["f0", "f1", "f2"]},
            metric="NEG_R2",
            higher_is_better=True,
        )

        assert result.features_within_tolerance(tolerance=0.1) == ["f0", "f1", "f2"]

    def test_features_within_tolerance_fallback_preserves_tied_importance_order(self):
        result = CatBoostSelectionResult(
            selected_features=["b", "a"],
            best_k=2,
            scores_by_k={2: 1.0},
            scores_std_by_k={},
            feature_importances=pd.Series({"b": 1.0, "a": 1.0, "c": 0.0}),
            features_by_k={},
            metric="AUC",
            higher_is_better=True,
        )

        assert result.features_within_tolerance(tolerance=0.0) == ["b", "a"]

    @pytest.mark.parametrize(
        ("selection_patience", "expected"),
        [(1, [f"f{i}" for i in range(20)]), (3, [f"f{i}" for i in range(5)])],
    )
    def test_features_within_tolerance_honors_selection_patience(
        self,
        selection_patience,
        expected,
    ):
        features = [f"f{i}" for i in range(20)]
        result = CatBoostSelectionResult(
            selected_features=features,
            best_k=20,
            scores_by_k={20: 0.30, 15: 0.31, 10: 0.305, 5: 0.302, 3: 0.60},
            scores_std_by_k={},
            feature_importances=pd.Series(
                np.arange(20, 0, -1, dtype=float),
                index=features,
            ),
            features_by_k={k: features[:k] for k in (20, 15, 10, 5, 3)},
            higher_is_better=False,
            selection_patience=selection_patience,
        )

        assert result.features_within_tolerance(tolerance=0.01) == expected

    @pytest.mark.parametrize("tolerance", [-0.1, np.nan, True])
    def test_features_within_tolerance_validates_tolerance(self, tolerance):
        result = CatBoostSelectionResult(
            selected_features=["f0"],
            best_k=1,
            scores_by_k={1: 1.0},
            scores_std_by_k={},
            feature_importances=pd.Series({"f0": 1.0}),
        )

        with pytest.raises(ValueError, match="tolerance"):
            result.features_within_tolerance(tolerance=tolerance)

    def test_features_within_tolerance_rejects_all_nonfinite_scores(self):
        result = CatBoostSelectionResult(
            selected_features=["f0"],
            best_k=1,
            scores_by_k={1: np.nan, 2: np.inf},
            scores_std_by_k={},
            feature_importances=pd.Series({"f0": 1.0}),
        )

        with pytest.raises(RuntimeError, match="No finite scores"):
            result.features_within_tolerance()

    def test_features_within_tolerance_skips_nonfinite_gaps(self):
        result = CatBoostSelectionResult(
            selected_features=["f0", "f1", "f2"],
            best_k=3,
            scores_by_k={3: 0.30, 2: np.nan, 1: 0.302},
            scores_std_by_k={},
            feature_importances=pd.Series({"f0": 1.0, "f1": 0.5, "f2": 0.25}),
            features_by_k={1: ["f0"], 3: ["f0", "f1", "f2"]},
            higher_is_better=False,
            selection_patience=1,
        )

        assert result.features_within_tolerance(tolerance=0.01) == ["f0"]

    def test_score_at_k(self):
        result = CatBoostSelectionResult(
            selected_features=['f0'],
            best_k=5,
            scores_by_k={5: 0.48, 10: 0.50},
            scores_std_by_k={5: 0.02, 10: 0.03},
            feature_importances=pd.Series(dtype=float),
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
            X, y, k=3,
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
            X, y, k=3,
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
            catboost_select(
                X, y, k=2,
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
                X, y, k=30,  # Request 30 features
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
                X, y, k=10,
                algorithm='forward_greedy',
                prefilter_k=None,  # Don't prefilter
                n_splits=2,
                verbose=False,
                random_state=0,
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
                X, y, k=35,  # Exceeds MAX_FORWARD_GREEDY_K=30
                algorithm='forward_greedy',
                n_splits=2,
                verbose=False,
                random_state=0,
            )
