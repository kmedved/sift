"""Public contracts for generic ModelSelector."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.utils.validation import check_is_fitted

from sift import GroupPurgedTimeSeriesSplit, ModelSelector, PurgedTimeSeriesSplit, as_result


SKLEARN_VERSION = tuple(int(part) for part in sklearn.__version__.split(".")[:2])


class NoWeightEstimator(BaseEstimator):
    def fit(self, X, y):
        values = np.asarray(X, dtype=np.float64)
        self.coef_ = values.mean(axis=0)
        self.n_features_in_ = values.shape[1]
        return self

    def predict(self, X):
        return np.asarray(X, dtype=np.float64) @ self.coef_

    def score(self, X, y):
        pred = self.predict(X)
        y_arr = np.asarray(y, dtype=np.float64)
        return float(1.0 - np.mean((y_arr - pred) ** 2))


class SpyRidge(Ridge):
    seen_index: list[set[int]] = []

    def fit(self, X, y, sample_weight=None):
        if isinstance(X, pd.DataFrame):
            SpyRidge.seen_index.append(set(int(i) for i in X.index.tolist()))
        else:
            SpyRidge.seen_index.append(set(range(np.asarray(X).shape[0])))
        return super().fit(X, y, sample_weight=sample_weight)


class BoomRidge(Ridge):
    def fit(self, X, y, sample_weight=None):
        raise ValueError("intentional fit failure")


def _regression_data(n=48, p=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = X[:, 0] + 0.9 * X[:, 1] + 0.05 * rng.normal(size=n)
    return X, y


def test_ridge_rfe_explicit_k_does_not_mutate_caller_estimator():
    X, y = _regression_data()
    estimator = Ridge()
    selector = ModelSelector(
        estimator, n_features_to_select=2, cv=2, random_state=0, verbose=False
    )
    fitted = selector.fit(X, y)
    assert fitted.selected_indices_.tolist() == [0, 1]
    assert fitted.n_features_to_select_ == 2
    assert fitted.scores_by_k_ is None
    with pytest.raises(Exception):
        check_is_fitted(estimator)
    cloned = clone(selector)
    assert cloned.estimator is not selector.estimator
    assert cloned.get_params()["n_features_to_select"] == 2
    Xt = fitted.transform(X)
    assert Xt.shape == (X.shape[0], 2)
    names = fitted.get_feature_names_out()
    assert list(names) == ["x0", "x1"]
    restored = fitted.inverse_transform(Xt)
    assert restored.shape == X.shape


def test_dataframe_names_and_original_output_order():
    X, y = _regression_data()
    frame = pd.DataFrame(X, columns=list("abcde"))
    selector = ModelSelector(
        Ridge(),
        method="forward",
        n_features_to_select=2,
        cv=2,
        output_order="original",
        random_state=0,
    ).fit(frame, y)
    assert set(selector.selected_features_) == {"a", "b"}
    np.testing.assert_array_equal(
        selector.get_support(indices=True),
        np.sort(selector.selected_indices_),
    )
    out = selector.transform(frame)
    assert list(out.columns) == ["a", "b"]


def test_classification_logistic_and_pipeline_fits_inside_fold():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 4))
    y = (X[:, 0] + 0.4 * rng.normal(size=60) > 0).astype(int)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200))
    selector = ModelSelector(
        pipe, n_features_to_select=1, cv=3, random_state=0, scoring="accuracy"
    ).fit(X, y)
    assert selector.n_features_selected_ == 1
    assert int(selector.selected_indices_[0]) == 0


def test_sample_weight_honored_or_rejected():
    X, y = _regression_data(n=40, p=3)
    weights = np.ones(40)
    weights[:10] = 0.0
    with_w = ModelSelector(
        Ridge(), n_features_to_select=1, cv=2, random_state=0
    ).fit(X, y, sample_weight=weights)
    without = ModelSelector(
        Ridge(), n_features_to_select=1, cv=2, random_state=0
    ).fit(X, y)
    assert with_w.selected_indices_.shape[0] == 1
    assert without.selected_indices_.shape[0] == 1
    with pytest.raises(TypeError, match="sample_weight"):
        ModelSelector(
            NoWeightEstimator(), n_features_to_select=1, cv=2, random_state=0
        ).fit(X, y, sample_weight=np.ones(40))


def test_searched_counts_are_selection_curve_not_nested_scores():
    X, y = _regression_data()
    selector = ModelSelector(
        Ridge(),
        n_features_to_select=[1, 2, 3],
        cv=3,
        nested=True,
        random_state=0,
        parsimony_tolerance=0.0,
    ).fit(X, y)
    assert selector.scores_by_k_ is not None
    assert set(selector.scores_by_k_) == {1, 2, 3}
    assert selector.nested_scores_ is not None
    assert selector.nested_scores_.shape[0] == 3
    view = selector.result_view()
    assert view.metadata["selection_curve_is_nested_score"] is False
    assert view.metadata["curve_available"] is True
    assert int(view.curve["selected"].sum()) == 1
    restored = as_result(selector)
    assert restored.features == view.features


def test_stability_returns_threshold_passers_without_padding():
    rng = np.random.default_rng(2)
    n, p = 50, 6
    X = rng.normal(size=(n, p))
    X[:, 0] += 3.0
    X[:, 1] += 2.5
    y = X[:, 0] + X[:, 1] + 0.05 * rng.normal(size=n)
    selector = ModelSelector(
        Ridge(),
        method="stability",
        n_features_to_select=5,
        threshold=0.6,
        n_resamples=12,
        random_state=0,
        cv=2,
    ).fit(X, y)
    assert selector.selection_frequencies_ is not None
    assert selector.selection_frequencies_.shape == (p,)
    assert selector.n_features_selected_ <= 5
    assert selector.n_features_selected_ >= 2
    assert 0 in selector.selected_indices_
    assert 1 in selector.selected_indices_
    assert selector.n_features_selected_ == int(
        (selector.selection_frequencies_ >= 0.6).sum()
        if (selector.selection_frequencies_ >= 0.6).sum() <= 5
        else 5
    )


def test_nested_spy_outer_validation_cannot_affect_fold_local_subset():
    X, y = _regression_data(n=30, p=4)
    frame = pd.DataFrame(X, columns=list("abcd"))
    SpyRidge.seen_index = []
    cv = KFold(n_splits=3, shuffle=False)
    outer = list(cv.split(frame, y))
    ModelSelector(
        SpyRidge(),
        n_features_to_select=2,
        cv=cv,
        nested=True,
        random_state=0,
    ).fit(frame, y)
    full = set(range(30))
    subset_seen = [seen for seen in SpyRidge.seen_index if seen != full]
    assert subset_seen
    outer_trains = [set(train.tolist()) for train, _val in outer]
    outer_vals = [set(val.tolist()) for _train, val in outer]
    for seen in subset_seen:
        matching = [i for i, train in enumerate(outer_trains) if seen <= train]
        assert matching, seen
        for i in matching:
            assert seen.isdisjoint(outer_vals[i])


class GroupLenSplitter(GroupKFold):
    """Group-aware splitter that checks local metadata length."""

    def __init__(self):
        super().__init__(n_splits=2)

    def split(self, X, y=None, groups=None):
        n = int(np.asarray(X).shape[0])
        assert groups is not None
        assert int(np.asarray(groups).shape[0]) == n
        mid = n // 2
        yield np.arange(mid), np.arange(mid, n)
        yield np.arange(mid, n), np.arange(mid)


def test_group_and_time_metadata_are_sliced_and_unused_rejected():
    X, y = _regression_data(n=24, p=3)
    groups = np.repeat(np.arange(6), 4)
    time = np.arange(24)
    fitted = ModelSelector(
        Ridge(), n_features_to_select=[1, 2], random_state=0
    ).fit(X, y, groups=groups, time=time)
    assert fitted.n_features_selected_ in {1, 2}
    ModelSelector(
        Ridge(),
        n_features_to_select=1,
        cv=GroupLenSplitter(),
        nested=True,
        random_state=0,
    ).fit(X, y, groups=groups)
    with pytest.raises(ValueError, match="does not consume groups"):
        ModelSelector(
            Ridge(),
            n_features_to_select=[1, 2],
            cv=KFold(n_splits=2, shuffle=False),
            random_state=0,
        ).fit(X, y, groups=groups)
    with pytest.raises(ValueError, match="event_end requires time"):
        ModelSelector(Ridge(), n_features_to_select=1, cv=2).fit(
            X, y, event_end=time
        )


def test_precomputed_pairs_validated_and_not_reused_inside_nested_subset():
    X, y = _regression_data(n=20, p=3)
    n = 20
    good = [
        (np.arange(0, 12), np.arange(12, 20)),
        (np.arange(8, 20), np.arange(0, 8)),
    ]
    ModelSelector(
        Ridge(), n_features_to_select=1, cv=good, nested=True, random_state=0
    ).fit(X, y)
    with pytest.raises(ValueError, match="overlapping"):
        ModelSelector(
            Ridge(),
            n_features_to_select=1,
            cv=[(np.arange(10), np.arange(8, 16))],
        ).fit(X, y)
    with pytest.raises(ValueError, match="boolean"):
        ModelSelector(
            Ridge(),
            n_features_to_select=1,
            cv=[(np.zeros(n, dtype=bool), np.ones(n, dtype=bool))],
        ).fit(X, y)
    with pytest.raises(ValueError, match="outside"):
        ModelSelector(
            Ridge(),
            n_features_to_select=1,
            cv=[(np.array([0, 1, 99]), np.array([2, 3]))],
        ).fit(X, y)


def test_failed_fit_clears_state_and_duplicate_columns_raise():
    X, y = _regression_data(n=20, p=3)
    selector = ModelSelector(BoomRidge(), n_features_to_select=1, cv=2)
    with pytest.raises(ValueError, match="intentional fit failure"):
        selector.fit(X, y)
    assert not hasattr(selector, "selected_indices_")
    frame = pd.DataFrame(X, columns=["a", "a", "b"])
    with pytest.raises(ValueError, match="Duplicate"):
        ModelSelector(Ridge(), n_features_to_select=1, cv=2).fit(frame, y)


def test_default_time_cv_is_purged_splitter():
    X, y = _regression_data(n=18, p=3)
    time = np.arange(18)
    selector = ModelSelector(
        Ridge(), n_features_to_select=[1, 2], random_state=0
    )
    selector.fit(X, y, time=time)
    assert isinstance(
        selector._default_splitter(groups=None, time=time, n_rows=18),
        PurgedTimeSeriesSplit,
    )


@pytest.mark.skipif(SKLEARN_VERSION < (1, 4), reason="fit metadata routing needs sklearn>=1.4")
def test_metadata_routing_grouped_outer_cv_keeps_unused_groups_off_selector():
    from sklearn import config_context
    from sklearn.model_selection import GroupKFold, cross_validate

    X, y = _regression_data(n=30, p=3)
    groups = np.repeat(np.arange(6), 5)
    selector = ModelSelector(Ridge(), n_features_to_select=1, random_state=0)
    pipe = make_pipeline(selector, Ridge())
    with config_context(enable_metadata_routing=True):
        routing = str(selector.get_metadata_routing())
        assert "groups" in routing
        scores = cross_validate(
            pipe, X, y, cv=GroupKFold(n_splits=3), params={"groups": groups}
        )
        assert scores["test_score"].shape == (3,)
        grouped_search = ModelSelector(
            Ridge(), n_features_to_select=[1, 2], random_state=0
        )
        grouped_search.set_fit_request(groups=True)
        grouped_search.fit(X, y, groups=groups)
        assert grouped_search.n_features_selected_ in {1, 2}


def test_forward_adds_conditionally_on_selected_training_columns():
    rng = np.random.default_rng(19)
    a = rng.normal(size=120)
    z = rng.normal(size=120)
    X = np.column_stack([a, a + 0.005 * rng.normal(size=120), z])
    y = 3 * a + 0.8 * z + 0.03 * rng.normal(size=120)
    selected = ModelSelector(
        Ridge(alpha=0.1), method="forward", n_features_to_select=2
    ).fit(X, y).selected_indices_.tolist()
    expected: list[int] = []
    for _ in range(2):
        choices = [j for j in range(3) if j not in expected]
        scores = {
            j: Ridge(alpha=0.1)
            .fit(X[:, expected + [j]], y)
            .score(X[:, expected + [j]], y)
            for j in choices
        }
        expected.append(max(scores, key=scores.get))
    assert selected == expected


def test_default_stability_uses_per_draw_count_search():
    rng = np.random.default_rng(51)
    X = rng.normal(size=(90, 4))
    y = 2 * X[:, 0] + 0.7 * X[:, 1] + rng.normal(size=90) * 0.05
    fitted = ModelSelector(
        Ridge(), method="stability", threshold=1.0, n_resamples=4, random_state=0
    ).fit(X, y)
    assert not np.allclose(fitted.selection_frequencies_, 1.0)
    noise = ModelSelector(
        Ridge(), method="stability", threshold=1.0, n_resamples=4, random_state=1
    ).fit(X, rng.normal(size=90))
    assert noise.n_features_selected_ < X.shape[1]


class WeightSpyRidge(Ridge):
    seen: list[np.ndarray | None] = []

    def fit(self, X, y, sample_weight=None):
        WeightSpyRidge.seen.append(
            None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64).copy()
        )
        return super().fit(X, y, sample_weight=sample_weight)


def test_pipeline_forwards_sample_weight_to_final_estimator():
    X, y = _regression_data(n=40, p=3)
    weights = np.linspace(0.5, 2.0, 40)
    WeightSpyRidge.seen = []
    pipe = make_pipeline(StandardScaler(), WeightSpyRidge())
    ModelSelector(pipe, n_features_to_select=2, random_state=0).fit(
        X, y, sample_weight=weights
    )
    forwarded = [w for w in WeightSpyRidge.seen if w is not None]
    assert forwarded
    assert any(np.allclose(w, weights) for w in forwarded)


def test_dataframe_identity_survives_scorers_and_permutation():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(48, 4))
    y = 2 * X[:, 0] + 0.8 * X[:, 1] + rng.normal(size=48) * 0.1
    frame = pd.DataFrame(X, columns=list("abcd"))
    ModelSelector(Ridge(), n_features_to_select=[1, 2], scoring="r2", cv=2).fit(frame, y)
    ModelSelector(Ridge(), n_features_to_select=[1, 2], scoring="neg_mse", cv=2).fit(
        frame, y
    )
    ModelSelector(
        Ridge(), n_features_to_select=2, importance="permutation", random_state=0
    ).fit(frame, y)
    from sklearn.compose import ColumnTransformer, make_column_selector

    pipe = make_pipeline(
        ColumnTransformer(
            [("scale", StandardScaler(), make_column_selector(dtype_include=np.number))]
        ),
        Ridge(),
    )
    ModelSelector(pipe, n_features_to_select=[1, 2], scoring="r2", cv=2).fit(frame, y)


class PredictOnly(BaseEstimator):
    def fit(self, X, y):
        self.model_ = Ridge().fit(X, y)
        self.coef_ = self.model_.coef_
        return self

    def predict(self, X):
        return self.model_.predict(X)


def test_permutation_keeps_sift_scoring_names():
    X, y = _regression_data(n=40, p=3)
    fitted = ModelSelector(
        PredictOnly(),
        n_features_to_select=2,
        importance="permutation",
        scoring="neg_mse",
        random_state=0,
    ).fit(X, y)
    assert fitted.n_features_selected_ == 2


def test_precomputed_split_generator_is_materialized_once():
    X, y = _regression_data(n=30, p=3)
    pairs = KFold(n_splits=3).split(X, y)
    fitted = ModelSelector(
        Ridge(), n_features_to_select=[1, 2], cv=pairs, random_state=0
    ).fit(X, y)
    assert set(fitted.scores_by_k_) == {1, 2}


def test_group_splitter_subclass_consumes_groups():
    from sklearn.model_selection import GroupKFold

    class EntityCV(GroupKFold):
        pass

    X, y = _regression_data(n=90, p=4, seed=51)
    groups = np.repeat(np.arange(9), 10)
    fitted = ModelSelector(
        Ridge(), n_features_to_select=[1, 2], cv=EntityCV(n_splits=3), random_state=0
    ).fit(X, y, groups=groups)
    assert fitted.n_features_selected_ in {1, 2}


def test_criterion_se_is_standard_error_not_std():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(90, 4))

    class MeanYScore(BaseEstimator):
        def fit(self, X, y):
            self.coef_ = np.arange(np.asarray(X).shape[1], 0, -1, dtype=float)
            return self

        def score(self, X, y):
            return float(np.mean(y))

    y = np.repeat([1.0, 4.0, 10.0], 30)
    fitted = ModelSelector(
        MeanYScore(), n_features_to_select=[1, 2], cv=KFold(3)
    ).fit(X, y)
    expected = float(np.std([1.0, 4.0, 10.0], ddof=1) / np.sqrt(3))
    np.testing.assert_allclose(fitted.result_view().curve["criterion_se"], expected)


def test_fitted_provenance_is_snapshot_not_live_params():
    X, y = _regression_data(n=30, p=3)
    selector = ModelSelector(
        Ridge(), n_features_to_select=[1, 2], scoring="r2", cv=2, random_state=0
    ).fit(X, y)
    before = selector.result_view().metadata["configured_options"]
    assert before.get("scoring") == "r2"
    assert "cv" in before
    selector.set_params(method="forward", random_state=999, scoring="neg_mse")
    after = selector.result_view().metadata["configured_options"]
    assert after == before
    assert after.get("method") == "rfe"


def test_searched_counts_reject_non_integers_and_accept_numpy_ints():
    X, y = _regression_data(n=20, p=3)
    for counts in ([True, 2], [1.9, 2.9], ["1", "2"]):
        with pytest.raises(ValueError, match="integer"):
            ModelSelector(Ridge(), n_features_to_select=counts, cv=2).fit(X, y)
    fitted = ModelSelector(
        Ridge(),
        n_features_to_select=np.array([1, 2], dtype=np.int64),
        cv=2,
        random_state=0,
    ).fit(X, y)
    assert set(fitted.scores_by_k_) == {1, 2}


def test_nested_precomputed_count_search_requires_reusable_splitter():
    X, y = _regression_data(n=20, p=3)
    pairs = [
        (np.arange(0, 12), np.arange(12, 20)),
        (np.arange(8, 20), np.arange(0, 8)),
    ]
    with pytest.raises(ValueError, match="reusable splitter"):
        ModelSelector(
            Ridge(), n_features_to_select=[1, 2], cv=pairs, nested=True
        ).fit(X, y)
    ModelSelector(
        Ridge(), n_features_to_select=1, cv=pairs, nested=True, random_state=0
    ).fit(X, y)


class RequestedPurged(PurgedTimeSeriesSplit):
    calls: list[tuple[int, int, object]] = []

    def split(self, X, y=None, groups=None, *, time=None, event_end=None):
        self.calls.append((int(np.asarray(X).shape[0]), int(self.n_splits), self.embargo))
        yield from super().split(X, y, groups, time=time, event_end=event_end)


def test_stability_honors_configured_splitter_on_each_draw():
    rng = np.random.default_rng(51)
    X = rng.normal(size=(90, 4))
    y = 2 * X[:, 0] + 0.7 * X[:, 1] + rng.normal(size=90) * 0.05
    RequestedPurged.calls = []
    cv = RequestedPurged(n_splits=2, embargo=2)
    ModelSelector(
        Ridge(), method="stability", n_resamples=3, cv=cv, random_state=0
    ).fit(X, y, time=np.arange(90))
    assert RequestedPurged.calls
    assert all(n_splits == 2 and embargo == 2 for _, n_splits, embargo in RequestedPurged.calls)
    assert all(n_rows < 90 for n_rows, _, _ in RequestedPurged.calls)

    class RecordingKFold(KFold):
        n_splits_seen: list[int] = []

        def split(self, X, y=None, groups=None):
            RecordingKFold.n_splits_seen.append(int(self.n_splits))
            return super().split(X, y, groups)

    RecordingKFold.n_splits_seen = []
    ModelSelector(
        Ridge(),
        method="stability",
        n_resamples=2,
        cv=RecordingKFold(n_splits=3, shuffle=False),
        random_state=0,
    ).fit(X, y)
    assert RecordingKFold.n_splits_seen
    assert set(RecordingKFold.n_splits_seen) == {3}

    with pytest.raises((TypeError, ValueError)):
        ModelSelector(Ridge(), method="stability", n_resamples=2, cv="bad").fit(X, y)


def test_stability_rejects_precomputed_cv_that_cannot_be_reused():
    X, y = _regression_data(n=90, p=4, seed=51)
    pairs = [(np.arange(60), np.arange(60, 90)), (np.arange(40), np.arange(40, 60))]
    with pytest.raises(ValueError, match="reusable splitter"):
        ModelSelector(
            Ridge(), method="stability", n_resamples=2, cv=pairs, random_state=0
        ).fit(X, y)
    with pytest.raises(ValueError, match="reusable splitter"):
        ModelSelector(
            Ridge(),
            method="stability",
            n_resamples=2,
            nested=True,
            cv=pairs,
            random_state=0,
        ).fit(X, y)


def test_nested_empty_stability_uses_weighted_dummy_baseline():
    from sklearn.dummy import DummyClassifier, DummyRegressor

    rng = np.random.default_rng(51)
    X = rng.normal(size=(90, 4))
    noise = rng.normal(size=90)
    weights = np.linspace(0.4, 1.8, 90)
    cv = KFold(n_splits=3, shuffle=False)
    fitted = ModelSelector(
        Ridge(),
        method="stability",
        threshold=1.0,
        n_resamples=4,
        nested=True,
        cv=cv,
        random_state=0,
    ).fit(X, noise, sample_weight=weights)
    assert fitted.n_features_selected_ == 0
    assert fitted.nested_scores_ is not None
    notes = fitted.nested_fold_diagnostics_
    assert notes is not None
    assert fitted.result_view().diagnostics["nested_fold_diagnostics"] == notes
    expected = []
    for fold, (train, val) in enumerate(cv.split(X, noise)):
        assert notes[fold]["empty_selection"] is True
        assert notes[fold]["baseline"] == "DummyRegressor"
        dummy = DummyRegressor(strategy="mean")
        dummy.fit(
            np.zeros((train.size, 0)),
            noise[train],
            sample_weight=weights[train],
        )
        expected.append(
            dummy.score(
                np.zeros((val.size, 0)),
                noise[val],
                sample_weight=weights[val],
            )
        )
    np.testing.assert_allclose(fitted.nested_scores_, expected)

    y_cls = (noise > np.median(noise)).astype(int)
    classif = ModelSelector(
        LogisticRegression(max_iter=200),
        method="stability",
        threshold=1.0,
        n_resamples=8,
        nested=True,
        cv=cv,
        random_state=0,
    ).fit(X, y_cls, sample_weight=weights)
    assert classif.n_features_selected_ == 0
    cls_expected = []
    for fold, (train, val) in enumerate(cv.split(X, y_cls)):
        assert classif.nested_fold_diagnostics_[fold]["baseline"] == "DummyClassifier"
        dummy = DummyClassifier(strategy="prior")
        dummy.fit(
            np.zeros((train.size, 0)),
            y_cls[train],
            sample_weight=weights[train],
        )
        cls_expected.append(
            dummy.score(
                np.zeros((val.size, 0)),
                y_cls[val],
                sample_weight=weights[val],
            )
        )
    np.testing.assert_allclose(classif.nested_scores_, cls_expected)


def test_fit_cv_snapshot_keeps_known_kfold_fields():
    X, y = _regression_data(n=40, p=3)
    cv = KFold(n_splits=2, shuffle=True, random_state=7)
    fitted = ModelSelector(
        Ridge(), n_features_to_select=[1, 2], cv=cv, random_state=0
    ).fit(X, y)
    snap = fitted.result_view().metadata["configured_options"]["cv"]
    assert snap["status"] == "params"
    assert snap["params"]["n_splits"] == 2
    assert snap["params"]["shuffle"] is True
    assert snap["params"]["random_state"] == 7
    other = ModelSelector(
        Ridge(),
        n_features_to_select=[1, 2],
        cv=KFold(n_splits=5, shuffle=True, random_state=99),
        random_state=0,
    ).fit(X, y)
    other_snap = other.result_view().metadata["configured_options"]["cv"]
    assert other_snap["params"]["n_splits"] == 5
    assert other_snap["params"]["random_state"] == 99
    assert snap != other_snap
    fitted.set_params(cv=KFold(n_splits=5, shuffle=True, random_state=99))
    cv.n_splits = 5
    after = fitted.result_view().metadata["configured_options"]["cv"]
    assert after == snap
    assert after["params"]["n_splits"] == 2


def test_purged_cv_snapshot_captures_constructor_policy():
    import json

    X, y = _regression_data(n=36, p=3)
    time = np.arange(36)
    first = PurgedTimeSeriesSplit(n_splits=2, embargo=3, mode="forward")
    days = np.arange("2020-01-01", "2020-02-06", dtype="datetime64[D]")
    second = PurgedTimeSeriesSplit(
        n_splits=2,
        embargo=np.timedelta64(1, "D"),
        mode="purged_kfold",
        test_size=4,
        max_train_size=10,
    )
    fitted = ModelSelector(
        Ridge(), n_features_to_select=[1, 2], cv=first, random_state=0
    ).fit(X, y, time=time)
    snap_a = fitted.result_view().metadata["configured_options"]["cv"]
    snap_b = (
        ModelSelector(Ridge(), n_features_to_select=[1, 2], cv=second, random_state=0)
        .fit(X, y, time=days)
        .result_view()
        .metadata["configured_options"]["cv"]
    )
    json.dumps(snap_a, allow_nan=False)
    json.dumps(snap_b, allow_nan=False)
    assert snap_a["params"]["n_splits"] == 2
    assert snap_a["params"]["embargo"] == 3
    assert snap_a["params"]["mode"] == "forward"
    assert snap_a["params"]["max_train_size"] is None
    assert snap_a["params"]["test_size"] is None
    assert snap_b["params"]["mode"] == "purged_kfold"
    assert snap_b["params"]["test_size"] == 4
    assert snap_b["params"]["max_train_size"] == 10
    assert isinstance(snap_b["params"]["embargo"], str)
    assert snap_b["params"]["embargo"].startswith("duration:")
    assert snap_a != snap_b
    first.embargo = 99
    first.mode = "purged_kfold"
    fitted.set_params(cv=second)
    after = fitted.result_view().metadata["configured_options"]["cv"]
    assert after == snap_a
    assert after["params"]["embargo"] == 3
    groups = np.repeat(np.arange(6), 6)
    grouped = GroupPurgedTimeSeriesSplit(n_splits=2, embargo=1, max_train_size=8)
    snap_g = (
        ModelSelector(Ridge(), n_features_to_select=[1, 2], cv=grouped, random_state=0)
        .fit(X, y, time=time, groups=groups)
        .result_view()
        .metadata["configured_options"]["cv"]
    )
    assert snap_g["params"]["embargo"] == 1
    assert snap_g["params"]["max_train_size"] == 8
    assert "GroupPurgedTimeSeriesSplit" in snap_g["type"]


def _reordered_ridge_frame():
    rng = np.random.default_rng(17)
    frame = pd.DataFrame(rng.normal(size=(120, 2)), columns=["signal", "noise"])
    y = 10.0 * frame["signal"]
    transformer = FunctionTransformer(
        lambda x: x.iloc[:, ::-1],
        feature_names_out=lambda self, names: names[::-1],
    )
    pipe = Pipeline(
        [("reverse", transformer), ("ridge", Ridge(alpha=1e-6))],
    )
    return frame, y, pipe


def test_model_selector_rejects_same_width_reordered_pipeline_coefficients():
    frame, y, pipe = _reordered_ridge_frame()
    with pytest.raises(ValueError, match="importance='permutation'"):
        ModelSelector(pipe, n_features_to_select=1, importance="auto").fit(frame, y)
    selected = (
        ModelSelector(
            pipe, n_features_to_select=1, importance="permutation", random_state=0
        )
        .fit(frame, y)
        .selected_features_
    )
    assert selected == ["signal"]


def test_model_selector_standardscaler_pipeline_keeps_auto_importance():
    frame, y, _ = _reordered_ridge_frame()
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=1e-6))
    selected = (
        ModelSelector(pipe, n_features_to_select=1, importance="auto")
        .fit(frame, y)
        .selected_features_
    )
    assert selected == ["signal"]


def test_model_selector_alignment_guard_covers_search_cv_and_pca():
    frame, y, pipe = _reordered_ridge_frame()
    wrapped = GridSearchCV(pipe, {"ridge__alpha": [1e-6, 1e-5]}, cv=2)
    with pytest.raises(ValueError, match="importance='permutation'"):
        ModelSelector(wrapped, n_features_to_select=1, importance="auto").fit(frame, y)
    mixed = make_pipeline(PCA(n_components=2), Ridge(alpha=1e-6))
    with pytest.raises(ValueError, match="callable"):
        ModelSelector(mixed, n_features_to_select=1, importance="auto").fit(frame, y)


def _nested_final_reverse():
    reverse = FunctionTransformer(
        lambda x: np.asarray(x)[:, ::-1],
        feature_names_out=lambda self, names: names[::-1],
    )
    return make_pipeline(reverse, Ridge(alpha=1e-6))


def test_model_selector_rejects_nested_final_reordered_pipeline_coefficients():
    frame, y, _ = _reordered_ridge_frame()
    inner = _nested_final_reverse()
    outer = Pipeline([("scale", StandardScaler()), ("inner", inner)])
    one_step = Pipeline([("inner", inner)])
    for estimator in (outer, one_step):
        with pytest.raises(ValueError, match="importance='permutation'"):
            ModelSelector(estimator, n_features_to_select=1, importance="auto").fit(
                frame, y
            )
    selected = (
        ModelSelector(
            outer, n_features_to_select=1, importance="permutation", random_state=0
        )
        .fit(frame, y)
        .selected_features_
    )
    assert selected == ["signal"]


def test_model_selector_nested_final_scaler_pipeline_keeps_auto_importance():
    frame, y, _ = _reordered_ridge_frame()
    outer = Pipeline(
        [
            ("scale", StandardScaler()),
            ("inner", make_pipeline(StandardScaler(), Ridge(alpha=1e-6))),
        ]
    )
    selected = (
        ModelSelector(outer, n_features_to_select=1, importance="auto")
        .fit(frame, y)
        .selected_features_
    )
    assert selected == ["signal"]
