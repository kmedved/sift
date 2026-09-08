"""Regression coverage for row-resampling composition and proxy availability."""

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectFromModel, SelectorMixin, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor

from sift import (
    AutoKConfig,
    CEFSPlusSelector,
    JMISelector,
    KnockoffSelector,
    MRMRSelector,
    ModelSelector,
    Stabilized,
    StabilitySelector,
    as_result,
    build_cache,
    build_classic_cache,
)


class _CVWeightRecorder(SelectorMixin, BaseEstimator):
    records = []

    def __init__(self, cv=2):
        self.cv = cv

    def fit(self, X, y, *, sample_weight=None, groups=None, time=None):
        values = np.asarray(X)
        type(self).records.append(
            tuple(np.asarray(v).copy() if v is not None else None
                  for v in (values, y, sample_weight, groups, time))
        )
        self.n_features_in_ = values.shape[1]
        self.support_ = np.ones(self.n_features_in_, dtype=bool)
        return self

    def _get_support_mask(self):
        return self.support_


@pytest.mark.parametrize("weighted", [False, True])
def test_inner_cv_receives_unique_rows_with_actual_draw_multiplicity(weighted):
    n = 40
    X = np.column_stack([np.arange(n), np.sin(np.arange(n))])
    y = np.arange(n) + 100
    caller_weight = np.arange(n) + 1.0 if weighted else None
    groups, time = np.arange(n) + 200, np.arange(n) + 300
    _CVWeightRecorder.records = []
    fitted = Stabilized(
        _CVWeightRecorder(), n_resamples=3, resample="bootstrap",
        random_state=17, verbose=False,
    ).fit(X, y, sample_weight=caller_weight, groups=groups, time=time)
    for record, child in zip(
        _CVWeightRecorder.records, np.random.SeedSequence(17).spawn(3), strict=True
    ):
        draw = np.random.default_rng(child).choice(n, size=n, replace=True)
        unique = np.unique(draw)
        values, target, weights, group, times = record
        np.testing.assert_array_equal(values, X[unique])
        np.testing.assert_array_equal(target, y[unique])
        expected = np.bincount(draw, minlength=n)[unique].astype(float)
        if weighted:
            expected *= caller_weight[unique]
        np.testing.assert_array_equal(weights, expected)
        np.testing.assert_array_equal(group, groups[unique])
        np.testing.assert_array_equal(times, time[unique])
        assert len(unique) < len(draw)
    assert as_result(fitted).diagnostics["resample_fit_policy"] == (
        "unique_rows_with_multiplicity_weights"
    )


def test_real_model_selector_cv_never_sees_duplicate_source_rows():
    class UniqueRowsKFold(KFold):
        def split(self, X, y=None, groups=None):
            assert len(np.unique(np.asarray(X)[:, 0])) == len(X)
            yield from super().split(X, y, groups)

    rng = np.random.default_rng(3)
    X = np.column_stack([np.arange(40), rng.normal(size=(40, 2))])
    y = X[:, 0] + rng.normal(size=40)
    fitted = Stabilized(
        ModelSelector(LinearRegression(), cv=UniqueRowsKFold(2),
                      n_features_to_select=[1, 2]),
        n_resamples=2, resample="bootstrap", random_state=5, verbose=False,
    ).fit(X, y)
    assert fitted.selection_frequencies_.shape == (3,)


def test_unweighted_cv_base_restricted_to_half():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 3))
    y = X[:, 0]
    base = SequentialFeatureSelector(
        KNeighborsRegressor(2), n_features_to_select=1, cv=2
    )
    with pytest.raises(ValueError, match="inner-CV.*sample_weight"):
        Stabilized(base, n_resamples=1, resample="bootstrap", verbose=False).fit(X, y)
    assert Stabilized(base, n_resamples=1, verbose=False).fit(X, y).get_support().sum() == 1


@pytest.mark.parametrize("resample", ["bootstrap", "blocks"])
def test_ksg_inner_cv_reports_replacement_restriction_and_half_works(resample):
    rng = np.random.default_rng(903)
    X = rng.normal(size=(80, 3))
    y = X[:, 0] + rng.normal(size=80)
    time = np.arange(80)
    base = JMISelector(
        k="auto", estimator="ksg", task="regression", verbose=False,
        auto_k_config=AutoKConfig(
            k_method="evaluate", strategy="time_holdout", min_k=1, max_k=2, val_frac=.3
        ),
    )
    row_context = {"time": time}
    if resample == "blocks":
        row_context["groups"] = np.repeat(np.arange(8), 10)
    with pytest.raises(ValueError, match="resample='half'"):
        Stabilized(base, n_resamples=1, resample=resample, verbose=False).fit(
            X, y, **row_context
        )
    fitted = Stabilized(base, n_resamples=1, resample="half", verbose=False).fit(
        X, y, time=time
    )
    assert fitted.get_support().any()


@pytest.mark.parametrize("kind", ["gaussian", "classic"])
@pytest.mark.parametrize("resample", ["half", "bootstrap"])
def test_prebuilt_row_cache_rejected_before_resampling(kind, resample):
    rng = np.random.default_rng(9)
    X = rng.normal(size=(40, 3))
    y = X[:, 0]
    if kind == "gaussian":
        base = CEFSPlusSelector(k=1, cache=build_cache(X, subsample=None), verbose=False)
    else:
        base = MRMRSelector(k=1, cache=build_classic_cache(X, subsample=None), verbose=False)
    with pytest.raises(ValueError, match="prebuilt feature cache"):
        Stabilized(base, n_resamples=1, resample=resample, verbose=False).fit(X, y)


def test_nested_cache_rejected_and_full_data_evalues_cache_retained():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(80, 12))
    y = X[:, 0]
    cache = build_cache(X, subsample=None)
    nested = SelectFromModel(CEFSPlusSelector(k=1, cache=cache, verbose=False))
    with pytest.raises(ValueError, match="nested estimator cache"):
        Stabilized(nested, n_resamples=1, verbose=False).fit(X, y)
    base = KnockoffSelector(cache=cache, q=0.2, verbose=False)
    result = Stabilized(base, aggregation="evalues", n_resamples=2, verbose=False).fit(X, y)
    assert result.selection_frequencies_.shape == (12,)


class _SelectAll(SelectorMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def _get_support_mask(self):
        return np.ones(self.n_features_in_, dtype=bool)


@pytest.mark.parametrize("kind", ["stabilized", "stability"])
def test_selected_constants_only_restrict_requested_proxy_storage(kind):
    rng = np.random.default_rng(21)
    X = np.column_stack([np.ones(40), rng.normal(size=(40, 2))])
    y = X[:, 1] + rng.normal(size=40)
    if kind == "stabilized":
        base = Stabilized(_SelectAll(), n_resamples=1, verbose=False)
    else:
        base = StabilitySelector(n_bootstrap=2, threshold=0.0, random_state=3, verbose=False)
    fitted = base.fit(X, y)
    assert fitted.get_support()[0]
    with pytest.raises(ValueError, match="store_proxies=True.*constant"):
        base.set_params(store_proxies=True).fit(X, y)
