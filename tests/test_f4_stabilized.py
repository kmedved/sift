"""Public-contract tests for the additive Stabilized meta-selector."""

from __future__ import annotations

import pickle
from packaging.version import Version

import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn import config_context
from sklearn.base import BaseEstimator, clone
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    SelectorMixin,
    VarianceThreshold,
    f_regression,
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted

from sift import CEFSPlusSelector, KnockoffSelector, Stabilized, as_result
from sift.selection.stabilized import _spawn_resample_rngs


SKLEARN_VERSION = Version(sklearn.__version__).release[:2]


class MeanSignSelector(SelectorMixin, BaseEstimator):
    """Select columns whose (weighted) mean is strictly above ``cutoff``."""

    def __init__(self, cutoff=0.0):
        self.cutoff = cutoff

    def fit(self, X, y=None, sample_weight=None, groups=None, time=None):
        values = np.asarray(X, dtype=np.float64)
        self.n_features_in_ = values.shape[1]
        if sample_weight is None:
            means = values.mean(axis=0)
        else:
            weights = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
            if weights.size != values.shape[0]:
                raise ValueError("sample_weight length mismatch")
            means = np.average(values, axis=0, weights=weights)
        if groups is not None and np.asarray(groups).reshape(-1).size != values.shape[0]:
            raise ValueError("groups length mismatch")
        if time is not None and np.asarray(time).reshape(-1).size != values.shape[0]:
            raise ValueError("time length mismatch")
        self.support_ = means > float(self.cutoff)
        return self

    def _get_support_mask(self):
        return np.asarray(self.support_, dtype=bool)


class BoomOnSmallN(SelectorMixin, BaseEstimator):
    def fit(self, X, y=None):
        values = np.asarray(X, dtype=np.float64)
        if values.shape[0] < 5:
            raise ValueError("too few rows")
        self.n_features_in_ = values.shape[1]
        self.support_ = np.ones(values.shape[1], dtype=bool)
        return self

    def _get_support_mask(self):
        return np.asarray(self.support_, dtype=bool)


class EmptySelector(SelectorMixin, BaseEstimator):
    def fit(self, X, y=None):
        values = np.asarray(X)
        self.n_features_in_ = values.shape[1]
        self.support_ = np.zeros(values.shape[1], dtype=bool)
        return self

    def _get_support_mask(self):
        return np.asarray(self.support_, dtype=bool)


def _oracle_frequencies(X, *, n_resamples, random_state, resample, sample_frac, cutoff=0.0):
    n, p = X.shape
    rngs = _spawn_resample_rngs(random_state, n_resamples)
    counts = np.zeros(p, dtype=np.int64)
    for rng in rngs:
        if resample == "half":
            size = max(1, min(n, int(n * sample_frac)))
            idx = rng.choice(n, size=size, replace=False)
        else:
            size = max(1, int(round(n * sample_frac)))
            idx = rng.choice(n, size=size, replace=True)
        means = X[idx].mean(axis=0)
        counts += means > cutoff
    return (counts / float(n_resamples)).astype(np.float64)


def test_seeded_half_frequencies_match_manual_oracle():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))
    X[:, 0] += 1.5
    X[:, 3] -= 1.5
    y = rng.normal(size=40)
    fitted = Stabilized(
        MeanSignSelector(),
        n_resamples=12,
        resample="half",
        threshold=0.6,
        random_state=7,
        verbose=False,
    ).fit(X, y)
    expected = _oracle_frequencies(
        X, n_resamples=12, random_state=7, resample="half", sample_frac=0.5
    )
    np.testing.assert_array_equal(fitted.selection_frequencies_, expected)
    assert fitted.selection_frequencies_.dtype == np.float64
    keep = np.flatnonzero(expected >= 0.6)
    order = np.argsort(-expected[keep], kind="mergesort")
    assert list(fitted.selected_features_) == [
        fitted.feature_names_in_[int(i)] for i in keep[order]
    ]


def test_bootstrap_draws_with_replacement_and_half_does_not():
    n = 24
    rngs_half = _spawn_resample_rngs(3, 6)
    rngs_boot = _spawn_resample_rngs(3, 6)
    half_unique = []
    boot_has_duplicate = False
    for rng in rngs_half:
        idx = rng.choice(n, size=max(1, int(n * 0.5)), replace=False)
        half_unique.append(len(set(idx.tolist())) == len(idx))
    for rng in rngs_boot:
        idx = rng.choice(n, size=n, replace=True)
        boot_has_duplicate = boot_has_duplicate or len(set(idx.tolist())) < len(idx)
    assert all(half_unique)
    assert boot_has_duplicate

    rng = np.random.default_rng(1)
    X = np.column_stack([np.arange(n, dtype=float), rng.normal(size=n)])
    y = rng.normal(size=n)

    class DuplicateFlagSelector(SelectorMixin, BaseEstimator):
        def fit(self, X, y=None):
            values = np.asarray(X)
            ids, counts = np.unique(values[:, 0], return_counts=True)
            self.n_features_in_ = values.shape[1]
            self.support_ = np.array([True, bool(np.any(counts > 1))], dtype=bool)
            return self

        def _get_support_mask(self):
            return np.asarray(self.support_, dtype=bool)

    half = Stabilized(
        DuplicateFlagSelector(),
        n_resamples=8,
        resample="half",
        threshold=0.0,
        random_state=3,
        verbose=False,
    ).fit(X, y)
    boot = Stabilized(
        DuplicateFlagSelector(),
        n_resamples=8,
        resample="bootstrap",
        threshold=0.0,
        random_state=3,
        verbose=False,
    ).fit(X, y)
    assert half.selection_frequencies_[1] == 0.0
    assert boot.selection_frequencies_[1] > 0.0


def test_sample_weight_and_metadata_are_sliced_with_rows():
    n = 30
    row_id = np.arange(n, dtype=np.float64)
    X = np.column_stack([row_id, np.zeros(n)])
    y = np.zeros(n)
    weights = 10.0 + row_id
    groups = row_id.copy()
    time = row_id.copy()

    class AlignedSelector(SelectorMixin, BaseEstimator):
        def fit(self, X, y=None, sample_weight=None, groups=None, time=None):
            values = np.asarray(X, dtype=np.float64)
            self.n_features_in_ = values.shape[1]
            weight = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
            group = np.asarray(groups).reshape(-1)
            clock = np.asarray(time).reshape(-1)
            aligned = (
                weight.size == values.shape[0]
                and np.allclose(weight, 10.0 + values[:, 0])
                and np.allclose(group, values[:, 0])
                and np.allclose(clock, values[:, 0])
            )
            self.support_ = np.array([aligned, False], dtype=bool)
            return self

        def _get_support_mask(self):
            return np.asarray(self.support_, dtype=bool)

    fitted = Stabilized(
        AlignedSelector(),
        n_resamples=8,
        resample="bootstrap",
        sample_frac=1.0,
        threshold=0.0,
        random_state=0,
        verbose=False,
    ).fit(X, y, sample_weight=weights, groups=groups, time=time)
    assert fitted._fit_used_sample_weight_ is True
    assert fitted._fit_used_groups_ is True
    assert fitted._fit_used_time_ is True
    assert fitted.selection_frequencies_[0] == 1.0


def test_raw_names_support_and_sklearn_selectkbest():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(80, 5)), columns=list("abcde"))
    y = X["a"] + 0.8 * X["b"] + 0.05 * rng.normal(size=80)
    fitted = Stabilized(
        SelectKBest(f_regression, k=2),
        n_resamples=10,
        resample="half",
        threshold=0.6,
        random_state=0,
        verbose=False,
    ).fit(X, y)
    assert fitted.selected_features_ == ["a", "b"]
    assert fitted.selected_indices_.tolist() == [0, 1]
    np.testing.assert_array_equal(
        fitted.get_support(), np.array([True, True, False, False, False])
    )
    transformed = fitted.transform(X)
    assert list(transformed.columns) == ["a", "b"]


def test_empty_selection_keeps_true_zero_frequencies():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    y = rng.normal(size=20)
    fitted = Stabilized(
        EmptySelector(),
        n_resamples=5,
        threshold=0.6,
        random_state=0,
        verbose=False,
    ).fit(X, y)
    np.testing.assert_array_equal(fitted.selection_frequencies_, np.zeros(3))
    assert fitted.selected_features_ == []
    assert fitted.selected_indices_.size == 0
    assert fitted.transform(X).shape == (20, 0)
    view = as_result(fitted)
    assert view.k == 0
    assert view.metadata["table_complete"] is True


def test_failed_refit_clears_fitted_state():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    y = rng.normal(size=20)
    selector = Stabilized(
        BoomOnSmallN(),
        n_resamples=4,
        resample="half",
        sample_frac=1.0,
        random_state=0,
        verbose=False,
    )
    selector.fit(X, y)
    check_is_fitted(selector, ["selection_frequencies_"])
    with pytest.raises(ValueError, match="too few rows"):
        selector.fit(X[:3], y[:3])
    with pytest.raises(Exception):
        check_is_fitted(selector, ["selection_frequencies_"])
    assert not hasattr(selector, "selected_features_")


def test_clone_pickle_and_pandas_output():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 4)), columns=list("abcd"))
    y = X["a"] + rng.normal(scale=0.1, size=40)
    selector = Stabilized(
        SelectKBest(f_regression, k=1),
        n_resamples=6,
        random_state=0,
        verbose=False,
    )
    cloned = clone(selector)
    fitted = cloned.fit(X, y)
    roundtrip = pickle.loads(pickle.dumps(fitted))
    np.testing.assert_array_equal(
        roundtrip.selection_frequencies_, fitted.selection_frequencies_
    )
    pandas_out = roundtrip.set_output(transform="pandas").transform(X)
    assert isinstance(pandas_out, pd.DataFrame)
    assert list(pandas_out.columns) == list(roundtrip.get_feature_names_out())


def test_opt_in_cluster_frequency_and_manifest_omits_matrices():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(50, 4)), columns=list("abcd"))
    X["b"] = X["a"] + 0.01 * rng.normal(size=50)
    X.loc[0, "c"] = 424242.424242
    y = X["a"] + rng.normal(scale=0.05, size=50)
    fitted = Stabilized(
        SelectKBest(f_regression, k=1),
        n_resamples=8,
        store_proxies=True,
        random_state=0,
        verbose=False,
    ).fit(X, y)
    view = fitted.result_view_
    clusters = view.proxy_clusters(r_min=0.8)
    assert view.metadata["cluster_frequencies_available"] is True
    assert "cluster_frequency" in clusters.columns
    manifest = view.reproducibility_()
    blob = str(manifest)
    assert "n_resamples" in manifest["configuration"]["configured"]
    assert manifest["configuration"]["seeds"]["random_state"] == 0
    assert "424242.424242" not in blob
    assert "selection_frequencies_" not in blob


def test_fixed_k_base_rejects_unused_groups_and_blocks_require_both():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))
    y = X[:, 0] + rng.normal(scale=0.1, size=40)
    groups = np.repeat(np.arange(8), 5)
    time = np.tile(np.arange(5), 8)
    selector = Stabilized(
        CEFSPlusSelector(k=1, verbose=False),
        n_resamples=3,
        resample="half",
        random_state=0,
        verbose=False,
    )
    with pytest.raises(ValueError, match="groups was supplied but not used"):
        selector.fit(X, y, groups=groups)
    with pytest.raises(ValueError, match="requires both groups and time"):
        Stabilized(
            MeanSignSelector(),
            resample="blocks",
            n_resamples=2,
            verbose=False,
        ).fit(X, y, groups=groups)
    fitted = Stabilized(
        MeanSignSelector(),
        n_resamples=4,
        resample="blocks",
        threshold=0.0,
        random_state=0,
        verbose=False,
    ).fit(X, y, groups=groups, time=time)
    assert fitted._fit_used_groups_ is True
    assert fitted._fit_used_time_ is True
    assert fitted.selection_frequencies_.dtype == np.float64


def test_atomic_blocks_stay_atomic_in_frequencies():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(80, 4)), columns=list("abcd"))
    X["b"] = X["a"] + 0.05 * rng.normal(size=80)
    y = X["a"] + X["b"] + rng.normal(scale=0.05, size=80)
    fitted = Stabilized(
        CEFSPlusSelector(k=1, feature_blocks={"ab": ["a", "b"]}, verbose=False),
        n_resamples=6,
        threshold=0.0,
        random_state=0,
        verbose=False,
    ).fit(X, y)
    np.testing.assert_array_equal(
        fitted.selection_frequencies_[:2],
        np.full(2, fitted.selection_frequencies_[0]),
    )


def test_evalues_mode_matches_native_knockoff_and_rejects_overrides():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(120, 8)), columns=[f"f{i}" for i in range(8)])
    y = X.iloc[:, :3].sum(axis=1).to_numpy() + 0.2 * rng.normal(size=120)
    native = KnockoffSelector(
        q=0.2,
        n_draws=4,
        aggregation="evalues",
        random_state=0,
        verbose=False,
        screen_pairs=None,
    ).fit(X, y)
    wrapped = Stabilized(
        KnockoffSelector(
            q=0.2,
            n_draws=4,
            aggregation="evalues",
            random_state=0,
            verbose=False,
            screen_pairs=None,
        ),
        n_resamples=4,
        aggregation="evalues",
        verbose=False,
    ).fit(X, y)
    assert list(wrapped.selected_features_) == list(native.selected_features_)
    np.testing.assert_array_equal(wrapped.selected_indices_, native.selected_indices_)
    assert wrapped.result_view_.metadata["aggregation"] == "evalues"
    assert wrapped.result_view_.metadata["fdr_control"] == native.result_.selector_metadata[
        "fdr_control"
    ]
    with pytest.raises(ValueError, match="does not resample rows"):
        Stabilized(
            KnockoffSelector(q=0.2, n_draws=4, aggregation="evalues", verbose=False),
            n_resamples=4,
            aggregation="evalues",
            resample="bootstrap",
            verbose=False,
        ).fit(X, y)
    with pytest.raises(TypeError, match="KnockoffSelector"):
        Stabilized(
            SelectKBest(f_regression, k=2),
            aggregation="evalues",
            verbose=False,
        ).fit(X, y)


@pytest.mark.skipif(
    SKLEARN_VERSION < (1, 4),
    reason="cross_validate(params=...) requires sklearn 1.4+",
)
def test_group_metadata_routes_through_pipeline_when_base_consumes_it():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(6), 10)
    X = rng.normal(size=(60, 4))
    y = 2.0 * X[:, 0] + rng.normal(scale=0.2, size=60)
    selector = Stabilized(
        MeanSignSelector(),
        n_resamples=4,
        resample="half",
        threshold=0.0,
        random_state=0,
        verbose=False,
    )
    with config_context(enable_metadata_routing=True):
        selector.set_fit_request(groups=True)
        result = cross_validate(
            make_pipeline(selector, Ridge()),
            X,
            y,
            cv=GroupKFold(3),
            params={"groups": groups},
            error_score="raise",
        )
    assert np.isfinite(result["test_score"]).all()


def test_dataframe_feature_names_require_exact_full_order():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(80, 4)), columns=list("abcd"))
    y = 3 * X.a + 1.5 * X.c + 0.1 * rng.normal(size=80)
    sel = Stabilized(
        CEFSPlusSelector(k=2, verbose=False, subsample=None),
        n_resamples=3,
        threshold=0.5,
        verbose=False,
    )
    with pytest.raises(ValueError, match="column order"):
        sel.fit(X, y, feature_names=["c", "a", "b", "d"])
    with pytest.raises(ValueError, match="one name per column"):
        sel.fit(X, y, feature_names=["a"])
    with pytest.raises(ValueError, match="one name per column"):
        sel.fit(X, y, feature_names=[])
    fitted = sel.fit(X, y, feature_names=["a", "b", "c", "d"])
    assert fitted.selected_features_ == [
        X.columns[int(i)] for i in fitted.selected_indices_
    ]


def test_ndarray_feature_names_reach_sift_bases_and_evalue_view():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(90, 4))
    y = X[:, 0] + 0.8 * X[:, 1] + 0.1 * rng.normal(size=90)
    names = list("abcd")
    fitted = Stabilized(
        CEFSPlusSelector(k=1, include=["a"], verbose=False, subsample=None),
        n_resamples=3,
        threshold=0.0,
        random_state=0,
        verbose=False,
    ).fit(X, y, feature_names=names)
    assert "a" in fitted.selected_features_
    blocked = Stabilized(
        CEFSPlusSelector(
            k=1, feature_blocks={"ab": ["a", "b"]}, verbose=False, subsample=None
        ),
        n_resamples=3,
        threshold=0.0,
        random_state=0,
        verbose=False,
    ).fit(X, y, feature_names=names)
    assert set(blocked.selected_features_) <= set(names)

    Xe = rng.normal(size=(160, 8))
    ye = Xe[:, :3].sum(axis=1) + 0.2 * rng.normal(size=160)
    e_names = [f"name{i}" for i in range(8)]
    wrapped = Stabilized(
        KnockoffSelector(
            q=0.2,
            n_draws=3,
            aggregation="evalues",
            random_state=0,
            verbose=False,
            screen_pairs=None,
        ),
        n_resamples=3,
        aggregation="evalues",
        verbose=False,
    ).fit(Xe, ye, feature_names=e_names)
    assert all(name in e_names for name in wrapped.selected_features_)
    view = as_result(wrapped)
    assert view.features == list(wrapped.selected_features_)
    assert view.indices == wrapped.selected_indices_.tolist()


def test_evalue_frequencies_reindex_dropped_constant_columns():
    rng = np.random.default_rng(11)
    X = pd.DataFrame(rng.normal(size=(200, 20)), columns=[f"f{i}" for i in range(20)])
    X["f7"] = 1.0
    y = X.iloc[:, :6].sum(axis=1).to_numpy() + 0.2 * rng.normal(size=200)
    native = KnockoffSelector(
        q=0.5,
        n_draws=5,
        aggregation="evalues",
        random_state=11,
        verbose=False,
        screen_pairs=None,
    ).fit(X, y)
    wrapped = Stabilized(
        KnockoffSelector(
            q=0.5,
            n_draws=5,
            aggregation="evalues",
            random_state=11,
            verbose=False,
            screen_pairs=None,
        ),
        n_resamples=5,
        aggregation="evalues",
        verbose=False,
    ).fit(X, y)
    expected = np.zeros(20, dtype=np.float64)
    table = native.result_.W
    expected[np.asarray(table["selected_index"], dtype=np.int64)] = np.nan_to_num(
        np.asarray(table["selection_frequency"], dtype=np.float64),
        nan=0.0,
    )
    np.testing.assert_allclose(wrapped.selection_frequencies_, expected)
    assert wrapped.selection_frequencies_[7] == 0.0
    assert np.any(wrapped.selection_frequencies_ > 0.0)
    assert np.any((wrapped.selection_frequencies_ > 0.0) & (wrapped.selection_frequencies_ < 1.0))


def test_blocks_do_not_require_oob_rows_on_small_or_full_group_panels():
    rng = np.random.default_rng(0)
    X_small = rng.normal(size=(8, 3))
    y_small = X_small[:, 0] + rng.normal(scale=0.1, size=8)
    groups_small = np.repeat(np.arange(2), 4)
    time_small = np.tile(np.arange(4), 2)
    small = Stabilized(
        MeanSignSelector(),
        n_resamples=4,
        resample="blocks",
        threshold=0.0,
        random_state=0,
        verbose=False,
    ).fit(X_small, y_small, groups=groups_small, time=time_small)
    assert small.n_features_in_ == 3

    X = rng.normal(size=(40, 3))
    y = X[:, 0] + rng.normal(scale=0.1, size=40)
    groups = np.repeat(np.arange(2), 20)
    time = np.tile(np.arange(20), 2)
    full = Stabilized(
        MeanSignSelector(),
        n_resamples=3,
        resample="blocks",
        block_size=20,
        threshold=0.0,
        random_state=0,
        verbose=False,
    ).fit(X, y, groups=groups, time=time)
    assert full.selection_frequencies_.shape == (3,)


def test_unsupervised_variance_threshold_fit_and_fit_transform():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 4))
    X[:, 3] = 0.0
    selector = Stabilized(
        VarianceThreshold(threshold=1e-8),
        n_resamples=3,
        threshold=0.6,
        random_state=0,
        verbose=False,
    )
    fitted = selector.fit(X)
    assert fitted.selection_frequencies_[3] == 0.0
    transformed = Stabilized(
        VarianceThreshold(threshold=1e-8),
        n_resamples=3,
        threshold=0.6,
        random_state=0,
        verbose=False,
    ).fit_transform(X)
    assert transformed.shape[0] == 30
    with pytest.raises(ValueError, match="target y is None"):
        Stabilized(
            SelectKBest(f_regression, k=1),
            n_resamples=2,
            verbose=False,
        ).fit(X)
    with pytest.raises(ValueError, match="one-dimensional"):
        Stabilized(
            SelectKBest(f_regression, k=1),
            n_resamples=2,
            verbose=False,
        ).fit(X, 1.0)


def test_tags_follow_the_base_selector():
    kbest = Stabilized(SelectKBest(f_regression, k=1), verbose=False)
    kbest_tags = kbest._more_tags()
    assert kbest_tags.get("allow_nan") is False
    variance = Stabilized(VarianceThreshold(), verbose=False)
    assert variance._more_tags().get("requires_y") is False
    knockoff = Stabilized(KnockoffSelector(verbose=False), verbose=False)
    assert knockoff._more_tags().get("non_deterministic") is True
    X = np.array([[1.0, np.nan], [2.0, np.nan], [4.0, np.nan], [6.0, np.nan]])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="NaN"):
        Stabilized(
            SelectKBest(f_regression, k=1),
            n_resamples=2,
            verbose=False,
        ).fit(X, y)


def test_manifest_records_actual_seeds_rows_and_route_identity():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 8)), columns=[f"f{i}" for i in range(8)])
    y = X.iloc[:, :3].sum(axis=1).to_numpy() + 0.2 * rng.normal(size=200)
    evalue = Stabilized(
        KnockoffSelector(
            q=0.5,
            n_draws=4,
            aggregation="evalues",
            random_state=42,
            subsample=80,
            verbose=False,
            screen_pairs=None,
        ),
        n_resamples=4,
        aggregation="evalues",
        verbose=False,
    ).fit(X, y)
    view = as_result(evalue)
    manifest = view.reproducibility_()
    assert manifest["configuration"]["configured"]["selector"] == "stabilized"
    assert manifest["configuration"]["configured"]["base_selector"]["type"].endswith(
        "KnockoffSelector"
    )
    assert manifest["configuration"]["seeds"]["random_state"] == 42
    assert view.diagnostics["rng"] == "KnockoffSelector.random_state"
    assert "resample" not in manifest["configuration"]["effective"]
    assert manifest["input"]["n_rows"] == 200
    assert manifest["input"]["n_rows_used"] == 80

    Xf = rng.normal(size=(100, 4))
    yf = Xf[:, 0] + rng.normal(scale=0.1, size=100)
    frequency = Stabilized(
        MeanSignSelector(),
        n_resamples=1,
        resample="half",
        sample_frac=0.2,
        threshold=0.0,
        random_state=0,
        verbose=False,
    ).fit(Xf, yf)
    freq_view = as_result(frequency)
    freq_manifest = freq_view.reproducibility_()
    assert freq_manifest["input"]["n_rows"] == 100
    assert freq_manifest["input"]["n_rows_used"] is None
    assert freq_manifest["input"]["n_rows_used_source"] == "unknown"
    assert freq_view.diagnostics["resample_n_rows"] == [20]
    assert freq_view.diagnostics["rng"] == "numpy.random.SeedSequence.spawn"


def test_rfe_and_select_from_model_forward_sliced_sample_weight():
    rng = np.random.default_rng(0)
    n = 40
    row_id = np.arange(n, dtype=np.float64)
    X = np.column_stack([row_id, rng.normal(size=(n, 3))])
    y = X[:, 1] + rng.normal(scale=0.1, size=n)
    weights = 10.0 + row_id
    seen = []

    class RecordingLinear(LinearRegression):
        def fit(self, X, y, sample_weight=None):
            values = np.asarray(X, dtype=np.float64)
            weight = None if sample_weight is None else np.asarray(
                sample_weight, dtype=np.float64
            ).reshape(-1)
            seen.append((values[:, 0].copy(), weight))
            return super().fit(X, y, sample_weight=sample_weight)

    Stabilized(
        RFE(RecordingLinear(), n_features_to_select=2),
        n_resamples=3,
        resample="bootstrap",
        sample_frac=1.0,
        threshold=0.0,
        random_state=0,
        verbose=False,
    ).fit(X, y, sample_weight=weights)
    assert seen
    aligned = 0
    for ids, weight in seen:
        assert weight is not None
        assert weight.shape[0] == ids.shape[0]
        if np.allclose(weight, 10.0 + ids):
            aligned += 1
    assert aligned >= 3

    seen.clear()

    class RecordingLasso(Lasso):
        def fit(self, X, y, sample_weight=None, **kwargs):
            values = np.asarray(X, dtype=np.float64)
            weight = None if sample_weight is None else np.asarray(
                sample_weight, dtype=np.float64
            ).reshape(-1)
            seen.append((values[:, 0].copy(), weight))
            return super().fit(X, y, sample_weight=sample_weight, **kwargs)

    lasso = Stabilized(
        SelectFromModel(RecordingLasso(alpha=0.05, max_iter=5000), max_features=2),
        n_resamples=3,
        resample="half",
        threshold=0.0,
        random_state=0,
        verbose=False,
    ).fit(X, y, sample_weight=weights)
    assert lasso._fit_used_sample_weight_ is True
    assert seen
    for ids, weight in seen:
        assert weight is not None
        assert weight.shape[0] == ids.shape[0]
        np.testing.assert_allclose(weight, 10.0 + ids)
    with pytest.raises(ValueError, match="does not accept"):
        Stabilized(
            SelectKBest(f_regression, k=1),
            n_resamples=2,
            verbose=False,
        ).fit(X, y, sample_weight=weights)


def test_generic_ndarray_keeps_mixed_hashable_names():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    y = X[:, 0] + 0.1 * rng.normal(size=80)
    names = [10, "b", "c"]
    fitted = Stabilized(
        SelectKBest(f_regression, k=1),
        n_resamples=4,
        threshold=0.5,
        random_state=0,
        verbose=False,
    ).fit(X, y, feature_names=names)
    assert list(fitted.feature_names_in_) == names
    assert fitted.selected_features_ == [10]
    out = fitted.transform(X)
    assert out.shape == (80, 1)
    view = as_result(fitted)
    assert view.features == [10]
    assert view.indices == [0]


def test_manifest_uses_fit_time_configuration_not_live_objects():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(80, 4)), columns=list("abcd"))
    y = X["a"] + 0.1 * rng.normal(size=80)
    base = CEFSPlusSelector(k=1, verbose=False, subsample=None, random_state=17)
    fitted = Stabilized(
        base,
        n_resamples=3,
        threshold=0.5,
        random_state=0,
        verbose=False,
    ).fit(X, y)
    first = list(fitted.selected_features_)
    manifest = as_result(fitted).reproducibility_()
    assert manifest["configuration"]["configured"]["base_selector"]["params"]["k"] == 1
    assert (
        manifest["configuration"]["configured"]["base_selector"]["params"]["random_state"]
        == 17
    )
    assert manifest["configuration"]["configured"]["n_resamples"] == 3
    base.set_params(k=3, random_state=999)
    fitted.set_params(n_resamples=9, random_state=4)
    later = as_result(fitted).reproducibility_()
    assert later["configuration"]["configured"]["base_selector"]["params"]["k"] == 1
    assert (
        later["configuration"]["configured"]["base_selector"]["params"]["random_state"]
        == 17
    )
    assert later["configuration"]["configured"]["n_resamples"] == 3
    assert later["configuration"]["configured"]["random_state"] == 0
    assert list(fitted.selected_features_) == first


def test_frequency_n_rows_used_is_base_consumption_not_draw_size():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 5)), columns=list("abcde"))
    y = X["a"] + 0.1 * rng.normal(size=200)
    fitted = Stabilized(
        CEFSPlusSelector(k=1, subsample=40, verbose=False, random_state=0),
        n_resamples=1,
        resample="half",
        threshold=0.0,
        random_state=0,
        verbose=False,
    ).fit(X, y)
    view = as_result(fitted)
    manifest = view.reproducibility_()
    assert view.diagnostics["resample_n_rows"] == [100]
    assert manifest["input"]["n_rows"] == 200
    assert manifest["input"]["n_rows_used"] == 40
