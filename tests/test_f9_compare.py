"""F9 leakage-safe sift.compare contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import GroupKFold, KFold

from sift import (
    AutoKConfig,
    CEFSPlusBinarySelector,
    CEFSPlusSelector,
    CompareResult,
    KnockoffSelector,
    compare,
)


def _regression_frame(n=90, p=6, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"x{i}" for i in range(p)])
    y = 2.2 * X["x0"] + 1.4 * X["x1"] + 0.15 * rng.normal(size=n)
    groups = np.repeat(np.arange(6), n // 6)[:n]
    return X, y, groups


def test_compare_cv_does_not_fit_on_held_out_rows_or_targets(monkeypatch):
    X, y, groups = _regression_frame()
    y_arr = np.asarray(y)
    selector_y = []
    estimator_y = []
    orig_sel = CEFSPlusSelector._fit_impl
    orig_est = Ridge.fit

    def spy_sel(self, X_fit, y_fit, **kwargs):
        selector_y.append(np.asarray(y_fit).reshape(-1).copy())
        return orig_sel(self, X_fit, y_fit, **kwargs)

    def spy_est(self, X_fit, y_fit, **kwargs):
        estimator_y.append(np.asarray(y_fit).reshape(-1).copy())
        return orig_est(self, X_fit, y_fit, **kwargs)

    monkeypatch.setattr(CEFSPlusSelector, "_fit_impl", spy_sel)
    monkeypatch.setattr(Ridge, "fit", spy_est)
    splitter = GroupKFold(n_splits=3)
    splits = list(splitter.split(X, y_arr, groups))
    train_ys = {tuple(y_arr[train_idx]) for train_idx, _val in splits}
    val_ys = {tuple(y_arr[val_idx]) for _train, val_idx in splits}
    result = compare(
        {"cefs": lambda: CEFSPlusSelector(k=2, verbose=False)},
        X,
        y_arr,
        estimator=Ridge(),
        cv=splitter,
        groups=groups,
    )
    assert result.mode == "cv"
    assert result.in_sample is False
    assert selector_y and estimator_y
    for seen in selector_y + estimator_y:
        key = tuple(seen)
        assert key in train_ys
        assert key not in val_ys


def test_compare_grouped_cefs_and_knockoff_and_empty_sets():
    X, y, groups = _regression_frame(n=120, p=8, seed=2)
    result = compare(
        {
            "cefs": lambda: CEFSPlusSelector(k="auto", verbose=False),
            "fdr": lambda: KnockoffSelector(q=0.2, random_state=0, verbose=False),
        },
        X,
        np.asarray(y),
        estimator=Ridge(),
        cv=GroupKFold(n_splits=3),
        groups=groups,
    )
    assert set(result.summary["selector"]) == {"cefs", "fdr"}
    assert result.k_unit in {"raw_features", "additional_blocks"}
    assert result.selection_identity == "raw_features"
    assert "mean_k" in result.summary.columns
    assert result.folds["train_index_sha256"].nunique() == 3
    assert {"selector_a", "selector_b", "mean_jaccard"} <= set(result.overlap.columns)
    assert result.prefix_scores.empty
    assert all(item["train_index_sha256"] for item in result.fold_bookkeeping)

    from sklearn.base import BaseEstimator, TransformerMixin

    class _EmptySelector(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None, **kwargs):
            n_features = int(np.asarray(X).shape[1])
            self.n_features_in_ = n_features
            cols = list(X.columns) if hasattr(X, "columns") else [f"x{i}" for i in range(n_features)]
            self.feature_names_in_ = np.asarray(cols, dtype=object)
            self.selected_features_ = []
            self.selected_indices_ = np.empty(0, dtype=np.int64)
            return self

        def transform(self, X):
            arr = np.asarray(X)
            return arr[:, :0]

    empty = compare(
        {"empty": lambda: _EmptySelector()},
        X,
        np.asarray(y),
        estimator=Ridge(),
        cv=KFold(n_splits=3, shuffle=True, random_state=0),
    )
    assert empty.scores["empty"].all()
    assert (empty.scores["k"] == 0).all()
    assert int(empty.summary["n_empty"].iloc[0]) == 3
    assert np.isfinite(empty.summary["score_mean"].iloc[0])


def test_compare_classification_and_sklearn_scorer():
    rng = np.random.default_rng(5)
    X = pd.DataFrame(rng.normal(size=(100, 5)), columns=list("abcde"))
    y = (X["a"] + 0.2 * rng.normal(size=100) > 0).astype(int)
    clf = compare(
        {"cefs": lambda: CEFSPlusBinarySelector(k=2, verbose=False)},
        X,
        y.to_numpy(),
        estimator=LogisticRegression(max_iter=200),
        cv=KFold(n_splits=3, shuffle=True, random_state=0),
        task="classification",
        scoring="accuracy",
    )
    assert clf.higher_is_better is True
    assert clf.scoring == "accuracy"
    assert np.isfinite(clf.summary["score_mean"].iloc[0])

    reg_scorer = compare(
        {"cefs": lambda: CEFSPlusSelector(k=2, verbose=False)},
        *_regression_frame(n=90, p=5, seed=6)[:2],
        estimator=Ridge(),
        cv=KFold(n_splits=3, shuffle=True, random_state=0),
        scoring=make_scorer(r2_score),
    )
    assert "sklearn:" in reg_scorer.scoring
    assert np.isfinite(reg_scorer.summary["score_mean"].iloc[0])


def test_compare_in_sample_path_is_labelled_and_fixed_k_rejects_groups(monkeypatch):
    X, y, groups = _regression_frame(n=90, p=5, seed=7)
    y_arr = np.asarray(y)
    seen_groups = []
    orig = CEFSPlusSelector.fit

    def spy(self, X_fit, y_fit, **kwargs):
        seen_groups.append("groups" in kwargs)
        return orig(self, X_fit, y_fit, **kwargs)

    monkeypatch.setattr(CEFSPlusSelector, "fit", spy)
    labelled = compare(
        {"cefs": lambda: CEFSPlusSelector(k=2, verbose=False)},
        X,
        y_arr,
        estimator=Ridge(),
        cv=GroupKFold(n_splits=3),
        groups=groups,
        mode="in_sample_path",
    )
    assert labelled.mode == "in_sample_path"
    assert labelled.in_sample is True
    assert labelled.diagnostics["in_sample"] is True
    assert not labelled.prefix_scores.empty
    assert labelled.prefix_scores["in_sample"].all()
    assert set(labelled.prefix_scores["mode"]) == {"in_sample_path"}
    assert labelled.scores["in_sample"].all()
    assert labelled.summary["in_sample"].all()
    assert not any(seen_groups)

    cv_fixed = compare(
        {"cefs": lambda: CEFSPlusSelector(k=2, verbose=False)},
        X,
        y_arr,
        estimator=Ridge(),
        cv=GroupKFold(n_splits=3),
        groups=groups,
    )
    assert cv_fixed.in_sample is False
    assert isinstance(labelled, CompareResult)


def test_compare_weights_are_fold_sliced(monkeypatch):
    X, y, _groups = _regression_frame(n=60, p=4, seed=8)
    y_arr = np.asarray(y)
    weights = np.linspace(0.5, 2.0, len(X))
    seen = []
    orig = CEFSPlusSelector._fit_impl

    def spy(self, X_fit, y_fit, **kwargs):
        seen.append(np.asarray(kwargs.get("sample_weight")).copy())
        return orig(self, X_fit, y_fit, **kwargs)

    monkeypatch.setattr(CEFSPlusSelector, "_fit_impl", spy)
    splitter = KFold(n_splits=3, shuffle=False)
    compare(
        {"cefs": lambda: CEFSPlusSelector(k=1, verbose=False)},
        X,
        y_arr,
        estimator=Ridge(),
        cv=splitter,
        sample_weight=weights,
    )
    for (train_idx, _val_idx), got in zip(splitter.split(X), seen):
        np.testing.assert_allclose(got, weights[train_idx])


def test_compare_reports_selectkbest_stability_and_block_units():
    rng = np.random.default_rng(19)
    X = pd.DataFrame(rng.normal(size=(90, 4)), columns=list("abcd"))
    y = 3.0 * X["a"] + 0.01 * rng.normal(size=90)
    from sklearn.feature_selection import SelectKBest, f_regression

    cv = KFold(n_splits=3, shuffle=True, random_state=0)
    result = compare(
        {
            "kbest": lambda: SelectKBest(f_regression, k=1),
            "cefs": lambda: CEFSPlusSelector(k=1, verbose=False),
        },
        X,
        np.asarray(y),
        estimator=Ridge(),
        cv=cv,
        scoring="neg_mse",
    )
    kbest = result.scores[result.scores["selector"] == "kbest"]
    assert not kbest["empty"].any()
    assert (kbest["k"] == 1).all()
    assert kbest["score"].mean() > -1.0

    y_b = 4.0 * X["b"] + 0.01 * rng.normal(size=90)
    from sift import StabilitySelector

    stab = compare(
        {
            "stab": lambda: StabilitySelector(
                alpha=0.1,
                n_bootstrap=3,
                threshold=0.3,
                max_features=1,
                random_state=0,
                verbose=False,
                n_jobs=1,
            )
        },
        X,
        np.asarray(y_b),
        estimator=Ridge(),
        cv=KFold(n_splits=2, shuffle=True, random_state=0),
    )
    freq = stab.selection_frequency
    assert float(freq.loc[freq["feature"] == "b", "frequency"].iloc[0]) > 0

    blocked = compare(
        {
            "blk": lambda: CEFSPlusSelector(
                k=1, feature_blocks={"ab": ["a", "b"]}, verbose=False
            )
        },
        X,
        np.asarray(y),
        estimator=Ridge(),
        cv=KFold(n_splits=3, shuffle=True, random_state=0),
    )
    assert (blocked.scores["k_unit"] == "additional_blocks").all()
    assert (blocked.scores["k"] == 1).all()
    assert (blocked.scores["n_columns"] == 2).all()


def test_compare_in_sample_path_keeps_encoding_and_atomic_blocks():
    rng = np.random.default_rng(21)
    n = 90
    X = pd.DataFrame(
        {
            "cat": np.array(["a", "b", "c"] * (n // 3), dtype=object),
            "num": rng.normal(size=n),
            "noise": rng.normal(size=n),
        }
    )
    y = 3.0 * (X["cat"] == "a").astype(float) + 0.1 * X["num"]
    out = compare(
        {"onehot": lambda: CEFSPlusSelector(k=1, cat_encoding="onehot", verbose=False)},
        X,
        np.asarray(y),
        estimator=Ridge(),
        cv=KFold(n_splits=3, shuffle=True, random_state=0),
        mode="in_sample_path",
    )
    assert np.isfinite(out.summary["score_mean"].iloc[0])
    assert (out.prefix_scores["in_sample"]).all()
    widths = set(out.prefix_scores.loc[out.prefix_scores["k"] == 1, "n_encoded_columns"])
    assert widths
    assert min(widths) >= 2

    Xn = pd.DataFrame(rng.normal(size=(90, 4)), columns=list("abcd"))
    yn = 3.0 * Xn["a"] + Xn["b"]
    blocked = compare(
        {
            "blk": lambda: CEFSPlusSelector(
                k=1, feature_blocks={"ab": ["a", "b"]}, verbose=False
            )
        },
        Xn,
        np.asarray(yn),
        estimator=Ridge(),
        cv=KFold(n_splits=3, shuffle=True, random_state=0),
        mode="in_sample_path",
    )
    k1 = blocked.prefix_scores[blocked.prefix_scores["k"] == 1]
    assert (k1["n_encoded_columns"] == 2).all()


def test_compare_forwards_accepted_row_context_and_scorer_direction():
    rng = np.random.default_rng(22)
    n = 60
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    y = X["a"] + 0.1 * rng.normal(size=n)
    groups = np.repeat(np.arange(6), 10)
    times = np.arange(n)
    seen = []
    from sklearn.base import BaseEstimator
    from sklearn.feature_selection import SelectorMixin

    class _Ctx(SelectorMixin, BaseEstimator):
        def fit(self, frame, target, sample_weight=None, groups=None, time=None):
            seen.append((groups is not None, time is not None))
            self.support_ = np.array([True, False, False])
            self._validate_data(frame, target)
            return self

        def _get_support_mask(self):
            return self.support_

    compare(
        {"ctx": lambda: _Ctx()},
        X,
        np.asarray(y),
        estimator=Ridge(),
        cv=GroupKFold(n_splits=3),
        groups=groups,
        time=times,
    )
    assert seen and all(seen)

    from sklearn.metrics import mean_absolute_error

    class _Good(SelectorMixin, BaseEstimator):
        def fit(self, frame, target, **kwargs):
            self.support_ = np.array([True, False, False])
            self._validate_data(frame, target)
            return self

        def _get_support_mask(self):
            return self.support_

    class _Bad(SelectorMixin, BaseEstimator):
        def fit(self, frame, target, **kwargs):
            self.support_ = np.array([False, False, True])
            self._validate_data(frame, target)
            return self

        def _get_support_mask(self):
            return self.support_

    ranked = compare(
        {"good": lambda: _Good(), "bad": lambda: _Bad()},
        X,
        np.asarray(y),
        estimator=Ridge(),
        cv=KFold(n_splits=3, shuffle=True, random_state=0),
        scoring=make_scorer(mean_absolute_error, greater_is_better=False),
    )
    assert ranked.higher_is_better is True
    means = ranked.summary.set_index("selector")["score_mean"]
    assert means.loc["good"] > means.loc["bad"]


def test_compare_empty_classifier_prior_and_json_labels():
    rng = np.random.default_rng(23)
    n = 60
    X = pd.DataFrame({"cat": np.array(["u", "v"] * (n // 2), dtype=object)})
    y = np.array([0, 0, 0, 1] * (n // 4))
    from sklearn.base import BaseEstimator
    from sklearn.feature_selection import SelectorMixin
    from sklearn.metrics import get_scorer

    class _Empty(SelectorMixin, BaseEstimator):
        def fit(self, frame, target, **kwargs):
            n_features = int(frame.shape[1])
            self.n_features_in_ = n_features
            self.support_ = np.zeros(n_features, dtype=bool)
            if hasattr(frame, "columns"):
                self.feature_names_in_ = np.asarray(list(frame.columns), dtype=object)
            return self

        def transform(self, X):
            return np.zeros((int(X.shape[0]), 0), dtype=np.float64)

        def _get_support_mask(self):
            return self.support_

    out = compare(
        {"empty": lambda: _Empty()},
        X,
        y,
        estimator=LogisticRegression(max_iter=200),
        cv=KFold(n_splits=3, shuffle=False),
        task="classification",
        scoring=get_scorer("neg_log_loss"),
    )
    assert out.scores["empty"].all()
    assert np.isfinite(out.summary["score_mean"].iloc[0])
    assert out.summary["score_mean"].iloc[0] > -2.0
    assert list(out.prefix_scores.columns) == list(
        __import__("sift.selection.compare", fromlist=["PREFIX_COLUMNS"]).PREFIX_COLUMNS
    )
    assert "in_sample" in out.folds.columns
    import json

    json.dumps(out.to_dict())


def test_compare_target_cv_matches_pipeline_oracle():
    rng = np.random.default_rng(28)
    n = 120
    levels = np.array(list("abcdefghijkl"))
    cat = rng.permutation(np.repeat(levels, n // len(levels)))[:n]
    X = pd.DataFrame({"cat": cat, "noise": rng.normal(size=n)})
    effect = pd.Series(cat).map({ch: i for i, ch in enumerate(levels)}).astype(float)
    y = 2.0 * effect.to_numpy() + rng.normal(size=n)
    cv = KFold(n_splits=3, shuffle=True, random_state=8)
    from sklearn.pipeline import make_pipeline

    factory = lambda: CEFSPlusSelector(k=1, cat_encoding="target_cv", verbose=False)
    compared = compare(
        {"cv": factory},
        X,
        y,
        estimator=Ridge(),
        cv=cv,
    )
    oracle = []
    for train_idx, val_idx in cv.split(X, y):
        pipe = make_pipeline(
            CEFSPlusSelector(k=1, cat_encoding="target_cv", verbose=False),
            Ridge(),
        )
        pipe.fit(X.iloc[train_idx], y[train_idx])
        oracle.append(float(pipe.score(X.iloc[val_idx], y[val_idx])))
    got = compared.scores.sort_values("split_id")["score"].to_numpy()
    np.testing.assert_allclose(got, oracle, rtol=1e-7, atol=1e-7)


def _prefix_path_frame():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(180, 4)), columns=list("abcd"))
    y = 5 * X.c + 0.7 * X.a + 0.01 * rng.normal(size=len(X))
    return X, y, KFold(3)


def test_compare_in_sample_path_follows_learned_order_not_output_order():
    X, y, cv = _prefix_path_frame()
    factory = lambda: CEFSPlusSelector(k=2, output_order="original", verbose=False)
    assert factory().fit(X, y).selected_features_ == ["c", "a"]
    result = compare(
        {"s": factory},
        X,
        y,
        estimator=Ridge(),
        cv=cv,
        mode="in_sample_path",
    )
    observed = result.prefix_scores.query("k == 1").score.to_numpy()
    expected = [
        r2_score(
            y.iloc[va],
            Ridge().fit(X.iloc[tr][["c"]], y.iloc[tr]).predict(X.iloc[va][["c"]]),
        )
        for tr, va in cv.split(X)
    ]
    np.testing.assert_allclose(observed, expected, atol=1e-10, rtol=1e-10)


def test_compare_in_sample_path_keeps_noncontiguous_blocks_atomic():
    X, y, cv = _prefix_path_frame()
    result = compare(
        {
            "s": lambda: CEFSPlusSelector(
                k=2,
                feature_blocks={"ac": ["a", "c"], "bd": ["b", "d"]},
                output_order="original",
                verbose=False,
            )
        },
        X,
        y,
        estimator=Ridge(),
        cv=cv,
        mode="in_sample_path",
    )
    assert set(result.prefix_scores.k) == {1, 2}
    assert (result.prefix_scores.query("k == 1").n_encoded_columns == 2).all()
    assert (result.prefix_scores.query("k == 2").n_encoded_columns == 4).all()


def test_compare_in_sample_path_integer_labels_are_not_positions():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(180, 3)), columns=[2, 0, 1])
    y = 5 * X[2] + 0.7 * X[0] + 0.01 * rng.normal(size=len(X))
    cv = KFold(3)
    for order in ("legacy", "original"):
        factory = lambda order=order: CEFSPlusSelector(
            k=2, output_order=order, verbose=False
        )
        assert factory().fit(X, y).selected_features_ == [2, 0]
        result = compare(
            {"s": factory},
            X,
            y,
            estimator=Ridge(),
            cv=cv,
            mode="in_sample_path",
        )
        assert set(result.prefix_scores.k) == {1, 2}
        assert (result.prefix_scores.query("k == 1").n_encoded_columns == 1).all()
        assert (result.prefix_scores.query("k == 2").n_encoded_columns == 2).all()
        np.testing.assert_allclose(
            result.prefix_scores.query("k == 2").score,
            result.scores.score,
            atol=1e-12,
            rtol=1e-12,
        )
        expected = [
            r2_score(
                y.iloc[va],
                Ridge().fit(X.iloc[tr][[2]], y.iloc[tr]).predict(X.iloc[va][[2]]),
            )
            for tr, va in cv.split(X)
        ]
        np.testing.assert_allclose(
            result.prefix_scores.query("k == 1").score,
            expected,
            atol=1e-10,
            rtol=1e-10,
        )


def test_compare_in_sample_path_scores_include_only_k0_prefix():
    X, y, cv = _prefix_path_frame()
    result = compare(
        {
            "s": lambda: CEFSPlusSelector(
                k="auto",
                include=["a", "c"],
                feature_blocks={"ac": ["a", "c"], "bd": ["b", "d"]},
                auto_k_config=AutoKConfig(
                    k_method="penalized_objective", min_k=0
                ),
                verbose=False,
            )
        },
        X,
        y,
        estimator=Ridge(),
        cv=cv,
        mode="in_sample_path",
    )
    assert (result.scores.k == 0).all()
    assert (result.prefix_scores.k == 0).all()
    assert (result.prefix_scores.n_encoded_columns == 2).all()
    np.testing.assert_allclose(
        result.prefix_scores.score, result.scores.score, atol=1e-12, rtol=1e-12
    )
