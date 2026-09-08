"""Public regressions for 0.9.1 audit stage-1 wrong-result/routing fixes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from sift import (
    AutoKConfig,
    ModelSelector,
    MRMRSelector,
    PurgedTimeSeriesSplit,
    GroupPurgedTimeSeriesSplit,
    compare,
    evaluate_feature_path,
    select_cefsplus,
    select_mrmr,
)
from sift.selection import filter_auto_k


class _OptionalTimeCV:
    """Splitter that declares optional time=None and doesونها consume it."""

    def split(self, X, y=None, groups=None, time=None):
        n = int(getattr(X, "shape", [len(X)])[0])
        mid = max(1, n // 2)
        yield np.arange(0, mid, dtype=np.int64), np.arange(mid, n, dtype=np.int64)

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1


class _NamedSelector(BaseEstimator, TransformerMixin):
    def fit(self, frame, target=None, **kwargs):
        self.n_features_in_ = int(frame.shape[1])
        self.feature_names_in_ = np.asarray(list(frame.columns), dtype=object)
        self.selected_indices_ = np.asarray([0, 1], dtype=np.int64)
        self.selected_features_ = [frame.columns[0], frame.columns[1]]
        return self

    def transform(self, frame):
        return np.asarray(frame, dtype=float)[:, self.selected_indices_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_features_, dtype=object)


def test_compare_forwards_time_to_purged_splitter():
    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    y = rng.normal(size=n)
    time = pd.date_range("2024-01-01", periods=n, freq="D").to_numpy()

    result = compare(
        {"s": lambda: _NamedSelector()},
        X,
        y,
        estimator=Ridge(),
        cv=PurgedTimeSeriesSplit(n_splits=3, embargo=pd.Timedelta("2D")),
        time=time,
    )
    assert len(result.folds) == 3
    assert (result.folds["n_train"] > 0).all()

    with pytest.raises(ValueError, match="PurgedTimeSeriesSplit requires time"):
        compare(
            {"s": lambda: _NamedSelector()},
            X,
            y,
            estimator=Ridge(),
            cv=PurgedTimeSeriesSplit(n_splits=3, embargo=pd.Timedelta("2D")),
        )

    groups = np.repeat(np.arange(5), n // 5)[:n]
    grouped = compare(
        {"s": lambda: _NamedSelector()},
        X,
        y,
        estimator=Ridge(),
        cv=GroupPurgedTimeSeriesSplit(n_splits=3),
        groups=groups,
        time=time,
    )
    assert len(grouped.folds) == 3


def test_optional_time_splitter_does_not_require_time():
    rng = np.random.default_rng(0)
    n = 40
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    y = X["a"].to_numpy() + 0.1 * rng.normal(size=n)
    compared = compare(
        {"s": lambda: _NamedSelector()},
        X,
        y,
        estimator=Ridge(),
        cv=_OptionalTimeCV(),
    )
    assert len(compared.folds) == 1
    assert int(compared.folds["n_train"].iloc[0]) == 20
    assert int(compared.folds["n_val"].iloc[0]) == 20

    evaluated = evaluate_feature_path(
        X,
        y,
        ["a", "b", "c"],
        [1, 2],
        estimator=Ridge(),
        splitter=_OptionalTimeCV(),
    )
    assert evaluated.k == [1, 2]
    assert evaluated.best_k in {1, 2}

    with pytest.raises(ValueError, match="PurgedTimeSeriesSplit requires time"):
        evaluate_feature_path(
            X,
            y,
            ["a", "b", "c"],
            [1, 2],
            estimator=Ridge(),
            splitter=PurgedTimeSeriesSplit(n_splits=3),
        )
    timed = evaluate_feature_path(
        X,
        y,
        ["a", "b", "c"],
        [1, 2],
        estimator=Ridge(),
        splitter=PurgedTimeSeriesSplit(n_splits=3),
        time=np.arange(n),
    )
    assert timed.k == [1, 2]
    assert np.isfinite(list(timed.scores.values())).all()


def test_compare_prefix_scores_path_order_not_transform_position():
    rng = np.random.default_rng(0)
    n = 180
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    y = (5 * X["d"] + 0.5 * X["a"] + 0.01 * rng.normal(size=n)).to_numpy()

    class PathSel(BaseEstimator, TransformerMixin):
        def fit(self, frame, target=None, **kwargs):
            self.n_features_in_ = int(frame.shape[1])
            self.feature_names_in_ = np.asarray(list(frame.columns), dtype=object)
            self.selected_indices_ = np.asarray([3, 0], dtype=np.int64)
            self.selected_features_ = ["d", "a"]
            return self

        def transform(self, frame):
            return np.asarray(frame, dtype=float)[:, [0, 3]]

        def get_feature_names_out(self, input_features=None):
            return np.asarray(["a", "d"], dtype=object)

    cv = KFold(3)
    result = compare(
        {"s": lambda: PathSel()},
        X,
        y,
        estimator=Ridge(),
        cv=cv,
        mode="in_sample_path",
    )
    got = result.prefix_scores.query("k == 1")["score"].to_numpy()
    expected = [
        r2_score(
            y[va],
            Ridge().fit(X.iloc[tr][["d"]], y[tr]).predict(X.iloc[va][["d"]]),
        )
        for tr, va in cv.split(X)
    ]
    np.testing.assert_allclose(got, expected, atol=1e-10, rtol=1e-10)

    class Nameless(PathSel):
        def get_feature_names_out(self, input_features=None):
            raise AttributeError("no names")

    with pytest.raises(ValueError, match="transformed-column identity"):
        compare(
            {"s": lambda: Nameless()},
            X,
            y,
            estimator=Ridge(),
            cv=cv,
            mode="in_sample_path",
        )


def test_conditioning_generators_match_lists():
    rng = np.random.default_rng(9)
    n, p = 120, 8
    Xa = rng.normal(size=(n, p))
    Xa[:, 1] = Xa[:, 0] * 0.8 + 0.5 * rng.normal(size=n)
    cols = [f"f{i}" for i in range(p)]
    X = pd.DataFrame(Xa, columns=cols)
    y = pd.Series(Xa[:, 0] + 0.8 * Xa[:, 2] + 0.5 * Xa[:, 4] + 0.3 * rng.normal(size=n))

    listed = select_cefsplus(X, y, k=2, exclude=["f2"], verbose=False)
    generated = select_cefsplus(X, y, k=2, exclude=(c for c in ["f2"]), verbose=False)
    assert generated == listed

    listed_inc = select_cefsplus(X, y, k=2, include=["f0"], verbose=False)
    generated_inc = select_cefsplus(X, y, k=2, include=(c for c in ["f0"]), verbose=False)
    assert generated_inc == listed_inc

    listed_cand = select_cefsplus(
        X, y, k=2, candidates=[c for c in cols if c != "f0"], verbose=False
    )
    generated_cand = select_cefsplus(
        X, y, k=2, candidates=filter(lambda c: c != "f0", cols), verbose=False
    )
    assert generated_cand == listed_cand

    cfg = AutoKConfig(k_method="penalized_objective", min_k=0, max_k=4)
    auto_list = select_cefsplus(
        X, y, k="auto", auto_k_config=cfg, exclude=["f2"], verbose=False
    )
    auto_gen = select_cefsplus(
        X, y, k="auto", auto_k_config=cfg, exclude=(c for c in ["f2"]), verbose=False
    )
    assert auto_gen == auto_list


def test_gaussian_elbow_keeps_additional_block_units_with_dropped_member():
    rng = np.random.default_rng(4)
    n, p = 200, 10
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    X["cst"] = 7.0
    y = pd.Series(X["f0"] + 0.9 * X["f1"] + 0.7 * X["f2"] + 0.3 * rng.normal(size=n))
    cfg = AutoKConfig(k_method="elbow", max_k=4, min_k=0)

    def _run(blocks, candidates):
        return select_mrmr(
            X,
            y,
            task="regression",
            estimator="gaussian",
            k="auto",
            auto_k_config=cfg,
            verbose=False,
            return_result=True,
            feature_blocks=blocks,
            candidates=candidates,
            random_state=0,
            subsample=None,
        )

    clean = _run({"g": ["f0", "f1"], "h": ["f2", "f3"]}, ["f0", "f1", "f2", "f3"])
    dropped = _run(
        {"g": ["f0", "f1", "cst"], "h": ["f2", "f3"]},
        ["f0", "f1", "cst", "f2", "f3"],
    )
    clean_curve = clean.diagnostics_["auto_k_curve"]["curve"]
    dropped_curve = dropped.diagnostics_["auto_k_curve"]["curve"]
    assert list(clean_curve["k"].astype(int)) == list(dropped_curve["k"].astype(int))
    np.testing.assert_allclose(
        clean_curve["criterion"].to_numpy(dtype=float),
        dropped_curve["criterion"].to_numpy(dtype=float),
        atol=1e-10,
        rtol=1e-10,
    )
    assert int(clean.selector_metadata["k"]) == int(dropped.selector_metadata["k"])
    assert list(clean.selector_metadata["selected_blocks"]) == list(
        dropped.selector_metadata["selected_blocks"]
    )


def test_purged_max_train_size_caps_after_purge():
    t = np.arange(40)
    X = np.zeros((40, 1))
    folds = list(
        PurgedTimeSeriesSplit(n_splits=3, max_train_size=5).split(
            X, time=t, event_end=t + 5
        )
    )
    assert len(folds) == 3
    for train, val in folds:
        assert train.size > 0
        assert val.size > 0
        assert len(np.unique(t[train])) <= 5
        assert np.max(t[train]) < np.min(t[val])


def test_model_selector_callable_nan_matches_extracted():
    rng = np.random.default_rng(0)
    n, p = 100, 4
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=list("abcd"))
    y = pd.Series(X["a"] * 3 + X["b"] * 2 + rng.normal(size=n) * 0.2)

    def imp_nan(est):
        values = np.abs(est.coef_).astype(float)
        values[-1] = np.nan
        return values

    class NanRidge(Ridge):
        @property
        def feature_importances_(self):
            values = np.abs(self.coef_).astype(float)
            values[-1] = np.nan
            return values

    callable_sel = ModelSelector(
        Ridge(), n_features_to_select=2, importance=imp_nan
    ).fit(X, y)
    extracted = ModelSelector(
        NanRidge(), n_features_to_select=2, importance="feature_importances"
    ).fit(X, y)
    assert callable_sel.selected_features_ == extracted.selected_features_
    np.testing.assert_array_equal(callable_sel.ranking_, extracted.ranking_)


@pytest.mark.parametrize("encoding", ["ordinal", "frequency", "onehot", "target_cv"])
def test_duplicate_labels_rejected_when_encoding(encoding):
    rng = np.random.default_rng(9)
    n = 80
    lev = rng.integers(0, 4, n)
    frame = pd.DataFrame(
        np.column_stack([rng.normal(size=n), rng.normal(size=n)]), columns=["a", "a"]
    )
    frame["c"] = [f"L{v}" for v in lev]
    y = pd.Series(rng.normal(size=n) + lev * 0.5)
    numeric = frame.drop(columns=["c"])
    ok = MRMRSelector(k=1).fit(numeric, y)
    assert len(ok.get_feature_names_out()) == 1
    with pytest.raises(ValueError, match="unique DataFrame column names"):
        MRMRSelector(
            k=2, cat_features=["c"], cat_encoding=encoding
        ).fit(frame, y)


def test_weighted_multi_target_auto_k_uses_supported_ebic_route():
    rng = np.random.default_rng(9)
    n, p = 200, 8
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    Y = np.column_stack(
        [
            X["f0"] + 0.4 * rng.normal(size=n),
            0.8 * X["f1"] + 0.4 * rng.normal(size=n),
        ]
    )
    weights = np.linspace(0.2, 5.0, n)
    result = select_cefsplus(
        X, Y, k="auto", sample_weight=weights, verbose=False, return_result=True
    )
    route = result.diagnostics_["auto_k"]["auto_routing"]
    assert route["chosen"] == "penalized_objective"
    assert route["objective_penalty"] == "ebic"
    assert route["reason"] == "heavy_weight_skew_multi_target_ebic"
    assert {"f0", "f1"} & set(result.selected_features)


def test_auto_dense_check_rejects_two_d_y_before_routed_selection(monkeypatch):
    rng = np.random.default_rng(0)
    n, p = 80, 8
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    Y = np.column_stack([X["f0"], X["f1"]])
    cfg = AutoKConfig(
        k_method="auto",
        min_k=0,
        max_k=4,
        auto_dense_check=True,
        auto_dense_min_k=0,
        auto_dense_min_frac=0.0,
    )
    routed = []
    progress = []

    def _forbid_routed(*args, **kwargs):
        routed.append(kwargs.get("auto_k_config"))
        raise AssertionError("routed selection ran before dense-check rejection")

    monkeypatch.setattr(filter_auto_k, "_run_gaussian_routed_path", _forbid_routed)
    with pytest.raises(ValueError, match="auto_dense_check is not supported for 2-D y"):
        select_cefsplus(
            X,
            Y,
            k="auto",
            auto_k_config=cfg,
            verbose=False,
            callback=lambda *args, **kwargs: progress.append((args, kwargs)),
        )
    assert routed == []
    assert progress == []
