import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_parameters_default_constructible

from sift import (
    BorutaSelector,
    CEFSPlusBinarySelector,
    CEFSPlusSelector,
    JMIMSelector,
    JMISelector,
    KnockoffSelector,
    MRMRSelector,
    StabilitySelector,
    build_cache,
)
import sift.selectors as selectors_mod
import sift.selection.auto_k_nested as auto_k_nested_module
from sift.selection.auto_k import AutoKConfig


@pytest.mark.parametrize(
    "selector_cls",
    [
        MRMRSelector,
        JMISelector,
        JMIMSelector,
        CEFSPlusSelector,
        CEFSPlusBinarySelector,
        KnockoffSelector,
        BorutaSelector,
        StabilitySelector,
    ],
)
def test_selector_parameters_are_default_constructible(selector_cls):
    selector = selector_cls()
    check_parameters_default_constructible(selector_cls.__name__, selector)


def test_selector_classes_default_to_no_categorical_encoding():
    for selector in (
        MRMRSelector(verbose=False),
        JMISelector(verbose=False),
        JMIMSelector(verbose=False),
        CEFSPlusSelector(verbose=False),
    ):
        assert selector.cat_encoding == "none"


def test_binary_nested_auto_k_scores_with_natural_weights(monkeypatch):
    rng = np.random.default_rng(19)
    X = pd.DataFrame(rng.normal(size=(80, 4)), columns=[f"f{i}" for i in range(4)])
    y = np.array([0] * 60 + [1] * 20)
    captured = {}

    class DummyNested:
        selected_k = 2
        diagnostics = pd.DataFrame({"k": [2], "score": [0.0]})

    def fake_select_k_nested(*args, sample_weight=None, **kwargs):
        captured["sample_weight"] = np.asarray(sample_weight, dtype=float)
        return DummyNested()

    def fake_fit_selector(self, X, y, **kwargs):
        self.feature_names_in_ = list(X.columns)
        self.n_features_in_ = X.shape[1]
        self.selected_features_ = ["f0", "f1"]
        self.selected_indices_ = np.array([0, 1])
        self.k_ = 2
        return self

    monkeypatch.setattr(selectors_mod, "select_k_nested", fake_select_k_nested)
    monkeypatch.setattr(CEFSPlusBinarySelector, "_fit_selector", fake_fit_selector)

    selector = CEFSPlusBinarySelector(
        k="auto",
        class_weight="balanced",
        auto_k_config=AutoKConfig(auto_k_mode="nested", min_k=1, max_k=3),
        verbose=False,
    )
    selector.fit(X, y, sample_weight=np.ones(len(y)))

    np.testing.assert_allclose(captured["sample_weight"], np.ones(len(y)))


@pytest.mark.parametrize(
    "selector_cls, kwargs",
    [
        (MRMRSelector, dict(k=2, task="regression", verbose=False)),
        (JMISelector, dict(k=2, task="regression", verbose=False)),
        (JMIMSelector, dict(k=2, task="regression", verbose=False)),
        (CEFSPlusSelector, dict(k=2, verbose=False)),
    ],
)
def test_selector_df_fit_transform_and_support(selector_cls, kwargs):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(120, 6)), columns=[f"f{i}" for i in range(6)])
    y = X["f0"] + 0.2 * X["f1"] + rng.normal(size=120) * 0.05

    selector = selector_cls(**kwargs)
    X_out = selector.fit_transform(X, y)

    assert isinstance(X_out, pd.DataFrame)
    assert X_out.shape[1] <= X.shape[1]
    assert X_out.shape[1] == len(selector.selected_features_)
    assert len(selector.selected_indices_) == len(selector.selected_features_)

    mask = selector.get_support()
    indices = selector.get_support(indices=True)
    assert mask.shape == (X.shape[1],)
    assert mask.dtype == bool
    assert np.issubdtype(indices.dtype, np.integer)
    assert np.array_equal(np.nonzero(mask)[0], indices)
    assert X_out.shape[1] == len(indices)
    assert list(X_out.columns) == [selector.feature_names_in_[i] for i in indices]


@pytest.mark.parametrize(
    "selector_cls, kwargs",
    [
        (MRMRSelector, dict(k=3, task="regression", verbose=False)),
        (JMISelector, dict(k=3, task="regression", verbose=False)),
        (JMIMSelector, dict(k=3, task="regression", verbose=False)),
        (CEFSPlusSelector, dict(k=3, verbose=False)),
    ],
)
def test_selector_ndarray_fit_transform_and_support(selector_cls, kwargs):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 5))
    y = X[:, 0] + 0.25 * X[:, 2] + rng.normal(size=150) * 0.1

    selector = selector_cls(**kwargs)
    X_out = selector.fit_transform(X, y)

    assert isinstance(X_out, np.ndarray)
    assert X_out.shape[1] == len(selector.selected_features_)
    assert X_out.shape[1] == len(selector.selected_indices_)

    mask = selector.get_support()
    indices = selector.get_support(indices=True)
    assert mask.shape == (X.shape[1],)
    assert np.array_equal(np.nonzero(mask)[0], indices)


@pytest.mark.parametrize(
    "selector_cls, kwargs",
    [
        (MRMRSelector, dict(k=2, task="regression", verbose=False)),
        (JMISelector, dict(k=2, task="regression", verbose=False)),
        (JMIMSelector, dict(k=2, task="regression", verbose=False)),
        (CEFSPlusSelector, dict(k=2, verbose=False)),
    ],
)
def test_selector_not_fitted_raises(selector_cls, kwargs):
    selector = selector_cls(**kwargs)
    with pytest.raises(NotFittedError):
        selector.get_support()
    with pytest.raises(NotFittedError):
        selector.get_support(indices=True)
    with pytest.raises(NotFittedError):
        selector.transform([[1, 2, 3], [4, 5, 6]])


@pytest.mark.parametrize(
    "selector_cls, kwargs",
    [
        (MRMRSelector, dict(k=2, task="regression", verbose=False)),
        (JMISelector, dict(k=2, task="regression", verbose=False)),
        (JMIMSelector, dict(k=2, task="regression", verbose=False)),
        (CEFSPlusSelector, dict(k=2, verbose=False)),
    ],
)
def test_selector_fit_rejects_return_result_override(selector_cls, kwargs):
    rng = np.random.default_rng(20)
    X = pd.DataFrame(rng.normal(size=(80, 4)), columns=[f"f{i}" for i in range(4)])
    y = X["f0"] + rng.normal(size=80) * 0.1

    selector = selector_cls(**kwargs)
    with pytest.raises(ValueError, match="return shape"):
        selector.fit(X, y, return_result=True)


@pytest.mark.parametrize(
    "selector_cls, kwargs",
    [
        (MRMRSelector, dict(k=2, task="regression", verbose=False)),
        (JMISelector, dict(k=2, task="regression", verbose=False)),
        (JMIMSelector, dict(k=2, task="regression", verbose=False)),
        (CEFSPlusSelector, dict(k=2, verbose=False)),
    ],
)
@pytest.mark.parametrize(
    "override",
    [
        {"cat_encoding": "loo"},
        {"cat_features": ["cat"]},
        {"allow_full_data_target_encoding": True},
    ],
)
def test_selector_fit_rejects_preprocessing_overrides(selector_cls, kwargs, override):
    rng = np.random.default_rng(21)
    X = pd.DataFrame(
        {
            "cat": np.where(np.arange(80) % 2 == 0, "a", "b"),
            "f0": rng.normal(size=80),
            "f1": rng.normal(size=80),
            "f2": rng.normal(size=80),
        }
    )
    y = X["f0"] + rng.normal(size=80) * 0.1

    selector = selector_cls(**kwargs)
    with pytest.raises(ValueError, match="preprocessing-affecting"):
        selector.fit(X, y, **override)


def test_selector_dataframe_transform_rejects_reordered_columns():
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(120, 5)), columns=[f"f{i}" for i in range(5)])
    y = X["f0"] + rng.normal(size=120) * 0.05

    selector = MRMRSelector(k=2, task="regression", verbose=False).fit(X, y)

    # String columns use sklearn's standard feature-name mismatch wording so
    # check_dataframe_column_names_consistency passes.
    with pytest.raises(
        ValueError,
        match="Feature names must be in the same order as they were in fit",
    ):
        selector.transform(X[list(reversed(X.columns))])

    # Non-string labels fall back to SIFT's own strict order/identity message.
    numeric = pd.DataFrame(X.to_numpy(), columns=range(5))
    numeric_selector = MRMRSelector(k=2, task="regression", verbose=False).fit(
        numeric, y
    )
    with pytest.raises(
        ValueError, match="DataFrame columns must match fitted columns and order"
    ):
        numeric_selector.transform(numeric[list(reversed(numeric.columns))])


def test_gaussian_selector_requires_positional_x_for_unnamed_cache():
    rng = np.random.default_rng(23)
    X = pd.DataFrame(rng.normal(size=(120, 5)), columns=[f"f{i}" for i in range(5)])
    y = X["f0"] + 0.3 * X["f2"] + rng.normal(size=120) * 0.05
    cache = build_cache(X.to_numpy(), subsample=None)

    with pytest.raises(ValueError, match="unnamed/positional"):
        MRMRSelector(
            k=2,
            task="regression",
            estimator="gaussian",
            cache=cache,
            verbose=False,
        ).fit(X, y)

    X_arr = X.to_numpy()
    selector = MRMRSelector(
        k=2,
        task="regression",
        estimator="gaussian",
        cache=cache,
        verbose=False,
    ).fit(X_arr, y)

    assert all(feature.startswith("x") for feature in selector.selected_features_)
    assert all(isinstance(index, (int, np.integer)) for index in selector.selected_indices_)
    assert selector.selected_features_ == [f"x{i}" for i in selector.selected_indices_]
    transformed = selector.transform(X_arr)
    assert transformed.shape[1] == 2
    np.testing.assert_array_equal(transformed, X_arr[:, selector.selected_indices_])


def test_selector_set_params_updates_fit_call():
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(140, 6)), columns=[f"f{i}" for i in range(6)])
    y = X["f0"] + X["f1"] * 0.5 + rng.normal(size=140) * 0.05

    selector = MRMRSelector(k=1, task="regression", verbose=False)
    selector.set_params(k=3)
    selector.fit(X, y)

    assert len(selector.selected_features_) == 3


@pytest.mark.categorical
def test_selector_class_fits_supervised_categorical_encoder_on_train_only():
    pytest.importorskip("category_encoders")

    X_train = pd.DataFrame(
        {
            "team": ["a"] * 40 + ["b"] * 40,
            "noise": np.linspace(0.0, 1.0, 80),
        }
    )
    y_train = np.r_[np.zeros(40), np.ones(40)]
    selector = MRMRSelector(
        k=1,
        task="regression",
        estimator="classic",
        cat_features=["team"],
        cat_encoding="target",
        verbose=False,
    ).fit(X_train, y_train)

    assert selector.selected_features_ == ["team"]
    assert selector.categorical_features_ == ["team"]

    X_test = pd.DataFrame({"team": ["a", "b", "future_only"], "noise": [0.2, 0.3, 0.4]})
    X_out = selector.transform(X_test)

    assert list(X_out.columns) == ["team"]
    assert np.isclose(float(X_out.iloc[2, 0]), float(np.mean(y_train)))


@pytest.mark.categorical
def test_selector_loo_fit_transform_uses_training_fit_transform_matrix():
    pytest.importorskip("category_encoders")

    n_pairs = 30
    X = pd.DataFrame(
        {
            "id": np.repeat([f"id_{i}" for i in range(n_pairs)], 2),
            "noise": np.random.default_rng(0).normal(size=n_pairs * 2),
        }
    )
    y = np.repeat(np.arange(n_pairs, dtype=float) * 100.0, 2) + np.tile(
        [0.0, 10.0],
        n_pairs,
    )

    selector = MRMRSelector(
        k=1,
        task="regression",
        estimator="classic",
        cat_features=["id"],
        cat_encoding="loo",
        verbose=False,
    )
    X_fit = selector.fit_transform(X, y)

    assert selector.selected_features_ == ["id"]
    assert list(X_fit.columns) == ["id"]
    assert np.allclose(X_fit["id"].iloc[:4].to_numpy(), [10.0, 0.0, 110.0, 100.0])
    assert not np.allclose(
        X_fit["id"].to_numpy(),
        selector.transform(X)["id"].to_numpy(),
    )


@pytest.mark.categorical
def test_selector_prefix_auto_k_rejects_supervised_class_encoder():
    pytest.importorskip("category_encoders")

    X = pd.DataFrame(
        {
            "id": [f"id_{i}" for i in range(100)],
            "x": np.random.default_rng(0).normal(size=100),
        }
    )
    y = np.r_[np.zeros(50, dtype=int), np.ones(50, dtype=int)]
    cfg = AutoKConfig(
        k_method="evaluate",
        auto_k_mode="prefix_only",
        strategy="time_holdout",
        min_k=1,
        max_k=2,
    )

    selector = MRMRSelector(
        k="auto",
        task="classification",
        cat_features=["id"],
        cat_encoding="target",
        auto_k_config=cfg,
        verbose=False,
    )

    with pytest.raises(ValueError, match="prefix_only auto-k"):
        selector.fit(X, y, time=np.arange(len(X)))


def test_selector_nested_target_cv_passes_context_to_fold_local_encoder():
    X = pd.DataFrame(
        {
            "category": np.tile(["a", "b"], 40),
            "signal": np.linspace(0.0, 1.0, 80),
        }
    )
    y = X["signal"].to_numpy()
    config = AutoKConfig(
        k_method="evaluate",
        auto_k_mode="nested",
        strategy="time_holdout",
        min_k=1,
        max_k=2,
    )

    selector = MRMRSelector(
        k="auto",
        cat_features=["category"],
        cat_encoding="target_cv",
        target_cv_n_splits=3,
        target_cv_smoothing=1.0,
        auto_k_config=config,
        verbose=False,
    ).fit(X, y, time=np.arange(len(X)))

    assert selector.categorical_encoding_metadata_ == {
        "kind": "time",
        "n_splits": 3,
    }
    np.testing.assert_array_equal(
        selector.categorical_encoder_.effective_sample_weight_[:3],
        np.zeros(3),
    )


def test_selector_nested_auto_k_time_holdout():
    rng = np.random.default_rng(4)
    X = pd.DataFrame(rng.normal(size=(160, 5)), columns=[f"f{i}" for i in range(5)])
    y = X["f0"] * 1.5 + X["f1"] * 0.25 + rng.normal(size=160) * 0.1
    cfg = AutoKConfig(
        auto_k_mode="nested",
        strategy="time_holdout",
        metric="rmse",
        min_k=1,
        max_k=4,
        val_frac=0.25,
    )

    selector = MRMRSelector(
        k="auto",
        task="regression",
        estimator="classic",
        auto_k_config=cfg,
        verbose=False,
    ).fit(X, y, time=np.arange(len(X)))

    assert 1 <= selector.k_ <= 4
    assert len(selector.selected_features_) == selector.k_
    assert selector.nested_auto_k_diagnostics_["mode"] == "nested"
    assert not selector.nested_auto_k_diagnostics_["scores"].empty


def test_cefsplus_selector_nested_auto_k_plateau_rule():
    rng = np.random.default_rng(45)
    X = pd.DataFrame(rng.normal(size=(160, 5)), columns=[f"f{i}" for i in range(5)])
    y = X["f0"] * 1.5 + X["f1"] * 0.25 + rng.normal(size=160) * 0.1
    cfg = AutoKConfig(
        auto_k_mode="nested",
        strategy="time_holdout",
        metric="rmse",
        selection_rule="plateau",
        score_rel_tol=0.05,
        plateau_prefer="smallest",
        min_k=1,
        max_k=4,
        val_frac=0.25,
    )

    selector = CEFSPlusSelector(k="auto", auto_k_config=cfg, verbose=False).fit(
        X,
        y,
        time=np.arange(len(X)),
    )

    assert 1 <= selector.k_ <= 4
    assert selector.nested_auto_k_diagnostics_["selection_rule"] == "plateau"
    scores = selector.nested_auto_k_diagnostics_["scores"]
    assert "score_se" in scores.columns
    assert "in_selected_plateau" in scores.columns


def test_cefsplus_selector_nested_auto_k_distinguishes_best_and_selected(monkeypatch):
    rng = np.random.default_rng(145)
    X = pd.DataFrame(rng.normal(size=(160, 5)), columns=[f"f{i}" for i in range(5)])
    y = X["f0"] * 1.5 + rng.normal(size=160) * 0.1
    cfg = AutoKConfig(
        auto_k_mode="nested",
        strategy="time_holdout",
        metric="rmse",
        selection_rule="plateau",
        score_rel_tol=0.06,
        plateau_prefer="smallest",
        min_k=1,
        max_k=4,
        val_frac=0.25,
    )

    def fake_evaluate_prefixes(*args, k_grid, **kwargs):
        scores = {1: 0.95, 3: 0.90, 4: 1.20}
        return {k: scores[k] for k in k_grid}

    monkeypatch.setattr(
        auto_k_nested_module,
        "evaluate_numeric_prefixes",
        fake_evaluate_prefixes,
    )

    selector = CEFSPlusSelector(k="auto", auto_k_config=cfg, verbose=False).fit(
        X,
        y,
        time=np.arange(len(X)),
    )

    diagnostics = selector.nested_auto_k_diagnostics_
    assert diagnostics["best_k"] == 3
    assert diagnostics["selected_k"] == 1
    assert selector.k_ == 1


@pytest.mark.categorical
def test_selector_nested_auto_k_allows_supervised_class_encoder():
    pytest.importorskip("category_encoders")

    rng = np.random.default_rng(44)
    X = pd.DataFrame(
        {
            "id": [f"id_{i}" for i in range(120)],
            "x": rng.normal(size=120),
        }
    )
    y = np.sin(np.arange(120) / 6.0) + rng.normal(scale=0.05, size=120)
    cfg = AutoKConfig(
        auto_k_mode="nested",
        strategy="time_holdout",
        metric="rmse",
        min_k=1,
        max_k=2,
        val_frac=0.25,
    )

    selector = MRMRSelector(
        k="auto",
        task="regression",
        estimator="classic",
        cat_features=["id"],
        cat_encoding="target",
        auto_k_config=cfg,
        verbose=False,
    ).fit(X, y, time=np.arange(len(X)))

    assert 1 <= selector.k_ <= 2
    assert not selector.nested_auto_k_diagnostics_["scores"].empty


@pytest.mark.categorical
def test_selector_nested_auto_k_uses_fit_transform_train_matrix(monkeypatch):
    pytest.importorskip("category_encoders")

    n_pairs = 40
    X = pd.DataFrame(
        {
            "id": np.repeat([f"id_{i}" for i in range(n_pairs)], 2),
            "noise": np.random.default_rng(45).normal(size=n_pairs * 2),
        }
    )
    y = np.repeat(np.arange(n_pairs, dtype=float) * 100.0, 2) + np.tile(
        [0.0, 10.0],
        n_pairs,
    )
    cfg = AutoKConfig(
        auto_k_mode="nested",
        strategy="time_holdout",
        metric="rmse",
        min_k=1,
        max_k=2,
        val_frac=0.25,
    )
    captured = []

    def capture_train_path(
        X_train_path,
        X_val_path,
        y_train,
        y_val,
        w_train,
        w_val,
        *,
        task,
        metric,
        k_grid,
    ):
        captured.append(X_train_path.copy())
        return {k: float(k - 1) for k in k_grid}

    monkeypatch.setattr(
        auto_k_nested_module,
        "evaluate_numeric_prefixes",
        capture_train_path,
    )

    selector = MRMRSelector(
        k="auto",
        task="regression",
        estimator="classic",
        cat_features=["id"],
        cat_encoding="loo",
        auto_k_config=cfg,
        verbose=False,
    )
    selector.fit(X, y, time=np.arange(len(X)))

    train_rows = np.arange(int(np.floor((1.0 - cfg.val_frac) * len(X))))
    expected = MRMRSelector(
        k=cfg.max_k,
        task="regression",
        estimator="classic",
        cat_features=["id"],
        cat_encoding="loo",
        verbose=False,
    ).fit_transform(X.iloc[train_rows], y[train_rows])

    assert captured
    pd.testing.assert_frame_equal(
        captured[0].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
    )


def test_selector_nested_auto_k_group_cv():
    rng = np.random.default_rng(5)
    X = pd.DataFrame(rng.normal(size=(120, 5)), columns=[f"f{i}" for i in range(5)])
    y = X["f0"] + rng.normal(size=120) * 0.1
    groups = np.repeat(np.arange(6), 20)
    cfg = AutoKConfig(
        auto_k_mode="nested",
        strategy="group_cv",
        metric="rmse",
        min_k=1,
        max_k=3,
        n_splits=3,
    )

    selector = MRMRSelector(
        k="auto",
        task="regression",
        estimator="classic",
        auto_k_config=cfg,
        verbose=False,
    ).fit(X, y, groups=groups)

    assert 1 <= selector.k_ <= 3
    assert selector.nested_auto_k_diagnostics_["scores"]["n_splits"].eq(3).all()


@pytest.mark.parametrize(
    "selector_cls, kwargs",
    [
        (MRMRSelector, dict(task="regression", estimator="classic")),
        (JMISelector, dict(task="regression", estimator="r2")),
        (JMIMSelector, dict(task="regression", estimator="r2")),
        (CEFSPlusSelector, dict()),
    ],
)
def test_selector_auto_k_groups_infers_group_cv_without_config(selector_cls, kwargs):
    rng = np.random.default_rng(46)
    X = pd.DataFrame(rng.normal(size=(120, 5)), columns=[f"f{i}" for i in range(5)])
    y = X["f0"] + 0.25 * X["f1"] + rng.normal(size=120) * 0.1
    groups = np.repeat(np.arange(6), 20)

    selector = selector_cls(k="auto", verbose=False, **kwargs)
    selector.fit(X, y, groups=groups)

    assert 1 <= len(selector.selected_features_) <= X.shape[1]


def test_selector_nested_auto_k_handles_empty_fold_paths(monkeypatch):
    rng = np.random.default_rng(49)
    X = pd.DataFrame(rng.normal(size=(80, 4)), columns=list("abcd"))
    y = rng.normal(size=80)
    cfg = AutoKConfig(
        auto_k_mode="nested",
        strategy="time_holdout",
        metric="rmse",
        min_k=1,
        max_k=3,
    )

    def empty_fit_transform(self, X, y=None, **fit_params):
        self.feature_names_in_ = list(X.columns)
        self.n_features_in_ = X.shape[1]
        self.selected_features_ = []
        self.selected_indices_ = np.asarray([], dtype=np.int64)
        return X.iloc[:, []]

    def empty_transform(self, X):
        return X.iloc[:, []]

    monkeypatch.setattr(MRMRSelector, "fit_transform", empty_fit_transform)
    monkeypatch.setattr(MRMRSelector, "transform", empty_transform)

    selector = MRMRSelector(
        k="auto",
        task="regression",
        estimator="classic",
        auto_k_config=cfg,
        verbose=False,
    )
    with pytest.warns(UserWarning, match="candidate score-curve values are non-finite"):
        selector.fit(X, y, time=np.arange(len(X)))

    scores = selector.nested_auto_k_diagnostics_["scores"]
    assert np.isinf(scores["score"]).all()
    assert 0 <= len(selector.selected_features_) <= selector.k_


def test_selector_auto_k_without_context_matches_public_error():
    rng = np.random.default_rng(47)
    X = pd.DataFrame(rng.normal(size=(80, 4)), columns=list("abcd"))
    y = X["a"] + rng.normal(size=80) * 0.1
    selector = MRMRSelector(k="auto", task="regression", verbose=False)

    with pytest.raises(
        ValueError,
        match="k='auto' requires time, groups, or auto_k_config",
    ):
        selector.fit(X, y)


def test_failed_refit_clears_previous_selector_state():
    rng = np.random.default_rng(48)
    X = pd.DataFrame(rng.normal(size=(80, 4)), columns=list("abcd"))
    y = X["a"] + rng.normal(size=80) * 0.1
    selector = MRMRSelector(k=1, task="regression", verbose=False).fit(X, y)

    assert selector.selected_features_
    selector.set_params(k="auto")
    with pytest.raises(ValueError, match="k='auto' requires time"):
        selector.fit(X, y)

    with pytest.raises(NotFittedError):
        selector.transform(X)
    with pytest.raises(NotFittedError):
        selector.get_support()


def test_failed_mid_fit_clears_partial_selector_state():
    rng = np.random.default_rng(49)
    X = pd.DataFrame(rng.normal(size=(80, 4)), columns=list("abcd"))
    y = X["a"] + rng.normal(size=80) * 0.1
    selector = MRMRSelector(k=1, task="regression", verbose=False).fit(X, y)

    bad_X = pd.DataFrame(
        {
            "team": ["a", "b"] * 40,
            "x": rng.normal(size=80),
        }
    )
    selector.set_params(cat_features=["team"], cat_encoding="bad")
    with pytest.raises(ValueError, match="cat_encoding"):
        selector.fit(bad_X, y)

    for attr in (
        "categorical_encoder_",
        "categorical_features_",
        "_categorical_encoding_applied_",
        "selected_features_",
        "selected_indices_",
        "feature_names_in_",
        "n_features_in_",
    ):
        assert not hasattr(selector, attr)

    with pytest.raises(NotFittedError):
        selector.transform(bad_X)
    with pytest.raises(NotFittedError):
        selector.get_support()


def test_selector_nested_auto_k_passes_fit_params_to_fold_and_final_paths(monkeypatch):
    rng = np.random.default_rng(50)
    X = pd.DataFrame(rng.normal(size=(80, 4)), columns=list("abcd"))
    y = X["a"] + rng.normal(size=80) * 0.1
    calls = []

    def fake_selector(X, y, k, **kwargs):
        calls.append(kwargs.get("top_m"))
        return list(X.columns[: min(int(k), X.shape[1])])

    monkeypatch.setattr(selectors_mod, "select_mrmr", fake_selector)

    cfg = AutoKConfig(
        auto_k_mode="nested",
        strategy="time_holdout",
        min_k=1,
        max_k=2,
        val_frac=0.25,
    )
    selector = MRMRSelector(k="auto", task="regression", auto_k_config=cfg, verbose=False)

    selector.fit(X, y, time=np.arange(len(X)), top_m=1)

    assert calls == [1, 1]


@pytest.mark.categorical
def test_selector_supervised_encoding_rejects_prebuilt_cache():
    pytest.importorskip("category_encoders")

    X = pd.DataFrame({"team": ["a"] * 20 + ["b"] * 20, "x": np.arange(40.0)})
    y = np.r_[np.zeros(20), np.ones(20)]
    selector = MRMRSelector(
        k=1,
        task="regression",
        cat_features=["team"],
        cat_encoding="target",
        cache=object(),
        verbose=False,
    )

    with pytest.raises(ValueError, match="prebuilt caches"):
        selector.fit(X, y)


def test_selector_nested_auto_k_rejects_cache():
    rng = np.random.default_rng(6)
    X = pd.DataFrame(rng.normal(size=(80, 4)), columns=[f"f{i}" for i in range(4)])
    y = X["f0"] + rng.normal(size=80) * 0.1
    cfg = AutoKConfig(auto_k_mode="nested", strategy="time_holdout", min_k=1, max_k=2)

    selector = MRMRSelector(k="auto", task="regression", auto_k_config=cfg, cache=object())
    with pytest.raises(ValueError, match="prebuilt caches"):
        selector.fit(X, y, time=np.arange(len(X)))


@pytest.mark.parametrize("k_method", ["elbow", "penalized_objective"])
def test_selector_nested_auto_k_rejects_non_evaluate_method(k_method):
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(80, 4)), columns=list("abcd"))
    y = X["a"] + rng.normal(size=80) * 0.1
    cfg = AutoKConfig(
        k_method=k_method,
        auto_k_mode="nested",
        strategy="time_holdout",
        min_k=1,
        max_k=3,
    )

    selector = MRMRSelector(k="auto", task="regression", auto_k_config=cfg, verbose=False)
    with pytest.raises(ValueError, match="k_method='evaluate'"):
        selector.fit(X, y, time=np.arange(len(X)))
