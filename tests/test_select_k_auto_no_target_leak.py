import importlib.util
import warnings
import numpy as np
import pandas as pd
import pytest

import sift.api as sift_api
from sift import select_cefsplus, select_jmi, select_jmim, select_mrmr
from sift.selection.auto_k import AutoKConfig, select_k_auto


NESTED_MODE_ERROR = "auto_k_mode='nested'.*not implemented"


def _numeric_auto_k_data():
    rng = np.random.default_rng(123)
    n = 80
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"x{i}" for i in range(6)])
    y = X["x0"].to_numpy() + 0.25 * rng.normal(size=n)
    time = np.arange(n)
    return X, y, time


def test_select_k_auto_prefix_only_matches_default():
    X, y, time = _numeric_auto_k_data()
    feature_path = list(X.columns)

    default_cfg = AutoKConfig(
        strategy="time_holdout",
        min_k=1,
        max_k=6,
        val_frac=0.25,
    )
    explicit_cfg = AutoKConfig(
        auto_k_mode="prefix_only",
        strategy="time_holdout",
        min_k=1,
        max_k=6,
        val_frac=0.25,
    )

    default_result = select_k_auto(X, y, feature_path, default_cfg, time=time)
    explicit_result = select_k_auto(X, y, feature_path, explicit_cfg, time=time)

    assert default_result[0] == explicit_result[0]
    assert default_result[1] == explicit_result[1]
    pd.testing.assert_frame_equal(default_result[2], explicit_result[2])


def test_select_k_auto_nested_mode_raises():
    X, y, time = _numeric_auto_k_data()
    cfg = AutoKConfig(
        auto_k_mode="nested",
        strategy="time_holdout",
        min_k=1,
        max_k=6,
        val_frac=0.25,
    )

    with pytest.raises(NotImplementedError, match=NESTED_MODE_ERROR):
        select_k_auto(X, y, list(X.columns), cfg, time=time)


def test_select_k_auto_rejects_elbow_method():
    X, y, time = _numeric_auto_k_data()
    cfg = AutoKConfig(
        k_method="elbow",
        strategy="time_holdout",
        min_k=1,
        max_k=6,
        val_frac=0.25,
    )

    with pytest.raises(ValueError, match="select_k_auto.*k_method='evaluate'"):
        select_k_auto(X, y, list(X.columns), cfg, time=time)


def test_select_k_auto_evaluate_honors_sample_weight():
    rng = np.random.default_rng(321)
    n_train = 40
    n_val = 20
    n = n_train + n_val
    x0 = rng.normal(size=n)
    x1 = np.zeros(n)
    x1[:n_train] = np.tile([0.0, 1.0], n_train // 2)
    x1[n_train:] = 1.0

    y = np.empty(n)
    y[:n_train] = 2.0 * x0[:n_train] + 10.0 * x1[:n_train]
    y[n_train : n - 1] = 2.0 * x0[n_train : n - 1]
    y[n - 1] = 2.0 * x0[n - 1] + 10.0

    X = pd.DataFrame({"x0": x0, "x1": x1})
    time = np.arange(n)
    cfg = AutoKConfig(
        strategy="time_holdout",
        metric="rmse",
        min_k=1,
        max_k=2,
        val_frac=n_val / n,
    )

    unweighted_k, _, unweighted_diag = select_k_auto(
        X,
        y,
        ["x0", "x1"],
        cfg,
        time=time,
        task="regression",
    )

    sample_weight = np.ones(n)
    sample_weight[-1] = 1000.0
    weighted_k, _, weighted_diag = select_k_auto(
        X,
        y,
        ["x0", "x1"],
        cfg,
        time=time,
        task="regression",
        sample_weight=sample_weight,
    )

    assert unweighted_k == 1
    assert weighted_k == 2
    assert unweighted_diag.loc[unweighted_diag["k"] == 1, "score"].iloc[0] < (
        unweighted_diag.loc[unweighted_diag["k"] == 2, "score"].iloc[0]
    )
    assert weighted_diag.loc[weighted_diag["k"] == 2, "score"].iloc[0] < (
        weighted_diag.loc[weighted_diag["k"] == 1, "score"].iloc[0]
    )


def test_public_auto_k_passes_sample_weight_to_prefix_evaluation(monkeypatch):
    X, y, time = _numeric_auto_k_data()
    sample_weight = np.linspace(1.0, 3.0, len(y))
    captured = {}

    def fake_select_k_auto(
        X,
        y,
        feature_path,
        config,
        *,
        sample_weight=None,
        **kwargs,
    ):
        captured["sample_weight"] = np.asarray(sample_weight)
        return 1, feature_path[:1], pd.DataFrame({"k": [1], "score": [0.0]})

    monkeypatch.setattr(sift_api, "select_k_auto", fake_select_k_auto)
    cfg = AutoKConfig(
        strategy="time_holdout",
        min_k=1,
        max_k=2,
        val_frac=0.25,
    )

    selected = select_mrmr(
        X,
        y,
        k="auto",
        task="regression",
        time=time,
        auto_k_config=cfg,
        sample_weight=sample_weight,
        subsample=None,
        verbose=False,
    )

    assert len(selected) == 1
    assert "sample_weight" in captured
    assert captured["sample_weight"].shape == sample_weight.shape
    assert np.isclose(captured["sample_weight"].mean(), 1.0)
    np.testing.assert_allclose(
        captured["sample_weight"] / captured["sample_weight"][0],
        sample_weight / sample_weight[0],
    )


@pytest.mark.parametrize(
    ("selector", "kwargs"),
    [
        (select_mrmr, {"task": "regression"}),
        (select_jmi, {"task": "regression"}),
        (select_jmim, {"task": "regression"}),
        (select_cefsplus, {}),
    ],
)
def test_public_selectors_reject_nested_auto_k_mode(selector, kwargs):
    X, y, time = _numeric_auto_k_data()
    cfg = AutoKConfig(
        auto_k_mode="nested",
        strategy="time_holdout",
        min_k=1,
        max_k=6,
        val_frac=0.25,
    )

    with pytest.raises(NotImplementedError, match=NESTED_MODE_ERROR):
        selector(
            X,
            y,
            k="auto",
            time=time,
            auto_k_config=cfg,
            verbose=False,
            **kwargs,
        )


@pytest.mark.parametrize(
    "config_kwargs, match",
    [
        ({"k_method": "bad"}, "k_method"),
        ({"strategy": "bad"}, "strategy"),
        ({"val_frac": 1.0}, "val_frac"),
        ({"val_frac": "0.2"}, "val_frac"),
        ({"min_k": 5, "max_k": 3}, "min_k"),
        ({"min_k": True}, "min_k"),
        ({"elbow_min_rel_gain": "0.02"}, "elbow_min_rel_gain"),
    ],
)
def test_public_selectors_validate_auto_k_config(config_kwargs, match):
    X, y, time = _numeric_auto_k_data()
    cfg = AutoKConfig(**config_kwargs)

    with pytest.raises(ValueError, match=match):
        select_mrmr(
            X,
            y,
            k="auto",
            task="regression",
            time=time,
            auto_k_config=cfg,
            verbose=False,
        )


@pytest.mark.parametrize(
    ("selector", "kwargs"),
    [
        (select_mrmr, {"task": "regression", "estimator": "classic"}),
        (select_jmi, {"task": "regression", "estimator": "r2"}),
        (select_jmim, {"task": "regression", "estimator": "r2"}),
    ],
)
def test_classic_public_auto_k_rejects_elbow_method(selector, kwargs):
    X, y, time = _numeric_auto_k_data()
    cfg = AutoKConfig(
        k_method="elbow",
        strategy="time_holdout",
        min_k=1,
        max_k=3,
        val_frac=0.25,
    )

    with pytest.raises(ValueError, match="k_method='elbow'.*classic"):
        selector(
            X,
            y,
            k="auto",
            time=time,
            auto_k_config=cfg,
            verbose=False,
            **kwargs,
        )


def test_gaussian_auto_k_elbow_still_works_without_split_context():
    X, y, _ = _numeric_auto_k_data()
    cfg = AutoKConfig(k_method="elbow", min_k=1, max_k=4)

    cefs = select_cefsplus(X, y, k="auto", auto_k_config=cfg, verbose=False)
    gaussian_mrmr = select_mrmr(
        X,
        y,
        k="auto",
        task="regression",
        estimator="gaussian",
        auto_k_config=cfg,
        verbose=False,
    )

    assert 1 <= len(cefs) <= 4
    assert 1 <= len(gaussian_mrmr) <= 4


def test_select_k_auto_target_encoding_not_leaky():
    pytest.importorskip("category_encoders")

    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({"id": [f"id_{i}" for i in range(n)]})
    y = rng.normal(size=n)
    feature_path = ["id"]

    cfg = AutoKConfig(
        strategy="time_holdout",
        metric="rmse",
        val_frac=0.25,
        min_k=1,
        max_k=1,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, _, diag = select_k_auto(
            X=X,
            y=y,
            feature_path=feature_path,
            config=cfg,
            time=np.arange(n),
            task="regression",
            cat_features=["id"],
            cat_encoding="target",
        )

    assert float(diag["score"].iloc[0]) > 0.5


def test_select_k_auto_cat_encoding_requires_category_encoders():
    if importlib.util.find_spec("category_encoders") is not None:
        pytest.skip("category_encoders installed; skipping dependency error test")

    X = pd.DataFrame({"id": ["a", "b", "c", "d"]})
    y = np.array([0.1, 0.2, 0.3, 0.4])
    cfg = AutoKConfig(
        strategy="time_holdout",
        metric="rmse",
        val_frac=0.5,
        min_k=1,
        max_k=1,
    )

    with pytest.raises(ImportError):
        select_k_auto(
            X=X,
            y=y,
            feature_path=["id"],
            config=cfg,
            time=np.arange(len(y)),
            task="regression",
            cat_features=["id"],
            cat_encoding="target",
        )
