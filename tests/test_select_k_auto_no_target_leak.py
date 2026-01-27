import importlib.util
import numpy as np
import pandas as pd
import pytest

from sift.selection.auto_k import AutoKConfig, select_k_auto


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
