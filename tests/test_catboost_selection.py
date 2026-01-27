import numpy as np
import pandas as pd
import pytest

from sift import catboost as cb


def test_catboost_select_features_reorders_survivors():
    pytest.importorskip("catboost")

    rng = np.random.default_rng(0)
    n, p = 300, 20
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = pd.Series(rng.normal(size=n))

    X_train = X.iloc[:200]
    X_val = X.iloc[200:]
    y_train = y.iloc[:200]
    y_val = y.iloc[200:]

    feature_counts = [5, 10]
    model_params = {
        "iterations": 50,
        "depth": 6,
        "learning_rate": 0.1,
        "loss_function": "RMSE",
        "random_seed": 0,
    }

    scores, features_by_k = cb._select_features_single_split(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        task="regression",
        model_params=model_params,
        feature_counts=feature_counts,
        features=list(X.columns),
        cat_features=[],
        text_features=[],
        eval_metric="RMSE",
        higher_is_better=False,
        algorithm="prediction",
        steps=2,
        train_early_stopping_rounds=10,
    )

    assert scores
    for k in feature_counts:
        assert k in features_by_k
        assert len(features_by_k[k]) == k
