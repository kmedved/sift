import pandas as pd

from sift import select_mrmr

columns = ["target_classif", "target_regression", "some_null", "feature_a", "constant", "feature_b"]
features = ["some_null", "feature_a", "constant", "feature_b"]

data = [
    ("a", 1.0, 1.0, 2.0, 7.0, 3.0),
    ("a", 2.0, float("NaN"), 2.0, 7.0, 2.0),
    ("b", 3.0, float("NaN"), 3.0, 7.0, 1.0),
    ("b", 4.0, 4.0, 3.0, 7.0, 2.0),
    ("b", 5.0, 5.0, 4.0, 7.0, 3.0),
]

df_pandas = pd.DataFrame(data=data, columns=columns)


def test_select_mrmr_classification_returns_k():
    selected_features = select_mrmr(
        X=df_pandas.loc[:, features],
        y=df_pandas.loc[:, "target_classif"],
        k=3,
        task="classification",
        verbose=False,
    )

    assert len(selected_features) == 3
    assert set(selected_features).issubset(set(features))


def test_select_mrmr_regression_returns_k():
    selected_features = select_mrmr(
        X=df_pandas.loc[:, features],
        y=df_pandas.loc[:, "target_regression"],
        k=2,
        task="regression",
        verbose=False,
    )

    assert len(selected_features) == 2
    assert set(selected_features).issubset(set(features))
