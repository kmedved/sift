import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from sift import CEFSPlusSelector, JMIMSelector, JMISelector, MRMRSelector


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


def test_selector_dataframe_transform_rejects_reordered_columns():
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(120, 5)), columns=[f"f{i}" for i in range(5)])
    y = X["f0"] + rng.normal(size=120) * 0.05

    selector = MRMRSelector(k=2, task="regression", verbose=False).fit(X, y)

    with pytest.raises(ValueError, match="columns"):
        selector.transform(X[list(reversed(X.columns))])


def test_selector_set_params_updates_fit_call():
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(140, 6)), columns=[f"f{i}" for i in range(6)])
    y = X["f0"] + X["f1"] * 0.5 + rng.normal(size=140) * 0.05

    selector = MRMRSelector(k=1, task="regression", verbose=False)
    selector.set_params(k=3)
    selector.fit(X, y)

    assert len(selector.selected_features_) == 3
