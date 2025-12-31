import numpy as np
import pandas as pd

from sift import select_jmi, select_mrmr


def test_select_mrmr_prefers_strong_signal():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 4)), columns=["a", "b", "c", "d"])
    y = X["a"] * 2.0 + rng.normal(scale=0.1, size=200)

    selected = select_mrmr(X, y, k=2, task="regression", verbose=False)

    assert len(selected) == 2
    assert selected[0] == "a"


def test_select_jmi_returns_k():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(200, 4)), columns=["a", "b", "c", "d"])
    y = (X["a"] + X["b"] > 0).astype(int)

    selected = select_jmi(X, y, k=3, task="classification", estimator="binned", verbose=False)

    assert len(selected) == 3
