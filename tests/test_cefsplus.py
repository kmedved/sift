import numpy as np
import pandas as pd

from sift import build_cache, select_cached, select_cefsplus


def test_select_cefsplus_regression():
    rng = np.random.default_rng(42)
    n, p = 500, 20
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = X["f0"] + 0.5 * X["f1"] + rng.normal(size=n) * 0.3

    selected = select_cefsplus(X, y, k=5, verbose=False)
    assert len(selected) == 5
    assert "f0" in selected[:3]


def test_select_cached_with_cache():
    rng = np.random.default_rng(42)
    n, p = 500, 20
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])

    cache = build_cache(X, subsample=400)

    y1 = X["f0"] + rng.normal(size=n) * 0.3
    y2 = X["f5"] + rng.normal(size=n) * 0.3

    sel1 = select_cached(cache, y1, k=5)
    sel2 = select_cached(cache, y2, k=5)

    assert len(sel1) == 5
    assert len(sel2) == 5
    assert "f0" in sel1[:2]
    assert "f5" in sel2[:2]


def test_build_cache_handles_nonfinite():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(200, 10)))
    X.iloc[0, 0] = np.nan
    X.iloc[1, 1] = np.inf
    X.iloc[2, 2] = -np.inf

    cache = build_cache(X, subsample=None)
    assert np.isfinite(cache.Z).all()
