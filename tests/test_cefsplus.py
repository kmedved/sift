import numpy as np
import pandas as pd
import sift


def test_cefsplus_regression():
    np.random.seed(42)
    n, p = 500, 20
    X = pd.DataFrame(np.random.randn(n, p), columns=[f'f{i}' for i in range(p)])
    y = X['f0'] + 0.5 * X['f1'] + np.random.randn(n) * 0.3

    selected = sift.cefsplus_regression(X, y, K=5)
    assert len(selected) == 5
    assert 'f0' in selected[:3]


def test_cefsplus_with_cache():
    np.random.seed(42)
    n, p = 500, 20
    X = pd.DataFrame(np.random.randn(n, p), columns=[f'f{i}' for i in range(p)])

    cache = sift.build_cache(X, subsample=None)

    y1 = X['f0'] + np.random.randn(n) * 0.3
    y2 = X['f5'] + np.random.randn(n) * 0.3

    sel1 = sift.select_features_cached(cache, y1, k=5)
    sel2 = sift.select_features_cached(cache, y2, k=5)

    assert len(sel1) == 5
    assert len(sel2) == 5
    assert "f0" in sel1[:2]
    assert "f5" in sel2[:2]


def test_build_cache_nan_handling():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 10))
    X.iloc[0:5, 0] = np.nan

    cache = sift.build_cache(X, impute='mean')
    assert not np.isnan(cache.Z).any()


def test_build_cache_inf_handling():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(200, 10))
    X.iloc[0, 0] = np.inf
    X.iloc[1, 1] = -np.inf

    cache = sift.build_cache(X, impute="mean", subsample=None)
    assert np.isfinite(cache.Z).all()


def test_ksg_sanity():
    """KSG should report higher MI for dependent variables."""
    from sift.fast_mi import _ksg_mi_joint
    np.random.seed(42)
    x1 = np.random.randn(1000)
    x2 = np.random.randn(1000)
    y_independent = np.random.randn(1000)
    y_dependent = x1 + x2 + np.random.randn(1000) * 0.1

    mi_independent = _ksg_mi_joint(x1, x2, y_independent, k=3)
    mi_dependent = _ksg_mi_joint(x1, x2, y_dependent, k=3)
    assert mi_dependent > mi_independent


def test_binned_mi_increases_with_dependence():
    """Binned MI should increase when y depends on x."""
    from sift.fast_mi import _binned_mi_single
    np.random.seed(42)
    x1 = np.random.randn(2000)
    x2 = np.random.randn(2000)

    y_independent = np.random.randn(2000)
    y_dependent = x1 + x2 + np.random.randn(2000) * 0.1

    mi_independent = _binned_mi_single(x1, x2, y_independent)
    mi_dependent = _binned_mi_single(x1, x2, y_dependent)

    assert mi_dependent > mi_independent * 1.5
