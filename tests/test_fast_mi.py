import numpy as np
import pandas as pd
from sift.fast_mi import regression_joint_mi, binned_joint_mi, ksg_joint_mi


def test_regression_joint_mi_block_consistency():
    """Results should be same regardless of block_size."""
    np.random.seed(42)
    n, p = 200, 50
    X = pd.DataFrame(np.random.randn(n, p), columns=[f'f{i}' for i in range(p)])
    y = pd.Series(np.random.randn(n))

    features = [f'f{i}' for i in range(1, p)]

    result_small_block = regression_joint_mi('f0', features, X, y, block_size=10)
    result_large_block = regression_joint_mi('f0', features, X, y, block_size=256)

    pd.testing.assert_series_equal(result_small_block, result_large_block, rtol=1e-10)


def test_jmi_methods_return_series():
    np.random.seed(42)
    n, p = 100, 10
    X = pd.DataFrame(np.random.randn(n, p), columns=[f'f{i}' for i in range(p)])
    y = pd.Series(np.random.randn(n))
    features = [f'f{i}' for i in range(1, p)]

    for func in [regression_joint_mi, binned_joint_mi, ksg_joint_mi]:
        result = func('f0', features, X, y, n_jobs=1)
        assert isinstance(result, pd.Series)
        assert len(result) == len(features)
        assert set(result.index) == set(features)
