import numpy as np
import pandas as pd
from sift import StabilitySelector, stability_select


def test_stability_selector_regression():
    np.random.seed(42)
    n, p = 200, 20
    X = pd.DataFrame(np.random.randn(n, p), columns=[f'f{i}' for i in range(p)])
    y = X['f0'] + 0.5 * X['f1'] + np.random.randn(n) * 0.5

    selector = StabilitySelector(n_bootstrap=10, threshold=0.3, verbose=False)
    selector.fit(X, y)

    assert selector.n_features_selected_ > 0
    assert 'f0' in selector.selected_feature_names_


def test_stability_selector_classification():
    np.random.seed(42)
    n, p = 200, 20
    X = pd.DataFrame(np.random.randn(n, p), columns=[f'f{i}' for i in range(p)])
    y = (X['f0'] + X['f1'] > 0).astype(int)

    selector = StabilitySelector(
        n_bootstrap=10, threshold=0.3, task='classification', verbose=False
    )
    selector.fit(X, y)

    assert selector.n_features_selected_ > 0


def test_stability_select_convenience():
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = X[:, 0] + np.random.randn(100) * 0.3

    selected, freqs = stability_select(X, y, n_bootstrap=10, threshold=0.3, verbose=False)

    assert len(selected) > 0
    assert len(freqs) == 10
