import numpy as np

from sift.stability import StabilitySelector, _block_bootstrap_indices


def test_block_bootstrap_indices_basic():
    n = 120
    groups = np.repeat(np.arange(6), 20)
    time = np.tile(np.arange(20), 6)
    splits = list(
        _block_bootstrap_indices(
            n=n,
            n_bootstrap=3,
            groups=groups,
            time=time,
            block_size=5,
            block_method="moving",
            random_state=0,
        )
    )

    assert len(splits) == 3
    for train_idx, val_idx in splits:
        assert train_idx.dtype == np.int64
        assert val_idx.dtype == np.int64
        assert train_idx.min() >= 0
        assert train_idx.max() < n
        assert val_idx.min() >= 0
        assert val_idx.max() < n


def test_stability_selector_block_bootstrap_runs():
    rng = np.random.default_rng(0)
    n, p = 120, 8
    X = rng.normal(size=(n, p))
    y = X[:, 0] + rng.normal(size=n) * 0.2
    groups = np.repeat(np.arange(6), 20)
    time = np.tile(np.arange(20), 6)

    selector = StabilitySelector(
        n_bootstrap=5,
        threshold=0.1,
        alpha=0.01,
        random_state=0,
        n_jobs=1,
        verbose=False,
        block_size=5,
        block_method="moving",
    )
    selector.fit(X, y, groups=groups, time=time)

    assert selector.n_features_selected_ > 0
