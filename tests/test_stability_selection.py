import numpy as np
import pandas as pd

from sift import StabilitySelector
from sift.stability import stability_select


def test_stability_selector_regression():
    np.random.seed(42)
    n, p = 200, 20
    X = pd.DataFrame(np.random.randn(n, p), columns=[f'f{i}' for i in range(p)])
    y = X['f0'] + 0.5 * X['f1'] + np.random.randn(n) * 0.5

    selector = StabilitySelector(
        n_bootstrap=10,
        threshold=0.1,
        alpha=0.01,
        random_state=0,
        n_jobs=1,
        verbose=False,
    )
    selector.fit(X, y)

    assert selector.n_features_selected_ > 0
    top = selector.get_feature_info()["feature"].head(5).tolist()
    assert len(top) > 0


def test_stability_selector_classification():
    np.random.seed(42)
    n, p = 200, 20
    X = pd.DataFrame(np.random.randn(n, p), columns=[f'f{i}' for i in range(p)])
    y = (X['f0'] + X['f1'] > 0).astype(int)

    selector = StabilitySelector(
        n_bootstrap=10,
        threshold=0.1,
        task="classification",
        alpha=0.1,
        random_state=0,
        n_jobs=1,
        verbose=False,
    )
    selector.fit(X, y)

    assert selector.n_features_selected_ > 0


def test_stability_select_convenience():
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = X[:, 0] + np.random.randn(100) * 0.3

    selected, freqs = stability_select(
        X,
        y,
        n_bootstrap=10,
        threshold=0.1,
        alpha=0.01,
        random_state=0,
        n_jobs=1,
        verbose=False,
    )

    assert len(selected) > 0
    assert len(freqs) == 10


def test_stability_regression_wrapper():
    """Test the stability_regression convenience function."""
    from sift import stability_regression

    np.random.seed(42)
    n, p = 200, 20
    X = pd.DataFrame(np.random.randn(n, p), columns=[f'f{i}' for i in range(p)])
    y = X['f0'] + 0.5 * X['f1'] + np.random.randn(n) * 0.5

    selected = stability_regression(
        X,
        y,
        k=10,
        n_bootstrap=10,
        threshold=0.1,
        alpha=0.01,
        random_state=0,
        n_jobs=1,
        verbose=False,
    )

    assert isinstance(selected, list)
    assert len(selected) <= 10
    assert all(isinstance(f, str) for f in selected)


def test_stability_classif_wrapper():
    """Test the stability_classif convenience function."""
    from sift import stability_classif

    np.random.seed(42)
    n, p = 200, 20
    X = pd.DataFrame(np.random.randn(n, p), columns=[f'f{i}' for i in range(p)])
    y = (X['f0'] + X['f1'] > 0).astype(int)

    selected = stability_classif(
        X,
        y,
        k=10,
        n_bootstrap=10,
        threshold=0.1,
        alpha=0.1,
        random_state=0,
        n_jobs=1,
        verbose=False,
    )

    assert isinstance(selected, list)
    assert len(selected) <= 10
    assert all(isinstance(f, str) for f in selected)


def test_prep_arrays_exclusion_only_when_smart_sampler_enabled():
    """Test that group/time columns are only excluded when use_smart_sampler=True."""
    from sift.sampling.smart import SmartSamplerConfig

    np.random.seed(42)
    n, p = 100, 10
    X = pd.DataFrame(np.random.randn(n, p), columns=[f'f{i}' for i in range(p)])
    X['group_id'] = np.repeat(np.arange(10), 10)
    X['timestamp'] = np.tile(np.arange(10), 10)
    y = X['f0'] + np.random.randn(n) * 0.3

    config = SmartSamplerConfig(group_col='group_id', time_col='timestamp')

    # With use_smart_sampler=False, group_id and timestamp should be treated as features
    selector = StabilitySelector(
        n_bootstrap=5,
        threshold=0.3,
        use_smart_sampler=False,
        sampler_config=config,
        verbose=False
    )
    selector.fit(X, y)

    assert 'group_id' in selector.feature_names_in_
    assert 'timestamp' in selector.feature_names_in_


def test_first_and_last_per_group_respects_time_order():
    """Test that first_and_last_per_group uses time_col for ordering."""
    from sift.sampling.anchors import first_and_last_per_group

    # Create data where row order != time order
    df = pd.DataFrame({
        'group': ['A', 'A', 'A', 'B', 'B'],
        'time': [3, 1, 2, 2, 1],  # Out of order
        'value': [30, 10, 20, 20, 10]
    })

    mask = first_and_last_per_group(df, 'group', 'time')

    # For group A: time 1 (row 1) is first, time 3 (row 0) is last
    # For group B: time 1 (row 4) is first, time 2 (row 3) is last
    expected = np.array([True, True, False, True, True])
    np.testing.assert_array_equal(mask, expected)


def test_periodic_anchors_respects_time_order():
    """Test that periodic_anchors uses time_col for ordering within periods."""
    from sift.sampling.anchors import periodic_anchors

    df = pd.DataFrame({
        'group': ['A', 'A', 'A', 'A'],
        'month': [1, 1, 2, 2],
        'time': [2, 1, 2, 1],  # Out of order within each month
        'value': [20, 10, 20, 10]
    })

    anchor_fn = periodic_anchors('month')
    mask = anchor_fn(df, 'group', 'time')

    # For month 1: time 1 (row 1) is first
    # For month 2: time 1 (row 3) is first
    expected = np.array([False, True, False, True])
    np.testing.assert_array_equal(mask, expected)


def test_anchor_max_share_zero_excludes_all_anchors():
    """Test that anchor_max_share=0 excludes all anchors."""
    from sift.sampling.smart import SmartSamplerConfig, smart_sample
    from sift.sampling.anchors import first_per_group

    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'f0': np.random.randn(n),
        'f1': np.random.randn(n),
        'group': np.repeat(np.arange(10), 10),
        'y': np.random.randn(n)
    })

    config = SmartSamplerConfig(
        sample_frac=0.5,
        group_col='group',
        anchor_fn=first_per_group,
        anchor_max_share=0.0,  # Should exclude all anchors
        verbose=False
    )

    # Should not raise and should return samples
    result = smart_sample(df, ['f0', 'f1'], 'y', config)
    assert len(result) > 0


def test_selected_features_sorted_by_frequency():
    """Test that selected features are ordered by selection frequency."""
    np.random.seed(42)
    n, p = 200, 5
    X = pd.DataFrame(np.random.randn(n, p), columns=[f'f{i}' for i in range(p)])
    y = 2.0 * X['f0'] + 0.2 * X['f1'] + np.random.randn(n) * 0.3

    selector = StabilitySelector(
        n_bootstrap=20,
        threshold=0.1,
        alpha=0.01,
        random_state=0,
        n_jobs=1,
        verbose=False,
    )
    selector.fit(X, y)

    selected_names = selector.selected_feature_names_
    selected_freqs = selector.selection_frequencies_[selector.selected_features_]

    assert len(selected_names) > 0
    assert np.all(selected_freqs[:-1] >= selected_freqs[1:])


def test_stability_regression_returns_features_ordered_by_frequency():
    """Test that selected features are ordered by selection frequency (descending)."""
    from sift import stability_regression

    np.random.seed(42)
    n, p = 300, 20
    X = pd.DataFrame(np.random.randn(n, p), columns=[f'f{i}' for i in range(p)])
    # f0 has strong signal, f1 has moderate signal, rest are noise
    y = 3 * X['f0'] + 1 * X['f1'] + np.random.randn(n) * 0.5

    selected = stability_regression(
        X,
        y,
        k=10,
        n_bootstrap=30,
        threshold=0.3,
        alpha=0.01,
        verbose=False,
        random_state=42
    )

    assert len(selected) >= 2, "Should select at least 2 features"
