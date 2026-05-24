import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from sift import StabilitySelector
import sift.stability as stability_module
from sift.sampling.smart import SmartSamplerConfig, smart_sample
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


def test_plot_coef_distributions_rejects_empty_feature_list():
    selector = StabilitySelector(verbose=False)
    selector.coef_bootstrap_ = np.empty((1, 0), dtype=np.float32)

    with pytest.raises(ValueError, match="features must contain"):
        selector.plot_coef_distributions(features=[])


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
    assert len(selected) == 10
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
    assert len(selected) == 10
    assert all(isinstance(f, str) for f in selected)


def test_stability_selector_validates_sample_weight():
    rng = np.random.default_rng(123)
    n, p = 40, 5
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = X["f0"] + rng.normal(size=n) * 0.1

    bad_weights = [
        np.ones(n - 1),
        np.r_[np.ones(n - 1), -1.0],
        np.r_[np.ones(n - 1), np.nan],
        np.zeros(n),
    ]

    for weights in bad_weights:
        selector = StabilitySelector(
            n_bootstrap=2,
            threshold=0.1,
            alpha=0.1,
            n_jobs=1,
            verbose=False,
        )
        with pytest.raises(ValueError, match="sample_weight"):
            selector.fit(X, y, sample_weight=weights)


@pytest.mark.parametrize("method_name", ["transform", "get_feature_info", "get_support"])
def test_stability_selector_public_methods_require_fit(method_name):
    selector = StabilitySelector(verbose=False)

    with pytest.raises(NotFittedError):
        if method_name == "transform":
            getattr(selector, method_name)(np.zeros((3, 2)))
        else:
            getattr(selector, method_name)()


@pytest.mark.parametrize(
    "selector_kwargs, match",
    [
        ({"task": "bad"}, "task must be"),
        ({"block_method": "bad"}, "block_method must be"),
        ({"parallel_backend": "bad"}, "parallel_backend must be"),
        ({"sample_frac": 0}, "sample_frac must be"),
        ({"sample_frac": 1.5}, "sample_frac must be"),
        ({"n_bootstrap": 0}, "n_bootstrap must be"),
        ({"threshold": -0.1}, "threshold must be"),
    ],
)
def test_stability_selector_validates_runtime_options(selector_kwargs, match):
    rng = np.random.default_rng(321)
    X = pd.DataFrame(rng.normal(size=(20, 4)), columns=[f"f{i}" for i in range(4)])
    y = X["f0"] + rng.normal(size=20) * 0.1

    selector = StabilitySelector(n_jobs=1, verbose=False, **selector_kwargs)

    with pytest.raises(ValueError, match=match):
        selector.fit(X, y)


def test_tune_threshold_reuses_fit_time_imputation(monkeypatch):
    rng = np.random.default_rng(777)
    X_fit = pd.DataFrame(
        {
            "f0": [1.0, 3.0, np.nan, 7.0, 9.0, 11.0],
            "f1": [2.0, np.nan, 6.0, 8.0, 10.0, 12.0],
        }
    )
    y_fit = X_fit["f0"].fillna(0).to_numpy() + rng.normal(size=len(X_fit)) * 0.01

    selector = StabilitySelector(
        n_bootstrap=2,
        threshold=0.1,
        alpha=0.1,
        n_jobs=1,
        verbose=False,
    )
    selector.fit(X_fit, y_fit)

    selector.selection_frequencies_ = np.array([1.0, 0.0], dtype=np.float32)

    X_tune = pd.DataFrame(
        {
            "f0": [100.0, np.nan, 300.0, 400.0],
            "f1": [np.nan, 500.0, 600.0, 700.0],
        }
    )
    y_tune = np.array([1.0, 2.0, 3.0, 4.0])

    captured = {}

    def fake_cross_val_score(model, X, y, cv=None, scoring=None):
        captured["X"] = np.array(X, copy=True)
        captured["y"] = np.array(y, copy=True)
        captured["cv"] = cv
        captured["scoring"] = scoring
        return np.array([0.25, 0.5, 0.75], dtype=np.float32)

    monkeypatch.setattr("sklearn.model_selection.cross_val_score", fake_cross_val_score)

    best_threshold, results = selector.tune_threshold(X_tune, y_tune, thresholds=[0.5], cv=3)

    assert best_threshold == 0.5
    assert results.loc[0, "n_features"] == 1
    assert captured["cv"] == 3
    assert captured["scoring"] == "r2"

    expected_imputed = np.array(
        [
            [100.0, 7.6],
            [6.2, 500.0],
            [300.0, 600.0],
            [400.0, 700.0],
        ],
        dtype=np.float32,
    )
    expected_scaled = selector._scaler.transform(expected_imputed)

    np.testing.assert_allclose(captured["X"], expected_scaled[:, [0]])


def test_smart_sampler_config_is_not_mutated():
    rng = np.random.default_rng(456)
    n, p = 80, 4
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = np.where(X["f0"] > 0, "win", "loss")
    config = SmartSamplerConfig(
        sample_frac=0.5,
        min_per_group=1,
        residual_weight_cap=0.4,
        random_state=None,
        verbose=True,
    )

    selector = StabilitySelector(
        n_bootstrap=2,
        threshold=0.1,
        alpha=0.1,
        task="classification",
        use_smart_sampler=True,
        sampler_config=config,
        n_jobs=1,
        random_state=7,
        verbose=False,
    )
    selector.fit(X, y)

    assert config.residual_weight_cap == 0.4
    assert config.random_state is None
    assert config.verbose is True


def test_smart_sample_full_fraction_keeps_all_rows_without_residual_y_check():
    rng = np.random.default_rng(457)
    df = pd.DataFrame(
        {
            "f0": rng.normal(size=24),
            "f1": rng.normal(size=24),
            "y": np.r_[np.nan, rng.normal(size=23)],
        }
    )
    config = SmartSamplerConfig(
        sample_frac=1.0,
        residual_weight_cap=0.0,
        random_state=5,
        verbose=False,
    )

    sampled = smart_sample(df, ["f0", "f1"], "y", config)

    assert len(sampled) == len(df)
    assert sampled.index.tolist() == list(range(len(df)))
    assert np.isfinite(sampled["sample_weight"]).all()
    np.testing.assert_allclose(sampled["sample_weight"].mean(), 1.0)


def test_smart_sample_accepts_list_anchor_mask():
    rng = np.random.default_rng(458)
    df = pd.DataFrame(
        {
            "row_id": np.arange(30),
            "f0": rng.normal(size=30),
            "f1": rng.normal(size=30),
            "y": rng.normal(size=30),
        }
    )

    def anchor_fn(frame, _group_col, _time_col):
        return [i < 3 for i in range(len(frame))]

    sampled = smart_sample(
        df,
        ["f0", "f1"],
        "y",
        SmartSamplerConfig(
            sample_frac=0.2,
            anchor_fn=anchor_fn,
            anchor_max_share=1.0,
            residual_weight_cap=0.0,
            random_state=6,
            verbose=False,
        ),
    )

    assert set(range(3)).issubset(set(sampled["row_id"]))


def test_smart_sample_rejects_wrong_length_anchor_mask():
    rng = np.random.default_rng(459)
    df = pd.DataFrame(
        {
            "f0": rng.normal(size=20),
            "f1": rng.normal(size=20),
            "y": rng.normal(size=20),
        }
    )

    def anchor_fn(_frame, _group_col, _time_col):
        return [True, False]

    with pytest.raises(ValueError, match="anchor_fn returned"):
        smart_sample(
            df,
            ["f0", "f1"],
            "y",
            SmartSamplerConfig(
                anchor_fn=anchor_fn,
                residual_weight_cap=0.0,
                verbose=False,
            ),
        )


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


def test_stability_regression_wrapper_fills_to_requested_k_when_threshold_high():
    from sift import stability_regression

    rng = np.random.default_rng(123)
    n, p = 160, 12
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = X["f0"] + 0.2 * X["f1"] + rng.normal(size=n) * 0.8

    selected = stability_regression(
        X,
        y,
        k=6,
        threshold=0.999,
        n_bootstrap=8,
        alpha=0.05,
        n_jobs=1,
        random_state=5,
        verbose=False,
    )

    assert len(selected) == 6


def test_stability_alpha_cv_uses_group_splits_without_group_overlap():
    selector = StabilitySelector(alpha=None, task="regression", verbose=False)
    idx = np.arange(12)
    groups = np.repeat(np.arange(4), 3)
    cv = selector._alpha_cv(idx, np.zeros(12), groups=groups, time=None)

    assert isinstance(cv, list)
    for train_idx, val_idx in cv:
        train_groups = set(groups[train_idx].tolist())
        val_groups = set(groups[val_idx].tolist())
        assert train_groups.isdisjoint(val_groups)


def test_stability_bootstrap_duplicate_rows_sum_weights(monkeypatch):
    captured = {}

    class FakeLasso:
        def __init__(self, *, alpha, max_iter):
            self.alpha = alpha
            self.max_iter = max_iter

        def fit(self, X, y, sample_weight):
            captured["X"] = np.asarray(X)
            captured["y"] = np.asarray(y)
            captured["sample_weight"] = np.asarray(sample_weight)
            self.coef_ = np.array([1.0, 0.0])
            return self

    monkeypatch.setattr(stability_module, "Lasso", FakeLasso)
    selector = StabilitySelector(alpha=0.1, task="regression", verbose=False)
    selector.alpha_ = 0.1
    X = np.arange(8, dtype=float).reshape(4, 2)
    y = np.arange(4, dtype=float)
    sample_weight = np.array([1.0, 3.0, 2.0, 5.0])

    selector._fit_single_stability_run(
        X,
        y,
        sample_weight,
        train_idx=np.array([2, 0, 2, 1]),
        seed=0,
    )

    np.testing.assert_array_equal(captured["X"], X[[0, 1, 2]])
    np.testing.assert_array_equal(captured["y"], y[[0, 1, 2]])
    np.testing.assert_allclose(captured["sample_weight"], [1.0, 3.0, 4.0])


def test_stability_refit_without_stored_coefs_clears_old_coef_matrix():
    rng = np.random.default_rng(50)
    X = pd.DataFrame(rng.normal(size=(120, 5)), columns=[f"f{i}" for i in range(5)])
    y = X["f0"] + rng.normal(size=120) * 0.1

    selector = StabilitySelector(
        n_bootstrap=5,
        threshold=0.1,
        alpha=0.05,
        store_coefs=True,
        n_jobs=1,
        random_state=0,
        verbose=False,
    ).fit(X, y)
    assert hasattr(selector, "coef_bootstrap_")

    selector.store_coefs = False
    selector.fit(X, y)

    assert not hasattr(selector, "coef_bootstrap_")
    with pytest.raises(ValueError, match="store_coefs=True"):
        selector.get_coef_stability()


def test_stability_failed_refit_clears_partial_and_old_state(monkeypatch):
    rng = np.random.default_rng(51)
    X = pd.DataFrame(rng.normal(size=(100, 4)), columns=[f"f{i}" for i in range(4)])
    y = X["f0"] + rng.normal(size=100) * 0.1
    selector = StabilitySelector(
        n_bootstrap=5,
        threshold=0.1,
        alpha=0.05,
        n_jobs=1,
        random_state=0,
        verbose=False,
    ).fit(X, y)
    assert hasattr(selector, "selection_frequencies_")

    X_new = X.rename(columns={col: f"new_{col}" for col in X.columns})

    def fail_run(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(selector, "_run_stability_chunks", fail_run)
    with pytest.raises(RuntimeError, match="boom"):
        selector.fit(X_new, y)

    with pytest.raises(NotFittedError):
        selector.get_feature_info()
    with pytest.raises(NotFittedError):
        selector.transform(X_new)
