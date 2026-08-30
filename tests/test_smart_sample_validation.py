import numpy as np
import pandas as pd
import pytest

import sift.sampling.smart as smart_module
from sift.sampling.smart import (
    SmartSamplerConfig,
    _compute_residual_scores,
    _prepare_smart_arrays,
    smart_sample,
)


def _sample_frame(n: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "group": np.repeat(["a", "b", "c"], n // 3),
            "time": np.arange(n),
            "f0": rng.normal(size=n),
            "f1": rng.normal(size=n),
            "y": rng.normal(size=n),
        }
    )


def test_smart_sample_does_not_mutate_caller_config_when_overrides_supplied():
    df = _sample_frame()
    config = SmartSamplerConfig(sample_frac=0.4, residual_weight_cap=0.0, verbose=False)

    smart_sample(
        df,
        ["f0", "f1"],
        "y",
        config=config,
        sample_frac=0.2,
        min_per_group=1,
    )

    assert config.sample_frac == 0.4
    assert config.min_per_group == 2
    assert config.residual_weight_cap == 0.0


@pytest.mark.parametrize(
    "override, match",
    [
        ({"sample_frac": 0.0}, "sample_frac"),
        ({"min_per_group": 0}, "min_per_group"),
        ({"pilot_sample_size": 0}, "pilot_sample_size"),
        ({"leverage_batch_size": 0}, "leverage_batch_size"),
        ({"svd_sample_size": 0}, "svd_sample_size"),
        ({"weight_clip_quantile": 1.5}, "weight_clip_quantile"),
        ({"residual_weight_cap": -0.1}, "residual_weight_cap"),
        ({"uniform_floor": -0.1}, "uniform_floor"),
        ({"anchor_max_share": 1.5}, "anchor_max_share"),
        ({"random_state": 1.2}, "random_state"),
        ({"anchor_fn": "not-callable"}, "anchor_fn"),
        ({"group_col": 1}, "group_col"),
        ({"time_col": 1}, "time_col"),
    ],
)
def test_smart_sample_rejects_invalid_config_overrides(override, match):
    df = _sample_frame()
    config = SmartSamplerConfig(sample_frac=0.4, residual_weight_cap=0.0, verbose=False)

    with pytest.raises((TypeError, ValueError), match=match):
        smart_sample(df, ["f0", "f1"], "y", config=config, **override)

    assert config.sample_frac == 0.4
    assert config.residual_weight_cap == 0.0


def test_smart_sampler_config_constructor_validates_bad_values():
    with pytest.raises(ValueError, match="sample_frac"):
        SmartSamplerConfig(sample_frac=float("nan"))


def test_smart_sample_revalidates_mutated_config_without_overrides():
    df = _sample_frame()
    config = SmartSamplerConfig(sample_frac=0.4, residual_weight_cap=0.0, verbose=False)
    config.residual_weight_cap = -1.0

    with pytest.raises(ValueError, match="residual_weight_cap"):
        smart_sample(df, ["f0", "f1"], "y", config=config)


def test_smart_sampler_accepts_numpy_bool_verbose():
    df = _sample_frame()
    config = SmartSamplerConfig(
        sample_frac=0.4,
        residual_weight_cap=0.0,
        verbose=np.bool_(False),
    )

    out = smart_sample(df, ["f0", "f1"], "y", config=config)

    assert len(out) > 0


def test_smart_sample_residual_disabled_accepts_non_float_target():
    df = _sample_frame(30)
    df["label"] = np.where(df["time"] % 2 == 0, "win", "loss")
    config = SmartSamplerConfig(
        sample_frac=0.4,
        residual_weight_cap=0.0,
        min_per_group=1,
        random_state=0,
        verbose=False,
    )

    out = smart_sample(df, ["f0", "f1"], "label", config=config)

    assert len(out) > 0
    assert "sample_weight" in out.columns
    assert np.isfinite(out["sample_weight"]).all()


def test_smart_sample_dense_probabilities_are_deterministic():
    df = _sample_frame(60)
    config = SmartSamplerConfig(
        sample_frac=0.35,
        residual_weight_cap=0.0,
        min_per_group=2,
        group_col="group",
        random_state=7,
        verbose=False,
    )

    out1 = smart_sample(df, ["f0", "f1"], "y", config=config)
    out2 = smart_sample(df, ["f0", "f1"], "y", config=config)

    pd.testing.assert_frame_equal(out1.reset_index(drop=True), out2.reset_index(drop=True))
    assert out1["time"].is_monotonic_increasing


def test_smart_sample_residual_enabled_is_deterministic_with_nonpilot_rows():
    df = _sample_frame(240)
    config = SmartSamplerConfig(
        sample_frac=0.35,
        pilot_sample_size=100,
        residual_weight_cap=0.4,
        min_per_group=2,
        group_col="group",
        random_state=8,
        verbose=False,
    )

    out1 = smart_sample(df, ["f0", "f1"], "y", config=config)
    out2 = smart_sample(df, ["f0", "f1"], "y", config=config)

    pd.testing.assert_frame_equal(out1.reset_index(drop=True), out2.reset_index(drop=True))


def test_prepare_smart_arrays_preserves_large_offset_target_precision():
    df = pd.DataFrame(
        {
            "f0": np.arange(128, dtype=float),
            "y": 1e12 + np.arange(128, dtype=float),
        }
    )
    config = SmartSamplerConfig(residual_weight_cap=0.4, verbose=False)

    arrays = _prepare_smart_arrays(df, ["f0"], "y", config)

    assert arrays.y is not None
    assert arrays.y.dtype == np.float64
    assert np.unique(arrays.y).size == len(df)


def test_residual_scores_are_invariant_to_large_target_offset():
    rng = np.random.default_rng(21)
    n = 240
    signal = np.tile(np.arange(12, dtype=float), n // 12)
    X = np.column_stack([signal, rng.normal(size=n)]).astype(np.float32)
    config = SmartSamplerConfig(
        pilot_sample_size=200,
        residual_weight_cap=0.4,
        random_state=7,
        verbose=False,
    )

    scores, beta = _compute_residual_scores(
        X,
        signal,
        config,
        np.random.default_rng(22),
    )
    shifted_scores, shifted_beta = _compute_residual_scores(
        X,
        1e12 + signal,
        config,
        np.random.default_rng(22),
    )

    assert beta > 0.0
    assert np.std(scores) > 0.0
    np.testing.assert_allclose(shifted_scores, scores, rtol=0.0, atol=1e-12)
    assert shifted_beta == pytest.approx(beta, abs=1e-12)


def test_residual_pilot_cross_fits_every_prediction(monkeypatch):
    class RecordingRegressor:
        instances = []
        overlap_seen = False

        def __init__(self, **_kwargs):
            self.train_ids = set()
            self.offset = 0.0
            self.instances.append(self)

        def fit(self, X, y):
            ids = np.asarray(X[:, 0], dtype=int)
            self.train_ids = set(ids.tolist())
            self.offset = float(np.mean(np.asarray(y) - 2.0 * ids))
            return self

        def predict(self, X):
            ids = np.asarray(X[:, 0], dtype=int)
            if self.train_ids.intersection(ids.tolist()):
                type(self).overlap_seen = True
            return 2.0 * ids + self.offset

    monkeypatch.setattr(
        smart_module,
        "HistGradientBoostingRegressor",
        RecordingRegressor,
    )
    row_ids = np.arange(120, dtype=np.float32)
    X = np.column_stack([row_ids, np.ones_like(row_ids)])
    y = 2.0 * row_ids.astype(np.float64) + (row_ids % 3)
    config = SmartSamplerConfig(
        pilot_sample_size=100,
        residual_weight_cap=0.4,
        random_state=0,
        verbose=False,
    )

    scores, beta = _compute_residual_scores(
        X,
        y,
        config,
        np.random.default_rng(0),
    )

    assert len(RecordingRegressor.instances) == 2
    assert RecordingRegressor.overlap_seen is False
    assert np.isfinite(scores).all()
    assert beta > 0.0


def test_nonpilot_rows_use_one_model_residuals_on_the_same_scale(monkeypatch):
    class SymmetricErrorRegressor:
        instances = []

        def __init__(self, **_kwargs):
            self.offset = 1.0 if not self.instances else -1.0
            self.center = 0.0
            self.instances.append(self)

        def fit(self, X, y):
            self.center = float(np.mean(np.asarray(y) - X[:, 0]))
            return self

        def predict(self, X):
            return X[:, 0] + self.center + self.offset

    monkeypatch.setattr(
        smart_module,
        "HistGradientBoostingRegressor",
        SymmetricErrorRegressor,
    )
    n = 240
    pilot_size = 100
    seed = 32
    row_ids = np.arange(n, dtype=np.float32)
    X = np.column_stack([row_ids, np.ones_like(row_ids)])
    y = row_ids.astype(np.float64)
    config = SmartSamplerConfig(
        pilot_sample_size=pilot_size,
        residual_weight_cap=0.4,
        random_state=9,
        verbose=False,
    )

    scores, beta = _compute_residual_scores(
        X,
        y,
        config,
        np.random.default_rng(seed),
    )
    pilot = np.random.default_rng(seed).choice(n, size=pilot_size, replace=False)
    nonpilot_mask = np.ones(n, dtype=bool)
    nonpilot_mask[pilot] = False

    assert len(SymmetricErrorRegressor.instances) == 2
    assert beta > 0.0
    assert np.mean(scores[pilot]) == pytest.approx(1.0, abs=1e-12)
    assert np.mean(scores[nonpilot_mask]) == pytest.approx(1.0, abs=1e-12)


def test_residual_pilot_centers_on_robust_pilot_location(monkeypatch):
    class RecordingRegressor:
        median_abs_targets = []

        def __init__(self, **_kwargs):
            pass

        def fit(self, X, y):
            self.median_abs_targets.append(float(np.median(np.abs(y))))
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=np.float64)

    monkeypatch.setattr(
        smart_module,
        "HistGradientBoostingRegressor",
        RecordingRegressor,
    )
    n = 120
    seed = 31
    pilot_order = np.random.default_rng(seed).choice(n, size=100, replace=False)
    y = 1e12 + np.arange(n, dtype=np.float64)
    y[pilot_order[0]] = 0.0
    X = np.column_stack(
        [np.arange(n, dtype=np.float32), np.ones(n, dtype=np.float32)]
    )
    config = SmartSamplerConfig(
        pilot_sample_size=100,
        residual_weight_cap=0.4,
        random_state=0,
        verbose=False,
    )

    _compute_residual_scores(X, y, config, np.random.default_rng(seed))

    assert len(RecordingRegressor.median_abs_targets) == 2
    assert max(RecordingRegressor.median_abs_targets) < 100.0


def test_constant_target_disables_residual_blend():
    rng = np.random.default_rng(23)
    X = rng.normal(size=(120, 3)).astype(np.float32)
    config = SmartSamplerConfig(
        pilot_sample_size=100,
        residual_weight_cap=0.4,
        random_state=0,
        verbose=False,
    )

    scores, beta = _compute_residual_scores(
        X,
        np.full(120, 1e12),
        config,
        np.random.default_rng(0),
    )

    assert beta == 0.0
    np.testing.assert_array_equal(scores, np.ones(120, dtype=np.float32))
