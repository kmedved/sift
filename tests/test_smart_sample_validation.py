import numpy as np
import pandas as pd
import pytest

from sift.sampling.smart import SmartSamplerConfig, smart_sample


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
