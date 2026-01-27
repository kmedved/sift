import numpy as np
import pandas as pd

from sift.sampling.smart import SmartSamplerConfig, smart_sample


def test_smart_sample_does_not_drop_nan_groups():
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame(
        {
            "group": rng.choice(["a", "b", None], size=n, p=[0.45, 0.45, 0.10]),
            "time": np.arange(n),
            "f0": rng.normal(size=n),
            "f1": rng.normal(size=n),
            "y": rng.normal(size=n),
            "row_id": np.arange(n),
        }
    )

    config = SmartSamplerConfig(
        sample_frac=0.5,
        group_col="group",
        time_col="time",
        residual_weight_cap=0.0,
        verbose=False,
    )
    out = smart_sample(df, ["f0", "f1"], "y", config)

    assert len(out) > 0
    assert out["row_id"].isin(df["row_id"]).all()

    config_all = SmartSamplerConfig(
        sample_frac=1.0,
        group_col="group",
        time_col="time",
        residual_weight_cap=0.0,
        verbose=False,
    )
    out_all = smart_sample(df, ["f0", "f1"], "y", config_all)
    assert len(out_all) == len(df)


def test_smart_sample_nan_group_does_not_collide_with_existing_group_value():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "group": ["__SIFT_MISSING_GROUP__", None, "__SIFT_MISSING_GROUP__", None],
            "time": [0, 1, 2, 3],
            "f0": rng.normal(size=4),
            "f1": rng.normal(size=4),
            "y": rng.normal(size=4),
            "row_id": np.arange(4),
        }
    )
    config = SmartSamplerConfig(
        sample_frac=0.1,
        min_per_group=1,
        group_col="group",
        time_col="time",
        residual_weight_cap=0.0,
        verbose=False,
    )
    out = smart_sample(df, ["f0", "f1"], "y", config)
    assert len(out) >= 2
    assert out["group"].isna().any()
    assert (out["group"] == "__SIFT_MISSING_GROUP__").any()
