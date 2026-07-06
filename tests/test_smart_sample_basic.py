import numpy as np
import pandas as pd

from sift.sampling.smart import SmartSamplerConfig, smart_sample


def test_smart_sample_basic_weights():
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame(
        {
            "f0": rng.normal(size=n),
            "f1": rng.normal(size=n),
            "y": rng.normal(size=n),
        }
    )

    config = SmartSamplerConfig(sample_frac=0.4, residual_weight_cap=0.0, verbose=False)
    out = smart_sample(df, ["f0", "f1"], "y", config)

    assert "sample_weight" in out.columns
    assert out["sample_weight"].notna().all()
    assert np.isfinite(out["sample_weight"]).all()


def test_smart_sample_small_svd_sample_size():
    rng = np.random.default_rng(1)
    n = 80
    df = pd.DataFrame(
        {
            "f0": rng.normal(size=n),
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "y": rng.normal(size=n),
        }
    )

    config = SmartSamplerConfig(
        sample_frac=0.5,
        residual_weight_cap=0.0,
        svd_sample_size=10,
        verbose=False,
    )
    out = smart_sample(df, ["f0", "f1", "f2"], "y", config)

    assert len(out) > 0
