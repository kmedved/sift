import numpy as np
import pandas as pd

from sift.sampling.anchors import first_and_last_per_group
from sift.sampling.smart import SmartSamplerConfig, _cap_group_anchors, smart_sample


def test_smart_sample_keeps_anchors():
    rng = np.random.default_rng(0)
    rows_per_group = 6
    groups = np.repeat(["a", "b", "c"], rows_per_group)
    n = groups.size
    df = pd.DataFrame(
        {
            "row_id": np.arange(n),
            "group": groups,
            "time": np.tile(np.arange(rows_per_group), 3),
            "f0": rng.normal(size=n),
            "f1": rng.normal(size=n),
            "y": rng.normal(size=n),
        }
    )

    config = SmartSamplerConfig(
        sample_frac=0.5,
        group_col="group",
        time_col="time",
        anchor_fn=first_and_last_per_group,
        anchor_max_share=1.0,
        residual_weight_cap=0.0,
        verbose=False,
    )

    out = smart_sample(df, ["f0", "f1"], "y", config)
    anchor_mask = first_and_last_per_group(df, "group", "time")
    anchor_ids = set(df.loc[anchor_mask, "row_id"])
    sampled_ids = set(out["row_id"])

    assert anchor_ids.issubset(sampled_ids)


def test_positive_anchor_max_share_keeps_at_least_one_mandatory_anchor():
    n = 10
    g_idx = np.arange(n)
    anchor_mask = np.ones(n, dtype=bool)
    base_scores = np.arange(n, dtype=float)
    config = SmartSamplerConfig(
        anchor_max_share=1e-9,
        verbose=False,
    )

    kept = _cap_group_anchors(g_idx, anchor_mask, base_scores, target_g=5, config=config)

    assert kept.size == 1
    assert kept[0] == 9
