import numpy as np
import pandas as pd

from sift import cross_section_config, panel_config, select_k_elbow, smart_sample
from sift.sampling.anchors import first_and_last_per_group


def test_select_k_elbow_stops_after_plateau_patience():
    path = np.concatenate([np.linspace(0.5, 3.0, 5), np.full(15, 3.0)])

    best_k, diag = select_k_elbow(path, min_k=2, max_k=20, min_rel_gain=0.02, patience=3)

    assert best_k == 6
    assert not diag.empty
    assert diag["k"].is_monotonic_increasing


def test_select_k_elbow_monotone_gains_hits_max_k():
    path = np.cumsum(np.full(30, 1.0))

    best_k, _ = select_k_elbow(path, min_k=2, max_k=10, min_rel_gain=0.02, patience=3)

    assert best_k == 10


def test_select_k_elbow_empty_path():
    best_k, diag = select_k_elbow(np.array([]))

    assert best_k == 0
    assert diag.empty


def test_panel_config_wires_fields_and_anchors():
    cfg = panel_config(group_col="g", time_col="t", sample_frac=0.3)

    assert cfg.group_col == "g"
    assert cfg.time_col == "t"
    assert cfg.sample_frac == 0.3
    assert cfg.min_per_group == 2
    assert cfg.anchor_fn is first_and_last_per_group


def test_panel_config_through_smart_sample_keeps_group_endpoints():
    rng = np.random.default_rng(0)
    n_groups, per_group = 30, 40
    df = pd.DataFrame(
        {
            "g": np.repeat(np.arange(n_groups), per_group),
            "t": np.tile(np.arange(per_group), n_groups),
            "x1": rng.normal(size=n_groups * per_group),
            "y": rng.normal(size=n_groups * per_group),
        }
    )

    out = smart_sample(df, ["x1"], "y", config=panel_config("g", "t", sample_frac=0.2))

    for _g, grp in out.groupby("g"):
        assert grp["t"].min() == 0
        assert grp["t"].max() == per_group - 1
    assert 0 < len(out) < len(df)


def test_cross_section_config_through_smart_sample():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x1": rng.normal(size=5000), "y": rng.normal(size=5000)})
    cfg = cross_section_config(sample_frac=0.15)

    assert cfg.group_col is None
    assert cfg.anchor_fn is None
    assert cfg.min_per_group == 1
    out = smart_sample(df, ["x1"], "y", config=cfg)

    assert 0 < len(out) < len(df)
    assert abs(len(out) / len(df) - 0.15) < 0.08
