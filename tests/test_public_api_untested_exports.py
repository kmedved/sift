import numpy as np
import pandas as pd
import pytest

from sift import panel_config, select_k_elbow, smart_sample
from sift.sampling.anchors import first_per_group


def test_select_k_elbow_stops_after_plateau_patience():
    path = np.concatenate([np.linspace(0.5, 3.0, 5), np.full(15, 3.0)])

    best_k, diag = select_k_elbow(path, min_k=2, max_k=20, min_rel_gain=0.02, patience=3)

    assert best_k == 5
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


def test_select_k_elbow_accepts_integer_paths_and_excludes_first_flat_gain():
    best_k, diag = select_k_elbow(
        np.array([1, 2, 3, 4, 5, 5, 5, 5]),
        min_k=1,
        max_k=8,
        min_rel_gain=0.02,
        patience=3,
    )

    assert best_k == 5
    assert diag[["objective", "delta", "rel_gain"]].dtypes.eq(np.dtype("float64")).all()


def test_select_k_elbow_counts_a_plateau_beginning_at_minimum_k():
    path = np.array([1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0, 10.0, 20.0, 30.0])

    best_k, _ = select_k_elbow(
        path,
        min_k=5,
        max_k=10,
        min_rel_gain=0.02,
        patience=3,
    )

    assert best_k == 5


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_k": -1}, "min_k"),
        ({"max_k": 0}, "max_k"),
        ({"min_k": 3, "max_k": 2}, "min_k"),
        ({"min_rel_gain": np.nan}, "min_rel_gain"),
        ({"patience": 0}, "patience"),
    ],
)
def test_select_k_elbow_validates_direct_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        select_k_elbow(np.array([1.0, 2.0, 3.0]), **kwargs)


def test_select_k_elbow_rejects_nonfinite_or_nonvector_paths():
    with pytest.raises(ValueError, match="finite"):
        select_k_elbow(np.array([1.0, np.nan]))
    with pytest.raises(ValueError, match="one-dimensional"):
        select_k_elbow(np.array([[1.0, 2.0]]))


def test_panel_config_wires_fields():
    cfg = panel_config(group_col="g", time_col="t", size=0.3)

    assert cfg.group_col == "g"
    assert cfg.time_col == "t"
    assert cfg.size == 0.3
    assert cfg.anchor_fn is None


def test_panel_config_with_first_per_group_anchor_keeps_every_entity():
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

    cfg = panel_config("g", "t", size=0.2)
    cfg.anchor_fn = first_per_group
    out = smart_sample(df, ["x1"], "y", config=cfg)

    assert len(out) == int(0.2 * len(df))
    assert out["g"].nunique() == n_groups
    for _g, grp in out.groupby("g"):
        assert grp["t"].min() == 0
