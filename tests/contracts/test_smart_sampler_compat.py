"""Public contracts for ``SmartSamplerConfig``, ``smart_sample`` and ``panel_config`` (sift 2.0)."""

from dataclasses import fields, replace
import warnings

import numpy as np
import pandas as pd
import pytest

import sift
from sift.sampling import first_per_group

EXPECTED_DEFAULTS = {
    "size": 0.2,
    "method": "residual",
    "group_col": None,
    "time_col": None,
    "strata_col": None,
    "task": "auto",
    "pilot": "auto",
    "pilot_size": 0.02,
    "uniform_mix": 0.5,
    "anchor_fn": None,
    "random_state": 42,
    "verbose": False,
}

REMOVED_IN_2_0 = (
    "sample_frac",
    "pilot_sample_size",
    "min_per_group",
    "leverage_batch_size",
    "svd_sample_size",
    "weight_clip_quantile",
    "residual_weight_cap",
    "uniform_floor",
    "anchor_max_share",
)


def _frame(n=1_200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    f0, f1 = rng.normal(size=n), rng.normal(size=n)
    return pd.DataFrame(
        {
            "group": np.repeat(["b", "a", "c"], n // 3),
            "time": np.tile(np.arange(n // 3), 3),
            "f0": f0,
            "f1": f1,
            "y": f0 + 0.2 * f1 + rng.standard_t(3, size=n),
            "label": np.arange(n).astype(str),
        },
        index=np.arange(10_000, 10_000 + n),
    )


def test_smart_sampler_config_exact_public_defaults():
    assert [f.name for f in fields(sift.SmartSamplerConfig)] == list(EXPECTED_DEFAULTS)
    config = sift.SmartSamplerConfig()
    assert {f.name: getattr(config, f.name) for f in fields(config)} == EXPECTED_DEFAULTS


@pytest.mark.parametrize("name", REMOVED_IN_2_0)
def test_removed_1x_fields_fail_with_a_migration_hint(name):
    with pytest.raises(TypeError, match=f"{name}:"):
        sift.smart_sample(_frame(), ["f0", "f1"], "y", **{name: 1})
    with pytest.raises(TypeError):
        sift.SmartSamplerConfig(**{name: 1})


def test_keyword_overrides_equal_explicit_config_and_are_warning_free():
    options = dict(size=0.25, method="chronological", group_col="group", time_col="time",
                   random_state=7)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        by_keyword = sift.smart_sample(_frame(), ["f0", "f1"], "y", **options)
        by_config = sift.smart_sample(_frame(), ["f0", "f1"], "y", sift.SmartSamplerConfig(**options))
    pd.testing.assert_frame_equal(by_keyword, by_config)
    assert type(by_keyword) is pd.DataFrame
    assert by_keyword.columns.tolist() == ["group", "time", "f0", "f1", "y", "label", "sample_weight"]
    assert len(by_keyword) == 300
    assert by_keyword.groupby("group", sort=False).size().to_dict() == {"b": 100, "a": 100, "c": 100}
    assert by_keyword.index.is_monotonic_increasing
    assert by_keyword.index.isin(_frame().index).all()
    assert by_keyword["sample_weight"].dtype == np.float64
    assert by_keyword["sample_weight"].mean() == pytest.approx(1.0)


def test_default_pilot_sample_is_exact_and_preserves_input_rows():
    source = _frame()
    out = sift.smart_sample(source, ["f0", "f1"], "y", size=240, random_state=1)
    assert len(out) == 240
    pd.testing.assert_frame_equal(out.drop(columns="sample_weight"), source.loc[out.index])
    assert "sample_weight" not in source.columns


def test_full_size_keeps_every_row_in_order_with_unit_weights():
    source = _frame(n=30)
    out = sift.smart_sample(source.copy(), ["f0", "f1"], "y", size=1.0)
    pd.testing.assert_frame_equal(out.drop(columns="sample_weight"), source)
    np.testing.assert_array_equal(out["sample_weight"].to_numpy(), np.ones(30))


def test_overrides_do_not_mutate_the_callers_config():
    config = sift.SmartSamplerConfig(size=0.5, random_state=3)
    sift.smart_sample(_frame(), ["f0", "f1"], "y", config, size=0.1, method="uniform")
    assert config.size == 0.5
    assert config.method == "residual"


def test_config_validates_on_construction_replace_and_use():
    with pytest.raises(ValueError, match="size"):
        sift.SmartSamplerConfig(size=1.5)
    with pytest.raises(ValueError, match="uniform_mix"):
        replace(sift.SmartSamplerConfig(), uniform_mix=2.0)
    config = sift.SmartSamplerConfig()
    config.task = "ranking"
    with pytest.raises(ValueError, match="task must be one of"):
        sift.smart_sample(_frame(), ["f0", "f1"], "y", config)
    with pytest.raises(TypeError, match="group_col"):
        sift.SmartSamplerConfig(group_col=3)
    with pytest.raises(ValueError, match="method must be one of"):
        sift.SmartSamplerConfig(method="smart")
    with pytest.raises(ValueError, match="shape only method='chronological'"):
        sift.SmartSamplerConfig(group_col="group")
    with pytest.raises(ValueError, match="strata_col applies only"):
        sift.SmartSamplerConfig(strata_col="season", method="uniform")
    with pytest.raises(ValueError, match="needs group_col or time_col"):
        sift.SmartSamplerConfig(method="chronological")
    with pytest.raises(ValueError, match="needs a pilot"):
        sift.SmartSamplerConfig(pilot=None)
    with pytest.raises(TypeError, match="anchor_fn"):
        sift.SmartSamplerConfig(anchor_fn="first")
    with pytest.raises(TypeError, match="verbose"):
        sift.SmartSamplerConfig(verbose="yes")
    assert sift.SmartSamplerConfig(verbose=np.bool_(True)).verbose
    with pytest.raises(TypeError, match="SmartSamplerConfig or None"):
        sift.smart_sample(_frame(), ["f0", "f1"], "y", {"size": 0.1})


def test_rejects_input_weight_column_instead_of_overwriting_it():
    source = _frame()
    source["sample_weight"] = 1.0
    with pytest.raises(ValueError, match="input already has one.*Rename or drop that column"):
        sift.smart_sample(source, ["f0", "f1"], "y")


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ({"method": "chronological", "group_col": "missing_group"}, "Missing columns.*missing_group"),
        ({"method": "chronological", "time_col": "missing_time"}, "Missing columns.*missing_time"),
        ({"unknown_option": 1}, "Unknown SmartSamplerConfig field"),
    ),
)
def test_rejects_unknown_columns_and_options(change, message):
    with pytest.raises((TypeError, ValueError), match=message):
        sift.smart_sample(_frame(), ["f0", "f1"], "y", **change)


def test_default_pilot_rejects_non_numeric_features_and_non_frames():
    with pytest.raises(TypeError, match="non-numeric: \\['label'\\]"):
        sift.smart_sample(_frame(), ["f0", "label"], "y")
    with pytest.raises(TypeError, match="requires a pandas DataFrame"):
        sift.smart_sample(_frame().to_numpy(), ["f0"], "y")
    with pytest.raises(ValueError, match="at least one column"):
        sift.smart_sample(_frame(), [], "y")


def test_anchor_fn_rows_are_always_kept():
    source = _frame()
    out = sift.smart_sample(source, ["f0", "f1"], "y", size=60, group_col="group",
                            time_col="time", anchor_fn=first_per_group, random_state=0)
    assert len(out) == 60
    first = source.sort_values(["group", "time"]).groupby("group").head(1).index
    assert set(first) <= set(out.index)

    def wrong_length(df, group_col, time_col):
        return np.ones(3, dtype=bool)

    with pytest.raises(ValueError, match="anchor_fn returned 3 rows"):
        sift.smart_sample(source, ["f0", "f1"], "y", anchor_fn=wrong_length)


def test_panel_config_is_a_chronological_preset():
    config = sift.panel_config("group", "time", size=0.3)
    assert (config.method, config.size, config.group_col, config.time_col, config.strata_col,
            config.anchor_fn) == ("chronological", 0.3, "group", "time", None, None)
    assert sift.panel_config("g").size == 0.2
    out = sift.smart_sample(_frame(), ["f0", "f1"], "y", sift.panel_config("group", "time", size=0.1))
    assert len(out) == 120
    assert out["group"].value_counts().to_dict() == {"b": 40, "a": 40, "c": 40}
    np.testing.assert_array_equal(out["sample_weight"].to_numpy(), np.ones(120))
    source = _frame()
    source["season"] = np.tile([2023, 2024], len(source) // 2)
    out = sift.smart_sample(source, ["f0"], "y", sift.panel_config("group", "time", 0.1, "season"))
    assert out["season"].value_counts().to_dict() == {2023: 60, 2024: 60}


def test_classification_target_uses_the_class_gradient():
    source = _frame()
    source["cls"] = np.where(source["y"] > 0, "pos", "neg")
    out = sift.smart_sample(source, ["f0", "f1"], "cls", size=0.25, random_state=0)
    assert len(out) == 300
    assert set(out["cls"]) == {"pos", "neg"}
