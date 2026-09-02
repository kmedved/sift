"""Public compatibility contracts for ``SmartSamplerConfig`` and sampling."""

from dataclasses import fields
import warnings

import numpy as np
import pandas as pd
import pytest

import sift


EXPECTED_DEFAULTS = {
    "sample_frac": 0.10,
    "group_col": None,
    "time_col": None,
    "min_per_group": 2,
    "pilot_sample_size": 50_000,
    "leverage_batch_size": 200_000,
    "svd_sample_size": None,
    "weight_clip_quantile": 0.99,
    "residual_weight_cap": 0.4,
    "uniform_floor": 0.05,
    "anchor_fn": None,
    "anchor_max_share": 0.4,
    "random_state": 42,
    "verbose": True,
}


def _frame() -> pd.DataFrame:
    n = 18
    row = np.arange(n, dtype=np.float64)
    return pd.DataFrame(
        {
            "group": np.repeat(["b", "a", "c"], 6),
            "time": np.tile(np.arange(6), 3),
            "f0": np.sin(0.70 * row),
            "f1": np.cos(0.37 * row),
            "y": np.sin(0.70 * row) + 0.2 * np.cos(0.37 * row),
            "label": list("ABCDEFGHIJKLMNOPQR"),
        },
        index=np.arange(100, 118),
    )


def _bounded_options():
    return {
        "sample_frac": 0.5,
        "group_col": "group",
        "time_col": "time",
        "min_per_group": 2,
        "pilot_sample_size": 10,
        "leverage_batch_size": 8,
        "svd_sample_size": None,
        "weight_clip_quantile": 0.99,
        "residual_weight_cap": 0.0,
        "uniform_floor": 0.05,
        "anchor_fn": None,
        "anchor_max_share": 0.4,
        "random_state": 7,
        "verbose": False,
    }


def test_smart_sampler_config_exact_public_defaults():
    assert {field.name for field in fields(sift.SmartSamplerConfig)} == set(
        EXPECTED_DEFAULTS
    )
    config = sift.SmartSamplerConfig()
    assert {
        field.name: getattr(config, field.name)
        for field in fields(sift.SmartSamplerConfig)
    } == EXPECTED_DEFAULTS


def test_smart_sample_omitted_config_matches_explicit_options_and_golden_order():
    """Keyword construction and an explicit config are the same public call."""
    source = _frame()
    options = _bounded_options()

    with warnings.catch_warnings(record=True) as keyword_warnings:
        warnings.simplefilter("always")
        keyword_result = sift.smart_sample(
            source.copy(),
            ["f0", "f1"],
            "y",
            **options,
        )
    with warnings.catch_warnings(record=True) as config_warnings:
        warnings.simplefilter("always")
        config_result = sift.smart_sample(
            source.copy(),
            ["f0", "f1"],
            "y",
            sift.SmartSamplerConfig(**options),
        )

    assert [(item.category, str(item.message)) for item in keyword_warnings] == []
    assert [(item.category, str(item.message)) for item in config_warnings] == []
    assert type(keyword_result) is pd.DataFrame
    assert type(config_result) is pd.DataFrame
    pd.testing.assert_frame_equal(keyword_result, config_result)
    assert keyword_result.columns.tolist() == [
        "group",
        "time",
        "f0",
        "f1",
        "y",
        "label",
        "sample_weight",
    ]
    assert keyword_result["label"].tolist() == [
        "A",
        "B",
        "C",
        "D",
        "G",
        "H",
        "I",
        "M",
        "N",
        "O",
    ]
    assert keyword_result.groupby("group", sort=False).size().to_dict() == {
        "b": 4,
        "a": 3,
        "c": 3,
    }
    assert keyword_result.index.tolist() == list(range(10))
    assert keyword_result["sample_weight"].dtype == np.float32
    assert np.isfinite(keyword_result["sample_weight"]).all()
    assert float(keyword_result["sample_weight"].mean()) == pytest.approx(
        1.0,
        abs=2e-7,
    )
    assert "sample_weight" not in source.columns


def test_smart_sample_full_fraction_preserves_row_and_column_order():
    source = _frame()
    config = sift.SmartSamplerConfig(
        sample_frac=1.0,
        residual_weight_cap=0.0,
        random_state=7,
        verbose=False,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sift.smart_sample(source.copy(), ["f0", "f1"], "y", config)

    assert [(item.category, str(item.message)) for item in caught] == []
    assert result["label"].tolist() == source["label"].tolist()
    assert result.index.tolist() == list(range(len(source)))
    pd.testing.assert_frame_equal(
        result.drop(columns="sample_weight"),
        source.reset_index(drop=True),
    )
    np.testing.assert_array_equal(
        result["sample_weight"].to_numpy(),
        np.ones(len(source), dtype=np.float32),
    )


def test_smart_sample_small_pilot_warning_contract():
    config = sift.SmartSamplerConfig(
        sample_frac=0.5,
        pilot_sample_size=50,
        residual_weight_cap=0.4,
        random_state=7,
        verbose=True,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sift.smart_sample(_frame(), ["f0", "f1"], "y", config)

    assert type(result) is pd.DataFrame
    assert [(item.category, str(item.message)) for item in caught] == [
        (
            RuntimeWarning,
            "Dataset too small for pilot model (n=18); using geometry only.",
        )
    ]


def test_smart_sample_rejects_input_weight_column_instead_of_overwriting_it():
    source = _frame()
    source["sample_weight"] = np.linspace(0.5, 1.5, len(source))
    with pytest.raises(
        ValueError,
        match="input already has one.*Rename or drop that column",
    ):
        sift.smart_sample(
            source,
            ["f0", "f1"],
            "y",
            residual_weight_cap=0.0,
            verbose=False,
        )


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ({"group_col": "missing_group"}, "Missing columns.*missing_group"),
        ({"time_col": "missing_time"}, "Missing columns.*missing_time"),
        ({"unknown_option": 1}, "Unknown SmartSamplerConfig override"),
    ),
)
def test_smart_sample_rejects_unsupported_column_or_override_contracts(
    change,
    message,
):
    config = sift.SmartSamplerConfig(
        sample_frac=0.5,
        residual_weight_cap=0.0,
        verbose=False,
    )
    with pytest.raises((TypeError, ValueError), match=message):
        sift.smart_sample(
            _frame(),
            ["f0", "f1"],
            "y",
            config=config,
            **change,
        )
