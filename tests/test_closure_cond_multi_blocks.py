"""Independent public checks for conditioned routing and block-unit auto-k."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sift import (
    AutoKConfig, build_cache, select_cached, select_cefsplus,
    select_cefsplus_binary, select_fdr, select_jmi, select_jmim, select_mrmr,
)


@pytest.mark.parametrize("selector", [select_mrmr, select_jmi, select_jmim])
def test_real_auto_router_rejects_conditioning_with_usable_advice(selector):
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(120, 7)), columns=[f"f{i}" for i in range(7)])
    y = X["f0"] + 0.7 * X["f1"] + 0.15 * rng.normal(size=len(X))
    options = {"task": "regression", "estimator": "gaussian", "include": ["f0"],
               "verbose": False}
    with pytest.raises(ValueError, match="k_method='auto'.*gaussian_cv.*k_method='elbow'"):
        selector(X, y, "auto", auto_k_config=AutoKConfig(k_method="auto"), **options)
    selected = selector(
        X, y, "auto", auto_k_config=AutoKConfig(k_method="elbow", min_k=1, max_k=3),
        **options,
    )
    assert selected[0] == "f0" and "f1" in selected


@pytest.mark.parametrize(
    "selector,extra", [
        (select_mrmr, {}),
        (select_mrmr, {"formula": "difference"}),
        (select_jmi, {}),
        (select_jmim, {}),
    ],
)
@pytest.mark.parametrize("dead", [7.0, np.nan])
def test_elbow_uses_block_units_after_dead_member_is_removed(selector, extra, dead):
    rng = np.random.default_rng(4)
    X = pd.DataFrame(rng.normal(size=(200, 10)), columns=[f"f{i}" for i in range(10)])
    X["dead"] = dead
    y = X["f0"] + 0.9 * X["f1"] + 0.7 * X["f2"] + 0.3 * rng.normal(size=len(X))
    options = {"task": "regression", "estimator": "gaussian", "k": "auto",
               "auto_k_config": AutoKConfig(k_method="elbow", max_k=4, min_k=0),
               "verbose": False, "return_result": True, "subsample": None, **extra}
    clean = selector(
        X, y, feature_blocks={"g": ["f0", "f1"], "h": ["f2", "f3"]},
        candidates=["f0", "f1", "f2", "f3"], **options,
    )
    dropped = selector(
        X, y, feature_blocks={"g": ["f0", "f1", "dead"], "h": ["f2", "f3"]},
        candidates=["f0", "f1", "dead", "f2", "f3"], **options,
    )
    clean_curve = clean.diagnostics_["auto_k_curve"]["curve"]
    dropped_curve = dropped.diagnostics_["auto_k_curve"]["curve"]
    assert clean_curve["k"].tolist() == dropped_curve["k"].tolist()
    assert max(clean_curve["k"]) <= 4
    np.testing.assert_allclose(
        clean_curve["criterion"], dropped_curve["criterion"],
        rtol=0.0, atol=1e-12,
    )
    assert clean.selector_metadata["selected_blocks"] == dropped.selector_metadata["selected_blocks"]


@pytest.mark.parametrize(
    "make_refs", [
        lambda: (name for name in ["f0"]),
        lambda: map(str, ["f0"]),
        lambda: filter(None, ["f0"]),
        lambda: iter(["f0"]),
    ],
)
def test_function_conditioning_consumes_one_shot_iterables_once(make_refs):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(80, 7)), columns=[f"f{i}" for i in range(7)])
    y = X["f0"] + 0.7 * X["f1"] + 0.2 * rng.normal(size=len(X))
    binary_y = (y > np.median(y)).astype(int)
    cache = build_cache(X, subsample=None)
    calls = (
        lambda value: select_mrmr(X, y, 1, task="regression", estimator="gaussian", exclude=value, verbose=False),
        lambda value: select_mrmr(X, y, 1, task="regression", estimator="classic", exclude=value, verbose=False),
        lambda value: select_jmi(X, y, 1, task="regression", estimator="gaussian", exclude=value, verbose=False),
        lambda value: select_jmim(X, y, 1, task="regression", estimator="gaussian", exclude=value, verbose=False),
        lambda value: select_cefsplus_binary(X, binary_y, 1, exclude=value, verbose=False),
        lambda value: select_cached(cache, y, 1, exclude=value),
        lambda value: select_fdr(
            X, y, q=0.5, offset=0, exclude=value,
            include_provenance="prespecified", random_state=0, verbose=False,
        ).selected_features,
        lambda value: select_cefsplus(
            X, y, "auto", auto_k_config=AutoKConfig(k_method="elbow", min_k=1, max_k=3),
            exclude=value, verbose=False,
        ),
    )
    for call in calls:
        expected = call(["f0"])
        assert expected and "f0" not in expected
        assert call(make_refs()) == expected
