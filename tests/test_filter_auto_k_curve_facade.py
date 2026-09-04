"""Facade aliases for the extracted auto-k curve payload cluster."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sift.selection import filter_auto_k, filter_auto_k_curve
from sift.selection.filter_auto_k import build_auto_k_curve_payload


def test_curve_cluster_names_are_facade_aliases() -> None:
    for name in (
        "AUTO_K_CURVE_KEY",
        "AUTO_K_CURVE_COLUMNS",
        "_AUTO_K_CURVE_CRITERIA",
        "_AUTO_K_CURVE_UNAVAILABLE",
        "_auto_k_curve_unavailable",
        "build_auto_k_curve_payload",
    ):
        assert getattr(filter_auto_k, name) is getattr(filter_auto_k_curve, name)


def test_normalized_payload_keeps_column_order_and_selected_k() -> None:
    diagnostics = pd.DataFrame(
        {
            "k": [2, 1, 3],
            "score": [0.2, 0.1, 0.3],
            "score_se": [0.02, 0.01, 0.03],
        }
    )
    payload = build_auto_k_curve_payload(
        k_method="evaluate",
        diagnostics=diagnostics,
        summary={"method": "evaluate", "selected_k": 2},
    )
    assert payload["available"] is True
    assert payload["route"] == "evaluate"
    assert payload["criterion"] == "score"
    assert payload["criterion_direction"] == "higher_is_better"
    assert tuple(payload["curve"].columns) == filter_auto_k.AUTO_K_CURVE_COLUMNS
    assert list(payload["curve"]["k"]) == [1, 2, 3]
    assert list(payload["curve"]["selected"]) == [False, True, False]
    assert payload["curve"]["k"].dtype == np.int64
    assert payload["curve"]["criterion"].dtype == np.float64
    assert payload["curve"]["criterion_se"].dtype == np.float64
    assert payload["curve"]["selected"].dtype == bool


def test_unavailable_route_payload_is_stable() -> None:
    payload = build_auto_k_curve_payload(
        k_method="knockoff_path",
        diagnostics=None,
        summary={"method": "knockoff_path"},
    )
    assert payload["available"] is False
    assert payload["route"] == "knockoff_path"
    assert payload["curve"] is None
    assert payload["unavailable_reason"] == (
        "knockoff_path diagnostics carry one row per candidate feature and knockoff "
        "draw rather than one row per k, so the route has no k-indexed criterion curve"
    )
