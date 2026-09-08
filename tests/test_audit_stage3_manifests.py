"""Small regressions for the bounded Stage 3 manifest fixes."""

import datetime
import json

import numpy as np
import pandas as pd
import pytest

from sift import FilterSelectionResult
from sift import build_classic_cache
from sift.selection.reproducibility import (
    _data_hash,
    _sanitize_param,
    _seed_block,
    snapshot_selector_kwargs,
)
from sift.selection.view import _columns_hash, _label_token


def _result():
    return FilterSelectionResult(
        selected_features=["a"],
        selected_indices=[0],
        selector_metadata={
            "selector": "test",
            "k": 1,
            "n_features": 2,
            "n_rows_original": 3,
        },
    )


def test_long_options_use_typed_ordered_digest_without_contents():
    left = _sanitize_param([f"x{i}" for i in range(33)])
    right = _sanitize_param([f"y{i}" for i in range(33)])
    assert left["status"] == right["status"] == "sequence_digest"
    assert left["length"] == right["length"] == 33
    assert left["sha256"] != right["sha256"]
    assert "x0" not in json.dumps(left)


def test_canonical_tokens_reject_arbitrary_objects_and_order_sets():
    assert _label_token(frozenset(["b", "a"])) == _label_token(
        frozenset(["a", "b"])
    )
    assert _label_token(b"ab")["value"]["encoding"] == "base64"
    with pytest.raises(TypeError, match="deterministic identity token"):
        _label_token(object())
    with pytest.raises(TypeError, match="deterministic identity token"):
        _columns_hash([object()])
    token = _label_token(datetime.timedelta(days=1, seconds=2, microseconds=3))
    assert token["value"] == {
        "days": 1,
        "seconds": 2,
        "microseconds": 3,
    }


def test_classic_cache_snapshot_omits_training_arrays():
    cache = build_classic_cache(np.arange(12.0).reshape(4, 3), subsample=None)
    snapshot = snapshot_selector_kwargs({"cache": cache})
    assert snapshot["cache"]["status"] == "cache_provenance"
    assert "X" not in json.dumps(snapshot)
    assert "row_idx" not in json.dumps(snapshot)


def test_opaque_configured_seed_is_not_claimed_reproducible():
    seeds = _seed_block({
        "configured_options": {
            "random_state": {
                "status": "opaque",
                "type": "numpy.random.Generator",
            }
        }
    })
    assert seeds["available"] is False
    assert seeds["random_state"]["status"] == "opaque"


def test_object_hash_stream_preserves_supported_string_hash():
    n, p = 2000, 10
    frame = pd.DataFrame(
        {f"c{j}": [f"level-{(i + j) % 97}" for i in range(n)] for j in range(p)}
    )
    assert _data_hash(frame) == (
        "47825e111e5b782c5bd9444eafcd9d43db55dddf7001b65f859748aef511d489"
    )


def test_context_hashes_are_opt_in_and_validate_rows():
    result = _result()
    X = np.zeros((3, 2))
    x_only = result.reproducibility_(X=X, hash_data=True)["input"]
    assert x_only["context_hashes_complete"] is False
    assert x_only["context_selection_time_verified"] is False
    unhashed = result.reproducibility_(
        X=X,
        y=np.arange(3),
        sample_weight=np.ones(3),
        groups=np.array(["a", "b", "a"]),
        time=np.arange(3),
        hash_data=False,
    )["input"]
    assert unhashed["y_hash"] is None
    assert unhashed["y_hash_source"] == "caller_unhashed"
    assert unhashed["context_hashes_complete"] is False

    hashed = result.reproducibility_(
        X=X,
        y=np.arange(3),
        sample_weight=np.ones(3),
        groups=np.array(["a", "b", "a"]),
        time=np.arange(3),
        hash_data=True,
    )["input"]
    assert hashed["y_hash"]
    assert hashed["sample_weight_hash"]
    assert hashed["groups_hash"]
    assert hashed["time_hash"]
    assert hashed["context_hashes_complete"] is True
    with pytest.raises(ValueError, match="rows"):
        result.reproducibility_(X=X, y=np.arange(2), hash_data=False)


def test_effective_manifest_keeps_multi_target_identity_and_legacy_metadata_safe():
    one = _result()
    two = _result()
    one.selector_metadata["n_targets"] = 1
    two.selector_metadata.update({"n_targets": 2, "ic_df_rule": "q_k"})
    first = one.reproducibility_()["configuration"]["effective"]
    second = two.reproducibility_()["configuration"]["effective"]
    assert first["n_targets"] == 1
    assert second["n_targets"] == 2
    assert second["ic_df_rule"] == "q_k"

    legacy = FilterSelectionResult(
        selected_features=["a"],
        selected_indices=[0],
        selector_metadata={
            "selector": "legacy",
            "k": 1,
            "n_features": 2,
            "estimator": object(),
        },
    )
    json.dumps(legacy.reproducibility_(), allow_nan=False)


def test_public_legacy_result_types_expose_manifest_delegate():
    from sift.boruta import BorutaResult
    from sift.catboost_common import CatBoostSelectionResult
    from sift.importance import ImportanceResult
    from sift.selection.path_eval import FeaturePathEvaluationResult

    for result_type in (
        BorutaResult,
        CatBoostSelectionResult,
        ImportanceResult,
        FeaturePathEvaluationResult,
    ):
        assert callable(getattr(result_type, "reproducibility_", None))
