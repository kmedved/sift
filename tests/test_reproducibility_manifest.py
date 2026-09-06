"""Public reproducibility-manifest contracts."""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold

from sift import (
    AutoKConfig,
    CEFSPlusSelector,
    FilterSelectionResult,
    as_result,
    build_cache,
    compare,
    select_cached,
    select_cefsplus,
    select_cefsplus_binary,
    select_fdr,
)
from sift.selection.view import _columns_hash


def _regression_frame(n=90, p=4, seed=4):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"x{i}" for i in range(p)])
    y = 2.2 * X["x0"] + 1.1 * X["x1"] + 0.1 * rng.normal(size=n)
    return X, y


def _assert_export_environment(payload):
    env = payload["environment"]
    assert env["captured_at"] == "export"
    assert env["sift"] == "0.9.1.dev0"
    assert isinstance(env["numpy"], str) and env["numpy"]
    assert isinstance(env["pandas"], str) and env["pandas"]
    assert isinstance(env["scikit-learn"], str) and env["scikit-learn"]
    assert env["scipy"] is None or isinstance(env["scipy"], str)
    assert env["numba"] is None or isinstance(env["numba"], str)
    assert isinstance(env["blas"], list)
    assert env["git_commit_source"] == "sift_package"
    commit = env["git_commit"]
    assert commit is None or (
        isinstance(commit, str)
        and len(commit) == 40
        and all(char in "0123456789abcdef" for char in commit)
    )


def test_selection_manifest_is_json_safe_and_does_not_retain_x():
    X, y = _regression_frame()
    result = select_cefsplus(X, y, k=2, verbose=False, return_result=True)
    view = as_result(result, input_features=list(X.columns))
    payload = view.reproducibility_()
    json.dumps(payload, allow_nan=False)
    assert payload["schema_version"] == "1"
    assert payload["kind"] == "selection"
    assert payload is not view.to_dict()
    assert payload["input"]["data_hash"] is None
    assert payload["input"]["data_hash_source"] is None
    assert payload["folds"] == []
    _assert_export_environment(payload)
    assert view.reproducibility_(X=X, hash_data=False)["input"]["data_hash"] is None
    assert not hasattr(view, "_X")
    assert "_X" not in view.__slots__
    pickled = pickle.loads(pickle.dumps(result))
    assert pickled.selected_features == result.selected_features


def test_typed_column_hash_is_order_and_type_sensitive():
    rng = np.random.default_rng(5)
    values = rng.normal(size=(80, 3))
    X_int = pd.DataFrame(values, columns=[2, 0, 1])
    X_str = pd.DataFrame(values, columns=["2", "0", "1"])
    y = 3.0 * X_int[2] + 0.2 * rng.normal(size=80)
    int_view = as_result(
        select_cefsplus(X_int, y, k=1, verbose=False, return_result=True),
        input_features=list(X_int.columns),
    )
    str_view = as_result(
        select_cefsplus(X_str, y, k=1, verbose=False, return_result=True),
        input_features=list(X_str.columns),
    )
    int_hash = int_view.reproducibility_()["input"]["columns_hash"]
    str_hash = str_view.reproducibility_()["input"]["columns_hash"]
    assert int_hash == _columns_hash([2, 0, 1])
    assert str_hash == _columns_hash(["2", "0", "1"])
    assert int_hash != str_hash
    reversed_hash = as_result(
        select_cefsplus(
            X_int[[1, 0, 2]], y, k=1, verbose=False, return_result=True
        ),
        input_features=[1, 0, 2],
    ).reproducibility_()["input"]["columns_hash"]
    assert reversed_hash != int_hash


def test_data_hash_is_opt_in_and_requires_x():
    X, y = _regression_frame()
    result = select_cefsplus(X, y, k=1, verbose=False, return_result=True)
    with pytest.raises(ValueError, match="hash_data=True requires X"):
        result.reproducibility_(hash_data=True)
    hashed = result.reproducibility_(X=X, hash_data=True, input_features=list(X.columns))
    assert hashed["input"]["data_hash"]
    assert hashed["input"]["data_hash_source"] == "caller"
    other = X.copy()
    other.iloc[0, 0] += 1.0
    other_hash = result.reproducibility_(
        X=other, hash_data=True, input_features=list(X.columns)
    )["input"]["data_hash"]
    assert other_hash != hashed["input"]["data_hash"]


def test_effective_k_differs_from_configured_auto_k():
    X, y = _regression_frame(n=120, p=6, seed=8)
    result = select_cefsplus(X, y, k="auto", verbose=False, return_result=True)
    payload = result.reproducibility_(input_features=list(X.columns))
    configured = payload["configuration"]["configured"]
    effective = payload["configuration"]["effective"]
    assert payload["configuration"]["captured_at"] == "selection"
    assert configured["k_requested"] == "auto"
    assert configured["auto_k"] is True
    assert isinstance(effective["k"], int)
    assert effective["k"] != "auto"


def test_legacy_result_keeps_unknown_provenance():
    result = FilterSelectionResult(
        selected_features=["f0", "f3"],
        selected_indices=[0, 3],
        selector_metadata={"selector": "mrmr", "k": 2, "n_features": 5},
    )
    view = as_result(result)
    payload = view.reproducibility_()
    assert view.metadata["input_kind"] == "unknown"
    assert payload["input"]["n_rows"] is None
    assert payload["input"]["n_rows_source"] == "unknown"
    assert payload["input"]["cache"]["available"] is False
    assert payload["input"]["cache"]["n_rows_original"] is None
    assert payload["input"]["cache"]["feature_names_are_synthetic"] is None
    assert payload["input"]["data_hash"] is None
    assert payload["configuration"]["seeds"]["available"] is False
    assert payload["configuration"]["seeds"]["random_state"] is None
    assert payload["environment"]["captured_at"] == "export"
    named = as_result(result, input_features=["f0", "f1", "f2", "f3", "f4"])
    named_payload = named.reproducibility_()
    assert named.metadata["input_kind"] == "unknown"
    assert named_payload["input"]["columns_hash"] == _columns_hash(
        ["f0", "f1", "f2", "f3", "f4"]
    )
    assert named_payload["input"]["columns_hash_source"] == "result"


def test_cache_provenance_is_retained_when_present():
    X, y = _regression_frame(n=80, p=5, seed=3)
    cache = build_cache(X)
    result = select_cached(cache, y, k=2, method="cefsplus", return_result=True)
    payload = result.reproducibility_()
    cache_block = payload["input"]["cache"]
    assert cache_block["available"] is True
    assert cache_block["n_rows_original"] == 80
    assert cache_block["feature_names_are_synthetic"] is False
    assert payload["input"]["n_rows"] == 80
    assert payload["input"]["n_rows_source"] == "cache"
    assert payload["configuration"]["effective"].get("cache_backed") is True


def test_compare_manifest_reuses_real_fold_bookkeeping():
    X, y = _regression_frame(n=90, p=4, seed=1)
    cv = KFold(n_splits=3, shuffle=True, random_state=0)
    result = compare(
        {"cefs": lambda: CEFSPlusSelector(k=1, verbose=False)},
        X,
        np.asarray(y),
        estimator=Ridge(),
        cv=cv,
        random_state=0,
    )
    payload = result.reproducibility_()
    json.dumps(payload, allow_nan=False)
    assert payload["kind"] == "compare"
    assert payload["folds"] == [dict(item) for item in result.fold_bookkeeping]
    assert payload["folds"][0]["train_index_sha256"] == result.folds.iloc[0][
        "train_index_sha256"
    ]
    assert payload["input"]["n_rows"] == 90
    assert payload["input"]["n_features"] == 4
    assert payload["input"]["columns_hash"] == _columns_hash(list(X.columns))
    assert payload["input"]["data_hash"] is None
    assert payload["configuration"]["captured_at"] == "compare"
    assert payload["configuration"]["seeds"]["available"] is True
    assert payload["configuration"]["seeds"]["compare_random_state"] == 0
    assert payload["configuration"]["seeds"]["split_random_state"] == 0
    assert payload["configuration"]["seeds"]["compare_random_state_used_for_split"] is False
    _assert_export_environment(payload)
    hashed = result.reproducibility_(X=X, hash_data=True)
    assert hashed["input"]["data_hash"]
    assert hashed["folds"] == payload["folds"]


def test_compare_manifest_records_actual_selector_estimator_and_split():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(90, 4)), columns=list("abcd"))
    y = 5 * X.c + 0.7 * X.a + 0.01 * rng.normal(size=len(X))
    cv = KFold(n_splits=3, shuffle=True, random_state=19)
    left = compare(
        {"s": lambda: CEFSPlusSelector(k=1, random_state=17, verbose=False)},
        X,
        y,
        estimator=Ridge(alpha=0.1),
        cv=cv,
        random_state=999,
    )
    right = compare(
        {"s": lambda: CEFSPlusSelector(k=3, random_state=29, verbose=False)},
        X,
        y,
        estimator=Ridge(alpha=50),
        cv=cv,
        random_state=999,
    )
    a = left.reproducibility_()
    b = right.reproducibility_()
    json.dumps(a, allow_nan=False)
    assert a["configuration"]["effective"]["selectors"]["s"]["params"]["k"] == 1
    assert a["configuration"]["effective"]["selectors"]["s"]["params"]["random_state"] == 17
    assert b["configuration"]["effective"]["selectors"]["s"]["params"]["k"] == 3
    assert b["configuration"]["effective"]["selectors"]["s"]["params"]["random_state"] == 29
    assert a["configuration"]["effective"]["estimator"]["params"]["alpha"] == 0.1
    assert b["configuration"]["effective"]["estimator"]["params"]["alpha"] == 50
    split = a["configuration"]["effective"]["split"]
    assert split["source"] == "caller"
    assert split["params"]["n_splits"] == 3
    assert split["params"]["shuffle"] is True
    assert split["params"]["random_state"] == 19
    seeds = a["configuration"]["seeds"]
    assert seeds["compare_random_state"] == 999
    assert seeds["split_random_state"] == 19
    assert seeds["compare_random_state_used_for_split"] is False
    assert a["configuration"]["effective"] != b["configuration"]["effective"]


def test_new_filter_and_knockoff_runs_retain_known_settings():
    X, y = _regression_frame(n=90, p=4, seed=6)
    cefs = select_cefsplus(
        X, y, k=2, random_state=17, subsample=50, verbose=False, return_result=True
    )
    cefs_m = cefs.reproducibility_(input_features=list(X.columns))
    assert cefs_m["configuration"]["seeds"]["available"] is True
    assert cefs_m["configuration"]["seeds"]["random_state"] == 17
    assert cefs_m["configuration"]["configured"]["subsample"] == 50
    assert cefs_m["configuration"]["configured"]["random_state"] == 17
    assert cefs_m["input"]["n_rows"] == 90
    assert cefs_m["input"]["n_rows_used"] == 50

    auto = select_cefsplus(
        X,
        y,
        k="auto",
        auto_k_config=AutoKConfig(
            k_method="penalized_objective", objective_penalty="ebic", min_k=1
        ),
        verbose=False,
        return_result=True,
    )
    auto_m = auto.reproducibility_(input_features=list(X.columns))
    assert auto_m["configuration"]["configured"]["k_method"] == "penalized_objective"
    assert auto_m["configuration"]["configured"]["objective_penalty"] == "ebic"
    assert auto_m["configuration"]["effective"]["objective_penalty"] == "ebic"

    rng = np.random.default_rng(0)
    Xb = pd.DataFrame(rng.normal(size=(80, 4)), columns=list("abcd"))
    yb = (Xb.a + 0.3 * rng.normal(size=len(Xb)) > 0).astype(int)
    binary = select_cefsplus_binary(
        Xb, yb, k=1, subsample=50, verbose=False, return_result=True
    )
    binary_m = binary.reproducibility_(input_features=list(Xb.columns))
    assert binary_m["configuration"]["configured"]["subsample"] == 50

    Xk = pd.DataFrame(rng.normal(size=(120, 6)), columns=[f"g{i}" for i in range(6)])
    yk = 3.0 * Xk["g0"] + 0.2 * rng.normal(size=len(Xk))
    rel = select_fdr(Xk, yk, q=0.3, statistic="relevance", random_state=7, verbose=False)
    ridge = select_fdr(Xk, yk, q=0.3, statistic="ridge", random_state=7, verbose=False)
    assert rel.reproducibility_()["configuration"]["configured"]["statistic"] == "relevance"
    assert ridge.reproducibility_()["configuration"]["configured"]["statistic"] == "ridge"
    assert rel.reproducibility_()["configuration"]["effective"]["statistic"] == "relevance"


def test_caller_x_must_match_known_rows_and_typed_columns():
    X, y = _regression_frame(n=90, p=4, seed=2)
    result = select_cefsplus(X, y, k=1, verbose=False, return_result=True)
    with pytest.raises(ValueError, match="2D"):
        result.reproducibility_(X=X["x0"], hash_data=True, input_features=list(X.columns))
    with pytest.raises(ValueError, match="rows"):
        result.reproducibility_(X=X.iloc[:7], hash_data=True, input_features=list(X.columns))
    with pytest.raises(ValueError, match="column identity"):
        result.reproducibility_(
            X=X.rename(columns={"x0": "z0"}),
            hash_data=True,
            input_features=list(X.columns),
        )
    with pytest.raises(ValueError, match="column identity"):
        result.reproducibility_(
            X=X[list(reversed(X.columns))],
            hash_data=True,
            input_features=list(X.columns),
        )
    perturbed = X.copy()
    perturbed.iloc[0, 0] += 1.0
    hashed = result.reproducibility_(
        X=perturbed, hash_data=True, input_features=list(X.columns)
    )
    assert hashed["input"]["data_hash_source"] == "caller"


def test_knockoff_original_and_used_rows_are_distinct():
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(400, 6)), columns=[f"g{i}" for i in range(6)])
    y = 2.5 * X["g0"] + 0.4 * rng.normal(size=len(X))
    result = select_fdr(X, y, q=0.3, random_state=7, subsample=200, verbose=False)
    payload = result.reproducibility_()
    assert payload["input"]["n_rows"] == 400
    assert payload["input"]["n_rows_source"] == "result"
    assert payload["input"]["n_rows_used"] == 200
    assert payload["input"]["n_rows_used_source"] == "result"
    accepted = result.reproducibility_(X=X, hash_data=True)
    assert accepted["input"]["n_rows"] == 400
    assert accepted["input"]["n_rows_used"] == 200
    with pytest.raises(ValueError, match="rows"):
        result.reproducibility_(X=X.iloc[:7], hash_data=True)


def test_git_commit_is_bound_to_sift_package(tmp_path, monkeypatch):
    X, y = _regression_frame()
    result = select_cefsplus(X, y, k=1, verbose=False, return_result=True)
    inside = result.reproducibility_()["environment"]
    monkeypatch.chdir(tmp_path)
    outside = result.reproducibility_()["environment"]
    assert inside["git_commit_source"] == "sift_package"
    assert outside["git_commit_source"] == "sift_package"
    assert inside["git_commit"] == outside["git_commit"]


def _probe_frame():
    rng = np.random.default_rng(37)
    X = pd.DataFrame(rng.normal(size=(90, 4)), columns=list("abcd"))
    y = 3 * X.a + X.c + 0.1 * rng.normal(size=len(X))
    return X, y


def test_prebuilt_cache_rows_are_measured_and_unused_defaults_omitted():
    X, y = _probe_frame()
    weights = np.zeros(len(X), dtype=np.float64)
    weights[:60] = 1.0
    cache = build_cache(X, sample_weight=weights, subsample=40, random_state=17)
    assert len(cache.row_idx) == 40
    result = select_cefsplus(X, y, k=1, cache=cache, verbose=False, return_result=True)
    payload = result.reproducibility_()
    assert payload["input"]["n_rows"] == 90
    assert payload["input"]["n_rows_used"] == 40
    assert payload["input"]["cache"]["available"] is True
    assert payload["input"]["cache"]["n_rows_original"] == 90
    assert payload["configuration"]["configured"].get("random_state") is None
    assert payload["configuration"]["configured"].get("subsample") is None
    assert payload["configuration"]["seeds"]["random_state"] is None

    fresh = select_cefsplus(
        X, y, k=1, sample_weight=weights, subsample=None, verbose=False, return_result=True
    )
    fresh_m = fresh.reproducibility_(input_features=list(X.columns))
    assert fresh_m["input"]["n_rows"] == 90
    assert fresh_m["input"]["n_rows_used"] == 60


def test_typed_feature_block_keys_survive_manifest_json():
    X, y = _probe_frame()
    result = compare(
        {
            "s": lambda: CEFSPlusSelector(
                k=1, feature_blocks={1: ["a"], "1": ["b"]}, verbose=False
            )
        },
        X,
        y,
        estimator=Ridge(),
        cv=KFold(n_splits=3, shuffle=True, random_state=0),
    )
    payload = result.reproducibility_()
    encoded = json.dumps(payload, allow_nan=False)
    blocks = json.loads(encoded)["configuration"]["effective"]["selectors"]["s"]["params"][
        "feature_blocks"
    ]
    assert blocks["__sift_mapping__"] == "typed_key_entries"
    keys = {(entry["key"]["type"], str(entry["key"]["value"])) for entry in blocks["entries"]}
    assert ("builtins.int", "1") in keys
    assert ("builtins.str", "1") in keys


def test_autok_custom_penalty_and_separate_rng_are_retained():
    X, y = _probe_frame()
    left = select_cefsplus(
        X,
        y,
        k="auto",
        random_state=17,
        auto_k_config=AutoKConfig(
            k_method="penalized_objective",
            objective_penalty="custom",
            objective_penalty_weight=0.001,
            min_k=1,
        ),
        verbose=False,
        return_result=True,
    )
    right = select_cefsplus(
        X,
        y,
        k="auto",
        random_state=17,
        auto_k_config=AutoKConfig(
            k_method="penalized_objective",
            objective_penalty="custom",
            objective_penalty_weight=100,
            min_k=1,
        ),
        verbose=False,
        return_result=True,
    )
    a = left.reproducibility_(input_features=list(X.columns))
    b = right.reproducibility_(input_features=list(X.columns))
    a_cfg = a["configuration"]["configured"]["auto_k_config"]["params"]
    b_cfg = b["configuration"]["configured"]["auto_k_config"]["params"]
    assert a_cfg["objective_penalty"] == "custom"
    assert a_cfg["objective_penalty_weight"] == 0.001
    assert b_cfg["objective_penalty_weight"] == 100
    assert a["configuration"] != b["configuration"]
    assert a["configuration"]["seeds"]["random_state"] == 17
    assert a_cfg["random_state"] == 42
    assert a["configuration"]["seeds"]["auto_k_random_state"] == 42


def test_compare_records_dummy_and_prefix_factory_models():
    X, y = _probe_frame()

    class _Empty(TransformerMixin, BaseEstimator):
        def fit(self, frame, target, **kwargs):
            self.n_features_in_ = int(frame.shape[1])
            if hasattr(frame, "columns"):
                self.feature_names_in_ = np.asarray(list(frame.columns), dtype=object)
            self.support_ = np.zeros(self.n_features_in_, dtype=bool)
            return self

        def transform(self, frame):
            return np.zeros((int(frame.shape[0]), 0), dtype=np.float64)

        def get_support(self, indices=False):
            if indices:
                return np.empty(0, dtype=np.int64)
            return self.support_

    empty = compare(
        {"s": lambda: _Empty()},
        X,
        y,
        estimator=Ridge(alpha=71),
        cv=KFold(n_splits=3, shuffle=True, random_state=0),
    )
    empty_m = empty.reproducibility_()
    assert empty.scores["empty"].all()
    configured = empty_m["configuration"]["configured"]["estimator"]
    actual = empty_m["configuration"]["effective"]["estimator"]
    assert configured["params"]["alpha"] == 71
    assert "DummyRegressor" in actual["type"]
    assert actual["params"]["strategy"] == "mean"

    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return Ridge(alpha=float(calls["n"]))

    path = compare(
        {"s": lambda: CEFSPlusSelector(k=1, verbose=False)},
        X,
        y,
        estimator_factory=factory,
        cv=KFold(n_splits=3, shuffle=True, random_state=0),
        mode="in_sample_path",
    )
    path_m = path.reproducibility_()
    actual = path_m["configuration"]["effective"]["estimator"]
    assert actual["status"] == "varies"
    alphas = [item["model"]["params"]["alpha"] for item in actual["by_fit"]]
    assert alphas == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    scopes = [(item["scope"], item.get("prefix_k"), item["split_id"]) for item in actual["by_fit"]]
    assert scopes[0][0] == "main"
    assert "prefix" in {item[0] for item in scopes}


def test_resolved_groupkfold_does_not_claim_compare_seed():
    X, y = _probe_frame()
    groups = np.repeat(np.arange(6), 15)
    result = compare(
        {"s": lambda: CEFSPlusSelector(k=1, verbose=False)},
        X,
        y,
        estimator=Ridge(),
        groups=groups,
        random_state=999,
    )
    payload = result.reproducibility_()
    split = payload["configuration"]["effective"]["split"]
    assert "GroupKFold" in split["type"]
    assert split["params"]["n_splits"] == 5
    seeds = payload["configuration"]["seeds"]
    assert seeds["compare_random_state"] == 999
    assert seeds["split_random_state"] is None
    assert seeds["compare_random_state_used_for_split"] is False
    assert split["uses_compare_random_state"] is False


def test_function_configured_options_preserve_non_data_settings():
    X, y = _probe_frame()
    none_prune = select_cefsplus(
        X, y, k=1, corr_prune=None, verbose=False, return_result=True
    )
    half_prune = select_cefsplus(
        X, y, k=1, corr_prune=0.5, verbose=False, return_result=True
    )
    none_m = none_prune.reproducibility_(input_features=list(X.columns))
    half_m = half_prune.reproducibility_(input_features=list(X.columns))
    assert none_m["configuration"]["configured"]["corr_prune"] is None
    assert half_m["configuration"]["configured"]["corr_prune"] == 0.5
    assert none_m["configuration"]["configured"] != half_m["configuration"]["configured"]
    assert none_m["configuration"]["effective"].get("top_m") != none_m["configuration"][
        "configured"
    ].get("top_m")

    included = select_cefsplus(
        X, y, k=1, include=["b"], verbose=False, return_result=True
    )
    inc_m = included.reproducibility_(input_features=list(X.columns))
    assert list(inc_m["configuration"]["configured"]["include"]) == ["b"]
    assert included.selected_features[0] == "b"

    excluded = select_cefsplus(
        X, y, k=1, exclude=["a"], verbose=False, return_result=True
    )
    exc_m = excluded.reproducibility_(input_features=list(X.columns))
    assert list(exc_m["configuration"]["configured"]["exclude"]) == ["a"]
    assert "a" not in excluded.selected_features

    blocked = select_cefsplus(
        X,
        y,
        k=1,
        feature_blocks={"pair": ["a", "b"]},
        verbose=False,
        return_result=True,
    )
    blk_m = blocked.reproducibility_(input_features=list(X.columns))
    blocks = blk_m["configuration"]["configured"]["feature_blocks"]
    assert blocks is not True
    assert list(blocks["pair"]) == ["a", "b"]
    assert set(blocked.selected_features) == {"a", "b"}


def test_in_sample_path_prebuilt_cache_is_provenance_only(monkeypatch):
    import dataclasses

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(12, 2)), columns=list("ab"))
    y = X.a + 0.1 * rng.normal(size=len(X))
    cache = build_cache(X)
    original_asdict = dataclasses.asdict

    def guarded_asdict(obj, *args, **kwargs):
        if type(obj).__name__ == "FeatureCache":
            raise AssertionError("FeatureCache must not be copied via asdict")
        return original_asdict(obj, *args, **kwargs)

    monkeypatch.setattr(dataclasses, "asdict", guarded_asdict)
    result = compare(
        {"s": lambda: CEFSPlusSelector(k=1, cache=cache, verbose=False)},
        X,
        y,
        estimator=Ridge(),
        cv=KFold(n_splits=2, shuffle=True, random_state=0),
        mode="in_sample_path",
    )
    payload = result.reproducibility_()
    encoded = json.dumps(payload, allow_nan=False)
    cache_desc = payload["configuration"]["configured"]["selectors"]["s"]["params"]["cache"]
    assert cache_desc["status"] == "cache_provenance"
    assert "Z" not in cache_desc
    assert "Rxx" not in cache_desc
    assert "sample_weight" not in cache_desc
    assert "row_idx" not in cache_desc
    assert cache_desc["n_rows_original"] == 12
    assert cache_desc["n_rows_cached"] == 12
    assert "feature_names_are_synthetic" in cache_desc
    loaded = json.loads(encoded)
    dumped_cache = loaded["configuration"]["configured"]["selectors"]["s"]["params"]["cache"]
    assert dumped_cache["status"] == "cache_provenance"
    assert "Z" not in dumped_cache
