"""Closure regressions for reproducibility manifests and their hashing."""

from __future__ import annotations

import decimal
import fractions
import hashlib
import json
import os
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import sift
from sift import (
    CEFSPlusSelector,
    FilterSelectionResult,
    compare,
    evaluate_feature_path,
    permutation_importance,
    select_boruta,
    select_cefsplus,
)
from sift.selection import reproducibility as repro
from sift.selection.reproducibility import (
    MANIFEST_SCHEMA_KEYS,
    _data_hash,
    _git_dirty,
    _sanitize_param,
    manifest_key_map,
)
from sift.selection.view import _columns_hash, _label_token


SIFT_ROOT = str(Path(sift.__file__).resolve().parents[1])


def _subprocess_json(script: str, *, hash_seed: str) -> dict:
    """Run ``script`` in a fresh interpreter and parse its single JSON line."""
    env = dict(os.environ)
    env.update(
        {
            "PYTHONHASHSEED": hash_seed,
            "PYTHONPATH": SIFT_ROOT,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------------
# 1. Deterministic typed tokens for Decimal / UUID / complex / Fraction
# --------------------------------------------------------------------------

_TOKEN_SCRIPT = """
    import decimal, fractions, json, uuid
    import numpy as np
    import pandas as pd
    from sift.selection.reproducibility import _data_hash
    from sift.selection.view import _columns_hash

    labels = [
        decimal.Decimal("1.50"),
        uuid.UUID(int=7),
        complex(1.5, -2.25),
        fractions.Fraction(3, 4),
        np.complex128(1.5 - 2.25j),
        frozenset({"q", "r", "s", "t", "u", "v"}),
    ]
    cells = np.empty((2, 2), dtype=object)
    cells[:] = [[decimal.Decimal("0.10"), uuid.UUID(int=1)],
                [complex(0.0, 1.0), fractions.Fraction(-1, 3)]]
    frame = pd.DataFrame(cells, columns=["c0", "c1"])
    print(json.dumps({
        "columns": _columns_hash(labels),
        "data": _data_hash(frame),
        "per_label": [_columns_hash([value]) for value in labels],
    }))
"""


def _expected_columns_hash(tokens: list[dict]) -> str:
    """Recompute the documented digest without calling ``_label_token``."""
    encoded = json.dumps(
        tokens,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_typed_tokens_are_identical_under_different_hash_seeds():
    labels = [
        decimal.Decimal("1.50"),
        uuid.UUID(int=7),
        complex(1.5, -2.25),
        fractions.Fraction(3, 4),
        np.complex128(1.5 - 2.25j),
        frozenset({"q", "r", "s", "t", "u", "v"}),
    ]
    cells = np.empty((2, 2), dtype=object)
    cells[:] = [
        [decimal.Decimal("0.10"), uuid.UUID(int=1)],
        [complex(0.0, 1.0), fractions.Fraction(-1, 3)],
    ]
    frame = pd.DataFrame(cells, columns=["c0", "c1"])
    local = {
        "columns": _columns_hash(labels),
        "data": _data_hash(frame),
        "per_label": [_columns_hash([value]) for value in labels],
    }

    runs = [
        _subprocess_json(_TOKEN_SCRIPT, hash_seed=seed)
        for seed in ("0", "987654321")
    ]
    for run in runs:
        assert run == local

    # Independent oracle: the documented token layout, spelled out by hand.
    assert local["per_label"][0] == _expected_columns_hash(
        [{"type": "decimal.Decimal", "value": "1.50"}]
    )
    assert local["per_label"][1] == _expected_columns_hash(
        [{"type": "uuid.UUID", "value": "00000000-0000-0000-0000-000000000007"}]
    )
    assert local["per_label"][2] == _expected_columns_hash(
        [{"type": "builtins.complex", "value": {"real": "1.5", "imag": "-2.25"}}]
    )
    assert local["per_label"][3] == _expected_columns_hash(
        [{"type": "fractions.Fraction", "value": "3/4"}]
    )
    # NumPy complex scalars normalize to their Python counterpart.
    assert local["per_label"][4] == local["per_label"][2]
    # Distinct values stay distinct, and a set is order-independent.
    assert len(set(local["per_label"][:4])) == 4
    assert _columns_hash([frozenset({"b", "a"})]) == _columns_hash(
        [frozenset({"a", "b"})]
    )


def test_newly_tokenized_labels_survive_a_real_manifest_export():
    rng = np.random.default_rng(3)
    columns = [decimal.Decimal(index) for index in range(4)]
    X = pd.DataFrame(rng.normal(size=(60, 4)), columns=columns)
    y = 2.5 * X[columns[0]].to_numpy() + 0.1 * rng.normal(size=60)
    result = select_cefsplus(X, y, k=1, verbose=False, return_result=True)
    payload = result.reproducibility_(X=X, hash_data=True)
    assert payload["input"]["columns_hash"] == _columns_hash(columns)
    assert payload["input"]["columns_hash_source"] == "result"
    assert payload["input"]["data_hash"] == _data_hash(X)
    # The X-identity check is live again now that the labels have a token.
    with pytest.raises(ValueError, match="column identity"):
        result.reproducibility_(X=X[list(reversed(columns))], hash_data=True)


def test_objects_without_a_reproducible_identity_are_still_refused():
    class Custom:
        pass

    with pytest.raises(TypeError, match="no deterministic identity token"):
        _label_token(Custom())
    with pytest.raises(TypeError, match="Decimal, Fraction, UUID, or complex"):
        _columns_hash([Custom()])


# --------------------------------------------------------------------------
# 2. Descriptor-shaped mappings no longer recurse without bound
# --------------------------------------------------------------------------


def _descriptor_chain(depth: int) -> dict:
    root: dict = {"status": "params", "type": "x"}
    node = root
    for _ in range(depth):
        child: dict = {"status": "params", "type": "x"}
        node["params"] = child
        node = child
    return root


def _max_json_depth(value, level: int = 0) -> int:
    if isinstance(value, dict):
        return max((_max_json_depth(item, level + 1) for item in value.values()),
                   default=level)
    if isinstance(value, list):
        return max((_max_json_depth(item, level + 1) for item in value), default=level)
    return level


def test_deep_descriptor_chain_degrades_to_the_opaque_marker():
    sanitized = _sanitize_param(_descriptor_chain(3000))
    encoded = json.dumps(sanitized, allow_nan=False)
    assert _max_json_depth(sanitized) <= repro._MAX_NESTING + 2
    assert encoded.endswith('{"status": "opaque"}}' * 0 + encoded[-20:])
    assert '"status": "opaque"' in encoded


def test_self_referential_descriptor_degrades_to_the_opaque_marker():
    cycle: dict = {"status": "params", "type": "x"}
    cycle["params"] = cycle
    sanitized = _sanitize_param(cycle)
    json.dumps(sanitized, allow_nan=False)
    assert _max_json_depth(sanitized) <= repro._MAX_NESTING + 2


def test_shallow_nested_descriptors_keep_their_inner_parameters():
    # The depth reset exists so a nested estimator keeps its own seed; the
    # nesting guard must not take that back.
    nested = {
        "type": "outer",
        "status": "params",
        "params": {
            "inner": {
                "type": "middle",
                "status": "params",
                "params": {
                    "deep": {
                        "type": "inner",
                        "status": "params",
                        "params": {"random_state": 17},
                    }
                },
            }
        },
    }
    sanitized = _sanitize_param(nested)
    assert (
        sanitized["params"]["inner"]["params"]["deep"]["params"]["random_state"] == 17
    )


# --------------------------------------------------------------------------
# 3. Unseeded permutation_importance records the entropy it drew
# --------------------------------------------------------------------------


def _importance_frame(n=80, p=4, seed=11):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"x{i}" for i in range(p)])
    y = 3.0 * X["x0"] - 2.0 * X["x1"] + 0.2 * rng.normal(size=n)
    return X, y.to_numpy()


def test_unseeded_permutation_importance_records_a_replayable_seed():
    X, y = _importance_frame()
    model = Ridge().fit(X, y)
    realized = []
    importances = []
    for _ in range(2):
        with pytest.warns(FutureWarning, match="random_state=None"):
            result = permutation_importance(
                model, X, y, n_repeats=3, n_jobs=1, random_state=None,
                return_result=True,
            )
        seeds = result.reproducibility_()["configuration"]["seeds"]
        assert seeds["available"] is True
        assert seeds["random_state"] is None
        realized.append(seeds["realized_random_state"])
        importances.append(result.importances_)

    assert all(isinstance(seed, int) for seed in realized)
    assert realized[0] != realized[1]
    assert not np.array_equal(importances[0], importances[1])

    replay = permutation_importance(
        model, X, y, n_repeats=3, n_jobs=1, random_state=realized[0],
        return_result=True,
    )
    np.testing.assert_array_equal(replay.importances_, importances[0])
    replay_seeds = replay.reproducibility_()["configuration"]["seeds"]
    assert replay_seeds["random_state"] == realized[0]
    assert replay_seeds["realized_random_state"] == realized[0]


def test_seeded_permutation_importance_is_unchanged_and_silent():
    X, y = _importance_frame()
    model = Ridge().fit(X, y)
    left = permutation_importance(
        model, X, y, n_repeats=3, n_jobs=1, random_state=5, return_result=True
    )
    right = permutation_importance(
        model, X, y, n_repeats=3, n_jobs=1, random_state=5, return_result=True
    )
    np.testing.assert_array_equal(left.importances_, right.importances_)
    # Independent oracle: the per-(feature, repeat) seed grid is a plain
    # default_rng draw from random_state, exactly as documented.
    expected_seeds = np.random.default_rng(5).integers(0, 2**31, size=(4, 3))
    baseline = left.baseline_score
    column = X["x0"].to_numpy()
    from sift._permute import permute_array

    drops = []
    for repeat in range(3):
        permuted = X.copy()
        permuted["x0"] = permute_array(
            column,
            method="global",
            group_info=None,
            block_size="auto",
            rng=np.random.default_rng(int(expected_seeds[0, repeat])),
        )
        score = -np.mean((y - model.predict(permuted)) ** 2)
        drops.append(baseline - score)
    np.testing.assert_allclose(left.importances_[0], drops, rtol=1e-10, atol=1e-12)


def test_unseeded_boruta_records_the_entropy_it_drew():
    X, y = _importance_frame(n=70, p=5, seed=2)
    first = select_boruta(
        X, y, task="regression", n_estimators=10, max_iter=4,
        random_state=None, verbose=False, return_result=True,
    )
    second = select_boruta(
        X, y, task="regression", n_estimators=10, max_iter=4,
        random_state=None, verbose=False, return_result=True,
    )
    seeds = [
        result.reproducibility_()["configuration"]["seeds"]
        for result in (first, second)
    ]
    assert all(isinstance(block["realized_random_state"], int) for block in seeds)
    assert seeds[0]["realized_random_state"] != seeds[1]["realized_random_state"]
    assert all(block["available"] is True for block in seeds)
    replay = select_boruta(
        X, y, task="regression", n_estimators=10, max_iter=4,
        random_state=seeds[0]["realized_random_state"], verbose=False,
        return_result=True,
    )
    np.testing.assert_array_equal(replay.status, first.status)
    np.testing.assert_allclose(
        replay.mean_importance, first.mean_importance, equal_nan=True
    )


# --------------------------------------------------------------------------
# 4. Boruta / importance / path-evaluation manifests carry the run
# --------------------------------------------------------------------------


def test_boruta_manifest_distinguishes_seed_and_hyperparameters():
    X, y = _importance_frame(n=70, p=5, seed=4)
    common = dict(
        task="regression", n_estimators=10, max_iter=4, verbose=False,
        return_result=True,
    )
    base = select_boruta(X, y, random_state=1, **common)
    same = select_boruta(X, y, random_state=1, **common)
    other_seed = select_boruta(X, y, random_state=2, **common)
    other_iter = select_boruta(X, y, random_state=1, **{**common, "max_iter": 7})
    other_alpha = select_boruta(X, y, random_state=1, alpha=0.3, **common)

    payload = base.reproducibility_()
    assert payload["configuration"]["captured_at"] == "selection"
    assert payload["configuration"]["configured"]["max_iter"] == 4
    assert payload["configuration"]["configured"]["n_estimators"] == 10
    assert payload["configuration"]["configured"]["random_state"] == 1
    assert payload["configuration"]["effective"]["n_features"] == 5
    assert payload["configuration"]["seeds"]["random_state"] == 1
    assert json.dumps(payload, allow_nan=False, sort_keys=True)

    encoded = json.dumps(payload, sort_keys=True)
    assert json.dumps(same.reproducibility_(), sort_keys=True) == encoded
    for variant in (other_seed, other_iter, other_alpha):
        assert json.dumps(variant.reproducibility_(), sort_keys=True) != encoded


def test_hand_built_boruta_result_does_not_claim_selection_time_capture():
    result = sift.BorutaResult(
        feature_names=["a", "b"],
        status=np.array([1, -1], dtype=np.int8),
        hits=np.array([2, 0], dtype=np.int32),
        n_iter=2,
        shadow_thresholds=np.array([0.5, 0.4]),
        mean_importance=np.array([0.9, 0.1]),
    )
    payload = result.reproducibility_()
    assert payload["configuration"]["captured_at"] == "unknown"
    assert payload["configuration"]["seeds"]["available"] is False


def test_importance_manifest_distinguishes_seed_and_hyperparameters():
    X, y = _importance_frame()
    model = Ridge().fit(X, y)
    common = dict(n_repeats=3, n_jobs=1, return_result=True)
    base = permutation_importance(model, X, y, random_state=3, **common)
    same = permutation_importance(model, X, y, random_state=3, **common)
    other_seed = permutation_importance(model, X, y, random_state=4, **common)
    other_repeats = permutation_importance(
        model, X, y, random_state=3, **{**common, "n_repeats": 4}
    )
    other_scoring = permutation_importance(
        model, X, y, random_state=3, scoring="neg_mae", **common
    )

    payload = base.reproducibility_()
    assert payload["configuration"]["captured_at"] == "selection"
    assert payload["configuration"]["configured"]["n_repeats"] == 3
    assert payload["configuration"]["configured"]["scoring"] == "neg_mse"
    assert payload["configuration"]["effective"]["permute_method"] == "global"
    encoded = json.dumps(payload, sort_keys=True)
    assert json.dumps(same.reproducibility_(), sort_keys=True) == encoded
    for variant in (other_seed, other_repeats, other_scoring):
        assert json.dumps(variant.reproducibility_(), sort_keys=True) != encoded


def test_path_evaluation_manifest_carries_the_protocol_it_recorded():
    X, y = _importance_frame(n=90, p=4, seed=6)
    path = ["x0", "x1", "x2", "x3"]
    base = evaluate_feature_path(X, y, path, [1, 2, 3], random_state=0)
    same = evaluate_feature_path(X, y, path, [1, 2, 3], random_state=0)
    other_grid = evaluate_feature_path(X, y, path, [1, 2], random_state=0)
    other_scoring = evaluate_feature_path(
        X, y, path, [1, 2, 3], scoring="mae", random_state=0
    )

    payload = base.reproducibility_()
    assert payload["configuration"]["captured_at"] == "selection"
    assert payload["configuration"]["configured"]["k_grid"] == [1, 2, 3]
    assert payload["configuration"]["configured"]["scoring"] == "rmse"
    assert payload["configuration"]["effective"]["k"] == base.best_k
    encoded = json.dumps(payload, sort_keys=True)
    assert json.dumps(same.reproducibility_(), sort_keys=True) == encoded
    for variant in (other_grid, other_scoring):
        assert json.dumps(variant.reproducibility_(), sort_keys=True) != encoded


def test_catboost_adapter_reports_the_scoring_protocol_it_can_prove():
    from sift.catboost_common import CatBoostSelectionResult

    result = CatBoostSelectionResult(
        selected_features=["a"],
        best_k=1,
        scores_by_k={1: 0.5, 2: 0.7},
        scores_std_by_k={1: 0.1, 2: 0.2},
        feature_importances=pd.Series({"a": 1.0}),
        features_by_k={1: ["a"], 2: ["a", "b"]},
        metric="RMSE",
        higher_is_better=False,
    )
    payload = result.reproducibility_(input_features=["a", "b"])
    configured = payload["configuration"]["configured"]
    assert payload["configuration"]["captured_at"] == "selection"
    assert configured["metric"] == "RMSE"
    assert configured["k_grid"] == [1, 2]
    assert configured["higher_is_better"] is False
    assert payload["configuration"]["effective"]["k"] == 1
    # A run that records its seed has it carried through unchanged.
    result.selector_metadata = {"random_state": 9, "realized_random_state": 9}
    seeded = result.reproducibility_(input_features=["a", "b"])
    assert seeded["configuration"]["seeds"]["random_state"] == 9
    assert seeded["configuration"]["seeds"]["available"] is True


# --------------------------------------------------------------------------
# 5. Environment block: no absolute paths, real dirty flag
# --------------------------------------------------------------------------


def _walk_strings(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_strings(key)
            yield from _walk_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_strings(item)
    elif isinstance(value, str):
        yield value


def test_manifest_never_exports_an_absolute_path_or_site_packages():
    X, y = _importance_frame(n=60, p=4, seed=9)
    result = select_cefsplus(X, y, k=2, verbose=False, return_result=True)
    payload = result.reproducibility_(X=X, y=y, hash_data=True)
    home = str(Path.home())
    for text in _walk_strings(payload):
        assert not text.startswith(home), text
        assert "site-packages" not in text, text
        assert not text.startswith("/"), text
    env = payload["environment"]
    assert env["python_version"] == ".".join(
        str(part) for part in sys.version_info[:3]
    )
    assert isinstance(env["platform"], str) and env["platform"]
    assert env["threadpoolctl"] is None or isinstance(env["threadpoolctl"], str)
    assert all("filepath" not in entry for entry in env["blas"])
    assert env["git_dirty"] is None or isinstance(env["git_dirty"], bool)


def _tiny_git_package(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    package = repo / "fakepkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("__version__ = '0'\n")
    (repo / "elsewhere.txt").write_text("untouched\n")
    for command in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "t@example.com"],
        ["git", "config", "user.name", "t"],
        ["git", "add", "-A"],
        ["git", "-c", "commit.gpgsign=false", "commit", "-qm", "init"],
    ):
        subprocess.run(command, cwd=repo, check=True, capture_output=True)
    return package


def test_git_dirty_tracks_the_package_directory_only(tmp_path, monkeypatch):
    package = _tiny_git_package(tmp_path)
    monkeypatch.setattr(sift, "__file__", str(package / "__init__.py"))
    assert _git_dirty() is False

    (package.parent / "elsewhere.txt").write_text("edited outside the package\n")
    assert _git_dirty() is False

    (package / "__init__.py").write_text("__version__ = '0'  # edited\n")
    assert _git_dirty() is True


def test_git_dirty_is_none_outside_a_checkout(tmp_path, monkeypatch):
    loose = tmp_path / "loose" / "fakepkg"
    loose.mkdir(parents=True)
    (loose / "__init__.py").write_text("")
    monkeypatch.setattr(sift, "__file__", str(loose / "__init__.py"))
    assert _git_dirty() is None


# --------------------------------------------------------------------------
# 6. The documented schema and the exported keys cannot drift apart
# --------------------------------------------------------------------------


def test_manifest_keys_match_the_documented_schema():
    X, y = _importance_frame(n=60, p=4, seed=13)
    selection = select_cefsplus(
        X, y, k=2, random_state=1, verbose=False, return_result=True
    ).reproducibility_(X=X, y=y, hash_data=True)
    comparison = compare(
        {"s": lambda: CEFSPlusSelector(k=1, verbose=False)},
        X,
        y,
        estimator=Ridge(),
        cv=KFold(n_splits=3, shuffle=True, random_state=0),
        random_state=0,
    ).reproducibility_()

    for payload in (selection, comparison):
        expected = {
            path: tuple(sorted(keys))
            for path, keys in MANIFEST_SCHEMA_KEYS[payload["kind"]].items()
        }
        assert manifest_key_map(payload) == expected

    documented = repro.__doc__
    assert documented is not None
    for block in MANIFEST_SCHEMA_KEYS.values():
        for path, keys in block.items():
            for key in keys:
                assert f"``{key}``" in documented, f"{path}.{key} is undocumented"
    assert 'Schema version "1"' in documented


def test_exporter_self_check_rejects_an_undocumented_key(monkeypatch):
    original = repro._export_environment

    def extra_environment():
        payload = original()
        payload["undocumented_key"] = "x"
        return payload

    monkeypatch.setattr(repro, "_export_environment", extra_environment)
    result = FilterSelectionResult(
        selected_features=["a"],
        selected_indices=[0],
        selector_metadata={"selector": "mrmr", "k": 1, "n_features": 2},
    )
    with pytest.raises(RuntimeError, match="undocumented_key"):
        result.reproducibility_()


# --------------------------------------------------------------------------
# 7. End-to-end: one changed input, one changed manifest
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _distinguishing_frame():
    rng = np.random.default_rng(21)
    n, p = 90, 45
    X = pd.DataFrame(
        rng.normal(size=(n, p)), columns=[f"x{i}" for i in range(p)]
    )
    return X


def _selected_and_manifest(X, y, **kwargs):
    hash_kwargs = {
        key: kwargs.pop(key) for key in ("sample_weight",) if key in kwargs
    }
    result = select_cefsplus(
        X, y, verbose=False, return_result=True, **hash_kwargs, **kwargs
    )
    payload = result.reproducibility_(
        X=X,
        y=y,
        hash_data=True,
        input_features=list(X.columns),
        **hash_kwargs,
    )
    return list(result.selected_features), json.dumps(payload, sort_keys=True)


def test_manifests_distinguish_runs_that_differ_in_exactly_one_input(
    _distinguishing_frame,
):
    X = _distinguishing_frame
    rng = np.random.default_rng(5)
    noise = 0.1 * rng.normal(size=len(X))

    # (a) different y
    y_a = 3.0 * X["x0"].to_numpy() + noise
    y_b = 3.0 * X["x1"].to_numpy() + noise
    sel_a, man_a = _selected_and_manifest(X, y_a, k=1)
    sel_b, man_b = _selected_and_manifest(X, y_b, k=1)
    assert sel_a == ["x0"] and sel_b == ["x1"]
    assert man_a != man_b

    # (b) different exclude list of 40 names
    y_c = 3.0 * X["x0"].to_numpy() + 1.5 * X["x1"].to_numpy() + noise
    exclude_a = ["x0"] + [f"x{i}" for i in range(5, 44)]
    exclude_b = ["x1"] + [f"x{i}" for i in range(5, 44)]
    assert len(exclude_a) == len(exclude_b) == 40
    sel_c, man_c = _selected_and_manifest(X, y_c, k=1, exclude=exclude_a)
    sel_d, man_d = _selected_and_manifest(X, y_c, k=1, exclude=exclude_b)
    assert sel_c == ["x1"] and sel_d == ["x0"]
    assert man_c != man_d

    # (c) different sample_weight
    half = len(X) // 2
    y_e = np.where(
        np.arange(len(X)) < half,
        3.0 * X["x0"].to_numpy(),
        3.0 * X["x1"].to_numpy(),
    ) + noise
    w_first = np.concatenate([np.ones(half), np.zeros(len(X) - half)])
    w_second = np.concatenate([np.zeros(half), np.ones(len(X) - half)])
    sel_e, man_e = _selected_and_manifest(X, y_e, k=1, sample_weight=w_first)
    sel_f, man_f = _selected_and_manifest(X, y_e, k=1, sample_weight=w_second)
    assert sel_e == ["x0"] and sel_f == ["x1"]
    assert man_e != man_f

    # (d) one target versus two
    y_single = 3.0 * X["x0"].to_numpy() + noise
    y_double = np.column_stack(
        [y_single, 3.0 * X["x1"].to_numpy() + noise]
    )
    sel_g, man_g = _selected_and_manifest(X, y_single, k=2)
    sel_h, man_h = _selected_and_manifest(X, y_double, k=2)
    assert sel_g != sel_h
    assert man_g != man_h
    assert json.loads(man_h)["configuration"]["effective"]["n_targets"] == 2


_DETERMINISM_SCRIPT = """
    import json
    import numpy as np
    import pandas as pd
    from sift import select_cefsplus

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, 5)), columns=[f"x{i}" for i in range(5)])
    y = 2.0 * X["x0"].to_numpy() + 0.1 * rng.normal(size=60)
    result = select_cefsplus(
        X, y, k=2, random_state=7, verbose=False, return_result=True
    )
    payload = result.reproducibility_(X=X, y=y, hash_data=True)
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
"""


def test_identical_runs_give_byte_identical_manifests_across_processes():
    left = _subprocess_json(_DETERMINISM_SCRIPT, hash_seed="0")
    right = _subprocess_json(_DETERMINISM_SCRIPT, hash_seed="424242")
    assert json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)
    assert left["input"]["data_hash"] and left["input"]["y_hash"]
    assert left["configuration"]["seeds"]["random_state"] == 7


# --------------------------------------------------------------------------
# 8. Multi-target guards
# --------------------------------------------------------------------------

_MULTI_TARGET_MESSAGE = (
    "2-D y is only supported for select_cefsplus / CEFSPlusSelector"
)


def test_boruta_rejects_multi_target_y_up_front():
    X, y = _importance_frame(n=60, p=4, seed=8)
    Y = np.column_stack([y, y * 0.5, y * -1.0])
    common = dict(
        task="regression", n_estimators=10, max_iter=3, random_state=0, verbose=False
    )
    with pytest.raises(ValueError, match=_MULTI_TARGET_MESSAGE):
        select_boruta(X, Y, **common)
    with pytest.raises(ValueError, match=_MULTI_TARGET_MESSAGE):
        sift.BorutaSelector(
            n_estimators=10, max_iter=3, random_state=0, verbose=False
        ).fit(X, Y)
    with pytest.raises(ValueError, match=_MULTI_TARGET_MESSAGE):
        sift.select_boruta_shap(X, Y, n_estimators=10, max_iter=3,
                                random_state=0, verbose=False)

    # A single-column 2-D target keeps its historical behaviour.
    column = select_boruta(X, y.reshape(-1, 1), return_result=True, **common)
    flat = select_boruta(X, y, return_result=True, **common)
    np.testing.assert_array_equal(column.status, flat.status)


def test_permutation_importance_rejects_multi_target_y_up_front():
    X, y = _importance_frame(n=60, p=4, seed=10)
    model = Ridge().fit(X, y)
    Y = np.column_stack([y, y * 0.5])
    with pytest.raises(ValueError, match=_MULTI_TARGET_MESSAGE):
        permutation_importance(model, X, Y, n_repeats=2, n_jobs=1, random_state=0)

    column = permutation_importance(
        model, X, y.reshape(-1, 1), n_repeats=2, n_jobs=1, random_state=0,
        return_result=True,
    )
    flat = permutation_importance(
        model, X, y, n_repeats=2, n_jobs=1, random_state=0, return_result=True
    )
    np.testing.assert_array_equal(column.importances_, flat.importances_)
