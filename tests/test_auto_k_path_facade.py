"""Facade aliases and public contracts for extracted compute_objective_for_path."""

from __future__ import annotations

import ast
import inspect
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sift
from sift import build_cache, compute_objective_for_path as top_level_compute
from sift.selection import auto_k as auto_k_module, auto_k_path
from sift.selection import compute_objective_for_path as selection_compute
from sift.selection.auto_k import compute_objective_for_path as facade_compute


def test_compute_objective_for_path_is_identical_across_public_import_routes() -> None:
    sibling = auto_k_path.compute_objective_for_path
    assert sibling is facade_compute
    assert sibling is auto_k_module.compute_objective_for_path
    assert sibling is selection_compute
    assert sibling is top_level_compute
    assert sibling is sift.compute_objective_for_path
    assert sibling.__module__ == "sift.selection.auto_k"
    assert pickle.loads(pickle.dumps(sibling)) is sibling


def test_compute_objective_for_path_signature_and_leaf_imports() -> None:
    signature = inspect.signature(facade_compute)
    assert list(signature.parameters) == ["cache", "y", "feature_path", "shrink", "eps"]
    defaults = {name: param.default for name, param in signature.parameters.items()}
    assert defaults["shrink"] == 1e-6
    assert defaults["eps"] == 1e-12

    tree = ast.parse(Path(auto_k_path.__file__).read_text())
    runtime_imported = []
    type_checking_imported = []
    for node in tree.body:
        in_type_checking = (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        )
        import_nodes = node.body if in_type_checking else [node]
        bucket = type_checking_imported if in_type_checking else runtime_imported
        for inner in import_nodes:
            if isinstance(inner, ast.Import):
                bucket.extend(alias.name for alias in inner.names)
            elif isinstance(inner, ast.ImportFrom) and inner.module is not None:
                bucket.append(inner.module)
    runtime_imported = [name for name in runtime_imported if name != "__future__"]
    assert "sift.selection.auto_k" not in runtime_imported
    assert all(not name.startswith("sift.selection.auto_k") for name in runtime_imported)
    assert set(runtime_imported) == {"typing", "numpy"}
    assert type_checking_imported == ["sift.estimators.copula"]

    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    local_modules = [
        node.module
        for node in fn.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    ]
    assert local_modules == [
        "sift.estimators.copula",
        "sift.selection.objective",
        "sift.selection.knockoff_filter",
    ]


def test_compute_objective_for_path_cached_uncached_and_errors() -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(80, 4)), columns=list("abcd"))
    y = (X["a"] + 0.7 * X["b"] + 0.2 * rng.normal(size=80)).to_numpy()
    cached = build_cache(X, compute_Rxx=True, subsample=None)
    uncached = build_cache(X, compute_Rxx=False, subsample=None)

    cached_obj = facade_compute(cached, y, ["a", "b", "c"])
    uncached_obj = facade_compute(uncached, y, ["a", "b", "c"])
    assert cached_obj.shape == (3,)
    assert cached_obj.dtype == np.float64
    assert np.isfinite(cached_obj).all()
    assert np.all(np.diff(cached_obj) >= -1e-12)
    np.testing.assert_allclose(cached_obj, uncached_obj, atol=1e-10)
    assert facade_compute(cached, y, ["missing"]).size == 0
    assert facade_compute(cached, y, []).size == 0
    int_path_obj = facade_compute(cached, y, [0, 1])
    assert int_path_obj.shape == (2,)

    with pytest.raises(ValueError, match="cache was built from 80 rows"):
        facade_compute(cached, y[:-1], ["a", "b"])
