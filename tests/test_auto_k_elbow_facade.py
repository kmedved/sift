"""Facade aliases and public contracts for extracted select_k_elbow."""

from __future__ import annotations

import ast
import inspect
import pickle
from pathlib import Path

import numpy as np
import pytest

import sift
from sift import select_k_elbow as top_level_select_k_elbow
from sift.api import select_k_elbow as api_select_k_elbow
from sift.selection import auto_k as auto_k_module, auto_k_elbow, filter_auto_k_common
from sift.selection import select_k_elbow as selection_select_k_elbow
from sift.selection.auto_k import select_k_elbow as facade_select_k_elbow


def test_select_k_elbow_is_identical_across_public_import_routes() -> None:
    sibling = auto_k_elbow.select_k_elbow
    assert sibling is facade_select_k_elbow
    assert sibling is auto_k_module.select_k_elbow
    assert sibling is selection_select_k_elbow
    assert sibling is api_select_k_elbow
    assert sibling is top_level_select_k_elbow
    assert sibling is sift.select_k_elbow
    assert sibling is filter_auto_k_common.auto_k_module.select_k_elbow
    assert sibling.__module__ == "sift.selection.auto_k"
    assert pickle.loads(pickle.dumps(sibling)) is sibling


def test_select_k_elbow_signature_and_leaf_imports() -> None:
    signature = inspect.signature(facade_select_k_elbow)
    assert list(signature.parameters) == [
        "objective_path",
        "min_k",
        "max_k",
        "min_rel_gain",
        "patience",
    ]
    defaults = {name: param.default for name, param in signature.parameters.items()}
    assert defaults["min_k"] == 5
    assert defaults["max_k"] == 100
    assert defaults["min_rel_gain"] == 0.02
    assert defaults["patience"] == 3

    tree = ast.parse(Path(auto_k_elbow.__file__).read_text())
    runtime_imported = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            runtime_imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            runtime_imported.append(node.module)
    runtime_imported = [name for name in runtime_imported if name != "__future__"]
    assert "sift.selection.auto_k" not in runtime_imported
    assert all(not name.startswith("sift.") for name in runtime_imported)
    assert set(runtime_imported) == {"typing", "numpy", "pandas"}


def test_select_k_elbow_representative_outputs_and_errors() -> None:
    empty_k, empty_diag = facade_select_k_elbow(np.array([]))
    assert empty_k == 0
    assert empty_diag.empty

    objective = np.array([1.0, 1.8, 2.4, 2.42, 2.43, 2.44])
    best_k, diag = facade_select_k_elbow(
        objective, min_k=1, max_k=6, min_rel_gain=0.05, patience=2
    )
    assert best_k == 3
    assert list(diag.columns) == ["k", "objective", "delta", "rel_gain"]
    assert diag["k"].tolist() == [1, 2, 3, 4, 5, 6]
    assert diag[["objective", "delta", "rel_gain"]].dtypes.eq(np.dtype("float64")).all()

    with pytest.raises(ValueError, match="finite"):
        facade_select_k_elbow(np.array([1.0, np.nan]))
    with pytest.raises(ValueError, match="one-dimensional"):
        facade_select_k_elbow(np.array([[1.0, 2.0]]))
    with pytest.raises(ValueError, match="min_k"):
        facade_select_k_elbow(np.array([1.0, 2.0, 3.0]), min_k=3, max_k=2)
