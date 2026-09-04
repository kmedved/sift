"""Facade aliases for extracted auto-k penalty/objective helpers."""

from __future__ import annotations

import ast
from pathlib import Path

from sift.selection import auto_k as auto_k_module, auto_k_objective, filter_auto_k_common
from sift.selection.auto_k import AutoKConfig


OBJECTIVE_NAMES = (
    "_resolve_n_eff_mode",
    "_penalty_weight",
    "_log_comb",
    "_resolve_ebic_gamma",
    "_penalty_array",
    "_objective_weight_diagnostics",
)


def test_objective_helper_names_are_facade_aliases() -> None:
    for name in OBJECTIVE_NAMES:
        assert getattr(auto_k_module, name) is getattr(auto_k_objective, name)
    assert filter_auto_k_common.auto_k_module is auto_k_module
    assert (
        filter_auto_k_common.auto_k_module._objective_weight_diagnostics
        is auto_k_objective._objective_weight_diagnostics
    )
    assert AutoKConfig.__module__ == "sift.selection.auto_k"


def test_objective_sibling_is_a_leaf() -> None:
    tree = ast.parse(Path(auto_k_objective.__file__).read_text())
    runtime_imported = []
    type_checking_imported = []
    for node in tree.body:
        targets = type_checking_imported if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ) else None
        if targets is type_checking_imported:
            import_nodes = node.body
        else:
            import_nodes = [node]
        for inner in import_nodes:
            if isinstance(inner, ast.Import):
                names = [alias.name for alias in inner.names]
            elif isinstance(inner, ast.ImportFrom) and inner.module is not None:
                names = [inner.module]
            else:
                continue
            if targets is type_checking_imported:
                type_checking_imported.extend(names)
            else:
                runtime_imported.extend(names)
    runtime_imported = [name for name in runtime_imported if name != "__future__"]
    assert "sift.selection.auto_k" not in runtime_imported
    assert all(not name.startswith("sift.selection.auto_k.") for name in runtime_imported)
    assert set(runtime_imported) == {"typing", "numpy", "scipy.special", "sift._preprocess"}
    assert type_checking_imported == ["sift.selection.auto_k"]
