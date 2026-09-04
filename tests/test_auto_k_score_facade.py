"""Facade aliases for extracted auto-k score-curve rule helpers."""

from __future__ import annotations

import ast
from pathlib import Path

from sift.selection import auto_k as auto_k_module, auto_k_score
from sift.selection.auto_k import AutoKConfig, choose_k_from_score_curve


SCORE_NAMES = (
    "_score_curve_tolerance",
    "_choose_best_rule",
    "_choose_one_se_rule",
    "_mark_tolerance",
    "_choose_tolerance_rule",
    "_selected_plateau_ks",
    "_choose_plateau_rule",
    "_RULE_SELECTORS",
)


def test_score_rule_names_are_facade_aliases() -> None:
    for name in SCORE_NAMES:
        assert getattr(auto_k_module, name) is getattr(auto_k_score, name)
    selectors = auto_k_module._RULE_SELECTORS
    assert selectors is auto_k_score._RULE_SELECTORS
    assert selectors["best"] is auto_k_score._choose_best_rule
    assert selectors["one_se"] is auto_k_score._choose_one_se_rule
    assert selectors["tolerance"] is auto_k_score._choose_tolerance_rule
    assert selectors["plateau"] is auto_k_score._choose_plateau_rule
    assert choose_k_from_score_curve.__module__ == "sift.selection.auto_k"
    assert AutoKConfig.__module__ == "sift.selection.auto_k"


def test_score_sibling_is_a_leaf() -> None:
    tree = ast.parse(Path(auto_k_score.__file__).read_text())
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
    assert all(not name.startswith("sift.selection.auto_k.") for name in runtime_imported)
    assert set(runtime_imported) == {"typing", "warnings", "numpy"}
    assert set(type_checking_imported) == {"pandas", "sift.selection.auto_k"}
