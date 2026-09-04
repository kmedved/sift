"""Facade aliases for shared auto-k count and guard primitives."""

from __future__ import annotations

import ast
from pathlib import Path

from sift.selection import auto_k as auto_k_module, filter_auto_k, filter_auto_k_common


COMMON_NAMES = (
    "auto_k_summary",
    "_zero_capable_effective_min_k",
    "_effective_max_k",
    "_require_eval_split_context",
    "_print_selected_k",
    "_select_elbow_count",
    "_select_penalized_count",
    "_select_posterior_count",
    "_objective_n_eff",
    "_gain_test_candidate_inputs",
)


def test_common_cluster_names_are_facade_aliases() -> None:
    for name in COMMON_NAMES:
        assert getattr(filter_auto_k, name) is getattr(filter_auto_k_common, name)
    assert filter_auto_k.auto_k_module is filter_auto_k_common.auto_k_module
    assert filter_auto_k.auto_k_module is auto_k_module


def test_common_sibling_is_a_leaf() -> None:
    tree = ast.parse(Path(filter_auto_k_common.__file__).read_text())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    assert "sift.selection.filter_auto_k" not in imported
    assert all(not name.startswith("sift.selection.filter_auto_k.") for name in imported)
    auto_k_binding = [
        (node.module, alias.name, alias.asname)
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.asname == "auto_k_module"
    ]
    assert auto_k_binding == [("sift.selection", "auto_k", "auto_k_module")]
