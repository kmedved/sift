"""Facade aliases for extracted binary auto-k route helpers."""

from __future__ import annotations

import ast
from pathlib import Path

from sift.selection import auto_k as auto_k_module, filter_auto_k, filter_auto_k_binary


BINARY_NAMES = (
    "select_binary_elbow",
    "select_binary_penalized",
    "select_binary_posterior",
    "select_binary_changepoint",
    "select_binary_evaluate",
)


def test_binary_route_names_are_facade_aliases() -> None:
    for name in BINARY_NAMES:
        assert getattr(filter_auto_k, name) is getattr(filter_auto_k_binary, name)
    assert filter_auto_k.auto_k_module is filter_auto_k_binary.auto_k_module
    assert filter_auto_k.auto_k_module is auto_k_module


def test_binary_sibling_is_a_leaf() -> None:
    tree = ast.parse(Path(filter_auto_k_binary.__file__).read_text())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    assert "sift.selection.filter_auto_k" not in imported
    assert all(not name.startswith("sift.selection.filter_auto_k.") for name in imported)
    assert "sift.selection.filter_auto_k_common" in imported
    auto_k_binding = [
        (node.module, alias.name, alias.asname)
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.asname == "auto_k_module"
    ]
    assert auto_k_binding == [("sift.selection", "auto_k", "auto_k_module")]
