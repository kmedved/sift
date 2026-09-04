"""Facade aliases for extracted cache/eval/classic auto-k helpers."""

from __future__ import annotations

import ast
from pathlib import Path

from sift.selection import auto_k as auto_k_module, filter_auto_k, filter_auto_k_cache


CACHE_NAMES = (
    "prepare_filter_eval_data",
    "_cached_filter_path",
    "_cache_uses_synthetic_feature_names",
    "_require_positional_cache_dataframe_alignment",
    "select_filter_classic_auto_k",
)


def test_cache_cluster_names_are_facade_aliases() -> None:
    for name in CACHE_NAMES:
        assert getattr(filter_auto_k, name) is getattr(filter_auto_k_cache, name)
    assert filter_auto_k.auto_k_module is filter_auto_k_cache.auto_k_module
    assert filter_auto_k.auto_k_module is auto_k_module


def test_cache_sibling_is_a_leaf() -> None:
    tree = ast.parse(Path(filter_auto_k_cache.__file__).read_text())
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
