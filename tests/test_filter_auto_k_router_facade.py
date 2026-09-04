"""Facade aliases for extracted auto-k router configuration helpers."""

from __future__ import annotations

import ast
from pathlib import Path

from sift.selection import filter_auto_k, filter_auto_k_router


ROUTER_NAMES = (
    "auto_k_mode_label",
    "_auto_route_facts",
    "_AUTOK_FIELD_DEFAULTS",
    "_strip_router_only_fields",
    "_auto_route_config",
)


def test_router_cluster_names_are_facade_aliases() -> None:
    for name in ROUTER_NAMES:
        assert getattr(filter_auto_k, name) is getattr(filter_auto_k_router, name)


def test_router_sibling_is_a_leaf() -> None:
    tree = ast.parse(Path(filter_auto_k_router.__file__).read_text())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    imported = [name for name in imported if name != "__future__"]
    assert "sift.selection.filter_auto_k" not in imported
    assert all(not name.startswith("sift.selection.filter_auto_k.") for name in imported)
    assert set(imported) == {"dataclasses", "numpy", "sift.selection.auto_k"}
