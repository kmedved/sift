"""Contract tests for the generated public API reference."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re

import sift


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_api_reference.py"
SPHINX_ROLE = re.compile(r":[A-Za-z][A-Za-z0-9_-]*:`[^`]+`")


def _generator_module():
    spec = importlib.util.spec_from_file_location("generate_api_reference", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_reference_covers_public_surface_exactly() -> None:
    generator = _generator_module()
    grouped = [
        name
        for _, names in generator.PUBLIC_API_GROUPS
        for name in names
    ]

    assert len(grouped) == len(set(grouped))
    assert set(grouped) == set(sift.__all__)


def test_generated_reference_files_are_current() -> None:
    generator = _generator_module()
    expected = generator.expected_files()
    actual = set(generator.REFERENCE_DIR.glob("*.md"))

    assert actual == set(expected)
    for path, content in expected.items():
        assert path.read_text(encoding="utf-8") == content


def test_python_docstrings_do_not_leak_sphinx_roles_into_mkdocstrings() -> None:
    offenders: list[str] = []
    for path in sorted((ROOT / "sift").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for match in SPHINX_ROLE.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{path.relative_to(ROOT)}:{line}: {match.group()}")

    assert offenders == []
