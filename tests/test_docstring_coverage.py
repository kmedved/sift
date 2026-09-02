"""Every public export carries a substantive numpydoc docstring (0.9 DoD item 6)."""

from __future__ import annotations

import inspect
import re

import pytest

import sift

MIN_DOC_LINES = 8
_SECTION = re.compile(r"^(Parameters|Attributes)\n-{6,}$", re.MULTILINE)
_PARAM_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]*)*\s*:", re.MULTILINE)

PUBLIC_NAMES = [name for name in sift.__all__ if name != "__version__"]


def _documented_names(doc: str, *sections: str) -> set[str]:
    """Names introduced as ``name : type`` entries under the given numpydoc sections."""
    names: set[str] = set()
    for section in sections:
        pattern = rf"^{section}\n-{{6,}}\n(.*?)(?=^\S[^\n]*\n-{{3,}}$|\Z)"
        match = re.search(pattern, doc, re.MULTILINE | re.DOTALL)
        if match is None:
            continue
        for line in match.group(1).splitlines():
            if line and not line[0].isspace() and ":" in line:
                head = line.split(":", 1)[0]
                names.update(part.strip().lstrip("*") for part in head.split(","))
    return names


@pytest.mark.parametrize("name", PUBLIC_NAMES)
def test_public_export_has_numpydoc_docstring(name: str) -> None:
    obj = getattr(sift, name)
    doc = inspect.getdoc(obj) or ""
    assert len(doc.splitlines()) >= MIN_DOC_LINES, f"{name}: docstring has fewer than {MIN_DOC_LINES} lines"
    if inspect.isclass(obj):
        assert _SECTION.search(doc), f"{name}: class docstring lacks a Parameters/Attributes section"
        target = obj.__init__
    else:
        assert callable(obj), name
        target = obj
    signature = inspect.signature(target)
    expected = {
        p.name
        for p in signature.parameters.values()
        if p.name not in {"self", "cls"} and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    }
    if not expected:
        return
    sections = ("Parameters", "Attributes") if inspect.isclass(obj) else ("Parameters",)
    documented = _documented_names(doc, *sections)
    missing = sorted(expected - documented)
    assert not missing, f"{name}: Parameters section is missing {missing}"
