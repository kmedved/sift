"""Contracts for the canonical glossary and its generated-doc links."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
GLOSSARY = ROOT / "docs" / "glossary.md"
API_SCRIPT = ROOT / "scripts" / "generate_api_reference.py"
MATRIX_SCRIPT = ROOT / "scripts" / "generate_data_type_matrix.py"
LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")

REQUIRED_HEADINGS = (
    "All-relevant",
    "Approximate plugin",
    "Auto-k",
    "Boruta",
    "Conditional gain",
    "False discovery proportion",
    "False discovery rate",
    "Feature cache",
    "Feature path",
    "Fixed-k",
    "Gaussian copula",
    "Groups",
    "Inclusion weights",
    "Joint mutual information",
    "k",
    "Knockoff plus",
    "Knockoffs",
    "Leakage",
    "Model-X",
    "Out-of-fold",
    "Permutation importance",
    "q",
    "Rank-Gaussian transform",
    "Redundancy",
    "Relevance",
    "Result view",
    "Row metadata",
    "Sample weight",
    "Selection curve",
    "Selection rule",
    "Shadow feature",
    "Smart sampling",
    "Stability frequency",
    "Stability selection",
    "Stopping rule",
    "Target encoding",
    "target_cv",
    "Time",
    "W statistic",
)


def _slugify(heading: str) -> str:
    value = re.sub(r"[^\w\s-]", "", heading, flags=re.UNICODE).strip().lower()
    return re.sub(r"[-\s]+", "-", value)


def _headings(text: str) -> list[str]:
    return re.findall(r"^## (.+)$", text, flags=re.M)


def _load(script: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_glossary_has_required_headings_in_alphabetical_order() -> None:
    headings = _headings(GLOSSARY.read_text(encoding="utf-8"))
    assert headings == sorted(headings, key=str.casefold)
    missing = [name for name in REQUIRED_HEADINGS if name not in headings]
    assert missing == []


def test_glossary_is_linked_from_nav_and_root_maps() -> None:
    assert "glossary.md" in (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "(glossary.md)" in (ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    assert "docs/glossary.md" in (ROOT / "DOCS.MD").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "https://github.com/kmedved/sift/blob/main/docs/glossary.md" in readme


def test_glossary_is_linked_from_canonical_guides() -> None:
    for relpath in (
        "docs/choosing-a-selector.md",
        "docs/user-guide.md",
        "docs/ALGORITHMS.md",
        "docs/ADVANCED.md",
        "docs/results.md",
        "docs/troubleshooting.md",
        "docs/runtime-scaling.md",
        "docs/knockoff-statistic-bakeoff.md",
    ):
        text = (ROOT / relpath).read_text(encoding="utf-8")
        assert "glossary.md" in text, relpath


def test_glossary_internal_links_resolve() -> None:
    text = GLOSSARY.read_text(encoding="utf-8")
    anchors = {_slugify(heading) for heading in _headings(text)}
    docs = ROOT / "docs"
    for href in LINK_RE.findall(text):
        if href.startswith(("http://", "https://", "mailto:")):
            continue
        path_part, _, fragment = href.partition("#")
        if path_part:
            target = (docs / path_part).resolve()
            assert target.is_file(), href
        if fragment:
            if path_part:
                page_headings = {
                    _slugify(heading)
                    for heading in _headings(target.read_text(encoding="utf-8"))
                }
                # Also allow H1/H3 slugs from destination pages.
                page_headings.update(
                    _slugify(heading)
                    for heading in re.findall(
                        r"^#{1,6} (.+)$",
                        target.read_text(encoding="utf-8"),
                        flags=re.M,
                    )
                )
                assert fragment in page_headings, href
            else:
                assert fragment in anchors, href


def test_glossary_fragment_links_from_docs_resolve() -> None:
    anchors = {_slugify(heading) for heading in _headings(GLOSSARY.read_text(encoding="utf-8"))}
    for path in sorted((ROOT / "docs").rglob("*.md")):
        for href in LINK_RE.findall(path.read_text(encoding="utf-8")):
            if "glossary.md#" not in href:
                continue
            fragment = href.split("#", 1)[1]
            assert fragment in anchors, f"{path.relative_to(ROOT)} -> {href}"


def test_generated_api_pages_link_to_the_glossary() -> None:
    generator = _load(API_SCRIPT, "generate_api_reference")
    for path, content in generator.expected_files().items():
        assert generator.GLOSSARY_LINK in content, path.name


def test_generated_data_type_page_links_to_the_glossary() -> None:
    generator = _load(MATRIX_SCRIPT, "generate_data_type_matrix")
    page = generator.render_page(generator.probe_published())
    assert "[glossary](glossary.md)" in page
