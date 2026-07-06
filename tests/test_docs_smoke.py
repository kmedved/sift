import ast
import pathlib
import re

import sift


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _extract_docs_import_names(text: str) -> set[str]:
    match = re.search(r"Top-level exports:\s*```(?:[a-z]+)?\n(.*?)\n```", text, re.S)
    assert match, "Could not find the top-level exports code block in DOCS.MD"
    module = ast.parse(match.group(1))
    names: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.ImportFrom) and node.module == "sift":
            for alias in node.names:
                names.add(alias.name)
    return names


def _read_pyproject_extras() -> set[str]:
    pyproject_text = (ROOT / "pyproject.toml").read_text(encoding="utf8")
    match = re.search(
        r"^\[project\.optional-dependencies\]\n(.*?)(?=^\[|\Z)",
        pyproject_text,
        re.S | re.M,
    )
    assert match, "Could not find project.optional-dependencies in pyproject.toml"
    return set(re.findall(r"^([A-Za-z0-9_-]+)\s*=", match.group(1), re.M))


def test_docs_top_level_exports_match_package_public_api():
    docs_text = (ROOT / "DOCS.MD").read_text(encoding="utf8")
    documented = _extract_docs_import_names(docs_text)
    expected = set(sift.__all__) - {"__version__"}

    assert documented == expected


def test_docs_install_extras_match_pyproject():
    docs_text = (ROOT / "DOCS.MD").read_text(encoding="utf8")
    docs_extras = set(re.findall(r'python -m pip install -e "?\.\[([^\]]+)\]"?', docs_text))
    pyproject_extras = _read_pyproject_extras()

    assert docs_extras == pyproject_extras
