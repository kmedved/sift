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


def _read_setup_extras() -> set[str]:
    module = ast.parse((ROOT / "setup.py").read_text(encoding="utf8"))
    for node in module.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if getattr(call.func, "id", None) != "setup":
            continue
        for keyword in call.keywords:
            if keyword.arg != "extras_require":
                continue
            extras = ast.literal_eval(keyword.value)
            return set(extras)
    raise AssertionError("Could not find extras_require in setup.py")


def test_docs_top_level_exports_match_package_public_api():
    docs_text = (ROOT / "DOCS.MD").read_text(encoding="utf8")
    documented = _extract_docs_import_names(docs_text)
    expected = set(sift.__all__) - {"__version__"}

    assert documented == expected


def test_docs_install_extras_match_setup_py():
    docs_text = (ROOT / "DOCS.MD").read_text(encoding="utf8")
    docs_extras = set(re.findall(r'python -m pip install -e "?\.\[([^\]]+)\]"?', docs_text))
    setup_extras = _read_setup_extras()

    assert docs_extras == setup_extras
