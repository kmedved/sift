"""The release notes reproduce the 0.9 deprecation ledger verbatim (0.9 DoD item 4)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTES_HEADER = r"^### Deprecation ledger \(flips in 1\.0\)$"
SPEC_HEADER = r"^## 4\. Deprecation ledger \(everything that flips in 1\.0\)$"


def _table_after(text: str, header_pattern: str) -> list[str]:
    match = re.search(header_pattern, text, re.MULTILINE)
    assert match is not None, f"header not found: {header_pattern}"
    rows: list[str] = []
    for line in text[match.end():].splitlines():
        if line.startswith("|"):
            rows.append(line.rstrip())
        elif rows:
            break
    return rows


def test_release_notes_reproduce_the_deprecation_ledger_verbatim() -> None:
    notes = (ROOT / "docs" / "release-notes.md").read_text(encoding="utf8")
    spec = (ROOT / "docs" / "specs" / "0.9-product-layer.md").read_text(encoding="utf8")
    notes_rows = _table_after(notes, NOTES_HEADER)
    spec_rows = _table_after(spec, SPEC_HEADER)
    assert len(spec_rows) >= 10, "ledger table unexpectedly short"
    assert notes_rows == spec_rows


def test_struck_alias_flips_are_recorded_as_permanent() -> None:
    spec = (ROOT / "docs" / "specs" / "0.9-product-layer.md").read_text(encoding="utf8")
    rows = _table_after(spec, SPEC_HEADER)
    for needle in ("`group_col`/`sample_weight_col`", "stability `alpha`"):
        row = next(r for r in rows if needle in r)
        assert "permanent alias" in row, row
