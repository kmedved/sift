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


def test_ledger_0_9_column_records_the_58_exports_v0_9_0_shipped() -> None:
    """The 0.9 column is history: v0.9.0 exported 58 names, 0.10.0 made it 66."""
    spec = (ROOT / "docs" / "specs" / "0.9-product-layer.md").read_text(encoding="utf8")
    row = next(r for r in _table_after(spec, SPEC_HEADER) if r.startswith("| `sift.__all__` |"))
    _, item, state_0_9, state_1_0, _ = row.split("|")
    assert state_0_9.strip() == "58 exports (0.10.0 added 8, for 66)"
    assert "retain all 66 exports" in state_1_0


def test_1_0_restore_table_names_real_parameters_whose_defaults_changed() -> None:
    """Each explicit 0.10 setting in the 1.0.0 notes exists and differs from 1.0."""
    import ast
    import inspect

    import sift

    notes = (ROOT / "docs" / "release-notes.md").read_text(encoding="utf8")
    rows = _table_after(notes, r"^To keep 0\.10 behavior, pass the old values explicitly\.")
    assert rows[0] == "| entry point | explicit 0.10 settings |"
    restored: dict[str, dict[str, object]] = {}
    for row in rows[2:]:
        names_cell, settings_cell = row.strip("|").split("|")
        settings_span = re.search(r"`([^`]+)`", settings_cell).group(1)
        call = ast.parse(f"f({settings_span})", mode="eval").body
        settings = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
        for name in re.findall(r"`(\w+)`", names_cell):
            assert name not in restored, name
            restored[name] = settings

    for name, settings in restored.items():
        parameters = inspect.signature(getattr(sift, name)).parameters
        for parameter, old_value in settings.items():
            assert parameter in parameters, (name, parameter)
            assert parameters[parameter].default != old_value, (name, parameter)
    assert restored["StabilitySelector"] == {
        "output_order": "legacy",
        "verbose": True,
        "n_jobs": -1,
        "random_state": None,
    }
    assert len(restored) == 21
