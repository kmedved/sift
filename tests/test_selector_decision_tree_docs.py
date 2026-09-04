"""The 0.9.1 selector guide is the one canonical choice map."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "choosing-a-selector.md"


def test_selector_decision_tree_covers_each_public_workflow_leaf() -> None:
    text = GUIDE.read_text(encoding="utf-8")

    assert text.count("```mermaid") == 1
    diagram = text.split("```mermaid\n", 1)[1].split("\n```", 1)[0]
    assert diagram.startswith("flowchart TD\n")
    for node in (
        'Sample["smart_sample"]',
        'Permutation["permutation_importance"]',
        'Knockoff["select_fdr or KnockoffSelector"]',
        'Boruta["select_boruta or BorutaSelector"]',
        'Stability["StabilitySelector"]',
        'CatBoost["catboost_select"]',
        'MRMR["select_mrmr"]',
        'JMI["select_jmi"]',
        'JMIM["select_jmim"]',
        'CEFS["select_cefsplus"]',
        'Binary["select_cefsplus_binary"]',
    ):
        assert node in diagram


def test_old_selector_choice_tables_are_replaced_by_links() -> None:
    expected_links = {
        "README.md": "docs/choosing-a-selector.md",
        "DOCS.MD": "docs/choosing-a-selector.md",
        "docs/user-guide.md": "choosing-a-selector.md",
        "docs/ALGORITHMS.md": "choosing-a-selector.md",
    }
    retired_headers = (
        "| Goal | Start with |",
        "| Feature | Main entry points | Best for |",
        "| Need | Start with |",
        "| Method | Type | Output contract | Best for |",
        "| Scenario | Start with |",
    )

    for relpath, link in expected_links.items():
        text = (ROOT / relpath).read_text(encoding="utf-8")
        assert link in text
        for header in retired_headers:
            assert header not in text
