"""Warning metadata and caller-location contracts."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sift.sampling.smart as smart_module
import sift.selection.knockoff_filter as knockoff_module


def _warning_calls(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "warnings"
            and node.func.attr == "warn"
        ):
            yield node


def test_all_package_warning_calls_declare_category_and_stacklevel():
    package_root = Path(__file__).parents[1] / "sift"
    missing = []
    for path in sorted(package_root.rglob("*.py")):
        for call in _warning_calls(path):
            keywords = {keyword.arg for keyword in call.keywords}
            has_category = len(call.args) >= 2 or "category" in keywords
            has_stacklevel = "stacklevel" in keywords
            if not (has_category and has_stacklevel):
                missing.append(f"{path.relative_to(package_root)}:{call.lineno}")

    assert missing == []


def test_smart_sample_runtime_warning_points_to_public_caller(monkeypatch):
    def fail_svd(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("forced SVD failure")

    monkeypatch.setattr(smart_module, "randomized_svd", fail_svd)
    frame = pd.DataFrame(
        {
            "x": np.linspace(-1.0, 1.0, 24),
            "y": np.linspace(0.0, 2.0, 24),
        }
    )
    config = smart_module.SmartSamplerConfig(
        sample_frac=0.5,
        residual_weight_cap=0.0,
        random_state=0,
        verbose=False,
    )

    with pytest.warns(RuntimeWarning, match="forced SVD failure") as caught:
        smart_module.smart_sample(frame, ["x"], "y", config=config)

    assert Path(caught[0].filename) == Path(__file__)


def test_select_fdr_warning_reports_effective_path_depth(monkeypatch):
    def saturated_scores(G, r, *, path_depth, **kwargs):
        del r, path_depth, kwargs
        n_pairs = G.shape[0] // 2
        return np.r_[np.arange(n_pairs, 0, -1), np.zeros(n_pairs)]

    monkeypatch.setattr(
        knockoff_module,
        "_cefsplus_incremental_scores",
        saturated_scores,
    )
    rng = np.random.default_rng(7)
    X = rng.normal(size=(120, 6))
    y = 2.0 * X[:, 0] - X[:, 1] + rng.normal(scale=0.5, size=120)

    with pytest.warns(UserWarning, match=r"effective path_depth=3") as caught:
        result = knockoff_module.select_fdr(
            X,
            y,
            q=0.5,
            offset=0,
            statistic="cefsplus",
            statistic_options={"path_depth": 3},
            random_state=0,
            verbose=False,
        )

    matching = [
        warning
        for warning in caught
        if "effective path_depth=3" in str(warning.message)
    ]
    assert len(matching) == 1
    assert matching[0].category is UserWarning
    assert Path(matching[0].filename) == Path(__file__)
    assert result.selector_metadata["path_depth"] == 3
    assert result.selector_metadata["path_depth_saturated"] is True
