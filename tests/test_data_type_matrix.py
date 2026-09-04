"""Contracts for the executable public data-type support matrix."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_data_type_matrix.py"
DOC = ROOT / "docs" / "data-type-support.md"

REQUIRED_ENTRIES = (
    "select_mrmr",
    "select_jmi",
    "select_jmim",
    "select_cefsplus",
    "select_cefsplus_binary",
    "MRMRSelector",
    "JMISelector",
    "JMIMSelector",
    "CEFSPlusSelector",
    "CEFSPlusBinarySelector",
    "select_fdr",
    "KnockoffSelector",
    "select_cached",
    "select_boruta",
    "BorutaSelector",
    "select_boruta_shap",
    "StabilitySelector",
    "stability_regression",
    "stability_classif",
    "permutation_importance",
    "smart_sample",
    "catboost_select",
    "catboost_regression",
    "catboost_classif",
)

REQUIRED_AXES = (
    "numeric_ndarray",
    "numeric_dataframe",
    "categorical",
    "sparse",
    "datetime_timedelta",
    "sample_weight",
    "groups",
    "time",
)


def _generator():
    spec = importlib.util.spec_from_file_location("generate_data_type_matrix", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def matrix_mod():
    return _generator()


@pytest.fixture(scope="module")
def published_cells(matrix_mod):
    return matrix_mod.probe_published()


def test_committed_page_matches_live_published_probe(matrix_mod, published_cells) -> None:
    assert DOC.read_text(encoding="utf-8") == matrix_mod.render_page(published_cells)


def test_matrix_covers_required_entries_and_axes(published_cells) -> None:
    names = [cell.entry for cell in published_cells]
    axes = [cell.axis for cell in published_cells]
    assert names[:: len(REQUIRED_AXES)] == list(REQUIRED_ENTRIES)
    assert set(axes) == set(REQUIRED_AXES)
    assert len(published_cells) == len(REQUIRED_ENTRIES) * len(REQUIRED_AXES)


def test_published_cells_use_the_four_documented_statuses(
    matrix_mod, published_cells
) -> None:
    allowed = {
        matrix_mod.SUPPORTED,
        matrix_mod.REJECTED,
        matrix_mod.CONDITIONAL,
        matrix_mod.DEPENDENCY_GATED,
    }
    assert {cell.status for cell in published_cells} <= allowed
    assert matrix_mod.CONDITIONAL in {cell.status for cell in published_cells}
    assert matrix_mod.DEPENDENCY_GATED in {cell.status for cell in published_cells}


def test_filter_groups_are_conditional_not_silently_supported(
    matrix_mod, published_cells
) -> None:
    grid = {(cell.entry, cell.axis): cell for cell in published_cells}
    for name in (
        "select_mrmr",
        "select_jmi",
        "select_jmim",
        "select_cefsplus",
        "select_cefsplus_binary",
        "MRMRSelector",
        "JMISelector",
        "JMIMSelector",
        "CEFSPlusSelector",
        "CEFSPlusBinarySelector",
    ):
        assert grid[(name, "groups")].status == matrix_mod.CONDITIONAL
        assert grid[(name, "time")].status == matrix_mod.CONDITIONAL
        assert grid[(name, "categorical")].status == matrix_mod.CONDITIONAL
        assert grid[(name, "sparse")].status == matrix_mod.REJECTED
        assert grid[(name, "datetime_timedelta")].status == matrix_mod.REJECTED


def test_knockoff_and_cached_paths_reject_row_context(
    matrix_mod, published_cells
) -> None:
    grid = {(cell.entry, cell.axis): cell for cell in published_cells}
    for name in ("select_fdr", "KnockoffSelector", "select_cached"):
        assert grid[(name, "groups")].status == matrix_mod.REJECTED
        assert grid[(name, "time")].status == matrix_mod.REJECTED
        assert grid[(name, "numeric_ndarray")].status == matrix_mod.SUPPORTED
        assert grid[(name, "numeric_dataframe")].status == matrix_mod.SUPPORTED
    assert grid[("select_fdr", "categorical")].status == matrix_mod.REJECTED
    assert grid[("select_cached", "categorical")].status == matrix_mod.REJECTED


def test_select_cached_weights_use_the_cache_workflow(
    matrix_mod, published_cells
) -> None:
    grid = {(cell.entry, cell.axis): cell for cell in published_cells}
    cell = grid[("select_cached", "sample_weight")]
    assert cell.status == matrix_mod.SUPPORTED
    assert cell.note == (
        "weights belong on `build_cache(X, sample_weight=w, subsample=None)`; "
        "`select_cached` has no call-time `sample_weight` because the cache "
        "already stores row weights"
    )


def test_knockoff_legacy_loo_logit_is_conditional(
    matrix_mod, published_cells
) -> None:
    grid = {(cell.entry, cell.axis): cell for cell in published_cells}
    cell = grid[("KnockoffSelector", "categorical")]
    assert cell.status == matrix_mod.CONDITIONAL
    assert "loo_logit" in cell.note
    assert "fdr_control='none'" in cell.note
    assert cell.enabled is not None and cell.enabled.ok
    assert any("no FDR claim applies" in warning for warning in cell.enabled.warnings)


def test_permutation_importance_reaches_sift_for_mixed_dtypes(
    matrix_mod, published_cells
) -> None:
    grid = {(cell.entry, cell.axis): cell for cell in published_cells}
    categorical = grid[("permutation_importance", "categorical")]
    temporal = grid[("permutation_importance", "datetime_timedelta")]
    sparse = grid[("permutation_importance", "sparse")]
    assert categorical.status == matrix_mod.SUPPORTED
    assert temporal.status == matrix_mod.SUPPORTED
    assert "model-dependent" in categorical.note
    assert "model-dependent" in temporal.note
    assert sparse.status == matrix_mod.REJECTED
    assert "independently" in sparse.note
    assert categorical.default.ok
    assert temporal.default.ok
    assert not sparse.default.ok


def test_published_notes_are_version_portable(published_cells) -> None:
    forbidden = (
        "DTypePromotionError",
        "setting an array element with a sequence",
        "numpy.dtypes",
        "could not be promoted",
        "ValueError:",
        "TypeError:",
        "AttributeError:",
        "ImportError:",
    )
    for cell in published_cells:
        for token in forbidden:
            assert token not in cell.note, (cell.entry, cell.axis, cell.note)


def test_catboost_family_publishes_the_post_install_contract(
    matrix_mod, published_cells
) -> None:
    grid = {(cell.entry, cell.axis): cell for cell in published_cells}
    for name in ("catboost_select", "catboost_regression", "catboost_classif"):
        for axis, status in matrix_mod.CATBOOST_PUBLISHED_STATUS.items():
            cell = grid[(name, axis)]
            assert cell.status == status, (name, axis, cell.status, cell.note)
            if axis == "datetime_timedelta":
                assert cell.status == matrix_mod.REJECTED
                assert "convert them to numeric features explicitly" in cell.note
                assert "ImportError" not in cell.note
            if status == matrix_mod.DEPENDENCY_GATED:
                assert "unlocks this axis" in cell.note


def test_published_catboost_cells_bind_live_missing_extra_attempts(
    matrix_mod, published_cells
) -> None:
    try:
        import catboost  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("CatBoost is present; missing-extra attempts are the base-env half")

    grid = {(cell.entry, cell.axis): cell for cell in published_cells}
    for name in ("catboost_select", "catboost_regression", "catboost_classif"):
        for axis in REQUIRED_AXES:
            cell = grid[(name, axis)]
            assert not cell.default.ok, (name, axis, cell.default.summary)
            assert matrix_mod._is_missing_dependency(cell.default, "catboost"), (
                name,
                axis,
                cell.default.error,
            )


def test_smart_sample_datetime_coercion_is_recorded(
    matrix_mod, published_cells
) -> None:
    grid = {(cell.entry, cell.axis): cell for cell in published_cells}
    cell = grid[("smart_sample", "datetime_timedelta")]
    assert cell.status == matrix_mod.SUPPORTED
    assert "float32" in cell.note
    assert "lossy" in cell.note
    assert "explicitly" in cell.note


def test_smart_sample_groups_and_time_use_config_columns(
    matrix_mod, published_cells
) -> None:
    grid = {(cell.entry, cell.axis): cell for cell in published_cells}
    groups = grid[("smart_sample", "groups")]
    time = grid[("smart_sample", "time")]
    assert groups.status == matrix_mod.SUPPORTED
    assert time.status == matrix_mod.SUPPORTED
    assert "group_col" in groups.note
    assert "groups=" in groups.note
    assert "time_col" in time.note
    assert "time=" in time.note


def test_boruta_shap_published_note_names_shap(
    matrix_mod, published_cells
) -> None:
    grid = {(cell.entry, cell.axis): cell for cell in published_cells}
    numeric = grid[("select_boruta_shap", "numeric_dataframe")]
    categorical = grid[("select_boruta_shap", "categorical")]
    assert numeric.status == matrix_mod.DEPENDENCY_GATED
    assert "shap" in numeric.note
    assert "CatBoost" in numeric.note or "catboost" in numeric.note.lower()
    assert categorical.status == matrix_mod.DEPENDENCY_GATED
    assert "target_cv" in categorical.note
    assert "shap" in categorical.note


@pytest.mark.catboost
@pytest.mark.parametrize(
    "name",
    ("catboost_select", "catboost_regression", "catboost_classif"),
)
def test_catboost_entry_inner_axes_when_extra_is_present(matrix_mod, name) -> None:
    pytest.importorskip("catboost")
    by_name = {entry.name: entry for entry in matrix_mod.entries()}
    entry = by_name[name]
    expected = {
        "numeric_ndarray": matrix_mod.REJECTED,
        "numeric_dataframe": matrix_mod.SUPPORTED,
        "categorical": matrix_mod.SUPPORTED,
        "sparse": matrix_mod.REJECTED,
        "datetime_timedelta": matrix_mod.REJECTED,
        "sample_weight": matrix_mod.SUPPORTED,
        "groups": matrix_mod.SUPPORTED,
        "time": matrix_mod.SUPPORTED,
    }
    for axis, status in expected.items():
        cell = matrix_mod.classify_cell(entry, axis)
        assert cell.status == status, (name, axis, cell.status, cell.note)


def test_boruta_shap_explicit_estimator_still_needs_shap(matrix_mod) -> None:
    try:
        import catboost  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("CatBoost is present; this pin is the no-CatBoost shap gate")

    by_name = {entry.name: entry for entry in matrix_mod.entries()}
    entry = by_name["select_boruta_shap"]
    cell = matrix_mod.classify_cell(entry, "numeric_dataframe")
    assert cell.status == matrix_mod.DEPENDENCY_GATED
    assert "shap" in cell.note
    assert cell.enabled is not None
    try:
        import shap  # noqa: F401
    except ImportError:
        assert matrix_mod._is_missing_dependency(cell.enabled, "shap")
    else:
        assert cell.enabled.ok


def test_boruta_shap_categorical_retry_keeps_target_cv(matrix_mod) -> None:
    try:
        import catboost  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("CatBoost is present; this pin is the no-CatBoost shap gate")

    by_name = {entry.name: entry for entry in matrix_mod.entries()}
    entry = by_name["select_boruta_shap"]
    cell = matrix_mod.classify_cell(entry, "categorical")
    assert cell.status == matrix_mod.DEPENDENCY_GATED
    assert "target_cv" in cell.note
    assert "shap" in cell.note
    assert cell.enabled is not None
    error = cell.enabled.error or ""
    assert "Non-numeric columns" not in error
    assert "could not convert string" not in error
    try:
        import shap  # noqa: F401
    except ImportError:
        assert matrix_mod._is_missing_dependency(cell.enabled, "shap")
    else:
        assert cell.enabled.ok


def test_shap_dependency_does_not_match_importance_substring(matrix_mod) -> None:
    importance = matrix_mod.Attempt(
        ok=False,
        error="ValueError: importance='shap' requires catboost or an explicit estimator",
    )
    shap_package = matrix_mod.Attempt(
        ok=False,
        error="ImportError: SHAP backend requires either catboost or shap package",
    )
    assert matrix_mod._is_missing_dependency(importance, "catboost")
    assert not matrix_mod._is_missing_dependency(importance, "shap")
    assert matrix_mod._is_missing_dependency(shap_package, "shap")


def test_stable_extra_notes_cover_only_boruta_shap(matrix_mod) -> None:
    assert set(matrix_mod.STABLE_EXTRA_NOTES) == {"select_boruta_shap"}
