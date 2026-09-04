"""Facade aliases and contracts for extracted AutoKConfig/validation."""

from __future__ import annotations

import ast
import copy
import dataclasses
import inspect
import pickle
import typing
import warnings
from pathlib import Path

import pytest

import sift
from sift.api import AutoKConfig as api_config
from sift.selection import auto_k as auto_k_module, auto_k_config, filter_auto_k
from sift.selection import AutoKConfig as selection_config
from sift.selection.auto_k import (
    AutoKConfig,
    resolve_auto_k_config,
    validate_auto_k_config,
    with_effective_k_bounds,
)


REEXPORTED_NAMES = (
    "AutoKConfig",
    "_NONNEGATIVE_INT_FIELDS",
    "_POSITIVE_INT_FIELDS",
    "_REAL_TYPES",
    "_VALID_BINARY_OBJECTIVE_MODES",
    "_VALID_BOOT_MODES",
    "_VALID_CONSENSUS_METHODS",
    "_VALID_GAP_RULES",
    "_VALID_K_METHODS",
    "_VALID_KNOCKOFF_RETURNS",
    "_VALID_KNOCKOFF_S_METHODS",
    "_VALID_M_MODES",
    "_VALID_N_EFF_MODES",
    "_VALID_OBJECTIVE_PENALTIES",
    "_VALID_PERM_NULLS",
    "_VALID_PLATEAU_PREFERS",
    "_VALID_POSTERIOR_PICKS",
    "_VALID_SELECTION_RULES",
    "_VALID_STABILITY_RULES",
    "_VALID_STRATEGIES",
    "_VALID_XFIT_MODES",
    "_WARN_UNUSED_METHOD_FIELDS",
    "_auto_k_method_tags",
    "_ensure_supported_auto_k_mode",
    "_is_real_number",
    "_suppress_auto_k_unused_field_warnings",
    "_warn_unused_method_fields",
    "resolve_auto_k_config",
    "validate_auto_k_config",
    "with_effective_k_bounds",
)


def test_config_cluster_names_are_facade_aliases() -> None:
    assert len(REEXPORTED_NAMES) == 30
    for name in REEXPORTED_NAMES:
        assert getattr(auto_k_module, name) is getattr(auto_k_config, name)
    assert AutoKConfig is auto_k_config.AutoKConfig
    assert AutoKConfig is auto_k_module.AutoKConfig
    assert AutoKConfig is selection_config
    assert AutoKConfig is api_config
    assert AutoKConfig is sift.AutoKConfig
    assert AutoKConfig is filter_auto_k.AutoKConfig
    assert AutoKConfig.__module__ == "sift.selection.auto_k"
    assert validate_auto_k_config.__module__ == "sift.selection.auto_k"
    assert resolve_auto_k_config.__module__ == "sift.selection.auto_k"
    assert with_effective_k_bounds.__module__ == "sift.selection.auto_k"
    assert auto_k_config._is_real_number.__module__ == "sift.selection.auto_k_config"


def test_autokconfig_pickle_hints_and_dataclass_contract() -> None:
    payload = pickle.dumps(AutoKConfig())
    assert b"auto_k_config" not in payload
    assert b"sift.selection.auto_k" in payload
    restored = pickle.loads(payload)
    assert restored == AutoKConfig()
    assert pickle.loads(pickle.dumps(validate_auto_k_config)) is validate_auto_k_config

    assert len(dataclasses.fields(AutoKConfig)) == 49
    assert len(inspect.signature(AutoKConfig).parameters) == 49
    assert len(typing.get_type_hints(AutoKConfig)) == 49
    config = AutoKConfig(alpha=0.2)
    assert "objective=" not in repr(config)
    assert dataclasses.replace(config, min_k=1).min_k == 1
    assert copy.copy(config) == config
    assert copy.deepcopy(config) == config
    assert dataclasses.asdict(config)["alpha"] == 0.2


def test_default_config_is_not_on_facade_and_lazily_rebinds() -> None:
    assert not hasattr(auto_k_module, "_DEFAULT_AUTOK_CONFIG")
    original = auto_k_config._DEFAULT_AUTOK_CONFIG
    try:
        auto_k_config._DEFAULT_AUTOK_CONFIG = None
        with pytest.warns(UserWarning, match="does not use it"):
            validate_auto_k_config(AutoKConfig(k_method="elbow", alpha=0.2))
        assert isinstance(auto_k_config._DEFAULT_AUTOK_CONFIG, AutoKConfig)
    finally:
        auto_k_config._DEFAULT_AUTOK_CONFIG = original


def test_unused_field_warning_suppression_and_caller_facing_location() -> None:
    def _trigger() -> None:
        validate_auto_k_config(AutoKConfig(k_method="elbow", alpha=0.2))

    with pytest.warns(UserWarning, match="does not use it") as caught:
        _trigger()
    assert Path(caught[0].filename).name == Path(__file__).name

    with auto_k_module._suppress_auto_k_unused_field_warnings():
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            _trigger()
        assert [w for w in recorded if "does not use it" in str(w.message)] == []
        with auto_k_module._suppress_auto_k_unused_field_warnings():
            with warnings.catch_warnings(record=True) as nested:
                warnings.simplefilter("always")
                _trigger()
            assert [w for w in nested if "does not use it" in str(w.message)] == []

    with pytest.warns(UserWarning, match="does not use it"):
        _trigger()


def test_shared_contextvar_and_validation_seams() -> None:
    assert (
        auto_k_module._WARN_UNUSED_METHOD_FIELDS
        is auto_k_config._WARN_UNUSED_METHOD_FIELDS
    )
    assert (
        auto_k_module.select_k_auto.__globals__["validate_auto_k_config"]
        is auto_k_config.validate_auto_k_config
    )
    assert resolve_auto_k_config.__globals__ is vars(auto_k_config)


def test_accepted_class_getsource_loss_and_method_source() -> None:
    with pytest.raises(OSError):
        inspect.getsource(AutoKConfig)
    assert "measured automatic router preset" in inspect.getsource(AutoKConfig.default)


def test_config_sibling_is_a_leaf() -> None:
    tree = ast.parse(Path(auto_k_config.__file__).read_text())
    runtime_imported = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            runtime_imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            runtime_imported.append(node.module)
    runtime_imported = [name for name in runtime_imported if name != "__future__"]
    assert "sift.selection.auto_k" not in runtime_imported
    assert all(not name.startswith("sift.selection.auto_k.") for name in runtime_imported)
    assert set(runtime_imported) == {
        "contextlib",
        "contextvars",
        "dataclasses",
        "typing",
        "numpy",
        "sift._deprecate",
        "sift.selection.auto_k_options",
    }
