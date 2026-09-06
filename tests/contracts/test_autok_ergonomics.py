"""Public-contract checks for the additive Workstream-C ergonomics."""

from __future__ import annotations

import dataclasses
import inspect
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

import sift
import sift.experimental as experimental
import sift.selection.filter_auto_k as filter_auto_k
from sift.selection.auto_k import AutoKConfig, validate_auto_k_config
from sift.selection.auto_k_options import (
    AutoKCVOptions,
    AutoKExperimentalOptions,
    AutoKKnockoffOptions,
    AutoKObjectiveOptions,
    AutoKPermutationOptions,
    AutoKStabilityOptions,
    AutoKTestOptions,
)


AUTOK_FIELDS = (
    "k_method",
    "strategy",
    "metric",
    "max_k",
    "min_k",
    "val_frac",
    "n_splits",
    "random_state",
    "elbow_min_rel_gain",
    "elbow_patience",
    "auto_k_mode",
    "selection_rule",
    "one_se_multiplier",
    "score_abs_tol",
    "score_rel_tol",
    "plateau_prefer",
    "plateau_min_points",
    "objective_penalty",
    "objective_penalty_weight",
    "objective_n_eff",
    "binary_objective_mode",
    "n_eff_mode",
    "alpha",
    "m_mode",
    "stop_patience",
    "perm_B",
    "perm_null",
    "gap_rule",
    "knockoff_q",
    "knockoff_draws",
    "knockoff_s_method",
    "knockoff_return",
    "xfit_folds",
    "xfit_mode",
    "xfit_ridge",
    "ebic_gamma",
    "posterior_level",
    "posterior_pick",
    "boot_B",
    "boot_mode",
    "stability_rule",
    "stability_pi",
    "floor_z",
    "floor_window",
    "consensus_methods",
    "auto_dense_check",
    "auto_dense_min_k",
    "auto_dense_min_frac",
    "auto_dense_disagreement_ratio",
)

EXPERIMENTAL_NAMES = [
    "FeaturePathEvaluationResult",
    "select_k_posterior",
    "path_gain_pvalues",
    "select_k_changepoint",
    "select_k_chi2_stop",
    "select_k_forward_stop",
    "bootstrap_paths",
    "null_objective_paths",
    "select_k_perm_gap",
    "select_k_stability",
    "gaussian_cv_curves",
    "select_k_gaussian_cv",
    "select_k_xfit_objective",
    "select_k_knockoff_path",
    "xfit_objective_curves",
    "compute_objective_for_path",
]


def test_autok_flat_dataclass_contract_is_unchanged():
    assert tuple(field.name for field in dataclasses.fields(AutoKConfig)) == AUTOK_FIELDS
    assert tuple(inspect.signature(AutoKConfig).parameters) == AUTOK_FIELDS
    config = AutoKConfig()
    assert tuple(dataclasses.asdict(config)) == AUTOK_FIELDS
    assert "objective=" not in repr(config)
    assert "experimental=" not in repr(config)
    assert dataclasses.replace(config, alpha=0.2).alpha == 0.2
    assert pickle.loads(pickle.dumps(config)) == config


def test_autok_presets_are_exact_flat_configs():
    assert AutoKConfig.default() == AutoKConfig(k_method="auto")
    assert AutoKConfig.predictive() == AutoKConfig(
        k_method="gaussian_cv",
        strategy="kfold",
        selection_rule="best",
        xfit_folds=5,
    )
    assert AutoKConfig.predictive(
        strategy="group_cv", rule="one_se", n_folds=7
    ) == AutoKConfig(
        k_method="gaussian_cv",
        strategy="group_cv",
        selection_rule="one_se",
        xfit_folds=7,
    )
    assert AutoKConfig.discovery(alpha=0.1) == AutoKConfig(
        k_method="chi2_stop",
        min_k=0,
        alpha=0.1,
    )
    assert AutoKConfig.downstream("group_cv", "rmse", "best") == AutoKConfig(
        k_method="evaluate",
        strategy="group_cv",
        metric="rmse",
        selection_rule="best",
    )
    with pytest.raises(ValueError, match="kfold.*gaussian_cv"):
        AutoKConfig.downstream("kfold", "rmse", "best")


def test_group_builder_flattens_all_groups_without_duplicate_storage():
    objective = AutoKObjectiveOptions(
        objective_penalty="custom",
        objective_penalty_weight=3.0,
        objective_n_eff=120.0,
        n_eff_mode="kish",
        ebic_gamma=0.4,
        binary_objective_mode="score_test",
    )
    test = AutoKTestOptions(alpha=0.1, m_mode="panel", stop_patience=4)
    perm = AutoKPermutationOptions(
        perm_B=7,
        perm_null="permute",
        gap_rule="gain_envelope",
    )
    knockoff = AutoKKnockoffOptions(
        knockoff_q=0.1,
        knockoff_draws=2,
        knockoff_s_method="mvr",
        knockoff_return="prefix",
    )
    cv = AutoKCVOptions(
        strategy="kfold",
        metric="rmse",
        val_frac=0.3,
        n_splits=3,
        xfit_folds=7,
        xfit_mode="shared_z",
        xfit_ridge=0.01,
        selection_rule="plateau",
        one_se_multiplier=1.5,
        score_abs_tol=0.1,
        score_rel_tol=0.05,
        plateau_prefer="center",
        plateau_min_points=3,
    )
    stability = AutoKStabilityOptions(
        boot_B=6,
        boot_mode="half",
        stability_rule="pi_threshold",
        stability_pi=0.7,
    )
    experimental_options = AutoKExperimentalOptions(
        elbow_min_rel_gain=0.03,
        elbow_patience=4,
        posterior_level=0.8,
        posterior_pick="smallest_in_hpd",
        floor_z=3.0,
        floor_window=5,
        consensus_methods=("chi2_stop",),
        auto_dense_check=True,
        auto_dense_min_k=10,
        auto_dense_min_frac=0.2,
        auto_dense_disagreement_ratio=3.0,
    )
    config = AutoKConfig.from_groups(
        k_method="auto",
        max_k=50,
        min_k=0,
        random_state=7,
        objective=objective,
        test=test,
        perm=perm,
        knockoff=knockoff,
        cv=cv,
        stability=stability,
        experimental=experimental_options,
    )
    expected_values = {
        "k_method": "auto",
        "max_k": 50,
        "min_k": 0,
        "random_state": 7,
    }
    for group in (
        objective,
        test,
        perm,
        knockoff,
        cv,
        stability,
        experimental_options,
    ):
        expected_values.update(dataclasses.asdict(group))
    assert config == AutoKConfig(**expected_values)
    assert config.objective == objective
    assert config.test == test
    assert config.perm == perm
    assert config.knockoff == knockoff
    assert config.cv == cv
    assert config.stability == stability
    assert config.experimental == experimental_options
    assert pickle.loads(pickle.dumps(config)) == config


def test_group_views_are_frozen_snapshots_and_builder_rejects_ambiguity():
    config = AutoKConfig(alpha=0.1)
    view = config.test
    with pytest.raises(dataclasses.FrozenInstanceError):
        view.alpha = 0.2
    config.alpha = 0.3
    assert view.alpha == 0.1
    assert config.test.alpha == 0.3

    with pytest.raises(ValueError, match="both.*perm"):
        AutoKConfig.from_groups(
            k_method="perm_gap",
            perm=AutoKPermutationOptions(perm_B=7),
            perm_B=8,
        )
    with pytest.raises(TypeError, match="Unknown AutoKConfig field"):
        AutoKConfig.from_groups(k_method="auto", unknown_option=True)
    with pytest.raises(TypeError, match="objective must be AutoKObjectiveOptions"):
        AutoKConfig.from_groups(k_method="auto", objective=AutoKTestOptions())
    with pytest.raises(TypeError, match="unexpected keyword argument 'objective'"):
        AutoKConfig(objective=AutoKObjectiveOptions())


@pytest.mark.parametrize(
    ("config", "field_name"),
    [
        (AutoKConfig(k_method="evaluate", elbow_patience=4), "elbow_patience"),
        (AutoKConfig(k_method="elbow", selection_rule="one_se"), "selection_rule"),
        (AutoKConfig(k_method="elbow", objective_penalty="aic"), "objective_penalty"),
        (AutoKConfig(k_method="elbow", objective_n_eff=80.0), "objective_n_eff"),
        (AutoKConfig(k_method="elbow", n_eff_mode="kish"), "n_eff_mode"),
        (
            AutoKConfig(k_method="elbow", binary_objective_mode="score_test"),
            "binary_objective_mode",
        ),
        (AutoKConfig(k_method="perm_gap", alpha=0.1), "alpha"),
        (
            AutoKConfig(
                k_method="stability",
                stability_rule="max_one_se",
                stability_pi=0.7,
            ),
            "stability_pi",
        ),
        (
            AutoKConfig(
                k_method="penalized_objective",
                objective_penalty="bic",
                ebic_gamma=0.4,
            ),
            "ebic_gamma",
        ),
        (
            AutoKConfig(k_method="evaluate", score_abs_tol=0.1),
            "score_abs_tol",
        ),
        (
            AutoKConfig(
                k_method="gaussian_cv",
                strategy="time_holdout",
                random_state=7,
            ),
            "random_state",
        ),
    ],
)
def test_unused_nondefault_method_fields_warn(config, field_name):
    with pytest.warns(UserWarning, match=rf"AutoKConfig\.{field_name}.*does not use"):
        validate_auto_k_config(config)


@pytest.mark.parametrize(
    "config",
    [
        AutoKConfig(k_method="elbow", elbow_patience=4),
        AutoKConfig(
            k_method="evaluate",
            selection_rule="one_se",
            one_se_multiplier=1.5,
        ),
        AutoKConfig(
            k_method="evaluate",
            selection_rule="plateau",
            score_abs_tol=0.1,
            plateau_prefer="center",
        ),
        AutoKConfig(
            k_method="penalized_objective",
            objective_penalty="ebic",
            ebic_gamma=0.4,
        ),
        AutoKConfig(
            k_method="perm_gap",
            gap_rule="gain_envelope",
            alpha=0.1,
            stop_patience=4,
        ),
        AutoKConfig(
            k_method="stability",
            stability_rule="pi_threshold",
            stability_pi=0.7,
        ),
        AutoKConfig(
            k_method="gaussian_cv",
            strategy="kfold",
            random_state=7,
        ),
    ],
)
def test_nondefault_fields_used_by_selected_method_do_not_warn(config):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_auto_k_config(config)
    assert [warning for warning in caught if "does not use it" in str(warning.message)] == []


def test_auto_router_does_not_validate_routed_copy_as_user_config(monkeypatch):
    class DummyCache:
        sample_weight = np.ones(20, dtype=np.float64)
        valid_cols = np.arange(5, dtype=np.int64)

    def routed_runner(config, **_kwargs):
        validate_auto_k_config(config)
        return (
            ["x0"],
            np.array([0], dtype=np.int64),
            pd.DataFrame(),
            {
                "selected_k": 1,
                "max_k": 5,
                "effective_max_k": 5,
                "path_length": 5,
                "stopped_by": "criterion",
            },
        )

    monkeypatch.setattr(filter_auto_k, "_run_gaussian_routed_path", routed_runner)
    monkeypatch.setattr(filter_auto_k, "_run_auto_dense_check", lambda **_kwargs: None)
    config = AutoKConfig(k_method="auto", elbow_min_rel_gain=0.3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        filter_auto_k.select_gaussian_auto_path(
            cache=DummyCache(),
            y=np.arange(20.0),
            method="cefsplus",
            max_k=5,
            top_m=5,
            auto_k_config=config,
            verbose=False,
        )
    assert [warning for warning in caught if "does not use it" in str(warning.message)] == []


def test_experimental_namespace_warns_without_changing_top_level_surface():
    assert experimental.__all__ == EXPERIMENTAL_NAMES
    original_all = list(sift.__all__)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for name in EXPERIMENTAL_NAMES:
            assert getattr(experimental, name) is getattr(sift, name)
    assert len(caught) == len(EXPERIMENTAL_NAMES)
    assert all(warning.category is FutureWarning for warning in caught)
    assert sift.__all__ == original_all
    assert len(sift.__all__) == 64
    assert "experimental" not in sift.__all__
    assert not hasattr(sift, "AutoKCVOptions")


def test_experimental_import_forms_and_missing_name_contract():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        namespace: dict[str, object] = {}
        exec("from sift.experimental import select_k_posterior", namespace)
    assert namespace["select_k_posterior"] is sift.select_k_posterior
    assert len(caught) == 1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        namespace = {}
        exec("from sift.experimental import *", namespace)
    assert {name for name in namespace if not name.startswith("__")} == set(
        EXPERIMENTAL_NAMES
    )
    assert len(caught) == len(EXPERIMENTAL_NAMES)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(AttributeError, match="missing_name"):
            getattr(experimental, "missing_name")
    assert caught == []

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exec("from sift import *", {})
    assert caught == []

    with pytest.warns(FutureWarning):
        result_type = experimental.FeaturePathEvaluationResult
    assert pickle.loads(pickle.dumps(result_type)) is sift.FeaturePathEvaluationResult
