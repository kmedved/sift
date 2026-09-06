"""Feature selection algorithms."""

from sift.selection.auto_k import (
    AutoKConfig,
    compute_objective_for_path,
    select_k_auto,
    select_k_elbow,
    select_k_penalized_objective,
    select_k_posterior,
)
from sift.selection.auto_k_stop import (
    path_gain_pvalues,
    select_k_changepoint,
    select_k_chi2_stop,
    select_k_forward_stop,
)
from sift.selection.auto_k_knockoff import select_k_knockoff_path
from sift.selection.auto_k_resample import (
    bootstrap_paths,
    null_objective_paths,
    select_k_perm_gap,
    select_k_stability,
)
from sift.selection.auto_k_xfit import (
    gaussian_cv_curves,
    select_k_gaussian_cv,
    select_k_xfit_objective,
    xfit_objective_curves,
)
from sift.selection.cefsplus import select_cached
from sift.selection.knockoff_filter import (
    KnockoffSelectionResult,
    knockoff_threshold,
    select_fdr,
)
from sift.selection.loops import jmi_select, mrmr_select
from sift.selection.path_eval import FeaturePathEvaluationResult, evaluate_feature_path
from sift.selection.model_selector import ModelSelector
from sift.selection.purged_cv import (
    GroupPurgedTimeSeriesSplit,
    PurgedTimeSeriesSplit,
)

__all__ = [
    "AutoKConfig",
    "compute_objective_for_path",
    "select_k_auto",
    "select_k_elbow",
    "select_k_penalized_objective",
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
    "select_cached",
    "select_fdr",
    "knockoff_threshold",
    "KnockoffSelectionResult",
    "jmi_select",
    "mrmr_select",
    "FeaturePathEvaluationResult",
    "evaluate_feature_path",
    "PurgedTimeSeriesSplit",
    "GroupPurgedTimeSeriesSplit",
    "ModelSelector",
]
