__version__ = "0.9.0a1"

from sift._logging import set_verbosity
from sift.api import (
    FeatureCache,
    build_cache,
    KnockoffSelectionResult,
    sample_knockoffs,
    select_cached,
    select_cefsplus,
    select_cefsplus_binary,
    select_fdr,
    select_jmi,
    select_jmim,
    select_mrmr,
)
from sift.boruta import BorutaResult, BorutaSelector, select_boruta, select_boruta_shap
from sift.importance import permutation_importance
from sift.sampling import SmartSamplerConfig, cross_section_config, panel_config, smart_sample
from sift.selectors import (
    CEFSPlusBinarySelector,
    CEFSPlusSelector,
    JMISelector,
    JMIMSelector,
    KnockoffSelector,
    MRMRSelector,
)
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
from sift.selection.path_eval import FeaturePathEvaluationResult, evaluate_feature_path
from sift.selection.result import FilterSelectionResult
from sift.selection.view import SelectionView, as_result
from sift.stability import StabilitySelector, stability_classif, stability_regression


def __getattr__(name):
    if name in ("catboost_select", "catboost_regression", "catboost_classif"):
        # CatBoost is optional; keep it lazy so importing sift does not require it.
        from sift import catboost

        return getattr(catboost, name)
    raise AttributeError(f"module 'sift' has no attribute '{name}'")


__all__ = [
    "__version__",
    "FeatureCache",
    "build_cache",
    "KnockoffSelectionResult",
    "sample_knockoffs",
    "select_cached",
    "select_cefsplus",
    "select_cefsplus_binary",
    "select_fdr",
    "select_jmi",
    "select_jmim",
    "select_mrmr",
    "AutoKConfig",
    "FeaturePathEvaluationResult",
    "FilterSelectionResult",
    "evaluate_feature_path",
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
    "compute_objective_for_path",
    "BorutaSelector",
    "BorutaResult",
    "select_boruta",
    "select_boruta_shap",
    "MRMRSelector",
    "JMISelector",
    "JMIMSelector",
    "CEFSPlusSelector",
    "CEFSPlusBinarySelector",
    "KnockoffSelector",
    "permutation_importance",
    "SmartSamplerConfig",
    "smart_sample",
    "panel_config",
    "cross_section_config",
    "StabilitySelector",
    "stability_regression",
    "stability_classif",
    "catboost_select",
    "catboost_regression",
    "catboost_classif",
    "set_verbosity",
    "SelectionView",
    "as_result",
]
