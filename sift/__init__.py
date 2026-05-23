__version__ = "0.6.0"

from sift.api import (
    FeatureCache,
    build_cache,
    select_cached,
    select_cefsplus,
    select_cefsplus_binary,
    select_jmi,
    select_jmim,
    select_mrmr,
)
from sift.boruta import BorutaResult, BorutaSelector, select_boruta, select_boruta_shap
from sift.importance import permutation_importance
from sift.sampling import SmartSamplerConfig, cross_section_config, panel_config, smart_sample
from sift.selectors import CEFSPlusBinarySelector, CEFSPlusSelector, JMISelector, JMIMSelector, MRMRSelector
from sift.selection.auto_k import (
    AutoKConfig,
    compute_objective_for_path,
    select_k_auto,
    select_k_elbow,
    select_k_penalized_objective,
)
from sift.selection.path_eval import FeaturePathEvaluationResult, evaluate_feature_path
from sift.stability import StabilitySelector
from sift.stability_api import stability_classif, stability_regression


def __getattr__(name):
    if name == "catboost_select":
        from sift import catboost

        return getattr(catboost, name)
    if name in ("catboost_regression", "catboost_classif"):
        from sift import catboost_api

        return getattr(catboost_api, name)
    raise AttributeError(f"module 'sift' has no attribute '{name}'")


__all__ = [
    "__version__",
    "FeatureCache",
    "build_cache",
    "select_cached",
    "select_cefsplus",
    "select_cefsplus_binary",
    "select_jmi",
    "select_jmim",
    "select_mrmr",
    "AutoKConfig",
    "FeaturePathEvaluationResult",
    "evaluate_feature_path",
    "select_k_auto",
    "select_k_elbow",
    "select_k_penalized_objective",
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
]
