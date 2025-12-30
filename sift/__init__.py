__version__ = "0.4.0"

# Core algorithms
from sift.core.algorithms import mrmr_base, jmi_base

# MI utilities
from sift.mi.copula import FeatureCache, build_cache
from sift.mi.estimators import regression_joint_mi, binned_joint_mi, ksg_joint_mi
from sift.mi.fast_selectors import jmi_fast, jmim_fast, cefsplus_regression, select_features_cached

# Main selectors (pandas)
from sift.selectors.mrmr import (
    mrmr_classif,
    mrmr_regression,
    jmi_classif,
    jmi_regression,
    jmim_classif,
    jmim_regression,
    cefsplus_select,
)

# Stability selection
from sift.selectors.stability import (
    StabilitySelector,
    stability_select,
    stability_regression,
    stability_classif,
)

# Sampling
from sift.sampling.smart import SmartSamplerConfig, smart_sample
from sift.sampling.anchors import (
    panel_config,
    cross_section_config,
)

# Lazy imports for optional dependencies
_LAZY_ATTRS = {
    "catboost_select": ("sift.selectors.catboost", "catboost_select"),
    "catboost_regression": ("sift.selectors.catboost", "catboost_regression"),
    "catboost_classif": ("sift.selectors.catboost", "catboost_classif"),
    "CatBoostSelectionResult": ("sift.selectors.catboost", "CatBoostSelectionResult"),
    "polars": ("sift.backends.polars", None),
}


def __getattr__(name):
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__} has no attribute {name}")
    import importlib
    module_name, attr = _LAZY_ATTRS[name]
    module = importlib.import_module(module_name)
    value = module if attr is None else getattr(module, attr)
    globals()[name] = value
    return value


__all__ = [
    "__version__",
    # Core
    "mrmr_base",
    "jmi_base",
    # MI
    "FeatureCache",
    "build_cache",
    "regression_joint_mi",
    "binned_joint_mi",
    "ksg_joint_mi",
    "jmi_fast",
    "jmim_fast",
    "cefsplus_regression",
    "select_features_cached",
    # Pandas selectors
    "mrmr_classif",
    "mrmr_regression",
    "jmi_classif",
    "jmi_regression",
    "jmim_classif",
    "jmim_regression",
    "cefsplus_select",
    # Stability
    "StabilitySelector",
    "stability_select",
    "stability_regression",
    "stability_classif",
    # Sampling
    "SmartSamplerConfig",
    "smart_sample",
    "panel_config",
    "cross_section_config",
    # Lazy (catboost)
    "catboost_select",
    "catboost_regression",
    "catboost_classif",
    "CatBoostSelectionResult",
    # Lazy (polars)
    "polars",
]
