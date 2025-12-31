__version__ = "0.5.0"

from sift.api import (
    FeatureCache,
    build_cache,
    select_cached,
    select_cefsplus,
    select_jmi,
    select_jmim,
    select_mrmr,
)
from sift._compat import (
    cefsplus_regression,
    jmi_classif,
    jmi_fast,
    jmi_regression,
    jmim_classif,
    jmim_fast,
    jmim_regression,
    mrmr_classif,
    mrmr_regression,
)
from sift.sampling import SmartSamplerConfig, cross_section_config, panel_config, smart_sample
from sift.stability import StabilitySelector, stability_classif, stability_regression


def __getattr__(name):
    if name in ("catboost_select", "catboost_regression", "catboost_classif"):
        from sift import catboost

        return getattr(catboost, name)
    raise AttributeError(f"module 'sift' has no attribute '{name}'")


__all__ = [
    "__version__",
    "FeatureCache",
    "build_cache",
    "select_cached",
    "select_cefsplus",
    "select_jmi",
    "select_jmim",
    "select_mrmr",
    "SmartSamplerConfig",
    "smart_sample",
    "panel_config",
    "cross_section_config",
    "StabilitySelector",
    "stability_regression",
    "stability_classif",
    "mrmr_regression",
    "mrmr_classif",
    "jmi_regression",
    "jmi_classif",
    "jmim_regression",
    "jmim_classif",
    "cefsplus_regression",
    "jmi_fast",
    "jmim_fast",
]
