import importlib

__version__ = "0.3.0"

_LAZY_ATTRS = {
    "mrmr_classif": ("sift.selectors_pandas", "mrmr_classif"),
    "mrmr_regression": ("sift.selectors_pandas", "mrmr_regression"),
    "jmi_classif": ("sift.selectors_pandas", "jmi_classif"),
    "jmi_regression": ("sift.selectors_pandas", "jmi_regression"),
    "jmim_classif": ("sift.selectors_pandas", "jmim_classif"),
    "jmim_regression": ("sift.selectors_pandas", "jmim_regression"),
    "cefsplus_select": ("sift.selectors_pandas", "cefsplus_select"),
    "mrmr_base": ("sift.main", "mrmr_base"),
    "jmi_base": ("sift.main", "jmi_base"),
    "regression_joint_mi": ("sift.fast_mi", "regression_joint_mi"),
    "binned_joint_mi": ("sift.fast_mi", "binned_joint_mi"),
    "ksg_joint_mi": ("sift.fast_mi", "ksg_joint_mi"),
    "cefsplus_regression": ("sift.cefsplus", "cefsplus_regression"),
    "build_cache": ("sift.gcmi", "build_cache"),
    "select_features_cached": ("sift.cefsplus", "select_features_cached"),
    "FeatureCache": ("sift.gcmi", "FeatureCache"),
    "StabilitySelector": ("sift.stability_selection", "StabilitySelector"),
    "stability_select": ("sift.stability_selection", "stability_select"),
    "stability_regression": ("sift.stability_selection", "stability_regression"),
    "stability_classif": ("sift.stability_selection", "stability_classif"),
    "smart_sample": ("sift.stability_selection", "smart_sample"),
    "SmartSamplerConfig": ("sift.stability_selection", "SmartSamplerConfig"),
    "panel_config": ("sift.stability_selection", "panel_config"),
    "cross_section_config": ("sift.stability_selection", "cross_section_config"),
    "stability_selection": ("sift.stability_selection", None),
    "selectors_pandas": ("sift.selectors_pandas", None),
    "fast_mi": ("sift.fast_mi", None),
    "gcmi": ("sift.gcmi", None),
    "polars": ("sift.polars", None),
    "catboost_select": ("sift.catboost", "catboost_select"),
    "catboost_regression": ("sift.catboost", "catboost_regression"),
    "catboost_classif": ("sift.catboost", "catboost_classif"),
    "CatBoostSelectionResult": ("sift.catboost", "CatBoostSelectionResult"),
}

__all__ = [name for name in _LAZY_ATTRS.keys() if name != "polars"] + ["__version__"]


def __getattr__(name):
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__} has no attribute {name}")
    module_name, attr = _LAZY_ATTRS[name]
    module = importlib.import_module(module_name)
    value = module if attr is None else getattr(module, attr)
    globals()[name] = value
    return value
