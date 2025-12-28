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
    "build_cache": ("sift.cefsplus", "build_cache"),
    "select_features_cached": ("sift.cefsplus", "select_features_cached"),
    "FeatureCache": ("sift.cefsplus", "FeatureCache"),
    "selectors_pandas": ("sift.selectors_pandas", None),
    "fast_mi": ("sift.fast_mi", None),
    "polars": ("sift.polars", None),
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
