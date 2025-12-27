import importlib

__version__ = "0.3.0"

_LAZY_ATTRS = {
    "mrmr_classif": ("mrmr.selectors_pandas", "mrmr_classif"),
    "mrmr_regression": ("mrmr.selectors_pandas", "mrmr_regression"),
    "jmi_classif": ("mrmr.selectors_pandas", "jmi_classif"),
    "jmi_regression": ("mrmr.selectors_pandas", "jmi_regression"),
    "jmim_classif": ("mrmr.selectors_pandas", "jmim_classif"),
    "jmim_regression": ("mrmr.selectors_pandas", "jmim_regression"),
    "cefsplus_select": ("mrmr.selectors_pandas", "cefsplus_select"),
    "mrmr_base": ("mrmr.main", "mrmr_base"),
    "jmi_base": ("mrmr.main", "jmi_base"),
    "regression_joint_mi": ("mrmr.fast_mi", "regression_joint_mi"),
    "binned_joint_mi": ("mrmr.fast_mi", "binned_joint_mi"),
    "ksg_joint_mi": ("mrmr.fast_mi", "ksg_joint_mi"),
    "cefsplus_regression": ("mrmr.cefsplus", "cefsplus_regression"),
    "build_cache": ("mrmr.cefsplus", "build_cache"),
    "select_features_cached": ("mrmr.cefsplus", "select_features_cached"),
    "FeatureCache": ("mrmr.cefsplus", "FeatureCache"),
    "selectors_pandas": ("mrmr.selectors_pandas", None),
    "fast_mi": ("mrmr.fast_mi", None),
    "bigquery": ("mrmr.bigquery", None),
    "polars": ("mrmr.polars", None),
    "spark": ("mrmr.spark", None),
}

__all__ = list(_LAZY_ATTRS.keys()) + ["__version__"]


def __getattr__(name):
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__} has no attribute {name}")
    module_name, attr = _LAZY_ATTRS[name]
    module = importlib.import_module(module_name)
    value = module if attr is None else getattr(module, attr)
    globals()[name] = value
    return value
