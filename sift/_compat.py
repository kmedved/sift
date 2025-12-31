"""Backward compatibility shims for deprecated API."""

from __future__ import annotations

import warnings
from typing import List

from sift.api import select_cefsplus, select_jmi, select_jmim, select_mrmr


def mrmr_regression(X, y, K=10, **kwargs) -> List[str]:
    warnings.warn(
        "mrmr_regression is deprecated; use select_mrmr(..., task='regression')",
        DeprecationWarning,
        stacklevel=2,
    )
    return select_mrmr(X, y, k=K, task="regression", **kwargs)


def mrmr_classif(X, y, K=10, **kwargs) -> List[str]:
    warnings.warn(
        "mrmr_classif is deprecated; use select_mrmr(..., task='classification')",
        DeprecationWarning,
        stacklevel=2,
    )
    return select_mrmr(X, y, k=K, task="classification", **kwargs)


def jmi_regression(X, y, K=10, **kwargs) -> List[str]:
    warnings.warn(
        "jmi_regression is deprecated; use select_jmi(..., task='regression')",
        DeprecationWarning,
        stacklevel=2,
    )
    return select_jmi(X, y, k=K, task="regression", **kwargs)


def jmi_classif(X, y, K=10, **kwargs) -> List[str]:
    warnings.warn(
        "jmi_classif is deprecated; use select_jmi(..., task='classification')",
        DeprecationWarning,
        stacklevel=2,
    )
    return select_jmi(X, y, k=K, task="classification", **kwargs)


def jmim_regression(X, y, K=10, **kwargs) -> List[str]:
    warnings.warn(
        "jmim_regression is deprecated; use select_jmim(..., task='regression')",
        DeprecationWarning,
        stacklevel=2,
    )
    return select_jmim(X, y, k=K, task="regression", **kwargs)


def jmim_classif(X, y, K=10, **kwargs) -> List[str]:
    warnings.warn(
        "jmim_classif is deprecated; use select_jmim(..., task='classification')",
        DeprecationWarning,
        stacklevel=2,
    )
    return select_jmim(X, y, k=K, task="classification", **kwargs)


def cefsplus_regression(X, y, K=10, **kwargs) -> List[str]:
    warnings.warn(
        "cefsplus_regression is deprecated; use select_cefsplus()",
        DeprecationWarning,
        stacklevel=2,
    )
    return select_cefsplus(X, y, k=K, **kwargs)


def jmi_fast(X, y, K=10, **kwargs) -> List[str]:
    warnings.warn(
        "jmi_fast is deprecated; use select_jmi(..., task='regression', estimator='gaussian')",
        DeprecationWarning,
        stacklevel=2,
    )
    return select_jmi(X, y, k=K, task="regression", estimator="gaussian", **kwargs)


def jmim_fast(X, y, K=10, **kwargs) -> List[str]:
    warnings.warn(
        "jmim_fast is deprecated; use select_jmim(..., task='regression', estimator='gaussian')",
        DeprecationWarning,
        stacklevel=2,
    )
    return select_jmim(X, y, k=K, task="regression", estimator="gaussian", **kwargs)
