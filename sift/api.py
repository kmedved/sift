"""User-facing API for feature selection."""

from __future__ import annotations

from sift.estimators.copula import FeatureCache, build_cache
from sift.selection.auto_k import (
    AutoKConfig,
    select_k_auto,
    select_k_elbow,
    select_k_penalized_objective,
)
from sift.selection.cefsplus import select_cached
from sift.selection.cefsplus_api import select_cefsplus
from sift.selection.cefsplus_binary_api import select_cefsplus_binary
from sift.selection.jmi_api import select_jmi, select_jmim
from sift.selection.mrmr_api import select_mrmr

__all__ = [
    "FeatureCache",
    "build_cache",
    "select_cached",
    "select_cefsplus",
    "select_cefsplus_binary",
    "select_jmi",
    "select_jmim",
    "select_mrmr",
]
