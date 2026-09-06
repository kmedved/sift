"""User-facing API for feature selection."""

from __future__ import annotations

from sift.estimators.classic_cache import ClassicFeatureCache, build_classic_cache
from sift.estimators.copula import FeatureCache, build_cache
from sift.selection.auto_k import (
    AutoKConfig,
    select_k_auto,
    select_k_elbow,
    select_k_penalized_objective,
)
from sift.selection.cefsplus import select_cached
from sift.selection.filter_api import (
    select_cefsplus,
    select_cefsplus_binary,
    select_jmi,
    select_jmim,
    select_mrmr,
)
from sift.selection.knockoff_filter import (
    KnockoffSelectionResult,
    sample_knockoffs,
    select_fdr,
)

__all__ = [
    "FeatureCache",
    "ClassicFeatureCache",
    "AutoKConfig",
    "build_cache",
    "build_classic_cache",
    "select_k_auto",
    "select_k_elbow",
    "select_k_penalized_objective",
    "select_cached",
    "select_cefsplus",
    "select_cefsplus_binary",
    "select_fdr",
    "select_jmi",
    "select_jmim",
    "select_mrmr",
    "KnockoffSelectionResult",
    "sample_knockoffs",
]
