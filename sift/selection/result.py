"""Shared result containers for filter selectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


_PROXY_CORRELATIONS_ATTR = "_proxy_correlations"


@dataclass(frozen=True)
class FilterSelectionResult:
    """Result object for conservative filter selection APIs.

    Attributes
    ----------
    selected_features : list[str]
        Ordered selected feature names.
    selected_indices : list[int] or None
        Selected feature indices in the same feature namespace as input `X`.
        `None` if indices cannot be resolved safely.
    selector_metadata : dict
        Configuration used by the selector.
    ranking_ : DataFrame, optional
        Optional diagnostic ranking. When omitted, a compact selected-feature
        ranking is built on demand by ``get_feature_ranking``.
    diagnostics_ : dict or DataFrame, optional
        Optional selector-specific diagnostics.

    Examples
    --------
    Filter selectors hand one back when ``return_result=True``; the dataclass
    itself is plain enough to build directly, which is what the example does so
    every field is visible:

    >>> from sift import FilterSelectionResult
    >>> result = FilterSelectionResult(
    ...     selected_features=["f0", "f3"],
    ...     selected_indices=[0, 3],
    ...     selector_metadata={"selector": "mrmr", "k": 2, "n_features": 5},
    ... )
    >>> result.selected_features
    ['f0', 'f3']
    >>> ranking = result.get_feature_ranking()
    >>> ranking["feature"].tolist(), ranking["rank"].tolist()
    (['f0', 'f3'], [1, 2])
    >>> view = result.result_view(input_features=["f0", "f1", "f2", "f3", "f4"])
    >>> view.features
    ['f0', 'f3']
    """

    selected_features: List[str]
    selected_indices: Optional[List[int]]
    selector_metadata: Dict[str, Any]
    ranking_: Optional[pd.DataFrame] = None
    diagnostics_: Optional[Any] = None

    def get_feature_ranking(self) -> pd.DataFrame:
        """Return a feature ranking diagnostic table.

        The base table is intentionally conservative: it reports known selected
        features and leaves metrics such as relevance as NaN unless a selector
        supplied them explicitly.
        """
        if self.ranking_ is not None:
            return self.ranking_.copy()

        selected_indices = self.selected_indices
        if selected_indices is None:
            selected_index_values = [np.nan] * len(self.selected_features)
        else:
            selected_index_values = [int(idx) for idx in selected_indices]

        return pd.DataFrame(
            {
                "feature": list(self.selected_features),
                "rank": np.arange(1, len(self.selected_features) + 1, dtype=np.int64),
                "selected": np.ones(len(self.selected_features), dtype=bool),
                "selected_index": selected_index_values,
                "relevance": np.full(len(self.selected_features), np.nan),
                "selector": self.selector_metadata.get("selector"),
            }
        )

    def result_view(self, input_features=None):
        """Return an additive normalized view without changing this result."""
        from sift.selection.view import as_result

        return as_result(self, input_features=input_features)


def build_selector_metadata(
    selector: str,
    *,
    k: int | str,
    k_requested: int | str,
    top_m: Optional[int],
    n_features: int,
    auto_k: bool,
    extra: Optional[dict] = None,
) -> dict:
    metadata = {
        "selector": selector,
        "k_requested": k_requested,
        "k": k,
        "top_m": top_m,
        "n_features": int(n_features),
        "auto_k": auto_k,
    }
    if extra:
        metadata.update(extra)
    return metadata
