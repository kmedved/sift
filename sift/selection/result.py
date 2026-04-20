"""Shared result containers for filter selectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


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
    """

    selected_features: List[str]
    selected_indices: Optional[List[int]]
    selector_metadata: Dict[str, Any]
