"""Feature-path evaluation adapter for additive SelectionView construction."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from sift.selection.view import (
    SelectionView,
    _coerce_feature_names,
    _coerce_indices,
    _label_token,
    _labels_equal,
    _numeric_vector,
    _strict_integer,
    _strict_integer_vector,
)


def _path_result_scores(result: Any, tested_k: list[int]) -> dict[int, float]:
    if not isinstance(result.scores, Mapping):
        raise ValueError("FeaturePathEvaluationResult scores must be a mapping")
    scores: dict[int, float] = {}
    for key, value in result.scores.items():
        key_int = _strict_integer(
            key,
            label="FeaturePathEvaluationResult score keys",
            minimum=1,
        )
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (Real, np.integer, np.floating)
        ):
            raise ValueError(
                "FeaturePathEvaluationResult scores must be real non-boolean numbers"
            )
        try:
            score = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "FeaturePathEvaluationResult scores must be real non-boolean numbers "
                "representable as float64"
            ) from exc
        if (
            isinstance(value, np.floating)
            and np.isfinite(value)
            and not math.isfinite(score)
        ):
            raise ValueError(
                "FeaturePathEvaluationResult scores must be representable as float64"
            )
        if math.isnan(score) or score == float("-inf"):
            raise ValueError(
                "FeaturePathEvaluationResult scores must be finite or positive infinity"
            )
        scores[key_int] = score
    if set(scores) != set(tested_k) or len(scores) != len(tested_k):
        raise ValueError(
            "FeaturePathEvaluationResult score keys must match the tested k grid"
        )
    return scores


def _numeric_values_equal(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=0.0, atol=0.0, equal_nan=True))


def _validate_path_diagnostics(
    diagnostics: Any,
    *,
    tested_k: list[int],
    scores: Mapping[int, float],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(diagnostics, pd.DataFrame):
        raise ValueError("FeaturePathEvaluationResult diagnostics must be a DataFrame")
    required = {"k", "score", "std", "n_finite", "n_splits", "best_score"}
    missing = sorted(required.difference(diagnostics.columns))
    if missing:
        raise ValueError(
            "FeaturePathEvaluationResult diagnostics is missing required columns: "
            f"{missing}"
        )
    if len(diagnostics) != len(tested_k):
        raise ValueError(
            "FeaturePathEvaluationResult diagnostics rows must match the tested k grid"
        )
    diagnostic_k = _strict_integer_vector(
        diagnostics["k"],
        label="FeaturePathEvaluationResult diagnostics k",
        length=len(tested_k),
        minimum=1,
    ).tolist()
    if diagnostic_k != tested_k:
        raise ValueError(
            "FeaturePathEvaluationResult diagnostics k order must match the tested grid"
        )
    diagnostic_scores = _numeric_vector(
        diagnostics["score"],
        label="FeaturePathEvaluationResult diagnostics score",
        length=len(tested_k),
    )
    if any(
        not _numeric_values_equal(score, scores[k])
        for k, score in zip(tested_k, diagnostic_scores)
    ):
        raise ValueError(
            "FeaturePathEvaluationResult diagnostics scores must match scores"
        )
    std = _numeric_vector(
        diagnostics["std"],
        label="FeaturePathEvaluationResult diagnostics std",
        length=len(tested_k),
    )
    if np.isinf(std).any() or (std[np.isfinite(std)] < 0.0).any():
        raise ValueError(
            "FeaturePathEvaluationResult diagnostics std must be non-negative or NaN"
        )
    n_finite = _strict_integer_vector(
        diagnostics["n_finite"],
        label="FeaturePathEvaluationResult diagnostics n_finite",
        length=len(tested_k),
        minimum=0,
    )
    n_splits = _strict_integer_vector(
        diagnostics["n_splits"],
        label="FeaturePathEvaluationResult diagnostics n_splits",
        length=len(tested_k),
        minimum=1,
    )
    if (n_finite > n_splits).any():
        raise ValueError(
            "FeaturePathEvaluationResult diagnostics n_finite cannot exceed n_splits"
        )
    if len(set(n_splits.tolist())) != 1:
        raise ValueError(
            "FeaturePathEvaluationResult diagnostics n_splits must be constant"
        )
    for score, spread, finite, splits in zip(
        diagnostic_scores, std, n_finite, n_splits
    ):
        if np.isfinite(score):
            if finite != splits or not np.isfinite(spread):
                raise ValueError(
                    "finite FeaturePathEvaluationResult scores require every split "
                    "to be finite and std to be finite"
                )
            if finite == 1 and spread != 0.0:
                raise ValueError(
                    "a finite single-split FeaturePathEvaluationResult must have std 0"
                )
            continue
        if score != float("inf") or finite >= splits:
            raise ValueError(
                "infinite FeaturePathEvaluationResult scores require at least one "
                "failed split"
            )
        if (finite > 1 and not np.isfinite(spread)) or (
            finite <= 1 and not np.isnan(spread)
        ):
            raise ValueError(
                "FeaturePathEvaluationResult diagnostics std is inconsistent with "
                "n_finite"
            )
    return diagnostics.copy(deep=True), diagnostic_scores, std, n_finite


def _resolve_path_positions(
    feature_path: list[Any],
    raw_features: list[Any],
) -> list[int]:
    raw_tokens = [_label_token(feature) for feature in raw_features]
    positions: list[int] = []
    for feature in feature_path:
        token = _label_token(feature)
        matches = [
            index for index, candidate in enumerate(raw_tokens) if candidate == token
        ]
        if len(matches) != 1:
            raise ValueError(
                f"feature_path feature {feature!r} is missing or ambiguous in input_features; "
                "FeaturePathEvaluationResult does not retain raw positions"
            )
        positions.append(matches[0])
    if len(set(positions)) != len(positions):
        raise ValueError(
            "feature_path entries do not map to unique positions in input_features"
        )
    return positions


def _as_feature_path_result(result: Any, input_features: Any) -> SelectionView:
    feature_path = _coerce_feature_names(result.feature_path)
    selected = _coerce_feature_names(result.features)
    if feature_path is None or not feature_path:
        raise ValueError("FeaturePathEvaluationResult feature_path must be non-empty")
    if selected is None:
        raise ValueError(
            "FeaturePathEvaluationResult features must be an ordered iterable"
        )

    tested_k = _coerce_indices(result.k, label="FeaturePathEvaluationResult k")
    if tested_k is None or not tested_k or any(k < 1 for k in tested_k):
        raise ValueError(
            "FeaturePathEvaluationResult k must contain unique positive integers"
        )
    if any(k > len(feature_path) for k in tested_k):
        raise ValueError(
            "FeaturePathEvaluationResult k cannot exceed the feature_path length"
        )
    scores = _path_result_scores(result, tested_k)
    diagnostics, diagnostic_scores, std, n_finite = _validate_path_diagnostics(
        result.diagnostics,
        tested_k=tested_k,
        scores=scores,
    )

    finite_candidates = [
        (score, k) for k, score in scores.items() if np.isfinite(score)
    ]
    if finite_candidates:
        expected_best_score, expected_best_k = min(
            finite_candidates,
            key=lambda item: (item[0], item[1]),
        )
    else:
        expected_best_k = 0
        expected_best_score = float("nan")
    best_k = _strict_integer(
        result.best_k,
        label="FeaturePathEvaluationResult best_k",
        minimum=0,
    )
    if best_k != expected_best_k:
        raise ValueError(
            "FeaturePathEvaluationResult best_k does not match the lower-is-better scores"
        )
    expected_features = feature_path[:best_k]
    if len(selected) != best_k or any(
        not _labels_equal(expected, observed)
        for expected, observed in zip(expected_features, selected)
    ):
        raise ValueError(
            "FeaturePathEvaluationResult features must equal feature_path[:best_k]"
        )
    best_score_values = _numeric_vector(
        diagnostics["best_score"],
        label="FeaturePathEvaluationResult diagnostics best_score",
        length=len(tested_k),
    )
    if any(
        not _numeric_values_equal(value, expected_best_score)
        for value in best_score_values
    ):
        raise ValueError(
            "FeaturePathEvaluationResult diagnostics best_score is inconsistent"
        )

    criterion_se = np.full(len(tested_k), np.nan, dtype=np.float64)
    for index, (score, spread, finite) in enumerate(
        zip(diagnostic_scores, std, n_finite)
    ):
        if np.isfinite(score) and np.isfinite(spread) and finite >= 2:
            criterion_se[index] = float(spread) / math.sqrt(int(finite) - 1)
    curve = pd.DataFrame(
        {
            "k": tested_k,
            "criterion": diagnostic_scores.copy(),
            "criterion_se": criterion_se,
            "selected": [best_k > 0 and k == best_k for k in tested_k],
        }
    )

    raw_features = _coerce_feature_names(input_features)
    if raw_features is None:
        selected_indices = None
        table = pd.DataFrame(
            {
                "feature": feature_path,
                "selected_index": pd.array([pd.NA] * len(feature_path), dtype="Int64"),
                "path_rank": pd.array(
                    list(range(1, best_k + 1)) + [pd.NA] * (len(feature_path) - best_k),
                    dtype="Int64",
                ),
                "selected": [index < best_k for index in range(len(feature_path))],
                "feature_path_rank": pd.array(
                    range(1, len(feature_path) + 1), dtype="Int64"
                ),
            }
        )
        table_complete = False
        n_raw_features = None
    else:
        path_positions = _resolve_path_positions(feature_path, raw_features)
        selected_indices = path_positions[:best_k]
        n_raw_features = len(raw_features)
        path_rank = pd.Series(pd.array([pd.NA] * n_raw_features, dtype="Int64"))
        feature_path_rank = pd.Series(pd.array([pd.NA] * n_raw_features, dtype="Int64"))
        for rank, position in enumerate(path_positions, start=1):
            feature_path_rank.iloc[position] = rank
        for rank, position in enumerate(selected_indices, start=1):
            path_rank.iloc[position] = rank
        selected_mask = np.zeros(n_raw_features, dtype=bool)
        selected_mask[selected_indices] = True
        table = pd.DataFrame(
            {
                "feature": raw_features,
                "selected_index": pd.array(range(n_raw_features), dtype="Int64"),
                "path_rank": path_rank,
                "selected": selected_mask,
                "feature_path_rank": feature_path_rank,
            }
        )
        table_complete = True

    metadata = {
        "adapter": "FeaturePathEvaluationResult",
        "curve_available": True,
        "criterion_direction": "minimize",
        "best_k": best_k,
        "best_score": expected_best_score,
        "tested_k": tested_k,
        "table_complete": table_complete,
        "input_kind": "unknown",
        "criterion_se_definition": "population_std/sqrt(n_finite-1)",
    }
    return SelectionView(
        features=selected,
        indices=selected_indices,
        raw_features=raw_features,
        n_raw_features=n_raw_features,
        raw_table=table,
        curve=curve,
        metadata=metadata,
        diagnostics=diagnostics,
    )
