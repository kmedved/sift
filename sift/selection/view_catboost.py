"""CatBoost-only adapter for additive SelectionView construction."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Set
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from sift.selection.view import (
    SelectionView,
    _coerce_feature_names,
    _label_token,
    _strict_integer,
)


def _catboost_label_key(value: Any) -> str:
    return json.dumps(
        _label_token(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _catboost_feature_list(
    values: Any,
    *,
    label: str,
    allow_none: bool = False,
) -> list[Any] | None:
    if values is None:
        if allow_none:
            return None
        raise ValueError(f"{label} must be an ordered iterable")
    try:
        features = _coerce_feature_names(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an ordered one-dimensional iterable") from exc
    if features is None:
        raise ValueError(f"{label} must be an ordered iterable")
    keys = [_catboost_label_key(feature) for feature in features]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{label} must contain unique feature identities")
    return features


def _catboost_float(
    value: Any,
    *,
    label: str,
    finite: bool,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise ValueError(f"{label} must be a real non-boolean number")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be representable as float64") from exc
    if (
        isinstance(value, np.floating)
        and np.isfinite(value)
        and not math.isfinite(converted)
    ):
        raise ValueError(f"{label} must be representable as float64")
    if finite and not math.isfinite(converted):
        raise ValueError(f"{label} must be finite")
    return converted


def _catboost_score_mapping(values: Any, *, label: str) -> dict[int, float]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{label} must be a mapping")
    scores: dict[int, float] = {}
    for key, value in values.items():
        k = _strict_integer(key, label=f"{label} keys", minimum=1)
        if k in scores:
            raise ValueError(f"{label} keys must be unique positive integers")
        scores[k] = _catboost_float(value, label=f"{label} values", finite=True)
    return scores


def _catboost_std_mapping(values: Any, *, score_keys: set[int]) -> dict[int, float]:
    std_by_k = _catboost_score_mapping(values, label="CatBoost scores_std_by_k")
    extra = sorted(set(std_by_k).difference(score_keys))
    if extra:
        raise ValueError(
            "CatBoost scores_std_by_k keys must be present in scores_by_k; "
            f"unexpected {extra}"
        )
    if any(value < 0.0 for value in std_by_k.values()):
        raise ValueError("CatBoost scores_std_by_k values must be non-negative")
    return std_by_k


def _catboost_all_scores(values: Any) -> dict[int, list[float]] | None:
    if values is None:
        return None
    if not isinstance(values, Mapping):
        raise ValueError("CatBoost all_scores must be a mapping or None")
    all_scores: dict[int, list[float]] = {}
    for key, raw_values in values.items():
        k = _strict_integer(key, label="CatBoost all_scores keys", minimum=1)
        if k in all_scores:
            raise ValueError(
                "CatBoost all_scores keys must be unique positive integers"
            )
        if isinstance(raw_values, (str, bytes, bytearray, Mapping, Set)):
            raise ValueError("CatBoost all_scores values must be ordered iterables")
        try:
            score_values = list(raw_values)
        except TypeError as exc:
            raise ValueError(
                "CatBoost all_scores values must be ordered iterables"
            ) from exc
        all_scores[k] = [
            _catboost_float(
                value,
                label="CatBoost all_scores observations",
                finite=False,
            )
            for value in score_values
        ]
    return all_scores


def _catboost_features_by_k(values: Any) -> dict[int, list[Any]]:
    if not isinstance(values, Mapping):
        raise ValueError("CatBoost features_by_k must be a mapping")
    features_by_k: dict[int, list[Any]] = {}
    for key, raw_features in values.items():
        k = _strict_integer(key, label="CatBoost features_by_k keys", minimum=1)
        if k in features_by_k:
            raise ValueError(
                "CatBoost features_by_k keys must be unique positive integers"
            )
        features = _catboost_feature_list(
            raw_features,
            label=f"CatBoost features_by_k[{k}]",
        )
        assert features is not None
        if len(features) != k:
            raise ValueError(
                f"CatBoost features_by_k[{k}] must contain exactly {k} features"
            )
        features_by_k[k] = features
    return features_by_k


def _catboost_numeric_series(
    values: Any,
    *,
    label: str,
    allow_none: bool = False,
    unit_interval: bool = False,
) -> pd.Series | None:
    if values is None:
        if allow_none:
            return None
        raise ValueError(f"{label} must be a pandas Series")
    if not isinstance(values, pd.Series):
        raise ValueError(f"{label} must be a pandas Series")
    features = _catboost_feature_list(values.index, label=f"{label} index")
    assert features is not None
    numeric = np.asarray(
        [
            _catboost_float(value, label=f"{label} values", finite=True)
            for value in values.tolist()
        ],
        dtype=np.float64,
    )
    if unit_interval and ((numeric < 0.0).any() or (numeric > 1.0).any()):
        raise ValueError(f"{label} values must be between 0 and 1")
    return pd.Series(numeric, index=pd.Index(features), name=values.name)


def _catboost_curve(
    *,
    scores_by_k: Mapping[int, float],
    scores_std_by_k: Mapping[int, float],
    all_scores: Mapping[int, list[float]] | None,
    best_k: int,
) -> pd.DataFrame:
    if not scores_by_k:
        raise ValueError("CatBoost scores_by_k must contain at least one finite score")
    if best_k not in scores_by_k:
        raise ValueError("CatBoost best_k must be present in scores_by_k")

    criterion_se: dict[int, float] = {k: float("nan") for k in scores_by_k}
    if all_scores is not None:
        for k, observations in all_scores.items():
            finite_values = np.asarray(observations, dtype=np.float64)
            finite_values = finite_values[np.isfinite(finite_values)]
            if k not in scores_by_k:
                if finite_values.size:
                    raise ValueError(
                        "CatBoost all_scores with finite observations must have a "
                        "matching scores_by_k entry"
                    )
                continue
            if not finite_values.size:
                raise ValueError(
                    "CatBoost all_scores must contain a finite observation for each "
                    "stored score"
                )
            mean = float(np.mean(finite_values))
            if not np.isclose(mean, scores_by_k[k], rtol=1e-12, atol=1e-15):
                raise ValueError(
                    "CatBoost scores_by_k must match the finite all_scores mean"
                )
            spread = float(np.std(finite_values))
            if k in scores_std_by_k and not np.isclose(
                spread,
                scores_std_by_k[k],
                rtol=1e-12,
                atol=1e-15,
            ):
                raise ValueError(
                    "CatBoost scores_std_by_k must match the finite all_scores "
                    "population standard deviation"
                )
            if finite_values.size >= 2:
                criterion_se[k] = spread / math.sqrt(int(finite_values.size) - 1)

    ks = sorted(scores_by_k)
    return pd.DataFrame(
        {
            "k": ks,
            "criterion": [scores_by_k[k] for k in ks],
            "criterion_se": [criterion_se[k] for k in ks],
            "selected": [k == best_k for k in ks],
        }
    )


def _catboost_known_features(
    *,
    selected: list[Any],
    features_by_k: Mapping[int, list[Any]],
    feature_importances: pd.Series,
    stability_scores: pd.Series | None,
    prefilter_features: list[Any] | None,
) -> list[Any]:
    known: list[Any] = []
    seen: set[str] = set()

    def extend(values: Iterable[Any]) -> None:
        for feature in values:
            key = _catboost_label_key(feature)
            if key not in seen:
                seen.add(key)
                known.append(feature)

    extend(selected)
    for k in sorted(features_by_k, reverse=True):
        extend(features_by_k[k])
    if stability_scores is not None:
        extend(stability_scores.index.tolist())
    extend(feature_importances.index.tolist())
    if prefilter_features is not None:
        extend(prefilter_features)
    return known


def _catboost_resolve_positions(
    known_features: list[Any],
    raw_features: list[Any],
) -> dict[str, int]:
    raw_keys = [_catboost_label_key(feature) for feature in raw_features]
    positions: dict[str, int] = {}
    for feature in known_features:
        key = _catboost_label_key(feature)
        matches = [index for index, candidate in enumerate(raw_keys) if candidate == key]
        if len(matches) != 1:
            raise ValueError(
                f"CatBoost feature {feature!r} is missing or ambiguous in input_features"
            )
        positions[key] = matches[0]
    return positions


def _as_catboost_result(result: Any, input_features: Any) -> SelectionView:
    selected = _catboost_feature_list(
        result.selected_features,
        label="CatBoost selected_features",
    )
    assert selected is not None
    if not selected:
        raise ValueError("CatBoost selected_features must be non-empty")
    best_k = _strict_integer(result.best_k, label="CatBoost best_k", minimum=1)
    if len(selected) > best_k:
        raise ValueError("CatBoost selected_features cannot contain more than best_k")

    scores_by_k = _catboost_score_mapping(
        result.scores_by_k,
        label="CatBoost scores_by_k",
    )
    if not scores_by_k:
        raise ValueError("CatBoost scores_by_k must contain at least one finite score")
    scores_std_by_k = _catboost_std_mapping(
        result.scores_std_by_k,
        score_keys=set(scores_by_k),
    )
    all_scores = _catboost_all_scores(result.all_scores)
    curve = _catboost_curve(
        scores_by_k=scores_by_k,
        scores_std_by_k=scores_std_by_k,
        all_scores=all_scores,
        best_k=best_k,
    )
    features_by_k = _catboost_features_by_k(result.features_by_k)
    feature_importances = _catboost_numeric_series(
        result.feature_importances,
        label="CatBoost feature_importances",
    )
    assert feature_importances is not None
    stability_scores = _catboost_numeric_series(
        result.stability_scores,
        label="CatBoost stability_scores",
        allow_none=True,
        unit_interval=True,
    )
    prefilter_features = _catboost_feature_list(
        result.prefilter_features,
        label="CatBoost prefilter_features",
        allow_none=True,
    )

    selected_keys = {_catboost_label_key(feature) for feature in selected}
    if best_k in features_by_k and not selected_keys.issubset(
        {_catboost_label_key(feature) for feature in features_by_k[best_k]}
    ):
        raise ValueError(
            "CatBoost selected_features must be contained in features_by_k[best_k]"
        )
    if not feature_importances.empty and selected_keys != {
        _catboost_label_key(feature) for feature in feature_importances.index
    }:
        raise ValueError(
            "non-empty CatBoost feature_importances must cover selected_features exactly"
        )
    if stability_scores is not None and not selected_keys.issubset(
        {_catboost_label_key(feature) for feature in stability_scores.index}
    ):
        raise ValueError(
            "CatBoost selected_features must be present in stability_scores"
        )

    if not isinstance(result.metric, str) or not result.metric:
        raise ValueError("CatBoost metric must be a non-empty string")
    if not isinstance(result.higher_is_better, (bool, np.bool_)):
        raise ValueError("CatBoost higher_is_better must be boolean")
    higher_is_better = bool(result.higher_is_better)
    selection_patience = _strict_integer(
        result.selection_patience,
        label="CatBoost selection_patience",
        minimum=1,
    )

    known_features = _catboost_known_features(
        selected=selected,
        features_by_k=features_by_k,
        feature_importances=feature_importances,
        stability_scores=stability_scores,
        prefilter_features=prefilter_features,
    )
    raw_features = _coerce_feature_names(input_features)
    if raw_features is None:
        selected_indices = None
        table_features = known_features
        selected_index = pd.array([pd.NA] * len(table_features), dtype="Int64")
        table_complete = False
        n_raw_features = None
        row_positions = {
            _catboost_label_key(feature): index
            for index, feature in enumerate(table_features)
        }
    else:
        raw_positions = _catboost_resolve_positions(known_features, raw_features)
        selected_indices = [
            raw_positions[_catboost_label_key(feature)] for feature in selected
        ]
        table_features = raw_features
        selected_index = pd.array(range(len(raw_features)), dtype="Int64")
        table_complete = True
        n_raw_features = len(raw_features)
        row_positions = raw_positions

    selected_ranks = {
        _catboost_label_key(feature): rank
        for rank, feature in enumerate(selected, start=1)
    }
    path_rank = pd.Series(pd.array([pd.NA] * len(table_features), dtype="Int64"))
    selected_mask = np.zeros(len(table_features), dtype=bool)
    for feature in selected:
        key = _catboost_label_key(feature)
        row = row_positions[key]
        path_rank.iloc[row] = selected_ranks[key]
        selected_mask[row] = True
    table = pd.DataFrame(
        {
            "feature": table_features,
            "selected_index": selected_index,
            "path_rank": path_rank,
            "selected": selected_mask,
        }
    )

    def add_metric(column: str, values: pd.Series) -> None:
        if values.empty:
            return
        metric_values = np.full(len(table), np.nan, dtype=np.float64)
        for feature, value in values.items():
            metric_values[row_positions[_catboost_label_key(feature)]] = float(value)
        table[column] = metric_values

    add_metric("gain", feature_importances)
    if stability_scores is not None:
        add_metric("selection_frequency", stability_scores)
    if prefilter_features is not None:
        prefiltered = {
            _catboost_label_key(feature) for feature in prefilter_features
        }
        table["prefiltered_first_split"] = [
            _catboost_label_key(feature) in prefiltered for feature in table_features
        ]

    if higher_is_better:
        best_scoring_k = min(
            scores_by_k,
            key=lambda k: (-scores_by_k[k], k),
        )
    else:
        best_scoring_k = min(
            scores_by_k,
            key=lambda k: (scores_by_k[k], k),
        )
    metadata = {
        "adapter": "CatBoostSelectionResult",
        "selector": "catboost",
        "curve_available": True,
        "criterion_direction": "maximize" if higher_is_better else "minimize",
        "criterion_se_definition": "population_std/sqrt(n_finite-1)",
        "metric": result.metric,
        "higher_is_better": higher_is_better,
        "target_k": best_k,
        "selected_feature_count": len(selected),
        "best_scoring_k": best_scoring_k,
        "best_scoring_score": scores_by_k[best_scoring_k],
        "gain_source": "final_model_feature_importance",
        "table_complete": table_complete,
        "input_kind": "unknown",
    }
    diagnostics = {
        "scores_std_by_k": scores_std_by_k,
        "all_scores": all_scores,
        "features_by_k": features_by_k,
        "stability_scores": stability_scores,
        "stability_scope": (
            "target_k_split_frequency" if stability_scores is not None else None
        ),
        "prefilter_features": prefilter_features,
        "prefilter_scope": (
            "first_split_only" if prefilter_features is not None else None
        ),
        "selection_patience": selection_patience,
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
