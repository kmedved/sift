"""Leakage-safe selector comparison."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
from typing import Any, Callable, Hashable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, KFold

from sift._metadata import resolve_row_metadata
from sift.scoring import (
    get_scoring,
    is_sklearn_scorer,
    score_with_sklearn_scorer,
    sklearn_scorer_label,
)
from sift.selection.reproducibility import (
    collapse_fold_snapshots,
    describe_estimator,
    describe_splitter,
)
from sift.selection.view import _columns_hash, _json_safe
from sift.selection.path_eval import (
    _accepts_keyword,
    _build_splits,
    _fit_estimator,
    _to_estimator,
)


SelectorFactory = Callable[[], Any]
CompareMode = Literal["cv", "in_sample_path"]
Task = Literal["regression", "classification"]

FOLDS_COLUMNS = (
    "split_id",
    "n_train",
    "n_val",
    "train_index_sha256",
    "val_index_sha256",
    "in_sample",
    "mode",
)
SCORE_COLUMNS = (
    "selector",
    "split_id",
    "score",
    "k",
    "k_unit",
    "n_raw_features",
    "n_blocks",
    "n_columns",
    "n_encoded_columns",
    "empty",
    "in_sample",
    "mode",
)
SUMMARY_COLUMNS = (
    "selector",
    "score_mean",
    "score_std",
    "mean_k",
    "k_unit",
    "n_empty",
    "n_splits",
    "in_sample",
    "mode",
)
FREQUENCY_COLUMNS = (
    "selector",
    "feature",
    "frequency",
    "n_folds",
    "selection_identity",
    "in_sample",
    "mode",
)
OVERLAP_COLUMNS = (
    "selector_a",
    "selector_b",
    "mean_jaccard",
    "selection_identity",
    "in_sample",
    "mode",
)
PREFIX_COLUMNS = (
    "selector",
    "split_id",
    "k",
    "score",
    "n_encoded_columns",
    "in_sample",
    "mode",
    "protocol",
)


def _fingerprint_indices(idx: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(idx, dtype=np.int64).reshape(-1))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _as_2d(X) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        return np.asarray(X, dtype=np.float64)
    arr = np.asarray(X)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return np.asarray(arr, dtype=np.float64)


def _n_rows(X) -> int:
    return int(X.shape[0])


def _slice_frame(X, idx: np.ndarray):
    if isinstance(X, pd.DataFrame):
        return X.iloc[np.asarray(idx, dtype=np.int64)]
    return np.asarray(X)[np.asarray(idx, dtype=np.int64)]


def _slice_1d(values, idx: np.ndarray):
    if values is None:
        return None
    arr = np.asarray(values)
    return arr[np.asarray(idx, dtype=np.int64)]


def _feature_names(X) -> list[Hashable]:
    if isinstance(X, pd.DataFrame):
        return list(X.columns)
    return [f"x{i}" for i in range(int(np.asarray(X).shape[1]))]


def _estimator_event(
    model_desc: dict[str, Any] | None,
    *,
    selector: str,
    split_id: int,
    scope: str,
    prefix_k: int | None = None,
) -> dict[str, Any]:
    event = {
        "selector": selector,
        "split_id": int(split_id),
        "scope": scope,
        "model": model_desc,
    }
    if prefix_k is not None:
        event["prefix_k"] = int(prefix_k)
    return event


def _collapse_estimator_snaps(snaps: list[dict[str, Any]]) -> dict[str, Any]:
    if not snaps:
        return {"status": "absent"}
    models = [item.get("model") for item in snaps]
    if all(item == models[0] for item in models):
        collapsed = dict(models[0] or {"status": "absent"})
        if len(snaps) > 1:
            collapsed["n_fits"] = len(snaps)
        return collapsed
    return {"status": "varies", "by_fit": list(snaps)}


def _fresh_selector(factory: SelectorFactory):
    produced = factory()
    if produced is None:
        raise TypeError("selector factories must return a selector instance")
    try:
        return clone(produced)
    except Exception:
        return copy.deepcopy(produced)


def _is_sift_fixed_k_filter(selector) -> bool:
    """SIFT filter wrappers reject unused groups/time at fixed k without within."""
    k = getattr(selector, "k", None)
    if k is None or k == "auto":
        return False
    if getattr(selector, "within", None) is not None:
        return False
    return hasattr(selector, "_selector_fn")


def _selector_accepts_row_context(selector, name: str) -> bool:
    if not _accepts_keyword(selector.fit, name):
        return False
    from sift.selectors import KnockoffSelector

    if isinstance(selector, KnockoffSelector):
        return False
    if _is_sift_fixed_k_filter(selector):
        return False
    return True


def _fit_kwargs(
    selector,
    *,
    sample_weight,
    groups,
    time,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if sample_weight is not None and _accepts_keyword(selector.fit, "sample_weight"):
        kwargs["sample_weight"] = sample_weight
    if groups is not None and _selector_accepts_row_context(selector, "groups"):
        kwargs["groups"] = groups
    if time is not None and _selector_accepts_row_context(selector, "time"):
        kwargs["time"] = time
    return kwargs


def _names_in(selector, feature_names: Sequence[Hashable]) -> list[Hashable]:
    fitted = getattr(selector, "feature_names_in_", None)
    if fitted is not None and len(np.asarray(fitted).reshape(-1)) == len(feature_names):
        return [name for name in np.asarray(fitted, dtype=object).reshape(-1)]
    return list(feature_names)


def _coerce_index_array(values) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values)
    if arr.size == 0:
        return np.empty(0, dtype=np.int64)
    if arr.dtype == object:
        if all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) for v in arr.reshape(-1)):
            return np.asarray(arr, dtype=np.int64).reshape(-1)
        return None
    if np.issubdtype(arr.dtype, np.integer) and not np.issubdtype(arr.dtype, np.bool_):
        return arr.astype(np.int64, copy=False).reshape(-1)
    return None


def _selected_indices_and_names(
    selector, feature_names: Sequence[Hashable]
) -> tuple[np.ndarray, list[Hashable]]:
    names_in = _names_in(selector, feature_names)
    if hasattr(selector, "get_support"):
        try:
            idx = np.asarray(selector.get_support(indices=True), dtype=np.int64).reshape(-1)
            return idx, [names_in[int(i)] for i in idx]
        except Exception:
            try:
                mask = np.asarray(selector.get_support(), dtype=bool).reshape(-1)
                idx = np.flatnonzero(mask).astype(np.int64)
                return idx, [names_in[int(i)] for i in idx]
            except Exception:
                pass
    named = getattr(selector, "selected_feature_names_", None)
    if named is not None:
        raw_names = list(named)
        index_map = {name: i for i, name in enumerate(names_in)}
        idx = np.asarray([index_map[name] for name in raw_names if name in index_map], dtype=np.int64)
        return idx, [name for name in raw_names if name in index_map]
    selected = getattr(selector, "selected_features_", None)
    indices = getattr(selector, "selected_indices_", None)
    as_index = _coerce_index_array(selected)
    if as_index is not None:
        return as_index, [names_in[int(i)] for i in as_index]
    if selected is not None:
        raw_names = list(selected)
        index_map = {name: i for i, name in enumerate(names_in)}
        idx = np.asarray(
            [index_map[name] for name in raw_names if name in index_map],
            dtype=np.int64,
        )
        return idx, [name for name in raw_names if name in index_map]
    as_index = _coerce_index_array(indices)
    if as_index is not None:
        return as_index, [names_in[int(i)] for i in as_index]
    return np.empty(0, dtype=np.int64), []


def _map_names(values, names_in: Sequence[Hashable]) -> list[Hashable]:
    index_map = {name: i for i, name in enumerate(names_in)}
    return [name for name in list(values) if name in index_map]


def _path_feature_names(
    selector, report: Mapping[str, Any], feature_names: Sequence[Hashable]
) -> list[Hashable]:
    """Learned discovery order, independent of transform ``output_order``.

    SIFT filters expose path order on ``selected_indices_`` and raw labels on
    ``selected_features_``. Stability exposes positions on ``selected_features_``
    and the named path on ``selected_feature_names_``. Integer raw labels are
    not column positions.
    """
    names_in = _names_in(selector, feature_names)
    indices = getattr(selector, "selected_indices_", None)
    as_index = _coerce_index_array(indices)
    if as_index is not None and as_index.size:
        return [names_in[int(i)] for i in as_index if 0 <= int(i) < len(names_in)]
    named = getattr(selector, "selected_feature_names_", None)
    if named is not None:
        mapped = _map_names(named, names_in)
        if mapped:
            return mapped
    selected = getattr(selector, "selected_features_", None)
    if selected is not None:
        mapped = _map_names(selected, names_in)
        if mapped:
            return mapped
    return list(report["features"])


def _include_names(selector, feature_names: Sequence[Hashable]) -> list[Hashable]:
    include = getattr(selector, "include", None)
    if not include:
        return []
    names_in = list(feature_names)
    index_map = {name: i for i, name in enumerate(names_in)}
    out: list[Hashable] = []
    for item in include:
        if item in index_map:
            out.append(item)
        elif isinstance(item, (int, np.integer)) and not isinstance(item, (bool, np.bool_)):
            idx = int(item)
            if 0 <= idx < len(names_in):
                out.append(names_in[idx])
    return out


def _resolved_blocks(selector, feature_names: Sequence[Hashable]):
    mapping = getattr(selector, "feature_blocks", None)
    if mapping is None:
        return None
    from sift.selection.blocks import resolve_feature_blocks

    try:
        return resolve_feature_blocks(mapping, feature_names=list(feature_names), named=True)
    except Exception:
        return None


def _block_metadata(selector, indices: np.ndarray, feature_names: Sequence[Hashable]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    stored = getattr(selector, "selector_metadata_", None)
    if stored:
        metadata.update(stored)
    result = getattr(selector, "result_", None)
    if result is not None:
        metadata.update(getattr(result, "selector_metadata", None) or {})
    if metadata.get("feature_blocks"):
        return metadata
    blocks = _resolved_blocks(selector, feature_names)
    if blocks is None or blocks.all_singletons():
        return metadata
    from sift.selection.blocks import block_result_metadata
    from sift.selection.conditioning import resolve_conditioning

    include = getattr(selector, "include", None)
    resolved = None
    if include:
        try:
            resolved = resolve_conditioning(
                include, None, None, feature_names=list(feature_names), named=True, k=1
            )
        except Exception:
            resolved = None
    include_idx = () if resolved is None else resolved.include
    computed = block_result_metadata(
        blocks,
        [int(i) for i in indices],
        include_idx,
        n_columns_selected=int(len(indices)),
    )
    metadata.update(computed)
    return metadata


def _selection_report(selector, feature_names: Sequence[Hashable]) -> dict[str, Any]:
    indices, names = _selected_indices_and_names(selector, feature_names)
    metadata = _block_metadata(selector, indices, feature_names)
    n_raw = len(names)
    n_blocks = int(metadata["n_blocks_selected"]) if "n_blocks_selected" in metadata else n_raw
    n_columns = (
        int(metadata["n_columns_selected"]) if "n_columns_selected" in metadata else n_raw
    )
    if metadata.get("feature_blocks"):
        k = n_blocks
        k_unit = "additional_blocks"
    else:
        k = n_raw
        k_unit = "raw_features"
    return {
        "features": names,
        "indices": indices,
        "k": int(k),
        "k_unit": k_unit,
        "n_raw_features": n_raw,
        "n_blocks": n_blocks,
        "n_columns": n_columns,
        "empty": n_raw == 0,
    }


def _empty_design(n_rows: int) -> np.ndarray:
    return np.zeros((int(n_rows), 0), dtype=np.float64)


def _empty_predictor(*, task: Task):
    if task == "classification":
        return DummyClassifier(strategy="prior")
    return DummyRegressor(strategy="mean")


def _score_model(
    model,
    X,
    y,
    sample_weight,
    scoring,
    *,
    sample_weight_supplied: bool,
) -> float:
    if X is None:
        X_arr = _empty_design(len(np.asarray(y).reshape(-1)))
    else:
        arr = np.asarray(X)
        if arr.ndim == 2 and arr.shape[1] == 0:
            X_arr = arr
        else:
            X_arr = _as_2d(X)
    y_arr = np.asarray(y).reshape(-1)
    if is_sklearn_scorer(scoring):
        return float(
            score_with_sklearn_scorer(
                scoring,
                model,
                X_arr,
                y_arr,
                sample_weight=sample_weight if sample_weight_supplied else None,
            )
        )
    weights = (
        np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if sample_weight is not None
        else np.ones(len(y_arr), dtype=np.float64)
    )
    spec = get_scoring(scoring)
    return float(spec(model, X_arr, y_arr, weights))


def _resolve_scoring(scoring, *, task: Task):
    if scoring is None:
        return ("accuracy" if task == "classification" else "r2", True)
    if is_sklearn_scorer(scoring):
        return (scoring, True)
    if isinstance(scoring, str):
        spec = get_scoring(scoring)
        return (scoring, bool(spec.higher_is_better))
    raise TypeError(
        "scoring must be None, a SIFT scoring name, or an sklearn scorer object"
    )


def _scoring_label(scoring) -> str:
    if is_sklearn_scorer(scoring):
        return sklearn_scorer_label(scoring)
    return str(scoring)


def _jaccard(left: set, right: set) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 1.0
    return float(len(left & right) / len(union))


def _as_selected_matrix(X):
    if getattr(X, "ndim", 1) == 1:
        return np.asarray(X).reshape(-1, 1)
    return X


def _fit_transform_selected(selector, X, y, kwargs):
    if hasattr(selector, "fit_transform"):
        try:
            return _as_selected_matrix(selector.fit_transform(X, y, **kwargs))
        except TypeError:
            pass
    selector.fit(X, y, **kwargs)
    return _as_selected_matrix(selector.transform(X))


def _score_empty(
    *,
    y_tr,
    y_va,
    w_tr,
    w_va,
    scoring,
    task: Task,
    sample_weight_supplied: bool,
) -> tuple[float, int, dict[str, Any] | None]:
    model = _empty_predictor(task=task)
    X_tr = _empty_design(len(np.asarray(y_tr).reshape(-1)))
    X_va = _empty_design(len(np.asarray(y_va).reshape(-1)))
    fit_kwargs = {}
    if w_tr is not None and _accepts_keyword(model.fit, "sample_weight"):
        fit_kwargs["sample_weight"] = w_tr
    model.fit(X_tr, np.asarray(y_tr).reshape(-1), **fit_kwargs)
    score = _score_model(
        model,
        X_va,
        y_va,
        w_va,
        scoring,
        sample_weight_supplied=sample_weight_supplied,
    )
    return score, 0, describe_estimator(model)


def _score_matrices(
    *,
    X_tr_sel,
    X_va_sel,
    y_tr,
    y_va,
    w_tr,
    w_va,
    estimator,
    estimator_factory,
    scoring,
    sample_weight_supplied: bool,
    task: Task = "regression",
) -> tuple[float, int, dict[str, Any] | None]:
    n_encoded = int(np.asarray(X_tr_sel).shape[1]) if np.asarray(X_tr_sel).ndim == 2 else 1
    if n_encoded == 0:
        return _score_empty(
            y_tr=y_tr,
            y_va=y_va,
            w_tr=w_tr,
            w_va=w_va,
            scoring=scoring,
            task=task,
            sample_weight_supplied=sample_weight_supplied,
        )
    model = _to_estimator(estimator=estimator, estimator_factory=estimator_factory)
    description = describe_estimator(model)
    _fit_estimator(
        model,
        _as_2d(X_tr_sel),
        np.asarray(y_tr).reshape(-1),
        np.ones(len(y_tr), dtype=np.float64) if w_tr is None else np.asarray(w_tr, dtype=np.float64),
    )
    score = _score_model(
        model,
        X_va_sel,
        y_va,
        w_va,
        scoring,
        sample_weight_supplied=sample_weight_supplied,
    )
    return score, n_encoded, description


def _cluster_discovery(selector, discovery: Sequence[Hashable], feature_names: Sequence[Hashable]):
    blocks = _resolved_blocks(selector, feature_names)
    if blocks is None:
        return [[name] for name in discovery]
    names = list(feature_names)
    index = {name: i for i, name in enumerate(names)}
    member_block: dict[Hashable, int | None] = {}
    by_block: dict[int, list[Hashable]] = {}
    for name in discovery:
        pos = index.get(name)
        bidx = None if pos is None else int(blocks.column_to_block[int(pos)])
        member_block[name] = bidx
        if bidx is not None:
            by_block.setdefault(bidx, []).append(name)
    clusters: list[list[Hashable]] = []
    seen_blocks: set[int] = set()
    for name in discovery:
        bidx = member_block[name]
        if bidx is None:
            clusters.append([name])
            continue
        if bidx in seen_blocks:
            continue
        seen_blocks.add(bidx)
        clusters.append(list(by_block[bidx]))
    return clusters


def _raw_prefixes(selector, report: dict[str, Any], feature_names: Sequence[Hashable]):
    if report["empty"]:
        return [(0, [])]
    path = _path_feature_names(selector, report, feature_names)
    ordered: list[Hashable] = []
    seen: set[Hashable] = set()
    for name in list(path) + list(report["features"]):
        if name in seen:
            continue
        ordered.append(name)
        seen.add(name)
    include = set(_include_names(selector, feature_names))
    base = [name for name in ordered if name in include]
    discovery = [name for name in ordered if name not in include]
    clusters = _cluster_discovery(selector, discovery, feature_names)
    prefixes: list[tuple[int, list[Hashable]]] = []
    acc = list(base)
    if not clusters:
        prefixes.append((int(report["k"]), list(acc if acc else ordered)))
        return prefixes
    for step, cluster in enumerate(clusters, start=1):
        acc.extend(cluster)
        prefixes.append((step, list(acc)))
    return prefixes


def _encoded_column_index(selector, raw_prefix: Sequence[Hashable], encoded_names: Sequence[Hashable]):
    wanted: list[Hashable]
    encoder = getattr(selector, "categorical_encoder_", None)
    if raw_prefix and hasattr(encoder, "expand_selected"):
        wanted = list(encoder.expand_selected(list(raw_prefix)))
    else:
        wanted = list(raw_prefix)
    wanted_set = set(wanted)
    if encoded_names:
        idx = [i for i, name in enumerate(encoded_names) if name in wanted_set]
        if idx:
            return np.asarray(idx, dtype=np.int64)
    widths = getattr(selector, "_encoded_prefix_widths_", None)
    if widths and raw_prefix:
        n_steps = max(1, len(_cluster_discovery(
            selector,
            [n for n in raw_prefix if n not in set(_include_names(selector, list(getattr(selector, "feature_names_in_", [])) or []))],
            list(getattr(selector, "feature_names_in_", []) or []),
        )))
        width = int(widths[min(n_steps, len(widths)) - 1])
        return np.arange(width, dtype=np.int64)
    return np.arange(len(wanted), dtype=np.int64)


def _slice_columns(matrix, idx: np.ndarray):
    if isinstance(matrix, pd.DataFrame):
        if idx.size == 0:
            return matrix.iloc[:, []]
        return matrix.iloc[:, idx]
    arr = np.asarray(matrix)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if idx.size == 0:
        return arr[:, :0]
    return arr[:, idx]


@dataclass(frozen=True)
class CompareResult:
    """Leakage-safe comparison of selector factories.

    Default ``mode="cv"`` refits each factory inside every training fold and
    scores a fresh downstream estimator on the untouched validation fold.
    ``mode="in_sample_path"`` first selects on the full sample and then scores
    prefixes of that path; every output is labelled in-sample.

    Attributes
    ----------
    mode : {'cv', 'in_sample_path'}
        Comparison protocol.
    in_sample : bool
        ``True`` only for ``mode="in_sample_path"``.
    scoring : str
        Scoring label. SIFT named scorers and sklearn scorers keep their
        native higher-is-better convention.
    higher_is_better : bool
        Direction of ``scoring``.
    k_unit : str
        Unit of reported ``k``: ``raw_features``, ``additional_blocks``, or
        ``mixed`` when compared selectors disagree.
    selection_identity : str
        Namespace of selection-frequency labels. Always ``raw_features``.
    folds : DataFrame
        One row per split with ``split_id``, ``n_train``, ``n_val``,
        ``train_index_sha256``, and ``val_index_sha256``.
    scores : DataFrame
        Per-selector, per-split scores with ``k``, ``k_unit``, empty-selection
        flag, encoded transform width, and ``in_sample``.
    summary : DataFrame
        Score mean/std, mean ``k`` in ``k_unit``, and empty-selection counts.
    selection_frequency : DataFrame
        Raw-feature selection rate across folds (or the single full-sample
        path when ``in_sample``).
    overlap : DataFrame
        Mean per-fold Jaccard overlap of raw selected sets.
    prefix_scores : DataFrame
        In-sample prefix curve; empty in ``mode="cv"``. Every row sets
        ``in_sample=True`` and ``mode="in_sample_path"``.
    fold_bookkeeping : tuple of dict
        Split fingerprints reusable by later reproducibility manifests.
    diagnostics : dict
        Extra protocol metadata, including ``in_sample`` and ``mode``.

    Methods
    -------
    to_dict()
        JSON-serializable snapshot of labels, tables, and fold fingerprints
        using the established SelectionView converters. Not a provenance
        manifest; use ``reproducibility_`` for that.
    reproducibility_(X=None, hash_data=False)
        JSON-safe provenance manifest. Fold fingerprints are the stored
        compare-time bookkeeping; environment is export-time.

    See Also
    --------
    compare : Produces this object.
    evaluate_feature_path : Explicit prefix evaluation of a caller-supplied path.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import Ridge
    >>> from sklearn.model_selection import KFold
    >>> from sift import CEFSPlusSelector, compare
    >>> rng = np.random.default_rng(1)
    >>> X = rng.normal(size=(80, 5))
    >>> y = X[:, 0] + 0.2 * rng.normal(size=80)
    >>> out = compare(
    ...     {"cefs": lambda: CEFSPlusSelector(k=1, verbose=False)},
    ...     X, y, estimator=Ridge(),
    ...     cv=KFold(n_splits=2, shuffle=True, random_state=0),
    ... )
    >>> out.in_sample, out.selection_identity
    (False, 'raw_features')
    """

    mode: str
    in_sample: bool
    scoring: str
    higher_is_better: bool
    k_unit: str
    selection_identity: str
    folds: pd.DataFrame
    scores: pd.DataFrame
    summary: pd.DataFrame
    selection_frequency: pd.DataFrame
    overlap: pd.DataFrame
    prefix_scores: pd.DataFrame
    fold_bookkeeping: tuple[dict[str, Any], ...]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot without changing this result."""
        return _json_safe(
            {
                "mode": self.mode,
                "in_sample": self.in_sample,
                "scoring": self.scoring,
                "higher_is_better": self.higher_is_better,
                "k_unit": self.k_unit,
                "selection_identity": self.selection_identity,
                "folds": self.folds,
                "scores": self.scores,
                "summary": self.summary,
                "selection_frequency": self.selection_frequency,
                "overlap": self.overlap,
                "prefix_scores": self.prefix_scores,
                "fold_bookkeeping": [dict(item) for item in self.fold_bookkeeping],
                "diagnostics": dict(self.diagnostics),
            }
        )

    def reproducibility_(self, *, X=None, hash_data: bool = False) -> dict[str, Any]:
        """Return a JSON-safe reproducibility manifest for this comparison.

        Per-fold split fingerprints reuse ``fold_bookkeeping``. Instantiated
        selector, estimator, and splitter snapshots are compare-time.
        Package versions, BLAS identity, and git commit are labelled
        export-time. ``X`` is never retained; data hashing is opt-in.

        Parameters
        ----------
        X : DataFrame or ndarray, optional
            Caller-supplied matrix used only for opt-in hashing or unknown
            shape. Not retained.
        hash_data : bool, default False
            If True, hash ``X``. Raises if ``X`` is omitted.

        Returns
        -------
        dict
            Schema ``"1"`` payload with compare-time folds and export-time
            environment. Safe for ``json.dumps``.
        """
        from sift.selection.reproducibility import manifest_from_compare

        return manifest_from_compare(self, X=X, hash_data=hash_data)


def compare(
    selectors: Mapping[str, SelectorFactory],
    X,
    y,
    *,
    estimator=None,
    estimator_factory: Callable[[], Any] | None = None,
    cv=None,
    scoring=None,
    groups=None,
    time=None,
    sample_weight=None,
    mode: CompareMode = "cv",
    task: Task = "regression",
    random_state: int = 0,
    val_frac: float = 0.2,
) -> CompareResult:
    """Compare selector factories with fold-local selection and scoring.

    Default ``mode="cv"`` calls each factory inside every training fold,
    fits a fresh downstream estimator on the training selected matrix, and
    scores the untouched validation fold. Held-out rows never enter training
    encoders, preprocessing, selection, or estimator fit. Empty selected
    sets score an intercept-only predictor; they are not filled with extra
    columns and are not treated as failures.

    ``mode="in_sample_path"`` fits each selector once on the full sample,
    then scores prefixes of that path on the same CV splits. Every returned
    table and diagnostic is labelled in-sample.

    Parameters
    ----------
    selectors : mapping of str to callable
        Factories that return a new unfitted selector. Each fold (or the
        single full-sample fit in ``in_sample_path``) calls the factory and
        clones the result.
    X : DataFrame or ndarray
        Feature matrix.
    y : array-like
        Target aligned with ``X``.
    estimator : estimator, optional
        Downstream template cloned per fold. Mutually exclusive with
        ``estimator_factory``. Default is ``Ridge()`` for regression and
        ``LogisticRegression()`` for classification.
    estimator_factory : callable, optional
        Called to build one downstream estimator per fold.
    cv : int, splitter, or None, default None
        Fold specification. ``None`` uses ``GroupKFold(5)`` when ``groups``
        is supplied and shuffled ``KFold(5)`` otherwise. An integer ``n`` is
        ``KFold(n)`` (or ``GroupKFold(n)`` with groups).
    scoring : str, sklearn scorer, or None, default None
        Scoring from ``sift.scoring`` names or an sklearn scorer object.
        Sklearn scorer outputs follow the maximize convention, so
        ``higher_is_better`` is True for those objects. ``None`` is ``r2``
        for regression and ``accuracy`` for classification.
    groups : array-like or str, optional
        Split groups, and selector ``fit`` groups when the fitted object
        accepts them. SIFT fixed-k filter wrappers and ``KnockoffSelector``
        do not receive ``groups``; Stability and other accepting selectors do.
    time : array-like or str, optional
        Selector ``fit`` time only when accepted. Not used to invent a
        time-series splitter.
    sample_weight : array-like, optional
        Row weights sliced per train/validation fold and consumed by
        selectors, estimators, and scorers that accept them.
    mode : {'cv', 'in_sample_path'}, default 'cv'
        Leakage-safe nested comparison, or the labelled in-sample prefix
        diagnostic.
    task : {'regression', 'classification'}, default 'regression'
        Downstream predictor family and default scorer.
    random_state : int, default 0
        Shuffle seed for default ``KFold``.
    val_frac : float, default 0.2
        Holdout fraction only when ``cv`` is a single pair constructed via
        the path-evaluation splitter helper with ``splitter=None``. Unused
        for sklearn ``KFold``/``GroupKFold`` objects.

    Returns
    -------
    CompareResult
        Score distributions, explicit-unit mean ``k``, selection-frequency
        overlap, fold fingerprints, and (when requested) in-sample prefix
        scores.

    See Also
    --------
    CompareResult : Returned comparison object.
    evaluate_feature_path : Prefix evaluation of an already chosen path.
    CEFSPlusSelector : Ordered filter used in the campaign example.
    KnockoffSelector : q-calibrated alternative; empty sets are valid.

    Notes
    -----
    Selection identity stays in the raw feature namespace, including E4
    one-hot blocks. ``k`` is ``len(selected_features_)`` unless the fitted
    metadata reports ``feature_blocks``, in which case mean ``k`` uses
    additional-block units and ``n_columns`` remains the raw width.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import Ridge
    >>> from sklearn.model_selection import KFold
    >>> from sift import CEFSPlusSelector, compare
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(90, 6))
    >>> y = 2.0 * X[:, 0] + 0.2 * rng.normal(size=90)
    >>> result = compare(
    ...     {"cefs": lambda: CEFSPlusSelector(k=2, verbose=False)},
    ...     X, y, estimator=Ridge(),
    ...     cv=KFold(n_splits=3, shuffle=True, random_state=0),
    ... )
    >>> result.mode, result.in_sample
    ('cv', False)
    >>> bool(np.isfinite(result.summary["score_mean"].iloc[0]))
    True
    """
    if not isinstance(selectors, Mapping) or not selectors:
        raise TypeError("selectors must be a non-empty mapping of name -> factory")
    for name, factory in selectors.items():
        if not isinstance(name, str) or not name:
            raise ValueError("selector names must be non-empty strings")
        if not callable(factory):
            raise TypeError(f"selectors[{name!r}] must be a callable factory")
    if mode not in {"cv", "in_sample_path"}:
        raise ValueError("mode must be 'cv' or 'in_sample_path'")
    if task not in {"regression", "classification"}:
        raise ValueError("task must be 'regression' or 'classification'")
    if estimator is not None and estimator_factory is not None:
        raise ValueError("Pass either estimator or estimator_factory, not both")
    if estimator is None and estimator_factory is None:
        estimator = LogisticRegression(max_iter=200) if task == "classification" else Ridge()

    metadata = resolve_row_metadata(
        X, groups=groups, time=time, sample_weight=sample_weight
    )
    X = metadata.X
    groups = metadata.groups
    time = metadata.time
    sample_weight = metadata.sample_weight
    y_arr = np.asarray(y).reshape(-1)
    n = _n_rows(X)
    if y_arr.shape[0] != n:
        raise ValueError(f"X has {n} rows but y has {y_arr.shape[0]}")
    names = _feature_names(X)
    scoring_obj, higher_is_better = _resolve_scoring(scoring, task=task)
    scoring_name = _scoring_label(scoring_obj)
    sample_weight_supplied = sample_weight is not None
    splitter = _resolve_cv(cv, groups=groups, random_state=random_state)
    split_source = (
        "caller"
        if cv is not None
        and not (isinstance(cv, (int, np.integer)) and not isinstance(cv, (bool, np.bool_)))
        else "resolved"
    )
    split_desc = describe_splitter(splitter)
    split_desc["source"] = split_source
    split_desc["uses_compare_random_state"] = bool(
        split_source == "resolved"
        and getattr(splitter, "shuffle", False)
        and getattr(splitter, "random_state", None) is not None
    )
    splits = _build_splits(
        n,
        splitter,
        random_state=random_state,
        val_frac=val_frac,
        groups=None if groups is None else np.asarray(groups).reshape(-1),
        y=y_arr,
    )
    fold_rows = []
    bookkeeping = []
    for split_id, (train_idx, val_idx) in enumerate(splits):
        train_idx = np.asarray(train_idx, dtype=np.int64)
        val_idx = np.asarray(val_idx, dtype=np.int64)
        record = {
            "split_id": int(split_id),
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "train_index_sha256": _fingerprint_indices(train_idx),
            "val_index_sha256": _fingerprint_indices(val_idx),
            "in_sample": mode == "in_sample_path",
            "mode": mode,
        }
        fold_rows.append(record)
        bookkeeping.append(dict(record))
    folds = pd.DataFrame(fold_rows)

    if mode == "in_sample_path":
        return _compare_in_sample_path(
            selectors=selectors,
            X=X,
            y_arr=y_arr,
            names=names,
            estimator=estimator,
            estimator_factory=estimator_factory,
            splits=splits,
            folds=folds,
            bookkeeping=tuple(bookkeeping),
            scoring_obj=scoring_obj,
            scoring_name=scoring_name,
            higher_is_better=higher_is_better,
            task=task,
            groups=groups,
            time=time,
            sample_weight=sample_weight,
            sample_weight_supplied=sample_weight_supplied,
            n_rows=n,
            random_state=random_state,
            input_kind="dataframe" if isinstance(X, pd.DataFrame) else "positional",
            split=split_desc,
        )
    return _compare_cv(
        selectors=selectors,
        X=X,
        y_arr=y_arr,
        names=names,
        estimator=estimator,
        estimator_factory=estimator_factory,
        splits=splits,
        folds=folds,
        bookkeeping=tuple(bookkeeping),
        scoring_obj=scoring_obj,
        scoring_name=scoring_name,
        higher_is_better=higher_is_better,
        task=task,
        groups=groups,
        time=time,
        sample_weight=sample_weight,
        sample_weight_supplied=sample_weight_supplied,
        n_rows=n,
        random_state=random_state,
        input_kind="dataframe" if isinstance(X, pd.DataFrame) else "positional",
        split=split_desc,
    )


def _resolve_cv(cv, *, groups, random_state: int):
    if cv is None:
        if groups is not None:
            return GroupKFold(n_splits=5)
        return KFold(n_splits=5, shuffle=True, random_state=int(random_state))
    if isinstance(cv, (int, np.integer)) and not isinstance(cv, (bool, np.bool_)):
        n_splits = int(cv)
        if n_splits < 2:
            raise ValueError("cv integer must be >= 2")
        if groups is not None:
            return GroupKFold(n_splits=n_splits)
        return KFold(n_splits=n_splits, shuffle=True, random_state=int(random_state))
    return cv


def _compare_cv(
    *,
    selectors,
    X,
    y_arr,
    names,
    estimator,
    estimator_factory,
    splits,
    folds,
    bookkeeping,
    scoring_obj,
    scoring_name,
    higher_is_better,
    task,
    groups,
    time,
    sample_weight,
    sample_weight_supplied,
    n_rows,
    random_state,
    input_kind,
    split,
) -> CompareResult:
    score_rows: list[dict[str, Any]] = []
    selected_by: dict[str, list[list[Hashable]]] = {name: [] for name in selectors}
    selector_snaps: dict[str, list[dict[str, Any]]] = {name: [] for name in selectors}
    estimator_snaps: list[dict[str, Any]] = []
    k_units: set[str] = set()
    for split_id, (train_idx, val_idx) in enumerate(splits):
        train_idx = np.asarray(train_idx, dtype=np.int64)
        val_idx = np.asarray(val_idx, dtype=np.int64)
        X_tr = _slice_frame(X, train_idx)
        X_va = _slice_frame(X, val_idx)
        y_tr = y_arr[train_idx]
        y_va = y_arr[val_idx]
        w_tr = _slice_1d(sample_weight, train_idx)
        w_va = _slice_1d(sample_weight, val_idx)
        g_tr = _slice_1d(groups, train_idx)
        t_tr = _slice_1d(time, train_idx)
        for sel_name, factory in selectors.items():
            selector = _fresh_selector(factory)
            selector_snaps[sel_name].append(describe_estimator(selector))
            fit_kwargs = _fit_kwargs(
                selector, sample_weight=w_tr, groups=g_tr, time=t_tr
            )
            X_tr_sel = _fit_transform_selected(selector, X_tr, y_tr, fit_kwargs)
            report = _selection_report(selector, names)
            k_units.add(report["k_unit"])
            selected_by[sel_name].append(list(report["features"]))
            if report["empty"] or int(np.asarray(X_tr_sel).shape[1]) == 0:
                score, n_encoded, est_desc = _score_empty(
                    y_tr=y_tr,
                    y_va=y_va,
                    w_tr=w_tr,
                    w_va=w_va,
                    scoring=scoring_obj,
                    task=task,
                    sample_weight_supplied=sample_weight_supplied,
                )
            else:
                X_va_sel = _as_selected_matrix(selector.transform(X_va))
                score, n_encoded, est_desc = _score_matrices(
                    X_tr_sel=X_tr_sel,
                    X_va_sel=X_va_sel,
                    y_tr=y_tr,
                    y_va=y_va,
                    w_tr=w_tr,
                    w_va=w_va,
                    estimator=estimator,
                    estimator_factory=estimator_factory,
                    scoring=scoring_obj,
                    sample_weight_supplied=sample_weight_supplied,
                    task=task,
                )
            estimator_snaps.append(
                _estimator_event(
                    est_desc,
                    selector=sel_name,
                    split_id=int(split_id),
                    scope="main",
                )
            )
            score_rows.append(
                {
                    "selector": sel_name,
                    "split_id": int(split_id),
                    "score": score,
                    "k": report["k"],
                    "k_unit": report["k_unit"],
                    "n_raw_features": report["n_raw_features"],
                    "n_blocks": report["n_blocks"],
                    "n_columns": report["n_columns"],
                    "n_encoded_columns": int(n_encoded),
                    "empty": bool(report["empty"]),
                    "in_sample": False,
                    "mode": "cv",
                }
            )
    return _assemble_result(
        mode="cv",
        in_sample=False,
        scoring_name=scoring_name,
        higher_is_better=higher_is_better,
        k_unit=("mixed" if len(k_units) > 1 else next(iter(k_units), "raw_features")),
        names=names,
        folds=folds,
        bookkeeping=bookkeeping,
        score_rows=score_rows,
        selected_by=selected_by,
        prefix_rows=[],
        n_splits=len(splits),
        n_rows=n_rows,
        random_state=random_state,
        input_kind=input_kind,
        split=split,
        selectors_config=collapse_fold_snapshots(selector_snaps),
        estimator_config=_collapse_estimator_snaps(estimator_snaps),
        configured_estimator=(
            describe_estimator(estimator)
            if estimator is not None
            else {"status": "factory"}
        ),
    )


def _compare_in_sample_path(
    *,
    selectors,
    X,
    y_arr,
    names,
    estimator,
    estimator_factory,
    splits,
    folds,
    bookkeeping,
    scoring_obj,
    scoring_name,
    higher_is_better,
    task,
    groups,
    time,
    sample_weight,
    sample_weight_supplied,
    n_rows,
    random_state,
    input_kind,
    split,
) -> CompareResult:
    score_rows: list[dict[str, Any]] = []
    prefix_rows: list[dict[str, Any]] = []
    selected_by: dict[str, list[list[Hashable]]] = {name: [] for name in selectors}
    selector_snaps: dict[str, list[dict[str, Any]]] = {name: [] for name in selectors}
    estimator_snaps: list[dict[str, Any]] = []
    k_units: set[str] = set()
    fitted: dict[str, Any] = {}
    fitted_reports: dict[str, dict[str, Any]] = {}
    for sel_name, factory in selectors.items():
        selector = _fresh_selector(factory)
        selector_snaps[sel_name].append(describe_estimator(selector))
        selector.fit(
            X,
            y_arr,
            **_fit_kwargs(
                selector, sample_weight=sample_weight, groups=groups, time=time
            ),
        )
        report = _selection_report(selector, names)
        k_units.add(report["k_unit"])
        fitted[sel_name] = selector
        fitted_reports[sel_name] = report
        selected_by[sel_name].append(list(report["features"]))

    for split_id, (train_idx, val_idx) in enumerate(splits):
        train_idx = np.asarray(train_idx, dtype=np.int64)
        val_idx = np.asarray(val_idx, dtype=np.int64)
        X_tr = _slice_frame(X, train_idx)
        X_va = _slice_frame(X, val_idx)
        y_tr = y_arr[train_idx]
        y_va = y_arr[val_idx]
        w_tr = _slice_1d(sample_weight, train_idx)
        w_va = _slice_1d(sample_weight, val_idx)
        for sel_name, selector in fitted.items():
            report = fitted_reports[sel_name]
            if report["empty"]:
                score, n_encoded, est_desc = _score_empty(
                    y_tr=y_tr,
                    y_va=y_va,
                    w_tr=w_tr,
                    w_va=w_va,
                    scoring=scoring_obj,
                    task=task,
                    sample_weight_supplied=sample_weight_supplied,
                )
                prefixes = [(0, np.empty(0, dtype=np.int64))]
                X_tr_full = _empty_design(len(y_tr))
                X_va_full = _empty_design(len(y_va))
            else:
                X_tr_full = _as_selected_matrix(selector.transform(X_tr))
                X_va_full = _as_selected_matrix(selector.transform(X_va))
                encoded_names = []
                if hasattr(selector, "get_feature_names_out"):
                    try:
                        encoded_names = [name for name in selector.get_feature_names_out()]
                    except Exception:
                        encoded_names = []
                prefixes = []
                for step, raw_prefix in _raw_prefixes(selector, report, names):
                    col_idx = _encoded_column_index(selector, raw_prefix, encoded_names)
                    prefixes.append((step, col_idx))
                score, n_encoded, est_desc = _score_matrices(
                    X_tr_sel=X_tr_full,
                    X_va_sel=X_va_full,
                    y_tr=y_tr,
                    y_va=y_va,
                    w_tr=w_tr,
                    w_va=w_va,
                    estimator=estimator,
                    estimator_factory=estimator_factory,
                    scoring=scoring_obj,
                    sample_weight_supplied=sample_weight_supplied,
                    task=task,
                )
            estimator_snaps.append(
                _estimator_event(
                    est_desc,
                    selector=sel_name,
                    split_id=int(split_id),
                    scope="main",
                )
            )
            score_rows.append(
                {
                    "selector": sel_name,
                    "split_id": int(split_id),
                    "score": score,
                    "k": report["k"],
                    "k_unit": report["k_unit"],
                    "n_raw_features": report["n_raw_features"],
                    "n_blocks": report["n_blocks"],
                    "n_columns": report["n_columns"],
                    "n_encoded_columns": int(n_encoded),
                    "empty": bool(report["empty"]),
                    "in_sample": True,
                    "mode": "in_sample_path",
                }
            )
            for prefix_k, col_idx in prefixes:
                if col_idx.size == 0:
                    p_score, p_encoded, p_est = _score_empty(
                        y_tr=y_tr,
                        y_va=y_va,
                        w_tr=w_tr,
                        w_va=w_va,
                        scoring=scoring_obj,
                        task=task,
                        sample_weight_supplied=sample_weight_supplied,
                    )
                else:
                    p_score, p_encoded, p_est = _score_matrices(
                        X_tr_sel=_slice_columns(X_tr_full, col_idx),
                        X_va_sel=_slice_columns(X_va_full, col_idx),
                        y_tr=y_tr,
                        y_va=y_va,
                        w_tr=w_tr,
                        w_va=w_va,
                        estimator=estimator,
                        estimator_factory=estimator_factory,
                        scoring=scoring_obj,
                        sample_weight_supplied=sample_weight_supplied,
                        task=task,
                    )
                estimator_snaps.append(
                    _estimator_event(
                        p_est,
                        selector=sel_name,
                        split_id=int(split_id),
                        scope="prefix",
                        prefix_k=int(prefix_k),
                    )
                )
                prefix_rows.append(
                    {
                        "selector": sel_name,
                        "split_id": int(split_id),
                        "k": int(prefix_k),
                        "score": p_score,
                        "n_encoded_columns": int(p_encoded),
                        "in_sample": True,
                        "mode": "in_sample_path",
                        "protocol": "in_sample_path",
                    }
                )
    return _assemble_result(
        mode="in_sample_path",
        in_sample=True,
        scoring_name=scoring_name,
        higher_is_better=higher_is_better,
        k_unit=("mixed" if len(k_units) > 1 else next(iter(k_units), "raw_features")),
        names=names,
        folds=folds,
        bookkeeping=bookkeeping,
        score_rows=score_rows,
        selected_by=selected_by,
        prefix_rows=prefix_rows,
        n_splits=len(splits),
        n_rows=n_rows,
        random_state=random_state,
        input_kind=input_kind,
        split=split,
        selectors_config=collapse_fold_snapshots(selector_snaps),
        estimator_config=_collapse_estimator_snaps(estimator_snaps),
        configured_estimator=(
            describe_estimator(estimator)
            if estimator is not None
            else {"status": "factory"}
        ),
    )


def _frame(rows, columns) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(columns))


def _assemble_result(
    *,
    mode,
    in_sample,
    scoring_name,
    higher_is_better,
    k_unit,
    names,
    folds,
    bookkeeping,
    score_rows,
    selected_by,
    prefix_rows,
    n_splits,
    n_rows=None,
    random_state=None,
    input_kind="unknown",
    split=None,
    selectors_config=None,
    estimator_config=None,
    configured_estimator=None,
) -> CompareResult:
    scores = _frame(score_rows, SCORE_COLUMNS)
    summary_rows = []
    if not scores.empty:
        for sel_name, group in scores.groupby("selector", sort=False):
            values = np.asarray(group["score"], dtype=np.float64)
            k_values = np.asarray(group["k"], dtype=np.float64)
            summary_rows.append(
                {
                    "selector": sel_name,
                    "score_mean": float(np.mean(values)),
                    "score_std": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
                    "mean_k": float(np.mean(k_values)),
                    "k_unit": str(group["k_unit"].iloc[0]),
                    "n_empty": int(np.asarray(group["empty"]).sum()),
                    "n_splits": int(len(group)),
                    "in_sample": bool(in_sample),
                    "mode": mode,
                }
            )
    summary = _frame(summary_rows, SUMMARY_COLUMNS)
    freq_rows = []
    n_sel_folds = max((len(v) for v in selected_by.values()), default=1)
    for sel_name, folds_selected in selected_by.items():
        counts = {name: 0 for name in names}
        for chosen in folds_selected:
            for feature in dict.fromkeys(chosen):
                if feature in counts:
                    counts[feature] += 1
        for feature, count in counts.items():
            freq_rows.append(
                {
                    "selector": sel_name,
                    "feature": feature,
                    "frequency": float(count / n_sel_folds),
                    "n_folds": int(n_sel_folds),
                    "selection_identity": "raw_features",
                    "in_sample": bool(in_sample),
                    "mode": mode,
                }
            )
    selection_frequency = _frame(freq_rows, FREQUENCY_COLUMNS)
    overlap_rows = []
    sel_names = list(selected_by)
    for i, left in enumerate(sel_names):
        for right in sel_names[i + 1 :]:
            n_pairs = min(len(selected_by[left]), len(selected_by[right]))
            values = [
                _jaccard(set(selected_by[left][j]), set(selected_by[right][j]))
                for j in range(n_pairs)
            ]
            overlap_rows.append(
                {
                    "selector_a": left,
                    "selector_b": right,
                    "mean_jaccard": float(np.mean(values)) if values else float("nan"),
                    "selection_identity": "raw_features",
                    "in_sample": bool(in_sample),
                    "mode": mode,
                }
            )
    overlap = _frame(overlap_rows, OVERLAP_COLUMNS)
    prefix_scores = _frame(prefix_rows, PREFIX_COLUMNS)
    folds = folds.reindex(columns=list(FOLDS_COLUMNS))
    diagnostics = {
        "mode": mode,
        "in_sample": bool(in_sample),
        "protocol": mode,
        "scoring": scoring_name,
        "higher_is_better": bool(higher_is_better),
        "k_unit": k_unit,
        "selection_identity": "raw_features",
        "n_splits": int(n_splits),
        "empty_selection": "intercept_only",
        "n_rows": None if n_rows is None else int(n_rows),
        "n_features": int(len(names)),
        "raw_columns_hash": _columns_hash(names),
        "input_kind": input_kind,
        "compare_random_state": random_state,
        "split": split,
        "selectors": selectors_config,
        "estimator": estimator_config,
        "configured_estimator": configured_estimator,
    }
    return CompareResult(
        mode=mode,
        in_sample=bool(in_sample),
        scoring=scoring_name,
        higher_is_better=bool(higher_is_better),
        k_unit=k_unit,
        selection_identity="raw_features",
        folds=folds,
        scores=scores,
        summary=summary,
        selection_frequency=selection_frequency,
        overlap=overlap,
        prefix_scores=prefix_scores,
        fold_bookkeeping=bookkeeping,
        diagnostics=diagnostics,
    )


__all__ = ["CompareResult", "compare"]
