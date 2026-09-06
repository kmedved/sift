"""Atomic feature-block resolution for filter selectors and knockoff aliases.

``feature_blocks`` is an additive grouping of *columns*. Row ``groups`` and
``time`` metadata are not feature blocks. The documented one-hot auto
convention is a double-underscore prefix ``{block}__{level}``; ordinary
single underscores are never split. Knockoff ``feature_blocks="auto"`` is a
different alias: it forwards to correlation-cluster ``feature_groups="auto"``.

Block auto-k support ledger (filter paths):

- Supported: ``auto``, ``evaluate``, ``elbow``, ``penalized_objective``,
  ``gaussian_cv``, ``xfit_objective``. Grids, ``min_k``/``max_k``, stopping,
  and diagnostics count additional blocks; selected prefixes expand to raw
  columns. ``k_method="auto"`` routes among those (perm-gap fallbacks become
  EBIC). Identity (all-singleton) maps reuse the no-block routes, including
  calibrated methods for selection, scoring, and cadence. Public metadata
  still uses additional-block units for every supplied map, including
  identities; ``n_blocks_selected_total`` is the include-plus-discovery
  total and ``view.k`` stays raw width. EBIC model dimension is copula
  rank; search multiplicity is ``log C(B, k)``. RIC block adaptation is
  ``2 df log(B)``.
- Unsupported, column-step null/df/FDR: ``stability``, ``knockoff_path``,
  ``consensus``, ``perm_gap``, ``chi2_stop``, ``forward_stop``,
  ``changepoint``, ``k_posterior``/``posterior``. These raise
  ``require_block_auto_k``.
- Binary log-loss CEFS+ supports ``auto``, ``evaluate``, ``elbow``, and
  ``penalized_objective`` with a joint logistic block score. Identity maps
  reuse the scalar path for selection, scoring, and cadence; metadata still
  uses additional-block units. ``gaussian_cv``/``xfit_objective`` and calibrated
  column-step rules raise ``require_binary_block_auto_k``. ``loss="brier"``
  delegates to Gaussian CEFS+ blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.errors import InvalidIndexError

from sift.selection.conditioning import ResolvedConditioning, _format_original_refs

ONEHOT_PREFIX_SEP = "__"
SUPPORTED_BLOCK_AUTO_K = frozenset(
    {
        "auto",
        "evaluate",
        "elbow",
        "penalized_objective",
        "gaussian_cv",
        "xfit_objective",
    }
)
SUPPORTED_BINARY_BLOCK_AUTO_K = frozenset(
    {
        "auto",
        "evaluate",
        "elbow",
        "penalized_objective",
    }
)
UNSUPPORTED_BLOCK_AUTO_K = frozenset(
    {
        "stability",
        "knockoff_path",
        "consensus",
        "perm_gap",
        "chi2_stop",
        "forward_stop",
        "changepoint",
        "k_posterior",
        "posterior",
    }
)
AUTO_K_BLOCKS_MESSAGE = (
    "feature_blocks is not supported with this auto-k rule: the method's "
    "null, degrees-of-freedom, or FDR calibration is defined on scalar "
    "column steps and has no justified block generalization. Use "
    "k_method in {'evaluate', 'elbow', 'penalized_objective', "
    "'gaussian_cv', 'xfit_objective'} or k_method='auto' (which routes "
    "among those). Binary log-loss CEFS+ supports evaluate, elbow, "
    "penalized_objective, and auto (EBIC); it does not use Gaussian CV/xfit."
)
BINARY_LOGLOSS_BLOCKS_MESSAGE = (
    "feature_blocks is not supported for this binary log-loss auto-k rule: "
    "the method's null or Gaussian-copula calibration does not apply to "
    "joint logistic block scores. Use evaluate, elbow, "
    "penalized_objective, or k_method='auto' (EBIC). loss='brier' delegates "
    "to Gaussian CEFS+ blocks."
)


@dataclass(frozen=True)
class ResolvedBlocks:
    """Resolved block-label → original-column membership.

    ``block_ids[i]`` owns ``members[i]`` (original positions, sorted). Every
    input column belongs to exactly one block. ``column_to_block[j]`` is the
    block index of original column ``j``.
    """

    block_ids: tuple[Any, ...]
    members: tuple[tuple[int, ...], ...]
    column_to_block: tuple[int, ...]
    named: bool
    spec: Any

    @property
    def n_blocks(self) -> int:
        return len(self.block_ids)

    @property
    def n_columns(self) -> int:
        return len(self.column_to_block)

    def members_of(self, block_index: int) -> tuple[int, ...]:
        return self.members[int(block_index)]

    def block_index_for(self, original: int) -> int:
        return self.column_to_block[int(original)]

    def all_singletons(self) -> bool:
        return all(len(group) == 1 for group in self.members)

    def expand(self, block_indices: Sequence[int]) -> list[int]:
        """Expand block indices to original columns, preserving block order."""
        out: list[int] = []
        seen: set[int] = set()
        for raw in block_indices:
            for col in self.members[int(raw)]:
                if col not in seen:
                    out.append(col)
                    seen.add(col)
        return out


def resolve_feature_blocks(
    feature_blocks,
    *,
    feature_names: Sequence[Any],
    named: bool,
) -> ResolvedBlocks | None:
    """Resolve ``feature_blocks`` against one feature namespace.

    ``None`` leaves the call on the historical column-as-unit path.
    A mapping is an explicit block-label → members dict. ``"auto"`` groups
    columns that share a ``{block}__{level}`` prefix when at least two
    columns share that prefix. Unlisted or unprefixed columns become
    singleton blocks labeled by the column identity.
    """
    if feature_blocks is None:
        return None
    names = list(feature_names)
    n_features = len(names)
    if n_features == 0:
        raise ValueError("feature_blocks requires a non-empty feature namespace")
    if isinstance(feature_blocks, str):
        if feature_blocks != "auto":
            raise ValueError(
                "feature_blocks must be None, 'auto', or a mapping of block "
                f"labels to member names or positions; got {feature_blocks!r}"
            )
        return _resolve_auto_prefix_blocks(names, named=named)
    if isinstance(feature_blocks, Mapping):
        return _resolve_explicit_blocks(feature_blocks, names=names, named=named)
    raise ValueError(
        "feature_blocks must be None, 'auto', or a mapping of block labels "
        "to member names or positions"
    )


def require_atomic_conditioning(
    resolved: ResolvedConditioning | None,
    blocks: ResolvedBlocks | None,
    *,
    feature_names: Sequence[Any],
) -> None:
    """Reject include/exclude/candidates that would split a block."""
    if blocks is None or resolved is None or not resolved.active:
        return
    names = list(feature_names)
    named = blocks.named
    _require_complete_restriction(
        resolved.include,
        blocks,
        names=names,
        named=named,
        label="include",
    )
    _require_complete_restriction(
        resolved.exclude,
        blocks,
        names=names,
        named=named,
        label="exclude",
    )
    if resolved.candidates is not None:
        _require_complete_restriction(
            resolved.candidates,
            blocks,
            names=names,
            named=named,
            label="candidates",
        )


def block_indices_for_columns(
    columns: Sequence[int],
    blocks: ResolvedBlocks,
) -> tuple[int, ...]:
    """Unique block indices covering ``columns``, in first-appearance order."""
    seen: set[int] = set()
    out: list[int] = []
    for col in columns:
        b = blocks.column_to_block[int(col)]
        if b not in seen:
            seen.add(b)
            out.append(b)
    return tuple(out)


def require_block_auto_k(method: str | None) -> None:
    """Reject auto-k rules whose column-step calibrations do not generalize."""
    if method is None or method in SUPPORTED_BLOCK_AUTO_K:
        return
    raise ValueError(
        f"feature_blocks is not supported with k_method={method!r}: "
        "that rule's null, degrees-of-freedom, or FDR calibration is defined "
        "on scalar column steps and has no justified block generalization. "
        "Use evaluate, elbow, penalized_objective, gaussian_cv, or "
        "xfit_objective (or k_method='auto', which routes among those)."
    )


def require_binary_block_auto_k(method: str | None) -> None:
    """Reject binary auto-k rules that do not generalize to joint log-loss blocks."""
    if method is None or method in SUPPORTED_BINARY_BLOCK_AUTO_K:
        return
    raise ValueError(
        f"feature_blocks is not supported for binary log-loss CEFS+ with "
        f"k_method={method!r}: that rule's calibration is defined on scalar "
        "column steps or is a Gaussian-copula method. Use evaluate, elbow, "
        "penalized_objective, or k_method='auto' (EBIC). loss='brier' "
        "delegates to Gaussian CEFS+ blocks."
    )


def discovery_prefix_widths(
    path_indices: Sequence[int],
    blocks: ResolvedBlocks | None,
) -> tuple[int, ...]:
    """Cumulative raw discovery widths at each additional-block boundary.

    Without blocks each column is one step, recovering the historical
    ``1..len(path)`` prefix grid.
    """
    indices = [int(i) for i in path_indices]
    if not indices:
        return ()
    if blocks is None:
        return tuple(range(1, len(indices) + 1))
    widths: list[int] = []
    i = 0
    n = len(indices)
    while i < n:
        bidx = blocks.column_to_block[indices[i]]
        member_set = set(int(c) for c in blocks.members[bidx])
        take = 0
        while i + take < n and indices[i + take] in member_set:
            take += 1
        if take == 0:
            take = 1
        i += take
        widths.append(i)
    return tuple(widths)


def slice_prefix_by_steps(
    values: Sequence[Any],
    selected_steps: int,
    prefix_widths: Sequence[int],
) -> list[Any]:
    """Slice a raw path by additional-block (or column) step count."""
    if int(selected_steps) <= 0 or not prefix_widths:
        return []
    n = int(prefix_widths[min(int(selected_steps), len(prefix_widths)) - 1])
    return list(values[:n])


def weighted_copula_design_rank(
    Z: np.ndarray,
    weights: np.ndarray,
    columns: Sequence[int] | None = None,
) -> int:
    """Rank of a weighted, centered rank-Gaussian design.

    Zero-weight rows are dropped. Constant or duplicated columns do not add
    dimension. Shrinkage is not applied.
    """
    cols = np.asarray(Z, dtype=np.float64)
    if cols.ndim == 1:
        cols = cols.reshape(-1, 1)
    if columns is not None:
        idx = [int(c) for c in columns]
        if not idx:
            return 0
        cols = cols[:, idx]
    if cols.size == 0:
        return 0
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.shape[0] != cols.shape[0]:
        raise ValueError("weights must match the number of copula rows")
    mask = w > 0.0
    if not np.any(mask):
        return 0
    cols = cols[mask]
    w = w[mask]
    mean = np.average(cols, axis=0, weights=w)
    design = (cols - mean) * np.sqrt(w)[:, None]
    return int(np.linalg.matrix_rank(design))


def gaussian_copula_prefix_df(
    cache,
    path_indices: Sequence[int],
    prefix_widths: Sequence[int],
    *,
    include_indices: Sequence[int] = (),
) -> np.ndarray:
    """Usable Gaussian model df at each additional-block prefix.

    Weighted rank of the cache rank-Gaussian design of cache-valid members,
    minus the rank of the include set. Constant padding dropped from the
    cache contributes 0. Exact duplicates do not add dimension. This is the
    likelihood dimension, distinct from block-search multiplicity.
    """
    lookup = {
        int(orig): int(local)
        for local, orig in enumerate(np.asarray(cache.valid_cols, dtype=np.int64))
    }
    Z = np.asarray(getattr(cache, "Z"), dtype=np.float64)
    w = np.asarray(cache.sample_weight, dtype=np.float64).reshape(-1)

    def _rank(origs: Sequence[int]) -> int:
        locals_ = [lookup[int(i)] for i in origs if int(i) in lookup]
        return weighted_copula_design_rank(Z, w, locals_)

    include = [int(i) for i in include_indices]
    base = _rank(include)
    indices = [int(i) for i in path_indices]
    dfs = np.zeros(len(prefix_widths), dtype=np.float64)
    for t, width in enumerate(prefix_widths):
        prefix = include + indices[: int(width)]
        dfs[t] = float(max(_rank(prefix) - base, 0))
    return dfs


def eligible_discovery_block_count(
    blocks: ResolvedBlocks,
    *,
    valid_cols: Sequence[int],
    resolved: ResolvedConditioning | None,
) -> int:
    """Cache-valid eligible discovery blocks before screen/prune.

    Included blocks are not discoveries. Excluded blocks and blocks outside
    ``candidates`` are omitted. A block with no cache-valid member cannot
    enter the copula search and is omitted.
    """
    valid = {int(i) for i in np.asarray(valid_cols, dtype=np.int64).ravel()}
    include = set(resolved.include) if resolved is not None and resolved.include else set()
    exclude = set(resolved.exclude) if resolved is not None and resolved.exclude else set()
    candidates = (
        set(resolved.candidates)
        if resolved is not None and resolved.candidates is not None
        else None
    )
    count = 0
    for members in blocks.members:
        member_set = {int(c) for c in members}
        if include and member_set <= include:
            continue
        if exclude and member_set & exclude:
            continue
        if candidates is not None and not member_set <= candidates:
            continue
        if not (member_set & valid):
            continue
        count += 1
    return count


def labels_for_columns(blocks: ResolvedBlocks) -> list[Any]:
    """Per-original-column block labels, knockoff ``feature_groups`` shape."""
    return [blocks.block_ids[blocks.column_to_block[i]] for i in range(blocks.n_columns)]


def screen_block_indices(
    block_members: Sequence[Sequence[int]],
    scores: np.ndarray,
    *,
    top_m: int,
    protect: Sequence[int] = (),
) -> list[int]:
    """Keep protected blocks plus the top-scoring remaining blocks, whole.

    ``scores`` is a per-column array. A block's screen score is the max of
    its members. ``top_m`` is the discovery-block budget and does not count
    protected/included blocks.
    """
    n_blocks = len(block_members)
    protect_unique: list[int] = []
    seen: set[int] = set()
    for raw in protect:
        idx = int(raw)
        if idx in seen:
            continue
        if idx < 0 or idx >= n_blocks:
            raise ValueError("protected block index is out of range")
        protect_unique.append(idx)
        seen.add(idx)
    remaining = [i for i in range(n_blocks) if i not in seen]
    if not remaining:
        return protect_unique
    block_scores = np.empty(len(remaining), dtype=np.float64)
    for pos, block_idx in enumerate(remaining):
        members = np.asarray(block_members[block_idx], dtype=np.int64)
        block_scores[pos] = float(np.max(scores[members])) if members.size else -np.inf
    budget = max(int(top_m), 0)
    if budget <= 0:
        return protect_unique
    if budget >= len(remaining):
        order = np.lexsort((np.asarray(remaining, dtype=np.int64), -block_scores))
        return protect_unique + [remaining[int(i)] for i in order]
    pick = np.argpartition(block_scores, -budget)[-budget:]
    chosen = [remaining[int(i)] for i in pick]
    chosen_scores = block_scores[pick]
    order = np.lexsort((np.asarray(chosen, dtype=np.int64), -chosen_scores))
    return protect_unique + [chosen[int(i)] for i in order]


def prune_blocks_by_corr(
    block_members: Sequence[Sequence[int]],
    R: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    protect: Sequence[int] = (),
) -> list[int]:
    """Greedy correlation prune at block granularity (never splits a block)."""
    n_blocks = len(block_members)
    protect_set = {int(i) for i in protect}
    discovery = [i for i in range(n_blocks) if i not in protect_set]
    block_scores = np.empty(n_blocks, dtype=np.float64)
    for i, members in enumerate(block_members):
        arr = np.asarray(members, dtype=np.int64)
        block_scores[i] = float(np.max(scores[arr])) if arr.size else -np.inf
    order = [i for i in discovery]
    order_scores = block_scores[np.asarray(order, dtype=np.int64)] if order else np.empty(0)
    ranked = [order[int(i)] for i in np.lexsort((np.asarray(order, dtype=np.int64), -order_scores))]
    kept: list[int] = list(protect_set)
    kept_members = [np.asarray(block_members[i], dtype=np.int64) for i in kept]
    for idx in ranked:
        members = np.asarray(block_members[idx], dtype=np.int64)
        drop = False
        for other in kept_members:
            if other.size == 0 or members.size == 0:
                continue
            if float(np.max(np.abs(R[np.ix_(members, other)]))) >= threshold:
                drop = True
                break
        if drop:
            continue
        kept.append(idx)
        kept_members.append(members)
    # Preserve protect-first then discovery-keep order by score.
    discovery_kept = [i for i in ranked if i in set(kept) and i not in protect_set]
    protect_order = [i for i in range(n_blocks) if i in protect_set]
    return protect_order + discovery_kept


def map_blocks_to_valid(
    blocks: ResolvedBlocks,
    valid_cols: np.ndarray,
) -> tuple[list[int], list[np.ndarray]]:
    """Map original blocks onto cache-valid local indices.

    Blocks with no valid member are omitted. Remaining valid members keep
    the original block identity; expansion to raw columns still uses
    ``ResolvedBlocks.members``.
    """
    lookup = {int(orig): int(local) for local, orig in enumerate(np.asarray(valid_cols))}
    kept_ids: list[int] = []
    members_valid: list[np.ndarray] = []
    for block_idx, orig_members in enumerate(blocks.members):
        mapped = [lookup[int(col)] for col in orig_members if int(col) in lookup]
        if not mapped:
            continue
        kept_ids.append(block_idx)
        members_valid.append(np.asarray(mapped, dtype=np.int64))
    return kept_ids, members_valid


def block_result_metadata(
    blocks: ResolvedBlocks,
    selected_indices: Sequence[int],
    include_indices: Sequence[int] | None = None,
    *,
    n_columns_selected: int | None = None,
) -> dict[str, Any]:
    """Public block counts: additional discovery vs explicitly labeled totals.

    ``k`` / ``n_blocks_selected`` / ``selected_blocks`` are additional
    discovery blocks. ``n_blocks_selected_total`` counts include blocks plus
    those discoveries. Raw width stays in ``n_columns_selected``.
    """
    include_block_ids: list[int] = []
    seen_include: set[int] = set()
    for pos in include_indices or ():
        bidx = int(blocks.column_to_block[int(pos)])
        if bidx not in seen_include:
            seen_include.add(bidx)
            include_block_ids.append(bidx)
    discovery_labels: list[Any] = []
    seen_discovery: set[int] = set()
    for pos in selected_indices:
        bidx = int(blocks.column_to_block[int(pos)])
        if bidx in seen_include or bidx in seen_discovery:
            continue
        seen_discovery.add(bidx)
        discovery_labels.append(blocks.block_ids[bidx])
    n_discovery = len(discovery_labels)
    n_include = len(include_block_ids)
    n_columns = (
        int(n_columns_selected)
        if n_columns_selected is not None
        else len(list(selected_indices))
    )
    return {
        "feature_blocks": True,
        "n_blocks": int(blocks.n_blocks),
        "n_blocks_selected": n_discovery,
        "n_blocks_selected_total": n_include + n_discovery,
        "n_columns_selected": n_columns,
        "k": n_discovery,
        "selected_blocks": discovery_labels,
    }


def _resolve_explicit_blocks(
    mapping: Mapping[Any, Any],
    *,
    names: Sequence[Any],
    named: bool,
) -> ResolvedBlocks:
    n_features = len(names)
    column_to_block = [-1] * n_features
    block_ids: list[Any] = []
    members: list[tuple[int, ...]] = []
    assigned: dict[int, Any] = {}
    for label, raw_members in mapping.items():
        _require_hashable_block_id(label)
        member_idx = _resolve_member_refs(
            raw_members,
            names=names,
            named=named,
            label=f"feature_blocks[{label!r}]",
        )
        if not member_idx:
            raise ValueError(f"feature_blocks[{label!r}] has no members")
        overlap = [i for i in member_idx if i in assigned]
        if overlap:
            refs = _format_original_refs(overlap, names, named=named)
            raise ValueError(
                f"feature_blocks members overlap across blocks: {refs}"
            )
        ordered = tuple(sorted(set(member_idx)))
        block_index = len(block_ids)
        block_ids.append(label)
        members.append(ordered)
        for col in ordered:
            assigned[col] = label
            column_to_block[col] = block_index
    for col in range(n_features):
        if column_to_block[col] >= 0:
            continue
        singleton_id = names[col] if named else col
        _require_hashable_block_id(singleton_id)
        block_index = len(block_ids)
        block_ids.append(singleton_id)
        members.append((col,))
        column_to_block[col] = block_index
    _reject_duplicate_block_labels(
        block_ids,
        members,
        names=names,
        named=named,
        origin="feature_blocks",
    )
    return ResolvedBlocks(
        block_ids=tuple(block_ids),
        members=tuple(members),
        column_to_block=tuple(column_to_block),
        named=named,
        spec=dict(mapping),
    )


def _resolve_auto_prefix_blocks(
    names: Sequence[Any],
    *,
    named: bool,
) -> ResolvedBlocks:
    n_features = len(names)
    groups: dict[str, list[int]] = {}
    singleton_cols: list[int] = []
    for col, name in enumerate(names):
        prefix = _onehot_prefix(name)
        if prefix is None:
            singleton_cols.append(col)
            continue
        groups.setdefault(prefix, []).append(col)
    multi = {prefix: cols for prefix, cols in groups.items() if len(cols) >= 2}
    for prefix, cols in groups.items():
        if len(cols) < 2:
            singleton_cols.extend(cols)
    singleton_cols = sorted(set(singleton_cols))
    block_ids: list[Any] = []
    members: list[tuple[int, ...]] = []
    column_to_block = [-1] * n_features
    multi_items = sorted(multi.items(), key=lambda item: (item[1][0], item[0]))
    for prefix, cols in multi_items:
        ordered = tuple(sorted(cols))
        block_index = len(block_ids)
        block_ids.append(prefix)
        members.append(ordered)
        for col in ordered:
            column_to_block[col] = block_index
    for col in singleton_cols:
        singleton_id = names[col] if named else col
        _require_hashable_block_id(singleton_id)
        block_index = len(block_ids)
        block_ids.append(singleton_id)
        members.append((col,))
        column_to_block[col] = block_index
    _reject_duplicate_block_labels(
        block_ids,
        members,
        names=names,
        named=named,
        origin="feature_blocks='auto'",
    )
    return ResolvedBlocks(
        block_ids=tuple(block_ids),
        members=tuple(members),
        column_to_block=tuple(column_to_block),
        named=named,
        spec="auto",
    )


def _reject_duplicate_block_labels(
    block_ids: Sequence[Any],
    members: Sequence[Sequence[int]],
    *,
    names: Sequence[Any],
    named: bool,
    origin: str,
) -> None:
    first: dict[Any, int] = {}
    collisions: list[str] = []
    for i, label in enumerate(block_ids):
        if label in first:
            left = _format_original_refs(members[first[label]], names, named=named)
            right = _format_original_refs(members[i], names, named=named)
            collisions.append(f"label {label!r} would merge {left} and {right}")
        else:
            first[label] = i
    if collisions:
        detail = "; ".join(collisions)
        raise ValueError(
            f"{origin} produced duplicate block labels, which would silently "
            f"merge distinct groups: {detail}. Rename an explicit block label "
            "or the colliding raw column so generated singleton identities "
            "cannot change the partition"
        )


def _onehot_prefix(name: Any) -> str | None:
    if not isinstance(name, str):
        return None
    if ONEHOT_PREFIX_SEP not in name:
        return None
    prefix, _, remainder = name.partition(ONEHOT_PREFIX_SEP)
    if not prefix or not remainder:
        return None
    return prefix


def _require_hashable_block_id(label: Any) -> None:
    if isinstance(label, (list, dict, set)):
        raise ValueError("feature_blocks labels must be hashable")
    try:
        hash(label)
    except TypeError as exc:
        raise ValueError("feature_blocks labels must be hashable") from exc
    missing = pd.isna(label)
    is_missing = bool(np.any(missing)) if isinstance(missing, np.ndarray) else bool(missing)
    if is_missing:
        raise ValueError("feature_blocks labels must not be missing")


def _resolve_member_refs(
    values,
    *,
    names: Sequence[Any],
    named: bool,
    label: str,
) -> list[int]:
    if isinstance(values, (str, bytes)):
        refs: tuple[Any, ...] = (values,)
    elif isinstance(values, (set, frozenset, dict)):
        raise ValueError(
            f"{label} members must be an ordered sequence of feature names or "
            "positions; sets and mappings are unordered"
        )
    elif isinstance(values, (bool, np.bool_)):
        raise ValueError(f"{label} members must be feature names or integer positions")
    elif isinstance(values, (int, np.integer)):
        refs = (int(values),)
    elif isinstance(values, np.ndarray):
        if values.ndim > 1:
            raise ValueError(f"{label} members must be a 1-d sequence")
        refs = tuple(values.tolist())
    elif isinstance(values, Iterable):
        refs = tuple(values)
    else:
        raise ValueError(
            f"{label} members must be a sequence of feature names or positions"
        )
    if not refs:
        return []
    index = pd.Index(list(names))
    indices: list[int] = []
    seen: set[int] = set()
    for ref in refs:
        idx = _resolve_one_member(ref, index=index, named=named, label=label)
        if idx in seen:
            raise ValueError(f"{label} contains duplicate member {ref!r}")
        seen.add(idx)
        indices.append(idx)
    return indices


def _resolve_one_member(
    ref,
    *,
    index: pd.Index,
    named: bool,
    label: str,
) -> int:
    n_features = len(index)
    if isinstance(ref, (bool, np.bool_)):
        raise ValueError(f"{label} members must be feature names or integer positions")
    if named:
        try:
            loc = index.get_loc(ref)
        except (KeyError, InvalidIndexError):
            raise ValueError(f"{label} contains unknown member {ref!r}") from None
        if isinstance(loc, slice):
            raise ValueError(f"{label} refers to a duplicate input name {ref!r}")
        if isinstance(loc, np.ndarray):
            hits = np.flatnonzero(loc)
            if hits.size != 1:
                raise ValueError(f"{label} refers to a duplicate input name {ref!r}")
            return int(hits[0])
        return int(loc)
    if isinstance(ref, (int, np.integer)):
        value = int(ref)
        if value < 0 or value >= n_features:
            raise ValueError(
                f"{label} contains out-of-range position {value}; "
                f"expected 0..{n_features - 1}"
            )
        return value
    if isinstance(ref, str):
        try:
            loc = index.get_loc(ref)
        except (KeyError, InvalidIndexError):
            raise ValueError(f"{label} contains unknown member {ref!r}") from None
        if isinstance(loc, (slice, np.ndarray)):
            raise ValueError(f"{label} refers to a duplicate input name {ref!r}")
        return int(loc)
    raise ValueError(f"{label} members must be feature names or integer positions")


def _require_complete_restriction(
    columns: Sequence[int],
    blocks: ResolvedBlocks,
    *,
    names: Sequence[Any],
    named: bool,
    label: str,
) -> None:
    if not columns:
        return
    chosen = {int(i) for i in columns}
    touched: set[int] = set()
    missing: list[int] = []
    for col in chosen:
        block_idx = blocks.column_to_block[int(col)]
        if block_idx in touched:
            continue
        touched.add(block_idx)
        for member in blocks.members[block_idx]:
            if member not in chosen:
                missing.append(member)
    if missing:
        refs = _format_original_refs(sorted(set(missing)), names, named=named)
        raise ValueError(
            f"{label} would split one or more feature_blocks; include every "
            f"member of each touched block or omit the restriction. Missing: {refs}"
        )
