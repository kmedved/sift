"""Shared include/exclude/candidates validation and selection composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.errors import InvalidIndexError

INCLUDE_PROVENANCE = ("prespecified", "sample_split", "data_derived")
FDR_COMPATIBLE_PROVENANCE = frozenset({"prespecified", "sample_split"})

# Auto-k methods that rebuild or re-score paths without the same conditional
# state. Prefix-truncation methods are supported because they consume a
# already-conditioned greedy path.
UNSUPPORTED_CONDITIONING_AUTO_K = frozenset(
    {
        "stability",
        "knockoff_path",
        "consensus",
        "gaussian_cv",
        "perm_gap",
        "xfit_objective",
    }
)

ConditioningRef = Any


@dataclass(frozen=True)
class ResolvedConditioning:
    """Resolved original-column positions for one filter/knockoff call.

    ``include`` is the conditioning set in caller order. ``exclude`` and
    ``candidates`` constrain the discovery pool; both the stored
    ``candidates`` tuple (when provided) and ``discovery`` are in stable
    original-column order so allow-list permutation cannot change ties.
    An omitted ``candidates`` argument is recorded as ``None``.
    ``discovery`` is the eligible pool before ``top_m`` relevance screening
    and ``corr_prune`` correlation pruning, which run on the candidate panel.
    """

    include: tuple[int, ...]
    exclude: tuple[int, ...]
    candidates: tuple[int, ...] | None
    discovery: tuple[int, ...]
    include_refs: tuple[Any, ...]
    named: bool

    @property
    def active(self) -> bool:
        return bool(self.include or self.exclude or self.candidates is not None)


def omitted_conditioning(include, exclude, candidates) -> bool:
    """Return True when every public conditioning keyword was omitted."""
    return include is None and exclude is None and candidates is None


def resolve_conditioning(
    include,
    exclude,
    candidates,
    *,
    feature_names: Sequence[str],
    named: bool,
    k: int | Literal["auto"] | None = None,
) -> ResolvedConditioning | None:
    """Resolve public feature-set arguments against one feature namespace.

    Named spaces (DataFrame columns, named caches) accept those labels, and
    reject integer entries that are not themselves column labels.
    Positional spaces (ndarrays, synthetic-name caches) accept integer
    positions and the synthetic ``x{{i}}`` names. Strings are never iterated
    as character sequences.
    """
    if omitted_conditioning(include, exclude, candidates):
        return None

    names = list(feature_names)
    n_features = len(names)
    include_idx, include_refs = _resolve_refs(
        include, names=names, named=named, label="include"
    )
    exclude_idx, _exclude_refs = _resolve_refs(
        exclude, names=names, named=named, label="exclude"
    )
    if candidates is None:
        candidate_idx: tuple[int, ...] | None = None
    else:
        candidate_idx, _cand_refs = _resolve_refs(
            candidates, names=names, named=named, label="candidates"
        )

    include_set = set(include_idx)
    exclude_set = set(exclude_idx)
    overlap_ie = sorted(include_set & exclude_set)
    if overlap_ie:
        raise ValueError(
            "include and exclude overlap: "
            + _format_original_refs(overlap_ie, names, named=named)
        )
    if candidate_idx is not None:
        candidate_set = set(candidate_idx)
        overlap_ce = sorted(candidate_set & exclude_set)
        if overlap_ce:
            raise ValueError(
                "candidates and exclude overlap: "
                + _format_original_refs(overlap_ce, names, named=named)
            )
        candidate_idx = tuple(i for i in range(n_features) if i in candidate_set)
        discovery = tuple(
            i for i in candidate_idx if i not in exclude_set and i not in include_set
        )
    else:
        discovery = tuple(
            i for i in range(n_features) if i not in exclude_set and i not in include_set
        )

    if not discovery and (k is None or k == "auto" or int(k) > 0):
        raise ValueError(
            "conditioning leaves no eligible features for discovery; "
            "include, exclude, and candidates produce an empty candidate pool"
        )

    return ResolvedConditioning(
        include=include_idx,
        exclude=exclude_idx,
        candidates=candidate_idx,
        discovery=discovery,
        include_refs=include_refs,
        named=named,
    )


def require_supported_auto_k(method: str | None) -> None:
    """Reject auto-k methods that cannot honor exact conditioning."""
    if method in UNSUPPORTED_CONDITIONING_AUTO_K:
        raise ValueError(
            f"k_method={method!r} cannot honor exact include/exclude/candidates "
            "conditioning; it rebuilds or re-scores an unconditioned path. "
            "Use a prefix-truncation method such as 'evaluate', 'elbow', or "
            "'penalized_objective', or omit the conditioning keywords."
        )


def require_include_provenance(
    include_provenance,
    *,
    conditioning_active: bool,
) -> str | None:
    """Validate knockoff include_provenance against the conditioning sets."""
    if not conditioning_active:
        if include_provenance is not None:
            raise ValueError(
                "include_provenance is only meaningful when include, exclude, "
                "or candidates is provided"
            )
        return None
    if include_provenance is None:
        raise ValueError(
            "include_provenance is required when include, exclude, or candidates "
            "is provided; pass 'prespecified', 'sample_split', or 'data_derived'"
        )
    if include_provenance not in INCLUDE_PROVENANCE:
        raise ValueError(
            "include_provenance must be one of 'prespecified', 'sample_split', "
            f"or 'data_derived'; got {include_provenance!r}"
        )
    return str(include_provenance)


def compose_selected(
    feature_names: Sequence[str],
    include_idx: Sequence[int],
    discovered_idx: Sequence[int],
) -> tuple[list[str], list[int]]:
    """Return include (caller order) followed by discoveries (greedy order)."""
    seen: set[int] = set()
    indices: list[int] = []
    for raw in include_idx:
        idx = int(raw)
        if idx in seen:
            continue
        indices.append(idx)
        seen.add(idx)
    for raw in discovered_idx:
        idx = int(raw)
        if idx in seen:
            continue
        indices.append(idx)
        seen.add(idx)
    names = [feature_names[i] for i in indices]
    return names, indices


def conditioning_record(
    resolved: ResolvedConditioning | None,
    *,
    feature_names: Sequence[str],
    discovered_idx: Sequence[int] | None = None,
    include_provenance: str | None = None,
    discovery_universe: Sequence[int] | None = None,
) -> dict[str, Any] | None:
    """Machine-readable conditioning payload for diagnostics/metadata.

    ``discovery_universe`` (and ``discovery_universe_indices``) is the
    *eligibility* pool: every original column left after ``exclude``,
    ``candidates`` and ``include`` are applied. It is recorded BEFORE
    relevance screening (``top_m``) and correlation pruning
    (``corr_prune``), which run later on the candidate panel and are not
    visible here, so it is not the set of columns the greedy loop actually
    scored: with ``top_m=3`` and seven eligible columns it still lists all
    seven. Read ``selector_metadata['top_m']`` for the screening width.
    The value answers "what was this call allowed to discover from", which
    is what manifests and conditioning tests compare.
    """
    if resolved is None or not resolved.active:
        return None
    names = list(feature_names)
    discovered = (
        []
        if discovered_idx is None
        else [int(i) for i in discovered_idx]
    )
    universe = (
        list(resolved.discovery)
        if discovery_universe is None
        else [int(i) for i in discovery_universe]
    )
    record: dict[str, Any] = {
        "include": _names_for(resolved.include, names),
        "include_indices": list(resolved.include),
        "exclude": _names_for(resolved.exclude, names),
        "exclude_indices": list(resolved.exclude),
        "candidates": (
            None
            if resolved.candidates is None
            else _names_for(resolved.candidates, names)
        ),
        "candidates_indices": (
            None if resolved.candidates is None else list(resolved.candidates)
        ),
        "discovered": _names_for(discovered, names),
        "discovered_indices": discovered,
        "discovery_universe": _names_for(universe, names),
        "discovery_universe_indices": universe,
        "include_are_discoveries": False,
        "k_counts": "additional_discoveries",
    }
    if include_provenance is not None:
        record["include_provenance"] = include_provenance
        record["fdr_compatible"] = include_provenance in FDR_COMPATIBLE_PROVENANCE
        record["exploratory"] = include_provenance == "data_derived"
    return record


def map_original_to_valid(
    original_idx: Sequence[int],
    valid_cols: np.ndarray,
    *,
    feature_names: Sequence[str] | None,
    label: str,
    missing: Literal["error", "drop"] = "error",
    raw_names: Mapping[Any, Any] | None = None,
    prebuilt_cache: bool = False,
) -> np.ndarray:
    """Map original column positions onto cache.valid_cols / Z columns.

    ``raw_names`` maps an internal column name back to the raw column the
    caller passed, so an error never quotes a name the user never wrote
    (one-hot dummies such as ``city__NY``). Omitting it reports the internal
    names unchanged. ``prebuilt_cache`` is set by callers that received a
    cache from the user: only those can report a column as "never cached",
    because a cache built from ``X`` in this call drops columns for exactly
    one reason.
    """
    lookup = {int(orig): int(local) for local, orig in enumerate(np.asarray(valid_cols))}
    mapped: list[int] = []
    missing_idx: list[int] = []
    for orig in original_idx:
        key = int(orig)
        if key not in lookup:
            missing_idx.append(key)
        else:
            mapped.append(lookup[key])
    if missing_idx and missing == "error":
        names = list(feature_names or [])
        refs = _format_original_refs(
            missing_idx, names, named=bool(names), raw_names=raw_names
        )
        if prebuilt_cache:
            raise ValueError(
                f"{label} features are not present in the cache valid columns "
                f"(dropped as constant/non-finite or never cached): {refs}. "
                f"Drop them from {label}, or rebuild the cache from columns "
                "that vary"
            )
        raise ValueError(
            f"{label} features were dropped as constant or non-finite, so they "
            f"carry no information to condition on: {refs}. Drop them from "
            f"{label}, or pass columns that vary on the retained rows"
        )
    return np.asarray(mapped, dtype=np.int64)


def named_feature_space(feature_names: Sequence[Any] | np.ndarray | None, *, synthetic: bool) -> bool:
    """Named caches/frames use labels; positional/synthetic spaces use indices."""
    if feature_names is None or synthetic:
        return False
    if isinstance(feature_names, np.ndarray):
        return int(feature_names.size) > 0
    return len(list(feature_names)) > 0


def _resolve_refs(
    values,
    *,
    names: Sequence[Any],
    named: bool,
    label: str,
) -> tuple[tuple[int, ...], tuple[Any, ...]]:
    if values is None:
        return (), ()
    refs = _as_refs(values, label=label)
    if not refs:
        return (), ()

    index = pd.Index(list(names))
    indices: list[int] = []
    seen: dict[int, Any] = {}
    for ref in refs:
        try:
            idx = _resolve_one(ref, index=index, named=named, label=label)
        except TypeError as exc:
            raise ValueError(
                f"{label} entries must be feature names or integer positions"
            ) from exc
        if idx in seen:
            raise ValueError(f"{label} contains duplicate feature {ref!r}")
        seen[idx] = ref
        indices.append(idx)
    return tuple(indices), tuple(refs)


def _as_refs(values, *, label: str) -> tuple[Any, ...]:
    if isinstance(values, (set, frozenset, dict)):
        raise ValueError(
            f"{label} must be an ordered sequence of feature names or positions; "
            "sets and mappings are unordered"
        )
    if isinstance(values, (str, bytes)):
        return (values,)
    if isinstance(values, (bool, np.bool_)):
        raise ValueError(f"{label} must be a sequence of feature names or positions")
    if isinstance(values, (int, np.integer)):
        return (int(values),)
    if isinstance(values, np.ndarray):
        if values.ndim > 1:
            raise ValueError(f"{label} must be a 1-d sequence of feature names or positions")
        return tuple(values.tolist())
    if isinstance(values, Iterable):
        return tuple(values)
    raise ValueError(f"{label} must be a sequence of feature names or positions")


def _resolve_one(
    ref,
    *,
    index: pd.Index,
    named: bool,
    label: str,
) -> int:
    n_features = len(index)
    if isinstance(ref, (bool, np.bool_)):
        raise ValueError(f"{label} entries must be feature names or integer positions")
    if named:
        try:
            loc = index.get_loc(ref)
        except KeyError:
            _reject_position_in_named_space(ref, index=index, label=label)
            raise ValueError(f"{label} contains unknown feature {ref!r}") from None
        except InvalidIndexError:
            _reject_position_in_named_space(ref, index=index, label=label)
            raise ValueError(f"{label} contains unknown feature {ref!r}") from None
        if isinstance(loc, slice):
            raise ValueError(f"{label} refers to duplicate input feature name {ref!r}")
        if isinstance(loc, np.ndarray):
            hits = np.flatnonzero(loc)
            if hits.size != 1:
                raise ValueError(f"{label} refers to duplicate input feature name {ref!r}")
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
        except KeyError:
            raise ValueError(f"{label} contains unknown feature {ref!r}") from None
        if isinstance(loc, (slice, np.ndarray)):
            raise ValueError(f"{label} refers to duplicate input feature name {ref!r}")
        return int(loc)
    raise ValueError(f"{label} entries must be feature names or integer positions")


def _reject_position_in_named_space(ref, *, index: pd.Index, label: str) -> None:
    """Reject an integer position against a named space; integer labels pass first.

    Only reached once ``index.get_loc(ref)`` has already failed, so an
    integer that *is* a column label never lands here.
    """
    if isinstance(ref, (bool, np.bool_)) or not isinstance(ref, (int, np.integer)):
        return
    value = int(ref)
    hint = ""
    if 0 <= value < len(index):
        hint = f"; position {value} is column {index[value]!r}"
    raise ValueError(
        f"{label} contains integer position {value}, but X has named columns: "
        "integer positions are accepted only for ndarray input, and DataFrame "
        f"entries must be column labels. Pass the column label in {label}{hint}"
    ) from None


def _names_for(indices: Sequence[int], names: Sequence[Any]) -> list[Any]:
    return [names[int(i)] for i in indices]


def _format_original_refs(
    indices: Sequence[int],
    names: Sequence[str],
    *,
    named: bool,
    raw_names: Mapping[Any, Any] | None = None,
) -> str:
    if named and names:
        labels = [names[int(i)] for i in indices]
        if raw_names is not None:
            labels = list(dict.fromkeys(raw_names.get(name, name) for name in labels))
        return ", ".join(repr(name) for name in labels)
    return ", ".join(str(int(i)) for i in indices)
