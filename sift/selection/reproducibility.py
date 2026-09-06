"""JSON-safe reproducibility manifests for selection results and compare."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from sift.selection.view import _columns_hash, _json_safe, _label_token


_CONFIGURED_KEYS = (
    "selector",
    "k_requested",
    "auto_k",
    "top_m",
    "method",
    "corr_prune",
    "q",
    "aggregation",
    "cat_encoding",
    "feature_blocks",
    "random_state",
    "subsample",
    "statistic",
    "s_method",
    "n_draws",
    "eta",
    "offset",
    "loss",
    "ridge",
    "k_method",
    "auto_k_mode",
    "objective_penalty",
    "auto_k_strategy",
    "selection_rule",
    "formula",
    "relevance",
    "estimator",
    "task",
    "within",
    "auto_k_config",
    "n_resamples",
    "resample",
    "threshold",
    "sample_frac",
    "store_proxies",
    "output_order",
    "block_size",
    "block_method",
    "base_selector",
)
_EFFECTIVE_KEYS = (
    "k",
    "n_features",
    "n_blocks_selected",
    "n_columns_selected",
    "k_unit",
    "cache_backed",
    "fdr_control",
    "n_rows_used",
    "n_rows_original",
    "statistic",
    "k_method",
    "objective_penalty",
    "auto_k_mode",
    "path_depth",
    "subsample",
    "top_m",
    "n_resamples",
    "resample",
    "threshold",
    "aggregation",
)
_COMPARE_PROTOCOL_KEYS = (
    "mode",
    "in_sample",
    "scoring",
    "higher_is_better",
    "k_unit",
    "n_splits",
    "selection_identity",
)
_SCALAR_TYPES = (bool, int, float, str, type(None))


def _module_version(module_name: str) -> str | None:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    version = getattr(module, "__version__", None)
    return None if version is None else str(version)


def _sift_source_root() -> Path:
    import sift

    path = Path(sift.__file__).resolve().parent
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return path


def _git_commit() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_sift_source_root()),
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    commit = proc.stdout.strip()
    if len(commit) == 40 and all(char in "0123456789abcdef" for char in commit):
        return commit
    return None


def _export_environment() -> dict[str, Any]:
    import sift
    from threadpoolctl import threadpool_info

    return {
        "captured_at": "export",
        "sift": str(sift.__version__),
        "numpy": _module_version("numpy"),
        "pandas": _module_version("pandas"),
        "scikit-learn": _module_version("sklearn"),
        "scipy": _module_version("scipy"),
        "numba": _module_version("numba"),
        "blas": threadpool_info(),
        "git_commit": _git_commit(),
        "git_commit_source": "sift_package",
    }


def _subset(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: mapping[key] for key in keys if key in mapping}


_CONFIGURED_IDENTITY_KEYS = (
    "selector",
    "k_requested",
    "auto_k",
    "auto_k_config",
    "k_method",
    "objective_penalty",
    "auto_k_mode",
    "auto_k_strategy",
    "selection_rule",
)


def _configured_from_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    options = metadata.get("configured_options")
    if isinstance(options, Mapping):
        configured = dict(options)
        for key in _CONFIGURED_IDENTITY_KEYS:
            if key in metadata:
                configured.setdefault(key, metadata[key])
        return configured
    return _subset(metadata, _CONFIGURED_KEYS)


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _require_2d(X) -> None:
    if isinstance(X, pd.DataFrame):
        if int(getattr(X, "ndim", 2)) != 2:
            raise ValueError("X must be a 2D feature matrix")
        return
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"X must be a 2D feature matrix; got {arr.ndim}D array")


def _n_features_of(X) -> int:
    _require_2d(X)
    return int(X.shape[1])


def _n_rows_of(X) -> int:
    _require_2d(X)
    return int(X.shape[0])


def _caller_columns(X) -> list[Any] | None:
    if isinstance(X, pd.DataFrame):
        return list(X.columns)
    return None


def _data_hash(X) -> str:
    _require_2d(X)
    if isinstance(X, pd.DataFrame):
        column_token = _columns_hash(list(X.columns))
        payload = np.ascontiguousarray(X.to_numpy())
    else:
        column_token = ""
        payload = np.ascontiguousarray(np.asarray(X))
    digest = hashlib.sha256()
    digest.update(column_token.encode("utf-8"))
    digest.update(np.asarray(payload.shape, dtype=np.int64).tobytes())
    digest.update(str(payload.dtype).encode("utf-8"))
    if payload.dtype == object:
        encoded = json.dumps(
            [_label_token(value) for value in payload.reshape(-1)],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest.update(encoded)
    else:
        digest.update(np.ascontiguousarray(payload).tobytes())
    return digest.hexdigest()


def _is_feature_cache(obj: Any) -> bool:
    cls = type(obj)
    return cls.__name__ == "FeatureCache" and str(cls.__module__).startswith("sift.")


def describe_feature_cache(obj: Any) -> dict[str, Any]:
    """Compact cache provenance without copying matrix, weights, or row indices."""
    row_idx = getattr(obj, "row_idx", None)
    valid_cols = getattr(obj, "valid_cols", None)
    n_cached = None if row_idx is None else int(np.asarray(row_idx).reshape(-1).size)
    n_valid = None if valid_cols is None else int(np.asarray(valid_cols).reshape(-1).size)
    return {
        "type": f"{type(obj).__module__}.{type(obj).__qualname__}",
        "status": "cache_provenance",
        "n_rows_original": int(getattr(obj, "n_rows_original")),
        "n_rows_cached": n_cached,
        "n_valid_features": n_valid,
        "feature_names_are_synthetic": bool(
            getattr(obj, "feature_names_are_synthetic", False)
        ),
        "has_rxx": getattr(obj, "Rxx", None) is not None,
    }


def snapshot_selector_kwargs(kwargs: Mapping[str, Any] | None, *, unused: tuple[str, ...] = ()) -> dict[str, Any]:
    """Typed snapshot of non-data function-selector options."""
    skip = {"callback", *unused}
    data = {key: value for key, value in dict(kwargs or {}).items() if key not in skip}
    sanitized = _sanitize_param(data)
    return sanitized if isinstance(sanitized, dict) else {"status": "opaque"}


def _sanitize_param(value: Any, *, depth: int = 0) -> Any:
    if depth > 3:
        return {"status": "opaque"}
    if _is_feature_cache(value):
        return describe_feature_cache(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, _SCALAR_TYPES):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    if isinstance(value, (list, tuple)):
        if len(value) > 32:
            return {"status": "partial", "reason": "sequence_too_long", "length": len(value)}
        return [_sanitize_param(item, depth=depth + 1) for item in value]
    if isinstance(value, np.ndarray):
        if value.size > 32:
            return {
                "status": "partial",
                "reason": "array_omitted",
                "shape": [int(dim) for dim in value.shape],
            }
        return _sanitize_param(value.tolist(), depth=depth + 1)
    if isinstance(value, Mapping):
        items = list(value.items())
        if len(items) > 256:
            return {
                "status": "partial",
                "reason": "mapping_truncated",
                "n_entries": len(items),
                "params": {
                    key: _sanitize_param(item, depth=depth + 1)
                    for key, item in items[:256]
                },
            }
        return {key: _sanitize_param(item, depth=depth + 1) for key, item in items}
    getter = getattr(value, "get_params", None)
    if callable(getter):
        return describe_estimator(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return describe_estimator(value)
    type_name = f"{type(value).__module__}.{type(value).__qualname__}"
    return {"status": "opaque", "type": type_name}


def describe_splitter(obj: Any) -> dict[str, Any]:
    """Describe a CV splitter without requiring sklearn estimator params."""
    desc = describe_estimator(obj)
    if desc.get("status") == "params":
        return desc
    params: dict[str, Any] = {}
    for name in ("n_splits", "shuffle", "random_state"):
        if hasattr(obj, name):
            params[name] = _sanitize_param(getattr(obj, name))
    type_name = f"{type(obj).__module__}.{type(obj).__qualname__}"
    if params:
        return {"type": type_name, "status": "params", "params": params}
    return desc


def describe_estimator(obj: Any) -> dict[str, Any]:
    """Compact JSON-safe constructor snapshot. Never retains a live object."""
    if obj is None:
        return {"status": "absent"}
    if _is_feature_cache(obj):
        return describe_feature_cache(obj)
    type_name = f"{type(obj).__module__}.{type(obj).__qualname__}"
    getter = getattr(obj, "get_params", None)
    if callable(getter):
        try:
            try:
                raw = getter(deep=False)
            except TypeError:
                raw = getter()
        except Exception:
            return {"type": type_name, "status": "opaque"}
        if not isinstance(raw, Mapping):
            return {"type": type_name, "status": "opaque"}
        return {
            "type": type_name,
            "status": "params",
            "params": {key: _sanitize_param(item) for key, item in raw.items()},
        }
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            "type": type_name,
            "status": "params",
            "params": {
                str(key): _sanitize_param(item)
                for key, item in dataclasses.asdict(obj).items()
            },
        }
    return {"type": type_name, "status": "opaque"}


def collapse_fold_snapshots(snapshots: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, items in snapshots.items():
        if items and all(item == items[0] for item in items):
            out[name] = items[0]
        else:
            out[name] = {"status": "varies", "by_fold": list(items)}
    return out


def _cache_block(metadata: Mapping[str, Any]) -> dict[str, Any]:
    synthetic = metadata.get("feature_names_are_synthetic")
    cache_backed = metadata.get("cache_backed")
    retained = bool(cache_backed) or isinstance(synthetic, (bool, np.bool_))
    n_rows = metadata.get("n_rows_original") if retained else None
    has_rows = _is_int(n_rows)
    has_synthetic = isinstance(synthetic, (bool, np.bool_))
    return {
        "available": bool(retained and (has_rows or has_synthetic)),
        "n_rows_original": int(n_rows) if has_rows else None,
        "feature_names_are_synthetic": bool(synthetic) if has_synthetic else None,
    }


def _seed_block(metadata: Mapping[str, Any]) -> dict[str, Any]:
    seeds: dict[str, Any] = {
        "available": False,
        "random_state": None,
        "auto_k_random_state": None,
    }
    if "random_state" in metadata:
        seeds["available"] = True
        seeds["random_state"] = metadata["random_state"]
    cfg = metadata.get("auto_k_config")
    params = None
    if isinstance(cfg, Mapping):
        params = cfg.get("params") if isinstance(cfg.get("params"), Mapping) else cfg
    if isinstance(params, Mapping) and "random_state" in params:
        seeds["auto_k_random_state"] = params["random_state"]
        seeds["available"] = True
    return seeds


def _int_or_none(value: Any) -> int | None:
    return int(value) if _is_int(value) else None


def _validate_optional_X(
    X,
    *,
    n_features: int | None,
    n_rows_original: int | None,
    columns_hash: str | None,
) -> None:
    if X is None:
        return
    _require_2d(X)
    width = _n_features_of(X)
    if n_features is not None and width != int(n_features):
        raise ValueError(
            f"X has {width} columns but the result describes {int(n_features)} features"
        )
    if n_rows_original is not None and _n_rows_of(X) != int(n_rows_original):
        raise ValueError(
            f"X has {_n_rows_of(X)} rows but the result describes "
            f"{int(n_rows_original)} original rows"
        )
    caller_names = _caller_columns(X)
    if columns_hash is not None and caller_names is not None:
        observed = _columns_hash(caller_names)
        if observed != columns_hash:
            raise ValueError(
                "X column identity does not match the result's ordered typed columns"
            )


def _row_fields(metadata: Mapping[str, Any], X) -> dict[str, Any]:
    cache = _cache_block(metadata)
    original = cache["n_rows_original"]
    original_source = "cache" if original is not None else None
    if original is None:
        original = _int_or_none(metadata.get("n_rows_original"))
        if original is not None:
            original_source = "result"
    used = _int_or_none(metadata.get("n_rows_used"))
    if used is None:
        used = _int_or_none(metadata.get("n_rows_cached"))
    if original is None and X is not None:
        original, original_source = _n_rows_of(X), "caller"
    if original is None:
        original_source = "unknown"
    if used is None:
        used_source = "unknown"
    else:
        used_source = "result"
    return {
        "n_rows": original,
        "n_rows_source": original_source,
        "n_rows_used": used,
        "n_rows_used_source": used_source,
        "cache": cache,
    }


def manifest_from_view(view, *, X=None, hash_data: bool = False) -> dict[str, Any]:
    """Build a JSON-safe manifest from a SelectionView.

    Environment, BLAS identity, and git commit are always labelled as
    export-time and bound to the installed sift package tree. Selection-time
    facts come only from what the view already retained. ``X`` is never stored.
    """
    if hash_data and X is None:
        raise ValueError("hash_data=True requires X")
    metadata = view.metadata
    raw_input = view.raw_input
    n_features = raw_input.get("n_features")
    columns_hash = raw_input.get("columns_hash")
    rows = _row_fields(metadata, None)
    _validate_optional_X(
        X,
        n_features=n_features if _is_int(n_features) else None,
        n_rows_original=rows["n_rows"] if rows["n_rows_source"] in {"cache", "result"} else None,
        columns_hash=columns_hash,
    )
    if rows["n_rows"] is None and X is not None:
        rows["n_rows"] = _n_rows_of(X)
        rows["n_rows_source"] = "caller"
    columns_source = "unknown" if columns_hash is None else "result"
    if columns_hash is None and X is not None:
        caller_names = _caller_columns(X)
        if caller_names is not None:
            columns_hash = _columns_hash(caller_names)
            columns_source = "caller"
    if n_features is None and X is not None:
        n_features = _n_features_of(X)
        n_features_source = "caller"
    elif n_features is None:
        n_features_source = "unknown"
    else:
        n_features_source = "result"
    data_hash = _data_hash(X) if hash_data else None
    configured = _configured_from_metadata(metadata)
    effective = _subset(metadata, _EFFECTIVE_KEYS)
    captured = (
        "selection"
        if configured or effective or _seed_block(metadata)["available"]
        else "unknown"
    )
    payload = {
        "schema_version": "1",
        "kind": "selection",
        "environment": _export_environment(),
        "input": {
            "n_rows": rows["n_rows"],
            "n_rows_source": rows["n_rows_source"],
            "n_rows_used": rows["n_rows_used"],
            "n_rows_used_source": rows["n_rows_used_source"],
            "n_features": None if n_features is None else int(n_features),
            "n_features_source": n_features_source,
            "columns_hash": columns_hash,
            "columns_hash_source": columns_source,
            "data_hash": data_hash,
            "data_hash_source": None if data_hash is None else "caller",
            "cache": rows["cache"],
        },
        "configuration": {
            "captured_at": captured,
            "configured": configured,
            "effective": effective,
            "seeds": _seed_block(metadata),
        },
        "folds": [],
    }
    return _json_safe(payload)


def manifest_from_compare(result, *, X=None, hash_data: bool = False) -> dict[str, Any]:
    """Build a JSON-safe manifest from a CompareResult.

    Fold fingerprints are the compare-time bookkeeping already stored on the
    result. Selector, estimator, and splitter snapshots are compare-time.
    Environment remains export-time. ``X`` is never stored.
    """
    if hash_data and X is None:
        raise ValueError("hash_data=True requires X")
    diagnostics = dict(result.diagnostics)
    n_features = diagnostics.get("n_features")
    n_rows = diagnostics.get("n_rows")
    columns_hash = diagnostics.get("raw_columns_hash")
    _validate_optional_X(
        X,
        n_features=int(n_features) if _is_int(n_features) else None,
        n_rows_original=int(n_rows) if _is_int(n_rows) else None,
        columns_hash=columns_hash if columns_hash else None,
    )
    if _is_int(n_features):
        n_features = int(n_features)
        n_features_source = "result"
    elif X is not None:
        n_features = _n_features_of(X)
        n_features_source = "caller"
    else:
        n_features = None
        n_features_source = "unknown"
    if _is_int(n_rows):
        n_rows = int(n_rows)
        n_rows_source = "result"
    elif X is not None:
        n_rows = _n_rows_of(X)
        n_rows_source = "caller"
    else:
        n_rows = None
        n_rows_source = "unknown"
    columns_source = "unknown" if not columns_hash else "result"
    if not columns_hash and X is not None:
        caller_names = _caller_columns(X)
        if caller_names is not None:
            columns_hash = _columns_hash(caller_names)
            columns_source = "caller"
    data_hash = _data_hash(X) if hash_data else None
    protocol = _subset(diagnostics, _COMPARE_PROTOCOL_KEYS)
    split = diagnostics.get("split")
    selectors = diagnostics.get("selectors")
    estimator = diagnostics.get("estimator")
    configured_estimator = diagnostics.get("configured_estimator")
    compare_seed = diagnostics.get("compare_random_state", diagnostics.get("random_state"))
    split_seed = None
    if isinstance(split, Mapping):
        params = split.get("params") if isinstance(split.get("params"), Mapping) else {}
        split_seed = params.get("random_state")
    used_for_split = bool(
        isinstance(split, Mapping) and split.get("uses_compare_random_state")
    )
    configured = dict(protocol)
    if split is not None:
        configured["split"] = split
    if selectors is not None:
        configured["selectors"] = selectors
    if configured_estimator is not None:
        configured["estimator"] = configured_estimator
    elif estimator is not None:
        configured["estimator"] = estimator
    if compare_seed is not None:
        configured["compare_random_state"] = compare_seed
    payload = {
        "schema_version": "1",
        "kind": "compare",
        "environment": _export_environment(),
        "input": {
            "n_rows": n_rows,
            "n_rows_source": n_rows_source,
            "n_rows_used": n_rows,
            "n_rows_used_source": n_rows_source if n_rows is not None else "unknown",
            "n_features": n_features,
            "n_features_source": n_features_source,
            "columns_hash": columns_hash,
            "columns_hash_source": columns_source,
            "data_hash": data_hash,
            "data_hash_source": None if data_hash is None else "caller",
            "cache": {
                "available": False,
                "n_rows_original": None,
                "feature_names_are_synthetic": None,
            },
        },
        "configuration": {
            "captured_at": "compare" if configured else "unknown",
            "configured": configured,
            "effective": {
                "split": split,
                "selectors": selectors,
                "estimator": estimator,
            },
            "seeds": {
                "available": compare_seed is not None or split_seed is not None,
                "compare_random_state": compare_seed,
                "split_random_state": split_seed,
                "compare_random_state_used_for_split": used_for_split,
            },
        },
        "folds": [dict(item) for item in result.fold_bookkeeping],
    }
    return _json_safe(payload)
