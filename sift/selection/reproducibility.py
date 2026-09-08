"""JSON-safe reproducibility manifests for selection results and compare."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import subprocess
from datetime import timedelta
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
    "n_targets",
    "ic_df_rule",
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
_DESCRIPTOR_STATUSES = {
    "params",
    "opaque",
    "cache_provenance",
    "sequence_digest",
    "array_digest",
    "varies",
    "partial",
}


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
        return _sanitize_param(configured)
    return _sanitize_param(_subset(metadata, _CONFIGURED_KEYS))


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
        # Keep the historical JSON-array framing byte-for-byte while hashing
        # one token at a time.  This avoids retaining a list of every object
        # token for large object frames.
        digest.update(b"[")
        for position, value in enumerate(payload.reshape(-1)):
            if position:
                digest.update(b",")
            token = _label_token(value)
            digest.update(
                json.dumps(
                    token,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            )
        digest.update(b"]")
    else:
        digest.update(np.ascontiguousarray(payload).tobytes())
    return digest.hexdigest()


def _is_feature_cache(obj: Any) -> bool:
    cls = type(obj)
    return cls.__name__ in {"FeatureCache", "ClassicFeatureCache"} and str(
        cls.__module__
    ).startswith("sift.")


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
        **{
            key: _sanitize_param(getattr(obj, key))
            for key in (
                "subsample",
                "random_state",
                "weights_supplied",
                "subsample_applied",
            )
            if hasattr(obj, key)
        },
    }


def snapshot_selector_kwargs(
    kwargs: Mapping[str, Any] | None,
    *,
    unused: tuple[str, ...] = (),
) -> dict[str, Any]:
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
            return _sequence_digest(value)
        return [_sanitize_param(item, depth=depth + 1) for item in value]
    if isinstance(value, np.ndarray):
        if value.size > 32:
            return _array_digest(value)
        return _sanitize_param(value.tolist(), depth=depth + 1)
    if isinstance(value, Mapping):
        # ``describe_estimator`` already returns this shape.  Re-sanitizing
        # nested descriptors with the ordinary depth counter used to erase
        # their inner estimator parameters and seeds.
        if value.get("status") in _DESCRIPTOR_STATUSES and (
            "type" in value or value.get("status") in {"varies", "partial"}
        ):
            return {
                key: _sanitize_param(item, depth=0)
                for key, item in value.items()
            }
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


def _sequence_digest(value: list | tuple) -> dict[str, Any]:
    digest = hashlib.sha256()
    digest.update(f"sequence:{type(value).__module__}.{type(value).__qualname__}:".encode())
    digest.update(str(len(value)).encode("ascii"))
    digest.update(b"[")
    try:
        for position, item in enumerate(value):
            if position:
                digest.update(b",")
            digest.update(
                json.dumps(
                    _label_token(item),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            )
    except TypeError as exc:
        return {
            "status": "opaque",
            "reason": "unsupported_sequence_value",
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "error": str(exc),
        }
    digest.update(b"]")
    return {
        "status": "sequence_digest",
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "length": len(value),
        "sha256": digest.hexdigest(),
    }


def _array_digest(value: np.ndarray) -> dict[str, Any]:
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    if value.dtype == object:
        try:
            for item in value.reshape(-1):
                digest.update(
                    json.dumps(
                        _label_token(item),
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("utf-8")
                )
        except TypeError as exc:
            return {
                "status": "opaque",
                "reason": "unsupported_array_value",
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
                "shape": [int(dim) for dim in value.shape],
                "error": str(exc),
            }
    else:
        digest.update(np.ascontiguousarray(value).tobytes())
    return {
        "status": "array_digest",
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "dtype": str(value.dtype),
        "shape": [int(dim) for dim in value.shape],
        "sha256": digest.hexdigest(),
    }


_PURGED_SPLITTER_FIELDS = (
    "n_splits",
    "max_train_size",
    "test_size",
    "embargo",
    "mode",
)


def _is_sift_purged_splitter(obj: Any) -> bool:
    cls = type(obj)
    return cls.__name__ in {
        "PurgedTimeSeriesSplit",
        "GroupPurgedTimeSeriesSplit",
    } and str(getattr(cls, "__module__", "")).startswith("sift.selection.purged_cv")


def _sanitize_splitter_param(value: Any) -> Any:
    if isinstance(value, (np.timedelta64, pd.Timedelta, timedelta)):
        # Keep a single JSON string so snapshot depth limits do not opaque it.
        return (
            "duration:"
            f"{type(value).__module__}.{type(value).__qualname__}:"
            f"{pd.Timedelta(value).isoformat()}"
        )
    return _sanitize_param(value)


def describe_splitter(obj: Any) -> dict[str, Any]:
    """Describe a CV splitter without requiring sklearn estimator params."""
    desc = describe_estimator(obj)
    type_name = f"{type(obj).__module__}.{type(obj).__qualname__}"
    params: dict[str, Any] = {}
    if desc.get("status") == "params":
        params.update(dict(desc.get("params") or {}))
    if _is_sift_purged_splitter(obj):
        for name in _PURGED_SPLITTER_FIELDS:
            if hasattr(obj, name):
                params[name] = _sanitize_splitter_param(getattr(obj, name))
        return {"type": type_name, "status": "params", "params": params}
    if desc.get("status") == "params":
        return desc
    for name in ("n_splits", "shuffle", "random_state"):
        if hasattr(obj, name):
            params[name] = _sanitize_param(getattr(obj, name))
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
        try:
            fields = dataclasses.asdict(obj)
        except Exception:
            return {"type": type_name, "status": "opaque"}
        return {
            "type": type_name,
            "status": "params",
            "params": {
                str(key): _sanitize_param(item)
                for key, item in fields.items()
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
        "realized_random_state": None,
        "base_seed_control": None,
    }
    configured = metadata.get("configured_options")
    configured_seed = (
        configured.get("random_state")
        if isinstance(configured, Mapping)
        else None
    )
    if "random_state" in metadata:
        configured_seed = metadata.get("random_state")
    if _is_int(configured_seed):
        seeds["available"] = True
    if configured_seed is not None:
        seeds["random_state"] = _sanitize_param(configured_seed)
    elif "random_state" in metadata:
        # Preserve an explicit configured None without treating it as a
        # reproducible seed.  This is distinct from an unavailable field.
        seeds["random_state"] = None
    realized = metadata.get("realized_random_state")
    if realized is not None:
        seeds["realized_random_state"] = _sanitize_param(realized)
        if _is_int(realized):
            seeds["available"] = True
    base_control = metadata.get("base_seed_control")
    if base_control is not None:
        seeds["base_seed_control"] = _sanitize_param(base_control)
    cfg = metadata.get("auto_k_config")
    params = None
    if isinstance(cfg, Mapping):
        params = cfg.get("params") if isinstance(cfg.get("params"), Mapping) else cfg
    if isinstance(params, Mapping) and params.get("random_state") is not None:
        auto_seed = params["random_state"]
        seeds["auto_k_random_state"] = _sanitize_param(auto_seed)
        if _is_int(auto_seed):
            seeds["available"] = True
    return seeds


def _context_hash(value: Any, *, label: str, n_rows: int | None) -> str:
    """Hash caller-supplied row context without retaining it."""
    array = (
        value.to_numpy()
        if isinstance(value, (pd.Series, pd.DataFrame))
        else np.asarray(value)
    )
    allowed_ndim = (1, 2) if label == "y" else (1,)
    if array.ndim not in allowed_ndim:
        expected = "1-D or 2-D" if label == "y" else "1-D"
        raise ValueError(f"{label} must be {expected} when supplied for hashing")
    if n_rows is not None and int(array.shape[0]) != int(n_rows):
        raise ValueError(
            f"{label} has {int(array.shape[0])} rows but the result describes "
            f"{int(n_rows)} rows"
        )
    digest = hashlib.sha256()
    digest.update(f"{label}:".encode("utf-8"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(str(array.dtype).encode("utf-8"))
    if array.dtype == object:
        digest.update(b"[")
        for position, item in enumerate(array.reshape(-1)):
            if position:
                digest.update(b",")
            digest.update(
                json.dumps(
                    _label_token(item),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            )
        digest.update(b"]")
    else:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _context_input_fields(
    *,
    y: Any,
    sample_weight: Any,
    groups: Any,
    time: Any,
    hash_data: bool,
    n_rows: int | None,
) -> dict[str, Any]:
    values = {
        "y": y,
        "sample_weight": sample_weight,
        "groups": groups,
        "time": time,
    }
    observed_rows: int | None = n_rows
    for label, value in values.items():
        if value is None:
            continue
        array = (
            value.to_numpy()
            if isinstance(value, (pd.Series, pd.DataFrame))
            else np.asarray(value)
        )
        allowed_ndim = (1, 2) if label == "y" else (1,)
        if array.ndim not in allowed_ndim:
            expected = "1-D or 2-D" if label == "y" else "1-D"
            raise ValueError(f"{label} must be {expected} when supplied for hashing")
        rows = int(array.shape[0])
        if observed_rows is None:
            observed_rows = rows
        elif rows != observed_rows:
            raise ValueError(
                f"{label} has {rows} rows but other manifest inputs describe "
                f"{observed_rows} rows"
            )
    fields: dict[str, Any] = {}
    for label, value in values.items():
        key = f"{label}_hash"
        source_key = f"{label}_hash_source"
        complete_key = f"{label}_hash_complete"
        if value is None:
            fields[key] = None
            fields[source_key] = None
            fields[complete_key] = None
        elif hash_data:
            fields[key] = _context_hash(value, label=label, n_rows=n_rows)
            fields[source_key] = "caller"
            fields[complete_key] = True
        else:
            fields[key] = None
            fields[source_key] = "caller_unhashed"
            fields[complete_key] = False
    fields["context_hashes_complete"] = bool(
        hash_data and all(value is not None for value in values.values())
    )
    fields["context_hash_scope"] = "caller_only"
    fields["context_selection_time_verified"] = False
    return fields


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


def manifest_from_view(
    view,
    *,
    X=None,
    y=None,
    sample_weight=None,
    groups=None,
    time=None,
    hash_data: bool = False,
) -> dict[str, Any]:
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
            try:
                columns_hash = _columns_hash(caller_names)
            except TypeError:
                columns_hash = None
            else:
                columns_source = "caller"
    if n_features is None and X is not None:
        n_features = _n_features_of(X)
        n_features_source = "caller"
    elif n_features is None:
        n_features_source = "unknown"
    else:
        n_features_source = "result"
    data_hash = _data_hash(X) if hash_data else None
    context_fields = _context_input_fields(
        y=y,
        sample_weight=sample_weight,
        groups=groups,
        time=time,
        hash_data=hash_data,
        n_rows=rows["n_rows"],
    )
    configured = _configured_from_metadata(metadata)
    effective = _sanitize_param(_subset(metadata, _EFFECTIVE_KEYS))
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
            **context_fields,
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


def manifest_from_compare(
    result,
    *,
    X=None,
    y=None,
    sample_weight=None,
    groups=None,
    time=None,
    hash_data: bool = False,
) -> dict[str, Any]:
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
            try:
                columns_hash = _columns_hash(caller_names)
            except TypeError:
                columns_hash = None
            else:
                columns_source = "caller"
    data_hash = _data_hash(X) if hash_data else None
    context_fields = _context_input_fields(
        y=y,
        sample_weight=sample_weight,
        groups=groups,
        time=time,
        hash_data=hash_data,
        n_rows=n_rows,
    )
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
    configured = _sanitize_param(configured)
    effective = _sanitize_param(
        {
            "split": split,
            "selectors": selectors,
            "estimator": estimator,
        }
    )
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
            **context_fields,
            "cache": {
                "available": False,
                "n_rows_original": None,
                "feature_names_are_synthetic": None,
            },
        },
        "configuration": {
            "captured_at": "compare" if configured else "unknown",
            "configured": configured,
            "effective": effective,
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
