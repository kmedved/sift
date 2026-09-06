"""Sklearn-style selector wrappers around top-level function selectors."""

from __future__ import annotations

import inspect
import importlib.util
import warnings
from dataclasses import replace
from typing import Callable, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.metadata_routing import UNUSED
from sklearn.utils.validation import check_is_fitted

from sift._metadata import drop_fitted_metadata_columns, resolve_row_metadata
from sift._progress import ProgressCallback
from sift._selector_compat import (
    check_fitted_column_identity,
    feature_names_array,
    inverse_selected_matrix,
    ordered_indices,
    reject_sparse,
    selector_tags,
    validate_fit_matrix,
    validate_output_order,
)
from sift._preprocess import (
    LeaveOneOutLogitEncoder,
    OneHotBlockEncoder,
    TargetCVEncoder,
    ensure_weights,
    extract_feature_names,
    suppress_category_encoder_pandas_warnings,
    validate_onehot_max_levels,
    validate_target_cv_encoding_flags,
)
from sift.api import (
    select_fdr,
    select_cefsplus_binary,
    select_cefsplus,
    select_jmim,
    select_jmi,
    select_mrmr,
)
from sift.selection.cefsplus_binary_common import (
    resolve_binary_weights,
    validate_binary_target,
)
from sift.selection.auto_k import AutoKConfig, resolve_auto_k_config
from sift.selection.auto_k_nested import NestedAutoKFold, select_k_nested
from sift.selection.knockoff_filter import (
    _SUBSAMPLE_DEFAULT,
    _validate_prebuilt_cache_structure,
)
from sift.selection.filter_api import _RANDOM_STATE_DEFAULT

_SUPERVISED_CLASS_ENCODINGS = frozenset(
    {"target_cv", "loo", "target", "james_stein", "loo_logit"}
)
_BINARY_PREPROCESSING_FIT_PARAM_OVERRIDES = frozenset(
    {
        "loss",
        "cat_features",
        "cat_encoding",
        "class_weight",
        "loo_smoothing",
        "loo_clip_min",
        "loo_clip_max",
    }
)
_SELECTOR_FORWARD_SKIP_PARAMS = frozenset(
    {
        "X",
        "y",
        "k",
        "cache",
        "groups",
        "time",
        "auto_k_config",
        "sample_weight",
        "return_result",
    }
)
_BLOCKED_FIT_PARAM_OVERRIDES = frozenset(
    {
        "return_result",
        "cat_features",
        "cat_encoding",
        "allow_full_data_target_encoding",
        "onehot_max_levels",
    }
)


def _as_label_list(value) -> list:
    """Convert sklearn/numpy feature-name containers without stringifying labels."""
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return []
        return list(np.asarray(value, dtype=object).ravel().tolist())
    try:
        return list(value)
    except TypeError:
        return [value]


def _effective_nested_blocks_and_conditioning(selector, fit_params):
    params = dict(fit_params or {})
    feature_blocks = params.get("feature_blocks", getattr(selector, "feature_blocks", None))
    include = params.get("include", getattr(selector, "include", None))
    exclude = params.get("exclude", getattr(selector, "exclude", None))
    candidates = params.get("candidates", getattr(selector, "candidates", None))
    return feature_blocks, include, exclude, candidates


def _nonconstant_column_indices(X) -> list[int]:
    """Unsupervised finite non-constant column positions; no target is used."""
    if hasattr(X, "to_numpy"):
        values = X.to_numpy()
    else:
        values = np.asarray(X)
    if values.ndim != 2:
        return []
    out: list[int] = []
    for j in range(values.shape[1]):
        col = np.asarray(values[:, j])
        try:
            numeric = np.asarray(col, dtype=np.float64)
        except (TypeError, ValueError):
            uniq = pd.unique(col)
            if uniq.size > 1:
                out.append(j)
            continue
        finite = numeric[np.isfinite(numeric)]
        if finite.size and np.any(finite != finite[0]):
            out.append(j)
    return out


def _nested_discovery_units(selector, X, fit_params) -> int | None:
    """Eligible additional-block count, or None to keep column units."""
    from sift.selection.blocks import eligible_discovery_block_count, resolve_feature_blocks
    from sift.selection.conditioning import resolve_conditioning

    feature_blocks, include, exclude, candidates = _effective_nested_blocks_and_conditioning(
        selector, fit_params
    )
    if feature_blocks is None:
        return None
    names = list(X.columns) if hasattr(X, "columns") else list(_feature_names_or_default(X))
    named = hasattr(X, "columns")
    blocks = resolve_feature_blocks(feature_blocks, feature_names=names, named=named)
    if blocks is None or blocks.all_singletons():
        return None
    resolved = resolve_conditioning(
        include, exclude, candidates, feature_names=names, named=named, k=1
    )
    return int(
        eligible_discovery_block_count(
            blocks, valid_cols=_nonconstant_column_indices(X), resolved=resolved
        )
    )


def _nested_block_prefix_sizes(
    selector,
    train_X,
    *,
    feature_blocks,
    include,
) -> tuple[int, ...] | None:
    """Raw transform-matrix widths after 1, 2, ... additional blocks.

    ``feature_blocks`` and ``include`` are the already-resolved effective
    values from the nested caller. Explicit ``None`` means omitted, not
    "use the constructor attribute".
    """
    encoded_widths = getattr(selector, "_encoded_prefix_widths_", None)
    if encoded_widths:
        return tuple(int(w) for w in encoded_widths)
    if feature_blocks is None:
        return None
    from sift.selection.blocks import discovery_prefix_widths, resolve_feature_blocks
    from sift.selection.conditioning import resolve_conditioning

    names = _as_label_list(getattr(selector, "feature_names_in_", None))
    if not names:
        names = list(train_X.columns) if hasattr(train_X, "columns") else []
    named = hasattr(train_X, "columns")
    blocks = resolve_feature_blocks(feature_blocks, feature_names=names, named=named)
    if blocks is None or blocks.all_singletons():
        return None
    resolved = resolve_conditioning(
        include, None, None, feature_names=names, named=named, k=1
    )
    include_idx = set(resolved.include) if resolved is not None else set()
    selected = _as_label_list(getattr(selector, "selected_features_", None))
    name_to_idx = {name: i for i, name in enumerate(names)}
    selected_idx = [name_to_idx[name] for name in selected if name in name_to_idx]
    n_include = sum(1 for idx in selected_idx if idx in include_idx)
    discoveries = [idx for idx in selected_idx if idx not in include_idx]
    widths = discovery_prefix_widths(discoveries, blocks)
    if not widths:
        return None
    return tuple(int(n_include + width) for width in widths)


def _coerce_selection_indices(
    feature_names: list[str], selected_features: list[str]
) -> np.ndarray:
    """Map selected feature names back to integer positions.

    Keep the first unmatched index for duplicate names so output remains aligned
    with a stable source-order selection path.
    """

    pools: dict[str, list[int]] = {}
    for i, name in enumerate(feature_names):
        pools.setdefault(name, []).append(i)

    used: dict[str, int] = {name: 0 for name in pools}
    indices: list[int] = []
    for name in selected_features:
        if name not in pools:
            raise ValueError(f"Selected feature '{name}' not found in fitted data.")
        pos = used[name]
        choices = pools[name]
        if pos >= len(choices):
            raise ValueError(f"Could not map selected feature '{name}' to a unique index.")
        indices.append(choices[pos])
        used[name] = pos + 1

    return np.asarray(indices, dtype=np.int64)


def _require_2d_x(X) -> None:
    """Reject non-2D feature input with a clear error instead of an IndexError."""
    validate_fit_matrix(X)


def _feature_names_or_default(X) -> list[str]:
    return _feature_names_with_provenance(X)[0]


def _feature_names_with_provenance(X) -> tuple[list[str], bool]:
    """Return fitted feature names plus whether they were generated positionally.

    Generated ``x0...`` names stay in the public ``feature_names_in_`` attribute
    because that is the established 0.8 behavior; the boolean is the private
    provenance marker used wherever named and positional fits must be told
    apart.
    """
    feature_names = extract_feature_names(X)
    if feature_names is not None:
        return list(feature_names), False
    n_features = np.asarray(X).shape[1]
    return [f"x{i}" for i in range(n_features)], True


def _slice_rows(X, idx: np.ndarray):
    if isinstance(X, pd.DataFrame):
        return X.iloc[idx]
    return np.asarray(X)[idx]


def _categorical_columns(X: pd.DataFrame, cat_features: list[str] | None) -> list[str]:
    if cat_features is None:
        return X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    return [col for col in cat_features if col in X.columns]


def _selected_training_output(X_fit, selected_indices: np.ndarray):
    """Return selected columns from the matrix used during selector fitting."""
    if isinstance(X_fit, pd.DataFrame):
        return X_fit.iloc[:, selected_indices].copy()
    return np.asarray(X_fit)[:, selected_indices].copy()


def _apply_onehot_call_params(selector, call_params, feature_names) -> tuple:
    encoder = getattr(selector, "categorical_encoder_", None)
    if not isinstance(encoder, OneHotBlockEncoder):
        return encoder, None, []
    from sift.selection.blocks import compose_raw_blocks_through_onehot, resolve_feature_blocks
    from sift.selection.conditioning import resolve_conditioning

    raw_names = list(feature_names)
    raw_blocks = resolve_feature_blocks(
        call_params.get("feature_blocks", getattr(selector, "feature_blocks", None)),
        feature_names=raw_names,
        named=True,
    )
    encoded_names = list(encoder.output_names_)
    composed = compose_raw_blocks_through_onehot(
        raw_blocks,
        raw_names=raw_names,
        encoded_names=encoded_names,
        parents=list(encoder.output_parents_),
    )
    call_params["feature_blocks"] = {
        composed.block_ids[i]: [encoded_names[j] for j in composed.members[i]]
        for i in range(composed.n_blocks)
    }
    resolved = resolve_conditioning(
        call_params.get("include", getattr(selector, "include", None)),
        call_params.get("exclude", getattr(selector, "exclude", None)),
        call_params.get("candidates", getattr(selector, "candidates", None)),
        feature_names=raw_names,
        named=True,
        k=1,
    )
    if resolved is not None and resolved.include:
        call_params["include"] = [
            dummy
            for raw_i in resolved.include
            for dummy in encoder.encoded_columns_for(raw_names[int(raw_i)])
        ]
    if resolved is not None and resolved.exclude:
        call_params["exclude"] = [
            dummy
            for raw_i in resolved.exclude
            for dummy in encoder.encoded_columns_for(raw_names[int(raw_i)])
        ]
    if resolved is not None and resolved.candidates is not None:
        call_params["candidates"] = [
            dummy
            for raw_i in resolved.candidates
            for dummy in encoder.encoded_columns_for(raw_names[int(raw_i)])
        ]
    return encoder, composed, encoded_names


def _make_category_encoder(
    method: str,
    columns: list[str],
    *,
    target_type: Literal["continuous", "binary"] = "continuous",
    loo_smoothing: float = 20.0,
    loo_clip_min: float = 1e-4,
    loo_clip_max: float = 1.0 - 1e-4,
    target_cv_n_splits: int = 5,
    target_cv_smoothing: Literal["auto"] | float = "auto",
    target_prior: float | None = None,
    warmup_policy: Literal["exclude", "zero_weight"] = "zero_weight",
    onehot_max_levels: int = 32,
):
    if method == "none" or not columns:
        return None
    if method == "target_cv":
        return TargetCVEncoder(
            columns,
            target_type=target_type,
            smooth=target_cv_smoothing,
            cv=target_cv_n_splits,
            target_prior=target_prior,
            warmup_policy=warmup_policy,
        )
    if method == "loo_logit":
        return LeaveOneOutLogitEncoder(
            columns,
            smoothing=loo_smoothing,
            clip_min=loo_clip_min,
            clip_max=loo_clip_max,
        )
    if method == "onehot":
        return OneHotBlockEncoder(
            columns,
            max_levels=validate_onehot_max_levels(onehot_max_levels),
        )
    if method not in {"loo", "target", "james_stein"}:
        raise ValueError(
            "cat_encoding must be one of 'none', 'target_cv', 'onehot', "
            "'target', 'loo', 'james_stein', or 'loo_logit'. "
            f"Got {method!r}."
        )
    if importlib.util.find_spec("category_encoders") is None:
        raise ImportError(
            "cat_encoding requires category_encoders. Install with: pip install category_encoders"
        )

    import category_encoders as ce

    encoders = {
        "loo": ce.LeaveOneOutEncoder,
        "target": ce.TargetEncoder,
        "james_stein": ce.JamesSteinEncoder,
    }
    Encoder = encoders[method]
    try:
        return Encoder(
            cols=columns,
            handle_missing="return_nan",
            handle_unknown="value",
        )
    except TypeError:
        return Encoder(cols=columns, handle_missing="return_nan")


class _BaseSelector(SelectorMixin, BaseEstimator):
    """Sklearn-style compatibility layer for function-based selectors."""

    __metadata_request__fit = {
        "cache": UNUSED,
        "auto_k_config": UNUSED,
    }
    _selector_fn: Callable
    _subsample_auto_is_cache_default = False
    _random_state_auto_is_cache_default = False

    def _init_selector(self, selector_fn: Callable, params: dict) -> None:
        for name, value in params.items():
            if name != "self":
                setattr(self, name, value)
        self._selector_fn = selector_fn

    def _selector_params(self) -> dict:
        params = {
            name: getattr(self, name)
            for name in inspect.signature(self._selector_fn).parameters
            if name not in _SELECTOR_FORWARD_SKIP_PARAMS and hasattr(self, name)
        }
        # sklearn estimator defaults must be immutable built-in values. Keep
        # the private identity sentinels inside the function API and expose the
        # literal ``"auto"`` as the estimator-facing omission marker.
        return self._resolve_auto_selector_params(params)

    def _output_indices(self) -> np.ndarray:
        check_is_fitted(self, ["selected_indices_", "n_features_in_"])
        return ordered_indices(self.selected_indices_, self.output_order)

    def _encoded_output_names(self) -> list[str]:
        names = list(getattr(self, "_encoded_selected_names_", None) or [])
        if not names:
            return [str(name) for name in self.feature_names_in_[self._output_indices()]]
        if getattr(self, "output_order", "legacy") != "original":
            return names
        encoder = getattr(self, "categorical_encoder_", None)
        parent_pos = {str(name): i for i, name in enumerate(self.feature_names_in_)}
        def _key(encoded_name: str):
            parent = encoder.parent_of(encoded_name) if encoder is not None else encoded_name
            return (parent_pos.get(str(parent), 0), names.index(encoded_name))
        return sorted(names, key=_key)

    def _get_support_mask(self) -> np.ndarray:
        check_is_fitted(self, ["selected_indices_", "n_features_in_"])
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.selected_indices_] = True
        return mask

    def _more_tags(self):
        parent = getattr(super(), "_more_tags", None)
        return selector_tags({} if parent is None else parent())

    def __sklearn_tags__(self):
        parent = getattr(super(), "__sklearn_tags__", None)
        if parent is None:
            return self._more_tags()
        return selector_tags(parent())

    def get_metadata_routing(self):
        routing = super().get_metadata_routing()
        if hasattr(self, "k") and self.k != "auto" and getattr(self, "within", None) is None:
            unsupported = [
                name
                for name in ("groups", "time")
                if routing.fit.requests.get(name) not in (None, False)
            ]
            if unsupported:
                raise ValueError(
                    f"{self.__class__.__name__} can request groups/time metadata only "
                    "when k='auto' or within is set; fixed-k fitting rejects unused "
                    "row context"
                )
        return routing

    def _resolve_auto_selector_params(self, params: dict) -> dict:
        """Translate sklearn-facing auto tokens only for supporting selectors."""
        subsample = params.get("subsample")
        if (
            self._subsample_auto_is_cache_default
            and isinstance(subsample, str)
            and subsample == "auto"
        ):
            params["subsample"] = _SUBSAMPLE_DEFAULT
        random_state = params.get("random_state")
        if (
            self._random_state_auto_is_cache_default
            and isinstance(random_state, str)
            and random_state == "auto"
        ):
            params["random_state"] = _RANDOM_STATE_DEFAULT
        return params

    def _clear_fit_state(self) -> None:
        for attr in (
            "_fit_transform_output_",
            "categorical_encoder_",
            "categorical_encoding_metadata_",
            "categorical_features_",
            "_categorical_encoding_applied_",
            "_fit_feature_names_generated_",
            "feature_names_in_",
            "n_features_in_",
            "selected_features_",
            "selected_indices_",
            "k_",
            "nested_auto_k_diagnostics_",
            "_row_metadata_columns_",
            "_encoded_selected_names_",
            "_encoded_prefix_widths_",
        ):
            if hasattr(self, attr):
                delattr(self, attr)

    def _task(self) -> str:
        return getattr(self, "task", "regression")

    def _supports_auto_k(self) -> bool:
        return True

    def _routes_no_config_auto_k(self) -> bool:
        """Whether k='auto' without a config should use the Auto-K router."""
        return False

    def _categorical_target(self, y):
        return y

    def _categorical_sample_weight(self, y, sample_weight):
        return sample_weight

    def _nested_eval_sample_weight(self, y, sample_weight):
        y_arr = np.asarray(y).reshape(-1)
        return ensure_weights(sample_weight, len(y_arr), normalize=True)

    def _validate_categorical_encoding_params(self) -> None:
        validate_target_cv_encoding_flags(
            getattr(self, "cat_encoding", "none"),
            getattr(self, "allow_full_data_target_encoding", False),
        )

    def _would_fit_supervised_categoricals(self, X) -> bool:
        cat_encoding = getattr(self, "cat_encoding", "none")
        if cat_encoding not in _SUPERVISED_CLASS_ENCODINGS or not isinstance(X, pd.DataFrame):
            return False
        return bool(_categorical_columns(X, getattr(self, "cat_features", None)))

    def _fit_transform_categoricals(
        self,
        X,
        y,
        sample_weight=None,
        groups=None,
        time=None,
    ):
        self.categorical_encoder_ = None
        self.categorical_features_ = []
        self._categorical_encoding_applied_ = False

        cat_encoding = getattr(self, "cat_encoding", "none")
        if cat_encoding == "none" or not isinstance(X, pd.DataFrame):
            return X
        if cat_encoding == "onehot":
            cfg = getattr(self, "auto_k_config", None)
            nested = getattr(cfg, "auto_k_mode", None) == "nested" if cfg is not None else False
            method = getattr(cfg, "k_method", None) if cfg is not None else None
            if (
                getattr(self, "k", None) == "auto"
                and not nested
                and method in {None, "auto", "evaluate", "gaussian_cv", "xfit_objective"}
            ):
                return X

        cat_features = _categorical_columns(X, getattr(self, "cat_features", None))
        self.categorical_features_ = list(cat_features)
        if not cat_features:
            return X

        encoder = _make_category_encoder(
            cat_encoding,
            cat_features,
            target_type="binary" if self._task() == "classification" else "continuous",
            loo_smoothing=getattr(self, "loo_smoothing", 20.0),
            loo_clip_min=getattr(self, "loo_clip_min", 1e-4),
            loo_clip_max=getattr(self, "loo_clip_max", 1.0 - 1e-4),
            target_cv_n_splits=getattr(self, "target_cv_n_splits", 5),
            target_cv_smoothing=getattr(self, "target_cv_smoothing", "auto"),
            target_prior=getattr(self, "target_prior", None),
            warmup_policy=getattr(self, "warmup_policy", "zero_weight"),
            onehot_max_levels=getattr(self, "onehot_max_levels", 32),
        )
        if sample_weight is not None and not isinstance(
            encoder,
            (LeaveOneOutLogitEncoder, TargetCVEncoder, OneHotBlockEncoder),
        ):
            raise ValueError(
                "sample_weight with selector-class categorical encoding is only "
                "supported for cat_encoding='loo_logit'. category_encoders-backed "
                "methods ('loo', 'target', 'james_stein') do not consume sample weights."
            )
        y_enc = self._categorical_target(y)
        with suppress_category_encoder_pandas_warnings():
            if isinstance(encoder, LeaveOneOutLogitEncoder):
                X_encoded = encoder.fit_transform(
                    X,
                    y_enc,
                    sample_weight=self._categorical_sample_weight(y, sample_weight),
                )
            elif isinstance(encoder, TargetCVEncoder):
                target_cv_weight = None
                if sample_weight is not None or getattr(self, "class_weight", None) is not None:
                    target_cv_weight = self._categorical_sample_weight(y, sample_weight)
                X_encoded = encoder.fit_transform(
                    X,
                    y_enc,
                    sample_weight=target_cv_weight,
                    groups=groups,
                    time=time,
                )
            elif isinstance(encoder, OneHotBlockEncoder):
                X_encoded = encoder.fit_transform(X, sample_weight=sample_weight)
            else:
                X_encoded = encoder.fit_transform(X, y_enc)

        self.categorical_encoder_ = encoder
        if hasattr(encoder, "encoding_cv_"):
            self.categorical_encoding_metadata_ = dict(encoder.encoding_cv_)
        self._categorical_encoding_applied_ = True
        return X_encoded

    def _transform_categoricals(self, X):
        if not getattr(self, "_categorical_encoding_applied_", False):
            return X
        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                "This selector was fitted with DataFrame categorical encoding; "
                "transform also requires a DataFrame with matching columns."
            )
        with suppress_category_encoder_pandas_warnings():
            return self.categorical_encoder_.transform(X)

    def _fit_selector(
        self,
        X,
        y,
        *,
        k,
        sample_weight=None,
        groups=None,
        time=None,
        cache=None,
        auto_k_config=None,
        fit_params=None,
        capture_training_output: bool = False,
        encoding_groups=None,
        encoding_time=None,
    ):
        call_params = dict(self._selector_params())

        if cache is not None:
            call_params["cache"] = cache
        if auto_k_config is not None:
            call_params["auto_k_config"] = auto_k_config

        if groups is not None:
            call_params["groups"] = groups
        if time is not None:
            call_params["time"] = time

        call_params["sample_weight"] = sample_weight
        if fit_params:
            blocked = sorted(_BLOCKED_FIT_PARAM_OVERRIDES.intersection(fit_params))
            if blocked:
                blocked_text = ", ".join(blocked)
                raise ValueError(
                    "selector fit-time overrides cannot change return shape or "
                    "preprocessing-affecting parameters: "
                    f"{blocked_text}"
                )
            call_params.update(fit_params)
        self._resolve_auto_selector_params(call_params)

        feature_names, names_generated = _feature_names_with_provenance(X)
        X_fit = self._fit_transform_categoricals(
            X,
            y,
            sample_weight=sample_weight,
            groups=encoding_groups,
            time=encoding_time,
        )
        effective_sample_weight = sample_weight
        if (
            getattr(self, "cat_encoding", "none") == "target_cv"
            and getattr(self, "categorical_encoder_", None) is not None
            and getattr(self.categorical_encoder_, "effective_sample_weight_", None)
            is not None
        ):
            effective_sample_weight = self.categorical_encoder_.effective_sample_weight_
        call_params["sample_weight"] = effective_sample_weight
        if getattr(self, "_categorical_encoding_applied_", False):
            call_params["cat_features"] = None
            call_params["cat_encoding"] = "none"
            call_params["allow_full_data_target_encoding"] = False
        onehot_encoder, composed, encoded_names = _apply_onehot_call_params(
            self, call_params, feature_names
        )

        result = self._selector_fn(
            X_fit,
            y,
            k=k,
            return_result=True,
            **call_params,
        )
        if hasattr(result, "selected_indices"):
            selected_features = list(result.selected_features)
            selected_indices = result.selected_indices
        else:
            selected_features = list(result)
            selected_indices = None
        if selected_indices is None:
            selected_indices = _coerce_selection_indices(
                feature_names,
                selected_features,
            ).tolist()

        self._encoded_selected_names_ = None
        self._encoded_prefix_widths_ = None
        if isinstance(onehot_encoder, OneHotBlockEncoder):
            encoded_selected = list(selected_features)
            self._encoded_selected_names_ = encoded_selected
            selected_features = onehot_encoder.collapse_to_raw(encoded_selected)
            selected_indices = _coerce_selection_indices(
                feature_names,
                selected_features,
            ).tolist()
            from sift.selection.blocks import discovery_prefix_widths

            dummy_index = {name: i for i, name in enumerate(encoded_names)}
            dummy_path = [dummy_index[name] for name in encoded_selected if name in dummy_index]
            include_dummy = set()
            if call_params.get("include"):
                include_dummy = set(call_params["include"])
            include_idx = {
                dummy_index[name] for name in include_dummy if name in dummy_index
            }
            discoveries = [i for i in dummy_path if i not in include_idx]
            n_include = sum(1 for i in dummy_path if i in include_idx)
            widths = discovery_prefix_widths(discoveries, composed)
            if widths:
                self._encoded_prefix_widths_ = tuple(int(n_include + w) for w in widths)
        elif (
            getattr(self, "cat_encoding", None) == "onehot"
            and isinstance(X, pd.DataFrame)
        ):
            cols = _categorical_columns(X, getattr(self, "cat_features", None))
            if cols:
                enc = OneHotBlockEncoder(
                    cols,
                    max_levels=validate_onehot_max_levels(
                        getattr(self, "onehot_max_levels", 32)
                    ),
                )
                enc.fit(X, sample_weight=sample_weight)
                self.categorical_encoder_ = enc
                self._categorical_encoding_applied_ = True
                self._encoded_selected_names_ = enc.expand_selected(selected_features)

        self.feature_names_in_ = feature_names_array(feature_names)
        self._fit_feature_names_generated_ = names_generated
        self.n_features_in_ = len(feature_names)
        self.selected_features_ = selected_features
        self.selected_indices_ = np.asarray(selected_indices, dtype=np.int64)
        if capture_training_output:
            if self._encoded_selected_names_ is not None and isinstance(X_fit, pd.DataFrame):
                self._fit_transform_output_ = X_fit.loc[:, self._encoded_output_names()].copy()
            else:
                self._fit_transform_output_ = _selected_training_output(
                    X_fit,
                    self._output_indices(),
                )
        return self

    def _fit_impl(
        self,
        X,
        y,
        *,
        sample_weight=None,
        groups=None,
        time=None,
        cache=None,
        auto_k_config=None,
        capture_training_output: bool = False,
        encoding_groups=None,
        encoding_time=None,
        **fit_params,
    ):
        _require_2d_x(X)
        validate_output_order(self.output_order)
        self._validate_categorical_encoding_params()
        resolved_cache = cache if cache is not None else getattr(self, "cache", None)
        resolved_auto_k = auto_k_config
        if resolved_auto_k is None:
            resolved_auto_k = getattr(self, "auto_k_config", None)

        self._clear_fit_state()
        metadata = resolve_row_metadata(X, groups=groups, time=time)
        X = metadata.X
        validate_fit_matrix(X)
        groups = metadata.groups
        time = metadata.time
        self._row_metadata_columns_ = metadata.extracted_columns
        has_supervised_categoricals = self._would_fit_supervised_categoricals(X)
        if (
            has_supervised_categoricals
            and getattr(self, "cat_encoding", "none") == "target_cv"
            and (groups is not None or time is not None)
            and self.k != "auto"
        ):
            raise ValueError(
                "cat_encoding='target_cv' with groups/time is supported only for "
                "k='auto' nested evaluate paths"
            )

        if resolved_cache is not None and has_supervised_categoricals:
            raise ValueError(
                "selector-class supervised categorical encoding does not support "
                "prebuilt caches. Use cat_encoding='none' with a cache, or omit the "
                "cache so the selector can fit encoders on the training rows."
            )
        if (
            resolved_cache is not None
            and getattr(self, "cat_encoding", "none") == "onehot"
        ):
            raise ValueError(
                "cat_encoding='onehot' cannot be combined with a prebuilt cache "
                "because the cache has no one-hot provenance"
            )
        if getattr(self, "cat_encoding", "none") == "onehot":
            validate_onehot_max_levels(getattr(self, "onehot_max_levels", 32))
            if getattr(self, "within", None) is not None:
                raise ValueError(
                    "cat_encoding='onehot' is not supported with within panel demeaning"
                )

        if self.k == "auto":
            if not self._supports_auto_k():
                raise ValueError(
                    f"{self.__class__.__name__} requires a fixed positive integer k; "
                    "k='auto' is not supported."
                )
            if resolved_auto_k is None and self._routes_no_config_auto_k():
                # Mirror the function API: CEFS+ selectors without an explicit
                # config use the measured Auto-K router instead of the legacy
                # evaluate/time_holdout inference.
                resolved_auto_k = AutoKConfig(k_method="auto")
            effective_auto_k = resolve_auto_k_config(
                resolved_auto_k,
                time,
                groups,
                allow_nested=True,
            )
            if (
                has_supervised_categoricals
                and getattr(self, "cat_encoding", "none") == "target_cv"
                and (groups is not None or time is not None)
                and not (
                    effective_auto_k.auto_k_mode == "nested"
                    and effective_auto_k.k_method == "evaluate"
                )
            ):
                raise ValueError(
                    "contextual cat_encoding='target_cv' requires an explicit "
                    "AutoKConfig(auto_k_mode='nested', k_method='evaluate')"
                )
            if getattr(self, "cat_encoding", "none") == "onehot":
                from sift.selection.blocks import require_onehot_auto_k

                require_onehot_auto_k(effective_auto_k.k_method)
            if effective_auto_k.auto_k_mode == "nested":
                if effective_auto_k.k_method != "evaluate":
                    raise ValueError(
                        "auto_k_mode='nested' currently supports only "
                        "k_method='evaluate'"
                    )
                return self._fit_nested_auto_k(
                    X,
                    y,
                    sample_weight=sample_weight,
                    groups=groups,
                    time=time,
                    cache=resolved_cache,
                    auto_k_config=effective_auto_k,
                    fit_params=fit_params,
                    capture_training_output=capture_training_output,
                )

            if (
                effective_auto_k.auto_k_mode == "prefix_only"
                and effective_auto_k.k_method == "evaluate"
                and has_supervised_categoricals
            ):
                raise ValueError(
                    "prefix_only auto-k with supervised selector-class categorical "
                    "encoding would evaluate target-encoded validation rows. Use "
                    "auto_k_mode='nested' or pre-encode/cross-fit categoricals "
                    "outside the selector."
                )

            resolved_auto_k = effective_auto_k

        return self._fit_selector(
            X,
            y,
            k=self.k,
            sample_weight=sample_weight,
            groups=groups,
            time=time,
            cache=resolved_cache,
            auto_k_config=resolved_auto_k,
            fit_params=fit_params,
            capture_training_output=capture_training_output,
            encoding_groups=encoding_groups,
            encoding_time=encoding_time,
        )

    def fit(
        self,
        X,
        y,
        *,
        sample_weight=None,
        groups=None,
        time=None,
        cache=None,
        auto_k_config=None,
        **fit_params,
    ):
        try:
            return self._fit_impl(
                X,
                y,
                sample_weight=sample_weight,
                groups=groups,
                time=time,
                cache=cache,
                auto_k_config=auto_k_config,
                capture_training_output=False,
                **fit_params,
            )
        except Exception:
            self._clear_fit_state()
            raise

    def fit_transform(self, X, y=None, **fit_params):
        """Fit the selector and return the training matrix used for fitting.

        For supervised categorical encoders this avoids sklearn's default
        ``fit(X, y).transform(X)`` behavior, which would call a target-blind
        transform on the training rows and could differ from the y-aware encoded
        matrix used during feature selection.
        """
        try:
            self._fit_impl(X, y, capture_training_output=True, **fit_params)
            captured = getattr(self, "_fit_transform_output_", None)
            if captured is not None:
                return captured
            return self.transform(X)
        except Exception:
            self._clear_fit_state()
            raise
        finally:
            if hasattr(self, "_fit_transform_output_"):
                delattr(self, "_fit_transform_output_")

    def _clone_for_nested_path(self, k: int):
        params = self.get_params(deep=False)
        params["k"] = k
        if "auto_k_config" in params:
            params["auto_k_config"] = None
        if "cache" in params:
            params["cache"] = None
        if "verbose" in params:
            params["verbose"] = False
        if "callback" in params:
            # Nested folds each build their own local path. Keep the public
            # callback sequence reserved for the final refit instead of
            # emitting several unrelated 1..max_k sequences.
            params["callback"] = None
        if "output_order" in params:
            # Prefix evaluation always follows the learned selector path.
            params["output_order"] = "legacy"
        return self.__class__(**params)

    def _fit_nested_auto_k(
        self,
        X,
        y,
        *,
        sample_weight=None,
        groups=None,
        time=None,
        cache=None,
        auto_k_config=None,
        fit_params=None,
        capture_training_output: bool = False,
        encoding_groups=None,
        encoding_time=None,
    ):
        if cache is not None:
            raise ValueError("auto_k_mode='nested' does not support prebuilt caches")
        if getattr(self, "within", None) is not None:
            raise ValueError(
                "within is not supported with auto_k_mode='nested'; use "
                "function-style prefix_only evaluate, gaussian_cv, or "
                "xfit_objective so demeaning stays fold-local"
            )

        y_arr = np.asarray(y).reshape(-1)
        n_units = _nested_discovery_units(self, X, fit_params)
        n_features = n_units if n_units is not None else len(_feature_names_or_default(X))
        config = auto_k_config
        fit_w_arr = (
            ensure_weights(sample_weight, len(y_arr), normalize=True)
            if sample_weight is not None
            else None
        )
        eval_w_arr = self._nested_eval_sample_weight(y, sample_weight)
        fold_fit_params = dict(fit_params or {})
        # A fit-time callback override belongs to the public fit, just like a
        # constructor callback. Fold-local paths restart at step one, so keep
        # them silent and report only the final full-data refit below.
        fold_fit_params.pop("callback", None)

        def build_fold_path(train_idx: np.ndarray, val_idx: np.ndarray, max_k: int):
            fold_selector = self._clone_for_nested_path(max_k)
            train_X = _slice_rows(X, train_idx)
            X_train_path = fold_selector.fit_transform(
                train_X,
                y_arr[train_idx],
                sample_weight=fit_w_arr[train_idx] if fit_w_arr is not None else None,
                encoding_groups=groups[train_idx] if groups is not None else None,
                encoding_time=time[train_idx] if time is not None else None,
                **fold_fit_params,
            )
            X_val_path = fold_selector.transform(_slice_rows(X, val_idx))
            feature_blocks, include, _exclude, _candidates = (
                _effective_nested_blocks_and_conditioning(self, fold_fit_params)
            )
            prefix_sizes = _nested_block_prefix_sizes(
                fold_selector,
                train_X,
                feature_blocks=feature_blocks,
                include=include,
            )
            n_include = 0
            if include is not None and include is not False:
                n_include = len(list(include))
            encoded_names = getattr(fold_selector, "_encoded_selected_names_", None)
            encoder = getattr(fold_selector, "categorical_encoder_", None)
            path_parents = None
            if encoded_names and isinstance(encoder, OneHotBlockEncoder):
                feature_path = list(encoded_names)
                path_parents = tuple(encoder.parent_of(name) for name in feature_path)
            else:
                feature_path = list(fold_selector.selected_features_)
            return NestedAutoKFold(
                train_path=X_train_path,
                val_path=X_val_path,
                feature_path=feature_path,
                prefix_sizes=prefix_sizes,
                n_include_features=int(n_include),
                path_parents=path_parents,
            )

        nested = select_k_nested(
            X,
            y_arr,
            n_features=n_features,
            config=config,
            build_fold_path=build_fold_path,
            groups=groups,
            time=time,
            sample_weight=eval_w_arr,
            task=self._task(),
        )
        self.nested_auto_k_diagnostics_ = nested.diagnostics
        self.k_ = nested.selected_k

        return self._fit_selector(
            X,
            y,
            k=nested.selected_k,
            sample_weight=sample_weight,
            # The selected k is now fixed; groups/time were used by the
            # nested evaluator and must not be forwarded to the public fixed-k
            # filter call, where they have no meaning.
            groups=None,
            time=None,
            cache=None,
            auto_k_config=None,
            fit_params=fit_params,
            capture_training_output=capture_training_output,
            encoding_groups=groups,
            encoding_time=time,
        )

    def transform(self, X):
        check_is_fitted(
            self,
            ["selected_indices_", "selected_features_", "feature_names_in_"],
        )
        reject_sparse(X, operation="transform")
        X = drop_fitted_metadata_columns(
            X,
            getattr(self, "_row_metadata_columns_", ()),
        )
        if isinstance(X, pd.DataFrame):
            check_fitted_column_identity(X, self.feature_names_in_)
            X = self._transform_categoricals(X)
            encoded_names = getattr(self, "_encoded_selected_names_", None)
            if encoded_names:
                return X.loc[:, self._encoded_output_names()]
            return X.iloc[:, self._output_indices()]
        X_arr = np.asarray(X)
        if getattr(self, "_categorical_encoding_applied_", False):
            raise ValueError(
                "This selector was fitted with categorical DataFrame encoding; "
                "transform also requires a DataFrame."
            )
        if X_arr.ndim != 2:
            raise ValueError(
                "X must be a 2D feature matrix. Reshape your data with "
                "X.reshape(-1, 1) for a single feature."
            )
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, but selector was fitted with "
                f"{self.n_features_in_}"
            )
        return X_arr[:, self._output_indices()]

    def get_support(self, indices: bool = False) -> np.ndarray:
        """Return selected-feature mask (default) or indices (indices=True)."""
        if indices:
            return self._output_indices()
        return self._get_support_mask()

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return names of selected features following sklearn's transformer API."""
        check_is_fitted(self, ["selected_indices_", "feature_names_in_", "n_features_in_"])
        fitted_names = feature_names_array(self.feature_names_in_)
        if input_features is not None:
            input_names = np.asarray(input_features, dtype=object)
            if input_names.ndim != 1 or input_names.shape[0] != self.n_features_in_:
                raise ValueError(
                    "input_features must have the same number of features as the fitted data"
                )
            if not np.array_equal(input_names, fitted_names):
                raise ValueError(
                    "input_features is not equal to feature_names_in_"
                )
        encoded_names = getattr(self, "_encoded_selected_names_", None)
        if encoded_names:
            return np.asarray(self._encoded_output_names(), dtype=object)
        return fitted_names[self._output_indices()]

    def inverse_transform(self, X):
        """Restore selected values to their fitted raw-column positions."""
        check_is_fitted(self, ["selected_indices_", "n_features_in_"])
        if getattr(self, "_categorical_encoding_applied_", False):
            raise NotImplementedError(
                "inverse_transform is unavailable after supervised categorical "
                "encoding because the fitted encoder is not invertible"
            )
        return inverse_selected_matrix(
            X,
            n_features=self.n_features_in_,
            selected_indices=self._output_indices(),
        )


class MRMRSelector(_BaseSelector):
    """Sklearn-style wrapper for ``sift.select_mrmr``.

    Minimum-redundancy maximum-relevance grows a greedy path that trades target
    relevance against redundancy with the features already chosen, and stops at
    ``k``. Use it as a fast, model-free pre-filter inside an sklearn
    ``Pipeline`` whenever a fixed-width, low-redundancy feature block is
    wanted. ``fit`` learns the path and records the selection; ``transform``
    returns the selected columns in the fitted container kind (DataFrame in,
    DataFrame out; ndarray in, ndarray out), ``set_output(transform="pandas")``
    is honored like any sklearn transformer, and ``inverse_transform`` restores
    a dense full-width matrix with unselected columns zero-filled. Sparse input
    is rejected in ``fit``, ``transform`` and ``inverse_transform``.

    ``k`` is an upper bound rather than a promise: constant-column filtering,
    relevance screening and non-positive objective checks can end the path with
    fewer than ``k`` features.

    Parameters
    ----------
    k : int or {"auto"}, default=10
        Upper bound on the number of selected features. ``"auto"`` defers the
        size to ``auto_k_config``; with no config, automatic sizing needs
        ``groups`` or ``time`` at fit time (they select ``strategy="group_cv"``
        or ``"time_holdout"``) and otherwise raises.
    task : {"regression", "classification"}, default="regression"
        Target kind. It picks the relevance scorer and the estimator variants.
    relevance : {"f", "ks", "rf"}, default="f"
        Relevance score. Regression accepts ``"f"`` and ``"rf"``;
        classification also accepts ``"ks"``. Any other pairing raises.
    estimator : {"classic", "gaussian"}, default="classic"
        Redundancy estimator. ``"classic"`` scores redundancy on the raw rows;
        ``"gaussian"`` is a regression-only fast path over the Gaussian-copula
        ``sift.FeatureCache`` and the only route that accepts ``cache``.
    formula : {"quotient", "difference"}, default="quotient"
        Objective shape: relevance divided by mean redundancy, or relevance
        minus mean redundancy.
    top_m : int or None, default=None
        Keep only the ``top_m`` most relevant valid candidates before the
        greedy loop. ``None`` resolves to ``max(5 * k, 250)``, never below
        ``k``.
    cat_features : list of str or None, default=None
        Categorical column names to encode. ``None`` auto-detects ``object``,
        ``category`` and ``string`` DataFrame columns. Unused when
        ``cat_encoding="none"`` or when ``X`` is an ndarray.
    cat_encoding : {"none", "target_cv", "onehot", "target", "loo", "james_stein", "loo_logit"}, default="none"
        One of ``"none"``, ``"target_cv"``, ``"target"``, ``"loo"``,
        ``"james_stein"`` or ``"loo_logit"``, fitted inside ``fit``.
        ``"target_cv"`` is the built-in leakage-safe contract: out-of-fold
        training rows receive
        ``fold_encoding - fold_training_prior`` while inference rows receive
        ``full_fit_encoding - full_training_prior``, so an unseen category maps
        to a centered zero and cannot identify its own fold. ``"target"``,
        ``"loo"`` and ``"james_stein"`` require the optional
        ``category_encoders`` package; ``"loo_logit"`` is SIFT's own
        leave-one-out logit encoder and the only one that accepts
        ``sample_weight`` besides ``"target_cv"``. Any supervised encoding
        makes ``fit_transform`` return the y-aware encoded training block and
        makes ``inverse_transform`` unavailable.
    target_cv_n_splits : int, default=5
        Fold count for ``cat_encoding="target_cv"``.
    target_cv_smoothing : {"auto"} or float, default="auto"
        Empirical-Bayes shrinkage for ``"target_cv"``. ``"auto"`` reproduces
        sklearn's ``TargetEncoder`` rule on unweighted fixed-k folds and uses
        weighted row mass elsewhere; an explicit non-negative float always
        works.
    target_prior : float or None, default=None
        Target-independent prior for time-aware ``"target_cv"`` fits, so the
        earliest block emits a centered neutral zero and stays in the fit.
    warmup_policy : {"exclude", "zero_weight"}, default="zero_weight"
        How to treat the earliest no-history block of a time-aware
        ``"target_cv"`` fit when no ``target_prior`` is given.
    allow_full_data_target_encoding : bool, default=False
        Opt in to fitting the 0.8 supervised encoders on the full matrix. It is
        rejected together with ``cat_encoding="target_cv"``, whose cross-fitted
        contract it contradicts.
    onehot_max_levels : int, default=32
        Cap on retained dummy levels per categorical when
        ``cat_encoding="onehot"``. Surplus levels share ``other``.
    subsample : int, None or {"auto"}, default="auto"
        Row cap for classic row sampling and for uncached Gaussian cache
        construction. ``"auto"`` is the sklearn-clonable spelling of the
        omitted default: 50,000 rows when fitting from ``X``, and "not
        supplied" when ``cache`` is given. ``None`` keeps every positively
        weighted row. An explicit value beside a ``cache`` raises.
    random_state : int or {"auto"}, default="auto"
        Seed for that row sampling and for uncached cache construction.
        ``"auto"`` is the omitted default and resolves to seed 0 when fitting
        from ``X``. An explicit value beside a ``cache`` raises.
    n_jobs : int, default=1
        Worker count for the redundancy loop.
    mrmr_backend : {"auto", "serial", "blas", "processes"}, default="auto"
        Redundancy-update backend. ``"auto"`` resolves to ``"blas"``
        regardless of ``n_jobs``, because the BLAS matvec update avoids
        process start-up and pickling costs; pass ``"processes"`` explicitly
        to opt into joblib workers.
    verbose : bool, default=True
        Emit progress at INFO on the ``sift`` logger.
    cache : FeatureCache or None, default=None
        Prebuilt Gaussian-copula cache to reuse with ``estimator="gaussian"``.
        A named cache requires a DataFrame with identical columns in identical
        order; a positional cache requires the matching ndarray. A cache cannot
        be combined with a supervised ``cat_encoding``.
    auto_k_config : AutoKConfig or None, default=None
        Automatic-sizing configuration, read only when ``k="auto"``. Selector
        classes additionally accept ``auto_k_mode="nested"`` together with
        ``k_method="evaluate"``, which refits a train-only path per split.
    within : {"groups", "two_way"} or None, default=None
        Panel demeaning applied after encoding and before ranks. ``"groups"``
        subtracts per-entity weighted means; ``"two_way"`` alternates entity
        and time demeaning for five iterations. Regression only. Fixed-``k``
        fits then require ``groups`` (and ``time`` for ``"two_way"``).
        ``transform`` still returns selected raw columns.
    callback : ProgressCallback or None, default=None
        ``callback(step, total, info)`` called after each completed greedy
        step. Nested auto-k folds stay silent; only the final refit reports.
    include : sequence of names or positions, optional
        Conditioning set. Selector state is initialized from these features
        before step 1. They appear in the fitted selection in caller order
        but are not discoveries; ``k`` counts additional features.
    exclude : sequence of names or positions, optional
        Features removed from the discovery pool. Cannot overlap ``include``.
    candidates : sequence of names or positions, optional
        Hard allow-list for discovery. ``include`` may sit outside it.
        Overlap with ``exclude`` is rejected. An empty remaining pool raises.
    feature_blocks : mapping, {"auto"} or None, default None
        Atomic column groups. ``k`` counts additional blocks; selected
        blocks expand to raw member columns. ``"auto"`` uses the
        ``{block}__{level}`` one-hot prefix; ordinary underscores are not
        split. ``k="auto"`` counts additional blocks on evaluate, elbow,
        penalized_objective, gaussian_cv, xfit_objective, and auto routing;
        calibrated column-step rules raise.
    output_order : {"legacy", "original"}, default="legacy"
        Order used by ``transform``, ``get_support(indices=True)``,
        ``get_feature_names_out`` and ``inverse_transform``. ``"legacy"`` keeps
        selection-path order; ``"original"`` emits ascending fitted column
        position. The boolean support mask is always positional.

    Attributes
    ----------
    selected_features_ : list
        Selected feature labels in selection-path order.
    selected_indices_ : ndarray of shape (n_selected,)
        Their positions in the fitted feature matrix, in path order.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        One-dimensional object array of fitted feature names. A positional
        ndarray fit stores generated ``x0...`` names here.
    n_features_in_ : int
        Number of candidate features seen during ``fit``.
    k_ : int
        Feature count chosen by nested automatic k. Set only on the
        ``auto_k_mode="nested"`` path, not by prefix-only auto-k.
    nested_auto_k_diagnostics_ : dict
        Fold scores, metric and selection rule behind ``k_``; same path only.
    categorical_features_ : list
        Categorical columns the fitted encoder covered.
    categorical_encoder_ : object or None
        The fitted encoder, reused target-blind by ``transform``.
    categorical_encoding_metadata_ : dict
        The encoder's own ``{"kind": ..., "n_splits": ...}``, present only when
        ``cat_encoding="target_cv"`` encoded at least one column.

    Raises
    ------
    ValueError
        If ``groups``/``time`` reach a fixed-``k`` fit, if ``k="auto"`` has
        neither ``auto_k_config`` nor row context, if ``subsample`` or
        ``random_state`` is explicit beside a ``cache``, if a supervised
        ``cat_encoding`` is combined with a ``cache``, or if ``X`` is sparse or
        not two-dimensional.
    NotImplementedError
        From ``inverse_transform`` after a supervised categorical encoding,
        because the fitted encoder is not invertible.

    Warns
    -----
    UserWarning
        When an explicit ``AutoKConfig(k_method="auto")`` router supports zero
        features, or saturates its effective ``max_k`` (a censored result).

    See Also
    --------
    sift.select_mrmr : Function-style mRMR with the same options.
    JMISelector : Complementarity-driven joint mutual information.
    CEFSPlusSelector : Gaussian log-determinant conditional-information path.
    KnockoffSelector : q-calibrated selection instead of a fixed ``k``.

    Notes
    -----
    The shared selector-class fit contract is
    ``fit(X, y, sample_weight=None, groups=None, time=None)``; ``cache`` and
    ``auto_k_config`` may also be passed per call and then win over the
    constructor. Fixed-``k`` fits reject unused ``groups``/``time`` unless
    ``within`` is set, while ``k="auto"`` accepts them for split construction,
    including the DataFrame shorthand ``groups="col"``/``time="col"``
    that moves the column out of the candidate features. Under sklearn >= 1.4
    metadata routing every datum must be requested explicitly with
    ``set_fit_request(...)``, and a fixed-``k`` estimator without ``within``
    refuses a ``groups``/``time`` request.

    Examples
    --------
    >>> import numpy as np
    >>> from sift import MRMRSelector
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 6))
    >>> y = X[:, 0] + 0.5 * X[:, 1] + 0.1 * rng.normal(size=200)
    >>> selector = MRMRSelector(k=2, task="regression", verbose=False)
    >>> selector.fit(X, y).selected_features_
    ['x0', 'x1']
    >>> selector.transform(X).shape
    (200, 2)
    """

    _subsample_auto_is_cache_default = True
    _random_state_auto_is_cache_default = True

    def __init__(
        self,
        k: int | str = 10,
        *,
        task: str = "regression",
        relevance: str = "f",
        estimator: str = "classic",
        formula: str = "quotient",
        top_m: int | None = None,
        cat_features: list[str] | None = None,
        cat_encoding: str = "none",
        target_cv_n_splits: int = 5,
        target_cv_smoothing: Literal["auto"] | float = "auto",
        target_prior: float | None = None,
        warmup_policy: Literal["exclude", "zero_weight"] = "zero_weight",
        allow_full_data_target_encoding: bool = False,
        onehot_max_levels: int = 32,
        subsample: int | None | Literal["auto"] = "auto",
        random_state: int | Literal["auto"] = "auto",
        n_jobs: int = 1,
        mrmr_backend: str = "auto",
        verbose: bool = True,
        cache=None,
        auto_k_config=None,
        within: str | None = None,
        include=None,
        exclude=None,
        candidates=None,
        feature_blocks=None,
        callback: ProgressCallback | None = None,
        output_order: str = "legacy",
    ):
        self._init_selector(select_mrmr, locals())


class JMISelector(_BaseSelector):
    """Sklearn-style wrapper for ``sift.select_jmi``.

    Joint mutual information adds, at every step, the candidate whose *summed*
    joint information with the target given each already-selected feature is
    largest, so it prefers features that complement the current set rather than
    merely scoring well alone. Reach for it over mRMR when interactions matter
    more than raw marginal relevance. ``fit`` learns the path and records the
    selection; ``transform`` returns the selected columns in the fitted
    container kind, ``set_output(transform="pandas")`` is honored like any
    sklearn transformer, and ``inverse_transform`` restores a dense full-width
    matrix with unselected columns zero-filled. Sparse input is rejected in
    ``fit``, ``transform`` and ``inverse_transform``.

    ``k`` is an upper bound rather than a promise: constant-column filtering,
    relevance screening and non-positive objective checks can end the path with
    fewer than ``k`` features.

    Parameters
    ----------
    k : int or {"auto"}, default=10
        Upper bound on the number of selected features. ``"auto"`` defers the
        size to ``auto_k_config``; with no config, automatic sizing needs
        ``groups`` or ``time`` at fit time (they select ``strategy="group_cv"``
        or ``"time_holdout"``) and otherwise raises.
    task : {"regression", "classification"}, default="regression"
        Target kind. It picks the relevance scorer and the estimator variants.
    estimator : {"auto", "binned", "r2", "ksg", "gaussian"}, default="auto"
        Mutual-information estimator for the joint terms. ``"auto"`` resolves
        to ``"binned"`` for classification and ``"r2"`` for regression;
        ``"gaussian"`` is the copula-cache fast path and the only route that
        accepts ``cache``; ``"ksg"`` is the nearest-neighbour estimator and
        rejects ``sample_weight``.
    relevance : {"f", "ks", "rf"}, default="f"
        Relevance score used for screening and the first step. Regression
        accepts ``"f"`` and ``"rf"``; classification also accepts ``"ks"``.
    top_m : int or None, default=None
        Keep only the ``top_m`` most relevant valid candidates before the
        greedy loop. ``None`` resolves to ``max(5 * k, 250)``, never below
        ``k``.
    cat_features : list of str or None, default=None
        Categorical column names to encode. ``None`` auto-detects ``object``,
        ``category`` and ``string`` DataFrame columns. Unused when
        ``cat_encoding="none"`` or when ``X`` is an ndarray.
    cat_encoding : {"none", "target_cv", "onehot", "target", "loo", "james_stein", "loo_logit"}, default="none"
        One of ``"none"``, ``"target_cv"``, ``"target"``, ``"loo"``,
        ``"james_stein"`` or ``"loo_logit"``, fitted inside ``fit``.
        ``"target_cv"`` is the built-in leakage-safe contract: out-of-fold
        training rows receive ``fold_encoding - fold_training_prior`` while
        inference rows receive ``full_fit_encoding - full_training_prior``, so
        an unseen category maps to a centered zero and cannot identify its own
        fold. ``"target"``, ``"loo"`` and ``"james_stein"`` require the
        optional ``category_encoders`` package; ``"loo_logit"`` is SIFT's own
        leave-one-out logit encoder and the only one that accepts
        ``sample_weight`` besides ``"target_cv"``. Any supervised encoding
        makes ``fit_transform`` return the y-aware encoded training block and
        makes ``inverse_transform`` unavailable.
    target_cv_n_splits : int, default=5
        Fold count for ``cat_encoding="target_cv"``.
    target_cv_smoothing : {"auto"} or float, default="auto"
        Empirical-Bayes shrinkage for ``"target_cv"``. ``"auto"`` reproduces
        sklearn's ``TargetEncoder`` rule on unweighted fixed-k folds and uses
        weighted row mass elsewhere; an explicit non-negative float always
        works.
    target_prior : float or None, default=None
        Target-independent prior for time-aware ``"target_cv"`` fits, so the
        earliest block emits a centered neutral zero and stays in the fit.
    warmup_policy : {"exclude", "zero_weight"}, default="zero_weight"
        How to treat the earliest no-history block of a time-aware
        ``"target_cv"`` fit when no ``target_prior`` is given.
    allow_full_data_target_encoding : bool, default=False
        Opt in to fitting the 0.8 supervised encoders on the full matrix. It is
        rejected together with ``cat_encoding="target_cv"``, whose cross-fitted
        contract it contradicts.
    onehot_max_levels : int, default=32
        Cap on retained dummy levels per categorical when
        ``cat_encoding="onehot"``. Surplus levels share ``other``.
    subsample : int, None or {"auto"}, default="auto"
        Row cap for classic row sampling and for uncached Gaussian cache
        construction. ``"auto"`` is the sklearn-clonable spelling of the
        omitted default: 50,000 rows when fitting from ``X``, and "not
        supplied" when ``cache`` is given. ``None`` keeps every positively
        weighted row. An explicit value beside a ``cache`` raises.
    random_state : int or {"auto"}, default="auto"
        Seed for that row sampling and for uncached cache construction.
        ``"auto"`` is the omitted default and resolves to seed 0 when fitting
        from ``X``. An explicit value beside a ``cache`` raises.
    verbose : bool, default=True
        Emit progress at INFO on the ``sift`` logger.
    cache : FeatureCache or None, default=None
        Prebuilt Gaussian-copula cache to reuse with ``estimator="gaussian"``.
        A named cache requires a DataFrame with identical columns in identical
        order; a positional cache requires the matching ndarray. A cache cannot
        be combined with a supervised ``cat_encoding``.
    auto_k_config : AutoKConfig or None, default=None
        Automatic-sizing configuration, read only when ``k="auto"``. Selector
        classes additionally accept ``auto_k_mode="nested"`` together with
        ``k_method="evaluate"``, which refits a train-only path per split.
    within : {"groups", "two_way"} or None, default=None
        Panel demeaning applied after encoding and before ranks. ``"groups"``
        subtracts per-entity weighted means; ``"two_way"`` alternates entity
        and time demeaning for five iterations. Regression only. Fixed-``k``
        fits then require ``groups`` (and ``time`` for ``"two_way"``).
        ``transform`` still returns selected raw columns.
    callback : ProgressCallback or None, default=None
        ``callback(step, total, info)`` called after each completed greedy
        step. Nested auto-k folds stay silent; only the final refit reports.
    include : sequence of names or positions, optional
        Conditioning set. Selector state is initialized from these features
        before step 1. They appear in the fitted selection in caller order
        but are not discoveries; ``k`` counts additional features.
    exclude : sequence of names or positions, optional
        Features removed from the discovery pool. Cannot overlap ``include``.
    candidates : sequence of names or positions, optional
        Hard allow-list for discovery. ``include`` may sit outside it.
        Overlap with ``exclude`` is rejected. An empty remaining pool raises.
    feature_blocks : mapping, {"auto"} or None, default None
        Atomic column groups. ``k`` counts additional blocks; selected
        blocks expand to raw member columns. ``"auto"`` uses the
        ``{block}__{level}`` one-hot prefix; ordinary underscores are not
        split. ``k="auto"`` counts additional blocks on evaluate, elbow,
        penalized_objective, gaussian_cv, xfit_objective, and auto routing;
        calibrated column-step rules raise.
    output_order : {"legacy", "original"}, default="legacy"
        Order used by ``transform``, ``get_support(indices=True)``,
        ``get_feature_names_out`` and ``inverse_transform``. ``"legacy"`` keeps
        selection-path order; ``"original"`` emits ascending fitted column
        position. The boolean support mask is always positional.

    Attributes
    ----------
    selected_features_ : list
        Selected feature labels in selection-path order.
    selected_indices_ : ndarray of shape (n_selected,)
        Their positions in the fitted feature matrix, in path order.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        One-dimensional object array of fitted feature names. A positional
        ndarray fit stores generated ``x0...`` names here.
    n_features_in_ : int
        Number of candidate features seen during ``fit``.
    k_ : int
        Feature count chosen by nested automatic k. Set only on the
        ``auto_k_mode="nested"`` path, not by prefix-only auto-k.
    nested_auto_k_diagnostics_ : dict
        Fold scores, metric and selection rule behind ``k_``; same path only.
    categorical_features_ : list
        Categorical columns the fitted encoder covered.
    categorical_encoder_ : object or None
        The fitted encoder, reused target-blind by ``transform``.
    categorical_encoding_metadata_ : dict
        The encoder's own ``{"kind": ..., "n_splits": ...}``, present only when
        ``cat_encoding="target_cv"`` encoded at least one column.

    Raises
    ------
    ValueError
        If ``groups``/``time`` reach a fixed-``k`` fit, if ``k="auto"`` has
        neither ``auto_k_config`` nor row context, if ``subsample`` or
        ``random_state`` is explicit beside a ``cache``, if a supervised
        ``cat_encoding`` is combined with a ``cache``, if
        ``estimator="ksg"`` is combined with ``sample_weight``, or if ``X`` is
        sparse or not two-dimensional.
    NotImplementedError
        From ``inverse_transform`` after a supervised categorical encoding,
        because the fitted encoder is not invertible.

    Warns
    -----
    UserWarning
        When an explicit ``AutoKConfig(k_method="auto")`` router supports zero
        features, or saturates its effective ``max_k`` (a censored result).

    See Also
    --------
    sift.select_jmi : Function-style JMI with the same options.
    JMIMSelector : Conservative minimum-pair variant of the same objective.
    MRMRSelector : Relevance-versus-redundancy greedy path.

    Notes
    -----
    The shared selector-class fit contract is
    ``fit(X, y, sample_weight=None, groups=None, time=None)``; ``cache`` and
    ``auto_k_config`` may also be passed per call and then win over the
    constructor. Fixed-``k`` fits reject unused ``groups``/``time`` unless
    ``within`` is set, while ``k="auto"`` accepts them for split construction,
    including the DataFrame shorthand ``groups="col"``/``time="col"``
    that moves the column out of the candidate features. Under sklearn >= 1.4
    metadata routing every datum must be requested explicitly with
    ``set_fit_request(...)``, and a fixed-``k`` estimator without ``within``
    refuses a ``groups``/``time`` request.

    Examples
    --------
    >>> import numpy as np
    >>> from sift import JMISelector
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 6))
    >>> y = X[:, 0] + 0.5 * X[:, 1] + 0.1 * rng.normal(size=200)
    >>> selector = JMISelector(k=2, task="regression", verbose=False)
    >>> selector.fit(X, y).selected_features_
    ['x0', 'x1']
    >>> selector.get_support(indices=True)
    array([0, 1])
    """

    _subsample_auto_is_cache_default = True
    _random_state_auto_is_cache_default = True

    def __init__(
        self,
        k: int | str = 10,
        *,
        task: str = "regression",
        estimator: str = "auto",
        relevance: str = "f",
        top_m: int | None = None,
        cat_features: list[str] | None = None,
        cat_encoding: str = "none",
        target_cv_n_splits: int = 5,
        target_cv_smoothing: Literal["auto"] | float = "auto",
        target_prior: float | None = None,
        warmup_policy: Literal["exclude", "zero_weight"] = "zero_weight",
        allow_full_data_target_encoding: bool = False,
        onehot_max_levels: int = 32,
        subsample: int | None | Literal["auto"] = "auto",
        random_state: int | Literal["auto"] = "auto",
        verbose: bool = True,
        cache=None,
        auto_k_config=None,
        within: str | None = None,
        include=None,
        exclude=None,
        candidates=None,
        feature_blocks=None,
        callback: ProgressCallback | None = None,
        output_order: str = "legacy",
    ):
        self._init_selector(select_jmi, locals())


class JMIMSelector(_BaseSelector):
    """Sklearn-style wrapper for ``sift.select_jmim``.

    JMI Maximization is the conservative sibling of ``JMISelector``: at
    each step it scores a candidate by the *minimum* joint information taken
    over the already-selected features instead of the sum, so one strongly
    redundant pairing is enough to hold a candidate back. Prefer it when a
    single redundant partner should veto a feature; prefer plain JMI when
    average complementarity is the better summary. ``fit`` learns the path and
    records the selection; ``transform`` returns the selected columns in the
    fitted container kind, ``set_output(transform="pandas")`` is honored like
    any sklearn transformer, and ``inverse_transform`` restores a dense
    full-width matrix with unselected columns zero-filled. Sparse input is
    rejected in ``fit``, ``transform`` and ``inverse_transform``.

    ``k`` is an upper bound rather than a promise: constant-column filtering,
    relevance screening and non-positive objective checks can end the path with
    fewer than ``k`` features.

    Parameters
    ----------
    k : int or {"auto"}, default=10
        Upper bound on the number of selected features. ``"auto"`` defers the
        size to ``auto_k_config``; with no config, automatic sizing needs
        ``groups`` or ``time`` at fit time (they select ``strategy="group_cv"``
        or ``"time_holdout"``) and otherwise raises.
    task : {"regression", "classification"}, default="regression"
        Target kind. It picks the relevance scorer and the estimator variants.
    estimator : {"auto", "binned", "r2", "ksg", "gaussian"}, default="auto"
        Mutual-information estimator for the joint terms. ``"auto"`` resolves
        to ``"binned"`` for classification and ``"r2"`` for regression;
        ``"gaussian"`` is the copula-cache fast path and the only route that
        accepts ``cache``; ``"ksg"`` is the nearest-neighbour estimator and
        rejects ``sample_weight``.
    relevance : {"f", "ks", "rf"}, default="f"
        Relevance score used for screening and the first step. Regression
        accepts ``"f"`` and ``"rf"``; classification also accepts ``"ks"``.
    top_m : int or None, default=None
        Keep only the ``top_m`` most relevant valid candidates before the
        greedy loop. ``None`` resolves to ``max(5 * k, 250)``, never below
        ``k``.
    cat_features : list of str or None, default=None
        Categorical column names to encode. ``None`` auto-detects ``object``,
        ``category`` and ``string`` DataFrame columns. Unused when
        ``cat_encoding="none"`` or when ``X`` is an ndarray.
    cat_encoding : {"none", "target_cv", "onehot", "target", "loo", "james_stein", "loo_logit"}, default="none"
        One of ``"none"``, ``"target_cv"``, ``"target"``, ``"loo"``,
        ``"james_stein"`` or ``"loo_logit"``, fitted inside ``fit``.
        ``"target_cv"`` is the built-in leakage-safe contract: out-of-fold
        training rows receive ``fold_encoding - fold_training_prior`` while
        inference rows receive ``full_fit_encoding - full_training_prior``, so
        an unseen category maps to a centered zero and cannot identify its own
        fold. ``"target"``, ``"loo"`` and ``"james_stein"`` require the
        optional ``category_encoders`` package; ``"loo_logit"`` is SIFT's own
        leave-one-out logit encoder and the only one that accepts
        ``sample_weight`` besides ``"target_cv"``. Any supervised encoding
        makes ``fit_transform`` return the y-aware encoded training block and
        makes ``inverse_transform`` unavailable.
    target_cv_n_splits : int, default=5
        Fold count for ``cat_encoding="target_cv"``.
    target_cv_smoothing : {"auto"} or float, default="auto"
        Empirical-Bayes shrinkage for ``"target_cv"``. ``"auto"`` reproduces
        sklearn's ``TargetEncoder`` rule on unweighted fixed-k folds and uses
        weighted row mass elsewhere; an explicit non-negative float always
        works.
    target_prior : float or None, default=None
        Target-independent prior for time-aware ``"target_cv"`` fits, so the
        earliest block emits a centered neutral zero and stays in the fit.
    warmup_policy : {"exclude", "zero_weight"}, default="zero_weight"
        How to treat the earliest no-history block of a time-aware
        ``"target_cv"`` fit when no ``target_prior`` is given.
    allow_full_data_target_encoding : bool, default=False
        Opt in to fitting the 0.8 supervised encoders on the full matrix. It is
        rejected together with ``cat_encoding="target_cv"``, whose cross-fitted
        contract it contradicts.
    onehot_max_levels : int, default=32
        Cap on retained dummy levels per categorical when
        ``cat_encoding="onehot"``. Surplus levels share ``other``.
    subsample : int, None or {"auto"}, default="auto"
        Row cap for classic row sampling and for uncached Gaussian cache
        construction. ``"auto"`` is the sklearn-clonable spelling of the
        omitted default: 50,000 rows when fitting from ``X``, and "not
        supplied" when ``cache`` is given. ``None`` keeps every positively
        weighted row. An explicit value beside a ``cache`` raises.
    random_state : int or {"auto"}, default="auto"
        Seed for that row sampling and for uncached cache construction.
        ``"auto"`` is the omitted default and resolves to seed 0 when fitting
        from ``X``. An explicit value beside a ``cache`` raises.
    verbose : bool, default=True
        Emit progress at INFO on the ``sift`` logger.
    cache : FeatureCache or None, default=None
        Prebuilt Gaussian-copula cache to reuse with ``estimator="gaussian"``.
        A named cache requires a DataFrame with identical columns in identical
        order; a positional cache requires the matching ndarray. A cache cannot
        be combined with a supervised ``cat_encoding``.
    auto_k_config : AutoKConfig or None, default=None
        Automatic-sizing configuration, read only when ``k="auto"``. Selector
        classes additionally accept ``auto_k_mode="nested"`` together with
        ``k_method="evaluate"``, which refits a train-only path per split.
    within : {"groups", "two_way"} or None, default=None
        Panel demeaning applied after encoding and before ranks. ``"groups"``
        subtracts per-entity weighted means; ``"two_way"`` alternates entity
        and time demeaning for five iterations. Regression only. Fixed-``k``
        fits then require ``groups`` (and ``time`` for ``"two_way"``).
        ``transform`` still returns selected raw columns.
    callback : ProgressCallback or None, default=None
        ``callback(step, total, info)`` called after each completed greedy
        step. Nested auto-k folds stay silent; only the final refit reports.
    include : sequence of names or positions, optional
        Conditioning set. Selector state is initialized from these features
        before step 1. They appear in the fitted selection in caller order
        but are not discoveries; ``k`` counts additional features.
    exclude : sequence of names or positions, optional
        Features removed from the discovery pool. Cannot overlap ``include``.
    candidates : sequence of names or positions, optional
        Hard allow-list for discovery. ``include`` may sit outside it.
        Overlap with ``exclude`` is rejected. An empty remaining pool raises.
    feature_blocks : mapping, {"auto"} or None, default None
        Atomic column groups. ``k`` counts additional blocks; selected
        blocks expand to raw member columns. ``"auto"`` uses the
        ``{block}__{level}`` one-hot prefix; ordinary underscores are not
        split. ``k="auto"`` counts additional blocks on evaluate, elbow,
        penalized_objective, gaussian_cv, xfit_objective, and auto routing;
        calibrated column-step rules raise.
    output_order : {"legacy", "original"}, default="legacy"
        Order used by ``transform``, ``get_support(indices=True)``,
        ``get_feature_names_out`` and ``inverse_transform``. ``"legacy"`` keeps
        selection-path order; ``"original"`` emits ascending fitted column
        position. The boolean support mask is always positional.

    Attributes
    ----------
    selected_features_ : list
        Selected feature labels in selection-path order.
    selected_indices_ : ndarray of shape (n_selected,)
        Their positions in the fitted feature matrix, in path order.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        One-dimensional object array of fitted feature names. A positional
        ndarray fit stores generated ``x0...`` names here.
    n_features_in_ : int
        Number of candidate features seen during ``fit``.
    k_ : int
        Feature count chosen by nested automatic k. Set only on the
        ``auto_k_mode="nested"`` path, not by prefix-only auto-k.
    nested_auto_k_diagnostics_ : dict
        Fold scores, metric and selection rule behind ``k_``; same path only.
    categorical_features_ : list
        Categorical columns the fitted encoder covered.
    categorical_encoder_ : object or None
        The fitted encoder, reused target-blind by ``transform``.
    categorical_encoding_metadata_ : dict
        The encoder's own ``{"kind": ..., "n_splits": ...}``, present only when
        ``cat_encoding="target_cv"`` encoded at least one column.

    Raises
    ------
    ValueError
        If ``groups``/``time`` reach a fixed-``k`` fit, if ``k="auto"`` has
        neither ``auto_k_config`` nor row context, if ``subsample`` or
        ``random_state`` is explicit beside a ``cache``, if a supervised
        ``cat_encoding`` is combined with a ``cache``, if
        ``estimator="ksg"`` is combined with ``sample_weight``, or if ``X`` is
        sparse or not two-dimensional.
    NotImplementedError
        From ``inverse_transform`` after a supervised categorical encoding,
        because the fitted encoder is not invertible.

    Warns
    -----
    UserWarning
        When an explicit ``AutoKConfig(k_method="auto")`` router supports zero
        features, or saturates its effective ``max_k`` (a censored result).

    See Also
    --------
    sift.select_jmim : Function-style JMIM with the same options.
    JMISelector : Summed-score variant of the same joint objective.
    MRMRSelector : Relevance-versus-redundancy greedy path.

    Notes
    -----
    The shared selector-class fit contract is
    ``fit(X, y, sample_weight=None, groups=None, time=None)``; ``cache`` and
    ``auto_k_config`` may also be passed per call and then win over the
    constructor. Fixed-``k`` fits reject unused ``groups``/``time`` unless
    ``within`` is set, while ``k="auto"`` accepts them for split construction,
    including the DataFrame shorthand ``groups="col"``/``time="col"``
    that moves the column out of the candidate features. Under sklearn >= 1.4
    metadata routing every datum must be requested explicitly with
    ``set_fit_request(...)``, and a fixed-``k`` estimator without ``within``
    refuses a ``groups``/``time`` request.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sift import JMIMSelector
    >>> rng = np.random.default_rng(0)
    >>> X = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
    >>> y = X["a"] + 0.5 * X["b"] + 0.1 * rng.normal(size=200)
    >>> selector = JMIMSelector(k=2, task="regression", verbose=False)
    >>> list(selector.fit(X, y).get_feature_names_out())
    ['a', 'b']
    >>> selector.transform(X).shape
    (200, 2)
    """

    _subsample_auto_is_cache_default = True
    _random_state_auto_is_cache_default = True

    def __init__(
        self,
        k: int | str = 10,
        *,
        task: str = "regression",
        estimator: str = "auto",
        relevance: str = "f",
        top_m: int | None = None,
        cat_features: list[str] | None = None,
        cat_encoding: str = "none",
        target_cv_n_splits: int = 5,
        target_cv_smoothing: Literal["auto"] | float = "auto",
        target_prior: float | None = None,
        warmup_policy: Literal["exclude", "zero_weight"] = "zero_weight",
        allow_full_data_target_encoding: bool = False,
        onehot_max_levels: int = 32,
        subsample: int | None | Literal["auto"] = "auto",
        random_state: int | Literal["auto"] = "auto",
        verbose: bool = True,
        cache=None,
        auto_k_config=None,
        within: str | None = None,
        include=None,
        exclude=None,
        candidates=None,
        feature_blocks=None,
        callback: ProgressCallback | None = None,
        output_order: str = "legacy",
    ):
        self._init_selector(select_jmim, locals())


class CEFSPlusSelector(_BaseSelector):
    """Sklearn-style wrapper for ``sift.select_cefsplus``.

    CEFS+ is a regression-only Gaussian-copula filter: it ranks a candidate by
    the log-determinant conditional-information gain it adds to the features
    already chosen, which makes it fast on wide numeric matrices and willing to
    keep suppressor variables that a pairwise-redundancy filter would drop.
    Use it as the default wide-data filter, and reach for ``k="auto"`` when the
    feature count itself should be measured rather than guessed. ``fit`` learns
    the path and records the selection; ``transform`` returns the selected
    columns in the fitted container kind, ``set_output(transform="pandas")`` is
    honored like any sklearn transformer, and ``inverse_transform`` restores a
    dense full-width matrix with unselected columns zero-filled. Sparse input
    is rejected in ``fit``, ``transform`` and ``inverse_transform``.

    ``k`` is an upper bound rather than a promise: constant-column filtering,
    relevance screening, correlation pruning and non-positive objective checks
    can end the path with fewer than ``k`` features.

    Parameters
    ----------
    k : int or {"auto"}, default=75
        Upper bound on the number of selected features. Unlike the mutual
        information selectors, ``k="auto"`` without an ``auto_k_config`` is
        supported with no row context: it routes through the measured Auto-K
        router, ``AutoKConfig(k_method="auto")``.
    top_m : int or None, default=None
        Keep only the ``top_m`` most relevant valid candidates before the
        greedy loop. ``None`` resolves to ``max(5 * k, 250)``, never below
        ``k``.
    corr_prune : float or None, default=None
        Absolute copula-correlation threshold for dropping duplicate-looking
        candidates. ``None`` prunes nothing and therefore preserves possible
        suppressor pairs; a float in ``(0, 1]`` such as ``0.95`` opts into
        duplicate-oriented pruning.
    cat_features : list of str or None, default=None
        Categorical column names to encode. ``None`` auto-detects ``object``,
        ``category`` and ``string`` DataFrame columns. Unused when
        ``cat_encoding="none"`` or when ``X`` is an ndarray.
    cat_encoding : {"none", "target_cv", "onehot", "target", "loo", "james_stein", "loo_logit"}, default="none"
        One of ``"none"``, ``"target_cv"``, ``"target"``, ``"loo"``,
        ``"james_stein"`` or ``"loo_logit"``, fitted inside ``fit``.
        ``"target_cv"`` is the built-in leakage-safe contract: out-of-fold
        training rows receive ``fold_encoding - fold_training_prior`` while
        inference rows receive ``full_fit_encoding - full_training_prior``, so
        an unseen category maps to a centered zero and cannot identify its own
        fold. ``"target"``, ``"loo"`` and ``"james_stein"`` require the
        optional ``category_encoders`` package; ``"loo_logit"`` is SIFT's own
        leave-one-out logit encoder and the only one that accepts
        ``sample_weight`` besides ``"target_cv"``. Any supervised encoding
        makes ``fit_transform`` return the y-aware encoded training block and
        makes ``inverse_transform`` unavailable.
    target_cv_n_splits : int, default=5
        Fold count for ``cat_encoding="target_cv"``.
    target_cv_smoothing : {"auto"} or float, default="auto"
        Empirical-Bayes shrinkage for ``"target_cv"``. ``"auto"`` reproduces
        sklearn's ``TargetEncoder`` rule on unweighted fixed-k folds and uses
        weighted row mass elsewhere; an explicit non-negative float always
        works.
    target_prior : float or None, default=None
        Target-independent prior for time-aware ``"target_cv"`` fits, so the
        earliest block emits a centered neutral zero and stays in the fit.
    warmup_policy : {"exclude", "zero_weight"}, default="zero_weight"
        How to treat the earliest no-history block of a time-aware
        ``"target_cv"`` fit when no ``target_prior`` is given.
    allow_full_data_target_encoding : bool, default=False
        Opt in to fitting the 0.8 supervised encoders on the full matrix. It is
        rejected together with ``cat_encoding="target_cv"``, whose cross-fitted
        contract it contradicts.
    onehot_max_levels : int, default=32
        Cap on retained dummy levels per categorical when
        ``cat_encoding="onehot"``. Surplus levels share ``other``.
    subsample : int, None or {"auto"}, default="auto"
        Row cap for uncached Gaussian cache construction. ``"auto"`` is the
        sklearn-clonable spelling of the omitted default: 50,000 rows when
        fitting from ``X``, and "not supplied" when ``cache`` is given.
        ``None`` keeps every positively weighted row. An explicit value beside
        a ``cache`` raises.
    random_state : int or {"auto"}, default="auto"
        Seed for that row sampling and cache construction. ``"auto"`` is the
        omitted default and resolves to seed 0 when fitting from ``X``. An
        explicit value beside a ``cache`` raises.
    verbose : bool, default=True
        Emit progress at INFO on the ``sift`` logger.
    cache : FeatureCache or None, default=None
        Prebuilt Gaussian-copula cache to reuse. A named cache requires a
        DataFrame with identical columns in identical order; a positional cache
        requires the matching ndarray. A cache carries its own row weights, so
        it cannot be combined with ``sample_weight`` or with a supervised
        ``cat_encoding``.
    auto_k_config : AutoKConfig or None, default=None
        Automatic-sizing configuration, read only when ``k="auto"``. Selector
        classes additionally accept ``auto_k_mode="nested"`` together with
        ``k_method="evaluate"``, which refits a train-only path per split.
    within : {"groups", "two_way"} or None, default=None
        Panel demeaning applied after encoding and before ranks. ``"groups"``
        subtracts per-entity weighted means; ``"two_way"`` alternates entity
        and time demeaning for five iterations. Regression only. Fixed-``k``
        fits then require ``groups`` (and ``time`` for ``"two_way"``).
        ``transform`` still returns selected raw columns.
    callback : ProgressCallback or None, default=None
        ``callback(step, total, info)`` called after each completed greedy
        step. Nested auto-k folds stay silent; only the final refit reports.
    include : sequence of names or positions, optional
        Conditioning set. Selector state is initialized from these features
        before step 1. They appear in the fitted selection in caller order
        but are not discoveries; ``k`` counts additional features.
    exclude : sequence of names or positions, optional
        Features removed from the discovery pool. Cannot overlap ``include``.
    candidates : sequence of names or positions, optional
        Hard allow-list for discovery. ``include`` may sit outside it.
        Overlap with ``exclude`` is rejected. An empty remaining pool raises.
    feature_blocks : mapping, {"auto"} or None, default None
        Atomic column groups. ``k`` counts additional blocks; selected
        blocks expand to raw member columns. ``"auto"`` uses the
        ``{block}__{level}`` one-hot prefix; ordinary underscores are not
        split. ``k="auto"`` counts additional blocks on evaluate, elbow,
        penalized_objective, gaussian_cv, xfit_objective, and auto routing;
        calibrated column-step rules raise.
    output_order : {"legacy", "original"}, default="legacy"
        Order used by ``transform``, ``get_support(indices=True)``,
        ``get_feature_names_out`` and ``inverse_transform``. ``"legacy"`` keeps
        selection-path order; ``"original"`` emits ascending fitted column
        position. The boolean support mask is always positional.

    Attributes
    ----------
    selected_features_ : list
        Selected feature labels in selection-path order.
    selected_indices_ : ndarray of shape (n_selected,)
        Their positions in the fitted feature matrix, in path order.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        One-dimensional object array of fitted feature names. A positional
        ndarray fit stores generated ``x0...`` names here.
    n_features_in_ : int
        Number of candidate features seen during ``fit``.
    k_ : int
        Feature count chosen by nested automatic k. Set only on the
        ``auto_k_mode="nested"`` path, not by prefix-only or routed auto-k.
    nested_auto_k_diagnostics_ : dict
        Fold scores, metric and selection rule behind ``k_``; same path only.
    categorical_features_ : list
        Categorical columns the fitted encoder covered.
    categorical_encoder_ : object or None
        The fitted encoder, reused target-blind by ``transform``.
    categorical_encoding_metadata_ : dict
        The encoder's own ``{"kind": ..., "n_splits": ...}``, present only when
        ``cat_encoding="target_cv"`` encoded at least one column.

    Raises
    ------
    ValueError
        If ``groups``/``time`` reach a fixed-``k`` fit, if ``subsample`` or
        ``random_state`` is explicit beside a ``cache``, if ``sample_weight``
        is passed beside a ``cache``, if a supervised ``cat_encoding`` is
        combined with a ``cache``, or if ``X`` is sparse or not
        two-dimensional. Contextual ``cat_encoding="target_cv"`` with
        ``groups``/``time`` additionally requires an explicit
        ``AutoKConfig(auto_k_mode="nested", k_method="evaluate")``.
    NotImplementedError
        From ``inverse_transform`` after a supervised categorical encoding,
        because the fitted encoder is not invertible.

    Warns
    -----
    UserWarning
        When the routed ``k="auto"`` criterion supports zero features, or when
        it saturates its effective ``max_k`` and the result is censored.

    See Also
    --------
    sift.select_cefsplus : Function-style CEFS+ with the same options.
    CEFSPlusBinarySelector : Bernoulli-deviance CEFS+ for binary targets.
    MRMRSelector : Relevance-versus-redundancy greedy path.

    Notes
    -----
    The shared selector-class fit contract is
    ``fit(X, y, sample_weight=None, groups=None, time=None)``; ``cache`` and
    ``auto_k_config`` may also be passed per call and then win over the
    constructor. Fixed-``k`` fits reject unused ``groups``/``time`` unless
    ``within`` is set, while ``k="auto"`` accepts them for split construction,
    including the DataFrame shorthand ``groups="col"``/``time="col"``
    that moves the column out of the candidate features. Under sklearn >= 1.4
    metadata routing every datum must be requested explicitly with
    ``set_fit_request(...)``, and a fixed-``k`` estimator without ``within``
    refuses a ``groups``/``time`` request.

    Examples
    --------
    >>> import numpy as np
    >>> from sift import CEFSPlusSelector
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 6))
    >>> y = X[:, 0] + 0.5 * X[:, 1] + 0.1 * rng.normal(size=200)
    >>> CEFSPlusSelector(k=2, verbose=False).fit(X, y).selected_features_
    ['x0', 'x1']
    >>> CEFSPlusSelector(k="auto", verbose=False).fit(X, y).selected_features_
    ['x0', 'x1']
    """

    _subsample_auto_is_cache_default = True
    _random_state_auto_is_cache_default = True

    def __init__(
        self,
        k: int | str = 75,
        *,
        top_m: int | None = None,
        corr_prune: float | None = None,
        cat_features: list[str] | None = None,
        cat_encoding: str = "none",
        target_cv_n_splits: int = 5,
        target_cv_smoothing: Literal["auto"] | float = "auto",
        target_prior: float | None = None,
        warmup_policy: Literal["exclude", "zero_weight"] = "zero_weight",
        allow_full_data_target_encoding: bool = False,
        onehot_max_levels: int = 32,
        subsample: int | None | Literal["auto"] = "auto",
        random_state: int | Literal["auto"] = "auto",
        verbose: bool = True,
        cache=None,
        auto_k_config=None,
        within: str | None = None,
        include=None,
        exclude=None,
        candidates=None,
        feature_blocks=None,
        callback: ProgressCallback | None = None,
        output_order: str = "legacy",
    ):
        self._init_selector(select_cefsplus, locals())

    def _routes_no_config_auto_k(self) -> bool:
        return True


class CEFSPlusBinarySelector(_BaseSelector):
    """Sklearn-style wrapper for ``sift.select_cefsplus_binary``.

    The binary CEFS+ path scores candidates by the conditional Bernoulli
    deviance they remove, refitting a ridge-penalized logistic model along a
    greedy prefix, so it sizes features against the loss a downstream binary
    classifier actually optimizes. Use it for two-class targets, especially
    imbalanced ones, where ``class_weight`` should shape the objective. ``fit``
    validates a two-class target and records the selection; ``transform``
    returns the selected columns in the fitted container kind,
    ``set_output(transform="pandas")`` is honored like any sklearn
    transformer, and ``inverse_transform`` restores a dense full-width matrix
    with unselected columns zero-filled. Sparse input is rejected in ``fit``,
    ``transform`` and ``inverse_transform``. Unlike the other filter selector
    classes this one has no ``cache`` option and rejects prebuilt caches.

    ``k`` is an upper bound rather than a promise: constant-column filtering,
    relevance screening, correlation pruning and non-positive gain checks can
    end the path with fewer than ``k`` features.

    Parameters
    ----------
    k : int or {"auto"}, default=75
        Upper bound on the number of selected features. ``k="auto"`` without an
        ``auto_k_config`` needs no row context: it routes through the measured
        Auto-K router, ``AutoKConfig(k_method="auto")``.
    loss : {"logloss", "brier"}, default="logloss"
        Objective driving the greedy path. ``"logloss"`` walks the logistic
        deviance path described above; ``"brier"`` delegates to the Gaussian
        CEFS+ proxy and therefore follows that route's contracts.
    top_m : int or None, default=None
        Keep only the ``top_m`` most relevant valid candidates before the
        greedy loop. ``None`` resolves to ``max(5 * k, 250)``, never below
        ``k``.
    corr_prune : float or None, default=None
        Absolute copula-correlation threshold for dropping duplicate-looking
        candidates. ``None`` prunes nothing and therefore preserves possible
        suppressor pairs; a float in ``(0, 1]`` opts into duplicate-oriented
        pruning.
    class_weight : None, {"balanced"} or dict, default=None
        Per-class multipliers applied on top of ``sample_weight``.
        ``"balanced"`` equalizes the two classes' total weight; a dict must
        provide a finite non-negative value for both raw class labels.
    ridge : float, default=1e-4
        Positive L2 penalty on the non-intercept coefficients of the logistic
        prefix fits.
    refit_every : int, default=1
        Positive integer stride between full logistic refits along the greedy
        path; larger values trade objective accuracy for speed. With opt-in
        ``feature_blocks``, the stride is in additional discovery blocks; the
        no-block and identity cadence is unchanged.
    cat_features : list of str or None, default=None
        Categorical column names to encode. ``None`` auto-detects ``object``,
        ``category`` and ``string`` DataFrame columns. Unused when
        ``cat_encoding="none"`` or when ``X`` is an ndarray.
    cat_encoding : {"none", "target_cv", "onehot", "target", "loo", "james_stein", "loo_logit"}, default="none"
        One of ``"none"``, ``"target_cv"``, ``"target"``, ``"loo"``,
        ``"james_stein"`` or ``"loo_logit"``, fitted inside ``fit`` against the
        validated 0/1 target. ``"target_cv"`` is the built-in leakage-safe
        contract: out-of-fold training rows receive
        ``fold_encoding - fold_training_prior`` while inference rows receive
        ``full_fit_encoding - full_training_prior``, so an unseen category maps
        to a centered zero and cannot identify its own fold. ``"target"``,
        ``"loo"`` and ``"james_stein"`` require the optional
        ``category_encoders`` package; ``"loo_logit"`` is SIFT's own
        leave-one-out logit encoder and the only one that accepts
        ``sample_weight`` besides ``"target_cv"``. Any supervised encoding
        makes ``fit_transform`` return the y-aware encoded training block and
        makes ``inverse_transform`` unavailable.
    target_cv_n_splits : int, default=5
        Fold count for ``cat_encoding="target_cv"``.
    target_cv_smoothing : {"auto"} or float, default="auto"
        Empirical-Bayes shrinkage for ``"target_cv"``. ``"auto"`` reproduces
        sklearn's ``TargetEncoder`` rule on unweighted fixed-k folds and uses
        weighted row mass elsewhere; an explicit non-negative float always
        works.
    target_prior : float or None, default=None
        Target-independent prior for time-aware ``"target_cv"`` fits, so the
        earliest block emits a centered neutral zero and stays in the fit.
    warmup_policy : {"exclude", "zero_weight"}, default="zero_weight"
        How to treat the earliest no-history block of a time-aware
        ``"target_cv"`` fit when no ``target_prior`` is given.
    loo_smoothing : float, default=20.0
        Positive smoothing constant of the ``cat_encoding="loo_logit"``
        encoder.
    loo_clip_min : float, default=1e-4
        Lower probability clip of that encoder; ``0 < loo_clip_min <
        loo_clip_max < 1`` is enforced.
    loo_clip_max : float, default=1.0 - 1e-4
        Upper probability clip of that encoder.
    allow_full_data_target_encoding : bool, default=False
        Opt in to fitting the 0.8 supervised encoders on the full matrix. It is
        rejected together with ``cat_encoding="target_cv"``, whose cross-fitted
        contract it contradicts.
    onehot_max_levels : int, default=32
        Cap on retained dummy levels per categorical when
        ``cat_encoding="onehot"``. Surplus levels share ``other``.
    subsample : int or None, default=None
        Row cap applied before the path is built. ``None`` keeps every
        positively weighted row. This selector takes no cache, so the numeric
        default needs no ``"auto"`` sentinel.
    random_state : int, default=0
        Seed for that row sampling.
    verbose : bool, default=True
        Emit progress at INFO on the ``sift`` logger.
    auto_k_config : AutoKConfig or None, default=None
        Automatic-sizing configuration, read only when ``k="auto"``. Selector
        classes additionally accept ``auto_k_mode="nested"`` together with
        ``k_method="evaluate"``, which refits a train-only path per split.
    callback : ProgressCallback or None, default=None
        ``callback(step, total, info)`` called after each completed greedy
        step. Nested auto-k folds stay silent; only the final refit reports.
    include : sequence of names or positions, optional
        Conditioning set. Selector state is initialized from these features
        before step 1. They appear in the fitted selection in caller order
        but are not discoveries; ``k`` counts additional features.
    exclude : sequence of names or positions, optional
        Features removed from the discovery pool. Cannot overlap ``include``.
    candidates : sequence of names or positions, optional
        Hard allow-list for discovery. ``include`` may sit outside it.
        Overlap with ``exclude`` is rejected. An empty remaining pool raises.
    feature_blocks : mapping, {"auto"} or None, default None
        Atomic column groups. ``k`` counts additional blocks; selected
        blocks expand to raw member columns. ``"auto"`` uses the
        ``{block}__{level}`` one-hot prefix; ordinary underscores are not
        split. ``k="auto"`` counts additional blocks on evaluate, elbow,
        penalized_objective, and auto (EBIC). Joint logistic block scores
        are used; Gaussian CV/xfit and calibrated column-step rules raise.
        ``loss="brier"`` delegates to Gaussian CEFS+ blocks.
    output_order : {"legacy", "original"}, default="legacy"
        Order used by ``transform``, ``get_support(indices=True)``,
        ``get_feature_names_out`` and ``inverse_transform``. ``"legacy"`` keeps
        selection-path order; ``"original"`` emits ascending fitted column
        position. The boolean support mask is always positional.

    Attributes
    ----------
    selected_features_ : list
        Selected feature labels in selection-path order.
    selected_indices_ : ndarray of shape (n_selected,)
        Their positions in the fitted feature matrix, in path order.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        One-dimensional object array of fitted feature names. A positional
        ndarray fit stores generated ``x0...`` names here.
    n_features_in_ : int
        Number of candidate features seen during ``fit``.
    k_ : int
        Feature count chosen by nested automatic k. Set only on the
        ``auto_k_mode="nested"`` path, not by prefix-only or routed auto-k.
    nested_auto_k_diagnostics_ : dict
        Fold scores, metric and selection rule behind ``k_``; same path only.
    categorical_features_ : list
        Categorical columns the fitted encoder covered.
    categorical_encoder_ : object or None
        The fitted encoder, reused target-blind by ``transform``.
    categorical_encoding_metadata_ : dict
        The encoder's own ``{"kind": ..., "n_splits": ...}``, present only when
        ``cat_encoding="target_cv"`` encoded at least one column.

    Raises
    ------
    ValueError
        If ``y`` is not two-class, if ``groups``/``time`` reach a fixed-``k``
        fit, if a prebuilt cache is supplied, if ``loss="brier"`` is combined
        with ``cat_encoding="loo_logit"`` on DataFrame categoricals (no
        function-API parity), or if ``X`` is sparse or not two-dimensional.
        Contextual ``cat_encoding="target_cv"`` with ``groups``/``time``
        additionally requires an explicit
        ``AutoKConfig(auto_k_mode="nested", k_method="evaluate")``.
    NotImplementedError
        From ``inverse_transform`` after a supervised categorical encoding,
        because the fitted encoder is not invertible.

    Warns
    -----
    UserWarning
        When the routed ``k="auto"`` criterion supports zero features, or when
        it saturates its effective ``max_k`` and the result is censored.

    See Also
    --------
    sift.select_cefsplus_binary : Function-style binary CEFS+.
    CEFSPlusSelector : The regression Gaussian CEFS+ path this delegates to
        under ``loss="brier"``.
    MRMRSelector : Relevance-versus-redundancy greedy path.

    Notes
    -----
    The shared selector-class fit contract is
    ``fit(X, y, sample_weight=None, groups=None, time=None)``;
    ``auto_k_config`` may also be passed per call and then wins over the
    constructor. Fixed-``k`` fits reject ``groups``/``time`` by design, since
    those only define automatic-k evaluation splits, while ``k="auto"`` accepts
    them, including the DataFrame shorthand ``groups="col"``/``time="col"``
    that moves the column out of the candidate features. Under sklearn >= 1.4
    metadata routing every datum must be requested explicitly with
    ``set_fit_request(...)``, and a fixed-``k`` estimator refuses a
    ``groups``/``time`` request. The estimator declares itself binary-only to
    sklearn's tag APIs so the common estimator checks feed it two classes.

    Examples
    --------
    >>> import numpy as np
    >>> from sift import CEFSPlusBinarySelector
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 6))
    >>> y = (X[:, 0] + 0.5 * X[:, 1] + 0.1 * rng.normal(size=200) > 0).astype(int)
    >>> selector = CEFSPlusBinarySelector(k=2, verbose=False)
    >>> selector.fit(X, y).selected_features_
    ['x0', 'x1']
    >>> selector.get_support()
    array([ True,  True, False, False, False, False])

    """

    def __init__(
        self,
        k: int | str = 75,
        *,
        loss: str = "logloss",
        top_m: int | None = None,
        corr_prune: float | None = None,
        class_weight=None,
        ridge: float = 1e-4,
        refit_every: int = 1,
        cat_features: list[str] | None = None,
        cat_encoding: str = "none",
        target_cv_n_splits: int = 5,
        target_cv_smoothing: Literal["auto"] | float = "auto",
        target_prior: float | None = None,
        warmup_policy: Literal["exclude", "zero_weight"] = "zero_weight",
        loo_smoothing: float = 20.0,
        loo_clip_min: float = 1e-4,
        loo_clip_max: float = 1.0 - 1e-4,
        allow_full_data_target_encoding: bool = False,
        onehot_max_levels: int = 32,
        subsample: int | None = None,
        random_state: int = 0,
        verbose: bool = True,
        auto_k_config=None,
        include=None,
        exclude=None,
        candidates=None,
        feature_blocks=None,
        callback: ProgressCallback | None = None,
        output_order: str = "legacy",
    ):
        self._init_selector(select_cefsplus_binary, locals())

    def _more_tags(self):
        # sklearn <1.6 returns a shared module-level default dict from
        # BaseEstimator._more_tags, so selector_tags copies it before setting
        # binary_only=True. The tag makes sklearn's common checks coerce y to
        # two classes instead of tripping this selector's own validation.
        return selector_tags(super()._more_tags(), binary_only=True)

    def __sklearn_tags__(self):
        """Expose the two-class requirement through sklearn's tag APIs.

        sklearn >=1.6 dropped the flat ``binary_only`` key; its replacement,
        ``Tags.classifier_tags.multi_class``, only exists for estimators typed
        as classifiers. This selector is a transformer, so the nearest valid
        representation is to leave ``classifier_tags`` unset (``None``) and let
        the fit-time two-class validation error stand, rather than misdeclaring
        the estimator type just to obtain a tag.
        """
        parent_tags = getattr(super(), "__sklearn_tags__", None)
        if parent_tags is None:  # sklearn <1.6 uses the dict API above.
            return self._more_tags()
        return selector_tags(parent_tags(), binary_only=True)

    def _routes_no_config_auto_k(self) -> bool:
        return True

    def _task(self) -> str:
        return "classification"

    def _categorical_target(self, y):
        y01, _, _ = validate_binary_target(y)
        return y01

    def _categorical_sample_weight(self, y, sample_weight):
        y01, raw_y, _ = validate_binary_target(y)
        weights, _ = resolve_binary_weights(
            y01,
            raw_y,
            sample_weight=sample_weight,
            class_weight=self.class_weight,
        )
        return weights

    def _nested_eval_sample_weight(self, y, sample_weight):
        y_arr = np.asarray(y).reshape(-1)
        return ensure_weights(sample_weight, len(y_arr), normalize=True)

    def _fit_selector(
        self,
        X,
        y,
        *,
        k,
        sample_weight=None,
        groups=None,
        time=None,
        cache=None,
        auto_k_config=None,
        fit_params=None,
        capture_training_output: bool = False,
        encoding_groups=None,
        encoding_time=None,
    ):
        if cache is not None:
            raise ValueError("CEFSPlusBinarySelector does not support prebuilt caches.")

        call_params = dict(self._selector_params())
        call_params["sample_weight"] = sample_weight
        if groups is not None:
            call_params["groups"] = groups
        if time is not None:
            call_params["time"] = time
        if auto_k_config is not None:
            call_params["auto_k_config"] = auto_k_config
        if fit_params:
            blocked = sorted(
                _BLOCKED_FIT_PARAM_OVERRIDES.union(
                    _BINARY_PREPROCESSING_FIT_PARAM_OVERRIDES
                ).intersection(fit_params)
            )
            if blocked:
                blocked_text = ", ".join(blocked)
                raise ValueError(
                    "CEFSPlusBinarySelector return-shape or preprocessing-affecting "
                    f"parameters must be set on the estimator before fit, not as fit-time "
                    f"overrides: {blocked_text}"
                )
            call_params.update(fit_params)
        self._resolve_auto_selector_params(call_params)

        loss_eff = str(self.loss).lower()
        if (
            loss_eff == "brier"
            and self.cat_encoding == "loo_logit"
            and isinstance(X, pd.DataFrame)
            and _categorical_columns(X, self.cat_features)
        ):
            raise ValueError(
                "CEFSPlusBinarySelector(loss='brier', cat_encoding='loo_logit') "
                "has no selector-class parity with the function API. Use "
                "cat_encoding='loo' for Brier proxy mode or loss='logloss' "
                "for logistic loo_logit encoding."
            )

        feature_names, names_generated = _feature_names_with_provenance(X)
        X_fit = self._fit_transform_categoricals(
            X,
            y,
            sample_weight=sample_weight,
            groups=encoding_groups,
            time=encoding_time,
        )
        effective_sample_weight = sample_weight
        if (
            self.cat_encoding == "target_cv"
            and getattr(self, "categorical_encoder_", None) is not None
            and getattr(self.categorical_encoder_, "effective_sample_weight_", None)
            is not None
        ):
            encoder_weights = self.categorical_encoder_.effective_sample_weight_
            if self.class_weight is None:
                effective_sample_weight = encoder_weights
            else:
                # Binary filter functions apply class_weight themselves. Preserve
                # that single application while carrying contextual warmup
                # exclusions from the encoder into the selector weights.
                warmup_mask = getattr(
                    self.categorical_encoder_, "warmup_mask_", None
                )
                if warmup_mask is not None and not np.all(warmup_mask):
                    base_weights = (
                        np.ones(len(y), dtype=float)
                        if sample_weight is None
                        else ensure_weights(sample_weight, len(y), normalize=False)
                    )
                    effective_sample_weight = base_weights * np.asarray(
                        warmup_mask, dtype=float
                    )
        call_params["sample_weight"] = effective_sample_weight
        if getattr(self, "_categorical_encoding_applied_", False):
            call_params["cat_features"] = None
            call_params["cat_encoding"] = "none"
            call_params["allow_full_data_target_encoding"] = False
        onehot_encoder, composed, encoded_names = _apply_onehot_call_params(
            self, call_params, feature_names
        )

        result = self._selector_fn(
            X_fit,
            y,
            k=k,
            return_result=True,
            **call_params,
        )
        selected_features = list(result.selected_features)
        selected_indices = result.selected_indices
        if selected_indices is None:
            selected_indices = _coerce_selection_indices(
                feature_names,
                selected_features,
            ).tolist()
        self._encoded_selected_names_ = None
        self._encoded_prefix_widths_ = None
        if isinstance(onehot_encoder, OneHotBlockEncoder):
            encoded_selected = list(selected_features)
            self._encoded_selected_names_ = encoded_selected
            selected_features = onehot_encoder.collapse_to_raw(encoded_selected)
            selected_indices = _coerce_selection_indices(
                feature_names,
                selected_features,
            ).tolist()
            from sift.selection.blocks import discovery_prefix_widths

            dummy_index = {name: i for i, name in enumerate(encoded_names)}
            dummy_path = [dummy_index[name] for name in encoded_selected if name in dummy_index]
            include_dummy = set(call_params.get("include") or ())
            include_idx = {dummy_index[name] for name in include_dummy if name in dummy_index}
            discoveries = [i for i in dummy_path if i not in include_idx]
            n_include = sum(1 for i in dummy_path if i in include_idx)
            widths = discovery_prefix_widths(discoveries, composed)
            if widths:
                self._encoded_prefix_widths_ = tuple(int(n_include + w) for w in widths)
        elif (
            getattr(self, "cat_encoding", None) == "onehot"
            and isinstance(X, pd.DataFrame)
        ):
            cols = _categorical_columns(X, getattr(self, "cat_features", None))
            if cols:
                enc = OneHotBlockEncoder(
                    cols,
                    max_levels=validate_onehot_max_levels(
                        getattr(self, "onehot_max_levels", 32)
                    ),
                )
                enc.fit(X, sample_weight=sample_weight)
                self.categorical_encoder_ = enc
                self._categorical_encoding_applied_ = True
                self._encoded_selected_names_ = enc.expand_selected(selected_features)

        self.feature_names_in_ = feature_names_array(feature_names)
        self._fit_feature_names_generated_ = names_generated
        self.n_features_in_ = len(feature_names)
        self.selected_features_ = selected_features
        self.selected_indices_ = np.asarray(selected_indices, dtype=np.int64)
        if capture_training_output:
            if self._encoded_selected_names_ is not None and isinstance(X_fit, pd.DataFrame):
                self._fit_transform_output_ = X_fit.loc[:, self._encoded_output_names()].copy()
            else:
                self._fit_transform_output_ = _selected_training_output(
                    X_fit,
                    self._output_indices(),
                )
        return self


_KNOCKOFF_LEGACY_CAT_ENCODINGS = frozenset(
    {"loo", "target", "james_stein", "loo_logit"}
)
_KNOCKOFF_SUPERVISED_ENCODING_NOTE = (
    "supervised categorical encoding derives features from y, so the Model-X "
    "exchangeability assumption behind the knockoff filter no longer holds and "
    "no FDR claim applies to this result"
)


class KnockoffSelector(_BaseSelector):
    """Sklearn-style wrapper for ``sift.select_fdr``.

    This selector is sized by a target false-discovery rate ``q`` rather than
    by a feature count: it builds or reuses a Gaussian-copula
    ``sift.FeatureCache``, samples second-order knockoffs, computes
    antisymmetric ``W`` statistics and applies the knockoff+ threshold. Use it
    when the question is "which features are trustworthy discoveries" rather
    than "give me the best ``k``". It has no ``k`` and no ``auto_k_config``.
    ``transform`` returns the selected columns in the fitted container kind,
    ``set_output(transform="pandas")`` is honored like any sklearn
    transformer, and ``inverse_transform`` restores a dense full-width matrix
    with unselected columns zero-filled.

    ``subsample="auto"`` resolves to 50,000 rows when fitting from X and acts
    as an omitted construction option with a prebuilt cache. Explicit
    subsample values are not valid with a cache. The stochastic knockoff
    construction is sensitive to input row order, so this estimator is
    explicitly marked non-deterministic for sklearn estimator checks.

    ``cat_encoding="target_cv"`` is rejected: cross-fitted target encoding is
    still target-derived preprocessing and would silently invalidate the
    Model-X claim. The 0.8 supervised encodings remain available for
    compatibility, but only with an explicit ``UserWarning`` and result
    metadata that downgrades ``fdr_control`` to ``"none"``.

    Parameters
    ----------
    q : float, default=0.1
        Target FDR level, a finite float in ``(0, 1)``. With the default
        frequency vote, ``n_draws > 1`` makes this a per-draw level, not a
        guarantee for the aggregated set. With ``aggregation="evalues"`` it
        is the e-BH level on averaged knockoff e-values.
    statistic : str, default="relevance"
        Feature-importance statistic behind ``W``. Enabled values are
        ``"relevance"`` (fast default), ``"lsm"`` (lasso signed max),
        ``"ridge"`` (analytic ridge coefficient difference) and ``"cefsplus"``
        (tie-safe greedy CEFS+). Other registry names are reserved and raise.
    n_draws : int, default=1
        Number of knockoff draws. Above one, the default keeps features whose
        selection frequency reaches ``eta``, ``threshold`` becomes ``None``,
        and the aggregated result reports ``fdr_control="none"``. Opt-in
        ``aggregation="evalues"`` averages knockoff e-values instead.
    eta : float, default=0.5
        Selection-frequency cut for derandomized runs, in ``(0, 1]``. Ignored
        when ``aggregation="evalues"``.
    aggregation : {None, "evalues", "selection_frequency"}, default=None
        How to combine ``n_draws > 1``. ``None`` keeps the legacy frequency
        vote. ``"evalues"`` requires ``n_draws > 1`` and ``offset=1``.
    offset : {1, 0}, default=1
        ``1`` is the knockoff+ threshold; ``0`` is the less conservative
        modified-knockoff (mFDR-style) threshold.
    s_method : {"equi", "mvr", "me"}, default="equi"
        Diagonal decorrelation objective. ``"equi"`` is fastest; ``"mvr"`` and
        ``"me"`` use coordinate descent and can add power on correlated
        designs.
    min_eig : float, default=1e-3
        Minimum eigenvalue enforced on the estimated feature correlation.
        Shrinking towards it emits a plug-in-validity ``UserWarning``.
    screen_pairs : int or None, default=2000
        Positive cap on the candidate pairs screened by statistics that need
        screening; ``None`` screens every pair.
    statistic_options : dict or None, default=None
        Extra options for the chosen statistic: ``{"max_steps": int}`` for
        ``"lsm"``, ``{"ridge_lambda": float}`` for ``"ridge"``, and
        ``{"path_depth": int, "min_gain_ratio": float}`` for ``"cefsplus"``.
        Unknown keys raise.
    feature_groups : sequence, {"auto"} or None, default=None
        Group labels for a heuristic signed-maximum group aggregation, or
        ``"auto"`` to cluster near-collinear features and run the filter on one
        representative per cluster. Either mode expands selected groups back to
        their members and establishes no group- or feature-level FDR; the
        metadata reports ``"none"``.
    group_corr_threshold : float, default=0.7
        Absolute-correlation cut used by ``feature_groups="auto"`` clustering.
    cat_features : list of str or None, default=None
        Categorical column names to encode. ``None`` auto-detects ``object``,
        ``category`` and ``string`` DataFrame columns.
    cat_encoding : {"none", "target_cv", "onehot", "target", "loo", "james_stein", "loo_logit"}, default="none"
        One of ``"none"``, ``"target"``, ``"loo"``, ``"james_stein"`` or
        ``"loo_logit"``. ``"target_cv"`` is rejected outright here.
        ``"none"`` is the only value that preserves the Model-X FDR claim; the
        four legacy supervised encodings warn and downgrade the claim as
        described above. Note that ``sift.select_fdr`` itself has no
        ``cat_encoding`` parameter: the encoders live in this class.
    target_cv_n_splits : int, default=5
        Inherited constructor option of the shared preprocessing block. It has
        no effect here, because ``cat_encoding="target_cv"`` is rejected.
    target_cv_smoothing : {"auto"} or float, default="auto"
        Inherited constructor option of the shared preprocessing block, unused
        for the same reason.
    target_prior : float or None, default=None
        Inherited constructor option of the shared preprocessing block, unused
        for the same reason.
    warmup_policy : {"exclude", "zero_weight"}, default="zero_weight"
        Inherited constructor option of the shared preprocessing block, unused
        for the same reason.
    allow_full_data_target_encoding : bool, default=False
        Opt in to fitting the legacy supervised encoders on the full matrix.
    onehot_max_levels : int, default=32
        Accepted for constructor parity and rejected with
        ``cat_encoding="onehot"``: one-hot blocks are not a knockoff FDR
        claim.
    loo_smoothing : float, default=20.0
        Positive smoothing constant of the ``cat_encoding="loo_logit"``
        encoder.
    loo_clip_min : float, default=1e-4
        Lower probability clip of that encoder; ``0 < loo_clip_min <
        loo_clip_max < 1`` is enforced.
    loo_clip_max : float, default=1.0 - 1e-4
        Upper probability clip of that encoder.
    subsample : int, None or {"auto"}, default="auto"
        Row cap for uncached cache construction. ``"auto"`` means the omitted
        default: 50,000 rows when fitting from ``X``, and "not supplied" with a
        cache. An explicit value beside a ``cache`` raises.
    random_state : int, default=0
        Seed for the knockoff draw. Unlike the filter selectors this stays
        numeric, because it seeds a fresh draw even when a cache is reused.
    n_jobs : int, default=1
        Worker count for cache construction and statistic evaluation.
    verbose : bool, default=True
        Emit progress at INFO on the ``sift`` logger.
    cache : FeatureCache or None, default=None
        Prebuilt Gaussian-copula cache to reuse. A named cache requires a
        DataFrame with identical columns in identical order; a cache built from
        positional features requires the matching ndarray. A cache already
        stores row weights, so ``sample_weight`` is rejected beside it, and a
        supervised ``cat_encoding`` is rejected too.
    include : sequence of names or positions, optional
        Conditioning set. These features are not tested by the knockoff
        filter; they are prepended to the selected set in caller order.
        Any of ``include``, ``exclude``, or ``candidates`` requires
        ``include_provenance``.
    exclude : sequence of names or positions, optional
        Features removed from the tested discovery universe. Requires
        ``include_provenance``.
    candidates : sequence of names or positions, optional
        Hard allow-list for the tested discovery universe. Requires
        ``include_provenance``.
    feature_blocks : mapping, {"auto"} or None, default None
        Additive alias of ``feature_groups``. A mapping is converted to
        per-column labels (unlisted columns stay singletons). ``"auto"``
        means the existing correlation-cluster ``feature_groups="auto"``,
        not the filter one-hot prefix convention. Conflicting
        ``feature_groups`` and ``feature_blocks`` raise. Grouped FDR
        validity is unchanged.
    include_provenance : {"prespecified", "sample_split", "data_derived"} or None
        Required when ``include``, ``exclude``, or ``candidates`` is
        provided. FDR-compatible wording is allowed only for
        ``prespecified`` and ``sample_split``. ``data_derived`` is labeled
        exploratory and reports ``fdr_control="none"``.
    output_order : {"legacy", "original"}, default="legacy"
        Order used by ``transform``, ``get_support(indices=True)``,
        ``get_feature_names_out`` and ``inverse_transform``. ``"legacy"`` keeps
        discovery order; ``"original"`` emits ascending fitted column position.
        The boolean support mask is always positional.

    Attributes
    ----------
    result_ : KnockoffSelectionResult
        The full result: ``selected_features``, ``selected_indices``, the ``W``
        diagnostics table, ``threshold``, ``selection_frequency`` and
        ``selector_metadata`` (including the validity keys). Pass it to
        ``sift.as_result`` for a normalized ``SelectionView``.
    selected_features_ : list
        Selected feature labels.
    selected_indices_ : ndarray of shape (n_selected,)
        Their positions in the fitted feature matrix.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        One-dimensional object array of fitted feature names. A positional
        ndarray fit stores generated ``x0...`` names here.
    n_features_in_ : int
        Number of candidate features seen during ``fit``.
    categorical_features_ : list
        Categorical columns the fitted encoder covered.
    categorical_encoder_ : object or None
        The fitted encoder, reused target-blind by ``transform``.

    Raises
    ------
    ValueError
        If ``groups`` or ``time`` is passed in any mode, if ``auto_k_config``
        is passed, if ``cat_encoding="target_cv"`` is requested, if
        ``sample_weight``, an explicit ``subsample`` or a supervised
        ``cat_encoding`` accompanies a ``cache``, or if ``X`` is sparse or not
        two-dimensional.
    NotImplementedError
        From ``inverse_transform`` after a supervised categorical encoding,
        because the fitted encoder is not invertible.

    Warns
    -----
    UserWarning
        When a legacy supervised ``cat_encoding`` downgrades ``fdr_control`` to
        ``"none"``, when the estimated correlation is shrunk towards
        ``min_eig``, when ``feature_groups="auto"`` is advisable because
        the median decorrelation ``s`` is tiny, and when knockoff+
        (``offset=1``) has one or more completed draws with ``m·q < 1``.

    See Also
    --------
    sift.select_fdr : Function-style knockoff filter with the same options.
    sift.build_cache : Build the Gaussian-copula cache this can reuse.
    CEFSPlusSelector : Fixed-``k`` or measured-``k`` filter alternative.

    Notes
    -----
    ``fit(X, y, sample_weight=None)`` is the supported contract; ``cache`` may
    also be passed per call. Row ``groups``/``time`` are refused in every mode
    (use ``feature_groups`` for grouped *feature* discoveries), and sklearn
    metadata routing exposes only ``sample_weight``. The 0.9 filter reports
    plug-in validity metadata: ``fdr_control="approximate_plugin"`` under the
    fitted Gaussian-copula feature model, so with estimated correlations,
    shrinkage, weights, derandomization or feature groups the result should be
    read as an approximate practical knockoff filter. ``min_feasible_q`` is a
    necessary count bound from completed draws, not a sufficient discovery
    condition, and an infeasible draw does not imply an empty aggregate.
    ``tested_state="not_run"`` means no draw or pair-screen ran.
    ``n_discoveries_offset_0`` counts reported features from the same ``W``.
    Opt-in ``aggregation="evalues"`` records the common tested universe and
    reports ``approximate_plugin`` only for ungrouped ``relevance`` and
    ``ridge`` with a screening universe fixed before the statistics.
    ``lsm``, ``cefsplus``, grouped runs, and supervised encodings stay
    ``fdr_control="none"``. The e-value bound is an aggregate null
    expectation, not a unit bound per feature.
    They do not change omitted-option selections, ``q``, the statistic, or
    the default FDR labels.

    Examples
    --------
    >>> import numpy as np
    >>> from sift import KnockoffSelector
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(400, 20))
    >>> y = 2.0 * X[:, :5].sum(axis=1) + 0.2 * rng.normal(size=400)
    >>> selector = KnockoffSelector(q=0.2, random_state=0, verbose=False)
    >>> sorted(selector.fit(X, y).selected_features_)
    ['x0', 'x1', 'x2', 'x3', 'x4']
    >>> selector.result_.selector_metadata["fdr_control"]
    'approximate_plugin'
    """

    _subsample_auto_is_cache_default = True
    __metadata_request__fit = {"groups": UNUSED, "time": UNUSED}

    def __init__(
        self,
        q: float = 0.1,
        *,
        statistic: str = "relevance",
        n_draws: int = 1,
        eta: float = 0.5,
        aggregation: str | None = None,
        offset: int = 1,
        s_method: str = "equi",
        min_eig: float = 1e-3,
        screen_pairs: int | None = 2000,
        statistic_options: dict | None = None,
        feature_groups=None,
        group_corr_threshold: float = 0.7,
        cat_features: list[str] | None = None,
        cat_encoding: str = "none",
        target_cv_n_splits: int = 5,
        target_cv_smoothing: Literal["auto"] | float = "auto",
        target_prior: float | None = None,
        warmup_policy: Literal["exclude", "zero_weight"] = "zero_weight",
        allow_full_data_target_encoding: bool = False,
        onehot_max_levels: int = 32,
        loo_smoothing: float = 20.0,
        loo_clip_min: float = 1e-4,
        loo_clip_max: float = 1.0 - 1e-4,
        subsample: int | None | Literal["auto"] = "auto",
        random_state: int = 0,
        n_jobs: int = 1,
        verbose: bool = True,
        cache=None,
        include=None,
        exclude=None,
        candidates=None,
        include_provenance=None,
        feature_blocks=None,
        output_order: str = "legacy",
    ):
        self._init_selector(select_fdr, locals())

    def _supports_auto_k(self) -> bool:
        return False

    def _validate_categorical_encoding_params(self) -> None:
        if self.cat_encoding == "onehot":
            raise ValueError(
                "KnockoffSelector does not support cat_encoding='onehot'. "
                "One-hot blocks are a filter-selector contract and are not a "
                "Model-X FDR claim. Pre-encode outside the selector or use a "
                "filter selector."
            )
        if self.cat_encoding == "target_cv":
            raise ValueError(
                "KnockoffSelector does not support cat_encoding='target_cv'. "
                "Cross-fitted target encoding is still target-derived "
                "preprocessing, which breaks the Model-X exchangeability the "
                "knockoff FDR claim rests on. Pre-encode categoricals "
                "leakage-safely outside the selector and pass "
                "cat_encoding='none', or use a filter selector when you need "
                "target_cv."
            )
        super()._validate_categorical_encoding_params()

    def _more_tags(self):
        # sklearn <1.6 returns a module-level default dict here.  Copy it
        # before overriding one tag, otherwise instantiating this selector
        # changes the non_deterministic tag for every BaseEstimator instance.
        return selector_tags(super()._more_tags(), non_deterministic=True)

    def __sklearn_tags__(self):
        """Expose the row-order sensitivity through sklearn's new tag API."""
        parent_tags = getattr(super(), "__sklearn_tags__", None)
        if parent_tags is None:  # sklearn <1.6 uses the dict API above.
            return self._more_tags()
        return selector_tags(parent_tags(), non_deterministic=True)

    def _clear_fit_state(self) -> None:
        super()._clear_fit_state()
        if hasattr(self, "result_"):
            delattr(self, "result_")

    def _fit_impl(
        self,
        X,
        y,
        *,
        sample_weight=None,
        groups=None,
        time=None,
        cache=None,
        auto_k_config=None,
        capture_training_output: bool = False,
        **fit_params,
    ):
        _require_2d_x(X)
        validate_output_order(self.output_order)
        if groups is not None:
            raise ValueError(
                "KnockoffSelector does not support row groups. Use feature_groups "
                "on the estimator for grouped feature discoveries."
            )
        if time is not None:
            raise ValueError("KnockoffSelector does not support time-aware fitting.")
        if auto_k_config is not None:
            raise ValueError("KnockoffSelector is q-based and does not support auto_k_config.")
        self._validate_categorical_encoding_params()

        resolved_cache = cache if cache is not None else getattr(self, "cache", None)
        if resolved_cache is not None and sample_weight is not None:
            raise ValueError(
                "sample_weight cannot be passed with a prebuilt cache; the cache "
                "already stores row weights"
            )

        self._clear_fit_state()
        has_supervised_categoricals = self._would_fit_supervised_categoricals(X)
        if resolved_cache is not None and has_supervised_categoricals:
            raise ValueError(
                "KnockoffSelector supervised categorical encoding does not support "
                "prebuilt caches. Use cat_encoding='none' with a cache, or omit the "
                "cache so the selector can fit encoders on the training rows."
            )

        call_params = dict(self._selector_params())
        if fit_params:
            blocked = sorted(_BLOCKED_FIT_PARAM_OVERRIDES.intersection(fit_params))
            if blocked:
                blocked_text = ", ".join(blocked)
                raise ValueError(
                    "KnockoffSelector return-shape or preprocessing-affecting "
                    f"parameters must be set on the estimator before fit, not as "
                    f"fit-time overrides: {blocked_text}"
                )
            call_params.update(fit_params)
        self._resolve_auto_selector_params(call_params)

        feature_names, names_generated = _feature_names_with_provenance(X)
        if resolved_cache is None:
            X_fit = self._fit_transform_categoricals(X, y, sample_weight=sample_weight)
            effective_sample_weight = sample_weight
            if (
                self.cat_encoding == "target_cv"
                and getattr(self, "categorical_encoder_", None) is not None
                and getattr(self.categorical_encoder_, "effective_sample_weight_", None)
                is not None
            ):
                effective_sample_weight = self.categorical_encoder_.effective_sample_weight_
            result = self._selector_fn(
                X_fit,
                y,
                sample_weight=effective_sample_weight,
                **call_params,
            )
        else:
            if sample_weight is not None:
                raise ValueError("sample_weight cannot be passed with a prebuilt cache")
            x_shape = X.shape if hasattr(X, "shape") else np.asarray(X).shape
            if len(x_shape) != 2:
                raise ValueError("X must be a 2D feature matrix")
            n_rows, n_features = int(x_shape[0]), int(x_shape[1])
            y_arr = np.asarray(y).reshape(-1)
            if y_arr.size != n_rows:
                raise ValueError(
                    f"X has {n_rows} rows but y has {y_arr.size} rows"
                )
            # Validate provenance before deciding whether generated-looking
            # cache names are positional placeholders or real DataFrame labels.
            _validate_prebuilt_cache_structure(
                resolved_cache,
                original_n_features=n_features,
                n_rows=n_rows,
                validate_rxx=False,
            )
            cache_names = resolved_cache.feature_names
            if resolved_cache.feature_names_are_synthetic and isinstance(X, pd.DataFrame):
                raise ValueError(
                    "A cache built from unnamed/positional features requires X to be "
                    "the compatible positional ndarray; rebuild the cache from this "
                    "DataFrame to establish column names and order"
                )
            if cache_names is not None and not resolved_cache.feature_names_are_synthetic:
                if list(feature_names) != list(cache_names):
                    raise ValueError(
                        "X columns do not match cache.feature_names (names and order must "
                        "be identical); fit the cache from the same matrix"
                    )
            elif cache_names is not None and len(feature_names) != len(cache_names):
                raise ValueError(
                    f"X has {len(feature_names)} columns but the cache was built from "
                    f"{len(cache_names)}"
                )
            X_fit = X
            result = self._selector_fn(
                None,
                y,
                cache=resolved_cache,
                sample_weight=None,
                **call_params,
            )

        result = self._downgrade_fdr_claim_for_supervised_encoding(result)
        selected_indices = result.selected_indices
        if selected_indices is None:
            selected_indices = _coerce_selection_indices(
                feature_names,
                list(result.selected_features),
            ).tolist()

        self.feature_names_in_ = feature_names_array(feature_names)
        self._fit_feature_names_generated_ = names_generated
        self.n_features_in_ = len(feature_names)
        self.selected_features_ = list(result.selected_features)
        self.selected_indices_ = np.asarray(selected_indices, dtype=np.int64)
        self.result_ = result
        if capture_training_output:
            self._fit_transform_output_ = _selected_training_output(
                X_fit,
                self._output_indices(),
            )
        return self

    def _downgrade_fdr_claim_for_supervised_encoding(self, result):
        """Warn and drop the FDR claim when 0.8 supervised encoding was used.

        The legacy encodings stay available for 0.8 compatibility, but their
        target-derived columns are not exchangeable with their knockoffs, so the
        result must not keep advertising an FDR guarantee.
        """
        if not getattr(self, "_categorical_encoding_applied_", False):
            return result
        if self.cat_encoding not in _KNOCKOFF_LEGACY_CAT_ENCODINGS:
            return result
        warnings.warn(
            f"KnockoffSelector(cat_encoding={self.cat_encoding!r}): "
            f"{_KNOCKOFF_SUPERVISED_ENCODING_NOTE}. The result reports "
            "fdr_control='none'. Pre-encode categoricals leakage-safely and "
            "pass cat_encoding='none' to keep the Model-X FDR claim.",
            UserWarning,
            stacklevel=3,
        )
        return self._apply_supervised_encoding_fdr_downgrade(result)

    def _apply_supervised_encoding_fdr_downgrade(self, result):
        metadata = dict(result.selector_metadata)
        metadata.update(
            {
                "fdr_control": "none",
                "per_draw_fdr_control": "none",
                "aggregation_preserves_per_draw_fdr": False,
                "cat_encoding": self.cat_encoding,
                "validity_note": _KNOCKOFF_SUPERVISED_ENCODING_NOTE,
            }
        )
        updates = {"selector_metadata": metadata}
        if (
            metadata.get("aggregation") in {"evalues", "evalues_then_cluster_expansion"}
            or "evalue_validated" in metadata
        ):
            metadata["aggregation_fdr_control"] = "none"
            metadata["evalue_validated"] = False
            reasons = list(metadata.get("evalue_exploratory_reasons") or [])
            if "supervised_categorical_encoding" not in reasons:
                reasons.append("supervised_categorical_encoding")
            metadata["evalue_exploratory_reasons"] = reasons
            metadata["exploratory"] = True
            for key in (
                "representative_fdr_control",
                "representative_per_draw_fdr_control",
            ):
                if key in metadata:
                    metadata[key] = "none"
            updates["selector_metadata"] = metadata
            diagnostics = result.diagnostics_
            nested = None if diagnostics is None else diagnostics.get("representative_result")
            if nested is not None:
                diagnostics = dict(diagnostics)
                diagnostics["representative_result"] = (
                    self._apply_supervised_encoding_fdr_downgrade(nested)
                )
                updates["diagnostics_"] = diagnostics
        return replace(result, **updates)


__all__ = [
    "MRMRSelector",
    "JMISelector",
    "JMIMSelector",
    "CEFSPlusSelector",
    "CEFSPlusBinarySelector",
    "KnockoffSelector",
]
