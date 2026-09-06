"""Sklearn integration contracts for SIFT selector estimators."""

from __future__ import annotations

import inspect
import pickle
import warnings
from unittest import SkipTest

import numpy as np
import pandas as pd
import pytest
import sklearn
from scipy import sparse
from sklearn import config_context
from sklearn.base import clone
from sklearn.feature_selection import SelectorMixin
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.utils.estimator_checks import (
    check_complex_data,
    check_estimators_unfitted,
    check_get_params_invariance,
    check_no_attributes_set_in_init,
    check_parameters_default_constructible,
    check_sample_weights_not_an_array,
    check_set_params,
    check_transformer_data_not_an_array,
    check_transformers_unfitted,
)

try:  # Added in sklearn 1.2; keep the pin tolerant of older/newer layouts.
    from sklearn.utils.estimator_checks import (
        check_dataframe_column_names_consistency,
    )
except ImportError:  # pragma: no cover - exercised only on unsupported sklearn
    check_dataframe_column_names_consistency = None

import sift


SELECTOR_CLASSES = (
    sift.MRMRSelector,
    sift.JMISelector,
    sift.JMIMSelector,
    sift.CEFSPlusSelector,
    sift.CEFSPlusBinarySelector,
    sift.KnockoffSelector,
    sift.BorutaSelector,
    sift.StabilitySelector,
)

NAN_SELECTOR_CASES = (
    (sift.MRMRSelector(k=1, task="regression", verbose=False), "regression"),
    (sift.JMISelector(k=1, task="regression", verbose=False), "regression"),
    (sift.JMIMSelector(k=1, task="regression", verbose=False), "regression"),
    (sift.CEFSPlusSelector(k=1, verbose=False), "regression"),
    (sift.CEFSPlusBinarySelector(k=1, verbose=False), "binary"),
    (
        sift.KnockoffSelector(
            q=0.5,
            offset=0,
            screen_pairs=None,
            verbose=False,
        ),
        "regression",
    ),
    (
        sift.BorutaSelector(
            n_estimators=10,
            max_iter=2,
            early_stop_rounds=2,
            verbose=False,
        ),
        "regression",
    ),
    (
        sift.StabilitySelector(
            n_bootstrap=2,
            alpha=0.5,
            threshold=0.0,
            n_jobs=1,
            random_state=0,
            verbose=False,
        ),
        "regression",
    ),
)

PINNED_GREEN_CHECKS = (
    check_no_attributes_set_in_init,
    check_parameters_default_constructible,
    check_estimators_unfitted,
    check_transformers_unfitted,
    check_get_params_invariance,
    check_set_params,
)

# Checks that exercise duck-typed, complex, and DataFrame-named input.  They fit
# real data, so they run against the cheap configurations below.
PINNED_INPUT_CONTRACT_CHECKS = (
    check_complex_data,
    check_sample_weights_not_an_array,
    check_transformer_data_not_an_array,
)

# sklearn refuses to run transform-equivalence checks against estimators tagged
# ``non_deterministic``; KnockoffSelector is the only such SIFT selector.
EXPECTED_CHECK_SKIPS = frozenset(
    {(sift.KnockoffSelector, "check_transformer_data_not_an_array")}
)

# BorutaSelector and StabilitySelector deliberately select fitted columns by
# name in ``transform``, so a reordered or widened DataFrame is honored instead
# of rejected.  sklearn's column-name check requires strict positional identity,
# which would contradict that documented contract.
NAME_BASED_TRANSFORM_CLASSES = (sift.BorutaSelector, sift.StabilitySelector)

SKLEARN_VERSION = tuple(
    int(part) for part in sklearn.__version__.split(".")[:2]
)


def _fast_selector(selector_cls):
    """Return a cheap but representative instance of ``selector_cls``."""
    if selector_cls is sift.BorutaSelector:
        return selector_cls(
            n_estimators=10,
            max_iter=2,
            early_stop_rounds=2,
            random_state=0,
            verbose=False,
        )
    if selector_cls is sift.StabilitySelector:
        return selector_cls(n_bootstrap=3, random_state=0, verbose=False)
    if selector_cls is sift.KnockoffSelector:
        return selector_cls(verbose=False)
    return selector_cls(k=2, verbose=False)


class _ArrayFunctionRefusingDuckArray:
    """Local stand-in for sklearn's private ``_NotAnArray`` check helper.

    It is convertible through ``__array__`` but raises from
    ``__array_function__``, which is exactly what made ``np.iscomplexobj`` on
    raw fit input crash. The helper is defined here because sklearn's own class
    is private and has moved between versions.
    """

    def __init__(self, data):
        self.data = np.asarray(data)

    def __array__(self, dtype=None, copy=None):
        return self.data

    def __array_function__(self, func, types, args, kwargs):
        raise TypeError(f"Don't want to call array_function {func.__name__}!")


def _fit_data(selector_cls, n_samples: int = 60, n_features: int = 4):
    """Return (X, y) that every selector class can fit."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_samples, n_features))
    signal = X[:, 0] + 0.5 * X[:, 1]
    if selector_cls is sift.CEFSPlusBinarySelector:
        return X, (signal > 0).astype(np.int64)
    return X, signal + rng.normal(scale=0.1, size=n_samples)


@pytest.mark.parametrize("selector_cls", SELECTOR_CLASSES)
def test_selector_mixin_default_and_pinned_green_checks(selector_cls):
    """Pin the deliberately supported common sklearn estimator checks."""
    assert issubclass(selector_cls, SelectorMixin)
    selector = selector_cls()
    assert selector.output_order == "legacy"
    for check in PINNED_GREEN_CHECKS:
        check(selector_cls.__name__, selector)


@pytest.mark.parametrize("selector_cls", SELECTOR_CLASSES)
def test_pinned_duck_typed_and_complex_input_checks(selector_cls):
    """Pin the input-validation checks fixed in the 0.9 sklearn repair."""
    for check in PINNED_INPUT_CONTRACT_CHECKS:
        selector = _fast_selector(selector_cls)
        expected_skip = (selector_cls, check.__name__) in EXPECTED_CHECK_SKIPS
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                check(selector_cls.__name__, selector)
            except SkipTest:
                assert expected_skip, f"{check.__name__} was unexpectedly skipped"
            else:
                assert not expected_skip, f"{check.__name__} unexpectedly ran"


@pytest.mark.skipif(
    check_dataframe_column_names_consistency is None,
    reason="check_dataframe_column_names_consistency requires sklearn >= 1.2",
)
@pytest.mark.parametrize("selector_cls", SELECTOR_CLASSES)
def test_pinned_dataframe_column_names_consistency(selector_cls):
    """Pin sklearn's feature-name contract for order-strict transforms."""
    if selector_cls in NAME_BASED_TRANSFORM_CLASSES:
        pytest.skip(
            f"{selector_cls.__name__}.transform selects fitted columns by name; "
            "the ndarray/dtype half of the contract is pinned separately"
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_dataframe_column_names_consistency(
            selector_cls.__name__,
            _fast_selector(selector_cls),
        )


@pytest.mark.parametrize("selector_cls", SELECTOR_CLASSES)
def test_selector_fit_accepts_array_function_refusing_duck_arrays(selector_cls):
    """Regression: dense validation must not dispatch ``__array_function__``."""
    X, y = _fit_data(selector_cls)
    selector = _fast_selector(selector_cls)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = selector.fit(_ArrayFunctionRefusingDuckArray(X), y)

    assert fitted is selector
    assert fitted.n_features_in_ == X.shape[1]
    assert fitted.get_support().shape == (X.shape[1],)


@pytest.mark.parametrize("selector_cls", SELECTOR_CLASSES)
def test_selector_fit_rejects_complex_data_with_sklearn_wording(selector_cls):
    """sklearn's ``check_complex_data`` matches "Complex data not supported"."""
    X, y = _fit_data(selector_cls, n_samples=20, n_features=2)
    X_complex = X.astype(np.complex128) + 1j
    selector = _fast_selector(selector_cls)
    with pytest.raises(ValueError, match="Complex data not supported"):
        selector.fit(X_complex, y)

    frame = pd.DataFrame(X_complex, columns=["a", "b"])
    with pytest.raises(ValueError, match="Complex data not supported"):
        _fast_selector(selector_cls).fit(frame, y)


@pytest.mark.parametrize("selector_cls", SELECTOR_CLASSES)
def test_feature_names_in_is_object_ndarray_for_named_and_positional_fits(
    selector_cls,
):
    """``feature_names_in_`` follows sklearn's dtype contract in both modes."""
    X, y = _fit_data(selector_cls)
    names = [f"col_{i}" for i in range(X.shape[1])]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        named = _fast_selector(selector_cls).fit(pd.DataFrame(X, columns=names), y)
        positional = _fast_selector(selector_cls).fit(X, y)

    for fitted, expected, generated in (
        (named, names, False),
        # Generated ``x0...`` names stay in the public attribute: this is
        # existing 0.8 behavior, and only the private marker records provenance.
        (positional, [f"x{i}" for i in range(X.shape[1])], True),
    ):
        assert isinstance(fitted.feature_names_in_, np.ndarray)
        assert fitted.feature_names_in_.ndim == 1
        assert fitted.feature_names_in_.dtype == object
        assert list(fitted.feature_names_in_) == expected
        assert fitted._fit_feature_names_generated_ is generated

        selected = fitted.get_feature_names_out()
        assert isinstance(selected, np.ndarray)
        assert selected.dtype == object
        assert set(selected) <= set(expected)

    # get_params/clone/pickle stay unaffected by the fitted-attribute dtype.
    assert clone(named).get_params() == _fast_selector(selector_cls).get_params()
    restored = pickle.loads(pickle.dumps(named))
    assert list(restored.feature_names_in_) == names
    assert restored.feature_names_in_.dtype == object


def test_cefsplus_binary_declares_binary_only_across_tag_apis():
    """C6: legacy ``binary_only`` tag plus an honest object-tag representation."""
    selector = sift.CEFSPlusBinarySelector(k=2, verbose=False)

    # Legacy dict tag API. sklearn <1.6 reads it through ``_safe_tags``; on
    # newer versions it stays as SIFT's own declaration of the constraint.
    assert selector._more_tags()["binary_only"] is True
    # No other selector opts in. sklearn <1.6 seeds the dict with
    # binary_only=False; on newer versions BaseEstimator contributes no
    # defaults, so the key is simply absent.
    assert sift.MRMRSelector()._more_tags().get("binary_only", False) is False

    if SKLEARN_VERSION < (1, 6):
        from sklearn.utils._tags import _safe_tags

        assert _safe_tags(selector)["binary_only"] is True
        assert _safe_tags(sift.MRMRSelector())["binary_only"] is False
    else:
        from sklearn.utils import get_tags

        tags = get_tags(selector)
        # sklearn >=1.6 replaced ``binary_only`` with
        # ``Tags.classifier_tags.multi_class``, which only exists for
        # estimators typed as classifiers. This selector is a transformer, so
        # the nearest valid representation leaves ``classifier_tags`` unset
        # instead of misdeclaring the estimator type to obtain a tag.
        assert tags.classifier_tags is None
        assert tags.estimator_type != "classifier"
        assert tags.transformer_tags is not None

    # Either way the two-class requirement is enforced at fit time.
    X = np.arange(30, dtype=np.float64).reshape(10, 3)
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    with pytest.raises(ValueError, match="exactly two target classes"):
        selector.fit(X, y)


def _fitted_filter_selector(output_order: str):
    selector = sift.MRMRSelector(k=2, output_order=output_order, verbose=False)
    selector.n_features_in_ = 4
    selector.feature_names_in_ = ["a", "b", "c", "d"]
    selector.selected_indices_ = np.array([2, 0], dtype=np.int64)
    selector.selected_features_ = ["c", "a"]
    selector._row_metadata_columns_ = ()
    selector._categorical_encoding_applied_ = False
    return selector


def _fitted_boruta_selector(output_order: str):
    selector = sift.BorutaSelector(output_order=output_order, verbose=False)
    selector.n_features_in_ = 4
    selector.feature_names_in_ = ["a", "b", "c", "d"]
    selector.status_ = np.array([1, -1, 1, -1], dtype=np.int8)
    selector._row_metadata_columns_ = ()
    selector._categorical_encoding_applied_ = False
    return selector


def _fitted_stability_selector(output_order: str):
    selector = sift.StabilitySelector(output_order=output_order, verbose=False)
    selector.n_features_in_ = 4
    selector.feature_names_in_ = ["a", "b", "c", "d"]
    selector.selected_features_ = np.array([2, 0], dtype=np.int64)
    selector.selected_feature_names_ = ["c", "a"]
    selector._fit_feature_names_generated_ = False
    return selector


@pytest.mark.parametrize(
    "factory,legacy_indices",
    (
        (_fitted_filter_selector, [2, 0]),
        (_fitted_boruta_selector, [0, 2]),
        (_fitted_stability_selector, [2, 0]),
    ),
    ids=("filter", "boruta", "stability"),
)
@pytest.mark.parametrize("output_order", ("legacy", "original"))
def test_output_order_support_transform_names_and_inverse(
    factory,
    legacy_indices,
    output_order,
):
    selector = factory(output_order)
    expected = legacy_indices if output_order == "legacy" else sorted(legacy_indices)
    X = np.arange(20, dtype=np.float64).reshape(5, 4)

    np.testing.assert_array_equal(selector.get_support(indices=True), expected)
    np.testing.assert_array_equal(
        selector.get_support(),
        np.isin(np.arange(4), expected),
    )
    assert selector.get_feature_names_out().tolist() == [
        selector.feature_names_in_[index] for index in expected
    ]

    transformed = selector.transform(X)
    np.testing.assert_array_equal(transformed, X[:, expected])
    restored = selector.inverse_transform(transformed)
    expected_restored = np.zeros_like(X)
    expected_restored[:, expected] = X[:, expected]
    np.testing.assert_array_equal(restored, expected_restored)


@pytest.mark.parametrize("selector_cls", SELECTOR_CLASSES)
def test_all_selector_fits_reject_sparse_input_consistently(selector_cls):
    X = sparse.csr_matrix(np.eye(4, dtype=np.float64))
    y = np.arange(4, dtype=np.float64)
    with pytest.raises(TypeError, match="Sparse matrices are not supported.*fit"):
        selector_cls(verbose=False).fit(X, y)


@pytest.mark.parametrize("selector_cls", SELECTOR_CLASSES)
def test_all_selector_fits_validate_output_order(selector_cls):
    X = np.eye(4, dtype=np.float64)
    y = np.arange(4, dtype=np.float64)
    with pytest.raises(ValueError, match="output_order must be 'legacy' or 'original'"):
        selector_cls(output_order="ranked", verbose=False).fit(X, y)


@pytest.mark.parametrize(
    "selector",
    (
        _fitted_filter_selector("legacy"),
        _fitted_boruta_selector("legacy"),
        _fitted_stability_selector("legacy"),
    ),
    ids=("filter", "boruta", "stability"),
)
def test_selector_transform_and_inverse_reject_sparse_input(selector):
    X = sparse.csr_matrix(np.eye(4, dtype=np.float64))
    with pytest.raises(TypeError, match="Sparse matrices are not supported.*transform"):
        selector.transform(X)
    with pytest.raises(
        TypeError,
        match="Sparse matrices are not supported.*inverse_transform",
    ):
        selector.inverse_transform(X[:, :2])


def _selector_tag_values(selector):
    if SKLEARN_VERSION >= (1, 6):
        from sklearn.utils import get_tags

        tags = get_tags(selector)
        return (
            tags.input_tags.allow_nan,
            tags.target_tags.required,
            tags.non_deterministic,
            tags.transformer_tags is not None,
        )

    from sklearn.utils._tags import _safe_tags

    tags = _safe_tags(selector)
    return (
        tags["allow_nan"],
        tags["requires_y"],
        tags["non_deterministic"],
        True,
    )


@pytest.mark.parametrize("selector_cls", SELECTOR_CLASSES)
def test_selector_tags_match_fitted_contract(selector_cls):
    allow_nan, requires_y, nondeterministic, is_transformer = _selector_tag_values(
        selector_cls()
    )
    assert allow_nan is True
    assert requires_y is True
    assert is_transformer is True
    assert nondeterministic is (selector_cls is sift.KnockoffSelector)


@pytest.mark.parametrize(
    "selector,target_kind",
    NAN_SELECTOR_CASES,
    ids=[type(case[0]).__name__ for case in NAN_SELECTOR_CASES],
)
def test_allow_nan_tag_matches_fit_behavior(selector, target_kind):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 3))
    X[0, 1] = np.nan
    y_regression = np.nan_to_num(X[:, 0]) + rng.normal(scale=0.1, size=len(X))
    y = (np.nan_to_num(X[:, 0]) > 0).astype(np.int64)
    if target_kind == "regression":
        y = y_regression

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selector.fit(X, y)
    assert selector.get_support().shape == (X.shape[1],)


def test_metadata_request_setters_expose_only_supported_fit_metadata():
    row_metadata = {"sample_weight", "groups", "time"}
    estimators = [
        cls()
        for cls in (
            sift.MRMRSelector,
            sift.JMISelector,
            sift.JMIMSelector,
            sift.CEFSPlusSelector,
            sift.CEFSPlusBinarySelector,
            sift.BorutaSelector,
            sift.StabilitySelector,
        )
    ]
    estimators.append(
        sift.Stabilized(sift.CEFSPlusSelector(k=1, verbose=False), verbose=False)
    )
    for estimator in estimators:
        parameters = set(inspect.signature(estimator.set_fit_request).parameters) - {
            "self"
        }
        assert parameters == row_metadata

    knockoff_parameters = set(
        inspect.signature(sift.KnockoffSelector().set_fit_request).parameters
    ) - {"self"}
    assert knockoff_parameters == {"sample_weight"}


def test_invalid_configured_metadata_requests_fail_before_fit():
    with config_context(enable_metadata_routing=True):
        fixed = sift.MRMRSelector(k=1, verbose=False).set_fit_request(groups=True)
        with pytest.raises(ValueError, match="only when k='auto'"):
            fixed.get_metadata_routing()

        smart = sift.StabilitySelector(
            use_smart_sampler=True,
            verbose=False,
        ).set_fit_request(sample_weight=True)
        with pytest.raises(ValueError, match="use_smart_sampler=True"):
            smart.get_metadata_routing()

        frequency = sift.Stabilized(
            sift.CEFSPlusSelector(k=1, verbose=False),
            resample="half",
            verbose=False,
        ).set_fit_request(groups=True)
        with pytest.raises(ValueError, match="resample='blocks'"):
            frequency.get_metadata_routing()


@pytest.mark.skipif(
    SKLEARN_VERSION < (1, 4),
    reason="cross_validate(params=...) requires sklearn 1.4+",
)
def test_group_metadata_routes_through_pipeline_cross_validate():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(6), 10)
    X = rng.normal(size=(60, 4))
    y = 2.0 * X[:, 0] + rng.normal(scale=0.2, size=60)
    selector = sift.MRMRSelector(
        k="auto",
        task="regression",
        estimator="classic",
        mrmr_backend="serial",
        subsample=None,
        verbose=False,
        auto_k_config=sift.AutoKConfig(
            k_method="evaluate",
            strategy="group_cv",
            metric="rmse",
            min_k=1,
            max_k=2,
            n_splits=3,
            random_state=42,
        ),
    )

    with config_context(enable_metadata_routing=True):
        selector.set_fit_request(groups=True)
        result = cross_validate(
            make_pipeline(selector, Ridge()),
            X,
            y,
            cv=GroupKFold(3),
            params={"groups": groups},
            error_score="raise",
            return_estimator=True,
        )

    assert np.isfinite(result["test_score"]).all()
    assert [
        estimator.steps[0][1].get_support(indices=True).tolist()
        for estimator in result["estimator"]
    ] == [[0], [0], [0]]


@pytest.mark.skipif(
    SKLEARN_VERSION < (1, 4),
    reason="stable metadata routing requires sklearn 1.4+",
)
def test_routed_stability_fit_isolates_private_grid_search():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 3))
    y = X[:, 0] + rng.normal(scale=0.1, size=30)
    weights = np.linspace(0.5, 1.5, len(y))

    with config_context(enable_metadata_routing=True):
        selector = sift.StabilitySelector(
            n_bootstrap=2,
            sample_frac=0.7,
            threshold=0.0,
            alpha=None,
            n_jobs=1,
            random_state=0,
            verbose=False,
        ).set_fit_request(sample_weight=True)
        final_model = Ridge().set_fit_request(sample_weight=False)
        fitted = make_pipeline(selector, final_model).fit(
            X,
            y,
            sample_weight=weights,
        )

    fitted_selector = fitted.steps[0][1]
    assert np.isfinite(fitted_selector.alpha_)
    np.testing.assert_array_equal(fitted_selector.get_support(), np.ones(3, dtype=bool))
