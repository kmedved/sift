import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from sift import (
    build_cache,
    select_cefsplus,
    select_fdr,
    select_jmi,
    select_jmim,
    select_mrmr,
)
from sift.selection.cefsplus import select_cached
from sift.selection.knockoff_filter import sample_knockoffs
from sift.selectors import (
    CEFSPlusBinarySelector,
    CEFSPlusSelector,
    JMISelector,
    JMIMSelector,
    KnockoffSelector,
    MRMRSelector,
)


def _data(n=80, p=4):
    rng = np.random.default_rng(19)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = X["f0"].to_numpy() + 0.1 * rng.normal(size=n)
    return X, y


def _non_deterministic_tag(estimator) -> bool:
    """Read the public sklearn tag across the pre/post-1.6 APIs."""
    try:
        from sklearn.utils import get_tags
    except ImportError:  # sklearn < 1.6
        return bool(estimator._get_tags()["non_deterministic"])

    tags = get_tags(estimator)
    if isinstance(tags, dict):
        return bool(tags["non_deterministic"])
    return bool(tags.non_deterministic)


def test_gaussian_named_cache_requires_exact_dataframe_columns_and_order():
    X, y = _data()
    cache = build_cache(X, subsample=None)

    with pytest.raises(ValueError, match="names and order"):
        select_cefsplus(X[["f1", "f0", "f2", "f3"]], y, k=1, cache=cache, verbose=False)


def test_named_cache_rejects_duplicate_feature_names_across_gaussian_consumers():
    X, y = _data()
    X.columns = ["f0", "f0", "f2", "f3"]
    cache = build_cache(X, subsample=None, compute_Rxx=True)

    consumers = [
        lambda: select_cached(cache, y, k=1),
        lambda: select_cefsplus(X, y, k=1, cache=cache, verbose=False),
        lambda: select_mrmr(
            X, y, k=1, task="regression", estimator="gaussian", cache=cache, verbose=False
        ),
        lambda: select_jmi(
            X, y, k=1, task="regression", estimator="gaussian", cache=cache, verbose=False
        ),
        lambda: select_jmim(
            X, y, k=1, task="regression", estimator="gaussian", cache=cache, verbose=False
        ),
    ]
    for consume in consumers:
        with pytest.raises(ValueError, match="Duplicate feature names"):
            consume()


def test_gaussian_positional_cache_uses_feature_count_and_omitted_overrides():
    X, y = _data()
    X_arr = X.to_numpy()
    cache = build_cache(X_arr, subsample=None)

    result = select_cefsplus(X_arr, y, k=1, cache=cache, verbose=False)
    assert result[0].startswith("x")

    with pytest.raises(ValueError, match="X has 3 columns"):
        select_cefsplus(X_arr[:, :3], y, k=1, cache=cache, verbose=False)

    with pytest.raises(ValueError, match="unnamed/positional"):
        select_cefsplus(X, y, k=1, cache=cache, verbose=False)
    with pytest.raises(ValueError, match="unnamed/positional"):
        KnockoffSelector(cache=cache, verbose=False).fit(X, y)

    with pytest.raises(ValueError, match="subsample"):
        select_cefsplus(X_arr, y, k=1, cache=cache, subsample=None, verbose=False)
    with pytest.raises(ValueError, match="random_state"):
        select_cefsplus(X_arr, y, k=1, cache=cache, random_state=3, verbose=False)


@pytest.mark.parametrize(
    "selector_cls, kwargs",
    [
        (MRMRSelector, {"task": "regression", "estimator": "gaussian"}),
        (JMISelector, {"task": "regression", "estimator": "gaussian"}),
        (JMIMSelector, {"task": "regression", "estimator": "gaussian"}),
        (CEFSPlusSelector, {}),
    ],
)
def test_gaussian_selector_auto_defaults_preserve_cache_override_contract(
    selector_cls, kwargs
):
    X, y = _data()
    cache = build_cache(X, subsample=None)
    selector = selector_cls(k=1, cache=cache, verbose=False, **kwargs)
    cloned = clone(selector)

    assert selector.subsample == "auto"
    assert selector.random_state == "auto"
    assert cloned.subsample == "auto"
    assert cloned.random_state == "auto"
    cloned.fit(X, y)

    with pytest.raises(ValueError, match="subsample"):
        selector_cls(
            k=1, cache=cache, subsample=50_000, verbose=False, **kwargs
        ).fit(X, y)
    with pytest.raises(ValueError, match="random_state"):
        selector_cls(
            k=1, cache=cache, random_state=0, verbose=False, **kwargs
        ).fit(X, y)


def test_gaussian_selector_fit_time_auto_overrides_are_normalized():
    X, y = _data()
    cache = build_cache(X, subsample=None)

    MRMRSelector(
        k=1,
        task="regression",
        estimator="gaussian",
        cache=cache,
        verbose=False,
    ).fit(X, y, subsample="auto", random_state="auto")


def test_binary_selector_does_not_accept_gaussian_auto_subsample_token():
    X, y_continuous = _data()
    y = (y_continuous > np.median(y_continuous)).astype(np.int64)

    with pytest.raises(ValueError, match="subsample"):
        CEFSPlusBinarySelector(k=1, subsample="auto", verbose=False).fit(X, y)


@pytest.mark.parametrize("metadata", ["groups", "time"])
def test_fixed_k_rejects_auto_k_evaluation_metadata(metadata):
    X, y = _data()
    with pytest.raises(ValueError, match="only meaningful for auto-k"):
        select_cefsplus(X, y, k=1, **{metadata: np.arange(len(X))}, verbose=False)


@pytest.mark.parametrize(
    "selector_cls, kwargs",
    [
        (MRMRSelector, {"task": "regression", "estimator": "gaussian"}),
        (JMISelector, {"task": "regression", "estimator": "gaussian"}),
        (JMIMSelector, {"task": "regression", "estimator": "gaussian"}),
        (CEFSPlusSelector, {}),
    ],
)
def test_selector_get_feature_names_out_preserves_fitted_names(selector_cls, kwargs):
    X, y = _data()
    selector = selector_cls(k=2, verbose=False, **kwargs).fit(X, y)

    expected = np.asarray([X.columns[i] for i in selector.selected_indices_], dtype=object)
    np.testing.assert_array_equal(selector.get_feature_names_out(), expected)
    np.testing.assert_array_equal(selector.get_feature_names_out(list(X.columns)), expected)
    with pytest.raises(ValueError, match="input_features"):
        selector.get_feature_names_out(["wrong"] * X.shape[1])


def test_knockoff_selector_exposes_non_deterministic_tag_for_row_order_sensitivity():
    assert _non_deterministic_tag(KnockoffSelector(verbose=False)) is True


def test_knockoff_tag_does_not_mutate_other_selector_tags():
    # sklearn <1.6 exposes a shared default tag dict through _more_tags(); newer
    # releases expose a Tags object through get_tags(). Neither path may leak.
    assert _non_deterministic_tag(MRMRSelector(verbose=False)) is False
    assert _non_deterministic_tag(KnockoffSelector(verbose=False)) is True
    assert _non_deterministic_tag(MRMRSelector(verbose=False)) is False


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda R: np.full_like(R, np.nan),
        lambda R: np.zeros_like(R),
        lambda R: R + np.triu(np.ones_like(R), 1) * 0.2,
        lambda R: R.astype(np.complex128) + 1j,
    ],
    ids=["nonfinite", "nonunit-diagonal", "nonsymmetric", "complex"],
)
def test_cached_gaussian_selection_rejects_invalid_correlation_matrix(corrupt):
    X, y = _data()
    cache = build_cache(X, subsample=None, compute_Rxx=True)
    cache.Rxx = corrupt(cache.Rxx)

    with pytest.raises(ValueError, match="cache.Rxx"):
        select_cached(cache, y, k=2)
    with pytest.raises(ValueError, match="cache.Rxx"):
        select_cefsplus(X, y, k=2, cache=cache, verbose=False)


@pytest.mark.parametrize("missing_field", ["row_idx", "sample_weight"])
@pytest.mark.parametrize("entrypoint", ["cached", "fdr", "sample"])
def test_public_cache_consumers_report_missing_required_fields(
    missing_field, entrypoint
):
    X, y = _data()
    cache = build_cache(X, subsample=None, compute_Rxx=True)
    delattr(cache, missing_field)

    with pytest.raises(ValueError, match="missing required structural fields"):
        if entrypoint == "cached":
            select_cached(cache, y, k=2)
        elif entrypoint == "fdr":
            select_fdr(cache=cache, y=y, verbose=False)
        else:
            sample_knockoffs(cache)
