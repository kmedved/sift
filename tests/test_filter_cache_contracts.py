import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from sift import (
    Stabilized,
    build_cache,
    build_classic_cache,
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
@pytest.mark.filterwarnings("ignore:Gaussian mRMR:UserWarning")
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


# --------------------------------------------------------------------------
# 1.0.1: one cache x cat_encoding rule at every entry point. A prebuilt cache
# stores no encoding provenance, so an encoding that would encode a column
# raises with one message; with no column to encode it is inert, exactly as
# it is without a cache.
# --------------------------------------------------------------------------

_CACHE_ENCODINGS = (
    "target_cv", "target", "loo", "james_stein", "loo_logit",
    "onehot", "ordinal", "frequency",
)


def _encoded_frames():
    rng = np.random.default_rng(23)
    n = 120
    numeric = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.normal(size=n),
            "cat": rng.integers(0, 3, size=n).astype(float),
        }
    )
    y = (
        numeric["a"].to_numpy()
        + 0.8 * numeric["b"].to_numpy()
        + 0.5 * numeric["cat"].to_numpy()
        + 0.3 * rng.normal(size=n)
    )
    raw = numeric.copy()
    raw["cat"] = np.array(["lo", "mid", "hi"])[numeric["cat"].astype(int)]
    return numeric, raw, y


def _cache_routes():
    """(id, cache kind, call(X, y, cache, **encoding kwargs) -> selected names)."""

    def fn(func, **fixed):
        return lambda X, y, cache, **kw: list(
            func(X, y, 2, cache=cache, verbose=False, **fixed, **kw)
        )

    def wrapper(cls, **fixed):
        def run(X, y, cache, **kw):
            est = cls(cache=cache, verbose=False, **fixed, **kw).fit(X, y)
            return list(est.selected_features_)

        return run

    def stabilized(X, y, cache, **kw):
        base = KnockoffSelector(cache=cache, q=0.5, verbose=False, **kw)
        est = Stabilized(base, aggregation="evalues", n_resamples=2).fit(X, y)
        return list(est.selected_features_)

    regression = {"task": "regression"}
    gaussian = {"task": "regression", "estimator": "gaussian"}
    return (
        ("select_mrmr-gaussian", "gaussian", fn(select_mrmr, **gaussian)),
        ("select_mrmr-classic", "classic", fn(select_mrmr, **regression)),
        ("select_jmi-gaussian", "gaussian", fn(select_jmi, **gaussian)),
        ("select_jmi-classic", "classic", fn(select_jmi, **regression)),
        ("select_jmim-gaussian", "gaussian", fn(select_jmim, **gaussian)),
        ("select_jmim-classic", "classic", fn(select_jmim, **regression)),
        ("select_cefsplus", "gaussian", fn(select_cefsplus)),
        ("MRMRSelector-gaussian", "gaussian", wrapper(MRMRSelector, k=2, **gaussian)),
        ("MRMRSelector-classic", "classic", wrapper(MRMRSelector, k=2, **regression)),
        ("JMISelector-gaussian", "gaussian", wrapper(JMISelector, k=2, **gaussian)),
        ("JMISelector-classic", "classic", wrapper(JMISelector, k=2, **regression)),
        ("JMIMSelector-gaussian", "gaussian", wrapper(JMIMSelector, k=2, **gaussian)),
        ("JMIMSelector-classic", "classic", wrapper(JMIMSelector, k=2, **regression)),
        ("CEFSPlusSelector", "gaussian", wrapper(CEFSPlusSelector, k=2)),
        ("KnockoffSelector", "gaussian", wrapper(KnockoffSelector, q=0.5)),
        ("Stabilized-KnockoffSelector", "gaussian", stabilized),
    )


_CACHE_ROUTES = _cache_routes()


def _route_encodings(route_id):
    # The knockoff filter refuses target_cv and one-hot with or without a
    # cache (no Model-X claim survives them); that rule is not about caches.
    if "Knockoff" in route_id:
        return tuple(e for e in _CACHE_ENCODINGS if e not in {"target_cv", "onehot"})
    return _CACHE_ENCODINGS


@pytest.mark.parametrize("route", _CACHE_ROUTES, ids=[r[0] for r in _CACHE_ROUTES])
def test_prebuilt_cache_rejects_an_encoding_it_would_have_to_apply(route):
    route_id, kind, run = route
    numeric, raw, y = _encoded_frames()
    cache = build_cache(numeric) if kind == "gaussian" else build_classic_cache(numeric)
    categorical = numeric.assign(cat=numeric["cat"].astype("category"))
    for encoding in _route_encodings(route_id):
        expected = (
            f"cat_encoding={encoding!r} cannot be combined with a prebuilt cache "
            "because the cache has no encoding provenance, so it cannot encode "
            "['cat']. Encode those columns before building the cache and pass "
            "cat_encoding='none', or omit the cache"
        )
        # A raw string column the encoding would pick up by itself ...
        with pytest.raises(ValueError) as caught:
            run(raw, y, cache, cat_encoding=encoding)
        assert str(caught.value) == expected, encoding
        # ... a (pre-encoded) numeric column named in cat_features ...
        with pytest.raises(ValueError) as caught:
            run(numeric, y, cache, cat_encoding=encoding, cat_features=["cat"])
        assert str(caught.value) == expected, encoding
        # ... and a category-dtype numeric column, picked up without cat_features.
        with pytest.raises(ValueError) as caught:
            run(categorical, y, cache, cat_encoding=encoding)
        assert str(caught.value) == expected, encoding


@pytest.mark.parametrize("route", _CACHE_ROUTES, ids=[r[0] for r in _CACHE_ROUTES])
def test_prebuilt_cache_encoding_with_nothing_to_encode_is_inert(route):
    route_id, kind, run = route
    numeric, _raw, y = _encoded_frames()
    cache = build_cache(numeric) if kind == "gaussian" else build_classic_cache(numeric)
    baseline = run(numeric, y, cache, cat_encoding="none")
    assert baseline
    for encoding in _route_encodings(route_id):
        assert run(numeric, y, cache, cat_encoding=encoding) == baseline, encoding


def _cache_rule_message(encoding, columns, hint=""):
    return (
        f"cat_encoding={encoding!r} cannot be combined with a prebuilt cache "
        "because the cache has no encoding provenance, so it cannot encode "
        f"{columns}. Encode those columns before building the cache and pass "
        f"cat_encoding='none', or omit the cache{hint}"
    )


@pytest.mark.parametrize("route", _CACHE_ROUTES, ids=[r[0] for r in _CACHE_ROUTES])
def test_prebuilt_cache_on_an_ndarray_counts_only_cat_features_naming_a_column(route):
    route_id, kind, run = route
    numeric, _raw, y = _encoded_frames()
    arr = numeric.to_numpy()
    cache = build_cache(arr) if kind == "gaussian" else build_classic_cache(arr)
    baseline = run(arr, y, cache, cat_encoding="none")
    assert baseline
    for encoding in _route_encodings(route_id):
        # An in-range position, its generated name, or an ndarray of positions
        # names a real column, so the cache would have to encode it.
        for cat_features, shown in (([3], "[3]"), (["x3"], "['x3']"), (np.array([2, 3]), "[2, 3]")):
            with pytest.raises(ValueError) as caught:
                run(arr, y, cache, cat_encoding=encoding, cat_features=cat_features)
            assert str(caught.value) == _cache_rule_message(encoding, shown), encoding
        # Entries that name no column leave the cache nothing to encode. One-hot
        # then meets its own ndarray rule, which holds with or without a cache.
        unresolved = [99, -1, "zzz", "x03", True]
        if encoding == "onehot":
            with pytest.raises(TypeError) as caught:
                run(arr, y, cache, cat_encoding=encoding, cat_features=unresolved)
            assert str(caught.value) == (
                "cat_encoding='onehot' with cat_features requires a pandas "
                "DataFrame; an ndarray has no categorical column metadata"
            )
            continue
        assert run(arr, y, cache, cat_encoding=encoding, cat_features=unresolved) == baseline


@pytest.mark.parametrize("route", _CACHE_ROUTES, ids=[r[0] for r in _CACHE_ROUTES])
def test_prebuilt_cache_rule_reads_a_str_cat_features_as_every_encoder_does(route):
    route_id, kind, run = route
    numeric, _raw, y = _encoded_frames()
    cache = build_cache(numeric) if kind == "gaussian" else build_classic_cache(numeric)
    hint = (
        ". cat_features='cat' is a str, so each character names a column; "
        "pass ['cat'] to name one column"
    )
    for encoding in _route_encodings(route_id):
        # "cat" is read as the columns "c", "a" and "t"; X has the first two.
        with pytest.raises(ValueError) as caught:
            run(numeric, y, cache, cat_encoding=encoding, cat_features="cat")
        assert str(caught.value) == _cache_rule_message(encoding, "['c', 'a']", hint)


@pytest.mark.parametrize("encoding", ("none",) + _CACHE_ENCODINGS)
def test_binary_selector_rejects_a_cache_before_the_encoding_rule(encoding):
    numeric, raw, y = _encoded_frames()
    y_binary = (y > np.median(y)).astype(int)
    cache = build_cache(numeric)
    selector = CEFSPlusBinarySelector(
        k=2, cat_features=["cat"], cat_encoding=encoding, verbose=False
    )
    for X in (raw, numeric):
        with pytest.raises(ValueError) as caught:
            selector.fit(X, y_binary, cache=cache)
        assert str(caught.value) == "CEFSPlusBinarySelector does not support prebuilt caches."


@pytest.mark.parametrize("encoding", ("none", "ordinal", "frequency", "target_cv"))
@pytest.mark.parametrize("selector_cls", (MRMRSelector, CEFSPlusSelector))
def test_nested_auto_k_rejects_a_cache_before_the_encoding_rule(selector_cls, encoding):
    from sift import AutoKConfig

    numeric, raw, y = _encoded_frames()
    groups = np.repeat(np.arange(12), 10)
    cache = build_cache(numeric)
    kwargs = {"task": "regression"} if selector_cls is MRMRSelector else {}
    selector = selector_cls(
        k="auto",
        auto_k_config=AutoKConfig(k_method="evaluate", strategy="group_cv", auto_k_mode="nested"),
        cat_features=["cat"],
        cat_encoding=encoding,
        verbose=False,
        **kwargs,
    )
    # Following the encoding rule's advice would only reach this rejection.
    for X in (raw, numeric):
        with pytest.raises(ValueError) as caught:
            selector.fit(X, y, groups=groups, cache=cache)
        assert str(caught.value) == "auto_k_mode='nested' does not support prebuilt caches"
