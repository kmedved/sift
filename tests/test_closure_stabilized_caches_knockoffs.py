"""Closure regressions for Stabilized resampling, feature caches, and knockoffs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.feature_selection import RFECV, SelectorMixin
from sklearn.linear_model import LinearRegression

from sift import (
    KnockoffSelector,
    Stabilized,
    as_result,
    bootstrap_paths,
    build_cache,
    build_classic_cache,
    compute_objective_for_path,
    null_objective_paths,
    sample_knockoffs,
    select_fdr,
    select_jmi,
    select_jmim,
    select_mrmr,
)
from sift.selection.knockoff_filter import e_bh_reject, e_bh_threshold


class WeightedCVSpy(SelectorMixin, BaseEstimator):
    """Inner-CV base that accepts weights and records what each fit received."""

    fits = []

    def __init__(self, cv=3):
        self.cv = cv

    def fit(self, X, y=None, sample_weight=None, groups=None, time=None):
        values = np.asarray(X, dtype=np.float64)
        type(self).fits.append(
            (
                values[:, 0].astype(np.int64).copy(),
                None if sample_weight is None else np.asarray(sample_weight).copy(),
            )
        )
        self.n_features_in_ = values.shape[1]
        return self

    def _get_support_mask(self):
        return np.ones(self.n_features_in_, dtype=bool)


def _panel(n=60, p=3, seed=0):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.arange(n, dtype=float), rng.normal(size=(n, p - 1))])
    y = X[:, 1] + rng.normal(scale=0.2, size=n)
    groups = np.repeat(np.arange(n // 5), 5)
    time = np.tile(np.arange(5), n // 5)
    return X, y, groups, time


# --------------------------------------------------------------------------
# Stabilized: replacement resampling with an inner-CV base
# --------------------------------------------------------------------------


@pytest.mark.parametrize("resample", ["bootstrap", "blocks"])
def test_inner_cv_rejection_names_the_mode_and_its_real_alternatives(resample):
    X, y, groups, time = _panel(n=40, p=4)
    base = RFECV(LinearRegression(), cv=3, min_features_to_select=1)
    row_context = {"groups": groups, "time": time} if resample == "blocks" else {}
    with pytest.raises(ValueError) as excinfo:
        Stabilized(
            base, n_resamples=2, resample=resample, random_state=0, verbose=False
        ).fit(X, y, **row_context)
    message = str(excinfo.value)
    # The message must describe the mode the caller actually asked for ...
    assert f"resample={resample!r}" in message
    assert ("blocks of rows" in message) is (resample == "blocks")
    # ... say why duplicated rows are refused ...
    assert "inner-CV" in message and "leak" in message
    # ... and offer both real routes, with half's caveat spelled out.
    assert "sample_weight" in message
    assert "resample='half'" in message
    assert "ignores groups and time" in message


def test_blocks_fit_weight_accepting_inner_cv_base_on_sorted_unique_rows():
    """Block draws are collapsed to weighted unique rows, like bootstrap draws."""
    n = 60
    X, y, groups, time = _panel(n=n, p=3)
    WeightedCVSpy.fits = []
    fitted = Stabilized(
        WeightedCVSpy(cv=3),
        n_resamples=3,
        resample="blocks",
        threshold=0.5,
        random_state=0,
        verbose=False,
    ).fit(X, y, groups=groups, time=time)
    assert fitted._resample_fit_policy_ == "unique_rows_with_multiplicity_weights"
    assert len(WeightedCVSpy.fits) == 3
    for rows, weights in WeightedCVSpy.fits:
        # Sorted, unique source rows...
        assert np.array_equal(rows, np.sort(rows))
        assert len(set(rows.tolist())) == len(rows)
        # ...strictly fewer than the 60 rows the block draw produced...
        assert 0 < len(rows) < n
        # ...and the multiplicities they carry add back up to that draw.
        assert weights is not None
        np.testing.assert_array_equal(weights, np.round(weights))
        assert weights.min() >= 1.0
        assert float(weights.sum()) == float(n)


def test_stabilized_diagnostics_report_synthesized_weights_and_fitted_rows():
    n = 60
    X, y, groups, time = _panel(n=n, p=3)
    WeightedCVSpy.fits = []
    fitted = Stabilized(
        WeightedCVSpy(cv=3),
        n_resamples=3,
        resample="blocks",
        threshold=0.5,
        random_state=0,
        verbose=False,
    ).fit(X, y, groups=groups, time=time)
    diagnostics = as_result(fitted).diagnostics
    fitted_rows = [len(rows) for rows, _ in WeightedCVSpy.fits]
    assert diagnostics["multiplicity_weights_synthesized"] is True
    # The caller supplied no weights; the wrapper synthesized them.
    assert diagnostics["fit_context"]["sample_weight"] is False
    assert diagnostics["resample_n_rows"] == [n, n, n]
    assert diagnostics["resample_n_rows_fitted"] == fitted_rows
    assert all(count < n for count in fitted_rows)

    # Without the unique-row policy the base is fitted on exactly the drawn rows.
    WeightedCVSpy.fits = []
    plain = Stabilized(
        WeightedCVSpy(cv=3),
        n_resamples=3,
        resample="half",
        threshold=0.5,
        random_state=0,
        verbose=False,
    ).fit(X, y)
    plain_diagnostics = as_result(plain).diagnostics
    assert plain_diagnostics["multiplicity_weights_synthesized"] is False
    assert (
        plain_diagnostics["resample_n_rows_fitted"]
        == plain_diagnostics["resample_n_rows"]
        == [len(rows) for rows, _ in WeightedCVSpy.fits]
    )
    assert all(weights is None for _, weights in WeightedCVSpy.fits)


def test_n_jobs_none_and_one_are_accepted_and_documented():
    X, y, _, _ = _panel(n=30, p=3)
    base = RFECV(LinearRegression(), cv=3, min_features_to_select=1)
    serial = Stabilized(
        base, n_resamples=2, n_jobs=1, random_state=0, verbose=False
    ).fit(X, y)
    none_jobs = Stabilized(
        base, n_resamples=2, n_jobs=None, random_state=0, verbose=False
    ).fit(X, y)
    np.testing.assert_array_equal(
        none_jobs.selection_frequencies_, serial.selection_frequencies_
    )
    with pytest.raises(ValueError, match="n_jobs must be 1"):
        Stabilized(base, n_resamples=2, n_jobs=2, verbose=False).fit(X, y)
    doc = Stabilized.__doc__
    n_jobs_entry = doc.split("n_jobs :", 1)[1].split("block_size :", 1)[0]
    assert "``None``" in n_jobs_entry and "``1``" in n_jobs_entry


def test_completed_resample_count_equals_the_request_because_failures_raise():
    class BoomOnThirdFit(SelectorMixin, BaseEstimator):
        fits = 0

        def fit(self, X, y=None):
            type(self).fits += 1
            if type(self).fits == 3:
                raise RuntimeError("resample 3 failed")
            values = np.asarray(X)
            self.n_features_in_ = values.shape[1]
            return self

        def _get_support_mask(self):
            return np.ones(self.n_features_in_, dtype=bool)

    X, y, _, _ = _panel(n=30, p=3)
    BoomOnThirdFit.fits = 0
    with pytest.raises(RuntimeError, match="resample 3 failed"):
        Stabilized(
            BoomOnThirdFit(), n_resamples=5, random_state=0, verbose=False
        ).fit(X, y)
    # Nothing partial survives: there is no fitted result to read a smaller
    # completed count from, which is why the reported count always equals the
    # request.
    BoomOnThirdFit.fits = 0
    view = as_result(
        Stabilized(
            BoomOnThirdFit(), n_resamples=2, random_state=0, verbose=False
        ).fit(X, y)
    )
    assert view.metadata["n_resamples_completed"] == view.metadata["n_resamples"] == 2
    assert view.diagnostics["n_completed_resamples"] == 2


# --------------------------------------------------------------------------
# Feature caches
# --------------------------------------------------------------------------


def _cache_frame(n=120, p=6, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"x{i}" for i in range(p)])
    y = pd.Series(1.5 * X["x0"] - X["x3"] + 0.5 * rng.normal(size=n))
    return X, y


@pytest.mark.parametrize("kind", ["classic", "gaussian"])
def test_row_permuted_x_with_a_prebuilt_cache_answers_for_the_cached_rows(kind):
    """The documented limit: rows are never checked, so this is silently wrong."""
    X, y = _cache_frame()
    rng = np.random.default_rng(1)
    permutation = rng.permutation(len(X))
    X_perm = X.iloc[permutation].reset_index(drop=True)
    y_perm = y.iloc[permutation].reset_index(drop=True)
    if kind == "classic":
        cache = build_classic_cache(X, subsample=None)
        kwargs = dict(k=3, task="regression", verbose=False)
        with_cache = select_mrmr(X_perm, y_perm, cache=cache, **kwargs)
        cached_rows_answer = select_mrmr(X, y_perm, cache=cache, **kwargs)
        honest_answer = select_mrmr(X_perm, y_perm, subsample=None, **kwargs)
    else:
        cache = build_cache(X, subsample=None, compute_Rxx=True)
        from sift import select_cefsplus

        kwargs = dict(k=3, verbose=False)
        with_cache = select_cefsplus(X_perm, y_perm, cache=cache, **kwargs)
        cached_rows_answer = select_cefsplus(X, y_perm, cache=cache, **kwargs)
        honest_answer = select_cefsplus(X_perm, y_perm, subsample=None, **kwargs)
    # No error is raised, and the answer is the one for the CACHED rows'
    # features paired with the new y -- not the answer for the matrix passed in.
    assert with_cache == cached_rows_answer
    assert with_cache != honest_answer


@pytest.mark.parametrize(
    "entry_point",
    [
        "sample_knockoffs",
        "select_fdr",
        "KnockoffSelector",
        "bootstrap_paths",
        "null_objective_paths",
        "compute_objective_for_path",
    ],
)
def test_classic_cache_on_gaussian_entry_points_names_both_cache_kinds(entry_point):
    X, y = _cache_frame(n=80, p=4)
    cache = build_classic_cache(X, subsample=None)
    calls = {
        "sample_knockoffs": lambda: sample_knockoffs(cache),
        "select_fdr": lambda: select_fdr(y=y, cache=cache, verbose=False),
        "KnockoffSelector": lambda: KnockoffSelector(
            cache=cache, random_state=0, verbose=False
        ).fit(X, y),
        "bootstrap_paths": lambda: bootstrap_paths(
            cache, y.to_numpy(), B=2, max_k=2, boot_mode="half", top_m=4,
            corr_prune=None, random_state=0,
        ),
        "null_objective_paths": lambda: null_objective_paths(
            cache, y.to_numpy(), B=2, max_k=2, null="permute", top_m=4,
            corr_prune=None, random_state=0,
        ),
        "compute_objective_for_path": lambda: compute_objective_for_path(
            cache, y.to_numpy(), ["x0", "x1"]
        ),
    }
    with pytest.raises(TypeError) as excinfo:
        calls[entry_point]()
    message = str(excinfo.value)
    assert "ClassicFeatureCache" in message
    assert "FeatureCache" in message
    assert "build_cache" in message
    assert "select_mrmr" in message


def test_gaussian_cache_still_reaches_the_same_entry_points():
    X, y = _cache_frame(n=80, p=4)
    cache = build_cache(X, subsample=None, compute_Rxx=True)
    assert sample_knockoffs(cache).shape == cache.Z.shape
    assert compute_objective_for_path(cache, y.to_numpy(), ["x0", "x1"]).shape == (2,)


@pytest.mark.parametrize("select_fn", [select_mrmr, select_jmi, select_jmim])
def test_classic_cache_weights_are_mean_one_only_up_to_rounding(select_fn):
    """The docstring promise is 'mean 1 up to rounding', and that is all."""
    rng = np.random.default_rng(0)
    n = 120
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    y = pd.Series(X["a"] + 0.3 * rng.normal(size=n))
    weights = np.abs(rng.normal(size=n)) + 0.1
    cache = build_classic_cache(X, sample_weight=weights, subsample=None)
    mean = float(cache.sample_weight.mean())
    assert mean == pytest.approx(1.0, abs=1e-12)
    # A run must not depend on the mean being bit-exactly 1.0.
    assert select_fn(X, y, k=2, task="regression", cache=cache, verbose=False)


# --------------------------------------------------------------------------
# Knockoffs
# --------------------------------------------------------------------------


def _clustered_frame(n=300, seed=4):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "a": z + 0.01 * rng.normal(size=n),
            "a_dup": z + 0.01 * rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.normal(size=n),
        }
    )
    y = pd.Series(2.0 * z + rng.normal(scale=0.3, size=n))
    return X, y


def _grouped_evalue_result(X, y):
    return select_fdr(
        X,
        y,
        q=0.5,
        n_draws=3,
        aggregation="evalues",
        feature_groups="auto",
        group_corr_threshold=0.9,
        random_state=4,
        verbose=False,
    )


def test_e_bh_helpers_explain_the_nan_evalues_of_a_grouped_result():
    X, y = _clustered_frame()
    result = _grouped_evalue_result(X, y)
    evalues = result.W["evalue"].to_numpy(dtype=float)
    representative = result.W["is_representative"].to_numpy(dtype=bool)
    assert np.isnan(evalues[~representative]).all()
    assert not np.isnan(evalues[representative]).any()

    for helper in (e_bh_threshold, e_bh_reject):
        with pytest.raises(ValueError) as excinfo:
            helper(evalues, 0.5)
        message = str(excinfo.value)
        assert "NaN" in message
        assert "is_representative" in message
        assert "evalue_universe" in message

    # The message's own recipe has to work: the tested universe is accepted
    # and reproduces the run's own rejections.
    tested = evalues[representative]
    universe = result.selector_metadata["evalue_universe"]
    assert len(universe) == int(representative.sum())
    rejected = e_bh_reject(tested, 0.5, m=len(tested))
    assert rejected.shape == tested.shape
    chosen = {
        str(name)
        for name, keep in zip(result.W["feature"][representative], rejected)
        if keep
    }
    assert chosen == {
        str(name)
        for name in result.W["feature"][representative & result.W["selected"]]
    }


def test_knockoff_view_table_carries_representative_evalue_beside_evalue():
    X, y = _clustered_frame()
    result = _grouped_evalue_result(X, y)
    table = as_result(result, input_features=list(X.columns)).table
    assert "evalue" in table.columns
    assert "representative_evalue" in table.columns
    members = ~result.W["is_representative"].to_numpy(dtype=bool)
    assert table.loc[members, "evalue"].isna().all()
    # The provenance column keeps the cluster representative's e-value, so the
    # NaN column is readable rather than a dead end.
    assert table.loc[members, "representative_evalue"].notna().all()
    np.testing.assert_allclose(
        table["representative_evalue"].to_numpy(dtype=float),
        result.W["representative_evalue"].to_numpy(dtype=float),
    )


def test_eta_does_not_change_the_evalue_selection_only_the_offset_zero_count():
    rng = np.random.default_rng(8)
    X = pd.DataFrame(rng.normal(size=(300, 12)), columns=[f"f{i}" for i in range(12)])
    y = pd.Series(
        2.0 * X["f0"] + X["f1"] + 0.5 * rng.normal(size=len(X))
    )
    selections = {}
    offset_zero = {}
    per_draw = {}
    for eta in (0.1, 0.5, 0.9):
        result = select_fdr(
            X, y, q=0.5, n_draws=3, eta=eta, aggregation="evalues",
            random_state=2, verbose=False,
        )
        selections[eta] = tuple(result.selected_features)
        offset_zero[eta] = result.selector_metadata["n_discoveries_offset_0"]
        per_draw[eta] = tuple(result.selector_metadata["n_discoveries_offset_0_per_draw"])
    # Documented contract: e-BH ignores eta, so the selection is identical ...
    assert len(set(selections.values())) == 1
    assert selections[0.5]
    # ... while the named offset-0 frequency-vote counterfactual may move, and
    # the per-draw counts it is built from may not.
    assert len(set(offset_zero.values())) > 1
    assert len(set(per_draw.values())) == 1


def test_selection_frequency_is_nan_for_a_single_draw_including_conditioned_features():
    rng = np.random.default_rng(3)
    n = 200
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"f{i}" for i in range(6)])
    y = pd.Series(2.0 * X["f0"] + 1.5 * X["f1"] + 0.4 * rng.normal(size=n))
    single = select_fdr(
        X, y, q=0.2, n_draws=1, include=["f5"], include_provenance="prespecified",
        random_state=0, verbose=False,
    )
    assert single.W["selection_frequency"].isna().all()
    assert bool(single.W.loc[single.W["feature"] == "f5", "selected"].iloc[0])
    assert single.selection_frequency is None

    multi = select_fdr(
        X, y, q=0.2, n_draws=3, aggregation="selection_frequency",
        include=["f5"], include_provenance="prespecified",
        random_state=0, verbose=False,
    )
    assert multi.W["selection_frequency"].notna().all()
    assert float(multi.W.loc[multi.W["feature"] == "f5", "selection_frequency"].iloc[0]) == 1.0
