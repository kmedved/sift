import argparse
import math
import warnings

import numpy as np
import pandas as pd
import pytest

import benchmarks.auto_k_designs as auto_k_designs
import benchmarks.bench_auto_k as bench_auto_k
from benchmarks.auto_k_designs import DESIGNS, _d6_monotone_transform, score_support
from sift import build_cache, select_cefsplus, select_cefsplus_binary, select_mrmr
from sift.estimators.copula import gaussian_mi_from_corr
import sift.selection.auto_k as auto_k_module
import sift.selection.auto_k_knockoff as auto_k_knockoff
import sift.selection.auto_k_resample as auto_k_resample
import sift.selection.filter_auto_k as filter_auto_k
from sift.selection.auto_k import (
    AutoKConfig,
    select_k_penalized_objective,
    select_k_posterior,
    validate_auto_k_config,
)
import sift.selection.auto_k_xfit as auto_k_xfit
from sift.selection.auto_k_stop import (
    path_gain_pvalues,
    select_k_changepoint,
    select_k_chi2_stop,
    select_k_forward_stop,
)
from sift.selection.auto_k_knockoff import select_k_knockoff_path
from sift.selection.auto_k_resample import (
    bootstrap_paths,
    null_objective_paths,
    select_k_perm_gap,
    select_k_stability,
)
from sift.selection.auto_k_xfit import (
    gaussian_cv_curves,
    select_k_gaussian_cv,
    select_k_xfit_objective,
    xfit_objective_curves,
)
from sift.selection.cefsplus import cefsplus_loop_with_objective, select_cached
from sift.selection.panel import build_candidate_panel, local_standardize
from scipy.special import digamma
from scipy.stats import f, kstest


def test_filter_elbow_clamps_minimum_to_short_candidate_path():
    config = AutoKConfig(k_method="elbow", min_k=5, max_k=100)

    selected_k, diagnostics = filter_auto_k._select_elbow_count(
        np.array([1.0, 2.0, 2.5]),
        config,
        path_length=3,
    )
    empty_k, empty_diagnostics = filter_auto_k._select_elbow_count(
        np.array([], dtype=np.float64),
        config,
        path_length=0,
    )

    assert selected_k == 3
    assert len(diagnostics) == 3
    assert empty_k == 0
    assert empty_diagnostics.empty


def test_candidate_panel_matches_select_cached_cefsplus_path():
    rng = np.random.default_rng(123)
    X = pd.DataFrame(rng.normal(size=(120, 12)), columns=[f"x{i}" for i in range(12)])
    y = X["x0"].to_numpy() + 0.5 * X["x3"].to_numpy() + rng.normal(scale=0.1, size=120)
    cache = build_cache(X, subsample=None, compute_Rxx=True)

    selected, selected_indices, objective = select_cached(
        cache,
        y,
        5,
        method="cefsplus",
        top_m=10,
        return_indices=True,
        return_objective=True,
    )
    panel = build_candidate_panel(cache, y, 5, top_m=10, method="cefsplus")
    local_path, panel_objective = cefsplus_loop_with_objective(
        panel.R,
        panel.r,
        min(5, len(panel.cand)),
        panel.rel,
    )

    assert selected_indices == panel.original[local_path].astype(int).tolist()
    assert selected == [cache.feature_names[i] for i in selected_indices]
    np.testing.assert_allclose(objective, panel_objective)


def _call_direct_cache_auto_k_api(api, cache, y):
    if api == "objective":
        return auto_k_module.compute_objective_for_path(cache, y, ["x0"])
    if api == "null":
        return null_objective_paths(
            cache,
            y,
            B=1,
            max_k=2,
            null="permute",
            top_m=5,
            corr_prune="auto",
            random_state=0,
        )
    if api == "bootstrap":
        return bootstrap_paths(
            cache,
            y,
            B=1,
            max_k=2,
            boot_mode="bayes",
            top_m=5,
            corr_prune="auto",
            random_state=0,
        )
    if api == "xfit":
        return xfit_objective_curves(
            cache,
            y,
            config=AutoKConfig(
                k_method="xfit_objective",
                strategy="kfold",
                min_k=1,
                max_k=2,
            ),
            top_m=5,
            corr_prune="auto",
            method="cefsplus",
        )
    if api == "cv":
        return gaussian_cv_curves(
            cache,
            y,
            config=AutoKConfig(
                k_method="gaussian_cv",
                strategy="kfold",
                min_k=1,
                max_k=2,
            ),
            top_m=5,
            corr_prune="auto",
            method="cefsplus",
        )
    raise AssertionError(f"unknown API: {api}")


@pytest.mark.parametrize("api", ["objective", "null", "bootstrap", "xfit", "cv"])
def test_direct_cache_auto_k_apis_require_provenance(api):
    rng = np.random.default_rng(1201)
    X = rng.normal(size=(40, 5))
    y = X[:, 0] + rng.normal(scale=0.1, size=40)
    cache = build_cache(X, subsample=None, compute_Rxx=True)
    delattr(cache, "feature_names_are_synthetic")

    with pytest.raises(ValueError, match="feature_names_are_synthetic|rebuild"):
        _call_direct_cache_auto_k_api(api, cache, y)


@pytest.mark.parametrize("api", ["objective", "null", "bootstrap", "xfit", "cv"])
def test_direct_cache_auto_k_apis_reject_duplicate_feature_names(api):
    rng = np.random.default_rng(1204)
    X = pd.DataFrame(rng.normal(size=(40, 5)), columns=["x0", "x0", "x2", "x3", "x4"])
    y = X.iloc[:, 0].to_numpy() + rng.normal(scale=0.1, size=40)
    cache = build_cache(X, subsample=None, compute_Rxx=True)

    with pytest.raises(ValueError, match="Duplicate feature names"):
        _call_direct_cache_auto_k_api(api, cache, y)


@pytest.mark.parametrize("mutation", ["duplicate_valid_cols", "duplicate_row_idx", "oob_row_idx"])
@pytest.mark.parametrize("api", ["objective", "null", "bootstrap", "xfit", "cv"])
def test_direct_cache_auto_k_apis_reject_malformed_indices(api, mutation):
    rng = np.random.default_rng(1202)
    X = rng.normal(size=(40, 5))
    y = X[:, 0] + rng.normal(scale=0.1, size=40)
    cache = build_cache(X, subsample=None, compute_Rxx=True)
    if mutation == "duplicate_valid_cols":
        cache.valid_cols = np.array([0, 0, 1, 2, 3])
    elif mutation == "duplicate_row_idx":
        cache.row_idx = np.r_[0, 0, np.arange(2, 40)]
    else:
        cache.row_idx = np.r_[np.arange(39), 40]

    with pytest.raises(ValueError, match="valid_cols|row_idx"):
        _call_direct_cache_auto_k_api(api, cache, y)


def test_bootstrap_paths_reject_short_y_before_subsampled_row_indexing():
    rng = np.random.default_rng(1203)
    X = rng.normal(size=(100, 5))
    y = X[:, 0] + rng.normal(scale=0.1, size=100)
    cache = build_cache(X, subsample=20, compute_Rxx=True, random_state=0)
    assert 99 not in cache.row_idx

    with pytest.raises(ValueError, match="y has 99 rows.*100 rows"):
        bootstrap_paths(
            cache,
            y[:-1],
            B=1,
            max_k=2,
            boot_mode="bayes",
            top_m=5,
            corr_prune="auto",
            random_state=0,
        )


def test_local_standardize_uses_local_weights_and_neutralizes_constant_columns():
    Z = np.array(
        [
            [0.0, 1.0, 5.0],
            [1.0, 1.0, 5.0],
            [3.0, 1.0, 5.0],
        ]
    )
    w = np.array([1.0, 2.0, 3.0])

    out = local_standardize(Z, w)
    weighted_mean = w @ out / w.sum()
    weighted_var = w @ (out * out) / w.sum()

    np.testing.assert_allclose(weighted_mean, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(weighted_var[0], 1.0)
    np.testing.assert_allclose(out[:, 1:], 0.0)


def test_local_corr_panel_matches_explicit_full_standardization():
    from sift.estimators.copula import (
        gaussian_mi_from_corr,
        weighted_corr_with_vector,
        weighted_correlation_matrix,
    )
    from sift.selection.panel import local_corr_panel

    rng = np.random.default_rng(901)
    Z = rng.normal(size=(80, 12)).astype(np.float32)
    zy = rng.normal(size=80)
    w = rng.uniform(0.2, 2.0, size=80)
    Z_full = local_standardize(Z, w)
    zy_full = local_standardize(zy, w).ravel()
    r_full = np.asarray(weighted_corr_with_vector(Z_full, zy_full, w), dtype=float)
    cand = np.argpartition(np.abs(r_full), -7)[-7:]
    R_full = weighted_correlation_matrix(Z_full[:, cand], w, backend="blas")

    panel = local_corr_panel(
        Z,
        zy,
        w,
        top_m=7,
        corr_prune=None,
        method="cefsplus",
        local_standardize=True,
    )

    assert set(panel.cand.tolist()) == set(cand.tolist())
    order = np.array([int(np.where(cand == value)[0][0]) for value in panel.cand])
    np.testing.assert_allclose(panel.r, r_full[panel.cand], rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(
        panel.R, R_full[np.ix_(order, order)], rtol=2e-6, atol=2e-7
    )
    np.testing.assert_allclose(panel.rel, gaussian_mi_from_corr(panel.r), rtol=1e-7)


def test_local_corr_panel_is_stable_for_large_offset_inputs():
    from sift.estimators.copula import weighted_corr_with_vector
    from sift.selection.panel import local_corr_panel

    rng = np.random.default_rng(902)
    Z = 1e12 + rng.normal(scale=2.0, size=(120, 9))
    zy = -1e11 + rng.normal(scale=3.0, size=120)
    w = rng.uniform(0.2, 2.0, size=120)
    Z_full = local_standardize(Z, w)
    zy_full = local_standardize(zy, w).ravel()
    expected_r = np.asarray(
        weighted_corr_with_vector(Z_full, zy_full, w), dtype=np.float64
    )

    panel = local_corr_panel(
        Z,
        zy,
        w,
        top_m=Z.shape[1],
        corr_prune=None,
        method="cefsplus",
        local_standardize=True,
    )

    assert np.isfinite(panel.r).all()
    np.testing.assert_allclose(panel.r, expected_r[panel.cand], rtol=1e-6, atol=1e-7)


def test_auto_k_designs_and_block_support_scoring_are_importable():
    X, y, meta = DESIGNS["D3"].make(0, False)
    assert X.shape[0] == y.shape[0]
    assert meta["support_type"] == "blocks"

    precision, recall, f1 = score_support([0, 1, 5, 42], meta)
    assert precision == 2 / 3
    assert recall == 2 / 8
    assert 0.0 < f1 < 1.0


def test_d6_heavy_tail_transforms_are_monotone():
    x = np.linspace(-5.0, 5.0, 101)
    for j in range(10):
        transformed = _d6_monotone_transform(x, j)
        assert np.all(np.diff(transformed) > 0.0)


def test_auto_k_harness_part2_regressions(monkeypatch):
    _n, _p, beta = auto_k_designs._d4_params(False)
    assert beta[:40].tolist() == pytest.approx([0.053] * 40)
    assert np.count_nonzero(beta) == 40

    args = argparse.Namespace(quick=True, full=False, seeds=30, n_test=20_000)
    bench_auto_k._normalize_args(args)
    assert args.seeds == 1
    assert args.n_test == 1_000

    with pytest.raises(ValueError, match="mutually exclusive"):
        bench_auto_k._normalize_args(
            argparse.Namespace(quick=True, full=True, seeds=30, n_test=20_000)
        )

    try:
        model = bench_auto_k._risk_model("catboost", seed=0)
    except RuntimeError as exc:
        assert "--model catboost requires" in str(exc)
    else:
        assert hasattr(model, "fit")

    X = pd.DataFrame(np.arange(12, dtype=np.float64).reshape(4, 3), columns=["x0", "x1", "x2"])
    y = np.arange(4, dtype=np.float64)

    class DummyDesign:
        def make(self, seed, full):
            return X, y, {"true_support": [0, 1], "k_star": 2}

        def sample_test(self, seed, n_test, full):
            return X, y

    class DummyCache:
        sample_weight = np.ones(4, dtype=np.float64)
        valid_cols = np.arange(3, dtype=np.int64)

    exact_calls = {}
    monkeypatch.setattr(bench_auto_k, "DESIGNS", {"DX": DummyDesign()})
    monkeypatch.setattr(bench_auto_k, "build_cache", lambda *_args, **_kwargs: DummyCache())
    monkeypatch.setattr(
        bench_auto_k,
        "select_cached",
        lambda *_args, **_kwargs: (
            ["x0", "x1", "x2"],
            [0, 1, 2],
            np.array([0.1, 0.2, 0.3]),
        ),
    )
    monkeypatch.setattr(bench_auto_k, "_path_max_k", lambda p, k_star: 3)
    monkeypatch.setattr(bench_auto_k, "_risk_grid", lambda max_k: [0, 1])
    monkeypatch.setattr(
        bench_auto_k,
        "_fit_rmse_curve",
        lambda *_args, **_kwargs: {0: 3.0, 1: 1.0},
    )
    method_runtimes = iter([99.0, 3.0, 1.0, 2.0])
    monkeypatch.setattr(
        bench_auto_k,
        "_method_k",
        lambda method, **_kwargs: (2, "", next(method_runtimes), None),
    )

    def fake_exact(_X, _y, _X_test, _y_test, selected_indices, **_kwargs):
        exact_calls["selected_indices"] = selected_indices
        return 0.5

    monkeypatch.setattr(bench_auto_k, "_fit_rmse_for_indices", fake_exact)

    rows = bench_auto_k.run(
        argparse.Namespace(
            designs="DX",
            methods="dummy",
            seeds=1,
            full=False,
            n_test=4,
            model="ridge",
        )
    )

    assert exact_calls["selected_indices"] == [0, 1]
    assert rows[0]["k_dispersion_group"] == "DX:dummy"
    assert rows[0]["runtime_s"] == 2.0
    assert "exact_off_grid_k" in rows[0]["notes"]


def test_benchmark_gaussian_cv_best_group_cv_spelling(monkeypatch):
    class DummyCache:
        sample_weight = np.ones(12, dtype=np.float64)
        valid_cols = np.arange(6, dtype=np.int64)

    captured = {}

    def fake_curves(*_args, config, groups=None, **_kwargs):
        captured["selection_rule"] = config.selection_rule
        captured["strategy"] = config.strategy
        captured["groups"] = groups
        return pd.DataFrame({"k": [3], "score_mean": [0.5], "score_se": [0.0]})

    monkeypatch.setattr(bench_auto_k, "gaussian_cv_curves", fake_curves)
    monkeypatch.setattr(bench_auto_k, "select_k_gaussian_cv", lambda *_args, **_kwargs: (3, pd.DataFrame()))

    k_hat, notes, _runtime, selected_override = bench_auto_k._method_k(
        "gaussian_cv/best/group_cv",
        X=pd.DataFrame(np.zeros((12, 6))),
        y=np.zeros(12),
        path_names=[f"x{i}" for i in range(6)],
        objective=np.linspace(0.1, 0.6, 6),
        cache=DummyCache(),
        meta={"groups": np.repeat(np.arange(4), 3)},
        max_k=6,
        seed=0,
    )

    assert k_hat == 3
    assert notes == ""
    assert selected_override is None
    assert captured["selection_rule"] == "best"
    assert captured["strategy"] == "group_cv"
    np.testing.assert_array_equal(captured["groups"], np.repeat(np.arange(4), 3))

    with pytest.raises(ValueError, match="Unsupported benchmark method"):
        bench_auto_k._method_k(
            "gaussian_cv_bad",
            X=pd.DataFrame(np.zeros((12, 6))),
            y=np.zeros(12),
            path_names=[f"x{i}" for i in range(6)],
            objective=np.linspace(0.1, 0.6, 6),
            cache=DummyCache(),
            meta={},
            max_k=6,
            seed=0,
        )


@pytest.mark.slow
def test_d10_dense_design_exposes_grouped_production_scale_metadata():
    X, y, meta = DESIGNS["D10"].make(0, False)

    assert X.shape == (12_000, 350)
    assert y.shape == (12_000,)
    assert len(meta["groups"]) == 12_000
    assert len(meta["true_support"]) == 120
    assert meta["benchmark_max_k"] == 250
    assert bench_auto_k._design_max_k(X.shape[1], meta) == 250


def test_ebic_penalty_uses_kish_by_default_and_requires_candidates():
    objective = np.array([0.08, 0.12, 0.13])
    weights = np.array([10.0, 1.0, 1.0, 1.0])
    bic_cfg = AutoKConfig(k_method="penalized_objective", objective_penalty="bic", min_k=1, max_k=3)
    ebic_cfg = AutoKConfig(k_method="penalized_objective", objective_penalty="ebic", min_k=0, max_k=3)

    _bic_k, bic_diag = select_k_penalized_objective(
        objective,
        bic_cfg,
        objective_scale="n_eff",
        n_samples=len(weights),
        sample_weight=weights,
    )
    assert bic_diag["n_eff_source"].iloc[0] == "selector_weight_sum"

    with pytest.raises(ValueError, match="n_candidates"):
        select_k_penalized_objective(
            objective,
            ebic_cfg,
            objective_scale="n_eff",
            n_samples=len(weights),
            sample_weight=weights,
        )

    _ebic_k, ebic_diag = select_k_penalized_objective(
        objective,
        ebic_cfg,
        objective_scale="n_eff",
        n_samples=len(weights),
        sample_weight=weights,
        n_candidates=10,
    )
    assert ebic_diag["k"].tolist()[0] == 0
    assert ebic_diag["n_eff_source"].iloc[0] == "kish"
    assert ebic_diag["penalty_kind"].iloc[0] == "ebic"
    assert ebic_diag["n_candidates"].iloc[0] == 10
    assert np.isfinite(ebic_diag["ebic_gamma"].iloc[0])


def test_ebic_ric_arithmetic_and_gamma_monotonicity():
    ks = np.arange(0, 6, dtype=np.int64)
    expected = np.array([math.log(math.comb(10, int(k))) for k in ks])
    np.testing.assert_allclose(auto_k_module._log_comb(10, ks), expected)

    auto_cfg = AutoKConfig(
        k_method="penalized_objective",
        objective_penalty="ebic",
        ebic_gamma="auto",
    )
    gamma = auto_k_module._resolve_ebic_gamma(auto_cfg, n_eff=50.0, n_candidates=1000)
    assert gamma == pytest.approx(1.0 - np.log(50.0) / (2.0 * np.log(1000.0)))
    assert auto_k_module._resolve_ebic_gamma(auto_cfg, n_eff=1_000_000.0, n_candidates=10) == 0.0

    objective = np.cumsum(np.array([0.20, 0.16, 0.10, 0.05, 0.025, 0.018, 0.012, 0.009]))
    selected_by_gamma = []
    for gamma_value in (0.0, 0.5, 1.0):
        cfg = AutoKConfig(
            k_method="penalized_objective",
            objective_penalty="ebic",
            ebic_gamma=gamma_value,
            min_k=0,
            max_k=len(objective),
        )
        k_hat, _diag = select_k_penalized_objective(
            objective,
            cfg,
            objective_scale="n_eff",
            n_samples=50,
            sample_weight=np.ones(50),
            n_candidates=100,
            max_k=len(objective),
        )
        selected_by_gamma.append(k_hat)
    assert selected_by_gamma == sorted(selected_by_gamma, reverse=True)

    ric_cfg = AutoKConfig(
        k_method="penalized_objective",
        objective_penalty="ric",
        min_k=0,
        max_k=len(objective),
    )
    ric_k, ric_diag = select_k_penalized_objective(
        objective,
        ric_cfg,
        objective_scale="n_eff",
        n_samples=50,
        sample_weight=np.ones(50),
        n_candidates=100,
        max_k=len(objective),
    )
    assert ric_diag["penalty_kind"].iloc[0] == "ric"
    assert ric_diag["penalty_weight"].iloc[0] == pytest.approx(2.0 * np.log(100.0))
    assert ric_k <= selected_by_gamma[0]


def test_k_posterior_map_matches_ebic_argmax_and_reports_hpd():
    objective = np.array([0.20, 0.35, 0.37, 0.371])
    weights = np.ones(80)
    ebic_cfg = AutoKConfig(k_method="penalized_objective", objective_penalty="ebic", min_k=0, max_k=4)
    post_cfg = AutoKConfig(k_method="k_posterior", min_k=0, max_k=4, posterior_level=0.9)

    ebic_k, _ebic_diag = select_k_penalized_objective(
        objective,
        ebic_cfg,
        objective_scale="n_eff",
        n_samples=len(weights),
        sample_weight=weights,
        n_candidates=20,
    )
    post_k, post_diag = select_k_posterior(
        objective,
        post_cfg,
        objective_scale="n_eff",
        n_samples=len(weights),
        sample_weight=weights,
        n_candidates=20,
    )

    assert post_k == ebic_k
    np.testing.assert_allclose(post_diag["post"].sum(), 1.0)
    assert post_diag["selected"].sum() == 1
    assert post_diag["in_hpd"].any()
    assert 0.0 <= post_diag["p_zero"].iloc[0] <= 1.0

    floor_cfg = AutoKConfig(k_method="k_posterior", min_k=2, max_k=4)
    floor_k, floor_diag = select_k_posterior(
        objective,
        floor_cfg,
        objective_scale="n_eff",
        n_samples=len(weights),
        sample_weight=weights,
        n_candidates=20,
    )
    assert floor_k >= 2
    assert floor_diag["k"].tolist()[0] == 0
    assert 1 not in floor_diag["k"].tolist()
    assert floor_diag["effective_min_k"].iloc[0] == 2
    assert not bool(floor_diag.loc[floor_diag["k"] == 0, "selected"].iloc[0])
    assert 0.0 <= floor_diag["p_zero"].iloc[0] <= 1.0

    hpd_floor_cfg = AutoKConfig(
        k_method="k_posterior",
        min_k=2,
        max_k=4,
        posterior_pick="smallest_in_hpd",
    )
    hpd_floor_k, hpd_floor_diag = select_k_posterior(
        np.zeros(4),
        hpd_floor_cfg,
        objective_scale="n_eff",
        n_samples=len(weights),
        sample_weight=weights,
        n_candidates=20,
    )
    assert hpd_floor_k >= 2
    assert not bool(hpd_floor_diag.loc[hpd_floor_diag["k"] == 0, "in_hpd"].iloc[0])


def test_cefsplus_public_dispatch_supports_ebic_and_k_posterior():
    rng = np.random.default_rng(321)
    X = pd.DataFrame(rng.normal(size=(160, 10)), columns=[f"x{i}" for i in range(10)])
    y = 1.5 * X["x0"].to_numpy() + rng.normal(scale=0.2, size=160)

    ebic_cfg = AutoKConfig(
        k_method="penalized_objective",
        objective_penalty="ebic",
        min_k=0,
        max_k=5,
    )
    ebic_result = select_cefsplus(X, y, k="auto", auto_k_config=ebic_cfg, return_result=True, verbose=False)
    assert ebic_result.diagnostics_["auto_k"]["objective_penalty"] == "ebic"
    assert "n_candidates" in ebic_result.diagnostics_["auto_k_diagnostics"]

    posterior_cfg = AutoKConfig(k_method="k_posterior", min_k=0, max_k=5)
    posterior_result = select_cefsplus(
        X,
        y,
        k="auto",
        auto_k_config=posterior_cfg,
        return_result=True,
        verbose=False,
    )
    assert posterior_result.diagnostics_["auto_k"]["method"] == "k_posterior"
    assert "post" in posterior_result.diagnostics_["auto_k_diagnostics"]


def test_path_gain_pvalues_match_f_sidak_formula():
    gain = 0.05
    n_eff = 100.0
    p_candidates = 10
    objective = np.array([gain])

    pvals = path_gain_pvalues(objective, n_eff=n_eff, p_candidates=p_candidates)
    nu = n_eff - 2.0
    F_stat = nu * np.expm1(gain)
    p_single = f.sf(F_stat, 1.0, nu)
    expected = 1.0 - (1.0 - p_single) ** p_candidates

    np.testing.assert_allclose(pvals[0], expected)


def test_path_gain_pvalues_are_uniform_under_exact_single_null():
    rng = np.random.default_rng(123)
    pvals = []
    for _ in range(200):
        Z = rng.normal(size=(500, 1))
        zy = rng.normal(size=500)
        Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)
        zy = (zy - zy.mean()) / zy.std()
        r = np.array([float(np.mean(Z[:, 0] * zy))])
        _path, objective = cefsplus_loop_with_objective(
            np.eye(1),
            r,
            1,
            gaussian_mi_from_corr(r),
            shrink=0.0,
        )
        pvals.append(path_gain_pvalues(objective, n_eff=500.0, p_candidates=1)[0])

    ks = kstest(np.asarray(pvals), "uniform")
    assert ks.pvalue > 0.01
    assert np.mean(pvals) == pytest.approx(0.5, abs=0.08)


def test_path_gain_pvalues_match_residual_partial_correlation_path():
    rng = np.random.default_rng(44)
    n = 250
    Z = rng.normal(size=(n, 4))
    Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)
    zy = 0.8 * Z[:, 2] + 0.4 * Z[:, 0] + rng.normal(size=n)
    zy = (zy - zy.mean()) / zy.std()
    R = np.corrcoef(Z, rowvar=False)
    r = Z.T @ zy / n
    path, objective = cefsplus_loop_with_objective(
        R,
        r,
        3,
        gaussian_mi_from_corr(r),
        shrink=0.0,
    )

    gains = np.diff(np.concatenate(([0.0], objective)))
    expected_p = []
    expected_gain = []
    prev: list[int] = []
    for step, feature_idx in enumerate(path, start=1):
        y_resid = zy.copy()
        x_resid = Z[:, int(feature_idx)].copy()
        if prev:
            X_prev = Z[:, prev]
            y_resid = zy - X_prev @ np.linalg.lstsq(X_prev, zy, rcond=None)[0]
            x_resid = Z[:, int(feature_idx)] - X_prev @ np.linalg.lstsq(
                X_prev,
                Z[:, int(feature_idx)],
                rcond=None,
            )[0]
        rho = float(np.corrcoef(x_resid, y_resid)[0, 1])
        gain = -np.log1p(-(rho * rho))
        nu = n - step - 1.0
        p_single = f.sf(nu * np.expm1(gain), 1.0, nu)
        m_eff = 4 - step + 1.0
        expected_gain.append(gain)
        expected_p.append(-np.expm1(m_eff * np.log1p(-p_single)))
        prev.append(int(feature_idx))

    np.testing.assert_allclose(gains, expected_gain, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        path_gain_pvalues(objective, n_eff=float(n), p_candidates=4),
        expected_p,
        rtol=1e-10,
        atol=1e-12,
    )


def test_zero_gain_sidak_path_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        pvals = path_gain_pvalues(
            np.zeros(3, dtype=np.float64),
            n_eff=80.0,
            p_candidates=20,
        )

    np.testing.assert_allclose(pvals, np.ones(3))


def test_chi2_stop_forward_stop_and_changepoint_rules():
    objective = np.cumsum(np.array([0.20, 0.15, 0.001, 0.001, 0.001]))
    chi_cfg = AutoKConfig(k_method="chi2_stop", min_k=0, max_k=5, alpha=0.05, stop_patience=2)
    chi_k, chi_diag = select_k_chi2_stop(
        objective,
        chi_cfg,
        n_eff=100.0,
        p_candidates=20,
    )
    assert chi_k == 2
    assert chi_diag["stopped_by"].iloc[0] == "test"
    clamp_cfg = AutoKConfig(k_method="chi2_stop", min_k=0, max_k=3, alpha=0.05)
    _clamp_k, clamp_diag = select_k_chi2_stop(
        np.cumsum(np.ones(8) * 0.2),
        clamp_cfg,
        n_eff=100.0,
        p_candidates=20,
    )
    assert clamp_diag["k"].max() == 3

    floor_cfg = AutoKConfig(
        k_method="chi2_stop",
        min_k=3,
        max_k=5,
        alpha=0.05,
        stop_patience=1,
    )
    floor_k, floor_diag = select_k_chi2_stop(
        np.zeros(5),
        floor_cfg,
        n_eff=100.0,
        p_candidates=20,
    )
    assert floor_k == 3
    assert floor_diag["stopped_by"].iloc[0] == "floored"

    low_cfg = AutoKConfig(k_method="forward_stop", min_k=0, max_k=5, alpha=0.05)
    high_cfg = AutoKConfig(k_method="forward_stop", min_k=0, max_k=5, alpha=0.20)
    low_k, _ = select_k_forward_stop(objective, low_cfg, n_eff=100.0, p_candidates=20)
    high_k, forward_diag = select_k_forward_stop(objective, high_cfg, n_eff=100.0, p_candidates=20)
    assert high_k >= low_k
    assert {"Y", "Y_running_mean", "eligible"} <= set(forward_diag.columns)

    gains = np.array([0.50] * 10 + [0.01] * 30)
    cp_cfg = AutoKConfig(k_method="changepoint", min_k=0, max_k=len(gains), floor_window=10)
    cp_k, cp_diag = select_k_changepoint(
        np.cumsum(gains),
        cp_cfg,
        objective_scale=100.0,
        n_eff=200.0,
        p_candidates=100,
    )
    assert cp_k == 10
    assert not bool(cp_diag["floor_not_reached"].iloc[0])


def test_forward_stop_gap_and_changepoint_hand_built_rules():
    n_eff = 120.0
    target_p = np.array([0.01, 0.20, 0.80], dtype=np.float64)
    gains = []
    for step, p_value in enumerate(target_p, start=1):
        nu = n_eff - step - 1.0
        gains.append(float(np.log1p(f.isf(p_value, 1.0, nu) / nu)))
    objective = np.cumsum(gains)
    low_k, low_diag = select_k_forward_stop(
        objective,
        AutoKConfig(k_method="forward_stop", min_k=0, max_k=3, alpha=0.10),
        n_eff=n_eff,
        p_candidates=1,
    )
    high_k, high_diag = select_k_forward_stop(
        objective,
        AutoKConfig(k_method="forward_stop", min_k=0, max_k=3, alpha=0.20),
        n_eff=n_eff,
        p_candidates=1,
    )
    assert low_k == 1
    assert high_k == 2
    np.testing.assert_allclose(low_diag["p_max"], target_p)
    np.testing.assert_allclose(high_diag["p_max"], target_p)

    gap_cfg = AutoKConfig(k_method="perm_gap", min_k=0, max_k=3, gap_rule="tibshirani")
    gap_k, gap_diag = select_k_perm_gap(
        np.array([4.0, 4.0, 4.0]),
        np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        gap_cfg,
    )
    assert gap_k == 1
    assert gap_diag.loc[gap_diag["selected"], "k"].tolist() == [1]

    with pytest.warns(UserWarning, match="noise floor was not reached"):
        cp_k, cp_diag = select_k_changepoint(
            np.cumsum(np.full(40, 0.1)),
            AutoKConfig(k_method="changepoint", min_k=0, max_k=40, floor_window=10),
            objective_scale=100.0,
            n_eff=200.0,
            p_candidates=100,
        )
    assert cp_k == 40
    assert bool(cp_diag["floor_not_reached"].iloc[0])


def test_cefsplus_public_dispatch_supports_phase2_path_methods():
    rng = np.random.default_rng(654)
    X = pd.DataFrame(rng.normal(size=(180, 12)), columns=[f"x{i}" for i in range(12)])
    y = 1.3 * X["x0"].to_numpy() + 0.9 * X["x1"].to_numpy() + rng.normal(scale=0.3, size=180)

    for method in ("chi2_stop", "forward_stop", "changepoint"):
        cfg = AutoKConfig(k_method=method, min_k=0, max_k=8)
        result = select_cefsplus(X, y, k="auto", auto_k_config=cfg, return_result=True, verbose=False)
        assert result.diagnostics_["auto_k"]["method"] == method
        assert not result.diagnostics_["auto_k_diagnostics"].empty

    weights = np.ones(len(y))
    weights[len(y) // 2 :] = 0.1
    cfg = AutoKConfig(k_method="chi2_stop", min_k=0, max_k=8)
    weighted = select_cefsplus(
        X,
        y,
        k="auto",
        auto_k_config=cfg,
        sample_weight=weights,
        return_result=True,
        verbose=False,
    )
    assert weighted.diagnostics_["auto_k"]["n_eff_source"] == "kish"


@pytest.mark.slow
def test_auto_k_null_calibration_and_signal_recovery_sims():
    null_results = {"chi2": [], "perm": [], "knockoff": [], "ebic": []}
    for seed in range(5):
        rng = np.random.default_rng(seed)
        X = pd.DataFrame(rng.normal(size=(500, 50)))
        y = rng.normal(size=500)
        cache = build_cache(X, subsample=None, compute_Rxx=True)
        _names, _indices, objective = select_cached(
            cache,
            y,
            20,
            method="cefsplus",
            top_m=50,
            return_indices=True,
            return_objective=True,
        )
        chi_k, _ = select_k_chi2_stop(
            objective,
            AutoKConfig(k_method="chi2_stop", min_k=0, max_k=20, alpha=0.05),
            n_eff=500.0,
            p_candidates=50,
        )
        ebic_k, _ = select_k_penalized_objective(
            objective,
            AutoKConfig(
                k_method="penalized_objective",
                objective_penalty="ebic",
                min_k=0,
                max_k=20,
            ),
            objective_scale="n_eff",
            n_samples=500,
            sample_weight=cache.sample_weight,
            n_candidates=50,
            max_k=20,
        )
        nulls = null_objective_paths(
            cache,
            y,
            B=10,
            max_k=20,
            null="permute",
            top_m=50,
            corr_prune="auto",
            random_state=seed,
        )
        perm_k, _ = select_k_perm_gap(
            objective,
            nulls,
            AutoKConfig(k_method="perm_gap", min_k=0, max_k=20, perm_B=10),
        )
        _selected, knock_k, _ = select_k_knockoff_path(
            cache,
            y,
            AutoKConfig(
                k_method="knockoff_path",
                min_k=0,
                max_k=20,
                knockoff_q=0.2,
                random_state=seed,
            ),
            top_m=50,
        )
        null_results["chi2"].append(chi_k)
        null_results["ebic"].append(ebic_k)
        null_results["perm"].append(perm_k)
        null_results["knockoff"].append(knock_k)

    assert max(null_results["chi2"]) <= 2
    assert max(null_results["ebic"]) <= 2
    assert sum(k <= 1 for k in null_results["perm"]) >= 4
    assert sum(k == 0 for k in null_results["knockoff"]) >= 4

    for seed in range(3):
        rng = np.random.default_rng(seed)
        X = pd.DataFrame(rng.normal(size=(1000, 50)))
        beta = np.zeros(50)
        beta[:8] = np.linspace(1.8, 0.8, 8)
        y = X.to_numpy() @ beta + rng.normal(size=1000)
        cache = build_cache(X, subsample=None, compute_Rxx=True)
        _names, _indices, objective = select_cached(
            cache,
            y,
            20,
            method="cefsplus",
            top_m=50,
            return_indices=True,
            return_objective=True,
        )
        chi_k, _ = select_k_chi2_stop(
            objective,
            AutoKConfig(k_method="chi2_stop", min_k=0, max_k=20),
            n_eff=1000.0,
            p_candidates=50,
        )
        ebic_k, _ = select_k_penalized_objective(
            objective,
            AutoKConfig(
                k_method="penalized_objective",
                objective_penalty="ebic",
                min_k=0,
                max_k=20,
            ),
            objective_scale="n_eff",
            n_samples=1000,
            sample_weight=cache.sample_weight,
            n_candidates=50,
            max_k=20,
        )
        cv_cfg = AutoKConfig(
            k_method="gaussian_cv",
            strategy="kfold",
            selection_rule="one_se",
            min_k=1,
            max_k=20,
            xfit_folds=3,
            random_state=seed,
        )
        cv_curves = gaussian_cv_curves(
            cache,
            y,
            config=cv_cfg,
            top_m=50,
            corr_prune="auto",
            method="cefsplus",
        )
        cv_k, _ = select_k_gaussian_cv(cv_curves, cv_cfg)
        assert 6 <= chi_k <= 10
        assert 6 <= ebic_k <= 10
        assert 6 <= cv_k <= 10


def test_perm_gap_null_paths_are_deterministic_and_rule_selects_from_gap():
    rng = np.random.default_rng(777)
    X = pd.DataFrame(rng.normal(size=(90, 8)), columns=[f"x{i}" for i in range(8)])
    y = X["x0"].to_numpy() + rng.normal(scale=0.5, size=90)
    cache = build_cache(X, subsample=None, compute_Rxx=True)

    null_1 = null_objective_paths(
        cache,
        y,
        B=3,
        max_k=4,
        null="permute",
        top_m=8,
        corr_prune="auto",
        random_state=123,
    )
    null_2 = null_objective_paths(
        cache,
        y,
        B=3,
        max_k=4,
        null="permute",
        top_m=8,
        corr_prune="auto",
        random_state=123,
    )
    assert null_1.shape == (3, 4)
    np.testing.assert_allclose(null_1, null_2)

    cfg = AutoKConfig(k_method="perm_gap", min_k=0, max_k=4, perm_B=3, gap_rule="argmax")
    k_hat, diag = select_k_perm_gap(np.array([0.4, 0.7, 0.72, 0.73]), null_1, cfg)
    assert 0 <= k_hat <= 4
    assert {"gap", "gap_se", "perm_B", "gap_rule"} <= set(diag.columns)

    with pytest.raises(ValueError, match="within_group.*requires groups"):
        null_objective_paths(
            cache,
            y,
            B=1,
            max_k=2,
            null="within_group",
            time=np.arange(len(y)),
            top_m=8,
            corr_prune="auto",
            random_state=123,
        )


def test_perm_gap_can_select_zero_when_real_curve_stays_under_null():
    objective = np.array([0.1, 0.2])
    nulls = np.array([[0.2, 0.4], [0.3, 0.5]])
    cfg = AutoKConfig(k_method="perm_gap", min_k=0, max_k=2, gap_rule="argmax")

    k_hat, diag = select_k_perm_gap(objective, nulls, cfg)

    assert k_hat == 0
    assert diag["k"].tolist()[0] == 0
    assert diag.loc[diag["selected"], "k"].tolist() == [0]


def test_perm_gap_gain_envelope_handles_single_null_replicate():
    objective = np.array([0.1, 0.2])
    nulls = np.array([[0.2, 0.4]])
    cfg = AutoKConfig(
        k_method="perm_gap",
        min_k=0,
        max_k=2,
        perm_B=1,
        gap_rule="gain_envelope",
        stop_patience=2,
    )

    k_hat, diag = select_k_perm_gap(objective, nulls, cfg)

    assert k_hat == 0
    assert np.isfinite(diag["gap_se"]).all()
    assert diag.loc[diag["selected"], "k"].tolist() == [0]


def test_zero_capable_gaussian_wrappers_report_zero_effective_min(monkeypatch):
    class DummyCache:
        sample_weight = np.ones(12, dtype=np.float64)
        valid_cols = np.arange(2, dtype=np.int64)

    def fake_cached_path(*_args, **_kwargs):
        return ["x0", "x1"], [0, 1], np.array([0.0, 0.0], dtype=np.float64)

    monkeypatch.setattr(filter_auto_k, "_cached_filter_path", fake_cached_path)
    monkeypatch.setattr(
        filter_auto_k,
        "null_objective_paths",
        lambda *_args, **_kwargs: np.array([[0.2, 0.4]], dtype=np.float64),
    )
    monkeypatch.setattr(filter_auto_k, "_consensus_method_k", lambda *_args, **_kwargs: (0, ""))

    calls = [
        (
            filter_auto_k.select_gaussian_penalized_path,
            AutoKConfig(
                k_method="penalized_objective",
                objective_penalty="ebic",
                min_k=0,
                max_k=2,
            ),
        ),
        (
            filter_auto_k.select_gaussian_perm_gap_path,
            AutoKConfig(k_method="perm_gap", min_k=0, max_k=2, gap_rule="argmax", perm_B=1),
        ),
        (
            filter_auto_k.select_gaussian_consensus_path,
            AutoKConfig(
                k_method="consensus",
                min_k=0,
                max_k=2,
                consensus_methods=("ebic", "chi2_stop"),
            ),
        ),
    ]

    for selector, cfg in calls:
        _selected, _indices, _diag, summary = selector(
            cache=DummyCache(),
            y=np.zeros(12, dtype=np.float64),
            method="cefsplus",
            max_k=2,
            top_m=2,
            auto_k_config=cfg,
            verbose=False,
        )
        assert summary["selected_k"] == 0
        assert summary["effective_min_k"] == 0
        assert summary["selected_at_min_k"] is True

    with pytest.warns(UserWarning, match="changepoint requires at least three"):
        _selected, _indices, _diag, summary = filter_auto_k.select_gaussian_changepoint_path(
            cache=DummyCache(),
            y=np.zeros(12, dtype=np.float64),
            method="cefsplus",
            max_k=2,
            top_m=2,
            auto_k_config=AutoKConfig(k_method="changepoint", min_k=0, max_k=2),
            verbose=False,
        )
    assert summary["selected_k"] == 0
    assert summary["effective_min_k"] == 0
    assert summary["selected_at_min_k"] is True


def test_consensus_rejects_unknown_or_failed_submethods(monkeypatch):
    class DummyCache:
        sample_weight = np.ones(12, dtype=np.float64)
        valid_cols = np.arange(2, dtype=np.int64)

    monkeypatch.setattr(
        filter_auto_k,
        "_cached_filter_path",
        lambda *_args, **_kwargs: (
            ["x0", "x1"],
            [0, 1],
            np.array([0.0, 0.0], dtype=np.float64),
        ),
    )
    cfg = AutoKConfig(
        k_method="consensus",
        min_k=0,
        max_k=2,
        consensus_methods=("gausian_cv", "ebic"),
    )

    with pytest.raises(ValueError, match="unsupported method.*gausian_cv"):
        filter_auto_k.select_gaussian_consensus_path(
            cache=DummyCache(),
            y=np.zeros(12, dtype=np.float64),
            method="cefsplus",
            max_k=2,
            top_m=2,
            auto_k_config=cfg,
            verbose=False,
        )

    def fake_consensus_method(name, **_kwargs):
        if name == "chi2_stop":
            raise ValueError("boom")
        return 0, ""

    monkeypatch.setattr(filter_auto_k, "_consensus_method_k", fake_consensus_method)
    cfg = AutoKConfig(
        k_method="consensus",
        min_k=0,
        max_k=2,
        consensus_methods=("ebic", "chi2_stop"),
    )

    with pytest.raises(ValueError, match="chi2_stop.*boom"):
        filter_auto_k.select_gaussian_consensus_path(
            cache=DummyCache(),
            y=np.zeros(12, dtype=np.float64),
            method="cefsplus",
            max_k=2,
            top_m=2,
            auto_k_config=cfg,
            verbose=False,
        )


def test_xfit_objective_and_gaussian_cv_curves_select_from_all_k_grid():
    rng = np.random.default_rng(888)
    X = pd.DataFrame(rng.normal(size=(120, 10)), columns=[f"x{i}" for i in range(10)])
    y = 1.4 * X["x0"].to_numpy() - 0.8 * X["x1"].to_numpy() + rng.normal(scale=0.4, size=120)
    cache = build_cache(X, subsample=None, compute_Rxx=True)

    xfit_cfg = AutoKConfig(
        k_method="xfit_objective",
        strategy="kfold",
        selection_rule="best",
        min_k=1,
        max_k=5,
        xfit_folds=3,
        random_state=7,
    )
    xfit_curves = xfit_objective_curves(
        cache,
        y,
        config=xfit_cfg,
        top_m=10,
        corr_prune="auto",
        method="cefsplus",
    )
    xfit_k, xfit_diag = select_k_xfit_objective(xfit_curves, xfit_cfg)
    assert 1 <= xfit_k <= 5
    assert {"score_mean", "score_se", "debias"} <= set(xfit_diag.columns)

    cv_cfg = AutoKConfig(
        k_method="gaussian_cv",
        strategy="kfold",
        selection_rule="best",
        min_k=1,
        max_k=5,
        xfit_folds=3,
        random_state=7,
    )
    cv_curves = gaussian_cv_curves(
        cache,
        y,
        config=cv_cfg,
        top_m=10,
        corr_prune="auto",
        method="cefsplus",
    )
    cv_k, cv_diag = select_k_gaussian_cv(cv_curves, cv_cfg)
    assert 1 <= cv_k <= 5
    assert {"proxy", "xfit_ridge", "split_scores"} <= set(cv_diag.columns)


def test_xfit_objective_uses_one_se_default_and_null_guard():
    curves = pd.DataFrame(
        {
            "k": [1, 2, 3],
            "score_mean": [2.0, 2.1, 2.2],
            "score_se": [0.5, 0.5, 0.5],
            "score": [2.0, 2.1, 2.2],
            "n_splits": [3, 3, 3],
        }
    )
    cfg = AutoKConfig(k_method="xfit_objective", min_k=1, max_k=3, selection_rule="best")

    k_hat, diag = select_k_xfit_objective(curves, cfg)

    assert k_hat == 1
    assert diag["selection_rule_effective"].iloc[0] == "one_se"
    assert diag["selection_rule_requested"].iloc[0] == "best"

    null_curves = curves.copy()
    null_curves["score_mean"] = [0.01, 0.02, 0.03]
    null_cfg = AutoKConfig(k_method="xfit_objective", min_k=0, max_k=3)
    null_k, null_diag = select_k_xfit_objective(null_curves, null_cfg)

    assert null_k == 0
    assert null_diag["stopped_by"].iloc[0] == "null_guard"
    assert null_diag["null_guard_z"].iloc[0] == 2.5
    assert not null_diag["selected"].any()


def test_gaussian_cv_empty_curves_fall_back_to_method_floor():
    empty = pd.DataFrame()
    empty.attrs["stopped_by"] = "degenerate_folds"
    cfg = AutoKConfig(k_method="gaussian_cv", min_k=3, max_k=5)

    k_hat, diag = select_k_gaussian_cv(empty, cfg)

    assert k_hat == 3
    assert diag.empty
    assert diag.attrs["stopped_by"] == "degenerate_folds"


def test_xfit_curve_builder_drops_degenerate_folds_or_stops():
    cfg = AutoKConfig(k_method="xfit_objective", min_k=1, max_k=3)
    extra = {
        "xfit_mode": "shared_z",
        "fold_max_k": (0, 3, 2),
        "fold_n_eff": (10.0, 10.0, 10.0),
    }

    with pytest.warns(UserWarning, match="dropped 1 degenerate"):
        diag = auto_k_xfit._curve_from_fold_scores(
            [np.array([]), np.array([1.0, 1.1, 1.2]), np.array([0.9, 1.0])],
            cfg,
            extra=extra,
            score_kind="xfit_objective",
        )

    assert diag["xfit_folds"].iloc[0] == 2
    assert diag["dropped_folds"].iloc[0] == 1
    assert diag["k"].tolist() == [1, 2]

    with pytest.warns(UserWarning, match="only 1 healthy fold"):
        stopped = auto_k_xfit._curve_from_fold_scores(
            [np.array([]), np.array([1.0, 1.1]), np.array([])],
            cfg,
            extra=extra,
            score_kind="xfit_objective",
        )

    assert stopped.empty
    assert stopped.attrs["stopped_by"] == "degenerate_folds"
    assert stopped.attrs["healthy_folds"] == 1
    assert stopped.attrs["dropped_folds"] == 2


def test_gaussian_cv_scores_match_direct_prefix_solves():
    rng = np.random.default_rng(890)

    def spd_corr() -> np.ndarray:
        A = rng.normal(size=(5, 5))
        cov = A @ A.T + 5.0 * np.eye(5)
        scale = np.sqrt(np.diag(cov))
        return cov / np.outer(scale, scale)

    R_train = spd_corr()
    R_val = spd_corr()
    r_train = rng.normal(scale=0.2, size=5)
    r_val = rng.normal(scale=0.2, size=5)
    ridge = 1e-3

    got = auto_k_xfit._gaussian_cv_scores(R_train, r_train, R_val, r_val, ridge=ridge)
    expected = []
    for k in range(1, 6):
        beta = np.linalg.solve(R_train[:k, :k] + ridge * np.eye(k), r_train[:k])
        expected.append(1.0 - 2.0 * beta @ r_val[:k] + beta @ R_val[:k, :k] @ beta)

    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)


def test_gaussian_cv_scores_match_actual_standardized_holdout_mse():
    rng = np.random.default_rng(891)
    Z_train = rng.normal(size=(1200, 6))
    Z_val = rng.normal(size=(800, 6))
    beta_true = np.array([1.2, -0.9, 0.7, 0.0, 0.0, 0.0])
    zy_train = Z_train @ beta_true + rng.normal(scale=0.8, size=1200)
    zy_val = Z_val @ beta_true + rng.normal(scale=0.8, size=800)
    Z_train = (Z_train - Z_train.mean(axis=0)) / Z_train.std(axis=0)
    Z_val = (Z_val - Z_val.mean(axis=0)) / Z_val.std(axis=0)
    zy_train = (zy_train - zy_train.mean()) / zy_train.std()
    zy_val = (zy_val - zy_val.mean()) / zy_val.std()
    R_train = Z_train.T @ Z_train / Z_train.shape[0]
    R_val = Z_val.T @ Z_val / Z_val.shape[0]
    r_train = Z_train.T @ zy_train / Z_train.shape[0]
    r_val = Z_val.T @ zy_val / Z_val.shape[0]
    ridge = 1e-3

    scores = auto_k_xfit._gaussian_cv_scores(R_train, r_train, R_val, r_val, ridge=ridge)
    actual_mse = []
    for k in range(1, 7):
        beta_hat = np.linalg.solve(R_train[:k, :k] + ridge * np.eye(k), r_train[:k])
        pred = Z_val[:, :k] @ beta_hat
        actual_mse.append(float(np.mean((zy_val - pred) ** 2)))

    np.testing.assert_allclose(scores, actual_mse, rtol=1e-12, atol=1e-12)
    assert np.corrcoef(scores, actual_mse)[0, 1] > 0.99


def test_xfit_null_debias_matches_digamma_drift():
    rng = np.random.default_rng(789)
    for nu in (30, 100, 400):
        r2 = rng.beta(0.5, nu / 2.0, size=5000)
        null_gains = -np.log1p(-r2)
        expected = digamma((nu + 1.0) / 2.0) - digamma(nu / 2.0)
        assert float(np.mean(null_gains)) == pytest.approx(float(expected), abs=0.002)

    n_eff = 80.0
    ks = np.arange(1, 6, dtype=np.float64)
    nu = n_eff - ks - 1.0
    drift = digamma((nu + 1.0) / 2.0) - digamma(nu / 2.0)
    debiased = auto_k_xfit._xfit_scores(np.cumsum(drift), n_eff_val=n_eff)
    np.testing.assert_allclose(debiased, np.zeros(5), atol=1e-12)


def test_xfit_helpers_slice_full_length_metadata_to_cache_rows():
    rng = np.random.default_rng(889)
    X = pd.DataFrame(rng.normal(size=(140, 10)), columns=[f"x{i}" for i in range(10)])
    y = X["x0"].to_numpy() + rng.normal(scale=0.4, size=140)
    cache = build_cache(X, subsample=70, random_state=5, compute_Rxx=True)
    time = np.arange(len(y))

    cfg = AutoKConfig(
        k_method="gaussian_cv",
        strategy="time_holdout",
        selection_rule="best",
        min_k=1,
        max_k=4,
    )
    curves = gaussian_cv_curves(
        cache,
        y,
        config=cfg,
        time=time,
        top_m=10,
        corr_prune="auto",
        method="cefsplus",
    )

    assert not curves.empty
    assert curves["n_splits"].iloc[0] == 1


def test_knockoff_path_and_stability_helpers_smoke():
    rng = np.random.default_rng(999)
    X = pd.DataFrame(rng.normal(size=(100, 8)), columns=[f"x{i}" for i in range(8)])
    y = 1.2 * X["x0"].to_numpy() + rng.normal(scale=0.5, size=100)
    cache = build_cache(X, subsample=None, compute_Rxx=True)

    knock_cfg = AutoKConfig(
        k_method="knockoff_path",
        min_k=0,
        max_k=4,
        knockoff_q=0.5,
        random_state=11,
    )
    selected_valid, k_hat, knock_diag = select_k_knockoff_path(cache, y, knock_cfg, top_m=8)
    assert k_hat == len(selected_valid)
    assert {"label", "fdp_hat", "q"} <= set(knock_diag.columns)
    assert "corr_prune_disabled" in knock_diag.columns
    assert bool(knock_diag["corr_prune_disabled"].iloc[0])

    paths = bootstrap_paths(
        cache,
        y,
        B=3,
        max_k=4,
        boot_mode="bayes",
        top_m=8,
        corr_prune="auto",
        random_state=12,
    )
    stab_cfg = AutoKConfig(k_method="stability", min_k=1, max_k=4, boot_B=3)
    stab_k, stab_diag = select_k_stability(paths, len(cache.valid_cols), stab_cfg)
    assert 0 <= stab_k <= 4
    assert {"phi", "mean_jaccard", "boot_B"} <= set(stab_diag.columns)


def test_knockoff_multi_draw_aggregation_clamps_to_max_k(monkeypatch):
    class DummyCache:
        Z = np.zeros((2, 4), dtype=np.float64)
        valid_cols = np.arange(4, dtype=np.int64)

    draws = iter(
        [
            np.array([0, 1], dtype=np.int64),
            np.array([2, 3], dtype=np.int64),
        ]
    )

    def fake_draw(_cache, _y, *, config, top_m, random_state, draw_state):
        del config, top_m, random_state, draw_state
        selected = next(draws)
        diag = pd.DataFrame(
            {
                "feature_index_valid": selected,
                "selected": True,
            }
        )
        return selected, int(selected.size), diag

    monkeypatch.setattr(auto_k_knockoff, "_draw_knockoff_path", fake_draw)
    monkeypatch.setattr(
        auto_k_knockoff,
        "_prepare_knockoff_draw_state",
        lambda cache, config: object(),
    )
    cfg = AutoKConfig(
        k_method="knockoff_path",
        min_k=0,
        max_k=2,
        knockoff_draws=2,
        random_state=123,
    )
    selected, k_hat, diag = auto_k_knockoff.select_k_knockoff_path(
        DummyCache(),
        np.zeros(2),
        cfg,
        top_m=4,
    )

    assert k_hat == 2
    assert selected.tolist() == [0, 1]
    assert diag["selected_final"].sum() == 2
    assert diag.attrs["q_scope"] == "per_draw"
    assert diag.attrs["aggregation_fdr_control"] == "none"
    assert not diag.attrs["aggregation_preserves_per_draw_fdr"]


def test_knockoff_multi_draw_fits_model_once(monkeypatch):
    rng = np.random.default_rng(1001)
    X = pd.DataFrame(rng.normal(size=(90, 7)))
    y = X.iloc[:, 0].to_numpy() + rng.normal(scale=0.5, size=len(X))
    cache = build_cache(X, subsample=None, compute_Rxx=True)
    original = auto_k_knockoff.fit_gaussian_knockoffs
    calls = 0

    def wrapped(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(auto_k_knockoff, "fit_gaussian_knockoffs", wrapped)
    cfg = AutoKConfig(
        k_method="knockoff_path",
        min_k=0,
        max_k=3,
        knockoff_draws=3,
        knockoff_q=0.5,
        random_state=1001,
    )

    select_k_knockoff_path(cache, y, cfg, top_m=7)

    assert calls == 1


def test_knockoff_pair_table_seqstep_hand_computation():
    selected, diag = auto_k_knockoff._knockoff_prefix_table(
        np.array([0, 4, 2, 3], dtype=np.int64),
        np.array([0.9, 0.4, 0.3, 0.2], dtype=np.float64),
        np.array([10, 11, 12], dtype=np.int64),
        np.arange(20, dtype=np.int64) + 100,
        q=1.0,
        max_k=3,
    )

    assert selected.tolist() == [10, 12]
    assert diag["label"].tolist() == [1, -1, 1]
    assert diag["fdp_hat"].tolist() == pytest.approx([1.0, 2.0, 1.0])
    assert diag["selected"].tolist() == [True, False, True]

    capped, capped_diag = auto_k_knockoff._knockoff_prefix_table(
        np.array([0, 4, 2], dtype=np.int64),
        np.ones(3, dtype=np.float64),
        np.array([10, 11, 12], dtype=np.int64),
        np.arange(20, dtype=np.int64),
        q=1.0,
        max_k=1,
    )
    assert capped.tolist() == [10]
    assert capped_diag["selected"].tolist() == [True, False, False]


def test_stability_pi_threshold_uses_phi_null_guard():
    paths = [
        np.array([0, 1], dtype=np.int64),
        np.array([0, 2], dtype=np.int64),
        np.array([1, 2], dtype=np.int64),
    ]
    cfg = AutoKConfig(
        k_method="stability",
        min_k=1,
        max_k=2,
        stability_rule="pi_threshold",
        stability_pi=0.6,
        boot_B=3,
    )

    k_hat, diag = select_k_stability(paths, 3, cfg)

    assert k_hat == 1
    assert diag["phi"].max() < 0.5
    assert diag["max_phi"].iloc[0] < 0.5
    assert diag["stopped_by"].iloc[0] == "stability_floor"
    assert diag.attrs["stopped_by"] == "stability_floor"
    assert diag.loc[diag["selected"], "k"].tolist() == [1]


def test_stability_pi_threshold_can_select_zero_when_allowed():
    paths = [
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
        np.array([2], dtype=np.int64),
    ]
    cfg = AutoKConfig(
        k_method="stability",
        min_k=0,
        max_k=1,
        stability_rule="pi_threshold",
        stability_pi=0.9,
        boot_B=3,
    )

    k_hat, diag = select_k_stability(paths, 3, cfg)

    assert k_hat == 0
    assert diag["stopped_by"].iloc[0] == "stability_floor"
    assert not diag["selected"].any()


def test_stability_max_one_se_can_select_zero_when_allowed():
    paths = [
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
        np.array([2], dtype=np.int64),
    ]
    cfg = AutoKConfig(
        k_method="stability",
        min_k=0,
        max_k=1,
        stability_rule="max_one_se",
        boot_B=3,
    )

    k_hat, diag = select_k_stability(paths, 3, cfg)

    assert k_hat == 0
    assert diag["stopped_by"].iloc[0] == "stability_floor"
    assert diag.attrs["max_phi"] < 0.5
    assert not diag["selected"].any()


def test_stability_rules_honor_effective_min_k():
    paths = [np.array([0, 1, 2], dtype=np.int64)] * 3

    default_cfg = AutoKConfig(k_method="stability", min_k=2, max_k=3, boot_B=3)
    default_k, default_diag = select_k_stability(paths, 3, default_cfg)
    assert default_cfg.stability_rule == "max_one_se"
    assert default_k == 2
    assert default_diag.loc[default_diag["selected"], "k"].tolist() == [2]
    assert default_diag["k"].max() == 2
    assert default_diag["phi_se"].fillna(0.0).max() == 0.0

    max_rule_cfg = AutoKConfig(
        k_method="stability",
        min_k=2,
        max_k=3,
        boot_B=3,
        stability_rule="max_one_se",
    )
    max_rule_k, max_rule_diag = select_k_stability(paths, 3, max_rule_cfg)
    assert max_rule_k == 2
    assert max_rule_diag.loc[max_rule_diag["selected"], "k"].tolist() == [2]

    pi_cfg = AutoKConfig(
        k_method="stability",
        min_k=2,
        max_k=2,
        stability_rule="pi_threshold",
        stability_pi=0.9,
        boot_B=3,
    )
    pi_k, pi_diag = select_k_stability(
        [
            np.array([0, 1], dtype=np.int64),
            np.array([2, 3], dtype=np.int64),
            np.array([4, 5], dtype=np.int64),
        ],
        6,
        pi_cfg,
    )
    assert pi_k == 2
    assert pi_diag.loc[pi_diag["selected"], "k"].tolist() == [2]


def test_nogueira_phi_ground_truth_and_jackknife_sanity():
    assert np.isnan(
        auto_k_resample._stability_phi_from_counts(
            np.array([3.0, 3.0, 3.0]),
            B=3,
            k=3,
            p=3,
        )
    )

    worked = auto_k_resample._stability_phi_from_counts(
        np.array([3.0, 1.0, 1.0, 1.0]),
        B=3,
        k=2,
        p=4,
    )
    assert worked == pytest.approx(0.0)

    identical = auto_k_resample._stability_phi_from_counts(
        np.array([3.0, 3.0, 0.0, 0.0]),
        B=3,
        k=2,
        p=4,
    )
    assert identical == pytest.approx(1.0)

    rng = np.random.default_rng(222)
    p = 50
    k = 5
    B = 400
    counts = np.zeros(p, dtype=np.float64)
    for _ in range(B):
        counts[rng.choice(p, size=k, replace=False)] += 1.0
    random_phi = auto_k_resample._stability_phi_from_counts(counts, B=B, k=k, p=p)
    assert random_phi == pytest.approx(0.0, abs=0.08)

    paths = [np.array([0, 1, 2], dtype=np.int64)] * 4
    _k_hat, diag = select_k_stability(
        paths,
        5,
        AutoKConfig(
            k_method="stability",
            min_k=1,
            max_k=3,
            boot_B=4,
            stability_rule="max_one_se",
        ),
    )
    assert diag["phi"].dropna().min() == pytest.approx(1.0)
    assert diag["phi_se"].fillna(0.0).max() == pytest.approx(0.0)


def test_perm_gap_within_group_negative_control_preserves_group_confound():
    rng = np.random.default_rng(7)
    n_groups = 30
    group_size = 10
    n = n_groups * group_size
    p = 20
    groups = np.repeat(np.arange(n_groups), group_size)
    group_effect = rng.normal(size=n_groups)
    X = rng.normal(scale=0.2, size=(n, p))
    for j in range(5):
        X[:, j] += group_effect[groups] + 0.05 * rng.normal(size=n)
    y = group_effect[groups] + rng.normal(scale=0.2, size=n)
    cache = build_cache(pd.DataFrame(X), subsample=None, compute_Rxx=True)
    _names, _indices, objective = select_cached(
        cache,
        y,
        10,
        method="cefsplus",
        top_m=20,
        return_indices=True,
        return_objective=True,
    )

    global_null = null_objective_paths(
        cache,
        y,
        B=10,
        max_k=10,
        null="permute",
        groups=groups,
        top_m=20,
        corr_prune="auto",
        random_state=7,
    )
    grouped_null = null_objective_paths(
        cache,
        y,
        B=10,
        max_k=10,
        null="within_group",
        groups=groups,
        top_m=20,
        corr_prune="auto",
        random_state=7,
    )
    cfg = AutoKConfig(k_method="perm_gap", min_k=0, max_k=10, perm_B=10)
    global_k, global_diag = select_k_perm_gap(objective, global_null, cfg)
    grouped_k, grouped_diag = select_k_perm_gap(objective, grouped_null, cfg)

    assert global_k >= 2
    assert grouped_k <= 1
    assert global_diag["gap"].max() > grouped_diag["gap"].max() + 1.0


def test_consensus_unused_field_warnings_include_submethods():
    cfg = AutoKConfig(
        k_method="consensus",
        alpha=0.1,
        perm_B=7,
        xfit_folds=3,
        consensus_methods=("chi2_stop", "perm_gap", "gaussian_cv"),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_auto_k_config(cfg)
    assert not caught

    with pytest.warns(UserWarning, match="stability_pi"):
        validate_auto_k_config(
            AutoKConfig(
                k_method="consensus",
                stability_pi=0.9,
                consensus_methods=("chi2_stop",),
            )
        )


def test_evaluate_rejects_kfold_strategy_during_validation():
    with pytest.raises(ValueError, match="kfold.*gaussian_cv"):
        validate_auto_k_config(AutoKConfig(k_method="evaluate", strategy="kfold"))


def test_knockoff_prefix_summary_clamps_to_rebuilt_path(monkeypatch):
    class DummyCache:
        valid_cols = np.arange(3, dtype=np.int64)
        feature_names = ["x0", "x1", "x2"]

    monkeypatch.setattr(
        filter_auto_k,
        "select_k_knockoff_path",
        lambda *_args, **_kwargs: (
            np.array([0, 1, 2], dtype=np.int64),
            3,
            pd.DataFrame({"k": [1, 2, 3]}),
        ),
    )
    monkeypatch.setattr(
        filter_auto_k,
        "_cached_filter_path",
        lambda *_args, **_kwargs: (["x0"], [0], np.array([], dtype=np.float64)),
    )
    cfg = AutoKConfig(
        k_method="knockoff_path",
        min_k=0,
        max_k=3,
        knockoff_return="prefix",
    )

    selected, selected_indices, _diag, summary = filter_auto_k.select_gaussian_knockoff_path(
        cache=DummyCache(),
        y=np.zeros(5, dtype=np.float64),
        method="cefsplus",
        max_k=3,
        top_m=3,
        auto_k_config=cfg,
        verbose=False,
    )

    assert selected == ["x0"]
    assert selected_indices == [0]
    assert summary["selected_k"] == 1
    assert summary["count_only"]


def test_public_chi2_panel_mode_uses_screened_panel_width():
    rng = np.random.default_rng(1000)
    X = pd.DataFrame(rng.normal(size=(120, 20)), columns=[f"x{i}" for i in range(20)])
    y = X["x0"].to_numpy() + rng.normal(scale=0.4, size=120)
    cfg = AutoKConfig(k_method="chi2_stop", min_k=0, max_k=4, m_mode="panel")

    result = select_cefsplus(
        X,
        y,
        k="auto",
        auto_k_config=cfg,
        top_m=6,
        return_result=True,
        verbose=False,
    )

    diag = result.diagnostics_["auto_k_diagnostics"]
    assert diag["m_eff"].iloc[0] <= 6


def test_consensus_gain_tests_preserve_panel_semantics(monkeypatch):
    class DummyCache:
        sample_weight = np.ones(40, dtype=np.float64)
        valid_cols = np.arange(20, dtype=np.int64)

    captured = {}
    panel_eigs = np.array([1.5, 0.5], dtype=np.float64)

    monkeypatch.setattr(
        filter_auto_k,
        "_gain_test_candidate_inputs",
        lambda *_args, **_kwargs: (6, panel_eigs),
    )

    def fake_chi(objective, config, *, n_eff, p_candidates, panel_eigs):
        captured.update(
            m_mode=config.m_mode,
            p_candidates=p_candidates,
            panel_eigs=panel_eigs,
        )
        return 2, pd.DataFrame()

    monkeypatch.setattr(filter_auto_k, "select_k_chi2_stop", fake_chi)
    cfg = AutoKConfig(
        k_method="consensus",
        min_k=0,
        max_k=4,
        m_mode="li_ji",
        consensus_methods=("chi2_stop",),
    )

    k_hat, _ = filter_auto_k._consensus_method_k(
        "chi2_stop",
        cache=DummyCache(),
        y=np.zeros(40),
        method="cefsplus",
        objective=np.arange(4, dtype=np.float64),
        config=cfg,
        top_m=6,
        corr_prune="auto",
        groups=None,
        time=None,
        source_groups=None,
        source_time=None,
        path_length=4,
    )

    assert k_hat == 2
    assert captured["m_mode"] == "li_ji"
    assert captured["p_candidates"] == 6
    assert captured["panel_eigs"] is panel_eigs


def test_consensus_submethods_get_distinct_deterministic_seeds():
    first = filter_auto_k._consensus_method_seed(123, "perm_gap")
    second = filter_auto_k._consensus_method_seed(123, "stability")

    assert first == filter_auto_k._consensus_method_seed(123, "perm_gap")
    assert first != second
    assert isinstance(filter_auto_k._consensus_method_seed(-1, "perm_gap"), int)


def test_gaussian_mrmr_stability_passes_selector_method(monkeypatch):
    rng = np.random.default_rng(1002)
    X = pd.DataFrame(rng.normal(size=(80, 8)), columns=[f"x{i}" for i in range(8)])
    y = X["x0"].to_numpy() + rng.normal(scale=0.5, size=80)
    captured = {}

    def fake_bootstrap_paths(*args, method, **kwargs):
        del args, kwargs
        captured["method"] = method
        return [np.array([0, 1, 2], dtype=np.int64)] * 3

    monkeypatch.setattr(filter_auto_k, "bootstrap_paths", fake_bootstrap_paths)
    cfg = AutoKConfig(k_method="stability", min_k=1, max_k=3, boot_B=3)

    selected = select_mrmr(
        X,
        y,
        k="auto",
        task="regression",
        estimator="gaussian",
        auto_k_config=cfg,
        verbose=False,
    )

    assert selected
    assert captured["method"] == "mrmr_quot"


def test_cefsplus_public_dispatch_supports_phase3_methods():
    rng = np.random.default_rng(1001)
    X = pd.DataFrame(rng.normal(size=(120, 9)), columns=[f"x{i}" for i in range(9)])
    y = 1.5 * X["x0"].to_numpy() - X["x1"].to_numpy() + rng.normal(scale=0.4, size=120)

    configs = [
        AutoKConfig(k_method="perm_gap", min_k=0, max_k=4, perm_B=3, random_state=1),
        AutoKConfig(
            k_method="xfit_objective",
            strategy="kfold",
            selection_rule="best",
            min_k=1,
            max_k=4,
            xfit_folds=3,
            random_state=1,
        ),
        AutoKConfig(
            k_method="gaussian_cv",
            strategy="kfold",
            selection_rule="best",
            min_k=1,
            max_k=4,
            xfit_folds=3,
            random_state=1,
        ),
        AutoKConfig(k_method="stability", min_k=1, max_k=4, boot_B=3, random_state=1),
        AutoKConfig(k_method="consensus", min_k=0, max_k=4, consensus_methods=("ebic", "chi2_stop")),
    ]
    for cfg in configs:
        result = select_cefsplus(
            X,
            y,
            k="auto",
            auto_k_config=cfg,
            return_result=True,
            verbose=False,
        )
        assert result.diagnostics_["auto_k"]["method"] == cfg.k_method
        assert "auto_k_diagnostics" in result.diagnostics_

    knock_cfg = AutoKConfig(
        k_method="knockoff_path",
        min_k=0,
        max_k=4,
        knockoff_q=0.5,
        random_state=2,
    )
    knock = select_cefsplus(
        X,
        y,
        k="auto",
        auto_k_config=knock_cfg,
        return_result=True,
        verbose=False,
    )
    assert knock.diagnostics_["auto_k"]["method"] == "knockoff_path"
    assert knock.diagnostics_["auto_k"]["corr_prune_disabled"]
    assert knock.diagnostics_["auto_k"]["fdr_control"] == "approximate_plugin"
    assert knock.diagnostics_["auto_k"]["approximate_fdr_control"]
    assert not knock.diagnostics_["auto_k"]["count_only"]


def test_consensus_clamps_default_min_k_on_short_paths():
    rng = np.random.default_rng(1003)
    X = pd.DataFrame(rng.normal(size=(80, 3)), columns=["x0", "x1", "x2"])
    y = X["x0"].to_numpy() + rng.normal(scale=0.5, size=80)

    with pytest.warns(UserWarning, match="consensus auto-k methods disagree"):
        result = select_cefsplus(
            X,
            y,
            k="auto",
            auto_k_config=AutoKConfig(k_method="consensus", perm_B=3, xfit_folds=3),
            top_m=3,
            return_result=True,
            verbose=False,
        )

    assert 0 <= result.diagnostics_["auto_k"]["selected_k"] <= 3


def test_auto_k_auto_router_defaults_cefsplus_to_measured_ebic():
    rng = np.random.default_rng(1100)
    X = pd.DataFrame(rng.normal(size=(160, 12)), columns=[f"x{i}" for i in range(12)])
    y = 1.2 * X["x0"].to_numpy() + rng.normal(scale=0.3, size=160)

    result = select_cefsplus(X, y, k="auto", return_result=True, verbose=False)
    summary = result.diagnostics_["auto_k"]

    assert summary["method"] == "auto"
    assert summary["auto_routing"]["chosen"] == "penalized_objective"
    assert summary["auto_routing"]["objective_penalty"] == "ebic"
    assert summary["auto_routing"]["reason"] == "measured_default_ebic"


def test_auto_k_auto_router_branches_are_reachable():
    rng = np.random.default_rng(1101)
    X_wide = pd.DataFrame(rng.normal(size=(80, 120)), columns=[f"x{i}" for i in range(120)])
    y_wide = 1.0 * X_wide["x0"].to_numpy() + rng.normal(scale=0.4, size=80)
    wide = select_cefsplus(
        X_wide,
        y_wide,
        k="auto",
        auto_k_config=AutoKConfig(k_method="auto", min_k=0, max_k=8),
        top_m=40,
        return_result=True,
        verbose=False,
    )
    assert wide.diagnostics_["auto_k"]["auto_routing"]["reason"] == "p_valid_exceeds_kish_n_eff"

    X = pd.DataFrame(rng.normal(size=(120, 8)), columns=[f"x{i}" for i in range(8)])
    y = X["x0"].to_numpy() + rng.normal(scale=0.4, size=120)
    weights = np.ones(120)
    weights[:8] = 30.0
    weighted = select_cefsplus(
        X,
        y,
        k="auto",
        sample_weight=weights,
        auto_k_config=AutoKConfig(k_method="auto", min_k=0, max_k=4, perm_B=2),
        return_result=True,
        verbose=False,
    )
    assert weighted.diagnostics_["auto_k"]["auto_routing"]["chosen"] == "perm_gap"
    assert weighted.diagnostics_["auto_k"]["auto_routing"]["reason"] == "heavy_weight_skew"

    gaussian = select_mrmr(
        X,
        y,
        k="auto",
        task="regression",
        estimator="gaussian",
        auto_k_config=AutoKConfig(k_method="auto", min_k=1, max_k=4, xfit_folds=3),
        return_result=True,
        verbose=False,
    )
    assert gaussian.diagnostics_["auto_k"]["auto_routing"]["chosen"] == "gaussian_cv"
    assert gaussian.diagnostics_["auto_k"]["auto_routing"]["reason"] == "non_cefsplus_gaussian_selector"


def test_auto_k_auto_router_fallback_records_metadata(monkeypatch):
    class DummyCache:
        sample_weight = np.ones(20, dtype=np.float64)
        valid_cols = np.arange(5, dtype=np.int64)

    calls = []

    def fake_runner(routed_config, **_kwargs):
        calls.append(routed_config.k_method)
        if len(calls) == 1:
            return [], [], pd.DataFrame(), {"selected_k": 0, "stopped_by": "degenerate_folds"}
        return ["x0"], [0], pd.DataFrame(), {"selected_k": 1}

    monkeypatch.setattr(filter_auto_k, "_run_gaussian_routed_path", fake_runner)

    selected, indices, _diag, summary = filter_auto_k.select_gaussian_auto_path(
        cache=DummyCache(),
        y=np.zeros(20),
        method="mrmr_quot",
        max_k=4,
        top_m=10,
        auto_k_config=AutoKConfig(k_method="auto", min_k=1, max_k=4),
        verbose=False,
    )

    assert selected == ["x0"]
    assert indices == [0]
    assert calls == ["gaussian_cv", "penalized_objective"]
    assert summary["auto_routing"]["primary"] == "gaussian_cv"
    assert summary["auto_routing"]["fallback"]["chosen"] == "penalized_objective"


def test_auto_k_router_falls_back_on_degenerate_placeholder_selection(monkeypatch):
    class DummyCache:
        sample_weight = np.ones(20, dtype=np.float64)
        valid_cols = np.arange(5, dtype=np.int64)

    calls = []

    def fake_runner(routed_config, **_kwargs):
        calls.append(routed_config.k_method)
        if len(calls) == 1:
            return ["x0"], [0], pd.DataFrame(), {
                "selected_k": 1,
                "stopped_by": "degenerate_folds",
            }
        return ["x1"], [1], pd.DataFrame(), {"selected_k": 1}

    monkeypatch.setattr(filter_auto_k, "_run_gaussian_routed_path", fake_runner)

    selected, indices, _diag, summary = filter_auto_k.select_gaussian_auto_path(
        cache=DummyCache(),
        y=np.zeros(20),
        method="mrmr_quot",
        max_k=4,
        top_m=10,
        auto_k_config=AutoKConfig(k_method="auto", min_k=1, max_k=4),
        verbose=False,
    )

    assert selected == ["x1"]
    assert indices == [1]
    assert calls == ["gaussian_cv", "penalized_objective"]
    assert summary["auto_routing"]["fallback"]["reason"].endswith("degenerate_folds")


def test_score_curve_rejects_partial_fold_coverage_and_all_invalid_uses_floor():
    from sift.selection.auto_k import choose_k_from_score_curve
    from sift.selection.auto_k_core import build_score_curve_diagnostics

    diagnostics = build_score_curve_diagnostics(
        [1, 2, 3],
        {
            1: [1.0, 1.0, 1.0, 1.0, 1.0],
            2: [0.5, np.inf, np.inf, np.inf, np.inf],
            3: [0.9, 0.9, 0.9, 0.9, 0.9],
        },
    )
    assert diagnostics.loc[diagnostics["k"] == 1, "score_mean"].iloc[0] == 1.0
    assert np.isinf(diagnostics.loc[diagnostics["k"] == 2, "score_mean"].iloc[0])
    assert diagnostics.loc[diagnostics["k"] == 2, "n_finite"].iloc[0] == 1

    selected_k, _ = choose_k_from_score_curve(
        diagnostics,
        AutoKConfig(k_method="evaluate", min_k=1, max_k=3, selection_rule="best"),
    )
    assert selected_k == 3

    with pytest.warns(UserWarning, match="method floor k=1"):
        selected_k, selected_diag = choose_k_from_score_curve(
            diagnostics.loc[diagnostics["k"] == 2],
            AutoKConfig(k_method="evaluate", min_k=1, max_k=2),
        )

    assert selected_k == 1
    assert not selected_diag["selected"].any()


def test_auto_k_auto_router_warns_and_records_saturation(monkeypatch):
    class DummyCache:
        sample_weight = np.ones(20, dtype=np.float64)
        valid_cols = np.arange(5, dtype=np.int64)

    def fake_runner(routed_config, **_kwargs):
        assert routed_config.k_method == "penalized_objective"
        return ["x0", "x1", "x2"], [0, 1, 2], pd.DataFrame(), {
            "selected_k": 3,
            "effective_max_k": 3,
            "path_length": 3,
            "selected_at_effective_max_k": True,
        }

    monkeypatch.setattr(filter_auto_k, "_run_gaussian_routed_path", fake_runner)

    with pytest.warns(UserWarning, match="configured max_k was reached"):
        selected, indices, _diag, summary = filter_auto_k.select_gaussian_auto_path(
            cache=DummyCache(),
            y=np.zeros(20),
            method="cefsplus",
            max_k=3,
            top_m=10,
            auto_k_config=AutoKConfig(k_method="auto", min_k=0, max_k=3),
            verbose=False,
        )

    assert selected == ["x0", "x1", "x2"]
    assert indices == [0, 1, 2]
    assert summary["auto_routing"]["chosen"] == "penalized_objective"
    assert summary["auto_routing"]["saturated"] is True
    assert summary["auto_routing"]["saturation_reason"] == "configured_max_k"


def test_auto_k_auto_router_reports_candidate_path_exhaustion_accurately(monkeypatch):
    class DummyCache:
        sample_weight = np.ones(20, dtype=np.float64)
        valid_cols = np.arange(3, dtype=np.int64)

    def fake_runner(_routed_config, **_kwargs):
        return ["x0", "x1", "x2"], [0, 1, 2], pd.DataFrame(), {
            "selected_k": 3,
            "effective_max_k": 3,
            "path_length": 3,
            "selected_at_effective_max_k": True,
            "path_exhausted_before_max_k": True,
        }

    monkeypatch.setattr(filter_auto_k, "_run_gaussian_routed_path", fake_runner)

    with pytest.warns(UserWarning, match="candidate path was exhausted") as warning_record:
        _selected, _indices, _diag, summary = filter_auto_k.select_gaussian_auto_path(
            cache=DummyCache(),
            y=np.zeros(20),
            method="cefsplus",
            max_k=5,
            top_m=10,
            auto_k_config=AutoKConfig(k_method="auto", min_k=0, max_k=5),
            verbose=False,
        )

    assert "Increase max_k or" not in str(warning_record[0].message)
    assert "Increasing max_k alone cannot" in str(warning_record[0].message)
    assert summary["auto_routing"]["saturation_reason"] == "candidate_path_exhausted"


def test_auto_k_auto_router_distinguishes_evaluation_curve_limit(monkeypatch):
    class DummyCache:
        sample_weight = np.ones(20, dtype=np.float64)
        valid_cols = np.arange(5, dtype=np.int64)

    def fake_runner(_routed_config, **_kwargs):
        return ["x0", "x1", "x2"], [0, 1, 2], pd.DataFrame(), {
            "selected_k": 3,
            "effective_max_k": 3,
            "path_length": 5,
            "selected_at_effective_max_k": True,
            "path_exhausted_before_max_k": False,
        }

    monkeypatch.setattr(filter_auto_k, "_run_gaussian_routed_path", fake_runner)

    with pytest.warns(UserWarning, match="evaluation curve ended") as warning_record:
        _selected, _indices, _diag, summary = filter_auto_k.select_gaussian_auto_path(
            cache=DummyCache(),
            y=np.zeros(20),
            method="mrmr",
            max_k=5,
            top_m=10,
            auto_k_config=AutoKConfig(k_method="auto", min_k=0, max_k=5),
            verbose=False,
        )

    assert "corr_prune/top_m" not in str(warning_record[0].message)
    assert summary["path_exhausted_before_max_k"] is False
    assert summary["evaluation_limited_before_path_end"] is True
    assert summary["auto_routing"]["saturation_reason"] == "evaluation_curve_limited"


def test_auto_k_auto_router_dense_check_warns_on_large_ebic_disagreement(monkeypatch):
    class DummyCache:
        sample_weight = np.ones(200, dtype=np.float64)
        valid_cols = np.arange(200, dtype=np.int64)

    def fake_runner(routed_config, **_kwargs):
        assert routed_config.k_method == "penalized_objective"
        return ["x"] * 120, list(range(120)), pd.DataFrame(), {
            "selected_k": 120,
            "effective_max_k": 160,
            "selected_at_effective_max_k": False,
        }

    captured = {}

    def fake_curves(*_args, config, **_kwargs):
        captured["config"] = config
        return pd.DataFrame({"k": [40], "score": [0.1], "se": [0.0], "n_splits": [3]})

    monkeypatch.setattr(filter_auto_k, "_run_gaussian_routed_path", fake_runner)
    monkeypatch.setattr(filter_auto_k, "gaussian_cv_curves", fake_curves)
    monkeypatch.setattr(filter_auto_k, "select_k_gaussian_cv", lambda *_args, **_kwargs: (40, pd.DataFrame()))

    with pytest.warns(UserWarning, match="dense-signal diagnostic"):
        _selected, _indices, _diag, summary = filter_auto_k.select_gaussian_auto_path(
            cache=DummyCache(),
            y=np.zeros(200),
            method="cefsplus",
            max_k=160,
            top_m=200,
            auto_k_config=AutoKConfig(
                k_method="auto",
                min_k=0,
                max_k=160,
                auto_dense_check=True,
            ),
            verbose=False,
        )

    check = summary["auto_routing"]["dense_check"]
    assert captured["config"].k_method == "gaussian_cv"
    assert captured["config"].selection_rule == "best"
    assert check["ran"] is True
    assert check["ebic_k"] == 120
    assert check["gaussian_cv_best_k"] == 40
    assert check["warned"] is True


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"auto_dense_check": 1}, "auto_dense_check"),
        ({"auto_dense_min_k": -1}, "auto_dense_min_k"),
        ({"auto_dense_min_frac": 1.5}, "auto_dense_min_frac"),
        ({"auto_dense_disagreement_ratio": 1.0}, "auto_dense_disagreement_ratio"),
    ],
)
def test_auto_dense_check_config_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        validate_auto_k_config(AutoKConfig(k_method="auto", **kwargs))


def test_binary_cefsplus_no_config_auto_routes_to_ebic():
    rng = np.random.default_rng(1102)
    X = pd.DataFrame(rng.normal(size=(140, 8)), columns=[f"x{i}" for i in range(8)])
    logits = 1.4 * X["x0"].to_numpy() - 0.8 * X["x1"].to_numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, probs, size=140)

    result = select_cefsplus_binary(
        X,
        y,
        k="auto",
        return_result=True,
        verbose=False,
    )

    summary = result.diagnostics_["auto_k"]
    assert summary["method"] == "auto"
    assert summary["auto_routing"]["chosen"] == "penalized_objective"
    assert summary["auto_routing"]["objective_penalty"] == "ebic"
