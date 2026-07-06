import numpy as np
import pandas as pd
import pytest

from sift import build_cache, select_mrmr
import sift.selection.filter_payloads as filter_payloads
import sift.selection.loops as loops_module


def _regression_data(n: int = 160, p: int = 12):
    rng = np.random.default_rng(123)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = (
        3.0 * X["f0"].to_numpy()
        - 2.0 * X["f1"].to_numpy()
        + 1.25 * X["f2"].to_numpy()
        + 0.5 * X["f3"].to_numpy()
        + rng.normal(scale=0.05, size=n)
    )
    return X, y


@pytest.mark.parametrize("formula", ["quotient", "difference"])
def test_classic_mrmr_backends_match_serial(formula):
    X, y = _regression_data()

    serial = select_mrmr(
        X,
        y,
        k=5,
        task="regression",
        formula=formula,
        mrmr_backend="serial",
        verbose=False,
    )
    blas = select_mrmr(
        X,
        y,
        k=5,
        task="regression",
        formula=formula,
        mrmr_backend="blas",
        verbose=False,
    )
    processes = select_mrmr(
        X,
        y,
        k=5,
        task="regression",
        formula=formula,
        n_jobs=2,
        mrmr_backend="processes",
        verbose=False,
    )

    assert blas == serial
    assert processes == serial


def test_classic_mrmr_weighted_process_and_blas_match_serial():
    X, y = _regression_data()
    sample_weight = np.linspace(0.5, 2.0, len(y))

    kwargs = dict(
        k=5,
        task="regression",
        sample_weight=sample_weight,
        top_m=10,
        verbose=False,
    )
    serial = select_mrmr(X, y, mrmr_backend="serial", **kwargs)
    blas = select_mrmr(X, y, mrmr_backend="blas", **kwargs)
    processes = select_mrmr(X, y, n_jobs=2, mrmr_backend="processes", **kwargs)

    assert blas == serial
    assert processes == serial


def test_mrmr_auto_backend_resolution_matches_public_contract():
    X, y = _regression_data(n=80, p=8)

    serial = select_mrmr(
        X,
        y,
        k=3,
        task="regression",
        mrmr_backend="serial",
        verbose=False,
    )
    default_auto = select_mrmr(X, y, k=3, task="regression", verbose=False)
    explicit_processes = select_mrmr(
        X,
        y,
        k=3,
        task="regression",
        n_jobs=2,
        mrmr_backend="processes",
        verbose=False,
    )
    auto_processes = select_mrmr(X, y, k=3, task="regression", n_jobs=2, verbose=False)

    assert default_auto == serial
    assert auto_processes == explicit_processes


def test_process_mrmr_k_one_does_not_start_process_pool(monkeypatch):
    def fail_parallel(*args, **kwargs):
        raise AssertionError("k=1 process mRMR should not start a process pool")

    monkeypatch.setattr(loops_module, "Parallel", fail_parallel)
    Z = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    relevance = np.array([0.1, 0.8, 0.3], dtype=np.float64)

    selected = loops_module._mrmr_loop_processes(
        Z,
        relevance,
        k=1,
        use_quotient=True,
        w=np.ones(Z.shape[0], dtype=np.float64),
        n_jobs=2,
    )

    np.testing.assert_array_equal(selected, np.array([1], dtype=np.int64))


def test_gaussian_mrmr_process_rank_transform_matches_serial():
    X, y = _regression_data(n=120, p=10)

    serial = select_mrmr(
        X,
        y,
        k=4,
        task="regression",
        estimator="gaussian",
        subsample=None,
        top_m=8,
        mrmr_backend="serial",
        verbose=False,
    )
    processes = select_mrmr(
        X,
        y,
        k=4,
        task="regression",
        estimator="gaussian",
        subsample=None,
        top_m=8,
        n_jobs=2,
        mrmr_backend="processes",
        verbose=False,
    )

    assert processes == serial


def test_gaussian_mrmr_prebuilt_cache_is_reused(monkeypatch):
    X, y = _regression_data(n=120, p=10)
    cache = build_cache(X, subsample=None, n_jobs=2, rank_backend="processes")

    def fail_build_cache(*args, **kwargs):
        raise AssertionError("select_mrmr should not rebuild a supplied cache")

    monkeypatch.setattr(filter_payloads, "build_cache", fail_build_cache)

    selected = select_mrmr(
        X,
        y,
        k=4,
        task="regression",
        estimator="gaussian",
        cache=cache,
        n_jobs=2,
        mrmr_backend="processes",
        verbose=False,
    )

    assert len(selected) == 4


def test_mrmr_parallel_validation_errors():
    X, y = _regression_data(n=40, p=5)

    with pytest.raises(ValueError, match="n_jobs"):
        select_mrmr(X, y, k=2, task="regression", n_jobs=0, verbose=False)

    with pytest.raises(ValueError, match="mrmr_backend"):
        select_mrmr(
            X,
            y,
            k=2,
            task="regression",
            mrmr_backend="bad",
            verbose=False,
        )


def test_process_mrmr_after_catboost_import_smoke():
    pytest.importorskip("catboost")
    X, y = _regression_data(n=80, p=8)

    selected = select_mrmr(
        X,
        y,
        k=3,
        task="regression",
        n_jobs=2,
        mrmr_backend="processes",
        verbose=False,
    )

    assert len(selected) == 3
