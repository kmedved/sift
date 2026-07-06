from __future__ import annotations

import pytest

import sift._numba as sift_numba


def test_njit_optional_cache_retries_only_cache_locator_failures(monkeypatch):
    calls: list[bool] = []

    def fake_njit(*args, **kwargs):
        del args
        calls.append(bool(kwargs["cache"]))

        def decorate(fn):
            if kwargs["cache"]:
                raise RuntimeError(
                    "cannot cache function 'kernel': no locator available for file '/tmp/kernel.py'"
                )
            return ("compiled", fn.__name__, kwargs["cache"])

        return decorate

    monkeypatch.setattr(sift_numba, "njit", fake_njit)

    def kernel():
        return 1

    assert sift_numba.njit_optional_cache(cache=True)(kernel) == ("compiled", "kernel", False)
    assert calls == [True, False]


def test_njit_optional_cache_keeps_real_runtime_errors(monkeypatch):
    def fake_njit(*args, **kwargs):
        del args, kwargs

        def decorate(fn):
            raise RuntimeError("real numba failure")

        return decorate

    monkeypatch.setattr(sift_numba, "njit", fake_njit)

    def kernel():
        return 1

    with pytest.raises(RuntimeError, match="real numba failure"):
        sift_numba.njit_optional_cache(cache=True)(kernel)
