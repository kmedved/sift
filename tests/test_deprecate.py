import inspect
import warnings

import pytest

from sift._deprecate import deprecated_alias, warn_deprecated


def _deprecated_api():
    warn_deprecated("legacy_api")


def test_warn_deprecated_uses_future_warning_and_points_to_caller():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call_line = inspect.currentframe().f_lineno + 1
        _deprecated_api()

    assert len(caught) == 1
    warning = caught[0]
    assert warning.category is FutureWarning
    assert str(warning.message) == (
        "legacy_api is deprecated since SIFT 0.9 and will be removed in SIFT 1.0."
    )
    assert warning.filename == __file__
    assert warning.lineno == call_line


def test_deprecated_alias_warns_once_per_call_and_forwards_exactly():
    calls = []
    positional = object()
    keyword = object()
    result = object()

    def target(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    alias = deprecated_alias(
        "legacy_alias",
        target,
        since="0.9.1",
        removal="1.1",
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first_call_line = inspect.currentframe().f_lineno + 1
        first = alias(positional, flag=keyword)
        second_call_line = inspect.currentframe().f_lineno + 1
        second = alias(positional, flag=keyword)

    assert first is result
    assert second is result
    assert calls == [
        ((positional,), {"flag": keyword}),
        ((positional,), {"flag": keyword}),
    ]
    assert len(caught) == 2
    assert all(warning.category is FutureWarning for warning in caught)
    assert [(warning.filename, warning.lineno) for warning in caught] == [
        (__file__, first_call_line),
        (__file__, second_call_line),
    ]
    assert all(
        str(warning.message)
        == (
            "legacy_alias is deprecated since SIFT 0.9.1 and will be removed "
            "in SIFT 1.1."
        )
        for warning in caught
    )


def test_deprecated_alias_preserves_target_metadata_and_error():
    error = RuntimeError("target failed")

    def target(required, *, option=None):
        """Target documentation."""
        raise error

    alias = deprecated_alias("legacy_alias", target)

    assert alias.__wrapped__ is target
    assert alias.__name__ == target.__name__
    assert alias.__doc__ == target.__doc__
    assert inspect.signature(alias) == inspect.signature(target)

    with pytest.warns(FutureWarning, match="legacy_alias") as caught:
        with pytest.raises(RuntimeError, match="target failed") as raised:
            alias("value", option="forwarded")

    assert len(caught) == 1
    assert raised.value is error
