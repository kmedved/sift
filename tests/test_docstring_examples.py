"""Execute the ``Examples`` doctests of every public export (0.9 DoD item 6).

``tests/test_docstring_coverage.py`` proves every export *has* an ``Examples``
section with at least one ``>>>`` statement.  This module proves the statements
actually run.

Contract
--------
* One test case per export in ``sift.__all__`` (minus ``__version__``) that has
  an ``Examples`` section, plus one per public method or property of an
  exported class when the method is defined inside the ``sift`` package and has
  its own ``Examples`` section.  The parametrization id is the export name, or
  ``Class.method``.
* The section is parsed with :class:`doctest.DocTestParser` and its statements
  are executed **in order, in one fresh namespace** per case, with the working
  directory set to a per-test temporary directory.
* Execution only.  Printed output is **not** compared: NumPy 2 changed scalar
  reprs (``np.float64(1.0)`` versus ``1.0``) and pandas keeps adjusting frame
  formatting, so output comparison would fail across the supported version
  matrix for reasons that have nothing to do with the example being correct.
  A raised exception fails the case, and so does a warning, because the suite
  runs with ``filterwarnings = ["error", ...]``.
* A statement whose expected output is a traceback is executed inside
  :func:`pytest.raises`, so "this call raises" stays a real assertion.
* Skips, and nothing else, are allowed to hide an example:

  - a statement carrying ``# doctest: +SKIP`` is not executed, and when the
    *first* statement carries it the whole case is skipped;
  - a case whose example needs the optional ``catboost`` extra is skipped when
    the extra is missing, via :func:`pytest.importorskip`;
  - a case whose ``Examples`` section is an illustrative literal block with no
    ``>>>`` statement at all is skipped.  That set is pinned exactly by
    ``tests/test_docstring_coverage.py``'s ``LITERAL_EXAMPLE_EXPORTS``, so it
    cannot grow unnoticed.

Keeping the examples cheap is part of the contract: they are documentation, so
each one should stay well under a second (tiny ``n``/``p``, ``random_state=0``,
``verbose=False``, small forests).
"""

from __future__ import annotations

import doctest
import inspect
from dataclasses import dataclass

import pytest

import sift
from tests.test_docstring_coverage import PUBLIC_NAMES, numpydoc_sections

#: Tokens that mark an example as needing the optional ``catboost`` extra.
CATBOOST_MARKERS = ("import catboost", "catboost_", "catboost.", "CatBoost")


@dataclass(frozen=True, eq=False)
class ExampleCase:
    """One export's (or method's) ``Examples`` section, ready to execute."""

    label: str
    body: str
    examples: list[doctest.Example]

    @property
    def needs_catboost(self) -> bool:
        return any(marker in self.body for marker in CATBOOST_MARKERS)

    @property
    def skipped_first(self) -> bool:
        return bool(self.examples) and bool(self.examples[0].options.get(doctest.SKIP))


def _examples_body(obj: object) -> str | None:
    doc = inspect.getdoc(obj)
    if not doc:
        return None
    return numpydoc_sections(doc).get("Examples")


def _make_case(label: str, obj: object) -> ExampleCase | None:
    body = _examples_body(obj)
    if body is None:
        return None
    return ExampleCase(label, body, doctest.DocTestParser().get_examples(body))


def _public_sift_members(name: str, cls: type):
    """Yield ``(label, member)`` for public members ``cls`` defines in ``sift``."""
    for attr in sorted(dir(cls)):
        if attr.startswith("_"):
            continue
        try:
            member = getattr(cls, attr)
        except AttributeError:  # pragma: no cover - dataclass field without default
            continue
        function = member.fget if isinstance(member, property) else member
        module = getattr(function, "__module__", None) or ""
        if not module.startswith("sift") or not inspect.isroutine(function):
            continue
        yield f"{name}.{attr}", member


def collect_cases() -> list[ExampleCase]:
    """Every export, then every documented public method of exported classes."""
    cases: list[ExampleCase] = []
    for name in PUBLIC_NAMES:
        obj = getattr(sift, name)
        case = _make_case(name, obj)
        if case is not None:
            cases.append(case)
        if inspect.isclass(obj):
            for label, member in _public_sift_members(name, obj):
                member_case = _make_case(label, member)
                if member_case is not None:
                    cases.append(member_case)
    return cases


CASES = collect_cases()


def execute_case(case: ExampleCase) -> int:
    """Run ``case``'s statements in one fresh namespace; return how many ran.

    Output is never compared.  A statement documented as raising is executed
    inside :func:`pytest.raises` and its exception type is checked against the
    documented one; any other exception -- a warning included, because the
    suite turns warnings into errors -- fails the case.
    """
    namespace: dict[str, object] = {"__name__": "__sift_docstring__"}
    executed = 0
    for position, example in enumerate(case.examples, start=1):
        if example.options.get(doctest.SKIP):
            continue
        where = f"{case.label} statement {position}"
        code = compile(example.source, f"<docstring:{case.label}>", "exec")
        executed += 1
        if example.exc_msg:
            documented = example.exc_msg.split(":", 1)[0].strip()
            with pytest.raises(Exception) as excinfo:
                exec(code, namespace)  # noqa: S102 - executing documentation on purpose
            raised = type(excinfo.value).__name__
            assert documented == raised or documented.endswith("." + raised), (
                f"{where}: documented {documented}, raised {raised}"
            )
            continue
        try:
            exec(code, namespace)  # noqa: S102 - executing documentation on purpose
        except Exception as exc:
            raise AssertionError(
                f"{where} failed:\n{example.source.rstrip()}\n{type(exc).__name__}: {exc}"
            ) from exc
    return executed


@pytest.mark.parametrize("case", CASES, ids=[case.label for case in CASES])
def test_docstring_example_executes(case: ExampleCase, tmp_path, monkeypatch) -> None:
    if not case.examples:
        pytest.skip(
            f"{case.label}: Examples section is an illustrative literal block with "
            "no '>>>' statement (pinned by LITERAL_EXAMPLE_EXPORTS)"
        )
    if case.skipped_first:
        pytest.skip(f"{case.label}: Examples section opens with '# doctest: +SKIP'")
    if case.needs_catboost:
        pytest.importorskip("catboost")

    monkeypatch.chdir(tmp_path)
    assert execute_case(case) > 0, f"{case.label}: nothing was executed"


def test_every_export_with_examples_has_a_case() -> None:
    """The parametrization covers every export, and never twice."""
    labels = [case.label for case in CASES]
    assert len(labels) == len(set(labels)), "duplicate docstring-example case ids"

    export_labels = {label for label in labels if "." not in label}
    documented = {
        name for name in PUBLIC_NAMES if _examples_body(getattr(sift, name)) is not None
    }
    assert export_labels == documented
    # test_docstring_coverage.py already asserts `documented` is every export;
    # this is the cheap cross-check that the two collectors agree.
    assert export_labels == set(PUBLIC_NAMES)


def _synthetic(body: str) -> ExampleCase:
    return ExampleCase("probe", body, doctest.DocTestParser().get_examples(body))


def test_runner_fails_on_an_example_that_raises() -> None:
    """The runner is falsifiable: a broken example is a failure, not a pass."""
    with pytest.raises(AssertionError, match="probe statement 1 failed"):
        execute_case(_synthetic(">>> 1 / 0\n"))


def test_runner_fails_on_an_example_that_warns() -> None:
    """The suite's warnings-as-errors filter reaches inside the executed example."""
    with pytest.raises(AssertionError, match="UserWarning"):
        execute_case(
            _synthetic(">>> import warnings\n>>> warnings.warn('boom', UserWarning)\n")
        )


def test_runner_treats_a_documented_traceback_as_an_assertion() -> None:
    case = _synthetic(
        ">>> 1 / 0\nTraceback (most recent call last):\n"
        "ZeroDivisionError: division by zero\n"
    )
    assert execute_case(case) == 1

    mismatched = _synthetic(
        ">>> 1 / 0\nTraceback (most recent call last):\nValueError: nope\n"
    )
    with pytest.raises(AssertionError, match="documented ValueError, raised"):
        execute_case(mismatched)

    never_raises = _synthetic(
        ">>> 1 + 1\nTraceback (most recent call last):\nValueError: nope\n"
    )
    with pytest.raises(BaseException, match="DID NOT RAISE"):
        execute_case(never_raises)


def test_runner_honours_an_inline_skip_marker() -> None:
    case = _synthetic(">>> 1 + 1\n2\n>>> 1 / 0  # doctest: +SKIP\n")
    assert not case.skipped_first
    assert execute_case(case) == 1

    leading = _synthetic(">>> 1 / 0  # doctest: +SKIP\n")
    assert leading.skipped_first


def test_runner_shares_one_namespace_across_the_statements() -> None:
    case = _synthetic(">>> value = 6 * 7\n>>> value\n42\n")
    assert execute_case(case) == 2


def test_catboost_detection_reads_the_whole_section() -> None:
    assert _synthetic(">>> import catboost\n").needs_catboost
    assert _synthetic(">>> sift.catboost_select(X, y, k=3)\n").needs_catboost
    assert not _synthetic(">>> sift.select_mrmr(X, y, k=3)\n").needs_catboost


def test_case_collection_finds_the_documented_class_methods() -> None:
    """Public methods of exported classes are collected, inherited noise is not."""
    method_labels = {case.label for case in CASES if "." in case.label}

    assert "SelectionView.to_dict" in method_labels
    assert "KnockoffSelectionResult.result_view" in method_labels
    # sklearn's own descriptors and inherited estimator plumbing stay out.
    assert not any(label.endswith(".set_fit_request") for label in method_labels)
    assert not any(label.endswith(".get_params") for label in method_labels)
