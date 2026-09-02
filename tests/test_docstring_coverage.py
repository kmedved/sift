"""Every public export carries a substantive numpydoc docstring (0.9 DoD item 6).

Contract
--------
For every name in ``sift.__all__`` except ``__version__``:

1. **Summary.** The docstring starts with a non-empty one-line summary, and
   that summary is followed by a blank line whenever more text follows.
2. **Substance.** At least :data:`MIN_DOC_LINES` *non-empty* lines.
3. **Parameters.** Every parameter of the signature -- ``__init__``'s
   signature for a class -- appears as a numpydoc entry: functions must
   document theirs under ``Parameters``, classes may use ``Parameters`` or
   ``Attributes``.  A regular parameter needs a type (``name : type``);
   ``*args``/``**kwargs`` must appear under exactly that spelling, with or
   without a type, which is what numpydoc conventionally emits for them.
4. **Returns.** Every exported *function* has a ``Returns`` or ``Yields``
   section.  Classes are exempt.
5. **Examples.** Every export has an ``Examples`` section, and that section
   contains at least one runnable ``>>>`` statement -- unless the export is
   one of :data:`LITERAL_EXAMPLE_EXPORTS`, the audited handful whose example
   demonstrates an optional dependency that SIFT's own test environment does
   not install and is therefore written as an indented literal block.

``tests/test_docstring_examples.py`` is the other half of the gate: it
*executes* the ``>>>`` statements this module only counts.

The parser deliberately understands nothing but numpydoc's underline headers
(``Parameters`` on one line, ``-----`` on the next) and its ``name : type``
entry lines, so it stays readable and does not need numpydoc installed.
"""

from __future__ import annotations

import inspect

import pytest

import sift

#: Minimum number of non-empty docstring lines for a public export.
MIN_DOC_LINES = 8

#: Exports whose ``Examples`` section is an indented literal block instead of a
#: doctest, because it demonstrates an optional dependency that SIFT's own test
#: environment does not install. Every entry must name that dependency in the
#: section text; the list is asserted exactly, so a new export cannot join it
#: silently.
LITERAL_EXAMPLE_EXPORTS = frozenset(
    {
        "select_boruta_shap",
        "catboost_select",
        "catboost_regression",
        "catboost_classif",
    }
)

#: Optional dependencies a literal example block is allowed to depend on.
OPTIONAL_DEPENDENCIES = ("catboost", "shap", "category_encoders")

PUBLIC_NAMES = [name for name in sift.__all__ if name != "__version__"]


def _is_underline(line: str, header: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and set(stripped) == {"-"} and len(stripped) >= len(header)


def numpydoc_sections(doc: str) -> dict[str, str]:
    """Return ``{section name: body}`` for the numpydoc sections of ``doc``.

    A section header is an unindented, non-empty line followed by a line of at
    least as many dashes.  ``doc`` is expected to come from
    :func:`inspect.getdoc`, which has already dedented it.
    """
    lines = doc.splitlines()
    starts: list[tuple[int, str]] = []
    for index in range(len(lines) - 1):
        header = lines[index]
        if not header or header[0].isspace() or not header.strip():
            continue
        if _is_underline(lines[index + 1], header.strip()):
            starts.append((index, header.strip()))

    sections: dict[str, str] = {}
    for position, (index, name) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        sections[name] = "\n".join(lines[index + 2 : end])
    return sections


def parameter_entries(body: str) -> tuple[set[str], set[str]]:
    """Split a section body into ``(typed, untyped)`` numpydoc entry names.

    An entry is an unindented line in the section body.  ``name : type`` is a
    typed entry, a bare ``name`` (what numpydoc conventionally writes for
    ``*args``/``**kwargs``) is an untyped one.  Comma-separated heads introduce
    several names at once.
    """
    typed: set[str] = set()
    untyped: set[str] = set()
    for line in body.splitlines():
        if not line.strip() or line[0].isspace():
            continue
        head, separator, rest = line.partition(":")
        target = typed if separator and rest.strip() else untyped
        for part in head.split(","):
            name = part.strip()
            if name:
                target.add(name)
    return typed, untyped


def expected_parameters(obj: object) -> dict[str, bool]:
    """Return ``{documented spelling: needs a type}`` for ``obj``'s signature."""
    target = obj.__init__ if inspect.isclass(obj) else obj
    expected: dict[str, bool] = {}
    for parameter in inspect.signature(target).parameters.values():
        if parameter.name in {"self", "cls"}:
            continue
        if parameter.kind is parameter.VAR_POSITIONAL:
            expected["*" + parameter.name] = False
        elif parameter.kind is parameter.VAR_KEYWORD:
            expected["**" + parameter.name] = False
        else:
            expected[parameter.name] = True
    return expected


def example_statements(body: str) -> list[str]:
    """Return the runnable ``>>>`` statement sources inside an ``Examples`` body.

    Statements carrying ``# doctest: +SKIP`` never count, and a section whose
    first statement is skipped counts as having none at all, because the
    docstring-examples runner then skips the whole case.
    """
    import doctest

    examples = doctest.DocTestParser().get_examples(body)
    if examples and examples[0].options.get(doctest.SKIP):
        return []
    return [
        example.source for example in examples if not example.options.get(doctest.SKIP)
    ]


@pytest.mark.parametrize("name", PUBLIC_NAMES)
def test_public_export_has_numpydoc_docstring(name: str) -> None:
    obj = getattr(sift, name)
    doc = inspect.getdoc(obj) or ""
    lines = doc.splitlines()

    assert lines and lines[0].strip(), f"{name}: docstring has no one-line summary"
    if len(lines) > 1:
        assert not lines[1].strip(), (
            f"{name}: the one-line summary must be followed by a blank line, "
            f"got {lines[1]!r}"
        )

    non_empty = [line for line in lines if line.strip()]
    assert len(non_empty) >= MIN_DOC_LINES, (
        f"{name}: docstring has {len(non_empty)} non-empty lines, "
        f"fewer than the {MIN_DOC_LINES} a public export must carry"
    )

    sections = numpydoc_sections(doc)
    is_class = inspect.isclass(obj)
    assert is_class or callable(obj), f"{name}: export is neither a class nor callable"

    typed: set[str] = set()
    untyped: set[str] = set()
    for section in ("Parameters", "Attributes") if is_class else ("Parameters",):
        section_typed, section_untyped = parameter_entries(sections.get(section, ""))
        typed |= section_typed
        untyped |= section_untyped

    expected = expected_parameters(obj)
    missing = sorted(key for key in expected if key not in typed | untyped)
    where = "Parameters/Attributes" if is_class else "Parameters"
    assert not missing, f"{name}: {where} is missing {missing}"
    untyped_but_should_be = sorted(
        key for key, needs_type in expected.items() if needs_type and key not in typed
    )
    assert not untyped_but_should_be, (
        f"{name}: {untyped_but_should_be} are documented without a type; "
        "numpydoc entries read 'name : type'"
    )

    if not is_class:
        assert "Returns" in sections or "Yields" in sections, (
            f"{name}: exported functions need a Returns or Yields section"
        )

    assert "Examples" in sections, f"{name}: docstring has no Examples section"
    statements = example_statements(sections["Examples"])
    if name in LITERAL_EXAMPLE_EXPORTS:
        assert not statements, (
            f"{name}: the Examples section now has runnable statements, so drop it "
            "from LITERAL_EXAMPLE_EXPORTS and let the doctest runner execute it"
        )
    else:
        assert statements, (
            f"{name}: the Examples section has no runnable '>>>' statement (a leading "
            "'# doctest: +SKIP' skips the whole case). Write a "
            "doctest, or -- only when it needs an optional dependency -- add the "
            "export to LITERAL_EXAMPLE_EXPORTS"
        )


def test_literal_example_exports_are_declared_and_justified() -> None:
    """The non-executed example blocks are exactly the declared, justified ones."""
    literal = set()
    for name in PUBLIC_NAMES:
        sections = numpydoc_sections(inspect.getdoc(getattr(sift, name)) or "")
        body = sections.get("Examples")
        if body is not None and not example_statements(body):
            literal.add(name)

    assert literal == set(LITERAL_EXAMPLE_EXPORTS), (
        "the set of exports whose Examples section is a non-executed literal block "
        f"drifted from LITERAL_EXAMPLE_EXPORTS: unexpected={sorted(literal - LITERAL_EXAMPLE_EXPORTS)}, "
        f"stale={sorted(LITERAL_EXAMPLE_EXPORTS - literal)}"
    )

    for name in sorted(literal):
        body = numpydoc_sections(inspect.getdoc(getattr(sift, name)))["Examples"]
        assert any(dependency in body for dependency in OPTIONAL_DEPENDENCIES), (
            f"{name}: a literal example block must say which optional dependency "
            f"keeps it from running; expected one of {OPTIONAL_DEPENDENCIES}"
        )
        assert any(line.startswith("    ") and line.strip() for line in body.splitlines()), (
            f"{name}: a literal example block must actually contain indented code"
        )


def test_numpydoc_section_parser_reads_headers_and_entries() -> None:
    """The parser itself, on a docstring with every shape it has to handle."""
    doc = inspect.cleandoc(
        """
        Summary line.

        Parameters
        ----------
        a, b : int
            Two names on one entry.
        bare
            An entry with no type, as numpydoc writes for var-args.
        **kwargs
            Passed through.

        Returns
        -------
        int
            Not an entry name we care about.

        Examples
        --------
        >>> 1 + 1
        2
        """
    )
    sections = numpydoc_sections(doc)

    assert set(sections) == {"Parameters", "Returns", "Examples"}
    typed, untyped = parameter_entries(sections["Parameters"])
    assert typed == {"a", "b"}
    assert untyped == {"bare", "**kwargs"}
    assert example_statements(sections["Examples"]) == ["1 + 1\n"]
    assert example_statements(sections["Returns"]) == []
