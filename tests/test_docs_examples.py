"""Execute every fenced ``python`` block in the user-facing manuals.

Stage 3 of the 0.9 documentation campaign: *every manual code block executes in
CI*.  This module discovers the fenced code blocks in :data:`DOC_FILES`,
parametrizes one test per block, and executes each block standalone.

Contract
--------
* Every ```` ```python ```` (or ```` ```python3 ````) fence is executed in a
  fresh namespace ``{"__name__": "__sift_doc__"}`` via
  ``exec(compile(block, f"{path}:L{line}", "exec"), ns)``.
* Blocks run with the process working directory set to a temporary directory so
  examples that write files do not pollute the checkout.
* The project's warnings-as-errors configuration stays in force.  A block that
  demonstrates a warning must catch the warning itself or carry a ``skip``
  directive; the runner never relaxes the filter globally.
* Each block gets a wall-clock budget of :data:`BLOCK_TIME_BUDGET_SECONDS`,
  measured with :func:`time.perf_counter` after the block returns.  The budget
  is a report-after-the-fact check, so no signal handling is involved.

Directives
----------
An HTML comment on the line immediately above the opening fence (one blank line
between the comment and the fence is tolerated) controls execution::

    <!-- sift-doc: skip reason="needs a live database" -->
    <!-- sift-doc: requires=catboost -->
    <!-- sift-doc: continues -->

``skip``
    The block is not executed.  ``reason="..."`` is mandatory; a ``skip``
    without a reason is a test failure, not a silent pass.
``requires=<module>``
    The block runs only when ``importlib.import_module("<module>")`` succeeds;
    otherwise the test is skipped.
``continues``
    The block executes in the namespace (and working directory) of the previous
    python block in the same file.  The previous block must exist and must not
    be skipped.  A continuation inherits its predecessor's ``requires``, because
    a chain is rebuilt from its head whenever the head was deselected.

Directives may be combined on one line, e.g.
``<!-- sift-doc: continues requires=catboost -->``.

Parser notes
------------
* Fences may be indented (list/table continuation) and may use more than three
  backticks; the closing fence must use at least as many backticks and the same
  indentation.  A CommonMark info string after the language word (``` ```python
  title="x" ```) is tolerated and ignored.
* Fences that are *inside a Markdown blockquote* (lines beginning with ``>``)
  are **not** collected.  Rather than silently ignoring such a block,
  :func:`test_no_blockquoted_python_fences` fails if one ever appears, so the
  convention has to be revisited deliberately.
* Non-python fences (``bash``, ``json``, plain) are ignored, but the parser
  still tracks them so that directive comments and prose inside them are never
  mistaken for real markup.
* Blocks that only contain ``...``/``pass`` or an illustrative signature are not
  special-cased by the runner.  They are either rewritten in the manual so they
  run, or carry an explicit ``skip`` directive with a reason.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import pathlib
import re
import shlex
import time
from dataclasses import dataclass

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Documentation files whose python blocks must execute.
DOC_FILES = (
    "README.md",
    "DOCS.MD",
    "docs/API.md",
    "docs/user-guide.md",
    "docs/ADVANCED.md",
    "docs/troubleshooting.md",
    "docs/results.md",
)

#: Per-block wall-clock budget, in seconds.
BLOCK_TIME_BUDGET_SECONDS = 20.0

PYTHON_LANGS = frozenset({"python", "python3"})

# CommonMark info string: the first word is the language, anything after it is
# metadata the runner ignores. Backticks may not appear in an info string, which
# is what keeps this from matching inline code spans.
_FENCE_RE = re.compile(r"^(?P<indent>[ \t]*)(?P<ticks>`{3,})[ \t]*(?P<lang>[^`\s]*)[^`]*$")
_DIRECTIVE_RE = re.compile(r"^[ \t]*<!--[ \t]*sift-doc:(?P<body>.*?)-->[ \t]*$")
_BLOCKQUOTE_FENCE_RE = re.compile(r"^[ \t]*>[ >\t]*`{3,}[ \t]*(?P<lang>[^`\s]*)[ \t]*$")


@dataclass
class DocBlock:
    """One fenced python block, with its directives resolved."""

    relpath: str
    line: int
    source: str
    skip_reason: str | None = None
    requires: tuple[str, ...] = ()
    continues: bool = False
    error: str | None = None
    chain_id: int = 0
    chain_pos: int = 0

    @property
    def block_id(self) -> str:
        return f"{self.relpath}:L{self.line}"


def _parse_directive(block: DocBlock, body: str) -> None:
    """Apply the ``sift-doc:`` directive tokens in ``body`` to ``block``."""
    try:
        tokens = shlex.split(body)
    except ValueError as exc:  # pragma: no cover - malformed quoting
        block.error = f"malformed sift-doc directive {body!r}: {exc}"
        return

    saw_skip = False
    reason: str | None = None
    requires: list[str] = []
    for token in tokens:
        if token == "skip":
            saw_skip = True
        elif token == "continues":
            block.continues = True
        elif token.startswith("reason="):
            reason = token[len("reason=") :]
        elif token.startswith("requires="):
            requires.append(token[len("requires=") :])
        else:
            block.error = (
                f"unknown sift-doc directive token {token!r}; "
                "supported: skip, reason=..., requires=..., continues"
            )
            return

    block.requires = tuple(requires)
    if saw_skip:
        if not reason:
            block.error = 'sift-doc: skip requires a non-empty reason="..."'
            return
        block.skip_reason = reason
    elif reason is not None:
        block.error = 'sift-doc: reason="..." is only meaningful together with skip'
        return

    if block.skip_reason and block.continues:
        block.error = "sift-doc: skip and continues are mutually exclusive"


def _find_directive(lines: list[str], fence_index: int) -> str | None:
    """Return the directive body attached to the fence at ``fence_index``."""
    for offset in (1, 2):
        candidate = fence_index - offset
        if candidate < 0:
            return None
        text = lines[candidate]
        match = _DIRECTIVE_RE.match(text)
        if match:
            return match.group("body")
        if text.strip():
            return None
        # A single blank line between the comment and the fence is tolerated.
    return None


def parse_doc_blocks(relpath: str, text: str) -> list[DocBlock]:
    """Extract the python blocks of ``text`` in document order."""
    lines = text.split("\n")
    blocks: list[DocBlock] = []
    index = 0
    chain_id = 0
    chain_pos = 0
    previous_was_python = False
    previous_was_skipped = True
    previous_requires: tuple[str, ...] = ()

    while index < len(lines):
        match = _FENCE_RE.match(lines[index])
        if match is None:
            index += 1
            continue

        indent = match.group("indent")
        ticks = match.group("ticks")
        lang = match.group("lang").lower()
        close_re = re.compile(rf"^{re.escape(indent)}`{{{len(ticks)},}}[ \t]*$")

        body: list[str] = []
        cursor = index + 1
        while cursor < len(lines) and not close_re.match(lines[cursor]):
            body.append(lines[cursor])
            cursor += 1

        if lang in PYTHON_LANGS:
            prefix = len(indent)
            source = "\n".join(
                line[prefix:] if line[:prefix] == indent else line.lstrip() for line in body
            )
            block = DocBlock(relpath=relpath, line=index + 1, source=source)
            directive = _find_directive(lines, index)
            if directive is not None:
                _parse_directive(block, directive)

            if block.continues:
                if not previous_was_python:
                    block.error = "sift-doc: continues has no preceding python block"
                elif previous_was_skipped:
                    block.error = "sift-doc: continues follows a skipped block"
                else:
                    chain_pos += 1
                    # A chain is rebuilt from its head when the head was
                    # deselected, so a continuation inherits whatever optional
                    # imports the earlier blocks needed.
                    block.requires = tuple(
                        dict.fromkeys(previous_requires + block.requires)
                    )
            if not block.continues:
                chain_id += 1
                chain_pos = 0
            block.chain_id = chain_id
            block.chain_pos = chain_pos
            blocks.append(block)

            previous_was_python = True
            previous_was_skipped = bool(block.skip_reason) or block.error is not None
            previous_requires = block.requires
        else:
            previous_was_python = False
            previous_was_skipped = True
            previous_requires = ()

        index = cursor + 1

    return blocks


def _collect_all_blocks() -> dict[str, list[DocBlock]]:
    collected: dict[str, list[DocBlock]] = {}
    for relpath in DOC_FILES:
        path = ROOT / relpath
        text = path.read_text(encoding="utf8")
        collected[relpath] = parse_doc_blocks(relpath, text)
    return collected


DOC_BLOCKS = _collect_all_blocks()
ALL_BLOCKS = [block for blocks in DOC_BLOCKS.values() for block in blocks]


@dataclass
class _ChainState:
    """Live namespace + working directory shared by a ``continues`` chain."""

    namespace: dict
    workdir: pathlib.Path
    position: int = -1


_CHAIN_STATE: dict[tuple[str, int], _ChainState] = {}
_FILE_TIMINGS: dict[str, float] = {}
_FILE_COUNTS: dict[str, tuple[int, int, int]] = {}


def _fresh_namespace() -> dict:
    return {"__name__": "__sift_doc__"}


def _run_source(block: DocBlock, namespace: dict, workdir: pathlib.Path) -> float:
    previous_cwd = pathlib.Path.cwd()
    os.chdir(workdir)
    start = time.perf_counter()
    try:
        exec(compile(block.source, block.block_id, "exec"), namespace)
    finally:
        elapsed = time.perf_counter() - start
        os.chdir(previous_cwd)
    return elapsed


def _chain_blocks(relpath: str, chain_id: int) -> list[DocBlock]:
    return [b for b in DOC_BLOCKS[relpath] if b.chain_id == chain_id]


@contextlib.contextmanager
def _uncaptured(config):
    """Let terminal writes escape pytest's output capture, when possible."""
    capman = config.pluginmanager.getplugin("capturemanager")
    if capman is None:  # pragma: no cover - capture plugin disabled
        yield
        return
    with capman.global_and_fixture_disabled():
        yield


@pytest.fixture(scope="session", autouse=True)
def _doc_block_timing_report(request):
    yield
    reporter = request.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None or not _FILE_TIMINGS:  # pragma: no cover - non-terminal runs
        return
    lines = ["", "----------------------- doc block timing -----------------------"]
    total = 0.0
    for relpath in DOC_FILES:
        if relpath not in _FILE_TIMINGS:
            continue
        seconds = _FILE_TIMINGS[relpath]
        total += seconds
        executed, skipped, continued = _FILE_COUNTS.get(relpath, (0, 0, 0))
        lines.append(
            f"{relpath:<26} {seconds:6.2f}s  executed={executed} "
            f"skipped={skipped} continues={continued}"
        )
    lines.append(f"{'total':<26} {total:6.2f}s")
    with _uncaptured(request.config):
        for line in lines:
            reporter.write_line(line)


def _record(
    relpath: str,
    seconds: float,
    *,
    executed: int = 0,
    skipped: int = 0,
    continued: int = 0,
) -> None:
    _FILE_TIMINGS[relpath] = _FILE_TIMINGS.get(relpath, 0.0) + seconds
    counts = _FILE_COUNTS.get(relpath, (0, 0, 0))
    _FILE_COUNTS[relpath] = (
        counts[0] + executed,
        counts[1] + skipped,
        counts[2] + continued,
    )


@pytest.mark.parametrize(
    "block",
    [pytest.param(block, id=block.block_id) for block in ALL_BLOCKS],
)
def test_doc_block_executes(block: DocBlock, tmp_path: pathlib.Path) -> None:
    if block.error is not None:
        pytest.fail(f"{block.block_id}: {block.error}")

    if block.skip_reason is not None:
        _record(block.relpath, 0.0, skipped=1)
        pytest.skip(f"sift-doc skip: {block.skip_reason}")

    for module_name in block.requires:
        try:
            importlib.import_module(module_name)
        except ImportError:
            _record(block.relpath, 0.0, skipped=1)
            pytest.skip(f"sift-doc requires={module_name} (not installed)")

    key = (block.relpath, block.chain_id)
    if block.continues:
        state = _CHAIN_STATE.get(key)
        if state is None or state.position != block.chain_pos - 1:
            # The chain head (or an intermediate block) was deselected or
            # reordered.  Rebuild the prefix so the block still runs against a
            # faithful namespace instead of failing for an unrelated reason.
            state = _ChainState(namespace=_fresh_namespace(), workdir=tmp_path)
            for earlier in _chain_blocks(block.relpath, block.chain_id):
                if earlier.chain_pos >= block.chain_pos:
                    break
                _run_source(earlier, state.namespace, state.workdir)
                state.position = earlier.chain_pos
            _CHAIN_STATE[key] = state
    else:
        state = _ChainState(namespace=_fresh_namespace(), workdir=tmp_path)
        _CHAIN_STATE[key] = state

    elapsed = _run_source(block, state.namespace, state.workdir)
    state.position = block.chain_pos
    _record(block.relpath, elapsed, executed=1, continued=1 if block.continues else 0)

    assert elapsed <= BLOCK_TIME_BUDGET_SECONDS, (
        f"{block.block_id} took {elapsed:.1f}s, over the "
        f"{BLOCK_TIME_BUDGET_SECONDS:.0f}s per-block budget. Every block in the "
        "manuals runs in under two seconds on an idle machine, so first check "
        "that nothing else is saturating the CPU; otherwise shrink the example "
        "(smaller n/p, fewer bootstraps or estimators) or mark it with a "
        "sift-doc skip."
    )


def test_doc_files_all_exist_and_have_blocks() -> None:
    missing = [relpath for relpath in DOC_FILES if not (ROOT / relpath).is_file()]
    assert missing == [], f"DOC_FILES lists files that do not exist: {missing}"
    empty = [relpath for relpath, blocks in DOC_BLOCKS.items() if not blocks]
    assert empty == [], f"no python blocks found in {empty}; did the parser break?"


def test_no_blockquoted_python_fences() -> None:
    """Blockquoted fences are not collected, so they must not exist."""
    offenders = []
    for relpath in DOC_FILES:
        text = (ROOT / relpath).read_text(encoding="utf8")
        for number, line in enumerate(text.split("\n"), 1):
            match = _BLOCKQUOTE_FENCE_RE.match(line)
            if match and match.group("lang").lower() in PYTHON_LANGS:
                offenders.append(f"{relpath}:L{number}")
    assert offenders == [], (
        "python fences inside blockquotes are not executed by the doc runner: "
        f"{offenders}. Unindent them out of the blockquote."
    )


def test_block_ids_are_unique_and_stable() -> None:
    ids = [block.block_id for block in ALL_BLOCKS]
    assert len(ids) == len(set(ids)), "duplicate doc block ids"
    for relpath, blocks in DOC_BLOCKS.items():
        lines = [block.line for block in blocks]
        assert lines == sorted(lines), f"{relpath}: blocks must stay in document order"


def test_every_skip_directive_states_a_reason() -> None:
    """A skipped manual block must say why; a bare `skip` is a parse error."""
    unexplained = [
        block.block_id
        for block in ALL_BLOCKS
        if block.error is not None and "reason" in block.error
    ]
    assert unexplained == [], f"skip without reason in {unexplained}"


# --------------------------------------------------------------------------
# Parser unit tests. These exercise the directive grammar on synthetic
# Markdown so the real manuals never have to encode a corner case.
# --------------------------------------------------------------------------


def _parse(text: str) -> list[DocBlock]:
    return parse_doc_blocks("synthetic.md", text)


def test_parser_collects_python_and_python3_and_skips_other_languages() -> None:
    blocks = _parse(
        "intro\n"
        "```bash\n"
        "echo not python\n"
        "```\n"
        "```python\n"
        "a = 1\n"
        "```\n"
        "```python3\n"
        "b = 2\n"
        "```\n"
        "```\n"
        "plain fence\n"
        "```\n"
    )
    assert [block.line for block in blocks] == [5, 8]
    assert [block.source for block in blocks] == ["a = 1", "b = 2"]


def test_parser_dedents_indented_fences_and_matches_longer_tick_runs() -> None:
    blocks = _parse("- item:\n\n  ````python\n  a = 1\n  if a:\n      a = 2\n  ````\n")
    assert len(blocks) == 1
    assert blocks[0].source == "a = 1\nif a:\n    a = 2"


def test_parser_reads_directive_above_the_fence_with_at_most_one_blank_line() -> None:
    attached = _parse('<!-- sift-doc: skip reason="x" -->\n\n```python\na = 1\n```\n')
    assert attached[0].skip_reason == "x"

    detached = _parse('<!-- sift-doc: skip reason="x" -->\n\n\n```python\na = 1\n```\n')
    assert detached[0].skip_reason is None
    assert detached[0].error is None


def test_parser_rejects_a_skip_without_a_reason() -> None:
    (block,) = _parse("<!-- sift-doc: skip -->\n```python\na = 1\n```\n")
    assert block.error is not None
    assert "reason" in block.error


def test_parser_rejects_unknown_directive_tokens_and_stray_reasons() -> None:
    (unknown,) = _parse("<!-- sift-doc: nonsense -->\n```python\na = 1\n```\n")
    assert unknown.error is not None and "unknown sift-doc directive" in unknown.error

    (stray,) = _parse('<!-- sift-doc: reason="x" -->\n```python\na = 1\n```\n')
    assert stray.error is not None and "only meaningful together with skip" in stray.error


def test_parser_rejects_continues_without_a_usable_predecessor() -> None:
    (orphan,) = _parse("<!-- sift-doc: continues -->\n```python\na = 1\n```\n")
    assert orphan.error is not None and "no preceding python block" in orphan.error

    after_skip = _parse(
        '<!-- sift-doc: skip reason="x" -->\n'
        "```python\n"
        "a = 1\n"
        "```\n"
        "<!-- sift-doc: continues -->\n"
        "```python\n"
        "b = a\n"
        "```\n"
    )
    assert after_skip[1].error is not None
    assert "follows a skipped block" in after_skip[1].error

    after_bash = _parse(
        "```python\na = 1\n```\n"
        "```bash\necho hi\n```\n"
        "<!-- sift-doc: continues -->\n```python\nb = a\n```\n"
    )
    assert after_bash[1].error is not None
    assert "no preceding python block" in after_bash[1].error


def test_parser_chains_consecutive_continues_blocks() -> None:
    blocks = _parse(
        "```python\na = 1\n```\n"
        "<!-- sift-doc: continues -->\n```python\nb = a\n```\n"
        "<!-- sift-doc: continues -->\n```python\nc = b\n```\n"
        "```python\nd = 4\n```\n"
    )
    assert [(b.chain_id, b.chain_pos) for b in blocks] == [(1, 0), (1, 1), (1, 2), (2, 0)]


def test_parser_tolerates_info_strings_and_unclosed_fences() -> None:
    with_info = _parse('```python title="setup"\na = 1\n```\n```python\nb = 2\n```\n')
    assert [block.source for block in with_info] == ["a = 1", "b = 2"]

    # An unterminated fence runs to end-of-file instead of desynchronising.
    unclosed = _parse("```python\na = 1\nb = 2\n")
    assert len(unclosed) == 1
    assert unclosed[0].source.strip() == "a = 1\nb = 2"


def test_parser_accepts_combined_directives() -> None:
    blocks = _parse(
        "```python\na = 1\n```\n"
        "<!-- sift-doc: continues requires=catboost requires=matplotlib -->\n"
        "```python\nb = a\n```\n"
    )
    assert blocks[1].continues is True
    assert blocks[1].requires == ("catboost", "matplotlib")
    assert blocks[1].error is None


def test_parser_propagates_requires_along_a_chain() -> None:
    blocks = _parse(
        "<!-- sift-doc: requires=catboost -->\n```python\na = 1\n```\n"
        "<!-- sift-doc: continues -->\n```python\nb = a\n```\n"
        "<!-- sift-doc: continues requires=matplotlib -->\n```python\nc = b\n```\n"
    )
    assert [block.requires for block in blocks] == [
        ("catboost",),
        ("catboost",),
        ("catboost", "matplotlib"),
    ]
