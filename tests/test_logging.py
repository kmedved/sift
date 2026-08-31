"""Focused contracts for SIFT's package logging surface."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import sift


@pytest.fixture(autouse=True)
def _restore_package_verbosity():
    sift.set_verbosity("info")
    yield
    sift.set_verbosity("info")


def _small_regression() -> tuple[np.ndarray, np.ndarray]:
    X = np.arange(48, dtype=np.float64).reshape(16, 3)
    X[:, 1] = np.sin(X[:, 0])
    X[:, 2] = np.cos(X[:, 0])
    y = 2.0 * X[:, 0] - X[:, 1]
    return X, y


def _run_mrmr(*, verbose: bool | None = None) -> None:
    X, y = _small_regression()
    kwargs = {} if verbose is None else {"verbose": verbose}
    sift.select_mrmr(
        X,
        y,
        k=1,
        task="regression",
        estimator="classic",
        mrmr_backend="serial",
        subsample=None,
        **kwargs,
    )


def test_default_verbose_selector_emits_info_record(caplog):
    caplog.set_level(logging.INFO)

    _run_mrmr()

    progress = [
        record
        for record in caplog.records
        if record.name == "sift" and "mRMR classic: selecting 1 feature" in record.getMessage()
    ]
    assert len(progress) == 1
    assert progress[0].levelno == logging.INFO


def test_default_progress_is_visible_and_none_suppresses_it_without_logging_config():
    script = textwrap.dedent(
        """
        import numpy as np
        import sift

        X = np.arange(48, dtype=np.float64).reshape(16, 3)
        X[:, 1] = np.sin(X[:, 0])
        X[:, 2] = np.cos(X[:, 0])
        y = 2.0 * X[:, 0] - X[:, 1]
        sift.select_mrmr(
            X,
            y,
            k=1,
            task="regression",
            estimator="classic",
            mrmr_backend="serial",
            subsample=None,
        )
        sift.set_verbosity(None)
        sift.select_mrmr(
            X,
            y,
            k=1,
            task="regression",
            estimator="classic",
            mrmr_backend="serial",
            subsample=None,
        )
        """
    )
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(repo_root), env.get("PYTHONPATH")) if part
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert completed.stderr.count("mRMR classic: selecting 1 feature") == 1


def test_verbose_false_is_silent(caplog):
    caplog.set_level(logging.DEBUG)
    sift.set_verbosity("debug")

    _run_mrmr(verbose=False)

    assert [record for record in caplog.records if record.name == "sift"] == []


def test_set_verbosity_controls_info_debug_and_none(caplog):
    caplog.set_level(logging.DEBUG)
    package_logger = logging.getLogger("sift")

    sift.set_verbosity("info")
    package_logger.debug("hidden-debug")
    package_logger.info("shown-info")
    sift.set_verbosity("debug")
    package_logger.debug("shown-debug")
    sift.set_verbosity(None)
    package_logger.info("hidden-info")

    messages = [record.getMessage() for record in caplog.records if record.name == "sift"]
    assert messages == ["shown-info", "shown-debug"]


def test_set_verbosity_rejects_unknown_levels():
    with pytest.raises(ValueError, match="'info', 'debug', or None"):
        sift.set_verbosity("warning")  # type: ignore[arg-type]


def test_repeated_configuration_has_one_handler_and_one_record(caplog):
    caplog.set_level(logging.INFO)
    package_logger = logging.getLogger("sift")

    for _ in range(4):
        sift.set_verbosity("info")
    package_logger.info("configured-once")

    owned_handlers = [
        handler for handler in package_logger.handlers if handler.get_name() == "sift.default"
    ]
    records = [
        record
        for record in caplog.records
        if record.name == "sift" and record.getMessage() == "configured-once"
    ]
    assert len(owned_handlers) == 1
    assert owned_handlers[0].level == logging.INFO
    assert len(records) == 1
