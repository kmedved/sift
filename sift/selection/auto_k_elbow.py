"""Elbow rule for automatic k selection."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def select_k_elbow(
    objective_path: np.ndarray,
    min_k: int = 5,
    max_k: int = 100,
    min_rel_gain: float = 0.02,
    patience: int = 3,
) -> Tuple[int, pd.DataFrame]:
    """Select the prefix before a patience-confirmed run of small gains.

    ``k`` is the number of retained features. Consequently, the first feature
    in a confirmed low-gain run is excluded from the selected prefix unless
    retaining it is required by the ``min_k`` floor.

    This is the rule behind ``AutoKConfig(k_method="elbow")``, which forwards
    ``elbow_min_rel_gain`` and ``elbow_patience`` here. It reads only the
    in-sample objective curve, so it costs nothing beyond the path itself, but
    its threshold is uncalibrated: gains shrink like ``1/n_eff`` under the
    null while the denominator tracks accumulated signal, so a fixed
    ``min_rel_gain`` means different things at different ``n`` and different
    signal strengths. Treat it as a fast heuristic for a first look and prefer
    a rule that cleared the Auto-K v2 campaign -- ``select_k_chi2_stop``, or
    ``select_k_penalized_objective`` with ``objective_penalty="ebic"`` -- for
    anything load-bearing.

    Parameters
    ----------
    objective_path : ndarray of shape (L,)
        Cumulative, non-decreasing objective after each path step, typically
        the CEFS+ objective ``obj[t] = -log(1 - R^2_t) = 2 I(y; S_t)``. Must
        be one-dimensional, numeric, and entirely finite.
    min_k : int, default 5
        Floor on the returned k; non-negative integer, clamped to the
        effective ``max_k``.
    max_k : int, default 100
        Ceiling on the returned k; positive integer, clamped to
        ``len(objective_path)``.
    min_rel_gain : float, default 0.02
        Relative-gain threshold. Step ``k`` counts as small when
        ``(obj[k-1] - obj[k-2]) / max(|obj[k-2]|, 1) < min_rel_gain``. Finite
        and non-negative.
    patience : int, default 3
        Consecutive small-gain steps required to stop; positive integer. On a
        confirmed run starting at step ``k``, the selection is
        ``max(min_k, k - patience)``.

    Returns
    -------
    best_k : int
        Selected prefix length, or ``0`` when the effective ``max_k`` is
        non-positive. Falls back to the effective ``max_k`` when no run of
        ``patience`` small gains is confirmed.
    diagnostics : DataFrame
        One row per evaluated k with ``k`` (1..effective max), ``objective``,
        ``delta`` (step gain, with ``delta[0] = obj[0]``), and ``rel_gain``
        (``inf`` at ``k=1``). Empty when the effective ``max_k`` is
        non-positive.

    Raises
    ------
    ValueError
        If ``objective_path`` is not a one-dimensional numeric array or holds
        a non-finite value; if ``min_k`` is not a non-negative integer, if
        ``max_k`` or ``patience`` is not a positive integer, if
        ``min_k > max_k``, or if ``min_rel_gain`` is not finite and
        non-negative.

    See Also
    --------
    select_k_changepoint : Same shape with an empirical noise floor
        (experimental).
    select_k_chi2_stop : Calibrated sequential test on the same gain path.
    select_k_penalized_objective : Information-criterion stop on the same
        objective path.
    AutoKConfig : ``elbow_min_rel_gain`` and ``elbow_patience`` fields.

    Notes
    -----
    Scanning is ``O(L)`` on top of the path, with no resampling and no model
    fits. Because the scan starts at ``max(min_k, 2)`` and stops at the first
    confirmed run, a single large interior gain resets the counter, which is
    what keeps a masked-then-revealed signal from truncating the path early.
    ``select_k_changepoint`` was written as the calibrated replacement for
    this rule, but it did not clear the campaign's null-calibration gate
    either; both stay available as diagnostics rather than as defaults.

    Examples
    --------
    >>> import numpy as np
    >>> from sift import select_k_elbow
    >>> objective = np.array([1.0, 1.8, 2.4, 2.42, 2.43, 2.44])
    >>> best_k, diag = select_k_elbow(
    ...     objective, min_k=1, max_k=6, min_rel_gain=0.05, patience=2
    ... )
    >>> best_k
    3
    >>> print(diag[["k", "delta", "rel_gain"]].round(3).to_string(index=False))
     k  delta  rel_gain
     1   1.00       inf
     2   0.80     0.800
     3   0.60     0.333
     4   0.02     0.008
     5   0.01     0.004
     6   0.01     0.004
    """
    raw_obj = np.asarray(objective_path)
    if raw_obj.ndim != 1:
        raise ValueError("objective_path must be a one-dimensional numeric array")
    try:
        obj = raw_obj.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("objective_path must be a one-dimensional numeric array") from exc

    for name, value, allow_zero in (
        ("min_k", min_k, True),
        ("max_k", max_k, False),
        ("patience", patience, False),
    ):
        lower = 0 if allow_zero else 1
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < lower
        ):
            qualifier = "a non-negative" if allow_zero else "a positive"
            raise ValueError(f"{name} must be {qualifier} integer")
    if int(min_k) > int(max_k):
        raise ValueError("min_k must be <= max_k")
    if (
        isinstance(min_rel_gain, (bool, np.bool_))
        or not isinstance(min_rel_gain, (int, float, np.integer, np.floating))
        or not np.isfinite(float(min_rel_gain))
        or float(min_rel_gain) < 0.0
    ):
        raise ValueError("min_rel_gain must be finite and non-negative")
    if obj.size and not np.isfinite(obj).all():
        raise ValueError("objective_path must contain only finite values")

    max_k = min(int(max_k), len(obj))

    if max_k <= 0:
        return 0, pd.DataFrame()

    min_k_eff = min(int(min_k), max_k)

    delta = np.zeros_like(obj, dtype=np.float64)
    delta[0] = obj[0]
    delta[1:] = obj[1:] - obj[:-1]

    rel_gain = np.zeros_like(obj, dtype=np.float64)
    rel_gain[0] = np.inf
    denom = np.maximum(np.abs(obj[:-1]), 1.0)
    rel_gain[1:] = delta[1:] / denom

    best_k = max_k
    run = 0

    for k in range(max(min_k_eff, 2), max_k + 1):
        if rel_gain[k - 1] < min_rel_gain:
            run += 1
            if run >= patience:
                best_k = max(min_k_eff, k - int(patience))
                break
        else:
            run = 0

    diag = pd.DataFrame(
        {
            "k": np.arange(1, max_k + 1),
            "objective": obj[:max_k],
            "delta": delta[:max_k],
            "rel_gain": rel_gain[:max_k],
        }
    )

    return best_k, diag


select_k_elbow.__module__ = "sift.selection.auto_k"
