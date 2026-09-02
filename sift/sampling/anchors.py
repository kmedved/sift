import numpy as np
import pandas as pd
from typing import List, Optional, Union

from sift.sampling.smart import SmartSamplerConfig


def no_anchors(df: pd.DataFrame, group_col: str, time_col: str) -> np.ndarray:
    """No anchor points."""
    return np.zeros(len(df), dtype=bool)


def first_per_group(df: pd.DataFrame, group_col: str, time_col: str) -> np.ndarray:
    """Anchor the first observation per group (by time if available)."""
    if group_col is None:
        return np.zeros(len(df), dtype=bool)

    if time_col and time_col in df.columns:
        df_sorted = df.sort_values([group_col, time_col], kind="mergesort")
        mask = ~df_sorted.duplicated(subset=[group_col], keep='first')
        return mask.reindex(df.index).values
    else:
        return (~df.duplicated(subset=[group_col], keep='first')).values


def first_and_last_per_group(df: pd.DataFrame, group_col: str, time_col: str) -> np.ndarray:
    """Anchor first and last observation per group (by time if available)."""
    if group_col is None:
        return np.zeros(len(df), dtype=bool)

    if time_col and time_col in df.columns:
        df_sorted = df.sort_values([group_col, time_col], kind="mergesort")
        first = ~df_sorted.duplicated(subset=[group_col], keep='first')
        last = ~df_sorted.duplicated(subset=[group_col], keep='last')
        mask = first | last
        return mask.reindex(df.index).values
    else:
        first = ~df.duplicated(subset=[group_col], keep='first')
        last = ~df.duplicated(subset=[group_col], keep='last')
        return (first | last).values


def periodic_anchors(period_cols: Union[str, List[str]], keep_first: bool = True):
    """
    Factory for anchoring first observation per (group, period).

    Parameters
    ----------
    period_cols : str or list of str
        Column(s) defining the period. Can be a single column like 'month'
        or multiple columns like ['month', 'year'].
    keep_first : bool
        If True, keep first observation per period. If False, keep last.

    Examples
    --------
    periodic_anchors('month')  # First row per group per month
    periodic_anchors(['season', 'month'])  # First row per group per season-month
    """
    # Normalize to list
    if isinstance(period_cols, str):
        period_cols = [period_cols]

    def anchor_fn(df: pd.DataFrame, group_col: str, time_col: str) -> np.ndarray:
        if group_col is None:
            return np.zeros(len(df), dtype=bool)

        # Check all period columns exist
        missing = [c for c in period_cols if c not in df.columns]
        if missing:
            return np.zeros(len(df), dtype=bool)

        subset = [group_col] + period_cols
        if time_col and time_col in df.columns:
            sort_cols = subset + [time_col]
            df_sorted = df.sort_values(sort_cols, kind="mergesort")
            mask = ~df_sorted.duplicated(subset=subset, keep='first' if keep_first else 'last')
            return mask.reindex(df.index).values

        return (~df.duplicated(subset=subset, keep='first' if keep_first else 'last')).values

    return anchor_fn


def quantile_anchors(
    value_col: str,
    quantile: float = 0.95,
    per_group: bool = True,
    subgroup_cols: Optional[List[str]] = None
):
    """
    Factory for anchoring rows above a quantile threshold.

    Parameters
    ----------
    value_col : str
        Column containing values to compute quantile on.
    quantile : float
        Quantile threshold (0-1). Rows at or above this quantile are anchored.
    per_group : bool
        If True, compute quantile within each group. If False, global quantile.
    subgroup_cols : list of str, optional
        Additional columns to group by when computing quantiles.
        E.g., ['season'] computes quantile per (group, season).

    Examples
    --------
    quantile_anchors('usage', 0.95)  # Top 5% usage per group
    quantile_anchors('poss_pct', 0.90, subgroup_cols=['season'])  # Top 10% per group-season
    """
    def anchor_fn(df: pd.DataFrame, group_col: str, time_col: str) -> np.ndarray:
        if value_col not in df.columns:
            return np.zeros(len(df), dtype=bool)

        if per_group and group_col:
            # Build groupby columns
            groupby_cols = [group_col]
            if subgroup_cols:
                # Filter to columns that exist
                valid_subgroups = [c for c in subgroup_cols if c in df.columns]
                groupby_cols.extend(valid_subgroups)

            threshold = df.groupby(groupby_cols)[value_col].transform(
                "quantile", quantile
            )
        else:
            threshold = df[value_col].quantile(quantile)

        return (df[value_col] >= threshold).values

    return anchor_fn




def event_window_anchors(event_col: str, window: int = 3):
    """
    Factory for anchoring rows within a window of an event.

    Example: event_window_anchors('team_change', 3) anchors ±3 rows around changes.
    """
    def anchor_fn(df: pd.DataFrame, group_col: str, time_col: str) -> np.ndarray:
        if event_col not in df.columns:
            return np.zeros(len(df), dtype=bool)

        mask = np.zeros(len(df), dtype=bool)

        if group_col is None:
            event_idx = np.where(df[event_col].values == 1)[0]
            for idx in event_idx:
                lo = max(0, idx - window)
                hi = min(len(df), idx + window + 1)
                mask[lo:hi] = True
        else:
            for _, g_idx in df.groupby(group_col, sort=False).indices.items():
                if len(g_idx) == 0:
                    continue

                # Sort by time within group if available
                if time_col and time_col in df.columns:
                    order = np.argsort(df[time_col].iloc[g_idx].values)
                    g_sorted = g_idx[order]
                else:
                    g_sorted = g_idx

                event_local = np.where(df[event_col].iloc[g_sorted].values == 1)[0]
                for loc in event_local:
                    lo = max(0, loc - window)
                    hi = min(len(g_sorted), loc + window + 1)
                    mask[g_sorted[lo:hi]] = True

        return mask

    return anchor_fn


def combine_anchors(*anchor_fns):
    """Combine multiple anchor functions with OR logic."""
    def combined(df: pd.DataFrame, group_col: str, time_col: str) -> np.ndarray:
        mask = np.zeros(len(df), dtype=bool)
        for fn in anchor_fns:
            mask |= fn(df, group_col, time_col)
        return mask
    return combined


# =============================================================================
# Domain-Specific Presets
# =============================================================================


def panel_config(
    group_col: str,
    time_col: Optional[str] = None,
    sample_frac: float = 0.15
) -> SmartSamplerConfig:
    """Generic preset for panel/longitudinal data.

    Builds a :class:`~sift.SmartSamplerConfig` for repeated observations of the
    same entities: rows are grouped by ``group_col``, at least two rows per
    group survive, and the first and last observation of every group are
    anchored so each entity keeps its endpoints instead of being reduced to a
    random interior slice. Use it whenever ``smart_sample`` runs on
    entity-over-time data (customers, patients, tickers, players); use
    :func:`cross_section_config` when rows are independent.

    Parameters
    ----------
    group_col : str
        Column naming the entity each row belongs to.
    time_col : str, optional
        Column ordering rows inside a group. When given, "first" and "last"
        anchors follow that order; when omitted, they fall back to the frame's
        own row order.
    sample_frac : float, default=0.15
        Target fraction of rows to keep, a finite float in ``(0, 1]``. Anchors
        and the per-group minimum can push the realized fraction above it.

    Returns
    -------
    SmartSamplerConfig
        A config with ``sample_frac``, ``group_col`` and ``time_col`` set from
        the arguments, ``anchor_fn=first_and_last_per_group`` and
        ``min_per_group=2``. Every other field keeps the dataclass default:
        ``pilot_sample_size=50_000``, ``leverage_batch_size=200_000``,
        ``svd_sample_size=None``, ``weight_clip_quantile=0.99``,
        ``residual_weight_cap=0.4``, ``uniform_floor=0.05``,
        ``anchor_max_share=0.4``, ``random_state=42`` and ``verbose=True``.
        The returned config is a mutable dataclass, so override any of those
        in place or through ``dataclasses.replace``.

    Raises
    ------
    TypeError
        If ``group_col`` or ``time_col`` is not a string or None, raised by the
        config's own validation.
    ValueError
        If ``sample_frac`` is not a finite value in ``(0, 1]``.

    See Also
    --------
    cross_section_config : Preset for independent, non-grouped rows.
    sift.smart_sample : Consumer of this config.
    sift.SmartSamplerConfig : The dataclass and its full field list.

    Notes
    -----
    ``smart_sample`` combines leverage-based geometric sampling with
    residual-based hard-case detection and attaches an approximate
    inverse-probability ``sample_weight`` column, so downstream selectors
    should pass those weights on. Anchored rows are kept with probability one
    but are capped at ``anchor_max_share`` of each group's quota. Keyword
    overrides passed to ``smart_sample`` win over the config's fields.

    Examples
    --------
    >>> from sift import panel_config
    >>> config = panel_config("entity_id", "date", sample_frac=0.25)
    >>> config.sample_frac, config.group_col, config.time_col
    (0.25, 'entity_id', 'date')
    >>> config.min_per_group, config.anchor_fn.__name__
    (2, 'first_and_last_per_group')
    """
    return SmartSamplerConfig(
        sample_frac=sample_frac,
        group_col=group_col,
        time_col=time_col,
        anchor_fn=first_and_last_per_group,
        min_per_group=2,
    )


def cross_section_config(sample_frac: float = 0.15) -> SmartSamplerConfig:
    """Preset for cross-sectional (non-grouped) data.

    Builds a :class:`~sift.SmartSamplerConfig` for independent rows: no entity
    column, no time ordering and no anchoring, so ``smart_sample`` treats the
    whole frame as one group and keeps rows purely on leverage and residual
    evidence. Use it for one-row-per-subject tables; use
    :func:`panel_config` when the same entity appears repeatedly, because
    ignoring that structure lets a large entity crowd the sample.

    Parameters
    ----------
    sample_frac : float, default=0.15
        Target fraction of rows to keep, a finite float in ``(0, 1]``. The
        realized fraction can exceed it slightly, because deterministic top-up
        enforces the per-group minimum.

    Returns
    -------
    SmartSamplerConfig
        A config with ``sample_frac`` set from the argument, ``group_col``,
        ``time_col`` and ``anchor_fn`` all None, and ``min_per_group=1``.
        Every other field keeps the dataclass default:
        ``pilot_sample_size=50_000``, ``leverage_batch_size=200_000``,
        ``svd_sample_size=None``, ``weight_clip_quantile=0.99``,
        ``residual_weight_cap=0.4``, ``uniform_floor=0.05``,
        ``anchor_max_share=0.4``, ``random_state=42`` and ``verbose=True``.
        The returned config is a mutable dataclass, so override any of those
        in place or through ``dataclasses.replace``.

    Raises
    ------
    ValueError
        If ``sample_frac`` is not a finite value in ``(0, 1]``.
    TypeError
        If ``sample_frac`` is not a real number, raised by the config's own
        validation.

    See Also
    --------
    panel_config : Preset for grouped, repeated-observation data.
    sift.smart_sample : Consumer of this config.
    sift.SmartSamplerConfig : The dataclass and its full field list.

    Notes
    -----
    ``smart_sample`` attaches an approximate inverse-probability
    ``sample_weight`` column, so downstream selectors should pass those weights
    on. With no ``anchor_fn`` the ``anchor_max_share`` field is inert here.
    Keyword overrides passed to ``smart_sample`` win over the config's fields.

    Examples
    --------
    >>> from sift import cross_section_config
    >>> config = cross_section_config(sample_frac=0.2)
    >>> config.sample_frac, config.group_col, config.time_col
    (0.2, None, None)
    >>> config.min_per_group, config.anchor_fn is None
    (1, True)
    """
    return SmartSamplerConfig(
        sample_frac=sample_frac,
        group_col=None,
        time_col=None,
        anchor_fn=None,
        min_per_group=1,
    )


def financial_config(
    ticker_col: str = 'ticker',
    date_col: str = 'date',
    sample_frac: float = 0.15
) -> SmartSamplerConfig:
    """Preset for financial panel data (stock returns, etc.)."""
    return SmartSamplerConfig(
        sample_frac=sample_frac,
        group_col=ticker_col,
        time_col=date_col,
        anchor_fn=combine_anchors(
            first_and_last_per_group,
            periodic_anchors('year'),
        ),
        min_per_group=5,
    )


def medical_config(
    patient_col: str = 'patient_id',
    time_col: Optional[str] = 'visit_date',
    sample_frac: float = 0.20
) -> SmartSamplerConfig:
    """Preset for medical/clinical longitudinal data."""
    return SmartSamplerConfig(
        sample_frac=sample_frac,
        group_col=patient_col,
        time_col=time_col,
        anchor_fn=first_and_last_per_group,
        min_per_group=2,
        # More conservative for medical data
        residual_weight_cap=0.3,
        weight_clip_quantile=0.95,
    )
