"""Period classification for SEC filing durations.

Single source of truth for the day-count thresholds that distinguish
quarterly from annual periods. Both the parser (within-filing dedup)
and the query layer (Q4 derivation, TTM, snapshots) classify periods
using these bounds.

Quarterly: 60-105 days covers standard 3-month periods including
fiscal-calendar quirks (e.g. JNJ Q4 spans Oct-Jan = ~97 days).
Annual: 340-380 days covers 12-month fiscal years including 52/53-week
calendars.
"""

import pandas as pd

Q_MIN, Q_MAX = 60, 105
A_MIN, A_MAX = 340, 380


def is_quarterly(duration_days: pd.Series) -> pd.Series:
    """Boolean mask: True where duration_days falls in the quarterly range."""
    return (duration_days >= Q_MIN) & (duration_days <= Q_MAX)


def is_annual(duration_days: pd.Series, form: pd.Series | None = None) -> pd.Series:
    """Boolean mask: True where duration_days falls in the annual range.

    If `form` is provided, additionally requires form == "10-K".
    """
    mask = (duration_days >= A_MIN) & (duration_days <= A_MAX)
    if form is not None:
        mask = mask & (form == "10-K")
    return mask
