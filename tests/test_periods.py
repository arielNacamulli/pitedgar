"""Boundary tests for pitedgar.periods."""

import pandas as pd

from pitedgar.periods import A_MAX, A_MIN, Q_MAX, Q_MIN, is_annual, is_quarterly


def test_is_quarterly_inclusive_bounds():
    s = pd.Series([Q_MIN - 1, Q_MIN, 91, Q_MAX, Q_MAX + 1])
    result = is_quarterly(s).tolist()
    assert result == [False, True, True, True, False]


def test_is_quarterly_handles_negative_and_zero():
    s = pd.Series([-1, 0, 30])
    assert is_quarterly(s).tolist() == [False, False, False]


def test_is_annual_inclusive_bounds_no_form():
    s = pd.Series([A_MIN - 1, A_MIN, 365, A_MAX, A_MAX + 1])
    result = is_annual(s).tolist()
    assert result == [False, True, True, True, False]


def test_is_annual_with_form_requires_10k():
    duration = pd.Series([365, 365, 365])
    form = pd.Series(["10-K", "10-Q", "10-K/A"])
    assert is_annual(duration, form).tolist() == [True, False, False]


def test_is_annual_with_form_still_checks_duration():
    duration = pd.Series([100, 365, 400])
    form = pd.Series(["10-K", "10-K", "10-K"])
    assert is_annual(duration, form).tolist() == [False, True, False]


def test_jnj_q4_at_upper_quarterly_bound():
    """Regression: JNJ Q4 spans ~97-105 days. Must classify as quarterly."""
    s = pd.Series([97, 101, 105])
    assert is_quarterly(s).all()
