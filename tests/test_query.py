"""Tests for query.py"""

from pathlib import Path

import pandas as pd
import pytest

from pitedgar.query import PitQuery

CONCEPT = "us-gaap:Revenues"


@pytest.fixture
def parquet_path(tmp_path):
    records = [
        {"ticker": "AAPL", "concept": CONCEPT, "end": "2021-12-31", "filed": "2022-01-28", "val": 365817000000.0, "form": "10-K", "accn": "A1"},
        {"ticker": "AAPL", "concept": CONCEPT, "end": "2022-12-31", "filed": "2023-02-02", "val": 394328000000.0, "form": "10-K", "accn": "A2"},
        {"ticker": "AAPL", "concept": CONCEPT, "end": "2022-06-25", "filed": "2022-07-29", "val": 82959000000.0,  "form": "10-Q", "accn": "A3"},
        {"ticker": "MSFT", "concept": CONCEPT, "end": "2022-06-30", "filed": "2022-07-28", "val": 198270000000.0, "form": "10-K", "accn": "B1"},
    ]
    df = pd.DataFrame(records)
    path = tmp_path / "pit_financials.parquet"
    df.to_parquet(path, index=False)
    return path


def test_as_of_no_lookahead(parquet_path):
    q = PitQuery(parquet_path)
    # Only AAPL FY2021 10-K was filed before 2022-06-01
    result = q.as_of("AAPL", CONCEPT, "2022-06-01", max_staleness_days=365)
    assert len(result) == 1
    assert result.iloc[0]["val"] == pytest.approx(365817000000.0)


def test_as_of_includes_most_recent(parquet_path):
    q = PitQuery(parquet_path)
    result = q.as_of("AAPL", CONCEPT, "2023-06-01", max_staleness_days=365)
    assert result.iloc[0]["val"] == pytest.approx(394328000000.0)


def test_as_of_staleness(parquet_path):
    q = PitQuery(parquet_path)
    # MSFT last filed 2022-07-28; query date is 2024-01-01 → stale at 180 days
    result = q.as_of("MSFT", CONCEPT, "2024-01-01", max_staleness_days=180)
    assert pd.isna(result.iloc[0]["val"])


def test_as_of_missing_ticker(parquet_path):
    q = PitQuery(parquet_path)
    result = q.as_of("FAKE", CONCEPT, "2023-01-01")
    assert pd.isna(result.iloc[0]["val"])


def test_history_freq_q(parquet_path):
    q = PitQuery(parquet_path)
    h = q.history("AAPL", CONCEPT, freq="Q")
    assert all(h["form"] == "10-Q")
    assert len(h) == 1


def test_history_freq_a(parquet_path):
    q = PitQuery(parquet_path)
    h = q.history("AAPL", CONCEPT, freq="A")
    assert all(h["form"] == "10-K")
    assert len(h) == 2


def test_history_date_filter(parquet_path):
    q = PitQuery(parquet_path)
    h = q.history("AAPL", CONCEPT, start_date="2022-01-01", end_date="2022-12-31")
    assert all((h["end"] >= pd.Timestamp("2022-01-01")) & (h["end"] <= pd.Timestamp("2022-12-31")))


def test_cross_section_returns_all_tickers(parquet_path):
    q = PitQuery(parquet_path)
    xs = q.cross_section(CONCEPT, "2023-01-01", max_staleness_days=365)
    assert "as_of_date" in xs.columns
    assert "AAPL" in set(xs["ticker"])
    assert "MSFT" in set(xs["ticker"])


def test_cross_section_subset(parquet_path):
    q = PitQuery(parquet_path)
    xs = q.cross_section(CONCEPT, "2023-01-01", tickers=["AAPL"], max_staleness_days=365)
    assert set(xs["ticker"]) == {"AAPL"}


def test_cross_section_multiple_dates(parquet_path):
    q = PitQuery(parquet_path)
    xs = q.cross_section(CONCEPT, ["2022-06-01", "2023-06-01"], tickers=["AAPL"], max_staleness_days=365)
    assert len(xs) == 2
    assert set(xs["as_of_date"]) == {pd.Timestamp("2022-06-01"), pd.Timestamp("2023-06-01")}
    early = xs[xs["as_of_date"] == pd.Timestamp("2022-06-01")].iloc[0]
    late = xs[xs["as_of_date"] == pd.Timestamp("2023-06-01")].iloc[0]
    # Early date: only AAPL FY2021 10-K was filed before 2022-06-01
    assert early["val"] == pytest.approx(365817000000.0)
    # Late date: AAPL FY2022 10-K is also available
    assert late["val"] == pytest.approx(394328000000.0)


@pytest.fixture
def ttm_parquet_path(tmp_path):
    """Four discrete quarters for AAPL so TTM = their sum."""
    records = [
        {"ticker": "AAPL", "concept": CONCEPT, "end": "2022-03-26", "filed": "2022-04-28", "val": 97278000000.0,  "form": "10-Q", "accn": "Q1"},
        {"ticker": "AAPL", "concept": CONCEPT, "end": "2022-06-25", "filed": "2022-07-28", "val": 82959000000.0,  "form": "10-Q", "accn": "Q2"},
        {"ticker": "AAPL", "concept": CONCEPT, "end": "2022-09-24", "filed": "2022-10-28", "val": 90146000000.0,  "form": "10-Q", "accn": "Q3"},
        {"ticker": "AAPL", "concept": CONCEPT, "end": "2022-12-31", "filed": "2023-02-02", "val": 117154000000.0, "form": "10-Q", "accn": "Q4"},
    ]
    df = pd.DataFrame(records)
    path = tmp_path / "pit_financials.parquet"
    df.to_parquet(path, index=False)
    return path


def test_as_of_restatement_pit_correctness(tmp_path):
    """Before a restatement is filed, as_of returns the original value; after, the restated value."""
    records = [
        # Q1 original
        {"ticker": "AAPL", "concept": CONCEPT, "end": "2023-03-31", "filed": "2023-04-28", "val": 100.0, "form": "10-Q", "accn": "R1"},
        # Q2 filed
        {"ticker": "AAPL", "concept": CONCEPT, "end": "2023-06-30", "filed": "2023-07-28", "val": 200.0, "form": "10-Q", "accn": "R2"},
        # Q1 restated (filed after Q2!)
        {"ticker": "AAPL", "concept": CONCEPT, "end": "2023-03-31", "filed": "2023-08-15", "val": 110.0, "form": "10-Q", "accn": "R3"},
    ]
    df = pd.DataFrame(records)
    path = tmp_path / "pit_financials.parquet"
    df.to_parquet(path, index=False)
    q = PitQuery(path)

    # Before Q1 restatement: as_of Q2 filing date → Q2 is the most recent period
    result = q.as_of("AAPL", CONCEPT, "2023-07-28", max_staleness_days=365)
    assert result.iloc[0]["val"] == pytest.approx(200.0)

    # After Q1 restatement: Q2 is still the most recent period (later end date)
    result = q.as_of("AAPL", CONCEPT, "2023-09-01", max_staleness_days=365)
    assert result.iloc[0]["val"] == pytest.approx(200.0)

    # Ask specifically for Q1's value before restatement
    q_hist = q.history("AAPL", CONCEPT, freq="Q")
    q1_row = q_hist[q_hist["end"] == pd.Timestamp("2023-03-31")].iloc[0]
    assert q1_row["val"] == pytest.approx(110.0)  # latest-filed (restated) value


def test_ttm_sum_of_four_quarters(ttm_parquet_path):
    q = PitQuery(ttm_parquet_path)
    result = q.ttm("AAPL", CONCEPT)
    # Only the last filing date has all 4 quarters available
    last = result.iloc[-1]
    expected = 97278 + 82959 + 90146 + 117154  # in millions → exact integer check
    assert last["ttm_val"] == pytest.approx(expected * 1e6, rel=1e-6)
    assert last["n_periods"] == 4
    assert last["ticker"] == "AAPL"


def test_ttm_no_lookahead(ttm_parquet_path):
    """After Q2 is filed, TTM should only use Q1+Q2 (2 periods < 4 → no row emitted)."""
    q = PitQuery(ttm_parquet_path)
    result = q.ttm("AAPL", CONCEPT, end_date="2022-07-28")
    # Only 2 quarters available by 2022-07-28 → no row meets min_periods=4
    assert result.empty


def test_ttm_cross_section_single_date(ttm_parquet_path):
    q = PitQuery(ttm_parquet_path)
    result = q.ttm_cross_section(CONCEPT, "2023-06-01", max_staleness_days=365)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["ticker"] == "AAPL"
    expected = (97278 + 82959 + 90146 + 117154) * 1e6
    assert row["ttm_val"] == pytest.approx(expected, rel=1e-6)


def test_ttm_cross_section_multiple_dates(ttm_parquet_path):
    q = PitQuery(ttm_parquet_path)
    # Date before 4 quarters are available → NaN; date after → full TTM
    result = q.ttm_cross_section(CONCEPT, ["2022-08-01", "2023-06-01"], max_staleness_days=365)
    assert len(result) == 2
    early = result[result["as_of_date"] == pd.Timestamp("2022-08-01")].iloc[0]
    late = result[result["as_of_date"] == pd.Timestamp("2023-06-01")].iloc[0]
    assert pd.isna(early["ttm_val"])   # only 2 quarters available → below min_periods=4
    assert not pd.isna(late["ttm_val"])


def test_ttm_cross_section_missing_ticker(ttm_parquet_path):
    q = PitQuery(ttm_parquet_path)
    result = q.ttm_cross_section(CONCEPT, "2023-06-01", tickers=["AAPL", "MSFT"])
    msft = result[result["ticker"] == "MSFT"].iloc[0]
    assert pd.isna(msft["ttm_val"])


def test_ttm_pit_no_lookahead(ttm_parquet_path):
    """TTM at Q3 filing date uses Q1+Q2+Q3 only — Q4 not yet filed."""
    q = PitQuery(ttm_parquet_path)
    result = q.ttm("AAPL", CONCEPT, min_periods=3, end_date="2022-10-28")
    last = result.iloc[-1]
    assert last["filed"] == pd.Timestamp("2022-10-28")
    assert last["n_periods"] == 3
    assert last["ttm_val"] == pytest.approx((97278 + 82959 + 90146) * 1e6, rel=1e-6)


def test_ttm_start_date_filter(ttm_parquet_path):
    q = PitQuery(ttm_parquet_path)
    result = q.ttm("AAPL", CONCEPT, start_date="2023-01-01")
    assert all(result["filed"] >= pd.Timestamp("2023-01-01"))


# ---------------------------------------------------------------------------
# TTM restatement correctness
# ---------------------------------------------------------------------------

@pytest.fixture
def restatement_parquet_path(tmp_path):
    """4 quarters + a Q1 restatement filed after Q4."""
    records = [
        {"ticker": "AAPL", "concept": CONCEPT, "end": "2022-03-31", "filed": "2022-04-28", "val": 100.0, "form": "10-Q", "accn": "Q1"},
        {"ticker": "AAPL", "concept": CONCEPT, "end": "2022-06-30", "filed": "2022-07-28", "val": 200.0, "form": "10-Q", "accn": "Q2"},
        {"ticker": "AAPL", "concept": CONCEPT, "end": "2022-09-30", "filed": "2022-10-28", "val": 300.0, "form": "10-Q", "accn": "Q3"},
        {"ticker": "AAPL", "concept": CONCEPT, "end": "2022-12-31", "filed": "2023-02-02", "val": 400.0, "form": "10-Q", "accn": "Q4"},
        # Q1 restated after Q4 is filed
        {"ticker": "AAPL", "concept": CONCEPT, "end": "2022-03-31", "filed": "2023-03-01", "val": 110.0, "form": "10-Q", "accn": "Q1R"},
    ]
    df = pd.DataFrame(records)
    path = tmp_path / "pit_financials.parquet"
    df.to_parquet(path, index=False)
    return path


def test_ttm_restatement_before_restate_uses_original(restatement_parquet_path):
    """TTM computed before the Q1 restatement date must use the original Q1 value."""
    q = PitQuery(restatement_parquet_path)
    result = q.ttm("AAPL", CONCEPT, end_date="2023-02-28")
    last = result.iloc[-1]
    assert last["ttm_val"] == pytest.approx(100 + 200 + 300 + 400)
    assert last["filed"] == pd.Timestamp("2023-02-02")


def test_ttm_restatement_after_restate_uses_restated(restatement_parquet_path):
    """TTM computed after the Q1 restatement date must use the restated Q1 value."""
    q = PitQuery(restatement_parquet_path)
    result = q.ttm("AAPL", CONCEPT)
    last = result.iloc[-1]
    assert last["ttm_val"] == pytest.approx(110 + 200 + 300 + 400)
    assert last["filed"] == pd.Timestamp("2023-03-01")


def test_ttm_cross_section_restatement(restatement_parquet_path):
    """ttm_cross_section must also reflect restated values after the restatement date."""
    q = PitQuery(restatement_parquet_path)
    result = q.ttm_cross_section(CONCEPT, ["2023-02-15", "2023-04-01"], max_staleness_days=365)
    before = result[result["as_of_date"] == pd.Timestamp("2023-02-15")].iloc[0]
    after = result[result["as_of_date"] == pd.Timestamp("2023-04-01")].iloc[0]
    assert before["ttm_val"] == pytest.approx(100 + 200 + 300 + 400)
    assert after["ttm_val"] == pytest.approx(110 + 200 + 300 + 400)


def test_ttm_cross_section_staleness(restatement_parquet_path):
    """ttm_cross_section returns NaN when the last filing is older than max_staleness_days."""
    q = PitQuery(restatement_parquet_path)
    result = q.ttm_cross_section(CONCEPT, "2025-01-01", max_staleness_days=100)
    assert pd.isna(result.iloc[0]["ttm_val"])


# ---------------------------------------------------------------------------
# cross_section correctness
# ---------------------------------------------------------------------------

def test_cross_section_no_lookahead(parquet_path):
    """Values filed after as_of_date must not appear."""
    q = PitQuery(parquet_path)
    # AAPL FY2022 10-K filed 2023-02-02; querying 2023-01-01 must not see it.
    # Latest available period is Q2 2022 10-Q (filed 2022-07-29, end 2022-06-25).
    xs = q.cross_section(CONCEPT, "2023-01-01", tickers=["AAPL"], max_staleness_days=365)
    row = xs.iloc[0]
    assert row["val"] == pytest.approx(82959000000.0)
    assert row["end"] == pd.Timestamp("2022-06-25")
    # After FY2022 is filed the value must update
    xs_after = q.cross_section(CONCEPT, "2023-03-01", tickers=["AAPL"], max_staleness_days=365)
    assert xs_after.iloc[0]["val"] == pytest.approx(394328000000.0)
    assert xs_after.iloc[0]["end"] == pd.Timestamp("2022-12-31")


def test_cross_section_staleness(parquet_path):
    """cross_section nullifies values whose last filing predates the as_of_date by too long."""
    q = PitQuery(parquet_path)
    # MSFT last filed 2022-07-28; querying 2024-01-01 at 100 days → stale
    xs = q.cross_section(CONCEPT, "2024-01-01", tickers=["MSFT"], max_staleness_days=100)
    assert pd.isna(xs.iloc[0]["val"])


def test_cross_section_missing_ticker(parquet_path):
    """Tickers not in the parquet are returned with NaN."""
    q = PitQuery(parquet_path)
    xs = q.cross_section(CONCEPT, "2023-01-01", tickers=["FAKE"], max_staleness_days=365)
    assert len(xs) == 1
    assert pd.isna(xs.iloc[0]["val"])


# ---------------------------------------------------------------------------
# history dedup
# ---------------------------------------------------------------------------

def test_history_returns_latest_filed_per_end(tmp_path):
    """history() must return the restated (latest-filed) value for each period end."""
    records = [
        {"ticker": "AAPL", "concept": CONCEPT, "end": "2023-03-31", "filed": "2023-04-28", "val": 50.0,  "form": "10-Q", "accn": "H1"},
        {"ticker": "AAPL", "concept": CONCEPT, "end": "2023-03-31", "filed": "2023-06-01", "val": 55.0,  "form": "10-Q", "accn": "H2"},
        {"ticker": "AAPL", "concept": CONCEPT, "end": "2023-06-30", "filed": "2023-07-28", "val": 60.0,  "form": "10-Q", "accn": "H3"},
    ]
    df = pd.DataFrame(records)
    path = tmp_path / "pit_financials.parquet"
    df.to_parquet(path, index=False)
    q = PitQuery(path)
    h = q.history("AAPL", CONCEPT, freq="Q")
    assert len(h) == 2  # one row per period end
    q1 = h[h["end"] == pd.Timestamp("2023-03-31")].iloc[0]
    assert q1["val"] == pytest.approx(55.0)  # restated value, not 50.0
