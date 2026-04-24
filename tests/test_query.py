"""Tests for query.py"""

import pandas as pd
import pytest

from pitedgar.query import PitQuery

CONCEPT = "us-gaap:Revenues"


@pytest.fixture
def parquet_path(tmp_path):
    records = [
        {
            "ticker": "AAPL",
            "concept": CONCEPT,
            "end": "2021-12-31",
            "filed": "2022-01-28",
            "val": 365817000000.0,
            "form": "10-K",
            "accn": "A1",
        },
        {
            "ticker": "AAPL",
            "concept": CONCEPT,
            "end": "2022-12-31",
            "filed": "2023-02-02",
            "val": 394328000000.0,
            "form": "10-K",
            "accn": "A2",
        },
        {
            "ticker": "AAPL",
            "concept": CONCEPT,
            "end": "2022-06-25",
            "filed": "2022-07-29",
            "val": 82959000000.0,
            "form": "10-Q",
            "accn": "A3",
        },
        {
            "ticker": "MSFT",
            "concept": CONCEPT,
            "end": "2022-06-30",
            "filed": "2022-07-28",
            "val": 198270000000.0,
            "form": "10-K",
            "accn": "B1",
        },
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
        {
            "ticker": "AAPL",
            "concept": CONCEPT,
            "end": "2022-03-26",
            "filed": "2022-04-28",
            "val": 97278000000.0,
            "form": "10-Q",
            "accn": "Q1",
        },
        {
            "ticker": "AAPL",
            "concept": CONCEPT,
            "end": "2022-06-25",
            "filed": "2022-07-28",
            "val": 82959000000.0,
            "form": "10-Q",
            "accn": "Q2",
        },
        {
            "ticker": "AAPL",
            "concept": CONCEPT,
            "end": "2022-09-24",
            "filed": "2022-10-28",
            "val": 90146000000.0,
            "form": "10-Q",
            "accn": "Q3",
        },
        {
            "ticker": "AAPL",
            "concept": CONCEPT,
            "end": "2022-12-31",
            "filed": "2023-02-02",
            "val": 117154000000.0,
            "form": "10-Q",
            "accn": "Q4",
        },
    ]
    df = pd.DataFrame(records)
    path = tmp_path / "pit_financials.parquet"
    df.to_parquet(path, index=False)
    return path


def test_as_of_restatement_pit_correctness(tmp_path):
    """Before a restatement is filed, as_of returns the original value; after, the restated value."""
    records = [
        # Q1 original
        {
            "ticker": "AAPL",
            "concept": CONCEPT,
            "end": "2023-03-31",
            "filed": "2023-04-28",
            "val": 100.0,
            "form": "10-Q",
            "accn": "R1",
        },
        # Q2 filed
        {
            "ticker": "AAPL",
            "concept": CONCEPT,
            "end": "2023-06-30",
            "filed": "2023-07-28",
            "val": 200.0,
            "form": "10-Q",
            "accn": "R2",
        },
        # Q1 restated (filed after Q2!)
        {
            "ticker": "AAPL",
            "concept": CONCEPT,
            "end": "2023-03-31",
            "filed": "2023-08-15",
            "val": 110.0,
            "form": "10-Q",
            "accn": "R3",
        },
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
    assert pd.isna(early["ttm_val"])  # only 2 quarters available → below min_periods=4
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
        {
            "ticker": "AAPL",
            "concept": CONCEPT,
            "end": "2022-03-31",
            "filed": "2022-04-28",
            "val": 100.0,
            "form": "10-Q",
            "accn": "Q1",
        },
        {
            "ticker": "AAPL",
            "concept": CONCEPT,
            "end": "2022-06-30",
            "filed": "2022-07-28",
            "val": 200.0,
            "form": "10-Q",
            "accn": "Q2",
        },
        {
            "ticker": "AAPL",
            "concept": CONCEPT,
            "end": "2022-09-30",
            "filed": "2022-10-28",
            "val": 300.0,
            "form": "10-Q",
            "accn": "Q3",
        },
        {
            "ticker": "AAPL",
            "concept": CONCEPT,
            "end": "2022-12-31",
            "filed": "2023-02-02",
            "val": 400.0,
            "form": "10-Q",
            "accn": "Q4",
        },
        # Q1 restated after Q4 is filed
        {
            "ticker": "AAPL",
            "concept": CONCEPT,
            "end": "2022-03-31",
            "filed": "2023-03-01",
            "val": 110.0,
            "form": "10-Q",
            "accn": "Q1R",
        },
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

# ---------------------------------------------------------------------------
# Q4-from-10-K derivation
# ---------------------------------------------------------------------------


@pytest.fixture
def dec_fy_parquet_path(tmp_path):
    """December-FY company: Q1/Q2/Q3 as 10-Qs, Q4 only in 10-K annual.

    Annual = Q1 + Q2 + Q3 + Q4 = 100 + 200 + 300 + 400 = 1000.
    The 10-K is filed 2026-02-05.  The last 10-Q (Q3) was filed 2025-11-10,
    which is ~135 days before a March-2026 query — exceeding the default
    max_staleness_days=100.  Without 10-K fallback these tickers go to NaN.
    """
    records = [
        # Q1 2025
        {
            "ticker": "TSLA",
            "concept": CONCEPT,
            "end": "2025-03-31",
            "filed": "2025-05-05",
            "val": 100.0,
            "form": "10-Q",
            "accn": "K1",
        },
        # Q2 2025
        {
            "ticker": "TSLA",
            "concept": CONCEPT,
            "end": "2025-06-30",
            "filed": "2025-08-05",
            "val": 200.0,
            "form": "10-Q",
            "accn": "K2",
        },
        # Q3 2025 — last 10-Q, filed ~135 days before 2026-03-28
        {
            "ticker": "TSLA",
            "concept": CONCEPT,
            "end": "2025-09-30",
            "filed": "2025-11-10",
            "val": 300.0,
            "form": "10-Q",
            "accn": "K3",
        },
        # FY2025 10-K — annual = Q1+Q2+Q3+Q4 = 1000; filed 2026-02-05 (~51 days ago)
        {
            "ticker": "TSLA",
            "concept": CONCEPT,
            "end": "2025-12-31",
            "filed": "2026-02-05",
            "val": 1000.0,
            "form": "10-K",
            "accn": "K4",
        },
    ]
    df = pd.DataFrame(records)
    path = tmp_path / "pit_financials.parquet"
    df.to_parquet(path, index=False)
    return path


def test_ttm_derives_q4_from_10k(dec_fy_parquet_path):
    """ttm() must use Annual − Q1 − Q2 − Q3 = 400 as Q4, giving TTM = 1000."""
    q = PitQuery(dec_fy_parquet_path)
    result = q.ttm("TSLA", CONCEPT)
    last = result.iloc[-1]
    assert last["ttm_val"] == pytest.approx(1000.0)
    assert last["n_periods"] == 4
    # The event is emitted at the 10-K filing date, not the last 10-Q date
    assert last["filed"] == pd.Timestamp("2026-02-05")


def test_ttm_cross_section_dec_fy_not_stale(dec_fy_parquet_path):
    """ttm_cross_section must not nullify December-FY companies whose 10-K is recent."""
    q = PitQuery(dec_fy_parquet_path)
    result = q.ttm_cross_section(CONCEPT, "2026-03-28", max_staleness_days=100)
    row = result.iloc[0]
    # 2026-03-28 minus 10-K filed 2026-02-05 = 51 days < 100 → not stale
    assert row["ttm_val"] == pytest.approx(1000.0)
    assert row["n_periods"] == 4


def test_ttm_cross_section_before_10k_filed_still_stale(dec_fy_parquet_path):
    """Before the 10-K is filed, the last 10-Q is too old → TTM is NaN."""
    q = PitQuery(dec_fy_parquet_path)
    # Query on 2026-01-01: 10-K not yet filed (2026-02-05); last 10-Q filed 2025-11-10
    # staleness = ~52 days at 100-day threshold → not stale yet; but test at shorter threshold
    result = q.ttm_cross_section(CONCEPT, "2026-01-20", max_staleness_days=60)
    row = result.iloc[0]
    # Last 10-Q was 2025-11-10 (~71 days before 2026-01-20) → stale at 60-day threshold
    assert pd.isna(row["ttm_val"])


def test_ttm_q4_not_injected_when_fewer_than_3_quarters(tmp_path):
    """If only Q1+Q2 exist for a fiscal year, Q4 is NOT synthesised (would absorb Q3 too)."""
    records = [
        {
            "ticker": "TSLA",
            "concept": CONCEPT,
            "end": "2025-03-31",
            "filed": "2025-05-05",
            "val": 100.0,
            "form": "10-Q",
            "accn": "K1",
        },
        {
            "ticker": "TSLA",
            "concept": CONCEPT,
            "end": "2025-06-30",
            "filed": "2025-08-05",
            "val": 200.0,
            "form": "10-Q",
            "accn": "K2",
        },
        # No Q3 10-Q
        {
            "ticker": "TSLA",
            "concept": CONCEPT,
            "end": "2025-12-31",
            "filed": "2026-02-05",
            "val": 1000.0,
            "form": "10-K",
            "accn": "K4",
        },
    ]
    df = pd.DataFrame(records)
    path = tmp_path / "pit_financials.parquet"
    df.to_parquet(path, index=False)
    q = PitQuery(path)
    # No synthetic Q4 → only 2 quarters available → min_periods=4 not met
    result = q.ttm("TSLA", CONCEPT, min_periods=4)
    assert result.empty


def test_ttm_q4_pit_no_lookahead_before_10k(dec_fy_parquet_path):
    """Before the 10-K is filed, the synthetic Q4 must not exist (PIT correctness)."""
    q = PitQuery(dec_fy_parquet_path)
    # end_date before 10-K filing → synthetic Q4 not yet known → min_periods=4 unmet
    result = q.ttm("TSLA", CONCEPT, end_date="2026-01-31", min_periods=4)
    assert result.empty


def test_ttm_uses_quarterly_data_from_10k_filings(tmp_path):
    """Discrete quarterly values inside 10-K filings (duration_days≈90, form='10-K')
    must be included in TTM — this is the JNJ pattern where EDGAR only has quarterly
    NetIncomeLoss as comparative data inside annual 10-K filings."""
    records = [
        # All 4 quarters appear as form='10-K' with quarterly duration
        {
            "ticker": "JNJ",
            "concept": CONCEPT,
            "end": "2022-03-31",
            "filed": "2023-02-15",
            "val": 100.0,
            "form": "10-K",
            "accn": "J1",
            "duration_days": 90,
        },
        {
            "ticker": "JNJ",
            "concept": CONCEPT,
            "end": "2022-06-30",
            "filed": "2023-02-15",
            "val": 200.0,
            "form": "10-K",
            "accn": "J2",
            "duration_days": 91,
        },
        {
            "ticker": "JNJ",
            "concept": CONCEPT,
            "end": "2022-09-30",
            "filed": "2023-02-15",
            "val": 300.0,
            "form": "10-K",
            "accn": "J3",
            "duration_days": 91,
        },
        {
            "ticker": "JNJ",
            "concept": CONCEPT,
            "end": "2022-12-31",
            "filed": "2023-02-15",
            "val": 400.0,
            "form": "10-K",
            "accn": "J4",
            "duration_days": 92,
        },
    ]
    df = pd.DataFrame(records)
    path = tmp_path / "pit_financials.parquet"
    df.to_parquet(path, index=False)
    q = PitQuery(path)
    result = q.ttm("JNJ", CONCEPT)
    assert not result.empty
    last = result.iloc[-1]
    assert last["ttm_val"] == pytest.approx(1000.0)
    assert last["n_periods"] == 4


def test_ttm_cross_section_quarterly_data_from_10k(tmp_path):
    """ttm_cross_section must also pick up quarterly data from 10-K filings."""
    records = [
        {
            "ticker": "JNJ",
            "concept": CONCEPT,
            "end": "2022-03-31",
            "filed": "2023-02-15",
            "val": 100.0,
            "form": "10-K",
            "accn": "J1",
            "duration_days": 90,
        },
        {
            "ticker": "JNJ",
            "concept": CONCEPT,
            "end": "2022-06-30",
            "filed": "2023-02-15",
            "val": 200.0,
            "form": "10-K",
            "accn": "J2",
            "duration_days": 91,
        },
        {
            "ticker": "JNJ",
            "concept": CONCEPT,
            "end": "2022-09-30",
            "filed": "2023-02-15",
            "val": 300.0,
            "form": "10-K",
            "accn": "J3",
            "duration_days": 91,
        },
        {
            "ticker": "JNJ",
            "concept": CONCEPT,
            "end": "2022-12-31",
            "filed": "2023-02-15",
            "val": 400.0,
            "form": "10-K",
            "accn": "J4",
            "duration_days": 92,
        },
    ]
    df = pd.DataFrame(records)
    path = tmp_path / "pit_financials.parquet"
    df.to_parquet(path, index=False)
    q = PitQuery(path)
    result = q.ttm_cross_section(CONCEPT, "2023-06-01", max_staleness_days=365)
    row = result.iloc[0]
    assert row["ttm_val"] == pytest.approx(1000.0)
    assert row["n_periods"] == 4


def test_as_of_10ka_supersedes_10k(tmp_path):
    """A 10-K/A amendment filed after the original 10-K must supersede the 10-K
    value in as_of() queries dated AFTER the amendment was filed."""
    records = [
        # Original 10-K
        {
            "ticker": "AAPL",
            "concept": CONCEPT,
            "end": "2022-12-31",
            "filed": "2023-02-02",
            "val": 1_000_000_000.0,
            "form": "10-K",
            "accn": "ORIG",
        },
        # 10-K/A restating the same period months later
        {
            "ticker": "AAPL",
            "concept": CONCEPT,
            "end": "2022-12-31",
            "filed": "2023-09-15",
            "val": 1_120_000_000.0,
            "form": "10-K/A",
            "accn": "AMEND",
        },
    ]
    df = pd.DataFrame(records)
    path = tmp_path / "pit_financials.parquet"
    df.to_parquet(path, index=False)
    q = PitQuery(path)

    # Before the amendment is filed, the original 10-K value is the latest known.
    before = q.as_of("AAPL", CONCEPT, "2023-06-01", max_staleness_days=365)
    assert before.iloc[0]["val"] == pytest.approx(1_000_000_000.0)
    assert before.iloc[0]["form"] == "10-K"

    # After the amendment is filed, it must supersede the original.
    after = q.as_of("AAPL", CONCEPT, "2023-10-01", max_staleness_days=365)
    assert after.iloc[0]["val"] == pytest.approx(1_120_000_000.0)
    assert after.iloc[0]["form"] == "10-K/A"


# ---------------------------------------------------------------------------
# age_days surfacing (v0.3.0)
# ---------------------------------------------------------------------------


def test_as_of_age_days_present_and_correct(parquet_path):
    """as_of always returns age_days = (as_of_date - filed) in days."""
    q = PitQuery(parquet_path)
    # AAPL FY2022 10-K filed 2023-02-02; as_of 2023-06-01 → 119 days.
    result = q.as_of("AAPL", CONCEPT, "2023-06-01")
    assert "age_days" in result.columns
    row = result.iloc[0]
    expected = (pd.Timestamp("2023-06-01") - pd.Timestamp("2023-02-02")).days
    assert int(row["age_days"]) == expected


def test_as_of_default_does_not_nullify_stale(parquet_path):
    """Default max_staleness_days=None must NOT nullify; surfaces age_days instead."""
    q = PitQuery(parquet_path)
    # MSFT last filed 2022-07-28; query date 2027-07-28 → ~5 years old (~1826 days).
    result = q.as_of("MSFT", CONCEPT, "2027-07-28")
    row = result.iloc[0]
    assert not pd.isna(row["val"])
    assert row["val"] == pytest.approx(198270000000.0)
    assert int(row["age_days"]) == pytest.approx(1826, abs=2)


def test_as_of_explicit_staleness_still_nullifies(parquet_path):
    """Passing max_staleness_days=100 must still nullify (back-compat)."""
    q = PitQuery(parquet_path)
    result = q.as_of("MSFT", CONCEPT, "2024-01-01", max_staleness_days=100)
    row = result.iloc[0]
    assert pd.isna(row["val"])
    # age_days is still surfaced even when nullified.
    assert "age_days" in result.columns
    assert not pd.isna(row["age_days"])


def test_as_of_missing_ticker_age_days_na(parquet_path):
    """Missing tickers get NA age_days (not 0)."""
    q = PitQuery(parquet_path)
    result = q.as_of("FAKE", CONCEPT, "2023-01-01")
    assert "age_days" in result.columns
    assert pd.isna(result.iloc[0]["age_days"])


def test_cross_section_age_days_present_and_correct(parquet_path):
    """cross_section always returns age_days."""
    q = PitQuery(parquet_path)
    xs = q.cross_section(CONCEPT, "2023-06-01", tickers=["AAPL"])
    assert "age_days" in xs.columns
    row = xs.iloc[0]
    expected = (pd.Timestamp("2023-06-01") - row["filed"]).days
    assert int(row["age_days"]) == expected


def test_cross_section_default_does_not_nullify_stale(parquet_path):
    """Default max_staleness_days=None must NOT nullify in cross_section."""
    q = PitQuery(parquet_path)
    xs = q.cross_section(CONCEPT, "2027-07-28", tickers=["MSFT"])
    row = xs.iloc[0]
    assert not pd.isna(row["val"])
    assert row["val"] == pytest.approx(198270000000.0)
    assert int(row["age_days"]) == pytest.approx(1826, abs=2)


def test_cross_section_explicit_staleness_still_nullifies(parquet_path):
    """Passing max_staleness_days=100 must still nullify in cross_section."""
    q = PitQuery(parquet_path)
    xs = q.cross_section(CONCEPT, "2024-01-01", tickers=["MSFT"], max_staleness_days=100)
    assert pd.isna(xs.iloc[0]["val"])
    assert "age_days" in xs.columns


def test_cross_section_missing_ticker_age_days_na(parquet_path):
    """Missing tickers get NA age_days (not 0) in cross_section."""
    q = PitQuery(parquet_path)
    xs = q.cross_section(CONCEPT, "2023-01-01", tickers=["FAKE"])
    assert "age_days" in xs.columns
    assert pd.isna(xs.iloc[0]["age_days"])


def test_ttm_cross_section_age_days_present_and_correct(ttm_parquet_path):
    """ttm_cross_section always returns age_days."""
    q = PitQuery(ttm_parquet_path)
    result = q.ttm_cross_section(CONCEPT, "2023-06-01")
    assert "age_days" in result.columns
    row = result.iloc[0]
    expected = (pd.Timestamp("2023-06-01") - row["filed"]).days
    assert int(row["age_days"]) == expected


def test_ttm_cross_section_default_does_not_nullify_stale(restatement_parquet_path):
    """Default max_staleness_days=None must NOT nullify TTM."""
    q = PitQuery(restatement_parquet_path)
    # last filing was 2023-03-01 → 2025-01-01 is ~671 days later, well past 100 days.
    result = q.ttm_cross_section(CONCEPT, "2025-01-01")
    row = result.iloc[0]
    assert not pd.isna(row["ttm_val"])
    assert row["n_periods"] == 4
    assert int(row["age_days"]) > 100


def test_ttm_cross_section_explicit_staleness_still_nullifies(restatement_parquet_path):
    """Passing max_staleness_days=100 must still nullify ttm_cross_section."""
    q = PitQuery(restatement_parquet_path)
    result = q.ttm_cross_section(CONCEPT, "2025-01-01", max_staleness_days=100)
    row = result.iloc[0]
    assert pd.isna(row["ttm_val"])
    assert row["n_periods"] == 0
    assert "age_days" in result.columns


def test_ttm_cross_section_missing_ticker_age_days_na(ttm_parquet_path):
    """Missing tickers get NA age_days (not 0) in ttm_cross_section."""
    q = PitQuery(ttm_parquet_path)
    result = q.ttm_cross_section(CONCEPT, "2023-06-01", tickers=["AAPL", "MSFT"])
    msft = result[result["ticker"] == "MSFT"].iloc[0]
    assert "age_days" in result.columns
    assert pd.isna(msft["age_days"])


def test_history_freq_q_picks_up_quarterly_data_from_10k(tmp_path):
    """history(freq='Q') must include discrete quarterly values reported inside a 10-K
    (e.g. AAPL us-gaap:Revenues comparative disclosures: form='10-K', duration≈90d)."""
    records = [
        # Quarterly values disclosed inside the annual 10-K as comparatives.
        {
            "ticker": "AAPL",
            "concept": CONCEPT,
            "end": "2022-03-26",
            "filed": "2023-02-02",
            "val": 97278000000.0,
            "form": "10-K",
            "accn": "AQ1",
            "duration_days": 90,
        },
        {
            "ticker": "AAPL",
            "concept": CONCEPT,
            "end": "2022-06-25",
            "filed": "2023-02-02",
            "val": 82959000000.0,
            "form": "10-K",
            "accn": "AQ2",
            "duration_days": 91,
        },
        {
            "ticker": "AAPL",
            "concept": CONCEPT,
            "end": "2022-09-24",
            "filed": "2023-02-02",
            "val": 90146000000.0,
            "form": "10-K",
            "accn": "AQ3",
            "duration_days": 91,
        },
    ]
    df = pd.DataFrame(records)
    path = tmp_path / "pit_financials.parquet"
    df.to_parquet(path, index=False)
    q = PitQuery(path)
    h = q.history("AAPL", CONCEPT, freq="Q")
    assert len(h) == 3
    assert set(h["form"]) == {"10-K"}
    assert all(h["end"].dt.year == 2022)


def test_history_freq_a_returns_annual_and_freq_q_does_not(tmp_path):
    """A 365-day 10-K row must be returned by freq='A' but NOT by freq='Q'."""
    records = [
        {
            "ticker": "AAPL",
            "concept": CONCEPT,
            "end": "2022-12-31",
            "filed": "2023-02-02",
            "val": 394328000000.0,
            "form": "10-K",
            "accn": "ANN1",
            "duration_days": 365,
        },
    ]
    df = pd.DataFrame(records)
    path = tmp_path / "pit_financials.parquet"
    df.to_parquet(path, index=False)
    q = PitQuery(path)
    h_a = q.history("AAPL", CONCEPT, freq="A")
    assert len(h_a) == 1
    assert h_a.iloc[0]["form"] == "10-K"
    h_q = q.history("AAPL", CONCEPT, freq="Q")
    assert h_q.empty


def test_history_freq_q_excludes_mid_year_10q_with_long_duration(tmp_path):
    """A 10-Q with duration_days=180 (outside 60-105 quarterly range) must be excluded by freq='Q'."""
    records = [
        {
            "ticker": "AAPL",
            "concept": CONCEPT,
            "end": "2022-06-30",
            "filed": "2022-07-28",
            "val": 123456789.0,
            "form": "10-Q",
            "accn": "WEIRD",
            "duration_days": 180,
        },
    ]
    df = pd.DataFrame(records)
    path = tmp_path / "pit_financials.parquet"
    df.to_parquet(path, index=False)
    q = PitQuery(path)
    h = q.history("AAPL", CONCEPT, freq="Q")
    assert h.empty
    # freq=None/other keeps it
    h_all = q.history("AAPL", CONCEPT, freq="all")
    assert len(h_all) == 1


def test_history_returns_latest_filed_per_end(tmp_path):
    """history() must return the restated (latest-filed) value for each period end."""
    records = [
        {
            "ticker": "AAPL",
            "concept": CONCEPT,
            "end": "2023-03-31",
            "filed": "2023-04-28",
            "val": 50.0,
            "form": "10-Q",
            "accn": "H1",
        },
        {
            "ticker": "AAPL",
            "concept": CONCEPT,
            "end": "2023-03-31",
            "filed": "2023-06-01",
            "val": 55.0,
            "form": "10-Q",
            "accn": "H2",
        },
        {
            "ticker": "AAPL",
            "concept": CONCEPT,
            "end": "2023-06-30",
            "filed": "2023-07-28",
            "val": 60.0,
            "form": "10-Q",
            "accn": "H3",
        },
    ]
    df = pd.DataFrame(records)
    path = tmp_path / "pit_financials.parquet"
    df.to_parquet(path, index=False)
    q = PitQuery(path)
    h = q.history("AAPL", CONCEPT, freq="Q")
    assert len(h) == 2  # one row per period end
    q1 = h[h["end"] == pd.Timestamp("2023-03-31")].iloc[0]
    assert q1["val"] == pytest.approx(55.0)  # restated value, not 50.0


# ---------------------------------------------------------------------------
# YTD-chain synthesis (AAPL post-2021 pattern)
# ---------------------------------------------------------------------------


AAPL_FY24_NI = [
    # AAPL FY2024 NetIncomeLoss, post-2021 tagging pattern: only YTD rows.
    # Values (in USD) from the SEC filings; TTM after the 10-K should equal
    # the published FY2024 net income of $93,736M.
    ("2023-12-30", "2024-02-02", 33916000000.0, "10-Q", 90),  # Q1 FY24 YTD
    ("2024-03-30", "2024-05-03", 57564000000.0, "10-Q", 181),  # Q2 FY24 YTD
    ("2024-06-29", "2024-08-02", 79000000000.0, "10-Q", 272),  # Q3 FY24 YTD
    ("2024-09-28", "2024-11-01", 93736000000.0, "10-K", 363),  # FY24 annual
]

AAPL_FY24_REV = [
    ("2023-12-30", "2024-02-02", 119575000000.0, "10-Q", 90),
    ("2024-03-30", "2024-05-03", 210328000000.0, "10-Q", 181),
    ("2024-06-29", "2024-08-02", 296135000000.0, "10-Q", 272),
    ("2024-09-28", "2024-11-01", 391035000000.0, "10-K", 363),
]


def _aapl_ytd_records(concept: str, entries: list[tuple], fy_start: str = "2023-10-01") -> list[dict]:
    """Build YTD-only records for a ticker/concept mimicking AAPL post-2021 tagging."""
    return [
        {
            "ticker": "AAPL",
            "concept": concept,
            "start": fy_start,
            "end": end,
            "filed": filed,
            "val": val,
            "form": form,
            "accn": f"YTD{i}",
            "duration_days": duration,
        }
        for i, (end, filed, val, form, duration) in enumerate(entries)
    ]


@pytest.fixture
def aapl_ytd_parquet(tmp_path):
    """AAPL FY2024 NetIncomeLoss + Revenues, post-2021 YTD-only tagging."""
    records = _aapl_ytd_records("us-gaap:NetIncomeLoss", AAPL_FY24_NI) + _aapl_ytd_records(
        "us-gaap:Revenues", AAPL_FY24_REV
    )
    df = pd.DataFrame(records)
    path = tmp_path / "pit_financials.parquet"
    df.to_parquet(path, index=False)
    return path


def test_ttm_netincome_from_ytd_chain(aapl_ytd_parquet):
    """AAPL FY2024 TTM NetIncomeLoss at 10-K date must equal FY value (~$93.7B)."""
    q = PitQuery(aapl_ytd_parquet)
    result = q.ttm("AAPL", "us-gaap:NetIncomeLoss")
    last = result.iloc[-1]
    # TTM = Q1_90d + (Q2_181d - Q1_90d) + (Q3_272d - Q2_181d) + (FY - Q3_272d) = FY.
    assert last["ttm_val"] == pytest.approx(93736000000.0, rel=0.01)
    assert last["n_periods"] == 4
    assert last["filed"] == pd.Timestamp("2024-11-01")


def test_ttm_revenues_from_ytd_chain(aapl_ytd_parquet):
    """AAPL FY2024 TTM Revenues at 10-K date must equal FY value (~$391B)."""
    q = PitQuery(aapl_ytd_parquet)
    result = q.ttm("AAPL", "us-gaap:Revenues")
    last = result.iloc[-1]
    assert last["ttm_val"] == pytest.approx(391035000000.0, rel=0.01)


def test_ttm_ytd_synthesis_no_lookahead(aapl_ytd_parquet):
    """Synthetic quarters must not appear before both source YTDs are filed."""
    q = PitQuery(aapl_ytd_parquet)
    # Before Q2 YTD is filed (2024-05-03), only Q1_90d is known → no synth Q2_3m yet.
    result = q.ttm("AAPL", "us-gaap:NetIncomeLoss", end_date="2024-04-30", min_periods=1)
    # Only the 90d Q1 YTD is known → n_periods == 1 at the Q1 filing date.
    assert not result.empty
    assert (result["n_periods"] <= 1).all()


def test_ttm_ytd_synthesis_after_two_ytds(aapl_ytd_parquet):
    """After Q2 YTD is filed, synthetic Q2_3m appears → 2 periods available."""
    q = PitQuery(aapl_ytd_parquet)
    result = q.ttm("AAPL", "us-gaap:NetIncomeLoss", end_date="2024-05-03", min_periods=2)
    last = result.iloc[-1]
    # Q1_90d (33916) + synth Q2_3m (57564 − 33916 = 23648) = 57564.
    assert last["ttm_val"] == pytest.approx(57564000000.0, rel=1e-6)
    assert last["n_periods"] == 2


def test_ttm_explicit_takes_precedence_over_synth(tmp_path):
    """When both explicit 3-month and synthesized YTD-diff produce a row for the same
    (end, filed), the explicit row wins (acceptance criterion)."""
    records = [
        # Discrete Q1 (90d)
        {
            "ticker": "X",
            "concept": CONCEPT,
            "start": "2024-01-01",
            "end": "2024-03-31",
            "filed": "2024-04-28",
            "val": 100.0,
            "form": "10-Q",
            "accn": "Q1",
            "duration_days": 91,
        },
        # Discrete Q2 (91d) — non-YTD, start at Q1 end. Value 200.
        {
            "ticker": "X",
            "concept": CONCEPT,
            "start": "2024-04-01",
            "end": "2024-06-30",
            "filed": "2024-07-28",
            "val": 200.0,
            "form": "10-Q",
            "accn": "Q2_EXPLICIT",
            "duration_days": 91,
        },
        # YTD_6m at same filed as explicit Q2 — creates a synth candidate at
        # (end=2024-06-30, filed=2024-07-28) with val = 999 − 100 = 899.
        {
            "ticker": "X",
            "concept": CONCEPT,
            "start": "2024-01-01",
            "end": "2024-06-30",
            "filed": "2024-07-28",
            "val": 999.0,
            "form": "10-Q",
            "accn": "YTD6",
            "duration_days": 182,
        },
    ]
    df = pd.DataFrame(records)
    path = tmp_path / "pit_financials.parquet"
    df.to_parquet(path, index=False)
    q = PitQuery(path)
    result = q.ttm("X", CONCEPT, min_periods=2)
    last = result.iloc[-1]
    # Explicit Q2 (200) must win, not synthesized (899). Expected TTM = 100 + 200 = 300.
    assert last["ttm_val"] == pytest.approx(300.0)


def test_ttm_no_regression_when_only_discrete_quarters(tmp_path):
    """When a filer uses only discrete 3-month rows, YTD synthesis is a no-op and
    TTM matches the pre-fix behavior."""
    records = [
        {
            "ticker": "MSFT",
            "concept": CONCEPT,
            "start": "2019-07-01",
            "end": "2019-09-30",
            "filed": "2019-10-23",
            "val": 100.0,
            "form": "10-Q",
            "accn": "D1",
            "duration_days": 92,
        },
        {
            "ticker": "MSFT",
            "concept": CONCEPT,
            "start": "2019-10-01",
            "end": "2019-12-31",
            "filed": "2020-01-29",
            "val": 200.0,
            "form": "10-Q",
            "accn": "D2",
            "duration_days": 92,
        },
        {
            "ticker": "MSFT",
            "concept": CONCEPT,
            "start": "2020-01-01",
            "end": "2020-03-31",
            "filed": "2020-04-29",
            "val": 300.0,
            "form": "10-Q",
            "accn": "D3",
            "duration_days": 91,
        },
        {
            "ticker": "MSFT",
            "concept": CONCEPT,
            "start": "2020-04-01",
            "end": "2020-06-30",
            "filed": "2020-07-22",
            "val": 400.0,
            "form": "10-Q",
            "accn": "D4",
            "duration_days": 91,
        },
    ]
    df = pd.DataFrame(records)
    path = tmp_path / "pit_financials.parquet"
    df.to_parquet(path, index=False)
    q = PitQuery(path)
    result = q.ttm("MSFT", CONCEPT)
    last = result.iloc[-1]
    assert last["ttm_val"] == pytest.approx(1000.0)
    assert last["n_periods"] == 4


# ---------------------------------------------------------------------------
# Issue #28: n_periods consistency between present-but-no-prior-filing and missing ticker
# ---------------------------------------------------------------------------


def test_ttm_cross_section_n_periods_zero_when_no_prior_filing(tmp_path):
    """Ticker present in data but all filings are AFTER as_of_date → n_periods == 0,
    ttm_val NaN. Must work WITHOUT passing max_staleness_days."""
    records = [
        {
            "ticker": "AAPL",
            "concept": CONCEPT,
            "end": "2023-03-31",
            "filed": "2023-04-28",
            "val": 100.0,
            "form": "10-Q",
            "accn": "Q1",
        },
        {
            "ticker": "AAPL",
            "concept": CONCEPT,
            "end": "2023-06-30",
            "filed": "2023-07-28",
            "val": 200.0,
            "form": "10-Q",
            "accn": "Q2",
        },
        {
            "ticker": "AAPL",
            "concept": CONCEPT,
            "end": "2023-09-30",
            "filed": "2023-10-28",
            "val": 300.0,
            "form": "10-Q",
            "accn": "Q3",
        },
        {
            "ticker": "AAPL",
            "concept": CONCEPT,
            "end": "2023-12-31",
            "filed": "2024-01-28",
            "val": 400.0,
            "form": "10-Q",
            "accn": "Q4",
        },
    ]
    df = pd.DataFrame(records)
    path = tmp_path / "pit_financials.parquet"
    df.to_parquet(path, index=False)
    q = PitQuery(path)

    # Query before ANY filing exists → ticker is present in the universe but has
    # no event before as_of_date; merge_asof previously left n_periods as NaN.
    result = q.ttm_cross_section(CONCEPT, "2023-01-01", tickers=["AAPL"])
    assert len(result) == 1
    row = result.iloc[0]
    assert row["ticker"] == "AAPL"
    assert pd.isna(row["ttm_val"])
    assert row["n_periods"] == 0  # must be 0, not NaN


def test_ttm_cross_section_n_periods_zero_for_missing_ticker_unchanged(tmp_path):
    """Regression: ticker absent from the data universe still gets n_periods == 0."""
    records = [
        {
            "ticker": "AAPL",
            "concept": CONCEPT,
            "end": "2023-03-31",
            "filed": "2023-04-28",
            "val": 100.0,
            "form": "10-Q",
            "accn": "Q1",
        },
        {
            "ticker": "AAPL",
            "concept": CONCEPT,
            "end": "2023-06-30",
            "filed": "2023-07-28",
            "val": 200.0,
            "form": "10-Q",
            "accn": "Q2",
        },
        {
            "ticker": "AAPL",
            "concept": CONCEPT,
            "end": "2023-09-30",
            "filed": "2023-10-28",
            "val": 300.0,
            "form": "10-Q",
            "accn": "Q3",
        },
        {
            "ticker": "AAPL",
            "concept": CONCEPT,
            "end": "2023-12-31",
            "filed": "2024-01-28",
            "val": 400.0,
            "form": "10-Q",
            "accn": "Q4",
        },
    ]
    df = pd.DataFrame(records)
    path = tmp_path / "pit_financials.parquet"
    df.to_parquet(path, index=False)
    q = PitQuery(path)

    # MSFT is not in the data at all → filler row.
    result = q.ttm_cross_section(CONCEPT, "2024-06-01", tickers=["AAPL", "MSFT"])
    msft = result[result["ticker"] == "MSFT"].iloc[0]
    assert pd.isna(msft["ttm_val"])
    assert msft["n_periods"] == 0


def test_ttm_cross_section_matches_ttm_on_ytd_universe(tmp_path):
    """ttm_cross_section must return the same TTM as per-ticker ttm() for a mixed
    universe — including AAPL-style YTD-only filers."""
    records: list[dict] = []
    # AAPL: YTD-only pattern (post-2021).
    records.extend(_aapl_ytd_records("us-gaap:NetIncomeLoss", AAPL_FY24_NI))
    # Four more tickers with discrete quarterly tagging.
    fy_quarters = [
        ("2023-03-31", "2023-04-28", 100.0, "10-Q", "2023-01-01"),
        ("2023-06-30", "2023-07-28", 200.0, "10-Q", "2023-04-01"),
        ("2023-09-30", "2023-10-28", 300.0, "10-Q", "2023-07-01"),
        ("2023-12-31", "2024-02-02", 400.0, "10-Q", "2023-10-01"),
    ]
    for t in ("MSFT", "GOOG", "META", "NVDA"):
        for i, (end, filed, val, form, start) in enumerate(fy_quarters):
            records.append(
                {
                    "ticker": t,
                    "concept": "us-gaap:NetIncomeLoss",
                    "start": start,
                    "end": end,
                    "filed": filed,
                    "val": val,
                    "form": form,
                    "accn": f"{t}{i}",
                    "duration_days": 91,
                }
            )
    df = pd.DataFrame(records)
    path = tmp_path / "pit_financials.parquet"
    df.to_parquet(path, index=False)
    q = PitQuery(path)

    universe = ["AAPL", "MSFT", "GOOG", "META", "NVDA"]
    xs = q.ttm_cross_section("us-gaap:NetIncomeLoss", "2024-12-01", tickers=universe, max_staleness_days=365)
    for t in universe:
        solo = q.ttm("AAPL" if t == "AAPL" else t, "us-gaap:NetIncomeLoss")
        expected = solo.iloc[-1]["ttm_val"] if not solo.empty else float("nan")
        row = xs[xs["ticker"] == t].iloc[0]
        if pd.isna(expected):
            assert pd.isna(row["ttm_val"])
        else:
            assert row["ttm_val"] == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# PitQuery.__init__ filter pushdown kwargs (#24)
# Issue #29: unknown concept validation
# ---------------------------------------------------------------------------


@pytest.fixture
def multi_ticker_parquet(tmp_path):
    """Parquet with AAPL, MSFT, and two concepts for filter tests."""
    records = [
        {"ticker": "AAPL", "concept": "us-gaap:Revenues", "end": "2022-12-31",
         "filed": "2023-02-02", "val": 394328000000.0, "form": "10-K", "accn": "A1"},
        {"ticker": "AAPL", "concept": "us-gaap:Assets", "end": "2022-12-31",
         "filed": "2023-02-02", "val": 352755000000.0, "form": "10-K", "accn": "A2"},
        {"ticker": "MSFT", "concept": "us-gaap:Revenues", "end": "2022-06-30",
         "filed": "2022-07-28", "val": 198270000000.0, "form": "10-K", "accn": "B1"},
        {"ticker": "MSFT", "concept": "us-gaap:Assets", "end": "2022-06-30",
         "filed": "2022-07-28", "val": 364840000000.0, "form": "10-K", "accn": "B2"},
        {"ticker": "AAPL", "concept": "us-gaap:Revenues", "end": "2021-12-31",
         "filed": "2022-01-28", "val": 365817000000.0, "form": "10-K", "accn": "A0"},
    ]
    df = pd.DataFrame(records)
    path = tmp_path / "pit_financials.parquet"
    df.to_parquet(path, index=False)
    return path


@pytest.fixture
def concept_check_parquet(tmp_path):
    """Small parquet with a single known concept: us-gaap:Revenues."""
    records = [
        {
            "ticker": "AAPL",
            "concept": "us-gaap:Revenues",
            "end": "2022-12-31",
            "filed": "2023-02-02",
            "val": 394328000000.0,
            "form": "10-K",
            "accn": "A1",
        },
        {
            "ticker": "AAPL",
            "concept": "us-gaap:Assets",
            "end": "2022-12-31",
            "filed": "2023-02-02",
            "val": 352755000000.0,
            "form": "10-K",
            "accn": "A2",
        },
        {
            "ticker": "MSFT",
            "concept": "us-gaap:Revenues",
            "end": "2022-06-30",
            "filed": "2022-07-28",
            "val": 198270000000.0,
            "form": "10-K",
            "accn": "B1",
        },
        {
            "ticker": "MSFT",
            "concept": "us-gaap:Assets",
            "end": "2022-06-30",
            "filed": "2022-07-28",
            "val": 364840000000.0,
            "form": "10-K",
            "accn": "B2",
        },
        # Old row that should be excluded by a `since` filter
        {
            "ticker": "AAPL",
            "concept": "us-gaap:Revenues",
            "end": "2021-12-31",
            "filed": "2022-01-28",
            "val": 365817000000.0,
            "form": "10-K",
            "accn": "A0",
        },
    ]
    df = pd.DataFrame(records)
    path = tmp_path / "pit_financials.parquet"
    df.to_parquet(path, index=False)
    return path


def test_init_with_ticker_filter_loads_subset(multi_ticker_parquet):
    """PitQuery(path, tickers=['AAPL']) loads only AAPL rows."""
    q = PitQuery(multi_ticker_parquet, tickers=["AAPL"])
    assert set(q.data["ticker"].unique()) == {"AAPL"}
    assert "MSFT" not in q.data["ticker"].values


def test_init_with_concept_filter(multi_ticker_parquet):
    """PitQuery(path, concepts=['us-gaap:Revenues']) loads only Revenues rows."""
    q = PitQuery(multi_ticker_parquet, concepts=["us-gaap:Revenues"])
    assert set(q.data["concept"].unique()) == {"us-gaap:Revenues"}
    assert "us-gaap:Assets" not in q.data["concept"].values


def test_init_with_since_filter_drops_old_rows(multi_ticker_parquet):
    """PitQuery(path, since='2023-01-01') drops rows filed before 2023-01-01."""
    q = PitQuery(multi_ticker_parquet, since="2023-01-01")
    assert (q.data["filed"] >= pd.Timestamp("2023-01-01")).all()
    # The AAPL row filed 2022-01-28 must be absent
    assert len(q.data[q.data["filed"] < pd.Timestamp("2023-01-01")]) == 0


def test_init_without_filters_is_backwards_compatible(multi_ticker_parquet):
    """PitQuery(path) with no filter kwargs loads all rows unchanged."""
    q = PitQuery(multi_ticker_parquet)
    assert set(q.data["ticker"].unique()) == {"AAPL", "MSFT"}
    assert set(q.data["concept"].unique()) == {"us-gaap:Revenues", "us-gaap:Assets"}
    assert len(q.data) == 5
def test_known_concepts_returns_sorted_list(concept_check_parquet):
    """known_concepts() returns a sorted list of all unique concept strings."""
    q = PitQuery(concept_check_parquet)
    concepts = q.known_concepts()
    assert isinstance(concepts, list)
    assert concepts == sorted(concepts)
    assert "us-gaap:Revenues" in concepts


def test_unknown_concept_warns_and_suggests(concept_check_parquet):
    """Typo'd concept emits a loguru warning mentioning a near-match suggestion."""
    import sys
    from loguru import logger

    q = PitQuery(concept_check_parquet)
    warnings: list[str] = []

    def _sink(message):
        warnings.append(str(message))

    handler_id = logger.add(_sink, level="WARNING")
    try:
        # "us-gaap:Revenue" is close to "us-gaap:Revenues" — difflib should suggest it
        q.as_of("AAPL", "us-gaap:Revenue", "2023-06-01")
    finally:
        logger.remove(handler_id)

    assert len(warnings) == 1
    assert "us-gaap:Revenue" in warnings[0]
    assert "Revenues" in warnings[0]


def test_unknown_concept_strict_raises(concept_check_parquet):
    """With strict_concepts=True an unknown concept raises KeyError."""
    q = PitQuery(concept_check_parquet, strict_concepts=True)
    with pytest.raises(KeyError, match="us-gaap:Revenue"):
        q.as_of("AAPL", "us-gaap:Revenue", "2023-06-01")


def test_unknown_concept_warn_once_per_concept_per_instance(concept_check_parquet):
    """The warning fires exactly once per unique unknown concept per PitQuery instance."""
    from loguru import logger

    q = PitQuery(concept_check_parquet)
    warnings: list[str] = []

    def _sink(message):
        warnings.append(str(message))

    handler_id = logger.add(_sink, level="WARNING")
    try:
        q.as_of("AAPL", "us-gaap:Revenue", "2023-06-01")
        q.as_of("AAPL", "us-gaap:Revenue", "2023-06-01")
        q.history("AAPL", "us-gaap:Revenue")
    finally:
        logger.remove(handler_id)

    # Three calls with the same typo on the same instance → only one warning
    assert len(warnings) == 1


def test_unknown_concept_warns_independently_per_new_instance(concept_check_parquet):
    """A new PitQuery instance resets the warned-concepts cache."""
    from loguru import logger

    warnings: list[str] = []

    def _sink(message):
        warnings.append(str(message))

    handler_id = logger.add(_sink, level="WARNING")
    try:
        q1 = PitQuery(concept_check_parquet)
        q1.as_of("AAPL", "us-gaap:Revenue", "2023-06-01")

        q2 = PitQuery(concept_check_parquet)
        q2.as_of("AAPL", "us-gaap:Revenue", "2023-06-01")
    finally:
        logger.remove(handler_id)

    # Two different instances → two warnings
    assert len(warnings) == 2
