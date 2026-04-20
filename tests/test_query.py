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
    assert all((h["end"].dt.year == 2022))


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
