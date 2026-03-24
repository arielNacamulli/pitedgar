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
    result = q.as_of("AAPL", CONCEPT, "2022-06-01")
    assert len(result) == 1
    assert result.iloc[0]["val"] == pytest.approx(365817000000.0)


def test_as_of_includes_most_recent(parquet_path):
    q = PitQuery(parquet_path)
    result = q.as_of("AAPL", CONCEPT, "2023-06-01")
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
    xs = q.cross_section(CONCEPT, "2023-01-01")
    tickers = set(xs["ticker"])
    assert "AAPL" in tickers
    assert "MSFT" in tickers


def test_cross_section_subset(parquet_path):
    q = PitQuery(parquet_path)
    xs = q.cross_section(CONCEPT, "2023-01-01", tickers=["AAPL"])
    assert set(xs["ticker"]) == {"AAPL"}
