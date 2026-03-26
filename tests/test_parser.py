"""Tests for parser.py"""

import json
from pathlib import Path

import pandas as pd
import pytest

from pitedgar.config import PitEdgarConfig
from pitedgar.parser import parse_company, parse_all

SAMPLE_FACTS = {
    "facts": {
        "us-gaap": {
            "Revenues": {
                "units": {
                    "USD": [
                        {"end": "2022-12-31", "filed": "2023-02-01", "val": 1000000, "form": "10-K", "accn": "A1"},
                        {"end": "2022-12-31", "filed": "2023-03-15", "val": 1050000, "form": "10-K", "accn": "A2"},
                        {"end": "2022-09-30", "filed": "2022-11-01", "val": 900000, "form": "10-Q", "accn": "A3"},
                    ]
                }
            },
            "EarningsPerShareBasic": {
                "units": {
                    "shares": [
                        {"end": "2022-12-31", "filed": "2023-02-01", "val": 5.5, "form": "10-K", "accn": "B1"},
                    ]
                }
            },
        }
    }
}


@pytest.fixture
def facts_dir(tmp_path):
    cik = "0000320193"
    json_path = tmp_path / f"CIK{cik}.json"
    json_path.write_text(json.dumps(SAMPLE_FACTS), encoding="utf-8")
    return tmp_path


def test_parse_company_returns_correct_columns(facts_dir):
    df = parse_company("0000320193", ["us-gaap:Revenues"], facts_dir, ["10-K", "10-Q"])
    assert set(df.columns) == {"cik", "concept", "end", "filed", "val", "form", "accn"}


def test_parse_company_pit_deduplication(facts_dir):
    df = parse_company("0000320193", ["us-gaap:Revenues"], facts_dir, ["10-K", "10-Q"])
    revenues_annual = df[(df["concept"] == "us-gaap:Revenues") & (df["form"] == "10-K")]
    # Two entries for same end date: dedup keeps the most recently filed
    assert len(revenues_annual) == 1
    assert revenues_annual.iloc[0]["val"] == 1050000
    assert str(revenues_annual.iloc[0]["accn"]) == "A2"


def test_parse_company_form_filter(facts_dir):
    df = parse_company("0000320193", ["us-gaap:Revenues"], facts_dir, ["10-K"])
    assert all(df["form"] == "10-K")


def test_parse_company_share_concept_uses_shares_units(facts_dir):
    df = parse_company("0000320193", ["us-gaap:EarningsPerShareBasic"], facts_dir, ["10-K"])
    assert len(df) == 1
    assert df.iloc[0]["val"] == pytest.approx(5.5)


def test_parse_company_missing_file(tmp_path):
    df = parse_company("9999999999", ["us-gaap:Revenues"], tmp_path, ["10-K"])
    assert df.empty


def test_parse_all(tmp_path):
    # Write fact for one company
    cik = "0000320193"
    facts_dir = tmp_path / "companyfacts"
    facts_dir.mkdir()
    (facts_dir / f"CIK{cik}.json").write_text(json.dumps(SAMPLE_FACTS), encoding="utf-8")

    config = PitEdgarConfig(
        edgar_identity="Test test@example.com",
        data_dir=tmp_path,
        facts_dir=facts_dir,
    )
    cik_map = pd.DataFrame({"cik": [cik]}, index=pd.Index(["AAPL"], name="ticker"))
    master = parse_all(config, cik_map)

    assert "ticker" in master.columns
    assert set(master["ticker"]) == {"AAPL"}
    assert (tmp_path / "pit_financials.parquet").exists()


def test_parse_all_skips_when_parquet_exists(tmp_path):
    """parse_all must return cached parquet without running the parse loop."""
    cik = "0000320193"
    facts_dir = tmp_path / "companyfacts"
    facts_dir.mkdir()
    (facts_dir / f"CIK{cik}.json").write_text(json.dumps(SAMPLE_FACTS), encoding="utf-8")

    config = PitEdgarConfig(
        edgar_identity="Test test@example.com",
        data_dir=tmp_path,
        facts_dir=facts_dir,
    )
    cik_map = pd.DataFrame({"cik": [cik]}, index=pd.Index(["AAPL"], name="ticker"))

    # First call: produces the parquet.
    first = parse_all(config, cik_map)

    # Second call (force=False): must return the cached file, not re-parse.
    # We verify by patching parse_company — it must NOT be called.
    from unittest.mock import patch

    with patch("pitedgar.parser.parse_company") as mock_pc:
        second = parse_all(config, cik_map, force=False)
        mock_pc.assert_not_called()

    assert list(second.columns) == list(first.columns)
    assert len(second) == len(first)


def test_parse_all_force_reparses(tmp_path):
    """force=True must re-run the parse loop even if parquet exists."""
    cik = "0000320193"
    facts_dir = tmp_path / "companyfacts"
    facts_dir.mkdir()
    (facts_dir / f"CIK{cik}.json").write_text(json.dumps(SAMPLE_FACTS), encoding="utf-8")

    config = PitEdgarConfig(
        edgar_identity="Test test@example.com",
        data_dir=tmp_path,
        facts_dir=facts_dir,
    )
    cik_map = pd.DataFrame({"cik": [cik]}, index=pd.Index(["AAPL"], name="ticker"))

    # Create the parquet first.
    parse_all(config, cik_map)

    from unittest.mock import patch, call
    import pitedgar.parser as parser_mod

    with patch.object(parser_mod, "parse_company", wraps=parser_mod.parse_company) as mock_pc:
        parse_all(config, cik_map, force=True)
        assert mock_pc.call_count >= 1
