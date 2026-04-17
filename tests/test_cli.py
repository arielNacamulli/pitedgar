"""Tests for the click-based CLI in pitedgar.cli.

These cover error paths that previously tracebacked with raw pandas/OS exceptions:
missing input files, unreadable tickers lists, invalid dates, missing build artifacts.
"""

import pandas as pd
import pytest
from click.testing import CliRunner

from pitedgar.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


def test_map_rejects_empty_tickers_file(runner, tmp_path):
    tickers_file = tmp_path / "tickers.txt"
    tickers_file.write_text("")

    result = runner.invoke(
        cli,
        [
            "map",
            "--tickers",
            str(tickers_file),
            "--identity",
            "Test test@example.com",
            "--data-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0
    assert "empty" in result.output.lower()


def test_map_rejects_missing_tickers_file(runner, tmp_path):
    result = runner.invoke(
        cli,
        [
            "map",
            "--tickers",
            str(tmp_path / "does_not_exist.txt"),
            "--identity",
            "Test test@example.com",
            "--data-dir",
            str(tmp_path),
        ],
    )
    # Click's Path(exists=True) validator rejects before our handler runs.
    assert result.exit_code != 0


def test_map_rejects_empty_identity(runner, tmp_path):
    tickers_file = tmp_path / "tickers.txt"
    tickers_file.write_text("AAPL\n")

    result = runner.invoke(
        cli,
        [
            "map",
            "--tickers",
            str(tickers_file),
            "--identity",
            "   ",
            "--data-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0
    assert "edgar_identity" in result.output or "identity" in result.output.lower()


def test_build_fails_helpfully_when_cik_map_missing(runner, tmp_path):
    # No ticker_cik_map.parquet exists: we expect a ClickException, not a
    # FileNotFoundError traceback.
    result = runner.invoke(
        cli,
        [
            "build",
            "--identity",
            "Test test@example.com",
            "--data-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0
    assert "ticker_cik_map.parquet" in result.output
    assert "pitedgar map" in result.output


def test_query_fails_helpfully_when_parquet_missing(runner, tmp_path):
    result = runner.invoke(
        cli,
        [
            "query",
            "--ticker",
            "AAPL",
            "--concept",
            "us-gaap:Revenues",
            "--as-of",
            "2023-01-01",
            "--data-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0
    assert "pit_financials.parquet" in result.output
    assert "pitedgar build" in result.output


def test_query_rejects_invalid_date(runner, tmp_path):
    # Even with a valid parquet in place, a garbage --as-of should fail usage-early.
    parquet = tmp_path / "pit_financials.parquet"
    pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "concept": ["us-gaap:Revenues"],
            "end": [pd.Timestamp("2022-12-31")],
            "filed": [pd.Timestamp("2023-02-01")],
            "val": [1.0e9],
            "form": ["10-K"],
            "duration_days": [365],
        }
    ).to_parquet(parquet)

    result = runner.invoke(
        cli,
        [
            "query",
            "--ticker",
            "AAPL",
            "--concept",
            "us-gaap:Revenues",
            "--as-of",
            "not-a-date",
            "--data-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0
    assert "as-of" in result.output.lower() or "not-a-date" in result.output


def test_query_happy_path(runner, tmp_path):
    """End-to-end cmd_query with a prebuilt parquet."""
    parquet = tmp_path / "pit_financials.parquet"
    pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "concept": ["us-gaap:Revenues"],
            "end": [pd.Timestamp("2022-12-31")],
            "filed": [pd.Timestamp("2023-02-01")],
            "val": [1.0e9],
            "form": ["10-K"],
            "duration_days": [365],
        }
    ).to_parquet(parquet)

    result = runner.invoke(
        cli,
        [
            "query",
            "--ticker",
            "aapl",  # lowercase should be uppercased
            "--concept",
            "us-gaap:Revenues",
            "--as-of",
            "2023-02-15",
            "--data-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "AAPL" in result.output


def test_map_happy_path_with_stubbed_edgar(runner, tmp_path, mocker):
    """cmd_map should write the parquet and echo a summary."""
    import pitedgar.mapping as mapping_mod

    mock_company = mocker.MagicMock()
    mock_company.cik = 320193
    mock_company.name = "Apple Inc."
    mock_company.sic = "3571"
    mock_company.fiscal_year_end = "09-28"
    mock_company.exchange = "NASDAQ"
    mocker.patch.object(mapping_mod.edgar, "set_identity")
    mocker.patch.object(mapping_mod.edgar, "Company", return_value=mock_company)
    mocker.patch.object(mapping_mod.time, "sleep")

    tickers_file = tmp_path / "tickers.txt"
    tickers_file.write_text("AAPL\n")

    result = runner.invoke(
        cli,
        [
            "map",
            "--tickers",
            str(tickers_file),
            "--identity",
            "Test test@example.com",
            "--data-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "ticker_cik_map.parquet").exists()
    assert "1" in result.output  # "Mapped 1 tickers"


def test_build_happy_path_with_stubbed_facts(runner, tmp_path):
    """cmd_build should read the CIK map and call parse_all, writing a parquet."""
    # Pre-seed the CIK map (normally produced by cmd_map).
    cik_map = pd.DataFrame({"cik": ["0000320193"]}, index=pd.Index(["AAPL"], name="ticker"))
    cik_map.to_parquet(tmp_path / "ticker_cik_map.parquet")

    # Pre-seed a minimal company facts JSON under the default facts_dir.
    facts_dir = tmp_path / "companyfacts"
    facts_dir.mkdir()
    _sample = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "start": "2022-01-01",
                                "end": "2022-12-31",
                                "filed": "2023-02-01",
                                "val": 1_000_000_000.0,
                                "form": "10-K",
                                "accn": "A1",
                            }
                        ]
                    }
                }
            }
        }
    }
    import json as _json

    (facts_dir / "CIK0000320193.json").write_text(_json.dumps(_sample))

    result = runner.invoke(
        cli,
        [
            "build",
            "--identity",
            "Test test@example.com",
            "--data-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "pit_financials.parquet").exists()


def test_help_works(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "map" in result.output
    assert "fetch" in result.output
    assert "build" in result.output
    assert "query" in result.output
