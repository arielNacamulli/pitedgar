"""Tests for mapping.py"""

from unittest.mock import MagicMock, patch

import pytest

from pitedgar.config import PitEdgarConfig
from pitedgar.mapping import build_cik_map


@pytest.fixture
def config(tmp_path):
    return PitEdgarConfig(edgar_identity="Test User test@example.com", data_dir=tmp_path)


def _make_company(cik: int, name: str, sic: str = "7372"):
    company = MagicMock()
    company.cik = cik
    company.name = name
    company.sic = sic
    company.fiscal_year_end = "12-31"
    company.exchange = "NASDAQ"
    return company


@patch("pitedgar.mapping.edgar.set_identity")
@patch("pitedgar.mapping.edgar.Company")
@patch("pitedgar.mapping.time.sleep")
def test_build_cik_map_success(mock_sleep, mock_company_cls, mock_set_identity, config):
    mock_company_cls.side_effect = lambda ticker: {
        "AAPL": _make_company(320193, "Apple Inc."),
        "MSFT": _make_company(789019, "Microsoft Corp."),
    }[ticker]

    result = build_cik_map(["AAPL", "MSFT"], config)

    assert set(result.index) == {"AAPL", "MSFT"}
    assert result.loc["AAPL", "cik"] == "0000320193"
    assert result.loc["MSFT", "cik"] == "0000789019"
    assert (config.data_dir / "ticker_cik_map.parquet").exists()


@patch("pitedgar.mapping.edgar.set_identity")
@patch("pitedgar.mapping.edgar.Company")
@patch("pitedgar.mapping.time.sleep")
def test_build_cik_map_bad_ticker_skipped(mock_sleep, mock_company_cls, mock_set_identity, config):
    def side_effect(ticker):
        if ticker == "FAKE":
            raise ValueError("Not found")
        return _make_company(320193, "Apple Inc.")

    mock_company_cls.side_effect = side_effect

    result = build_cik_map(["AAPL", "FAKE"], config)

    assert "AAPL" in result.index
    assert "FAKE" not in result.index


@patch("pitedgar.mapping.edgar.set_identity")
@patch("pitedgar.mapping.edgar.Company")
@patch("pitedgar.mapping.time.sleep")
def test_build_cik_map_cik_zero_padded(mock_sleep, mock_company_cls, mock_set_identity, config):
    mock_company_cls.return_value = _make_company(1, "Tiny Corp.")
    result = build_cik_map(["TINY"], config)
    assert result.loc["TINY", "cik"] == "0000000001"


@patch("pitedgar.mapping.edgar.set_identity")
@patch("pitedgar.mapping.edgar.Company")
@patch("pitedgar.mapping.time.sleep")
def test_cache_used_on_second_call(mock_sleep, mock_company_cls, mock_set_identity, config):
    """Second call with same tickers must not hit EDGAR."""
    mock_company_cls.return_value = _make_company(320193, "Apple Inc.")

    build_cik_map(["AAPL"], config)
    call_count_after_first = mock_company_cls.call_count

    result = build_cik_map(["AAPL"], config)

    assert mock_company_cls.call_count == call_count_after_first  # no new calls
    assert result.loc["AAPL", "cik"] == "0000320193"


@patch("pitedgar.mapping.edgar.set_identity")
@patch("pitedgar.mapping.edgar.Company")
@patch("pitedgar.mapping.time.sleep")
def test_incremental_update(mock_sleep, mock_company_cls, mock_set_identity, config):
    """Second call adds only new tickers, keeps existing ones."""
    mock_company_cls.side_effect = lambda ticker: {
        "AAPL": _make_company(320193, "Apple Inc."),
        "MSFT": _make_company(789019, "Microsoft Corp."),
    }[ticker]

    build_cik_map(["AAPL"], config)
    assert mock_company_cls.call_count == 1

    result = build_cik_map(["AAPL", "MSFT"], config)
    assert mock_company_cls.call_count == 2  # only MSFT was fetched

    assert set(result.index) == {"AAPL", "MSFT"}
    assert result.loc["AAPL", "cik"] == "0000320193"
    assert result.loc["MSFT", "cik"] == "0000789019"


@patch("pitedgar.mapping.edgar.set_identity")
@patch("pitedgar.mapping.edgar.Company")
@patch("pitedgar.mapping.time.sleep")
def test_force_reruns_all_tickers(mock_sleep, mock_company_cls, mock_set_identity, config):
    """force=True must re-resolve all tickers ignoring cache."""
    mock_company_cls.return_value = _make_company(320193, "Apple Inc.")

    build_cik_map(["AAPL"], config)
    assert mock_company_cls.call_count == 1

    build_cik_map(["AAPL"], config, force=True)
    assert mock_company_cls.call_count == 2  # called again despite cache


@patch("pitedgar.mapping.edgar.set_identity")
@patch("pitedgar.mapping.edgar.Company")
@patch("pitedgar.mapping.time.sleep")
def test_lowercase_tickers_normalized(mock_sleep, mock_company_cls, mock_set_identity, config):
    """Lowercase tickers should be uppercased and resolved correctly."""
    mock_company_cls.return_value = _make_company(320193, "Apple Inc.")

    result = build_cik_map(["aapl"], config)
    assert "AAPL" in result.index

    # Second call with uppercase should hit cache, not EDGAR again
    build_cik_map(["AAPL"], config)
    assert mock_company_cls.call_count == 1


# --- CIK validation tests (#36) ---


@patch("pitedgar.mapping.logger")
@patch("pitedgar.mapping.edgar.set_identity")
@patch("pitedgar.mapping.edgar.Company")
@patch("pitedgar.mapping.time.sleep")
def test_negative_cik_skipped(mock_sleep, mock_company_cls, mock_set_identity, mock_logger, config):
    """A company with a negative CIK must be skipped with a warning."""
    bad = MagicMock()
    bad.cik = -1
    bad.name = "Bad Corp"
    mock_company_cls.return_value = bad

    result = build_cik_map(["BAD"], config)

    assert "BAD" not in result.index
    warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
    assert any("Invalid CIK" in msg for msg in warning_calls)


@patch("pitedgar.mapping.logger")
@patch("pitedgar.mapping.edgar.set_identity")
@patch("pitedgar.mapping.edgar.Company")
@patch("pitedgar.mapping.time.sleep")
def test_too_large_cik_skipped(mock_sleep, mock_company_cls, mock_set_identity, mock_logger, config):
    """A company with a CIK > 9_999_999_999 must be skipped with a warning."""
    bad = MagicMock()
    bad.cik = 10_000_000_000
    bad.name = "Huge Corp"
    mock_company_cls.return_value = bad

    result = build_cik_map(["HUGE"], config)

    assert "HUGE" not in result.index
    warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
    assert any("Invalid CIK" in msg for msg in warning_calls)


@patch("pitedgar.mapping.logger")
@patch("pitedgar.mapping.edgar.set_identity")
@patch("pitedgar.mapping.edgar.Company")
@patch("pitedgar.mapping.time.sleep")
def test_non_int_cik_skipped(mock_sleep, mock_company_cls, mock_set_identity, mock_logger, config):
    """A company whose CIK is a string must be skipped with a warning."""
    bad = MagicMock()
    bad.cik = "320193"
    bad.name = "String CIK Corp"
    mock_company_cls.return_value = bad

    result = build_cik_map(["SCIK"], config)

    assert "SCIK" not in result.index
    warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
    assert any("Invalid CIK" in msg for msg in warning_calls)


@patch("pitedgar.mapping.edgar.set_identity")
@patch("pitedgar.mapping.edgar.Company")
@patch("pitedgar.mapping.time.sleep")
def test_valid_cik_regression(mock_sleep, mock_company_cls, mock_set_identity, config):
    """Regression: a normal valid CIK is still padded and stored correctly."""
    mock_company_cls.return_value = _make_company(320193, "Apple Inc.")

    result = build_cik_map(["AAPL"], config)

    assert "AAPL" in result.index
    assert result.loc["AAPL", "cik"] == "0000320193"
