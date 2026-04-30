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


# ---------------------------------------------------------------------------
# New tests for retry, schema-drift guard, rate limit, and failure-rate warn
# ---------------------------------------------------------------------------


@patch("pitedgar.mapping.edgar.set_identity")
@patch("pitedgar.mapping.edgar.Company")
@patch("pitedgar.mapping.time.sleep")
def test_retry_recovers_from_transient_timeout(mock_sleep, mock_company_cls, mock_set_identity, config):
    """edgar.Company raising TimeoutError twice, then succeeding, should resolve."""
    success_company = _make_company(320193, "Apple Inc.")
    call_count = {"n": 0}

    def side_effect(ticker):
        call_count["n"] += 1
        if call_count["n"] <= 2:
            raise TimeoutError("connection timed out")
        return success_company

    mock_company_cls.side_effect = side_effect

    result = build_cik_map(["AAPL"], config)

    assert "AAPL" in result.index
    assert result.loc["AAPL", "cik"] == "0000320193"
    assert call_count["n"] == 3  # two failures + one success


@patch("pitedgar.mapping.edgar.set_identity")
@patch("pitedgar.mapping.edgar.Company")
@patch("pitedgar.mapping.time.sleep")
def test_three_consecutive_attribute_errors_raise(mock_sleep, mock_company_cls, mock_set_identity, config):
    """Three consecutive AttributeErrors must raise RuntimeError mentioning 'schema'."""
    mock_company_cls.side_effect = AttributeError("'NoneType' has no attribute 'cik'")

    with pytest.raises(RuntimeError, match="schema"):
        build_cik_map(["T1", "T2", "T3"], config)


@patch("pitedgar.mapping.edgar.set_identity")
@patch("pitedgar.mapping.edgar.Company")
def test_rate_limit_adaptive(mock_company_cls, mock_set_identity, config, monkeypatch):
    """sleep should be called with roughly max(0, 0.105 - latency)."""
    mock_company_cls.return_value = _make_company(320193, "Apple Inc.")

    # Simulate a call that takes 0.040 s → expected sleep ≈ 0.065 s.
    simulated_latency = 0.040
    time_calls = {"n": 0}
    sleep_calls = []

    def fake_time():
        # First call is t0, second call is after the edgar.Company() returns.
        v = time_calls["n"] * simulated_latency
        time_calls["n"] += 1
        return v

    def fake_sleep(seconds):
        sleep_calls.append(seconds)

    import pitedgar.mapping as mapping_module

    monkeypatch.setattr(mapping_module.time, "time", fake_time)
    monkeypatch.setattr(mapping_module.time, "sleep", fake_sleep)

    build_cik_map(["AAPL"], config)

    # There should be exactly one sleep call (from the rate limiter in finally).
    assert len(sleep_calls) == 1
    expected = max(0.0, 0.105 - simulated_latency)
    assert abs(sleep_calls[0] - expected) < 0.005, (
        f"Expected sleep ≈ {expected:.4f}s, got {sleep_calls[0]:.4f}s"
    )


@patch("pitedgar.mapping.edgar.set_identity")
@patch("pitedgar.mapping.edgar.Company")
@patch("pitedgar.mapping.time.sleep")
def test_high_failure_rate_warns(mock_sleep, mock_company_cls, mock_set_identity, config):
    """More than 20 % of tickers failing should emit a warning via loguru."""
    from loguru import logger as loguru_logger

    tickers = [f"T{i}" for i in range(10)]

    def side_effect(ticker):
        # Fail 4 out of 10 (40 %) → above 20 % threshold.
        if ticker in {"T0", "T1", "T2", "T3"}:
            raise ValueError("not found")
        return _make_company(int(ticker[1:]) + 1, f"Company {ticker}")

    mock_company_cls.side_effect = side_effect

    warning_messages: list[str] = []

    def capture_sink(message):
        if message.record["level"].no >= 30:  # WARNING = 30
            warning_messages.append(message.record["message"])

    handler_id = loguru_logger.add(capture_sink, level="WARNING")
    try:
        build_cik_map(tickers, config)
    finally:
        loguru_logger.remove(handler_id)

    assert any("failed" in msg.lower() or "%" in msg for msg in warning_messages), (
        f"Expected a high-failure-rate warning, got: {warning_messages}"
    )


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
