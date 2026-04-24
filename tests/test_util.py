"""Tests for pitedgar.util."""

from pitedgar.util import normalize_ticker


def test_normalize_ticker_strips_and_uppercases():
    assert normalize_ticker("  aapl  ") == "AAPL"
    assert normalize_ticker("msft") == "MSFT"
    assert normalize_ticker(" goog ") == "GOOG"


def test_normalize_ticker_idempotent():
    assert normalize_ticker(normalize_ticker("aapl")) == normalize_ticker("aapl")
    assert normalize_ticker("AAPL") == "AAPL"


def test_normalize_ticker_handles_unicode_spaces():
    assert normalize_ticker("\tAAPL\n") == "AAPL"
