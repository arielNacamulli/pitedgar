"""pitedgar — Point-in-time SEC EDGAR financial data pipeline.

Typical usage::

    from pitedgar import PitQuery, PitEdgarConfig, DEFAULT_CONCEPTS

    q = PitQuery("data/pit_financials.parquet")

    # Balance sheet snapshot for S&P 500 across 20 quarterly rebalance dates
    assets = q.cross_section("us-gaap:Assets", rebalance_dates)

    # TTM revenue for S&P 500 across the same dates
    revenue = q.ttm_cross_section("us-gaap:Revenues", rebalance_dates)

    # Ad-hoc PIT lookup
    row = q.as_of("AAPL", "us-gaap:Revenues", "2024-01-01")

Pipeline stages (run once to build the parquet)::

    from pitedgar import PitEdgarConfig, build_cik_map, download_bulk, parse_all

    config = PitEdgarConfig(edgar_identity="Your Name your@email.com", data_dir="data")
    cik_map = build_cik_map(tickers, config)
    download_bulk(config)
    parse_all(config, cik_map)
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from pitedgar.config import (
    CONCEPT_ALIASES,
    DEFAULT_CONCEPTS,
    DEFAULT_FORMS,
    PitEdgarConfig,
)

try:
    __version__ = _pkg_version("pitedgar")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
from pitedgar.downloader import download_bulk
from pitedgar.mapping import build_cik_map
from pitedgar.parser import is_scale_corrected, parse_all
from pitedgar.query import PitQuery
from pitedgar.util import normalize_ticker

__all__ = [
    "CONCEPT_ALIASES",
    "DEFAULT_CONCEPTS",
    "DEFAULT_FORMS",
    "PitEdgarConfig",
    "PitQuery",
    "build_cik_map",
    "download_bulk",
    "is_scale_corrected",
    "normalize_ticker",
    "parse_all",
]
