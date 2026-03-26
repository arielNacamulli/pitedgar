# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-03-26

### Changed
- `build_cik_map()` now caches results to `ticker_cik_map.parquet` and performs incremental updates — only unresolved tickers are fetched from EDGAR, avoiding redundant HTTP calls (~50s saved per run on S&P 500)
- Added `force: bool = False` parameter to `build_cik_map()` to bypass the cache when needed
- Added `--force` flag to `pitedgar map` CLI command

### Added
- `tests/conftest.py` — lightweight `edgar` stub injected via `sys.modules` to keep the test suite fast and offline
- 4 new unit tests covering cache hit, incremental update, `force` flag, and ticker normalization

## [0.1.0] - 2024-03-24

### Added
- 4-stage point-in-time pipeline: ticker mapping → bulk download → parquet build → query
- `PitEdgarConfig` Pydantic config model with 16 default XBRL concepts
- `build_cik_map()` — resolves ticker symbols to CIK numbers via edgartools
- `download_bulk()` — downloads `companyfacts.zip` (~1.5 GB) from SEC EDGAR
- `parse_all()` — parses per-company JSON into `pit_financials.parquet` with deduplication by `filed` date
- `PitQuery` — `as_of()`, `history()`, `cross_section()` methods with zero look-ahead bias
- Click-based CLI (`pitedgar map / fetch / build / query`)
- S&P 500 FCF benchmark example (`examples/fcf_sp500.py`)
