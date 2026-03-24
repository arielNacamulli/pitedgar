# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
poetry install

# Run all tests
pytest

# Run a single test file
pytest tests/test_parser.py

# Run a single test
pytest tests/test_parser.py::test_deduplication

# Run the CLI
pitedgar --help
pitedgar map --tickers tickers.txt --identity "Name name@email.com"
pitedgar fetch --identity "Name name@email.com"
pitedgar build
pitedgar query --ticker AAPL --concept Revenues --as-of 2023-01-01
```

## Architecture

This is a 4-stage point-in-time financial data pipeline for SEC EDGAR data. "Point-in-time" means data is timestamped by `filed` date, not period-end date, to prevent look-ahead bias in backtesting.

**Pipeline stages:**

1. **`mapping.py`** — Resolve ticker symbols → CIK numbers via `edgartools.Company()`, cached to `ticker_cik_map.parquet`
2. **`downloader.py`** — Bulk download `companyfacts.zip` (~1.5GB) from SEC EDGAR and extract to `data/companyfacts/`
3. **`parser.py`** — Parse per-company JSON files into a master `pit_financials.parquet`. Key logic: for duplicate `(concept, end)` pairs, keeps the most recently `filed` record (not the most recent `end` date)
4. **`query.py`** — `PitQuery` class loads the parquet and exposes `as_of()`, `history()`, and `cross_section()` methods, all enforcing no look-ahead bias

**`config.py`** — `PitEdgarConfig` (Pydantic model) is the single config object threaded through all stages. Contains `DEFAULT_CONCEPTS` (16 XBRL metrics) and `DEFAULT_FORMS` (`["10-K", "10-Q"]`).

**`cli.py`** — Click-based CLI wrapping the 4 pipeline stages. All commands share `--data-dir` and `--identity` (SEC User-Agent string, required by EDGAR).

**Data flow:** `data/ticker_cik_map.parquet` → `data/companyfacts/*.json` → `data/pit_financials.parquet`

**Unit selection in parser:** EPS/share-count concepts use `"shares"` units; all other concepts use `"USD"`.
