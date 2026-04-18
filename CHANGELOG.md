# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.6] - 2026-04-18

### Added
- **Expanded `CONCEPT_ALIASES`** for revenue, net income, cash, and long-term debt families. Companies that file under deprecated or variant XBRL tags (e.g. `SalesRevenueNet`, `RevenueFromContractWithCustomerIncludingAssessedTax`, `Cash`, `CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents`, `LongTermDebtNoncurrent`, `ProfitLoss`) now appear in the parquet under the canonical concept, eliminating coverage gaps for filers that never used the canonical tag. The canonical-first lookup in `parse_company` ensures the canonical value always wins when both tags are present, so existing parquets are unaffected for those filers — only previously-empty rows are filled in. Rebuild with `pitedgar build --force` to backfill historical coverage.
- **Tests**: alias-only filers now resolve under the canonical concept; canonical-vs-alias precedence is verified for every newly added family.

## [0.2.4] - 2026-04-18

### Added
- **QA tooling baseline**: ruff (lint + format), mypy with pandas-stubs, pre-commit hooks, and a coverage gate (>= 90%) enforced in CI.
- **`scale_corrected` column** in the parser output: marks rows where the 1000x thousands-to-dollars rescale fired, making the correction auditable downstream.
- **CLI, downloader-error, and end-to-end integration tests**: coverage raised from 85% to 94% (cli.py was previously excluded entirely).

### Fixed
- **`__version__` drift**: `pitedgar.__version__` is now read from package metadata instead of a hardcoded literal that had fallen behind `pyproject.toml`.
- **CLI tracebacks on missing inputs**: `pitedgar build` and `pitedgar query` now emit a ClickException pointing users at the prior pipeline step instead of a raw `FileNotFoundError`. `--as-of` is validated up-front and unreadable/empty tickers files are rejected early.
- **Downloader atomicity and resilience**: the 1.5 GB bulk ZIP is now streamed to a `.part` sidecar and atomically renamed; transient `ConnectionError`/`Timeout`/`ChunkedEncodingError` are retried with exponential backoff; a corrupt ZIP on disk is deleted with a clear error so the next run re-fetches.
- **Atomic parquet writes** in `mapping.py` and `parser.py`: a crash mid-write no longer leaves a truncated parquet that later reads would choke on.
- **Narrowed `build_cik_map` exception handling**: only `ValueError`/`KeyError`/`LookupError`/`AttributeError` are treated as unresolvable-ticker skips; genuine network or auth failures now propagate.

## [0.2.3] - 2026-04-12

### Fixed
- **Period classification inconsistency between parser and query**: the parser used 60–100 days for "quarterly" during within-filing dedup while the query layer used 60–105 days for Q4 derivation, TTM, and snapshot selection. Companies with ~101–105 day quarters (e.g. JNJ Q4) could be misclassified during dedup. Both layers now share a single source of truth in the new `pitedgar.periods` module (60–105 days).

### Changed
- **New `pitedgar.periods` module**: centralizes day-count thresholds (`Q_MIN`/`Q_MAX`/`A_MIN`/`A_MAX`) and exposes `is_quarterly` / `is_annual` boolean masks. Internal refactor; no public API changes.

## [0.2.2] - 2026-04-12

### Fixed
- **`build_cik_map` crashed when no new tickers resolved**: if every new ticker failed resolution (e.g. delisted/historical symbols), the empty `records` list raised `KeyError: 'ticker'` on `set_index`. Now returns the existing map unchanged, or an empty typed frame when no map exists yet.

## [0.2.1] - 2026-03-29

### Fixed
- **Parser cross-filing dedup kept latest instead of earliest**: the initial parser kept the most-recently-filed record per `(concept, end)`, inflating PIT filing dates by months or years for companies with restatements (e.g. JNJ post-Kenvue spinoff). The dedup now keeps the **earliest** distinct-value filing, preserving the original PIT date. Affected parquets must be rebuilt with `pitedgar build --force`.
- **`pitedgar build` missing `--force` flag**: the CLI `build` command did not expose the `force` parameter of `parse_all()`. Added `--force` flag so re-parsing can be triggered without manually deleting the parquet.

### Changed
- **Example `pit_earnings_csv.py`**: `max_staleness_days` corrected to 100 (was inflated to work around the parser bug above).

## [0.2.0] - 2026-03-28

### Fixed
- **Restatement history preserved**: parser no longer deduplicates across different filing dates; the query layer now enforces PIT dedup at query time (`filed <= as_of_date`, then latest-filed per period end). A Q1 restatement filed after Q2 now correctly supersedes the original Q1 value from its restatement date onward.
- **Wrong CapEx concept**: replaced `CapitalExpendituresIncurredButNotYetPaid` (accrued-but-unpaid capex) with `PaymentsToAcquirePropertyPlantAndEquipment` (actual cash capex) in `DEFAULT_CONCEPTS`.
- **Duplicate OCF concept**: removed `OperatingCashFlow` from `DEFAULT_CONCEPTS`; `NetCashProvidedByUsedInOperatingActivities` is the canonical tag.
- **`as_of()` returns latest period end, not latest filing**: after PIT dedup by `(ticker, end)`, the method now picks `idxmax("end")` per ticker so a late restatement of an old period never shadows the most recent quarter.
- **Quarterly dedup window widened**: lower bound changed from 70 to 60 days to correctly identify discrete quarters in 52/53-week fiscal calendars.
- **USD scale correction**: companies that report USD values in thousands (max absolute value < $1M) are silently corrected by 1000× at parse time. Share/EPS concepts are unaffected.

### Added
- **`CONCEPT_ALIASES`**: maps deprecated/variant XBRL tags to canonical names (e.g. `RevenueFromContractWithCustomerExcludingAssessedTax` → `Revenues`). Applied at parse time so the parquet always uses canonical names; post-ASC 606 filers are now captured under `us-gaap:Revenues` automatically.
- **`PitQuery.ttm()`**: point-in-time trailing-twelve-month series for a single ticker/concept from 10-Q filings. Uses a running-state dict (O(n log n)) instead of a per-date DataFrame scan.
- **`PitQuery.ttm_cross_section()`**: bulk TTM for a full universe across multiple rebalance dates. Optimized with a single grouped `_ttm_events` pass + one vectorized `merge_asof(by="ticker")` — no per-date or per-ticker Python loops.
- **`PitQuery.cross_section()` multi-date support**: now accepts a list or `DatetimeIndex` of dates (same interface as `ttm_cross_section`). Uses `_snapshot_events` step-functions + single `merge_asof(by="ticker")` for S&P 500 / Russell 3000 scale.
- **Clean public API**: `DEFAULT_CONCEPTS`, `DEFAULT_FORMS`, and `CONCEPT_ALIASES` are now exported from the top-level `pitedgar` package alongside `PitQuery`, `PitEdgarConfig`, and the pipeline functions.
- **`max_staleness_days` default reduced**: 180 → 100 days, matching a quarterly filing cycle.
- 15 new tests covering restatement PIT correctness, concept aliasing, scale correction, `ttm()`, `ttm_cross_section()`, and multi-date `cross_section()`.

## [0.1.2] - 2026-03-26

### Changed
- `download_bulk()` now skips the HTTP download if `companyfacts.zip` already exists and `force=False` (~1.27 GB saved per run)
- `parse_all()` now skips the parse loop if `pit_financials.parquet` already exists and `force=False` (~16s saved per run); added `force: bool = False` parameter

### Added
- 5 new unit tests covering cache-skip and `force` flag behaviour for both `download_bulk` and `parse_all`

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
