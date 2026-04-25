# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-04-25

Adversarial-review hardening pass: correctness fixes (PIT, scale, aliases),
security hardening (zip-slip, zip-bomb, ZIP integrity, HTTP retry), and
robustness improvements (mapping retry/rate-limit, concept validation,
contiguity checks). 27 issues landed (#12–#38).

### Changed (potentially breaking)
- **Scale correction is opt-in** (#12): `PitEdgarConfig.scale_correction`
  defaults to `"off"`. Pass `"auto"` for the heuristic (now requires ≥2 USD
  concepts below `scale_correction_threshold` and emits a `WARNING`) or
  `"force"` for unconditional ×1000. Previously legitimate micro/nano-caps
  with revenues under $1M were silently multiplied.
- **Lossy aliases are opt-in** (#14): `PitEdgarConfig.lossy_aliases_enabled`
  (default `False`) gates the three financially non-equivalent aliases
  (`ProfitLoss → NetIncomeLoss`, `LongTermDebtNoncurrent → LongTermDebt`,
  `CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents → CashAndCashEquivalentsAtCarryingValue`).
  When enabled the parser records the original tag in the new `alias_source` column.
- **TTM rejects non-contiguous quarters by default** (#18): `ttm()` /
  `ttm_cross_section()` now drop top-4 sets whose end span exceeds
  `max_ttm_span_days=400`. Pass `None` to disable.

### Added
- `PitQuery.history(as_of=...)` (#13): explicit PIT filter; default `None`
  preserves legacy "latest filed" behaviour with a docstring warning.
- `PitQuery(parquet_path, tickers=, concepts=, since=)` (#24): pyarrow
  filter pushdown to cut peak memory on R3000-scale parquets.
- `PitQuery(strict_concepts=True)` and `q.known_concepts()` (#29): typo'd
  concept names now warn (or raise) with `difflib` suggestions instead of
  silently returning NaN.
- `pitedgar.is_scale_corrected(df)` helper (#31).
- `pitedgar.normalize_ticker(t)` helper (#37).
- `tests/test_adversarial.py` umbrella + `pytest -m adversarial` marker (#38).
- CLI `pitedgar query --format {table,json,csv}` (#35).
- Adversarial fix coverage: future-date warning (#34), Q4 structural validation (#30),
  worker error isolation in parse_all (#33).

### Fixed (correctness)
- **Float-tolerance dedup** (#15): cross-filing dedup uses `np.isclose` with
  concept-appropriate atol (0.005 USD / 1e-6 shares) so 1-ULP serialisation
  drift no longer breaks the "unchanged comparative re-filing" guard.
- **Q4 derivation for 52/53-week fiscal years** (#16): prefers the 10-K's own
  `start` when present, falls back to a 355-day window (was 366), and validates
  monotonic ends + per-quarter gaps + Q1→Q3 span before computing.
- **YTD synthesis O(N+M) sweep** (#17): replaces the O(N×M) cartesian product
  on heavily-restated YTD pairs.
- **Explicit alias priority** (#25): new `CONCEPT_ALIAS_PRIORITY` makes
  precedence between competing aliases independent of dict insertion order.
- **Unit selection union** (#26): `parse_company` tries canonical-class then
  candidate-class unit preferences (defensive against future cross-class aliases).
- **Legacy parquet `duration_days` fallback** (#27): balance-sheet concepts
  (Assets, Liabilities, etc.) get `duration_days=-1` instead of the form-derived
  365 so they no longer leak into `is_annual` / TTM filters.
- **`ttm_cross_section` n_periods=0** for missing matches (#28): unifies "no
  data ever" and "no data before as_of_date" semantics.

### Fixed (security / robustness)
- **Zip-slip protection** (#19): every ZIP member now routes through
  `_safe_extract` which rejects absolute paths, parent-traversal, symlinks,
  and any target resolving outside `facts_dir`.
- **Zip-bomb cap** (#20): `_check_zip_size` rejects archives whose declared
  decompressed size exceeds `PitEdgarConfig.max_extracted_bytes` (default 25 GiB)
  or any single member > 10 GiB.
- **Bulk ZIP integrity** (#21): SHA-256 + `companyfacts.zip.meta.json` sidecars
  are written on download and verified on cache hit. Mismatch raises;
  legacy caches without sidecar warn and proceed.
- **Retry 429/5xx with `Retry-After`** (#22): 429/502/503/504 now retried
  (cap 5 min total wait) instead of propagating immediately.
- **`build_cik_map` retry + adaptive rate limit** (#23): per-call latency is
  measured to keep ≥105 ms between requests (well under SEC's 10 req/s cap);
  `AttributeError` / `KeyError` no longer silently swallowed (3 consecutive
  raises a schema-drift `RuntimeError`).
- **Worker error isolation** (#33): a per-worker exception in `parse_all` no
  longer discards results from already-completed workers; pass `strict=True`
  for the previous fail-fast behaviour.
- **`edgar_identity` regex validation** (#32): rejects malformed strings up
  front to avoid SEC IP bans.
- **CIK range validation** (#36): `build_cik_map` skips and warns on negative
  or oversized CIKs instead of writing 11-char padded strings.

### Tooling
- 229 tests (was 122 on master); regression coverage doubled.
- Adversarial test suite runnable in isolation: `pytest -m adversarial`.

## [0.3.2] - 2026-04-21

### Fixed
- **`PitQuery.ttm()` / `ttm_cross_section()` inflated TTM for YTD-only filers**: companies that tag their quarterly disclosures only as year-to-date cumulative values (notably AAPL post-2021 for `us-gaap:NetIncomeLoss` / `us-gaap:Revenues` — Q1=90d, Q2=181d, Q3=272d, 10-K=363d, all sharing the same fiscal-year `start`) had no discrete 3-month rows for the TTM algorithm to sum. Result: `_ttm_events` fell back to summing the four most recent 90-day Q1 entries from consecutive fiscal years, producing a ~35% inflated TTM (AAPL TTM NetIncomeLoss at 2024-11-01 returned ~$127B instead of the published FY2024 figure of $93.7B). The query layer now synthesizes the missing 3-month rows from consecutive YTD records sharing the same `start`: `Q2_3m = YTD_6m − YTD_3m`, `Q3_3m = YTD_9m − YTD_6m`, `Q4_3m = FY − YTD_9m`. Synthetic rows are filed at `max(prev.filed, curr.filed)` (no look-ahead) and their `accn` is suffixed with `:DERIVED_YTD_DIFF` for audit. Explicit 3-month rows continue to take precedence over synthesized ones at the same `(end, filed)`, so filers with discrete quarterly tags (e.g. pre-2020 MSFT) see identical TTM output.
- **`_parse_one_for_pool` type annotation**: `config.concepts` is `list[str] | None` (None means "parse every us-gaap concept"), which `parse_company` already accepts, but the pool helper's arg-tuple type was narrower — mypy flagged the mismatch and CI had been red on `master` since v0.3.1 ship.

## [0.3.1] - 2026-04-20

### Fixed
- **Parser alias resolution silently dropped post-ASC 606 data**: `parse_company` stopped at the first candidate tag (canonical or alias) that had entries, ignoring all others. Filers that reported `us-gaap:Revenues` even for a single historical period had their modern `RevenueFromContractWithCustomerExcludingAssessedTax` / `SalesRevenueNet` records discarded — e.g. AAPL's 109 post-2017 revenue records were lost because 11 pre-2019 rows under canonical `Revenues` matched first. The parser now unions entries from every candidate and dedups by `(end, filed, form)` with canonical-first precedence. Rebuild with `pitedgar build --force` to recover the missing coverage.
- **`PitQuery.history(freq='Q')` returned empty for companies reporting quarterly breakdowns in 10-K filings**: the filter previously required `form ∈ {10-Q, 10-Q/A}`, but companies like AAPL report historical quarterly `Revenues` as comparative data inside the annual 10-K. `history()` now uses `is_quarterly(duration_days)` (aligning with `ttm()`, which already did the right thing) so quarterly values filed on any form are surfaced.

## [0.3.0] - 2026-04-20

### Changed
- **BREAKING for callers relying on auto-nullification**: `PitQuery.as_of`,
  `PitQuery.cross_section`, and `PitQuery.ttm_cross_section` now default
  `max_staleness_days=None`. Previously the default was `100`, which silently
  set `val` / `ttm_val` to NaN whenever the most recent filing was older than
  100 days. The new default returns the value as-is and lets the caller decide.
  To restore the previous behavior, pass `max_staleness_days=100` explicitly.

### Added
- **`age_days` column** on every `as_of`, `cross_section`, and
  `ttm_cross_section` result: `(as_of_date − filed)` in days as a nullable
  `Int64`. Missing-data rows surface `<NA>` (not `0`) so they are
  distinguishable from a same-day filing. Use this to threshold staleness in
  user code instead of (or in addition to) `max_staleness_days`.

## [0.2.9] - 2026-04-18

### Changed
- **Parser default now extracts every us-gaap concept**: `PitEdgarConfig.concepts` defaults to `None`, which causes `parse_company` / `parse_all` to iterate every `us-gaap:*` tag present in the per-company JSON instead of only the 15 in `DEFAULT_CONCEPTS`. Disk is cheap and parquet compresses well, so the broader output eliminates the need to re-parse the 1.5 GB cache when iterating on new quant signals. `CONCEPT_ALIASES` is still applied, so e.g. a filer reporting only `RevenueFromContractWithCustomerExcludingAssessedTax` still lands under `us-gaap:Revenues` in the parquet. **Migration**: rebuild the parquet with `pitedgar build --force` to populate the broader concept set; pass `concepts=DEFAULT_CONCEPTS` to keep the curated subset.

## [0.2.8] - 2026-04-18

### Added
- **`n_workers` parameter on `parse_all`**: parallelizes JSON parsing across worker processes via `concurrent.futures.ProcessPoolExecutor`. Default `None` uses `os.cpu_count()`; `n_workers=1` keeps the legacy serial loop for debugging and single-core environments.
- **`--workers` / `-j` flag on `pitedgar build`**: surfaces the new parallelism on the CLI (e.g. `pitedgar build -j 8`).

### Changed
- **`parse_all` is parallel by default**: JSON parsing is CPU-bound, so fanning out across all cores yields a roughly 5–10x speedup on a typical laptop for ~500 companies. Pass `n_workers=1` (or `--workers 1`) to restore the old serial behavior.

## [0.2.7] - 2026-04-18

### Added
- **Amendments and foreign private issuers in `DEFAULT_FORMS`**: `10-K/A`, `10-Q/A`, `20-F`, and `20-F/A` are now ingested by default. Amendments carry corrected (often substantially restated) data and were previously dropped silently; foreign private issuers reporting on `20-F` were missing entirely. The parser's `(concept, end)` PIT dedup keeps both the original and the amendment as separate rows so the query layer can still pick the latest-filed value at any as-of date.
- **Tests**: parser keeps a `10-K/A` alongside the original `10-K` for the same `period_end`; `as_of()` returns the `10-K/A` restated value once the amendment has been filed; `is_annual()` accepts `10-K/A`, `20-F`, and `20-F/A`.

### Changed
- **Form-based filtering centralised**: `pitedgar.periods` now exposes `_ANNUAL_FORMS = {10-K, 10-K/A, 20-F, 20-F/A}` and `_QUARTERLY_FORMS = {10-Q, 10-Q/A}`. `is_annual()` and `PitQuery.history(freq=...)` use these sets instead of bare `form == "10-K"` / `"10-Q"` comparisons, so amendments and 20-F filings flow through the annual/quarterly code paths consistently.

### Migration
- Run `pitedgar build --force` to pick up `10-K/A`, `10-Q/A`, `20-F`, and `20-F/A` filings already present under `data/companyfacts/` — the parquet must be rebuilt for the new forms to appear.

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
