"""Step 1: resolve ticker → CIK via edgartools."""

import time

import edgar
import pandas as pd
from loguru import logger

from pitedgar.config import PitEdgarConfig
from pitedgar.util import normalize_ticker

# Exceptions we expect edgartools to raise for unresolvable tickers (delisted,
# typos, historical symbols). Anything outside this set — network/auth/parse
# errors — should propagate so operators notice and fix the environment.
#
# NOTE: AttributeError and KeyError are intentionally excluded. If edgartools
# starts raising those for valid tickers it likely indicates schema drift in
# the library. We surface that via the consecutive-error guard below rather
# than silently skipping tickers.
_TICKER_LOOKUP_SKIPPABLE: tuple[type[BaseException], ...] = (
    ValueError,
    LookupError,
)

# Exceptions that indicate a transient network problem worth retrying.
_TRANSIENT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
)

_MAX_RETRIES = 3
_MIN_CALL_INTERVAL = 0.105  # seconds — keeps us safely under SEC's 10 req/s cap
_CONSECUTIVE_SCHEMA_ERRORS_LIMIT = 3
_HIGH_FAILURE_RATE_THRESHOLD = 0.20  # 20 %


def _fetch_company_with_retry(ticker: str) -> object:
    """Call ``edgar.Company(ticker)`` with up to *_MAX_RETRIES* retries.

    Retries on :class:`ConnectionError` / :class:`TimeoutError` with
    exponential back-off (``2**attempt`` seconds).  All other exceptions
    propagate immediately.
    """
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            return edgar.Company(ticker)
        except _TRANSIENT_EXCEPTIONS as exc:
            last_exc = exc
            backoff = 2 ** attempt
            logger.warning(
                f"Transient error for '{ticker}' (attempt {attempt + 1}/{_MAX_RETRIES}): "
                f"{exc!r}. Retrying in {backoff}s…"
            )
            time.sleep(backoff)
    # All retries exhausted — propagate the last transient exception.
    raise last_exc  # type: ignore[misc]


def _write_parquet_atomic(df: pd.DataFrame, dest_path) -> None:
    """Write *df* to *dest_path* via a .tmp sidecar + atomic rename.

    Prevents a crash mid-write from leaving a truncated parquet file behind
    that a later run would then try to read and fail on.
    """
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    try:
        df.to_parquet(tmp_path)
        tmp_path.replace(dest_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def build_cik_map(tickers: list[str], config: PitEdgarConfig, force: bool = False) -> pd.DataFrame:
    """Resolve a list of tickers to CIK numbers via edgartools.

    Saves the result to data_dir/ticker_cik_map.parquet.
    Returns a DataFrame indexed by ticker with columns:
        cik, name, sic, fiscal_year_end, exchange

    If the parquet cache already exists, only resolves tickers not yet present.
    Pass force=True to re-resolve all tickers from scratch.
    """
    edgar.set_identity(config.edgar_identity)
    config.ensure_dirs()

    out_path = config.data_dir / "ticker_cik_map.parquet"

    existing_df = pd.DataFrame()
    if not force and out_path.exists():
        existing_df = pd.read_parquet(out_path)
        logger.info(f"Loaded existing CIK map: {len(existing_df)} tickers cached")

    tickers_upper = [normalize_ticker(t) for t in tickers]
    cached_tickers = set(existing_df.index) if not existing_df.empty else set()
    new_tickers = [t for t in tickers_upper if t not in cached_tickers]

    if not new_tickers:
        logger.info("All tickers already cached, nothing to resolve")
        return existing_df

    logger.info(f"Resolving {len(new_tickers)} new ticker(s) via EDGAR...")
    records: list[dict] = []
    failed_count = 0
    consecutive_schema_errors = 0

    for ticker in new_tickers:
        t0 = time.time()
        try:
            company = _fetch_company_with_retry(ticker)
            cik = company.cik
            if not isinstance(cik, int) or cik < 0 or cik > 9_999_999_999:
                logger.warning(f"Invalid CIK for ticker {ticker!r}: {cik!r} (skipping)")
                continue
            cik_padded = f"{cik:010d}"
            records.append(
                {
                    "ticker": ticker,
                    "cik": cik_padded,
                    "name": getattr(company, "name", None),
                    "sic": getattr(company, "sic", None),
                    "fiscal_year_end": getattr(company, "fiscal_year_end", None),
                    "exchange": getattr(company, "exchange", None),
                }
            )
            logger.debug(f"{ticker} → CIK {cik_padded}")
            consecutive_schema_errors = 0
        except _TICKER_LOOKUP_SKIPPABLE as exc:
            logger.warning(f"Could not resolve ticker '{ticker}': {exc!r}")
            failed_count += 1
            consecutive_schema_errors = 0
        except (AttributeError, KeyError) as exc:
            consecutive_schema_errors += 1
            logger.warning(
                f"Schema-like error for '{ticker}' "
                f"({consecutive_schema_errors} consecutive): {exc!r}"
            )
            failed_count += 1
            if consecutive_schema_errors >= _CONSECUTIVE_SCHEMA_ERRORS_LIMIT:
                raise RuntimeError(
                    f"{consecutive_schema_errors} consecutive schema-drift errors encountered "
                    f"(last ticker: '{ticker}'). This likely indicates edgartools has changed "
                    "its internal schema. Please update or pin edgartools and re-run."
                ) from exc
        finally:
            # Adaptive rate limit: sleep only the remainder of the minimum interval.
            latency = time.time() - t0
            time.sleep(max(0.0, _MIN_CALL_INTERVAL - latency))

    # Aggregate failure warning.
    if new_tickers and failed_count / len(new_tickers) > _HIGH_FAILURE_RATE_THRESHOLD:
        logger.warning(
            f"{failed_count}/{len(new_tickers)} tickers ({failed_count / len(new_tickers):.0%}) "
            "failed to resolve. Verify your --identity string and consider retrying. "
            "Some tickers may be delisted or historical."
        )

    if not records:
        # All new tickers failed resolution (e.g. historical/delisted symbols).
        # Return the existing map unchanged rather than crashing on set_index.
        logger.info("No new tickers resolved; returning existing CIK map unchanged.")
        if not existing_df.empty:
            return existing_df
        return pd.DataFrame(
            index=pd.Index([], name="ticker"),
            columns=["cik", "name", "sic", "fiscal_year_end", "exchange"],
        )

    new_df = pd.DataFrame(records).set_index("ticker")

    df = pd.concat([existing_df, new_df]) if not existing_df.empty else new_df

    _write_parquet_atomic(df, out_path)
    logger.info(f"CIK map saved: {out_path} ({len(df)} rows, {len(new_df)} new)")
    return df
