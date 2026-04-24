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
_TICKER_LOOKUP_SKIPPABLE: tuple[type[BaseException], ...] = (
    ValueError,
    KeyError,
    LookupError,
    AttributeError,
)


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

    for ticker in new_tickers:
        try:
            company = edgar.Company(ticker)
            cik_padded = f"{company.cik:010d}"
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
        except _TICKER_LOOKUP_SKIPPABLE as exc:
            logger.warning(f"Could not resolve ticker '{ticker}': {exc!r}")
        time.sleep(0.1)

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
