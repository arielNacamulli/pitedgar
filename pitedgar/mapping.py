"""Step 1: resolve ticker → CIK via edgartools."""

import time
import edgar
import pandas as pd
from loguru import logger
from pitedgar.config import PitEdgarConfig


def build_cik_map(tickers: list[str], config: PitEdgarConfig) -> pd.DataFrame:
    """Resolve a list of tickers to CIK numbers via edgartools.

    Saves the result to data_dir/ticker_cik_map.parquet.
    Returns a DataFrame indexed by ticker with columns:
        cik, name, sic, fiscal_year_end, exchange
    """
    edgar.set_identity(config.edgar_identity)
    config.ensure_dirs()

    records: list[dict] = []

    for ticker in tickers:
        try:
            company = edgar.Company(ticker)
            cik_padded = f"{company.cik:010d}"
            records.append(
                {
                    "ticker": ticker.upper(),
                    "cik": cik_padded,
                    "name": getattr(company, "name", None),
                    "sic": getattr(company, "sic", None),
                    "fiscal_year_end": getattr(company, "fiscal_year_end", None),
                    "exchange": getattr(company, "exchange", None),
                }
            )
            logger.debug(f"{ticker} → CIK {cik_padded}")
        except Exception as exc:
            logger.warning(f"Could not resolve ticker '{ticker}': {exc}")
        time.sleep(0.1)

    df = pd.DataFrame(records).set_index("ticker")
    out_path = config.data_dir / "ticker_cik_map.parquet"
    df.to_parquet(out_path)
    logger.info(f"CIK map saved: {out_path} ({len(df)} rows)")
    return df
