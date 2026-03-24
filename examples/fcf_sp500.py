"""
Feasibility benchmark: S&P 500 FCF last 5 years, quarterly.

Stages timed separately so you can see where time is spent:
  1. Fetch S&P 500 constituent list from GitHub   (~1s, CIKs included)
  2. Build CIK map from CSV                        (instant — no API calls)
  3. Parse local JSON → parquet                    (~20s for 503 companies)
  4. Query FCF cross-section × 20 quarters         (<1s)

Prerequisites:
  - companyfacts.zip already downloaded and extracted to data/companyfacts/
    (run: pitedgar fetch --identity "Name email" OR download_bulk(config))

Run:
    python benchmark_fcf.py --identity "Name name@email.com"
    python benchmark_fcf.py --identity "Name name@email.com" --skip-parse
"""

import argparse
import time
from pathlib import Path

import pandas as pd

FCF_CONCEPTS = [
    "us-gaap:NetCashProvidedByUsedInOperatingActivities",
    "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",  # capex proxy
]

DATA_DIR = Path("./data")

SP500_CSV_URL = (
    "https://raw.githubusercontent.com/hanshof/sp500_constituents"
    "/main/sp500_constituents.csv"
)


def fetch_sp500_constituents() -> pd.DataFrame:
    t0 = time.perf_counter()
    df = pd.read_csv(SP500_CSV_URL)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df["symbol"] = df["symbol"].str.strip().str.upper()
    df["cik"] = df["cik"].astype(str).str.strip().str.zfill(10)
    elapsed = time.perf_counter() - t0
    print(f"[1] Fetched {len(df)} constituents in {elapsed:.1f}s  (CIKs included — no edgar API needed)")
    return df


def build_cik_map_from_csv(df: pd.DataFrame) -> pd.DataFrame:
    return df.set_index("symbol")[["cik", "security"]].rename(columns={"security": "name"})


def stage_parse(config, cik_map):
    from pitedgar.parser import parse_all
    t0 = time.perf_counter()
    master = parse_all(config, cik_map)
    elapsed = time.perf_counter() - t0
    rows = len(master)
    speed = rows / elapsed if elapsed > 0 else float("inf")
    print(f"[3] Parsed {rows:,} rows in {elapsed:.1f}s  ({speed:,.0f} rows/s)")
    return master


def stage_query(config, tickers):
    from pitedgar.query import PitQuery

    t0 = time.perf_counter()
    q = PitQuery(config.data_dir / "pit_financials.parquet")

    today = pd.Timestamp.today()
    quarters = pd.date_range(end=today, periods=20, freq="QE")

    results = []
    for date in quarters:
        cfo = (
            q.cross_section(
                "us-gaap:NetCashProvidedByUsedInOperatingActivities",
                as_of_date=str(date.date()),
                tickers=tickers,
                max_staleness_days=180,
            )
            .set_index("ticker")[["val"]]
            .rename(columns={"val": "cfo"})
        )
        capex = (
            q.cross_section(
                "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
                as_of_date=str(date.date()),
                tickers=tickers,
                max_staleness_days=180,
            )
            .set_index("ticker")[["val"]]
            .rename(columns={"val": "capex"})
        )
        combined = cfo.join(capex, how="outer")
        combined["fcf"] = combined["cfo"] - combined["capex"].abs()
        combined["date"] = date
        results.append(combined.reset_index())

    fcf_df = pd.concat(results, ignore_index=True)
    elapsed = time.perf_counter() - t0

    coverage = fcf_df.groupby("date")["fcf"].count()
    print(f"[4] Query: {len(fcf_df):,} cells × {len(quarters)} quarters in {elapsed:.2f}s")
    print(f"    Avg tickers with FCF data per quarter: {coverage.mean():.0f}")
    print(f"\nSample — most recent quarter ({quarters[-1].date()}):")
    sample = fcf_df[fcf_df.date == quarters[-1]].dropna(subset=["fcf"]).head(10)
    print(sample[["ticker", "cfo", "capex", "fcf"]].to_string(index=False))
    return fcf_df


def main():
    parser = argparse.ArgumentParser(description="S&P 500 FCF feasibility benchmark")
    parser.add_argument("--identity", required=True,
                        help='SEC identity string, e.g. "Name name@email.com"')
    parser.add_argument("--skip-parse", action="store_true",
                        help="Skip parsing step; load existing pit_financials.parquet")
    args = parser.parse_args()

    from pitedgar.config import PitEdgarConfig

    config = PitEdgarConfig(
        edgar_identity=args.identity,
        data_dir=DATA_DIR,
        concepts=FCF_CONCEPTS,
    )
    config.ensure_dirs()

    t_total = time.perf_counter()

    constituents = fetch_sp500_constituents()
    tickers = constituents["symbol"].tolist()

    cik_map = build_cik_map_from_csv(constituents)
    cik_map.to_parquet(config.data_dir / "ticker_cik_map.parquet")
    print(f"[2] CIK map saved: {len(cik_map)} entries (instant — from CSV)")

    if not args.skip_parse:
        stage_parse(config, cik_map)

    stage_query(config, tickers)

    total = time.perf_counter() - t_total
    print(f"\nTotal wall time (excl. zip download): {total:.1f}s")


if __name__ == "__main__":
    main()
