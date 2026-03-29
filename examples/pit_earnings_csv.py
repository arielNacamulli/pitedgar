"""Genera un CSV con Net Income TTM (PIT) per ogni business day.

Righe = business days, colonne = ticker, valori = TTM NetIncomeLoss
(somma degli ultimi 4 quarter 10-Q disponibili al momento, no look-ahead bias).

Output: examples/pit_earnings.csv
"""

from pathlib import Path

import pandas as pd

from pitedgar.query import PitQuery

_ROOT = Path(__file__).parent.parent
PARQUET = _ROOT / "data/pit_financials.parquet"

TICKERS = ["TSLA", "NVDA", "MSFT", "GOOGL", "JNJ"]
CONCEPT = "us-gaap:NetIncomeLoss"

START_DATE = "2010-01-01"
END_DATE = pd.Timestamp.today().normalize()

business_days = pd.bdate_range(start=START_DATE, end=END_DATE)

print(f"Carico {PARQUET} ...")
pq = PitQuery(PARQUET)

print(f"Calcolo TTM per {len(TICKERS)} tickers x {len(business_days)} business days ...")
raw = pq.ttm_cross_section(
    concept=CONCEPT,
    as_of_dates=business_days,
    tickers=TICKERS,
    max_staleness_days=100,
)

result = raw.pivot(index="as_of_date", columns="ticker", values="ttm_val")[TICKERS]
result.index.name = "date"
result = result.loc[result.notna().any(axis=1)]

out_path = Path(__file__).parent / "pit_earnings.csv"
result.to_csv(out_path)
print(f"\nSalvato: {out_path}")
print(f"Shape: {result.shape}  ({result.shape[0]} business days x {result.shape[1]} tickers)")
print(f"\nPrime 5 righe:")
print(result.head())
print(f"\nUltime 5 righe:")
print(result.tail())
