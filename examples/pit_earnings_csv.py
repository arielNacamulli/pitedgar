"""Genera un CSV con Net Income (PIT) per ogni business day.

Righe = business days, colonne = ticker, valori = ultimo NetIncomeLoss
disponibile al momento (no look-ahead bias).

Output: examples/pit_earnings.csv
"""

from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).parent.parent
PARQUET = _ROOT / "data/pit_financials.parquet"

TICKERS = ["TSLA", "NVDA", "MSFT", "GOOGL", "JNJ"]
CONCEPT = "us-gaap:NetIncomeLoss"

# Intervallo temporale (modifica se necessario)
START_DATE = "2010-01-01"
END_DATE = pd.Timestamp.today().normalize()

# ------------------------------------------------------------------
# Carica e filtra i dati
# ------------------------------------------------------------------
print(f"Carico {PARQUET} ...")
df = pd.read_parquet(PARQUET)
df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
df["end"] = pd.to_datetime(df["end"], errors="coerce")

mask = df["ticker"].isin(TICKERS) & (df["concept"] == CONCEPT)
df = df.loc[mask, ["ticker", "filed", "end", "val"]].copy()
df = df.dropna(subset=["filed", "val"])

print(f"Righe filtrate: {len(df)}  (tickers: {df['ticker'].nunique()})")

# ------------------------------------------------------------------
# Per ogni ticker: serie ordinata per filed, dedup per (end, filed)
# Il parser garantisce già dedup, ma riapplicare non fa male.
# Teniamo solo il val più recente per ogni data di filing.
# ------------------------------------------------------------------
business_days = pd.bdate_range(start=START_DATE, end=END_DATE)

series_list = []
for ticker, grp in df.groupby("ticker"):
    # Ordina per filed; in caso di più record con stesso filed prendi il più recente end
    grp = (
        grp.sort_values(["filed", "end"])
        .drop_duplicates(subset=["filed"], keep="last")
        .set_index("filed")["val"]
        .sort_index()
    )

    # Reindex sui business day: forward-fill (PIT: usa l'ultimo valore noto)
    # Allinea prima al range, poi ffill
    combined_index = grp.index.union(business_days)
    s = grp.reindex(combined_index).ffill()
    s = s.reindex(business_days)
    s.name = ticker
    series_list.append(s)

# ------------------------------------------------------------------
# Assembla il pivot DataFrame
# ------------------------------------------------------------------
result = pd.concat(series_list, axis=1)[TICKERS]
result.index.name = "date"

# Rimuovi le date iniziali in cui tutti i ticker sono NaN
result = result.loc[result.notna().any(axis=1)]

out_path = Path(__file__).parent / "pit_earnings.csv"
result.to_csv(out_path)
print(f"\nSalvato: {out_path}")
print(f"Shape: {result.shape}  ({result.shape[0]} business days x {result.shape[1]} tickers)")
print(f"\nPrime 5 righe:")
print(result.head())
print(f"\nUltime 5 righe:")
print(result.tail())
