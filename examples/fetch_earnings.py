"""Scarica i company facts per 5 ticker e mostra gli earnings TTM (trailing 12 mesi).

Usa solo dati 10-Q (trimestrali) e somma rolling degli ultimi 4 quarter.
I concept alias sono presi direttamente da edgartools/concept_mappings.json.
"""

import json
import time
from pathlib import Path

import edgar
import pandas as pd
import requests

_ROOT = Path(__file__).parent.parent

IDENTITY = "Ariel Nacamulli ariel.nacamulli@gmail.com"
TICKERS = ["TSLA", "NVDA", "MSFT", "GOOGL", "JNJ"]
DATA_DIR = _ROOT / "data/companyfacts_sample"
DATA_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": IDENTITY}

# Carica gli alias da edgartools concept_mappings.json
_MAPPING_PATH = _ROOT / ".venv/lib/python3.11/site-packages/edgar/xbrl/standardization/concept_mappings.json"
_RAW_MAPPINGS = json.loads(_MAPPING_PATH.read_text())

def _aliases(label: str) -> list[str]:
    """Restituisce i concept XBRL (short name) per un label standardizzato."""
    return [c.replace("us-gaap_", "") for c in _RAW_MAPPINGS.get(label, [])]

# Metriche da calcolare: label → lista di alias in ordine di priorità
# Per Revenue: prova prima i concept generici, poi il contract revenue come fallback
METRICS = {
    "Revenue (TTM)": _aliases("Revenue") + _aliases("Contract Revenue"),
    "Gross Profit (TTM)": _aliases("Gross Profit"),
    "Operating Income (TTM)": _aliases("Operating Income"),
    "Net Income (TTM)": _aliases("Net Income"),
    "EPS Basic (TTM)": _aliases("Earnings Per Share (Basic)"),
    "EPS Diluted (TTM)": _aliases("Earnings Per Share (Diluted)"),
}

EPS_METRICS = {"EPS Basic (TTM)", "EPS Diluted (TTM)"}

# --- Step 1: resolve tickers → CIK ---
print("=== Step 1: Mapping tickers → CIK ===")
edgar.set_identity(IDENTITY)
cik_map = {}
for ticker in TICKERS:
    company = edgar.Company(ticker)
    cik_padded = f"{company.cik:010d}"
    cik_map[ticker] = cik_padded
    print(f"  {ticker} → CIK {cik_padded}  ({company.name})")
    time.sleep(0.2)

# --- Step 2: download company facts JSON ---
print("\n=== Step 2: Downloading company facts JSON ===")
for ticker, cik in cik_map.items():
    json_path = DATA_DIR / f"CIK{cik}.json"
    if json_path.exists():
        print(f"  {ticker}: già in cache")
        continue
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    print(f"  {ticker}: GET {url} ...", end="", flush=True)
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    json_path.write_bytes(resp.content)
    print(f" {len(resp.content)/1024:.0f} KB")
    time.sleep(0.5)


# --- Step 3: parsing + TTM ---

def get_quarterly_series(usgaap: dict, concept_short: str, is_eps: bool) -> pd.DataFrame:
    """Estrae la serie trimestrale (10-Q) per un singolo concept, con dedup PIT."""
    concept_data = usgaap.get(concept_short)
    if not concept_data:
        return pd.DataFrame()

    units_dict = concept_data.get("units", {})
    unit_keys = ["shares", "USD"] if is_eps else ["USD", "shares"]
    entries = None
    for uk in unit_keys:
        if uk in units_dict:
            entries = units_dict[uk]
            break
    if not entries:
        return pd.DataFrame()

    rows = []
    for e in entries:
        if e.get("form") != "10-Q":
            continue
        if not (e.get("start") and e.get("end") and e.get("filed") and e.get("val") is not None):
            continue
        # Filtra solo quarter discreti (70-100 giorni): esclude YTD cumulativi (H1=~180gg, 9M=~270gg)
        duration = (pd.Timestamp(e["end"]) - pd.Timestamp(e["start"])).days
        if not (70 <= duration <= 100):
            continue
        rows.append({"end": e["end"], "filed": e["filed"], "val": float(e["val"])})
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    # PIT dedup: per ogni end, tieni il filing più recente
    df = (
        df.sort_values("filed")
        .drop_duplicates(subset=["end"], keep="last")
        .sort_values("end")
        .reset_index(drop=True)
    )
    return df


def resolve_metric(usgaap: dict, aliases: list[str], is_eps: bool) -> pd.DataFrame:
    """Prova gli alias in ordine e usa la serie con più dati."""
    best = pd.DataFrame()
    for alias in aliases:
        df = get_quarterly_series(usgaap, alias, is_eps)
        if len(df) > len(best):
            best = df
            best.attrs["concept"] = alias
    return best


def compute_ttm(series: pd.DataFrame) -> pd.DataFrame:
    """Calcola il TTM come rolling sum degli ultimi 4 quarter per ogni data.

    Restituisce un DataFrame con colonne: end, filed, ttm_val, quarters_used.
    Solo le righe con almeno 4 quarter disponibili sono incluse.
    """
    if series.empty or len(series) < 4:
        return pd.DataFrame()

    rows = []
    for i in range(3, len(series)):
        window = series.iloc[i - 3 : i + 1]
        # Controlla che i 4 quarter coprano ~12 mesi (non più di 400 giorni di gap)
        date_span = (window["end"].iloc[-1] - window["end"].iloc[0]).days
        if date_span > 400:
            continue
        rows.append({
            "end": window["end"].iloc[-1],
            "filed": window["filed"].max(),  # PIT: disponibile solo quando tutti e 4 sono stati depositati
            "ttm_val": window["val"].sum(),
            "quarters_used": 4,
        })
    return pd.DataFrame(rows)


print("\n=== Step 3: Parsing + calcolo TTM ===")

all_results = {}
for ticker, cik in cik_map.items():
    data = json.loads((DATA_DIR / f"CIK{cik}.json").read_text())
    usgaap = data.get("facts", {}).get("us-gaap", {})
    ticker_data = {}
    for metric_label, aliases in METRICS.items():
        is_eps = metric_label in EPS_METRICS
        series = resolve_metric(usgaap, aliases, is_eps)
        if not series.empty:
            ttm = compute_ttm(series)
            if not ttm.empty:
                ticker_data[metric_label] = (series.attrs.get("concept", "?"), ttm)
    all_results[ticker] = ticker_data
    resolved = {k: v[0] for k, v in ticker_data.items()}
    print(f"  {ticker}: {resolved}")


# --- Step 4: output ---
print("\n" + "=" * 70)
print("  EARNINGS TTM — ultimi 6 periodi (solo 10-Q, rolling 4 quarter)")
print("=" * 70)

for ticker in TICKERS:
    print(f"\n{'─'*70}")
    print(f"  {ticker}")
    print(f"{'─'*70}")
    td = all_results[ticker]
    if not td:
        print("  (nessun dato)")
        continue

    for metric_label, (concept_used, ttm_df) in td.items():
        recent = ttm_df.sort_values("end", ascending=False).head(6)
        is_eps = metric_label in EPS_METRICS
        print(f"\n  [{metric_label}]  concept: {concept_used}")
        print(f"  {'end':<12} {'filed':<12} {'TTM':>20}")
        print(f"  {'─'*46}")
        for _, row in recent.iterrows():
            val = row["ttm_val"]
            if is_eps:
                val_str = f"{val:>20.2f} $/sh"
            else:
                val_str = f"{val/1e9:>18.3f} B$"
            print(f"  {str(row['end'].date()):<12} {str(row['filed'].date()):<12} {val_str}")
