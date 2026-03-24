# pitedgar

Point-in-time SEC EDGAR financial data pipeline.

Downloads SEC EDGAR `companyfacts.zip`, parses XBRL JSON facts into a local
parquet file, and exposes a query API with **zero look-ahead bias** — every
value is stamped with the `filed` date (when the data was actually available
to the market), not the period-end date.

---

## Installation

```bash
pip install pitedgar
# or with Poetry
poetry install
```

---

## Quick start

```python
from pathlib import Path
from pitedgar import PitEdgarConfig, build_cik_map, download_bulk, parse_all, PitQuery

config = PitEdgarConfig(
    edgar_identity="Mario Rossi mario@example.com",  # required by SEC
    data_dir=Path("./data"),
)

# Step 1 — one-shot ticker → CIK mapping
tickers = ["AAPL", "MSFT", "JPM", "GOOGL"]
cik_map = build_cik_map(tickers, config)

# Step 2 — download ~1.5 GB bulk ZIP (do this periodically, not every run)
download_bulk(config)

# Step 3 — parse JSON → parquet (sub-minute for 500 companies)
master = parse_all(config, cik_map)

# Step 4 — query
q = PitQuery(config.data_dir / "pit_financials.parquet")

# What revenue figure was available to the market on 2022-06-30?
result = q.as_of(["AAPL", "MSFT"], "us-gaap:Revenues", "2022-06-30")

# Full history
hist = q.history("AAPL", "us-gaap:NetIncomeLoss", freq="A")

# Portfolio cross-section signal
xs = q.cross_section("us-gaap:NetIncomeLoss", "2023-12-31")
```

---

## CLI

```bash
# Resolve tickers (tickers.txt has one ticker per line)
pitedgar map --tickers tickers.txt --identity "Name name@email.com"

# Download bulk ZIP
pitedgar fetch --identity "Name name@email.com"

# Parse to parquet
pitedgar build --identity "Name name@email.com"

# Query a single value
pitedgar query --ticker AAPL --concept us-gaap:Revenues --as-of 2023-06-30
```

---

## Key design decisions

| Decision | Rationale |
|---|---|
| `filed` as PIT timestamp | The date the filing was submitted to SEC — this is when information became public |
| Deduplication keeps latest `filed` per `(concept, end)` | Companies sometimes refile restated figures; keep the superseding value |
| Raw USD values, no scale conversion | SEC reports values as-filed; downstream code applies any needed normalization |
| Local parquet, no runtime HTTP | Queries run at DataFrame speed with no network dependency |

---

## Supported XBRL concepts (defaults)

See `pitedgar.config.DEFAULT_CONCEPTS` for the full list, which includes
revenues, net income, assets, liabilities, equity, EPS, cash, debt, operating
cash flow, capex, and R&D expense.
