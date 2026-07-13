"""Extract ``dei:EntityCommonStockSharesOutstanding`` from companyfacts.

Cover-page share counts are the natural pairing for *raw* (as-printed)
prices when computing implied market caps. The raw facts, however, carry
real defects — found by pitdata's mcap-band cross-check (2026-07-13):

- companies occasionally file garbage values (CSX filed ``1``, CRM
  ``134``, AAP ``7.4e10``);
- zero placeholders exist (BRK filed ``0`` for 2010-Q1);
- multi-class issuers may cover only one class in the undimensioned
  fact (Berkshire's series counts class A only, and dies in 2011).

Rows are therefore **kept, not dropped**, with a ``quality`` column so
downstream consumers choose their own strictness:

- ``ok`` — passed all guards;
- ``nonpositive`` — val <= 0;
- ``magnitude_outlier`` — >10**1.5x (~30x) away from the rolling median
  of the issuer's neighbouring reports. Wide on purpose: as-reported
  counts legitimately jump at splits, so only absurd factors trip it.

Multi-class under-counting is NOT detectable from the dei series alone
(a class-A-only series is internally consistent); consumers should pair
this data with a staleness cap and cross-source checks.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

CONCEPT = "EntityCommonStockSharesOutstanding"
OUTLIER_LOG10 = 1.5  # ~30x from the rolling neighbourhood median
NEIGHBOURHOOD = 5  # facts on each side


def _quality(vals: list[float]) -> list[str]:
    """Per-fact quality flags for one issuer's chronological values."""
    flags: list[str] = []
    logs = [math.log10(v) if v > 0 else None for v in vals]
    for i, v in enumerate(vals):
        if v <= 0:
            flags.append("nonpositive")
            continue
        window = [
            x
            for j, x in enumerate(logs)
            if x is not None and j != i and abs(j - i) <= NEIGHBOURHOOD
        ]
        if window:
            window.sort()
            median = window[len(window) // 2]
            if abs(logs[i] - median) > OUTLIER_LOG10:  # type: ignore[operator]
                flags.append("magnitude_outlier")
                continue
        flags.append("ok")
    return flags


def extract_dei_shares(
    companyfacts_dir: str | Path,
    ticker_by_cik: dict[str, str],
) -> pd.DataFrame:
    """One row per dei share-count fact for the given issuers.

    Columns: ``cticker, cik, filed, end, val, form, quality`` —
    superset of the legacy ``dei_shares.parquet`` schema (which lacked
    ``cik`` and ``quality``).
    """
    companyfacts_dir = Path(companyfacts_dir)
    rows: list[dict] = []
    for cik, ticker in sorted(ticker_by_cik.items()):
        path = companyfacts_dir / f"CIK{cik.zfill(10)}.json"
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        facts = data.get("facts", {}).get("dei", {}).get(CONCEPT, {}).get("units", {})
        share_facts = facts.get("shares", [])
        share_facts = [f for f in share_facts if f.get("val") is not None and f.get("filed")]
        share_facts.sort(key=lambda f: (f["filed"], f.get("end") or ""))
        flags = _quality([float(f["val"]) for f in share_facts])
        for f, q in zip(share_facts, flags, strict=True):
            rows.append(
                {
                    "cticker": ticker,
                    "cik": cik.zfill(10),
                    "filed": f["filed"],
                    "end": f.get("end"),
                    "val": float(f["val"]),
                    "form": f.get("form"),
                    "quality": q,
                }
            )
    df = pd.DataFrame(rows, columns=["cticker", "cik", "filed", "end", "val", "form", "quality"])
    if len(df):
        df["filed"] = pd.to_datetime(df["filed"])
        df["end"] = pd.to_datetime(df["end"])
        df = df.sort_values(["cticker", "filed"]).reset_index(drop=True)
    return df


def ticker_map_from_financials(pit_financials_path: str | Path) -> dict[str, str]:
    """cik -> ticker from an existing pit_financials.parquet (first ticker wins)."""
    pairs = pd.read_parquet(pit_financials_path, columns=["ticker", "cik"]).drop_duplicates()
    out: dict[str, str] = {}
    for r in pairs.itertuples():
        cik = str(r.cik).zfill(10)
        out.setdefault(cik, str(r.ticker))
    return out


__all__ = ["extract_dei_shares", "ticker_map_from_financials"]
