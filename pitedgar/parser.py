"""Step 3: parse local JSON facts into a point-in-time parquet master."""

import json
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm

from pitedgar.config import PitEdgarConfig

# Units to attempt per concept, in priority order.
# EPS and share concepts use "shares"; everything else USD.
_SHARE_CONCEPTS = {
    "EarningsPerShareBasic",
    "EarningsPerShareDiluted",
    "CommonStockSharesOutstanding",
}


def _preferred_units(concept_short: str) -> list[str]:
    if concept_short in _SHARE_CONCEPTS:
        return ["shares", "USD"]
    return ["USD", "shares"]


def parse_company(
    cik_padded: str,
    concepts: list[str],
    facts_dir: Path,
    forms: list[str],
) -> pd.DataFrame:
    """Parse a single company's JSON into a tidy PIT DataFrame.

    Args:
        cik_padded: zero-padded 10-digit CIK string.
        concepts:   list of "us-gaap:ConceptName" strings.
        facts_dir:  directory containing CIK*.json files.
        forms:      filing forms to keep, e.g. ["10-K", "10-Q"].

    Returns:
        DataFrame with columns: cik, concept, end, filed, val, form, accn
    """
    json_path = facts_dir / f"CIK{cik_padded}.json"
    if not json_path.exists():
        logger.debug(f"No JSON for CIK {cik_padded}, skipping.")
        return pd.DataFrame()

    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    usgaap = data.get("facts", {}).get("us-gaap", {})
    if not usgaap:
        return pd.DataFrame()

    rows: list[dict] = []

    for concept_full in concepts:
        # concept_full is like "us-gaap:Revenues"
        parts = concept_full.split(":", 1)
        concept_short = parts[1] if len(parts) == 2 else parts[0]

        concept_data = usgaap.get(concept_short)
        if concept_data is None:
            continue

        units_dict: dict = concept_data.get("units", {})
        unit_entries: list[dict] | None = None

        for unit_key in _preferred_units(concept_short):
            if unit_key in units_dict:
                unit_entries = units_dict[unit_key]
                break

        if not unit_entries:
            continue

        for entry in unit_entries:
            form = entry.get("form", "")
            if form not in forms:
                continue
            end = entry.get("end")
            filed = entry.get("filed")
            val = entry.get("val")
            accn = entry.get("accn", "")
            if end is None or filed is None or val is None:
                continue
            rows.append(
                {
                    "cik": cik_padded,
                    "concept": concept_full,
                    "end": end,
                    "filed": filed,
                    "val": float(val),
                    "form": form,
                    "accn": accn,
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    df = df.dropna(subset=["end", "filed"])

    # PIT deduplication: for each (concept, end), keep the most recently filed record.
    df = (
        df.sort_values("filed")
        .drop_duplicates(subset=["concept", "end"], keep="last")
        .sort_values("filed")
        .reset_index(drop=True)
    )

    return df


def parse_all(config: PitEdgarConfig, cik_map: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """Parse all companies in cik_map into a single PIT master parquet.

    Args:
        config:  pipeline configuration.
        cik_map: DataFrame indexed by ticker with a 'cik' column.
        force:   re-parse even if pit_financials.parquet already exists.

    Returns:
        Master DataFrame with an additional 'ticker' column.
    """
    config.ensure_dirs()
    out_path = config.data_dir / "pit_financials.parquet"

    if not force and out_path.exists():
        logger.info(f"Parquet already exists at {out_path}, skipping parse (use force=True to override).")
        return pd.read_parquet(out_path)

    all_frames: list[pd.DataFrame] = []

    for ticker, row in tqdm(cik_map.iterrows(), total=len(cik_map), desc="Parsing"):
        cik_padded = str(row["cik"])
        df = parse_company(
            cik_padded=cik_padded,
            concepts=config.concepts,
            facts_dir=config.facts_dir,
            forms=config.forms,
        )
        if df.empty:
            continue
        df.insert(0, "ticker", ticker)
        all_frames.append(df)

    if not all_frames:
        logger.warning("No data parsed — check facts_dir and CIK map.")
        return pd.DataFrame()

    master = pd.concat(all_frames, ignore_index=True)

    master.to_parquet(out_path, index=False)
    logger.info(f"Master parquet saved: {out_path} ({len(master):,} rows)")
    return master
