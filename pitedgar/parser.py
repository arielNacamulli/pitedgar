"""Step 3: parse local JSON facts into a point-in-time parquet master."""

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm

from pitedgar.config import CONCEPT_ALIASES, PitEdgarConfig
from pitedgar.periods import is_quarterly

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

    with open(json_path, encoding="utf-8") as fh:
        data = json.load(fh)

    usgaap = data.get("facts", {}).get("us-gaap", {})
    if not usgaap:
        return pd.DataFrame()

    rows: list[dict] = []

    # Build reverse alias map: canonical -> [alias, ...]
    alias_lookup: dict[str, list[str]] = {c: [] for c in concepts}
    for alias, canonical in CONCEPT_ALIASES.items():
        if canonical in alias_lookup:
            alias_lookup[canonical].append(alias)

    for concept_full in concepts:
        # Try the canonical concept first, then any aliases (e.g. post-ASC 606 revenue tag).
        # Store all rows under the canonical name regardless of which tag matched.
        candidates = [concept_full, *alias_lookup.get(concept_full, [])]
        unit_entries: list[dict] | None = None
        canonical_short = concept_full.split(":", 1)[-1]

        for candidate_full in candidates:
            concept_short = candidate_full.split(":", 1)[-1]
            concept_data = usgaap.get(concept_short)
            if concept_data is None:
                continue
            units_dict: dict = concept_data.get("units", {})
            for unit_key in _preferred_units(canonical_short):
                if unit_key in units_dict:
                    unit_entries = units_dict[unit_key]
                    break
            if unit_entries:
                break

        if not unit_entries:
            continue

        for entry in unit_entries:
            form = entry.get("form", "")
            if form not in forms:
                continue
            start = entry.get("start")
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
                    "start": start,
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
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    df = df.dropna(subset=["end", "filed"])

    # Scale detection: some filers report USD values in thousands instead of dollars.
    # If the maximum absolute USD value across all USD concepts is < $1M, the company
    # is almost certainly mis-scaled — rescale by 1000 and flag the affected rows in
    # `scale_corrected` so downstream consumers can audit which values were adjusted.
    df["scale_corrected"] = False
    usd_mask = ~df["concept"].str.split(":").str[-1].isin(_SHARE_CONCEPTS)
    if usd_mask.any() and df.loc[usd_mask, "val"].abs().max() < 1_000_000:
        df.loc[usd_mask, "val"] *= 1000
        df.loc[usd_mask, "scale_corrected"] = True
        logger.debug(f"CIK {cik_padded}: applied 1000x scale correction to {int(usd_mask.sum())} USD rows")

    # Duration in days; -1 if start missing (instantaneous entries, e.g. balance sheet).
    df["duration_days"] = (df["end"] - df["start"]).dt.days.where(df["start"].notna(), -1)

    # Within-filing dedup: a single filing can contain both a discrete quarter and a YTD
    # cumulative entry for the same (concept, end). Prefer the discrete quarter.
    # We dedup on (concept, end, filed) so restatements filed on different dates are preserved.
    df["_is_quarterly"] = is_quarterly(df["duration_days"]).astype(int)
    df = (
        df.sort_values(["filed", "_is_quarterly"])
        .drop_duplicates(subset=["concept", "end", "filed"], keep="last")
        .drop(columns=["_is_quarterly"])
    )

    # Cross-filing dedup: for each (concept, end), keep only the EARLIEST filing of
    # each distinct value. Later re-filings of the same unchanged value — e.g. a prior
    # period appearing as comparative data in a subsequent 10-K — would inflate the
    # PIT "known date" by years and must be dropped. Genuine restatements (value changes
    # between filings) are preserved as new rows so the query layer can track them.
    df = df.sort_values(["concept", "end", "filed"])
    prev_val = df.groupby(["concept", "end"], sort=False)["val"].shift(1)
    keep = prev_val.isna() | (df["val"] != prev_val)
    df = df[keep].sort_values("filed").reset_index(drop=True)

    return df


def _parse_one_for_pool(
    args: tuple[str, str, list[str], Path, list[str]],
) -> tuple[str, pd.DataFrame]:
    """Top-level helper for ProcessPoolExecutor (must be picklable, hence not a closure).

    Args:
        args: tuple of (ticker, cik_padded, concepts, facts_dir, forms).

    Returns:
        (ticker, parsed DataFrame). DataFrame may be empty.
    """
    ticker, cik_padded, concepts, facts_dir, forms = args
    df = parse_company(
        cik_padded=cik_padded,
        concepts=concepts,
        facts_dir=facts_dir,
        forms=forms,
    )
    return ticker, df


def parse_all(
    config: PitEdgarConfig,
    cik_map: pd.DataFrame,
    force: bool = False,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """Parse all companies in cik_map into a single PIT master parquet.

    Args:
        config:    pipeline configuration.
        cik_map:   DataFrame indexed by ticker with a 'cik' column.
        force:     re-parse even if pit_financials.parquet already exists.
        n_workers: number of worker processes for parallel parsing.
                   ``None`` (default) uses ``os.cpu_count()`` (or 1 if unavailable).
                   ``1`` runs serially, skipping the process pool entirely — useful
                   for debugging (cleaner stack traces) and single-core environments.

    Returns:
        Master DataFrame with an additional 'ticker' column.
    """
    config.ensure_dirs()
    out_path = config.data_dir / "pit_financials.parquet"

    if not force and out_path.exists():
        logger.info(f"Parquet already exists at {out_path}, skipping parse (use force=True to override).")
        return pd.read_parquet(out_path)

    if n_workers is None:
        n_workers = os.cpu_count() or 1
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")

    all_frames: list[pd.DataFrame] = []

    if n_workers == 1:
        # Serial path: no pool overhead, clean stack traces for debugging.
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
            df.insert(0, "ticker", str(ticker))
            all_frames.append(df)
    else:
        # Parallel path: fan out across processes — JSON parsing is CPU-bound.
        tasks = [
            (str(ticker), str(row["cik"]), config.concepts, config.facts_dir, config.forms)
            for ticker, row in cik_map.iterrows()
        ]
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_parse_one_for_pool, task) for task in tasks]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc=f"Parsing ({n_workers} workers)"
            ):
                ticker, df = future.result()
                if df.empty:
                    continue
                df.insert(0, "ticker", ticker)
                all_frames.append(df)

    if not all_frames:
        logger.warning("No data parsed — check facts_dir and CIK map.")
        return pd.DataFrame()

    master = pd.concat(all_frames, ignore_index=True)

    # Atomic write: write to .tmp then rename, so a crash mid-write never leaves
    # a truncated parquet behind that later reads would choke on.
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        master.to_parquet(tmp_path, index=False)
        tmp_path.replace(out_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    logger.info(f"Master parquet saved: {out_path} ({len(master):,} rows)")
    return master
