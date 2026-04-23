"""Step 3: parse local JSON facts into a point-in-time parquet master."""

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
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


def _values_equal_tol(a: pd.Series, b: pd.Series, concept: pd.Series) -> pd.Series:
    """True where a ≈ b under concept-appropriate tolerance; NaN in b → False.

    Share/EPS concepts use a tighter absolute tolerance (1e-6) since values
    are small. All other (USD) concepts use atol=0.005 (half a cent) to absorb
    1-ULP float representation drift between JSON serialisations.
    """
    is_share = concept.str.split(":").str[-1].isin(_SHARE_CONCEPTS)
    atol = np.where(is_share, 1e-6, 0.005)
    # rtol=0: purely absolute tolerance. Using a non-zero rtol on USD values
    # would scale the tolerance with magnitude (e.g. rtol=1e-9 on $1B yields
    # $1 tolerance, swallowing genuine $1 restatements).
    rtol = 0.0
    a_np = a.to_numpy(dtype=float, na_value=np.nan)
    b_np = b.to_numpy(dtype=float, na_value=np.nan)
    # np.isclose returns False when either operand is NaN, which is the
    # desired behaviour: NaN in b means "no previous value" → not equal.
    result = np.isclose(a_np, b_np, atol=atol, rtol=rtol, equal_nan=False)
    return pd.Series(result, index=a.index)


def _preferred_units(concept_short: str) -> list[str]:
    if concept_short in _SHARE_CONCEPTS:
        return ["shares", "USD"]
    return ["USD", "shares"]


def parse_company(
    cik_padded: str,
    concepts: list[str] | None,
    facts_dir: Path,
    forms: list[str],
) -> pd.DataFrame:
    """Parse a single company's JSON into a tidy PIT DataFrame.

    Args:
        cik_padded: zero-padded 10-digit CIK string.
        concepts:   list of "us-gaap:ConceptName" strings to extract. Pass ``None``
                    or an empty list to extract every us-gaap concept present in
                    the JSON (recommended — the full parquet supports flexible
                    quant signal iteration without re-parsing the 1.5 GB cache).
        facts_dir:  directory containing CIK*.json files.
        forms:      filing forms to keep, e.g. ["10-K", "10-Q"].

    Returns:
        DataFrame with columns: cik, concept, end, filed, val, form, accn.
        The ``concept`` column always contains the canonical full name
        ``"us-gaap:<Name>"`` — alias tags are mapped to their canonical target.
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

    # When ``concepts`` is None or empty, derive the canonical concept set from
    # whatever us-gaap tags this company actually filed. Aliases are mapped to
    # their canonical name so e.g. a filer that only reports the post-ASC 606
    # revenue tag still ends up under ``us-gaap:Revenues`` in the parquet.
    if not concepts:
        canonical_set: set[str] = set()
        for short_name in usgaap:
            full_name = f"us-gaap:{short_name}"
            canonical = CONCEPT_ALIASES.get(full_name, full_name)
            canonical_set.add(canonical)
        concepts = sorted(canonical_set)

    rows: list[dict] = []

    # Build reverse alias map: canonical -> [alias, ...]
    alias_lookup: dict[str, list[str]] = {c: [] for c in concepts}
    for alias, canonical in CONCEPT_ALIASES.items():
        if canonical in alias_lookup:
            alias_lookup[canonical].append(alias)

    for concept_full in concepts:
        # Union data from the canonical concept AND every alias — a filer may
        # report different periods under different tags (e.g. pre-ASC 606 years
        # under ``SalesRevenueNet`` and post-ASC 606 years under
        # ``RevenueFromContractWithCustomerExcludingAssessedTax``), so stopping at
        # the first tag with data would silently drop all the other periods.
        #
        # Canonical precedence: when the same ``(end, filed, form, accn)`` key is
        # present under multiple tags, the canonical value wins. The candidate
        # list places canonical first and we dedup ``keep='first'`` below.
        candidates = [concept_full, *alias_lookup.get(concept_full, [])]
        canonical_short = concept_full.split(":", 1)[-1]

        candidate_rows: list[dict] = []
        for candidate_full in candidates:
            concept_short = candidate_full.split(":", 1)[-1]
            concept_data = usgaap.get(concept_short)
            if concept_data is None:
                continue
            units_dict: dict = concept_data.get("units", {})
            # Pick the preferred unit per candidate (USD vs shares). This runs
            # independently for each candidate so e.g. a share-concept alias
            # reported in "shares" is still picked up alongside the canonical.
            unit_entries: list[dict] | None = None
            for unit_key in _preferred_units(canonical_short):
                if unit_key in units_dict:
                    unit_entries = units_dict[unit_key]
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
                candidate_rows.append(
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

        if not candidate_rows:
            # Concept reported neither USD nor shares (e.g. pure/EUR/etc.) — skip
            # silently to keep the parquet schema homogeneous.
            continue

        # Dedup across candidates by (end, filed, form). Because the canonical
        # candidate is processed first, the first-seen row wins — preserving
        # canonical precedence when a filer reports the same reporting instance
        # (same period, same filing date, same form) under both canonical and
        # alias tags. Different ``accn`` values at the same (end, filed, form)
        # cannot legitimately coexist for one company, so excluding accn from
        # the dedup key is safe and also matches the legacy "canonical wins"
        # behaviour exercised by ``test_parse_company_canonical_wins_over_alias``.
        # This dedup only collapses overlaps between canonical and alias tags;
        # non-overlapping periods (e.g. pre-ASC 606 years under one tag and
        # post-ASC 606 years under another) are preserved via the union.
        if len(candidates) > 1:
            seen: set[tuple] = set()
            deduped: list[dict] = []
            for row in candidate_rows:
                key = (row["end"], row["filed"], row["form"])
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(row)
            candidate_rows = deduped

        rows.extend(candidate_rows)

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
    keep = prev_val.isna() | ~_values_equal_tol(df["val"], prev_val, df["concept"])
    df = df[keep].sort_values("filed").reset_index(drop=True)

    return df


def _parse_one_for_pool(
    args: tuple[str, str, list[str] | None, Path, list[str]],
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
