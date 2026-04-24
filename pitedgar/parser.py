"""Step 3: parse local JSON facts into a point-in-time parquet master."""

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from pitedgar.config import (
    CONCEPT_ALIAS_PRIORITY,
    CONCEPT_ALIASES,
    LOSSY_CONCEPT_ALIASES,
    PitEdgarConfig,
)
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


def is_scale_corrected(df: pd.DataFrame) -> pd.Series:
    """Boolean series indicating which rows had the 1000x scale correction applied.

    Use this to filter or exclude corrected rows when merging parquets from
    different parse runs, since the correction is applied in-place and not
    re-detectable from values alone.
    """
    if "scale_corrected" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["scale_corrected"].astype(bool)


def parse_company(
    cik_padded: str,
    concepts: list[str] | None,
    facts_dir: Path,
    forms: list[str],
    scale_correction: str = "off",
    scale_correction_threshold: float = 1_000_000.0,
    alias_map: dict[str, str] | None = None,
    lossy_alias_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Parse a single company's JSON into a tidy PIT DataFrame.

    Args:
        cik_padded:      zero-padded 10-digit CIK string.
        concepts:        list of "us-gaap:ConceptName" strings, or None for all.
        facts_dir:       directory containing CIK*.json files.
        forms:           filing forms to keep, e.g. ["10-K", "10-Q"].
        scale_correction: "off" | "auto" | "force" — controls 1000x USD correction.
        scale_correction_threshold: threshold USD value for the "auto" heuristic.
        alias_map:       non-lossy alias dict (defaults to ``CONCEPT_ALIASES``).
        lossy_alias_map: lossy alias dict (defaults to ``{}``). Rows produced via a
                         lossy alias carry the original tag in ``alias_source``.

    Returns:
        DataFrame with columns: ticker, cik, concept, start, end, duration_days,
        filed, val, form, accn, scale_corrected, alias_source.

    Note:
        The ``scale_corrected`` column is an audit marker. Do NOT re-run scale
        detection on an already-parsed parquet — use ``is_scale_corrected`` to
        identify affected rows before merging with another parse.
    """
    effective_alias_map: dict[str, str] = CONCEPT_ALIASES if alias_map is None else alias_map
    effective_lossy_map: dict[str, str] = lossy_alias_map if lossy_alias_map is not None else {}
    combined_alias_map: dict[str, str] = {**effective_alias_map, **effective_lossy_map}
    lossy_tags: set[str] = set(effective_lossy_map)
    warned_lossy: set[str] = set()

    json_path = facts_dir / f"CIK{cik_padded}.json"
    if not json_path.exists():
        logger.debug(f"No JSON for CIK {cik_padded}, skipping.")
        return pd.DataFrame()

    with open(json_path, encoding="utf-8") as fh:
        data = json.load(fh)

    usgaap = data.get("facts", {}).get("us-gaap", {})
    if not usgaap:
        return pd.DataFrame()

    if not concepts:
        canonical_set: set[str] = set()
        for short_name in usgaap:
            full_name = f"us-gaap:{short_name}"
            canonical = combined_alias_map.get(full_name, full_name)
            canonical_set.add(canonical)
        concepts = sorted(canonical_set)

    rows: list[dict] = []

    # Build reverse alias map: canonical -> [alias, ...] in explicit priority order.
    # CONCEPT_ALIAS_PRIORITY is the source of truth for non-lossy ordering;
    # lossy aliases (when enabled) are appended after the priority-ordered aliases.
    alias_lookup: dict[str, list[str]] = {
        c: list(CONCEPT_ALIAS_PRIORITY.get(c, [])) for c in concepts
    }
    # Append lossy alias candidates to the lookup for canonicals that have them,
    # so the candidate loop picks them up when lossy_alias_map is non-empty.
    for lossy_alias, canonical in effective_lossy_map.items():
        if canonical in alias_lookup and lossy_alias not in alias_lookup[canonical]:
            alias_lookup[canonical].append(lossy_alias)

    for concept_full in concepts:
        candidates = [concept_full, *alias_lookup.get(concept_full, [])]
        canonical_short = concept_full.split(":", 1)[-1]

        candidate_rows: list[dict] = []
        for candidate_full in candidates:
            concept_short = candidate_full.split(":", 1)[-1]
            concept_data = usgaap.get(concept_short)
            if concept_data is None:
                continue
            units_dict: dict = concept_data.get("units", {})
            # Union canonical-class and candidate-class unit preferences so a
            # cross-class alias still finds its entries. Canonical takes priority.
            units_to_try = _preferred_units(canonical_short)
            if concept_short != canonical_short:
                for u in _preferred_units(concept_short):
                    if u not in units_to_try:
                        units_to_try.append(u)
            unit_entries: list[dict] | None = None
            for unit_key in units_to_try:
                if unit_key in units_dict:
                    unit_entries = units_dict[unit_key]
                    break
            if not unit_entries:
                continue

            is_lossy = candidate_full in lossy_tags
            alias_source_value: str | None = candidate_full if is_lossy else None

            if is_lossy and candidate_full not in warned_lossy:
                logger.warning(
                    f"CIK {cik_padded}: lossy alias {candidate_full!r} → "
                    f"{concept_full!r}; rows will have alias_source set."
                )
                warned_lossy.add(candidate_full)

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
                        "alias_source": alias_source_value,
                    }
                )

        if not candidate_rows:
            continue

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

    # Scale correction (config-driven, opt-in).
    df["scale_corrected"] = False
    usd_mask = ~df["concept"].str.split(":").str[-1].isin(_SHARE_CONCEPTS)

    if scale_correction == "force":
        if usd_mask.any():
            df.loc[usd_mask, "val"] *= 1000
            df.loc[usd_mask, "scale_corrected"] = True
            logger.debug(
                f"CIK {cik_padded}: force scale correction applied to {int(usd_mask.sum())} USD rows"
            )
    elif scale_correction == "auto":
        if usd_mask.any():
            usd_df = df.loc[usd_mask]
            concepts_below = (
                usd_df.groupby("concept")["val"]
                .apply(lambda s: s.abs().max() < scale_correction_threshold)
            )
            n_below = int(concepts_below.sum())
            if n_below >= 2:
                max_val = usd_df["val"].abs().max()
                df.loc[usd_mask, "val"] *= 1000
                df.loc[usd_mask, "scale_corrected"] = True
                logger.warning(
                    f"CIK {cik_padded}: auto scale correction applied "
                    f"({n_below} USD concepts below threshold; "
                    f"max_val={max_val:.2f}, threshold={scale_correction_threshold:.0f})"
                )
    # scale_correction == "off": do nothing

    df["duration_days"] = (df["end"] - df["start"]).dt.days.where(df["start"].notna(), -1)

    df["_is_quarterly"] = is_quarterly(df["duration_days"]).astype(int)
    df = (
        df.sort_values(["filed", "_is_quarterly"])
        .drop_duplicates(subset=["concept", "end", "filed"], keep="last")
        .drop(columns=["_is_quarterly"])
    )

    df = df.sort_values(["concept", "end", "filed"])
    prev_val = df.groupby(["concept", "end"], sort=False)["val"].shift(1)
    keep = prev_val.isna() | ~_values_equal_tol(df["val"], prev_val, df["concept"])
    df = df[keep].sort_values("filed").reset_index(drop=True)

    return df


def _parse_one_for_pool(
    args: tuple[
        str, str, list[str] | None, Path, list[str], str, float,
        dict[str, str], dict[str, str],
    ],
) -> tuple[str, pd.DataFrame]:
    """Top-level helper for ProcessPoolExecutor (must be picklable)."""
    (
        ticker, cik_padded, concepts, facts_dir, forms,
        scale_correction, scale_correction_threshold,
        alias_map, lossy_alias_map,
    ) = args
    df = parse_company(
        cik_padded=cik_padded,
        concepts=concepts,
        facts_dir=facts_dir,
        forms=forms,
        scale_correction=scale_correction,
        scale_correction_threshold=scale_correction_threshold,
        alias_map=alias_map,
        lossy_alias_map=lossy_alias_map,
    )
    return ticker, df


def parse_all(
    config: PitEdgarConfig,
    cik_map: pd.DataFrame,
    force: bool = False,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """Parse all companies in cik_map into a single PIT master parquet."""
    config.ensure_dirs()
    out_path = config.data_dir / "pit_financials.parquet"

    if not force and out_path.exists():
        logger.info(f"Parquet already exists at {out_path}, skipping parse (use force=True to override).")
        return pd.read_parquet(out_path)

    if n_workers is None:
        n_workers = os.cpu_count() or 1
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")

    effective_lossy_map: dict[str, str] = (
        LOSSY_CONCEPT_ALIASES if config.lossy_aliases_enabled else {}
    )

    all_frames: list[pd.DataFrame] = []

    if n_workers == 1:
        for ticker, row in tqdm(cik_map.iterrows(), total=len(cik_map), desc="Parsing"):
            cik_padded = str(row["cik"])
            df = parse_company(
                cik_padded=cik_padded,
                concepts=config.concepts,
                facts_dir=config.facts_dir,
                forms=config.forms,
                scale_correction=config.scale_correction,
                scale_correction_threshold=config.scale_correction_threshold,
                alias_map=CONCEPT_ALIASES,
                lossy_alias_map=effective_lossy_map,
            )
            if df.empty:
                continue
            df.insert(0, "ticker", str(ticker))
            all_frames.append(df)
    else:
        tasks = [
            (
                str(ticker),
                str(row["cik"]),
                config.concepts,
                config.facts_dir,
                config.forms,
                config.scale_correction,
                config.scale_correction_threshold,
                CONCEPT_ALIASES,
                effective_lossy_map,
            )
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

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        master.to_parquet(tmp_path, index=False)
        tmp_path.replace(out_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    logger.info(f"Master parquet saved: {out_path} ({len(master):,} rows)")
    return master
