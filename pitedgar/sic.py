"""PIT SIC extraction from the SEC Financial Statement Data Sets.

Every quarterly Financial Statement Data Set (DERA,
https://www.sec.gov/dera/data/financial-statement-data-sets, 2009q2 ->
today) ships a ``sub.txt`` with one row per XBRL submission carrying
the registrant's SIC **as stated at filing time** — point-in-time by
construction, in bulk, from an official SEC product. This module turns
those files into ``pit_sic.parquet``: for each CIK, the SIC in force as
of any date, with filed-date semantics.

Decision record (arielNacamulli/pitedgar#40, approved 2026-07-15):

- **Source**: FSDS ``sub.txt`` beats per-filing header scraping
  (~60-100k rate-limited requests, brittle SGML) and periodic
  submissions-endpoint snapshots (current-only, no history). Coverage
  is the XBRL era (2009q2+), which matches the downstream model start
  (2010). Empirics on 2015q1/2024q1: SIC present on >97% of
  submissions; 92-99% of then-current S&P 500 members appear in any
  single quarter (100% of active filers over a rolling year).
- **All forms are kept** (with a ``form`` column): the SIC stated on a
  registration statement is as point-in-time as one on a 10-K;
  consumers filter if they want periodic reports only.
- Quarterly slices are cached as slim parquet (``fsds/sub_YYYYqN.
  parquet``); the ~100 MB source zips are discarded after extraction.
  Published FSDS quarters are immutable, so the cache never goes stale;
  a new quarter appears roughly one month after quarter end.

Defects are counted and reported, never silently dropped: rows with a
missing or unparseable SIC, and same-day conflicting SICs for one CIK
(latest accession wins).
"""

from __future__ import annotations

import datetime as dt
import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import requests

FSDS_URL = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/{quarter}.zip"
FIRST_QUARTER = (2009, 2)  # the series starts at 2009q2
SUB_COLUMNS = ["adsh", "cik", "name", "sic", "form", "filed"]
# FSDS quarters publish with a lag; a quarter is requested only once its
# end is at least this many days in the past.
PUBLICATION_LAG_DAYS = 35


class SicExtractionError(RuntimeError):
    """The PIT SIC series cannot be built as requested."""


def quarters_through(today: dt.date) -> list[str]:
    """All FSDS quarter labels from 2009q2 through the last published one."""
    out = []
    y, q = FIRST_QUARTER
    while True:
        quarter_end = dt.date(y, q * 3, 1) + dt.timedelta(days=31)
        quarter_end = quarter_end.replace(day=1) - dt.timedelta(days=1)
        if (today - quarter_end).days < PUBLICATION_LAG_DAYS:
            return out
        out.append(f"{y}q{q}")
        q += 1
        if q == 5:
            q, y = 1, y + 1


def parse_sub_txt(raw: bytes) -> pd.DataFrame:
    """The SIC-relevant columns of one ``sub.txt``, types normalized."""
    df = pd.read_csv(
        io.BytesIO(raw), sep="\t", dtype=str, usecols=lambda c: c in SUB_COLUMNS, low_memory=False
    )
    missing = set(SUB_COLUMNS) - set(df.columns)
    if missing:
        raise SicExtractionError(f"sub.txt layout changed: missing columns {sorted(missing)}")
    return df[SUB_COLUMNS]


def _fetch_quarter(quarter: str, identity: str, timeout: int = 300) -> pd.DataFrame:
    url = FSDS_URL.format(quarter=quarter)
    resp = requests.get(url, headers={"User-Agent": identity}, timeout=timeout)
    if resp.status_code != 200:
        raise SicExtractionError(f"FSDS download failed ({resp.status_code}): {url}")
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        if "sub.txt" not in zf.namelist():
            raise SicExtractionError(f"{quarter}.zip has no sub.txt (members: {zf.namelist()})")
        return parse_sub_txt(zf.read("sub.txt"))


def download_fsds_quarters(
    cache_dir: str | Path,
    identity: str,
    *,
    quarters: list[str] | None = None,
    force: bool = False,
    progress: bool = False,
) -> list[Path]:
    """Ensure a slim per-quarter parquet exists for every FSDS quarter.

    Downloads only quarters whose slice is missing (published quarters
    are immutable). Returns the cached paths, oldest first.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    quarters = quarters or quarters_through(dt.date.today())
    paths = []
    for q in quarters:
        path = cache / f"sub_{q}.parquet"
        if force or not path.exists():
            df = _fetch_quarter(q, identity)
            df.to_parquet(path, index=False)
            if progress:
                print(f"  {q}: {len(df):,} submissions")
        paths.append(path)
    return paths


def build_pit_sic(slice_paths: list[Path]) -> tuple[pd.DataFrame, dict]:
    """Assemble the PIT SIC series from cached quarterly slices.

    Returns ``(df, report)``. ``df`` columns: ``cik`` (10-padded),
    ``sic`` (int), ``filed`` (datetime), ``form``, ``adsh`` — one row
    per (cik, filed day), latest accession winning same-day conflicts.
    The report counts what was dropped and why.
    """
    if not slice_paths:
        raise SicExtractionError("no FSDS quarter slices supplied")
    frames = [pd.read_parquet(p) for p in slice_paths]
    df = pd.concat(frames, ignore_index=True)
    report: dict = {"quarters": len(slice_paths), "submissions": len(df)}

    df["sic_num"] = pd.to_numeric(df["sic"], errors="coerce")
    bad_sic = df["sic_num"].isna() | (df["sic_num"] <= 0) | (df["sic_num"] > 9999)
    report["dropped_missing_or_invalid_sic"] = int(bad_sic.sum())
    df = df[~bad_sic].copy()
    df["sic"] = df["sic_num"].astype(int)
    df["cik"] = df["cik"].str.zfill(10)
    df["filed"] = pd.to_datetime(df["filed"], format="%Y%m%d")

    # same-day duplicates: keep the latest accession; count real conflicts
    df = df.sort_values(["cik", "filed", "adsh"])
    day_groups = df.groupby(["cik", "filed"])["sic"]
    report["same_day_conflicting_sic"] = int((day_groups.nunique() > 1).sum())
    df = df.drop_duplicates(subset=["cik", "filed"], keep="last")

    out = df[["cik", "sic", "filed", "form", "adsh"]].reset_index(drop=True)
    report["rows"] = len(out)
    report["ciks"] = int(out["cik"].nunique())
    report["filed_span"] = [str(out["filed"].min().date()), str(out["filed"].max().date())]
    changes = out.groupby("cik")["sic"].nunique()
    report["ciks_with_sic_changes"] = int((changes > 1).sum())
    return out, report


def sic_as_of(pit_sic: pd.DataFrame, cik: str, as_of: dt.date | str) -> int | None:
    """The SIC in force for ``cik`` as of a date (filed-date semantics)."""
    ts = pd.Timestamp(as_of)
    sub = pit_sic[(pit_sic["cik"] == str(cik).zfill(10)) & (pit_sic["filed"] <= ts)]
    if sub.empty:
        return None
    return int(sub.sort_values("filed").iloc[-1]["sic"])


def coverage_report(pit_sic: pd.DataFrame, universe_ciks: list[str]) -> dict:
    """How much of a caller-supplied CIK universe the series covers."""
    have = set(pit_sic["cik"].unique())
    want = {str(c).zfill(10) for c in universe_ciks}
    missing = sorted(want - have)
    return {
        "universe_ciks": len(want),
        "covered": len(want & have),
        "missing": len(missing),
        "missing_ciks": missing[:50],
    }


def write_outputs(
    df: pd.DataFrame, report: dict, out_path: str | Path, report_path: str | Path | None = None
) -> None:
    out_path = Path(out_path)
    df.to_parquet(out_path, index=False)
    rp = Path(report_path) if report_path else out_path.with_suffix(".report.json")
    rp.write_text(json.dumps(report, indent=2), encoding="utf-8")


__all__ = [
    "FIRST_QUARTER",
    "FSDS_URL",
    "SicExtractionError",
    "build_pit_sic",
    "coverage_report",
    "download_fsds_quarters",
    "parse_sub_txt",
    "quarters_through",
    "sic_as_of",
    "write_outputs",
]
