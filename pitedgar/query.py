"""PIT query API over the master parquet."""

import logging
from pathlib import Path

import pandas as pd

from pitedgar.periods import Q_MAX, Q_MIN, is_annual, is_quarterly

logger = logging.getLogger(__name__)


def _snapshot_events(grp: pd.DataFrame) -> pd.DataFrame:
    """Compute point-in-time snapshot events for one ticker/concept.

    At each filing date, records the current best value: the latest period end
    whose most-recently-filed value is known as of that date.
    Used to build the step function consumed by cross_section.

    Args:
        grp: DataFrame with columns end, filed, val, form, sorted by filed ascending.

    Returns DataFrame with columns: filed, end, val, form.
    """
    state: dict = {}  # {end_timestamp: (val, form)}
    rows = []
    filed_arr = grp["filed"].to_numpy()
    end_arr = grp["end"].to_numpy()
    val_arr = grp["val"].to_numpy()
    form_arr = grp["form"].to_numpy()

    i, n_total = 0, len(filed_arr)
    while i < n_total:
        current_filed = filed_arr[i]
        while i < n_total and filed_arr[i] == current_filed:
            state[end_arr[i]] = (val_arr[i], form_arr[i])
            i += 1
        latest_end = max(state.keys())
        val, form = state[latest_end]
        rows.append({"filed": current_filed, "end": latest_end, "val": val, "form": form})

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["filed", "end", "val", "form"])


def _derive_q4_rows(df_q: pd.DataFrame, df_k: pd.DataFrame) -> pd.DataFrame:
    """Derive synthetic Q4 rows from 10-K annual filings for one ticker.

    For each 10-K (end = FY_end, filed = K_filed, val = Annual), finds the
    three 10-Q filings within that fiscal year (end strictly in
    (FY_end − 366d, FY_end)) that were themselves filed on or before K_filed.
    If exactly 3 such quarterly periods are present (Q1, Q2, Q3), computes:

        Q4_val = Annual_val − Q1_val − Q2_val − Q3_val

    and returns a row with end=FY_end, filed=K_filed, val=Q4_val.

    Rows where fewer than 3 quarters are found are skipped: returning
    Annual − Q1 − Q2 would silently absorb multiple missing quarters into
    a single synthetic value, producing a wrong TTM.

    The returned rows carry the 10-K's filing date, so the staleness check in
    ttm / ttm_cross_section uses the 10-K date (not the last 10-Q date).

    Args:
        df_q: All 10-Q rows for one ticker (any sort order).
        df_k: All 10-K rows for one ticker (any sort order).

    Returns DataFrame with columns: end, filed, val.
    """
    if df_k.empty or df_q.empty:
        return pd.DataFrame(columns=["end", "filed", "val"])

    # Detect whether the 10-K rows carry a `start` column (parser may have
    # written it; older fixtures / legacy parquets may not).
    k_has_start = "start" in df_k.columns

    rows = []
    for _, k_row in df_k.iterrows():
        fy_end = k_row["end"]
        k_filed = k_row["filed"]
        annual_val = k_row["val"]

        # Prefer the 10-K's own fiscal-year start when available.  This is the
        # critical fix for 52/53-week fiscal calendars (Target, Walmart, Costco):
        # adjacent fiscal years can otherwise have Q4 end-dates that fall within
        # a 366-day window of the following fiscal year, inflating the count.
        # Fall back to a conservative 355-day window (less than 366 to avoid the
        # cross-year bleed) when the start column is missing or NaT.
        if k_has_start and pd.notna(k_row["start"]):
            fy_start = k_row["start"]
        else:
            fy_start = fy_end - pd.Timedelta(days=355)

        # 10-Q quarters within this fiscal year that were filed by the 10-K date.
        q_in_year = df_q[
            (df_q["end"] > fy_start) & (df_q["end"] < fy_end) & (df_q["filed"] <= k_filed)
        ].copy()

        if q_in_year.empty:
            continue

        # PIT dedup: keep latest-filed version of each period end.
        q_in_year = q_in_year.sort_values("filed").drop_duplicates(subset=["end"], keep="last")

        if len(q_in_year) != 3:
            # Require all three quarters present to safely isolate Q4.
            continue

        # --- Structural validation of the 3 matched quarters ---
        q_ends = q_in_year.sort_values("end")["end"].tolist()

        # 1. Ends must be strictly monotonically increasing.
        if q_ends[0] >= q_ends[1] or q_ends[1] >= q_ends[2]:
            logger.debug(
                "_derive_q4_rows: skipping fy_end=%s — quarter ends not monotonically increasing: %s",
                fy_end.date(),
                [e.date() for e in q_ends],
            )
            continue

        # 2. Cumulative span (Q1_end → Q3_end) must fall in [150, 280] days.
        span = (q_ends[2] - q_ends[0]).days
        if not (150 <= span <= 280):
            logger.debug(
                "_derive_q4_rows: skipping fy_end=%s — Q1→Q3 span %d days outside [150, 280]",
                fy_end.date(),
                span,
            )
            continue

        # 3. Each successive gap must be within the quarterly range [Q_MIN, Q_MAX].
        gap_01 = (q_ends[1] - q_ends[0]).days
        gap_12 = (q_ends[2] - q_ends[1]).days
        if not (Q_MIN <= gap_01 <= Q_MAX) or not (Q_MIN <= gap_12 <= Q_MAX):
            logger.debug(
                "_derive_q4_rows: skipping fy_end=%s — quarter gaps %d/%d outside [%d, %d]",
                fy_end.date(),
                gap_01,
                gap_12,
                Q_MIN,
                Q_MAX,
            )
            continue

        q4_val = annual_val - q_in_year["val"].sum()

        # 4. Sanity check: |Q4| must not exceed 2× the annual value.
        if abs(q4_val) > abs(annual_val) * 2:
            logger.warning(
                "_derive_q4_rows: skipping fy_end=%s — synthetic Q4 value %g seems absurd "
                "(annual=%g); likely wrong quarters matched",
                fy_end.date(),
                q4_val,
                annual_val,
            )
            continue

        rows.append({"end": fy_end, "filed": k_filed, "val": q4_val})

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["end", "filed", "val"])


_SYNTH_COLS = ["end", "filed", "val", "duration_days", "form", "accn", "start"]


def _derive_quarterly_from_ytd(df: pd.DataFrame) -> pd.DataFrame:
    """Synthesize 3-month rows from consecutive YTD records sharing a `start`.

    Some filers (notably AAPL post-2021 for NetIncomeLoss/Revenues) tag their
    quarterly disclosures only as year-to-date cumulative values: Q1 is a 90-day
    record, Q2 a 181-day record, Q3 a 272-day record, and the 10-K a 363-day
    record — all sharing the same fiscal-year start. Without discrete 3-month
    rows, ``_ttm_events`` has nothing to sum and ``ttm()`` falls back to summing
    four Q1s from different years, producing an inflated TTM.

    For each ``start`` group, this walks the sorted-by-end records and emits
    a synthetic 3-month row for every consecutive YTD pair whose end-to-end
    delta lands in the quarterly range:

        Q2_3m = YTD_6m − YTD_3m     (quarters 181d, 90d → 91d synthetic)
        Q3_3m = YTD_9m − YTD_6m     (quarters 272d, 181d → 91d synthetic)
        Q4_3m = FY − YTD_9m         (quarters 363d, 272d → 91d synthetic)

    Restatement handling: if either YTD end has multiple filings (different
    `filed`/`val`), this emits one synthesized row per (prev_record, curr_record)
    combination. The TTM state dict keyed by ``end`` absorbs duplicates at the
    same (end, filed).

    No look-ahead: synthesized `filed` = max(prev.filed, curr.filed), so the
    row is only "known" after both YTD disclosures were filed.

    Args:
        df: rows for a single (ticker, concept). Must contain the columns
            ``start``, ``end``, ``filed``, ``val``; ``form`` and ``accn`` are
            used only for audit metadata on the emitted rows. Rows with a
            missing ``start`` (e.g. balance-sheet instants) are ignored.

    Returns DataFrame with columns: end, filed, val, duration_days, form,
    accn, start. Empty when no YTD chain can be synthesized — including when
    ``start`` is not present on the input.
    """
    empty = pd.DataFrame(columns=_SYNTH_COLS)
    if df.empty or "start" not in df.columns:
        return empty

    ytd = df[df["start"].notna()]
    if ytd.empty:
        return empty

    rows: list[dict] = []
    for _, grp in ytd.groupby("start", sort=True):
        ends = sorted(grp["end"].unique())
        if len(ends) < 2:
            continue
        for i in range(1, len(ends)):
            prev_end, curr_end = ends[i - 1], ends[i]
            duration = (curr_end - prev_end).days
            if duration < Q_MIN or duration > Q_MAX:
                continue
            prev_rows = grp[grp["end"] == prev_end]
            curr_rows = grp[grp["end"] == curr_end]
            for _, p in prev_rows.iterrows():
                for _, c in curr_rows.iterrows():
                    filed = p["filed"] if p["filed"] >= c["filed"] else c["filed"]
                    src_accn = c.get("accn", "")
                    src_form = c.get("form", "")
                    rows.append(
                        {
                            "end": curr_end,
                            "filed": filed,
                            "val": c["val"] - p["val"],
                            "duration_days": duration,
                            "form": src_form,
                            "accn": f"{src_accn}:DERIVED_YTD_DIFF",
                            "start": prev_end,
                        }
                    )
    return pd.DataFrame(rows, columns=_SYNTH_COLS) if rows else empty


def _combine_quarterly_sources(
    sub_all: pd.DataFrame,
    quarterly_mask: pd.Series,
) -> pd.DataFrame:
    """Return the full 3-month row set for a (ticker, concept): explicit ∪ synthesized.

    Explicit quarterly rows (those with ``is_quarterly(duration_days)``) take
    precedence over synthesized ones at the same ``(end, filed)``. Returned
    DataFrame has columns ``end, filed, val`` — the minimum needed by
    ``_ttm_events`` and ``_derive_q4_rows``.
    """
    explicit = sub_all.loc[quarterly_mask]
    synth = _derive_quarterly_from_ytd(sub_all)

    if synth.empty:
        return explicit[["end", "filed", "val"]].copy()

    # Concat synthesized first, explicit second → drop_duplicates keep='last'
    # so explicit wins when both are present for the same (end, filed).
    synth_keep = synth[["end", "filed", "val"]]
    explicit_keep = explicit[["end", "filed", "val"]]
    combined = pd.concat([synth_keep, explicit_keep], ignore_index=True)
    combined = combined.drop_duplicates(subset=["end", "filed"], keep="last")
    return combined


def _ttm_events(grp: pd.DataFrame, min_periods: int = 4) -> pd.DataFrame:
    """Compute TTM step-function events for one ticker from sorted 10-Q rows.

    Uses a running-state dict: each filing date updates the current value for
    that period end, then emits one TTM row per unique filing date.
    Complexity: O(n log n) — dominated by the top-4 sort, not a DataFrame scan.

    Args:
        grp:         DataFrame with columns end, filed, val, sorted by filed ascending.
        min_periods: minimum distinct quarters required before emitting a row.

    Returns DataFrame with columns: filed, ttm_val, n_periods.
    """
    state: dict = {}  # {end_timestamp: val}
    rows = []
    filed_arr = grp["filed"].to_numpy()
    end_arr = grp["end"].to_numpy()
    val_arr = grp["val"].to_numpy()

    i, n_total = 0, len(filed_arr)
    while i < n_total:
        current_filed = filed_arr[i]
        # Absorb all rows sharing this filed date (same-day filings / restatements)
        while i < n_total and filed_arr[i] == current_filed:
            state[end_arr[i]] = val_arr[i]
            i += 1
        top = sorted(state.keys(), reverse=True)[:4]
        n = len(top)
        if n >= min_periods:
            rows.append({"filed": current_filed, "ttm_val": sum(state[k] for k in top), "n_periods": n})

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["filed", "ttm_val", "n_periods"])


class PitQuery:
    """Query point-in-time financial data from a master parquet file."""

    def __init__(self, parquet_path: Path) -> None:
        self.data = pd.read_parquet(parquet_path)
        self.data["filed"] = pd.to_datetime(self.data["filed"], errors="coerce")
        self.data["end"] = pd.to_datetime(self.data["end"], errors="coerce")
        # `start` is required by YTD-chain synthesis in ttm(); older fixtures and
        # legacy parquets may lack it, in which case synthesis silently no-ops.
        if "start" in self.data.columns:
            self.data["start"] = pd.to_datetime(self.data["start"], errors="coerce")
        # Ensure duration_days exists for period classification. The parser always
        # writes this column; for test fixtures or legacy parquets without it,
        # infer from form (10-Q → 91, 10-K → 365) as a safe approximation.
        if "duration_days" not in self.data.columns:
            self.data["duration_days"] = self.data["form"].map({"10-Q": 91, "10-K": 365}).fillna(-1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def as_of(
        self,
        tickers: list[str] | str,
        concept: str,
        as_of_date: str | pd.Timestamp,
        max_staleness_days: int | None = None,
    ) -> pd.DataFrame:
        """Last known value of a concept for one or a few tickers at a single date.

        Use this for ad-hoc, interactive queries on a small number of tickers.
        For bulk use across many tickers or many dates, use cross_section() (balance
        sheet concepts) or ttm_cross_section() (flow concepts) instead — they are
        significantly faster due to vectorized merge_asof with a single pre-filter pass.

        Only filings with filed <= as_of_date are considered (no look-ahead bias).
        The result always includes an ``age_days`` column (as_of_date − filed in
        days) so callers can decide what to do with stale values themselves. When
        ``max_staleness_days`` is set to an int, values whose most recent filing
        predates ``as_of_date`` by more than that many days are nullified.

        v0.2.x changed the default from 100 to None to surface signal; pass an
        int to restore the previous behavior.

        Args:
            tickers:            one ticker string or a list of ticker strings.
            concept:            XBRL concept, e.g. "us-gaap:Assets".
            as_of_date:         the observation date (ISO string or Timestamp).
            max_staleness_days: maximum age (in days) of the last filing before
                                the value is considered stale and set to NaN.
                                Default ``None`` keeps every value; ``age_days``
                                is always returned regardless.

        Returns DataFrame with columns: ticker, val, filed, end, form, age_days
        """
        if isinstance(tickers, str):
            tickers = [tickers]
        as_of_ts = pd.Timestamp(as_of_date)

        mask = (
            self.data["ticker"].isin(tickers)
            & (self.data["concept"] == concept)
            & (self.data["filed"] <= as_of_ts)
        )
        sub = self.data.loc[mask].copy()

        if sub.empty:
            return self._empty_result(tickers)

        # PIT dedup: for each (ticker, end) keep the latest-filed version as of as_of_ts.
        # This makes restatements filed after their original date correctly supersede the original.
        sub = sub.sort_values("filed").drop_duplicates(subset=["ticker", "end"], keep="last")

        # Among all periods available, take the most recent period end per ticker.
        idx = sub.groupby("ticker")["end"].idxmax()
        result = sub.loc[idx, ["ticker", "val", "filed", "end", "form"]].copy()

        # Always surface age_days so callers can decide what to do with stale rows.
        result["age_days"] = (as_of_ts - result["filed"]).dt.days.astype("Int64")

        # Only nullify when the caller opts in by passing an int.
        if max_staleness_days is not None:
            stale = result["age_days"] > max_staleness_days
            result.loc[stale, "val"] = float("nan")

        # Add tickers with no data at all
        missing = set(tickers) - set(result["ticker"])
        if missing:
            filler = pd.DataFrame(
                {
                    "ticker": list(missing),
                    "val": float("nan"),
                    "filed": pd.NaT,
                    "end": pd.NaT,
                    "form": None,
                    "age_days": pd.array([pd.NA] * len(missing), dtype="Int64"),
                }
            )
            result = pd.concat([result, filler], ignore_index=True)

        return result[["ticker", "val", "filed", "end", "form", "age_days"]].reset_index(drop=True)

    def history(
        self,
        ticker: str,
        concept: str,
        start_date: str | None = None,
        end_date: str | None = None,
        freq: str = "Q",
    ) -> pd.DataFrame:
        """Point-in-time history of a concept for a single ticker.

        Args:
            ticker:     company ticker symbol.
            concept:    XBRL concept, e.g. "us-gaap:Revenues".
            start_date: filter end >= start_date (ISO string).
            end_date:   filter end <= end_date (ISO string).
            freq:       "Q" → quarterly periods (60-105 days),
                        "A" → annual filings (10-K/A/20-F with 340-380 day duration),
                        anything else → all.

        Returns DataFrame with columns: ticker, concept, end, filed, val, form, accn.
        For each period end, shows the latest-filed (restated) value.
        """
        mask = (self.data["ticker"] == ticker) & (self.data["concept"] == concept)

        # Classify by duration_days, not form: some filers (notably Apple for
        # us-gaap:Revenues) report quarterly breakdowns inside the annual 10-K
        # as comparative disclosures. A form-based filter would miss them.
        # Annual still requires an annual form so non-annual filings with
        # coincidentally long durations aren't misclassified.
        if freq == "Q":
            mask &= is_quarterly(self.data["duration_days"])
        elif freq == "A":
            mask &= is_annual(self.data["duration_days"], self.data["form"])

        sub = self.data.loc[mask].copy()

        if start_date is not None:
            sub = sub[sub["end"] >= pd.Timestamp(start_date)]
        if end_date is not None:
            sub = sub[sub["end"] <= pd.Timestamp(end_date)]

        # Return one row per period end using the latest-filed (restated) value.
        sub = sub.sort_values("filed").drop_duplicates(subset=["end"], keep="last")
        return sub.sort_values("end").reset_index(drop=True)

    def ttm(
        self,
        ticker: str,
        concept: str,
        start_date: str | None = None,
        end_date: str | None = None,
        min_periods: int = 4,
    ) -> pd.DataFrame:
        """Point-in-time trailing-twelve-month series from quarterly filings.

        For each 10-Q filing date, sums the 4 most recently available discrete
        quarters as of that date (no look-ahead bias). Rows with fewer than
        min_periods available quarters are dropped.

        Args:
            ticker:      company ticker symbol.
            concept:     XBRL concept, e.g. "us-gaap:Revenues".
            start_date:  include only rows with filed >= start_date.
            end_date:    include only rows with filed <= end_date.
            min_periods: minimum quarters required to emit a TTM row (default 4).

        Returns DataFrame with columns: ticker, concept, filed, ttm_val, n_periods
        """
        base_mask = (self.data["ticker"] == ticker) & (self.data["concept"] == concept)
        sub_all = self.data.loc[base_mask]
        sub_k = sub_all.loc[is_annual(sub_all["duration_days"], sub_all["form"])]

        # Union explicit 3-month rows (duration_days 60–105, regardless of form —
        # some companies like JNJ report quarterly breakdowns inside 10-K filings)
        # with synthesized 3-month rows from consecutive YTD pairs (AAPL post-2021
        # pattern: Q2/Q3 tagged only as YTD, no discrete 3-month disclosure).
        sub = _combine_quarterly_sources(sub_all, is_quarterly(sub_all["duration_days"])).sort_values("filed")

        if sub.empty:
            return pd.DataFrame(columns=["ticker", "concept", "filed", "ttm_val", "n_periods"])

        # Inject synthetic Q4 rows derived from 10-K annual filings so that
        # December-FY companies (and others whose Q4 only appears in a 10-K)
        # are not falsely nullified by the staleness check.
        q4_rows = _derive_q4_rows(sub, sub_k)
        if not q4_rows.empty:
            sub = pd.concat([sub, q4_rows], ignore_index=True).sort_values("filed")

        result = _ttm_events(sub, min_periods=min_periods)
        if result.empty:
            return pd.DataFrame(columns=["ticker", "concept", "filed", "ttm_val", "n_periods"])

        result.insert(0, "ticker", ticker)
        result.insert(1, "concept", concept)

        if start_date is not None:
            result = result[result["filed"] >= pd.Timestamp(start_date)]
        if end_date is not None:
            result = result[result["filed"] <= pd.Timestamp(end_date)]

        return result.reset_index(drop=True)

    def ttm_cross_section(
        self,
        concept: str,
        as_of_dates: str | list[str] | pd.DatetimeIndex,
        tickers: list[str] | None = None,
        min_periods: int = 4,
        max_staleness_days: int | None = None,
    ) -> pd.DataFrame:
        """Trailing-twelve-month values for a full universe across one or more dates.

        This is the bulk API for flow concepts (Revenues, NetIncomeLoss,
        NetCashProvidedByUsedInOperatingActivities, PaymentsToAcquirePropertyPlantAndEquipment,
        ResearchAndDevelopmentExpense, GrossProfit, OperatingIncomeLoss). TTM is the
        sum of the four most recent discrete 10-Q quarters available as of each date,
        enforcing strict point-in-time correctness (no look-ahead bias).

        For balance sheet concepts (Assets, Liabilities, Equity, Cash, Debt) use
        cross_section() instead — those are point-in-time snapshots, not period sums.

        For quick ad-hoc TTM lookups on a single ticker use ttm().

        Optimized for large universes (S&P 500 / Russell 3000 + many rebalance dates):
        - Computes TTM step-functions for all tickers in one grouped O(n log n) pass
        - Resolves all as_of_dates in a single vectorized merge_asof (no per-date loop)

        The result always includes an ``age_days`` column (as_of_date − filed in
        days) so callers can decide what to do with stale values themselves. When
        ``max_staleness_days`` is set to an int, TTM values whose underlying
        filing is older than that many days are nullified.

        v0.2.x changed the default from 100 to None to surface signal; pass an
        int to restore the previous behavior.

        Args:
            concept:            XBRL concept, e.g. "us-gaap:Revenues".
            as_of_dates:        one date string or a list/DatetimeIndex of dates.
            tickers:            subset of tickers; defaults to full universe.
            min_periods:        minimum distinct quarters required before emitting a
                                TTM value (default 4). Set to 3 for companies with
                                less than one year of quarterly history.
            max_staleness_days: nullify TTM if the most recent quarterly filing
                                predates the as_of_date by more than this many days.
                                Default ``None`` keeps every value; ``age_days``
                                is always returned regardless.

        Returns DataFrame with columns:
            as_of_date, ticker, ttm_val, n_periods, filed, age_days
        """
        if isinstance(as_of_dates, str):
            as_of_dates = [as_of_dates]
        dates = pd.DatetimeIndex(pd.to_datetime(as_of_dates)).sort_values()

        universe = tickers if tickers is not None else self.data["ticker"].unique().tolist()

        # Pre-filter once — avoid redundant scans inside the per-ticker loop.
        # Use duration_days to classify periods, not form: some companies (e.g. JNJ)
        # only report discrete quarterly values as comparative data inside 10-K filings.
        concept_mask = self.data["concept"] == concept
        if tickers is not None:
            concept_mask &= self.data["ticker"].isin(universe)
        df_all = self.data.loc[concept_mask].sort_values(["ticker", "filed"])

        # Compute TTM events for all tickers in one grouped pass.
        # Per ticker: build the unified 3-month row set (explicit ∪ YTD-synthesized),
        # then inject a synthetic Q4 from 10-K when needed. The YTD step covers
        # AAPL post-2021 (Q2/Q3 tagged only as YTD); the 10-K step covers December-FY
        # companies whose Q4 never appears in a 10-Q.
        ttm_frames: list[pd.DataFrame] = []
        tickers_with_data: set = set()
        q_mask = is_quarterly(df_all["duration_days"])
        k_mask = is_annual(df_all["duration_days"], df_all["form"])
        for ticker, grp_all in df_all.groupby("ticker", sort=False):
            grp_q_mask = q_mask.loc[grp_all.index]
            grp_k = grp_all.loc[k_mask.loc[grp_all.index]]
            combined = _combine_quarterly_sources(grp_all, grp_q_mask).sort_values("filed")
            if combined.empty:
                continue
            q4_rows = _derive_q4_rows(combined, grp_k)
            if not q4_rows.empty:
                combined = pd.concat([combined, q4_rows], ignore_index=True).sort_values("filed")
            events = _ttm_events(combined, min_periods=min_periods)
            if not events.empty:
                events["ticker"] = ticker
                ttm_frames.append(events)
                tickers_with_data.add(ticker)

        # Build cross-product (universe × dates).
        # merge_asof requires the key column to be globally monotonic, so sort by
        # as_of_date only (not ticker first); by="ticker" handles per-group matching.
        cross = (
            pd.DataFrame(
                [(t, d) for t in universe for d in dates],
                columns=["ticker", "as_of_date"],
            )
            .sort_values("as_of_date")
            .reset_index(drop=True)
        )

        if not ttm_frames:
            cross[["ttm_val", "n_periods", "filed"]] = [float("nan"), 0, pd.NaT]  # type: ignore[list-item,assignment]
            cross["age_days"] = pd.array([pd.NA] * len(cross), dtype="Int64")
            return (
                cross[["as_of_date", "ticker", "ttm_val", "n_periods", "filed", "age_days"]]
                .sort_values(["as_of_date", "ticker"])
                .reset_index(drop=True)
            )

        # Sort by filed globally so the key column is monotonic for merge_asof.
        all_events = pd.concat(ttm_frames, ignore_index=True).sort_values("filed")

        # Single vectorized lookup: merge_asof groups by ticker in one C-level pass.
        result = pd.merge_asof(
            cross,
            all_events[["ticker", "filed", "ttm_val", "n_periods"]],
            left_on="as_of_date",
            right_on="filed",
            by="ticker",
            direction="backward",
        )

        # Always surface age_days so callers can decide what to do with stale rows.
        result["age_days"] = (result["as_of_date"] - result["filed"]).dt.days.astype("Int64")

        # Only nullify when the caller opts in by passing an int.
        if max_staleness_days is not None:
            stale = result["filed"].isna() | (result["age_days"] > max_staleness_days)
            result.loc[stale, "ttm_val"] = float("nan")
            result.loc[stale, "n_periods"] = 0

        # Add tickers that had no 10-Q data at all.
        missing = [t for t in universe if t not in tickers_with_data]
        if missing:
            filler = pd.DataFrame(
                [(t, d) for t in missing for d in dates],
                columns=["ticker", "as_of_date"],
            )
            filler[["ttm_val", "n_periods", "filed"]] = [float("nan"), 0, pd.NaT]  # type: ignore[list-item,assignment]
            filler["age_days"] = pd.array([pd.NA] * len(filler), dtype="Int64")
            result = pd.concat([result, filler], ignore_index=True)

        return (
            result[["as_of_date", "ticker", "ttm_val", "n_periods", "filed", "age_days"]]
            .sort_values(["as_of_date", "ticker"])
            .reset_index(drop=True)
        )

    def cross_section(
        self,
        concept: str,
        as_of_dates: str | list[str] | pd.DatetimeIndex,
        tickers: list[str] | None = None,
        max_staleness_days: int | None = None,
    ) -> pd.DataFrame:
        """Snapshot values for a full universe of tickers across one or more dates.

        This is the bulk API for balance sheet concepts (Assets, Liabilities, Equity,
        Cash, LongTermDebt). It pre-filters the data once and resolves all tickers and
        dates with a single vectorized merge_asof — suitable for S&P 500 / Russell 3000
        scale and multiple rebalance dates.

        For flow concepts (Revenues, NetIncome, OperatingCashFlow, CapEx, R&D) use
        ttm_cross_section() instead, which sums the four most recent quarters to give
        a trailing-twelve-month value rather than a single-period snapshot.

        For quick ad-hoc lookups on one or a few tickers at a single date, as_of()
        is more ergonomic (no as_of_date column in the result).

        The result always includes an ``age_days`` column (as_of_date − filed in
        days) so callers can decide what to do with stale values themselves. When
        ``max_staleness_days`` is set to an int, values whose underlying filing
        is older than that many days are nullified.

        v0.2.x changed the default from 100 to None to surface signal; pass an
        int to restore the previous behavior.

        Args:
            concept:            XBRL concept, e.g. "us-gaap:Assets".
            as_of_dates:        one date string or a list/DatetimeIndex of dates.
            tickers:            subset of tickers; defaults to full universe.
            max_staleness_days: nullify value if the last filing predates the
                                as_of_date by more than this many days. Default
                                ``None`` keeps every value; ``age_days`` is always
                                returned regardless.

        Returns DataFrame with columns:
            as_of_date, ticker, val, filed, end, form, age_days
        """
        if isinstance(as_of_dates, str):
            as_of_dates = [as_of_dates]
        dates = pd.DatetimeIndex(pd.to_datetime(as_of_dates)).sort_values()

        universe = tickers if tickers is not None else self.data["ticker"].unique().tolist()

        # Pre-filter once.
        mask = self.data["concept"] == concept
        if tickers is not None:
            mask &= self.data["ticker"].isin(universe)
        df_c = self.data.loc[mask].sort_values(["ticker", "filed"])

        # Compute snapshot step-functions for all tickers in one grouped pass.
        snap_frames: list[pd.DataFrame] = []
        tickers_with_data: set = set()
        for ticker, grp in df_c.groupby("ticker", sort=False):
            events = _snapshot_events(grp)
            if not events.empty:
                events["ticker"] = ticker
                snap_frames.append(events)
                tickers_with_data.add(ticker)

        # Build cross-product (universe × dates).
        # merge_asof requires the key column to be globally monotonic, so sort by
        # as_of_date only (not ticker first); by="ticker" handles per-group matching.
        cross = (
            pd.DataFrame(
                [(t, d) for t in universe for d in dates],
                columns=["ticker", "as_of_date"],
            )
            .sort_values("as_of_date")
            .reset_index(drop=True)
        )

        if not snap_frames:
            cross[["val", "filed", "end", "form"]] = [float("nan"), pd.NaT, pd.NaT, None]
            cross["age_days"] = pd.array([pd.NA] * len(cross), dtype="Int64")
            return (
                cross[["as_of_date", "ticker", "val", "filed", "end", "form", "age_days"]]
                .sort_values(["as_of_date", "ticker"])
                .reset_index(drop=True)
            )

        # Sort by filed globally (not ticker-first) so the key column is monotonic.
        all_events = pd.concat(snap_frames, ignore_index=True).sort_values("filed")

        # Single vectorized lookup.
        result = pd.merge_asof(
            cross,
            all_events[["ticker", "filed", "end", "val", "form"]],
            left_on="as_of_date",
            right_on="filed",
            by="ticker",
            direction="backward",
        )

        # Always surface age_days so callers can decide what to do with stale rows.
        result["age_days"] = (result["as_of_date"] - result["filed"]).dt.days.astype("Int64")

        # Only nullify when the caller opts in by passing an int.
        if max_staleness_days is not None:
            stale = result["filed"].isna() | (result["age_days"] > max_staleness_days)
            result.loc[stale, "val"] = float("nan")

        # Add tickers with no data at all.
        missing = [t for t in universe if t not in tickers_with_data]
        if missing:
            filler = pd.DataFrame(
                [(t, d) for t in missing for d in dates],
                columns=["ticker", "as_of_date"],
            )
            filler[["val", "filed", "end", "form"]] = [float("nan"), pd.NaT, pd.NaT, None]
            filler["age_days"] = pd.array([pd.NA] * len(filler), dtype="Int64")
            result = pd.concat([result, filler], ignore_index=True)

        return (
            result[["as_of_date", "ticker", "val", "filed", "end", "form", "age_days"]]
            .sort_values(["as_of_date", "ticker"])
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result(tickers: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ticker": tickers,
                "val": float("nan"),
                "filed": pd.NaT,
                "end": pd.NaT,
                "form": None,
                "age_days": pd.array([pd.NA] * len(tickers), dtype="Int64"),
            }
        )
