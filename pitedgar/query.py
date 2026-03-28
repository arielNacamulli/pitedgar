"""PIT query API over the master parquet."""

from pathlib import Path

import pandas as pd


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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def as_of(
        self,
        tickers: list[str] | str,
        concept: str,
        as_of_date: str | pd.Timestamp,
        max_staleness_days: int = 100,
    ) -> pd.DataFrame:
        """Last known value of a concept for one or a few tickers at a single date.

        Use this for ad-hoc, interactive queries on a small number of tickers.
        For bulk use across many tickers or many dates, use cross_section() (balance
        sheet concepts) or ttm_cross_section() (flow concepts) instead — they are
        significantly faster due to vectorized merge_asof with a single pre-filter pass.

        Only filings with filed <= as_of_date are considered (no look-ahead bias).
        Values whose most recent filing predates as_of_date by more than
        max_staleness_days are returned as NaN.

        Args:
            tickers:            one ticker string or a list of ticker strings.
            concept:            XBRL concept, e.g. "us-gaap:Assets".
            as_of_date:         the observation date (ISO string or Timestamp).
            max_staleness_days: maximum age (in days) of the last filing before
                                the value is considered stale and set to NaN.

        Returns DataFrame with columns: ticker, val, filed, end, form
        """
        if isinstance(tickers, str):
            tickers = [tickers]
        as_of_ts = pd.Timestamp(as_of_date)
        cutoff = as_of_ts - pd.Timedelta(days=max_staleness_days)

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

        # Nullify stale values
        stale = result["filed"] < cutoff
        result.loc[stale, "val"] = float("nan")

        # Add tickers with no data at all
        missing = set(tickers) - set(result["ticker"])
        if missing:
            filler = pd.DataFrame(
                {"ticker": list(missing), "val": float("nan"), "filed": pd.NaT, "end": pd.NaT, "form": None}
            )
            result = pd.concat([result, filler], ignore_index=True)

        return result.reset_index(drop=True)

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
            freq:       "Q" → 10-Q only, "A" → 10-K only, anything else → all.

        Returns DataFrame with columns: ticker, concept, end, filed, val, form, accn.
        For each period end, shows the latest-filed (restated) value.
        """
        mask = (self.data["ticker"] == ticker) & (self.data["concept"] == concept)
        sub = self.data.loc[mask].copy()

        if freq == "Q":
            sub = sub[sub["form"] == "10-Q"]
        elif freq == "A":
            sub = sub[sub["form"] == "10-K"]

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
        mask = (
            (self.data["ticker"] == ticker)
            & (self.data["concept"] == concept)
            & (self.data["form"] == "10-Q")
        )
        sub = self.data.loc[mask].sort_values("filed")

        if sub.empty:
            return pd.DataFrame(columns=["ticker", "concept", "filed", "ttm_val", "n_periods"])

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
        max_staleness_days: int = 100,
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

        Args:
            concept:            XBRL concept, e.g. "us-gaap:Revenues".
            as_of_dates:        one date string or a list/DatetimeIndex of dates.
            tickers:            subset of tickers; defaults to full universe.
            min_periods:        minimum distinct quarters required before emitting a
                                TTM value (default 4). Set to 3 for companies with
                                less than one year of quarterly history.
            max_staleness_days: nullify TTM if the most recent quarterly filing
                                predates the as_of_date by more than this many days.

        Returns DataFrame with columns: as_of_date, ticker, ttm_val, n_periods, filed
        """
        if isinstance(as_of_dates, str):
            as_of_dates = [as_of_dates]
        dates = pd.DatetimeIndex(pd.to_datetime(as_of_dates)).sort_values()

        universe = tickers if tickers is not None else self.data["ticker"].unique().tolist()

        # Pre-filter once — avoid redundant scans inside the per-ticker loop.
        mask = (self.data["concept"] == concept) & (self.data["form"] == "10-Q")
        if tickers is not None:
            mask &= self.data["ticker"].isin(universe)
        df_q = self.data.loc[mask].sort_values(["ticker", "filed"])

        # Compute TTM events for all tickers in one grouped pass.
        ttm_frames: list[pd.DataFrame] = []
        tickers_with_data: set[str] = set()
        for ticker, grp in df_q.groupby("ticker", sort=False):
            events = _ttm_events(grp, min_periods=min_periods)
            if not events.empty:
                events["ticker"] = ticker
                ttm_frames.append(events)
                tickers_with_data.add(ticker)

        # Build cross-product (universe × dates).
        # merge_asof requires the key column to be globally monotonic, so sort by
        # as_of_date only (not ticker first); by="ticker" handles per-group matching.
        cross = pd.DataFrame(
            [(t, d) for t in universe for d in dates],
            columns=["ticker", "as_of_date"],
        ).sort_values("as_of_date").reset_index(drop=True)

        if not ttm_frames:
            cross[["ttm_val", "n_periods", "filed"]] = [float("nan"), 0, pd.NaT]
            return cross.sort_values(["as_of_date", "ticker"]).reset_index(drop=True)

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

        # Staleness check.
        staleness = (result["as_of_date"] - result["filed"]).dt.days
        stale = result["filed"].isna() | (staleness > max_staleness_days)
        result.loc[stale, "ttm_val"] = float("nan")
        result.loc[stale, "n_periods"] = 0

        # Add tickers that had no 10-Q data at all.
        missing = [t for t in universe if t not in tickers_with_data]
        if missing:
            filler = pd.DataFrame(
                [(t, d) for t in missing for d in dates],
                columns=["ticker", "as_of_date"],
            )
            filler[["ttm_val", "n_periods", "filed"]] = [float("nan"), 0, pd.NaT]
            result = pd.concat([result, filler], ignore_index=True)

        return (
            result[["as_of_date", "ticker", "ttm_val", "n_periods", "filed"]]
            .sort_values(["as_of_date", "ticker"])
            .reset_index(drop=True)
        )

    def cross_section(
        self,
        concept: str,
        as_of_dates: str | list[str] | pd.DatetimeIndex,
        tickers: list[str] | None = None,
        max_staleness_days: int = 100,
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

        Args:
            concept:            XBRL concept, e.g. "us-gaap:Assets".
            as_of_dates:        one date string or a list/DatetimeIndex of dates.
            tickers:            subset of tickers; defaults to full universe.
            max_staleness_days: nullify value if the last filing predates the
                                as_of_date by more than this many days.

        Returns DataFrame with columns: as_of_date, ticker, val, filed, end, form
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
        tickers_with_data: set[str] = set()
        for ticker, grp in df_c.groupby("ticker", sort=False):
            events = _snapshot_events(grp)
            if not events.empty:
                events["ticker"] = ticker
                snap_frames.append(events)
                tickers_with_data.add(ticker)

        # Build cross-product (universe × dates).
        # merge_asof requires the key column to be globally monotonic, so sort by
        # as_of_date only (not ticker first); by="ticker" handles per-group matching.
        cross = pd.DataFrame(
            [(t, d) for t in universe for d in dates],
            columns=["ticker", "as_of_date"],
        ).sort_values("as_of_date").reset_index(drop=True)

        if not snap_frames:
            cross[["val", "filed", "end", "form"]] = [float("nan"), pd.NaT, pd.NaT, None]
            return cross.sort_values(["as_of_date", "ticker"]).reset_index(drop=True)

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

        # Staleness check.
        staleness = (result["as_of_date"] - result["filed"]).dt.days
        stale = result["filed"].isna() | (staleness > max_staleness_days)
        result.loc[stale, "val"] = float("nan")

        # Add tickers with no data at all.
        missing = [t for t in universe if t not in tickers_with_data]
        if missing:
            filler = pd.DataFrame(
                [(t, d) for t in missing for d in dates],
                columns=["ticker", "as_of_date"],
            )
            filler[["val", "filed", "end", "form"]] = [float("nan"), pd.NaT, pd.NaT, None]
            result = pd.concat([result, filler], ignore_index=True)

        return (
            result[["as_of_date", "ticker", "val", "filed", "end", "form"]]
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
            }
        )
