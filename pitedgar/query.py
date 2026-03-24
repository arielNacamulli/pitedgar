"""PIT query API over the master parquet."""

from pathlib import Path

import pandas as pd


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
        max_staleness_days: int = 180,
    ) -> pd.DataFrame:
        """Last known value of a concept for each ticker as of a given date.

        Only filings with filed <= as_of_date are considered (no look-ahead).
        Tickers whose most recent filing is older than max_staleness_days
        return NaN for val.

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

        # Most recently filed record per ticker
        idx = sub.groupby("ticker")["filed"].idxmax()
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

        Returns DataFrame with columns: ticker, concept, end, filed, val, form, accn
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

        return sub.sort_values("filed").reset_index(drop=True)

    def cross_section(
        self,
        concept: str,
        as_of_date: str,
        tickers: list[str] | None = None,
        max_staleness_days: int = 180,
    ) -> pd.DataFrame:
        """Cross-section of all (or a subset of) tickers for a concept at a date.

        Useful for building portfolio signals.

        Returns DataFrame with columns: ticker, val, filed, end, form
        """
        universe = tickers if tickers is not None else self.data["ticker"].unique().tolist()
        return self.as_of(
            tickers=universe,
            concept=concept,
            as_of_date=as_of_date,
            max_staleness_days=max_staleness_days,
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
