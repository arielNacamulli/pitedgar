"""CLI entry points for pitedgar."""

from pathlib import Path

import click
import pandas as pd

from pitedgar.config import PitEdgarConfig
from pitedgar.downloader import download_bulk
from pitedgar.mapping import build_cik_map
from pitedgar.parser import parse_all
from pitedgar.query import PitQuery


@click.group()
def cli() -> None:
    """pitedgar — SEC EDGAR point-in-time financial data pipeline."""


@cli.command("map")
@click.option("--tickers", "tickers_file", required=True, type=click.Path(exists=True),
              help="Path to a text file with one ticker per line.")
@click.option("--identity", required=True,
              help='SEC identity string, e.g. "Name name@email.com".')
@click.option("--data-dir", default="./data", show_default=True,
              help="Directory where outputs are saved.")
def cmd_map(tickers_file: str, identity: str, data_dir: str) -> None:
    """Resolve tickers to CIK numbers and save the mapping."""
    tickers = Path(tickers_file).read_text().splitlines()
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    config = PitEdgarConfig(edgar_identity=identity, data_dir=Path(data_dir))
    cik_map = build_cik_map(tickers, config)
    click.echo(f"Mapped {len(cik_map)} tickers → {config.data_dir / 'ticker_cik_map.parquet'}")


@cli.command("fetch")
@click.option("--force", is_flag=True, default=False,
              help="Re-extract even if facts_dir already populated.")
@click.option("--identity", required=True,
              help='SEC identity string, e.g. "Name name@email.com".')
@click.option("--data-dir", default="./data", show_default=True)
def cmd_fetch(force: bool, identity: str, data_dir: str) -> None:
    """Download and extract the SEC companyfacts bulk ZIP."""
    config = PitEdgarConfig(edgar_identity=identity, data_dir=Path(data_dir))
    facts_dir = download_bulk(config, force=force)
    click.echo(f"Facts extracted to {facts_dir}")


@cli.command("build")
@click.option("--identity", required=True,
              help='SEC identity string, e.g. "Name name@email.com".')
@click.option("--data-dir", default="./data", show_default=True)
def cmd_build(identity: str, data_dir: str) -> None:
    """Parse all local JSON facts into the master PIT parquet."""
    config = PitEdgarConfig(edgar_identity=identity, data_dir=Path(data_dir))
    cik_map = pd.read_parquet(config.data_dir / "ticker_cik_map.parquet")
    master = parse_all(config, cik_map)
    click.echo(f"Built master parquet: {len(master):,} rows")


@cli.command("query")
@click.option("--ticker", required=True)
@click.option("--concept", required=True, help='e.g. "us-gaap:Revenues"')
@click.option("--as-of", "as_of", required=True, help="ISO date, e.g. 2023-06-30")
@click.option("--data-dir", default="./data", show_default=True)
def cmd_query(ticker: str, concept: str, as_of: str, data_dir: str) -> None:
    """Query the latest PIT value for a ticker/concept as of a date."""
    parquet_path = Path(data_dir) / "pit_financials.parquet"
    q = PitQuery(parquet_path)
    result = q.as_of([ticker.upper()], concept, as_of)
    click.echo(result.to_string(index=False))
