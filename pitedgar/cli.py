"""CLI entry points for pitedgar."""

from pathlib import Path

import click
import pandas as pd
from pydantic import ValidationError

from pitedgar.config import PitEdgarConfig
from pitedgar.downloader import download_bulk
from pitedgar.mapping import build_cik_map
from pitedgar.parser import parse_all
from pitedgar.query import PitQuery
from pitedgar.util import normalize_ticker


def _build_config(identity: str, data_dir: str) -> PitEdgarConfig:
    """Build a PitEdgarConfig, translating validation errors into a Click usage error."""
    try:
        return PitEdgarConfig(edgar_identity=identity, data_dir=Path(data_dir))
    except ValidationError as exc:
        raise click.UsageError(f"Invalid configuration: {exc}") from exc


def _require_file(path: Path, *, hint: str) -> None:
    """Raise a Click error if *path* does not exist, pointing users at the missing step."""
    if not path.exists():
        raise click.ClickException(f"Required file not found: {path}\n{hint}")


@click.group()
def cli() -> None:
    """pitedgar — SEC EDGAR point-in-time financial data pipeline."""


@cli.command("map")
@click.option(
    "--tickers",
    "tickers_file",
    required=True,
    type=click.Path(exists=True),
    help="Path to a text file with one ticker per line.",
)
@click.option("--identity", required=True, help='SEC identity string, e.g. "Name name@email.com".')
@click.option("--data-dir", default="./data", show_default=True, help="Directory where outputs are saved.")
@click.option(
    "--force", is_flag=True, default=False, help="Re-resolve all tickers, ignoring the existing cache."
)
def cmd_map(tickers_file: str, identity: str, data_dir: str, force: bool) -> None:
    """Resolve tickers to CIK numbers and save the mapping."""
    try:
        raw = Path(tickers_file).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise click.ClickException(f"Cannot read tickers file {tickers_file!r}: {exc}") from exc
    tickers = [normalize_ticker(t) for t in raw.splitlines() if t.strip()]
    if not tickers:
        raise click.ClickException(f"Tickers file {tickers_file!r} is empty.")
    config = _build_config(identity, data_dir)
    cik_map = build_cik_map(tickers, config, force=force)
    click.echo(f"Mapped {len(cik_map)} tickers → {config.data_dir / 'ticker_cik_map.parquet'}")


@cli.command("fetch")
@click.option("--force", is_flag=True, default=False, help="Re-extract even if facts_dir already populated.")
@click.option("--identity", required=True, help='SEC identity string, e.g. "Name name@email.com".')
@click.option("--data-dir", default="./data", show_default=True)
def cmd_fetch(force: bool, identity: str, data_dir: str) -> None:
    """Download and extract the SEC companyfacts bulk ZIP."""
    config = _build_config(identity, data_dir)
    facts_dir = download_bulk(config, force=force)
    click.echo(f"Facts extracted to {facts_dir}")


@cli.command("build")
@click.option("--identity", required=True, help='SEC identity string, e.g. "Name name@email.com".')
@click.option("--data-dir", default="./data", show_default=True)
@click.option(
    "--force", is_flag=True, default=False, help="Re-parse even if pit_financials.parquet already exists."
)
@click.option(
    "--workers",
    "-j",
    "workers",
    type=int,
    default=None,
    help="Number of worker processes for parallel parsing. Default: all CPU cores. Use 1 for serial.",
)
def cmd_build(identity: str, data_dir: str, force: bool, workers: int | None) -> None:
    """Parse all local JSON facts into the master PIT parquet."""
    config = _build_config(identity, data_dir)
    cik_map_path = config.data_dir / "ticker_cik_map.parquet"
    _require_file(cik_map_path, hint="Run `pitedgar map` first to create it.")
    cik_map = pd.read_parquet(cik_map_path)
    master = parse_all(config, cik_map, force=force, n_workers=workers)
    click.echo(f"Built master parquet: {len(master):,} rows")


@cli.command("dei")
@click.option("--data-dir", default="./data", show_default=True)
@click.option("--facts-dir", default=None, help="companyfacts dir (default: <data-dir>/companyfacts).")
@click.option(
    "--map-parquet",
    default=None,
    help="Parquet with (ticker, cik) columns for the issuer map "
    "(default: <data-dir>/ticker_cik_map.parquet, else pit_financials.parquet).",
)
@click.option("--out", default=None, help="Output path (default: <data-dir>/dei_shares.parquet).")
def cmd_dei(data_dir: str, facts_dir: str | None, map_parquet: str | None, out: str | None) -> None:
    """Extract dei cover-page share counts with per-fact quality flags."""
    from pitedgar.dei import extract_dei_shares, ticker_map_from_financials

    base = Path(data_dir)
    facts = Path(facts_dir) if facts_dir else base / "companyfacts"
    _require_file(facts, hint="Run `pitedgar fetch` first to download companyfacts.")
    map_path = Path(map_parquet) if map_parquet else base / "ticker_cik_map.parquet"
    if not map_path.exists():
        map_path = base / "pit_financials.parquet"
    _require_file(map_path, hint="Need ticker_cik_map.parquet or pit_financials.parquet for tickers.")
    tick_map = ticker_map_from_financials(map_path)
    df = extract_dei_shares(facts, tick_map)
    out_path = Path(out) if out else base / "dei_shares.parquet"
    df.to_parquet(out_path, index=False)
    by_q = df["quality"].value_counts().to_dict() if len(df) else {}
    n_issuers = df["cticker"].nunique() if len(df) else 0
    click.echo(f"dei_shares: {len(df):,} rows for {n_issuers} issuers")
    click.echo(f"quality: {by_q}")
    click.echo(f"written: {out_path}")


@cli.command("query")
@click.option("--ticker", required=True)
@click.option("--concept", required=True, help='e.g. "us-gaap:Revenues"')
@click.option("--as-of", "as_of", required=True, help="ISO date, e.g. 2023-06-30")
@click.option("--data-dir", default="./data", show_default=True)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["table", "json", "csv"], case_sensitive=False),
    default="table",
    show_default=True,
    help="Output format.",
)
def cmd_query(ticker: str, concept: str, as_of: str, data_dir: str, fmt: str) -> None:
    """Query the latest PIT value for a ticker/concept as of a date."""
    parquet_path = Path(data_dir) / "pit_financials.parquet"
    _require_file(parquet_path, hint="Run `pitedgar build` first to create it.")
    try:
        as_of_ts = pd.Timestamp(as_of)
    except ValueError as exc:
        raise click.UsageError(f"Invalid --as-of date {as_of!r}: {exc}") from exc
    q = PitQuery(parquet_path)
    result = q.as_of([normalize_ticker(ticker)], concept, as_of_ts)
    if fmt == "json":
        click.echo(result.to_json(orient="records", date_format="iso", indent=2))
    elif fmt == "csv":
        click.echo(result.to_csv(index=False))
    else:
        click.echo(result.to_string(index=False))
