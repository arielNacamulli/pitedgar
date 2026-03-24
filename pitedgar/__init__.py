"""pitedgar — Point-in-time SEC EDGAR financial data pipeline."""

__version__ = "0.1.0"

from pitedgar.config import PitEdgarConfig
from pitedgar.mapping import build_cik_map
from pitedgar.downloader import download_bulk
from pitedgar.parser import parse_all
from pitedgar.query import PitQuery

__all__ = [
    "PitEdgarConfig",
    "build_cik_map",
    "download_bulk",
    "parse_all",
    "PitQuery",
]
