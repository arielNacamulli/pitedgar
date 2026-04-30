"""Adversarial regression tests: one per category surfaced in the
adversarial review (issue #38). Run with `pytest -m adversarial`.

This module is the umbrella for cross-cutting regressions that do not
belong to any single parser/query/downloader unit test. Per-fix tests
still live in the respective module's test file; entries here are the
minimum-viable smoke tests that must survive any future refactor.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pitedgar.parser import parse_company
from pitedgar.query import PitQuery

pytestmark = pytest.mark.adversarial


def test_parser_handles_empty_us_gaap(tmp_path: Path) -> None:
    """Filer JSON with no us-gaap section must produce an empty DataFrame, not crash."""
    cik = "0000000001"
    (tmp_path / f"CIK{cik}.json").write_text(json.dumps({"facts": {}}), encoding="utf-8")
    df = parse_company(cik, ["us-gaap:Revenues"], tmp_path, ["10-K"])
    assert df.empty


def test_parser_handles_missing_json_file(tmp_path: Path) -> None:
    """Parser must not crash when the CIK JSON is absent."""
    df = parse_company("9999999999", ["us-gaap:Revenues"], tmp_path, ["10-K"])
    assert df.empty


def test_query_empty_parquet_degrades_gracefully(tmp_path: Path) -> None:
    """An empty (but schema-valid) parquet must load without error and return empty results."""
    path = tmp_path / "pit_financials.parquet"
    pd.DataFrame(
        {
            "ticker": pd.Series([], dtype="string"),
            "concept": pd.Series([], dtype="string"),
            "end": pd.Series([], dtype="datetime64[ns]"),
            "filed": pd.Series([], dtype="datetime64[ns]"),
            "val": pd.Series([], dtype="float64"),
            "form": pd.Series([], dtype="string"),
            "duration_days": pd.Series([], dtype="int64"),
        }
    ).to_parquet(path)
    q = PitQuery(path)
    # All queries on a missing ticker must produce NaN without crashing.
    result = q.as_of("FAKE", "us-gaap:Revenues", "2023-01-01")
    assert len(result) == 1
    assert pd.isna(result.iloc[0]["val"])
