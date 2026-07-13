"""dei share-count extraction with quality flags (synthetic companyfacts)."""

from __future__ import annotations

import json

from pitedgar.dei import extract_dei_shares


def _facts(share_facts):
    return {"facts": {"dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": share_facts}}}}}


def _write(tmp_path, cik, payload):
    (tmp_path / f"CIK{cik.zfill(10)}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_extracts_and_flags(tmp_path):
    facts = [
        {"end": "2010-03-31", "val": 390_000_000, "form": "10-Q", "filed": "2010-04-20"},
        {"end": "2010-06-30", "val": 391_000_000, "form": "10-Q", "filed": "2010-07-20"},
        # the CSX-style garbage value
        {"end": "2011-03-31", "val": 1, "form": "10-Q", "filed": "2011-04-20"},
        {"end": "2011-06-30", "val": 392_000_000, "form": "10-Q", "filed": "2011-07-20"},
        # a BRK-style zero placeholder
        {"end": "2011-09-30", "val": 0, "form": "10-Q", "filed": "2011-10-20"},
    ]
    _write(tmp_path, "277948", _facts(facts))
    df = extract_dei_shares(tmp_path, {"0000277948": "CSX"})
    assert list(df["quality"]) == ["ok", "ok", "magnitude_outlier", "ok", "nonpositive"]
    assert (df["cticker"] == "CSX").all()
    assert df["cik"].iloc[0] == "0000277948"


def test_split_jump_is_not_an_outlier(tmp_path):
    facts = [
        {"end": "2020-03-31", "val": 100_000_000, "form": "10-Q", "filed": "2020-04-20"},
        # 4:1 split -> legitimate 4x jump in as-reported count
        {"end": "2020-09-30", "val": 400_000_000, "form": "10-Q", "filed": "2020-10-20"},
        {"end": "2021-03-31", "val": 401_000_000, "form": "10-Q", "filed": "2021-04-20"},
    ]
    _write(tmp_path, "320193", _facts(facts))
    df = extract_dei_shares(tmp_path, {"0000320193": "AAPL"})
    assert list(df["quality"]) == ["ok", "ok", "ok"]


def test_missing_file_and_missing_concept(tmp_path):
    _write(tmp_path, "111", {"facts": {"us-gaap": {}}})
    df = extract_dei_shares(tmp_path, {"0000000111": "NOP", "0000000222": "GONE"})
    assert len(df) == 0
