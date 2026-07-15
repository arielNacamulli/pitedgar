"""PIT SIC extraction tests — synthetic sub.txt content, no network."""

from __future__ import annotations

import datetime as dt
import io
import zipfile

import pytest

from pitedgar import sic


def sub_bytes(rows: list[tuple[str, str, str, str, str, str]]) -> bytes:
    """Serialize (adsh, cik, name, sic, form, filed) rows as a sub.txt."""
    header = "adsh\tcik\tname\tsic\tcountryba\tform\tfiled\taccepted\n"
    lines = [
        f"{adsh}\t{cik}\t{name}\t{s}\tUS\t{form}\t{filed}\t{filed} 12:00:00\n"
        for adsh, cik, name, s, form, filed in rows
    ]
    return (header + "".join(lines)).encode()


def slice_path(tmp_path, name, rows):
    df = sic.parse_sub_txt(sub_bytes(rows))
    p = tmp_path / f"sub_{name}.parquet"
    df.to_parquet(p, index=False)
    return p


class TestQuarters:
    def test_series_starts_2009q2_and_respects_lag(self):
        qs = sic.quarters_through(dt.date(2010, 3, 1))
        assert qs == ["2009q2", "2009q3", "2009q4"]  # 2009q4 ended 35+ days ago
        assert sic.quarters_through(dt.date(2010, 1, 20)) == ["2009q2", "2009q3"]

    def test_recent_today_includes_many_quarters(self):
        qs = sic.quarters_through(dt.date(2026, 7, 15))
        assert qs[0] == "2009q2"
        assert qs[-1] == "2026q1"
        assert len(qs) == 68


class TestParseSub:
    def test_extracts_needed_columns_only(self):
        df = sic.parse_sub_txt(sub_bytes([("a-1", "320193", "APPLE INC", "3571", "10-Q", "20150128")]))
        assert list(df.columns) == sic.SUB_COLUMNS
        assert df.iloc[0]["sic"] == "3571"

    def test_layout_change_fails_loud(self):
        with pytest.raises(sic.SicExtractionError, match="layout changed"):
            sic.parse_sub_txt(b"foo\tbar\n1\t2\n")


class TestBuildPitSic:
    def test_sic_change_lands_on_filed_date(self, tmp_path):
        p1 = slice_path(tmp_path, "2015q1", [("a-1", "816761", "TERADATA", "3571", "10-K", "20150227")])
        p2 = slice_path(tmp_path, "2018q1", [("a-2", "816761", "TERADATA", "7372", "10-K", "20180223")])
        df, report = sic.build_pit_sic([p1, p2])
        assert report["ciks_with_sic_changes"] == 1
        assert sic.sic_as_of(df, "816761", "2015-03-01") == 3571
        assert sic.sic_as_of(df, "816761", "2018-02-22") == 3571  # day before the new filing
        assert sic.sic_as_of(df, "816761", "2018-02-23") == 7372
        assert sic.sic_as_of(df, "816761", "2014-01-01") is None  # before first filing

    def test_stable_name_stays_flat(self, tmp_path):
        rows = [(f"a-{i}", "320193", "APPLE", "3571", "10-Q", f"201{i}0501") for i in range(5)]
        df, report = sic.build_pit_sic([slice_path(tmp_path, "x", rows)])
        assert report["ciks_with_sic_changes"] == 0
        assert df["sic"].unique().tolist() == [3571]

    def test_invalid_sic_dropped_and_counted(self, tmp_path):
        rows = [
            ("a-1", "1", "GOOD", "2911", "10-K", "20200228"),
            ("a-2", "2", "EMPTY", "", "10-K", "20200228"),
            ("a-3", "3", "JUNK", "notasic", "10-K", "20200228"),
        ]
        df, report = sic.build_pit_sic([slice_path(tmp_path, "x", rows)])
        assert report["dropped_missing_or_invalid_sic"] == 2
        assert df["cik"].tolist() == ["0000000001"]

    def test_same_day_conflict_latest_accession_wins(self, tmp_path):
        rows = [
            ("0001-20-000001", "9", "DUAL", "1000", "8-K", "20200228"),
            ("0001-20-000002", "9", "DUAL", "2000", "10-K", "20200228"),
        ]
        df, report = sic.build_pit_sic([slice_path(tmp_path, "x", rows)])
        assert report["same_day_conflicting_sic"] == 1
        assert sic.sic_as_of(df, "9", "2020-03-01") == 2000

    def test_dead_filer_keeps_last_sic(self, tmp_path):
        rows = [("a-1", "77", "DEAD CO", "6021", "10-K", "20120228")]
        df, _ = sic.build_pit_sic([slice_path(tmp_path, "x", rows)])
        assert sic.sic_as_of(df, "77", "2026-01-01") == 6021  # last known SIC persists

    def test_empty_input_fails_loud(self):
        with pytest.raises(sic.SicExtractionError, match="no FSDS"):
            sic.build_pit_sic([])


class TestCoverage:
    def test_missing_ciks_listed(self, tmp_path):
        df, _ = sic.build_pit_sic(
            [slice_path(tmp_path, "x", [("a-1", "1", "A", "2911", "10-K", "20200228")])]
        )
        rep = sic.coverage_report(df, ["1", "2"])
        assert rep["covered"] == 1
        assert rep["missing"] == 1
        assert rep["missing_ciks"] == ["0000000002"]


class TestZipPath:
    def test_fetch_quarter_reads_sub_from_zip(self, monkeypatch):
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w") as zf:
            zf.writestr("sub.txt", sub_bytes([("a-1", "1", "A", "2911", "10-K", "20200228")]))

        class FakeResp:
            status_code = 200
            content = payload.getvalue()

        monkeypatch.setattr(sic.requests, "get", lambda *a, **k: FakeResp())
        df = sic._fetch_quarter("2020q1", identity="Test test@example.com")
        assert df.iloc[0]["sic"] == "2911"
