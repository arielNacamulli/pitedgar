"""End-to-end pipeline integration tests.

Exercises the four pipeline stages (map → [stub fetch] → build → query) without
hitting SEC EDGAR. This catches cross-module regressions that per-module unit
tests miss: column-name drift between parser/query, facts_dir plumbing in the
config, ticker normalisation across stages, and parquet round-trips.
"""

import json

import pandas as pd
import pytest

from pitedgar.config import PitEdgarConfig
from pitedgar.mapping import build_cik_map
from pitedgar.parser import parse_all
from pitedgar.query import PitQuery

SAMPLE_FACTS = {
    "facts": {
        "us-gaap": {
            "Revenues": {
                "units": {
                    "USD": [
                        {
                            "start": "2022-01-01",
                            "end": "2022-03-31",
                            "filed": "2022-05-01",
                            "val": 100_000_000.0,
                            "form": "10-Q",
                            "accn": "A1",
                        },
                        {
                            "start": "2022-04-01",
                            "end": "2022-06-30",
                            "filed": "2022-08-01",
                            "val": 110_000_000.0,
                            "form": "10-Q",
                            "accn": "A2",
                        },
                        {
                            "start": "2022-07-01",
                            "end": "2022-09-30",
                            "filed": "2022-11-01",
                            "val": 120_000_000.0,
                            "form": "10-Q",
                            "accn": "A3",
                        },
                        {
                            "start": "2022-01-01",
                            "end": "2022-12-31",
                            "filed": "2023-02-01",
                            "val": 500_000_000.0,
                            "form": "10-K",
                            "accn": "A4",
                        },
                    ]
                }
            },
            "Assets": {
                "units": {
                    "USD": [
                        {
                            "end": "2022-12-31",
                            "filed": "2023-02-01",
                            "val": 2_000_000_000.0,
                            "form": "10-K",
                            "accn": "A4",
                        }
                    ]
                }
            },
        }
    }
}


@pytest.fixture
def pipeline_dirs(tmp_path):
    """Full data_dir layout: ticker_cik_map.parquet + companyfacts/CIK*.json."""
    data_dir = tmp_path / "data"
    facts_dir = data_dir / "companyfacts"
    facts_dir.mkdir(parents=True)

    cik = "0000320193"
    (facts_dir / f"CIK{cik}.json").write_text(json.dumps(SAMPLE_FACTS))

    cik_map = pd.DataFrame({"cik": [cik]}, index=pd.Index(["AAPL"], name="ticker"))
    cik_map.to_parquet(data_dir / "ticker_cik_map.parquet")

    return data_dir


def test_map_build_query_roundtrip(pipeline_dirs, mocker):
    """map → build → query on a tiny fabricated filings set."""
    config = PitEdgarConfig(edgar_identity="Test test@example.com", data_dir=pipeline_dirs)

    # map: stub edgar so we don't hit the network; re-resolve AAPL to verify
    # the mapping stage round-trips through the same parquet the parser reads.
    import pitedgar.mapping as mapping_mod

    mock_company = mocker.MagicMock()
    mock_company.cik = 320193
    mock_company.name = "Apple Inc."
    mock_company.sic = "3571"
    mock_company.fiscal_year_end = "09-28"
    mock_company.exchange = "NASDAQ"
    mocker.patch.object(mapping_mod.edgar, "set_identity")
    mocker.patch.object(mapping_mod.edgar, "Company", return_value=mock_company)
    mocker.patch.object(mapping_mod.time, "sleep")

    cik_map = build_cik_map(["AAPL"], config, force=True)
    assert cik_map.loc["AAPL", "cik"] == "0000320193"

    # build: parse local JSON into the master parquet.
    master = parse_all(config, cik_map, force=True)
    assert not master.empty
    assert "ticker" in master.columns
    assert "scale_corrected" in master.columns
    assert "duration_days" in master.columns
    assert (master["ticker"] == "AAPL").all()

    # Scale-correction must not fire: all USD values are >= $100M.
    assert not master["scale_corrected"].any()

    # query: PIT lookups via the same parquet.
    q = PitQuery(pipeline_dirs / "pit_financials.parquet")

    # Annual revenue known after 10-K filed (10-K filed 2023-02-01 → query 3 weeks later).
    rev = q.as_of("AAPL", "us-gaap:Revenues", "2023-02-20")
    assert len(rev) == 1
    assert rev["val"].iloc[0] == 500_000_000.0

    # Look-ahead guard: before 10-K filed, the 10-K row must not leak.
    rev_before_10k = q.as_of("AAPL", "us-gaap:Revenues", "2023-01-01", max_staleness_days=365)
    assert len(rev_before_10k) == 1
    # Should see the most recent pre-10-K filing (Q3 10-Q filed 2022-11-01).
    assert rev_before_10k["val"].iloc[0] == 120_000_000.0
    assert rev_before_10k["form"].iloc[0] == "10-Q"

    # Balance sheet snapshot.
    assets = q.as_of("AAPL", "us-gaap:Assets", "2023-02-20")
    assert assets["val"].iloc[0] == 2_000_000_000.0


def test_build_triggers_scale_correction_for_thousands_filer(tmp_path):
    """Filers reporting in thousands ($) must be rescaled 1000x and flagged when
    scale_correction='auto' and at least 2 distinct USD concepts are below threshold."""
    data_dir = tmp_path / "data"
    facts_dir = data_dir / "companyfacts"
    facts_dir.mkdir(parents=True)

    tiny_facts = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "start": "2022-01-01",
                                "end": "2022-12-31",
                                "filed": "2023-02-01",
                                # $500 reported — i.e. $500k, filer meant $500k but put 500.
                                "val": 500,
                                "form": "10-K",
                                "accn": "A1",
                            }
                        ]
                    }
                },
                # Second USD concept below threshold — needed for auto heuristic to fire.
                "Assets": {
                    "units": {
                        "USD": [
                            {
                                "end": "2022-12-31",
                                "filed": "2023-02-01",
                                "val": 800,
                                "form": "10-K",
                                "accn": "A2",
                            }
                        ]
                    }
                },
            }
        }
    }

    cik = "0000999999"
    (facts_dir / f"CIK{cik}.json").write_text(json.dumps(tiny_facts))
    cik_map = pd.DataFrame({"cik": [cik]}, index=pd.Index(["TINY"], name="ticker"))
    cik_map.to_parquet(data_dir / "ticker_cik_map.parquet")

    config = PitEdgarConfig(
        edgar_identity="Test test@example.com",
        data_dir=data_dir,
        scale_correction="auto",
    )
    master = parse_all(config, cik_map, force=True)

    assert master["scale_corrected"].all()
    rev_row = master[master["concept"] == "us-gaap:Revenues"].iloc[0]
    assert rev_row["val"] == 500 * 1000


def test_build_is_atomic_under_simulated_crash(pipeline_dirs, mocker):
    """If the parquet write fails mid-way, no corrupt file must remain."""
    config = PitEdgarConfig(edgar_identity="Test test@example.com", data_dir=pipeline_dirs)
    cik_map = pd.read_parquet(pipeline_dirs / "ticker_cik_map.parquet")

    out_path = pipeline_dirs / "pit_financials.parquet"

    # Sabotage pandas.DataFrame.to_parquet so it raises AFTER writing to the
    # temp sidecar — mirror a real disk-full / SIGTERM failure.
    real_to_parquet = pd.DataFrame.to_parquet

    def exploding_to_parquet(self, path, *args, **kwargs):
        # Actually write, then blow up — ensures the temp file exists first.
        real_to_parquet(self, path, *args, **kwargs)
        raise OSError("simulated disk-full")

    mocker.patch.object(pd.DataFrame, "to_parquet", exploding_to_parquet)

    with pytest.raises(OSError, match="simulated disk-full"):
        parse_all(config, cik_map, force=True)

    # Neither the final parquet nor the .tmp sidecar must survive.
    assert not out_path.exists()
    assert not out_path.with_suffix(out_path.suffix + ".tmp").exists()
