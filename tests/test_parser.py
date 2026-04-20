"""Tests for parser.py"""

import json

import pandas as pd
import pytest

from pitedgar.config import PitEdgarConfig
from pitedgar.parser import parse_all, parse_company

SAMPLE_FACTS = {
    "facts": {
        "us-gaap": {
            "Revenues": {
                "units": {
                    "USD": [
                        {
                            "start": "2022-01-01",
                            "end": "2022-12-31",
                            "filed": "2023-02-01",
                            "val": 1000000,
                            "form": "10-K",
                            "accn": "A1",
                        },
                        {
                            "start": "2022-01-01",
                            "end": "2022-12-31",
                            "filed": "2023-03-15",
                            "val": 1050000,
                            "form": "10-K",
                            "accn": "A2",
                        },
                        # Q3 discreto (92 giorni) — deve essere preferito nel dedup
                        {
                            "start": "2022-07-01",
                            "end": "2022-09-30",
                            "filed": "2022-11-01",
                            "val": 300000,
                            "form": "10-Q",
                            "accn": "A3",
                        },
                        # Q3 YTD (272 giorni) — stesso end, deve essere scartato a favore del discreto
                        {
                            "start": "2022-01-01",
                            "end": "2022-09-30",
                            "filed": "2022-11-01",
                            "val": 900000,
                            "form": "10-Q",
                            "accn": "A4",
                        },
                    ]
                }
            },
            "EarningsPerShareBasic": {
                "units": {
                    "shares": [
                        {
                            "start": "2022-01-01",
                            "end": "2022-12-31",
                            "filed": "2023-02-01",
                            "val": 5.5,
                            "form": "10-K",
                            "accn": "B1",
                        },
                    ]
                }
            },
        }
    }
}


@pytest.fixture
def facts_dir(tmp_path):
    cik = "0000320193"
    json_path = tmp_path / f"CIK{cik}.json"
    json_path.write_text(json.dumps(SAMPLE_FACTS), encoding="utf-8")
    return tmp_path


def test_parse_company_returns_correct_columns(facts_dir):
    df = parse_company("0000320193", ["us-gaap:Revenues"], facts_dir, ["10-K", "10-Q"])
    assert set(df.columns) == {
        "cik",
        "concept",
        "start",
        "end",
        "duration_days",
        "filed",
        "val",
        "form",
        "accn",
        "scale_corrected",
    }


def test_parse_company_pit_deduplication(facts_dir):
    df = parse_company("0000320193", ["us-gaap:Revenues"], facts_dir, ["10-K", "10-Q"])
    revenues_annual = df[(df["concept"] == "us-gaap:Revenues") & (df["form"] == "10-K")]
    # Two entries for same end date but different filed dates — both preserved for PIT query layer.
    assert len(revenues_annual) == 2
    assert set(revenues_annual["accn"]) == {"A1", "A2"}


def test_parse_company_prefers_discrete_quarter_over_ytd(facts_dir):
    """Per lo stesso end date, il parser deve tenere il quarter discreto (90gg) e scartare il YTD."""
    df = parse_company("0000320193", ["us-gaap:Revenues"], facts_dir, ["10-Q"])
    q3 = df[df["end"] == "2022-09-30"]
    assert len(q3) == 1
    assert q3.iloc[0]["accn"] == "A3"  # discreto, non A4 (YTD)
    assert q3.iloc[0]["duration_days"] == 91


def test_parse_company_form_filter(facts_dir):
    df = parse_company("0000320193", ["us-gaap:Revenues"], facts_dir, ["10-K"])
    assert all(df["form"] == "10-K")


def test_parse_company_share_concept_uses_shares_units(facts_dir):
    df = parse_company("0000320193", ["us-gaap:EarningsPerShareBasic"], facts_dir, ["10-K"])
    assert len(df) == 1
    assert df.iloc[0]["val"] == pytest.approx(5.5)


def test_parse_company_concept_alias(tmp_path):
    """Companies that use the post-ASC 606 revenue tag should be stored under the canonical us-gaap:Revenues."""
    facts = {
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [
                            {
                                "start": "2023-01-01",
                                "end": "2023-12-31",
                                "filed": "2024-02-01",
                                "val": 500000000,
                                "form": "10-K",
                                "accn": "Z1",
                            },
                        ]
                    }
                }
            }
        }
    }
    cik = "0000000001"
    (tmp_path / f"CIK{cik}.json").write_text(json.dumps(facts), encoding="utf-8")
    df = parse_company(cik, ["us-gaap:Revenues"], tmp_path, ["10-K"])
    assert len(df) == 1
    assert df.iloc[0]["concept"] == "us-gaap:Revenues"
    assert df.iloc[0]["val"] == pytest.approx(500_000_000)


def test_parse_company_scale_correction(tmp_path):
    """Filers that report USD values in thousands (max < $1M) are auto-corrected by 1000x."""
    facts = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            # Values in thousands: 500_000 = $500M reported as $500
                            {
                                "start": "2022-01-01",
                                "end": "2022-12-31",
                                "filed": "2023-02-01",
                                "val": 500,
                                "form": "10-K",
                                "accn": "S1",
                            },
                        ]
                    }
                }
            }
        }
    }
    cik = "0000000002"
    (tmp_path / f"CIK{cik}.json").write_text(json.dumps(facts), encoding="utf-8")
    df = parse_company(cik, ["us-gaap:Revenues"], tmp_path, ["10-K"])
    assert df.iloc[0]["val"] == pytest.approx(500_000)


def test_parse_company_scale_correct_does_not_affect_shares(tmp_path):
    """Scale correction must not touch share/EPS concepts."""
    facts = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "start": "2022-01-01",
                                "end": "2022-12-31",
                                "filed": "2023-02-01",
                                "val": 400,
                                "form": "10-K",
                                "accn": "S2",
                            },
                        ]
                    }
                },
                "EarningsPerShareBasic": {
                    "units": {
                        "shares": [
                            {
                                "start": "2022-01-01",
                                "end": "2022-12-31",
                                "filed": "2023-02-01",
                                "val": 2.5,
                                "form": "10-K",
                                "accn": "S3",
                            },
                        ]
                    }
                },
            }
        }
    }
    cik = "0000000003"
    (tmp_path / f"CIK{cik}.json").write_text(json.dumps(facts), encoding="utf-8")
    df = parse_company(cik, ["us-gaap:Revenues", "us-gaap:EarningsPerShareBasic"], tmp_path, ["10-K"])
    eps_row = df[df["concept"] == "us-gaap:EarningsPerShareBasic"].iloc[0]
    assert eps_row["val"] == pytest.approx(2.5)  # unchanged


def test_parse_company_scale_no_correction_above_threshold(tmp_path):
    """Values with max >= $1M must NOT be multiplied — they are already in dollars."""
    facts = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "start": "2022-01-01",
                                "end": "2022-12-31",
                                "filed": "2023-02-01",
                                "val": 1_500_000,
                                "form": "10-K",
                                "accn": "T1",
                            },
                        ]
                    }
                }
            }
        }
    }
    cik = "0000000004"
    (tmp_path / f"CIK{cik}.json").write_text(json.dumps(facts), encoding="utf-8")
    df = parse_company(cik, ["us-gaap:Revenues"], tmp_path, ["10-K"])
    assert df.iloc[0]["val"] == pytest.approx(1_500_000)  # unchanged


def test_parse_company_canonical_wins_over_alias(tmp_path):
    """If a company reports both the canonical and alias tag, canonical takes priority
    and the result has exactly one row (no double-counting)."""
    facts = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "start": "2023-01-01",
                                "end": "2023-12-31",
                                "filed": "2024-02-01",
                                "val": 800_000_000,
                                "form": "10-K",
                                "accn": "C1",
                            },
                        ]
                    }
                },
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [
                            {
                                "start": "2023-01-01",
                                "end": "2023-12-31",
                                "filed": "2024-02-01",
                                "val": 900_000_000,
                                "form": "10-K",
                                "accn": "C2",
                            },
                        ]
                    }
                },
            }
        }
    }
    cik = "0000000005"
    (tmp_path / f"CIK{cik}.json").write_text(json.dumps(facts), encoding="utf-8")
    df = parse_company(cik, ["us-gaap:Revenues"], tmp_path, ["10-K"])
    assert len(df) == 1
    # Canonical (Revenues) is tried first, so its value wins
    assert df.iloc[0]["val"] == pytest.approx(800_000_000)


def test_parse_company_restatements_preserve_filed_dates(tmp_path):
    """Both the original and restated filings for the same period must be retained
    with their exact filed dates intact."""
    facts = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "start": "2022-01-01",
                                "end": "2022-12-31",
                                "filed": "2023-02-01",
                                "val": 1_000_000_000,
                                "form": "10-K",
                                "accn": "R1",
                            },
                            {
                                "start": "2022-01-01",
                                "end": "2022-12-31",
                                "filed": "2023-04-15",
                                "val": 1_050_000_000,
                                "form": "10-K",
                                "accn": "R2",
                            },
                        ]
                    }
                }
            }
        }
    }
    cik = "0000000006"
    (tmp_path / f"CIK{cik}.json").write_text(json.dumps(facts), encoding="utf-8")
    df = parse_company(cik, ["us-gaap:Revenues"], tmp_path, ["10-K"])
    assert len(df) == 2
    filed_dates = set(df["filed"].dt.strftime("%Y-%m-%d"))
    assert filed_dates == {"2023-02-01", "2023-04-15"}
    vals = set(df["val"])
    assert vals == {1_000_000_000.0, 1_050_000_000.0}


def test_parse_company_drops_unchanged_comparative_filings(tmp_path):
    """Later re-filings of the same unchanged value (comparative periods in
    subsequent 10-Ks) must be dropped to preserve PIT accuracy."""
    facts = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            # Original FY2022 10-K (filed Feb 2023)
                            {
                                "start": "2022-01-01",
                                "end": "2022-12-31",
                                "filed": "2023-02-01",
                                "val": 1_000_000_000,
                                "form": "10-K",
                                "accn": "P1",
                            },
                            # Same period/value re-filed 2 years later as comparative in FY2024 10-K
                            {
                                "start": "2022-01-01",
                                "end": "2022-12-31",
                                "filed": "2025-02-01",
                                "val": 1_000_000_000,
                                "form": "10-K",
                                "accn": "P2",
                            },
                        ]
                    }
                }
            }
        }
    }
    cik = "0000000007"
    (tmp_path / f"CIK{cik}.json").write_text(json.dumps(facts), encoding="utf-8")
    df = parse_company(cik, ["us-gaap:Revenues"], tmp_path, ["10-K"])
    # Only the earliest filing must survive — the later re-filing adds no new information
    assert len(df) == 1
    assert df.iloc[0]["filed"] == pd.Timestamp("2023-02-01")
    assert df.iloc[0]["accn"] == "P1"


def test_parse_company_restated_value_preserved_after_dedup(tmp_path):
    """A genuine restatement (changed value) must be kept even after cross-filing dedup."""
    facts = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "start": "2022-01-01",
                                "end": "2022-12-31",
                                "filed": "2023-02-01",
                                "val": 1_000_000_000,
                                "form": "10-K",
                                "accn": "R1",
                            },
                            {
                                "start": "2022-01-01",
                                "end": "2022-12-31",
                                "filed": "2023-04-01",
                                "val": 1_050_000_000,
                                "form": "10-K",
                                "accn": "R2",
                            },
                        ]
                    }
                }
            }
        }
    }
    cik = "0000000008"
    (tmp_path / f"CIK{cik}.json").write_text(json.dumps(facts), encoding="utf-8")
    df = parse_company(cik, ["us-gaap:Revenues"], tmp_path, ["10-K"])
    assert len(df) == 2
    assert set(df["accn"]) == {"R1", "R2"}


def test_parse_company_sales_revenue_net_alias(tmp_path):
    """A filer that uses only the deprecated SalesRevenueNet tag (no us-gaap:Revenues)
    must still produce rows under the canonical us-gaap:Revenues concept."""
    facts = {
        "facts": {
            "us-gaap": {
                "SalesRevenueNet": {
                    "units": {
                        "USD": [
                            {
                                "start": "2017-01-01",
                                "end": "2017-12-31",
                                "filed": "2018-02-15",
                                "val": 250_000_000,
                                "form": "10-K",
                                "accn": "SRN1",
                            },
                        ]
                    }
                }
            }
        }
    }
    cik = "0000000010"
    (tmp_path / f"CIK{cik}.json").write_text(json.dumps(facts), encoding="utf-8")
    df = parse_company(cik, ["us-gaap:Revenues"], tmp_path, ["10-K"])
    assert len(df) == 1
    assert df.iloc[0]["concept"] == "us-gaap:Revenues"
    assert df.iloc[0]["val"] == pytest.approx(250_000_000)


def test_parse_company_canonical_revenues_wins_over_sales_revenue_net(tmp_path):
    """When both us-gaap:Revenues and the deprecated SalesRevenueNet are present,
    the canonical Revenues value must win (no double-counting)."""
    facts = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "start": "2017-01-01",
                                "end": "2017-12-31",
                                "filed": "2018-02-15",
                                "val": 700_000_000,
                                "form": "10-K",
                                "accn": "REV1",
                            },
                        ]
                    }
                },
                "SalesRevenueNet": {
                    "units": {
                        "USD": [
                            {
                                "start": "2017-01-01",
                                "end": "2017-12-31",
                                "filed": "2018-02-15",
                                "val": 250_000_000,
                                "form": "10-K",
                                "accn": "SRN1",
                            },
                        ]
                    }
                },
            }
        }
    }
    cik = "0000000011"
    (tmp_path / f"CIK{cik}.json").write_text(json.dumps(facts), encoding="utf-8")
    df = parse_company(cik, ["us-gaap:Revenues"], tmp_path, ["10-K"])
    assert len(df) == 1
    assert df.iloc[0]["concept"] == "us-gaap:Revenues"
    # Canonical (Revenues) is tried first, so its value wins (no double-counting)
    assert df.iloc[0]["val"] == pytest.approx(700_000_000)


def test_parse_company_missing_file(tmp_path):
    df = parse_company("9999999999", ["us-gaap:Revenues"], tmp_path, ["10-K"])
    assert df.empty


def test_parse_all(tmp_path):
    # Write fact for one company
    cik = "0000320193"
    facts_dir = tmp_path / "companyfacts"
    facts_dir.mkdir()
    (facts_dir / f"CIK{cik}.json").write_text(json.dumps(SAMPLE_FACTS), encoding="utf-8")

    config = PitEdgarConfig(
        edgar_identity="Test test@example.com",
        data_dir=tmp_path,
        facts_dir=facts_dir,
    )
    cik_map = pd.DataFrame({"cik": [cik]}, index=pd.Index(["AAPL"], name="ticker"))
    master = parse_all(config, cik_map)

    assert "ticker" in master.columns
    assert set(master["ticker"]) == {"AAPL"}
    assert (tmp_path / "pit_financials.parquet").exists()


def test_parse_all_skips_when_parquet_exists(tmp_path):
    """parse_all must return cached parquet without running the parse loop."""
    cik = "0000320193"
    facts_dir = tmp_path / "companyfacts"
    facts_dir.mkdir()
    (facts_dir / f"CIK{cik}.json").write_text(json.dumps(SAMPLE_FACTS), encoding="utf-8")

    config = PitEdgarConfig(
        edgar_identity="Test test@example.com",
        data_dir=tmp_path,
        facts_dir=facts_dir,
    )
    cik_map = pd.DataFrame({"cik": [cik]}, index=pd.Index(["AAPL"], name="ticker"))

    # First call: produces the parquet.
    first = parse_all(config, cik_map)

    # Second call (force=False): must return the cached file, not re-parse.
    # We verify by patching parse_company — it must NOT be called.
    from unittest.mock import patch

    with patch("pitedgar.parser.parse_company") as mock_pc:
        second = parse_all(config, cik_map, force=False)
        mock_pc.assert_not_called()

    assert list(second.columns) == list(first.columns)
    assert len(second) == len(first)


def test_parse_all_force_reparses(tmp_path):
    """force=True must re-run the parse loop even if parquet exists."""
    cik = "0000320193"
    facts_dir = tmp_path / "companyfacts"
    facts_dir.mkdir()
    (facts_dir / f"CIK{cik}.json").write_text(json.dumps(SAMPLE_FACTS), encoding="utf-8")

    config = PitEdgarConfig(
        edgar_identity="Test test@example.com",
        data_dir=tmp_path,
        facts_dir=facts_dir,
    )
    cik_map = pd.DataFrame({"cik": [cik]}, index=pd.Index(["AAPL"], name="ticker"))

    # Create the parquet first.
    parse_all(config, cik_map)

    from unittest.mock import patch

    import pitedgar.parser as parser_mod

    with patch.object(parser_mod, "parse_company", wraps=parser_mod.parse_company) as mock_pc:
        parse_all(config, cik_map, force=True)
        assert mock_pc.call_count >= 1


def test_parse_company_keeps_10ka_alongside_10k(tmp_path):
    """A 10-K/A amendment with a restated value for the same period_end as the
    original 10-K must be retained in the parquet so the query layer can pick
    the latest-filed value. This is the core invariant that makes including
    amendments in DEFAULT_FORMS safe — the parser does not collapse them."""
    facts = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            # Original 10-K
                            {
                                "start": "2022-01-01",
                                "end": "2022-12-31",
                                "filed": "2023-02-01",
                                "val": 1_000_000_000,
                                "form": "10-K",
                                "accn": "ORIG",
                            },
                            # 10-K/A amendment with a restated value for the same period
                            {
                                "start": "2022-01-01",
                                "end": "2022-12-31",
                                "filed": "2023-09-15",
                                "val": 1_120_000_000,
                                "form": "10-K/A",
                                "accn": "AMEND",
                            },
                        ]
                    }
                }
            }
        }
    }
    cik = "0000000099"
    (tmp_path / f"CIK{cik}.json").write_text(json.dumps(facts), encoding="utf-8")
    df = parse_company(cik, ["us-gaap:Revenues"], tmp_path, ["10-K", "10-K/A"])
    assert len(df) == 2
    assert set(df["form"]) == {"10-K", "10-K/A"}
    assert set(df["accn"]) == {"ORIG", "AMEND"}
    assert set(df["val"]) == {1_000_000_000.0, 1_120_000_000.0}
