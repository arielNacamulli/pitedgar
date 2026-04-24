"""Tests for parser.py"""

import json

import pandas as pd
import pytest

from pitedgar.config import DEFAULT_CONCEPTS, PitEdgarConfig
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
    """Filers that report USD values in thousands (max < $1M) are auto-corrected by 1000x
    when scale_correction='auto' and at least 2 distinct USD concepts are below threshold."""
    facts = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            # Values in thousands: 500 reported → $500k after correction
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
                },
                # Second USD concept below threshold — required for auto heuristic to fire.
                "Assets": {
                    "units": {
                        "USD": [
                            {
                                "end": "2022-12-31",
                                "filed": "2023-02-01",
                                "val": 800,
                                "form": "10-K",
                                "accn": "S1A",
                            },
                        ]
                    }
                },
            }
        }
    }
    cik = "0000000002"
    (tmp_path / f"CIK{cik}.json").write_text(json.dumps(facts), encoding="utf-8")
    df = parse_company(
        cik,
        ["us-gaap:Revenues", "us-gaap:Assets"],
        tmp_path,
        ["10-K"],
        scale_correction="auto",
    )
    rev_row = df[df["concept"] == "us-gaap:Revenues"].iloc[0]
    assert rev_row["val"] == pytest.approx(500_000)


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


def test_parse_company_unions_canonical_and_alias_across_periods(tmp_path):
    """A filer may report disjoint periods under canonical and alias tags (e.g.
    pre-ASC 606 years under us-gaap:Revenues and post-ASC 606 years under
    us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax). The parser
    must union all rows — not stop at the first tag that has data."""
    facts = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            # Period A — only reported under canonical
                            {
                                "start": "2016-01-01",
                                "end": "2016-12-31",
                                "filed": "2017-02-15",
                                "val": 200_000_000,
                                "form": "10-K",
                                "accn": "OLD1",
                            },
                        ]
                    }
                },
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [
                            # Periods B, C, D — only reported under the post-ASC 606 alias
                            {
                                "start": "2018-01-01",
                                "end": "2018-12-31",
                                "filed": "2019-02-15",
                                "val": 300_000_000,
                                "form": "10-K",
                                "accn": "NEW1",
                            },
                            {
                                "start": "2019-01-01",
                                "end": "2019-12-31",
                                "filed": "2020-02-15",
                                "val": 400_000_000,
                                "form": "10-K",
                                "accn": "NEW2",
                            },
                            {
                                "start": "2020-01-01",
                                "end": "2020-12-31",
                                "filed": "2021-02-15",
                                "val": 500_000_000,
                                "form": "10-K",
                                "accn": "NEW3",
                            },
                        ]
                    }
                },
            }
        }
    }
    cik = "0000000020"
    (tmp_path / f"CIK{cik}.json").write_text(json.dumps(facts), encoding="utf-8")
    df = parse_company(cik, ["us-gaap:Revenues"], tmp_path, ["10-K"])
    # 1 canonical + 3 alias = 4 rows, all stored under the canonical concept name.
    assert len(df) == 4
    assert set(df["concept"]) == {"us-gaap:Revenues"}
    assert set(df["accn"]) == {"OLD1", "NEW1", "NEW2", "NEW3"}
    assert set(df["val"]) == {200_000_000.0, 300_000_000.0, 400_000_000.0, 500_000_000.0}


def test_parse_company_canonical_wins_over_alias_same_period_different_value(tmp_path):
    """When canonical and alias both report the same (end, filed, form), the
    canonical value must win — no double-counting, canonical takes precedence."""
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
                                "val": 1_000_000_000,
                                "form": "10-K",
                                "accn": "CANON",
                            },
                        ]
                    }
                },
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [
                            # Same (end, filed, form) as the canonical row, different value
                            {
                                "start": "2023-01-01",
                                "end": "2023-12-31",
                                "filed": "2024-02-01",
                                "val": 1_200_000_000,
                                "form": "10-K",
                                "accn": "ALIAS",
                            },
                        ]
                    }
                },
            }
        }
    }
    cik = "0000000021"
    (tmp_path / f"CIK{cik}.json").write_text(json.dumps(facts), encoding="utf-8")
    df = parse_company(cik, ["us-gaap:Revenues"], tmp_path, ["10-K"])
    assert len(df) == 1
    assert df.iloc[0]["val"] == pytest.approx(1_000_000_000)
    assert df.iloc[0]["accn"] == "CANON"


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


def test_parse_company_none_concepts_extracts_non_default_concept(tmp_path):
    """concepts=None must pull in tags that are NOT part of the curated DEFAULT_CONCEPTS list."""
    obscure_concept = "us-gaap:GoodwillImpairmentLoss"
    assert obscure_concept not in DEFAULT_CONCEPTS  # guard against future curation changes
    facts = {
        "facts": {
            "us-gaap": {
                "GoodwillImpairmentLoss": {
                    "units": {
                        "USD": [
                            {
                                "start": "2022-01-01",
                                "end": "2022-12-31",
                                "filed": "2023-02-01",
                                "val": 25_000_000,
                                "form": "10-K",
                                "accn": "G1",
                            },
                        ]
                    }
                },
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "start": "2022-01-01",
                                "end": "2022-12-31",
                                "filed": "2023-02-01",
                                "val": 800_000_000,
                                "form": "10-K",
                                "accn": "R1",
                            },
                        ]
                    }
                },
            }
        }
    }
    cik = "0000000100"
    (tmp_path / f"CIK{cik}.json").write_text(json.dumps(facts), encoding="utf-8")

    df = parse_company(cik, None, tmp_path, ["10-K"])
    assert obscure_concept in set(df["concept"])
    assert "us-gaap:Revenues" in set(df["concept"])
    goodwill_row = df[df["concept"] == obscure_concept].iloc[0]
    assert goodwill_row["val"] == pytest.approx(25_000_000)


def test_parse_company_empty_concepts_behaves_like_none(tmp_path):
    """An empty concepts list must trigger the same "parse all" code path as None."""
    facts = {
        "facts": {
            "us-gaap": {
                "AccruedLiabilitiesCurrent": {
                    "units": {
                        "USD": [
                            {
                                "start": "2022-01-01",
                                "end": "2022-12-31",
                                "filed": "2023-02-01",
                                "val": 50_000_000,
                                "form": "10-K",
                                "accn": "AL1",
                            },
                        ]
                    }
                },
            }
        }
    }
    cik = "0000000101"
    (tmp_path / f"CIK{cik}.json").write_text(json.dumps(facts), encoding="utf-8")

    df = parse_company(cik, [], tmp_path, ["10-K"])
    assert "us-gaap:AccruedLiabilitiesCurrent" in set(df["concept"])


def test_parse_company_none_canonicalizes_aliases(tmp_path):
    """When parsing all concepts, alias tags must still be stored under the canonical name."""
    facts = {
        "facts": {
            "us-gaap": {
                # Alias-only filer (post-ASC 606 revenue tag, no canonical Revenues)
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [
                            {
                                "start": "2023-01-01",
                                "end": "2023-12-31",
                                "filed": "2024-02-01",
                                "val": 600_000_000,
                                "form": "10-K",
                                "accn": "AL1",
                            },
                        ]
                    }
                },
            }
        }
    }
    cik = "0000000102"
    (tmp_path / f"CIK{cik}.json").write_text(json.dumps(facts), encoding="utf-8")

    df = parse_company(cik, None, tmp_path, ["10-K"])
    # Concept must appear as the canonical name, not the alias
    assert set(df["concept"]) == {"us-gaap:Revenues"}
    assert df.iloc[0]["val"] == pytest.approx(600_000_000)


def test_parse_company_none_skips_unknown_non_usd_non_shares(tmp_path):
    """Concepts reported only in foreign/exotic units must be silently skipped, not crash."""
    facts = {
        "facts": {
            "us-gaap": {
                "SomeForeignCurrencyConcept": {
                    "units": {
                        "EUR": [
                            {
                                "start": "2022-01-01",
                                "end": "2022-12-31",
                                "filed": "2023-02-01",
                                "val": 1_000_000,
                                "form": "10-K",
                                "accn": "F1",
                            },
                        ]
                    }
                },
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "start": "2022-01-01",
                                "end": "2022-12-31",
                                "filed": "2023-02-01",
                                "val": 800_000_000,
                                "form": "10-K",
                                "accn": "R1",
                            },
                        ]
                    }
                },
            }
        }
    }
    cik = "0000000103"
    (tmp_path / f"CIK{cik}.json").write_text(json.dumps(facts), encoding="utf-8")

    df = parse_company(cik, None, tmp_path, ["10-K"])
    # The EUR-only concept must NOT appear; Revenues must.
    assert "us-gaap:SomeForeignCurrencyConcept" not in set(df["concept"])
    assert "us-gaap:Revenues" in set(df["concept"])


def test_parse_all_default_config_parses_all_concepts(tmp_path):
    """A PitEdgarConfig with no explicit concepts should extract every us-gaap tag."""
    cik = "0000320193"
    facts_dir = tmp_path / "companyfacts"
    facts_dir.mkdir()
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
                                "accn": "X1",
                            },
                        ]
                    }
                },
                "GoodwillImpairmentLoss": {
                    "units": {
                        "USD": [
                            {
                                "start": "2022-01-01",
                                "end": "2022-12-31",
                                "filed": "2023-02-01",
                                "val": 25_000_000,
                                "form": "10-K",
                                "accn": "X2",
                            },
                        ]
                    }
                },
            }
        }
    }
    (facts_dir / f"CIK{cik}.json").write_text(json.dumps(facts), encoding="utf-8")

    config = PitEdgarConfig(
        edgar_identity="Test test@example.com",
        data_dir=tmp_path,
        facts_dir=facts_dir,
    )
    # Sanity check: the new default is "parse everything"
    assert config.concepts is None

    master = parse_all(config, pd.DataFrame({"cik": [cik]}, index=pd.Index(["AAPL"], name="ticker")))
    assert "us-gaap:GoodwillImpairmentLoss" in set(master["concept"])
    assert "us-gaap:Revenues" in set(master["concept"])


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

    # Create the parquet first. Use n_workers=1 to keep mock-based assertions simple.
    parse_all(config, cik_map, n_workers=1)

    from unittest.mock import patch

    import pitedgar.parser as parser_mod

    with patch.object(parser_mod, "parse_company", wraps=parser_mod.parse_company) as mock_pc:
        parse_all(config, cik_map, force=True, n_workers=1)
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


def _write_multi_company_facts(facts_dir):
    """Create a small multi-company facts directory for parallel-equivalence tests."""
    companies = {
        "0000000101": ("AAA", 100_000_000, "M1"),
        "0000000102": ("BBB", 250_000_000, "M2"),
        "0000000103": ("CCC", 500_000_000, "M3"),
        "0000000104": ("DDD", 750_000_000, "M4"),
    }
    for cik, (_ticker, val, accn) in companies.items():
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
                                    "val": val,
                                    "form": "10-K",
                                    "accn": accn,
                                },
                            ]
                        }
                    }
                }
            }
        }
        (facts_dir / f"CIK{cik}.json").write_text(json.dumps(facts), encoding="utf-8")
    cik_map = pd.DataFrame(
        {"cik": list(companies)},
        index=pd.Index([t for _c, (t, _v, _a) in companies.items()], name="ticker"),
    )
    return cik_map


@pytest.mark.parametrize("n_workers", [1, 2])
def test_parse_all_parallel_equivalence(tmp_path, n_workers):
    """parse_all with n_workers > 1 must produce identical output to the serial path."""
    facts_dir = tmp_path / "companyfacts"
    facts_dir.mkdir()
    cik_map = _write_multi_company_facts(facts_dir)

    # Each call uses its own data_dir so the cached parquet doesn't short-circuit.
    serial_dir = tmp_path / "serial"
    serial_dir.mkdir()
    parallel_dir = tmp_path / "parallel"
    parallel_dir.mkdir()

    serial_cfg = PitEdgarConfig(
        edgar_identity="Test test@example.com",
        data_dir=serial_dir,
        facts_dir=facts_dir,
    )
    parallel_cfg = PitEdgarConfig(
        edgar_identity="Test test@example.com",
        data_dir=parallel_dir,
        facts_dir=facts_dir,
    )

    serial = parse_all(serial_cfg, cik_map, n_workers=1)
    parallel = parse_all(parallel_cfg, cik_map, n_workers=n_workers)

    # Order of returned rows is not guaranteed across workers — sort before comparing.
    sort_cols = ["ticker", "concept", "end", "filed"]
    serial_sorted = serial.sort_values(sort_cols).reset_index(drop=True)
    parallel_sorted = parallel.sort_values(sort_cols).reset_index(drop=True)

    pd.testing.assert_frame_equal(serial_sorted, parallel_sorted)
    assert set(parallel_sorted["ticker"]) == {"AAA", "BBB", "CCC", "DDD"}


def test_parse_all_n_workers_default_uses_cpu_count(tmp_path):
    """n_workers=None should default to os.cpu_count() (or 1) — verify it runs without error."""
    facts_dir = tmp_path / "companyfacts"
    facts_dir.mkdir()
    cik_map = _write_multi_company_facts(facts_dir)

    config = PitEdgarConfig(
        edgar_identity="Test test@example.com",
        data_dir=tmp_path,
        facts_dir=facts_dir,
    )
    master = parse_all(config, cik_map)  # n_workers=None
    assert set(master["ticker"]) == {"AAA", "BBB", "CCC", "DDD"}


def test_parse_all_invalid_n_workers(tmp_path):
    """n_workers < 1 must raise ValueError."""
    facts_dir = tmp_path / "companyfacts"
    facts_dir.mkdir()
    cik_map = _write_multi_company_facts(facts_dir)

    config = PitEdgarConfig(
        edgar_identity="Test test@example.com",
        data_dir=tmp_path,
        facts_dir=facts_dir,
    )
    with pytest.raises(ValueError, match="n_workers"):
        parse_all(config, cik_map, n_workers=0)


# ---------------------------------------------------------------------------
# Scale correction mode tests (Issue #12)
# ---------------------------------------------------------------------------

def _microcap_facts_single_concept():
    """A micro-cap filer with $500k revenue — one USD concept below $1M threshold."""
    return {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "start": "2022-01-01",
                                "end": "2022-12-31",
                                "filed": "2023-02-01",
                                "val": 500_000,
                                "form": "10-K",
                                "accn": "MC1",
                            },
                        ]
                    }
                }
            }
        }
    }


def _microcap_facts_two_concepts():
    """A micro-cap filer with 2 USD concepts both below $1M threshold."""
    return {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "start": "2022-01-01",
                                "end": "2022-12-31",
                                "filed": "2023-02-01",
                                "val": 500_000,
                                "form": "10-K",
                                "accn": "MC2",
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
                                "val": 750_000,
                                "form": "10-K",
                                "accn": "MC3",
                            },
                        ]
                    }
                },
            }
        }
    }


def test_microcap_not_corrected_by_default(tmp_path):
    """Micro-cap with $500k revenue must NOT be corrected when scale_correction='off'
    (the default), protecting legitimate small companies from silent corruption."""
    cik = "0000111001"
    (tmp_path / f"CIK{cik}.json").write_text(
        json.dumps(_microcap_facts_single_concept()), encoding="utf-8"
    )
    df = parse_company(cik, ["us-gaap:Revenues"], tmp_path, ["10-K"])
    # Default is "off" — value must be untouched.
    assert df.iloc[0]["val"] == pytest.approx(500_000)
    assert not df.iloc[0]["scale_corrected"]


def test_microcap_auto_single_concept_not_corrected(tmp_path):
    """With scale_correction='auto', a micro-cap with only ONE USD concept below
    the threshold must NOT be corrected — the heuristic requires at least two."""
    cik = "0000111002"
    (tmp_path / f"CIK{cik}.json").write_text(
        json.dumps(_microcap_facts_single_concept()), encoding="utf-8"
    )
    df = parse_company(
        cik, ["us-gaap:Revenues"], tmp_path, ["10-K"], scale_correction="auto"
    )
    assert df.iloc[0]["val"] == pytest.approx(500_000)
    assert not df.iloc[0]["scale_corrected"]


def test_auto_two_concepts_below_threshold_corrects_and_warns(tmp_path, capsys):
    """With scale_correction='auto' AND 2+ concepts below threshold, correction fires
    and a visible warning is emitted (not just a debug message)."""
    from loguru import logger

    cik = "0000111003"
    (tmp_path / f"CIK{cik}.json").write_text(
        json.dumps(_microcap_facts_two_concepts()), encoding="utf-8"
    )

    # Capture loguru output by adding a temporary sink.
    warning_messages: list[str] = []

    def _sink(message):
        if message.record["level"].no >= 30:  # WARNING = 30
            warning_messages.append(str(message))

    sink_id = logger.add(_sink, level="WARNING")
    try:
        df = parse_company(
            cik,
            ["us-gaap:Revenues", "us-gaap:Assets"],
            tmp_path,
            ["10-K"],
            scale_correction="auto",
        )
    finally:
        logger.remove(sink_id)

    rev_row = df[df["concept"] == "us-gaap:Revenues"].iloc[0]
    assert rev_row["val"] == pytest.approx(500_000 * 1000)
    assert rev_row["scale_corrected"]
    # A WARNING-level log entry must mention the CIK.
    assert any(cik in msg for msg in warning_messages), (
        f"Expected a WARNING mentioning CIK {cik}; got: {warning_messages}"
    )


def test_scale_correction_force_always_multiplies(tmp_path):
    """With scale_correction='force', USD values are always multiplied ×1000
    regardless of their magnitude."""
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
                                "val": 5_000_000,  # already well above $1M threshold
                                "form": "10-K",
                                "accn": "F1",
                            },
                        ]
                    }
                }
            }
        }
    }
    cik = "0000111004"
    (tmp_path / f"CIK{cik}.json").write_text(json.dumps(facts), encoding="utf-8")
    df = parse_company(
        cik, ["us-gaap:Revenues"], tmp_path, ["10-K"], scale_correction="force"
    )
    assert df.iloc[0]["val"] == pytest.approx(5_000_000 * 1000)
    assert df.iloc[0]["scale_corrected"]


def test_scale_correction_off_never_multiplies(tmp_path):
    """With scale_correction='off', values are never multiplied even if they look
    like thousands-reporting (only 1 concept, tiny value)."""
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
                                "val": 250,
                                "form": "10-K",
                                "accn": "O1",
                            },
                        ]
                    }
                }
            }
        }
    }
    cik = "0000111005"
    (tmp_path / f"CIK{cik}.json").write_text(json.dumps(facts), encoding="utf-8")
    df = parse_company(
        cik, ["us-gaap:Revenues"], tmp_path, ["10-K"], scale_correction="off"
    )
    assert df.iloc[0]["val"] == pytest.approx(250)
    assert not df.iloc[0]["scale_corrected"]


# ---------------------------------------------------------------------------
# Alias priority tests (issue #25)
# ---------------------------------------------------------------------------

def test_alias_precedence_follows_priority_list_order(tmp_path):
    """When two aliases collide on (end, filed, form), the alias listed earlier in
    CONCEPT_ALIAS_PRIORITY wins — independent of dict insertion order."""
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
                                "val": 111_000_000,
                                "form": "10-K",
                                "accn": "EXCL",
                            },
                        ]
                    }
                },
                "SalesRevenueNet": {
                    "units": {
                        "USD": [
                            {
                                "start": "2023-01-01",
                                "end": "2023-12-31",
                                "filed": "2024-02-01",
                                "val": 999_000_000,
                                "form": "10-K",
                                "accn": "SRN",
                            },
                        ]
                    }
                },
            }
        }
    }
    cik = "0000000200"
    (tmp_path / f"CIK{cik}.json").write_text(json.dumps(facts), encoding="utf-8")
    df = parse_company(cik, ["us-gaap:Revenues"], tmp_path, ["10-K"])
    assert len(df) == 1
    assert df.iloc[0]["val"] == pytest.approx(111_000_000)
    assert df.iloc[0]["accn"] == "EXCL"


def test_reversing_priority_changes_winner(tmp_path, monkeypatch):
    """Monkeypatching CONCEPT_ALIAS_PRIORITY to reverse order must flip the winner."""
    import pitedgar.config as config_mod
    import pitedgar.parser as parser_mod

    reversed_priority = {
        "us-gaap:Revenues": list(
            reversed(config_mod.CONCEPT_ALIAS_PRIORITY["us-gaap:Revenues"])
        ),
    }
    monkeypatch.setattr(config_mod, "CONCEPT_ALIAS_PRIORITY", reversed_priority)
    monkeypatch.setattr(parser_mod, "CONCEPT_ALIAS_PRIORITY", reversed_priority)

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
                                "val": 111_000_000,
                                "form": "10-K",
                                "accn": "EXCL",
                            },
                        ]
                    }
                },
                "SalesRevenueNet": {
                    "units": {
                        "USD": [
                            {
                                "start": "2023-01-01",
                                "end": "2023-12-31",
                                "filed": "2024-02-01",
                                "val": 999_000_000,
                                "form": "10-K",
                                "accn": "SRN",
                            },
                        ]
                    }
                },
            }
        }
    }
    cik = "0000000201"
    (tmp_path / f"CIK{cik}.json").write_text(json.dumps(facts), encoding="utf-8")
    df = parse_company(cik, ["us-gaap:Revenues"], tmp_path, ["10-K"])
    assert len(df) == 1
    assert df.iloc[0]["val"] == pytest.approx(999_000_000)
    assert df.iloc[0]["accn"] == "SRN"


# ---------------------------------------------------------------------------
# is_scale_corrected helper tests (issue #31)
# ---------------------------------------------------------------------------

def test_is_scale_corrected_returns_boolean_series():
    """Mixed scale_corrected column returns correct True/False mask."""
    from pitedgar.parser import is_scale_corrected

    df = pd.DataFrame({
        "val": [1_000_000, 2_000_000, 3_000_000],
        "scale_corrected": [True, False, True],
    })
    result = is_scale_corrected(df)

    assert isinstance(result, pd.Series)
    assert result.dtype == bool
    assert list(result) == [True, False, True]
    assert result.index.equals(df.index)


def test_is_scale_corrected_missing_column_returns_false_series():
    """Legacy parquet without scale_corrected column yields an all-False series."""
    from pitedgar.parser import is_scale_corrected

    df = pd.DataFrame({
        "val": [1_000_000, 2_000_000],
        "concept": ["us-gaap:Revenues", "us-gaap:Assets"],
    })
    result = is_scale_corrected(df)

    assert isinstance(result, pd.Series)
    assert result.dtype == bool
    assert not result.any()
    assert result.index.equals(df.index)


def test_is_scale_corrected_exported_from_package():
    """is_scale_corrected must be importable from both pitedgar.parser and pitedgar."""
    from pitedgar.parser import is_scale_corrected as from_parser
    import pitedgar

    assert hasattr(pitedgar, "is_scale_corrected")
    assert pitedgar.is_scale_corrected is from_parser


# ---------------------------------------------------------------------------
# Float-tolerance dedup tests (issue #15)
# ---------------------------------------------------------------------------

def test_dedup_tolerates_float_representation_drift(tmp_path):
    """Same revenue $1B serialized with float drift must dedupe."""
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
                                "val": 1_000_000_000.0,
                                "form": "10-K",
                                "accn": "F1",
                            },
                            {
                                "start": "2022-01-01",
                                "end": "2022-12-31",
                                "filed": "2024-02-01",
                                "val": 1_000_000_000.0 + 1e-7,
                                "form": "10-K",
                                "accn": "F2",
                            },
                        ]
                    }
                }
            }
        }
    }
    cik = "0001000001"
    (tmp_path / f"CIK{cik}.json").write_text(json.dumps(facts), encoding="utf-8")
    df = parse_company(cik, ["us-gaap:Revenues"], tmp_path, ["10-K"])
    assert len(df) == 1
    assert df.iloc[0]["accn"] == "F1"


def test_dedup_preserves_cent_level_restatement(tmp_path):
    """Revenue $1B restated by exactly $1 (well above 0.005 USD atol) must produce 2 rows."""
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
                                "val": 1_000_000_000.00,
                                "form": "10-K",
                                "accn": "CR1",
                            },
                            {
                                "start": "2022-01-01",
                                "end": "2022-12-31",
                                "filed": "2023-04-01",
                                "val": 1_000_000_001.00,
                                "form": "10-K",
                                "accn": "CR2",
                            },
                        ]
                    }
                }
            }
        }
    }
    cik = "0001000002"
    (tmp_path / f"CIK{cik}.json").write_text(json.dumps(facts), encoding="utf-8")
    df = parse_company(cik, ["us-gaap:Revenues"], tmp_path, ["10-K"])
    assert len(df) == 2
    assert set(df["accn"]) == {"CR1", "CR2"}


def test_dedup_preserves_small_restatement_proportional(tmp_path):
    """Revenue $1B restated to $1.01B (1% change) must produce 2 rows."""
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
                                "accn": "SR1",
                            },
                            {
                                "start": "2022-01-01",
                                "end": "2022-12-31",
                                "filed": "2023-04-01",
                                "val": 1_010_000_000,
                                "form": "10-K",
                                "accn": "SR2",
                            },
                        ]
                    }
                }
            }
        }
    }
    cik = "0001000003"
    (tmp_path / f"CIK{cik}.json").write_text(json.dumps(facts), encoding="utf-8")
    df = parse_company(cik, ["us-gaap:Revenues"], tmp_path, ["10-K"])
    assert len(df) == 2
    assert set(df["accn"]) == {"SR1", "SR2"}


def test_dedup_share_concept_tolerates_float_drift(tmp_path):
    """EPS 5.50 and 5.499999999 must dedupe under tight share tolerance."""
    facts = {
        "facts": {
            "us-gaap": {
                "EarningsPerShareBasic": {
                    "units": {
                        "shares": [
                            {
                                "start": "2022-01-01",
                                "end": "2022-12-31",
                                "filed": "2023-02-01",
                                "val": 5.5,
                                "form": "10-K",
                                "accn": "EP1",
                            },
                            {
                                "start": "2022-01-01",
                                "end": "2022-12-31",
                                "filed": "2024-02-01",
                                "val": 5.499999999,
                                "form": "10-K",
                                "accn": "EP2",
                            },
                        ]
                    }
                }
            }
        }
    }
    cik = "0001000004"
    (tmp_path / f"CIK{cik}.json").write_text(json.dumps(facts), encoding="utf-8")
    df = parse_company(cik, ["us-gaap:EarningsPerShareBasic"], tmp_path, ["10-K"])
    assert len(df) == 1
    assert df.iloc[0]["accn"] == "EP1"
