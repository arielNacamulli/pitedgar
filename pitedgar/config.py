"""Configuration for pitedgar via Pydantic BaseModel."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator

DEFAULT_CONCEPTS = [
    "us-gaap:Revenues",
    "us-gaap:NetIncomeLoss",
    "us-gaap:Assets",
    "us-gaap:Liabilities",
    "us-gaap:StockholdersEquity",
    "us-gaap:OperatingIncomeLoss",
    "us-gaap:GrossProfit",
    "us-gaap:EarningsPerShareBasic",
    "us-gaap:EarningsPerShareDiluted",
    "us-gaap:CommonStockSharesOutstanding",
    "us-gaap:CashAndCashEquivalentsAtCarryingValue",
    "us-gaap:LongTermDebt",
    "us-gaap:NetCashProvidedByUsedInOperatingActivities",
    "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
    "us-gaap:ResearchAndDevelopmentExpense",
]

# Explicit priority ordering for (non-lossy) alias resolution. When two aliases both
# report the same (end, filed, form) for a canonical concept, the alias that appears
# earlier in the list wins. List order = priority; canonical always wins over aliases.
#
# Only genuinely synonymous / deprecated tags belong here. Tags that are financially
# non-equivalent to their canonical target are in LOSSY_CONCEPT_ALIASES below and
# require opt-in via PitEdgarConfig.lossy_aliases_enabled.
CONCEPT_ALIAS_PRIORITY: dict[str, list[str]] = {
    # --- Revenue family -> us-gaap:Revenues ---
    # Post-ASC 606 standard tag (mandatory from 2018 onward) has highest priority;
    # deprecated pre-ASC 606 tags are tried in descending order.
    "us-gaap:Revenues": [
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
        "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax",
        "us-gaap:SalesRevenueNet",
        "us-gaap:SalesRevenueGoodsNet",
        "us-gaap:Revenue",
    ],
    # --- Cash family -> us-gaap:CashAndCashEquivalentsAtCarryingValue ---
    # Bare "Cash" is a generic fallback used by a minority of (older/smaller) filers.
    "us-gaap:CashAndCashEquivalentsAtCarryingValue": [
        "us-gaap:Cash",
    ],
    # --- Operating cash flow ---
    "us-gaap:NetCashProvidedByUsedInOperatingActivities": ["us-gaap:OperatingCashFlow"],
}

# Maps deprecated/variant XBRL tags to their canonical concept in DEFAULT_CONCEPTS.
# Derived from CONCEPT_ALIAS_PRIORITY so the public API remains stable and the parser
# priority order no longer depends on Python dict insertion order.
CONCEPT_ALIASES: dict[str, str] = {
    alias: canonical
    for canonical, aliases in CONCEPT_ALIAS_PRIORITY.items()
    for alias in aliases
}

# Aliases that are financially NON-EQUIVALENT to their canonical targets.
# These are kept separate because substituting them silently can mislead analysts:
#
#   ProfitLoss → NetIncomeLoss
#       ProfitLoss is the consolidated bottom line BEFORE allocating income to
#       non-controlling (minority) interests; NetIncomeLoss is attributable to
#       the parent only.
#
#   LongTermDebtNoncurrent → LongTermDebt
#       LongTermDebtNoncurrent excludes the current portion of long-term debt;
#       LongTermDebt includes it.
#
#   CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents
#       → CashAndCashEquivalentsAtCarryingValue
#       The post-ASU 2016-18 tag bundles restricted cash with unrestricted cash,
#       inflating the figure relative to the canonical carrying-value tag.
#
# These aliases are DISABLED by default (PitEdgarConfig.lossy_aliases_enabled=False).
# When enabled the parser records the original tag in the `alias_source` column so
# analysts can filter or audit the substitutions.
LOSSY_CONCEPT_ALIASES: dict[str, str] = {
    "us-gaap:ProfitLoss": "us-gaap:NetIncomeLoss",
    "us-gaap:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents": "us-gaap:CashAndCashEquivalentsAtCarryingValue",
    "us-gaap:LongTermDebtNoncurrent": "us-gaap:LongTermDebt",
}

# Amendments (10-K/A, 10-Q/A) carry corrected — and often substantially restated —
# data; they are typically filed weeks or months after the original. Foreign private
# issuers file annual reports on Form 20-F (and amendments on 20-F/A). We include all
# of these by default so the parquet captures the full universe of filings; the
# parser's PIT dedup keeps the correct row because `(concept, end)` dedup retains the
# latest distinct value (i.e. an amendment supersedes the original from its filed date).
DEFAULT_FORMS = ["10-K", "10-K/A", "10-Q", "10-Q/A", "20-F", "20-F/A"]
BULK_ZIP_URL = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"


class PitEdgarConfig(BaseModel):
    edgar_identity: str
    data_dir: Path
    facts_dir: Path = None  # type: ignore[assignment]
    zip_url: str = BULK_ZIP_URL
    # ``concepts`` controls which us-gaap tags are extracted from the per-company JSON.
    # ``None`` (the default) or an empty list means "parse every us-gaap concept present
    # in the JSON" — recommended so the parquet contains the full universe of tags and
    # iterating on new signals does not require rebuilding the 1.5 GB cache. Pass an
    # explicit list (e.g. ``DEFAULT_CONCEPTS``) to opt back into the curated subset.
    concepts: list[str] | None = None
    forms: list[str] = DEFAULT_FORMS
    scale_correction: Literal["off", "auto", "force"] = "off"
    scale_correction_threshold: float = 1_000_000.0
    lossy_aliases_enabled: bool = False
    max_extracted_bytes: int = 25 * 1024**3  # 25 GiB decompressed-size cap (zip-bomb guard)

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def set_facts_dir(self) -> "PitEdgarConfig":
        if self.facts_dir is None:
            self.facts_dir = self.data_dir / "companyfacts"
        return self

    @model_validator(mode="after")
    def validate_edgar_identity(self) -> "PitEdgarConfig":
        if not self.edgar_identity or not self.edgar_identity.strip():
            raise ValueError("edgar_identity must be a non-empty SEC User-Agent string")
        return self

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.facts_dir.mkdir(parents=True, exist_ok=True)
