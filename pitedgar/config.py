"""Configuration for pitedgar via Pydantic BaseModel."""

from pathlib import Path

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

# Maps deprecated/variant XBRL tags to their canonical concept in DEFAULT_CONCEPTS.
# Applied at parse time so the parquet always uses canonical names. The parser tries
# the canonical tag first, falling back to each alias only if the canonical is absent —
# so when both are present the canonical value wins (no double-counting).
CONCEPT_ALIASES: dict[str, str] = {
    # --- Revenue family -> us-gaap:Revenues ---
    # Post-ASC 606 standard tag (mandatory from 2018 onward for most filers)
    "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax": "us-gaap:Revenues",
    # ASC 606 variant that includes assessed taxes (e.g. sales tax) in the topline
    "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax": "us-gaap:Revenues",
    # Pre-ASC 606 deprecated tags still used by some filers / historical filings
    "us-gaap:SalesRevenueNet": "us-gaap:Revenues",
    "us-gaap:SalesRevenueGoodsNet": "us-gaap:Revenues",
    "us-gaap:Revenue": "us-gaap:Revenues",
    # --- Net income family -> us-gaap:NetIncomeLoss ---
    # CAVEAT: ProfitLoss is NOT identical to NetIncomeLoss — ProfitLoss is the
    # consolidated bottom line BEFORE allocating income to non-controlling (minority)
    # interests, while NetIncomeLoss is attributable to the parent only. We map it
    # as a fallback so companies that only file ProfitLoss are still represented;
    # the canonical-first lookup ensures NetIncomeLoss wins whenever both are present.
    "us-gaap:ProfitLoss": "us-gaap:NetIncomeLoss",
    # --- Cash family -> us-gaap:CashAndCashEquivalentsAtCarryingValue ---
    # Post-ASU 2016-18 tag that bundles restricted cash with cash & equivalents
    "us-gaap:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents": "us-gaap:CashAndCashEquivalentsAtCarryingValue",
    # Bare "Cash" tag used by a minority of filers (typically smaller / older filings)
    "us-gaap:Cash": "us-gaap:CashAndCashEquivalentsAtCarryingValue",
    # --- Long-term debt -> us-gaap:LongTermDebt ---
    # Many filers report only the noncurrent portion under this tag
    "us-gaap:LongTermDebtNoncurrent": "us-gaap:LongTermDebt",
    # --- Operating cash flow ---
    "us-gaap:OperatingCashFlow": "us-gaap:NetCashProvidedByUsedInOperatingActivities",
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
