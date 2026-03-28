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
# Applied at parse time so the parquet always uses canonical names.
CONCEPT_ALIASES: dict[str, str] = {
    "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax": "us-gaap:Revenues",
    "us-gaap:OperatingCashFlow": "us-gaap:NetCashProvidedByUsedInOperatingActivities",
}

DEFAULT_FORMS = ["10-K", "10-Q"]
BULK_ZIP_URL = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"


class PitEdgarConfig(BaseModel):
    edgar_identity: str
    data_dir: Path
    facts_dir: Path | None = None
    zip_url: str = BULK_ZIP_URL
    concepts: list[str] = DEFAULT_CONCEPTS
    forms: list[str] = DEFAULT_FORMS

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def set_facts_dir(self) -> "PitEdgarConfig":
        if self.facts_dir is None:
            self.facts_dir = self.data_dir / "companyfacts"
        return self

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.facts_dir.mkdir(parents=True, exist_ok=True)
