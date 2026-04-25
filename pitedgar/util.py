"""Shared utilities."""


def normalize_ticker(t: str) -> str:
    """Canonicalise a user-supplied ticker symbol: strip + upper.

    This is the single source of truth for ticker canonicalisation; both
    CLI input and library-level mapping should route through it to
    guarantee identical treatment.
    """
    return t.strip().upper()
