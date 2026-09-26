"""Typed contracts shared by the India Market Investigator agent.

The contracts intentionally contain only JSON-serialisable values.  Agent 3 and
the future decision synthesizer can therefore consume ``MarketReport`` without
depending on provider response formats or database cursor objects.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


SUPPORTED_RETAILERS: tuple[str, ...] = (
    "amazon.in",
    "flipkart.com",
    "croma.com",
    "reliancedigital.in",
    "vijaysales.com",
)


@dataclass(frozen=True)
class MarketAgentRequest:
    query: str
    canonical_id: str | None = None
    deadline_days: int = 30
    force_refresh: bool = False
    provider_policy: Literal["api_first", "database_first", "database_only"] = "api_first"
    bank: str | None = None
    card_type: str | None = None
    wants_emi: bool | None = None

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query must not be empty")
        if self.deadline_days < 0 or self.deadline_days > 365:
            raise ValueError("deadline_days must be between 0 and 365")
        if self.provider_policy not in {"api_first", "database_first", "database_only"}:
            raise ValueError("provider_policy must be api_first, database_first, or database_only")


@dataclass(frozen=True)
class FreshnessPolicy:
    price_minutes: int = 60
    availability_minutes: int = 30
    delivery_minutes: int = 30
    promotion_minutes: int = 180
    seller_minutes: int = 1_440
    product_minutes: int = 10_080


@dataclass
class MarketReport:
    schema_version: str = "1.1"
    agent: str = "market_investigator"
    status: Literal["complete", "partial", "insufficient_evidence", "error"] = (
        "insufficient_evidence"
    )
    product: dict[str, Any] = field(default_factory=dict)
    analysis_timestamp: str = ""
    provider_runs: list[dict[str, Any]] = field(default_factory=list)
    coverage: dict[str, Any] = field(default_factory=dict)
    freshness: dict[str, Any] = field(default_factory=dict)
    best_listed_offer: dict[str, Any] | None = None
    best_unconditional_offer: dict[str, Any] | None = None
    best_conditional_offer: dict[str, Any] | None = None
    ranked_offers: list[dict[str, Any]] = field(default_factory=list)
    upcoming_sales: list[dict[str, Any]] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)
    confidence: float = 0.0
    missing_inputs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    match_mode: Literal[
        "exact_variant",
        "expanded_colors",
        "expanded_variants",
        "product_family_not_found",
    ] = "product_family_not_found"
    requested_attributes: dict[str, Any] = field(default_factory=dict)
    requested_match: dict[str, Any] | None = None
    variant_groups: list[dict[str, Any]] = field(default_factory=list)
    lowest_starting_price: float | None = None
    agent_trace: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
