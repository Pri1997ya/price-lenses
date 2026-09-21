"""Provider orchestration for the India Market Investigator."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from .market_apify_provider import ApifyProvider, specs_from_settings
from .market_config import ConfigurationError, MarketSettings
from .market_db import MarketDatabase
from .market_matching import ProductResolver
from .market_models import Offer
from .market_serpapi_provider import SerpApiProvider

log = logging.getLogger(__name__)


class MarketProvider(Protocol):
    name: str

    def search(self, query: str, limit: int = 20) -> list[Offer]: ...


@dataclass(frozen=True)
class RunResult:
    provider: str
    count: int
    status: str
    error: str | None = None
    warnings: tuple[str, ...] = ()


def build_providers(
    settings: MarketSettings,
    names: tuple[str, ...] = ("serpapi", "apify"),
) -> list[MarketProvider]:
    """Construct configured providers without making any network calls."""
    providers: list[MarketProvider] = []
    for name in names:
        normalized = name.strip().lower()
        if normalized == "serpapi":
            if not settings.serpapi_key:
                raise ConfigurationError("SERPAPI_API_KEY is required for the SerpAPI provider")
            providers.append(
                SerpApiProvider(
                    settings.serpapi_key,
                    country=settings.country,
                    google_domain=settings.google_domain,
                    amazon_domain=settings.amazon_domain,
                    connect_timeout=settings.serpapi_connect_timeout,
                    read_timeout=settings.serpapi_read_timeout,
                    retries=settings.serpapi_retries,
                    enrich_amazon=settings.serpapi_enrich_amazon,
                )
            )
        elif normalized == "apify":
            if not settings.apify_token:
                raise ConfigurationError("APIFY_API_TOKEN is required for the Apify provider")
            providers.append(
                ApifyProvider(
                    settings.apify_token,
                    specs=specs_from_settings(settings),
                    enrich=bool(settings.apify_enrichers),
                    enrich_limit=settings.apify_enrich_limit,
                )
            )
        else:
            raise ConfigurationError(f"Unknown market provider: {name}")
    return providers


class MarketInvestigatorService:
    def __init__(self, database: MarketDatabase, providers: list[MarketProvider]):
        if not providers:
            raise ValueError("At least one market provider is required")
        self.database = database
        self.providers = providers

    def search(self, query: str, limit: int = 20) -> list[RunResult]:
        """Fetch, match, and persist offers while isolating provider failures."""
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")

        resolver = ProductResolver(self.database.known_products())
        results: list[RunResult] = []
        for provider in self.providers:
            run_id = self.database.start_run(normalized_query, provider.name)
            try:
                offers = provider.search(normalized_query, limit)
                offers.sort(key=lambda offer: (offer.asin is None, offer.gtin is None))
                rows: list[tuple[str, Offer]] = []
                for offer in offers:
                    resolved_id = resolver.resolve(offer)
                    canonical_id = self.database.upsert_product(resolved_id, offer)
                    rows.append((canonical_id, offer))
                count = self.database.insert_offers(run_id, rows)
                warnings = tuple(getattr(provider, "warnings", ()) or ())
                status = "partial" if warnings else "ok"
                self.database.finish_run(
                    run_id,
                    count,
                    status=status,
                    error="; ".join(warnings) or None,
                )
                results.append(RunResult(provider.name, count, status, warnings=warnings))
            except Exception as exc:  # provider failures must not discard other results
                log.exception("market provider %s failed", provider.name)
                self.database.finish_run(run_id, 0, status="error", error=str(exc))
                results.append(RunResult(provider.name, 0, "error", error=str(exc)))
        return results
