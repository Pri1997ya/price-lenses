"""Provider orchestration for the India Market Investigator."""
from __future__ import annotations

import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

from .market_apify_provider import ApifyProvider, specs_from_settings
from .market_config import ConfigurationError, MarketSettings
from .market_db import MarketDatabase
from .market_agent_models import SUPPORTED_RETAILERS
from .market_analysis import marketplace_name
from .market_matching import ProductResolver, product_relevance
from .market_models import Offer
from .market_normalize import extract_asin, is_url, search_query_from_url
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
    run_id: str | None = None


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
        """Fetch providers concurrently, then validate and persist sequentially."""
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")

        resolver = ProductResolver(self.database.known_products())
        results: list[RunResult] = []
        runs = [
            (provider, self.database.start_run(normalized_query, provider.name))
            for provider in self.providers
        ]
        with ThreadPoolExecutor(max_workers=len(runs), thread_name_prefix="market-provider") as pool:
            futures = {
                provider.name: pool.submit(provider.search, normalized_query, limit)
                for provider, _run_id in runs
            }
        for provider, run_id in runs:
            try:
                offers = futures[provider.name].result()
                offers, validation_warnings = self._validated_offers(
                    normalized_query, offers
                )
                offers.sort(key=lambda offer: (offer.asin is None, offer.gtin is None))
                rows: list[tuple[str, Offer]] = []
                for offer in offers:
                    resolved_id = resolver.resolve(offer)
                    canonical_id = self.database.upsert_product(resolved_id, offer)
                    rows.append((canonical_id, offer))
                count = self.database.insert_offers(run_id, rows)
                warnings = tuple(getattr(provider, "warnings", ()) or ()) + validation_warnings
                status = "partial" if warnings else "ok"
                self.database.finish_run(
                    run_id,
                    count,
                    status=status,
                    error="; ".join(warnings) or None,
                )
                results.append(
                    RunResult(provider.name, count, status, warnings=warnings, run_id=run_id)
                )
            except Exception as exc:  # provider failures must not discard other results
                log.exception("market provider %s failed", provider.name)
                self.database.finish_run(run_id, 0, status="error", error=str(exc))
                results.append(
                    RunResult(provider.name, 0, "error", error=str(exc), run_id=run_id)
                )
        return results

    @staticmethod
    def _validated_offers(query: str, offers: list[Offer]) -> tuple[list[Offer], tuple[str, ...]]:
        """Reject accessories, unsupported stores, missing prices, and non-INR rows."""
        query_asin = extract_asin(query)
        reference = search_query_from_url(query) if is_url(query) else query
        if query_asin:
            exact = next((offer for offer in offers if offer.asin == query_asin and offer.title), None)
            if exact:
                reference = exact.title

        accepted: list[Offer] = []
        rejected: Counter[str] = Counter()
        for offer in offers:
            marketplace = marketplace_name(offer.marketplace)
            if marketplace not in SUPPORTED_RETAILERS:
                rejected["unsupported retailer"] += 1
                continue
            if offer.price is None or offer.price <= 0:
                rejected["missing price"] += 1
                continue
            currency = (offer.currency or "INR").upper()
            if currency != "INR":
                rejected["non-INR price"] += 1
                continue
            if query_asin and offer.asin == query_asin:
                relevance = 1.0
            else:
                relevance = product_relevance(reference, offer.title)
            if relevance < 0.25:
                rejected["irrelevant product or accessory"] += 1
                continue
            offer.marketplace = marketplace
            offer.currency = "INR"
            accepted.append(offer)

        warnings = tuple(
            f"validation excluded {count} {reason} result{'s' if count != 1 else ''}"
            for reason, count in sorted(rejected.items())
        )
        return accepted, warnings
