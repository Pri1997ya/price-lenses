"""Configuration for the India Market Investigator.

The existing application convention is intentionally retained: ``DATABASE_URL``
is the runtime PostgreSQL connection string. API credentials are read only from
the environment and are never logged.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


class ConfigurationError(ValueError):
    """Raised when Market Investigator configuration is invalid."""


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if value < 1:
        raise ConfigurationError(f"{name} must be greater than zero")
    return value


def _nonnegative_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if value < 0:
        raise ConfigurationError(f"{name} must be zero or greater")
    return value


def _json_object(name: str) -> dict:
    raw = os.getenv(name)
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"{name} must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"{name} must contain a JSON object")
    return value


@dataclass(frozen=True)
class MarketSettings:
    database_url: str
    serpapi_key: str | None
    apify_token: str | None
    country: str = "in"
    google_domain: str = "google.co.in"
    amazon_domain: str = "amazon.in"
    serpapi_connect_timeout: int = 10
    serpapi_read_timeout: int = 120
    serpapi_retries: int = 2
    serpapi_enrich_amazon: bool = False
    apify_ecom_actor: str = "apify/e-commerce-scraping-tool"
    apify_ecom_mode: str = "Sellers"
    apify_flipkart_details_actor: str = "piotrv1001/flipkart-product-details-scraper"
    apify_bank_offers_actor: str = "pale_tapestry/bank-offers-aggregator-actor"
    apify_enrichers: tuple[str, ...] = ("flipkart_details", "bank_offers")
    apify_enrich_limit: int = 5
    apify_extra_input: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "MarketSettings":
        load_dotenv()
        database_url = (
            os.getenv("DATABASE_URL") or os.getenv("PL_DATABASE_URL") or ""
        ).strip()
        if not database_url:
            raise ConfigurationError(
                "DATABASE_URL is required (PL_DATABASE_URL is also supported)"
            )

        country = os.getenv("MARKET_COUNTRY", "in").strip().lower()
        google_domain = os.getenv("MARKET_GOOGLE_DOMAIN", "google.co.in").strip().lower()
        amazon_domain = os.getenv("MARKET_AMAZON_DOMAIN", "amazon.in").strip().lower()
        if country != "in" or google_domain != "google.co.in" or amazon_domain != "amazon.in":
            raise ConfigurationError(
                "Market Investigator currently supports India only: use MARKET_COUNTRY=in, "
                "MARKET_GOOGLE_DOMAIN=google.co.in and MARKET_AMAZON_DOMAIN=amazon.in"
            )

        enrichers = tuple(
            value.strip()
            for value in os.getenv(
                "APIFY_ENRICHERS", "flipkart_details,bank_offers"
            ).split(",")
            if value.strip()
        )
        allowed_enrichers = {"flipkart_details", "bank_offers"}
        unknown = set(enrichers) - allowed_enrichers
        if unknown:
            raise ConfigurationError(
                "Unsupported APIFY_ENRICHERS: " + ", ".join(sorted(unknown))
            )

        return cls(
            database_url=database_url,
            serpapi_key=os.getenv("SERPAPI_API_KEY") or None,
            apify_token=os.getenv("APIFY_API_TOKEN") or None,
            country=country,
            google_domain=google_domain,
            amazon_domain=amazon_domain,
            serpapi_connect_timeout=_positive_int("SERPAPI_CONNECT_TIMEOUT", 10),
            serpapi_read_timeout=_positive_int("SERPAPI_READ_TIMEOUT", 120),
            serpapi_retries=_nonnegative_int("SERPAPI_RETRIES", 2),
            serpapi_enrich_amazon=os.getenv("SERPAPI_ENRICH_AMAZON", "false").lower()
            in {"1", "true", "yes"},
            apify_ecom_actor=os.getenv(
                "APIFY_ECOM_ACTOR", "apify/e-commerce-scraping-tool"
            ),
            apify_ecom_mode=os.getenv("APIFY_ECOM_MODE", "Sellers"),
            apify_flipkart_details_actor=os.getenv(
                "APIFY_FLIPKART_DETAILS_ACTOR",
                "piotrv1001/flipkart-product-details-scraper",
            ),
            apify_bank_offers_actor=os.getenv(
                "APIFY_BANK_OFFERS_ACTOR",
                "pale_tapestry/bank-offers-aggregator-actor",
            ),
            apify_enrichers=enrichers,
            apify_enrich_limit=_positive_int("APIFY_ENRICH_LIMIT", 5),
            apify_extra_input=_json_object("APIFY_EXTRA_INPUT_JSON"),
        )
