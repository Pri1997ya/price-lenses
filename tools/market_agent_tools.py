"""Controlled tools available to the Market Investigator agent.

The agent receives these typed operations instead of unrestricted SQL or direct
provider clients.  This keeps provider spend and database writes auditable.
"""
from __future__ import annotations

from dataclasses import asdict
import re
from typing import Any

from .market_matching import product_relevance
from .market_normalize import is_url, search_query_from_url
from .market_service import MarketInvestigatorService, MarketProvider


_CAPACITY_RE = re.compile(r"\b(\d+)\s*(gb|tb)\b", re.I)
_RAM_RE = re.compile(r"\b(\d+)\s*gb\s*(?:ram|memory)\b", re.I)
_STORAGE_RE = re.compile(r"\b(\d+)\s*(gb|tb)\s*(?:storage|rom|ssd)\b", re.I)
_SCREEN_SIZE_RE = re.compile(r"\b(\d{1,3}(?:\.\d+)?)\s*(?:inch(?:es)?|in\b|\")", re.I)
_DEVICE_SIZE_RE = re.compile(r"\b(\d{2,3}(?:\.\d+)?)\s*mm\b", re.I)
_GENERATION_RE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s*(?:gen|generation)\b", re.I)
_COLOURS = (
    "silk white", "sapphire blue", "natural titanium", "titanium black",
    "titanium blue", "midnight", "starlight", "graphite", "black", "white",
    "blue", "green", "silver", "gold", "purple", "pink", "grey", "gray",
)
_CONDITIONS = ("renewed", "refurbished", "open box", "used")
_BUNDLES = (
    "with charger", "without charger", "with keyboard", "with pen",
    "with stylus",
)


def variant_facets(title: str | None) -> dict[str, str | None]:
    text = (title or "").lower()
    ram_match = _RAM_RE.search(text)
    storage_match = _STORAGE_RE.search(text)
    capacities = [f"{number}{unit.upper()}" for number, unit in _CAPACITY_RE.findall(text)]
    ram = f"{ram_match.group(1)}GB" if ram_match else None
    storage = (
        f"{storage_match.group(1)}{storage_match.group(2).upper()}"
        if storage_match
        else next((value for value in reversed(capacities) if value != ram), None)
    )
    colour = next((colour.title() for colour in _COLOURS if colour in text), None)
    connectivity = "5G" if re.search(r"\b5g\b", text) else ("4G" if re.search(r"\b4g\b", text) else None)
    screen_match = _SCREEN_SIZE_RE.search(text)
    size_match = _DEVICE_SIZE_RE.search(text)
    generation_match = _GENERATION_RE.search(text)
    condition = next((item.title() for item in _CONDITIONS if item in text), None)
    bundle = next((item.title() for item in _BUNDLES if item in text), None)
    return {
        "storage": storage,
        "ram": ram,
        "color": colour,
        "connectivity": connectivity,
        "screen_size": f"{screen_match.group(1)} inch" if screen_match else None,
        "device_size": f"{size_match.group(1)}mm" if size_match else None,
        "generation": f"{generation_match.group(1)} Gen" if generation_match else None,
        "condition": condition,
        "bundle": bundle,
    }


def _normalized_facet(value: Any) -> str | None:
    if value is None:
        return None
    return re.sub(r"\s+", "", str(value)).lower().replace("grey", "gray")


def variant_match_tier(
    requested: dict[str, str | None], candidate: dict[str, str | None]
) -> str:
    """Classify a product-family member without silently relaxing hard attributes."""
    hard_fields = (
        "storage", "ram", "connectivity", "screen_size", "device_size",
        "generation", "condition", "bundle",
    )
    hard_conflict = any(
        requested.get(field)
        and _normalized_facet(requested[field]) != _normalized_facet(candidate.get(field))
        for field in hard_fields
    )
    if hard_conflict:
        return "other_configuration"
    requested_color = _normalized_facet(requested.get("color"))
    candidate_color = _normalized_facet(candidate.get("color"))
    if requested_color and requested_color != candidate_color:
        return "same_configuration_other_color"
    if any(requested.values()):
        return "exact_variant"
    return "other_configuration"


class MarketAgentTools:
    def __init__(self, database: Any, providers: list[MarketProvider] | None = None):
        self.database = database
        self.providers = list(providers or [])

    @staticmethod
    def _rank_candidates(query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        reference = search_query_from_url(query) if is_url(query) else query
        ranked = []
        for candidate in candidates:
            score = product_relevance(reference, candidate.get("title") or "")
            if score < 0.25:
                continue
            item = dict(candidate)
            item["relevance"] = round(score, 3)
            item["variant"] = variant_facets(item.get("title"))
            ranked.append(item)
        ranked.sort(
            key=lambda product: (
                product["relevance"], int(product.get("observed_offer_count") or 0)
            ),
            reverse=True,
        )
        return ranked[:10]

    @staticmethod
    def _unresolved_fields(query: str, options: list[dict[str, Any]]) -> list[str]:
        requested = variant_facets(query)
        unresolved = []
        for field in (
            "storage", "ram", "color", "connectivity", "screen_size",
            "device_size", "generation", "condition", "bundle",
        ):
            values = {
                option.get("variant", {}).get(field)
                for option in options
                if option.get("variant", {}).get(field)
            }
            if len(values) > 1 and not requested.get(field):
                unresolved.append(field)
        return unresolved

    def _select_product(
        self, query: str, canonical_id: str | None
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
        if canonical_id:
            product = self.database.product(canonical_id)
            if product:
                option = dict(product)
                option["variant"] = variant_facets(option.get("title"))
                return product, [option], []
        candidates = self.database.products_for_query(query)
        options = self._rank_candidates(query, candidates)
        if not options:
            return {}, [], []
        return options[0], options, self._unresolved_fields(query, options)

    def get_market_snapshot(
        self,
        query: str,
        canonical_id: str | None = None,
        run_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        if run_ids:
            offers = self.database.offers_for_run_ids(run_ids)
            if offers:
                counts: dict[str, int] = {}
                titles: dict[str, str] = {}
                prices: dict[str, list[float]] = {}
                for row in offers:
                    product_id = str(row.get("canonical_id"))
                    counts[product_id] = counts.get(product_id, 0) + 1
                    titles.setdefault(product_id, row.get("product_title") or row.get("title") or "")
                    if row.get("price") is not None:
                        prices.setdefault(product_id, []).append(float(row["price"]))
                candidates = [
                    {
                        **(self.database.product(product_id) or {
                            "canonical_id": product_id, "title": titles[product_id]
                        }),
                        "observed_offer_count": counts[product_id],
                        "lowest_price": min(prices.get(product_id, []), default=None),
                    }
                    for product_id in counts
                ]
                options = self._rank_candidates(query, candidates)
                if not options:
                    return {
                        "product": {}, "offers": [], "promotions": [],
                        "variant_options": [], "unresolved_variant_fields": [],
                    }
                selected_id = options[0]["canonical_id"]
                allowed_ids = {str(option["canonical_id"]) for option in options}
                offers = [
                    row for row in offers
                    if str(row.get("canonical_id")) in allowed_ids
                ]
                product = self.database.product(selected_id) or {
                    "canonical_id": selected_id,
                    "title": titles[selected_id],
                }
                unresolved = self._unresolved_fields(query, options)
            else:
                product, options, unresolved = {}, [], []
            promotions = self.database.promotions_for_offer_ids(
                row["offer_id"] for row in offers if row.get("offer_id")
            )
            return {
                "product": product,
                "offers": offers,
                "promotions": promotions,
                "variant_options": options,
                "unresolved_variant_fields": unresolved,
            }

        product, options, unresolved = self._select_product(query, canonical_id)
        if canonical_id and product.get("canonical_id"):
            offers = self.database.offers_for_product(product["canonical_id"])
        else:
            offers = self.database.offers_for_query(query)
            allowed_ids = {str(option["canonical_id"]) for option in options}
            if allowed_ids:
                offers = [
                    row for row in offers
                    if str(row.get("canonical_id")) in allowed_ids
                ]
            if not product and offers:
                selected_id = offers[0].get("canonical_id")
                product = self.database.product(selected_id) or {
                    "canonical_id": selected_id,
                    "title": offers[0].get("product_title") or offers[0].get("title"),
                }
        promotions = self.database.promotions_for_offer_ids(
            row["offer_id"] for row in offers if row.get("offer_id")
        )
        return {
            "product": product,
            "offers": offers,
            "promotions": promotions,
            "variant_options": options,
            "unresolved_variant_fields": unresolved,
        }

    def refresh_market_snapshot(
        self,
        query: str,
        providers: list[MarketProvider] | None = None,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        selected = list(self.providers if providers is None else providers)
        if not selected:
            return []
        results = MarketInvestigatorService(self.database, selected).search(query, limit)
        return [asdict(result) for result in results]

    def get_upcoming_sales(self, deadline_days: int) -> list[dict[str, Any]]:
        return self.database.upcoming_sales(deadline_days)
