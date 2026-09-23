"""SerpAPI discovery and Amazon detail retrieval.

Google Shopping is the cross-retailer discovery source. Amazon Search discovers ASINs
from text; Amazon Product verifies a known ASIN and supplies detail/promotions.

Docs: https://serpapi.com/google-shopping-api  https://serpapi.com/amazon-search-api
      https://serpapi.com/amazon-product-api
Field names below follow SerpApi's documented responses but parsing is defensive; verify
against a real response once you have a key (raw JSON is stored in the DB for that).
"""
from __future__ import annotations

from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .market_models import Offer
from .market_normalize import (extract_asin, is_url, marketplace_from_url, parse_float,
                               parse_int, parse_price, search_query_from_url)
from .market_promotions import classify_promo

BASE_URL = "https://serpapi.com/search.json"

SUPPORTED_RETAILERS = {
    "amazon.in": ("amazon.in", "amazon"),
    "flipkart.com": ("flipkart.com", "flipkart"),
    "croma.com": ("croma.com", "croma"),
    "reliancedigital.in": ("reliancedigital.in", "reliance digital"),
    "vijaysales.com": ("vijaysales.com", "vijay sales"),
}


class SerpApiError(RuntimeError):
    pass


class SerpApiProvider:
    name = "serpapi"

    def __init__(self, api_key: str, *, country: str = "in", google_domain: str = "google.co.in",
                 amazon_domain: str = "amazon.in", session: requests.Session | None = None,
                 connect_timeout: int = 10, read_timeout: int = 120, retries: int = 2,
                 timeout: int | None = None, enrich_amazon: bool = False):
        if not api_key:
            raise SerpApiError("SERPAPI_API_KEY is not set")
        self.api_key = api_key
        self.country = country
        self.google_domain = google_domain
        self.amazon_domain = amazon_domain
        self.session = session or self._session_with_retries(retries)
        self.timeout = (connect_timeout, timeout or read_timeout)
        self.enrich_amazon = enrich_amazon  # 1 extra credit per ASIN
        self.warnings: list[str] = []

    @staticmethod
    def _session_with_retries(retries: int) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=max(0, retries), connect=max(0, retries), read=max(0, retries),
            status=max(0, retries), backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}), respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        return session

    # -- http ---------------------------------------------------------------
    def _get(self, **params: Any) -> dict[str, Any]:
        params["api_key"] = self.api_key
        engine = params.get("engine", "unknown")
        try:
            resp = self.session.get(BASE_URL, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.Timeout as exc:
            raise SerpApiError(
                f"{engine} timed out after {self.timeout[1]}s; retry later or increase "
                "SERPAPI_READ_TIMEOUT"
            ) from exc
        except requests.RequestException as exc:
            raise SerpApiError(f"{engine} request failed: {exc}") from exc
        except ValueError as exc:
            raise SerpApiError(f"{engine} returned invalid JSON") from exc
        if data.get("error"):
            raise SerpApiError(data["error"])
        return data

    # -- public -------------------------------------------------------------
    def search(self, query: str, limit: int = 20) -> list[Offer]:
        self.warnings = []
        asin = extract_asin(query)
        if asin:
            # Amazon Search remains the required discovery API. Searching the ASIN
            # resolves the canonical title without making Product API mandatory.
            amazon_offers, amazon_failed = self._safe_source(
                "amazon_search", lambda: self.search_amazon(asin, limit))
            exact = next((o for o in amazon_offers if o.asin == asin and o.title), None)
            resolved_query = exact.title if exact else asin
            shopping_offers, shopping_failed = self._safe_source(
                "google_shopping", lambda: self.search_google_shopping(resolved_query, limit))
            if amazon_failed and shopping_failed:
                raise SerpApiError("; ".join(self.warnings))
            offers = shopping_offers + amazon_offers
            if self.enrich_amazon:
                self._enrich_amazon_offers(offers)
            return offers

        resolved_query = search_query_from_url(query) if is_url(query) else query.strip()
        shopping_offers, shopping_failed = self._safe_source(
            "google_shopping", lambda: self.search_google_shopping(resolved_query, limit))
        amazon_offers, amazon_failed = self._safe_source(
            "amazon_search", lambda: self.search_amazon(resolved_query, limit))
        if shopping_failed and amazon_failed:
            raise SerpApiError("; ".join(self.warnings))
        offers = shopping_offers + amazon_offers
        if self.enrich_amazon:
            self._enrich_amazon_offers(offers)
        return offers

    def _safe_source(self, label: str, call) -> tuple[list[Offer], bool]:
        try:
            return call(), False
        except Exception as exc:  # one engine must not discard the other engine's results
            self.warnings.append(f"{label}: {exc}")
            return [], True

    def search_google_shopping(self, query: str, limit: int = 20) -> list[Offer]:
        data = self._get(engine="google_shopping", q=query, gl=self.country,
                         hl="en", google_domain=self.google_domain)
        return [o for o in self.parse_google_shopping(data)
                if o.marketplace in SUPPORTED_RETAILERS][:limit]

    def search_amazon(self, query: str, limit: int = 20) -> list[Offer]:
        data = self._get(engine="amazon", k=query, amazon_domain=self.amazon_domain)
        return self.parse_amazon_search(data)[:limit]

    def google_search(self, query: str, num_results: int = 5) -> list[dict[str, Any]]:
        """Plain Google web search (used by the Market Investigator)."""
        data = self._get(engine="google", q=query, gl=self.country, hl="en",
                         google_domain=self.google_domain, num=num_results)
        return self.parse_google_search(data)[:num_results]

    @staticmethod
    def parse_google_search(data: dict[str, Any]) -> list[dict[str, Any]]:
        return [{"title": r.get("title", ""), "snippet": r.get("snippet", ""),
                 "link": r.get("link", ""), "date": r.get("date")}
                for r in data.get("organic_results", []) or []]

    # -- parsers (pure; unit-tested with fixtures) ----------------------------
    def parse_google_shopping(self, data: dict[str, Any]) -> list[Offer]:
        out: list[Offer] = []
        for r in data.get("shopping_results", []) or []:
            link = r.get("link") or r.get("product_link")
            price, cur = parse_price(r.get("extracted_price", r.get("price")))
            if cur is None:
                _, cur = parse_price(r.get("price"))
            old, _ = parse_price(r.get("extracted_old_price", r.get("old_price")))
            badges = [b for b in (r.get("tag"), r.get("delivery"), r.get("snippet")) if b]
            marketplace = _supported_marketplace(link, r.get("source"))
            if marketplace is None:
                continue
            out.append(Offer(
                provider=self.name,
                marketplace=marketplace,
                title=r.get("title", ""),
                url=link,
                external_id=str(r.get("product_id")) if r.get("product_id") else None,
                asin=extract_asin(link, r.get("product_link")),
                price=price, currency=cur, original_price=old,
                rating=parse_float(r.get("rating")),
                review_count=parse_int(r.get("reviews")),
                seller_name=r.get("source"),
                shipping=r.get("delivery"),
                offers=badges,
                raw=r,
            ))
        return out

    def fetch_amazon_product(self, asin: str) -> list[Offer]:
        data = self._get(engine="amazon_product", asin=asin,
                         amazon_domain=self.amazon_domain, other_sellers="true")
        return self.parse_amazon_product(data, asin)

    def parse_amazon_product(self, data: dict[str, Any], asin: str) -> list[Offer]:
        pr = data.get("product_results", {}) or {}
        title = pr.get("title") or ""
        url = pr.get("link") or f"https://www.{self.amazon_domain}/dp/{asin}"
        price, cur = parse_price(pr.get("extracted_price", pr.get("price")))
        old, _ = parse_price(pr.get("extracted_old_price", pr.get("old_price")))
        seller = _seller_name(pr.get("seller") or pr.get("sold_by"))
        promo_values = list(pr.get("offers") or [])
        promo_values += list(pr.get("promotions") or [])
        promo_values += list(pr.get("coupons") or [])
        promos = []
        for value in promo_values:
            text = _promotion_text(value)
            if text:
                promos.append(classify_promo(text, source="serpapi_amazon_product"))
        base = Offer(
            provider=self.name, marketplace=self.amazon_domain, title=title,
            url=url, external_id=asin, asin=asin, brand=pr.get("brand"),
            price=price, currency=cur or "INR", original_price=old,
            rating=parse_float(pr.get("rating")), review_count=parse_int(pr.get("reviews")),
            seller_name=seller or "Amazon listing",
            availability=_join(pr.get("availability")), shipping=_join(pr.get("delivery")),
            offers=[p.description for p in promos], promos=promos, raw=pr,
        )
        offers = [base]
        for row in data.get("other_sellers", []) or []:
            other_price, other_cur = parse_price(row.get("extracted_price", row.get("price")))
            offers.append(Offer(
                provider=self.name, marketplace=self.amazon_domain, title=title,
                url=row.get("link") or url, external_id=asin, asin=asin,
                price=other_price, currency=other_cur or base.currency,
                original_price=parse_price(row.get("extracted_old_price", row.get("old_price")))[0],
                seller_name=_seller_name(row.get("seller")) or _seller_name(row.get("sold_by")),
                seller_rating=parse_float(row.get("rating")),
                review_count=parse_int(row.get("reviews")), shipping=_join(row.get("delivery")),
                offers=[str(v) for v in row.get("notes", []) or []], raw=row,
            ))
        return offers

    def parse_amazon_search(self, data: dict[str, Any]) -> list[Offer]:
        out: list[Offer] = []
        for r in data.get("organic_results", []) or []:
            if r.get("sponsored"):
                continue
            price, cur = parse_price(r.get("extracted_price", r.get("price")))
            if cur is None:
                _, cur = parse_price(r.get("price"))
            old, _ = parse_price(r.get("extracted_old_price", r.get("old_price")))
            badges = [str(b) for b in (r.get("coupon"), r.get("save_with_coupon"),
                                       r.get("bought_last_month"), r.get("tag"),
                                       "Prime" if r.get("prime") else None) if b]
            badges += [str(v) for v in r.get("offers", []) or []]
            promos = [classify_promo(text, source="serpapi_amazon_search")
                      for text in badges if _looks_like_promotion(text)]
            asin = extract_asin(r.get("asin"), r.get("link_clean"), r.get("link"))
            out.append(Offer(
                provider=self.name,
                marketplace=self.amazon_domain,
                title=r.get("title", ""),
                url=r.get("link_clean") or r.get("link"),
                external_id=asin,
                asin=asin,
                price=price, currency=cur or ("INR" if self.amazon_domain.endswith(".in") else None),
                original_price=old,
                rating=parse_float(r.get("rating")),
                review_count=parse_int(r.get("reviews")),
                seller_name="Amazon listing",  # search results don't expose the buy-box seller
                shipping=_join(r.get("delivery")),
                offers=badges, promos=promos,
                raw=r,
            ))
        return out

    # -- enrichment ---------------------------------------------------------
    def _enrich_from_amazon_product(self, offer: Offer) -> None:
        try:
            rich = self.fetch_amazon_product(offer.asin or "")
        except Exception as exc:
            self.warnings.append(f"amazon_product ({offer.asin}): {exc}")
            return
        if not rich:
            return
        detail = rich[0]
        offer.seller_name = detail.seller_name or offer.seller_name
        offer.brand = detail.brand or offer.brand
        offer.availability = detail.availability or offer.availability
        offer.promos = detail.promos
        offer.offers = list(dict.fromkeys(offer.offers + detail.offers))
        offer.raw["product_detail"] = detail.raw

    def _enrich_amazon_offers(self, offers: list[Offer]) -> None:
        """Fetch each ASIN once even when Shopping and Amazon Search both found it."""
        by_asin: dict[str, list[Offer]] = {}
        for offer in offers:
            if offer.marketplace == self.amazon_domain and offer.asin:
                by_asin.setdefault(offer.asin, []).append(offer)
        for asin, matching in by_asin.items():
            try:
                rich = self.fetch_amazon_product(asin)
            except Exception as exc:
                self.warnings.append(f"amazon_product ({asin}): {exc}")
                continue
            if not rich:
                continue
            detail = rich[0]
            for offer in matching:
                offer.seller_name = detail.seller_name or offer.seller_name
                offer.brand = detail.brand or offer.brand
                offer.availability = detail.availability or offer.availability
                offer.promos = detail.promos or offer.promos
                offer.offers = list(dict.fromkeys(offer.offers + detail.offers))
                offer.raw["product_detail"] = detail.raw


def _join(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, list):
        return "; ".join(map(str, v))
    return str(v)


def _supported_marketplace(url: str | None, source: str | None) -> str | None:
    host = marketplace_from_url(url)
    source_text = (source or "").lower()
    for marketplace, aliases in SUPPORTED_RETAILERS.items():
        if (host and (host == marketplace or host.endswith("." + marketplace))) or any(
            alias in source_text for alias in aliases
        ):
            return marketplace
    return None


def _seller_name(value: Any) -> str | None:
    if isinstance(value, dict):
        return value.get("name") or value.get("business_name")
    return str(value) if value else None


def _promotion_text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("text") or value.get("description") or value.get("discount") or "")
    return str(value or "")


def _looks_like_promotion(text: str) -> bool:
    value = text.lower()
    return any(token in value for token in (
        "off", "coupon", "offer", "cashback", "emi", "save", "discount", "deal",
    ))
