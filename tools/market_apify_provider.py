"""Apify provider: two-stage fetch.

  Stage 1  DISCOVER  keyword -> listings / sellers        (apify/e-commerce-scraping-tool, search-engine mode)
  Stage 2  ENRICH    product URL -> full page details     (site-specific actors: sellers, bank offers, EMI ...)

Why two stages: the generic e-commerce tool returns schema.org-style data (name, price, currency ...).
Bank/exchange offers, seller scores, COD, warranty etc. only come from actors that read the product page
itself, so we discover products first and then enrich the cheapest few per marketplace.

Every actor has its own input/output shape, so each one is an `ActorSpec` (actor id + input builder +
normaliser). Output mappings below follow each actor's documented sample output; verify with one real
run per actor (raw JSON of every item is kept in the DB) and adjust the small normaliser if needed.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from .market_matching import title_similarity
from .market_models import Offer, Promo
from .market_normalize import (extract_asin, marketplace_from_url, parse_float, parse_int,
                               parse_price)
from .market_promotions import best_instant_bank_discount, classify_promo


class ApifyError(RuntimeError):
    pass


# --------------------------------------------------------------------------- run helpers
def _field(run: Any, snake: str, camel: str) -> Any:
    """apify-client v1 returns dicts (camelCase); v2+ returns a pydantic `Run` (snake_case)."""
    if isinstance(run, dict):
        return run.get(camel, run.get(snake))
    return getattr(run, snake, None) or getattr(run, camel, None)


def dataset_id_of(run: Any) -> str | None:
    return _field(run, "default_dataset_id", "defaultDatasetId")


def check_run_succeeded(run: Any, actor_id: str) -> None:
    status = _field(run, "status", "status")
    status = str(getattr(status, "value", status) or "")
    if status and "SUCCEEDED" not in status.upper():
        raise ApifyError(f"Actor {actor_id} finished with status {status} "
                         f"(run id: {_field(run, 'id', 'id')}). Check the run log in the Apify console.")


# --------------------------------------------------------------------------- spec
@dataclass
class ActorSpec:
    key: str                                             # ecom_search | flipkart_details | bank_offers
    actor_id: str
    stage: str                                           # "discover" | "enrich"
    build_input: Callable[[Any, int], dict[str, Any]]    # (query | [urls], limit) -> run_input
    normalize: Callable[[dict[str, Any]], list[Offer]]
    hosts: tuple[str, ...] = ()                          # enrich only: marketplaces this actor understands


_BULKY = {"reviews", "specifications", "images", "description", "highlights", "manufacturerInfo"}


def _slim(item: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in item.items() if k not in _BULKY}


# --------------------------------------------------------------------------- 1) official e-commerce tool
def normalize_ecom_item(item: dict[str, Any]) -> list[Offer]:
    """apify/e-commerce-scraping-tool. Two shapes are handled:
    * search-engine 'Sellers' mode: {"product": {...}, "sellers": [{merchantName, price, url, shipping}]}
    * product rows (schema.org-like): {name, url, brand, offers: {price, priceCurrency, ...}, ...}
    """
    if isinstance(item.get("sellers"), list):
        prod = item.get("product") or {}
        title = prod.get("title") or item.get("title") or item.get("name") or ""
        out = []
        for s in item["sellers"]:
            if not isinstance(s, dict):
                continue
            url = s.get("url") or s.get("link")
            price, cur = parse_price(s.get("price"))
            seller = s.get("merchantName") or s.get("name") or s.get("seller")
            host = marketplace_from_url(url)
            if not host or "google." in host:        # google redirect links -> use the merchant name
                host = (seller or "google_shopping").lower()
            out.append(Offer(
                provider="apify", marketplace=host, title=s.get("title") or title, url=url,
                asin=extract_asin(url), price=price, currency=cur or s.get("currency"),
                seller_name=seller, shipping=s.get("shipping"), condition=s.get("condition"),
                rating=parse_float(s.get("rating")),
                offers=[str(s["offer"])] if s.get("offer") else [], raw=s))
        return out

    title = item.get("name") or item.get("title") or ""
    if not title:
        return []
    offers_field = item.get("offers")
    off_list = offers_field if isinstance(offers_field, list) else ([offers_field] if isinstance(offers_field, dict) else [{}])
    brand = item.get("brand")
    brand = brand if isinstance(brand, str) else (brand or {}).get("name")
    agg = item.get("aggregateRating") or {}
    url = item.get("url")
    gtin = next((str(item[k]) for k in ("gtin", "gtin13", "gtin12", "gtin14", "ean", "upc") if item.get(k)), None)
    out = []
    for off in off_list:
        off = off or {}
        price, cur = parse_price(off.get("price") or off.get("lowPrice") or item.get("price"))
        seller = off.get("seller")
        seller = seller.get("name") if isinstance(seller, dict) else seller
        availability = off.get("availability") or item.get("availability")
        out.append(Offer(
            provider="apify", marketplace=marketplace_from_url(url) or "", title=title, url=url,
            external_id=str(item.get("sku") or item.get("productID") or "") or None,
            asin=extract_asin(item.get("asin"), url), gtin=gtin, brand=brand if isinstance(brand, str) else None,
            price=price, currency=off.get("priceCurrency") or cur or item.get("currency"),
            rating=parse_float(agg.get("ratingValue") or item.get("rating") or item.get("stars")),
            review_count=parse_int(agg.get("reviewCount") or agg.get("ratingCount") or item.get("reviewsCount")),
            seller_name=seller, availability=str(availability).rsplit("/", 1)[-1] if availability else None,
            raw=_slim(item)))
    return out


def ecom_search_spec(actor_id: str, *, country: str = "in", mode: str = "Products",
                     sellers_per_product: int = 10, extra: dict | None = None) -> ActorSpec:
    # Field names follow the actor's Input tab. Its README uses slightly different names
    # (SearchEngineSearchKeyword / scrapeSellersFromSearchEngine); if a run ignores your keyword, copy
    # the JSON from the Console "Input" tab and put the differences in APIFY_EXTRA_INPUT_JSON.
    def build(query: str, limit: int) -> dict[str, Any]:
        inp = {"searchEngineKeyword": query, "scrapeModeSearchEngine": mode, "countryCode": country,
               "additionalPropertiesSearchEngine": True, "maxSearchEngineProducts": limit,
               "maxSearchEngineSellersPerProduct": sellers_per_product, "maxSearchEngineResults": limit}
        normalized_mode = mode.strip().lower()
        # Keep the current mode field and the actor's documented compatibility flags.
        # Exactly one mode is enabled so Products cannot silently become Sellers.
        inp["scrapeProductsFromSearchEngine"] = normalized_mode == "products"
        inp["scrapeSellersFromSearchEngine"] = normalized_mode == "sellers"
        inp.update(extra or {})
        return inp
    return ActorSpec("ecom_search", actor_id, "discover", build, normalize_ecom_item)


# --------------------------------------------------------------------------- 2) Flipkart product details
def normalize_flipkart_item(item: dict[str, Any]) -> list[Offer]:
    """piotrv1001/flipkart-product-details-scraper (one row per product; offers[] nested)."""
    if item.get("status") not in (None, "ok") or not item.get("title"):
        return []
    promos: list[Promo] = []
    for off in item.get("offers") or []:
        if isinstance(off, dict):
            text = " ".join(str(x) for x in (off.get("title"), off.get("description"), off.get("value")) if x)
            hint = off.get("type")
        else:
            text, hint = str(off), None
        if text.strip():
            promos.append(classify_promo(text, hint, source="flipkart_details"))
    if item.get("specialPrice"):
        promos.append(Promo("SPECIAL_PRICE", "Special price", source="flipkart_details"))
    warranty = item.get("warranty")
    if isinstance(warranty, dict) and warranty:
        warranty = warranty.get("Warranty Summary") or next(iter(warranty.values()))
    delivery = " ".join(str(x) for x in (item.get("deliveryBy"), item.get("deliveryDays") and f"({item['deliveryDays']})") if x) or None
    return [Offer(
        provider="apify", marketplace="flipkart.com", title=item["title"], url=item.get("url"),
        external_id=item.get("pid"), brand=item.get("brand"),
        price=parse_price(item.get("price"))[0], currency=item.get("currency") or "INR",
        original_price=parse_price(item.get("mrp"))[0], price_with_offers=parse_price(item.get("priceWithOffers"))[0],
        rating=parse_float(item.get("rating")), review_count=parse_int(item.get("ratingCount")),
        seller_name=item.get("sellerName"), seller_id=item.get("sellerId"),
        seller_rating=parse_float(item.get("sellerRating")),
        availability=item.get("availability"), shipping=delivery, delivery_by=item.get("deliveryBy"),
        is_assured=item.get("isFlipkartAssured"), cod_available=item.get("codAvailable"),
        no_cost_emi=item.get("noCostEmi"), return_policy=item.get("returnPolicy"),
        warranty=warranty if isinstance(warranty, str) else None,
        offers=[p.description for p in promos], promos=promos, raw=_slim(item))]


def flipkart_details_spec(actor_id: str, extra: dict | None = None) -> ActorSpec:
    def build(urls: list[str], limit: int) -> dict[str, Any]:
        return {"productUrls": urls, "maxItems": len(urls), **(extra or {})}
    return ActorSpec("flipkart_details", actor_id, "enrich", build, normalize_flipkart_item,
                     hosts=("flipkart.com",))


# --------------------------------------------------------------------------- 3) bank-offer aggregator
def normalize_bank_offers_item(item: dict[str, Any]) -> list[Offer]:
    """pale_tapestry/bank-offers-aggregator-actor -> {title, price, offers:[{rawText, bank, cardType, isEmi, discountPercent}]}"""
    promos: list[Promo] = []
    for off in item.get("offers") or []:
        raw = off.get("rawText") if isinstance(off, dict) else str(off)
        if not raw:
            continue
        p = classify_promo(raw, "BANK", source="bank_offers")
        if isinstance(off, dict):
            p.bank = off.get("bank") or p.bank
            p.card_type = off.get("cardType") or p.card_type
            if off.get("isEmi") is not None:
                p.is_emi = bool(off["isEmi"])
            if off.get("discountPercent") is not None:
                p.percent = float(off["discountPercent"])
        promos.append(p)
    url = item.get("url") or item.get("productUrl")
    return [Offer(provider="apify", marketplace=marketplace_from_url(url) or "", title=item.get("title") or "",
                  url=url, price=parse_price(item.get("price"))[0], offers=[p.description for p in promos],
                  promos=promos, raw=_slim(item))]


def bank_offers_spec(actor_id: str, extra: dict | None = None) -> ActorSpec:
    def build(urls: list[str], limit: int) -> dict[str, Any]:
        return {"productUrls": urls, "maxConcurrency": 1, "useProxy": False, **(extra or {})}
    return ActorSpec("bank_offers", actor_id, "enrich", build, normalize_bank_offers_item,
                     hosts=("flipkart.com", "amazon.in"))


# --------------------------------------------------------------------------- matching / merging
def _host(url: str | None) -> str | None:
    h = marketplace_from_url(url)
    return "flipkart.com" if h and "flipkart" in h else h


def offer_key(o: Offer) -> str | None:
    """Identity of a listing across actors: ASIN, Flipkart pid, or host+path."""
    if o.asin:
        return f"asin:{o.asin}"
    if o.external_id and "flipkart" in (o.marketplace or ""):
        return f"flipkart.com:pid:{o.external_id}"
    if o.url:
        u = urlparse(o.url)
        pid = parse_qs(u.query).get("pid")
        if pid:
            return f"{_host(o.url)}:pid:{pid[0]}"
        return f"{_host(o.url)}{u.path.rstrip('/')}"
    return None


def find_match(rich: Offer, offers: list[Offer], threshold: float = 0.85) -> Offer | None:
    key = offer_key(rich)
    if key:
        for o in offers:
            if offer_key(o) == key:
                return o
    best, score = None, 0.0
    for o in offers:
        if rich.marketplace and o.marketplace != rich.marketplace:
            continue
        s = title_similarity(rich.title, o.title) if rich.title and o.title else 0.0
        if s > score:
            best, score = o, s
    return best if score >= threshold else None


_RICH_WINS = {"price", "currency", "original_price", "rating", "review_count", "seller_name", "seller_id",
              "seller_rating", "availability", "shipping", "brand", "gtin", "asin", "external_id",
              "price_with_offers", "is_assured", "cod_available", "no_cost_emi", "return_policy",
              "delivery_by", "warranty", "condition"}


def merge_offer(base: Offer, rich: Offer) -> Offer:
    """Copy detail-level data from `rich` onto `base` (rich wins where it has a value)."""
    for f in fields(Offer):
        if f.name in _RICH_WINS:
            v = getattr(rich, f.name)
            if v is not None and v != "":
                setattr(base, f.name, v)
    base.offers = list(dict.fromkeys(base.offers + rich.offers))
    seen = {(p.promo_type, p.description) for p in base.promos}
    base.promos += [p for p in rich.promos if (p.promo_type, p.description) not in seen]
    base.raw.setdefault("enrichment", []).append(rich.raw)
    return base


def apply_effective_price(o: Offer) -> None:
    """If no source gave an effective price, estimate it from the best single bank offer."""
    if o.price_with_offers is None and o.price:
        disc = best_instant_bank_discount(o.price, o.promos)
        if disc > 0:
            o.price_with_offers = round(o.price - disc, 2)


# --------------------------------------------------------------------------- provider
class ApifyProvider:
    name = "apify"

    def __init__(self, token: str, *, specs: list[ActorSpec] | None = None, enrich: bool = True,
                 enrich_limit: int = 5, client: Any | None = None):
        if not token and client is None:
            raise ApifyError("APIFY_API_TOKEN is not set")
        if client is None:
            from apify_client import ApifyClient  # lazy import
            client = ApifyClient(token)
        self.client = client
        self.specs = specs or []
        self.enrich = enrich
        self.enrich_limit = enrich_limit
        self.warnings: list[str] = []

    def search(self, query: str, limit: int = 20) -> list[Offer]:
        self.warnings = []
        discover = [s for s in self.specs if s.stage == "discover"]
        if not discover:
            raise ApifyError("No discovery actor configured (APIFY_ECOM_ACTOR)")
        offers: list[Offer] = []
        failed = 0
        for spec in discover:
            try:
                offers += self._run(spec, query, limit, item_cap=limit)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                self.warnings.append(f"{spec.key}: {exc}")
        if failed == len(discover) and not offers:
            raise ApifyError("; ".join(self.warnings))
        if self.enrich and offers:
            offers = self._enrich(offers)
        for o in offers:
            apply_effective_price(o)
        return offers

    # -- internals ---------------------------------------------------------
    def _run(self, spec: ActorSpec, arg: Any, limit: int, item_cap: int) -> list[Offer]:
        run = self.client.actor(spec.actor_id).call(run_input=spec.build_input(arg, limit))
        if not run:
            raise ApifyError(f"Actor {spec.actor_id} returned no run")
        check_run_succeeded(run, spec.actor_id)
        dataset_id = dataset_id_of(run)
        if not dataset_id:
            raise ApifyError(f"Actor {spec.actor_id} run has no default dataset")
        out: list[Offer] = []
        for i, item in enumerate(self.client.dataset(dataset_id).iterate_items()):
            if i >= item_cap:
                break
            out.extend(spec.normalize(item))
        return out

    def _enrich(self, offers: list[Offer]) -> list[Offer]:
        result = list(offers)
        for spec in (s for s in self.specs if s.stage == "enrich"):
            for host in spec.hosts:
                cands = [o for o in result if o.url and _host(o.url) == host]
                # cheapest first: those are the listings worth spending detail-scrape credits on
                cands.sort(key=lambda o: (o.price is None, o.price or 0))
                urls = list(dict.fromkeys(o.url for o in cands))[: self.enrich_limit]
                if not urls:
                    continue
                # This community actor's documented output can omit the source URL.
                # Run one URL at a time so a promotion is bound to an exact listing
                # by the input URL/ASIN/PID rather than guessed from a title batch.
                if spec.key == "bank_offers":
                    for source_url in urls:
                        try:
                            rich = self._run(spec, [source_url], 1, item_cap=2)
                        except Exception as exc:  # noqa: BLE001
                            self.warnings.append(f"{spec.key} ({host}, {source_url}): {exc}")
                            continue
                        source_key = offer_key(next(
                            o for o in cands if o.url == source_url
                        ))
                        for r in rich:
                            r.url = r.url or source_url
                            r.marketplace = r.marketplace or host
                            r.asin = r.asin or extract_asin(source_url)
                            target = next((o for o in result if offer_key(o) == source_key), None)
                            if target is None:
                                self.warnings.append(
                                    f"{spec.key} ({host}): result could not be bound to {source_url}")
                                continue
                            merge_offer(target, r)
                    continue
                try:
                    rich = self._run(spec, urls, len(urls), item_cap=len(urls) * 2)
                except Exception as exc:  # noqa: BLE001
                    self.warnings.append(f"{spec.key} ({host}): {exc}")
                    continue
                for r in rich:
                    r.marketplace = r.marketplace or host
                    target = find_match(r, result)
                    if target is not None:
                        merge_offer(target, r)
                    else:
                        result.append(r)
        return result


def specs_from_settings(s: Any) -> list[ActorSpec]:
    """Build the actor list from Settings (see .env.example)."""
    extra = s.apify_extra_input or {}
    specs = [ecom_search_spec(s.apify_ecom_actor, country=s.country, mode=s.apify_ecom_mode,
                              extra=extra.get("ecom_search"))]
    if "flipkart_details" in s.apify_enrichers:
        specs.append(flipkart_details_spec(s.apify_flipkart_details_actor, extra.get("flipkart_details")))
    if "bank_offers" in s.apify_enrichers:
        specs.append(bank_offers_spec(s.apify_bank_offers_actor, extra.get("bank_offers")))
    return specs
