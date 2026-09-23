"""Normalisers are tested against the sample outputs published on each actor's Apify page."""
from tools.market_models import Offer
from tools.market_apify_provider import (
    ApifyError, ApifyProvider, bank_offers_spec, ecom_search_spec, find_match, flipkart_details_spec,
    merge_offer, normalize_bank_offers_item, normalize_ecom_item, normalize_flipkart_item, offer_key)

FLIPKART_URL = "https://www.flipkart.com/apple-iphone-16-black-128-gb/p/itmb07d67f995271?pid=MOBH4DQFG8NKFRDY"

# sample from piotrv1001/flipkart-product-details-scraper (trimmed)
FLIPKART_ITEM = {
    "pid": "MOBH4DQFG8NKFRDY", "url": FLIPKART_URL, "title": "iPhone 16 (Black, 128 GB)", "brand": "Apple",
    "price": 69900, "mrp": 69900, "priceWithOffers": 66405, "currency": "INR", "availability": "IN_STOCK",
    "isFlipkartAssured": True, "codAvailable": True, "noCostEmi": False,
    "returnPolicy": "7_day_replacement_service_centre_action", "deliveryBy": "Friday, 18 Sep", "deliveryDays": "3d",
    "sellerId": "f6d15b4a4f304426", "sellerName": "TrueComRetail", "sellerRating": 4.6,
    "offers": [
        {"type": "EXCHANGE", "title": "Exchange offer", "description": "Change pincode to exchange item", "value": "Up to ₹40,900"},
        {"type": "PBO", "title": "Bank offers", "description": None, "value": "₹3,495 off"}],
    "warranty": {"Warranty Summary": "1 year warranty for phone and 1 year for in Box Accessories."},
    "rating": 4.6, "ratingCount": 198569, "reviewCount": 8581, "status": "ok",
    "reviews": [{"text": "bulky field that should not be stored"}]}

# sample from pale_tapestry/bank-offers-aggregator-actor (no url in the documented sample)
BANK_ITEM = {"title": "Apple iPhone 16 (Black, 128 GB)", "price": 69900, "offers": [
    {"rawText": "10% off on HDFC Bank Credit Card EMI, up to ₹1,500", "bank": "HDFC", "cardType": "Credit Card",
     "isEmi": True, "discountPercent": 10}]}

SELLERS_ITEM = {"product": {"title": "Apple iPhone 16 128GB Black", "price": "₹69,900"}, "sellers": [
    {"merchantName": "Flipkart", "price": "₹68,999", "url": FLIPKART_URL, "shipping": "Free delivery"},
    {"merchantName": "Croma", "price": "₹69,900", "url": "https://www.croma.com/apple-iphone-16/p/300001"},
    {"merchantName": "Amazon.in", "price": "₹67,900", "url": "https://www.amazon.in/dp/B0DGJ6Q7LM?tag=x"}]}


def test_flipkart_normaliser_maps_seller_offers_and_flags():
    [o] = normalize_flipkart_item(FLIPKART_ITEM)
    assert (o.price, o.original_price, o.price_with_offers) == (69900, 69900, 66405)
    assert (o.seller_name, o.seller_id, o.seller_rating) == ("TrueComRetail", "f6d15b4a4f304426", 4.6)
    assert o.is_assured and o.cod_available and o.no_cost_emi is False
    assert o.external_id == "MOBH4DQFG8NKFRDY" and o.review_count == 198569
    assert o.warranty.startswith("1 year") and "Friday" in o.shipping
    assert {(p.promo_type, p.amount) for p in o.promos} == {("EXCHANGE", 40900), ("BANK", 3495)}
    assert "reviews" not in o.raw                       # bulky fields are not stored
    assert normalize_flipkart_item({"pid": "X", "status": "not_found"}) == []


def test_bank_offer_normaliser_prefers_actor_fields():
    [o] = normalize_bank_offers_item(BANK_ITEM)
    p = o.promos[0]
    assert (p.promo_type, p.bank, p.card_type, p.is_emi, p.percent, p.amount) == \
        ("BANK", "HDFC", "Credit Card", True, 10.0, 1500.0)


def test_ecom_sellers_mode_yields_one_offer_per_seller_with_asin():
    offers = normalize_ecom_item(SELLERS_ITEM)
    assert [o.seller_name for o in offers] == ["Flipkart", "Croma", "Amazon.in"]
    assert [o.marketplace for o in offers] == ["flipkart.com", "croma.com", "amazon.in"]
    assert offers[2].asin == "B0DGJ6Q7LM" and offers[2].price == 67900 and offers[0].shipping == "Free delivery"


def test_ecom_schemaorg_product_row():
    item = {"url": "https://www.amazon.in/dp/B0DGJ6Q7LM", "name": "iPhone 16", "brand": {"slogan": None},
            "offers": {"price": 67900, "priceCurrency": "INR", "availability": "https://schema.org/InStock",
                       "seller": {"name": "Appario Retail"}},
            "aggregateRating": {"ratingValue": "4.5", "reviewCount": "1,200"}, "gtin13": "0194253000000"}
    [o] = normalize_ecom_item(item)
    assert (o.asin, o.price, o.currency, o.seller_name, o.availability) == \
        ("B0DGJ6Q7LM", 67900, "INR", "Appario Retail", "InStock")
    assert (o.rating, o.review_count, o.gtin, o.brand) == (4.5, 1200, "0194253000000", None)


def test_offer_key_and_merge():
    thin = Offer("apify", "flipkart.com", "Apple iPhone 16 128GB Black", url=FLIPKART_URL.split("?")[0] + "?pid=MOBH4DQFG8NKFRDY&lid=x",
                 price=68999, offers=["Free delivery"])
    [rich] = normalize_flipkart_item(FLIPKART_ITEM)
    assert offer_key(thin) == offer_key(rich) == "flipkart.com:pid:MOBH4DQFG8NKFRDY"
    assert find_match(rich, [thin]) is thin
    merge_offer(thin, rich)
    assert thin.seller_name == "TrueComRetail" and thin.price == 69900 and thin.price_with_offers == 66405
    assert thin.promos and "Free delivery" in thin.offers and thin.is_assured


# ------------------------------------------------------------------ provider flow with a fake Apify client
class RunV2:                                            # apify-client >= 2 returns an object, not a dict
    def __init__(self, ds, status="SUCCEEDED"):
        self.id, self.status, self.default_dataset_id = "run1", status, ds


class FakeClient:
    def __init__(self, datasets, fail=(), status="SUCCEEDED"):
        self.datasets, self.fail, self.status, self.calls, self._actor, self._ds = datasets, set(fail), status, [], None, None

    def actor(self, actor_id): self._actor = actor_id; return self
    def call(self, run_input):
        self.calls.append((self._actor, run_input))
        if self._actor in self.fail:
            raise RuntimeError("boom")
        return RunV2(self._actor, self.status)
    def dataset(self, ds): self._ds = ds; return self
    def iterate_items(self): yield from self.datasets.get(self._ds, [])


def _provider(client, enrich=True, limit=5):
    specs = [ecom_search_spec("ecom"), flipkart_details_spec("fk"), bank_offers_spec("bank")]
    return ApifyProvider("", specs=specs, enrich=enrich, enrich_limit=limit, client=client)


def test_two_stage_flow_enriches_and_computes_effective_price():
    client = FakeClient({"ecom": [SELLERS_ITEM], "fk": [FLIPKART_ITEM], "bank": [BANK_ITEM]})
    offers = _provider(client).search("iphone 16", 10)
    by_mkt = {o.marketplace: o for o in offers}
    fk = by_mkt["flipkart.com"]
    assert fk.seller_name == "TrueComRetail" and fk.is_assured           # from flipkart_details
    types = {p.promo_type for p in fk.promos}
    assert {"EXCHANGE", "BANK"} <= types and any(p.bank == "HDFC" for p in fk.promos)   # + bank_offers merged
    assert fk.price_with_offers == 66405                                # actor-provided value kept
    assert by_mkt["croma.com"].promos == []                             # not a supported enrich host
    # discovery ran once; details only for flipkart.com; bank actor per host (flipkart.com and amazon.in)
    actors = [a for a, _ in client.calls]
    assert actors.count("ecom") == 1 and actors.count("fk") == 1 and actors.count("bank") == 2
    assert dict(client.calls)["fk"]["productUrls"] == [FLIPKART_URL]
    assert dict(client.calls)["ecom"]["searchEngineKeyword"] == "iphone 16"


def test_effective_price_estimated_from_bank_offer_when_missing():
    item = {**FLIPKART_ITEM}
    item.pop("priceWithOffers")
    client = FakeClient({"ecom": [SELLERS_ITEM], "fk": [item], "bank": []})
    fk = {o.marketplace: o for o in _provider(client).search("iphone 16", 10)}["flipkart.com"]
    assert fk.price_with_offers == 69900 - 3495


def test_enrich_failure_is_a_warning_not_an_error():
    client = FakeClient({"ecom": [SELLERS_ITEM]}, fail={"fk", "bank"})
    p = _provider(client)
    offers = p.search("iphone 16", 10)
    assert len(offers) == 3 and any("flipkart_details" in w for w in p.warnings)


def test_enrich_limit_and_disable():
    many = {"product": {"title": "x"}, "sellers": [
        {"merchantName": f"s{i}", "price": f"₹{60000 + i}", "url": f"https://www.flipkart.com/p{i}/p/itm{i}"} for i in range(8)]}
    client = FakeClient({"ecom": [many], "fk": [], "bank": []})
    _provider(client, limit=3).search("q", 10)
    assert len(dict(client.calls)["fk"]["productUrls"]) == 3
    bank_inputs = [inp for actor, inp in client.calls if actor == "bank"]
    assert len(bank_inputs) == 3 and all(len(inp["productUrls"]) == 1 for inp in bank_inputs)
    client2 = FakeClient({"ecom": [many]})
    _provider(client2, enrich=False).search("q", 10)
    assert [a for a, _ in client2.calls] == ["ecom"]


def test_all_discovery_failing_raises_and_failed_run_status_is_clear():
    try:
        _provider(FakeClient({}, fail={"ecom"})).search("q", 5)
    except ApifyError as e:
        assert "ecom_search" in str(e)
    else:
        raise AssertionError("expected ApifyError")
    try:
        _provider(FakeClient({}, status="FAILED")).search("q", 5)
    except ApifyError as e:
        assert "FAILED" in str(e) and "run1" in str(e)
    else:
        raise AssertionError("expected ApifyError")
