import requests
import pytest

from tools.market_serpapi_provider import SerpApiError, SerpApiProvider

GOOGLE = {"shopping_results": [{
    "title": "Apple iPhone 17 Pro 256GB", "product_id": "123",
    "link": "https://www.amazon.in/dp/B0FQFB8FMG?tag=x", "source": "Amazon.in",
    "extracted_price": 134900, "price": "₹1,34,900", "old_price": "₹1,49,900",
    "rating": 4.6, "reviews": 1200, "delivery": "Free delivery", "tag": "10% OFF"}]}

AMAZON = {"organic_results": [
    {"asin": "B0FQFB8FMG", "title": "Apple iPhone 17 Pro", "link_clean": "https://www.amazon.in/dp/B0FQFB8FMG",
     "extracted_price": 134900, "price": "₹1,34,900", "rating": 4.5, "reviews": 3400, "prime": True,
     "save_with_coupon": "Save ₹2,000 with coupon", "offers": ["10% off on HDFC Credit Card"]},
    {"asin": "B0SPONSORED", "title": "Ad", "sponsored": True}]}

AMAZON_PRODUCT = {
    "product_results": {
        "title": "Apple iPhone 16 128GB Black", "brand": "Apple",
        "extracted_price": 69900, "price": "₹69,900", "seller": {"name": "Appario"},
        "promotions": [{"type": "card", "text": "10% off on HDFC Credit Card up to ₹1,500"}],
    },
    "other_sellers": [{"extracted_price": 70900, "seller": {"name": "Seller Two"}, "rating": 4.2}],
}


def test_serpapi_google_parse():
    p = SerpApiProvider("k")
    o = p.parse_google_shopping(GOOGLE)[0]
    assert o.asin == "B0FQFB8FMG" and o.price == 134900 and o.currency == "INR"
    assert o.original_price == 149900 and o.seller_name == "Amazon.in"
    assert o.marketplace == "amazon.in" and o.review_count == 1200


def test_serpapi_amazon_parse_skips_sponsored():
    p = SerpApiProvider("k")
    offers = p.parse_amazon_search(AMAZON)
    assert len(offers) == 1 and offers[0].asin == "B0FQFB8FMG" and "Prime" in offers[0].offers
    assert {promo.promo_type for promo in offers[0].promos} == {"COUPON", "BANK"}


def test_amazon_product_parses_seller_other_sellers_and_promotions():
    p = SerpApiProvider("k")
    offers = p.parse_amazon_product(AMAZON_PRODUCT, "B0DGJ6Q7LM")
    assert [o.seller_name for o in offers] == ["Appario", "Seller Two"]
    assert offers[0].price == 69900 and offers[0].promos[0].bank == "HDFC Bank"
    assert offers[1].price == 70900 and offers[1].seller_rating == 4.2


def test_google_shopping_excludes_unapproved_retailers():
    p = SerpApiProvider("k")
    data = {"shopping_results": GOOGLE["shopping_results"] + [{
        "title": "Apple iPhone", "link": "https://example.com/iphone",
        "source": "Unknown Shop", "extracted_price": 1,
    }]}
    assert [o.marketplace for o in p.parse_google_shopping(data)] == ["amazon.in"]


class _Response:
    def __init__(self, data): self.data = data
    def raise_for_status(self): return None
    def json(self): return self.data


class _Session:
    def __init__(self): self.calls, self.timeouts = [], []
    def get(self, url, params, timeout):
        self.calls.append(params.copy())
        self.timeouts.append(timeout)
        data = {"amazon_product": AMAZON_PRODUCT, "amazon": AMAZON,
                "google_shopping": GOOGLE}[params["engine"]]
        return _Response(data)


def test_amazon_url_uses_search_then_searches_across_stores():
    session = _Session()
    p = SerpApiProvider("k", session=session)
    offers = p.search("https://www.amazon.in/dp/B0FQFB8FMG", 10)
    assert [call["engine"] for call in session.calls] == ["amazon", "google_shopping"]
    assert session.calls[0]["k"] == "B0FQFB8FMG"
    assert session.calls[1]["q"] == "Apple iPhone 17 Pro"
    assert session.timeouts == [(10, 120), (10, 120)]
    assert {o.marketplace for o in offers} == {"amazon.in"}


def test_amazon_product_is_optional_enrichment_after_search():
    session = _Session()
    p = SerpApiProvider("k", session=session, enrich_amazon=True)
    offers = p.search("https://www.amazon.in/dp/B0FQFB8FMG", 10)
    assert [call["engine"] for call in session.calls] == [
        "amazon", "google_shopping", "amazon_product",
    ]
    assert all(o.seller_name == "Appario" for o in offers if o.marketplace == "amazon.in")


class _TimeoutSession(_Session):
    def __init__(self, failed_engines):
        super().__init__()
        self.failed_engines = set(failed_engines)

    def get(self, url, params, timeout):
        if params["engine"] in self.failed_engines:
            raise requests.ReadTimeout("slow upstream")
        return super().get(url, params, timeout)


def test_one_serpapi_engine_timeout_preserves_partial_results():
    p = SerpApiProvider("k", session=_TimeoutSession({"google_shopping"}), read_timeout=7)
    offers = p.search("iPhone 17 Pro", 10)
    assert len(offers) == 1 and offers[0].marketplace == "amazon.in"
    assert p.warnings and "timed out after 7s" in p.warnings[0]


def test_all_serpapi_engines_timing_out_fails_clearly():
    p = SerpApiProvider("k", session=_TimeoutSession({"google_shopping", "amazon"}))
    with pytest.raises(SerpApiError, match="google_shopping.*amazon_search"):
        p.search("iPhone 17 Pro", 10)
