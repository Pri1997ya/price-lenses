from tools.market_matching import ProductResolver
from tools.market_models import Offer
from tools.market_normalize import (extract_asin, normalize_title, parse_int, parse_price,
                                    search_query_from_url)


def test_extract_asin_from_urls():
    assert extract_asin("https://www.amazon.in/Apple-iPhone-17-Pro/dp/B0FQFB8FMG/ref=sr_1_1?x=1") == "B0FQFB8FMG"
    assert extract_asin("https://www.amazon.in/gp/product/B0FQFB8FMG") == "B0FQFB8FMG"
    assert extract_asin("b0fqfb8fmg") == "B0FQFB8FMG"
    assert extract_asin("https://www.flipkart.com/apple-iphone/p/itm123") is None
    assert extract_asin(None, "") is None


def test_parse_price():
    assert parse_price("₹1,34,900.00") == (134900.0, "INR")
    assert parse_price("$1,099") == (1099.0, "USD")
    assert parse_price(1299) == (1299.0, None)
    assert parse_price({"value": 99.5, "currency": "USD"}) == (99.5, "USD")
    assert parse_price("N/A") == (None, None)


def test_parse_int():
    assert parse_int("1,234") == 1234
    assert parse_int("(2.5K)") == 2500
    assert parse_int(None) is None


def test_asin_offers_share_one_product():
    r = ProductResolver()
    a = Offer("serpapi", "amazon.in", "Apple iPhone 17 Pro 256GB", asin="B0FQFB8FMG")
    b = Offer("apify", "amazon.in", "iPhone 17 Pro (256 GB) Apple", asin="B0FQFB8FMG")
    assert r.resolve(a) == r.resolve(b) == "asin:B0FQFB8FMG"


def test_title_fallback_attaches_to_known_product():
    r = ProductResolver()
    pid = r.resolve(Offer("serpapi", "amazon.in", "Apple iPhone 17 Pro 256GB Deep Blue", asin="B0FQFB8FMG"))
    other = r.resolve(Offer("apify", "flipkart.com", "Apple iPhone 17 Pro (Deep Blue, 256 GB)"))
    assert other == pid


def test_different_storage_is_a_different_product():
    r = ProductResolver()
    a = r.resolve(Offer("serpapi", "x", "Apple iPhone 17 Pro 256GB", asin="B0AAAAAAA1"))
    b = r.resolve(Offer("apify", "flipkart.com", "Apple iPhone 17 Pro 512GB"))
    assert a != b


def test_normalize_title():
    assert normalize_title("The New iPhone 17 Pro (5G)!") == "iphone 17 pro"


def test_product_query_from_retailer_url():
    url = "https://www.flipkart.com/apple-iphone-16-black-128-gb/p/itm123?pid=MOB123"
    assert search_query_from_url(url) == "apple iphone 16 black 128 gb"
