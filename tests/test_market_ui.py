from tools import market_ui


def row(
    canonical_id,
    marketplace,
    seller,
    price,
    currency="INR",
    original=None,
    rating=None,
    reviews=None,
    title="iPhone 16",
):
    return {
        "offer_id": f"{canonical_id}-{marketplace}-{seller}",
        "canonical_id": canonical_id,
        "product_title": title,
        "title": title,
        "marketplace": marketplace,
        "seller_name": seller,
        "price": price,
        "price_with_offers": None,
        "currency": currency,
        "original_price": original,
        "rating": rating,
        "review_count": reviews,
        "offers_json": ["10% OFF"],
        "url": "https://example.test/product",
        "provider": "serpapi",
    }


ROWS = market_ui.enrich_rows(
    [
        row("B0A", "amazon.in", "Appario", 69900, original=79900, rating=4.5),
        row("B0A", "flipkart.com", "RetailNet", 67900, rating=4.6),
        row("B0B", "croma.com", "Croma", 70900),
    ]
)


def test_enrichment_and_summary_use_effective_indian_prices():
    assert ROWS[0]["offer_labels"] == ["10% OFF"]
    assert ROWS[0]["discount_percent"] == 12.5

    summary = market_ui.summarize(ROWS)
    assert summary["currency"] == "INR"
    assert summary["cheapest"]["seller_name"] == "RetailNet"
    assert summary["marketplaces"] == 3


def test_filters_and_product_groups_use_main_canonical_id():
    assert len(market_ui.filter_rows(ROWS, canonical_id="B0A")) == 2
    assert len(market_ui.filter_rows(ROWS, marketplaces=["croma.com"])) == 1
    assert len(market_ui.filter_rows(ROWS, minimum_rating=4.6)) == 1

    groups = market_ui.group_products(ROWS)
    assert groups[0]["canonical_id"] == "B0A"
    assert groups[0]["offers"] == 2


def test_offer_and_promotion_tables_preserve_provider_identity():
    offer = market_ui.offers_table(ROWS)[0]
    assert offer["Offers"] == "10% OFF"
    assert offer["Link"] == "https://example.test/product"

    promotions = [
        {
            "offer_id": ROWS[0]["offer_id"],
            "canonical_id": "B0A",
            "product_title": "iPhone 16",
            "external_id": "provider-123",
            "marketplace": "amazon.in",
            "seller_name": "Appario",
            "promotion_type": "BANK",
            "bank": "HDFC Bank",
            "card_type": "Credit Card",
            "description": "10% instant discount",
            "amount": 1500,
            "percent": 10,
            "is_emi": False,
            "source": "serpapi_amazon_product",
            "url": "https://example.test/product",
        }
    ]
    table = market_ui.promotions_table(promotions, {ROWS[0]["offer_id"]})
    assert table[0]["Product ID"] == "B0A"
    assert table[0]["Provider product ID"] == "provider-123"
    assert market_ui.promotion_summary(promotions) == {"BANK": 1}


def test_empty_market_summary():
    summary = market_ui.summarize([])
    assert summary["cheapest"] is None
    assert summary["average"] is None
    assert summary["offers"] == 0
