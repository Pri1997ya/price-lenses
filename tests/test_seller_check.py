from datetime import datetime, timedelta, timezone

import pytest

from tools.seller_check import (
    AVOID,
    CAUTION,
    IN_STOCK,
    LOW_STOCK,
    OK,
    OUT_OF_STOCK,
    PREORDER,
    TRUSTED,
    UNKNOWN,
    assess_seller,
    check_sellers,
    classify_stock,
    normalize_rating,
)

NOW = datetime(2026, 9, 24, 12, tzinfo=timezone.utc)


def row(**overrides):
    base = {
        "offer_id": "o1",
        "marketplace": "amazon.in",
        "seller_name": "Appario Retail",
        "seller_rating": 4.5,
        "availability": "In stock",
        "price": 70000.0,
        "price_with_offers": None,
        "is_assured": None,
        "item_condition": None,
        "fetched_at": NOW - timedelta(hours=2),
        "url": "https://www.amazon.in/dp/B0CS5XW6TN",
        "title": "Samsung Galaxy S24 Ultra",
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    "text, expected",
    [
        ("In stock", (IN_STOCK, None)),
        ("InStock", (IN_STOCK, None)),
        ("IN_STOCK", (IN_STOCK, None)),
        ("Only 3 left in stock.", (LOW_STOCK, 3)),
        ("LimitedAvailability", (LOW_STOCK, None)),
        ("Currently unavailable.", (OUT_OF_STOCK, None)),
        ("OutOfStock", (OUT_OF_STOCK, None)),
        ("Sold Out", (OUT_OF_STOCK, None)),
        ("Not available", (OUT_OF_STOCK, None)),
        ("PreOrder", (PREORDER, None)),
        ("Coming soon", (PREORDER, None)),
        (None, (UNKNOWN, None)),
        ("", (UNKNOWN, None)),
        ("Blue colour", (UNKNOWN, None)),
    ],
)
def test_classify_stock(text, expected):
    assert classify_stock(text) == expected


def test_normalize_rating():
    assert normalize_rating(4.2) == 4.2
    assert normalize_rating(92) == 4.6
    assert normalize_rating("bad") is None
    assert normalize_rating(500) is None


@pytest.mark.parametrize(
    "overrides, tier",
    [
        ({"marketplace": "croma.com", "seller_name": "Croma", "seller_rating": None}, TRUSTED),
        ({"marketplace": "flipkart.com", "is_assured": True, "seller_rating": None}, TRUSTED),
        ({}, OK),
        ({"seller_rating": 3.5}, CAUTION),
        ({"seller_rating": 2.1}, AVOID),
        ({"seller_name": "Amazon.in", "seller_rating": None}, CAUTION),
        ({"item_condition": "RENEWED"}, CAUTION),
        ({"marketplace": "croma.com", "item_condition": "OPEN_BOX"}, CAUTION),
    ],
)
def test_seller_tiers(overrides, tier):
    assert assess_seller(row(**overrides))[0] == tier


def test_trusted_seller_allowlist():
    tier, reasons = assess_seller(row(seller_rating=None), trusted_names=("appario",))
    assert tier == TRUSTED
    assert "TRUSTED_SELLERS" in reasons[0]


def test_report_picks_cheapest_and_safest_and_warns():
    rows = [
        row(offer_id="oos", price=60000, availability="Currently unavailable"),
        row(offer_id="shady", price=62000, seller_rating=2.0),
        row(offer_id="amz", price=70000),
        row(offer_id="croma", marketplace="croma.com", seller_name="Croma", seller_rating=None, price=72000),
        row(offer_id="unk", marketplace="flipkart.com", seller_name="RetailNet", availability=None, price=65000),
    ]
    report = check_sellers(rows, now=NOW, stale_hours=24, trusted_names=())
    assert report.cheapest_available.offer_id == "amz"
    assert report.safest_available.offer_id == "croma"
    assert report.cheapest_unverified.offer_id == "unk"
    assert any("out of stock" in warning for warning in report.warnings)
    assert report.stock_by_retailer["amazon"]["out_of_stock"] == 1
    assert report.stock_by_retailer["croma"]["best_price"] == 72000


def test_effective_price_uses_bank_offer_price():
    rows = [row(offer_id="a", price=70000, price_with_offers=65000), row(offer_id="b", price=67000)]
    report = check_sellers(rows, now=NOW, trusted_names=())
    assert report.cheapest_available.offer_id == "a"


def test_stale_and_empty_data_warnings():
    stale = check_sellers([row(fetched_at=NOW - timedelta(days=3))], now=NOW, stale_hours=24, trusted_names=())
    assert any("older than 24 hours" in warning for warning in stale.warnings)
    empty = check_sellers([], now=NOW, trusted_names=())
    assert empty.cheapest_available is None
    assert "No stored offers" in empty.warnings[0]
