"""Pure presentation helpers for the Market Investigator Streamlit tab."""
from __future__ import annotations

import json
from collections import Counter
from typing import Any

CURRENCY_SYMBOLS = {"INR": "₹", "USD": "$", "GBP": "£", "EUR": "€"}


def format_money(price: float | None, currency: str | None) -> str:
    if price is None:
        return "-"
    symbol = CURRENCY_SYMBOLS.get(
        currency or "", f"{currency} " if currency else ""
    )
    return f"{symbol}{price:,.0f}"


def discount_percent(price: float | None, original: float | None) -> float | None:
    if price is None or not original or original <= price:
        return None
    return round((original - price) / original * 100, 1)


def parse_offer_labels(value: str | list | None) -> list[str]:
    try:
        parsed = value if isinstance(value, list) else json.loads(value or "[]")
    except (TypeError, ValueError):
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def enrich_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = []
    for source in rows:
        row = dict(source)
        row["offer_labels"] = parse_offer_labels(row.get("offers_json"))
        row["discount_percent"] = discount_percent(
            row.get("price"), row.get("original_price")
        )
        row["effective_price"] = row.get("price_with_offers") or row.get("price")
        enriched.append(row)
    return enriched


def filter_rows(
    rows: list[dict[str, Any]],
    marketplaces: list[str] | None = None,
    minimum_rating: float = 0.0,
    canonical_id: str | None = None,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if (not canonical_id or row["canonical_id"] == canonical_id)
        and (not marketplaces or row["marketplace"] in marketplaces)
        and (not minimum_rating or (row.get("rating") or 0) >= minimum_rating)
    ]


def group_products(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        canonical_id = row["canonical_id"]
        group = groups.setdefault(
            canonical_id,
            {
                "canonical_id": canonical_id,
                "title": row.get("product_title") or row.get("title"),
                "offers": 0,
                "cheapest": None,
            },
        )
        group["offers"] += 1
        price = row.get("effective_price")
        if price is not None and (
            group["cheapest"] is None or price < group["cheapest"]
        ):
            group["cheapest"] = price
    return sorted(groups.values(), key=lambda group: -group["offers"])


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    currency_counts = Counter(
        row["currency"]
        for row in rows
        if row.get("effective_price") is not None and row.get("currency")
    )
    currency = currency_counts.most_common(1)[0][0] if currency_counts else None
    priced = [
        row
        for row in rows
        if row.get("effective_price") is not None and row.get("currency") == currency
    ]
    rated = [row for row in rows if row.get("rating") is not None]
    cheapest = min(priced, key=lambda row: row["effective_price"], default=None)
    best_rated = max(
        rated,
        key=lambda row: (row["rating"], row.get("review_count") or 0),
        default=None,
    )
    return {
        "currency": currency,
        "offers": len(rows),
        "marketplaces": len({row["marketplace"] for row in rows}),
        "cheapest": cheapest,
        "average": (
            sum(row["effective_price"] for row in priced) / len(priced)
            if priced
            else None
        ),
        "best_rated": best_rated,
    }


def offer_label(row: dict[str, Any]) -> str:
    seller = row.get("seller_name")
    return f"{row['marketplace']} · {seller}" if seller else row["marketplace"]


def offers_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "Product": row.get("product_title") or row.get("title"),
            "Marketplace": row["marketplace"],
            "Seller": row.get("seller_name"),
            "Price": row.get("price"),
            "Effective price": row.get("effective_price"),
            "Currency": row.get("currency"),
            "Was": row.get("original_price"),
            "Discount %": row.get("discount_percent"),
            "Rating": row.get("rating"),
            "Reviews": row.get("review_count"),
            "Seller rating": row.get("seller_rating"),
            "Assured": row.get("is_assured"),
            "COD": row.get("cod_available"),
            "No-cost EMI": row.get("no_cost_emi"),
            "Return policy": row.get("return_policy"),
            "Delivery by": row.get("delivery_by"),
            "Availability": row.get("availability"),
            "Delivery": row.get("shipping"),
            "Offers": " | ".join(row.get("offer_labels", [])),
            "Source": row.get("provider"),
            "Link": row.get("url"),
        }
        for row in rows
    ]


def promotions_table(
    rows: list[dict[str, Any]], offer_ids: set[str] | None = None
) -> list[dict[str, Any]]:
    return [
        {
            "Product": row.get("product_title"),
            "Product ID": row.get("canonical_id"),
            "Provider product ID": row.get("external_id"),
            "Marketplace": row["marketplace"],
            "Seller": row.get("seller_name"),
            "Type": row.get("promotion_type"),
            "Bank": row.get("bank"),
            "Card": row.get("card_type"),
            "Offer": row.get("description"),
            "Amount": row.get("amount"),
            "Percent": row.get("percent"),
            "EMI": row.get("is_emi"),
            "Source": row.get("source"),
            "Product link": row.get("url"),
        }
        for row in rows
        if offer_ids is None or row["offer_id"] in offer_ids
    ]


def promotion_summary(
    rows: list[dict[str, Any]], offer_ids: set[str] | None = None
) -> dict[str, int]:
    counts: Counter = Counter()
    for row in rows:
        if offer_ids is None or row["offer_id"] in offer_ids:
            counts[row.get("promotion_type") or "OTHER"] += 1
    return dict(counts)
