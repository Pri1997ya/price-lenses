from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Promo:
    """One structured promotion attached to an offer (bank card, exchange, coupon, festive ...)."""

    promo_type: str                  # BANK | EXCHANGE | COUPON | CASHBACK | EMI | FESTIVE | SPECIAL_PRICE | OTHER
    description: str
    bank: str | None = None
    card_type: str | None = None     # "Credit Card", "Debit Card", "UPI", ...
    amount: float | None = None      # rupees; for "10% up to Rs1,500" this is the cap
    percent: float | None = None
    is_emi: bool = False
    source: str | None = None        # which actor/provider reported it


@dataclass
class Offer:
    """One listing of a product on one marketplace/seller, from one provider."""

    provider: str                    # "serpapi" | "apify"
    marketplace: str                 # "amazon.in", "flipkart.com", "google_shopping", ...
    title: str
    url: str | None = None
    external_id: str | None = None   # marketplace-native id (ASIN, Google product_id, ...)
    asin: str | None = None
    gtin: str | None = None          # UPC / EAN / GTIN if the source exposes it
    brand: str | None = None
    price: float | None = None
    currency: str | None = None
    original_price: float | None = None
    rating: float | None = None
    review_count: int | None = None
    seller_name: str | None = None
    seller_rating: float | None = None
    availability: str | None = None
    shipping: str | None = None
    offers: list[str] = field(default_factory=list)   # coupons, bank offers, badges
    raw: dict[str, Any] = field(default_factory=dict)
    # ---- richer fields (filled by detail-level scrapers) ----
    seller_id: str | None = None
    price_with_offers: float | None = None   # effective price after (stackable) bank offers
    is_assured: bool | None = None           # Flipkart Assured / authorised-seller style flag
    cod_available: bool | None = None
    no_cost_emi: bool | None = None
    return_policy: str | None = None
    delivery_by: str | None = None
    warranty: str | None = None
    condition: str | None = None             # NEW / RENEWED / USED ...
    promos: list[Promo] = field(default_factory=list)
