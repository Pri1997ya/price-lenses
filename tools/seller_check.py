"""Seller authorization and stock verification for Agent 3 (Eligibility & Safety).

Pure, deterministic rules over the offers the Market Investigator already
stores (``latest_market_offers`` joined with ``market_offer_details``). No LLM
and no network access: every verdict carries the reasons that produced it.

Stock status  : IN_STOCK | LOW_STOCK | OUT_OF_STOCK | PREORDER | UNKNOWN
Seller trust  : TRUSTED | OK | CAUTION | AVOID
"""
from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from .policy_corpus import RETAILER_LABELS, retailer_for_marketplace

IN_STOCK, LOW_STOCK, OUT_OF_STOCK, PREORDER, UNKNOWN = (
    "IN_STOCK", "LOW_STOCK", "OUT_OF_STOCK", "PREORDER", "UNKNOWN",
)
TRUSTED, OK, CAUTION, AVOID = "TRUSTED", "OK", "CAUTION", "AVOID"
TIER_RANK = {TRUSTED: 3, OK: 2, CAUTION: 1, AVOID: 0}
BUYABLE_STOCK = {IN_STOCK, LOW_STOCK}

# Retailers that sell their own inventory on their own site.
FIRST_PARTY_RETAILERS = {"croma", "reliance_digital", "vijay_sales"}
# Names that identify the marketplace, not the actual seller (e.g. Google
# Shopping reports "Amazon.in" as the source for every Amazon listing).
MARKETPLACE_ONLY_NAMES = {"amazon", "amazon.in", "flipkart", "flipkart.com"}
NON_NEW_CONDITIONS = {"RENEWED", "REFURBISHED", "USED", "OPEN_BOX", "OPEN BOX", "PRE-OWNED", "PREOWNED"}

_PREORDER = re.compile(r"pre[-\s_]?order|coming\s*soon|back[-\s_]?order|releases?\s+on", re.I)
_OUT = re.compile(
    r"out[\s_-]*of[\s_-]*stock|currently\s+unavailable|\bunavailable\b|sold[\s_-]*out|"
    r"discontinued|not\s+available|no\s+longer\s+available|notify\s+me",
    re.I,
)
_LOW_COUNT = re.compile(r"only\s+(\d+)\s+(?:left|remaining)|(\d+)\s+left\s+in\s+stock", re.I)
_LOW = re.compile(r"few\s+left|limited\s*(?:stock|availability)|hurry", re.I)
_IN = re.compile(
    r"in[\s_-]*stock|\bavailable\b|usually\s+dispatched|ships\s+(?:in|within)|delivery\s+by", re.I
)


def classify_stock(availability: str | None) -> tuple[str, int | None]:
    """Map a provider's free-text availability to a status and optional unit count."""
    if not availability or not str(availability).strip():
        return UNKNOWN, None
    text = str(availability).strip()
    if _PREORDER.search(text):
        return PREORDER, None
    if _OUT.search(text):
        return OUT_OF_STOCK, None
    match = _LOW_COUNT.search(text)
    if match:
        return LOW_STOCK, int(match.group(1) or match.group(2))
    if _LOW.search(text):
        return LOW_STOCK, None
    if _IN.search(text):
        return IN_STOCK, None
    return UNKNOWN, None


def normalize_rating(value) -> float | None:
    """Return a 0–5 seller rating; percentages (e.g. 92) are converted."""
    if value is None:
        return None
    try:
        rating = float(value)
    except (TypeError, ValueError):
        return None
    if rating < 0:
        return None
    if rating > 5:
        return round(rating / 20, 2) if rating <= 100 else None
    return rating


def trusted_sellers_from_env() -> tuple[str, ...]:
    raw = os.getenv("TRUSTED_SELLERS", "")
    return tuple(name.strip().lower() for name in raw.split(",") if name.strip())


def _as_datetime(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


@dataclass
class OfferAssessment:
    offer_id: str | None
    retailer: str | None
    marketplace: str
    seller: str | None
    title: str | None
    url: str | None
    price: float | None
    effective_price: float | None
    stock_status: str
    units_left: int | None
    trust_tier: str
    trust_reasons: list[str]
    condition: str | None
    age_hours: float | None
    stale: bool
    purchasable: bool

    @property
    def retailer_label(self) -> str:
        return RETAILER_LABELS.get(self.retailer or "", self.marketplace)


def assess_seller(row: dict, trusted_names: tuple[str, ...] = ()) -> tuple[str, list[str]]:
    retailer = retailer_for_marketplace(row.get("marketplace"))
    seller = (row.get("seller_name") or "").strip()
    seller_key = seller.lower()
    identified = bool(seller) and seller_key not in MARKETPLACE_ONLY_NAMES
    reasons: list[str] = []
    tier = CAUTION

    if retailer in FIRST_PARTY_RETAILERS:
        tier = TRUSTED
        reasons.append(f"Sold on {RETAILER_LABELS[retailer]}'s own store")
    if row.get("is_assured") is True:
        tier = TRUSTED
        reasons.append("Flipkart Assured listing")
    if identified and any(name in seller_key for name in trusted_names):
        tier = TRUSTED
        reasons.append("Seller is on your TRUSTED_SELLERS list")

    rating = normalize_rating(row.get("seller_rating"))
    if rating is not None:
        if rating < 3.0:
            tier = AVOID
            reasons.append(f"Low seller rating ({rating:.1f}/5)")
        elif rating < 4.0:
            if TIER_RANK[tier] > TIER_RANK[CAUTION]:
                tier = CAUTION
            reasons.append(f"Middling seller rating ({rating:.1f}/5)")
        else:
            if tier == CAUTION:
                tier = OK
            reasons.append(f"Seller rated {rating:.1f}/5")
    elif tier == CAUTION:
        reasons.append(
            "Seller not identified by the data source" if not identified else "No seller rating available"
        )

    condition = (row.get("item_condition") or row.get("condition") or "").strip().upper()
    if condition in NON_NEW_CONDITIONS:
        if TIER_RANK[tier] > TIER_RANK[CAUTION]:
            tier = CAUTION
        reasons.append(f"Item condition is {condition.title()}, not new")
    return tier, reasons


def assess_offer(
    row: dict,
    now: datetime | None = None,
    stale_hours: float = 24.0,
    trusted_names: tuple[str, ...] = (),
) -> OfferAssessment:
    now = now or datetime.now(timezone.utc)
    stock, units = classify_stock(row.get("availability"))
    tier, reasons = assess_seller(row, trusted_names)
    fetched = _as_datetime(row.get("fetched_at"))
    age = round((now - fetched).total_seconds() / 3600, 1) if fetched else None
    price = row.get("price")
    effective = row.get("price_with_offers") or price
    return OfferAssessment(
        offer_id=row.get("offer_id"),
        retailer=retailer_for_marketplace(row.get("marketplace")),
        marketplace=row.get("marketplace") or "",
        seller=row.get("seller_name"),
        title=row.get("title"),
        url=row.get("url"),
        price=float(price) if price is not None else None,
        effective_price=float(effective) if effective is not None else None,
        stock_status=stock,
        units_left=units,
        trust_tier=tier,
        trust_reasons=reasons,
        condition=(row.get("item_condition") or None),
        age_hours=age,
        stale=age is None or age > stale_hours,
        purchasable=stock in BUYABLE_STOCK and tier != AVOID,
    )


@dataclass
class SellerStockReport:
    offers: list[OfferAssessment] = field(default_factory=list)
    cheapest_available: OfferAssessment | None = None
    safest_available: OfferAssessment | None = None
    cheapest_unverified: OfferAssessment | None = None
    stock_by_retailer: dict[str, dict] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _price_key(offer: OfferAssessment) -> float:
    return offer.effective_price if offer.effective_price is not None else float("inf")


def check_sellers(
    rows: list[dict],
    now: datetime | None = None,
    stale_hours: float | None = None,
    trusted_names: tuple[str, ...] | None = None,
) -> SellerStockReport:
    stale_hours = float(os.getenv("OFFER_STALE_HOURS", "24")) if stale_hours is None else stale_hours
    trusted_names = trusted_sellers_from_env() if trusted_names is None else trusted_names
    offers = [assess_offer(row, now, stale_hours, trusted_names) for row in rows]
    report = SellerStockReport(offers=offers)
    if not offers:
        report.warnings.append("No stored offers for this product. Fetch live offers in the Market Investigator first.")
        return report

    priced = [offer for offer in offers if offer.effective_price is not None]
    buyable = sorted((o for o in priced if o.purchasable), key=_price_key)
    if buyable:
        report.cheapest_available = buyable[0]
        report.safest_available = sorted(
            buyable, key=lambda o: (-TIER_RANK[o.trust_tier], _price_key(o))
        )[0]
    unverified = sorted(
        (o for o in priced if o.stock_status == UNKNOWN and o.trust_tier != AVOID), key=_price_key
    )
    if unverified and (not buyable or _price_key(unverified[0]) < _price_key(buyable[0])):
        report.cheapest_unverified = unverified[0]

    for offer in offers:
        key = offer.retailer or offer.marketplace
        summary = report.stock_by_retailer.setdefault(
            key,
            {"label": offer.retailer_label, "listings": 0, "in_stock": 0, "out_of_stock": 0,
             "unknown": 0, "best_price": None},
        )
        summary["listings"] += 1
        if offer.stock_status in BUYABLE_STOCK:
            summary["in_stock"] += 1
            if offer.effective_price is not None and (
                summary["best_price"] is None or offer.effective_price < summary["best_price"]
            ):
                summary["best_price"] = offer.effective_price
        elif offer.stock_status in {OUT_OF_STOCK, PREORDER}:
            summary["out_of_stock"] += 1
        else:
            summary["unknown"] += 1

    cheapest_any = min(priced, key=_price_key) if priced else None
    if cheapest_any and cheapest_any is not report.cheapest_available:
        if cheapest_any.stock_status in {OUT_OF_STOCK, PREORDER}:
            report.warnings.append(
                f"The lowest listed price (₹{cheapest_any.effective_price:,.0f} on "
                f"{cheapest_any.retailer_label}) is {cheapest_any.stock_status.replace('_', ' ').lower()}."
            )
        elif cheapest_any.trust_tier == AVOID:
            report.warnings.append(
                f"The lowest listed price (₹{cheapest_any.effective_price:,.0f} on "
                f"{cheapest_any.retailer_label}) comes from a seller rated AVOID: "
                + "; ".join(cheapest_any.trust_reasons)
            )
    if report.cheapest_available and report.cheapest_available.trust_tier == CAUTION:
        report.warnings.append(
            f"The cheapest in-stock offer ({report.cheapest_available.retailer_label}) has seller "
            "CAUTION flags: " + "; ".join(report.cheapest_available.trust_reasons)
        )
    if not buyable:
        report.warnings.append("No offer is confirmed in stock from an acceptable seller.")
    if all(offer.stale for offer in offers):
        report.warnings.append(
            f"All stock data is older than {stale_hours:g} hours; refresh live offers before buying."
        )
    return report
