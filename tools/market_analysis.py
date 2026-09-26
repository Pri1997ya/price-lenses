"""Deterministic calculations for Agent 2.

No LLM is allowed to calculate prices, join promotions, or decide whether two
variants are comparable.  This module is deliberately pure so those decisions
remain testable and reproducible.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Iterable

from .market_agent_models import FreshnessPolicy, MarketReport, SUPPORTED_RETAILERS


_MARKETPLACE_ALIASES = {
    "amazon": "amazon.in",
    "amazon india": "amazon.in",
    "www.amazon.in": "amazon.in",
    "flipkart": "flipkart.com",
    "www.flipkart.com": "flipkart.com",
    "croma": "croma.com",
    "www.croma.com": "croma.com",
    "reliance digital": "reliancedigital.in",
    "www.reliancedigital.in": "reliancedigital.in",
    "vijay sales": "vijaysales.com",
    "www.vijaysales.com": "vijaysales.com",
}
_CAPACITY_RE = re.compile(r"\b(\d+)\s*(gb|tb)\b", re.I)
_MODEL_RE = re.compile(r"\b(?=[a-z0-9-]*\d)[a-z]+[a-z0-9-]*\b|\b\d{1,4}[a-z]+\b", re.I)
_COLOURS = {
    "black", "white", "blue", "green", "red", "pink", "purple", "yellow",
    "silver", "gold", "grey", "gray", "titanium", "graphite", "midnight",
    "starlight", "natural", "cream", "lavender", "teal", "orange",
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def marketplace_name(value: str | None) -> str | None:
    if not value:
        return None
    candidate = value.strip().lower().removeprefix("https://").removeprefix("http://")
    candidate = candidate.split("/", 1)[0]
    if candidate in SUPPORTED_RETAILERS:
        return candidate
    if candidate in _MARKETPLACE_ALIASES:
        return _MARKETPLACE_ALIASES[candidate]
    for marketplace in SUPPORTED_RETAILERS:
        if candidate.endswith("." + marketplace):
            return marketplace
    return None


def _material_attributes(title: str | None) -> dict[str, set[str]]:
    text = (title or "").lower().replace("gray", "grey")
    capacities = {f"{number}{unit.lower()}" for number, unit in _CAPACITY_RE.findall(text)}
    models = {token.lower() for token in _MODEL_RE.findall(text)}
    colours = {colour for colour in _COLOURS if re.search(rf"\b{re.escape(colour)}\b", text)}
    conditions = {
        condition for condition in ("renewed", "refurbished", "used", "open box")
        if condition in text
    }
    connectivity = {
        item for item in ("wifi", "cellular", "lte", "5g", "4g")
        if re.search(rf"\b{re.escape(item)}\b", text)
    }
    return {
        "capacity": capacities,
        "model": models,
        "colour": colours,
        "condition": conditions,
        "connectivity": connectivity,
    }


def variants_compatible(reference: str | None, candidate: str | None) -> tuple[bool, list[str]]:
    """Reject a candidate only when both titles expose conflicting material facts."""
    left, right = _material_attributes(reference), _material_attributes(candidate)
    conflicts: list[str] = []
    for field in ("capacity", "condition", "connectivity"):
        if left[field] and right[field] and not left[field].issubset(right[field]):
            conflicts.append(field)
    # Colours are often omitted.  When both sides declare one, they must overlap.
    if left["colour"] and right["colour"] and not (left["colour"] & right["colour"]):
        conflicts.append("colour")
    # Model tokens such as A2890, S24, 15R and 16E are strong conflict evidence.
    if left["model"] and right["model"] and not left["model"].issubset(right["model"]):
        conflicts.append("model")
    return not conflicts, conflicts


def age_minutes(value: Any, now: datetime | None = None) -> float | None:
    if value is None:
        return None
    current = now or utc_now()
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return max(0.0, (current - value.astimezone(timezone.utc)).total_seconds() / 60)


def freshness_for_rows(
    rows: Iterable[dict[str, Any]],
    policy: FreshnessPolicy,
    now: datetime | None = None,
) -> dict[str, Any]:
    ages = [age for row in rows if (age := age_minutes(row.get("fetched_at"), now)) is not None]
    if not ages:
        return {
            "status": "empty",
            "oldest_observation_at": None,
            "oldest_age_minutes": None,
            "price_ttl_minutes": policy.price_minutes,
        }
    oldest_age = max(ages)
    newest_age = min(ages)
    if oldest_age <= policy.price_minutes:
        status = "fresh"
    elif newest_age > policy.price_minutes:
        status = "stale"
    else:
        status = "mixed"
    return {
        "status": status,
        "oldest_observation_at": iso(min(row.get("fetched_at") for row in rows if row.get("fetched_at"))),
        "oldest_age_minutes": round(oldest_age, 1),
        "newest_age_minutes": round(newest_age, 1),
        "price_ttl_minutes": policy.price_minutes,
    }


def _offer_evidence(row: dict[str, Any], *, price_key: str = "price") -> dict[str, Any]:
    return {
        "offer_id": row.get("offer_id"),
        "canonical_id": row.get("canonical_id"),
        "retailer": marketplace_name(row.get("marketplace")) or row.get("marketplace"),
        "seller": row.get("seller_name"),
        "price": row.get(price_key),
        "currency": row.get("currency") or "INR",
        "url": row.get("url"),
        "provider": row.get("provider"),
        "external_id": row.get("external_id"),
        "fetched_at": iso(row.get("fetched_at")),
        "availability": row.get("availability"),
        "delivery": row.get("delivery_by") or row.get("shipping"),
    }


def _promo_discount(price: float, promo: dict[str, Any]) -> float:
    amount = float(promo.get("amount") or 0)
    percent = float(promo.get("percent") or 0)
    percentage_value = price * percent / 100 if percent else 0
    if amount and percentage_value:
        return min(amount, percentage_value)
    return percentage_value or amount


def _eligibility_status(promo: dict[str, Any], context: dict[str, Any]) -> str:
    bank = promo.get("bank")
    card = promo.get("card_type")
    is_emi = bool(promo.get("is_emi"))
    if bank and context.get("bank") and bank.lower() != str(context["bank"]).lower():
        return "ineligible"
    if card and context.get("card_type") and card.lower() not in str(context["card_type"]).lower():
        return "ineligible"
    if is_emi and context.get("wants_emi") is False:
        return "ineligible"
    missing = []
    if bank and not context.get("bank"):
        missing.append("bank")
    if card and not context.get("card_type"):
        missing.append("card_type")
    if is_emi and context.get("wants_emi") is None:
        missing.append("emi_preference")
    return "unknown" if missing else "eligible"


def promotion_scenarios(
    row: dict[str, Any],
    promotions: list[dict[str, Any]],
    context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Calculate non-stacked, offer-bound promotion scenarios.

    Cashback, exchange, and generic festive text are evidence but never reduce
    checkout price.  Bank discounts are conditional unless eligibility is known.
    """
    price = row.get("price")
    if price is None or (row.get("currency") or "INR") != "INR":
        return []
    price = float(price)
    context = context or {}
    scenarios: list[dict[str, Any]] = []
    detail_price = row.get("price_with_offers")
    if detail_price is not None and 0 < float(detail_price) < price:
        scenarios.append({
            **_offer_evidence(row),
            "price": float(detail_price),
            "listed_price": price,
            "promotion_ids": [],
            "conditions": ["Provider-reported offer price; verify checkout eligibility"],
            "eligibility": "unknown",
            "scenario_type": "PROVIDER_EFFECTIVE_PRICE",
        })
    for promo in promotions:
        if promo.get("offer_id") != row.get("offer_id"):
            continue
        promo_type = str(promo.get("promotion_type") or "OTHER").upper()
        if promo_type not in {"BANK", "COUPON", "SPECIAL_PRICE"}:
            continue
        description = str(promo.get("description") or "")
        if promo_type == "BANK" and "cashback" in description.lower():
            continue
        discount = _promo_discount(price, promo)
        if discount <= 0 or discount >= price:
            continue
        eligibility = _eligibility_status(promo, context)
        if eligibility == "ineligible":
            continue
        scenarios.append({
            **_offer_evidence(row),
            "price": round(price - discount, 2),
            "listed_price": price,
            "promotion_ids": [promo.get("promotion_id")],
            "conditions": [description],
            "eligibility": eligibility,
            "scenario_type": promo_type,
        })
    return sorted(scenarios, key=lambda scenario: scenario["price"])


def build_market_report(
    *,
    product: dict[str, Any],
    offers: list[dict[str, Any]],
    promotions: list[dict[str, Any]],
    sales: list[dict[str, Any]],
    provider_runs: list[dict[str, Any]] | None = None,
    policy: FreshnessPolicy | None = None,
    context: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    now: datetime | None = None,
) -> MarketReport:
    policy = policy or FreshnessPolicy()
    now = now or utc_now()
    warnings = list(warnings or [])
    reference = product.get("title")
    comparable: list[dict[str, Any]] = []
    for source in offers:
        row = dict(source)
        row["marketplace"] = marketplace_name(row.get("marketplace"))
        if row["marketplace"] not in SUPPORTED_RETAILERS:
            continue
        if (row.get("currency") or "INR") != "INR":
            warnings.append(f"Excluded non-INR offer {row.get('offer_id')}")
            continue
        compatible, conflicts = variants_compatible(reference, row.get("title"))
        if not compatible:
            warnings.append(
                f"Excluded variant mismatch {row.get('offer_id')}: {', '.join(conflicts)}"
            )
            continue
        if row.get("price") is None or float(row["price"]) <= 0:
            continue
        row["price"] = float(row["price"])
        comparable.append(row)

    comparable.sort(key=lambda row: row["price"])
    verified = sorted({row["marketplace"] for row in comparable})
    missing = [retailer for retailer in SUPPORTED_RETAILERS if retailer not in verified]
    freshness = freshness_for_rows(comparable, policy, now)
    promo_by_offer: dict[str, list[dict[str, Any]]] = {}
    for promo in promotions:
        promo_by_offer.setdefault(str(promo.get("offer_id")), []).append(promo)

    ranked: list[dict[str, Any]] = []
    all_scenarios: list[dict[str, Any]] = []
    for row in comparable:
        evidence = _offer_evidence(row)
        scenarios = promotion_scenarios(row, promo_by_offer.get(str(row.get("offer_id")), []), context)
        all_scenarios.extend(scenarios)
        evidence["unconditional_price"] = row["price"]
        evidence["conditional_price"] = scenarios[0]["price"] if scenarios else None
        evidence["promotion_count"] = len(promo_by_offer.get(str(row.get("offer_id")), []))
        ranked.append(evidence)

    ranked.sort(key=lambda row: (row["unconditional_price"], row["retailer"]))
    all_scenarios.sort(key=lambda scenario: scenario["price"])
    best_listed = _offer_evidence(comparable[0]) if comparable else None
    best_unconditional = dict(best_listed) if best_listed else None
    best_conditional = all_scenarios[0] if all_scenarios else None

    normalized_sales = []
    for sale in sales:
        item = dict(sale)
        item["approx_start_date"] = iso(item.get("approx_start_date"))
        item["approx_end_date"] = iso(item.get("approx_end_date"))
        normalized_sales.append(item)

    signals: list[str] = []
    if best_unconditional:
        signals.append("BEST_CURRENT_VERIFIED_PRICE")
    if best_conditional:
        signals.append("ACTIVE_PROMOTION")
    if normalized_sales:
        signals.append("SALE_SOON")
    if len(comparable) > 1:
        spread = (max(row["price"] for row in comparable) - comparable[0]["price"]) / comparable[0]["price"]
        if spread >= 0.10:
            signals.append("LARGE_RETAILER_PRICE_SPREAD")
    if not comparable:
        signals = ["INSUFFICIENT_EVIDENCE"]
    elif not signals:
        signals = ["NO_STRONG_SIGNAL"]

    coverage_ratio = len(verified) / len(SUPPORTED_RETAILERS)
    fresh_factor = {"fresh": 1.0, "mixed": 0.6, "stale": 0.25}.get(freshness["status"], 0.0)
    confidence = round(min(1.0, 0.55 * coverage_ratio + 0.30 * fresh_factor + (0.15 if comparable else 0)), 2)
    status = "complete" if len(verified) == len(SUPPORTED_RETAILERS) and freshness["status"] == "fresh" else "partial"
    if not comparable:
        status = "insufficient_evidence"

    missing_inputs: list[str] = []
    if best_conditional and best_conditional.get("eligibility") == "unknown":
        if any(promo.get("bank") for promo in promotions) and not (context or {}).get("bank"):
            missing_inputs.append("bank")
        if any(promo.get("card_type") for promo in promotions) and not (context or {}).get("card_type"):
            missing_inputs.append("card_type")
        if any(promo.get("is_emi") for promo in promotions) and (context or {}).get("wants_emi") is None:
            missing_inputs.append("emi_preference")

    if best_unconditional:
        summary = (
            f"The lowest verified unconditional price is ₹{best_unconditional['price']:,.0f} "
            f"from {best_unconditional['retailer']}. Coverage includes {len(verified)} of "
            f"{len(SUPPORTED_RETAILERS)} supported Indian retailers. This is a market-price "
            "signal only; purchase timing and seller safety require Agents 1 and 3."
        )
    else:
        summary = "No comparable INR offer was verified for the requested product variant."

    return MarketReport(
        status=status,
        product={key: product.get(key) for key in (
            "canonical_id", "title", "brand", "model", "storage", "ram", "color"
        )},
        analysis_timestamp=now.isoformat(),
        provider_runs=list(provider_runs or []),
        coverage={
            "expected_retailers": list(SUPPORTED_RETAILERS),
            "verified_retailers": verified,
            "missing_retailers": missing,
        },
        freshness=freshness,
        best_listed_offer=best_listed,
        best_unconditional_offer=best_unconditional,
        best_conditional_offer=best_conditional,
        ranked_offers=ranked,
        upcoming_sales=normalized_sales,
        signals=signals,
        confidence=confidence,
        missing_inputs=sorted(set(missing_inputs)),
        warnings=list(dict.fromkeys(warnings)),
        evidence=[_offer_evidence(row) for row in comparable],
        summary=summary,
    )
