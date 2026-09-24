"""Agent 3: Eligibility & Safety Analyst.

Combines the deterministic seller/stock check (``seller_check``) with the
policy RAG (``policy_rag``) into the ``EligibilityReport`` consumed by the
LangGraph DAG and the Streamlit dashboard.

Every dependency is optional at runtime: a missing database, an unbuilt policy
index or an offline LLM downgrades the report (``status="partial"``) with an
explanation instead of failing the graph.
"""
from __future__ import annotations

import os
import re
from dataclasses import asdict
from typing import Callable

from .policy_corpus import RETAILER_LABELS
from .seller_check import OfferAssessment, check_sellers

CATEGORY_PATTERNS = [
    ("mobile phones", r"iphone|galaxy\s+[sazm]\d|smartphone|\bphone\b|pixel\s*\d|oneplus|redmi|\b5g\b"),
    ("laptops", r"laptop|macbook|notebook|vivobook|zenbook|thinkpad|ideapad|inspiron|pavilion"),
    ("tablets", r"ipad|tablet|galaxy\s+tab|\btab\s+[as]\d"),
    ("headphones and earbuds", r"earbuds|headphone|airpods|\bbuds\b|earphone|neckband"),
    ("speakers", r"speaker|soundbar"),
    ("gaming consoles", r"playstation|\bps5\b|xbox|nintendo|switch\s*2?"),
    ("televisions", r"\btv\b|television|oled|qled"),
    ("smartwatches", r"watch|band\s*\d"),
]
CATEGORY_TERMS = {
    "mobile phones": ("mobile", "phone", "smartphone"),
    "laptops": ("laptop", "notebook", "computer"),
    "tablets": ("tablet",),
    "headphones and earbuds": ("headphone", "earbud", "earphone", "audio"),
    "speakers": ("speaker", "audio"),
    "gaming consoles": ("console", "gaming"),
    "televisions": ("television", " tv", "tv "),
    "smartwatches": ("watch", "wearable"),
}
MAX_POLICY_RETAILERS = 3


def infer_category(title: str | None) -> str:
    text = (title or "").lower()
    for category, pattern in CATEGORY_PATTERNS:
        if re.search(pattern, text):
            return category
    return "electronics"


def _offer_summary(offer: OfferAssessment | None) -> dict | None:
    if offer is None:
        return None
    return {
        "retailer": offer.retailer,
        "label": offer.retailer_label,
        "seller": offer.seller,
        "price": offer.effective_price,
        "url": offer.url,
        "stock_status": offer.stock_status,
        "units_left": offer.units_left,
        "trust_tier": offer.trust_tier,
        "trust_reasons": offer.trust_reasons,
        "age_hours": offer.age_hours,
    }


def _policy_retailers(report) -> list[str]:
    ordered: list[str] = []
    for offer in (report.cheapest_available, report.safest_available, report.cheapest_unverified):
        if offer and offer.retailer and offer.retailer not in ordered:
            ordered.append(offer.retailer)
    # Only retailers the buyer could actually use: buyable or stock-unverified.
    for offer in sorted(
        (
            o for o in report.offers
            if o.retailer and o.effective_price is not None
            and (o.purchasable or (o.stock_status == "UNKNOWN" and o.trust_tier != "AVOID"))
        ),
        key=lambda o: o.effective_price,
    ):
        if offer.retailer not in ordered:
            ordered.append(offer.retailer)
    return ordered[:MAX_POLICY_RETAILERS]


def default_offers_loader(canonical_id: str) -> list[dict]:
    from dotenv import load_dotenv

    from .market_db import MarketDatabase

    load_dotenv()
    database_url = os.getenv("DATABASE_URL") or os.getenv("PL_DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured")
    with MarketDatabase(database_url) as database:
        return database.offers_for_product(canonical_id)


def default_advisor():
    from .policy_rag import PolicyAdvisor, PolicyIndex, embedder_from_env

    return PolicyAdvisor(PolicyIndex(embedder_from_env()))


def run_eligibility_analysis(
    canonical_id: str,
    product_title: str | None = None,
    policy_question: str | None = None,
    *,
    offers_loader: Callable[[str], list[dict]] = default_offers_loader,
    advisor_factory: Callable[[], object] = default_advisor,
) -> dict:
    trace: list[str] = []
    errors: list[str] = []
    category = infer_category(product_title)
    trace.append(f"Product category inferred from title: {category}")

    # 1. Seller authorization + stock (deterministic).
    try:
        rows = offers_loader(canonical_id)
        trace.append(f"Loaded {len(rows)} stored market offers for {canonical_id}")
    except Exception as exc:
        rows = []
        errors.append(f"Stored offers unavailable: {exc}")
        trace.append(f"Offer lookup failed: {exc}")
    if category == "electronics" and rows:
        # The resolver may only have the ASIN; fall back to the stored catalog title.
        for row in rows:
            fallback = infer_category(row.get("product_title") or row.get("title"))
            if fallback != "electronics":
                category = fallback
                trace.append(f"Category inferred from stored offer title instead: {category}")
                break
    seller_report = check_sellers(rows)
    for offer in seller_report.offers:
        trace.append(
            f"Offer {offer.retailer_label} / {offer.seller or 'unknown seller'}: "
            f"stock={offer.stock_status}, trust={offer.trust_tier} ({'; '.join(offer.trust_reasons)})"
        )

    # 2. Policy RAG for the retailers the buyer would actually use.
    policies: dict[str, dict] = {}
    user_answer = None
    restrictions: list[dict] = []
    retailers = _policy_retailers(seller_report)
    try:
        advisor = advisor_factory()
        targets = retailers or [None]
        for retailer in targets:
            label = RETAILER_LABELS.get(retailer, "Indian e-commerce retailers") if retailer else "Indian e-commerce retailers"
            question = (
                f"What is the return, replacement and refund policy for {category} bought on {label}? "
                "Include the time window and any conditions."
            )
            answer = advisor.answer(
                question, [retailer] if retailer else None,
                boost_terms=CATEGORY_TERMS.get(category, ()),
            )
            policies[retailer or "general"] = answer.to_dict()
            restrictions.extend(answer.restrictions)
            trace.append(
                f"Policy RAG ({label}): mode={answer.mode}, {len(answer.citations)} passages"
                + (f", note: {answer.note}" if answer.note else "")
            )
        if policy_question and policy_question.strip():
            user_answer = advisor.answer(policy_question.strip(), retailers or None).to_dict()
            trace.append(f"Answered user policy question (mode={user_answer['mode']})")
    except Exception as exc:
        errors.append(f"Policy RAG unavailable: {exc}")
        trace.append(f"Policy RAG failed: {exc}")

    relevant = [r for r in restrictions if r["retailer"] in set(retailers) | {"regulation"}]
    warning = None
    if relevant:
        seen = []
        for item in relevant:
            text = f"{RETAILER_LABELS.get(item['retailer'], item['retailer'])}: {item['restriction']}"
            if text not in seen:
                seen.append(text)
        warning = "Policy restrictions found in retrieved passages — " + "; ".join(seen)

    status = "ok"
    if errors or not seller_report.offers:
        status = "partial"
    if errors and not seller_report.offers and not policies:
        status = "error"

    return {
        "status": status,
        "canonical_id": canonical_id,
        "category": category,
        "cheapest_store": _offer_summary(seller_report.cheapest_available),
        "safest_store": _offer_summary(seller_report.safest_available),
        "cheapest_unverified": _offer_summary(seller_report.cheapest_unverified),
        "stock_summary": seller_report.stock_by_retailer,
        "offers": [asdict(offer) for offer in seller_report.offers],
        "policies": policies,
        "user_policy_answer": user_answer,
        "return_policy_warning": warning,
        "warnings": seller_report.warnings,
        "errors": errors,
        "agent_trace": trace,
    }
