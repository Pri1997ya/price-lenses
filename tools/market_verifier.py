"""Ground Agent 2 output against the database snapshot used to create it."""
from __future__ import annotations

from typing import Any


class MarketReportVerificationError(ValueError):
    pass


def verify_market_report(
    report: dict[str, Any],
    offers: list[dict[str, Any]],
    promotions: list[dict[str, Any]],
) -> None:
    offer_by_id = {str(row.get("offer_id")): row for row in offers if row.get("offer_id")}
    promo_by_id = {
        str(row.get("promotion_id")): row
        for row in promotions
        if row.get("promotion_id")
    }
    errors: list[str] = []

    selected_fields = [
        (field, report.get(field))
        for field in ("best_listed_offer", "best_unconditional_offer", "best_conditional_offer")
    ]
    for index, group in enumerate(report.get("variant_groups") or []):
        selected_fields.extend([
            (f"variant_groups[{index}].best_unconditional_offer", group.get("best_unconditional_offer")),
            (f"variant_groups[{index}].best_conditional_offer", group.get("best_conditional_offer")),
        ])

    for field, selected in selected_fields:
        if not selected:
            continue
        offer_id = str(selected.get("offer_id") or "")
        source = offer_by_id.get(offer_id)
        if source is None:
            errors.append(f"{field} references unknown offer {offer_id}")
            continue
        is_conditional = field.endswith("best_conditional_offer")
        if not is_conditional:
            actual = source.get("price")
            if actual is None or round(float(actual), 2) != round(float(selected.get("price")), 2):
                errors.append(f"{field} price is not grounded in offer {offer_id}")
        elif selected.get("scenario_type") == "PROVIDER_EFFECTIVE_PRICE":
            actual = source.get("price_with_offers")
            if actual is None or round(float(actual), 2) != round(float(selected.get("price")), 2):
                errors.append(f"{field} is not grounded in provider effective price")
        for promotion_id in selected.get("promotion_ids", []) or []:
            promo = promo_by_id.get(str(promotion_id))
            if promo is None:
                errors.append(f"unknown promotion {promotion_id}")
            elif str(promo.get("offer_id")) != offer_id:
                errors.append(f"promotion {promotion_id} belongs to another offer")
            else:
                listed = float(source.get("price") or 0)
                amount = float(promo.get("amount") or 0)
                percent = float(promo.get("percent") or 0)
                percentage_value = listed * percent / 100 if percent else 0
                discount = min(amount, percentage_value) if amount and percentage_value else (percentage_value or amount)
                expected = round(listed - discount, 2)
                if expected != round(float(selected.get("price")), 2):
                    errors.append(f"promotion {promotion_id} does not produce the reported price")

    ranked_ids = {str(row.get("offer_id")) for row in report.get("ranked_offers", [])}
    if not ranked_ids.issubset(offer_by_id):
        errors.append("ranked_offers contains evidence outside the snapshot")
    for ranked in report.get("ranked_offers", []):
        source = offer_by_id.get(str(ranked.get("offer_id")))
        if source and round(float(source.get("price") or 0), 2) != round(
            float(ranked.get("unconditional_price") or 0), 2
        ):
            errors.append(f"ranked offer {ranked.get('offer_id')} has an ungrounded price")
    forbidden = {"BUY_NOW", "WAIT", "BUY_ABROAD"}
    if forbidden.intersection(report.get("signals", [])):
        errors.append("Agent 2 returned a final purchase decision")

    if errors:
        raise MarketReportVerificationError("; ".join(errors))
