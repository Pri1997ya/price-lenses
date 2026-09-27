"""Streamlit rendering helpers for Agent 3 (Eligibility & Safety)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .policy_corpus import RETAILER_LABELS

STOCK_BADGES = {
    "IN_STOCK": "🟢 In stock",
    "LOW_STOCK": "🟠 Low stock",
    "OUT_OF_STOCK": "🔴 Out of stock",
    "PREORDER": "🔵 Pre-order",
    "UNKNOWN": "⚪ Unknown",
}
TRUST_BADGES = {
    "TRUSTED": "🛡️ Trusted",
    "OK": "✅ OK",
    "CAUTION": "⚠️ Caution",
    "AVOID": "⛔ Avoid",
}
MODE_LABELS = {
    "llm": "Answer grounded in retrieved policy passages",
    "extractive": "Retrieved policy passages (no generated answer)",
    "no_match": "Not found in indexed policies",
}


def _price(value) -> str:
    return f"₹{value:,.0f}" if isinstance(value, (int, float)) else "—"


def offers_frame(offers: list[dict]) -> pd.DataFrame:
    rows = []
    for offer in offers:
        rows.append(
            {
                "Retailer": RETAILER_LABELS.get(offer.get("retailer") or "", offer.get("marketplace")),
                "Seller": offer.get("seller") or "—",
                "Effective price": offer.get("effective_price"),
                "Stock": STOCK_BADGES.get(offer.get("stock_status"), offer.get("stock_status"))
                + (f" ({offer['units_left']} left)" if offer.get("units_left") else ""),
                "Seller trust": TRUST_BADGES.get(offer.get("trust_tier"), offer.get("trust_tier")),
                "Why": "; ".join(offer.get("trust_reasons") or []),
                "Data age (h)": offer.get("age_hours"),
                "Link": offer.get("url"),
            }
        )
    return pd.DataFrame(rows)


def render_offers_table(offers: list[dict]) -> None:
    if not offers:
        st.info("No stored offers for this product. Fetch live offers in the Market Investigator tab first.")
        return
    st.dataframe(
        offers_frame(offers),
        hide_index=True,
        width="stretch",
        column_config={
            "Effective price": st.column_config.NumberColumn(format="₹%.0f"),
            "Link": st.column_config.LinkColumn(display_text="Open"),
        },
    )


def _store_card(column, title: str, store: dict | None, empty: str) -> None:
    with column.container(border=True):
        st.caption(title)
        if not store:
            st.write(empty)
            return
        st.markdown(f"**{store['label']}** · {store.get('seller') or 'seller not identified'}")
        st.markdown(f"### {_price(store.get('price'))}")
        st.write(
            f"{STOCK_BADGES.get(store['stock_status'], store['stock_status'])} · "
            f"{TRUST_BADGES.get(store['trust_tier'], store['trust_tier'])}"
        )
        if store.get("url"):
            st.link_button("Open listing", store["url"])


def render_policy_answer(answer: dict, *, show_question: bool = False) -> None:
    if show_question:
        st.markdown(f"**Q:** {answer['question']}")
    mode = answer.get("mode")
    st.caption(MODE_LABELS.get(mode, mode))
    if mode == "no_match":
        st.warning(answer["answer"])
    else:
        st.markdown(answer["answer"])
    if answer.get("note"):
        st.caption(f"ℹ️ {answer['note']}")
    for restriction in answer.get("restrictions") or []:
        st.warning(
            f"**{restriction['restriction']}** "
            f"({RETAILER_LABELS.get(restriction['retailer'], restriction['retailer'])}, [{restriction['citation']}]): "
            f"“{restriction['excerpt']}”"
        )
    citations = answer.get("citations") or []
    if citations:
        with st.expander(f"Sources ({len(citations)})"):
            for hit in citations:
                label = RETAILER_LABELS.get(hit["retailer"], hit["retailer"])
                heading = f" — {hit['heading']}" if hit.get("heading") else ""
                st.markdown(
                    f"**[{hit['number']}] {label}{heading}**  \n"
                    f"[{hit['source_url']}]({hit['source_url']}) · retrieved {hit['retrieved_at']} · "
                    f"relevance {hit['relevance']:.2f}"
                )
                st.text(hit["text"][:1200])


def render_defects(defects: dict | None) -> None:
    if not defects:
        return
    total = defects.get("reviews_analyzed", 0)
    if not total:
        st.caption(defects.get("note", "No reviews collected for this product."))
        return
    sources = ", ".join(f"{name} {count}" for name, count in defects.get("by_source", {}).items())
    st.markdown(f"**Reported defects** ({total} reviews: {sources})")
    if not defects.get("findings"):
        st.caption("No defect is reported by enough reviewers to count as a pattern.")
    for finding in defects.get("findings", []):
        recent = finding["recent_mentions"]
        title = (
            f"{finding['label']}: {finding['mentions']} of {total} reviews "
            f"({finding['share']:.1%}), {recent} in the last year"
        )
        with st.expander(title):
            st.caption("Seen on " + ", ".join(finding["sources"]))
            for example in finding["examples"]:
                meta = " · ".join(
                    part for part in (
                        example["source"], example.get("date"),
                        f"{example['rating']:g}★" if example.get("rating") else None,
                    ) if part
                )
                link = f" [open]({example['url']})" if example.get("url") else ""
                st.markdown(f"> {example['snippet']}\n\n{meta}{link}")


def render_eligibility_report(report: dict) -> None:
    if not report:
        return
    for error in report.get("errors", []):
        st.error(error)

    left, right, third = st.columns(3)
    _store_card(left, "Cheapest in stock (acceptable seller)", report.get("cheapest_store"),
                "No in-stock offer from an acceptable seller.")
    _store_card(right, "Safest in stock", report.get("safest_store"), "—")
    _store_card(third, "Cheaper but stock unverified", report.get("cheapest_unverified"),
                "No cheaper listing with unknown stock.")

    if report.get("return_policy_warning"):
        st.warning(report["return_policy_warning"])
    for warning in report.get("warnings", []):
        st.warning(warning)

    if report.get("defect_warning"):
        st.warning(report["defect_warning"])

    st.markdown("**All stored listings**")
    render_offers_table(report.get("offers", []))

    windows = report.get("return_windows") or {}
    if windows:
        st.markdown("**Return windows (reviewed)**")
        for row in windows.values():
            st.markdown(
                f"- {row['summary']} — [source]({row['source_url']}), retrieved {row['retrieved_at']}"
            )

    policies = report.get("policies") or {}
    if policies:
        st.markdown(f"**Return & replacement policy ({report.get('category', 'electronics')})**")
        for retailer, answer in policies.items():
            with st.expander(RETAILER_LABELS.get(retailer, retailer.title()), expanded=len(policies) == 1):
                render_policy_answer(answer)
    render_defects(report.get("defects"))
    if report.get("user_policy_answer"):
        st.markdown("**Your policy question**")
        render_policy_answer(report["user_policy_answer"], show_question=True)
    if report.get("agent_trace"):
        with st.expander("View Eligibility Agent trace"):
            for entry in report["agent_trace"]:
                st.code(entry, language="text")
