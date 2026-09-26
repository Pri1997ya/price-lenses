"""PriceLens Streamlit application with history and live-market workspaces."""
from __future__ import annotations

from dataclasses import replace

import pandas as pd
import plotly.express as px
import streamlit as st

from orchestrator import graph
from tools import market_ui
from tools.analytics import get_db_connection
from tools.market_agent import MarketInvestigatorAgent
from tools.market_agent_models import FreshnessPolicy, MarketAgentRequest
from tools.market_config import ConfigurationError, MarketSettings
from tools.market_db import MarketDatabase, MarketSchemaError
from tools.market_service import build_providers


st.set_page_config(page_title="PriceLens Advisor", page_icon="🔍", layout="wide")


def fetch_price_history(canonical_id: str) -> pd.DataFrame:
    """Fetch the existing historical time series used by the History Agent."""
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT recorded_date, price
                FROM price_history
                WHERE canonical_id = %s
                ORDER BY recorded_date ASC
                """,
                (canonical_id,),
            )
            return pd.DataFrame(
                cursor.fetchall(), columns=["recorded_date", "price"]
            )
    finally:
        connection.close()


def run_history_analysis(query: str) -> dict | None:
    initial_state = {"query": query}
    result_state = initial_state.copy()
    with st.status("🚀 Executing History Agent workflow...", expanded=True) as status:
        try:
            for event in graph.stream(initial_state, {"recursion_limit": 15}):
                for node_name, node_state in event.items():
                    st.write(f"✅ **{node_name.replace('_', ' ').title()}** completed")
                    result_state.update(node_state)
                    if result_state.get("errors"):
                        break
            errors = result_state.get("errors", [])
            if errors:
                status.update(label="Analysis failed", state="error", expanded=True)
                st.error(errors[0])
                return None
            status.update(label="History analysis complete", state="complete", expanded=False)
            return result_state
        except Exception as exc:
            status.update(label="History analysis failed", state="error", expanded=True)
            st.error(f"History Agent error: {exc}")
            return None


def render_history_results(result_state: dict) -> None:
    canonical_id = result_state.get("canonical_id")
    product_title = result_state.get("product_title") or canonical_id or "Product"
    history_report = result_state.get("history_report", {})
    trend = history_report.get("trend", {})
    drops = history_report.get("drops", {})

    st.header(product_title)
    st.caption(f"Product ID: {canonical_id}")

    metric_columns = st.columns(4)
    metric_columns[0].metric("Current Price", f"₹{trend.get('current_price', 0):,.0f}")
    metric_columns[1].metric("All-Time Low", f"₹{trend.get('true_atl', 0):,.0f}")
    metric_columns[2].metric("30-Day Average", f"₹{trend.get('avg_30d', 0):,.0f}")
    metric_columns[3].metric("DHI Score", f"{trend.get('s_history', 0)} / 100")

    verdict_column, chart_column = st.columns(2)
    with verdict_column:
        st.subheader("Verdict & Timing")
        with st.container(border=True):
            stance = trend.get("historical_stance", "UNKNOWN")
            if stance == "BUY_NOW":
                st.success("### 🟢 BUY NOW")
                st.write("The price is near its verified historical low.")
            else:
                st.warning("### 🟡 WAIT")
                st.write(drops.get("rationale", "The current price is elevated."))
                if drops.get("safe_target_price") is not None:
                    st.markdown(f"#### Target price: ₹{drops['safe_target_price']:,.0f}")
                    st.info(
                        f"Expected {drops.get('expected_discount_pct', 0)}% reduction "
                        f"during {drops.get('upcoming_sale', 'an upcoming sale')}."
                    )
        if history_report.get("llm_analysis"):
            st.write("**History Agent analysis**")
            st.info(history_report["llm_analysis"])
        if history_report.get("agent_trace"):
            with st.expander("View LLM tool trace"):
                for log_entry in history_report["agent_trace"]:
                    st.code(log_entry, language="text")

    with chart_column:
        st.subheader("Price Trajectory")
        with st.container(border=True):
            try:
                history = fetch_price_history(canonical_id)
            except Exception as exc:
                st.error(f"Unable to load price history: {exc}")
                history = pd.DataFrame()
            if history.empty:
                st.info("No time-series data is available for this product.")
            else:
                figure = px.line(history, x="recorded_date", y="price")
                figure.update_layout(
                    margin=dict(l=20, r=20, t=20, b=20),
                    xaxis_title=None,
                    yaxis_title="Price (₹)",
                    showlegend=False,
                )
                figure.add_hline(
                    y=trend.get("current_price", 0),
                    line_dash="dot",
                    line_color="red",
                    annotation_text="Today",
                )
                figure.add_hline(
                    y=trend.get("true_atl", 0),
                    line_dash="dash",
                    line_color="green",
                    annotation_text="ATL",
                )
                st.plotly_chart(figure, width="stretch")

    with st.container(border=True):
        st.info("⏳ Review Intelligence Agent is pending implementation.")


def render_history_tab() -> None:
    st.subheader("History & Purchase Timing")
    st.caption(
        "Analyze stored PostgreSQL price history and receive a BUY NOW or WAIT recommendation."
    )
    with st.form("history_search_form"):
        input_column, button_column = st.columns([5, 1])
        query = input_column.text_input(
            "Amazon URL, ASIN, or product name",
            value=st.session_state.get("history_query", "B0CS5XW6TN"),
            help="Try an ASIN, a product name, or an Amazon /dp/ link.",
        )
        submitted = button_column.form_submit_button(
            "Analyze", type="primary", width="stretch"
        )

    if submitted:
        if not query.strip():
            st.warning("Enter a product to analyze.")
        else:
            st.session_state["history_query"] = query.strip()
            result = run_history_analysis(query.strip())
            if result:
                st.session_state["history_result"] = result

    result = st.session_state.get("history_result")
    if result:
        render_history_results(result)
    else:
        st.info("Enter a product above to inspect its historical price behavior.")


def market_database(settings: MarketSettings) -> MarketDatabase | None:
    try:
        return MarketDatabase(settings.database_url)
    except MarketSchemaError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Could not connect to the Market Investigator database: {exc}")
    return None


def fetch_market_results(query: str, settings: MarketSettings) -> tuple[list[dict], list[dict]]:
    database = market_database(settings)
    if database is None:
        return [], []
    try:
        return database.offers_for_query(query), database.promotions_for_query(query)
    finally:
        database.close()


def render_market_results(
    query: str,
    offer_rows: list[dict],
    promotion_rows: list[dict],
    report: dict | None = None,
) -> None:
    if report and report.get("unresolved_variant_fields"):
        st.info(
            "Run the search again with a specific storage/colour or paste a product URL. "
            "Raw provider matches are hidden to prevent accessory or cross-variant comparisons."
        )
        return
    if report and report.get("evidence"):
        evidence_ids = {
            item.get("offer_id") for item in report["evidence"] if item.get("offer_id")
        }
        offer_rows = [row for row in offer_rows if row.get("offer_id") in evidence_ids]
    rows = market_ui.enrich_rows(offer_rows)
    st.subheader(f"Results for “{query}”")
    if not rows:
        st.info("No stored offers were found for this exact query.")
        return

    products = market_ui.group_products(rows)
    product_labels = {
        product["canonical_id"]: (
            f"{(product['title'] or product['canonical_id'])[:70]} "
            f"({product['offers']} offers)"
        )
        for product in products
    }
    marketplaces = sorted({row["marketplace"] for row in rows})
    product_column, market_column, rating_column = st.columns([3, 2, 1])
    chosen_product = product_column.selectbox(
        "Product",
        [""] + list(product_labels),
        format_func=lambda value: "All matched products" if not value else product_labels[value],
        key=f"market_product_{query}",
    )
    chosen_markets = market_column.multiselect(
        "Marketplace",
        marketplaces,
        default=marketplaces,
        key=f"market_marketplaces_{query}",
    )
    minimum_rating = rating_column.slider(
        "Min rating", 0.0, 5.0, 0.0, 0.5, key=f"market_rating_{query}"
    )
    filtered = market_ui.filter_rows(
        rows, chosen_markets, minimum_rating, chosen_product or None
    )
    if not filtered:
        st.warning("No offers match the selected filters.")
        return

    summary = market_ui.summarize(filtered)
    metric_columns = st.columns(4)
    cheapest = summary["cheapest"]
    metric_columns[0].metric(
        "Lowest effective price",
        market_ui.format_money(
            cheapest["effective_price"] if cheapest else None, summary["currency"]
        ),
    )
    metric_columns[1].metric(
        "Average effective price",
        market_ui.format_money(summary["average"], summary["currency"]),
    )
    metric_columns[2].metric(
        "Offers · Marketplaces", f"{summary['offers']} · {summary['marketplaces']}"
    )
    best_rated = summary["best_rated"]
    metric_columns[3].metric(
        "Best rated", f"{best_rated['rating']:.1f} ★" if best_rated else "-"
    )

    table = pd.DataFrame(market_ui.offers_table(filtered))
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Price": st.column_config.NumberColumn(format="₹ %.0f"),
            "Effective price": st.column_config.NumberColumn(format="₹ %.0f"),
            "Was": st.column_config.NumberColumn(format="₹ %.0f"),
            "Discount %": st.column_config.NumberColumn(format="%.1f%%"),
            "Rating": st.column_config.NumberColumn(format="%.1f ★"),
            "Link": st.column_config.LinkColumn("Link", display_text="Open"),
        },
    )

    visible_offer_ids = {row["offer_id"] for row in filtered if row.get("offer_id")}
    promotions = market_ui.promotions_table(promotion_rows, visible_offer_ids)
    st.markdown("**Product-specific offers and promotions**")
    if promotions:
        counts = market_ui.promotion_summary(promotion_rows, visible_offer_ids)
        st.caption(" · ".join(f"{name}: {count}" for name, count in sorted(counts.items())))
        st.dataframe(
            pd.DataFrame(promotions),
            hide_index=True,
            width="stretch",
            column_config={
                "Amount": st.column_config.NumberColumn(format="₹ %.0f"),
                "Percent": st.column_config.NumberColumn(format="%.1f%%"),
                "Product link": st.column_config.LinkColumn(
                    "Product link", display_text="Open"
                ),
            },
        )
    else:
        st.caption(
            "No structured product-specific promotions are stored for these offers. "
            "Enable provider enrichment and fetch again."
        )

    st.download_button(
        "Download offers CSV",
        table.to_csv(index=False).encode("utf-8"),
        "market-offers.csv",
        "text/csv",
    )

    currency = summary["currency"]
    chart_rows = [
        row
        for row in filtered
        if row.get("effective_price") is not None and row.get("currency") == currency
    ]
    if chart_rows:
        chart = pd.DataFrame(
            {
                "Seller": [market_ui.offer_label(row) for row in chart_rows],
                "Effective price": [row["effective_price"] for row in chart_rows],
            }
        ).sort_values("Effective price")
        st.bar_chart(chart.set_index("Seller"))


def render_market_agent_report(report: dict) -> None:
    st.markdown("**Agent 2 market report**")
    coverage = report.get("coverage") or {}
    freshness = report.get("freshness") or {}
    requested_match = report.get("requested_match") or {}
    best = (
        requested_match.get("best_unconditional_offer")
        or report.get("best_unconditional_offer")
    )
    conditional = (
        requested_match.get("best_conditional_offer")
        or report.get("best_conditional_offer")
    )
    columns = st.columns(4)
    columns[0].metric(
        "Requested variant" if requested_match else "Starting price",
        market_ui.format_money(
            best.get("price") if best else None,
            best.get("currency") if best else "INR",
        ),
    )
    columns[1].metric(
        "Potential conditional",
        market_ui.format_money(
            conditional.get("price") if conditional else None,
            conditional.get("currency") if conditional else "INR",
        ),
    )
    columns[2].metric(
        "Retailer coverage",
        f"{len(coverage.get('verified_retailers', []))}/{len(coverage.get('expected_retailers', []))}",
    )
    columns[3].metric("Confidence", f"{float(report.get('confidence') or 0):.0%}")
    st.write(report.get("summary") or "No Agent 2 summary is available.")
    st.caption(
        f"Match: {str(report.get('match_mode', 'unknown')).replace('_', ' ')} · "
        f"Status: {report.get('status', 'unknown')} · Freshness: "
        f"{freshness.get('status', 'unknown')} · Signals: "
        f"{', '.join(report.get('signals') or []) or 'none'}"
    )
    if report.get("missing_inputs"):
        st.info("Conditional-price eligibility still needed: " + ", ".join(report["missing_inputs"]))
    if report.get("variant_groups"):
        st.markdown("**Verified product variants**")
        variants = []
        for group in report["variant_groups"]:
            variant = group.get("variant") or {}
            group_best = group.get("best_unconditional_offer") or {}
            group_conditional = group.get("best_conditional_offer") or {}
            variants.append({
                "Match": str(group.get("match_type") or "").replace("_", " ").title(),
                "Product": group.get("title"),
                "Storage": variant.get("storage"),
                "RAM": variant.get("ram"),
                "Colour": variant.get("color"),
                "Connectivity": variant.get("connectivity"),
                "Screen / device size": (
                    variant.get("screen_size") or variant.get("device_size")
                ),
                "Generation": variant.get("generation"),
                "Condition": variant.get("condition"),
                "Bundle": variant.get("bundle"),
                "Best price": group_best.get("price"),
                "Conditional price": group_conditional.get("price"),
                "Retailer": group_best.get("retailer"),
                "Seller": group_best.get("seller"),
                "Offers": group.get("offer_count"),
                "Promotions": group.get("promotion_count"),
                "Link": group_best.get("url"),
            })
        st.dataframe(
            pd.DataFrame(variants),
            hide_index=True,
            width="stretch",
            column_config={
                "Best price": st.column_config.NumberColumn(format="₹ %.0f"),
                "Conditional price": st.column_config.NumberColumn(format="₹ %.0f"),
                "Link": st.column_config.LinkColumn("Link", display_text="Open"),
            },
        )
    for warning in report.get("warnings") or []:
        st.warning(warning)
    with st.expander("How Agent 2 investigated this product"):
        trace = report.get("agent_trace") or []
        if not trace:
            st.caption("No execution trace is available for this analysis.")
        for index, event in enumerate(trace, start=1):
            status_icon = {
                "completed": "✓",
                "skipped": "○",
                "fallback": "△",
                "error": "✕",
            }.get(event.get("status"), "•")
            st.markdown(
                f"**{index}. {status_icon} {str(event.get('stage') or '').replace('_', ' ').title()}** "
                f"— {float(event.get('duration_ms') or 0):,.0f} ms"
            )
            st.caption(
                f"Tool: {event.get('tool', '-')} · Source: {event.get('source', '-')} · "
                f"Status: {event.get('status', 'unknown')}"
            )
            st.caption(f"Input: {event.get('input_summary') or 'No input summary.'}")
            st.write(event.get("output_summary") or "No result summary.")
            if event.get("display_prompt"):
                st.code(event["display_prompt"], language="text")
            else:
                st.caption("Prompt: Not applicable — this stage does not use an LLM.")


def render_market_tab() -> None:
    st.subheader("Live Market Investigator")
    st.caption(
        "Compare India prices, sellers, availability, and product-specific promotions "
        "from SerpAPI and Apify."
    )
    st.warning(
        "Analyze market checks SerpAPI/Apify first and may consume credits. "
        "Analyze stored data only never calls providers."
    )

    try:
        settings = MarketSettings.from_env()
    except ConfigurationError as exc:
        st.error(f"Market configuration error: {exc}")
        st.code("DATABASE_URL=<postgresql connection string>", language="bash")
        return

    with st.form("market_search_form"):
        query = st.text_input(
            "Amazon/Flipkart URL, ASIN, or product name",
            value=st.session_state.get("market_query", ""),
            placeholder="e.g. Apple iPhone 16 128GB",
        )
        option_columns = st.columns(4)
        use_serpapi = option_columns[0].checkbox("SerpAPI", value=True)
        use_apify = option_columns[1].checkbox("Apify", value=True)
        enrich_amazon = option_columns[2].checkbox(
            "Amazon details",
            value=settings.serpapi_enrich_amazon,
            help="Uses extra SerpAPI Product API credits. Amazon Search remains the discovery API.",
        )
        enrich_apify = option_columns[3].checkbox(
            "Seller/promotion enrichment",
            value=bool(settings.apify_enrichers),
            help="Runs product-detail actors and may consume additional Apify credits.",
        )
        limit = st.slider("Results per provider", 5, 50, 20, step=5)
        fetch_button, load_button = st.columns(2)
        analyze_market = fetch_button.form_submit_button(
            "Analyze market", type="primary", width="stretch"
        )
        load_stored = load_button.form_submit_button(
            "Analyze stored data only", width="stretch"
        )

    normalized_query = query.strip()
    if analyze_market or load_stored:
        if not normalized_query:
            st.warning("Enter a product name or product URL.")
        else:
            st.session_state["market_query"] = normalized_query
            st.session_state["market_active_query"] = normalized_query

    if (analyze_market or load_stored) and normalized_query:
        provider_names = [
            name
            for name, enabled in (("serpapi", use_serpapi), ("apify", use_apify))
            if enabled and analyze_market
        ]
        if analyze_market and not provider_names:
            st.warning("Select at least one provider.")
        else:
            runtime_settings = replace(
                settings,
                serpapi_enrich_amazon=enrich_amazon,
                apify_enrichers=settings.apify_enrichers if enrich_apify else (),
            )
            providers = []
            for provider_name in provider_names:
                try:
                    providers.extend(build_providers(runtime_settings, (provider_name,)))
                except ConfigurationError as exc:
                    st.error(f"{provider_name}: {exc}")
            database = market_database(runtime_settings)
            if database is not None:
                try:
                    freshness = FreshnessPolicy(
                        price_minutes=runtime_settings.market_price_freshness_minutes,
                        availability_minutes=runtime_settings.market_availability_freshness_minutes,
                        delivery_minutes=runtime_settings.market_delivery_freshness_minutes,
                        promotion_minutes=runtime_settings.market_promotion_freshness_minutes,
                        seller_minutes=runtime_settings.market_seller_freshness_minutes,
                        product_minutes=runtime_settings.market_product_freshness_minutes,
                    )
                    with st.spinner("Agent 2 is checking stored evidence and provider freshness..."):
                        report = MarketInvestigatorAgent(
                            database,
                            providers,
                            freshness=freshness,
                            enable_llm_summary=runtime_settings.market_agent_llm_enabled,
                        ).analyze(
                            MarketAgentRequest(
                                query=normalized_query,
                                provider_policy=(
                                    "api_first" if analyze_market else "database_only"
                                ),
                            ),
                            limit=limit,
                        )
                    st.session_state["market_agent_report"] = report
                    st.session_state["market_run_results"] = report.get("provider_runs", [])
                finally:
                    database.close()

    for result in st.session_state.get("market_run_results", []):
        if result["status"] == "error":
            st.error(f"{result['provider']}: {result['error']}")
        else:
            st.success(
                f"{result['provider']}: {result['count']} offers stored "
                f"({result['status']})"
            )
            for warning in result.get("warnings", []):
                st.warning(f"{result['provider']}: {warning}")

    active_query = st.session_state.get("market_active_query")
    if active_query:
        report = st.session_state.get("market_agent_report")
        if report:
            render_market_agent_report(report)
        offers, promotions = fetch_market_results(active_query, settings)
        render_market_results(active_query, offers, promotions, report)
    else:
        st.info("Fetch live offers or load an exact query already stored in PostgreSQL.")


st.title("🔍 PriceLens: Autonomous Deal Advisor")
st.caption(
    "Use historical pricing to decide when to buy, then compare live Indian-market offers."
)

history_tab, market_tab = st.tabs(["📈 History & Timing", "🛒 Market Investigator"])
with history_tab:
    render_history_tab()
with market_tab:
    render_market_tab()
