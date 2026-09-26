from datetime import datetime, timedelta, timezone

from tools.market_agent import MarketInvestigatorAgent
from tools.market_agent_models import MarketAgentRequest, SUPPORTED_RETAILERS
from tools.market_agent_tools import MarketAgentTools, variant_facets
from tools.market_analysis import build_market_report, variants_compatible
from tools.market_models import Offer
from tools.market_verifier import MarketReportVerificationError, verify_market_report


def offer_row(
    offer_id,
    marketplace,
    price,
    fetched_at=None,
    title="Apple iPhone 16 128GB Black",
    canonical_id="B0EXAMPLE1",
):
    return {
        "offer_id": offer_id,
        "canonical_id": canonical_id,
        "provider": "serpapi",
        "marketplace": marketplace,
        "external_id": offer_id,
        "title": title,
        "product_title": "Apple iPhone 16 128GB Black",
        "url": f"https://{marketplace}/product/{offer_id}",
        "price": price,
        "currency": "INR",
        "seller_name": marketplace,
        "availability": "In stock",
        "shipping": "Free delivery",
        "fetched_at": fetched_at or datetime.now(timezone.utc),
    }


class FakeDatabase:
    def __init__(self, offers, promotions=None, recent_runs=None):
        self.offers = list(offers)
        self.promotions = list(promotions or [])
        self.recent = list(recent_runs or [])
        self.started = []

    def product(self, canonical_id):
        return {
            "canonical_id": canonical_id,
            "title": "Apple iPhone 16 128GB Black",
            "brand": "Apple",
            "model": "iPhone 16",
            "storage": "128GB",
            "ram": None,
            "color": "Black",
        }

    def products_for_query(self, _query):
        return [self.product("B0EXAMPLE1")]

    def offers_for_product(self, _canonical_id):
        return list(self.offers)

    def offers_for_query(self, _query):
        return list(self.offers)

    def offers_for_run_ids(self, run_ids):
        identifiers = set(run_ids)
        return [row for row in self.offers if row.get("run_id") in identifiers]

    def promotions_for_offer_ids(self, offer_ids):
        identifiers = set(offer_ids)
        return [row for row in self.promotions if row["offer_id"] in identifiers]

    def upcoming_sales(self, _deadline_days):
        return []

    def recent_runs_for_query(self, _query):
        return list(self.recent)

    def known_products(self):
        return {"B0EXAMPLE1": "apple iphone 16 128gb black"}

    def start_run(self, query, provider):
        run_id = f"run-{provider}-{len(self.started)}"
        self.started.append((run_id, query, provider))
        return run_id

    def upsert_product(self, _resolved_id, _offer):
        return "B0EXAMPLE1"

    def insert_offers(self, run_id, rows):
        rows = list(rows)
        for index, (_canonical_id, offer) in enumerate(rows):
            row = offer_row(
                f"{run_id}-{index}", offer.marketplace, offer.price, title=offer.title
            )
            row["run_id"] = run_id
            self.offers.append(row)
        return len(rows)

    def finish_run(self, *_args, **_kwargs):
        return None


class CountingProvider:
    name = "serpapi"
    warnings = []

    def __init__(self):
        self.calls = 0

    def search(self, _query, limit=20):
        self.calls += 1
        return [
            Offer(
                provider="serpapi",
                marketplace="amazon.in",
                title="Apple iPhone 16 128GB Black",
                asin="B0EXAMPLE1",
                price=67900,
                currency="INR",
            )
        ][:limit]


class EmptyProvider(CountingProvider):
    def search(self, _query, limit=20):
        self.calls += 1
        return []


class VariantDatabase(FakeDatabase):
    PRODUCTS = {
        "PHONE128W": "S2 5G (Silk White, 8GB RAM, 128GB Storage) | 50MP Camera",
        "PHONE128B": "S2 5G (Sapphire Blue, 8GB RAM, 128GB Storage) | 50MP Camera",
        "PHONE256W": "S2 5G (Silk White, 8GB RAM, 256GB Storage) | 50MP Camera",
        "ACCESSORY": "Myflips Flip Cover For Vivo S2 5G",
    }

    def __init__(self):
        rows = [
            offer_row(
                product_id.lower(), "amazon.in", price, title=title,
                canonical_id=product_id,
            )
            for product_id, title, price in (
                ("PHONE128W", self.PRODUCTS["PHONE128W"], 39999),
                ("PHONE128B", self.PRODUCTS["PHONE128B"], 39999),
                ("PHONE256W", self.PRODUCTS["PHONE256W"], 44999),
                ("ACCESSORY", self.PRODUCTS["ACCESSORY"], 300),
            )
        ]
        super().__init__(rows)

    def product(self, canonical_id):
        title = self.PRODUCTS.get(canonical_id)
        return {"canonical_id": canonical_id, "title": title} if title else None

    def products_for_query(self, _query):
        return [
            {
                "canonical_id": product_id,
                "title": title,
                "observed_offer_count": 1,
                "lowest_price": next(
                    row["price"] for row in self.offers
                    if row["canonical_id"] == product_id
                ),
            }
            for product_id, title in self.PRODUCTS.items()
        ]

    def offers_for_product(self, canonical_id):
        return [row for row in self.offers if row["canonical_id"] == canonical_id]


def test_fresh_complete_snapshot_avoids_provider_calls():
    rows = [offer_row(f"offer-{index}", retailer, 70000 + index) for index, retailer in enumerate(SUPPORTED_RETAILERS)]
    database = FakeDatabase(rows, promotions=[{
        "promotion_id": "promo-cache",
        "offer_id": "offer-0",
        "promotion_type": "FESTIVE",
        "description": "Product-specific festive price",
        "fetched_at": datetime.now(timezone.utc),
    }])
    provider = CountingProvider()

    report = MarketInvestigatorAgent(database, [provider]).analyze(
        MarketAgentRequest(
            "Apple iPhone 16 128GB Black",
            canonical_id="B0EXAMPLE1",
            provider_policy="database_first",
        )
    )

    assert provider.calls == 0
    assert report["status"] == "complete"
    assert report["refresh"]["performed"] is False
    assert report["best_unconditional_offer"]["offer_id"] == "offer-0"


def test_stale_snapshot_refreshes_and_keeps_provider_run_id():
    old = datetime.now(timezone.utc) - timedelta(hours=3)
    database = FakeDatabase([offer_row("old", "amazon.in", 71000, old)])
    provider = CountingProvider()

    report = MarketInvestigatorAgent(database, [provider]).analyze(
        MarketAgentRequest(
            "Apple iPhone 16 128GB Black",
            canonical_id="B0EXAMPLE1",
            provider_policy="database_first",
        )
    )

    assert provider.calls == 1
    assert report["refresh"]["reason"] == "stale stored prices"
    assert report["provider_runs"][0]["run_id"].startswith("run-serpapi")
    assert report["best_unconditional_offer"]["price"] == 67900


def test_api_first_calls_provider_before_using_database_fallback():
    database = FakeDatabase([offer_row("stored", "flipkart.com", 68900)])
    provider = CountingProvider()

    report = MarketInvestigatorAgent(database, [provider]).analyze(
        MarketAgentRequest("Apple iPhone 16 128GB Black", canonical_id="B0EXAMPLE1")
    )

    assert provider.calls == 1
    assert report["refresh"]["policy"] == "api_first"
    assert report["refresh"]["live_offer_count"] == 1
    assert report["refresh"]["source_mode"] == "live_with_database_fallback"
    assert report["refresh"]["database_fallback_retailers"] == ["flipkart.com"]
    assert report["best_unconditional_offer"]["price"] == 67900


def test_api_first_uses_database_when_provider_returns_no_offers():
    database = FakeDatabase([offer_row("stored", "flipkart.com", 68900)])
    provider = EmptyProvider()

    report = MarketInvestigatorAgent(database, [provider]).analyze(
        MarketAgentRequest("Apple iPhone 16 128GB Black", canonical_id="B0EXAMPLE1")
    )

    assert provider.calls == 1
    assert report["refresh"]["source_mode"] == "database_fallback"
    assert report["best_unconditional_offer"]["offer_id"] == "stored"


def test_broad_phone_query_returns_all_variants_and_excludes_accessories():
    report = MarketInvestigatorAgent(VariantDatabase(), []).analyze(
        MarketAgentRequest("vivo s2", provider_policy="database_only")
    )

    assert report["status"] == "partial"
    assert report["match_mode"] == "expanded_variants"
    assert report["lowest_starting_price"] == 39999
    assert report["unresolved_variant_fields"] == []
    assert report["missing_inputs"] == []
    assert all("Cover" not in option["title"] for option in report["variant_options"])
    assert {option["canonical_id"] for option in report["variant_options"]} == {
        "PHONE128W", "PHONE128B", "PHONE256W"
    }
    assert len(report["variant_groups"]) == 3
    assert {group["variant"]["storage"] for group in report["variant_groups"]} == {
        "128GB", "256GB"
    }


def test_specific_variant_can_be_priced_without_cross_variant_comparison():
    report = MarketInvestigatorAgent(VariantDatabase(), []).analyze(
        MarketAgentRequest(
            "vivo s2 128GB silk white", provider_policy="database_only"
        )
    )

    assert report["unresolved_variant_fields"] == []
    assert report["product"]["canonical_id"] == "PHONE128W"
    assert report["best_unconditional_offer"]["price"] == 39999
    assert report["best_unconditional_offer"]["canonical_id"] == "PHONE128W"
    assert report["match_mode"] == "exact_variant"
    assert report["requested_match"]["variant"]["storage"] == "128GB"
    assert [group["match_type"] for group in report["variant_groups"]] == [
        "exact_variant", "same_configuration_other_color", "other_configuration"
    ]


def test_live_snapshot_keeps_product_family_variants_when_anchor_id_is_known():
    database = VariantDatabase()
    for row in database.offers:
        row["run_id"] = "live-run"

    snapshot = MarketAgentTools(database).get_market_snapshot(
        "vivo s2 128GB silk white",
        canonical_id="PHONE128W",
        run_ids=["live-run"],
    )

    assert {row["canonical_id"] for row in snapshot["offers"]} == {
        "PHONE128W", "PHONE128B", "PHONE256W"
    }


def test_missing_requested_colour_expands_same_configuration_before_other_variants():
    report = MarketInvestigatorAgent(VariantDatabase(), []).analyze(
        MarketAgentRequest(
            "vivo s2 256GB sapphire blue", provider_policy="database_only"
        )
    )

    assert report["match_mode"] == "expanded_colors"
    assert report["requested_match"] is None
    assert report["variant_groups"][0]["variant"]["storage"] == "256GB"
    assert report["variant_groups"][0]["match_type"] == "same_configuration_other_color"


def test_agent_trace_exposes_only_display_safe_prompts_for_real_llm_stages():
    report = MarketInvestigatorAgent(VariantDatabase(), []).analyze(
        MarketAgentRequest("vivo s2", provider_policy="database_only")
    )

    assert [event["tool"] for event in report["agent_trace"]] == [
        "parse_product_request",
        "get_market_snapshot",
        "group_product_variants",
        "verify_market_report",
        "generate_market_summary",
    ]
    assert all(
        event["display_prompt"] is None
        for event in report["agent_trace"][:-1]
    )
    assert report["agent_trace"][-1]["status"] == "skipped"
    assert "verified Indian-market" in report["agent_trace"][-1]["display_prompt"]


def test_agent2_openai_configuration_is_isolated_and_requires_a_key(monkeypatch):
    for name in (
        "MARKET_AGENT_LLM_BASE_URL", "MARKET_AGENT_LLM_API_KEY",
        "MARKET_AGENT_LLM_MODEL", "OPENAI_API_KEY", "LLM_BASE_URL",
        "LLM_API_KEY", "LLM_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)

    try:
        MarketInvestigatorAgent._llm_runtime_config()
    except ValueError as exc:
        assert "MARKET_AGENT_LLM_API_KEY" in str(exc)
    else:
        raise AssertionError("OpenAI configuration accepted a missing key")

    monkeypatch.setenv("MARKET_AGENT_LLM_API_KEY", "test-only-key")
    assert MarketInvestigatorAgent._llm_runtime_config() == (
        "https://api.openai.com/v1", "test-only-key", "gpt-6-luna"
    )


def test_variant_parser_supports_non_phone_electronics():
    television = variant_facets('Samsung 55 inch 5th Gen TV Black')
    watch = variant_facets('Apple Watch 45mm GPS Starlight with charger')
    laptop = variant_facets('Laptop 16GB RAM 1TB SSD Refurbished')

    assert television["screen_size"] == "55 inch"
    assert television["generation"] == "5 Gen"
    assert watch["device_size"] == "45mm"
    assert watch["bundle"] == "With Charger"
    assert laptop["ram"] == "16GB"
    assert laptop["storage"] == "1TB"
    assert laptop["condition"] == "Refurbished"


def test_conditional_bank_price_is_separate_and_offer_bound():
    rows = [offer_row("offer-1", "amazon.in", 70000)]
    promotions = [{
        "promotion_id": "promo-1",
        "offer_id": "offer-1",
        "promotion_type": "BANK",
        "bank": "HDFC Bank",
        "card_type": "Credit Card",
        "description": "10% instant discount up to ₹2,000",
        "amount": 2000,
        "percent": 10,
        "is_emi": False,
    }]

    report = build_market_report(
        product={"canonical_id": "B0EXAMPLE1", "title": "Apple iPhone 16 128GB Black"},
        offers=rows,
        promotions=promotions,
        sales=[],
        context={},
    ).to_dict()

    assert report["best_unconditional_offer"]["price"] == 70000
    assert report["best_conditional_offer"]["price"] == 68000
    assert report["best_conditional_offer"]["promotion_ids"] == ["promo-1"]
    assert report["best_conditional_offer"]["eligibility"] == "unknown"
    assert set(report["missing_inputs"]) == {"bank", "card_type"}


def test_material_variant_conflicts_are_rejected():
    compatible, conflicts = variants_compatible(
        "Apple iPhone 16 128GB Black", "Apple iPhone 16 256GB Black"
    )
    assert compatible is False
    assert "capacity" in conflicts


def test_verifier_rejects_a_tampered_conditional_price():
    rows = [offer_row("offer-1", "amazon.in", 70000)]
    promotions = [{
        "promotion_id": "promo-1",
        "offer_id": "offer-1",
        "promotion_type": "BANK",
        "description": "₹2,000 instant discount",
        "amount": 2000,
        "percent": None,
        "is_emi": False,
    }]
    report = build_market_report(
        product={"canonical_id": "B0EXAMPLE1", "title": "Apple iPhone 16 128GB Black"},
        offers=rows,
        promotions=promotions,
        sales=[],
    ).to_dict()
    report["best_conditional_offer"]["price"] = 100

    try:
        verify_market_report(report, rows, promotions)
    except MarketReportVerificationError as exc:
        assert "does not produce" in str(exc)
    else:
        raise AssertionError("tampered price passed verification")


def test_verifier_checks_prices_inside_every_variant_group():
    database = VariantDatabase()
    report = MarketInvestigatorAgent(database, []).analyze(
        MarketAgentRequest("vivo s2", provider_policy="database_only")
    )
    report["variant_groups"][1]["best_unconditional_offer"]["price"] = 1

    try:
        verify_market_report(report, database.offers, [])
    except MarketReportVerificationError as exc:
        assert "variant_groups[1]" in str(exc)
    else:
        raise AssertionError("tampered variant price passed verification")
