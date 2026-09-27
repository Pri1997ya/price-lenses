from datetime import datetime, timezone

import orchestrator
from tools.eligibility_agent import infer_category, run_eligibility_analysis
from tools.policy_corpus import PolicyDocument
from tools.policy_rag import HashEmbedder, PolicyAdvisor, PolicyIndex


def offers(_canonical_id):
    now = datetime.now(timezone.utc)
    return [
        {"offer_id": "a", "marketplace": "flipkart.com", "seller_name": "RetailNet", "seller_rating": 4.4,
         "availability": "In stock", "price": 69999.0, "fetched_at": now, "url": "https://fk", "title": "Phone"},
        {"offer_id": "b", "marketplace": "croma.com", "seller_name": "Croma", "seller_rating": None,
         "availability": "In stock", "price": 72999.0, "fetched_at": now, "url": "https://croma", "title": "Phone"},
    ]


def advisor_factory(tmp_path):
    index = PolicyIndex(HashEmbedder(), tmp_path / "chroma")
    index.build(
        [
            PolicyDocument("fk", "flipkart", "return_policy", "https://fk/policy", "2026-09-24",
                           "# Mobiles\n\nMobile phones are replacement only within 7 days of delivery."),
            PolicyDocument("cr", "croma", "return_policy", "https://croma/policy", "2026-09-24",
                           "# Returns\n\nMobile phones bought at Croma can be returned within 7 days if unopened."),
        ]
    )
    return lambda: PolicyAdvisor(index, llm=None)


def test_infer_category():
    assert infer_category("Apple iPhone 16 (128 GB) - Black") == "mobile phones"
    assert infer_category("Apple MacBook Air M3") == "laptops"
    assert infer_category("Sony WH-1000XM5 Headphones") == "headphones and earbuds"
    assert infer_category("Mystery gadget") == "electronics"


def test_full_report(tmp_path, monkeypatch):
    monkeypatch.setenv("POLICY_MIN_RELEVANCE", "0")
    report = run_eligibility_analysis(
        "B0TEST0001", "Samsung Galaxy S24 5G", "Can I get a refund on a phone?",
        offers_loader=offers, advisor_factory=advisor_factory(tmp_path),
    )
    assert report["status"] == "ok"
    assert report["category"] == "mobile phones"
    assert report["cheapest_store"]["label"] == "Flipkart"
    assert report["safest_store"]["label"] == "Croma"
    assert set(report["policies"]) == {"flipkart", "croma"}
    assert report["policies"]["flipkart"]["citations"][0]["source_url"] == "https://fk/policy"
    assert "replacement only" in report["return_policy_warning"]
    assert report["user_policy_answer"]["question"] == "Can I get a refund on a phone?"


def test_failures_downgrade_instead_of_raising():
    def broken_loader(_):
        raise ConnectionError("database down")

    def broken_advisor():
        raise RuntimeError("index missing")

    report = run_eligibility_analysis(
        "B0TEST0001", "Phone", offers_loader=broken_loader, advisor_factory=broken_advisor
    )
    assert report["status"] == "error"
    assert len(report["errors"]) == 2


def test_graph_node_never_raises(monkeypatch):
    def explode(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("tools.eligibility_agent.run_eligibility_analysis", explode)
    result = orchestrator.eligibility_agent_node({"canonical_id": "B0TEST0001"})
    assert result["eligibility_report"]["status"] == "error"
    assert orchestrator.eligibility_agent_node({})["eligibility_report"] == {}
