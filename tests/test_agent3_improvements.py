import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from tools.eligibility_agent import category_terms, infer_category
from tools.policy_corpus import PolicyDocument, PolicySource
from tools.policy_rag import HashEmbedder, PolicyAdvisor, PolicyIndex, about_other_product

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "policies"))
import eval_policy_questions  # noqa: E402
import fetch_policies  # noqa: E402


# ------------------------------------------------------------ product types
@pytest.mark.parametrize(
    "title, expected",
    [
        ("OnePlus Nord Buds 4 Pro E518A Raven Black in", "headphones and earbuds"),
        ("OnePlus Bullets Wireless Z3 in-Ear Neckband with 12.4mm Drivers", "headphones and earbuds"),
        ("boAt Bassheads 300C Wired Earphones,Type-C Jack, 120cm Cable (Active Black)",
         "headphones and earbuds"),
        ("realme Buds Air 7 Pro with Ai Live Translation", "headphones and earbuds"),
        ("Redmi Pad 2, 11-inch (27.94 CM CM), LCD Display, 4GB RAM, 128GB ROM", "tablets"),
        ("Pikkme Flip Cover Leather Finish | Inside TPU with Card Pockets", "accessories"),
        ("OpenTech® Military-Grade Gorilla Tempered Glass for iPhone 15 / iPhone 16", "accessories"),
        ("Samsung Original 25W USB Type-C Travel Adaptor Without Cable for Google Pixel",
         "accessories"),
        ("Ambrane Unbreakable 3A Fast Charging 1.5m Braided Type C Cable for Smartphones",
         "accessories"),
        ("Samsung Galaxy M35 5G (Thunder Grey,6GB RAM,128GB Storage)| Corning Gorilla Glass "
         "Victus+| 120Hz Super AMOLED Display| AI| Without Charger", "mobile phones"),
        ("Motorola Edge 70 ( Pantone Bronze Green, 8GB RAM, 256GB Storage)", "mobile phones"),
        ("Nothing Phone (3A) 5G (Blue, 8GB RAM, 128GB Storage)", "mobile phones"),
        ("iQOO Z11x 5G (Eclipse Black, 6GB RAM, 128GB Storage)", "mobile phones"),
        ("OnePlus 13R 12GB/256GB", "mobile phones"),
        ("Apple iPhone 15 (128 GB) - Black", "mobile phones"),
        ("Noise Pulse 2 Max 1.85\" Display, Bluetooth Calling Smart Watch", "smartwatches"),
        ("boAt Lunar Vista Smartwatch", "smartwatches"),
        ("Apple MacBook Air M3", "laptops"),
        ("Mystery gadget", "electronics"),
    ],
)
def test_real_catalog_titles_get_the_right_product_type(title, expected):
    assert infer_category(title) == expected


def test_every_title_in_the_repo_data_is_classified():
    titles = set()
    for path in (ROOT / "data" / "raw_apify").glob("*.json"):
        titles.update(item["name"] for item in json.loads(path.read_text()) if item.get("name"))
    assert titles
    assert all(infer_category(title) != "electronics" for title in titles)


def test_category_terms_split_own_and_other():
    own, other = category_terms("mobile phones")
    assert "phone" in own and "laptop" in other and "phone" not in other
    assert category_terms("electronics") == ((), ())


# ------------------------------------------------------------ hybrid search
def documents():
    return [
        PolicyDocument("fk", "flipkart", "return_policy", "https://fk", "2026-09-27",
                       "# Mobiles\n\nMobile phones are eligible for replacement only within 7 days "
                       "of delivery. A technician visit may be required.\n\n"
                       "# Laptops\n\nLaptops can be returned within 7 days for a full refund if the "
                       "brand seal is intact. A restocking fee of 10 percent applies to opened laptops."),
        PolicyDocument("cr", "croma", "return_policy", "https://croma", "2026-09-27",
                       "# Returns\n\nItems that are non-returnable include software and consumables. "
                       "Refunds are processed within 7 working days."),
    ]


@pytest.fixture
def index(tmp_path):
    built = PolicyIndex(HashEmbedder(), tmp_path / "chroma")
    built.build(documents())
    return built


def test_keyword_match_survives_a_strict_cut_off(index):
    hits = index.search("restocking fee for laptops", ["flipkart"], min_relevance=0.99)
    assert hits and hits[0].heading == "Laptops"
    assert hits[0].matched_by == ("keyword",)
    assert not index.search("restocking fee for laptops", ["flipkart"],
                            min_relevance=0.99, hybrid=False)


def test_single_common_word_does_not_pull_in_everything(index):
    # Keyword search needs two matching terms, so a lone word adds nothing.
    assert index.search("fee zebra", ["croma"], min_relevance=0.99) == []


def test_hybrid_marks_passages_found_both_ways(index):
    hits = index.search("replacement for mobile phones", ["flipkart"], min_relevance=0.0)
    assert hits[0].heading == "Mobiles"
    assert set(hits[0].matched_by) == {"semantic", "keyword"}


# ------------------------------------------------------------ restriction flags
def test_laptop_clause_is_not_flagged_for_a_phone(index, monkeypatch):
    monkeypatch.setenv("POLICY_MIN_RELEVANCE", "0")
    own, other = category_terms("mobile phones")
    answer = PolicyAdvisor(index, llm=None).answer(
        "phone return window and conditions", ["flipkart"],
        include_regulations=False, boost_terms=own, exclude_terms=other,
    )
    labels = {r["restriction"] for r in answer.restrictions}
    assert "replacement only (no refund)" in labels
    assert "seal / packaging must be intact" not in labels  # laptop-only clause


def test_generic_clause_is_still_flagged(index, monkeypatch):
    monkeypatch.setenv("POLICY_MIN_RELEVANCE", "0")
    own, other = category_terms("mobile phones")
    answer = PolicyAdvisor(index, llm=None).answer(
        "which items are non-returnable", ["croma"],
        include_regulations=False, boost_terms=own, exclude_terms=other,
    )
    assert "non-returnable" in {r["restriction"] for r in answer.restrictions}


def test_about_other_product():
    hit = SimpleNamespace(heading="Laptops", text="Laptops can be returned")
    own, other = category_terms("mobile phones")
    assert about_other_product(hit, own, other) is True
    assert about_other_product(hit, own, ()) is False
    generic = SimpleNamespace(heading="Returns", text="Most items can be returned")
    assert about_other_product(generic, own, other) is False


# ------------------------------------------------------------ fetch script
def source(url, fmt="html"):
    return PolicySource(id="s1", retailer="croma", doc_type="return_policy", url=url, format=fmt)


def test_missing_packages_and_pdf_detection(monkeypatch):
    import importlib.util

    assert fetch_policies.missing_packages() == []
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name: None if name == "bs4" else real(name))
    assert fetch_policies.missing_packages() == ["beautifulsoup4"]
    assert fetch_policies.is_pdf_source(source("https://x.test/rules.pdf?x=1"))
    assert not fetch_policies.is_pdf_source(source("https://x.test/returns"))


@pytest.fixture
def one_source(tmp_path):
    path = tmp_path / "sources.json"
    path.write_text(json.dumps({"sources": [
        {"id": "s1", "retailer": "croma", "doc_type": "return_policy",
         "url": "https://x.test/rules.pdf", "format": "pdf"}]}))
    return path


def http_error(status):
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError(f"{status} error", response=response)


@pytest.mark.parametrize("status, label", [(403, "[blocked]"), (404, "[gone   ]"), (500, "[failed ]")])
def test_http_errors_get_specific_messages(one_source, tmp_path, monkeypatch, capsys, status, label):
    def fail(*_args, **_kwargs):
        raise http_error(status)

    monkeypatch.setattr(fetch_policies, "fetch_text", fail)
    assert fetch_policies.main(["--sources", str(one_source), "--out", str(tmp_path)]) == 1
    assert label in capsys.readouterr().out


def test_textless_pdf_message(one_source, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(fetch_policies, "fetch_text", lambda *_a, **_k: ("", None))
    fetch_policies.main(["--sources", str(one_source), "--out", str(tmp_path)])
    assert "no text layer" in capsys.readouterr().out


def test_missing_package_stops_before_fetching(one_source, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(fetch_policies, "missing_packages", lambda: ["pypdf"])
    assert fetch_policies.main(["--sources", str(one_source), "--out", str(tmp_path)]) == 2
    assert "pip install -r requirements.txt" in capsys.readouterr().out


# ------------------------------------------------------------ evaluation script
def test_evaluation_reports_passes_and_misses(index, capsys):
    result = eval_policy_questions.evaluate(index, [
        {"question": "replacement for mobile phones", "retailer": "flipkart",
         "expect_text": "replacement"},
        {"question": "restocking fee laptops", "retailer": "croma", "expect_text": "restocking"},
        {"question": "zebra giraffe", "expect": "no_match"},
    ])
    assert (result["passed"], result["total"]) == (2, 3)
    assert "MISS" in capsys.readouterr().out


def test_policy_llm_sends_no_temperature_unless_configured(monkeypatch):
    from tools import policy_rag

    monkeypatch.delenv("POLICY_LLM_TEMPERATURE", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "test-only-key")
    assert policy_rag.default_llm().temperature is None
    monkeypatch.setenv("POLICY_LLM_TEMPERATURE", "0")
    assert policy_rag.default_llm().temperature == 0
