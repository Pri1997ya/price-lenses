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


# ------------------------------------------------------------ search fixes
FLIPKART_HTML = """<html><body><main><h2>Returns Policy</h2>
<p>Returns is a scheme provided by respective sellers directly under this policy.</p>
<table>
<tr><th>Category</th><th>Returns Window, Actions Possible and Conditions (if any)</th></tr>
<tr><td>Mobiles (non-premium brands)</td><td><p>7 days Replacement only</p>
<p>Free replacement will be provided within 7 days if the product is delivered in
defective/damaged condition or different from the ordered item.</p></td></tr>
<tr><td>Mobiles (premium brands: Apple, Samsung)</td><td>7 Days Service Center
Replacement/Repair only <ul><li>Brand assistance for device related issues is subject
to brand warranty guidelines and service policies.</li></ul></td></tr>
<tr><td>Furniture</td><td>10 days Replacement only</td></tr>
</table>
<h2>Samsung DOA</h2><p>If DOA is approved by the brand, share the approval certificate
with Flipkart support to process the complaint for your device.</p>
</main></body></html>"""


def test_table_rows_stay_on_one_line():
    from tools.policy_corpus import html_to_markdown

    text, _ = html_to_markdown(FLIPKART_HTML)
    assert "- Mobiles (non-premium brands) | 7 days Replacement only Free replacement" in text
    assert "- Mobiles (premium brands: Apple, Samsung) | 7 Days Service Center" in text


def test_retailers_named_in_question():
    from tools.policy_rag import retailers_in_question

    assert retailers_in_question("Can I return a phone bought on Flipkart?") == ["flipkart"]
    assert retailers_in_question("Reliance Digital vs Croma returns") == ["croma", "reliance_digital"]
    assert retailers_in_question("Can I return a phone?") == []


def test_phone_question_finds_the_mobiles_row(tmp_path, monkeypatch):
    from tools.policy_corpus import html_to_markdown

    monkeypatch.setenv("POLICY_MIN_RELEVANCE", "0.1")
    text, _ = html_to_markdown(FLIPKART_HTML)
    index = PolicyIndex(HashEmbedder(), tmp_path / "chroma")
    index.build([
        PolicyDocument("fk", "flipkart", "return_policy", "https://fk", "2026-09-27", text),
        PolicyDocument("rd", "reliance_digital", "return_policy", "https://rd", "2026-09-27",
                       "# Returns\n\nMobile phones can be returned within 7 days at Reliance "
                       "Digital stores for a refund if unopened."),
    ])
    answer = PolicyAdvisor(index, llm=None).answer("Can I return a phone bought on Flipkart?")
    assert {hit.retailer for hit in answer.citations} == {"flipkart"}
    assert "7 days Replacement only" in answer.citations[0].text
    assert "keyword" in answer.citations[0].matched_by  # "phone" matched "Mobiles"


# ------------------------------------------------------------ corpus clean-up
def test_link_only_menus_are_dropped_but_policy_lists_kept():
    from tools.policy_corpus import html_to_markdown

    html = """<html><body><div class="menu"><ul>
      <li><a href="/b">boAt</a></li><li><a href="/a">Apple</a></li>
      <li><a href="/j">JBL</a></li><li><a href="/s">Sony</a></li></ul></div>
    <main><h2>Returns</h2><ul>
      <li>Items can be returned within 7 days.</li>
      <li>See the <a href="/faq">FAQ</a> for pickup details.</li>
      <li>Keep the invoice.</li><li>Keep the original box.</li></ul>
    <ul><li><a href="#pickup">Return Pickup</a></li><li><a href="#self">Self-Ship</a></li>
      <li><a href="#refund">Refunds</a></li><li><a href="#faq">FAQ</a></li></ul></main></body></html>"""
    text, _ = html_to_markdown(html)
    assert "boAt" not in text and "Self-Ship" not in text  # menu and table of contents
    assert "- Items can be returned within 7 days." in text
    assert "- See the FAQ for pickup details." in text  # a link inside real text stays


def test_long_paragraph_chunks_restart_on_a_word():
    from tools.policy_corpus import chunk_document

    words = " ".join(f"word{n:03d}" for n in range(400))
    document = PolicyDocument("s", "croma", "return_policy", "https://c", "2026-09-27", f"# T\n\n{words}")
    for chunk in chunk_document(document, size=300, overlap=60)[1:]:
        assert chunk.text.split("\n", 1)[1].startswith("word")


# ------------------------------------------------------------ cut-off suggestion
def _hit(relevance, text, matched_by):
    from tools.policy_rag import PolicyHit

    return PolicyHit(1, text, relevance, "amazon", "return_policy", "s", "https://a", "2026-09-27",
                     matched_by=matched_by)


class _FakeIndex:
    """Replays hits; applies the cut-off the way PolicyIndex.search does."""

    def __init__(self, hits_by_question, threshold):
        self.hits, self.threshold = hits_by_question, threshold

    def search(self, question, retailers=None, k=3, min_relevance=None):
        cut = self.threshold if min_relevance is None else min_relevance
        return [h for h in self.hits[question] if "keyword" in h.matched_by or h.relevance >= cut]


def test_suggestion_counts_keyword_matches_as_safe(capsys):
    # The reported run: a correct answer at 0.28 (also a keyword match) and a
    # meaning-only stray at 0.33. Raising the cut-off above 0.33 is safe.
    index = _FakeIndex({
        "replacement window": [_hit(0.28, "Replacement within 7 days", ("semantic", "keyword"))],
        "moon": [_hit(0.33, "Self-ship the item", ("semantic",))],
    }, threshold=0.25)
    result = eval_policy_questions.evaluate(index, [
        {"question": "replacement window", "expect_text": "days"},
        {"question": "moon", "expect": "no_match"},
    ])
    assert result["suggested_min_relevance"] == 0.35
    assert "also match by keyword" in capsys.readouterr().out


def test_no_suggestion_when_a_meaning_only_answer_is_weaker(capsys):
    index = _FakeIndex({
        "window": [_hit(0.28, "within 7 days", ("semantic",))],
        "moon": [_hit(0.33, "Self-ship", ("semantic",))],
    }, threshold=0.25)
    result = eval_policy_questions.evaluate(index, [
        {"question": "window", "expect_text": "days"}, {"question": "moon", "expect": "no_match"},
    ])
    assert result["suggested_min_relevance"] is None
    assert "Improve that source" in capsys.readouterr().out
