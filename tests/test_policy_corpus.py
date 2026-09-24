from pathlib import Path

import pytest

from tools.policy_corpus import (
    DEFAULT_SOURCES_FILE,
    CorpusError,
    PolicyDocument,
    chunk_document,
    html_to_markdown,
    load_corpus,
    load_sources,
    parse_document,
    render_document,
    retailer_for_marketplace,
)

HEADER = """---
source_id: flipkart-return-policy
retailer: flipkart
doc_type: return_policy
source_url: https://www.flipkart.com/pages/returnpolicy
retrieved_at: 2026-09-24
---
"""


def test_sources_file_is_valid_and_excludes_cross_check_and_portals():
    sources = load_sources(DEFAULT_SOURCES_FILE)
    by_id = {source.id: source for source in sources}
    assert by_id["zlash-croma-summary"].ingest is False
    assert by_id["ejagriti-portal"].ingest is False
    assert by_id["dca-consumer-protection-acts-index"].ingest is False
    ingest = {source.retailer for source in sources if source.ingest}
    assert ingest == {"amazon", "flipkart", "croma", "reliance_digital", "vijay_sales", "regulation"}


def test_parse_and_render_round_trip():
    document = parse_document(HEADER + "# Returns\n\nMobiles: 7 days replacement only.")
    assert document.retailer == "flipkart"
    assert document.retrieved_at == "2026-09-24"
    assert parse_document(render_document(document)).body == document.body


@pytest.mark.parametrize(
    "text",
    [
        "# no header at all",
        "---\nsource_id: x\n---\nbody",  # missing required fields
        HEADER + "   ",  # empty body
    ],
)
def test_parse_rejects_bad_documents(text):
    with pytest.raises(CorpusError):
        parse_document(text)


def test_load_corpus_skips_bad_and_duplicate_files(tmp_path: Path):
    (tmp_path / "flipkart").mkdir()
    (tmp_path / "flipkart" / "good.md").write_text(HEADER + "Body text")
    (tmp_path / "flipkart" / "dupe.md").write_text(HEADER + "Other text")
    (tmp_path / "flipkart" / "bad.md").write_text("no header")
    (tmp_path / "README.md").write_text("# readme is ignored")
    documents, problems = load_corpus(tmp_path)
    assert len(documents) == 1
    assert len(problems) == 2


def test_html_to_markdown_keeps_headings_and_drops_navigation():
    html = """
    <html><head><title>Return Policy</title><script>var x=1</script></head>
    <body><nav>Home | Cart</nav><main>
      <h1>Returns</h1><p>Mobiles can be replaced within 7 days.</p>
      <ul><li>Laptops: 10 days</li><li>Laptops: 10 days</li></ul>
    </main><footer>© Flipkart</footer></body></html>
    """
    text, title = html_to_markdown(html)
    assert title == "Return Policy"
    assert "# Returns" in text
    assert "- Laptops: 10 days" in text
    assert text.count("Laptops: 10 days") == 1
    assert "Cart" not in text and "var x" not in text and "©" not in text


def test_chunking_keeps_heading_and_metadata():
    body = "# Mobiles\n\n" + ("Replacement within 7 days of delivery. " * 60) + "\n\n# Laptops\n\nReturn in 10 days."
    document = PolicyDocument("fk", "flipkart", "return_policy", "https://f", "2026-09-24", body)
    chunks = chunk_document(document, size=500, overlap=80)
    assert len(chunks) >= 3
    assert all(len(chunk.text) <= 500 + len("Mobiles\n") for chunk in chunks)
    assert chunks[0].text.startswith("Mobiles")
    assert chunks[-1].text.startswith("Laptops")
    assert chunks[-1].metadata["retailer"] == "flipkart"
    assert chunks[-1].metadata["source_url"] == "https://f"
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_marketplace_mapping():
    assert retailer_for_marketplace("amazon.in") == "amazon"
    assert retailer_for_marketplace("www.flipkart.com") == "flipkart"
    assert retailer_for_marketplace("reliancedigital.in") == "reliance_digital"
    assert retailer_for_marketplace("google_shopping") is None
