from types import SimpleNamespace

import pytest

from tools.policy_corpus import PolicyDocument
from tools.policy_rag import (
    COLLECTION_NAME,
    HashEmbedder,
    PolicyAdvisor,
    PolicyIndex,
    PolicyIndexError,
    validate_citations,
)


def _documents():
    return [
        PolicyDocument(
            "flipkart-return-policy", "flipkart", "return_policy",
            "https://www.flipkart.com/pages/returnpolicy", "2026-09-24",
            "# Mobiles\n\nMobile phones are eligible for replacement only within 7 days of delivery "
            "for damaged or defective items. No refund is offered on mobiles.",
        ),
        PolicyDocument(
            "amazon-returns-policy", "amazon", "return_policy",
            "https://www.amazon.in/returns", "2026-09-23",
            "# Laptops\n\nLaptops can be returned within 10 days of delivery for a full refund "
            "if the brand seal is intact.",
        ),
        PolicyDocument(
            "ecommerce-rules", "regulation", "regulation",
            "https://example.gov/rules.pdf", "2026-09-20",
            "# Refunds\n\nEvery e-commerce entity shall effect all accepted refund requests "
            "within a reasonable period of time.",
        ),
    ]


class FakeLLM:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, 0

    def invoke(self, messages):
        self.calls += 1
        if self.error:
            raise self.error
        return SimpleNamespace(content=self.reply)


@pytest.fixture
def index(tmp_path, monkeypatch):
    # Hash embeddings give low absolute similarities; relevance filtering is
    # tuned for the real model, so disable it for these behaviour tests.
    monkeypatch.setenv("POLICY_MIN_RELEVANCE", "0")
    built = PolicyIndex(HashEmbedder(), tmp_path / "chroma")
    built.build(_documents())
    return built


def test_build_and_filtered_search(index):
    hits = index.search("replacement window for mobile phones", ["flipkart"], min_relevance=0.0)
    assert hits
    assert {hit.retailer for hit in hits} <= {"flipkart", "regulation"}
    assert hits[0].retailer == "flipkart"
    assert hits[0].source_url == "https://www.flipkart.com/pages/returnpolicy"
    no_rules = index.search("refund", ["amazon"], include_regulations=False, min_relevance=0.0)
    assert {hit.retailer for hit in no_rules} == {"amazon"}


def test_stats_report_documents_and_dates(index):
    stats = index.stats()
    assert stats["documents"] == 3
    assert stats["oldest_retrieval"] == "2026-09-20"
    assert stats["retailers"]["flipkart"] == ["flipkart-return-policy"]


def test_rebuild_replaces_old_index(index):
    index.build(_documents()[:1])
    assert index.stats()["documents"] == 1


def test_failed_rebuild_keeps_previous_index(index):
    class Exploding(HashEmbedder):
        def embed(self, texts):
            raise RuntimeError("embedding service down")

    broken = PolicyIndex(Exploding(), index.persist_dir)
    broken.embedder.name = index.embedder.name
    with pytest.raises(RuntimeError):
        broken.build(_documents())
    assert index.stats()["documents"] == 3
    assert [c.name for c in index._client.list_collections()] == [COLLECTION_NAME]


def test_missing_index_and_embedder_mismatch(tmp_path, index):
    with pytest.raises(PolicyIndexError, match="not been built"):
        PolicyIndex(HashEmbedder(), tmp_path / "empty").search("anything")
    with pytest.raises(PolicyIndexError, match="Rebuild"):
        PolicyIndex(HashEmbedder(dimensions=64), index.persist_dir).search("anything")


def test_grounded_llm_answer_is_accepted(index):
    llm = FakeLLM("Flipkart offers replacement only within 7 days for mobiles [1].")
    answer = PolicyAdvisor(index, llm=llm).answer("mobile phone replacement", ["flipkart"])
    assert answer.mode == "llm"
    assert "[1]" in answer.answer
    assert any(r["restriction"].startswith("replacement only") for r in answer.restrictions)


@pytest.mark.parametrize(
    "reply", ["Flipkart gives 30 days.", "See passage [9] for details."]
)
def test_ungrounded_llm_answer_falls_back_to_passages(index, reply):
    answer = PolicyAdvisor(index, llm=FakeLLM(reply)).answer("mobile phone replacement", ["flipkart"])
    assert answer.mode == "extractive"
    assert "grounding check" in answer.note
    assert "[1]" in answer.answer


def test_offline_llm_falls_back_and_is_not_retried(index):
    llm = FakeLLM(error=ConnectionError("gateway down"))
    advisor = PolicyAdvisor(index, llm=llm)
    first = advisor.answer("mobile phone replacement", ["flipkart"])
    second = advisor.answer("laptop return", ["amazon"])
    assert first.mode == second.mode == "extractive"
    assert llm.calls == 1


def test_no_match_says_not_stated(index):
    answer = PolicyAdvisor(index, llm=None).answer("mobile replacement", ["croma"], include_regulations=False)
    assert answer.mode == "no_match"
    assert "not stated" in answer.answer


def test_validate_citations():
    hits = [SimpleNamespace(number=1), SimpleNamespace(number=2)]
    assert validate_citations("Yes [1][2].", hits) is None
    assert "cited no passages" in validate_citations("Yes.", hits)
    assert "[3]" in validate_citations("Yes [3].", hits) or "3" in validate_citations("Yes [3].", hits)
