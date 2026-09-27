import json
import sys
from datetime import date
from pathlib import Path

import pytest

from tests.test_eligibility_agent import advisor_factory, offers
from tools.eligibility_agent import CATEGORY_PATTERNS, run_eligibility_analysis
from tools.policy_corpus import CorpusError, PolicyDocument, stale_sources
from tools.policy_rag import HashEmbedder, PolicyIndex
from tools.return_windows import (
    DEFAULT_RETURN_WINDOWS_FILE,
    find_return_window,
    load_return_windows,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "policies"))
import eval_policy_questions  # noqa: E402

CATEGORIES = {name for name, _ in CATEGORY_PATTERNS}


# ------------------------------------------------------------ staleness
def test_stale_sources_oldest_first_and_unreadable_dates():
    today = date(2026, 9, 27)
    stale = stale_sources(
        {"fresh": "2026-09-01", "old": "2026-05-01", "older": "2025-01-01", "bad": "last week"},
        max_age_days=90, today=today,
    )
    assert stale == [("bad", None), ("older", 634), ("old", 149)]
    assert stale_sources({"edge": "2026-06-29"}, 90, today) == []  # exactly 90 days is fine


def test_index_stats_report_stale_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("POLICY_STALE_DAYS", "30")
    index = PolicyIndex(HashEmbedder(), tmp_path / "chroma")
    index.build([
        PolicyDocument("new", "amazon", "return_policy", "https://a", date.today().isoformat(),
                       "# Returns\n\nItems can be returned within 10 days."),
        PolicyDocument("old", "croma", "return_policy", "https://c", "2020-01-01",
                       "# Returns\n\nItems can be returned within 7 days."),
    ])
    assert [source_id for source_id, _ in index.stats()["stale_sources"]] == ["old"]


# ------------------------------------------------------------ return windows
def row(**changes):
    base = {"retailer": "flipkart", "category": "mobile phones", "window_days": 7,
            "action": "replacement", "conditions": "Only if delivered damaged",
            "source_id": "test-source", "source_url": "https://example.test/policy",
            "retrieved_at": "2026-09-27"}
    return {**base, **changes}


def write_table(tmp_path, *rows):
    path = tmp_path / "return_windows.json"
    path.write_text(json.dumps({"rows": list(rows)}))
    return path


def test_lookup_prefers_category_then_retailer_default(tmp_path):
    rows = load_return_windows(write_table(
        tmp_path, row(), row(category="*", window_days=10, action="return", conditions=""),
    ), CATEGORIES)
    phone = find_return_window(rows, "flipkart", "mobile phones")
    assert phone.summary == "Flipkart, mobile phones: 7 days, replacement only (Only if delivered damaged)"
    assert find_return_window(rows, "flipkart", "laptops").summary == (
        "Flipkart, all products: 10 days, return for a refund"
    )
    assert find_return_window(rows, "croma", "mobile phones") is None


@pytest.mark.parametrize("bad, message", [
    (row(retailer="ebay"), "unknown retailer"),
    (row(retailer="regulation"), "unknown retailer"),
    (row(category="toasters"), "unknown category"),
    (row(action="maybe"), "action must be one of"),
    (row(window_days="7"), "whole number"),
    (row(window_days=-1), "whole number"),
    (row(retrieved_at="yesterday"), "YYYY-MM-DD"),
    (row(source_url=""), "missing source_url"),
])
def test_bad_rows_are_rejected(tmp_path, bad, message):
    with pytest.raises(CorpusError, match=message):
        load_return_windows(write_table(tmp_path, bad), CATEGORIES)


def test_duplicate_rows_are_rejected(tmp_path):
    with pytest.raises(CorpusError, match="two rows"):
        load_return_windows(write_table(tmp_path, row(), row(window_days=10)), CATEGORIES)


def test_missing_table_means_no_rows(tmp_path):
    assert load_return_windows(tmp_path / "absent.json") == []


def test_committed_table_is_valid():
    rows = load_return_windows(DEFAULT_RETURN_WINDOWS_FILE, CATEGORIES)
    source_ids = {s["id"] for s in json.loads((ROOT / "data/policies/sources.json").read_text())["sources"]}
    assert all(r.source_id in source_ids for r in rows)


def test_agent_reports_reviewed_window(tmp_path, monkeypatch):
    monkeypatch.setenv("POLICY_MIN_RELEVANCE", "0")
    table = write_table(tmp_path, row())
    report = run_eligibility_analysis(
        "B0TEST0001", "Samsung Galaxy S24 5G",
        offers_loader=offers, advisor_factory=advisor_factory(tmp_path),
        return_windows_loader=lambda: load_return_windows(table, CATEGORIES),
    )
    assert set(report["return_windows"]) == {"flipkart"}
    assert report["return_windows"]["flipkart"]["window_days"] == 7
    assert any("Reviewed return window: Flipkart" in line for line in report["agent_trace"])


def test_broken_table_downgrades_instead_of_raising(tmp_path):
    def broken():
        raise CorpusError("return_windows.json row 1: unknown retailer 'ebay'")

    report = run_eligibility_analysis(
        "B0TEST0001", "Samsung Galaxy S24 5G",
        offers_loader=offers, advisor_factory=advisor_factory(tmp_path), return_windows_loader=broken,
    )
    assert report["return_windows"] == {}
    assert any("Return window table unavailable" in error for error in report["errors"])


# ------------------------------------------------------------ evaluation set
def test_every_retailer_has_evaluation_questions():
    questions = eval_policy_questions.DEFAULT_QUESTIONS
    covered = {q.get("retailer") for q in questions}
    assert {"amazon", "flipkart", "croma", "reliance_digital", "vijay_sales", "regulation"} <= covered
    assert sum(q.get("expect") == "no_match" for q in questions) >= 4
