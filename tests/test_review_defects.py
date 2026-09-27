import json
import sys
from datetime import date
from pathlib import Path

import pytest
import requests

from tests.test_eligibility_agent import advisor_factory, offers
from tools.eligibility_agent import run_eligibility_analysis
from tools.review_corpus import (
    Review,
    load_reviews,
    normalize_marketplace_review,
    normalize_reddit_item,
    normalize_youtube_comment,
    parse_date,
    parse_rating,
    review_path,
    save_reviews,
)
from tools.review_defects import DEFECTS, detect_defects, reported_sentence
from tools.review_sources import ApifyReviewSource, RedditSource, YouTubeSource, _fill

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "reviews"))
import fetch_reviews  # noqa: E402

TODAY = date(2026, 9, 27)
DEFECT = {defect.key: defect for defect in DEFECTS}


def review(text, n=0, source="amazon", when="2026-08-01", rating=None):
    return Review(f"{source}:{n}", "B0TEST0001", source, text, url=f"https://x.test/{n}", date=when, rating=rating)


# ------------------------------------------------------------ normalisers
def test_marketplace_review_accepts_common_field_names():
    item = {"reviewId": "R1", "reviewTitle": "Heats up", "reviewDescription": "Gets  very hot\nwhile gaming",
            "ratingScore": "2.0 out of 5 stars", "date": "Reviewed in India on 3 March 2025",
            "reviewUrl": "https://amazon.in/r/R1", "verifiedPurchase": True}
    r = normalize_marketplace_review(item, "B0X", "amazon")
    assert (r.review_id, r.rating, r.date, r.verified) == ("amazon:R1", 2.0, "2025-03-03", True)
    assert r.text == "Gets very hot while gaming" and r.full_text.startswith("Heats up. ")
    assert normalize_marketplace_review({"rating": 5}, "B0X", "amazon") is None  # no text


@pytest.mark.parametrize("value, expected", [
    ("2025-03-03T10:00:00Z", "2025-03-03"), (1735689600, "2025-01-01"), ("12th Jan, 2025", "2025-01-12"),
    ("Mar 2025", "2025-03-01"), ("someday", None), (None, None),
])
def test_parse_date(value, expected):
    assert parse_date(value) == expected


def test_parse_rating():
    assert parse_rating("4.0 out of 5 stars") == 4.0
    assert parse_rating(5) == 5.0
    assert parse_rating("10") is None and parse_rating(None) is None


def test_reddit_posts_and_comments():
    post = {"kind": "t3", "data": {"name": "t3_a", "title": "S24 Ultra green line", "selftext": "Appeared at 8 months",
                                   "permalink": "/r/x/comments/a/", "created_utc": 1735689600}}
    comment = {"kind": "t1", "data": {"name": "t1_b", "body": "Mine heats up too", "permalink": "/r/x/c/b/",
                                      "created_utc": 1735689600}}
    deleted = {"kind": "t1", "data": {"name": "t1_c", "body": "[deleted]"}}
    p = normalize_reddit_item(post, "B0X")
    assert p.url == "https://www.reddit.com/r/x/comments/a/" and "green line" in p.full_text
    assert normalize_reddit_item(comment, "B0X").text == "Mine heats up too"
    assert normalize_reddit_item(deleted, "B0X") is None


def test_youtube_comment():
    thread = {"id": "T", "snippet": {"videoId": "V", "topLevelComment": {"id": "C", "snippet": {
        "textOriginal": "Battery drains fast", "publishedAt": "2026-01-02T00:00:00Z"}}}}
    r = normalize_youtube_comment(thread, "B0X")
    assert (r.url, r.date, r.source) == ("https://www.youtube.com/watch?v=V&lc=C", "2026-01-02", "youtube")


# ------------------------------------------------------------ storage
LONG = "The phone started heating up badly after the last update and the battery drains fast."


def test_save_merges_and_dedupes(tmp_path):
    first = [review("Heats up", 1), review(LONG, 2)]
    assert save_reviews("B0TEST0001", first, tmp_path) == (2, 2)
    again = [review("Heats up", 1), review(LONG, 99), review("Heats up", 3)]
    # Same id kept once; the same long text under a new id is the same review;
    # the same short text from another buyer is a separate review.
    assert save_reviews("B0TEST0001", again, tmp_path) == (1, 3)
    assert sorted(r.review_id for r in load_reviews("B0TEST0001", tmp_path)) == ["amazon:1", "amazon:2", "amazon:3"]
    assert load_reviews("B0NONE", tmp_path) == []


def test_review_path_rejects_unsafe_ids(tmp_path):
    with pytest.raises(ValueError):
        review_path("../etc/passwd", tmp_path)


# ------------------------------------------------------------ detection
@pytest.mark.parametrize("key, text, reported", [
    ("overheating", "Phone heats up a lot while gaming.", True),
    ("overheating", "No heating issue at all.", False),
    ("overheating", "Does it overheat?", False),
    ("overheating", "Not a bad phone but it heats up badly.", True),
    ("overheating", "I don't have any heating issues.", False),
    ("software", "Smooth, lag-free performance.", False),
    ("software", "Never lags. Great!", False),
    ("software", "Phone hangs every day", True),
    ("not_genuine", "I have used phone for 2 months, great.", False),
    ("not_genuine", "They delivered a used phone with scratches.", True),
    ("charging", "Phone is not charging after update.", True),
    ("display_lines", "Green line appeared on the screen after 8 months.", True),
    ("audio", "No sound from the speaker.", True),
    ("battery_drain", "Battery life is poor.", True),
    ("battery_drain", "Battery life is not poor at all.", False),
])
def test_reported_sentence(key, text, reported):
    assert bool(reported_sentence(text, DEFECT[key])) is reported


def corpus():
    reviews = [review(f"Great phone, number {n}.", n) for n in range(100)]
    reviews += [review("A green line appeared on the display.", 200 + n, rating=1) for n in range(3)]
    reviews += [review("It heats up while charging.", 300, when="2024-01-01")]  # 1 mention only
    reviews += [review("The hinge broke.", 400 + n) for n in range(5)]  # laptop defect, product is a phone
    reviews += [review("Screen has green lines everywhere. Green line again!", 500)]  # counts once
    return reviews


def test_detect_reports_patterns_above_threshold():
    report = detect_defects(corpus(), "mobile phones", today=TODAY, min_mentions=3, min_share=0.01)
    assert report["reviews_analyzed"] == 110
    assert [f["defect"] for f in report["findings"]] == ["display_lines"]
    lines = report["findings"][0]
    assert (lines["mentions"], lines["recent_mentions"], lines["low_rating_mentions"]) == (4, 4, 3)
    assert lines["examples"][0]["url"].startswith("https://x.test/")
    assert {"defect": "overheating", "label": "Overheating", "mentions": 1, "share": 0.0091} in report["below_threshold"]
    assert all(f["defect"] != "hinge_keyboard" for f in report["findings"] + report["below_threshold"])


def test_share_threshold_hides_rare_mentions():
    report = detect_defects(corpus(), "mobile phones", today=TODAY, min_mentions=3, min_share=0.05)
    assert report["findings"] == []


def test_no_reviews_gives_a_note():
    report = detect_defects([], "mobile phones", today=TODAY)
    assert report["reviews_analyzed"] == 0 and "fetch_reviews.py" in report["note"]


# ------------------------------------------------------------ sources
def test_fill_keeps_types():
    template = {"productUrls": [{"url": "{url}"}], "maxReviews": "{limit}", "note": "for {url}"}
    assert _fill(template, {"url": "https://a", "limit": 50}) == {
        "productUrls": [{"url": "https://a"}], "maxReviews": 50, "note": "for https://a"}


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}", response=self)


class FakeSession:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def _answer(self, url):
        self.calls.append(url)
        for fragment, response in self.routes.items():
            if fragment in url:
                return response
        raise AssertionError(f"unexpected call {url}")

    def get(self, url, **_kwargs):
        return self._answer(url)

    def post(self, url, **_kwargs):
        return self._answer(url)


def test_reddit_source_reads_posts_and_comments():
    session = FakeSession({
        "access_token": FakeResponse({"access_token": "tok"}),
        "/search": FakeResponse({"data": {"children": [
            {"kind": "t3", "data": {"id": "a", "name": "t3_a", "title": "Heating?", "selftext": "It heats up",
                                    "permalink": "/r/x/a/"}}]}}),
        "/comments/a": FakeResponse([{}, {"data": {"children": [
            {"kind": "t1", "data": {"name": "t1_b", "body": "Same, heats up", "permalink": "/r/x/a/b/"}},
            {"kind": "more", "data": {}}]}}]),
    })
    source = RedditSource("id", "secret", "IndianGaming", session=session)
    reviews = source.fetch("B0X", "Galaxy S24 Ultra")
    assert [r.review_id for r in reviews] == ["reddit:t3_a", "reddit:t1_b"]
    assert any("/r/IndianGaming/search" in call for call in session.calls)


def test_youtube_source_skips_videos_with_comments_disabled():
    thread = {"id": "T", "snippet": {"topLevelComment": {"id": "C", "snippet": {"textOriginal": "Lags a lot"}}}}
    session = FakeSession({
        "/search": FakeResponse({"items": [{"id": {"videoId": "V1"}}, {"id": {"videoId": "V2"}}]}),
        "videoId=V1": FakeResponse({}, 403),
        "commentThreads": FakeResponse({"items": [thread]}),
    })

    def get(url, params=None, **_kwargs):
        key = f"{url}?videoId={params.get('videoId')}" if params and "videoId" in params else url
        return session._answer(key)

    session.get = get
    reviews = YouTubeSource("key", session=session).fetch("B0X", "Galaxy S24 Ultra")
    assert [r.url for r in reviews] == ["https://www.youtube.com/watch?v=V2&lc=C"]


def test_apify_source_handles_nested_review_lists():
    class Actor:
        def call(self, run_input):
            assert run_input == {"url": "https://amazon.in/dp/B0X", "max": 10}
            return {"status": "SUCCEEDED", "defaultDatasetId": "D"}

    class Dataset:
        def iterate_items(self):
            yield {"reviews": [{"reviewId": "1", "text": "Heats up"}, {"reviewId": "2", "text": "Lags"}]}
            yield {"reviewId": "3", "text": "Good"}

    class Client:
        def actor(self, _):
            return Actor()

        def dataset(self, _):
            return Dataset()

    source = ApifyReviewSource("amazon", "me/actor", {"url": "{url}", "max": "{limit}"}, "tok", client=Client())
    assert [r.review_id for r in source.fetch("B0X", "https://amazon.in/dp/B0X", limit=10)] == [
        "amazon:1", "amazon:2", "amazon:3"]


# ------------------------------------------------------------ fetch script
def test_fetch_script_skips_unconfigured_and_saves(tmp_path, monkeypatch, capsys):
    class Collector:
        def fetch(self, product_id, target, **_kwargs):
            return [Review(f"reddit:{target}", product_id, "reddit", "It heats up")]

    monkeypatch.setattr(fetch_reviews, "configured_sources",
                        lambda: {"amazon": None, "flipkart": None, "reddit": Collector(), "youtube": None})
    code = fetch_reviews.main(["--asin", "B0CS5XW6TN", "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "[skip   ] amazon: not configured" in out
    assert "B0CS5XW6TN reddit: 1 reviews (Samsung Galaxy S24 Ultra)" in out
    assert load_reviews("B0CS5XW6TN", tmp_path)[0].text == "It heats up"


def test_fetch_script_without_sources(monkeypatch, capsys):
    monkeypatch.setattr(fetch_reviews, "configured_sources", lambda: dict.fromkeys(("amazon", "flipkart", "reddit", "youtube")))
    assert fetch_reviews.main(["--asin", "B0CS5XW6TN"]) == 1
    assert "No review source is configured" in capsys.readouterr().out


def test_catalog_search_names():
    catalog = json.loads(fetch_reviews.CATALOG.read_text())
    names = {p["asin"]: fetch_reviews.search_name(p) for p in catalog}
    assert names["B0CS5XW6TN"] == "Samsung Galaxy S24 Ultra"


# ------------------------------------------------------------ agent
def test_agent_reports_defects(tmp_path, monkeypatch):
    monkeypatch.setenv("POLICY_MIN_RELEVANCE", "0")
    reviews = [review(f"Nice phone {n}.", n) for n in range(40)]
    reviews += [review("It heats up badly while gaming.", 100 + n) for n in range(4)]
    report = run_eligibility_analysis(
        "B0TEST0001", "Samsung Galaxy S24 5G",
        offers_loader=offers, advisor_factory=advisor_factory(tmp_path), reviews_loader=lambda _id: reviews,
    )
    assert report["defects"]["reviews_analyzed"] == 44
    assert report["defect_warning"] == "Defects reported in reviews — Overheating: 4 of 44 reviews (9.1%)"


def test_agent_survives_review_failure(tmp_path):
    def broken(_id):
        raise OSError("disk unavailable")

    report = run_eligibility_analysis(
        "B0TEST0001", "Samsung Galaxy S24 5G",
        offers_loader=offers, advisor_factory=advisor_factory(tmp_path), reviews_loader=broken,
    )
    assert report["defects"] is None
    assert any("Review analysis unavailable" in error for error in report["errors"])
