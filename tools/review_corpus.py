"""User reviews and discussions of a product, stored per product (Agent 3).

Reviews come from Amazon and Flipkart (Apify review actors), Reddit (official
API) and YouTube comments (Data API). Each source has its own shape, so each
has a small normaliser that returns ``Review`` objects in one format.

Storage is one JSON-lines file per product, ``data/reviews/<product_id>.jsonl``,
keyed by the catalog ASIN. Saving merges with what is already there, so the
fetch script can be re-run to add newer reviews. Review text is third-party
content: the folder is git-ignored and is not committed.

Author names are never stored; only what is needed to count a defect and link
back to the original.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .policy_corpus import REPOSITORY_ROOT

DEFAULT_REVIEW_DIR = REPOSITORY_ROOT / "data" / "reviews"
SOURCES = ("amazon", "flipkart", "reddit", "youtube")
SOURCE_LABELS = {"amazon": "Amazon", "flipkart": "Flipkart", "reddit": "Reddit", "youtube": "YouTube"}

_TEXT_KEYS = ("text", "reviewText", "reviewDescription", "review", "content", "body", "comment", "description")
_TITLE_KEYS = ("title", "reviewTitle", "heading", "summary")
_RATING_KEYS = ("rating", "ratingScore", "stars", "reviewRating", "score")
_DATE_KEYS = ("date", "reviewDate", "datePublished", "publishedAt", "createdAt", "reviewedAt", "time")
_URL_KEYS = ("url", "reviewUrl", "link", "permalink")
_ID_KEYS = ("id", "reviewId", "review_id")


@dataclass(frozen=True)
class Review:
    review_id: str
    product_id: str
    source: str
    text: str
    url: str | None = None
    date: str | None = None          # YYYY-MM-DD
    rating: float | None = None      # 1-5 for marketplaces; None for Reddit/YouTube
    title: str = ""
    verified: bool | None = None

    @property
    def full_text(self) -> str:
        return f"{self.title}. {self.text}" if self.title else self.text


def _first(item: dict, keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return value
    return None


def parse_date(value: Any) -> str | None:
    """Best-effort date -> YYYY-MM-DD. Handles ISO strings, epoch seconds and
    Amazon's "Reviewed in India on 3 March 2025"."""
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc).date().isoformat()
    text = str(value).strip()
    iso = re.match(r"(\d{4}-\d{2}-\d{2})", text)
    if iso:
        return iso.group(1)
    text = re.sub(r"^.*\bon\s+", "", text)  # "Reviewed in India on 3 March 2025"
    text = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", text).replace(",", " ")
    text = re.sub(r"\s+", " ", text).strip()
    for fmt in ("%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y", "%b %Y", "%B %Y", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def parse_rating(value: Any) -> float | None:
    if value in (None, ""):
        return None
    match = re.search(r"\d+(?:\.\d+)?", str(value))  # "4.0 out of 5 stars" -> 4.0
    if not match:
        return None
    rating = float(match.group())
    return rating if 0 < rating <= 5 else None


def _clean(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _review_id(source: str, raw_id: Any, text: str) -> str:
    if raw_id not in (None, ""):
        return f"{source}:{raw_id}"
    return f"{source}:sha1:{hashlib.sha1(text.lower().encode('utf-8')).hexdigest()[:16]}"


# ------------------------------------------------------------- normalisers ----
def normalize_marketplace_review(item: dict, product_id: str, source: str) -> Review | None:
    """One review from an Amazon or Flipkart review actor. Field names differ
    between actors, so the common spellings are all accepted."""
    text = _clean(_first(item, _TEXT_KEYS))
    if not text:
        return None
    verified = item.get("verified", item.get("isVerified", item.get("verifiedPurchase")))
    return Review(
        review_id=_review_id(source, _first(item, _ID_KEYS), text),
        product_id=product_id,
        source=source,
        text=text,
        url=_first(item, _URL_KEYS),
        date=parse_date(_first(item, _DATE_KEYS)),
        rating=parse_rating(_first(item, _RATING_KEYS)),
        title=_clean(_first(item, _TITLE_KEYS)),
        verified=bool(verified) if verified is not None else None,
    )


def normalize_reddit_item(thing: dict, product_id: str) -> Review | None:
    """A post (kind t3) or comment (kind t1) from the Reddit API listing JSON."""
    data = thing.get("data", thing)
    is_post = thing.get("kind") == "t3" or "selftext" in data
    text = _clean(data.get("selftext") if is_post else data.get("body"))
    title = _clean(data.get("title")) if is_post else ""
    if text in ("[deleted]", "[removed]"):
        text = ""
    if not text and not title:
        return None
    permalink = data.get("permalink")
    return Review(
        review_id=_review_id("reddit", data.get("name") or data.get("id"), title + text),
        product_id=product_id,
        source="reddit",
        text=text or title,
        url=f"https://www.reddit.com{permalink}" if permalink else data.get("url"),
        date=parse_date(data.get("created_utc")),
        title=title if text else "",
    )


def normalize_youtube_comment(thread: dict, product_id: str) -> Review | None:
    """A commentThreads.list item from the YouTube Data API (top-level comment)."""
    snippet = thread.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
    text = _clean(snippet.get("textOriginal") or snippet.get("textDisplay"))
    if not text:
        return None
    comment_id = thread.get("snippet", {}).get("topLevelComment", {}).get("id") or thread.get("id")
    video_id = snippet.get("videoId") or thread.get("snippet", {}).get("videoId")
    url = f"https://www.youtube.com/watch?v={video_id}&lc={comment_id}" if video_id and comment_id else None
    return Review(
        review_id=_review_id("youtube", comment_id, text),
        product_id=product_id,
        source="youtube",
        text=text,
        url=url,
        date=parse_date(snippet.get("publishedAt")),
    )


# ----------------------------------------------------------------- storage ----
def review_path(product_id: str, review_dir: Path = DEFAULT_REVIEW_DIR) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", product_id or ""):
        raise ValueError(f"Invalid product id {product_id!r}")
    return Path(review_dir) / f"{product_id}.jsonl"


def load_reviews(product_id: str, review_dir: Path = DEFAULT_REVIEW_DIR) -> list[Review]:
    path = review_path(product_id, review_dir)
    if not path.exists():
        return []
    reviews = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            reviews.append(Review(**json.loads(line)))
    return reviews


def save_reviews(product_id: str, reviews: list[Review], review_dir: Path = DEFAULT_REVIEW_DIR) -> tuple[int, int]:
    """Merge ``reviews`` into the product's file. Returns (added, total).

    The same review fetched twice (same id, or the same long text from the
    same source) is kept once; a later fetch replaces the stored copy.
    """
    existing = {review.review_id: review for review in load_reviews(product_id, review_dir)}
    text_keys = {dedupe_key(r): r.review_id for r in existing.values()}
    added = 0
    for review in reviews:
        duplicate = text_keys.get(dedupe_key(review))
        if duplicate and duplicate != review.review_id:
            continue
        if review.review_id not in existing:
            added += 1
        existing[review.review_id] = review
        text_keys[dedupe_key(review)] = review.review_id
    path = review_path(product_id, review_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(existing.values(), key=lambda r: (r.date or "", r.review_id), reverse=True)
    path.write_text(
        "".join(json.dumps(asdict(review), ensure_ascii=False) + "\n" for review in ordered),
        encoding="utf-8",
    )
    return added, len(ordered)


# Identical text this long is the same review fetched twice; shorter text
# ("Heats up", "Good product") is often written by different buyers.
SAME_TEXT_MIN_CHARS = 60


def dedupe_key(review: Review) -> tuple[str, str]:
    if len(review.full_text) >= SAME_TEXT_MIN_CHARS:
        return (review.source, review.full_text.lower())
    return ("id", review.review_id)


def age_days(review: Review, today: date | None = None) -> int | None:
    if not review.date:
        return None
    try:
        return ((today or date.today()) - date.fromisoformat(review.date)).days
    except ValueError:
        return None
