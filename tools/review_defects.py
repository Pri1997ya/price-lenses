"""Find hardware and quality defects that many reviewers report (Agent 3).

Detection is a fixed pattern scan, like the policy restriction flags: every
reported defect can be traced to the exact sentences that triggered it, and
the same reviews always give the same result.

A review counts toward a defect only when a sentence in it reports the problem:

* negated mentions are skipped ("no heating issue", "never lags");
* questions are skipped ("does it overheat?"), which matter on Reddit;
* each review counts once per defect, however often it repeats the words.

A defect is reported only when enough distinct reviews mention it
(``REVIEW_MIN_MENTIONS``, default 3) and they are a meaningful share of all
reviews collected (``REVIEW_MIN_SHARE``, default 1%), so one angry review or a
handful out of thousands is not presented as a pattern.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date

from .review_corpus import SOURCE_LABELS, Review, age_days, dedupe_key

PHONES = "mobile phones"
SCREENS = {PHONES, "tablets", "laptops", "televisions", "smartwatches"}
BATTERY = {PHONES, "tablets", "laptops", "smartwatches", "headphones and earbuds", "speakers"}
RADIOS = {PHONES, "tablets", "laptops"}
AUDIO = {"headphones and earbuds", "speakers"}
ALL = None  # applies to every product type

RECENT_DAYS = 365
MAX_EXAMPLES = 5


@dataclass(frozen=True)
class Defect:
    key: str
    label: str
    patterns: tuple[str, ...]
    categories: frozenset[str] | None = ALL

    def applies_to(self, category: str) -> bool:
        # Unknown product type ("electronics"): check everything.
        return self.categories is None or category == "electronics" or category in self.categories


def _defect(key, label, patterns, categories=ALL) -> Defect:
    return Defect(key, label, tuple(patterns), frozenset(categories) if categories else None)


DEFECTS: tuple[Defect, ...] = (
    _defect("display_lines", "Lines on the display (green/pink line)", [
        r"\b(?:green|pink|purple|white|vertical|horizontal)\s+lines?\b",
        r"\blines?\s+(?:on|in|across)\s+(?:the\s+)?(?:screen|display)\b",
    ], SCREENS),
    _defect("display_other", "Display fault (dead pixels, flicker, burn-in)", [
        r"\bdead\s+pixels?\b", r"\b(?:screen|display)\s+flicker", r"\bflickering\b",
        r"\bburn[\s-]?in\b", r"\b(?:screen|display)\s+(?:went\s+)?(?:black|blank)\b",
    ], SCREENS),
    _defect("ghost_touch", "Touch problems (ghost touch, unresponsive)", [
        r"\bghost\s+touch", r"\btouch\s+(?:screen\s+)?(?:is\s+)?(?:not\s+working|unresponsive|issues?|problems?)\b",
    ], {PHONES, "tablets", "smartwatches"}),
    _defect("overheating", "Overheating", [
        r"\bover\s?heat", r"\bheats?\s+up\b", r"\bheating\s+(?:issues?|problems?)\b",
        r"\b(?:gets?|getting|becomes?|very|too|so|extremely)\s+hot\b",
    ], ALL),
    _defect("battery_drain", "Poor battery life or fast drain", [
        r"\bbattery\s+drain", r"\bdrains?\s+(?:very\s+|too\s+|so\s+)?(?:fast|quickly)\b",
        r"\bbattery\s+(?:life|backup)\s+(?:is\s+)?(?:very\s+)?(?:poor|bad|terrible|worst|pathetic)\b",
        r"\b(?:poor|bad|terrible|worst|pathetic)\s+battery\b",
    ], BATTERY),
    _defect("battery_swelling", "Battery swelling", [
        r"\bbattery\s+(?:swell|bulg|bloat)", r"\bswollen\s+battery\b",
    ], BATTERY),
    _defect("charging", "Charging problems", [
        r"\bnot\s+charging\b", r"\bstopped\s+charging\b", r"\bcharging\s+(?:port\s+)?(?:issues?|problems?|fault)\b",
        r"\bwon'?t\s+charge\b", r"\bslow\s+charging\b",
    ], BATTERY),
    _defect("network", "Network, SIM or Wi-Fi problems", [
        r"\bnetwork\s+(?:issues?|problems?|drops?)\b", r"\bcall\s+drops?\b", r"\bno\s+signal\b",
        r"\bsim\s+(?:card\s+)?not\s+(?:detected|working)\b", r"\b5g\s+not\s+working\b",
        r"\bwi-?fi\s+(?:keeps\s+)?(?:disconnect|drops?|issues?|problems?)",
    ], RADIOS),
    _defect("camera", "Camera problems", [
        r"\bcamera\s+(?:is\s+)?(?:not\s+(?:working|focusing)|blurry|blurs?|shak|issues?|problems?)",
        r"\b(?:focus|focusing)\s+(?:issues?|problems?)\b",
    ], {PHONES, "tablets", "laptops"}),
    _defect("audio", "Speaker, microphone or sound faults", [
        r"\bspeaker\s+(?:is\s+)?(?:crackl|not\s+working|issues?|problems?|stopped)",
        r"\b(?:mic|microphone)\s+(?:is\s+)?(?:not\s+working|issues?|problems?)\b", r"\bno\s+sound\b",
        r"\bsound\s+(?:issues?|problems?|crackl)",
    ], ALL),
    _defect("one_side_audio", "One earbud or side stops working", [
        r"\b(?:left|right)\s+(?:ear\s*bud|bud|side|earphone)\s+(?:is\s+)?(?:not\s+working|stopped|dead)",
        r"\bone\s+(?:ear\s*bud|bud|side)\s+(?:is\s+)?(?:not\s+working|stopped|dead)",
    ], AUDIO),
    _defect("bluetooth", "Bluetooth disconnects", [
        r"\bbluetooth\s+(?:keeps\s+)?(?:disconnect|drops?|issues?|problems?)",
        r"\bkeeps\s+disconnecting\b",
    ], AUDIO | {"smartwatches"}),
    _defect("software", "Lag, hangs or random restarts", [
        r"\bhangs?\b", r"\bhanging\b", r"\blag(?:s|gy|ging)?\b", r"\bboot\s?loop",
        r"\brestarts?\s+(?:by\s+itself|automatically|randomly|on\s+its\s+own)",
        r"\b(?:apps?|phone)\s+(?:keeps\s+)?crash",
    ], {PHONES, "tablets", "laptops", "televisions", "smartwatches"}),
    _defect("hinge_keyboard", "Hinge or keyboard faults", [
        r"\bhinge\s+(?:is\s+)?(?:broke|broken|crack|loose)", r"\bkeys?\s+(?:stopped|not)\s+working\b",
        r"\bkeyboard\s+(?:stopped|not)\s+working\b",
    ], {"laptops"}),
    _defect("dead_unit", "Dead on arrival or stopped working", [
        r"\bdead\s+on\s+arrival\b", r"\bdoa\b", r"\b(?:not|won'?t)\s+turn(?:ing)?\s+on\b",
        r"\bstopped\s+working\b", r"\bdoes\s*n[o']t\s+(?:turn|switch)\s+on\b",
    ], ALL),
    _defect("not_genuine", "Used, refurbished or fake unit received", [
        r"\b(?:fake|duplicate|counterfeit|refurbished)\s+(?:product|phone|unit|item|piece)\b",
        r"\b(?:received|got|delivered|sent)\s+(?:me\s+)?(?:an?\s+)?(?:used|old|refurbished)\s+"
        r"(?:product|phone|unit|item|piece)\b",
        r"\bseal\s+(?:was\s+)?(?:broken|open)", r"\bopen(?:ed)?\s+box\b", r"\bnot\s+(?:genuine|original)\b",
    ], ALL),
)

_NEGATION = re.compile(
    r"\b(?:no|not|never|without|zero|nil|none|hardly|any|free\s+(?:of|from)"
    r"|don'?t|didn'?t|doesn'?t|isn'?t|wasn'?t|haven'?t|hasn'?t|won'?t|cannot|can'?t)\b"
    r"(?:\s+\w+){0,3}\s*$"
)
_SENTENCES = re.compile(r"(?<=[.!?])\s+|\n+")


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCES.split(text or "") if part.strip()]


def reported_sentence(text: str, defect: Defect) -> str | None:
    """The first sentence in ``text`` that reports ``defect``, else None."""
    for sentence in _sentences(text):
        if sentence.endswith("?"):
            continue  # a question, not a report
        lowered = sentence.lower()
        for pattern in defect.patterns:
            for match in re.finditer(pattern, lowered):
                if re.match(r"[\s-]*free\b", lowered[match.end():]):
                    continue  # "lag-free", "heating free"
                before = lowered[max(0, match.start() - 40):match.start()]
                if match.group().startswith(("not ", "no ")) or not _NEGATION.search(before):
                    return sentence
    return None


def _snippet(sentence: str, limit: int = 220) -> str:
    return sentence if len(sentence) <= limit else sentence[:limit].rsplit(" ", 1)[0] + "…"


def detect_defects(
    reviews: list[Review],
    category: str = "electronics",
    *,
    today: date | None = None,
    min_mentions: int | None = None,
    min_share: float | None = None,
) -> dict:
    """Count, per defect, the distinct reviews that report it. Returns a
    JSON-ready report with shares, recency and links to example reviews."""
    min_mentions = min_mentions if min_mentions is not None else int(os.getenv("REVIEW_MIN_MENTIONS", "3"))
    min_share = min_share if min_share is not None else float(os.getenv("REVIEW_MIN_SHARE", "0.01"))
    today = today or date.today()

    unique: dict[tuple[str, str], Review] = {}
    for review in reviews:
        unique.setdefault(dedupe_key(review), review)
    reviews = list(unique.values())
    total = len(reviews)
    by_source: dict[str, int] = {}
    for review in reviews:
        by_source[review.source] = by_source.get(review.source, 0) + 1
    dates = sorted(review.date for review in reviews if review.date)
    report = {
        "reviews_analyzed": total,
        "by_source": {SOURCE_LABELS.get(k, k): v for k, v in sorted(by_source.items())},
        "oldest_review": dates[0] if dates else None,
        "newest_review": dates[-1] if dates else None,
        "category": category,
        "findings": [],
        "below_threshold": [],
    }
    if not total:
        report["note"] = "No reviews collected for this product. Run scripts/reviews/fetch_reviews.py."
        return report

    for defect in DEFECTS:
        if not defect.applies_to(category):
            continue
        hits = []
        for review in reviews:
            sentence = reported_sentence(review.full_text, defect)
            if sentence:
                hits.append((review, sentence))
        if not hits:
            continue
        ages = [age_days(review, today) for review, _ in hits]
        recent = sum(1 for age in ages if age is not None and age <= RECENT_DAYS)
        rated = [review for review, _ in hits if review.rating is not None]
        # Newest first, preferring low-rated marketplace reviews as examples.
        ordered = sorted(
            hits,
            key=lambda item: (item[0].date or "", -(item[0].rating or 3)),
            reverse=True,
        )
        finding = {
            "defect": defect.key,
            "label": defect.label,
            "mentions": len(hits),
            "share": round(len(hits) / total, 4),
            "recent_mentions": recent,
            "low_rating_mentions": sum(1 for review in rated if review.rating <= 2),
            "sources": sorted({SOURCE_LABELS.get(r.source, r.source) for r, _ in hits}),
            "examples": [
                {
                    "source": SOURCE_LABELS.get(review.source, review.source),
                    "url": review.url,
                    "date": review.date,
                    "rating": review.rating,
                    "snippet": _snippet(sentence),
                }
                for review, sentence in ordered[:MAX_EXAMPLES]
            ],
        }
        if len(hits) >= min_mentions and len(hits) / total >= min_share:
            report["findings"].append(finding)
        else:
            report["below_threshold"].append(
                {key: finding[key] for key in ("defect", "label", "mentions", "share")}
            )
    report["findings"].sort(key=lambda item: item["mentions"], reverse=True)
    return report


def defect_summary(finding: dict, total: int) -> str:
    text = f"{finding['label']}: {finding['mentions']} of {total} reviews ({finding['share']:.1%})"
    if finding["mentions"] and finding["recent_mentions"] == 0:
        text += ", none in the last year"
    return text
