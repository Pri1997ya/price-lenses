"""Decide which canonical product an Offer belongs to.

Priority:
  1. ASIN   -> "asin:B0XXXXXXXX"   (exact; Amazon's identifier, also present in Amazon URLs)
  2. GTIN   -> "gtin:0194253..."   (exact; UPC/EAN, shared by many retailers)
  3. Title  -> fuzzy match against products already known in this run; otherwise a new
               "title:<hash>" product is created.

NOTE: ASIN is Amazon's id. Flipkart, Walmart, eBay etc. do not use it, so cross-site
matching only works via ASIN when the site links to / reports an Amazon ASIN.
GTIN and fuzzy title matching cover the rest.
"""
from __future__ import annotations

import hashlib
from difflib import SequenceMatcher

from .market_models import Offer
from .market_normalize import normalize_title

TITLE_MATCH_THRESHOLD = 0.82

ACCESSORY_TERMS = {
    "back cover", "case", "flip cover", "screen guard", "screen protector",
    "tempered glass", "camera lens", "lens guard", "protector", "skin",
    "charger", "charging cable", "usb cable", "adapter", "stand", "holder",
    "replacement", "spare", "pouch", "sleeve",
}

_PRODUCT_SPEC_TERMS = {
    "ram", "storage", "battery", "mah", "camera", "display", "processor",
    "smartphone", "laptop", "television", "oled", "amoled", "ssd",
}


def _token_set(s: str) -> set[str]:
    return set(s.split())


def title_similarity(a: str, b: str) -> float:
    na, nb = normalize_title(a), normalize_title(b)
    if not na or not nb:
        return 0.0
    seq = SequenceMatcher(None, na, nb).ratio()
    ta, tb = _token_set(na), _token_set(nb)
    jac = len(ta & tb) / len(ta | tb)
    # Model numbers/storage sizes must agree: penalise if numeric tokens differ.
    nums_a = {t for t in ta if any(c.isdigit() for c in t)}
    nums_b = {t for t in tb if any(c.isdigit() for c in t)}
    if nums_a and nums_b and not (nums_a & nums_b):
        return 0.0
    return max(seq * 0.6 + jac * 0.4, 0.0)


def is_accessory_title(title: str | None) -> bool:
    normalized = normalize_title(title or "")
    return any(term in normalized for term in ACCESSORY_TERMS)


def product_relevance(query: str, title: str | None) -> float:
    """Score whether a listing is the requested product rather than an accessory.

    Model tokens containing digits (``s2``, ``s24``, ``16``) are mandatory when
    the query supplies them.  Accessory listings are rejected unless the query
    itself explicitly asks for an accessory.
    """
    normalized_query = normalize_title(query)
    normalized_title = normalize_title(title or "")
    if not normalized_query or not normalized_title:
        return 0.0
    query_accessory = is_accessory_title(normalized_query)
    if is_accessory_title(normalized_title) and not query_accessory:
        return 0.0

    query_tokens = set(normalized_query.split())
    title_tokens = set(normalized_title.split())
    model_tokens = {
        token for token in query_tokens
        if any(character.isdigit() for character in token)
        and not token.endswith(("gb", "tb", "mah"))
    }
    if model_tokens and not model_tokens.issubset(title_tokens):
        return 0.0

    overlap = len(query_tokens & title_tokens) / max(1, len(query_tokens))
    score = 0.65 * title_similarity(normalized_query, normalized_title) + 0.35 * overlap
    if model_tokens:
        score += 0.20
    if _PRODUCT_SPEC_TERMS & title_tokens:
        score += 0.10
    return min(1.0, score)


class ProductResolver:
    """Stateful resolver; feed it known products first (from DB), then resolve offers."""

    def __init__(self, known: dict[str, str] | None = None):
        # product_id -> normalized title
        self.known: dict[str, str] = dict(known or {})
        self.gtin_index: dict[str, str] = {}
        self.asin_index: dict[str, str] = {}

    def resolve(self, offer: Offer) -> str:
        if offer.asin:
            pid = f"asin:{offer.asin.upper()}"
            self._register(pid, offer)
            return pid
        if offer.gtin:
            pid = self.gtin_index.get(offer.gtin) or f"gtin:{offer.gtin}"
            self.gtin_index[offer.gtin] = pid
            self._register(pid, offer)
            return pid
        best_id, best = None, 0.0
        for pid, known_title in self.known.items():
            score = title_similarity(offer.title, known_title)
            if score > best:
                best_id, best = pid, score
        if best_id and best >= TITLE_MATCH_THRESHOLD:
            return best_id
        pid = "title:" + hashlib.sha1(normalize_title(offer.title).encode()).hexdigest()[:12]
        self._register(pid, offer)
        return pid

    def _register(self, pid: str, offer: Offer) -> None:
        self.known.setdefault(pid, offer.title)
