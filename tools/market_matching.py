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
