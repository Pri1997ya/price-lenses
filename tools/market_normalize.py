"""Pure helpers: ASIN extraction, price parsing, title normalisation."""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import unquote, urlparse

_ASIN_URL_PATTERNS = [
    re.compile(r"/(?:dp|gp/product|gp/aw/d|product-reviews|exec/obidos/ASIN)/([A-Z0-9]{10})(?:[/?#]|$)", re.I),
    re.compile(r"[?&]asin=([A-Z0-9]{10})(?:&|$)", re.I),
]
_ASIN_RE = re.compile(r"^[A-Z0-9]{10}$")

_CURRENCY_SYMBOLS = {"₹": "INR", "$": "USD", "£": "GBP", "€": "EUR", "¥": "JPY", "Rs.": "INR", "Rs": "INR"}


def is_asin(value: str | None) -> bool:
    return bool(value) and bool(_ASIN_RE.match(value.strip().upper()))


def extract_asin(*candidates: Any) -> str | None:
    """Return an ASIN from a bare id or from any amazon-style URL among candidates."""
    for c in candidates:
        if not isinstance(c, str) or not c:
            continue
        s = c.strip()
        if is_asin(s) and not s.isdigit():
            return s.upper()
        for pat in _ASIN_URL_PATTERNS:
            m = pat.search(s)
            if m:
                return m.group(1).upper()
    return None


def parse_price(value: Any) -> tuple[float | None, str | None]:
    """'₹1,34,900.00' -> (134900.0, 'INR'); 1299 -> (1299.0, None); {'value':9,'currency':'USD'} ok."""
    if value is None or value == "":
        return None, None
    if isinstance(value, dict):
        amount, _ = parse_price(value.get("value") or value.get("amount") or value.get("price"))
        return amount, value.get("currency")
    if isinstance(value, (int, float)):
        return float(value), None
    s = str(value).strip()
    currency = None
    for sym, code in _CURRENCY_SYMBOLS.items():
        if sym in s:
            currency = code
            break
    m = re.search(r"\d[\d,]*(?:\.\d+)?", s)
    if not m:
        return None, currency
    return float(m.group(0).replace(",", "")), currency


def parse_int(value: Any) -> int | None:
    """'1,234' -> 1234, '2.5K' -> 2500, '(1.2K)' -> 1200."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value).replace(",", "").strip("() ").upper()
    m = re.search(r"(\d+(?:\.\d+)?)\s*([KM]?)", s)
    if not m:
        return None
    n = float(m.group(1)) * {"K": 1_000, "M": 1_000_000}.get(m.group(2), 1)
    return int(n)


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(str(value).split()[0].replace(",", ""))
    except ValueError:
        return None


def marketplace_from_url(url: str | None) -> str | None:
    if not url:
        return None
    m = re.match(r"https?://(?:www\.)?([^/]+)", url)
    return m.group(1).lower() if m else None


def is_url(value: str | None) -> bool:
    if not value:
        return False
    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def search_query_from_url(value: str) -> str:
    """Best-effort product query from a retailer URL slug.

    Amazon URLs are resolved by ASIN through the Amazon Product API. This fallback is
    mainly for Flipkart/Croma/Reliance/Vijay Sales URLs whose title appears in the path.
    """
    if not is_url(value):
        return value.strip()
    parsed = urlparse(value.strip())
    ignored = {"p", "product", "products", "dp", "gp", "buy", "online"}
    candidates = []
    for segment in unquote(parsed.path).split("/"):
        clean = re.sub(r"[-_]+", " ", segment).strip()
        if (not clean or clean.lower() in ignored or
                re.fullmatch(r"(?:itm|pid|sku)?[a-z0-9]{12,}", clean, re.I)):
            continue
        candidates.append(clean)
    return max(candidates, key=len, default=value.strip())


_STOPWORDS = {"the", "and", "with", "for", "new", "latest", "smartphone", "phone", "mobile", "5g", "unlocked"}


def normalize_title(title: str) -> str:
    t = title.lower()
    t = re.sub(r"(\d+)\s*(gb|tb)\b", r"\1\2", t)   # "256 GB" == "256GB"
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return " ".join(w for w in t.split() if w not in _STOPWORDS)


def canonical_id_for(asin: str | None = None, ean: str | None = None) -> str:
    """LLD format: EAN_<13 digits> if a valid EAN is known, else ASIN_<asin>."""
    if ean and re.fullmatch(r"\d{13}", ean.strip()):
        return f"EAN_{ean.strip()}"
    if asin and is_asin(asin):
        return f"ASIN_{asin.strip().upper()}"
    raise ValueError("Need a 13-digit EAN or a valid 10-character ASIN")
