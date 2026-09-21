"""Turn free-text offer strings ("10% off on HDFC Credit Card EMI, up to Rs1,500") into Promo objects."""
from __future__ import annotations

import re

from .market_models import Promo

BANKS = [
    ("HDFC Bank", r"\bHDFC\b"), ("ICICI Bank", r"\bICICI\b"), ("Axis Bank", r"\bAxis\b"),
    ("SBI", r"\bSBI\b|State Bank of India"), ("Kotak", r"\bKotak\b"), ("IDFC First", r"\bIDFC\b"),
    ("Federal Bank", r"\bFederal Bank\b"), ("Yes Bank", r"\bYes Bank\b"),
    ("American Express", r"\bAmex\b|American Express"), ("Citi", r"\bCiti(?:bank)?\b"),
    ("RBL Bank", r"\bRBL\b"), ("IndusInd", r"\bIndusInd\b"), ("OneCard", r"\bOneCard\b"),
    ("Bank of Baroda", r"\bBOB\b|Bank of Baroda"), ("Bajaj Finserv", r"\bBajaj\b"),
    ("Paytm", r"\bPaytm\b"), ("AU Bank", r"\bAU (?:Small Finance|Bank)\b"),
]

_RS = r"(?:₹|Rs\.?|INR)\s?"
_NUM = r"([\d,]+(?:\.\d+)?)"
_AMOUNT_OFF = re.compile(rf"{_RS}{_NUM}\s*(?:instant\s+)?(?:off|discount|cashback|back)\b", re.I)
_AMOUNT_UPTO = re.compile(rf"(?:up\s*to|upto|max(?:imum)?)\s*{_RS}{_NUM}", re.I)
_AMOUNT_ANY = re.compile(rf"{_RS}{_NUM}", re.I)
_PCT = re.compile(r"(\d{1,2}(?:\.\d+)?)\s?%")
_FESTIVE = re.compile(r"festive|big billion|great indian|\bbbd\b|diwali|\bsale\b|deal of the day|"
                      r"limited[- ]time|special price|republic day|independence day", re.I)
_HINTS = {"EXCHANGE": "EXCHANGE", "PBO": "BANK", "BANK": "BANK", "COUPON": "COUPON",
          "CASHBACK": "CASHBACK", "EMI": "EMI"}


def _num(text: str) -> float:
    return float(text.replace(",", ""))


def classify_promo(text: str, type_hint: str | None = None, source: str | None = None) -> Promo:
    t = " ".join(str(text).split())
    low = t.lower()
    bank = next((name for name, pat in BANKS if re.search(pat, t, re.I)), None)
    is_emi = "emi" in low
    if "credit" in low and "debit" in low:
        card = "Credit/Debit Card"
    elif "credit" in low:
        card = "Credit Card"
    elif "debit" in low:
        card = "Debit Card"
    elif "upi" in low:
        card = "UPI"
    else:
        card = None

    m = _AMOUNT_OFF.search(t) or _AMOUNT_UPTO.search(t) or _AMOUNT_ANY.search(t)
    amount = _num(m.group(1)) if m else None
    pm = _PCT.search(t)
    percent = float(pm.group(1)) if pm else None

    if "exchange" in low:
        ptype = "EXCHANGE"
    elif "cashback" in low:
        ptype = "CASHBACK"
    elif "coupon" in low:
        ptype = "COUPON"
    elif bank or "bank" in low or card:
        ptype = "BANK"
    elif is_emi:
        ptype = "EMI"
    elif _FESTIVE.search(low):
        ptype = "FESTIVE"
    else:
        ptype = _HINTS.get((type_hint or "").upper(), "OTHER")
    return Promo(promo_type=ptype, description=t[:300], bank=bank, card_type=card, amount=amount,
                 percent=percent, is_emi=is_emi, source=source)


def promo_value(price: float | None, promo: Promo) -> float:
    """Rupee value of one promo at a given price (percent offers are capped by `amount`)."""
    by_pct = price * promo.percent / 100 if price and promo.percent else None
    if by_pct is not None and promo.amount is not None:
        return min(by_pct, promo.amount)
    return by_pct if by_pct is not None else (promo.amount or 0.0)


def best_instant_bank_discount(price: float | None, promos: list[Promo]) -> float:
    """Best single bank-card discount (offers are assumed NOT stackable unless the source says so)."""
    if not price:
        return 0.0
    return max((promo_value(price, p) for p in promos if p.promo_type == "BANK"), default=0.0)
