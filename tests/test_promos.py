from tools.market_models import Promo
from tools.market_promotions import best_instant_bank_discount, classify_promo


def test_bank_emi_percent_with_cap():
    p = classify_promo("10% off on HDFC Bank Credit Card EMI, up to ₹1,500")
    assert (p.promo_type, p.bank, p.card_type, p.is_emi) == ("BANK", "HDFC Bank", "Credit Card", True)
    assert p.percent == 10 and p.amount == 1500


def test_flipkart_style_offers_use_type_hint_and_text():
    b = classify_promo("Bank offers ₹3,495 off", "PBO")
    assert b.promo_type == "BANK" and b.amount == 3495
    e = classify_promo("Exchange offer Change pincode to exchange item Up to ₹40,900", "EXCHANGE")
    assert e.promo_type == "EXCHANGE" and e.amount == 40900


def test_amount_picks_the_discount_not_the_minimum_order():
    p = classify_promo("Purchase above ₹50,000 to get ₹5,000 off on ICICI Debit Card")
    assert p.amount == 5000 and p.bank == "ICICI Bank" and p.card_type == "Debit Card"


def test_festive_and_plain():
    assert classify_promo("Big Billion Days special price").promo_type == "FESTIVE"
    assert classify_promo("Free delivery").promo_type == "OTHER"
    assert classify_promo("No Cost EMI available").promo_type == "EMI"


def test_best_bank_discount_caps_percent_and_ignores_exchange():
    promos = [Promo("BANK", "10% up to 1500", percent=10, amount=1500),
              Promo("BANK", "flat 3495", amount=3495),
              Promo("EXCHANGE", "up to 40900", amount=40900)]
    assert best_instant_bank_discount(69900, promos) == 3495          # not the exchange value
    assert best_instant_bank_discount(69900, promos[:1]) == 1500      # 10% = 6990, capped at 1500
    assert best_instant_bank_discount(None, promos) == 0.0
