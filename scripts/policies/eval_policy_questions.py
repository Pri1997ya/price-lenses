"""Check policy search quality against a fixed question set, and suggest a cut-off.

    python scripts/policies/eval_policy_questions.py          # data/policies/eval_questions.json
    python scripts/policies/eval_policy_questions.py --questions my_questions.json

Runs retrieval only (no LLM), so it is fast and repeatable. For every question
it shows the top passages and whether the expected retailer, and optionally an
expected phrase, came back. Questions whose ``expect`` is ``"no_match"`` must
return nothing; they show whether the relevance cut-off is too loose.

The question file is a JSON list:

    [{"question": "Can I return a phone bought on Flipkart?",
      "retailer": "flipkart", "expect_text": "replacement"},
     {"question": "Do you deliver to the moon?", "expect": "no_match"}]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv  # noqa: E402

from tools.policy_rag import PolicyIndex, embedder_from_env  # noqa: E402

DEFAULT_QUESTIONS_FILE = Path(__file__).resolve().parents[2] / "data" / "policies" / "eval_questions.json"
DEFAULT_QUESTIONS = [
    {"question": "Can I return a phone bought on Flipkart?", "retailer": "flipkart"},
    {"question": "What is the replacement window for mobile phones on Amazon?",
     "retailer": "amazon", "expect_text": "days"},
    {"question": "How long does a refund take on Amazon?", "retailer": "amazon", "expect_text": "refund"},
    {"question": "Can I return an opened phone at Croma?", "retailer": "croma"},
    {"question": "How do I cancel an order on Reliance Digital?", "retailer": "reliance_digital",
     "expect_text": "cancel"},
    {"question": "What is Vijay Sales' return policy for electronics?", "retailer": "vijay_sales"},
    {"question": "Is there a restocking fee for returned laptops?", "retailer": None,
     "expect_text": "fee"},
    # One or more per ingested source, so a missing or broken page shows up as a MISS.
    {"question": "Which items are not eligible for replacement on Amazon?", "retailer": "amazon",
     "expect_text": "replacement"},
    {"question": "How do I schedule a pickup for an Amazon return?", "retailer": "amazon",
     "expect_text": "pickup"},
    {"question": "Can I self-ship a return to Amazon if pickup is unavailable?", "retailer": "amazon",
     "expect_text": "ship"},
    {"question": "Are Apple and Samsung phones on Flipkart replaced by the service center?",
     "retailer": "flipkart", "expect_text": "service"},
    {"question": "How do I contact Flipkart customer support?", "retailer": "flipkart"},
    {"question": "Can I cancel a Croma order after it is shipped?", "retailer": "croma",
     "expect_text": "cancel"},
    {"question": "Does Croma refund to the original payment method?", "retailer": "croma",
     "expect_text": "refund"},
    {"question": "Which payment methods does Reliance Digital accept?", "retailer": "reliance_digital",
     "expect_text": "payment"},
    {"question": "What is Reliance Digital's return window for defective products?",
     "retailer": "reliance_digital", "expect_text": "days"},
    {"question": "How does Vijay Sales handle refunds for cancelled orders?", "retailer": "vijay_sales",
     "expect_text": "refund"},
    {"question": "Must an e-commerce seller appoint a grievance officer?", "retailer": "regulation",
     "expect_text": "grievance"},
    {"question": "Can a marketplace charge cancellation fees after a buyer cancels?",
     "retailer": "regulation", "expect_text": "cancellation"},
    # Unanswerable: these must return nothing.
    {"question": "What is the best pizza topping?", "expect": "no_match"},
    {"question": "Do you deliver to the moon?", "expect": "no_match"},
    {"question": "Who won the cricket world cup?", "expect": "no_match"},
    {"question": "How do I bake sourdough bread?", "expect": "no_match"},
]


def evaluate(index: PolicyIndex, questions: list[dict], k: int = 3) -> dict:
    passed = 0
    # Lowest cut-off each answerable question needs to keep a correct passage;
    # None when a keyword match keeps one whatever the cut-off.
    needed: list[float | None] = []
    stray_scores: list[float] = []
    for number, item in enumerate(questions, start=1):
        retailer = item.get("retailer")
        retailers = [retailer] if retailer else None
        # Retrieve with no cut-off so we can see where the scores fall.
        hits = index.search(item["question"], retailers, k=k, min_relevance=-1.0)
        print(f"\n[{number}] {item['question']}")
        for hit in hits:
            where = f" > {hit.heading}" if hit.heading else ""
            print(
                f"    {hit.relevance:5.2f}  {'+'.join(hit.matched_by) or 'below cut-off':16s} "
                f"{hit.retailer}{where}: {hit.text[:90].replace(chr(10), ' ')}…"
            )
        # Pass/fail uses the real behaviour, including the configured cut-off.
        real = index.search(item["question"], retailers, k=k)
        if item.get("expect") == "no_match":
            stray_scores.extend(h.relevance for h in hits if "keyword" not in h.matched_by)
            ok = not real
            verdict = "OK (nothing returned)" if ok else (
                f"returned {len(real)} passages for an unanswerable question (cut-off too loose)"
            )
        else:
            ok = bool(real)
            if ok and item.get("expect_text"):
                ok = any(item["expect_text"].lower() in h.text.lower() for h in real)
            correct = [
                h for h in hits
                if not item.get("expect_text") or item["expect_text"].lower() in h.text.lower()
            ]
            if correct:
                needed.append(
                    None if any("keyword" in h.matched_by for h in correct)
                    else max(h.relevance for h in correct)
                )
            verdict = "OK" if ok else (
                "MISS: nothing passed the cut-off" if not real
                else f"MISS: no returned passage mentions '{item['expect_text']}'"
            )
        passed += ok
        print(f"    -> {verdict}")

    print(f"\n{passed}/{len(questions)} questions passed.")
    suggestion = None
    strongest_stray = max(stray_scores) if stray_scores else None
    meaning_only = [score for score in needed if score is not None]
    weakest_good = min(meaning_only) if meaning_only else None
    if strongest_stray is not None:
        print(f"Strongest meaning-only match for an unanswerable question: {strongest_stray:.2f}")
    if weakest_good is not None:
        print(f"Weakest correct passage found by meaning only: {weakest_good:.2f}")
    if needed and len(meaning_only) < len(needed):
        print(f"{len(needed) - len(meaning_only)} answerable questions also match by keyword, "
              "so they keep their passages at any cut-off.")
    if strongest_stray is not None:
        if weakest_good is None:
            suggestion = round(strongest_stray + 0.02, 2)
            reason = "just above the strongest unanswerable match"
        elif strongest_stray < weakest_good:
            suggestion = round((strongest_stray + weakest_good) / 2, 2)
            reason = "midway between the two"
    if suggestion is not None:
        print(f"Suggested POLICY_MIN_RELEVANCE={suggestion} ({reason}). "
              "Set it and run this again to confirm.")
    elif strongest_stray is not None:
        print("No cut-off separates these: an answerable question's only correct passage scores "
              "below an unanswerable one. Improve that source rather than the cut-off.")
    return {"passed": passed, "total": len(questions), "suggested_min_relevance": suggestion}


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Evaluate policy search on a question set.")
    parser.add_argument(
        "--questions", type=Path, default=DEFAULT_QUESTIONS_FILE,
        help="JSON question file (see module docstring); default: data/policies/eval_questions.json",
    )
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args(argv)
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    result = evaluate(PolicyIndex(embedder_from_env()), questions, k=args.k)
    return 0 if result["passed"] == result["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
