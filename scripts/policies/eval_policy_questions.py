"""Check policy search quality against a fixed question set, and suggest a cut-off.

    python scripts/policies/eval_policy_questions.py
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
    {"question": "What is the best pizza topping?", "expect": "no_match"},
    {"question": "Do you deliver to the moon?", "expect": "no_match"},
]


def evaluate(index: PolicyIndex, questions: list[dict], k: int = 3) -> dict:
    passed = 0
    good_scores: list[float] = []
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
            if ok:
                good_scores.append(real[0].relevance)
            verdict = "OK" if ok else (
                "MISS: nothing passed the cut-off" if not real
                else f"MISS: no returned passage mentions '{item['expect_text']}'"
            )
        passed += ok
        print(f"    -> {verdict}")

    print(f"\n{passed}/{len(questions)} questions passed.")
    suggestion = None
    if good_scores:
        weakest_good = min(good_scores)
        strongest_stray = max(stray_scores) if stray_scores else None
        print(f"Weakest correct top result: {weakest_good:.2f}")
        if strongest_stray is not None:
            print(f"Strongest match for an unanswerable question: {strongest_stray:.2f}")
            if strongest_stray < weakest_good:
                suggestion = round((strongest_stray + weakest_good) / 2, 2)
        if suggestion is not None:
            print(f"Suggested POLICY_MIN_RELEVANCE={suggestion} (midway between the two).")
        else:
            print("No clean cut-off separates these; keep the default and rely on keyword matches.")
    return {"passed": passed, "total": len(questions), "suggested_min_relevance": suggestion}


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Evaluate policy search on a question set.")
    parser.add_argument("--questions", type=Path, help="JSON question file (see module docstring)")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args(argv)
    questions = (
        json.loads(args.questions.read_text(encoding="utf-8")) if args.questions else DEFAULT_QUESTIONS
    )
    result = evaluate(PolicyIndex(embedder_from_env()), questions, k=args.k)
    return 0 if result["passed"] == result["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
