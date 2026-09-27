"""Chunk and embed the policy corpus into the Chroma ``retailer_policies`` collection.

    python scripts/policies/build_policy_index.py
    python scripts/policies/build_policy_index.py --ask "Can I return a phone bought on Flipkart?"

The rebuild is staged: the new collection is written separately and swapped in
only when complete, so an interrupted run leaves the previous index usable.
Embedding backend: POLICY_EMBEDDINGS=local (default) | gateway | hash.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv  # noqa: E402

from tools.policy_corpus import DEFAULT_POLICY_DIR, load_corpus, stale_sources  # noqa: E402
from tools.policy_rag import (  # noqa: E402
    DEFAULT_CHROMA_DIR, PolicyAdvisor, PolicyIndex, embedder_from_env, max_age_days,
)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Build the Chroma policy index.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_POLICY_DIR)
    parser.add_argument(
        "--chroma", type=Path, default=None,
        help=f"Index folder (default: POLICY_CHROMA_DIR or {DEFAULT_CHROMA_DIR})",
    )
    parser.add_argument("--ask", help="Ask a test question after building")
    parser.add_argument("--retailer", action="append", help="Limit --ask to a retailer (repeatable)")
    args = parser.parse_args(argv)

    documents, problems = load_corpus(args.corpus)
    for problem in problems:
        print(f"[skipped] {problem}")
    if not documents:
        print("No valid policy documents found. Run scripts/policies/fetch_policies.py first.")
        return 1

    index = PolicyIndex(embedder_from_env(), args.chroma)
    report = index.build(documents)
    print(
        f"Indexed {report.documents} documents as {report.chunks} chunks "
        f"with {report.embedder} embeddings."
    )
    for retailer, count in sorted(report.by_retailer.items()):
        print(f"  {retailer:18s} {count} chunks")
    limit = max_age_days()
    for source_id, age in stale_sources({d.source_id: d.retrieved_at for d in documents}, limit):
        when = f"{age} days old" if age is not None else "unreadable retrieved_at date"
        print(f"[stale  ] {source_id}: {when} (limit {limit}). Re-fetch it or save a fresh copy by hand.")

    if args.ask:
        answer = PolicyAdvisor(index).answer(args.ask, args.retailer)
        print(f"\nQ: {args.ask}\n[{answer.mode}] {answer.answer}")
        if answer.note:
            print(f"note: {answer.note}")
        for hit in answer.citations:
            print(f"  [{hit.number}] {hit.retailer} {hit.source_url} (retrieved {hit.retrieved_at}, relevance {hit.relevance})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
