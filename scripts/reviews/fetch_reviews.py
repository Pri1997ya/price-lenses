"""Collect reviews for catalog products into data/reviews/<ASIN>.jsonl.

    python scripts/reviews/fetch_reviews.py                          # every catalog product
    python scripts/reviews/fetch_reviews.py --asin B0CS5XW6TN --sources reddit,youtube
    python scripts/reviews/fetch_reviews.py --dry-run                 # show what would run

Sources that are not configured are skipped (see .env.example). Amazon reviews
use the catalog's Amazon URL; Flipkart reviews need a stored Flipkart offer
(run scripts/market_search.py first, with DATABASE_URL set). Re-running adds
new reviews to what is already stored.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv  # noqa: E402

from tools.review_corpus import DEFAULT_REVIEW_DIR, SOURCES, save_reviews  # noqa: E402
from tools.review_sources import configured_sources  # noqa: E402

CATALOG = Path(__file__).resolve().parents[1] / "scrapers" / "curated_100_electronics.json"
SETUP_HINT = {
    "amazon": "set APIFY_API_TOKEN, APIFY_AMAZON_REVIEWS_ACTOR and APIFY_AMAZON_REVIEWS_INPUT",
    "flipkart": "set APIFY_API_TOKEN, APIFY_FLIPKART_REVIEWS_ACTOR and APIFY_FLIPKART_REVIEWS_INPUT",
    "reddit": "set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET",
    "youtube": "set YOUTUBE_API_KEY",
}


def search_name(product: dict) -> str:
    """"Samsung Galaxy S24 Ultra": what owners call it on Reddit and YouTube."""
    brand, model = product.get("brand", "").strip(), product.get("model", "").strip()
    if not model:
        return product["name"]
    return model if model.lower().startswith(brand.lower()) else f"{brand} {model}".strip()


def flipkart_urls(asins: list[str]) -> dict[str, str]:
    database_url = os.getenv("DATABASE_URL") or os.getenv("PL_DATABASE_URL")
    if not database_url:
        return {}
    from tools.market_db import MarketDatabase

    found = {}
    with MarketDatabase(database_url) as database:
        for asin in asins:
            for offer in database.offers_for_product(asin):
                if "flipkart" in (offer.get("marketplace") or "") and offer.get("url"):
                    found[asin] = offer["url"]
                    break
    return found


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Collect product reviews for defect detection.")
    parser.add_argument("--asin", action="append", help="Catalog ASIN (repeatable); default: all")
    parser.add_argument("--sources", default=",".join(SOURCES), help="Comma-separated: " + ",".join(SOURCES))
    parser.add_argument("--limit", type=int, default=200, help="Marketplace reviews per product")
    parser.add_argument("--out", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    wanted = [name.strip() for name in args.sources.split(",") if name.strip()]
    unknown = set(wanted) - set(SOURCES)
    if unknown:
        parser.error(f"unknown sources: {', '.join(sorted(unknown))}")
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    products = [p for p in catalog if not args.asin or p["asin"] in args.asin]
    if not products:
        print("No matching catalog products.")
        return 1

    collectors = configured_sources()
    active = {}
    for name in wanted:
        if collectors.get(name) is None:
            print(f"[skip   ] {name}: not configured ({SETUP_HINT[name]}).")
        else:
            active[name] = collectors[name]
    if not active:
        print("No review source is configured.")
        return 1
    fk_urls = flipkart_urls([p["asin"] for p in products]) if "flipkart" in active else {}

    failures = 0
    for product in products:
        asin, name = product["asin"], search_name(product)
        collected = []
        for source, collector in active.items():
            if source == "amazon":
                target = product.get("url") or f"https://www.amazon.in/dp/{asin}"
            elif source == "flipkart":
                target = fk_urls.get(asin)
                if not target:
                    print(f"[skip   ] {asin} flipkart: no stored Flipkart offer URL.")
                    continue
            else:
                target = name
            if args.dry_run:
                print(f"[dry-run] {asin} {source}: {target}")
                continue
            try:
                if source in ("amazon", "flipkart"):
                    reviews = collector.fetch(asin, target, limit=args.limit)
                else:
                    reviews = collector.fetch(asin, target)
            except Exception as exc:  # one failing source must not stop the batch
                failures += 1
                print(f"[failed ] {asin} {source}: {exc}")
                continue
            print(f"[ok     ] {asin} {source}: {len(reviews)} reviews ({target})")
            collected.extend(reviews)
        if collected:
            added, total = save_reviews(asin, collected, args.out)
            print(f"          {asin}: {added} new, {total} stored")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
