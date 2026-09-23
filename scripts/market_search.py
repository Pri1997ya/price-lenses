#!/usr/bin/env python3
"""Fetch Indian market offers from SerpAPI and Apify and store them in PostgreSQL."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.market_config import MarketSettings  # noqa: E402
from tools.market_db import MarketDatabase  # noqa: E402
from tools.market_service import MarketInvestigatorService, build_providers  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Amazon/Flipkart URL, ASIN, or product name")
    parser.add_argument("--limit", type=int, default=20, help="maximum results per provider")
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["serpapi", "apify"],
        choices=["serpapi", "apify"],
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    settings = MarketSettings.from_env()
    providers = build_providers(settings, tuple(args.providers))
    with MarketDatabase(settings.database_url) as database:
        results = MarketInvestigatorService(database, providers).search(
            args.query, args.limit
        )
    print(json.dumps([asdict(result) for result in results], indent=2))


if __name__ == "__main__":
    main()
