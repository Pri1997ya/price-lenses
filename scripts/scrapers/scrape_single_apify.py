#!/usr/bin/env python3
"""
Single Product Apify Scraper
============================
Runs 'apify/e-commerce-scraping-tool' for Product #2:
Apple iPhone 15 Pro Max (256 GB) - Natural Titanium (ASIN: B0CHX1W1XY)
Saves the raw JSON output directly to data/raw_apify/B0CHX1W1XY.json
"""

import os
import json
from dotenv import load_dotenv
from apify_client import ApifyClient

import argparse

from batch_runner import CATALOG_PATH, RAW_APIFY_DIR

# 1. Load environment variables
load_dotenv()

APIFY_TOKEN = os.getenv("APIFY_API_TOKEN")

if not APIFY_TOKEN:
    print("❌ ERROR: APIFY_API_TOKEN not found in .env file.")
    print("Please add APIFY_API_TOKEN=your_token_here to your .env file.")
    exit(1)

# Parse arguments
parser = argparse.ArgumentParser(description="Single Product Apify Scraper")
parser.add_argument("--asin", type=str, default="B0CS5XW6TN", help="ASIN to scrape (e.g. B0CS5XW6TN)")
parser.add_argument("--url", type=str, default=None, help="Custom Amazon URL (optional)")
parser.add_argument("--force", action="store_true", help="Force re-scrape even if cached locally")
args = parser.parse_args()

ASIN = args.asin
URL = args.url or f"https://www.amazon.in/dp/{ASIN}"

# Check catalog for friendly product name
NAME = f"Product {ASIN}"
catalog_path = CATALOG_PATH
if os.path.exists(catalog_path):
    with open(catalog_path, "r") as f:
        catalog = json.load(f)
        for item in catalog:
            if item.get("asin") == ASIN:
                NAME = item.get("name")
                if not args.url:
                    URL = item.get("url")
                break

cache_file = os.path.join(RAW_APIFY_DIR, f"{ASIN}.json")
if os.path.exists(cache_file) and not args.force:
    print(f"📦 [Cache HIT] Payload already exists locally at: {cache_file}")
    print("💡 To force a live paid Apify run, re-run with: --force")
    with open(cache_file, "r") as f:
        data = json.load(f)
    first_item = data[0] if isinstance(data, list) and data else data
    print(f"   • Name: {first_item.get('name')}")
    print(f"   • Offers: {first_item.get('offers')}")
    exit(0)

print("=" * 60)
print(f"🚀 Launching Apify Actor for Product")
print(f"📦 Product: {NAME}")
print(f"🔗 ASIN: {ASIN}")
print(f"🌐 URL: {URL}")
print("=" * 60)

# 3. Define payload matching exact single-product parameters (Direct URL scrape, no keyword search)
run_input = {
    "detailsUrls": [
        {
            "url": URL
        }
    ],
    "countryCode": "in",
    "additionalProperties": True,
    "additionalReviewProperties": True,
    "fieldsToAnalyze": [
        "offers",
        "brand",
        "description",
        "additionalProperties",
        "url",
    ],
    "scrapeInfluencerProducts": False,
    "scrapeReviewsDelivery": False,
}

# 4. Initialize client and execute actor
client = ApifyClient(APIFY_TOKEN)

print("⏳ Calling Actor: apify/e-commerce-scraping-tool (Memory: 512MB, Timeout: 900s)...")
run = client.actor("apify/e-commerce-scraping-tool").call(
    run_input=run_input,
    memory_mbytes=512,
)

status = getattr(run, "status", None) or (run.get("status") if isinstance(run, dict) else "UNKNOWN")
dataset_id = getattr(run, "default_dataset_id", None) or (run.get("defaultDatasetId") if isinstance(run, dict) else None)
status_msg = getattr(run, "status_message", None)

print(f"✅ Run finished with status: {status}")
if status_msg:
    print(f"ℹ️ Status message: {status_msg}")

# 5. Fetch dataset items
dataset_items = list(client.dataset(dataset_id).iterate_items()) if dataset_id else []
print(f"📊 Retrieved {len(dataset_items)} item(s) from Apify dataset.")

if dataset_items:
    # Save output to disk immediately
    os.makedirs(RAW_APIFY_DIR, exist_ok=True)
    out_path = os.path.join(RAW_APIFY_DIR, f"{ASIN}.json")
    with open(out_path, "w") as f:
        json.dump(dataset_items, f, indent=2)
    print(f"💾 Successfully cached raw Apify payload to: {out_path}")

    # Display preview
    first_item = dataset_items[0]
    print("\n--- Summary Preview ---")
    print(f"Name: {first_item.get('name')}")
    print(f"Offers: {first_item.get('offers')}")
    props = first_item.get("additionalProperties", {})
    print(f"Seller: {props.get('seller')}")
    print(f"Stars: {props.get('stars')}")
    print(f"In Stock: {props.get('inStock')}")
else:
    print("⚠️ No items returned in the dataset.")
