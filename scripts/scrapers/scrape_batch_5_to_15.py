#!/usr/bin/env python3
"""
Batch Apify Scraper for Products 5 to 15
=======================================
Scrapes product metadata and reviews for the 11 target products in a single optimized run,
caches each output to data/raw_apify/{asin}.json, and ingests into Neon PostgreSQL.
"""

import os
import json
import re
from dotenv import load_dotenv
from apify_client import ApifyClient
from batch_runner import RAW_APIFY_DIR, ingest_to_postgres, RAW_HISTORY_DIR, CATALOG_PATH

load_dotenv()
token = os.getenv("APIFY_API_TOKEN")
if not token:
    print("❌ ERROR: APIFY_API_TOKEN not found in .env")
    exit(1)

client = ApifyClient(token)

with open(CATALOG_PATH) as f:
    catalog = json.load(f)

targets = [p for p in catalog if 5 <= p.get("id", 0) <= 15]
asin_to_item = {p["asin"]: p for p in targets}

print(f"🚀 Starting batch Apify run for {len(targets)} products (IDs 5 to 15)...")
details_urls = [{"url": p["url"]} for p in targets]

run_input = {
    "detailsUrls": details_urls,
    "countryCode": "in",
    "additionalProperties": True,
    "additionalReviewProperties": True,
    "fieldsToAnalyze": [
        "offers",
        "brand",
        "description",
        "additionalProperties",
        "url"
    ],
    "scrapeInfluencerProducts": False,
    "scrapeReviewsDelivery": False
}

print("⏳ Calling Actor: apify/e-commerce-scraping-tool (Memory: 1024MB)...")
run = client.actor("apify/e-commerce-scraping-tool").call(
    run_input=run_input,
    memory_mbytes=1024,
)

status = getattr(run, "status", None) or (run.get("status") if isinstance(run, dict) else "UNKNOWN")
dataset_id = getattr(run, "default_dataset_id", None) or (run.get("defaultDatasetId") if isinstance(run, dict) else None)
print(f"✅ Run completed with status: {status}")

dataset_items = list(client.dataset(dataset_id).iterate_items()) if dataset_id else []
print(f"📊 Retrieved {len(dataset_items)} total item(s) from dataset.")

os.makedirs(RAW_APIFY_DIR, exist_ok=True)

# Map items back to their target ASIN
for item in dataset_items:
    props = item.get("additionalProperties") or {}
    matched_asin = props.get("originalAsin") or props.get("asin")
    
    if not matched_asin or matched_asin not in asin_to_item:
        # Fallback: check URL or inputUrl
        item_url = item.get("url", "") + " " + item.get("inputUrl", "")
        for asin in asin_to_item:
            if asin in item_url:
                matched_asin = asin
                break
                
    if matched_asin:
        out_path = os.path.join(RAW_APIFY_DIR, f"{matched_asin}.json")
        with open(out_path, "w") as f:
            json.dump([item], f, indent=2)
        print(f"💾 Saved {matched_asin} -> {out_path} ({item.get('name', 'N/A')[:40]}...)")
    else:
        print(f"⚠️ Could not map item: {item.get('name')}")

print("\n🚀 Now ingesting products 5 to 15 into Neon PostgreSQL...")
for t in targets:
    asin = t["asin"]
    apify_file = os.path.join(RAW_APIFY_DIR, f"{asin}.json")
    hist_file = os.path.join(RAW_HISTORY_DIR, f"{asin}.json")
    
    apify_data = None
    if os.path.exists(apify_file):
        with open(apify_file) as f:
            apify_data = json.load(f)
            
    hist_data = None
    if os.path.exists(hist_file):
        with open(hist_file) as f:
            hist_data = json.load(f)
            
    ingest_to_postgres(asin, apify_data, hist_data)

print("\n✅ All finished!")
