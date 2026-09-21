#!/usr/bin/env python3
"""
PriceLens Batch Ingestion Runner
=================================
Automates data extraction for the 100 curated electronics:
1. Scrapes PriceHistoryApp (Free: 1-year daily price history + competitor pricing)
2. Calls Apify Amazon Scraper (Costly: Cached strictly to disk to prevent re-billing)
3. Merges and ingests unified records into PostgreSQL

Usage:
    # 1. Run for a single ASIN to test:
    python batch_runner.py --asin B0CS5XW6TN

    # 2. Run for all 100 products:
    python batch_runner.py --all

    # 3. Only ingest existing cached files into PostgreSQL:
    python batch_runner.py --ingest-only
"""

import os
import re
import json
import time
import argparse
import asyncio
from datetime import datetime, timedelta
import httpx
import psycopg2
from psycopg2.extras import execute_values

from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
CATALOG_PATH = os.path.join(SCRIPT_DIR, "curated_100_electronics.json")
if not os.path.exists(CATALOG_PATH):
    CATALOG_PATH = os.path.join(WORKSPACE_DIR, "curated_100_electronics.json")
RAW_APIFY_DIR = os.path.join(WORKSPACE_DIR, "data", "raw_apify")
RAW_HISTORY_DIR = os.path.join(WORKSPACE_DIR, "data", "raw_history")

# Environment
DATABASE_URL = os.getenv("DATABASE_URL")
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN", "")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


# ==============================================================================
# 1. PriceHistoryApp Scraper (Free Time-Series & Competitor Prices)
# ==============================================================================
async def scrape_price_history(product_name: str, amazon_url: str, custom_url: str = None) -> dict:
    """
    Scrapes 1-year historical price series and competitor offers from PriceHistoryApp.
    """
    if custom_url:
        target_url = custom_url
    else:
        # Construct slug from product name or search
        slug = re.sub(r'[^a-zA-Z0-9]+', '-', product_name.lower()).strip('-')
        target_url = f"https://pricehistoryapp.com/product/{slug}"

    async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as client:
        try:
            response = await client.get(target_url, headers=HEADERS)
            html_text = response.text
        except Exception as e:
            print(f"   [Error] Failed fetching price history for {product_name}: {e}")
            return None

    # 1. Extract 1-year time series
    matches = re.findall(
        r'date\\*"[^\d]*(\d{4}-\d{2}-\d{2})\\*"[^\d]*price\\*"[^\d]*(\d+)',
        html_text,
    )

    # Deduplicate and sort chronologically
    unique_map = {d: int(p) for d, p in matches}
    sorted_points = [{"date": k, "price": v} for k, v in sorted(unique_map.items())]

    one_year_ago = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    clean_history = [p for p in sorted_points if p["date"] >= one_year_ago]
    if not clean_history and sorted_points:
        clean_history = sorted_points[-365:]  # Fallback to latest available points

    final_1yr_data = clean_history

    # 2. Extract core pricing metadata
    meta_match = re.search(r'\\"meta\\":(\{.*?\\"is_synthetic\\":(?:true|false)\})', html_text)
    meta_data = json.loads(meta_match.group(1).replace('\\"', '"')) if meta_match else {}

    # 3. Extract competitor store comparisons
    raw_stores = []
    stores_match = re.search(r'\\"storeComparisons\\":\s*(\[\{.*?\}\])(?=,\\"|\}\])', html_text)
    if stores_match:
        cleaned_json = stores_match.group(1).replace('\\"', '"')
        try:
            raw_stores = json.loads(cleaned_json)
        except json.JSONDecodeError:
            blocks = re.findall(r'\{[^{}]*\\"store\\":[^{}]*\}', stores_match.group(1))
            raw_stores = [json.loads(b.replace('\\"', '"')) for b in blocks if "store" in b]

    # Fallback to schema offers
    if len(raw_stores) <= 1:
        schema_offers = re.findall(
            r'\\"@type\\":\\"Offer\\",\\"url\\":\\"([^\\"]+)\\",\\"priceCurrency\\":\\"INR\\",\\"price\\":(\d+).*?\\"seller\\":\{\\"@type\\":\\"Organization\\",\\"name\\":\\"([^\\"]+)\\"\}',
            html_text,
        )
        for offer_url, price, seller_name in schema_offers:
            if not any(s.get("store") == seller_name for s in raw_stores):
                raw_stores.append({
                    "store": seller_name,
                    "price": int(price),
                    "isCurrent": seller_name.lower() in target_url.lower(),
                    "url": offer_url.replace(r"\u0026", "&"),
                })

    competitors = {
        s.get("store"): {
            "price": s.get("price"),
            "is_current": s.get("isCurrent", False),
            "url": s.get("url"),
        }
        for s in raw_stores if s.get("store")
    }

    # 4. Extract deal score rating
    score_total_match = re.search(r'\\"scoreBreakup\\":\{.*?\\"total\\":(\d+)', html_text)
    score_tag_match = re.search(r'\\"verdict\\":\{.*?\\"tag\\":\\"([^\\"]+)\\"', html_text)
    deal_score = int(score_total_match.group(1)) if score_total_match else None
    deal_verdict = score_tag_match.group(1) if score_tag_match else None

    return {
        "price_overview": {
            "current_price": meta_data.get("price"),
            "all_time_low": meta_data.get("lowest"),
            "all_time_high": meta_data.get("highest"),
            "avg_30_days": meta_data.get("avg30"),
            "overall_average": meta_data.get("average"),
            "low_6_months": meta_data.get("low180"),
            "trend_percentage": meta_data.get("trend_pct"),
        },
        "deal_evaluation": {
            "score": deal_score,
            "verdict": deal_verdict,
        },
        "competitor_pricing": competitors,
        "time_series": final_1yr_data,
        "historical_data_points": len(final_1yr_data)
    }


# ==============================================================================
# 2. Apify Scraper (Costly - Checked against local cache first)
# ==============================================================================
def fetch_apify_payload(asin: str, amazon_url: str) -> dict:
    """
    Fetches Amazon rich product details from Apify.
    Uses cached JSON if available on disk.
    """
    cache_file = os.path.join(RAW_APIFY_DIR, f"{asin}.json")
    if os.path.exists(cache_file):
        print(f"   [Cache HIT] Apify payload found locally for {asin}")
        with open(cache_file, "r") as f:
            return json.load(f)

    if not APIFY_API_TOKEN:
        print(f"   [Warning] APIFY_API_TOKEN not set. Cannot fetch new data from Apify.")
        return None

    print(f"   [Apify Call] Fetching live payload for {asin} (Billable)...")
    try:
        from apify_client import ApifyClient
        client = ApifyClient(APIFY_API_TOKEN)
        run_input = {
            "detailsUrls": [{"url": amazon_url}],
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
        run = client.actor("apify/e-commerce-scraping-tool").call(
            run_input=run_input,
            memory_mbytes=512
        )
        dataset_id = getattr(run, "default_dataset_id", None) or (run.get("defaultDatasetId") if isinstance(run, dict) else None)
        dataset_items = list(client.dataset(dataset_id).iterate_items()) if dataset_id else []
        if dataset_items:
            apify_data = dataset_items
            with open(cache_file, "w") as f:
                json.dump(apify_data, f, indent=2)
            print(f"   [Saved] Cached Apify response to {cache_file}")
            return apify_data
    except Exception as e:
        print(f"   [Error] Apify execution failed: {e}")

    return None


# ==============================================================================
# 3. PostgreSQL Database Ingestion
# ==============================================================================
def ingest_to_postgres(asin: str, apify_data: dict, history_data: dict):
    """
    Merges metadata from Apify and price history from PriceHistoryApp into PostgreSQL.
    """
    if not apify_data and not history_data:
        print(f"   [Skip] No data to ingest for {asin}")
        return

    if isinstance(apify_data, list) and apify_data:
        apify_data = apify_data[0]

    props = (apify_data or {}).get("additionalProperties") or {}
    attributes = props.get("attributes") or []

    def get_attr(key_name):
        for attr in attributes:
            if isinstance(attr, dict) and attr.get("key") and key_name.lower() in attr["key"].lower():
                return attr.get("value")
        return None

    title = (apify_data or {}).get("name") or f"Product {asin}"
    brand_obj = (apify_data or {}).get("brand")
    brand = brand_obj.get("slogan") if isinstance(brand_obj, dict) else (brand_obj if isinstance(brand_obj, str) else "Unknown")
    model = get_attr("Model Name") or get_attr("Processor Series") or "Flagship"
    color = get_attr("Colour")
    storage = get_attr("Memory Storage")
    ram = get_attr("RAM Memory")
    warranty = get_attr("Warranty Description")
    image_url = (apify_data or {}).get("image")
    
    raw_list_price = props.get("listPrice")
    list_price = raw_list_price.get("value") if isinstance(raw_list_price, dict) else raw_list_price
    
    rating = props.get("stars")
    review_count = (apify_data or {}).get("reviewCount")
    in_stock = props.get("inStock", True)
    
    raw_seller = props.get("seller")
    seller_name = raw_seller.get("name") if isinstance(raw_seller, dict) else (raw_seller if isinstance(raw_seller, str) else None)
    
    ai_summary = (apify_data or {}).get("aiSummary")

    price_overview = (history_data or {}).get("price_overview", {})
    current_price = price_overview.get("current_price") or (apify_data or {}).get("offers", {}).get("price")
    atl = price_overview.get("all_time_low")
    ath = price_overview.get("all_time_high")
    avg_30 = price_overview.get("avg_30_days")
    overall_avg = price_overview.get("overall_average")

    def get_db_connection():
        try:
            return psycopg2.connect(DATABASE_URL)
        except Exception:
            from urllib.parse import urlparse
            p = urlparse(DATABASE_URL)
            return psycopg2.connect(
                dbname=p.path.lstrip("/"),
                user=p.username,
                password=p.password,
                host=p.hostname,
                port=p.port or 5432,
                hostaddr="52.95.251.153",
                sslmode="require"
            )

    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # 1. Upsert Products
            cur.execute("""
                INSERT INTO products (
                    canonical_id, title, brand, model, color, storage, ram, image_url,
                    list_price, current_price, all_time_low, all_time_high, avg_30_days,
                    overall_avg, customer_rating, review_count, in_stock, seller_name,
                    warranty_description, ai_reviews_summary
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (canonical_id) DO UPDATE SET
                    current_price = COALESCE(EXCLUDED.current_price, products.current_price),
                    all_time_low = COALESCE(EXCLUDED.all_time_low, products.all_time_low),
                    all_time_high = COALESCE(EXCLUDED.all_time_high, products.all_time_high),
                    avg_30_days = COALESCE(EXCLUDED.avg_30_days, products.avg_30_days),
                    overall_avg = COALESCE(EXCLUDED.overall_avg, products.overall_avg),
                    in_stock = EXCLUDED.in_stock;
            """, (
                asin, title, brand, model, color, storage, ram, image_url,
                list_price, current_price, atl, ath, avg_30, overall_avg,
                rating, review_count, in_stock, seller_name, warranty, ai_summary
            ))

            # 2. Upsert Time Series History
            time_series = (history_data or {}).get("time_series", [])
            if time_series:
                history_tuples = [(asin, p["date"], p["price"]) for p in time_series]
                execute_values(cur, """
                    INSERT INTO price_history (canonical_id, recorded_date, price)
                    VALUES %s
                    ON CONFLICT (canonical_id, recorded_date) DO UPDATE SET price = EXCLUDED.price;
                """, history_tuples)

            # 3. Upsert Competitor Offers
            competitors = (history_data or {}).get("competitor_pricing", {})
            if competitors:
                offer_tuples = [
                    (asin, store_name, data["price"], data["url"], data.get("is_current", False))
                    for store_name, data in competitors.items() if data.get("price")
                ]
                if offer_tuples:
                    cur.execute("DELETE FROM live_store_offers WHERE canonical_id = %s;", (asin,))
                    execute_values(cur, """
                        INSERT INTO live_store_offers (canonical_id, retailer, price, product_url, is_current)
                        VALUES %s;
                    """, offer_tuples)

            conn.commit()
        conn.close()
        print(f"   [DB Success] Ingested {asin} into PostgreSQL.")
    except Exception as e:
        print(f"   [DB Error] Could not ingest {asin}: {e}")


# ==============================================================================
# 4. Orchestration Loop
# ==============================================================================
async def process_product(item: dict):
    asin = item["asin"]
    url = item["url"]
    name = item["name"]

    print(f"\n==================================================")
    print(f"Processing [{item.get('id', '?')}/100]: {name}")
    print(f"ASIN: {asin} | URL: {url}")

    os.makedirs(RAW_APIFY_DIR, exist_ok=True)
    os.makedirs(RAW_HISTORY_DIR, exist_ok=True)

    # 1. Price History
    slug = item.get("slug")
    custom_url = f"https://pricehistoryapp.com/product/{slug}" if slug else None
    hist_cache = os.path.join(RAW_HISTORY_DIR, f"{asin}.json")
    if os.path.exists(hist_cache):
        print(f"   [Cache HIT] Historical data found locally.")
        with open(hist_cache, "r") as f:
            hist_data = json.load(f)
    else:
        print(f"   [Scraping] PriceHistoryApp...")
        hist_data = await scrape_price_history(name, url, custom_url)
        if hist_data:
            with open(hist_cache, "w") as f:
                json.dump(hist_data, f, indent=2)

    # 2. Apify
    apify_data = fetch_apify_payload(asin, url)

    # 3. PostgreSQL Ingestion
    ingest_to_postgres(asin, apify_data, hist_data)


async def main():
    parser = argparse.ArgumentParser(description="PriceLens Batch Data Pipeline")
    parser.add_argument("--asin", type=str, help="Process a single ASIN (e.g. B0CS5XW6TN)")
    parser.add_argument("--range", type=str, help="Process a range of IDs (e.g. 5-15)")
    parser.add_argument("--ids", type=str, help="Comma-separated IDs (e.g. 5,6,7)")
    parser.add_argument("--all", action="store_true", help="Process all products in catalog")
    parser.add_argument("--ingest-only", action="store_true", help="Re-ingest all cached files to DB")
    args = parser.parse_args()

    if not os.path.exists(CATALOG_PATH):
        print(f"Error: Catalog not found at {CATALOG_PATH}")
        return

    with open(CATALOG_PATH, "r") as f:
        catalog = json.load(f)

    if args.asin:
        matched = [p for p in catalog if p["asin"] == args.asin]
        if not matched:
            print(f"ASIN {args.asin} not in catalog. Creating ad-hoc item...")
            matched = [{"id": 0, "asin": args.asin, "name": f"Product {args.asin}", "url": f"https://www.amazon.in/dp/{args.asin}"}]
        await process_product(matched[0])

    elif args.range:
        start_id, end_id = map(int, args.range.split("-"))
        subset = [p for p in catalog if start_id <= p.get("id", 0) <= end_id]
        print(f"Starting batch run for {len(subset)} products (IDs {start_id} to {end_id})...")
        for p in subset:
            await process_product(p)
            if not args.ingest_only:
                await asyncio.sleep(1)

    elif args.ids:
        target_ids = set(map(int, args.ids.split(",")))
        subset = [p for p in catalog if p.get("id", 0) in target_ids]
        print(f"Starting batch run for {len(subset)} products...")
        for p in subset:
            await process_product(p)
            if not args.ingest_only:
                await asyncio.sleep(1)

    elif args.all or args.ingest_only:
        print(f"Starting batch run for {len(catalog)} products...")
        for p in catalog:
            await process_product(p)
            if not args.ingest_only:
                await asyncio.sleep(1) # Polite pause
    else:
        print("Please provide --asin <ASIN>, --range <START-END>, --ids <1,2,3>, --all, or --ingest-only.")


if __name__ == "__main__":
    asyncio.run(main())
