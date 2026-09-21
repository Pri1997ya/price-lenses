#!/usr/bin/env python3
"""
PriceLens PostgreSQL Database Initializer
=========================================
Initializes all relational and time-series tables in PostgreSQL,
and seeds the reference tables (sales_calendar, authorized_sellers).
"""

import os
import psycopg2
from dotenv import load_dotenv

# Load .env file
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("❌ ERROR: DATABASE_URL not found in .env file.")
    exit(1)

SCHEMA_SQL = """
-- 1. Master Product Catalog
CREATE TABLE IF NOT EXISTS products (
    canonical_id VARCHAR(50) PRIMARY KEY,       -- ASIN (e.g. 'B0CS5XW6TN')
    title TEXT NOT NULL,
    brand VARCHAR(100),
    model VARCHAR(100),
    color VARCHAR(50),
    storage VARCHAR(50),
    ram VARCHAR(50),
    image_url TEXT,
    list_price DOUBLE PRECISION,
    current_price DOUBLE PRECISION,
    all_time_low DOUBLE PRECISION,
    all_time_high DOUBLE PRECISION,
    avg_30_days DOUBLE PRECISION,
    overall_avg DOUBLE PRECISION,
    customer_rating DOUBLE PRECISION,
    review_count INTEGER,
    in_stock BOOLEAN DEFAULT TRUE,
    seller_name VARCHAR(150),
    warranty_description TEXT,
    ai_reviews_summary TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 2. Dynamic Price History (N points per product)
CREATE TABLE IF NOT EXISTS price_history (
    id BIGSERIAL PRIMARY KEY,
    canonical_id VARCHAR(50) NOT NULL REFERENCES products(canonical_id) ON DELETE CASCADE,
    recorded_date DATE NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    CONSTRAINT unique_product_date UNIQUE(canonical_id, recorded_date)
);
CREATE INDEX IF NOT EXISTS idx_history_date ON price_history(canonical_id, recorded_date DESC);

-- 3. Competitor Live Offers (Flipkart, Amazon, Croma)
CREATE TABLE IF NOT EXISTS live_store_offers (
    offer_id BIGSERIAL PRIMARY KEY,
    canonical_id VARCHAR(50) NOT NULL REFERENCES products(canonical_id) ON DELETE CASCADE,
    retailer VARCHAR(100) NOT NULL,             -- 'Amazon', 'Flipkart', 'Croma'
    price DOUBLE PRECISION NOT NULL,
    product_url TEXT NOT NULL,
    is_current BOOLEAN DEFAULT FALSE,
    fetched_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 4. Annual E-Commerce Sales Calendar (Upcoming Sales Engine)
CREATE TABLE IF NOT EXISTS sales_calendar (
    sale_id SERIAL PRIMARY KEY,
    sale_name VARCHAR(150) NOT NULL,
    retailer VARCHAR(150) NOT NULL,
    approx_start_date DATE NOT NULL,
    approx_end_date DATE NOT NULL,
    typical_category_discount_pct DOUBLE PRECISION,
    sale_type VARCHAR(50),
    CONSTRAINT unique_sale_window UNIQUE(sale_name, approx_start_date)
);

-- 5. Authorized Seller Registry (Reference Lookup)
CREATE TABLE IF NOT EXISTS authorized_sellers (
    id SERIAL PRIMARY KEY,
    brand VARCHAR(100) NOT NULL,
    retailer VARCHAR(100) NOT NULL,
    seller_name VARCHAR(150) NOT NULL,
    is_authorized BOOLEAN DEFAULT TRUE,
    CONSTRAINT unique_brand_retailer_seller UNIQUE(brand, retailer, seller_name)
);

-- Seed Reference Sales Calendar
INSERT INTO sales_calendar (sale_name, retailer, approx_start_date, approx_end_date, typical_category_discount_pct, sale_type) VALUES
('Big Billion Days & Great Indian Festival', 'Amazon & Flipkart', '2026-10-06', '2026-10-15', 0.18, 'MAJOR_FESTIVE'),
('Republic Day Sale', 'Amazon & Flipkart', '2027-01-19', '2027-01-26', 0.10, 'MAJOR_FESTIVE'),
('Amazon Prime Day', 'Amazon', '2027-07-15', '2027-07-17', 0.12, 'SEASONAL')
ON CONFLICT (sale_name, approx_start_date) DO NOTHING;

-- Seed Authorized Sellers Reference
INSERT INTO authorized_sellers (brand, retailer, seller_name, is_authorized) VALUES
('Apple', 'Amazon', 'Appario Retail Private Ltd', TRUE),
('Apple', 'Amazon', 'Indiflash', TRUE),
('Apple', 'Amazon', 'Cocoblu Retail', TRUE),
('Apple', 'Flipkart', 'SuperComNet', TRUE),
('Apple', 'Flipkart', 'IndiFlashMart', TRUE),
('Samsung', 'Amazon', 'Appario Retail Private Ltd', TRUE),
('Samsung', 'Amazon', 'Indiflash', TRUE),
('Samsung', 'Flipkart', 'SuperComNet', TRUE),
('All', 'Croma', 'Croma', TRUE),
('All', 'Croma', 'Infiniti Retail Ltd', TRUE),
('All', 'Reliance', 'Reliance Digital', TRUE),
('All', 'Vijay Sales', 'Vijay Sales', TRUE)
ON CONFLICT (brand, retailer, seller_name) DO NOTHING;
"""

def main():
    print("🔌 Connecting to PostgreSQL...")
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            print("🚀 Executing DDL Schema creation...")
            cur.execute(SCHEMA_SQL)
            conn.commit()

            # Verify tables
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                ORDER BY table_name;
            """)
            tables = [row[0] for row in cur.fetchall()]
            print("✅ Successfully initialized PostgreSQL. Active tables:")
            for t in tables:
                print(f"   • {t}")

        conn.close()
    except Exception as e:
        print(f"❌ Failed connecting or initializing database: {e}")
        exit(1)

if __name__ == "__main__":
    main()
