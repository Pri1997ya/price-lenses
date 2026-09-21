import os
import psycopg2
from urllib.parse import urlparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    """Establishes a robust connection to Neon PostgreSQL, with DNS fallback."""
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL is missing in environment.")
    try:
        return psycopg2.connect(DATABASE_URL, sslmode="require", connect_timeout=5)
    except Exception:
        # Fallback for macOS local DNS resolution issues with Neon poolers
        p = urlparse(DATABASE_URL)
        return psycopg2.connect(
            dbname=p.path.lstrip("/"),
            user=p.username,
            password=p.password,
            host=p.hostname,
            port=p.port or 5432,
            hostaddr="52.95.251.153",
            sslmode="require",
            connect_timeout=10
        )

def query_historical_trend(canonical_id: str) -> dict:
    """
    TOOL 1: The Baseline Engine
    Calculates time-series mathematical truth from price_history.
    Returns boundaries, percentiles, and S_history (Deal Health factor).
    """
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # 1. Get current price
            cur.execute("SELECT current_price FROM products WHERE canonical_id = %s", (canonical_id,))
            prod_row = cur.fetchone()
            if not prod_row or not prod_row[0]:
                return {"error": f"Product {canonical_id} or current_price not found."}
            current_price = float(prod_row[0])

            # 2. Get Overall Stats from history
            cur.execute("""
                SELECT MIN(price), MAX(price), AVG(price), COUNT(*) 
                FROM price_history 
                WHERE canonical_id = %s
            """, (canonical_id,))
            hist_stats = cur.fetchone()
            
            if not hist_stats or hist_stats[3] == 0:
                # Edge case: No history points
                return {
                    "canonical_id": canonical_id,
                    "current_price": current_price,
                    "true_atl": current_price,
                    "true_ath": current_price,
                    "total_history_days": 0,
                    "s_history": 50.0,
                    "historical_stance": "WAIT",
                    "note": "No price history available."
                }
            
            true_atl = float(hist_stats[0])
            true_ath = float(hist_stats[1])
            overall_avg = float(hist_stats[2])
            total_days = int(hist_stats[3])

            # 3. Get 30-Day Moving Average
            cur.execute("""
                SELECT AVG(price) FROM (
                    SELECT price FROM price_history 
                    WHERE canonical_id = %s 
                    ORDER BY recorded_date DESC LIMIT 30
                ) sub
            """, (canonical_id,))
            avg_30d_row = cur.fetchone()
            avg_30d = float(avg_30d_row[0]) if avg_30d_row and avg_30d_row[0] else overall_avg

            # 4. Calculate Price Percentile (how many days were more expensive than today)
            cur.execute("""
                SELECT COUNT(*) FROM price_history 
                WHERE canonical_id = %s AND price > %s
            """, (canonical_id, current_price))
            days_more_expensive = cur.fetchone()[0]
            percentile = (days_more_expensive / total_days) * 100.0

            # 5. Calculate S_history Formula
            if true_ath > true_atl:
                s_history = 100.0 * (1.0 - (current_price - true_atl) / (true_ath - true_atl))
                # Clamp between 0 and 100
                s_history = max(0.0, min(100.0, s_history))
            else:
                # Flat price history
                s_history = 50.0
            
            # 6. Determine Stance
            historical_stance = "BUY_NOW" if s_history >= 75.0 else "WAIT"

            return {
                "canonical_id": canonical_id,
                "current_price": current_price,
                "true_atl": true_atl,
                "true_ath": true_ath,
                "avg_30d": round(avg_30d, 2),
                "overall_avg": round(overall_avg, 2),
                "total_history_days": total_days,
                "price_percentile": round(percentile, 2),
                "s_history": round(s_history, 2),
                "historical_stance": historical_stance
            }
    except Exception as e:
        return {"error": str(e)}
    finally:
        if 'conn' in locals() and conn:
            conn.close()

def query_sale_event_drops(canonical_id: str) -> dict:
    """
    TOOL 2: The Target Price Engine
    Finds historical drop prices during mega-sales (Sep/Oct, Jan, Jul) 
    or computes a safe fallback estimate for WAIT recommendations.
    """
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # 1. Get baseline prices
            cur.execute("SELECT current_price FROM products WHERE canonical_id = %s", (canonical_id,))
            prod_row = cur.fetchone()
            if not prod_row:
                return {"error": f"Product {canonical_id} not found."}
            current_price = float(prod_row[0])
            
            # Also get 30-day avg and true ATL
            cur.execute("SELECT MIN(price) FROM price_history WHERE canonical_id = %s", (canonical_id,))
            atl_row = cur.fetchone()
            true_atl = float(atl_row[0]) if atl_row and atl_row[0] else current_price

            cur.execute("""
                SELECT AVG(price) FROM (
                    SELECT price FROM price_history 
                    WHERE canonical_id = %s ORDER BY recorded_date DESC LIMIT 30
                ) sub
            """, (canonical_id,))
            avg30_row = cur.fetchone()
            avg_30d = float(avg30_row[0]) if avg30_row and avg30_row[0] else current_price

            # 2. Look for historical festive drops (Months 9, 10 for BBD/Diwali, 1 for Republic Day, 7 for Prime Day)
            cur.execute("""
                SELECT MIN(price) 
                FROM price_history 
                WHERE canonical_id = %s 
                  AND (EXTRACT(MONTH FROM recorded_date) IN (1, 7, 9, 10))
            """, (canonical_id,))
            festive_row = cur.fetchone()
            
            festive_low = float(festive_row[0]) if festive_row and festive_row[0] else None

            # 3. Look up upcoming sale in calendar
            cur.execute("""
                SELECT sale_name, approx_start_date, typical_category_discount_pct 
                FROM sales_calendar 
                WHERE approx_start_date >= CURRENT_DATE 
                ORDER BY approx_start_date ASC LIMIT 1
            """)
            next_sale = cur.fetchone()
            
            sale_name = next_sale[0] if next_sale else "Upcoming Sale"
            discount_pct = float(next_sale[2]) if next_sale else 0.15

            # 4. Resolve Target Price
            if festive_low and festive_low < current_price:
                target_price = festive_low
                is_estimated = False
                rationale = f"Based on verified historical data during major sales (Jan/Jul/Sep/Oct), this product drops to ₹{target_price:,.0f}."
            else:
                # Fallback: estimate based on sales calendar applied to moving average
                estimated_drop = avg_30d * (1.0 - discount_pct)
                # Never recommend a price lower than true_atl just on estimation
                target_price = max(estimated_drop, true_atl) 
                is_estimated = True
                rationale = f"No prior festive history available. Using typical category discount ({discount_pct*100:.0f}%) applied to 30-day average."

            # Calculate expected discount % against current price
            expected_discount = ((current_price - target_price) / current_price) * 100.0

            return {
                "canonical_id": canonical_id,
                "current_price": current_price,
                "upcoming_sale": sale_name,
                "safe_target_price": round(target_price, 2),
                "is_estimated": is_estimated,
                "expected_discount_pct": round(expected_discount, 2),
                "rationale": rationale
            }
            
    except Exception as e:
        return {"error": str(e)}
    finally:
        if 'conn' in locals() and conn:
            conn.close()

if __name__ == "__main__":
    # Quick test when script is executed directly
    test_asin = "B0CS5XW6TN" # Samsung Galaxy S24 Ultra
    print(f"--- Testing Tools for {test_asin} ---")
    
    print("\\n1. query_historical_trend():")
    trend = query_historical_trend(test_asin)
    for k, v in trend.items():
        print(f"  {k}: {v}")
        
    print("\\n2. query_sale_event_drops():")
    drops = query_sale_event_drops(test_asin)
    for k, v in drops.items():
        print(f"  {k}: {v}")
