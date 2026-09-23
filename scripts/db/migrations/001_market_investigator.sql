-- Live Market Investigator tables. These extend, and do not replace, the
-- products and price_history tables inherited from main.

CREATE TABLE IF NOT EXISTS market_search_runs (
    run_id VARCHAR(32) PRIMARY KEY,
    query TEXT NOT NULL,
    provider VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    error TEXT,
    result_count INTEGER NOT NULL DEFAULT 0 CHECK (result_count >= 0),
    started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_market_search_runs_query
    ON market_search_runs (lower(query), started_at DESC);

CREATE TABLE IF NOT EXISTS market_offers (
    offer_id VARCHAR(32) PRIMARY KEY,
    run_id VARCHAR(32) NOT NULL REFERENCES market_search_runs(run_id),
    canonical_id VARCHAR(50) NOT NULL REFERENCES products(canonical_id),
    provider VARCHAR(50) NOT NULL,
    marketplace VARCHAR(100) NOT NULL,
    external_id VARCHAR(255),
    title TEXT NOT NULL,
    url TEXT,
    price NUMERIC(14,2),
    currency VARCHAR(3) NOT NULL DEFAULT 'INR',
    original_price NUMERIC(14,2),
    rating NUMERIC(4,2),
    review_count BIGINT,
    seller_name VARCHAR(255),
    seller_rating NUMERIC(4,2),
    availability VARCHAR(255),
    shipping TEXT,
    offers_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    raw_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    fetched_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_market_offers_product
    ON market_offers (canonical_id, fetched_at DESC);
CREATE INDEX IF NOT EXISTS idx_market_offers_run
    ON market_offers (run_id);
CREATE INDEX IF NOT EXISTS idx_market_offers_marketplace
    ON market_offers (marketplace, fetched_at DESC);

CREATE TABLE IF NOT EXISTS market_offer_details (
    offer_id VARCHAR(32) PRIMARY KEY
        REFERENCES market_offers(offer_id) ON DELETE CASCADE,
    seller_id VARCHAR(255),
    price_with_offers NUMERIC(14,2),
    is_assured BOOLEAN,
    cod_available BOOLEAN,
    no_cost_emi BOOLEAN,
    return_policy TEXT,
    delivery_by TEXT,
    warranty TEXT,
    item_condition VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS market_offer_promotions (
    promotion_id VARCHAR(32) PRIMARY KEY,
    offer_id VARCHAR(32) NOT NULL
        REFERENCES market_offers(offer_id) ON DELETE CASCADE,
    promotion_type VARCHAR(50) NOT NULL,
    bank VARCHAR(100),
    card_type VARCHAR(100),
    description TEXT NOT NULL,
    amount NUMERIC(14,2),
    percent NUMERIC(7,3),
    is_emi BOOLEAN NOT NULL DEFAULT FALSE,
    source VARCHAR(100)
);

CREATE INDEX IF NOT EXISTS idx_market_promotions_offer
    ON market_offer_promotions (offer_id);

CREATE OR REPLACE VIEW latest_market_offers AS
SELECT offer_id, run_id, canonical_id, provider, marketplace, external_id,
       title, url, price, currency, original_price, rating, review_count,
       seller_name, seller_rating, availability, shipping, offers_json,
       raw_json, fetched_at
FROM (
    SELECT offer.*,
           ROW_NUMBER() OVER (
               PARTITION BY canonical_id, marketplace,
                            COALESCE(seller_name, ''), COALESCE(external_id, '')
               ORDER BY fetched_at DESC, offer_id DESC
           ) AS row_number
    FROM market_offers offer
) ranked
WHERE row_number = 1;
