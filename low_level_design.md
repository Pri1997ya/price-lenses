# PriceLens — Low-Level Design (LLD) Specification
**Project:** Autonomous Purchase-Timing Advisor  
**Architecture:** LangGraph Multi-Agent DAG + RAG (Chroma) + Deterministic Tooling  
**Data Sources:** Keepa API (Amazon History) + SerpApi (Google Shopping & Search)  
**Database:** DuckDB (Relational/Time-series) + Chroma (Vector Store)

---

## 1. System Overview & Graph Topology

PriceLens answers one specific question:  
> *"Given this product, my budget, and my deadline (7, 30, or 90 days) — should I buy now or wait? From which seller, in what condition, at what price, and on what verified evidence?"*

The system orchestrates three parallel specialist agents via a **LangGraph StateGraph**, enforces an explicit **Hard Join Barrier**, synthesizes their findings through a **Decision Agent**, and subjects the final draft to a **Deterministic (Non-LLM) Verifier Gate**.

### Execution Graph Topology

```
                                  [ START ]
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼ (Parallel ReAct)           ▼ (Parallel ReAct)           ▼ (Parallel ReAct)
┌─────────────────┐          ┌─────────────────┐          ┌──────────────────┐
│ History Analyst │          │ Market Investi- │          │  Eligibility     │
│ - DuckDB Stats  │          │   gator Agent   │          │  Analyst Agent   │
│ - Price RAG     │          │ - Google Search │          │ - SerpApi Offers │
│ - Max 6 tools   │          │ - Bank Promos   │          │ - Warranty RAG   │
└────────┬────────┘          └────────┬────────┘          └────────┬─────────┘
         │                            │                            │
         │ HistoryReport              │ MarketReport               │ EligibilityReport
         └────────────────────────────┼────────────────────────────┘
                                      ▼ (Hard Join Barrier)
                             ┌─────────────────┐
                             │ Decision Agent  │
                             │ (Mid-Tier LLM)  │
                             └────────┬────────┘
                                      │ DraftVerdictProposal
                                      ▼
                             ┌─────────────────┐
                             │ Verifier Gate   │  ◄── Pure Python / SQL
                             │ (Grounding DB)  │      (Zero Hallucination)
                             └────────┬────────┘
                                      │
                                      ▼
                             [ FINAL VERDICT ]
```

---

## 2. Database Architecture & Storage Schemas

The persistence layer uses **DuckDB** for structured/time-series data and **Chroma** for unstructured qualitative embeddings.

### 2.1 DuckDB Relational Schemas

```
┌─────────────────────────────────────────────────────────────┐
│                 canonical_products (Table 1)                │
│  canonical_id (PK) | asin | ean | mpn | brand | model       │
└──────────────────────────────┬──────────────────────────────┘
                               │ 1:N
        ┌──────────────────────┴──────────────────────┐
        ▼                                             ▼
┌──────────────────────────────┐ ┌──────────────────────────────┐
│    price_history (Table 2)   │ │  live_store_offers (Table 3) │
│  (Keepa Time-Series Archive) │ │  (Google Shopping Multi-Site)│
│  canonical_id | recorded_at  │ │  canonical_id | retailer     │
│  buy_box_price | amazon_price│ │  seller_name | base_price    │
│  marketplace_price           │ │  effective_price | condition │
└──────────────────────────────┘ └──────────────────────────────┘
```

#### Table 1: `canonical_products` (Master Catalog)
Stores canonical product metadata across all marketplaces, keyed by global barcode or ASIN.

| Column Name | Data Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `canonical_id` | `VARCHAR` | `PRIMARY KEY` | Global unique identifier (format: `EAN_<13digits>` or `ASIN_<id>`) |
| `asin` | `VARCHAR` | `INDEXED` | Amazon Standard Identification Number |
| `ean` | `VARCHAR` | `INDEXED` | Global 13-digit EAN/GTIN barcode |
| `upc` | `VARCHAR` | `NULLABLE` | Universal Product Code |
| `mpn` | `VARCHAR` | `INDEXED` | Manufacturer Part Number (e.g., `MYND3HN/A`) |
| `brand` | `VARCHAR` | `NOT NULL` | Brand name (e.g., `Apple`, `Lenovo`) |
| `model` | `VARCHAR` | `NOT NULL` | Model name (e.g., `iPhone 16 Pro`) |
| `title` | `VARCHAR` | `NOT NULL` | Canonical product title |
| `category` | `VARCHAR` | `NOT NULL` | Category hierarchy (e.g., `Electronics > Laptops`) |
| `created_at` | `TIMESTAMP` | `DEFAULT CURRENT_TIMESTAMP` | Ingestion timestamp |

#### Table 2: `price_history` (1 year Time-Series Records)
Stores timestamped historical prices extracted from Keepa minute-by-minute arrays.

| Column Name | Data Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `BIGINT` | `PRIMARY KEY` | Auto-increment sequence ID |
| `canonical_id` | `VARCHAR` | `FOREIGN KEY` | Reference to `canonical_products(canonical_id)` |
| `recorded_at` | `TIMESTAMP` | `NOT NULL, INDEXED` | Standard UTC timestamp (converted from Keepa minutes) |
| `buy_box_price` | `DOUBLE` | `NULLABLE` | Amazon Buy Box price in INR (`NULL` if out of stock) |
| `amazon_retail_price` | `DOUBLE` | `NULLABLE` | Direct Amazon retail price in INR |
| `marketplace_new_price`| `DOUBLE` | `NULLABLE` | Lowest 3rd-party New price |
| `used_price` | `DOUBLE` | `NULLABLE` | Lowest 3rd-party Used/Renewed price |
| `sales_rank` | `INTEGER` | `NULLABLE` | Amazon Best Sellers Rank at that timestamp |
| `is_out_of_stock` | `BOOLEAN` | `NOT NULL` | `TRUE` if item had no active offers |

#### Table 3: `live_store_offers` (Current Multi-Store Listings)
Stores real-time seller listings extracted from SerpApi Google Shopping.

| Column Name | Data Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `offer_id` | `VARCHAR` | `PRIMARY KEY` | Unique offer ID (format: `<canonical_id>_<store>_<seller_hash>`) |
| `canonical_id` | `VARCHAR` | `FOREIGN KEY` | Reference to `canonical_products(canonical_id)` |
| `retailer_name` | `VARCHAR` | `NOT NULL` | Store name (e.g., `Amazon.in`, `Flipkart`, `Croma`, `Reliance Digital`) |
| `seller_name` | `VARCHAR` | `NOT NULL` | Direct merchant (e.g., `Appario Retail`, `SuperComNet`, `Indiflash`) |
| `seller_rating` | `DOUBLE` | `NULLABLE` | Merchant rating out of 5.0 |
| `seller_review_count` | `INTEGER` | `NULLABLE` | Total customer reviews for this merchant |
| `base_price` | `DOUBLE` | `NOT NULL` | Listed product price in INR |
| `shipping_cost` | `DOUBLE` | `DEFAULT 0.0` | Delivery charge in INR |
| `condition` | `VARCHAR` | `NOT NULL` | `NEW`, `RENEWED`, `USED_LIKE_NEW`, `OPEN_BOX` |
| `is_authorized_seller`| `BOOLEAN` | `NOT NULL` | `TRUE` if retailer is an official brand-authorized partner |
| `active_bank_offer` | `VARCHAR` | `NULLABLE` | Promo description (e.g., *"₹5,000 instant discount on ICICI Cards"*) |
| `effective_price` | `DOUBLE` | `NOT NULL` | Net price after factoring instant bank promotions |
| `product_url` | `VARCHAR` | `NOT NULL` | Direct checkout or product listing link |
| `fetched_at` | `TIMESTAMP` | `DEFAULT CURRENT_TIMESTAMP` | Time this snapshot was captured |

---

### 2.2 Chroma Vector Database Collections (RAG Layer)

The RAG layer indexes two distinct corpora into local **Chroma DB** collections:

#### Collection 1: `price_behavior_corpus`
* **Purpose:** Stores pre-computed qualitative narrative summaries of seasonal discount patterns per product.
* **Chunk Unit:** 1 document per canonical product (approx. 200–300 words).
* **Metadata Schema:**
  * `canonical_id`: String
  * `brand`: String
  * `model`: String
  * `has_festive_drops`: Boolean
  * `typical_discount_bracket`: String (e.g., `"10-15%"`, `"25-30%"`)
* **Document Content Template:**
  > `Product: [Model] | Brand: [Brand] | ASIN: [ASIN]`  
  > `Historical Cadence: Historically drops price every 45 to 60 days. During major Indian festive sales (Amazon Great Indian Festival, Flipkart Big Billion Days), recorded maximum discounts reaching 18% below standard Buy Box. Rebounds to launch MSRP within 72 hours of initial flash sale stock depletion.`

#### Collection 2: `warranty_condition_corpus`
* **Purpose:** Stores policy rules, merchant integrity track records, and brand warranty terms.
* **Chunk Unit:** Section-level chunks (150–250 words per policy).
* **Metadata Schema:**
  * `brand`: String
  * `policy_category`: String (`"WARRANTY"`, `"SELLER_AUTHORIZATION"`, `"RENEWED_RISK"`)
  * `retailer_scope`: String (`"ALL"`, `"FLIPKART"`, `"AMAZON"`)
* **Document Content Template:**
  > `Brand: Apple India | Category: Warranty Coverage`  
  > `Policy: Full 1-year manufacturer warranty is valid only if purchased from Apple Authorised Resellers (including official stores on Amazon like Indiflash/Appario and Flipkart like SuperComNet). Open-box, grey-market imports, or unauthorized 3P sellers carry seller-only warranty, not recognized at official Apple Authorized Service Centers.`

---

## 3. External API Contracts & Ingestion Specifications

### 3.1 Keepa Product API Contract
* **Base URL:** `https://api.keepa.com/product`
* **Method:** `GET`
* **Query Parameters:**

| Parameter | Type | Required | Value |
| :--- | :--- | :---: | :--- |
| `key` | String | Yes | Keepa API Key |
| `domain` | Integer | Yes | `10` (Amazon India - `amazon.in`) |
| `asin` | String | Yes | Target ASIN (e.g., `B0DGJ6Q7LM`) |
| `history` | Integer | Yes | `1` (Full historical arrays) |
| `offers` | Integer | Yes | `20` (Returns active marketplace seller offers) |
| `stats` | Integer | Yes | `180` (180-day price statistics) |

* **Response JSON Schema (Selected Fields):**
```json
{
  "timestamp": "integer (current Keepa time)",
  "tokensLeft": "integer",
  "products": [
    {
      "asin": "string",
      "title": "string",
      "eanList": ["string"],
      "upcList": ["string"],
      "mpn": "string",
      "brand": "string",
      "stats": {
        "current": ["array of current prices across types"],
        "avg30": ["array of 30-day averages"],
        "min": ["array of all-time minimums"],
        "max": ["array of all-time maximums"]
      },
      "csv": [
        "csv[0]: Amazon retail price array [time, price, ...]",
        "csv[1]: Marketplace New price array",
        "csv[2]: Used price array",
        "csv[18]: Buy Box price array [time, price, ...]"
      ],
      "offers": [
        {
          "offerId": "integer",
          "sellerId": "string",
          "condition": "integer (1=New, 2=Like New, 6=Refurbished)",
          "price": "integer (in INR)",
          "shipping": "integer"
        }
      ]
    }
  ]
}
```

---

### 3.2 SerpApi Google Shopping API Contract
* **Base URL:** `https://serpapi.com/search.json`
* **Method:** `GET`
* **Query Parameters:**

| Parameter | Type | Required | Value |
| :--- | :--- | :---: | :--- |
| `engine` | String | Yes | `google_shopping` |
| `q` | String | Yes | Search query (Primary: `EAN`, Fallback: `MPN` or Title) |
| `google_domain`| String | Yes | `google.co.in` |
| `gl` | String | Yes | `in` (India) |
| `hl` | String | Yes | `en` |
| `api_key` | String | Yes | SerpApi Key |

* **Response JSON Schema (Selected Fields):**
```json
{
  "search_parameters": {
    "engine": "google_shopping",
    "q": "string"
  },
  "product_results": {
    "title": "string",
    "typical_price_range": ["string (low)", "string (high)"]
  },
  "sellers_results": {
    "online_sellers": [
      {
        "name": "string (Store Name, e.g., Flipkart, Croma)",
        "price": "string (e.g., '₹1,26,999')",
        "total_price": "string",
        "shipping": "string (e.g., 'Free delivery')",
        "condition": "string (e.g., 'New', 'Refurbished')",
        "seller_rating": "number",
        "offer": "string (e.g., '10% instant discount on Axis Bank')",
        "link": "string (URL)"
      }
    ]
  }
}
```

---

### 3.3 SerpApi Google Search API Contract
* **Base URL:** `https://serpapi.com/search.json`
* **Method:** `GET`
* **Query Parameters:**

| Parameter | Type | Required | Value |
| :--- | :--- | :---: | :--- |
| `engine` | String | Yes | `google` |
| `q` | String | Yes | Query (e.g., `"iPhone 16 Pro" HDFC ICICI bank offer India`) |
| `gl` | String | Yes | `in` |
| `api_key` | String | Yes | SerpApi Key |

* **Response JSON Schema (Selected Fields):**
```json
{
  "organic_results": [
    {
      "position": "integer",
      "title": "string",
      "link": "string (URL)",
      "snippet": "string",
      "date": "string"
    }
  ]
}
```

---

### 3.4 Product Entity Resolution & Matching Funnel

To ensure cross-retailer listings refer to the exact same physical product without code, the system adheres to this 3-tier deterministic decision tree:

```
[ Incoming Amazon Product ]
       │
       ▼
Tier 1: Global Barcode Check
  ├── Does Keepa provide EAN or GTIN?
  │     ├── YES ──► Query Google Shopping by exact EAN (q=<EAN>)
  │     │             └── Verify Google Shopping result barcode == Keepa EAN
  │     │                   └── MATCH CONFIRMED (100% Identity)
  │     └── NO  ──► Proceed to Tier 2
       ▼
Tier 2: Canonical Spec Fingerprint
  ├── Extract 4 Mandatory Attributes: Brand + Model + Storage/RAM + Color
  │     ├── Example: "apple" + "iphone_16_pro" + "128gb" + "natural_titanium"
  │     └── Check Candidate Offer Attributes:
  │           ├── Storage or Model Mismatch? ──► REJECT / DISQUALIFY
  │           └── All 4 Attributes Identical? ─► MATCH CONFIRMED
       ▼
Tier 3: Ambiguity Resolution
  └── If attributes cannot be cleanly extracted, flag as "AMBIGUOUS_REJECT"
      (Never compare ambiguous or missing specifications)
```

---

## 4. LangGraph Shared State Contract (`PriceLensState`)

The shared state passed through all nodes in the LangGraph DAG is defined as follows:

```json
{
  "asin": "string (Amazon identifier)",
  "canonical_id": "string (Primary EAN or ASIN)",
  "product_title": "string",
  "user_budget": "number (in INR)",
  "deadline_days": "integer (7, 30, or 90)",
  "evaluation_cutoff_date": "string or null (ISO-8601 for time-travel backtesting)",

  "ean": "string or null",
  "mpn": "string or null",
  "brand": "string",

  "history_report": {
    "type": "object (Conforms to HistoryAnalystReport schema)",
    "nullable": true
  },
  "market_report": {
    "type": "object (Conforms to MarketInvestigatorReport schema)",
    "nullable": true
  },
  "eligibility_report": {
    "type": "object (Conforms to EligibilityAnalystReport schema)",
    "nullable": true
  },

  "tool_call_count": "integer (Accumulates total tool invocations across agents)",
  "shared_evidence_pool": [
    {
      "source_type": "string ('DUCKDB_RECORD', 'SERP_SHOPPING_OFFER', 'WEB_SEARCH_URL')",
      "reference_id_or_url": "string",
      "fact_summary": "string",
      "verified": "boolean"
    }
  ],

  "draft_verdict": {
    "type": "object (Conforms to DraftVerdictProposal schema)",
    "nullable": true
  },
  "final_verified_verdict": {
    "type": "object (Conforms to FinalVerifiedVerdict schema)",
    "nullable": true
  },
  "verification_errors": ["array of string error messages"],
  "execution_status": "string ('RUNNING', 'SUCCESS', 'REJECTED_REFUSAL', 'REJECTED_UNVERIFIED')"
}
```

---

## 5. Agent Specifications, Tools, and Output Contracts

### 5.1 Agent 1: History Analyst

* **Role:** Analyzes price trends, discount cadence, volatility, and historical festive drops.
* **Model Tier:** Fast, cost-efficient small model (e.g., Gemini Flash).
* **Constraints:** ReAct loop capped at **maximum 6 tool calls**; final report $\le 200$ words.

#### Tool 1: `query_price_statistics`
* **Input Schema:**
  * `canonical_id` (string, required)
  * `window_days` (integer, default: 180)
* **Output Schema:**
  * `all_time_low` (number): Lowest recorded Buy Box price.
  * `current_price` (number): Latest recorded Buy Box price.
  * `average_price` (number): Mean price over window.
  * `price_percentile` (number): 0.0 (all-time low) to 1.0 (all-time high).
  * `days_since_last_drop` (integer): Days elapsed since a $>5\%$ drop.
  * `typical_drop_depth_pct` (number): Average depth of historical price drops.

#### Tool 2: `search_price_behavior_corpus`
* **Input Schema:**
  * `canonical_id` (string, required)
  * `query` (string, required): e.g., *"festive sale discount depth Diwali"*
* **Output Schema:**
  * `chunks` (array of objects): `[{"chunk_id": "string", "text": "string", "similarity_score": "number"}]`

#### Output Report Contract: `HistoryAnalystReport`
```json
{
  "all_time_low": "number",
  "current_price": "number",
  "price_percentile_180d": "number (0.0 to 1.0)",
  "typical_drop_depth_pct": "number",
  "days_since_last_drop": "integer",
  "drop_probability_within_deadline": "number (0.0 to 1.0)",
  "recommended_timing_stance": "string ('STRONGLY_WAIT', 'LEAN_WAIT', 'BUY_NOW')",
  "expected_target_price": "number or null",
  "evidence_chunk_ids": ["array of string chunk IDs"],
  "summary_rationale": "string (max 150 words)"
}
```

---

### 5.2 Agent 2: Market Investigator

* **Role:** Searches open web for off-graph real-world signals: festive sale dates, product lifecycle refresh, and active bank discounts.
* **Model Tier:** Fast, cost-efficient small model.
* **Constraints:** ReAct loop capped at **maximum 4 tool calls**; final report $\le 200$ words.

#### Tool 1: `google_market_search`
* **Input Schema:**
  * `query` (string, required): e.g., *"Amazon Great Indian Festival 2026 dates"*
  * `num_results` (integer, default: 5)
* **Output Schema:**
  * `results` (array of objects): `[{"title": "string", "snippet": "string", "link": "string"}]`

#### Tool 2: `check_bank_discount_offers`
* **Input Schema:**
  * `brand` (string, required)
  * `model` (string, required)
* **Output Schema:**
  * `promos` (array of objects): `[{"bank_name": "string", "discount_amount": "number", "card_type": "string", "source_url": "string"}]`

#### Output Report Contract: `MarketInvestigatorReport`
```json
{
  "upcoming_sales_detected": "boolean",
  "sale_name": "string or null",
  "estimated_sale_start_days": "integer or null",
  "successor_hardware_launch": "boolean",
  "hardware_launch_details": "string or null",
  "active_bank_promos": [
    {
      "bank_name": "string",
      "discount_rupees": "number",
      "applicable_card": "string",
      "source_url": "string"
    }
  ],
  "market_pressure_verdict": "string ('PRICES_LIKELY_FALLING', 'PRICES_STABLE', 'PRICES_LIKELY_RISING')",
  "evidence_urls": ["array of valid URLs"],
  "summary_rationale": "string (max 150 words)"
}
```

---

### 5.3 Agent 3: Eligibility Analyst

* **Role:** Filters, validates, and ranks current multi-store offers across Amazon, Flipkart, Croma, and Reliance Digital.
* **Model Tier:** Fast, cost-efficient small model.
* **Constraints:** ReAct loop capped at **maximum 4 tool calls**; final report $\le 200$ words.

#### Tool 1: `query_live_store_offers`
* **Input Schema:**
  * `canonical_id` (string, required)
* **Output Schema:**
  * `offers` (array of objects): All rows from DuckDB `live_store_offers` matching `canonical_id`.

#### Tool 2: `search_warranty_condition_policy`
* **Input Schema:**
  * `brand` (string, required)
  * `condition_type` (string, required): e.g., `"OPEN_BOX"` or `"RENEWED"`
* **Output Schema:**
  * `policy_summary` (string): Official manufacturer warranty standing.
  * `is_manufacturer_warranty_honored`: (boolean).

#### Deterministic Deal Scoring Formula
Before reporting, the agent calculates a composite score (0 to 100) for every offer:

$$\text{Score} = (0.40 \times S_{\text{price}}) + (0.30 \times S_{\text{seller}}) + (0.20 \times S_{\text{warranty}}) + (0.10 \times S_{\text{delivery}})$$

* **$S_{\text{price}}$:** $100 \times \left(1 - \frac{\text{Effective Price}}{\text{User Budget}}\right)$, clamped to $[0, 100]$.
* **$S_{\text{seller}}$:** $100$ if Brand-Authorized; otherwise $(\text{Rating} / 5.0) \times 70$.
* **$S_{\text{warranty}}$:** $100$ if Manufacturer Warranty; $30$ if Seller Warranty; $0$ if No Warranty.
* **$S_{\text{delivery}}$:** $100$ if Prime / 1-Day; $70$ if Standard; $0$ if $>7$ days.
* **Penalty:** $-50$ points deduction if Condition is Renewed/Open-Box and user did not explicitly request used.

#### Output Report Contract: `EligibilityAnalystReport`
```json
{
  "total_offers_evaluated": "integer",
  "cheapest_raw_offer": {
    "retailer": "string",
    "seller_name": "string",
    "condition": "string",
    "listed_price": "number",
    "effective_price": "number",
    "deal_score": "number",
    "url": "string"
  },
  "recommended_safe_offer": {
    "retailer": "string",
    "seller_name": "string",
    "condition": "string",
    "listed_price": "number",
    "effective_price": "number",
    "deal_score": "number",
    "is_authorized": "boolean",
    "warranty_valid": "boolean",
    "url": "string"
  },
  "cheapest_is_safe": "boolean",
  "safety_disqualification_notes": "string or null",
  "evidence_offer_ids": ["array of string offer IDs"]
}
```

---

### 5.4 Agent 4: Decision Synthesizer

* **Role:** Acts as central arbiter at the LangGraph **Hard Join Barrier**. Reconciles specialist reports, handles trade-offs, and drafts the final verdict.
* **Model Tier:** Mid-Tier Frontier Model (e.g., Claude Sonnet / Gemini Pro).
* **Constraints:** Single-pass synthesis (0 tool calls); input strictly limited to the 3 specialist reports.

#### Conflict Resolution Matrix

| Scenario | History Analyst | Market Investigator | Eligibility Analyst | Decision Synthesizer Output |
| :--- | :--- | :--- | :--- | :--- |
| **1. Impending Sale** | *"Current price is average."* | *"Diwali Sale announced in 12 days."* | *"Current offers carry zero bank discounts."* | **`WAIT`**<br>Target: Historical Festive Low |
| **2. Active Flash Deal** | *"Price at 6-month average."* | *"Prices stable."* | *"Authorized seller has instant ₹5,000 card offer today."* | **`BUY_NOW`**<br>Take the active bank promo |
| **3. Fake / Unsafe Low** | *"Price is at all-time low!"* | *"No external sales."* | *"Cheapest seller is unverified 3P renewed unit (3.1★)."* | **`WAIT`** or **`BUY_NOW` from 2nd-ranked safe seller** |
| **4. Zero History** | *"Product has 0 days of recorded history."* | Any | Any | **`REFUSE_NO_HISTORY`**<br>(Refuses to guess) |

#### Output Proposal Contract: `DraftVerdictProposal`
```json
{
  "decision": "string ('BUY_NOW', 'WAIT', 'REFUSE_NO_HISTORY')",
  "target_price": "number",
  "recommended_retailer": "string",
  "recommended_seller": "string",
  "condition": "string ('NEW', 'RENEWED', 'USED')",
  "confidence_score": "number (0.0 to 1.0)",
  "primary_rationale": "string",
  "key_evidence": [
    {
      "source_type": "string ('DUCKDB_RECORD', 'SERP_SHOPPING_OFFER', 'WEB_SEARCH_URL')",
      "reference_id_or_url": "string",
      "fact_summary": "string",
      "verified": "boolean"
    }
  ]
}
```

---

## 6. Deterministic Verifier Gate (Non-LLM Guardrail)

The Verifier Gate is a **pure Python/SQL deterministic validation rule engine**. It strictly checks the Decision Agent's proposal against the DuckDB database before releasing the answer.

### Verification Checklist:

```
                      [ DraftVerdictProposal ]
                                 │
                                 ▼
         ┌───────────────────────────────────────────────┐
         │ Check 1: Zero-History Refusal Check           │
         │ - If history < 18 months:                     │
         │   Must strictly equal "REFUSE_NO_HISTORY"     │
         └───────────────────────┬───────────────────────┘
                                 │ Pass
                                 ▼
         ┌───────────────────────────────────────────────┐
         │ Check 2: Database Price & Seller Grounding    │
         │ - SELECT base_price FROM live_store_offers    │
         │   WHERE retailer = recommended_retailer       │
         │ - Target price must equal base_price OR       │
         │   base_price - verified_bank_discount         │
         │ - Discrepancy > tolerance? ──► REJECT ANSWER  │
         └───────────────────────┬───────────────────────┘
                                 │ Pass
                                 ▼
         ┌───────────────────────────────────────────────┐
         │ Check 3: Evidence Lineage Integrity           │
         │ - DUCKDB_RECORD: ID must exist in table       │
         │ - WEB_SEARCH_URL: Must be valid HTTP format   │
         │ - Any invented / ungrounded ID? ──► REJECT    │
         └───────────────────────┬───────────────────────┘
                                 │ Pass
                                 ▼
                    [ FINAL VERIFIED VERDICT ]
```

---

## 7. End-to-End System Output Contract

This is the final Pydantic JSON structure returned to the user:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "PriceLensFinalVerdict",
  "type": "object",
  "properties": {
    "decision": {
      "type": "string",
      "enum": ["BUY_NOW", "WAIT", "REFUSE_NO_HISTORY"]
    },
    "target_price": {
      "type": "number",
      "description": "Recommended purchase price in INR"
    },
    "recommended_retailer": {
      "type": "string",
      "description": "e.g., 'Amazon.in', 'Flipkart', 'Croma'"
    },
    "recommended_seller": {
      "type": "string",
      "description": "e.g., 'SuperComNet', 'Indiflash'"
    },
    "condition": {
      "type": "string",
      "enum": ["NEW", "RENEWED", "USED"]
    },
    "confidence_score": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0
    },
    "effective_deal_score": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 100.0
    },
    "key_evidence": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "source_type": { "type": "string" },
          "reference_id_or_url": { "type": "string" },
          "fact_summary": { "type": "string" }
        },
        "required": ["source_type", "reference_id_or_url", "fact_summary"]
      }
    },
    "timing_and_safety_rationale": {
      "type": "string",
      "description": "Clear explanation covering price trend, deal safety, and sale proximity"
    }
  },
  "required": [
    "decision",
    "target_price",
    "recommended_retailer",
    "recommended_seller",
    "condition",
    "confidence_score",
    "key_evidence",
    "timing_and_safety_rationale"
  ]
}
```
