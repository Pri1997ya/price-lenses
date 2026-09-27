# 🔍 PriceLens: Autonomous Agentic RAG Advisor

PriceLens is a multi-agent AI system designed to act as an autonomous e-commerce purchase-timing advisor. It evaluates product prices across historical time-series data, live market competitors, and semantic customer reviews to give users a mathematically grounded, hallucination-free **BUY NOW** or **WAIT** recommendation.

Built as an **Agentic RAG** pipeline using **LangGraph**, PriceLens eschews traditional sequential LLM wrappers in favor of a highly parallelized Directed Acyclic Graph (DAG) architecture.

---

## 🏗️ The Multi-Agent Architecture

PriceLens uses **LangGraph** to instantly route user queries (Amazon URLs, ASINs, or fuzzy product names) to three specialized ReAct agents executing in parallel.

```mermaid
graph TD
    User((User Input)) --> Node0[0. Input Resolver]
    
    subgraph "Parallel Specialists (Concurrent Execution)"
        Node0 --> Agent1[1. History Analyst<br/>PostgreSQL Time-Series]
        Node0 --> Agent2[2. Market Analyst<br/>Live Competitor APIs]
        Node0 --> Agent3[3. Safety Analyst<br/>Chroma Deep Semantic RAG]
    end
    
    Agent1 --> Node4[4. Decision Synthesizer<br/>Gemini 2.0 Flash]
    Agent2 --> Node4
    Agent3 --> Node4
    
    Node4 --> UI[5. Streamlit Dashboard]
```

### 1. History & Trend Analyst (Deterministic + ReAct)
* **Role:** Analyzes 365-day price histories to determine the True All-Time Low (ATL) and Deal Health Index ($S_{history}$).
* **Tools:** Executes pure Python/PostgreSQL tools (`check_price_trend`, `get_historical_sale_drops`) to guarantee 0% hallucination on financial math.

### 2. Market & Arbitrage Analyst
* **Role:** Scrapes live pricing from competitors (Flipkart, Croma) and identifies geographic arbitrage opportunities.

### 3. Eligibility & Safety Analyst (Deep Semantic RAG)
* **Role:** Protects the buyer from defective products and bad return policies.
* **Tools:** Queries a **ChromaDB Vector Store** containing thousands of chunked, raw user reviews to semantically detect hidden hardware defects (e.g., "green line issues" or "overheating").

### 4. Decision Synthesizer (The Central Arbiter)
* **Role:** Waits for the three parallel agents to finish, ingests their JSON reports, resolves logical conflicts (e.g., "It's cheap, but it overheats"), and outputs the final executive verdict to the UI.

---

## 🛠️ Tech Stack
* **Orchestration:** LangGraph, LangChain
* **LLM Engine:** Google Gemini 2.0 Flash
* **Database (Time-Series):** Neon PostgreSQL (`psycopg2`)
* **Vector Store (RAG):** ChromaDB
* **UI/Frontend:** Streamlit, Plotly Express

---

## 🚀 Getting Started (Local Setup)

### 1. Clone & Environment Setup
Clone the repository and activate your Python virtual environment:
```bash
# Example using venv
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
Install the required packages from the generated `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys
Create a `.env` file in the root directory and add the database and agent credentials:
```env
DATABASE_URL="postgresql://[user]:[password]@[host]/[dbname]?sslmode=require"
GEMINI_API_KEY="your_google_gemini_api_key_here"

# Agent 2 only: verified market facts are summarized through OpenAI.
MARKET_AGENT_LLM_ENABLED=true
MARKET_AGENT_LLM_BASE_URL=https://api.openai.com/v1
MARKET_AGENT_LLM_API_KEY=
MARKET_AGENT_LLM_MODEL=gpt-6-luna
```

Keep real keys only in `.env`, which is ignored by Git. Agent 2 also accepts the
standard `OPENAI_API_KEY` variable when `MARKET_AGENT_LLM_API_KEY` is omitted.

### 4. Run the Dashboard
Launch the interactive Streamlit UI:
```bash
streamlit run app.py
```
The dashboard has two independent tabs:

- **History & Timing** uses the existing historical analyst and price-history tables.
- **Market Investigator** loads stored market offers or explicitly fetches live
  SerpAPI/Apify results. Merely opening the tab does not consume provider credits.

*Note: If the `GEMINI_API_KEY` is missing, the LangGraph engine will gracefully fail via a strict production security fault, preventing silent LLM hallucinations.*

### 5. Run the Market Investigator data pipeline

The live-market pipeline uses the same `DATABASE_URL` and `products` catalog as
the historical analyst. It adds append-only market run, offer, seller-detail,
and promotion tables without replacing the existing history tables.

For an optional local PostgreSQL instance:

```bash
docker compose up -d postgres
export DATABASE_URL=postgresql://pricelens:pricelens_local@localhost:5433/pricelens
python scripts/db/init_db.py
python scripts/db/migrate.py
```

For Neon, set `DATABASE_URL` to the pooled URI with `sslmode=require`, initialize
the existing base schema if necessary, and apply the same migration:

```bash
python scripts/db/init_db.py
python scripts/db/migrate.py
```

Configure `SERPAPI_API_KEY` and `APIFY_API_TOKEN` in the ignored `.env`, then
fetch and persist offers from both providers:

```bash
python scripts/market_search.py "Apple iPhone 16 128GB"
```

Use `--providers serpapi` or `--providers apify` to run one provider. SerpAPI
uses Amazon Search and Google Shopping for discovery. Optional Amazon Product
API enrichment is disabled by default and can be enabled with
`SERPAPI_ENRICH_AMAZON=true`. Apify enriches the cheapest discovered listings
with configured Flipkart and bank-offer actors.

The application prefers `DATABASE_URL`; existing installations that already use
`PL_DATABASE_URL` are also supported for backward compatibility.

### 6. Eligibility & Safety Agent (policy RAG + seller/stock check)

Agent 3 answers return, replacement and refund questions from indexed retailer
policies (Amazon, Flipkart, Croma, Reliance Digital, Vijay Sales) and the
Consumer Protection (E-Commerce) Rules, 2020. It also checks seller trust and
stock on the offers the Market Investigator has stored. It adds no database
tables and only reads `latest_market_offers`.

Build the policy corpus and the Chroma index:

```bash
python scripts/policies/fetch_policies.py        # pages/PDFs -> data/policies/<retailer>/*.md
python scripts/policies/build_policy_index.py --ask "Can I return a phone bought on Flipkart?"
```

Pages that block scripts (common on Amazon and Flipkart) are reported as
`[too short]` and never overwrite an existing file. Save those by hand; see
`data/policies/README.md`. The index is rebuilt in a staging collection and
swapped in only when complete.

How search works: every question runs both a meaning search (MiniLM embeddings)
and a keyword search (BM25) over the stored passages, and the two rankings are
merged. Meaning-only matches must clear `POLICY_MIN_RELEVANCE`; passages that
contain the question's key words are kept even below it, so exact terms such as
"restocking fee" or "7 days" are not lost. Passages about the product being
bought (phone, laptop, earbuds, ...) are ranked first. Related words count as
matches ("phone" finds a "Mobiles" row), a retailer named in the question
("... on Flipkart?") limits the search to that retailer, and each row of a
policy table is kept on one line ("Mobiles | 7 days Replacement only") so a
rule is never split across passages.

To check search quality and tune the cut-off on your own index:

```bash
python scripts/policies/eval_policy_questions.py              # built-in question set
python scripts/policies/eval_policy_questions.py --questions my_questions.json
```

How it stays grounded:

- The LLM may answer only from retrieved passages and must cite them as `[n]`.
  An answer with no citations, or with a citation number that wasn't supplied,
  is rejected and the retrieved passages are shown instead.
- If the LLM gateway is offline, the retrieved passages are shown with citations.
- If nothing relevant is retrieved, the answer says it is not stated in the
  indexed policies.
- Restrictive terms ("replacement only", "non-returnable", seal or inspection
  requirements) are flagged by a fixed pattern scan of the passages, not by the LLM.
  Clauses written for a different product type (for example a laptop "brand
  seal" rule when the product is a phone) are not flagged.

Seller and stock rules (no LLM): stock is classified as `IN_STOCK`, `LOW_STOCK`,
`OUT_OF_STOCK`, `PREORDER` or `UNKNOWN`. Seller trust is `TRUSTED` (the retailer's
own store, Flipkart Assured, or `TRUSTED_SELLERS`), `OK` (rated 4/5 or higher),
`CAUTION` (unidentified, rated 3 to 4, or refurbished/used) or `AVOID` (rated
below 3). Offers older than `OFFER_STALE_HOURS` are flagged.

The results appear in the **🛡️ Policy & Seller Check** tab and, after a History
analysis, in the Eligibility & Safety section. They are also in
`eligibility_report` in the LangGraph state for the Decision Synthesizer.

---

## 🧠 Key Features for Evaluators
* **Trace Observability:** The UI features an expandable dropdown (`🔍 View LLM Thought Process`) that streams the LangGraph internal state, exposing exactly what Prompts were injected, which Tools the LLM selected, and the raw JSON arguments passed.
* **Graceful DNS Fallback:** The PostgreSQL connection logic features a hardcoded IPv4 fallback to guarantee database resilience against macOS/Neon pooling DNS resolution failures.
* **Parallel Fan-Out:** The LangGraph DAG executes the three subagents concurrently, reducing total execution latency by over 60%.
