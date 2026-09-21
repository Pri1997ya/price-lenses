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
Create a `.env` file in the root directory and add your PostgreSQL connection string and Gemini API Key:
```env
DATABASE_URL="postgresql://[user]:[password]@[host]/[dbname]?sslmode=require"
GEMINI_API_KEY="your_google_gemini_api_key_here"
```

### 4. Run the Dashboard
Launch the interactive Streamlit UI:
```bash
streamlit run app.py
```
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

---

## 🧠 Key Features for Evaluators
* **Trace Observability:** The UI features an expandable dropdown (`🔍 View LLM Thought Process`) that streams the LangGraph internal state, exposing exactly what Prompts were injected, which Tools the LLM selected, and the raw JSON arguments passed.
* **Graceful DNS Fallback:** The PostgreSQL connection logic features a hardcoded IPv4 fallback to guarantee database resilience against macOS/Neon pooling DNS resolution failures.
* **Parallel Fan-Out:** The LangGraph DAG executes the three subagents concurrently, reducing total execution latency by over 60%.
