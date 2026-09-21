# 📊 Agent 1: The History & Trend Analyst

## 1. Overview
The **History & Trend Analyst** is the foundational time-series agent of the PriceLens multi-agent system. Its primary objective is to evaluate a product's price momentum over the last 365 days and determine if today is a mathematically optimal time to buy.

Unlike traditional LLMs that guess or hallucinate prices, this agent operates on a **Deterministic-First Agentic RAG** architecture. The LLM handles the *reasoning*, but relies entirely on pure PostgreSQL SQL queries to do the *math*.

---

## 2. Core Responsibilities
* **Establish Mathematical Truth:** Calculate the True All-Time Low (ATL), All-Time High (ATH), and 30-Day Moving Average.
* **Calculate Deal Health ($S_{history}$):** Generate a deterministic 0-100 score based on where today's price sits in the historical spectrum.
* **Project Target Prices:** If a product is inflated, scan past mega-sales (e.g., Big Billion Days, Prime Day) to predict a highly accurate target price for the user to wait for.
* **Provide Financial Context:** Output a human-readable, nuanced executive summary explaining seasonality and momentum.

---

## 3. The ReAct Architecture (LangGraph)
The agent is implemented in `orchestrator.py` as a **LangChain ReAct (Reason + Act) Agent** powered by Gemini 2.0 Flash.

### The System Prompt
The LLM is initialized with a strict persona:
> *"You are the History & Trend Analyst Agent for PriceLens. Your objective is to analyze the historical price trajectory of a product... Write a 3-sentence financial analysis explaining the momentum and seasonality. End with a clear BUY_NOW or WAIT stance."*

### The Agentic Loop
1. **Receive Task:** The LangGraph orchestrator passes the `canonical_id` (e.g., `B0CS5XW6TN`) to the agent.
2. **First Tool Call:** Gemini autonomously calls `check_price_trend`.
3. **Reasoning:** Gemini reads the returned JSON. If $S_{history} \ge 75$, it stops and issues a `BUY_NOW`. If it is $< 75$, it proceeds to step 4.
4. **Second Tool Call:** Gemini autonomously calls `get_historical_sale_drops` to find a mathematically safe target price.
5. **Synthesis:** Gemini writes the final human-readable `llm_analysis`.

---

## 4. The Deterministic Tools (`tools/analytics.py`)

To prevent hallucinations, the agent relies on two strict Python/SQL tools.

### Tool 1: `query_historical_trend(canonical_id)`
**Purpose:** Generates the baseline metrics from the `price_history` table.
* **$S_{history}$ Formula:** 
  `100.0 * (1.0 - (current_price - true_atl) / (true_ath - true_atl))`
* **Percentile Rank:** Computes exactly how many days out of the last 365 were more expensive than today.
* **Output Stance:** Automatically flags `"WAIT"` if the score drops below 75.

### Tool 2: `query_sale_event_drops(canonical_id)`
**Purpose:** Finds a repeatable, data-backed target price for buyers told to WAIT.
* **Historical Scan:** Runs an SQL filter for months `1, 7, 9, 10` (Republic Day, Prime Day, BBD, Diwali) to find the absolute minimum price the product dropped to during a past mega-sale.
* **Sales Calendar Fallback:** If the product is too new to have festive history, it cross-references the `sales_calendar` table for the next upcoming sale, fetches the `typical_category_discount_pct` (e.g., 18%), and applies it to the 30-Day Moving Average.
* **Safety Clamp:** Enforces `MAX(estimated_drop, true_atl)` to guarantee the tool never predicts an impossibly low target price.

---

## 5. Output Contract
The History Agent finishes its execution by returning a structured dictionary (`history_report`) to the LangGraph state. This report is subsequently passed to the **Decision Synthesizer (Node 4)** and the **Streamlit UI**.

```json
{
  "history_report": {
    "trend": {
      "true_atl": 71999.0,
      "s_history": 0.0,
      "historical_stance": "WAIT"
    },
    "drops": {
      "safe_target_price": 71999.0,
      "expected_discount_pct": 30.77,
      "upcoming_sale": "Big Billion Days & Great Indian Festival"
    },
    "llm_analysis": "Based on the time-series analysis, the current price of ₹1,03,999 is highly inflated, sitting at its all-time high... WAIT for the Big Billion Days sale.",
    "agent_trace": [
       "📥 SYSTEM PROMPT PASSED TO LLM...",
       "🛠️ LLM DECIDED TO CALL TOOL: check_price_trend..."
    ]
  }
}
```

---

## 6. Engineering Highlights: Graceful Degradation
The History Agent is designed with **Production Security Fault** tolerance. 
In the event that the LLM API (Gemini) goes down, or the API key is missing, the agent does not crash. It gracefully degrades by bypassing the ReAct LLM loop entirely and populating the `trend` and `drops` JSON using pure Python execution. The UI continues to function perfectly using the deterministic math, proving the resilience of the architecture.
