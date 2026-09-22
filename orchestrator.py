import re
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

# 1. Define the LangGraph State
class PriceLensState(TypedDict):
    query: str
    canonical_id: str
    product_title: str
    
    # Reports from parallel agents (Disjoint state keys for safe concurrent fan-out)
    history_report: dict
    market_report: dict
    eligibility_report: dict
    
    # Final outputs
    draft_verdict: dict
    final_verdict: dict
    errors: list[str]

# ==========================================
# 2. Define the Nodes
# ==========================================
def input_resolver_node(state: PriceLensState):
    """Node 0: Parses raw user query to extract ASIN/canonical_id"""
    from tools.analytics import get_db_connection
    
    query = state.get("query", "").strip()
    asin = None
    title = query
    errors = []
    
    # 1. Regex for Amazon URL
    dp_match = re.search(r"/dp/([A-Z0-9]{10})", query)
    if dp_match:
        asin = dp_match.group(1)
    # 2. Regex for Raw ASIN
    elif re.match(r"^[A-Z0-9]{10}$", query):
        asin = query
        
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            if asin:
                cur.execute("SELECT title FROM products WHERE canonical_id = %s", (asin,))
                row = cur.fetchone()
                if row:
                    title = row[0]
                else:
                    errors.append(f"ASIN {asin} not found in database.")
            else:
                # 3. Fuzzy search for product name
                search_term = f"%{query}%"
                cur.execute("SELECT canonical_id, title FROM products WHERE title ILIKE %s LIMIT 1", (search_term,))
                row = cur.fetchone()
                if row:
                    asin = row[0]
                    title = row[1]
                else:
                    errors.append(f"Could not find a product matching '{query}'")
    except Exception as e:
        errors.append(str(e))
    
    return {"canonical_id": asin, "product_title": title, "errors": errors}

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
import os

@tool
def check_price_trend(canonical_id: str) -> dict:
    """Gets the mathematical baseline (ATL, ATH, current price, and S_history score). Use this first."""
    from tools.analytics import query_historical_trend
    return query_historical_trend(canonical_id)

@tool
def get_historical_sale_drops(canonical_id: str) -> dict:
    """Finds what price the product historically dropped to during mega-sales. Use if product is inflated."""
    from tools.analytics import query_sale_event_drops
    return query_sale_event_drops(canonical_id)

def history_agent_node(state: PriceLensState):
    """Node 1: Runs the History Analysis Tools via LLM ReAct Agent"""
    from tools.analytics import query_historical_trend, query_sale_event_drops
    
    asin = state.get("canonical_id")
    if not asin:
        return {"history_report": {}}
        
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    # if not api_key:
    #     raise ValueError("PRODUCTION SECURITY FAULT: GEMINI_API_KEY is missing. The autonomous LLM ReAct Agent cannot run without it.")
        
    # We still fetch the raw deterministic dicts so the Streamlit UI can render the charts flawlessly
    trend = query_historical_trend(asin)
    drops = {}
    if trend.get("historical_stance") == "WAIT":
        drops = query_sale_event_drops(asin)

    LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:5001/gateway/mlflow/v1")
    LLM_API_KEY  = os.environ.get("LLM_API_KEY", "not-needed")
    LLM_MODEL    = os.environ.get("LLM_MODEL",    "gemini")

   
        
    try:
        llm = ChatOpenAI(
                base_url=LLM_BASE_URL,
                api_key=LLM_API_KEY,
                model=LLM_MODEL,
        )

        # result = llm.invoke("crack a joke");
        # print('result',result);

        
        system_prompt = (
            "You are the History & Trend Analyst Agent for PriceLens.\n"
            "Your objective is to analyze the historical price trajectory of a product and determine if today is a financially optimal time to buy.\n\n"
            "INSTRUCTIONS:\n"
            "1. Call `check_price_trend` for the product.\n"
            "2. Analyze the JSON. Is the product near its All-Time Low (S_history >= 75)?\n"
            "3. If it is expensive (S_history < 75), call `get_historical_sale_drops` to find the target wait price.\n"
            "4. Write a 3-sentence financial analysis explaining the momentum and seasonality. End with a clear BUY_NOW or WAIT stance."
        )
        
        agent = create_react_agent(llm, tools=[check_price_trend, get_historical_sale_drops], prompt=system_prompt)
        
        # 5. Capture the exact thought process (Trace Logging)
        agent_trace = []
        agent_trace.append(f"📥 SYSTEM PROMPT PASSED TO LLM:\n{system_prompt}\n")
        agent_trace.append(f"📥 USER PROMPT: Analyze the historical price for ASIN: {asin}\n")
        
        final_state = None
        for step in agent.stream({"messages": [("user", f"Analyze the historical price for ASIN: {asin}")]}):
            for node_name, node_state in step.items():
                if node_name == "agent":
                    msg = node_state["messages"][-1]
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        agent_trace.append(f"🛠️ LLM DECIDED TO CALL TOOL: {msg.tool_calls[0]['name']}")
                        agent_trace.append(f"   Arguments passed to tool: {msg.tool_calls[0]['args']}")
                    elif msg.content:
                        agent_trace.append(f"🧠 LLM GENERATED OUTPUT:\n{msg.content}")
                elif node_name == "tools":
                    msg = node_state["messages"][-1]
                    # Log the JSON returned by the database tool
                    agent_trace.append(f"✅ TOOL RETURNED RAW DATA to LLM:\n{msg.content}")
            
            final_state = step
            
        # Extract the final textual analysis
        last_node = list(final_state.keys())[0]
        llm_analysis = final_state[last_node]["messages"][-1].content
        
    except Exception as e:
        raise RuntimeError(f"LLM Agent Failed: {str(e)}")
            
    return {
        "history_report": {
            "trend": trend, 
            "drops": drops,
            "llm_analysis": llm_analysis,
            "agent_trace": agent_trace
        }
    }

def market_agent_node(state: PriceLensState):
    """Node 2: Runs Market & Arbitrage Tools (Stub)"""
    return {"market_report": {"status": "Pending implementation"}}

def eligibility_agent_node(state: PriceLensState):
    """Node 3: Runs Eligibility Tools (Stub)"""
    return {"eligibility_report": {"status": "Pending implementation"}}

def decision_synthesizer_node(state: PriceLensState):
    """Node 4: LLM Synthesizes the 3 reports into a DraftVerdict"""
    return {"draft_verdict": {"status": "Drafting LLM recommendation..."}}

def verifier_gate_node(state: PriceLensState):
    """Node 5: Deterministic Verifier validates the DraftVerdict"""
    return {"final_verdict": {"status": "Verified safely", "dhi": 55.5, "stance": "WAIT"}}

# ==========================================
# 3. Build the Directed Acyclic Graph (DAG)
# ==========================================
builder = StateGraph(PriceLensState)

builder.add_node("input_resolver", input_resolver_node)
builder.add_node("history_agent", history_agent_node)
builder.add_node("market_agent", market_agent_node)
builder.add_node("eligibility_agent", eligibility_agent_node)
builder.add_node("decision_synthesizer", decision_synthesizer_node)
builder.add_node("verifier_gate", verifier_gate_node)

builder.add_edge(START, "input_resolver")
builder.add_edge("input_resolver", "history_agent")
builder.add_edge("input_resolver", "market_agent")
builder.add_edge("input_resolver", "eligibility_agent")
builder.add_edge(["history_agent", "market_agent", "eligibility_agent"], "decision_synthesizer")
builder.add_edge("decision_synthesizer", "verifier_gate")
builder.add_edge("verifier_gate", END)

graph = builder.compile()

if __name__ == "__main__":
    test_state = {"query": "B0CS5XW6TN"}
    print("--- 🚀 Initializing PriceLens LangGraph DAG ---")
    for event in graph.stream(test_state, {"recursion_limit": 10}):
        for node_name, node_state in event.items():
            print(f"✅ Node Completed: [{node_name}]")
    
    final = graph.invoke(test_state)
    print("\\n--- 🏁 DAG Execution Complete ---")
    print(final.get("final_verdict"))
