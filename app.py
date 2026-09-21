import streamlit as st
import pandas as pd
import plotly.express as px
from orchestrator import graph
from tools.analytics import get_db_connection

# Streamlit Page Config
st.set_page_config(page_title="PriceLens Advisor", page_icon="🔍", layout="wide")

def fetch_price_history(canonical_id):
    """Fetches the raw time-series data for the Plotly chart."""
    conn = get_db_connection()
    try:
        query = "SELECT recorded_date, price FROM price_history WHERE canonical_id = %s ORDER BY recorded_date ASC"
        df = pd.read_sql_query(query, conn, params=(canonical_id,))
        return df
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

# ---------------------------------------------------------
# UI Layout
# ---------------------------------------------------------
st.title("🔍 PriceLens: Autonomous Deal Advisor")
st.markdown("Enter an Amazon URL, ASIN, or product name below. The LangGraph Multi-Agent system will evaluate the historical trends and market conditions to give you a definitive **BUY NOW** or **WAIT** recommendation.")

# Sidebar Input
st.sidebar.header("Product Search")
user_input = st.sidebar.text_input(
    "Amazon URL, ASIN, or Name:", 
    value="B0CS5XW6TN",
    help="Try 'B0CS5XW6TN', 'iPhone 15', or paste an Amazon /dp/ link."
)
analyze_btn = st.sidebar.button("Analyze Deal", type="primary")

st.sidebar.markdown("---")
st.sidebar.markdown("**Agent Status**")
st.sidebar.markdown("✅ **Input Resolver:** Active")
st.sidebar.markdown("✅ **History Agent:** Active")
st.sidebar.markdown("⏳ **Market Agent:** Stubbed")
st.sidebar.markdown("⏳ **Eligibility Agent:** Stubbed")

# Execution Block
if analyze_btn and user_input:
    with st.status("🚀 Executing Multi-Agent DAG...", expanded=True) as status:
        # 1. Trigger the DAG with Streaming
        initial_state = {"query": user_input}
        result_state = initial_state.copy()
        
        try:
            for event in graph.stream(initial_state, {"recursion_limit": 15}):
                for node_name, node_state in event.items():
                    # Format node name (e.g. "history_agent" -> "History Agent")
                    formatted_name = node_name.replace("_", " ").title()
                    st.write(f"✅ **{formatted_name}** node completed.")
                    
                    # Merge state updates
                    result_state.update(node_state)
                    
                    if result_state.get("errors"):
                        break
                        
            errors = result_state.get("errors", [])
            if errors:
                status.update(label="Execution Failed", state="error", expanded=True)
                st.error(f"Error during input resolution: {errors[0]}")
                st.stop()
                
            status.update(label="Analysis Complete!", state="complete", expanded=False)
            
        except Exception as e:
            status.update(label="System Error Encountered", state="error", expanded=True)
            st.error(f"FATAL ERROR: {str(e)}")
            st.stop()
            
    canonical_id = result_state.get("canonical_id")
    product_title = result_state.get("product_title")
    
    # 2. Extract Agent Reports
    history_report = result_state.get("history_report", {})
    trend = history_report.get("trend", {})
    drops = history_report.get("drops", {})
    
    # --- Main Dashboard ---
    st.header(product_title)
    st.caption(f"ASIN: {canonical_id}")
    
    # Top Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price", f"₹{trend.get('current_price', 0):,.0f}")
    c2.metric("All-Time Low", f"₹{trend.get('true_atl', 0):,.0f}")
    c3.metric("30-Day Average", f"₹{trend.get('avg_30d', 0):,.0f}")
    c4.metric("DHI Score (0-100)", f"{trend.get('s_history', 0)} / 100")

    st.divider()

    # 4-Card Layout
    col1, col2 = st.columns([1, 1])
    
    # CARD 1: Final Verdict & Timing
    with col1:
        st.subheader("Card 1: Final Verdict & Timing")
        stance = trend.get("historical_stance", "UNKNOWN")
        
        with st.container(border=True):
            if stance == "BUY_NOW":
                st.success("### 🟢 BUY NOW (Steal Deal)")
                st.write("This product is near its all-time low. The History Agent has mathematically verified that this is an optimal time to purchase.")
            else:
                st.warning("### 🟡 WAIT (Price Inflated)")
                st.write(drops.get("rationale", "Price is currently inflated."))
                
                if "safe_target_price" in drops:
                    st.markdown(f"#### 🎯 Target Wait Price: **₹{drops['safe_target_price']:,.0f}**")
                    st.info(f"Expect a **{drops.get('expected_discount_pct', 0)}%** drop during the {drops.get('upcoming_sale')}.")
                    
        # Show the LLM's natural language analysis
        if "llm_analysis" in history_report:
            st.write("**🤖 History Agent Analysis (Gemini):**")
            st.info(history_report["llm_analysis"])
            
        # Log the minute details (Thought Process)
        if "agent_trace" in history_report:
            with st.expander("🔍 View LLM Thought Process & Tool Calls"):
                for log in history_report["agent_trace"]:
                    st.code(log, language="text")

    # CARD 2: Price Trajectory
    with col2:
        st.subheader("Card 2: Price Trajectory")
        with st.container(border=True):
            df = fetch_price_history(canonical_id)
            if not df.empty:
                fig = px.line(df, x="recorded_date", y="price")
                fig.update_layout(
                    margin=dict(l=20, r=20, t=20, b=20),
                    xaxis_title=None,
                    yaxis_title="Price (₹)",
                    showlegend=False
                )
                
                # Add current price and ATL lines
                curr_val = trend.get('current_price', 0)
                atl_val = trend.get('true_atl', 0)
                
                fig.add_hline(y=curr_val, line_dash="dot", line_color="red", annotation_text="Today")
                fig.add_hline(y=atl_val, line_dash="dash", line_color="green", annotation_text="ATL")
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No time-series data available to plot.")

    # Lower Cards (Stubs for future agents)
    st.divider()
    col3, col4 = st.columns([1, 1])
    
    with col3:
        st.subheader("Card 3: Smart Arbitrage")
        with st.container(border=True):
            st.info("⏳ Market & Arbitrage Agent (Pending)")
            st.write("Will display live Flipkart vs Croma vs Amazon prices, plus Dubai/US travel arbitrage.")
        
    with col4:
        st.subheader("Card 4: Review Intelligence")
        with st.container(border=True):
            st.info("⏳ Eligibility Agent (Pending)")
            st.write("Will display ChromaDB warranty policies and AI defect summaries (e.g., heating, battery life, screen lines).")