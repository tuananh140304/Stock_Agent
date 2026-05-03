import re
import pandas as pd
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from Stock_Agent_Official import initialize_messages, get_agent_response
import plotly.graph_objects as go

company_logo = "images/Stock_image.jpg"

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="StockSense – Personal Stock Advisor",
    layout="centered"
)

# -----------------------------
# STYLING
# -----------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #eef2f3 0%, #dfe9f3 100%);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.image(company_logo, width=150)
st.sidebar.title("📊 StockSense")
st.sidebar.write("Your beginner friendly stock advisor.")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "- Stock explanations\n"
    "- Trend analysis\n"
    "- Real-time prices\n"
    "- Simple definitions\n"
    "- Price charts"
)
st.sidebar.markdown("---")
st.sidebar.caption("📚 Powered by SEC Investor Guides, yfinance & Finnhub")

# -----------------------------
# HEADER
# -----------------------------
st.image(company_logo)
st.title("📈 StockSense – Personal Stock Advisor")
st.write("Ask about any stock, company, or financial concept.")

# -----------------------------
# SESSION STATE
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages      = initialize_messages()
if "chart_history" not in st.session_state:
    st.session_state.chart_history = {}  # {message_index: {price_history, ticker, period_label}}

# -----------------------------
# CHART RENDERER
# -----------------------------
def render_price_chart(price_history: list, ticker: str = "", period_label: str = ""):
    """
    Renders a price chart from a list of {"date": ..., "price": ...} dicts.
    Chart title reflects the actual period the user asked for.
    """
    if not price_history:
        return

    try:
        df = pd.DataFrame(price_history)
        df["date"] = pd.to_datetime(df["date"])

        min_price = df["price"].min()
        max_price = df["price"].max()
        y_min = min_price - 10

        parts = []
        if ticker:
            parts.append(ticker)
        if period_label:
            parts.append(period_label.title())
        title = " — ".join(parts) + " Price Chart" if parts else "Price Chart"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["price"],
            mode="lines",
            name=ticker or "Price",
            line=dict(color="#1f77b4", width=2)
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            yaxis=dict(range=[y_min, max_price + 5]),
            xaxis=dict(showgrid=False),
            height=400,
            margin=dict(l=40, r=20, t=50, b=40)
        )

        with st.expander(f"📊 {title}", expanded=True):
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"Source: Yahoo Finance via yfinance | "
                f"Showing {period_label} of price history | "
                f"Data refreshes every 24 hours"
            )

    except Exception as e:
        st.warning(f"Chart could not be rendered: {e}")


# -----------------------------
# DISPLAY CHAT HISTORY
# Re-renders messages and charts on page refresh
# -----------------------------
for i, msg in enumerate(st.session_state.messages):
    if isinstance(msg, HumanMessage):
        st.chat_message("user", avatar="👤").write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant", avatar="📈"):
            st.write(msg.content)
            if i in st.session_state.chart_history:
                chart_data = st.session_state.chart_history[i]
                render_price_chart(
                    chart_data["price_history"],
                    chart_data.get("ticker", ""),
                    chart_data.get("period_label", "")
                )

# -----------------------------
# CHAT INPUT
# -----------------------------
user_input = st.chat_input("Ask about a stock or financial concept...")

if user_input:
    st.chat_message("user", avatar="👤").write(user_input)

    with st.spinner("Analyzing..."):
        # get_agent_response now returns 5 values
        response, updated_messages, price_history, period_label, ticker = get_agent_response(
            st.session_state.messages,
            user_input
        )

    st.session_state.messages = updated_messages

    # Strip any leaked debug lines
    clean_response = "\n".join(
        line for line in response.split("\n")
        if not line.strip().startswith("Context:")
    ).strip()

    with st.chat_message("assistant", avatar="📈"):
        if clean_response:
            st.write(clean_response)
        else:
            st.write("Sorry, I couldn't generate a response. Please try again.")

        # Render chart if stock history data was returned
        if price_history:
            #ticker comes directly from the tool result — always accurate

            render_price_chart(price_history, ticker, period_label)

            # Store chart so it re-renders on page refresh
            ai_msg_index = len(st.session_state.messages) - 1
            st.session_state.chart_history[ai_msg_index] = {
                "price_history": price_history,
                "ticker":        ticker or "",
                "period_label":  period_label
            }