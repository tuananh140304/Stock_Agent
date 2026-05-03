import os
import json
import logging
import requests
import chromadb
import yfinance as yf
from datetime import datetime, timedelta

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage

import streamlit as st
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["FINNHUB_API_KEY"] = st.secrets["FINNHUB_API_KEY"]

# -----------------------------
# LOGGING
# -----------------------------
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------
# CONFIG
# -----------------------------
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
MODEL_LLM       = "gpt-4o-mini"
CHROMA_PATH     = "chroma_db"
COLLECTION      = "stock_sense"

if not FINNHUB_API_KEY:
    raise EnvironmentError("FINNHUB_API_KEY is not set in your .env file.")

# LLM
llm = ChatOpenAI(
    model=MODEL_LLM,
    temperature=0.8,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# -----------------------------
# CHROMADB CONNECTION
# -----------------------------
try:
    _chroma_client     = chromadb.PersistentClient(path=CHROMA_PATH)
    _chroma_collection = _chroma_client.get_collection(name=COLLECTION)
    _embeddings        = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    logger.info(f"ChromaDB connected — {_chroma_collection.count()} chunks available")
except Exception as e:
    logger.warning(f"ChromaDB connection failed: {e}. query_knowledge_base will be unavailable.")
    _chroma_collection = None
    _embeddings        = None


# -----------------------------
# IN-MEMORY CACHE
# Cache key includes the period so different time ranges are cached separately.
# e.g. "overview:AAPL:180" and "overview:AAPL:365" are stored independently.
#
# Cache durations:
#   overview  — 24 hours  (historical data doesn't change intraday)
#   price     — 15 minutes (prices move constantly during market hours)
#   news      — 1 hour    (headlines refresh throughout the day)
# -----------------------------
_cache = {}

CACHE_DURATIONS = {
    "overview": timedelta(hours=24),
    "price":    timedelta(minutes=15),
    "news":     timedelta(hours=1),
}

def _get_cache(cache_type: str, ticker: str, period_days: int = 180):
    """Returns cached data if it exists and hasn't expired, else None."""
    key   = f"{cache_type}:{ticker.upper()}:{period_days}"
    entry = _cache.get(key)
    if not entry:
        return None
    age = datetime.now() - entry["timestamp"]
    if age > CACHE_DURATIONS[cache_type]:
        logger.debug(f"Cache expired for {key} (age: {age})")
        return None
    logger.debug(f"Cache hit for {key} (age: {age})")
    return entry["data"]


def _set_cache(cache_type: str, ticker: str, data: dict, period_days: int = 180):
    """Stores data in the cache with the current timestamp."""
    key         = f"{cache_type}:{ticker.upper()}:{period_days}"
    _cache[key] = {"data": data, "timestamp": datetime.now()}
    logger.debug(f"Cache set for {key}")


# -----------------------------
# PERIOD PARSER
# Converts natural language time expressions into a number of days.
# The agent passes whatever the user said (e.g. "last year", "3 months",
# "past quarter") and this function normalises it to days.
# Falls back to 180 days (6 months) if nothing is recognised.
# -----------------------------
def parse_period_to_days(period_str: str) -> tuple[int, str]:
    """
    Converts a natural language period string into (days, human_label).

    Examples:
        "1 year"      → (365, "1 year")
        "2 years"     → (730, "2 years")
        "3 months"    → (90,  "3 months")
        "last quarter"→ (90,  "last quarter")
        "ytd"         → (days since Jan 1, "year to date")
        "all time"    → (3650, "10 years")
        "6 months"    → (180, "6 months")   ← default

    Returns (days, label) so both the API call and the chart title are accurate.
    """
    p = period_str.lower().strip()

    # --- Year to date ---
    if any(x in p for x in ["ytd", "year to date", "this year"]):
        days  = (datetime.today() - datetime(datetime.today().year, 1, 1)).days
        return days, "year to date"

    # --- All time / max ---
    if any(x in p for x in ["all time", "all-time", "max", "maximum", "since ipo"]):
        return 3650, "10 years (max available)"

    # --- Quarters ---
    if any(x in p for x in ["quarter", "3 month", "three month"]):
        return 90, "3 months"

    # --- Weeks ---
    import re
    week_match = re.search(r'(\d+)\s*week', p)
    if week_match:
        weeks = int(week_match.group(1))
        return weeks * 7, f"{weeks} week{'s' if weeks > 1 else ''}"

    # --- Months ---
    month_match = re.search(r'(\d+)\s*month', p)
    if month_match:
        months = int(month_match.group(1))
        days   = months * 30
        return days, f"{months} month{'s' if months > 1 else ''}"

    # --- Years ---
    year_match = re.search(r'(\d+)\s*year', p)
    if year_match:
        years = int(year_match.group(1))
        days  = years * 365
        return days, f"{years} year{'s' if years > 1 else ''}"

    # --- Named periods ---
    if "1 month" in p or "one month" in p or "30 day" in p:
        return 30, "1 month"
    if "5 year" in p or "five year" in p:
        return 1825, "5 years"
    if "10 year" in p or "ten year" in p or "decade" in p:
        return 3650, "10 years"

    # --- Default: 6 months ---
    return 180, "6 months"


# -----------------------------
# SYSTEM PROMPT
# -----------------------------
SYSTEM_PROMPT = """
You are StockSense, an AI Stock Advisor designed to help beginner investors understand companies,
stock performance, financial terminology, and market trends. Your purpose is to make investing feel
simple, approachable, and non-intimidating — without giving financial advice.

🎯 Identity & Voice
You speak as a friendly, patient, and knowledgeable financial guide.
Your tone is beginner friendly, clear, and conversational.
You avoid jargon unless the user asks for deeper detail.
You automatically define financial terms in simple language.
You never assume prior financial knowledge.

📊 Core Responsibilities
When responding about a stock, ALWAYS structure your response in these five sections:

  1. 🏢 Company Overview     — What the company does, its sector, and why people follow it
  2. 📊 Performance Summary  — Key numbers: current price, start price, % change
  3. 📈 Trend Interpretation  — Plain-English classification of the trend (upward, stable, declining)
  4. 💡 Beginner Insights     — 1-2 observations a new investor should know, written without jargon
  5. ⚠️  Disclaimer           — Always end with: "This is for educational purposes only and is not financial advice."

🛠️ Tool Usage Rules
- Use get_stock_overview for company background and historical price data.
  IMPORTANT: Always pass the period the user asked for. Examples:
    • "How has AAPL done over the last year?"    → period="1 year"
    • "Show me Tesla's 3-month performance"      → period="3 months"
    • "What has NVDA done since the start of the year?" → period="ytd"
    • If no period is mentioned, default to period="6 months"
- Use get_realtime_price_and_news for current price and recent news headlines
- Use analyze_trend ONLY after you have real price data from the above tools — never with guessed numbers
- Use query_knowledge_base whenever:
    • The user asks what a financial term means (e.g. "what is a P/E ratio?")
    • You need to explain a concept in beginner-friendly language
    • The user asks how something works (e.g. "how do ETFs work?")
    • You want to add educational context to a stock response

- - Use get_market_news_by_date when the user asks WHY a stock moved, dipped, or 
  spiked on or around a specific date. Extract the date from the user's message 
  and pass it in YYYY-MM-DD format. Never claim you lack access to recent events 
  — always use this tool first.
  
- - Use get_general_market_news when the user asks about general financial news,
  top market stories, or what's happening in the markets — with no specific 
  ticker mentioned. Never claim you lack access to current news — always use 
  this tool first.

📚 Source Citation Rules — IMPORTANT
When you use information from query_knowledge_base, you MUST cite the source naturally
in your response. Use this format:

    "According to the [document title], [explanation in your own words]."

Examples:
    "According to the SEC guide on Stocks, a stock represents a share of ownership in a company."
    "The SEC Investor Guide on Diversification explains that spreading investments across
     different assets helps reduce risk."

Never present RAG content as your own knowledge — always attribute it to its source.
If the data came from a cached result, mention it naturally:
    "Based on the most recently available data, [ticker] is trading at $[price]."

🧠 Behavioral Guidelines
If data is unavailable:
- Explain what analysis you *would* run, what data is needed, and why it matters

If asked for predictions:
- Explain uncertainty and provide historical context instead

If asked for recommendations:
- Never say "buy" or "sell"
- Describe factors beginners often consider without directing action

🚫 Out-of-Scope Fallback
If asked about taxes, options trading, portfolio allocation, retirement planning, or legal advice,
respond with: "That's outside what I can help with — I focus on helping beginners understand
stock data and company information. For [topic], I'd recommend speaking with a licensed financial advisor."

🚫 Hard Limits — Never:
- Give financial advice or predict future prices
- Tell users what to buy or sell
- Invent data, prices, or news headlines
"""


# -----------------------------
# TOOLS
# -----------------------------

@tool
def get_stock_overview(ticker: str, period: str = "6 months") -> dict:
    """
    Use this tool when the user asks about a company's background, what it does,
    its sector, market cap, or historical stock performance over any time period.

    Examples:
        "How has AAPL done over the last year?"         → ticker="AAPL", period="1 year"
        "Show me Tesla's 3-month chart"                 → ticker="TSLA", period="3 months"
        "What has NVDA done since the start of the year?" → ticker="NVDA", period="ytd"
        "Tell me about Amazon"                          → ticker="AMZN", period="6 months"

    DO NOT use this tool for real-time prices — use get_realtime_price_and_news instead.

    Input:
        ticker — stock ticker symbol (e.g. 'AAPL', 'NVDA', 'AMZN')
        period — natural language time period (e.g. '3 months', '1 year', 'ytd', '2 years')
                 Defaults to '6 months' if not specified.
    Output:
        Company overview + historical price data for the requested period
    """
    logger.debug(f"Tool called: get_stock_overview | ticker={ticker} | period={period}")
    ticker = ticker.upper()

    # Convert natural language period to days + a clean label
    period_days, period_label = parse_period_to_days(period)

    # --- Check cache (keyed by ticker + period_days) ---
    cached = _get_cache("overview", ticker, period_days)
    if cached:
        cached["_cache_note"] = f"Data retrieved from cache (refreshes every 24 hours)."
        return cached

    try:
        stock = yf.Ticker(ticker)
        info  = stock.info

        company_name = info.get("longName", "N/A")
        sector       = info.get("sector", "N/A")
        industry     = info.get("industry", "N/A")
        description  = info.get("longBusinessSummary", "No description available.")
        market_cap   = info.get("marketCap", None)
        dividend     = info.get("dividendYield", None)

        if market_cap:
            if market_cap >= 1_000_000_000_000:
                market_cap_str = f"${market_cap / 1_000_000_000_000:.2f}T"
            elif market_cap >= 1_000_000_000:
                market_cap_str = f"${market_cap / 1_000_000_000:.2f}B"
            else:
                market_cap_str = f"${market_cap / 1_000_000:.2f}M"
        else:
            market_cap_str = "N/A"

        dividend_str = f"{dividend * 100:.2f}%" if dividend else "N/A"

        # Fetch history for the requested period
        end     = datetime.today()
        start   = end - timedelta(days=period_days)
        history = stock.history(start=start, end=end)

        if history.empty:
            historical_summary = f"Historical price data unavailable for the requested period ({period_label})."
        else:
            start_price = round(float(history["Close"].iloc[0]), 2)
            end_price   = round(float(history["Close"].iloc[-1]), 2)
            pct_change  = round(((end_price - start_price) / start_price) * 100, 2)

            trend = (
                "upward 📈"       if pct_change > 5  else
                "downward 📉"     if pct_change < -5 else
                "mostly stable ➖"
            )

            # Full price history list for the chart
            price_history = [
                {
                    "date":  str(date.date()),
                    "price": round(float(price), 2)
                }
                for date, price in history["Close"].items()
            ]

            historical_summary = {
                "start_price":   start_price,
                "end_price":     end_price,
                "pct_change":    pct_change,
                "trend":         trend,
                "period":        period_label,
                "period_days":   period_days,
                "price_history": price_history
            }

        result = {
            "ticker": ticker,
            "company_name":    company_name,
            "sector":          sector,
            "industry":        industry,
            "description":     description[:300] + "..." if len(description) > 300 else description,
            "market_cap":      market_cap_str,
            "dividend_yield":  dividend_str,
            "historical_data": historical_summary,
            "_cache_note":     None
        }

        _set_cache("overview", ticker, result, period_days)
        return result

    except Exception as e:
        logger.error(f"get_stock_overview failed for '{ticker}': {e}")
        stale = _cache.get(f"overview:{ticker}:{period_days}")
        if stale:
            logger.warning(f"Returning stale cache for {ticker} overview")
            stale["data"]["_cache_note"] = (
                "⚠️ Live data unavailable — showing last known data. Please try again later."
            )
            return stale["data"]
        return {"error": f"Could not retrieve data for ticker '{ticker}': {str(e)}"}


@tool
def get_realtime_price_and_news(ticker: str) -> dict:
    """
    Use this tool when the user asks for a stock's current/real-time price,
    today's market update, or recent news about a company
    (e.g. 'What is Tesla's price right now?' or 'Any recent news about MSFT?').
    DO NOT use this tool for historical trends — use get_stock_overview instead.

    Input:  A stock ticker symbol (e.g. 'TSLA', 'MSFT', 'GOOGL')
    Output: Current stock price + up to 5 recent news headlines with summaries
    """
    logger.debug(f"Tool called: get_realtime_price_and_news | ticker={ticker}")
    ticker = ticker.upper()
    result = {}

    # --- Real-Time Price (cache: 15 minutes) ---
    cached_price = _get_cache("price", ticker)
    if cached_price:
        result["price"]       = cached_price
        result["_price_note"] = "Based on the most recently available data (refreshes every 15 minutes)."
    else:
        try:
            quote_url  = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={FINNHUB_API_KEY}"
            quote_data = requests.get(quote_url).json()

            current_price = quote_data.get("c")
            open_price    = quote_data.get("o")
            high_price    = quote_data.get("h")
            low_price     = quote_data.get("l")
            prev_close    = quote_data.get("pc")

            if current_price:
                day_change     = round(current_price - prev_close, 2) if prev_close else None
                day_change_pct = round((day_change / prev_close) * 100, 2) if prev_close and day_change else None

                price_data = {
                    "current":        round(current_price, 2),
                    "open":           round(open_price, 2) if open_price else "N/A",
                    "high":           round(high_price, 2) if high_price else "N/A",
                    "low":            round(low_price, 2)  if low_price  else "N/A",
                    "previous_close": round(prev_close, 2) if prev_close else "N/A",
                    "day_change":     day_change,
                    "day_change_pct": day_change_pct
                }
                result["price"]       = price_data
                result["_price_note"] = None
                _set_cache("price", ticker, price_data)
            else:
                result["price"] = "Real-time price unavailable."

        except Exception as e:
            logger.error(f"Price fetch error for '{ticker}': {e}")
            stale = _cache.get(f"price:{ticker}:180")
            if stale:
                result["price"]       = stale["data"]
                result["_price_note"] = "⚠️ Live price unavailable — showing last known price."
            else:
                result["price"] = f"Price fetch failed: {str(e)}"

    # --- Recent News (cache: 1 hour) ---
    cached_news = _get_cache("news", ticker)
    if cached_news:
        result["news"]       = cached_news
        result["_news_note"] = "News refreshes every hour."
    else:
        try:
            today    = datetime.today().strftime("%Y-%m-%d")
            week_ago = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")
            news_url = (
                f"https://finnhub.io/api/v1/company-news"
                f"?symbol={ticker}&from={week_ago}&to={today}&token={FINNHUB_API_KEY}"
            )
            news_data = requests.get(news_url).json()

            if isinstance(news_data, list) and len(news_data) > 0:
                news = [
                    {
                        "headline": a.get("headline", "No headline"),
                        "source":   a.get("source", "Unknown"),
                        "summary":  a.get("summary", "No summary available.")[:200]
                    }
                    for a in news_data[:5]
                ]
                result["news"]       = news
                result["_news_note"] = None
                _set_cache("news", ticker, news)
            else:
                result["news"] = "No recent news found for this ticker."

        except Exception as e:
            logger.error(f"News fetch error for '{ticker}': {e}")
            stale = _cache.get(f"news:{ticker}:180")
            if stale:
                result["news"]       = stale["data"]
                result["_news_note"] = "⚠️ Live news unavailable — showing last known headlines."
            else:
                result["news"] = f"News fetch failed: {str(e)}"

    return result


@tool
def analyze_trend(start_price: float, end_price: float, period_label: str) -> dict:
    """
    Use this tool to calculate percentage return and classify the trend direction
    for a stock over a given period.
    ONLY call this tool AFTER you have already retrieved price data from
    get_stock_overview or get_realtime_price_and_news.
    DO NOT call this tool with estimated or assumed numbers.

    Input:  start_price (float), end_price (float), period_label (str, e.g. '1 year')
    Output: % return, trend classification, and a beginner-friendly interpretation
    """
    logger.debug(f"Tool called: analyze_trend | {start_price} -> {end_price} over {period_label}")

    if start_price <= 0:
        return {"error": "start_price must be greater than 0."}

    pct_change    = round(((end_price - start_price) / start_price) * 100, 2)
    dollar_change = round(end_price - start_price, 2)

    if pct_change > 10:
        trend         = "strongly upward 📈"
        beginner_note = (
            f"The stock grew by {pct_change}% over {period_label}. "
            "That's a strong positive move. Keep in mind past performance doesn't guarantee future growth."
        )
    elif pct_change > 5:
        trend         = "upward 📈"
        beginner_note = (
            f"The stock is up {pct_change}% over {period_label}. "
            "A moderate positive trend — it has generally been moving in a good direction."
        )
    elif pct_change < -10:
        trend         = "strongly downward 📉"
        beginner_note = (
            f"The stock dropped {abs(pct_change)}% over {period_label}. "
            "A significant decline — could reflect company challenges or broader market conditions."
        )
    elif pct_change < -5:
        trend         = "downward 📉"
        beginner_note = (
            f"The stock is down {abs(pct_change)}% over {period_label}. "
            "A moderate decline — not necessarily alarming, but worth understanding why."
        )
    else:
        trend         = "mostly stable ➖"
        beginner_note = (
            f"The stock moved only {pct_change}% over {period_label}. "
            "Relatively flat — it hasn't grown much but hasn't lost much either."
        )

    return {
        "start_price":   start_price,
        "end_price":     end_price,
        "dollar_change": dollar_change,
        "pct_change":    pct_change,
        "trend":         trend,
        "period":        period_label,
        "beginner_note": beginner_note
    }


@tool
def query_knowledge_base(query: str) -> str:
    """
    Use this tool to look up definitions, explanations, and educational content
    from the SEC investor guides and financial glossary stored in the knowledge base.

    Use this tool when:
    - The user asks what a financial term means (e.g. 'what is a P/E ratio?', 'what is beta?')
    - The user asks how something works (e.g. 'how do ETFs work?', 'what is diversification?')
    - You want to add beginner-friendly context after showing stock data
    - The user asks general investing questions not covered by live data tools

    Input:  A natural language question or financial term
    Output: Relevant educational content with source document title and page number for citation
    """
    logger.debug(f"Tool called: query_knowledge_base | query={query}")

    if _chroma_collection is None or _embeddings is None:
        return "Knowledge base is currently unavailable. Please ensure ChromaDB is set up correctly."

    try:
        query_vector = _embeddings.embed_query(query)

        results = _chroma_collection.query(
            query_embeddings=[query_vector],
            n_results=4,
            include=["documents", "metadatas"]
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        if not documents:
            return "No relevant information found in the knowledge base for that query."

        formatted = []
        for doc, meta in zip(documents, metadatas):
            source   = meta.get("title", "SEC Investor Guide")
            page     = meta.get("page", "?")
            citation = f"the SEC guide on '{source}' (page {page})"
            formatted.append(f"[Source: {citation}]\n{doc}")

        return "\n\n---\n\n".join(formatted)

    except Exception as e:
        logger.error(f"query_knowledge_base failed: {e}")
        return f"Knowledge base query failed: {str(e)}"

@tool
def get_market_news_by_date(ticker: str, date: str) -> dict:
    """
    Use this tool when the user asks WHY a stock moved, dipped, or spiked
    on or around a specific date. It fetches company-specific news from
    Finnhub for a 5-day window around the date provided.

    Examples:
        "Why did AAPL dip on April 21?"     → ticker="AAPL", date="2025-04-21"
        "What happened to TSLA last week?"  → ticker="TSLA", date="2025-04-15"

    Input:  ticker (str), date (str in YYYY-MM-DD format)
    Output: News headlines and summaries around that date
    """
    logger.debug(f"Tool called: get_market_news_by_date | ticker={ticker} | date={date}")
    ticker = ticker.upper()

    try:
        target   = datetime.strptime(date, "%Y-%m-%d")
        date_from = (target - timedelta(days=3)).strftime("%Y-%m-%d")
        date_to   = (target + timedelta(days=2)).strftime("%Y-%m-%d")

        url  = (
            f"https://finnhub.io/api/v1/company-news"
            f"?symbol={ticker}&from={date_from}&to={date_to}&token={FINNHUB_API_KEY}"
        )
        data = requests.get(url).json()

        if isinstance(data, list) and len(data) > 0:
            articles = [
                {
                    "headline": a.get("headline", "No headline"),
                    "source":   a.get("source",   "Unknown"),
                    "date":     datetime.fromtimestamp(a.get("datetime", 0)).strftime("%Y-%m-%d"),
                    "summary":  a.get("summary",  "No summary available.")[:300]
                }
                for a in data[:8]
            ]
            return {
                "ticker":   ticker,
                "period":   f"{date_from} to {date_to}",
                "articles": articles
            }
        else:
            return {"message": f"No news found for {ticker} around {date}."}

    except Exception as e:
        logger.error(f"get_market_news_by_date failed: {e}")
        return {"error": str(e)}

@tool
def get_general_market_news() -> dict:
    """
    Use this tool when the user asks about general financial news, market trends,
    or top stories — without mentioning a specific stock or ticker.

    Examples:
        "What are the biggest financial news stories right now?"
        "What's happening in the markets today?"
        "What are the top financial stories of 2026?"

    Input:  None
    Output: Latest general market news headlines from Finnhub
    """
    logger.debug("Tool called: get_general_market_news")

    try:
        url  = f"https://finnhub.io/api/v1/news?category=general&token={FINNHUB_API_KEY}"
        data = requests.get(url).json()

        if isinstance(data, list) and len(data) > 0:
            articles = [
                {
                    "headline": a.get("headline", "No headline"),
                    "source":   a.get("source",   "Unknown"),
                    "date":     datetime.fromtimestamp(a.get("datetime", 0)).strftime("%Y-%m-%d"),
                    "summary":  a.get("summary",  "No summary available.")[:300]
                }
                for a in data[:10]
            ]
            return {"articles": articles}
        else:
            return {"message": "No general market news available right now."}

    except Exception as e:
        logger.error(f"get_general_market_news failed: {e}")
        return {"error": str(e)}

# -----------------------------
# AGENT SETUP
# -----------------------------
tools = [
    get_stock_overview,
    get_realtime_price_and_news,
    analyze_trend,
    query_knowledge_base,
    get_market_news_by_date,
    get_general_market_news
]

agent_executor = create_agent(llm, tools, system_prompt=SYSTEM_PROMPT)


# -----------------------------
# CONVERSATION STATE
# -----------------------------
def initialize_messages():
    """Returns a fresh empty chat history."""
    return []


# -----------------------------
# MAIN RESPONSE FUNCTION
# -----------------------------
def get_agent_response(chat_history: list, user_input: str):
    """
    Sends user_input to the agent along with the full chat_history.
    Returns (assistant_message, updated_chat_history, price_history).
    price_history is passed to app.py for chart rendering — None if no stock data was retrieved.
    """
    logger.debug(f"get_agent_response called | input={repr(user_input)}")

    try:
        result = agent_executor.invoke(
            {"messages": chat_history + [HumanMessage(content=user_input)]}
        )

        assistant_message = result["messages"][-1].content

        # Extract price_history from tool result messages for chart rendering
        price_history = None
        period_label  = "6 months"
        ticker = None
        for msg in result.get("messages", []):
            content = getattr(msg, "content", "")
            if isinstance(content, str) and "price_history" in content:
                try:
                    parsed = json.loads(content)
                    hist   = parsed.get("historical_data", {})
                    if isinstance(hist, dict) and "price_history" in hist:
                        price_history = hist["price_history"]
                        period_label  = hist.get("period", "6 months")
                        break
                except Exception:
                    pass

        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=assistant_message))

        return assistant_message, chat_history, price_history, period_label, ticker

    except Exception as e:
        logger.error(f"get_agent_response failed: {e}")
        error_message = f"Sorry, I ran into an issue: {str(e)}"
        chat_history.append(AIMessage(content=error_message))
        return error_message, chat_history, None, "6 months", None