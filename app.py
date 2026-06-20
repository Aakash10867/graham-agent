"""
GRAHAM INVESTMENT AGENT — Web App
==================================
Streamlit version of week8_full_agent.py
Run locally:  streamlit run app.py
Deploy free:  Streamlit Cloud (see instructions below)
"""

import streamlit as st
from google import genai
from google.genai import types
import chromadb
import pymupdf
import yfinance as yf
import json

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Graham Investment Agent",
    page_icon="📈",
    layout="centered"
)

st.title("📈 Graham Investment Agent")
st.caption("Ask about stocks, investing, or Benjamin Graham's principles. Powered by *The Intelligent Investor*.")

# New Chat button — clears conversation memory
if st.button("🔄 New Chat"):
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.rerun()

# ──────────────────────────────────────────────
# LOAD BOOK INTO CHROMADB (runs once, cached)
# ──────────────────────────────────────────────
@st.cache_resource
def load_book():
    doc = pymupdf.open("The Intelligent Investor.pdf")  # put the PDF in the same folder as app.py
    full_text = "\n".join(page.get_text() for page in doc)
    doc.close()

    # Paragraph chunking (Week 6 lesson)
    raw_paragraphs = full_text.split("\n\n")
    chunks = []
    current = ""
    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) < 1500:
            current = current + "\n\n" + para if current else para
        else:
            if len(current) >= 100:
                chunks.append(current)
            current = para
    if current and len(current) >= 100:
        chunks.append(current)

    # Store in ChromaDB
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection("graham_book")
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        collection.add(
            documents=batch,
            ids=[f"chunk_{j}" for j in range(i, i + len(batch))]
        )
    return collection

collection = load_book()

# ──────────────────────────────────────────────
# TOOLS (same as week8_full_agent.py)
# ──────────────────────────────────────────────

def search_book(query: str) -> dict:
    """Search The Intelligent Investor for passages relevant to a query.
    Use this when the user asks about investing concepts, Graham's advice,
    value investing principles, margin of safety, or anything the book covers.

    Args:
        query: What to search for in the book, e.g. "margin of safety" or "defensive investor criteria"
    """
    stop_words = {"what", "does", "the", "a", "an", "is", "are", "how", "why",
                  "when", "where", "about", "for", "and", "or", "of", "in",
                  "to", "on", "with", "say", "says", "said", "his", "her",
                  "do", "did", "can", "should", "would", "it", "its", "that",
                  "this", "by", "from", "was", "were", "be", "been", "has", "have"}

    # Semantic search
    sem_results = collection.query(query_texts=[query], n_results=5)
    sem_docs = sem_results["documents"][0]
    sem_ids = sem_results["ids"][0]
    sem_dists = sem_results["distances"][0]

    # Keyword search
    keywords = [w for w in query.lower().split() if w not in stop_words and len(w) > 2]
    keyword_docs = []
    keyword_ids = []
    for kw in keywords[:3]:
        try:
            kw_results = collection.get(where_document={"$contains": kw}, limit=3)
            for doc, doc_id in zip(kw_results["documents"], kw_results["ids"]):
                if doc_id not in keyword_ids and doc_id not in sem_ids:
                    keyword_docs.append(doc)
                    keyword_ids.append(doc_id)
        except Exception:
            pass

    formatted = []
    for i, (text, dist) in enumerate(zip(sem_docs, sem_dists)):
        formatted.append(f"[Semantic match {i+1}, relevance={1-dist:.2f}]:\n{text}")
    for i, text in enumerate(keyword_docs[:3]):
        formatted.append(f"[Keyword match {i+1}]:\n{text}")

    return {"passages": "\n\n".join(formatted)}


def get_stock_data(ticker: str) -> dict:
    """Get real financial data for a stock using its ticker symbol.
    Use this when the user asks about a specific company's financials,
    P/E ratio, earnings, dividend yield, book value, or any fundamental data.

    For Indian stocks on NSE, append .NS (e.g., TCS.NS, INFY.NS, RELIANCE.NS).
    For US stocks, use plain ticker (e.g., AAPL for Apple, MSFT for Microsoft).
    For Mahindra, use MAHINDRA (the function resolves it automatically).

    Args:
        ticker: Stock ticker symbol, e.g. AAPL, MSFT, TCS.NS, MAHINDRA, RELIANCE.NS
    """
    # Resolve common aliases (Gemini struggles with & in tickers)
    TICKER_ALIASES = {
        "MAHINDRA": "M&M.NS",
        "MAHINDRA.NS": "M&M.NS",
        "M_M.NS": "M&M.NS",
        "MM.NS": "M&M.NS",
        "MNM.NS": "M&M.NS",
        "M&M": "M&M.NS",
        "L&T": "LT.NS",
        "L&T.NS": "LT.NS",
        "L_T.NS": "LT.NS",
    }
    resolved = TICKER_ALIASES.get(ticker.upper(), ticker)

    try:
        stock = yf.Ticker(resolved)
        info = stock.info

        if not info or info.get("regularMarketPrice") is None:
            return {"error": f"No data found for '{ticker}' (tried '{resolved}'). Check the ticker. Indian stocks need .NS suffix (e.g., TCS.NS, INFY.NS)."}

        return {
            "symbol": info.get("symbol"),
            "name": info.get("longName") or info.get("shortName"),
            "sector": info.get("sector"),
            "currency": info.get("currency"),
            "current_price": info.get("regularMarketPrice") or info.get("currentPrice"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "price_to_book": info.get("priceToBook"),
            "book_value": info.get("bookValue"),
            "eps": info.get("trailingEps"),
            "dividend_yield": info.get("dividendYield"),
            "profit_margin": info.get("profitMargins"),
            "return_on_equity": info.get("returnOnEquity"),
            "revenue": info.get("totalRevenue"),
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
        }
    except Exception as e:
        return {"error": f"Failed to fetch data for '{ticker}': {str(e)}"}


def calculator(expression: str) -> dict:
    """Evaluate a math expression. Use for any calculation:
    ratios, percentages, comparisons, margin of safety computations, etc.

    Args:
        expression: A Python math expression, e.g. "45000 / 1200" or "(52.3 - 41.8) / 52.3 * 100"
    """
    try:
        result = eval(expression)
        return {"expression": expression, "result": round(result, 4)}
    except Exception as e:
        return {"error": f"Could not evaluate '{expression}': {str(e)}"}


tool_functions = {
    "search_book": search_book,
    "get_stock_data": get_stock_data,
    "calculator": calculator,
}

# ──────────────────────────────────────────────
# SYSTEM PROMPT & CONFIG (no persistent client)
# ──────────────────────────────────────────────

SYSTEM_INSTRUCTION = """You are an investment analysis assistant grounded in Benjamin Graham's principles.

You have three tools:
1. search_book — searches The Intelligent Investor for relevant passages. USE THIS when the user asks about investing concepts, Graham's philosophy, or wants book-based advice.
2. get_stock_data — pulls real financial data for a stock ticker. USE THIS when the user asks about specific companies or wants fundamental data.
3. calculator — evaluates math expressions. USE THIS for any computation.

RULES:
- When you use search_book, base your answer on the retrieved passages. If the passages don't contain the answer, say so honestly.
- When analyzing a stock, connect the data back to Graham's principles when relevant.
- Be concise and direct. No filler.
- If the user asks something outside investing/finance, just answer normally without using tools.
- Remember the full conversation — the user may refer to earlier questions."""

TOOLS = [search_book, get_stock_data, calculator]


def agent_turn(user_message):
    """Create a fresh client every turn — no stale connections."""

    # Fresh client each time (Streamlit kills persistent ones between reruns)
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

    # Rebuild chat with stored history for memory across turns
    history = st.session_state.get("chat_history", [])

    chat = client.chats.create(
        model="gemini-2.5-flash-lite",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            tools=TOOLS,
        ),
        history=history,
    )

    response = chat.send_message(user_message)

    while response.function_calls:
        function_responses = []
        for fc in response.function_calls:
            if fc.name in tool_functions:
                result = tool_functions[fc.name](**fc.args)
            else:
                result = {"error": f"Unknown tool: {fc.name}"}
            function_responses.append(
                types.Part.from_function_response(name=fc.name, response=result)
            )
        response = chat.send_message(function_responses)

    # Save conversation history for next turn
    st.session_state.chat_history = chat.get_history()

    return response.text

# ──────────────────────────────────────────────
# CHAT UI
# ──────────────────────────────────────────────

# Initialize message history for display and chat history for Gemini
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new input
if prompt := st.chat_input("Ask about a stock, Graham's principles, or anything..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = agent_turn(prompt)
            except Exception as e:
                answer = f"Something went wrong: {str(e)}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})