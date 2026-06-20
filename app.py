"""
GRAHAM INVESTMENT AGENT — Web App (Visual Overhaul)
====================================================
Same logic as original. Radically different skin.
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

# ──────────────────────────────────────────────
# CUSTOM CSS — the entire visual overhaul
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* ═══════════════════════════════════════════════
   ANIMATED AURORA GRADIENT BACKGROUND
   ═══════════════════════════════════════════════ */
@keyframes aurora {
    0%   { background-position: 0% 50%; }
    25%  { background-position: 50% 100%; }
    50%  { background-position: 100% 50%; }
    75%  { background-position: 50% 0%; }
    100% { background-position: 0% 50%; }
}

/* Root app background */
.stApp {
    background: linear-gradient(
        -45deg,
        #0f0c29,
        #1a1a40,
        #302b63,
        #24243e,
        #0f0c29,
        #1b1145,
        #0d2137,
        #0a1628
    );
    background-size: 400% 400%;
    animation: aurora 20s ease infinite;
}

/* ═══════════════════════════════════════════════
   FLOATING PARTICLE OVERLAY (pure CSS)
   ═══════════════════════════════════════════════ */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        radial-gradient(2px 2px at 20% 30%, rgba(120, 200, 255, 0.3), transparent),
        radial-gradient(2px 2px at 40% 70%, rgba(200, 120, 255, 0.2), transparent),
        radial-gradient(1px 1px at 60% 20%, rgba(255, 200, 100, 0.3), transparent),
        radial-gradient(2px 2px at 80% 50%, rgba(100, 255, 200, 0.2), transparent),
        radial-gradient(1px 1px at 10% 80%, rgba(255, 150, 200, 0.25), transparent),
        radial-gradient(1px 1px at 70% 90%, rgba(150, 200, 255, 0.2), transparent),
        radial-gradient(2px 2px at 50% 10%, rgba(200, 255, 150, 0.15), transparent),
        radial-gradient(1px 1px at 90% 15%, rgba(255, 100, 200, 0.2), transparent);
    pointer-events: none;
    z-index: 0;
    animation: aurora 30s ease infinite reverse;
    background-size: 300% 300%;
}

/* ═══════════════════════════════════════════════
   TYPOGRAPHY & TITLE — NEON GLOW
   ═══════════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

.stApp, .stApp * {
    font-family: 'Inter', sans-serif !important;
}

/* Main title glow */
.stApp h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 2.6rem !important;
    background: linear-gradient(135deg, #00f5d4, #7bf1a8, #fee440, #f15bb5, #9b5de5);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: aurora 6s ease infinite;
    text-shadow: none;
    filter: drop-shadow(0 0 30px rgba(0, 245, 212, 0.3));
    padding-bottom: 4px;
}

/* Caption */
.stApp .stCaption, .stApp [data-testid="stCaptionContainer"] p {
    color: rgba(200, 210, 230, 0.6) !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.3px;
}

/* ═══════════════════════════════════════════════
   GLASSMORPHISM CHAT BUBBLES
   ═══════════════════════════════════════════════ */

/* Chat message containers */
[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.04) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    padding: 1.2rem 1.4rem !important;
    margin-bottom: 12px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow:
        0 8px 32px rgba(0, 0, 0, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
}

[data-testid="stChatMessage"]:hover {
    background: rgba(255, 255, 255, 0.07) !important;
    border-color: rgba(120, 200, 255, 0.15) !important;
    box-shadow:
        0 8px 32px rgba(0, 0, 0, 0.3),
        0 0 20px rgba(120, 200, 255, 0.05),
        inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
    transform: translateY(-1px);
}

/* Chat text color */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {
    color: rgba(230, 235, 245, 0.92) !important;
    line-height: 1.7 !important;
    font-size: 0.95rem !important;
}

[data-testid="stChatMessage"] strong {
    color: #7bf1a8 !important;
}

[data-testid="stChatMessage"] code {
    background: rgba(0, 245, 212, 0.1) !important;
    color: #00f5d4 !important;
    border-radius: 4px !important;
    padding: 2px 6px !important;
}

/* User avatar ring */
[data-testid="stChatMessage"] [data-testid="stAvatar"] {
    border: 2px solid rgba(0, 245, 212, 0.4) !important;
    border-radius: 50% !important;
    box-shadow: 0 0 12px rgba(0, 245, 212, 0.15) !important;
}

/* ═══════════════════════════════════════════════
   CHAT INPUT — GLOWING BAR
   ═══════════════════════════════════════════════ */
[data-testid="stChatInput"],
[data-testid="stChatInputContainer"] {
    background: transparent !important;
}

[data-testid="stChatInput"] textarea,
[data-testid="stChatInputContainer"] textarea {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 14px !important;
    color: #e6ebf5 !important;
    font-size: 0.95rem !important;
    padding: 14px 18px !important;
    transition: all 0.3s ease !important;
}

[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInputContainer"] textarea:focus {
    border-color: rgba(0, 245, 212, 0.5) !important;
    box-shadow:
        0 0 20px rgba(0, 245, 212, 0.1),
        0 0 40px rgba(0, 245, 212, 0.05),
        inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
    outline: none !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: rgba(200, 210, 230, 0.35) !important;
}

/* Send button */
[data-testid="stChatInput"] button,
[data-testid="stChatInputContainer"] button {
    background: linear-gradient(135deg, #00f5d4, #9b5de5) !important;
    border: none !important;
    border-radius: 10px !important;
    transition: all 0.3s ease !important;
}

[data-testid="stChatInput"] button:hover,
[data-testid="stChatInputContainer"] button:hover {
    filter: brightness(1.2) !important;
    box-shadow: 0 0 20px rgba(0, 245, 212, 0.3) !important;
}

/* ═══════════════════════════════════════════════
   NEW CHAT BUTTON — PILL STYLE
   ═══════════════════════════════════════════════ */
.stButton > button {
    background: rgba(255, 255, 255, 0.06) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 50px !important;
    color: rgba(200, 210, 230, 0.8) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 8px 24px !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    text-transform: uppercase !important;
}

.stButton > button:hover {
    background: rgba(0, 245, 212, 0.12) !important;
    border-color: rgba(0, 245, 212, 0.4) !important;
    color: #00f5d4 !important;
    box-shadow: 0 0 25px rgba(0, 245, 212, 0.1) !important;
    transform: translateY(-2px) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* ═══════════════════════════════════════════════
   SPINNER — CUSTOM COLOR
   ═══════════════════════════════════════════════ */
.stSpinner > div {
    border-top-color: #00f5d4 !important;
}

[data-testid="stSpinnerContainer"] {
    color: rgba(200, 210, 230, 0.6) !important;
}

/* ═══════════════════════════════════════════════
   SCROLLBAR — THIN & THEMED
   ═══════════════════════════════════════════════ */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}

::-webkit-scrollbar-track {
    background: rgba(0, 0, 0, 0.2);
}

::-webkit-scrollbar-thumb {
    background: rgba(0, 245, 212, 0.2);
    border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(0, 245, 212, 0.4);
}

/* ═══════════════════════════════════════════════
   SIDEBAR (if ever used)
   ═══════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.95) !important;
    backdrop-filter: blur(20px) !important;
}

/* ═══════════════════════════════════════════════
   HIDE STREAMLIT DEFAULTS
   ═══════════════════════════════════════════════ */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ═══════════════════════════════════════════════
   BOTTOM GRADIENT FADE ON INPUT DOCK
   ═══════════════════════════════════════════════ */
[data-testid="stBottom"] {
    background: linear-gradient(
        to top,
        rgba(15, 12, 41, 0.95) 60%,
        transparent 100%
    ) !important;
    padding-top: 2rem !important;
}

/* ═══════════════════════════════════════════════
   WELCOME CARD (empty state)
   ═══════════════════════════════════════════════ */
.welcome-card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin: 2rem auto;
    max-width: 520px;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
}

.welcome-card h2 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.3rem;
    color: rgba(230, 235, 245, 0.9);
    margin-bottom: 1rem;
    font-weight: 600;
}

.welcome-card p {
    color: rgba(200, 210, 230, 0.5);
    font-size: 0.88rem;
    line-height: 1.7;
    margin: 0;
}

.welcome-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    margin-top: 1.4rem;
}

.welcome-pill {
    background: rgba(0, 245, 212, 0.08);
    border: 1px solid rgba(0, 245, 212, 0.15);
    border-radius: 50px;
    padding: 6px 16px;
    color: rgba(0, 245, 212, 0.7);
    font-size: 0.78rem;
    letter-spacing: 0.3px;
}

/* ═══════════════════════════════════════════════
   RESPONSIVE
   ═══════════════════════════════════════════════ */
@media (max-width: 768px) {
    .stApp h1 {
        font-size: 1.8rem !important;
    }
    .welcome-card {
        margin: 1rem;
        padding: 1.5rem 1.2rem;
    }
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.title("📈 Graham Investment Agent")
st.caption("Ask about stocks, investing, or Benjamin Graham's principles. Powered by *The Intelligent Investor*.")

# New Chat button
if st.button("🔄 New Chat"):
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.rerun()

# ──────────────────────────────────────────────
# LOAD BOOK INTO CHROMADB (runs once, cached)
# ──────────────────────────────────────────────
@st.cache_resource
def load_book():
    doc = pymupdf.open("The Intelligent Investor.pdf")
    full_text = "\n".join(page.get_text() for page in doc)
    doc.close()

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
# TOOLS (unchanged)
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

    sem_results = collection.query(query_texts=[query], n_results=5)
    sem_docs = sem_results["documents"][0]
    sem_ids = sem_results["ids"][0]
    sem_dists = sem_results["distances"][0]

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
# SYSTEM PROMPT & AGENT
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
    """Create a fresh client every turn."""
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
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

    st.session_state.chat_history = chat.get_history()
    return response.text

# ──────────────────────────────────────────────
# CHAT UI
# ──────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Welcome card when empty
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-card">
        <h2>What would you like to explore?</h2>
        <p>I can pull live stock data, search Graham's <em>Intelligent Investor</em>, and crunch the numbers — all in one conversation.</p>
        <div class="welcome-pills">
            <span class="welcome-pill">Margin of Safety</span>
            <span class="welcome-pill">Analyze AAPL</span>
            <span class="welcome-pill">P/E Ratios</span>
            <span class="welcome-pill">Defensive Investing</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new input
if prompt := st.chat_input("Ask about a stock, Graham's principles, or anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = agent_turn(prompt)
            except Exception as e:
                answer = f"Something went wrong: {str(e)}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
