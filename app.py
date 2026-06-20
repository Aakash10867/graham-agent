import requests
# --- SQLITE PATCH FOR STREAMLIT CLOUD ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# ----------------------------------------

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import streamlit as st
from google import genai
from google.genai import types
import chromadb
import pymupdf
import yfinance as yf
import json
import re

FREE_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-pro",
    "gemini-3.1-pro-preview",
]

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
   DEEP SPACE PARALLAX
   ═══════════════════════════════════════════════ */
.stApp::before,
.stApp::after,
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: -100vh; left: -100vw; right: -100vw; bottom: -100vh;
    pointer-events: none;
    z-index: 0;
}

.stApp::before {
    background-image:
        radial-gradient(1px 1px at 10% 20%, rgba(255, 255, 255, 0.7) 50%, transparent),
        radial-gradient(1.5px 1.5px at 80% 40%, rgba(255, 255, 255, 0.4) 50%, transparent),
        radial-gradient(1px 1px at 30% 70%, rgba(0, 245, 212, 0.6) 50%, transparent),
        radial-gradient(1px 1px at 60% 90%, rgba(255, 255, 255, 0.8) 50%, transparent);
    background-size: 150px 150px;
    animation: starLayer1 25s linear infinite;
}

.stApp::after {
    background-image:
        radial-gradient(1.5px 1.5px at 15% 15%, rgba(0, 245, 212, 0.8) 50%, transparent),
        radial-gradient(2px 2px at 75% 25%, rgba(255, 255, 255, 0.9) 50%, transparent),
        radial-gradient(1.5px 1.5px at 25% 85%, rgba(0, 245, 212, 0.5) 50%, transparent),
        radial-gradient(2px 2px at 85% 65%, rgba(255, 255, 255, 0.7) 50%, transparent);
    background-size: 200px 200px;
    animation: starLayer2 18s linear infinite;
}

[data-testid="stAppViewContainer"]::before {
    background-image:
        radial-gradient(2px 2px at 5% 5%, rgba(255, 255, 255, 1) 50%, transparent),
        radial-gradient(2.5px 2.5px at 95% 95%, rgba(200, 210, 230, 0.9) 50%, transparent),
        radial-gradient(2px 2px at 45% 45%, rgba(0, 245, 212, 0.7) 50%, transparent);
    background-size: 300px 300px;
    animation: starLayer3 12s linear infinite;
}

@keyframes starLayer1 {
    0%   { transform: translateY(0); opacity: 0.3; }
    50%  { opacity: 0.9; }
    100% { transform: translateY(-150px); opacity: 0.3; }
}

@keyframes starLayer2 {
    0%   { transform: translate(0, 0); opacity: 0.2; }
    25%  { opacity: 1; }
    75%  { opacity: 0.3; }
    100% { transform: translate(-200px, -200px); opacity: 0.2; }
}

@keyframes starLayer3 {
    0%   { transform: translate(0, 0); opacity: 0.5; }
    33%  { opacity: 0.9; }
    66%  { opacity: 0.4; }
    100% { transform: translate(300px, -300px); opacity: 0.5; }
}

/* ═══════════════════════════════════════════════
   3D GLASSMORPHISM CHAT INPUT BAR
   ═══════════════════════════════════════════════ */
[data-testid="stChatInput"] {
    background: rgba(20, 25, 45, 0.3) !important;
    backdrop-filter: blur(24px) saturate(200%) !important;
    -webkit-backdrop-filter: blur(24px) saturate(200%) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-top: 1px solid rgba(0, 245, 212, 0.4) !important;
    border-radius: 24px !important;
    box-shadow:
        0 20px 40px rgba(0, 0, 0, 0.6),
        inset 0 1px 3px rgba(255, 255, 255, 0.15) !important;
    transition: all 0.3s ease !important;
}

[data-testid="stChatInput"] > div {
    background: transparent !important;
    background-color: transparent !important;
}

[data-testid="stChatInput"] textarea {
    background: transparent !important;
    background-color: transparent !important;
    color: #00f5d4 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1rem !important;
}

[data-testid="stChatInput"]:hover, [data-testid="stChatInput"]:focus-within {
    border-top: 1px solid rgba(0, 245, 212, 0.8) !important;
    box-shadow:
        0 20px 50px rgba(0, 245, 212, 0.15),
        inset 0 1px 3px rgba(255, 255, 255, 0.2) !important;
    transform: translateY(-2px);
}

/* ═══════════════════════════════════════════════
   TYPOGRAPHY & TITLE — NEON GLOW
   ═══════════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

.stApp, .stApp * {
    font-family: 'Inter', sans-serif !important;
}

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

.stApp .stCaption, .stApp [data-testid="stCaptionContainer"] p {
    color: rgba(200, 210, 230, 0.6) !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.3px;
}

/* ═══════════════════════════════════════════════
   GLASSMORPHISM CHAT BUBBLES
   ═══════════════════════════════════════════════ */
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
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(0, 0, 0, 0.2); }
::-webkit-scrollbar-thumb { background: rgba(0, 245, 212, 0.2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0, 245, 212, 0.4); }

/* ═══════════════════════════════════════════════
   SIDEBAR
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

/* Content above stars */
[data-testid="stAppViewContainer"] {
    background: transparent !important;
    position: relative !important;
    z-index: 1 !important;
}

/* ═══════════════════════════════════════════════
   WELCOME CARD
   ═══════════════════════════════════════════════ */
.welcome-card {
    background: rgba(15, 12, 41, 0.4);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(0, 245, 212, 0.15);
    border-radius: 8px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin: 2rem auto;
    max-width: 550px;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4), inset 0 0 20px rgba(0, 245, 212, 0.05);
}

.welcome-card h2 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.2rem;
    color: #00f5d4;
    margin-bottom: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
}

.welcome-card p {
    color: rgba(200, 210, 230, 0.6);
    font-size: 0.9rem;
    font-family: 'Inter', sans-serif !important;
    line-height: 1.6;
    margin: 0;
}

.welcome-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    justify-content: center;
    margin-top: 1.8rem;
}

.welcome-pill {
    background: rgba(0, 245, 212, 0.05);
    border: 1px solid rgba(0, 245, 212, 0.25);
    border-radius: 4px;
    padding: 8px 16px;
    color: #00f5d4;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1px;
    transition: all 0.2s ease;
}

.welcome-pill:hover {
    background: rgba(0, 245, 212, 0.15);
    border-color: rgba(0, 245, 212, 0.5);
    box-shadow: 0 0 15px rgba(0, 245, 212, 0.2);
    cursor: default;
}

/* ═══════════════════════════════════════════════
   RESPONSIVE
   ═══════════════════════════════════════════════ */
@media (max-width: 768px) {
    .stApp h1 { font-size: 1.8rem !important; }
    .welcome-card { margin: 1rem; padding: 1.5rem 1.2rem; }
}

/* ═══════════════════════════════════════════════
   KILL RED FOCUS OUTLINE
   ═══════════════════════════════════════════════ */
[data-testid="stChatInput"] [data-baseweb="textarea"] {
    outline: none !important;
    box-shadow: none !important;
}

[data-testid="stChatInput"] [data-baseweb="base-input"] {
    outline: none !important;
    box-shadow: none !important;
    border-color: rgba(0, 245, 212, 0.3) !important;
}

[data-testid="stChatInput"] [data-baseweb="base-input"]:focus-within {
    border-color: rgba(0, 245, 212, 0.6) !important;
    box-shadow: 0 0 15px rgba(0, 245, 212, 0.1) !important;
}

*:focus-visible {
    outline: none !important;
}

div[data-baseweb] [aria-invalid] {
    box-shadow: none !important;
}

/* ═══════════════════════════════════════════════
   BOTTOM DOCK — TRANSLUCENT FROSTED BAR
   ═══════════════════════════════════════════════ */
[data-testid="stBottom"] {
    z-index: 10 !important;
    background: rgba(15, 12, 41, 0.6) !important;
    background-color: rgba(15, 12, 41, 0.6) !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    border-top: 1px solid rgba(0, 245, 212, 0.15) !important;
}

[data-testid="stBottom"] > div {
    background: transparent !important;
    background-color: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.title("🏛️ AlphaConsensus Terminal")
st.caption("Quantitative Multi-Agent Investment Committee. Operating on Graham, Greenblatt, and Dorsey frameworks.")

if st.button("🔄 Reset Terminal"):
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.rerun()

# ──────────────────────────────────────────────
# LOAD BOOKS INTO CHROMADB (runs once, cached)
# ──────────────────────────────────────────────
@st.cache_resource
def load_books():
    books = {
        "Graham": "The Intelligent Investor.pdf",
        "Greenblatt": "The Little Book That Still Beats the Market.pdf",
        "Dorsey": "The Five Rules for Successful Stock Investing.pdf"
    }

    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection("investment_committee")

    if collection.count() > 0:
        return collection

    for author, filename in books.items():
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found.")
            continue

        doc = pymupdf.open(filename)
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

        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            collection.add(
                documents=batch,
                metadatas=[{"author": author} for _ in batch],
                ids=[f"{author}_chunk_{j}" for j in range(i, i + len(batch))]
            )

    return collection

collection = load_books()

# ──────────────────────────────────────────────
# TOOLS
# ──────────────────────────────────────────────

def search_book(query: str) -> dict:
    """Search the combined knowledge base of Graham, Greenblatt, and Dorsey.
    Use this when you need specific philosophical frameworks, formulas, or rules
    from any of the three investment authors.

    Args:
        query: What to search for, e.g. "magic formula return on capital" or "economic moat"
    """
    sem_results = collection.query(query_texts=[query], n_results=5)

    if not sem_results["documents"][0]:
        return {"error": "No relevant passages found."}

    sem_docs = sem_results["documents"][0]
    sem_meta = sem_results["metadatas"][0]
    sem_dists = sem_results["distances"][0]

    formatted = []
    for text, meta, dist in zip(sem_docs, sem_meta, sem_dists):
        author = meta.get("author", "Unknown")
        formatted.append(f"[Source: {author} | Relevance: {1-dist:.2f}]:\n{text}")

    return {"passages": "\n\n".join(formatted)}


def get_stock_data(company_query: str) -> dict:
    """Get real financial data for a stock using a ticker symbol OR company name.
    Use this when the user asks about a specific company's financials.

    Args:
        company_query: Stock ticker or company name, e.g. AAPL, TCS, "Mahindra", "Groww"
    """
    resolved_ticker = company_query.upper()

    if ".NS" not in resolved_ticker and ".BO" not in resolved_ticker:
        try:
            url = f"https://query2.finance.yahoo.com/v1/finance/search?q={company_query}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=5)
            data = response.json()

            if "quotes" in data and len(data["quotes"]) > 0:
                quotes = data["quotes"]
                indian_match = next((q for q in quotes if q.get("exchange") in ["NSI", "BSE"]), None)
                if indian_match:
                    resolved_ticker = indian_match["symbol"]
                else:
                    resolved_ticker = quotes[0]["symbol"]
        except Exception:
            pass

    try:
        stock = yf.Ticker(resolved_ticker)
        info = stock.info

        if not info or info.get("regularMarketPrice") is None:
            return {"error": f"No quantitative data found for '{company_query}'. Resolved to ticker [{resolved_ticker}] but it may be a private entity, mutual fund, or invalid."}

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
            "debt_to_equity": info.get("debtToEquity"),
        }
    except Exception as e:
        return {"error": f"Data retrieval failed for [{resolved_ticker}]: {str(e)}"}


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


def lookup_ticker(company_name: str) -> dict:
    """Find the stock ticker symbol for a company given its name.
    Use this FIRST whenever the user mentions a company by name
    without providing a ticker symbol.

    Args:
        company_name: The company name, e.g. "Groww", "Apple", "Tata Motors"
    """
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={company_name}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        data = response.json()

        if "quotes" in data and len(data["quotes"]) > 0:
            matches = []
            for q in data["quotes"][:5]:
                matches.append({
                    "symbol": q.get("symbol"),
                    "name": q.get("longname") or q.get("shortname"),
                    "exchange": q.get("exchange"),
                    "type": q.get("quoteType"),
                })
            if matches:
                return {"matches": matches}
        return {"error": f"No ticker found for '{company_name}'. It may not be publicly listed."}
    except Exception as e:
        return {"error": f"Ticker lookup failed: {str(e)}"}


# Register all tools
tool_functions = {
    "search_book": search_book,
    "get_stock_data": get_stock_data,
    "calculator": calculator,
    "lookup_ticker": lookup_ticker,
}

TOOLS = [search_book, get_stock_data, calculator, lookup_ticker]


def fallback_router(prompt: str) -> str:
    """Deterministic routing engine that triggers when the LLM is offline."""
    prompt_lower = prompt.lower()
    response_blocks = []

    potential_tickers = re.findall(r'\b[A-Z]{1,6}(?:\.NS)?\b', prompt)

    if "mahindra" in prompt_lower: potential_tickers.append("M&M.NS")
    if "apple" in prompt_lower: potential_tickers.append("AAPL")

    tickers_to_check = list(set(potential_tickers))
    valid_stock_found = False

    for ticker in tickers_to_check:
        if ticker in ["I", "A", "THE", "WHAT", "WHY", "HOW", "IS", "YES", "NO"]:
            continue

        data = get_stock_data(ticker)
        if "error" not in data:
            valid_stock_found = True
            table = f"### 📊 Auto-Fetched Data for {data.get('symbol', ticker)}\n"
            table += "| Metric | Value |\n| :--- | :--- |\n"
            table += f"| **Price** | {data.get('currency', '')} {data.get('current_price', 'N/A')} |\n"
            table += f"| **P/E Ratio** | {data.get('pe_ratio', 'N/A')} |\n"
            table += f"| **P/B Ratio** | {data.get('price_to_book', 'N/A')} |\n"
            table += f"| **ROE** | {round(data.get('return_on_equity', 0) * 100, 2) if data.get('return_on_equity') else 'N/A'}% |\n\n"
            response_blocks.append(table)

    book_keywords = ["graham", "greenblatt", "dorsey", "moat", "margin", "safety", "value", "formula", "rule"]
    if not valid_stock_found or any(kw in prompt_lower for kw in book_keywords):
        book_data = search_book(prompt)
        if "error" not in book_data:
            response_blocks.append("### 📚 Auto-Fetched Knowledge Base Passages\n")
            for p in book_data["passages"].split("\n\n"):
                response_blocks.append(f"> {p}\n\n")

    if not response_blocks:
        return "❌ *Fallback System:* Could not identify a valid ticker or knowledge base match from the prompt syntax."

    return "".join(response_blocks)

# ──────────────────────────────────────────────
# SYSTEM PROMPT & AGENT
# ──────────────────────────────────────────────

SYSTEM_INSTRUCTION = """You are a highly structured Quantitative Investment Committee acting as a single agent.

Your knowledge base consists of three frameworks:
1. Benjamin Graham (Defensive Value, Margin of Safety, P/B, P/E)
2. Joel Greenblatt (The Magic Formula, Return on Capital, Earnings Yield)
3. Pat Dorsey (Economic Moats, Consistent FCF, ROE > 15%, Low Debt)

You have four tools:
1. search_book — queries the texts of Graham, Greenblatt, and Dorsey.
2. get_stock_data — pulls live fundamental data for a ticker symbol or company name.
3. calculator — evaluates mathematical expressions.
4. lookup_ticker — finds the stock ticker symbol for a company name. Use this when the user mentions a company by name without a ticker.

RULES:
- When the user mentions a company by NAME (not a ticker symbol), call lookup_ticker FIRST to find the correct ticker, then call get_stock_data with that ticker.
- When you use search_book, base your answer on the retrieved passages. If the passages don't contain the answer, say so honestly.
- When analyzing a stock, connect the data back to the three frameworks when relevant.
- Be concise and direct. No filler.
- If the user asks something outside investing/finance, just answer normally without using tools.
- Remember the full conversation — the user may refer to earlier questions.

CRITICAL INSTRUCTIONS FOR STOCK ANALYSIS:
When the user asks you to evaluate a stock, you MUST NOT write a generic summary. You must execute a Three-Factor Committee Analysis using Markdown tables and provide a definitive YES/NO verdict.

PASS/FAIL THRESHOLDS — apply these mechanically. Do NOT override with qualitative judgment.

Graham Pass requires ALL of:
  - P/E ratio ≤ 15
  - P/B ratio ≤ 1.5 (or if one is slightly above, P/E × P/B ≤ 22.5)
  - Dividend Yield > 0%

Greenblatt Pass requires BOTH of:
  - Return on Equity > 15%
  - Earnings Yield (1 ÷ P/E × 100) > 5%

Dorsey Pass requires ALL of:
  - ROE > 15%
  - Debt/Equity < 50%
  - Identifiable economic moat (brand, switching costs, network effects, or cost advantage)

DECISION RULE:
- Graham has VETO POWER. If Graham fails, the final verdict is always NO, even if the other two pass. Margin of safety is non-negotiable.
- If Graham passes: 2 out of 3 passing = YES, otherwise NO.
- State each threshold and the actual value side by side in the verdict (e.g. "P/E of 12.3 vs threshold of 15 — Pass").
- Do NOT override these rules with qualitative reasoning. The framework IS the answer.

Format your response EXACTLY like this:

### 1. Live Fundamentals
[Render a clean Markdown table with the stock's current price, P/E, Forward P/E, P/B, ROE, Debt/Equity, and Dividend Yield]

### 2. The Committee Verdict

* **Graham's Verdict:** [Pass/Fail] — [State threshold vs actual for P/E, P/B, and dividend yield]
* **Greenblatt's Verdict:** [Pass/Fail] — [State threshold vs actual for ROE and earnings yield]
* **Dorsey's Verdict:** [Pass/Fail] — [State threshold vs actual for ROE, D/E, and name the moat or lack thereof]

### 3. Final Decision
**[YES or NO]** — [One paragraph. If Graham vetoed, say so explicitly. No hedging.]


def agent_turn(user_message):
    """Try each free model until one responds."""
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    history = st.session_state.get("chat_history", [])

    last_error = None

    for model_name in FREE_MODELS:
        try:
            chat = client.chats.create(
                model=model_name,
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
            return response.text, model_name

        except Exception as e:
            error_msg = str(e)
            last_error = error_msg
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                continue
            else:
                raise

    raise Exception(f"All models rate-limited. Last error: {last_error}")

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
        <h2>SYSTEM INITIALIZED</h2>
        <p>AlphaConsensus engine online. Awaiting ticker input or framework query.</p>
        <div class="welcome-pills">
            <span class="welcome-pill">> ANALYZE AAPL</span>
            <span class="welcome-pill">> QUERY MOAT RULES</span>
            <span class="welcome-pill">> MAGIC FORMULA YIELD</span>
            <span class="welcome-pill">> GRAHAM CRITERIA</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

USER_AVATAR = "👤"
AGENT_AVATAR = "📈"

# Display past messages
for msg in st.session_state.messages:
    avatar = USER_AVATAR if msg["role"] == "user" else AGENT_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("model"):
            st.caption(f"⚡ Powered by `{msg['model']}`")

# Handle new input
if prompt := st.chat_input("Ask about a stock, Graham's principles, or anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=AGENT_AVATAR):
        with st.spinner("Executing multi-factor analysis..."):
            try:
                answer, model_used = agent_turn(prompt)
                st.markdown(answer)
                st.caption(f"⚡ Powered by `{model_used}`")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "model": model_used
                })

            except Exception as e:
                error_msg = str(e)

                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "All models rate-limited" in error_msg:
                    st.warning("⚠️ **AlphaConsensus Engine Offline (API Limit).** Engaging deterministic fallback routing...")
                    fallback_answer = fallback_router(prompt)
                    st.markdown(fallback_answer)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"*(Deterministic Fallback Engaged)*\n\n{fallback_answer}"
                    })
                else:
                    st.error(f"🛑 **System Error:** Unable to process request. \n\n`{error_msg[:100]}...`")
