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
   DEEP SPACE PARALLAX (Independent / Chaotic Movement)
   ═══════════════════════════════════════════════ */

/* Base setup for all 3 star layers */
.stApp::before,
.stApp::after,
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    /* Oversize the layers massively so we don't see edges when moving diagonally */
    top: -100vh; left: -100vw; right: -100vw; bottom: -100vh;
    pointer-events: none;
    z-index: 0;
}

/* LAYER 1: Distant Stars (Slow, moving straight up) */
.stApp::before {
    background-image:
        radial-gradient(1px 1px at 10% 20%, rgba(255, 255, 255, 0.7) 50%, transparent),
        radial-gradient(1.5px 1.5px at 80% 40%, rgba(255, 255, 255, 0.4) 50%, transparent),
        radial-gradient(1px 1px at 30% 70%, rgba(0, 245, 212, 0.6) 50%, transparent),
        radial-gradient(1px 1px at 60% 90%, rgba(255, 255, 255, 0.8) 50%, transparent);
    background-size: 150px 150px;
    animation: starLayer1 25s linear infinite;
}

/* LAYER 2: Midground Stars (Medium speed, moving up-left, erratic twinkle) */
.stApp::after {
    background-image:
        radial-gradient(1.5px 1.5px at 15% 15%, rgba(0, 245, 212, 0.8) 50%, transparent),
        radial-gradient(2px 2px at 75% 25%, rgba(255, 255, 255, 0.9) 50%, transparent),
        radial-gradient(1.5px 1.5px at 25% 85%, rgba(0, 245, 212, 0.5) 50%, transparent),
        radial-gradient(2px 2px at 85% 65%, rgba(255, 255, 255, 0.7) 50%, transparent);
    background-size: 200px 200px;
    animation: starLayer2 18s linear infinite;
}

/* LAYER 3: Foreground Stars (Fast, moving up-right, bright) */
[data-testid="stAppViewContainer"]::before {
    background-image:
        radial-gradient(2px 2px at 5% 5%, rgba(255, 255, 255, 1) 50%, transparent),
        radial-gradient(2.5px 2.5px at 95% 95%, rgba(200, 210, 230, 0.9) 50%, transparent),
        radial-gradient(2px 2px at 45% 45%, rgba(0, 245, 212, 0.7) 50%, transparent);
    background-size: 300px 300px;
    animation: starLayer3 12s linear infinite;
}

/* --- INDEPENDENT ANIMATIONS --- */
/* Note: The translate values MUST exactly match the background-size above to loop seamlessly without jumping */

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
   3D GLASSMORPHISM CHAT INPUT BAR (The bottom typing area)
   ═══════════════════════════════════════════════ */

/* 1. Target the main outer container of the chat input */
[data-testid="stChatInput"] {
    background: rgba(20, 25, 45, 0.3) !important; /* Highly transparent frosted base */
    backdrop-filter: blur(24px) saturate(200%) !important;
    -webkit-backdrop-filter: blur(24px) saturate(200%) !important;
    
    /* 3D Edges */
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-top: 1px solid rgba(0, 245, 212, 0.4) !important; /* Bright neon light catch on top */
    border-radius: 24px !important; /* Sleek pill shape */
    
    /* Deep floating shadow */
    box-shadow: 
        0 20px 40px rgba(0, 0, 0, 0.6), 
        inset 0 1px 3px rgba(255, 255, 255, 0.15) !important;
        
    transition: all 0.3s ease !important;
}

/* 2. Strip the solid background from Streamlit's inner wrapper */
[data-testid="stChatInput"] > div {
    background: transparent !important;
    background-color: transparent !important;
}

/* 3. Target the actual text area where you type */
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    background-color: transparent !important;
    color: #00f5d4 !important; /* Makes the text you type neon cyan */
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1rem !important;
}

/* 4. Hover effect to make it feel responsive */
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
   AGGRESSIVE TRANSPARENT INPUT DOCK
   ═══════════════════════════════════════════════ */
[data-testid="stBottom"], 
[data-testid="stBottom"] > div,
[data-testid="stBottom"] [data-testid="stVerticalBlock"] {
    background: transparent !important;
    background-color: transparent !important;
    background-image: none !important;
}

/* Ensure the main view wrapper isn't creating a solid block behind it */
[data-testid="stAppViewContainer"] {
    background: transparent !important;
}

/* ═══════════════════════════════════════════════
   WELCOME CARD (Terminal Dashboard)
   ═══════════════════════════════════════════════ */
.welcome-card {
    background: rgba(15, 12, 41, 0.4);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(0, 245, 212, 0.15);
    border-radius: 8px; /* Sharper corners */
    padding: 2.5rem 2rem;
    text-align: center;
    margin: 2rem auto;
    max-width: 550px;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4), inset 0 0 20px rgba(0, 245, 212, 0.05);
}

.welcome-card h2 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.2rem;
    color: #00f5d4; /* Neon terminal green/cyan */
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
    border-radius: 4px; /* Tech/Command box look instead of round pill */
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
st.title("🏛️ AlphaConsensus Terminal")
st.caption("Quantitative Multi-Agent Investment Committee. Operating on Graham, Greenblatt, and Dorsey frameworks.")

# New Chat button
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
    
    # Write to a local folder in the container instead of RAM
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    # Use get_or_create to avoid duplication errors on hot reloads
    collection = chroma_client.get_or_create_collection("investment_committee")
    
    # Check if we already loaded them
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
# TOOLS (unchanged)
# ──────────────────────────────────────────────
def lookup_ticker(company_name: str) -> dict:
    """Find the stock ticker symbol for a company given its name.
    Use this FIRST whenever the user mentions a company by name
    without providing a ticker symbol.

    Args:
        company_name: The company name, e.g. "Groww", "Apple", "Tata Motors"
    """
    try:
        results = yf.search(company_name, max_results=5)
        if results and "quotes" in results:
            matches = []
            for q in results["quotes"]:
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
        # Prepend the author so the LLM explicitly knows the source
        formatted.append(f"[Source: {author} | Relevance: {1-dist:.2f}]:\n{text}")

    return {"passages": "\n\n".join(formatted)}


def get_stock_data(company_query: str) -> dict:
    """Get real financial data for a stock using a ticker symbol OR company name.
    Use this when the user asks about a specific company's financials.
    
    Args:
        company_query: Stock ticker or company name, e.g. AAPL, TCS, "Mahindra", "Groww"
    """
    # 1. THE AUTO-RESOLUTION LAYER
    # If the input doesn't already look like an explicit NSE ticker, search for it
    resolved_ticker = company_query.upper()
    
    if ".NS" not in resolved_ticker and ".BO" not in resolved_ticker:
        try:
            url = f"https://query2.finance.yahoo.com/v1/finance/search?q={company_query}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=5)
            data = response.json()
            
            if "quotes" in data and len(data["quotes"]) > 0:
                quotes = data["quotes"]
                
                # Try to prioritize Indian markets (NSE/BSE) first
                indian_match = next((q for q in quotes if q.get("exchange") in ["NSI", "BSE"]), None)
                if indian_match:
                    resolved_ticker = indian_match["symbol"]
                else:
                    # Otherwise, grab the top global result
                    resolved_ticker = quotes[0]["symbol"]
        except Exception:
            pass # If the search fails, just try running the original query

    # 2. THE DATA EXTRACTION LAYER
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


tool_functions = {
    "search_book": search_book,
    "get_stock_data": get_stock_data,
    "calculator": calculator,
    "lookup_ticker": lookup_ticker,
}

TOOLS = [search_book, get_stock_data, calculator, lookup_ticker]

import re

def fallback_router(prompt: str) -> str:
    """Deterministic routing engine that triggers when the LLM is offline."""
    prompt_lower = prompt.lower()
    response_blocks = []

    # 1. HEURISTIC FOR STOCK DATA
    # Look for uppercase words (e.g., AAPL) or words ending in .NS (e.g., TCS.NS)
    potential_tickers = re.findall(r'\b[A-Z]{1,6}(?:\.NS)?\b', prompt)
    
    # Add manual aliases for common names
    if "mahindra" in prompt_lower: potential_tickers.append("M&M.NS")
    if "apple" in prompt_lower: potential_tickers.append("AAPL")
    
    tickers_to_check = list(set(potential_tickers))
    valid_stock_found = False

    for ticker in tickers_to_check:
        # Ignore common English uppercase words
        if ticker in ["I", "A", "THE", "WHAT", "WHY", "HOW", "IS", "YES", "NO"]: continue
        
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

    # 2. HEURISTIC FOR BOOK SEARCH
    # If no valid stock was found, or if specific keywords are present, search the books
    book_keywords = ["graham", "greenblatt", "dorsey", "moat", "margin", "safety", "value", "formula", "rule"]
    if not valid_stock_found or any(kw in prompt_lower for kw in book_keywords):
        book_data = search_book(prompt)
        if "error" not in book_data:
            response_blocks.append("### 📚 Auto-Fetched Knowledge Base Passages\n")
            # Format passages as blockquotes
            for p in book_data["passages"].split("\n\n"):
                response_blocks.append(f"> {p}\n\n")
                
    if not response_blocks:
        return "❌ *Fallback System:* Could not identify a valid ticker or knowledge base match from the prompt syntax."
        
    return "".join(response_blocks)

# ──────────────────────────────────────────────
# SYSTEM PROMPT & AGENT
# ──────────────────────────────────────────────

SYSTEM_INSTRUCTION = """You are an investment analysis assistant grounded in Benjamin Graham's principles.

You have four tools:
1. search_book — searches The Intelligent Investor for relevant passages. USE THIS when the user asks about investing concepts, Graham's philosophy, or wants book-based advice.
2. get_stock_data — pulls real financial data for a stock ticker. USE THIS when the user asks about specific companies or wants fundamental data.
3. calculator — evaluates math expressions. USE THIS for any computation.
4. lookup_ticker — finds the stock ticker symbol for a company name. USE THIS when the user mentions a company by name without a ticker.

RULES:
- When the user mentions a company by NAME (not a ticker symbol), call lookup_ticker FIRST to find the correct ticker, then call get_stock_data with that ticker.
- When you use search_book, base your answer on the retrieved passages. If the passages don't contain the answer, say so honestly.
- When analyzing a stock, connect the data back to Graham's principles when relevant.
- Be concise and direct. No filler.
- If the user asks something outside investing/finance, just answer normally without using tools.
- Remember the full conversation — the user may refer to earlier questions.

CRITICAL INSTRUCTIONS FOR STOCK ANALYSIS:
When the user asks you to evaluate a stock, you MUST NOT write a generic summary. You must execute a "Three-Factor Committee Analysis" using Markdown tables and provide a definitive YES/NO verdict.

Format your response EXACTLY like this:

### 1. Live Fundamentals
[Render a clean Markdown table with the stocks current price, P/E, Forward P/E, P/B, ROE, Debt/Equity, and Dividend Yield]

### 2. The Committee Verdict
Evaluate the data against the three frameworks. If you are unsure of a specific rule, use the search_books tool.

* **Graham's Verdict:** [Pass/Fail] - [One sentence justification based on Margin of Safety/Valuation]
* **Greenblatt's Verdict:** [Pass/Fail] - [One sentence justification based on Earnings Yield/Capital Efficiency]
* **Dorsey's Verdict:** [Pass/Fail] - [One sentence justification based on inferred Moat/Financial Health]

### 3. Final Decision
**[YES or NO]** [If 2 out of 3 Pass, it is a YES. If not, it is a NO. Provide a brief, blunt, one-paragraph explanation of the final ruling.]
"""

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


# Define your avatars
USER_AVATAR = "👤"
AGENT_AVATAR = "📈"

# Display past messages
for msg in st.session_state.messages:
    # Assign the correct avatar based on the role
    avatar = USER_AVATAR if msg["role"] == "user" else AGENT_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Handle new input
if prompt := st.chat_input("Ask about a stock, Graham's principles, or anything..."):
    # 1. Save user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. Display user message in UI
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    # 3. Handle agent response
    with st.chat_message("assistant", avatar=AGENT_AVATAR):  # ← same level as user block
            with st.spinner("Executing multi-factor analysis..."):
                try:
                    # Attempt primary LLM orchestration
                    answer = agent_turn(prompt)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    error_msg = str(e)
                    
                    # --- AUTOMATED FALLBACK INTERCEPTION ---
                    if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                        # 1. Inform the user of the system state
                        st.warning("⚠️ **AlphaConsensus Engine Offline (API Limit).** Engaging deterministic fallback routing...")
                        
                        # 2. Run the fallback router directly in the UI
                        fallback_answer = fallback_router(prompt)
                        st.markdown(fallback_answer)
                        
                        # 3. Save the fallback data to the chat history so it persists
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": f"*(Deterministic Fallback Engaged)*\n\n{fallback_answer}"
                        })
                    else:
                        # Generic system crash
                        st.error(f"🛑 **System Error:** Unable to process request. \n\n`{error_msg[:100]}...`")
