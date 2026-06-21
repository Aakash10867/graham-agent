"""
ALPHACONSENSUS TERMINAL
========================
Quantitative Multi-Agent Investment Committee.
Operating on Graham, Greenblatt, Dorsey, and Trajectory frameworks.

Streamlit web app with Gemini LLM, ChromaDB RAG, and yfinance tools.
"""

# --- SQLITE PATCH FOR STREAMLIT CLOUD ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# ----------------------------------------

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import datetime
import streamlit as st
from google import genai
from google.genai import types
import chromadb
import pymupdf
import yfinance as yf
import json
import re
import requests
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# ──────────────────────────────────────────────
# FREE MODEL FALLBACK LIST
# ──────────────────────────────────────────────
FREE_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-pro",
    "gemini-3.1-pro-preview",
]

# ──────────────────────────────────────────────
# TICKER ALIAS MAP
# ──────────────────────────────────────────────
TICKER_ALIASES = {
    # ── Nifty 50 & common Indian abbreviations ──
    "RIL": "RELIANCE.NS",
    "RELIANCE": "RELIANCE.NS",
    "RELIANCE INDUSTRIES": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "TATA CONSULTANCY": "TCS.NS",
    "TATA CONSULTANCY SERVICES": "TCS.NS",
    "INFY": "INFY.NS",
    "INFOSYS": "INFY.NS",
    "HDFC": "HDFCBANK.NS",
    "HDFC BANK": "HDFCBANK.NS",
    "ICICI": "ICICIBANK.NS",
    "ICICI BANK": "ICICIBANK.NS",
    "SBI": "SBIN.NS",
    "STATE BANK": "SBIN.NS",
    "STATE BANK OF INDIA": "SBIN.NS",
    "WIPRO": "WIPRO.NS",
    "ITC": "ITC.NS",
    "LT": "LT.NS",
    "L&T": "LT.NS",
    "LARSEN": "LT.NS",
    "LARSEN AND TOUBRO": "LT.NS",
    "M&M": "M&M.NS",
    "MAHINDRA": "M&M.NS",
    "BAJAJ FINANCE": "BAJFINANCE.NS",
    "BAJAJ FINSERV": "BAJAJFINSV.NS",
    "KOTAK": "KOTAKBANK.NS",
    "KOTAK BANK": "KOTAKBANK.NS",
    "KOTAK MAHINDRA": "KOTAKBANK.NS",
    "MARUTI": "MARUTI.NS",
    "MARUTI SUZUKI": "MARUTI.NS",
    "TATA MOTORS": "TATAMOTORS.NS",
    "TATA STEEL": "TATASTEEL.NS",
    "AIRTEL": "BHARTIARTL.NS",
    "BHARTI AIRTEL": "BHARTIARTL.NS",
    "HUL": "HINDUNILVR.NS",
    "HINDUSTAN UNILEVER": "HINDUNILVR.NS",
    "ASIAN PAINTS": "ASIANPAINT.NS",
    "SUN PHARMA": "SUNPHARMA.NS",
    "SUNPHARMA": "SUNPHARMA.NS",
    "ADANI": "ADANIENT.NS",
    "ADANI ENTERPRISES": "ADANIENT.NS",
    "ADANI PORTS": "ADANIPORTS.NS",
    "ZOMATO": "ZOMATO.NS",
    "PAYTM": "PAYTM.NS",
    "NYKAA": "NYKAA.NS",
    "DMART": "DMART.NS",
    "AVENUE SUPERMARTS": "DMART.NS",
    "TITAN": "TITAN.NS",
    "NESTLE": "NESTLEIND.NS",
    "NESTLE INDIA": "NESTLEIND.NS",
    "POWER GRID": "POWERGRID.NS",
    "NTPC": "NTPC.NS",
    "COAL INDIA": "COALINDIA.NS",
    "ONGC": "ONGC.NS",
    "AXIS": "AXISBANK.NS",
    "AXIS BANK": "AXISBANK.NS",
    "TECH MAHINDRA": "TECHM.NS",
    "HCL": "HCLTECH.NS",
    "HCLTECH": "HCLTECH.NS",
    "HCL TECH": "HCLTECH.NS",
    "ULTRATECH": "ULTRACEMCO.NS",
    "ULTRATECH CEMENT": "ULTRACEMCO.NS",
    "BAJAJ AUTO": "BAJAJ-AUTO.NS",
    "HERO": "HEROMOTOCO.NS",
    "HERO MOTOCORP": "HEROMOTOCO.NS",
    "BRITANNIA": "BRITANNIA.NS",
    "CIPLA": "CIPLA.NS",
    "DR REDDY": "DRREDDY.NS",
    "DR REDDYS": "DRREDDY.NS",
    "EICHER": "EICHERMOT.NS",
    "EICHER MOTORS": "EICHERMOT.NS",
    "GRASIM": "GRASIM.NS",
    "HINDALCO": "HINDALCO.NS",
    "INDUSIND": "INDUSINDBK.NS",
    "INDUSIND BANK": "INDUSINDBK.NS",
    "JSW STEEL": "JSWSTEEL.NS",
    "TATA CONSUMER": "TATACONSUM.NS",
    "UPL": "UPL.NS",
    "DIVIS": "DIVISLAB.NS",
    "DIVIS LAB": "DIVISLAB.NS",
    "SHREE CEMENT": "SHREECEM.NS",
    "SBI LIFE": "SBILIFE.NS",
    "SBILIFE": "SBILIFE.NS",
    "HDFC LIFE": "HDFCLIFE.NS",
    "HDFCLIFE": "HDFCLIFE.NS",
    "TATA POWER": "TATAPOWER.NS",
    "TATA ELXSI": "TATAELXSI.NS",
    "HAL": "HAL.NS",
    "BEL": "BEL.NS",
    "IRCTC": "IRCTC.NS",
    "VEDANTA": "VEDL.NS",
    "VEDL": "VEDL.NS",
    "SAIL": "SAIL.NS",
    "IOC": "IOC.NS",
    "INDIAN OIL": "IOC.NS",
    "BPCL": "BPCL.NS",
    "HPCL": "HINDPETRO.NS",
    "PNB": "PNB.NS",
    "BANK OF BARODA": "BANKBARODA.NS",
    "BOB": "BANKBARODA.NS",
    "CANARA BANK": "CANBK.NS",
    # ── Major US stocks ──
    "APPLE": "AAPL",
    "MICROSOFT": "MSFT",
    "GOOGLE": "GOOGL",
    "ALPHABET": "GOOGL",
    "AMAZON": "AMZN",
    "META": "META",
    "FACEBOOK": "META",
    "TESLA": "TSLA",
    "NVIDIA": "NVDA",
    "NETFLIX": "NFLX",
    "BERKSHIRE": "BRK-B",
    "JPMORGAN": "JPM",
    "JP MORGAN": "JPM",
    "GOLDMAN": "GS",
    "GOLDMAN SACHS": "GS",
    "DISNEY": "DIS",
    "COCA COLA": "KO",
    "PEPSI": "PEP",
    "JOHNSON AND JOHNSON": "JNJ",
    "WALMART": "WMT",
    "VISA": "V",
    "MASTERCARD": "MA",
}

# ──────────────────────────────────────────────
# SCREENING UNIVERSE — stocks to scan
# ──────────────────────────────────────────────
SCREENING_UNIVERSE = {
    "Nifty 50": [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
        "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS", "HINDUNILVR.NS",
        "KOTAKBANK.NS", "AXISBANK.NS", "BAJFINANCE.NS", "ASIANPAINT.NS",
        "MARUTI.NS", "TATAMOTORS.NS", "SUNPHARMA.NS", "TITAN.NS",
        "WIPRO.NS", "HCLTECH.NS", "ULTRACEMCO.NS", "NTPC.NS",
        "POWERGRID.NS", "TATASTEEL.NS", "NESTLEIND.NS", "TECHM.NS",
        "BAJAJ-AUTO.NS", "INDUSINDBK.NS", "JSWSTEEL.NS", "M&M.NS",
        "ADANIENT.NS", "ADANIPORTS.NS", "COALINDIA.NS", "ONGC.NS",
        "BAJAJFINSV.NS", "BRITANNIA.NS", "CIPLA.NS", "DRREDDY.NS",
        "EICHERMOT.NS", "GRASIM.NS", "HEROMOTOCO.NS", "HINDALCO.NS",
        "DIVISLAB.NS", "SBILIFE.NS", "HDFCLIFE.NS", "TATACONSUM.NS",
        "SHREECEM.NS", "TATAPOWER.NS", "BEL.NS", "HAL.NS",
    ],
    "US Large Cap": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA",
        "BRK-B", "JPM", "V", "JNJ", "WMT", "PG", "MA", "UNH",
        "HD", "DIS", "KO", "PEP", "NFLX", "COST", "ABBV", "MRK",
        "CRM", "AMD", "INTC", "GS", "BA", "CAT", "GE",
    ],
}


# ──────────────────────────────────────────────
# TICKER RESOLUTION HELPERS
# ──────────────────────────────────────────────
def _search_yahoo(query):
    """Search Yahoo Finance for ticker matches."""
    try:
        search_result = yf.Search(query)
        quotes = getattr(search_result, "quotes", None)
        if quotes:
            return [
                {
                    "symbol": q.get("symbol"),
                    "name": q.get("longname") or q.get("shortname"),
                    "exchange": q.get("exchange"),
                    "type": q.get("quoteType"),
                }
                for q in quotes[:5]
            ]
    except Exception:
        pass

    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=5)
        data = resp.json()
        if "quotes" in data and data["quotes"]:
            return [
                {
                    "symbol": q.get("symbol"),
                    "name": q.get("longname") or q.get("shortname"),
                    "exchange": q.get("exchange"),
                    "type": q.get("quoteType"),
                }
                for q in data["quotes"][:5]
            ]
    except Exception:
        pass

    return None


def _resolve_ticker(query):
    """Central ticker resolution: alias map -> yf.Search -> raw fallback."""
    key = query.strip().upper()

    if key in TICKER_ALIASES:
        return TICKER_ALIASES[key]

    if ".NS" in key or ".BO" in key:
        return key

    results = _search_yahoo(query)
    if results:
        indian = next(
            (q for q in results if q.get("exchange") in ("NSI", "BSE", "NSE")),
            None,
        )
        if indian and indian.get("symbol"):
            return indian["symbol"]
        if results[0].get("symbol"):
            return results[0]["symbol"]

    return key


# ──────────────────────────────────────────────
# SCREENER ENGINE
# ──────────────────────────────────────────────
def _fetch_single_ticker(ticker):
    """Fetch key metrics + trajectory data for one ticker. Returns dict or None."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info or not info.get("regularMarketPrice"):
            return None

        pe = info.get("trailingPE")

        data = {
            "ticker": ticker,
            "name": info.get("longName") or info.get("shortName", ticker),
            "sector": info.get("sector", "N/A"),
            "price": info.get("regularMarketPrice") or info.get("currentPrice"),
            "pe": pe,
            "pb": info.get("priceToBook"),
            "roe": info.get("returnOnEquity"),
            "de": info.get("debtToEquity"),
            "dividend_yield": info.get("dividendYield"),
            "eps": info.get("trailingEps"),
            "earnings_yield": round(1.0 / pe * 100, 2) if pe and pe > 0 else None,
            "profit_margin": info.get("profitMargins"),
            "market_cap": info.get("marketCap"),
            "rev_growth": None,
            "ni_growth": None,
            "debt_growth": None,
        }

        # Trajectory: 1-year YoY growth from financial statements
        try:
            income_stmt = stock.financials
            if income_stmt is not None and not income_stmt.empty and len(income_stmt.columns) >= 2:
                cols = sorted(income_stmt.columns)[-2:]

                try:
                    rev = [income_stmt.loc["Total Revenue", c] for c in cols]
                    if all(pd.notna(v) and v > 0 for v in rev):
                        data["rev_growth"] = round((rev[1] / rev[0] - 1) * 100, 2)
                except (KeyError, ZeroDivisionError):
                    pass

                try:
                    ni = [income_stmt.loc["Net Income", c] for c in cols]
                    if all(pd.notna(v) for v in ni) and ni[0] != 0:
                        data["ni_growth"] = round((ni[1] / ni[0] - 1) * 100, 2)
                except (KeyError, ZeroDivisionError):
                    pass
        except Exception:
            pass

        try:
            balance_sheet = stock.balance_sheet
            if balance_sheet is not None and not balance_sheet.empty and len(balance_sheet.columns) >= 2:
                cols = sorted(balance_sheet.columns)[-2:]
                try:
                    debt = [balance_sheet.loc["Total Debt", c] for c in cols]
                    if all(pd.notna(v) for v in debt) and debt[0] > 0:
                        data["debt_growth"] = round((debt[1] / debt[0] - 1) * 100, 2)
                except (KeyError, ZeroDivisionError):
                    pass
        except Exception:
            pass

        return data
    except Exception:
        return None


@st.cache_data(ttl=21600, show_spinner="Scanning stock universe (fundamentals + trajectory)... First run takes ~60s, then cached for 6 hours.")
def _fetch_universe_data():
    """Fetch metrics for all stocks in the screening universe (parallel)."""
    all_tickers = []
    for group in SCREENING_UNIVERSE.values():
        all_tickers.extend(group)

    results = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_fetch_single_ticker, t): t for t in all_tickers}
        for future in as_completed(futures):
            data = future.result()
            if data:
                results[data["ticker"]] = data

    return results


# ══════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="AlphaConsensus Terminal",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════
# PRESET PROMPTS — reduced to essentials
# ══════════════════════════════════════════════
STOCK_PRESETS = [
    ("📊 Full Analysis",
     "Give me a complete investment analysis of {company} — valuation, financials, growth, and recommendation using all frameworks."),
    ("💰 Graham Value",
     "Calculate the Graham intrinsic value for {company}. Is it undervalued or overvalued? What is the margin of safety?"),
    ("📈 Performance & Chart",
     "How has {company} stock performed over the last 1 year? Show me returns, highs/lows, volatility, and a price chart."),
    ("🎯 Analyst View",
     "What do analysts recommend for {company}? What are the price targets?"),
    ("💸 Dividends",
     "Does {company} pay dividends? Show me the full dividend track record, growth rate, and current yield."),
    ("⚖️ Compare",
     "Compare {company} as investments — valuation, growth, profitability, and which is the better buy."),
]

SCREENER_PRESETS = [
    ("🇮🇳 Screen Indian Stocks",
     "Find the best Indian stocks to invest in right now. Show me which Nifty 50 stocks pass all 4 frameworks and which pass 3 out of 4 and which pass 2 out of 4, with upto top 10 from each tier. Explain why each tier is a good investment using the book philosophies."),
    ("🇺🇸 Screen US Stocks",
     "Find the best US stocks to invest in right now. Show me which large cap stocks pass all 4 frameworks and which pass 3 out of 4 and which pass 2 out of 4, with upto top 10 from each tier. Explain why each tier is a good investment using the book philosophies."),
    ("🌍 Screen All Markets",
     "Screen all stocks across India and US markets. Show me the best investment candidates that pass all 4 frameworks or 3 out of 4. Explain why each category is good for long-term returns based on Graham, Greenblatt, and Dorsey."),
]

# ══════════════════════════════════════════════
# CSS — CLEAN, SOLID, MINIMAL
# ══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

/* ── Base ── */
.stApp {
    background-color: #0f1117 !important;
}

.stApp, .stApp * {
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background: transparent !important;
}

/* ── Hide Streamlit chrome but KEEP sidebar toggle ── */
#MainMenu, footer { visibility: hidden !important; }

/* ── 1. Force the transparent header to stay alive permanently ── */
header[data-testid="stHeader"] {
    background: transparent !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 99999 !important;
}

/* ── 2. Kill the right-side toolbar (Deploy, Menu) ── */
[data-testid="stToolbar"] {
    display: none !important;
}

/* ── 3. Force the Sidebar Toggle Button to render boldly when closed ── */
[data-testid="collapsedControl"] {
    visibility: visible !important;
    display: flex !important;
    opacity: 1 !important;
    background-color: #161b22 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    top: 15px !important;
    left: 15px !important;
    color: #00f5d4 !important;
    transition: all 0.2s ease !important;
    z-index: 99999 !important;
}

[data-testid="collapsedControl"]:hover {
    background-color: rgba(0, 245, 212, 0.08) !important;
    border-color: rgba(0, 245, 212, 0.4) !important;
    box-shadow: 0 0 0 1px rgba(0, 245, 212, 0.15) !important;
}

[data-testid="collapsedControl"] svg {
    fill: #00f5d4 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #161b22 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdown"] p {
    color: #9ca3af !important;
    font-size: 0.85rem !important;
}

[data-testid="stSidebar"] h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #00f5d4 !important;
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
}

[data-testid="stSidebar"] h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #e5e7eb !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    margin-top: 1.5rem !important;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.06) !important;
}

/* ── Title ── */
.stApp h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 2rem !important;
    color: #00f5d4 !important;
    padding-bottom: 2px;
}

.stApp .stCaption, .stApp [data-testid="stCaptionContainer"] p {
    color: #6b7280 !important;
    font-size: 0.88rem !important;
}

/* ── Chat bubbles ── */
[data-testid="stChatMessage"] {
    background: #161b22 !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
    padding: 1rem 1.2rem !important;
    margin-bottom: 10px !important;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {
    color: #e5e7eb !important;
    line-height: 1.7 !important;
    font-size: 0.93rem !important;
}

[data-testid="stChatMessage"] strong {
    color: #00f5d4 !important;
}

[data-testid="stChatMessage"] code {
    background: rgba(0, 245, 212, 0.08) !important;
    color: #00f5d4 !important;
    border-radius: 4px !important;
    padding: 2px 6px !important;
}

[data-testid="stChatMessage"] [data-testid="stAvatar"] {
    border: 1px solid rgba(0, 245, 212, 0.3) !important;
    border-radius: 50% !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"],
[data-testid="stChatInputContainer"] {
    background: transparent !important;
}

[data-testid="stChatInput"] textarea,
[data-testid="stChatInputContainer"] textarea {
    background: #161b22 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: #e5e7eb !important;
    font-size: 0.93rem !important;
    padding: 12px 16px !important;
}

[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInputContainer"] textarea:focus {
    border-color: rgba(0, 245, 212, 0.4) !important;
    box-shadow: 0 0 0 1px rgba(0, 245, 212, 0.15) !important;
    outline: none !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #4b5563 !important;
}

[data-testid="stChatInput"] button,
[data-testid="stChatInputContainer"] button {
    background: #00f5d4 !important;
    border: none !important;
    border-radius: 8px !important;
}

[data-testid="stChatInput"] button:hover,
[data-testid="stChatInputContainer"] button:hover {
    background: #00dfc0 !important;
}

/* ── Kill red focus outlines ── */
[data-testid="stChatInput"] > div:focus-within,
[data-testid="stChatInputContainer"] > div:focus-within {
    outline: none !important;
    box-shadow: none !important;
    border: none !important;
}

[data-testid="stChatInput"] [data-baseweb="textarea"],
[data-testid="stChatInput"] [data-baseweb="base-input"] {
    outline: none !important;
    box-shadow: none !important;
    background-color: transparent !important;
}

[data-testid="stChatInput"] [data-baseweb="base-input"]:focus-within {
    border-color: rgba(0, 245, 212, 0.4) !important;
    box-shadow: none !important;
}

*:focus, *:active, *:focus-visible { outline: none !important; }
div[data-baseweb] [aria-invalid] { box-shadow: none !important; }

/* ── Buttons — clean pill ── */
.stButton > button {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #d1d5db !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    padding: 8px 20px !important;
    transition: all 0.15s ease !important;
}

.stButton > button:hover {
    background: rgba(0, 245, 212, 0.08) !important;
    border-color: rgba(0, 245, 212, 0.3) !important;
    color: #00f5d4 !important;
}

/* ── Text input ── */
.stTextInput > div > div > input {
    background: #161b22 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #e5e7eb !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.93rem !important;
    padding: 10px 14px !important;
    text-align: center !important;
}

.stTextInput > div > div > input::placeholder {
    color: #4b5563 !important;
}

.stTextInput > div > div > input:focus {
    border-color: rgba(0, 245, 212, 0.4) !important;
    box-shadow: 0 0 0 1px rgba(0, 245, 212, 0.1) !important;
    outline: none !important;
}

.stTextInput label {
    color: #6b7280 !important;
    font-size: 0.78rem !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}

/* ── Bottom dock ── */
[data-testid="stBottom"] {
    background: #0f1117 !important;
    background-color: #0f1117 !important;
    border-top: 1px solid rgba(255,255,255,0.06) !important;
}

[data-testid="stBottom"] > div {
    background: transparent !important;
    background-color: transparent !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #00f5d4 !important; }
[data-testid="stSpinnerContainer"] { color: #6b7280 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }

/* ── Tables — scoped to actual tables only ── */
.stDataFrame, .stTable {
    max-width: 100% !important;
    overflow-x: auto !important;
}

[data-testid="stChatMessage"] table {
    display: block !important;
    overflow-x: auto !important;
    white-space: nowrap !important;
    max-width: 100% !important;
}

/* ── Responsive ── */
@media (max-width: 768px) {
    .stApp h1 { font-size: 1.5rem !important; }
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.text_input(
        "TARGET COMPANY",
        placeholder="e.g. TCS, Reliance, Apple",
        key="target_company",
    )

    if st.button("🔄 New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown("### How it works")
    st.markdown(
        "Ask about any stock by name or ticker. "
        "The engine pulls live data from Yahoo Finance, "
        "scores it against four investment frameworks, "
        "and grounds its reasoning in classic investment books."
    )

    st.markdown("### Frameworks")
    st.markdown(
        "**Graham** — Deep value, margin of safety\n\n"
        "**Greenblatt** — Magic formula, capital efficiency\n\n"
        "**Dorsey** — Economic moats, financial health\n\n"
        "**Trajectory** — Revenue & earnings momentum"
    )

    st.markdown("---")
    st.markdown(
        "<p style='color: #4b5563; font-size: 0.75rem; text-align: center;'>"
        "Not financial advice. For educational and informational purposes only."
        "</p>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════
st.markdown("# AlphaConsensus Terminal")
st.caption("Quantitative investment analysis — Graham, Greenblatt, Dorsey, and Trajectory frameworks.")


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
# TOOL FUNCTIONS
# ──────────────────────────────────────────────

def show_stock_chart(ticker: str) -> dict:
    """Render a 13-month closing price chart for a stock directly in the terminal UI."""
    try:
        import pandas as pd
        import yfinance as yf
        import streamlit as st
        import altair as alt

        resolved = _resolve_ticker(ticker)
        resolved_upper = str(resolved).strip().upper()

        data_feed = yf.Ticker(resolved_upper).history(period="2y")
        if data_feed.empty and not resolved_upper.endswith((".NS", ".BSE")):
            data_feed = yf.Ticker(f"{resolved_upper}.NS").history(period="2y")
            if not data_feed.empty:
                resolved_upper = f"{resolved_upper}.NS"

        if not data_feed.empty:
            df = data_feed.tail(275).reset_index()
            df["Close"] = pd.to_numeric(df["Close"])

            y_min = float(df["Close"].min()) * 0.98
            y_max = float(df["Close"].max()) * 1.02

            st.write(f"### 📈 13-Month Trend: {resolved_upper}")

            chart = alt.Chart(df).mark_line(color="#00f5d4").encode(
                x=alt.X('Date:T', title='Date'),
                y=alt.Y('Close:Q', title='Price', scale=alt.Scale(domain=[y_min, y_max])),
                tooltip=['Date', 'Close']
            ).properties(height=400)

            st.altair_chart(chart, use_container_width=True)

            return {"success": f"Chart successfully rendered for {resolved_upper}."}
        else:
            return {"error": "Failed to fetch chart data."}

    except Exception as e:
        st.error(f"Chart Error: {str(e)}")
        return {"error": str(e)}


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
    Use this when the user asks about a specific company financials.

    Args:
        company_query: Stock ticker or company name, e.g. "AAPL", "RELIANCE.NS",
                       "TCS", "Mahindra", "Groww". Indian tickers should end in .NS
                       (NSE) or .BO (BSE). Common names like RIL, HDFC, SBI are
                       resolved automatically.
    """
    resolved_ticker = _resolve_ticker(company_query)

    try:
        stock = yf.Ticker(resolved_ticker)
        info = stock.info

        if not info or info.get("regularMarketPrice") is None:
            return {"error": f"No quantitative data found for '{company_query}'. "
                    f"Resolved to ticker [{resolved_ticker}] but it may be a "
                    f"private entity, mutual fund, or invalid."}

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


def get_historical_trends(company_query: str) -> dict:
    """Get 1-year historical trends (Year-over-Year) for Revenue, Net Income, and Debt.
    Use this when evaluating the immediate recent trajectory of a company.

    Args:
        company_query: Stock ticker or company name. Common names like TCS, Reliance,
                       Mahindra are resolved automatically.
    """
    resolved_ticker = _resolve_ticker(company_query)

    try:
        stock = yf.Ticker(resolved_ticker)
        income_stmt = stock.financials
        balance_sheet = stock.balance_sheet

        if income_stmt.empty or balance_sheet.empty:
            return {"error": "Historical financial statements not available."}

        recent_cols = sorted(income_stmt.columns, reverse=True)[:2]
        cols = sorted(recent_cols)

        if len(cols) < 2:
            return {"error": "Not enough historical data to establish a 1-year trend."}

        trends = {}

        def extract_metric(df, row_name):
            try:
                return [df.loc[row_name, col] for col in cols if pd.notna(df.loc[row_name, col])]
            except KeyError:
                return []

        rev_history = extract_metric(income_stmt, "Total Revenue")
        ni_history = extract_metric(income_stmt, "Net Income")
        debt_history = extract_metric(balance_sheet, "Total Debt")

        if len(rev_history) == 2:
            rev_growth = (rev_history[1] / rev_history[0]) - 1
            trends["1Y_Revenue_Growth"] = round(rev_growth * 100, 2)

        if len(ni_history) == 2:
            ni_growth = (ni_history[1] / ni_history[0]) - 1
            trends["1Y_NetIncome_Growth"] = round(ni_growth * 100, 2)

        if len(debt_history) == 2:
            debt_variance = ((debt_history[1] - debt_history[0]) / debt_history[0]) * 100
            trends["Debt_Growth_Trend"] = round(debt_variance, 2)

        return {
            "symbol": resolved_ticker,
            "data_years_analyzed": len(cols),
            "trends": trends
        }
    except Exception as e:
        return {"error": f"Trend data retrieval failed for [{resolved_ticker}]: {str(e)}"}


def get_financial_statements(ticker: str, statement: str) -> dict:
    """Get annual financial statements for a stock.
    Use this to answer questions about revenue, profits, expenses, assets,
    liabilities, debt levels, cash flow, margins, or multi-year growth trends.

    Args:
        ticker: Stock ticker symbol in Yahoo Finance format.
                Indian stocks need .NS suffix (e.g. RELIANCE.NS, TCS.NS).
                US stocks use plain symbol (e.g. AAPL, MSFT).
                Common names like Reliance, TCS, Infosys are also accepted.
        statement: Which financial statement to retrieve. Must be one of:
                   income   - Revenue, EBITDA, net income, operating expenses
                   balance  - Total assets, total debt, shareholder equity, cash
                   cashflow - Operating cash flow, capital expenditure, free cash flow
    """
    resolved = _resolve_ticker(ticker)
    try:
        t = yf.Ticker(resolved)

        if statement == "income":
            df = t.financials
        elif statement == "balance":
            df = t.balance_sheet
        elif statement == "cashflow":
            df = t.cashflow
        else:
            return {"error": f"Invalid statement type: '{statement}'. Use 'income', 'balance', or 'cashflow'."}

        if df is None or df.empty:
            return {"error": f"No {statement} statement data available for {resolved}"}

        data = {}
        for col in df.columns[:4]:
            year_key = str(col.date()) if hasattr(col, "date") else str(col)
            year_data = {}
            for idx in df.index:
                val = df.at[idx, col]
                if val is not None and val == val:
                    year_data[str(idx)] = round(float(val), 2)
            data[year_key] = year_data

        return {"ticker": resolved, "statement_type": statement, "data": data}

    except Exception as e:
        return {"error": f"Failed to get {statement} statement for {resolved}: {str(e)}"}


def get_price_history(ticker: str, period: str) -> dict:
    """Get historical stock price data with performance metrics.
    Use this when the user asks how a stock has performed over time,
    what the 52-week high/low is, price returns, volatility, or moving averages.

    Args:
        ticker: Stock ticker symbol (e.g. RELIANCE.NS, AAPL, TCS).
        period: Lookback period. Must be one of:
                1mo, 3mo, 6mo, 1y, 2y, 5y
    """
    resolved = _resolve_ticker(ticker)
    try:
        t = yf.Ticker(resolved)
        hist = t.history(period=period)

        if hist.empty:
            return {"error": f"No price history available for {resolved} over {period}"}

        start_price = float(hist["Close"].iloc[0])
        end_price = float(hist["Close"].iloc[-1])
        high = float(hist["High"].max())
        low = float(hist["Low"].min())
        total_return = ((end_price - start_price) / start_price) * 100
        avg_volume = float(hist["Volume"].mean())

        sma_50 = float(hist["Close"].tail(50).mean()) if len(hist) >= 50 else None
        sma_200 = float(hist["Close"].tail(200).mean()) if len(hist) >= 200 else None

        daily_returns = hist["Close"].pct_change().dropna()
        if len(daily_returns) > 1:
            volatility = float(daily_returns.std() * (252 ** 0.5) * 100)
        else:
            volatility = None

        return {
            "ticker": resolved,
            "period": period,
            "start_date": str(hist.index[0].date()),
            "end_date": str(hist.index[-1].date()),
            "start_price": round(start_price, 2),
            "current_price": round(end_price, 2),
            "period_high": round(high, 2),
            "period_low": round(low, 2),
            "total_return_pct": round(total_return, 2),
            "avg_daily_volume": int(avg_volume),
            "sma_50": round(sma_50, 2) if sma_50 else "Insufficient data",
            "sma_200": round(sma_200, 2) if sma_200 else "Insufficient data",
            "annualized_volatility_pct": round(volatility, 2) if volatility else "N/A",
        }

    except Exception as e:
        return {"error": f"Failed to get price history for {resolved}: {str(e)}"}


def get_analyst_recommendations(ticker: str) -> dict:
    """Get analyst recommendations, consensus rating, and price targets.
    Use this when the user asks what analysts think, buy/sell ratings,
    target prices, or broker recommendations.

    Args:
        ticker: Stock ticker symbol (e.g. RELIANCE.NS, AAPL, TCS).
    """
    resolved = _resolve_ticker(ticker)
    try:
        t = yf.Ticker(resolved)
        info = t.info
        result = {"ticker": resolved}

        result["current_price"] = round(
            float(info.get("currentPrice") or info.get("regularMarketPrice", 0)), 2
        )

        try:
            targets = t.analyst_price_targets
            if targets is not None:
                result["price_targets"] = {
                    "low": targets.get("low"),
                    "mean": targets.get("mean"),
                    "median": targets.get("median"),
                    "high": targets.get("high"),
                    "number_of_analysts": targets.get("numberOfAnalystOpinions"),
                }
            else:
                result["price_targets"] = "Not available"
        except Exception:
            result["price_targets"] = "Not available"

        try:
            recs = t.recommendations
            if recs is not None and not recs.empty:
                rec_list = []
                for _, row in recs.tail(12).iterrows():
                    rec_list.append({
                        "firm": str(row.get("Firm", row.get("firm", "Unknown"))),
                        "grade": str(row.get("To Grade", row.get("toGrade", "N/A"))),
                        "action": str(row.get("Action", row.get("action", "N/A"))),
                    })
                result["recent_recommendations"] = rec_list
            else:
                result["recent_recommendations"] = "Not available"
        except Exception:
            result["recent_recommendations"] = "Not available"

        try:
            summary = t.recommendations_summary
            if summary is not None and not summary.empty:
                result["recommendation_summary"] = summary.to_dict(orient="records")
        except Exception:
            pass

        return result

    except Exception as e:
        return {"error": f"Failed to get analyst data for {resolved}: {str(e)}"}


def get_stock_news(ticker: str) -> dict:
    """Get recent news articles about a stock.
    Use this when the user asks about recent news, developments, events,
    announcements, or what is happening with a company.

    Args:
        ticker: Stock ticker symbol (e.g. RELIANCE.NS, AAPL, TCS).
    """
    resolved = _resolve_ticker(ticker)
    try:
        t = yf.Ticker(resolved)
        news = t.news

        if not news:
            return {"ticker": resolved, "news": "No recent news available for this stock."}

        articles = []
        for item in news[:8]:
            published = item.get("providerPublishTime", "")
            if isinstance(published, (int, float)) and published > 0:
                from datetime import datetime
                try:
                    published = datetime.fromtimestamp(published).strftime("%Y-%m-%d %H:%M")
                except Exception:
                    published = str(published)

            articles.append({
                "title": item.get("title", "No title"),
                "publisher": item.get("publisher", "Unknown"),
                "link": item.get("link", ""),
                "published": str(published),
            })

        return {"ticker": resolved, "news_count": len(articles), "articles": articles}

    except Exception as e:
        return {"error": f"Failed to get news for {resolved}: {str(e)}"}


def get_ownership_info(ticker: str) -> dict:
    """Get major shareholders, institutional holders, and insider transactions.
    Use this when the user asks who owns the stock, promoter holding,
    FII/DII holding, institutional investors, or insider buying/selling.

    Args:
        ticker: Stock ticker symbol (e.g. RELIANCE.NS, AAPL, TCS).
    """
    resolved = _resolve_ticker(ticker)
    try:
        t = yf.Ticker(resolved)
        result = {"ticker": resolved}

        try:
            major = t.major_holders
            if major is not None and not major.empty:
                breakdown = {}
                for _, row in major.iterrows():
                    breakdown[str(row.iloc[1]).strip()] = str(row.iloc[0]).strip()
                result["holder_breakdown"] = breakdown
            else:
                result["holder_breakdown"] = "Not available"
        except Exception:
            result["holder_breakdown"] = "Not available"

        try:
            inst = t.institutional_holders
            if inst is not None and not inst.empty:
                holders = []
                for _, row in inst.head(10).iterrows():
                    pct = row.get("pctHeld", row.get("pctheld", None))
                    holders.append({
                        "name": str(row.get("Holder", row.get("holder", "Unknown"))),
                        "shares": int(row.get("Shares", row.get("shares", 0))),
                        "pct_held": round(float(pct) * 100, 2) if pct and pct == pct else "N/A",
                        "value": round(float(row.get("Value", row.get("value", 0))), 2),
                    })
                result["top_institutional_holders"] = holders
            else:
                result["top_institutional_holders"] = "Not available"
        except Exception:
            result["top_institutional_holders"] = "Not available"

        try:
            insider = t.insider_transactions
            if insider is not None and not insider.empty:
                txns = []
                for _, row in insider.head(10).iterrows():
                    shares = row.get("Shares", row.get("shares", 0))
                    txns.append({
                        "insider": str(row.get("Insider", row.get("insider", "Unknown"))),
                        "relation": str(row.get("Relation", row.get("relation", ""))),
                        "transaction": str(row.get("Transaction", row.get("transaction", ""))),
                        "shares": int(shares) if shares and shares == shares else 0,
                        "date": str(row.get("Start Date", row.get("startDate", ""))),
                    })
                result["recent_insider_transactions"] = txns
            else:
                result["recent_insider_transactions"] = "Not available"
        except Exception:
            result["recent_insider_transactions"] = "Not available"

        return result

    except Exception as e:
        return {"error": f"Failed to get ownership info for {resolved}: {str(e)}"}


def get_dividend_history(ticker: str) -> dict:
    """Get the full dividend payment history and growth trend for a stock.
    Use this when the user asks about dividend consistency, payout history,
    dividend growth, whether a company has paid dividends regularly, or
    dividend yield trends.

    Args:
        ticker: Stock ticker symbol (e.g. RELIANCE.NS, AAPL, TCS).
    """
    resolved = _resolve_ticker(ticker)
    try:
        t = yf.Ticker(resolved)
        divs = t.dividends

        if divs is None or divs.empty:
            return {
                "ticker": resolved,
                "has_dividends": False,
                "message": "No dividend history found. This company may not pay dividends.",
            }

        total_payments = len(divs)
        years_of_data = (divs.index[-1] - divs.index[0]).days / 365.25
        latest = float(divs.iloc[-1])

        annual = divs.resample("YE").sum()
        annual_dict = {}
        for date, val in annual.tail(5).items():
            annual_dict[str(date.year)] = round(float(val), 2)

        cagr = None
        if len(annual) >= 3:
            first_val = float(annual.iloc[-min(5, len(annual))])
            last_val = float(annual.iloc[-1])
            n = min(5, len(annual)) - 1
            if first_val > 0 and n > 0:
                cagr = round(((last_val / first_val) ** (1 / n) - 1) * 100, 2)

        info = t.info
        current_yield = info.get("dividendYield")
        if current_yield and current_yield == current_yield:
            current_yield = round(float(current_yield) * 100, 2)
        else:
            current_yield = "N/A"

        return {
            "ticker": resolved,
            "has_dividends": True,
            "total_payments": total_payments,
            "years_of_data": round(years_of_data, 1),
            "latest_dividend_per_share": round(latest, 2),
            "annual_dividends_last_5y": annual_dict,
            "dividend_cagr_pct": cagr if cagr else "Insufficient data for CAGR",
            "current_dividend_yield_pct": current_yield,
        }

    except Exception as e:
        return {"error": f"Failed to get dividend history for {resolved}: {str(e)}"}


def calculate_graham_value(ticker: str) -> dict:
    """Calculate Benjamin Grahams intrinsic value for a stock using his formula:
    V = EPS x (8.5 + 2g) x 4.4 / Y

    Where EPS = trailing earnings per share, g = expected growth rate (capped at 15%),
    Y = current AAA corporate bond yield (approximated at 5%).
    Graham recommended buying only when price is at least 33% below intrinsic value.

    Use this when the user asks for Graham valuation, intrinsic value,
    whether a stock is undervalued or overvalued, or margin of safety.

    Args:
        ticker: Stock ticker symbol (e.g. RELIANCE.NS, AAPL, TCS).
    """
    resolved = _resolve_ticker(ticker)
    try:
        t = yf.Ticker(resolved)
        info = t.info

        eps = info.get("trailingEps")
        if not eps or eps <= 0:
            return {
                "ticker": resolved,
                "error": f"Cannot compute Graham value: trailing EPS is {eps} (negative or unavailable). "
                         "Grahams formula only works for profitable companies.",
            }

        current_price = info.get("currentPrice") or info.get("regularMarketPrice")

        growth = info.get("earningsGrowth")
        if growth and growth > 0:
            g = min(growth * 100, 15.0)
        else:
            rev_growth = info.get("revenueGrowth")
            if rev_growth and rev_growth > 0:
                g = min(rev_growth * 100, 15.0)
            else:
                g = 5.0

        Y = 5.0
        intrinsic_value = eps * (8.5 + 2 * g) * 4.4 / Y

        if current_price and current_price > 0:
            margin = ((intrinsic_value - current_price) / current_price) * 100
            if margin > 33:
                verdict = "UNDERVALUED — meets Grahams 33% margin of safety"
            elif margin > 0:
                verdict = "SLIGHTLY UNDERVALUED — but does NOT meet 33% margin of safety"
            else:
                verdict = "OVERVALUED — price exceeds Graham intrinsic value"
        else:
            margin = None
            verdict = "Cannot determine (price data unavailable)"

        return {
            "ticker": resolved,
            "current_price": round(current_price, 2) if current_price else "N/A",
            "trailing_eps": round(eps, 2),
            "growth_rate_used_pct": round(g, 2),
            "aaa_bond_yield_used_pct": Y,
            "graham_intrinsic_value": round(intrinsic_value, 2),
            "margin_of_safety_pct": round(margin, 2) if margin is not None else "N/A",
            "verdict": verdict,
            "formula_breakdown": f"V = {round(eps,2)} x (8.5 + 2x{round(g,2)}) x 4.4 / {Y} = {round(intrinsic_value,2)}",
            "note": "Growth rate capped at 15% per Grahams conservatism. AAA yield approximated at 5%. "
                    "Graham recommended buying ONLY with >33% margin of safety.",
        }

    except Exception as e:
        return {"error": f"Failed to calculate Graham value for {resolved}: {str(e)}"}


def find_investments(market: str) -> dict:
    """Find the best investment candidates by scoring stocks against ALL 4 frameworks.
    Returns two tiers: Perfect Consensus (4/4 pass) and Strong Consensus (3/4 pass),
    with the top 3 from each tier and which frameworks each stock passed or failed.

    Use this when the user asks to find, discover, or recommend stocks to invest in,
    or asks which stocks are the best buys, or wants investment ideas.

    This screens ~80 stocks (Nifty 50 + US Large Cap 30). First call takes ~60 seconds
    to fetch all data; subsequent calls are cached for 6 hours.

    The 4 frameworks scored are:
    1. Graham — P/E <= 15 AND P/B <= 1.5 (deep value)
    2. Greenblatt — ROE > 15% AND Earnings Yield > 5% (magic formula / capital efficiency)
    3. Dorsey — ROE > 15% AND D/E < 50% (quality + financial health; moat is qualitative)
    4. Trajectory — (Revenue Growth > 0% OR Net Income Growth > 0%) AND (Debt Growth < 0% OR D/E < 50%)

    Args:
        market: Which market to screen. Must be one of:
                india — Nifty 50 stocks only
                us — US Large Cap stocks only
                all — Both markets combined
    """
    universe_data = _fetch_universe_data()

    tier_4 = []
    tier_3 = []
    tier_2 = []

    screened_count = 0

    for ticker, m in universe_data.items():
        is_indian = ticker.endswith(".NS") or ticker.endswith(".BO")
        if market == "india" and not is_indian:
            continue
        if market == "us" and is_indian:
            continue

        screened_count += 1

        pe = m.get("pe")
        pb = m.get("pb")
        roe = m.get("roe")
        de = m.get("de")
        ey = m.get("earnings_yield")
        rev_g = m.get("rev_growth")
        ni_g = m.get("ni_growth")
        debt_g = m.get("debt_growth")

        graham = bool(pe and pb and pe <= 15 and pb <= 1.5)
        greenblatt = bool(roe and ey and roe > 0.15 and ey > 5)
        dorsey = bool(roe and de is not None and roe > 0.15 and de < 50)

        growth_ok = (rev_g is not None and rev_g > 0) or (ni_g is not None and ni_g > 0)
        debt_ok = (debt_g is not None and debt_g < 0) or (de is not None and de < 50)
        trajectory = bool(growth_ok and debt_ok)

        results_map = {
            "Graham": graham,
            "Greenblatt": greenblatt,
            "Dorsey": dorsey,
            "Trajectory": trajectory,
        }
        score = sum(results_map.values())

        if score >= 2:
            entry = {
                "ticker": ticker,
                "name": m.get("name", ticker),
                "sector": m.get("sector", "N/A"),
                "price": round(m["price"], 2) if m.get("price") else "N/A",
                "pe": round(pe, 2) if pe else "N/A",
                "pb": round(pb, 2) if pb else "N/A",
                "roe_pct": round(roe * 100, 2) if roe else "N/A",
                "de_pct": round(de, 2) if de is not None else "N/A",
                "earnings_yield_pct": round(ey, 2) if ey else "N/A",
                "dividend_yield_pct": round(m["dividend_yield"] * 100, 2) if m.get("dividend_yield") else "N/A",
                "rev_growth_pct": rev_g if rev_g is not None else "N/A",
                "ni_growth_pct": ni_g if ni_g is not None else "N/A",
                "debt_growth_pct": debt_g if debt_g is not None else "N/A",
                "score": f"{score}/4",
                "passed": [name for name, passed in results_map.items() if passed],
                "failed": [name for name, passed in results_map.items() if not passed],
            }

            if score == 4:
                tier_4.append(entry)
            elif score == 3:
                tier_3.append(entry)
            else:
                tier_2.append(entry)

    def apply_rank_sum(tier_list):
        if not tier_list:
            return tier_list

        tier_list.sort(key=lambda x: x["pe"] if isinstance(x["pe"], (int, float)) else 9999)
        for i, item in enumerate(tier_list): item["value_rank"] = i + 1

        tier_list.sort(key=lambda x: x["roe_pct"] if isinstance(x["roe_pct"], (int, float)) else -9999, reverse=True)
        for i, item in enumerate(tier_list): item["quality_rank"] = i + 1

        tier_list.sort(key=lambda x: x["rev_growth_pct"] if isinstance(x["rev_growth_pct"], (int, float)) else -9999, reverse=True)
        for i, item in enumerate(tier_list): item["momentum_rank"] = i + 1

        for item in tier_list:
            item["composite_rank_score"] = item["value_rank"] + item["quality_rank"] + item["momentum_rank"]

        tier_list.sort(key=lambda x: x["composite_rank_score"])
        return tier_list

    tier_4 = apply_rank_sum(tier_4)
    tier_3 = apply_rank_sum(tier_3)
    tier_2 = apply_rank_sum(tier_2)

    return {
        "market": market,
        "stocks_screened": screened_count,
        "perfect_consensus_4_of_4": {
            "count": len(tier_4),
            "top_10": tier_4[:10],
            "investment_style": "Rare finds where deep value, capital efficiency, quality, and positive momentum ALL align. These represent the strongest quantitative buy signals across all philosophies.",
        },
        "strong_consensus_3_of_4": {
            "count": len(tier_3),
            "top_10": tier_3[:10],
            "investment_style": "Strong candidates that pass 3 frameworks. The single failing framework identifies the specific risk to monitor. Still well above average conviction.",
        },
        "moderate_consensus_2_of_4": {
            "count": len(tier_2),
            "top_10": tier_2[:10],
            "investment_style": "Partial alignment — these stocks show strength in 2 areas but have 2 gaps. May suit investors with specific theses (e.g. a cheap turnaround, or a quality grower at a premium). Requires more due diligence on the failing frameworks before committing.",
        },
        "note": "Screened Nifty 50 + US Large Cap 30. Dorsey moat is qualitative and checked only on quantitative criteria (ROE, D/E) here. Data cached for 6 hours. After presenting results, use search_book to explain WHY each investment style delivers returns, citing Graham, Greenblatt, and Dorsey.",
    }


# ──────────────────────────────────────────────
# TOOLS REGISTRY
# ──────────────────────────────────────────────
TOOLS = [
    search_book,
    get_stock_data,
    calculator,
    get_historical_trends,
    get_financial_statements,
    get_price_history,
    get_analyst_recommendations,
    get_stock_news,
    get_ownership_info,
    get_dividend_history,
    calculate_graham_value,
    find_investments,
    show_stock_chart,
]

tool_functions = {
    "search_book": search_book,
    "get_stock_data": get_stock_data,
    "calculator": calculator,
    "get_historical_trends": get_historical_trends,
    "get_financial_statements": get_financial_statements,
    "get_price_history": get_price_history,
    "get_analyst_recommendations": get_analyst_recommendations,
    "get_stock_news": get_stock_news,
    "get_ownership_info": get_ownership_info,
    "get_dividend_history": get_dividend_history,
    "calculate_graham_value": calculate_graham_value,
    "find_investments": find_investments,
    "show_stock_chart": show_stock_chart,
}


# ──────────────────────────────────────────────
# FALLBACK ROUTER
# ──────────────────────────────────────────────
def fallback_router(prompt):
    """Deterministic routing engine that triggers when the LLM is offline."""
    prompt_lower = prompt.lower()
    response_blocks = []

    potential_tickers = re.findall(r'\b[A-Z]{1,6}(?:\.NS)?\b', prompt)

    if "mahindra" in prompt_lower: potential_tickers.append("M&M.NS")
    if "apple" in prompt_lower: potential_tickers.append("AAPL")
    if "reliance" in prompt_lower or "ril" in prompt_lower: potential_tickers.append("RELIANCE.NS")

    tickers_to_check = list(set(potential_tickers))
    valid_stock_found = False

    for ticker in tickers_to_check:
        if ticker in ["I", "A", "THE", "WHAT", "WHY", "HOW", "IS", "YES", "NO"]:
            continue

        resolved = _resolve_ticker(ticker)
        data = get_stock_data(resolved)
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
# SYSTEM PROMPT
# ──────────────────────────────────────────────
SYSTEM_INSTRUCTION = """You are a highly structured Quantitative Investment Committee acting as a single agent.

Your knowledge base consists of four frameworks:
1. Benjamin Graham (Defensive Value, Margin of Safety)
2. Joel Greenblatt (The Magic Formula, Capital Efficiency)
3. Pat Dorsey (Economic Moats, Financial Health)
4. Historical Trajectory (1-Year Momentum & Growth)

You have 11 tools available. Pick the right combination for each question — you can call multiple tools in sequence.
1. search_book — Search The Intelligent Investor and other loaded books for Graham/Greenblatt/Dorsey investment philosophy. Use for conceptual or philosophical investing questions.
2. get_stock_data — Get current snapshot: price, P/E, P/B, market cap, dividend yield, 52-week range, sector. Use for quick overviews and valuation ratios.
3. calculator — Evaluate a math expression. Use for any arithmetic.
4. get_historical_trends — Get 1-year YoY trends for Revenue, Net Income, and Debt. Use for the Trajectory framework evaluation.
5. get_financial_statements — Get 4 years of income statement, balance sheet, OR cash flow data. Call with statement='income', 'balance', or 'cashflow'. You can call this multiple times with different statement types.
6. get_price_history — Get historical price performance over 1mo/3mo/6mo/1y/2y/5y. Returns total return, high/low, moving averages, and volatility.
7. get_analyst_recommendations — Get analyst buy/hold/sell ratings and consensus price targets.
8. get_stock_news — Get recent news headlines about a company.
9. get_ownership_info — Get major shareholders, institutional holders, and insider transactions.
10. get_dividend_history — Get complete dividend payment history, annual totals, growth rate, and yield.
11. calculate_graham_value — Compute Grahams intrinsic value formula (V = EPS x (8.5 + 2g) x 4.4/Y) and margin of safety.
12. find_investments — Screen ~80 stocks (Nifty 50 + US Large Cap 30) against ALL 4 frameworks and return two tiers: Perfect Consensus (4/4 pass) and Strong Consensus (3/4 pass), top 3 each. Use when the user asks to find, discover, or recommend stocks, or wants investment ideas. Call with market='india', 'us', or 'all'.
13. show_stock_chart — Renders a visual 13-month line chart of a stock's closing price directly in the UI. Use this whenever the user asks for a chart, graph, or visual trajectory.

TOOL SELECTION RULES:
- For a comprehensive stock analysis: call get_stock_data + get_historical_trends + get_financial_statements (income) + calculate_graham_value + search_book.
- For "is this stock a good investment" type questions: use at minimum get_stock_data + get_historical_trends + calculate_graham_value + search_book.
- For "how has X performed" questions: use get_price_history.
- For "any news about X" questions: use get_stock_news.
- For "what do analysts think" questions: use get_analyst_recommendations.
- For "does X pay dividends" or dividend history questions: use get_dividend_history.
- For "who owns X" or insider activity questions: use get_ownership_info.
- For "find me stocks" or "recommend stocks" or "where should I invest" or "best stocks" or "screen": call find_investments, THEN call search_book to explain WHY each investment tier is attractive. Follow the SCREENING OUTPUT PROTOCOL below.
- When comparing two stocks: call the relevant tools for BOTH tickers and synthesize.
- Always prefer calling a tool over guessing. If in doubt, call it.
- For "show me a chart" or "graph" questions: use show_stock_chart.

SCREENING OUTPUT PROTOCOL (use ONLY when find_investments is called):
After calling find_investments, you MUST also call search_book with queries like "margin of safety value investing" and "economic moat competitive advantage" and "magic formula return on capital" to ground your explanation in the actual books. Then present results as follows:

### Perfect Consensus (4/4 Frameworks Pass)
Show the top 3 stocks in a table with key metrics. Then explain:
- WHY this tier represents the strongest buy signal, citing specific concepts from the books (Graham margin of safety, Greenblatt capital efficiency, Dorsey moat durability)
- What kind of returns and risk profile an investor should expect (long-term compounding, downside protection)
- Use specific philosophy from the book passages you retrieved

### Strong Consensus (3/4 Frameworks Pass)
Show the top 3 stocks in a table with key metrics AND which framework they failed. Then explain:
- What the failing framework means as a specific risk (e.g., failing Graham means overvalued despite quality; failing Trajectory means growth is slowing)
- Why 3/4 is still a strong signal and what kind of investor this suits
- Ground the explanation in book concepts

### Moderate Consensus (2/4 Frameworks Pass)
Show the top 3 stocks in a table with key metrics AND which 2 frameworks they passed and which 2 they failed. Then explain:
- What combination of passes and fails this represents (e.g., passes Graham + Trajectory = cheap and growing but low quality; passes Greenblatt + Dorsey = high quality but expensive)
- Why this tier requires more caution and due diligence, but can still be attractive for investors with a specific thesis
- What additional research or conditions would strengthen conviction
- Ground the explanation in book concepts

If no stocks pass 4/4, say so clearly. If fewer than 3 pass in a tier, show however many exist.

CRITICAL RULES:
- For full analyses, you MUST call get_stock_data AND get_historical_trends.
- You MUST evaluate the thresholds silently before generating the output.
- Do NOT "think out loud" or correct yourself in the output.
- Do NOT copy the instruction text into your response.
- Each framework MUST be evaluated using ONLY its own criteria. Cross-contamination between frameworks is an error.

PASS/FAIL THRESHOLDS (Apply mechanically):
1. Graham: PASS IF (P/E <= 15) AND (P/B <= 1.5).
2. Greenblatt: PASS ONLY IF (ROE > 15%) AND (Earnings Yield > 5%).
3. Dorsey: PASS ONLY IF (ROE > 15%) AND (Debt/Equity < 50%) AND (You explicitly identify a business moat). The moat criterion is binary: does or does not have an identifiable moat. This is independent of Graham or Greenblatt results.
4. Trajectory: PASS ONLY IF (1Y Rev Growth > 0% OR 1Y Net Income Growth > 0%) AND (Debt Growth < 0% OR Current D/E < 50%).

VERDICT RULE:
- PASS CONDITION (YES): If ANY 2 out of the 4 frameworks PASS, the VERDICT decision is YES.
- VALUE EXCEPTION (YES): If Graham PASSES but the score is only 1/4, the VERDICT decision is YES (Deep Value).

EXECUTION PROTOCOL:
You MUST output your response EXACTLY following the template below. Use proper Markdown tables with pipes (|) and a blank line before the table. Do not add any text outside of this template.

<output_template>
### 1. Live Fundamentals & Trajectory

| Metric | Value | 1-Year YoY Trend |
| :--- | :--- | :--- |
| **Price** | [Value] | N/A |
| **P/E** | [Value] | N/A |
| **Forward P/E** | [Value] | N/A |
| **P/B** | [Value] | N/A |
| **ROE** | [Value]% | [Value]% Growth |
| **Debt/Equity** | [Value]% | [Value]% Growth |
| **Dividend Yield** | [Value]% | N/A |

### 2. The Committee Verdict

* **Graham:** P/E is [X] (Limit 15). P/B is [Y] (Limit 1.5). Yield is [Z]% (Limit >0%). -> **Verdict: [PASS or FAIL]**
* **Greenblatt:** ROE is [X]% (Limit >15%). Earnings Yield is [Y]% (Limit >5%). -> **Verdict: [PASS or FAIL]**
* **Dorsey:** ROE is [X]% (Limit >15%). D/E is [Y]% (Limit <50%). Moat: [Briefly name moat]. -> **Verdict: [PASS or FAIL]**
* **Trajectory:** Growth: [State Metric]. Debt: [State Metric]. -> **Verdict: [PASS or FAIL]**

### 3. Final Decision

* **Verdict:** [YES or NO]
* **Primary Driver:** [Strictly one sentence summarizing the 4-pillar vote count.]
* **Context:** [Strictly one sentence highlighting the main risk or overriding factor.]
* **Exit Strategy:** [Strictly one sentence detailing the exact quantitative or qualitative conditions that would trigger a SELL (e.g., P/E expands beyond historical norms, moat deteriorates, or growth turns negative). Only provide if Verdict is YES. If NO, put N/A.]
</output_template>"""


# ──────────────────────────────────────────────
# AGENT
# ──────────────────────────────────────────────
def sanitize_history(history):
    """Filters out malformed messages missing a role."""
    clean = []
    for msg in history:
        if isinstance(msg, dict):
            if msg.get("role") in ["user", "model"]:
                clean.append(msg)
        else:
            if hasattr(msg, 'role') and msg.role in ["user", "model"]:
                clean.append(msg)
    return clean


def agent_turn(user_message):
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

    raw_history = st.session_state.get("chat_history", [])
    history = sanitize_history(raw_history)

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
            last_error = str(e)
            if "429" in last_error or "RESOURCE_EXHAUSTED" in last_error:
                continue
            raise e

    raise Exception(f"All models rate-limited. Last error: {last_error}")


# ══════════════════════════════════════════════
# CHAT UI
# ══════════════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

USER_AVATAR = "👤"
AGENT_AVATAR = "📈"

# ── Reserve a container for all chat content (renders ABOVE buttons) ──
chat_area = st.container()

# ── Preset buttons (ALWAYS below all messages) ──
target = st.session_state.get("target_company", "").strip()

if target:
    st.markdown("")
    st.caption(f"Quick analysis for **{target}**")
    cols_per_row = 3
    for i in range(0, len(STOCK_PRESETS), cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            idx = i + j
            if idx < len(STOCK_PRESETS):
                label, template = STOCK_PRESETS[idx]
                with cols[j]:
                    if st.button(label, key=f"preset_{idx}", use_container_width=True):
                        st.session_state.pending_prompt = template.format(company=target)
                        st.rerun()

st.markdown("")
st.caption("Market screeners")
scr_cols = st.columns(3)
for i, (label, template) in enumerate(SCREENER_PRESETS):
    with scr_cols[i]:
        if st.button(label, key=f"screener_{i}", use_container_width=True):
            st.session_state.pending_prompt = template
            st.rerun()

# ── Chat input (pinned to bottom by Streamlit) ──
prompt = st.chat_input("Ask about any stock, or type a question...")

if not prompt and "pending_prompt" in st.session_state:
    prompt = st.session_state.pop("pending_prompt")

# ── All chat content renders inside the container (above buttons) ──
with chat_area:
    # Welcome text (shown only when chat is empty)
    if not st.session_state.messages:
        st.markdown("")

    # Display past messages
    for msg in st.session_state.messages:
        avatar = USER_AVATAR if msg["role"] == "user" else AGENT_AVATAR
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("model"):
                st.caption(f"⚡ {msg['model']}")

    # Handle new input (renders inside container = above buttons)
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=AGENT_AVATAR):
            with st.spinner("Analyzing..."):
                try:
                    answer, model_used = agent_turn(prompt)
                    st.markdown(answer)
                    st.caption(f"⚡ {model_used}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "model": model_used,
                    })

                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "All models rate-limited" in error_msg:
                        st.warning("API limit reached. Using fallback system...")
                        fallback_answer = fallback_router(prompt)
                        st.markdown(fallback_answer)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"*(Fallback)*\n\n{fallback_answer}",
                        })
                    else:
                        st.error(f"Error: {error_msg[:150]}")
