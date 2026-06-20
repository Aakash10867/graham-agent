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
# TICKER ALIAS MAP — Layer 1 of resolution
# Handles common abbreviations and Indian names
# that don't match Yahoo Finance ticker symbols.
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


def _search_yahoo(query: str) -> list[dict] | None:
    """Search Yahoo Finance for ticker matches.
    Layer 2: uses yf.Search() which handles Yahoo's cookie/crumb auth.
    Falls back to raw API as a last resort.
    """
    # Try yfinance's built-in Search (handles auth properly)
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

    # Last resort: raw API (may fail, but costs nothing to try)
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


def _resolve_ticker(query: str) -> str:
    """Central ticker resolution: alias map → yf.Search → raw fallback.
    Returns the best ticker string it can find.
    """
    key = query.strip().upper()

    # Layer 1: alias map (instant, zero API calls)
    if key in TICKER_ALIASES:
        return TICKER_ALIASES[key]

    # If it already has .NS or .BO suffix, trust it
    if ".NS" in key or ".BO" in key:
        return key

    # Layer 2: dynamic search via yfinance
    results = _search_yahoo(query)
    if results:
        # Prefer Indian exchange matches
        indian = next(
            (q for q in results if q.get("exchange") in ("NSI", "BSE", "NSE")),
            None,
        )
        if indian and indian.get("symbol"):
            return indian["symbol"]
        if results[0].get("symbol"):
            return results[0]["symbol"]

    # Last resort: return the input uppercased and hope yfinance can handle it
    return key


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
   KILL RED FOCUS OUTLINE (UPDATED)
   ═══════════════════════════════════════════════ */

/* 1. Nuke the Streamlit wrapper's default focus shadow */
[data-testid="stChatInput"] > div:focus-within,
[data-testid="stChatInputContainer"] > div:focus-within {
    outline: none !important;
    box-shadow: none !important;
    border: none !important;
}

/* 2. Strip internal baseweb outlines */
[data-testid="stChatInput"] [data-baseweb="textarea"] {
    outline: none !important;
    box-shadow: none !important;
}

[data-testid="stChatInput"] [data-baseweb="base-input"] {
    outline: none !important;
    box-shadow: none !important;
    border-color: rgba(0, 245, 212, 0.3) !important;
    background-color: transparent !important;
}

/* 3. Re-apply our neon glow instead of the red ring */
[data-testid="stChatInput"] [data-baseweb="base-input"]:focus-within {
    border-color: rgba(0, 245, 212, 0.6) !important;
    box-shadow: 0 0 15px rgba(0, 245, 212, 0.1) !important;
}

/* 4. Global fallback for any residual focus states */
*:focus, *:active, *:focus-visible {
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

/* ═══════════════════════════════════════════════
   FLOATING ACTION BUTTON (MOBILE & DESKTOP)
   ═══════════════════════════════════════════════ */
button:has(p:contains("🔄")) {
    position: fixed !important;
    top: 65px !important; 
    right: 15px !important;
    z-index: 99999 !important;
    width: auto !important;
    padding: 4px 16px !important;
    background: rgba(15, 12, 41, 0.85) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(0, 245, 212, 0.4) !important;
    border-radius: 50px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5), 0 0 10px rgba(0, 245, 212, 0.1) !important;
}

button:has(p:contains("🔄")):hover {
    background: rgba(0, 245, 212, 0.15) !important;
    border-color: #00f5d4 !important;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.title("🏛️ AlphaConsensus Terminal")
st.caption("Quantitative Multi-Agent Investment Committee. Operating on Graham, Greenblatt, and Dorsey frameworks.")

# ──────────────────────────────────────────────
# FLOATING RESET BUTTON
# ──────────────────────────────────────────────
if st.button("🔄 Reset"):
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
import pandas as pd
import numpy as np

def get_historical_trends(company_query: str) -> dict:
    """Get 1-year historical trends (Year-over-Year) for Revenue, Net Income, and Debt.
    Use this when evaluating the immediate recent trajectory of a company.
    """
    resolved_ticker = company_query.upper()
    # ... [Insert your existing ticker resolution block here if needed] ...
    
    try:
        stock = yf.Ticker(resolved_ticker)
        income_stmt = stock.financials
        balance_sheet = stock.balance_sheet
        
        if income_stmt.empty or balance_sheet.empty:
            return {"error": "Historical financial statements not available."}
            
        # yfinance returns columns newest to oldest. Grab the 2 most recent years.
        recent_cols = sorted(income_stmt.columns, reverse=True)[:2]
        # Sort chronologically (Oldest first, Newest second) so math flows forward
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
        
        # 1-Year YoY Growth calculations
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


def lookup_ticker(company_name: str) -> dict:
    """Find the stock ticker symbol for a company given its name.
    Use this FIRST whenever the user mentions a company by name
    without providing a ticker symbol.

    Args:
        company_name: The company name, e.g. "Groww", "Apple", "Tata Motors", "RIL"
    """
    key = company_name.strip().upper()

    # Check alias map first (instant, no API call)
    if key in TICKER_ALIASES:
        ticker = TICKER_ALIASES[key]
        return {
            "matches": [{
                "symbol": ticker,
                "name": company_name,
                "exchange": "Alias Map",
                "type": "EQUITY",
            }]
        }

    # Fall back to dynamic search
    results = _search_yahoo(company_name)
    if results:
        return {"matches": results}

    return {"error": f"No ticker found for '{company_name}'. "
            "It may not be publicly listed."}


# Register all tools
tool_functions = {
    "search_book": search_book,
    "get_stock_data": get_stock_data,
    "get_historical_trends": get_historical_trends,
    "calculator": calculator,
    "lookup_ticker": lookup_ticker,
}

TOOLS = [search_book, get_stock_data, get_historical_trends, calculator, lookup_ticker]


def fallback_router(prompt: str) -> str:
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

        # Use the alias-aware resolver instead of raw get_stock_data
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
# SYSTEM PROMPT & AGENT
# ──────────────────────────────────────────────

SYSTEM_INSTRUCTION = """You are a highly structured Quantitative Investment Committee acting as a single agent.

Your knowledge base consists of four frameworks:
1. Benjamin Graham (Defensive Value, Margin of Safety)
2. Joel Greenblatt (The Magic Formula, Capital Efficiency)
3. Pat Dorsey (Economic Moats, Financial Health)
4. Historical Trajectory (1-Year Momentum & Growth)

You have FIVE tools:
1. search_book — queries the texts of Graham, Greenblatt, and Dorsey.
2. get_stock_data — pulls live fundamental data.
3. get_historical_trends — pulls 1-Year YoY growth for Revenue, Net Income, and Debt.
4. calculator — evaluates mathematical expressions.
5. lookup_ticker — finds the stock ticker symbol. Use FIRST when given a company name.

CRITICAL RULES:
- You MUST call `get_stock_data` AND `get_historical_trends`.
- You MUST evaluate the thresholds silently before generating the output. 
- Do NOT "think out loud" or correct yourself in the output.
- Do NOT copy the instruction text into your response.

PASS/FAIL THRESHOLDS (Apply mechanically):
1. Graham: PASS IF (P/E ≤ 15) AND (P/B ≤ 1.5). 
2. Greenblatt: PASS ONLY IF (ROE > 15%) AND (Earnings Yield > 5%).
3. Dorsey: PASS ONLY IF (ROE > 15%) AND (Debt/Equity < 50%) AND (You explicitly identify a business moat).
4. Trajectory: PASS ONLY IF (1Y Rev Growth > 0% OR 1Y Net Income Growth > 0%) AND (Debt Growth < 0% OR Current D/E < 50%).

DECISION RULE:
- PASS CONDITION (YES): If ANY 2 out of the 4 frameworks PASS, the final decision is YES. 
- VALUE EXCEPTION (YES): If Graham PASSES but the score is only 1/4, the decision is YES (Deep Value).

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
</output_template>"""

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

# ──────────────────────────────────────────────
# NEW: BOTTOM RESET BUTTON
# ──────────────────────────────────────────────
if st.session_state.messages:
    st.write("") # Small spacer
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Clear & Reset Terminal", key="bottom_reset", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()
# ──────────────────────────────────────────────



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
