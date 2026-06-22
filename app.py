"""
DeepMoat
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

from supabase import create_client
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

def get_supabase():
    client = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    if st.session_state.get("sb_access_token"):
        try:
            resp = client.auth.set_session(
                st.session_state.sb_access_token,
                st.session_state.sb_refresh_token
            )
            # Update tokens in case set_session refreshed them
            st.session_state.sb_access_token = resp.session.access_token
            st.session_state.sb_refresh_token = resp.session.refresh_token
        except Exception:
            # Refresh token expired — force re-login
            st.session_state.sb_access_token = None
            st.session_state.sb_refresh_token = None
            st.session_state.sb_user_email = None
            st.session_state.sb_user_id = None
    return client

def allocate_shares(stocks, sip_amount):
    result = []
    for s in stocks:
        price = s["price"]
        target = sip_amount * s["allocation_pct"] / 100
        shares = int(target // price) if price > 0 else 0
        result.append({**s, "shares": shares, "actual_amount": shares * price})

    remaining = sip_amount - sum(s["actual_amount"] for s in result)

    while remaining > 0:
        best = None
        best_gap = -1
        for s in result:
            target = sip_amount * s["allocation_pct"] / 100
            gap = target - s["actual_amount"]
            if s["price"] <= remaining and gap > best_gap:
                best = s
                best_gap = gap
        if best is None:
            break
        best["shares"] += 1
        best["actual_amount"] = best["shares"] * best["price"]
        remaining = sip_amount - sum(s["actual_amount"] for s in result)

    return result, round(remaining, 2)

def find_replacement_candidates(investor_type, time_horizon, exclude_tickers, current_sectors):
    """Find replacement stocks when review flags sells."""
    df = universe_df.copy()

    if "quality_pass" in df.columns:
        df = df[df["quality_pass"] != False]
    df = df[df["years_of_data"] >= 2]
    df = df[pd.notna(df["pe"]) & pd.notna(df["roe_pct"]) & pd.notna(df["de"])]
    df = df[df["pe"] > 0]

    # Same profile filtering as get_sip_candidates
    if investor_type == "defensive":
        df = df[df["score"] >= 3]
        mask = df["graham_pass"] == True
        if mask.sum() >= 5:
            df = df[mask]
    elif investor_type == "enterprising":
        df = df[df["score"] >= 2]
        mask = df["trajectory_pass"] == True
        if mask.sum() >= 5:
            df = df[mask]
    else:
        df = df[df["score"] >= 2]
        mask = (df["greenblatt_pass"] == True) | (df["dorsey_pass"] == True)
        if mask.sum() >= 5:
            df = df[mask]

    if time_horizon == "short":
        high_score = df[df["score"] >= 3]
        if len(high_score) >= 5:
            df = high_score

    # Exclude stocks already in portfolio
    df = df[~df["ticker"].isin(exclude_tickers)]

    # Exclude sectors at the 2-stock cap
    from collections import Counter
    sector_counts = Counter(current_sectors)
    full_sectors = [s for s, c in sector_counts.items() if c >= 2]
    if full_sectors:
        df = df[~df["sector"].isin(full_sectors)]

    # Sort
    df = df.copy()
    df["_sort_score"] = -df["score"]
    df["_sort_pe"] = df["pe"].apply(lambda x: x if pd.notna(x) else 9999)
    df["_sort_roe"] = df["roe_pct"].apply(lambda x: -x if pd.notna(x) else 9999)
    df = df.sort_values(["_sort_score", "_sort_pe", "_sort_roe"])

    candidates = []
    for _, row in df.head(5).iterrows():
        candidates.append({
            "ticker": row["ticker"],
            "name": row.get("name", "N/A") if pd.notna(row.get("name")) else "N/A",
            "sector": row.get("sector", "N/A") if pd.notna(row.get("sector")) else "N/A",
            "price": round(row["price"], 2) if pd.notna(row.get("price")) else 0,
            "score": int(row["score"]),
            "pe": round(row["pe"], 2) if pd.notna(row.get("pe")) else "N/A",
            "roe_pct": round(row["roe_pct"], 2) if pd.notna(row.get("roe_pct")) else "N/A",
        })
    return candidates

def get_nifty_return(days):
    """Get Nifty 50 return over a given number of days."""
    try:
        nifty = yf.Ticker("^NSEI")
        hist = nifty.history(period=f"{max(days + 10, 30)}d")
        if len(hist) < 2:
            return None
        end_price = float(hist["Close"].iloc[-1])
        start_idx = max(0, len(hist) - days)
        start_price = float(hist["Close"].iloc[start_idx])
        return round(((end_price - start_price) / start_price) * 100, 2)
    except Exception:
        return None


def build_review_context(holdings, port):
    """Gather enriched data per holding: market context, earnings quality, ROE trend, book passage."""
    today = datetime.date.today()
    try:
        created = datetime.date.fromisoformat(str(port["created_at"])[:10])
        holding_days = (today - created).days
    except Exception:
        holding_days = 30

    nifty_return = get_nifty_return(holding_days)

    enriched = []
    for h in holdings:
        ticker = h["ticker"]
        entry_price = h.get("price_at_entry") or 0
        entry_score = h.get("score_at_entry") or 0
        shares = h.get("shares") or 0

        try:
            cinfo = yf.Ticker(ticker).info
            now_price = cinfo.get("currentPrice") or cinfo.get("regularMarketPrice") or 0
        except Exception:
            now_price = 0

        urow = universe_df[universe_df["ticker"] == ticker]
        now_score = int(urow["score"].iloc[0]) if len(urow) and pd.notna(urow["score"].iloc[0]) else 0

        roe_trend = []
        for y in ["roe_y0", "roe_y1", "roe_y2", "roe_y3"]:
            if len(urow) and y in urow.columns and pd.notna(urow[y].iloc[0]):
                roe_trend.append(round(float(urow[y].iloc[0]), 2))

        quality = get_earnings_quality_metrics(ticker)
        if "error" not in quality:
            quality_flags = quality.get("anomaly_flags", ["Unable to check"])
            cash_conversion = quality.get("cash_conversion_ratio", "N/A")
        else:
            quality_flags = ["Unable to check"]
            cash_conversion = "N/A"

        stock_return = ((now_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
        pnl = (now_price - entry_price) * shares if entry_price > 0 else 0
        score_change = now_score - entry_score
        market_relative = round(stock_return - nifty_return, 2) if nifty_return is not None else None
        roe_declining = len(roe_trend) >= 3 and roe_trend[0] < roe_trend[-1]
        has_red_flags = any("RED FLAG" in f for f in quality_flags) if isinstance(quality_flags, list) else False

        # Pattern-specific book query
        if has_red_flags:
            book_query = "Graham warnings about earnings quality non-recurring income value traps"
        elif score_change <= -2 and market_relative is not None and market_relative > -5:
            book_query = "Dorsey signs of eroding economic moat competitive advantage deterioration"
        elif stock_return < -10 and nifty_return is not None and nifty_return < -5:
            book_query = "Graham holding through market declines Mr Market temporary price drops"
        elif roe_declining:
            book_query = "Dorsey declining return on equity moat erosion when to sell"
        elif score_change >= 1:
            book_query = "Graham margin of safety increases buying more undervalued stocks"
        elif score_change == 0 and stock_return > 20:
            book_query = "Greenblatt when to take profits selling appreciated stocks"
        else:
            book_query = "Graham intelligent investor patience holding quality companies"

        book_result = search_book(book_query)
        book_passage = ""
        if "error" not in book_result:
            passages = book_result["passages"].split("\n\n")
            book_passage = passages[0][:500] if passages else ""

        enriched.append({
            "ticker": ticker, "name": h.get("name") or ticker, "sector": h.get("sector", ""),
            "shares": shares, "entry_price": entry_price, "now_price": now_price,
            "entry_score": entry_score, "now_score": now_score, "score_change": score_change,
            "stock_return": round(stock_return, 2), "pnl": round(pnl, 0),
            "nifty_return": nifty_return, "market_relative": market_relative,
            "roe_trend": roe_trend, "roe_declining": roe_declining,
            "quality_flags": quality_flags, "cash_conversion": cash_conversion,
            "has_red_flags": has_red_flags, "book_query": book_query,
            "book_passage": book_passage, "holding_days": holding_days,
            "holding_id": h.get("id"),
        })

    return enriched


def generate_review_recommendations(enriched_holdings, investor_type, time_horizon):
    """LLM-powered review recommendations grounded in book philosophy."""
    holdings_text = ""
    for i, h in enumerate(enriched_holdings):
        holdings_text += (
            f"\nStock {i+1}: {h['name']} ({h['ticker']})\n"
            f"- Shares: {h['shares']}, Entry: INR {h['entry_price']:.2f}, Now: INR {h['now_price']:.2f}\n"
            f"- Stock return: {h['stock_return']:+.1f}%, Nifty return: {h['nifty_return']}%, Market-relative: {h['market_relative']}%\n"
            f"- Score: {h['entry_score']} to {h['now_score']} (change: {h['score_change']:+d})\n"
            f"- ROE trend (recent to oldest): {h['roe_trend']}\n"
            f"- Earnings quality: {', '.join(h['quality_flags']) if isinstance(h['quality_flags'], list) else h['quality_flags']}\n"
            f"- Cash conversion ratio: {h['cash_conversion']}\n"
            f"- Held for: {h['holding_days']} days\n"
            f"- Relevant book passage: {h['book_passage']}\n"
        )

    review_prompt = (
        f"You are the DeepMoat Investment Committee reviewing a {investor_type} investor's "
        f"portfolio with a {time_horizon}-term horizon.\n\n"
        f"For each stock below, provide a recommendation.\n\n"
        f"DECISION FRAMEWORK (apply in order):\n"
        f"1. RED FLAGS OVERRIDE: If earnings quality has RED FLAGS, recommend SELL ALL. Cite Graham on value traps.\n"
        f"2. MOAT EROSION: If ROE declined for 3+ years AND stock underperformed market, recommend SELL HALF. Cite Dorsey.\n"
        f"3. MARKET EFFECT: If stock dropped BUT Nifty also dropped similarly (within 5%), recommend HOLD. "
        f"Cite Graham on Mr. Market. The business hasn't changed.\n"
        f"4. THESIS INTACT: If score stable or improved AND no red flags AND cash conversion > 0.5, "
        f"recommend HOLD or BUY MORE. Cite the relevant framework.\n"
        f"5. OVERVALUATION: If stock gained >30% and score dropped, recommend HOLD but note reduced margin of safety.\n"
        f"6. INVESTOR PROFILE: "
        f"{'Be conservative. Prefer HOLD over BUY MORE, SELL sooner on red flags.' if investor_type == 'defensive' else 'Balance risk and reward.' if investor_type == 'balanced' else 'Tolerate volatility. HOLD through short-term drops if moat is intact.'}\n\n"
        f"{holdings_text}\n\n"
        f"Respond ONLY with a JSON array (no markdown, no backticks, no preamble). Each element:\n"
        f'{{"ticker": "TICKER.NS", "action": "HOLD", "sell_qty": 0, "reasoning": "2-3 sentences grounded in Graham/Greenblatt/Dorsey.", "confidence": "high"}}\n'
        f"action must be one of: SELL ALL, SELL HALF, HOLD, BUY MORE\n"
        f"sell_qty: number of shares to sell (0 for HOLD/BUY MORE, all shares for SELL ALL, half for SELL HALF)\n"
    )

    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    for model_name in FREE_MODELS:
        try:
            response = client.models.generate_content(model=model_name, contents=review_prompt)
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            if text.startswith("json"):
                text = text[4:].strip()
            return json.loads(text)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                continue
            break
    return None



def register_portfolio(portfolio_name: str, investor_type: str, sip_amount: int, time_horizon: str, review_days: int = 90, stocks_json: str = "[]") -> dict:
    """Register a finalized SIP portfolio so the user can save it to their account.
    Call this ONLY after you have presented the final portfolio table with all stocks and allocations.

    Args:
        portfolio_name: Short descriptive name, e.g. 'Conservative Growth SIP - June 2026'
        investor_type: The investor profile - defensive, balanced, or enterprising
        sip_amount: Monthly SIP amount in INR
        time_horizon: Investment time horizon from the questionnaire
        review_days: Number of days between portfolio reviews. Convert the user preference to days. Monthly=30, Quarterly=90, Semi-annually=180, Annually=365. Any number is valid.
        stocks_json: A JSON string representing a list of stock objects. Each object must have keys: ticker (str), name (str), sector (str), allocation_pct (number). Example: [{"ticker":"TCS.NS","name":"TCS","sector":"Technology","allocation_pct":20}]
    """
    try:
        stocks = json.loads(stocks_json) if isinstance(stocks_json, str) else stocks_json
    except json.JSONDecodeError:
        return {"error": f"Could not parse stocks_json: {stocks_json[:200]}"}

    if not stocks:
        return {"error": "No stocks provided."}

    st.session_state.pending_portfolio = {
        "name": portfolio_name,
        "investor_type": investor_type,
        "sip_amount": sip_amount,
        "time_horizon": time_horizon,
        "review_days": int(review_days),
        "stocks": stocks
    }
    return {"status": f"Portfolio '{portfolio_name}' registered with {len(stocks)} stocks. Review every {review_days} days. The user can now save it."}


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


# ══════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="DeepMoat",
    page_icon="logo.svg",
    layout="centered",
)

for _key in ["sb_access_token", "sb_refresh_token", "sb_user_email", "sb_user_id"]:
    if _key not in st.session_state:
        st.session_state[_key] = None
if "pending_portfolio" not in st.session_state:
    st.session_state.pending_portfolio = None
if "pending_retry" not in st.session_state:
    st.session_state.pending_retry = None


if "sb_view_mode" not in st.session_state:
    st.session_state.sb_view_mode = "chat"

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
     "Find the best Indian stocks to invest in right now. Show me which stocks pass all 4 frameworks and which pass 3 out of 4 and which pass 2 out of 4, with upto top 10 from each tier. Explain why each tier is a good investment using the book philosophies."),
    ("💎 Find Hidden Gems",
     "Find hidden gem stocks — small and mid cap Indian companies outside the Nifty 50 that pass at least 3 out of 4 frameworks. Show top 10 with key metrics. Explain why each is a good investment using book philosophies."),
    ("💼 Build SIP Portfolio",
     "I want to build a SIP portfolio. Help me pick the right stocks based on my goals and investment amount."),
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

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

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
    # ── Auth ──
    if st.session_state.sb_user_email is None:
        auth_mode = st.radio(
            "Account", ["Login", "Sign Up"],
            horizontal=True, label_visibility="collapsed"
        )
        auth_email = st.text_input("Email", key="auth_email_input")
        auth_password = st.text_input("Password", type="password", key="auth_password_input")

        if auth_mode == "Login":
            if st.button("Log In", use_container_width=True):
                if not auth_email or not auth_password:
                    st.warning("Enter email and password.")
                else:
                    try:
                        sb = get_supabase()
                        resp = sb.auth.sign_in_with_password({
                            "email": auth_email,
                            "password": auth_password
                        })
                        st.session_state.sb_access_token = resp.session.access_token
                        st.session_state.sb_refresh_token = resp.session.refresh_token
                        st.session_state.sb_user_email = resp.user.email
                        st.session_state.sb_user_id = str(resp.user.id)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Login failed: {e}")
        else:
            if st.button("Sign Up", use_container_width=True):
                if not auth_email or not auth_password:
                    st.warning("Enter email and password.")
                elif len(auth_password) < 6:
                    st.warning("Password must be at least 6 characters.")
                else:
                    try:
                        sb = get_supabase()
                        resp = sb.auth.sign_up({
                            "email": auth_email,
                            "password": auth_password
                        })
                        st.session_state.sb_access_token = resp.session.access_token
                        st.session_state.sb_refresh_token = resp.session.refresh_token
                        st.session_state.sb_user_email = resp.user.email
                        st.session_state.sb_user_id = str(resp.user.id)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Sign up failed: {e}")
    else:
        st.caption(f"Logged in as {st.session_state.sb_user_email}")
        if st.session_state.sb_view_mode == "chat":
            if st.button("📁 My Portfolios", use_container_width=True):
                st.session_state.sb_view_mode = "portfolios"
                st.rerun()
        else:
            if st.button("← Back to Chat", use_container_width=True):
                st.session_state.sb_view_mode = "chat"
                st.rerun()
        if st.button("Log Out", use_container_width=True):
            try:
                sb = get_supabase()
                sb.auth.sign_out()
            except Exception:
                pass
            st.session_state.sb_access_token = None
            st.session_state.sb_refresh_token = None
            st.session_state.sb_user_email = None
            st.session_state.sb_user_id = None
            st.rerun()

    st.divider()
    st.markdown("Multi-framework investment analysis powered by Graham, Greenblatt, Dorsey, and momentum scoring.")

    st.markdown("---")

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
# Lock the logo and title into a tight horizontal grid
h_col1, h_col2 = st.columns([1, 11])

with h_col1:
    st.image("logo.svg", width=54) # Precise, discrete sizing

with h_col2:
    st.markdown("<h1 style='margin-top: -15px; padding-bottom: 0px;'>DeepMoat</h1>", unsafe_allow_html=True)

st.caption("Quantitative investment analysis — Graham, Greenblatt, Dorsey, and Trajectory frameworks.")
st.markdown("---")

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

import os
import pandas as pd
import streamlit as st

# Anchor the path absolutely relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "universe_scored.csv")

@st.cache_data(show_spinner=False)
def load_universe(file_path: str):
    """
    Passing the file_path as an argument allows Streamlit to hash the 
    file metadata. If the CSV is updated, the cache invalidates automatically.
    """
    if not os.path.exists(file_path):
        # Fallback empty dataframe to prevent fatal app crashes if file is missing
        st.error(f"Critical System Error: {file_path} not found.")
        return pd.DataFrame()
        
    return pd.read_csv(file_path)

# Initialize the global dataframe safely
universe_df = load_universe(CSV_PATH)


# ──────────────────────────────────────────────
# TOOL FUNCTIONS
# ──────────────────────────────────────────────

def get_earnings_quality_metrics(ticker: str) -> dict:
    resolved = _resolve_ticker(ticker)
    try:
        t = yf.Ticker(resolved)
        inc = t.financials
        cf = t.cashflow

        if inc.empty or cf.empty:
            return {"error": "Financial statements unavailable."}

        def get_latest(df, row_names):
            for name in row_names:
                if name in df.index:
                    val = df.loc[name].dropna()
                    if not val.empty:
                        return float(val.iloc[0])
            return 0.0

        def get_series(df, row_names, n=3):
            """Get up to n years of a metric."""
            for name in row_names:
                if name in df.index:
                    vals = df.loc[name].dropna().tolist()
                    return [float(v) for v in vals[:n]]
            return []

        net_income = get_latest(inc, ['Net Income', 'Net Income Common Stockholders'])
        ocf = get_latest(cf, ['Operating Cash Flow', 'Total Cash From Operating Activities'])
        operating_income = get_latest(inc, ['Operating Income', 'EBIT'])
        total_revenue = get_latest(inc, ['Total Revenue'])

        if net_income == 0:
            return {"error": "Net income is 0 or missing."}

        flags = []

        # CHECK 1: Cash conversion
        cash_conversion = ocf / net_income if net_income > 0 else 0
        if cash_conversion < 0.5 and net_income > 0:
            flags.append(
                f"RED FLAG: Cash conversion is {round(cash_conversion, 2)}. "
                f"Only {round(cash_conversion * 100)}% of reported profit is real cash."
            )

        # CHECK 2: Earnings spike (net income vs prior years)
        ni_series = get_series(inc, ['Net Income', 'Net Income Common Stockholders'], n=4)
        if len(ni_series) >= 3:
            prior_avg = sum(ni_series[1:]) / len(ni_series[1:])
            current = ni_series[0]
            if prior_avg > 0 and current > 3 * prior_avg:
                spike_multiple = round(current / prior_avg, 1)
                flags.append(
                    f"RED FLAG: Net income is {spike_multiple}x the prior-year average. "
                    f"Current: {current:,.0f}, Prior avg: {prior_avg:,.0f}. "
                    f"Likely driven by non-recurring event."
                )

        # CHECK 3: Non-operating income gap
        if operating_income > 0 and net_income > 0:
            non_op_gap = (net_income - operating_income) / net_income
            if non_op_gap > 0.4:
                flags.append(
                    f"RED FLAG: {round(non_op_gap * 100)}% of net income comes from "
                    f"below the operating line (non-operational sources). "
                    f"Operating income: {operating_income:,.0f}, Net income: {net_income:,.0f}."
                )

        # ALSO check the legacy unusual items field (catch it if available)
        unusual_items = get_latest(inc, ['Unusual Items', 'Extraordinary Items',
                                         'Special Items', 'Other Special Charges'])
        unusual_pct = abs(unusual_items / net_income) * 100 if net_income != 0 else 0

        if unusual_pct > 20:
            flags.append(
                f"RED FLAG: Tagged non-recurring items are {round(unusual_pct, 1)}% of net income."
            )

        return {
            "ticker": resolved,
            "net_income_reported": net_income,
            "operating_income": operating_income,
            "operating_cash_flow": ocf,
            "cash_conversion_ratio": round(cash_conversion, 2),
            "unusual_items_pct_of_income": round(unusual_pct, 2),
            "anomaly_flags": flags if flags else ["CLEAN: No major anomalies detected."],
            "directive": "If ANY RED FLAG is present, reject positive framework scores."
        }
    except Exception as e:
        return {"error": f"Failed anomaly check: {str(e)}"}


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

        result = {
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

        # Auto-inject earnings quality — LLM sees flags whether it asks or not
        quality = get_earnings_quality_metrics(resolved_ticker)
        if "error" not in quality:
            result["earnings_quality"] = {
                "cash_conversion_ratio": quality["cash_conversion_ratio"],
                "unusual_items_pct": quality["unusual_items_pct_of_income"],
                "anomaly_flags": quality["anomaly_flags"],
            }

        return result
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
    """Find the best investment candidates from the pre-scored universe of ~4500 Indian stocks.
    Reads from universe_scored.csv which is updated monthly via universe_updater.py.

    Use this when the user asks to find, discover, or recommend stocks to invest in,
    or asks which stocks are the best buys, or wants investment ideas.

    The 4 frameworks scored are:
    1. Graham — P/E <= 15 AND P/B <= 1.5 (deep value)
    2. Greenblatt — ROE > 15% AND Earnings Yield > 5% (magic formula / capital efficiency)
    3. Dorsey — ROE > 15% AND D/E < 50% (quality + financial health; moat is qualitative)
    4. Trajectory — (Revenue Growth > 0% OR Net Income Growth > 0%) AND (Debt Growth < 0% OR D/E < 50%)

    Args:
        market: Which market to screen. Use 'india' or 'all' (both return Indian stocks).
    """
    df = universe_df
    # Strip value traps pre-flagged by universe_updater
    if "quality_pass" in df.columns:
        df = df[df["quality_pass"] != False]

    tier_4 = df[df["score"] == 4].copy()
    tier_3 = df[df["score"] == 3].copy()
    tier_2 = df[df["score"] == 2].copy()


    

    # Rank-sum sorting within each tier (value + quality + momentum)
    def apply_rank_sort(tier_df):
        if tier_df.empty:
            return tier_df
        t = tier_df.copy()
        t["_pe_sort"] = t["pe"].apply(lambda x: x if pd.notna(x) else 9999)
        t["_roe_sort"] = t["roe_pct"].apply(lambda x: -x if pd.notna(x) else 9999)
        t["_rev_sort"] = t["rev_growth"].apply(lambda x: -x if pd.notna(x) else 9999)
        t = t.sort_values(["_pe_sort", "_roe_sort", "_rev_sort"])
        return t.drop(columns=["_pe_sort", "_roe_sort", "_rev_sort"])

    tier_4 = apply_rank_sort(tier_4)
    tier_3 = apply_rank_sort(tier_3)
    tier_2 = apply_rank_sort(tier_2)

    def to_list(tier_df, max_n=10):
        entries = []
        for _, row in tier_df.head(max_n).iterrows():
            entries.append({
                "ticker": row["ticker"],
                "name": row.get("name", row["ticker"]) if pd.notna(row.get("name")) else row["ticker"],
                "sector": row.get("sector", "N/A") if pd.notna(row.get("sector")) else "N/A",
                "price": round(row["price"], 2) if pd.notna(row.get("price")) else "N/A",
                "pe": round(row["pe"], 2) if pd.notna(row.get("pe")) else "N/A",
                "pb": round(row["pb"], 2) if pd.notna(row.get("pb")) else "N/A",
                "roe_pct": round(row["roe_pct"], 2) if pd.notna(row.get("roe_pct")) else "N/A",
                "de_pct": round(row["de"], 2) if pd.notna(row.get("de")) else "N/A",
                "earnings_yield_pct": round(row["earnings_yield"], 2) if pd.notna(row.get("earnings_yield")) else "N/A",
                "dividend_yield_pct": round(row["dividend_yield_pct"], 2) if pd.notna(row.get("dividend_yield_pct")) else "N/A",
                "rev_growth_pct": round(row["rev_growth"], 2) if pd.notna(row.get("rev_growth")) else "N/A",
                "ni_growth_pct": round(row["ni_growth"], 2) if pd.notna(row.get("ni_growth")) else "N/A",
                "debt_growth_pct": round(row["debt_growth"], 2) if pd.notna(row.get("debt_growth")) else "N/A",
                "score": f"{int(row['score'])}/4",
                "passed": [f for f in ["Graham", "Greenblatt", "Dorsey", "Trajectory"]
                           if pd.notna(row.get(f"{f.lower()}_pass")) and row.get(f"{f.lower()}_pass")],
                "failed": [f for f in ["Graham", "Greenblatt", "Dorsey", "Trajectory"]
                           if pd.notna(row.get(f"{f.lower()}_pass")) and not row.get(f"{f.lower()}_pass")],
                "years_of_data": int(row["years_of_data"]) if pd.notna(row.get("years_of_data")) else 0,
            })
        return entries

    updated = df["updated_date"].iloc[0] if "updated_date" in df.columns else "Unknown"

    return {
        "market": "india",
        "stocks_in_universe": len(df),
        "data_as_of": updated,
        "perfect_consensus_4_of_4": {
            "count": len(tier_4),
            "top_10": to_list(tier_4),
        },
        "strong_consensus_3_of_4": {
            "count": len(tier_3),
            "top_10": to_list(tier_3),
        },
        "moderate_consensus_2_of_4": {
            "count": len(tier_2),
            "top_10": to_list(tier_2),
        },
        "note": "Pre-scored universe of ~4500 Indian stocks (NSE + BSE). Data updated monthly. After presenting results, use search_book to explain WHY each investment style delivers returns, citing Graham, Greenblatt, and Dorsey.",
    }

def get_sip_candidates(sip_amount: int, time_horizon: str, investor_type: str, review_freq: str) -> dict:
    """Filter the pre-scored universe to 30-50 SIP-suitable candidates based on investor profile.
    The LLM then selects the final 5-8 stocks using book wisdom and qualitative judgment.

    Use this when the user wants to build a SIP portfolio. First collect the 4 inputs
    through natural conversation, then call this tool.

    Args:
        sip_amount: Monthly SIP amount in INR (e.g. 5000, 25000, 50000)
        time_horizon: Investment duration. Must be one of:
                      short  - 1 to 3 years
                      medium - 3 to 7 years
                      long   - 7+ years
        investor_type: Risk profile derived from goal question. Must be one of:
                       defensive    - wants to beat FD returns with safety
                       balanced     - wants to build wealth steadily over time
                       enterprising - wants maximum growth, patient through volatility
        review_freq: How often the investor wants to monitor. Must be one of:
                     passive  - set it and forget for years
                     moderate - review every few months
                     active   - likes staying informed and adjusting
    """
    df = universe_df.copy()

    # ── Base quality filter (all profiles) ──
    # ── Base quality filter (all profiles) ──
    if "quality_pass" in df.columns:
        df = df[df["quality_pass"] != False]
    df = df[df["years_of_data"] >= 2]
    df = df[pd.notna(df["pe"]) & pd.notna(df["roe_pct"]) & pd.notna(df["de"])]
    df = df[df["pe"] > 0]  # Exclude negative P/E (loss-making)

    # ── Profile-specific filtering ──
    if investor_type == "defensive":
        # Graham-focused: strict value, prefer dividends
        df = df[df["score"] >= 3]
        df = df[df["graham_pass"] == True]
        # Prefer dividend payers but don't exclude non-payers if pool is small
        div_payers = df[pd.notna(df["dividend_yield_pct"]) & (df["dividend_yield_pct"] > 0)]
        if len(div_payers) >= 15:
            df = div_payers
        target_count = 30

    elif investor_type == "balanced":
        # Quality + value balance
        df = df[df["score"] >= 2]
        # Prefer stocks passing at least Greenblatt or Dorsey (quality signal)
        quality = df[(df["greenblatt_pass"] == True) | (df["dorsey_pass"] == True)]
        if len(quality) >= 20:
            df = quality
        target_count = 40

    elif investor_type == "enterprising":
        # Growth-tilted: Greenblatt + Trajectory preferred
        df = df[df["score"] >= 2]
        # Prefer stocks with positive trajectory
        growers = df[df["trajectory_pass"] == True]
        if len(growers) >= 20:
            df = growers
        target_count = 50

    else:
        df = df[df["score"] >= 2]
        target_count = 40

    # ── Time horizon adjustments ──
    if time_horizon == "short":
        # Short horizon: prefer lower volatility, higher score
        df = df[df["score"] >= 3] if len(df[df["score"] >= 3]) >= 10 else df
        # Prefer larger, established companies
        large = df[pd.notna(df["market_cap"]) & (df["market_cap"] > 1e10)]
        if len(large) >= 10:
            df = large

    elif time_horizon == "long":
        # Long horizon: can include smaller companies with growth
        pass  # No additional filtering, broader pool is fine

    # ── Sort by composite score (value + quality + growth) ──
    df = df.copy()
    df["_sort_pe"] = df["pe"].apply(lambda x: x if pd.notna(x) else 9999)
    df["_sort_roe"] = df["roe_pct"].apply(lambda x: -x if pd.notna(x) else 9999)
    df["_sort_rev"] = df["rev_growth"].apply(lambda x: -x if pd.notna(x) else 9999)
    df["_sort_score"] = -df["score"]
    df = df.sort_values(["_sort_score", "_sort_pe", "_sort_roe", "_sort_rev"])

    # ── Trim to target count ──
    df = df.head(target_count)

    # ── Build output ──
    candidates = []
    for _, row in df.iterrows():
        candidate = {
            "ticker": row["ticker"],
            "name": row.get("name", "N/A") if pd.notna(row.get("name")) else "N/A",
            "sector": row.get("sector", "N/A") if pd.notna(row.get("sector")) else "N/A",
            "price": round(row["price"], 2) if pd.notna(row.get("price")) else "N/A",
            "market_cap": round(float(row["market_cap"]), 0) if pd.notna(row.get("market_cap")) else "N/A",
            "pe": round(row["pe"], 2) if pd.notna(row.get("pe")) else "N/A",
            "pb": round(row["pb"], 2) if pd.notna(row.get("pb")) else "N/A",
            "roe_pct": round(row["roe_pct"], 2) if pd.notna(row.get("roe_pct")) else "N/A",
            "de": round(row["de"], 2) if pd.notna(row.get("de")) else "N/A",
            "earnings_yield": round(row["earnings_yield"], 2) if pd.notna(row.get("earnings_yield")) else "N/A",
            "dividend_yield_pct": round(row["dividend_yield_pct"], 2) if pd.notna(row.get("dividend_yield_pct")) else "N/A",
            "rev_growth": round(row["rev_growth"], 2) if pd.notna(row.get("rev_growth")) else "N/A",
            "ni_growth": round(row["ni_growth"], 2) if pd.notna(row.get("ni_growth")) else "N/A",
            "debt_growth": round(row["debt_growth"], 2) if pd.notna(row.get("debt_growth")) else "N/A",
            "years_of_data": int(row["years_of_data"]) if pd.notna(row.get("years_of_data")) else 0,
            "score": int(row["score"]),
            "graham_pass": bool(row.get("graham_pass")) if pd.notna(row.get("graham_pass")) else False,
            "greenblatt_pass": bool(row.get("greenblatt_pass")) if pd.notna(row.get("greenblatt_pass")) else False,
            "dorsey_pass": bool(row.get("dorsey_pass")) if pd.notna(row.get("dorsey_pass")) else False,
            "trajectory_pass": bool(row.get("trajectory_pass")) if pd.notna(row.get("trajectory_pass")) else False,
            # Historical trends for qualitative LLM assessment
            "roe_y0": round(row["roe_y0"], 2) if pd.notna(row.get("roe_y0")) else None,
            "roe_y1": round(row["roe_y1"], 2) if pd.notna(row.get("roe_y1")) else None,
            "roe_y2": round(row["roe_y2"], 2) if pd.notna(row.get("roe_y2")) else None,
            "roe_y3": round(row["roe_y3"], 2) if pd.notna(row.get("roe_y3")) else None,
            "revenue_y0": row.get("revenue_y0") if pd.notna(row.get("revenue_y0")) else None,
            "revenue_y1": row.get("revenue_y1") if pd.notna(row.get("revenue_y1")) else None,
        }
        candidates.append(candidate)

    # Sanitize: replace any NaN/inf values that would break JSON serialization
    def _sanitize(obj):
        if isinstance(obj, float) and (pd.isna(obj) or np.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    candidates = _sanitize(candidates)

    return {
        "investor_profile": {
            "sip_amount_inr": sip_amount,
            "time_horizon": time_horizon,
            "investor_type": investor_type,
            "review_frequency": review_freq,
        },
        "candidates_count": len(candidates),
        "candidates": candidates,
        "selection_instruction": (
            f"You have {len(candidates)} pre-filtered candidates. "
            f"Select 5-8 stocks for this {investor_type} investor with a {time_horizon}-term horizon. "
            f"Use search_book to pull Graham/Greenblatt/Dorsey wisdom relevant to this profile. "
            f"Apply qualitative moat assessment (Dorsey) — check ROE trends to see if moat is stable or eroding. "
            f"Enforce max 2 stocks per sector for diversification. "
            f"Allocate the monthly SIP of INR {sip_amount} across selected stocks. "
            f"For each pick, explain WHY it fits this investor using book philosophy. "
            f"Output the final portfolio as a clean table with: ticker, name, sector, allocation_pct, sip_amount_inr, score, and a one-line thesis."
        ),
    }



def get_csv_financial_data(ticker: str) -> dict:
    """
    Reads the pre-scored universe database and returns the specific row for the requested ticker.
    Extracts core metrics, trajectories, and the boolean pass/fail status for the 4 investment frameworks.
    Use this when you need proprietary framework scores or specific local data for a single company.
    """
    resolved = _resolve_ticker(ticker)
    try:
        # universe_df is globally cached at the top of your script
        company_data = universe_df[universe_df['ticker'] == resolved]
        
        if company_data.empty:
            # Fallback to name search if ticker fails
            company_data = universe_df[universe_df['name'].str.contains(ticker, case=False, na=False)]
            
        if company_data.empty:
            return {"error": f"No proprietary CSV data found for {ticker}."}
            
        # Return the specific row as a dictionary
        return company_data.iloc[0].fillna("N/A").to_dict()
    except Exception as e:
        return {"error": f"Error reading CSV data: {str(e)}"}

def get_macro_context(ticker: str) -> dict:
    """
    Uses Yahoo Finance to return the company's sector and the 5-day 
    performance of the broader market index (Nifty 50: ^NSEI) to gauge macro momentum.
    """
    resolved = _resolve_ticker(ticker)
    try:
        stock = yf.Ticker(resolved)
        sector = stock.info.get('sector', 'Unknown Sector')
        
        # Get Nifty 50 momentum
        nifty = yf.Ticker("^NSEI")
        hist = nifty.history(period="5d")
        
        if not hist.empty and len(hist) >= 1:
            start_price = float(hist['Close'].iloc[0])
            end_price = float(hist['Close'].iloc[-1])
            pct_change = ((end_price - start_price) / start_price) * 100
        else:
            pct_change = None
            
        return {
            "ticker": resolved,
            "sector": sector,
            "nifty_50_5d_performance_pct": round(pct_change, 2) if pct_change is not None else "N/A"
        }
    except Exception as e:
        return {"error": f"Error fetching macro context: {str(e)}"}




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
    get_csv_financial_data,
    get_macro_context,
    get_sip_candidates,
    register_portfolio,
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
    "get_csv_financial_data": get_csv_financial_data,
    "get_macro_context": get_macro_context,
    "get_sip_candidates": get_sip_candidates,
    "register_portfolio": register_portfolio,
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

import datetime
current_date = datetime.date.today().strftime("%B %Y")

SYSTEM_INSTRUCTION = f"""You are a highly structured Quantitative Investment Committee acting as a single agent.
CURRENT DATE: {current_date}

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
12. find_investments — Screen ~4500 Indian stocks (NSE + BSE) from a pre-scored universe against ALL 4 frameworks. Returns three tiers: Perfect Consensus (4/4 pass), Strong Consensus (3/4 pass), and Moderate Consensus (2/4 pass), top 10 each. Use when the user asks to find, discover, or recommend stocks, or wants investment ideas. Call with market='india' or 'all'.
13. show_stock_chart — Renders a visual 13-month line chart of a stock's closing price directly in the UI. Use this whenever the user asks for a chart, graph, or visual trajectory.
14. get_csv_financial_data — Reads the pre-scored universe database for a specific ticker to get proprietary framework scores (Graham, Greenblatt, Dorsey, Trajectory pass/fail flags).
15. get_macro_context — Gets the sector and 5-day performance of the broader market (Nifty 50) to gauge macro momentum versus asset momentum.
16. get_sip_candidates — Build a SIP portfolio. Collects investor profile (sip_amount, time_horizon, investor_type, review_freq) and returns 30-50 pre-filtered candidates. You then select 5-8 using book wisdom and qualitative judgment. Use when the user wants to start a SIP, build a portfolio, or asks where to invest monthly.
17. register_portfolio — After presenting your finalized SIP portfolio to the user, call this to register it for saving. Pass portfolio_name, investor_type, sip_amount, time_horizon, review_days (integer number of days between reviews), and stocks_json (a JSON string list where each item has ticker, name, sector, allocation_pct). ALWAYS call this after presenting the final SIP portfolio table.

SIP PORTFOLIO PROTOCOL:
When the user wants to build a SIP portfolio, you MUST ask exactly ONE question per message. Wait for the answer before asking the next. The sequence is:

Message 1: Ask ONLY "How much do you want to invest monthly in INR?"
Message 2 (after they answer): Ask ONLY "How long do you plan to keep investing? 1-3 years, 3-7 years, or 7+ years?"
Message 3 (after they answer): Ask ONLY "What is your goal with this SIP?" and give three options: "Steady returns that beat savings/FDs" / "Build long-term wealth with a good balance" / "Maximum growth — I am patient through market ups and downs"
Message 4 (after they answer): Ask ONLY "How often do you want to check on your investments?" and give three options: "Set it and forget for years" / "Glance every few months" / "I like staying active and informed"

NEVER ask more than one question in a single message. If the user gives multiple answers at once, accept them and skip ahead.

After selecting your final 5-8 stocks with allocations, call register_portfolio with the structured data. Write your analysis and reasoning about the picks in the same response. Do NOT ask the user for permission to save — just call the tool.

CRITICAL: Frame question 3 around GOALS, never around LOSSES or RISK. Do NOT mention portfolio drops, drawdowns, or volatility in the question itself.

After all 4 answers are collected, map them:
- Goal answer 1 → investor_type="defensive"
- Goal answer 2 → investor_type="balanced"  
- Goal answer 3 → investor_type="enterprising"
- Convert the user's review preference to a number of days (review_days). Use your judgment — "monthly" is 30, "every two weeks" is 14, "quarterly" is 90, "twice a year" is 180, "set and forget" is 365. Any reasonable number is valid; do not constrain to fixed options.
- Time 1-3yr → time_horizon="short", 3-7yr → "medium", 7+yr → "long"

Then call get_sip_candidates with the mapped parameters.

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
SYNTHESIS PROTOCOL:
- If the local CSV tool returns framework flags (e.g., graham_pass = True/False), you MUST query search_book for the theoretical definition of that framework (e.g., 'Margin of Safety').
- Cross-reference company metrics with get_macro_context to determine if the company is outperforming or being dragged by market beta.

PASS/FAIL THRESHOLDS (Apply mechanically):
1. Graham: PASS IF (P/E <= 15) AND (P/B <= 1.5).
2. Greenblatt: PASS ONLY IF (ROE > 15%) AND (Earnings Yield > 5%).
3. Dorsey: PASS ONLY IF (ROE > 15%) AND (Debt/Equity < 50%) AND (You explicitly identify a business moat). The moat criterion is binary: does or does not have an identifiable moat. This is independent of Graham or Greenblatt results.
4. Trajectory: PASS ONLY IF (1Y Rev Growth > 0% OR 1Y Net Income Growth > 0%) AND (Debt Growth < 0% OR Current D/E < 50%).

VERDICT RULE:
- PASS CONDITION (YES): If ANY 2 out of the 4 frameworks PASS, the VERDICT decision is YES.
- VALUE EXCEPTION (YES): If Graham PASSES but the score is only 1/4, the VERDICT decision is YES (Deep Value).

VERIFICATION PROTOCOL (Mandatory — runs before ANY "YES" or portfolio inclusion):
You operate in two phases: DRAFT then VERIFY. Never skip VERIFY.
PHASE 1 — DRAFT:
Analyze the stock or build the candidate list normally using framework scores and tools.
PHASE 2 — VERIFY (loop for each stock you are about to recommend):
Before you write your final output, for EVERY stock you plan to say YES to or include in a portfolio:
Step A: Call get_stock_data for that ticker. Read the earnings_quality block in the response.
        If anomaly_flags contains ANY "RED FLAG" entry → that stock is REJECTED. Remove it. Move to next candidate.
Step B: Call search_book with a query relevant to the risk you see in the data. Examples:
        - If P/E is abnormally low (<3): search "Graham warnings non-recurring income one-time gains"
        - If ROE is abnormally high (>50%): search "Dorsey unsustainable returns on equity financial leverage"
        - If debt dropped dramatically in one year: search "Graham balance sheet manipulation debt restructuring"
        - If revenue grew but cash flow didn't: search "Dorsey earnings quality cash flow vs net income"
        Pick the query based on what looks unusual in the ACTUAL numbers, not a fixed checklist.
Step C: Cross-reference. Does the book passage describe a pattern that matches this stock's data?
        If yes → REJECT that stock with a one-line explanation citing the book.
        If no → KEEP.
Step D: If you rejected a stock from a portfolio, pull the next-best candidate from the tool results and run Steps A-C on it.
VERIFICATION APPLIES TO:
- Single stock YES verdicts
- Every stock in the final SIP portfolio table (all 5-8 picks must survive)
- Screener results when you highlight "top picks" or "best buys"
VERIFICATION DOES NOT APPLY TO:
- Simple data lookups ("what is TCS's P/E ratio")
- NO verdicts (if you're already saying no, no need to verify)
- Conversational messages (asking user questions, greetings, etc.)
- The raw tier listings from find_investments (only verify when you editorialize about specific picks)
LOOP LIMIT: Maximum 3 replacement rounds per portfolio. If you burn through 3 replacements for one slot, leave the slot empty and tell the user the pool didn't have enough quality candidates.


EXECUTION PROTOCOL:
You are an intelligent, conversational, and highly analytical Quantitative Investment Committee. You are free from rigid formatting templates, but you are BOUND by strict quantitative logic. 

EARNINGS QUALITY (AUTO-INJECTED):
Earnings quality flags are automatically included in every get_stock_data response under the "earnings_quality" key. If ANY anomaly flags say "RED FLAG", you MUST OVERRIDE positive framework scores and issue a "NO" verdict regardless of how many frameworks pass. A low P/E driven by unusual items is a value trap, not a bargain.

Follow these core behavioral directives:
1. The Binary Verdict (No Waffling): Answer the user's specific question immediately. You MUST explicitly state your final investment decision as a bold "YES" or "NO" in the opening paragraph. 
   - YES CONDITION: If ANY 2 out of the 4 frameworks PASS, the verdict is YES.
   - YES EXCEPTION: If Graham PASSES but the score is only 1/4, the verdict is YES (Deep Value).
   - NO CONDITION: If fewer than 2 frameworks pass (and Graham fails), the verdict is NO.
2. Fluid Integration: Weave the quantitative data (fundamentals, Graham/Greenblatt/Dorsey/Trajectory pass/fail states) naturally into your prose. Explain the *why* behind the numbers instead of just listing them. 
3. Dynamic Formatting: Use markdown headers, bullet points, and bold text organically to make your analysis readable. 
4. Grounded Wisdom: Conclude your analysis with a bolded "Committee Note" providing actionable risk management advice or psychological grounding derived directly from Graham, Greenblatt, or Dorsey.
"""

AUDITOR_SYSTEM_PROMPT = """You are the Chief Risk Officer and Auditor for an Investment Committee.
You are a truthful, disagreeable, first-principle thinker. Your sole job is to catch the Analyst making mistakes, specifically falling for statistical illusions.

You receive THREE inputs:
1. The user's original query
2. The Analyst's draft response
3. Independent Earnings Quality Data — hard numbers YOU verify against

AUDIT CHECKLIST (use the Independent data, not the Analyst's claims):
1. For every stock where the Analyst recommends YES: check if unusual_items_pct > 20%. If so, the YES is invalid.
2. For every stock where the Analyst recommends YES: check if cash_conversion < 0.5. If so, the YES is invalid.
3. If the Independent data contains RED FLAG entries for a stock the Analyst recommended, but the Analyst did not mention or address those flags, the draft is invalid.
4. If Independent Earnings Quality Data is empty (no tickers found or no flags raised), the draft is likely safe on this dimension.

CRITICAL BYPASS RULES (Auto-Approve):
- If the Analyst is simply asking the user a question (such as the 4-step SIP portfolio sequence), reply EXACTLY with: [APPROVED]
- If the Analyst issued a "NO" verdict or is simply conversing, reply EXACTLY with: [APPROVED]

If the Analyst's draft is fundamentally sound and no Independent data contradicts it, reply EXACTLY with: [APPROVED]
If the Independent data contradicts the Analyst's verdict, reply with: [REJECT] followed by which specific tickers failed quality checks and what the Analyst must change."""

# ──────────────────────────────────────────────
# AGENT
# ──────────────────────────────────────────────
def intercept_and_rewrite_query(user_query: str) -> str:
    """
    Intercepts the layman question and translates it into strict technical 
    directives for the main execution agent using a fast model.
    """
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    
    router_prompt = f"""
    You are the Pre-Processing Routing Agent for a quantitative financial system.
    Translate this layman user query into a strict, step-by-step technical directive for the Execution Agent.

    The Execution Agent has tools: get_csv_financial_data, get_macro_context, search_book, get_stock_data, get_price_history, etc.

    User Query: "{user_query}"

    Identify the ticker symbol. Tell the agent EXACTLY which tools to use and what to cross-reference based on the query intent. 
    DO NOT ANSWER THE QUESTION. ONLY OUTPUT THE DIRECTIVE.
    """
    try:
        for model_name in FREE_MODELS:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=router_prompt,
                )
                return f"SYSTEM DIRECTIVE (Translated Intent): {response.text}"
            except Exception as inner_e:
                if "429" in str(inner_e) or "RESOURCE_EXHAUSTED" in str(inner_e):
                    continue
                raise inner_e
        return user_query
    except Exception:
        return user_query


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
            # --- PHASE 1: ANALYST DRAFTS THESIS ---
            analyst_chat = client.chats.create(
                model=model_name,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    tools=TOOLS,
                ),
                history=history,
            )

            analyst_response = analyst_chat.send_message(user_message)
            all_text_parts = []

            def _extract_text(resp):
                """Get text from response even when function calls coexist."""
                try:
                    for part in resp.candidates[0].content.parts:
                        if hasattr(part, 'text') and part.text:
                            return part.text
                except (AttributeError, IndexError):
                    pass
                try:
                    return resp.text or ""
                except Exception:
                    return ""

            while analyst_response.function_calls:
                text_chunk = _extract_text(analyst_response)
                if text_chunk:
                    all_text_parts.append(text_chunk)
                function_responses = []
                for fc in analyst_response.function_calls:
                    if fc.name in tool_functions:
                        result = tool_functions[fc.name](**fc.args)
                    else:
                        result = {"error": f"Unknown tool: {fc.name}"}
                    function_responses.append(
                        types.Part.from_function_response(name=fc.name, response=result)
                    )
                analyst_response = analyst_chat.send_message(function_responses)

            final_chunk = _extract_text(analyst_response)
            if final_chunk:
                all_text_parts.append(final_chunk)
            draft_text = "\n\n".join(all_text_parts)

            # --- PHASE 2: AUDITOR REVIEWS DRAFT (with independent data) ---
            # Extract tickers mentioned in draft and run quality checks
            NOISE_WORDS = {"PASS", "FAIL", "YES", "NO", "ROE", "EPS", "SIP",
                           "AND", "THE", "FOR", "NOT", "USE", "ALL", "WHY",
                           "HOW", "BUY", "TOP", "LOW", "HIGH", "CAP", "NET",
                           "YOY", "INR", "USD", "FY", "PE", "PB", "DE",
                           "SMA", "CAGR", "NAV", "IPO", "ETF", "PDF", "CSV"}
            mentioned_tickers = set(re.findall(r'\b[A-Z]{2,15}(?:\.NS|\.BO)?\b', draft_text))
            mentioned_tickers -= NOISE_WORDS

            quality_checks = {}
            for t in mentioned_tickers:
                qc = get_earnings_quality_metrics(t)
                if "error" not in qc and qc.get("anomaly_flags"):
                    quality_checks[t] = {
                        "cash_conversion": qc["cash_conversion_ratio"],
                        "unusual_items_pct": qc["unusual_items_pct_of_income"],
                        "flags": qc["anomaly_flags"],
                    }

            auditor_input = (
                f"User Query: {user_message}\n\n"
                f"Analyst Draft:\n{draft_text}\n\n"
                f"Independent Earnings Quality Data:\n{json.dumps(quality_checks, indent=2)}"
            )

            auditor_response = client.models.generate_content(
                model=model_name,
                contents=auditor_input,
                config=types.GenerateContentConfig(system_instruction=AUDITOR_SYSTEM_PROMPT)
            )

            audit_result = auditor_response.text.strip()

            # --- PHASE 3: RESOLUTION ---
            if audit_result.startswith("[REJECT]"):
                # Force the Analyst to read the Auditor's rejection and rewrite
                correction_prompt = f"The Chief Risk Officer REJECTED your draft with the following feedback:\n\n{audit_result}\n\nRewrite your entire analysis to comply with this feedback. Change your verdict if necessary."
                final_response = analyst_chat.send_message(correction_prompt)
                st.session_state.chat_history = analyst_chat.get_history()
                
                # Append an internal note to the UI so the user sees the system working
                return f"*(Internal Audit Triggered: Adjusted thesis based on earnings quality)*\n\n{final_response.text}", model_name
            else:
                # Auditor approved
                st.session_state.chat_history = analyst_chat.get_history()
                return draft_text, model_name

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
AGENT_AVATAR = "logo.svg"

if st.session_state.sb_view_mode == "chat":
    chat_area = st.container()

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

    prompt = st.chat_input("Ask about any stock, or type a question...")

    if not prompt and "pending_prompt" in st.session_state:
        prompt = st.session_state.pop("pending_prompt")

    with chat_area:
        if not st.session_state.messages:
            st.markdown("")
            st.info("Choose a company in the sidebar then select a framework below, or find best stocks by clicking below buttons.")

        for msg in st.session_state.messages:
            avatar = USER_AVATAR if msg["role"] == "user" else AGENT_AVATAR
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])
                if msg.get("model"):
                    st.caption(f"⚡ {msg['model']}")

        if st.session_state.get("pending_portfolio"):
            portfolio = st.session_state.pending_portfolio

            if st.session_state.sb_user_id is None:
                st.info("💡 Log in to save this portfolio to your account.")
            else:
                st.markdown("### 📋 Your SIP Portfolio")
                preview_data = []
                for s in portfolio["stocks"]:
                    preview_data.append({
                        "Stock": s.get("name", s["ticker"]),
                        "Ticker": s["ticker"],
                        "Sector": s.get("sector", "—"),
                        "Allocation": f"{s.get('allocation_pct', 0)}%",
                        "Monthly": f"₹{portfolio['sip_amount'] * s.get('allocation_pct', 0) / 100:,.0f}",
                    })
                st.dataframe(pd.DataFrame(preview_data), hide_index=True, use_container_width=True)
                st.caption(f"Total SIP: ₹{portfolio['sip_amount']:,}/month · {portfolio.get('investor_type', '')} · {portfolio.get('time_horizon', '')} horizon")
                if st.button("💾 Save Portfolio", use_container_width=True):
                    try:
                        sb = get_supabase()
                        review_days = portfolio.get("review_days", 90)
                        next_review = (datetime.date.today() + datetime.timedelta(days=review_days)).isoformat()
                        port_resp = sb.table("portfolios").insert({
                            "user_id": st.session_state.sb_user_id,
                            "name": portfolio["name"],
                            "investor_type": portfolio["investor_type"],
                            "sip_amount": portfolio["sip_amount"],
                            "time_horizon": portfolio["time_horizon"],
                            "review_freq": str(review_days),
                            "next_review_date": next_review,
                        }).execute()
                        portfolio_id = port_resp.data[0]["id"]
                        stocks_for_alloc = []
                        for stock in portfolio["stocks"]:
                            ticker = stock["ticker"]
                            try:
                                info = yf.Ticker(ticker).info
                                price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
                            except Exception:
                                price = 0
                            row = universe_df[universe_df["ticker"] == ticker]
                            pe = float(row["pe"].iloc[0]) if len(row) and pd.notna(row["pe"].iloc[0]) else None
                            roe = float(row["roe_y0"].iloc[0]) if len(row) and "roe_y0" in row.columns and pd.notna(row["roe_y0"].iloc[0]) else None
                            score = int(row["score"].iloc[0]) if len(row) and pd.notna(row["score"].iloc[0]) else None
                            sector = stock.get("sector", "") or (str(row["sector"].iloc[0]) if len(row) and "sector" in row.columns and pd.notna(row["sector"].iloc[0]) else "")
                            stocks_for_alloc.append({
                                "ticker": ticker, "name": stock.get("name", ""), "sector": sector,
                                "allocation_pct": stock.get("allocation_pct", 0), "price": price,
                                "pe": pe, "roe": roe, "score": score,
                            })
                        allocated, unallocated = allocate_shares(stocks_for_alloc, portfolio["sip_amount"])
                        for s in allocated:
                            sb.table("holdings").insert({
                                "portfolio_id": portfolio_id, "ticker": s["ticker"], "name": s["name"],
                                "sector": s["sector"], "allocation_pct": s["allocation_pct"], "shares": s["shares"],
                                "sip_amount_inr": s["actual_amount"], "price_at_entry": s["price"],
                                "pe_at_entry": s["pe"], "roe_at_entry": s["roe"], "score_at_entry": s["score"],
                            }).execute()
                        st.success(f"Portfolio saved! Invested ₹{portfolio['sip_amount'] - unallocated:,.0f} of ₹{portfolio['sip_amount']:,}.")
                        if unallocated > 0:
                            st.info(f"₹{unallocated:,.0f} unallocated (not enough for another share of any holding).")
                        breakdown_data = []
                        for s in allocated:
                            breakdown_data.append({
                                "Stock": s["name"] or s["ticker"], "Price": f"₹{s['price']:,.2f}",
                                "Shares": s["shares"], "Invested": f"₹{s['actual_amount']:,.0f}",
                                "Target": f"₹{portfolio['sip_amount'] * s['allocation_pct'] / 100:,.0f}",
                            })
                        st.dataframe(pd.DataFrame(breakdown_data), hide_index=True, use_container_width=True)
                        st.session_state.pending_portfolio = None
                    except Exception as e:
                        st.error(f"Save failed: {e}")

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar=USER_AVATAR):
                st.markdown(prompt)
            with st.chat_message("assistant", avatar=AGENT_AVATAR):
                response_placeholder = st.empty()
                answer = None
                model_used = None
                with st.spinner("Routing & Analyzing..."):
                    try:
                        if len(st.session_state.messages) <= 1:
                            rewritten_directive = intercept_and_rewrite_query(prompt)
                        else:
                            rewritten_directive = prompt
                        answer, model_used = agent_turn(rewritten_directive)
                    except Exception as e:
                        error_msg = str(e)
                        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "All models rate-limited" in error_msg:
                            st.warning("API limit reached. Using fallback system...")
                            fallback_answer = fallback_router(prompt)
                            response_placeholder.markdown(fallback_answer)
                            st.session_state.messages.append({"role": "assistant", "content": f"*(Fallback)*\n\n{fallback_answer}"})
                        else:
                            st.error(f"Error: {error_msg[:150]}")
                            if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                                st.session_state.messages.pop()
                            st.session_state.pending_retry = prompt
                if answer:
                    response_placeholder.markdown(answer)
                    st.caption(f"⚡ {model_used} | Routed via Interceptor")
                    st.session_state.messages.append({"role": "assistant", "content": answer, "model": model_used})
                    if st.session_state.get("pending_portfolio"):
                        st.rerun()

        if st.session_state.get("pending_retry"):
            if st.button("🔄 Retry last query", use_container_width=True):
                st.session_state.pending_prompt = st.session_state.pop("pending_retry")
                st.rerun()

else:
    st.markdown("### 📁 My Portfolios")
    sb = get_supabase()
    try:
        port_resp = sb.table("portfolios").select("*").eq(
            "user_id", st.session_state.sb_user_id
        ).order("created_at", desc=True).execute()
        portfolios = port_resp.data
    except Exception as e:
        st.error(f"Failed to load portfolios: {e}")
        portfolios = []

    if not portfolios:
        st.info("No saved portfolios yet. Use the SIP Portfolio builder to create one!")
    else:
        for port in portfolios:
            with st.container(border=True):
                st.markdown(f"**{port['name']}**")
                st.caption(
                    f"Created: {port['created_at'][:10]} · "
                    f"{port.get('investor_type', '—')} · "
                    f"₹{port.get('sip_amount', 0):,}/mo · "
                    f"{port.get('time_horizon', '—')} horizon · "
                    f"Review: every {port.get('review_freq', '90')} days · "
                    f"Next: {port.get('next_review_date', '—')}"
                )
                try:
                    hold_resp = sb.table("holdings").select("*").eq("portfolio_id", port["id"]).execute()
                    holdings = hold_resp.data
                except Exception:
                    holdings = []

                if holdings:
                    hold_df = pd.DataFrame(holdings)
                    display_cols = {
                        "name": "Stock", "ticker": "Ticker", "sector": "Sector", "shares": "Shares",
                        "price_at_entry": "Entry Price", "sip_amount_inr": "Invested",
                        "allocation_pct": "Alloc %", "score_at_entry": "Score",
                    }
                    available = {k: v for k, v in display_cols.items() if k in hold_df.columns}
                    st.dataframe(hold_df[list(available.keys())].rename(columns=available), hide_index=True, use_container_width=True)
                else:
                    st.caption("No holdings found.")

                today = datetime.date.today()
                review_date = None
                if port.get("next_review_date"):
                    try:
                        review_date = datetime.date.fromisoformat(str(port["next_review_date"]))
                    except (ValueError, TypeError):
                        review_date = None

                if review_date and holdings:
                    days_until = (review_date - today).days
                    if days_until > 7:
                        st.caption(f"📅 Next review in {days_until} days ({review_date.isoformat()})")
                    else:
                        if days_until > 0:
                            st.warning(f"📅 Review due in {days_until} days!")
                        elif days_until == 0:
                            st.warning("📅 Review due today!")
                        else:
                            st.error(f"📅 Review overdue by {abs(days_until)} days!")

                        if st.button("🔄 Review Portfolio", key=f"review_{port['id']}", use_container_width=True):
                            with st.spinner("Analyzing holdings with market context and book philosophy..."):
                                enriched = build_review_context(holdings, port)
                                llm_recs = generate_review_recommendations(
                                    enriched, port.get("investor_type", "balanced"),
                                    port.get("time_horizon", "medium")
                                )

                                # Merge LLM recommendations with enriched data
                                total_entry = 0
                                total_current = 0
                                review_rows = []

                                for h in enriched:
                                    total_entry += h["entry_price"] * h["shares"]
                                    total_current += h["now_price"] * h["shares"]

                                    # Find LLM recommendation for this ticker
                                    llm_rec = None
                                    if llm_recs:
                                        llm_rec = next((r for r in llm_recs if r.get("ticker") == h["ticker"]), None)

                                    if llm_rec:
                                        raw_action = llm_rec.get("action", "HOLD").upper()
                                        reasoning = llm_rec.get("reasoning", "")
                                        confidence = llm_rec.get("confidence", "medium")
                                        sell_qty = llm_rec.get("sell_qty", 0)

                                        if "SELL ALL" in raw_action:
                                            action = f"🔴 SELL ALL ({h['shares']})"
                                            sell_qty = h["shares"]
                                        elif "SELL HALF" in raw_action:
                                            sell_qty = max(1, h["shares"] // 2)
                                            action = f"🟠 SELL {sell_qty} of {h['shares']}"
                                        elif "BUY" in raw_action:
                                            action = "🟢 BUY MORE"
                                            sell_qty = 0
                                        else:
                                            action = "🟢 HOLD"
                                            sell_qty = 0
                                    else:
                                        # Mechanical fallback
                                        sc = h["score_change"]
                                        if h["has_red_flags"]:
                                            action = f"🔴 SELL ALL ({h['shares']})"
                                            reasoning = "Earnings quality red flags detected. Graham warns against value traps."
                                            confidence = "high"
                                            sell_qty = h["shares"]
                                        elif sc <= -2:
                                            sell_qty = max(1, h["shares"] // 2)
                                            action = f"🟠 SELL {sell_qty} of {h['shares']}"
                                            reasoning = "Score dropped sharply. Review fundamentals."
                                            confidence = "medium"
                                        elif sc <= -1:
                                            action = "🟡 HOLD (watch)"
                                            reasoning = "Slight deterioration. Monitor next review."
                                            confidence = "medium"
                                            sell_qty = 0
                                        elif sc == 0:
                                            action = "🟢 HOLD"
                                            reasoning = "Fundamentals stable."
                                            confidence = "high"
                                            sell_qty = 0
                                        else:
                                            action = "🟢 BUY MORE"
                                            reasoning = "Score improved."
                                            confidence = "medium"
                                            sell_qty = 0

                                    mkt_note = ""
                                    if h["market_relative"] is not None:
                                        if h["market_relative"] > 5:
                                            mkt_note = f"Outperformed Nifty by {h['market_relative']:+.1f}%"
                                        elif h["market_relative"] < -5:
                                            mkt_note = f"Underperformed Nifty by {h['market_relative']:+.1f}%"
                                        else:
                                            mkt_note = f"In line with market ({h['market_relative']:+.1f}% vs Nifty)"

                                    review_rows.append({
                                        "Stock": h["name"], "Shares": h["shares"],
                                        "Entry": f"₹{h['entry_price']:,.2f}", "Now": f"₹{h['now_price']:,.2f}",
                                        "P&L": f"₹{h['pnl']:,.0f}", "Return": f"{h['stock_return']:+.1f}%",
                                        "Score": f"{h['entry_score']}→{h['now_score']}", "Action": action,
                                        "_reasoning": reasoning, "_confidence": confidence,
                                        "_market_note": mkt_note, "_book_passage": h["book_passage"],
                                        "_sell_qty": sell_qty, "_holding_id": h["holding_id"],
                                        "_ticker": h["ticker"], "_sector": h["sector"],
                                        "_entry_price": h["entry_price"], "_now_price": h["now_price"],
                                    })

                                st.session_state[f"review_data_{port['id']}"] = {
                                    "rows": review_rows, "total_entry": total_entry,
                                    "total_current": total_current, "holdings": holdings,
                                    "enriched": enriched,
                                }

                                try:
                                    next_days = int(port.get("review_freq", 90))
                                except (ValueError, TypeError):
                                    next_days = 90
                                new_review = (today + datetime.timedelta(days=next_days)).isoformat()
                                try:
                                    sb.table("portfolios").update({"next_review_date": new_review}).eq("id", port["id"]).execute()
                                except Exception:
                                    pass

                review_state = st.session_state.get(f"review_data_{port['id']}")
                if review_state:
                    review_rows = review_state["rows"]
                    total_entry = review_state["total_entry"]
                    total_current = review_state["total_current"]
                    rev_holdings = review_state["holdings"]

                    port_pnl = total_current - total_entry
                    port_ret = (port_pnl / total_entry * 100) if total_entry > 0 else 0

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Invested", f"₹{total_entry:,.0f}")
                    m2.metric("Current Value", f"₹{total_current:,.0f}")
                    m3.metric("Total Return", f"{port_ret:+.1f}%", delta=f"₹{port_pnl:,.0f}")

                    # Nifty comparison
                    if review_rows and review_rows[0].get("_market_note"):
                        nifty_note = review_rows[0]["_market_note"]
                        st.caption(f"📊 Market context: {nifty_note}")

                    display_df = pd.DataFrame(review_rows).drop(columns=[c for c in review_rows[0] if c.startswith("_")])
                    st.dataframe(display_df, hide_index=True, use_container_width=True)

                    # Per-stock reasoning with book grounding
                    for r in review_rows:
                        if "SELL" in r["Action"]:
                            st.error(f"**{r['Stock']}** — {r['Action']}\n\n{r['_reasoning']}")
                        elif "BUY" in r["Action"]:
                            st.success(f"**{r['Stock']}** — {r['Action']}\n\n{r['_reasoning']}")
                        else:
                            st.info(f"**{r['Stock']}** — {r['Action']}\n\n{r['_reasoning']}")
                    # ── Update form — ALL holdings ──
                    sell_stocks = [(i, r) for i, r in enumerate(review_rows) if "SELL" in r["Action"]]

                    st.markdown("---")

                    # Calculate this cycle's SIP allocation
                    monthly_sip = port.get("sip_amount", 0)
                    try:
                        review_days = int(port.get("review_freq", 30))
                    except (ValueError, TypeError):
                        review_days = 30
                    cycle_amount = round(monthly_sip * review_days / 30)

                    sip_stocks = []
                    for r in review_rows:
                        if "SELL" not in r["Action"]:
                            h_match = next((h for h in rev_holdings if h.get("id") == r["_holding_id"]), {})
                            alloc = h_match.get("allocation_pct", 0)
                            non_sell_count = len([x for x in review_rows if "SELL" not in x["Action"]])
                            if alloc == 0 and non_sell_count > 0:
                                alloc = 100 / non_sell_count
                            sip_stocks.append({
                                "ticker": r["_ticker"], "name": r["Stock"],
                                "allocation_pct": alloc, "price": r["_now_price"],
                            })
                    sip_alloc = {}
                    if sip_stocks and cycle_amount > 0:
                        allocated, unallocated = allocate_shares(sip_stocks, cycle_amount)
                        for a in allocated:
                            sip_alloc[a["ticker"]] = a["shares"]
                        st.caption(f"💰 This cycle ({review_days} days): ₹{cycle_amount:,} to invest — suggested shares pre-filled below")
                        if unallocated > 0:
                            st.caption(f"₹{unallocated:,.0f} unallocatable (not enough for another share)")
                    else:
                        st.caption("Update what you actually did at your broker since last review:")

                    for i, r in enumerate(review_rows):
                        h_id = r["_holding_id"]
                        if "SELL" in r["Action"]:
                            st.number_input(
                                f"🔴 {r['Stock']} — shares sold (of {r['Shares']})",
                                min_value=0, max_value=r["Shares"], value=r["_sell_qty"],
                                key=f"sold_{port['id']}_{h_id}"
                            )
                        else:
                            c1, c2 = st.columns(2)
                            with c1:
                                suggested = sip_alloc.get(r["_ticker"], 0)
                                st.number_input(
                                    f"{'🟢' if 'BUY' in r['Action'] else '📥'} {r['Stock']} — shares bought",
                                    min_value=0, value=suggested, key=f"add_qty_{port['id']}_{h_id}"
                                )
                            with c2:
                                st.number_input(
                                    f"{r['Stock']} — price paid (₹)",
                                    min_value=0.0, value=float(r["_now_price"]), format="%.2f", key=f"add_price_{port['id']}_{h_id}"
                                )

                    # ── Replacement candidates if sells exist ──
                    candidates = []
                    if sell_stocks:
                        freed = 0
                        for idx, r in sell_stocks:
                            sell_qty = st.session_state.get(f"sold_{port['id']}_{r['_holding_id']}", 0)
                            price = r["_now_price"]
                            freed += sell_qty * price
                        remaining_sectors = []
                        for i, r in enumerate(review_rows):
                            is_sell = any(si == i for si, _ in sell_stocks)
                            if not is_sell:
                                remaining_sectors.append(r.get("_sector", ""))
                            else:
                                sold_qty = st.session_state.get(f"sold_{port['id']}_{r['_holding_id']}", 0)
                                if r["Shares"] - sold_qty > 0:
                                    remaining_sectors.append(r.get("_sector", ""))
                        all_tickers = [r["_ticker"] for r in review_rows]
                        candidates = find_replacement_candidates(
                            port.get("investor_type", "balanced"), port.get("time_horizon", "medium"),
                            all_tickers, remaining_sectors
                        )
                        if candidates:
                            st.markdown("---")
                            st.markdown(f"**Replacement candidates** (≈₹{freed:,.0f} freed from sells)")
                            cand_df = pd.DataFrame(candidates)
                            cand_display = cand_df[["name", "ticker", "sector", "price", "score", "pe", "roe_pct"]].rename(columns={
                                "name": "Stock", "ticker": "Ticker", "sector": "Sector",
                                "price": "Price", "score": "Score", "pe": "P/E", "roe_pct": "ROE %"
                            })
                            st.dataframe(cand_display, hide_index=True, use_container_width=True)
                            for c in candidates:
                                col_sel, col_qty, col_px = st.columns([1, 2, 2])
                                with col_sel:
                                    st.checkbox(c["name"][:15], key=f"repl_sel_{port['id']}_{c['ticker']}", value=False)
                                with col_qty:
                                    st.number_input(f"Shares", min_value=0, value=0, key=f"repl_qty_{port['id']}_{c['ticker']}")
                                with col_px:
                                    st.number_input(f"Price (₹)", min_value=0.0, value=float(c["price"]), format="%.2f", key=f"repl_px_{port['id']}_{c['ticker']}")

                    # ── Single update button ──
                    if st.button("✅ Portfolio Updated", key=f"apply_{port['id']}", use_container_width=True):
                        for i, r in enumerate(review_rows):
                            h_id = r["_holding_id"]
                            if "SELL" in r["Action"]:
                                sold = st.session_state.get(f"sold_{port['id']}_{h_id}", 0)
                                if sold > 0:
                                    new_shares = r["Shares"] - sold
                                    if new_shares <= 0:
                                        sb.table("holdings").delete().eq("id", h_id).execute()
                                    else:
                                        new_invested = new_shares * r["_entry_price"]
                                        sb.table("holdings").update({"shares": new_shares, "sip_amount_inr": round(new_invested, 2)}).eq("id", h_id).execute()
                            else:
                                new_qty = st.session_state.get(f"add_qty_{port['id']}_{h_id}", 0)
                                buy_price = st.session_state.get(f"add_price_{port['id']}_{h_id}", 0.0)
                                if new_qty > 0 and buy_price > 0:
                                    old_shares = r["Shares"]
                                    old_price = r["_entry_price"]
                                    total_shares = old_shares + new_qty
                                    avg_price = ((old_shares * old_price) + (new_qty * buy_price)) / total_shares
                                    sb.table("holdings").update({
                                        "shares": total_shares,
                                        "price_at_entry": round(avg_price, 2),
                                        "sip_amount_inr": round(total_shares * avg_price, 2),
                                    }).eq("id", h_id).execute()
                        if sell_stocks and candidates:
                            for c in candidates:
                                selected = st.session_state.get(f"repl_sel_{port['id']}_{c['ticker']}", False)
                                qty = st.session_state.get(f"repl_qty_{port['id']}_{c['ticker']}", 0)
                                px = st.session_state.get(f"repl_px_{port['id']}_{c['ticker']}", 0.0)
                                if selected and qty > 0 and px > 0:
                                    urow = universe_df[universe_df["ticker"] == c["ticker"]]
                                    sc_val = int(urow["score"].iloc[0]) if len(urow) and pd.notna(urow["score"].iloc[0]) else None
                                    pe_val = float(urow["pe"].iloc[0]) if len(urow) and pd.notna(urow["pe"].iloc[0]) else None
                                    roe_val = float(urow["roe_y0"].iloc[0]) if len(urow) and "roe_y0" in urow.columns and pd.notna(urow["roe_y0"].iloc[0]) else None
                                    sb.table("holdings").insert({
                                        "portfolio_id": port["id"], "ticker": c["ticker"], "name": c["name"],
                                        "sector": c["sector"], "allocation_pct": 0, "shares": qty,
                                        "sip_amount_inr": round(qty * px, 2), "price_at_entry": round(px, 2),
                                        "pe_at_entry": pe_val, "roe_at_entry": roe_val, "score_at_entry": sc_val,
                                    }).execute()
                        st.session_state.pop(f"review_data_{port['id']}", None)
                        st.success("Portfolio updated.")
                        st.rerun()

                    if st.button("✕ Close Review", key=f"close_review_{port['id']}"):
                        st.session_state.pop(f"review_data_{port['id']}", None)
                        st.rerun()


                col_r, col_d = st.columns([3, 1])
                with col_r:
                    new_name = st.text_input("Rename", value=port["name"], key=f"rename_{port['id']}", label_visibility="collapsed")
                    if new_name != port["name"]:
                        if st.button("Save Name", key=f"save_name_{port['id']}"):
                            try:
                                sb.table("portfolios").update({"name": new_name}).eq("id", port["id"]).execute()
                                st.success("Renamed!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Rename failed: {e}")
                with col_d:
                    if st.button("🗑️ Delete", key=f"delete_{port['id']}", type="secondary"):
                        st.session_state[f"confirm_delete_{port['id']}"] = True

                if st.session_state.get(f"confirm_delete_{port['id']}"):
                    st.warning("Are you sure? This cannot be undone.")
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("Yes, delete", key=f"confirm_yes_{port['id']}"):
                            try:
                                sb.table("holdings").delete().eq("portfolio_id", port["id"]).execute()
                                sb.table("portfolios").delete().eq("id", port["id"]).execute()
                                st.session_state.pop(f"confirm_delete_{port['id']}", None)
                                st.success("Deleted.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Delete failed: {e}")
                    with c2:
                        if st.button("Cancel", key=f"confirm_no_{port['id']}"):
                            st.session_state.pop(f"confirm_delete_{port['id']}", None)
                            st.rerun()
