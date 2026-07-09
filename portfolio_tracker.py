# TODO Sprint 8: Polish, XIRR everywhere, review reminders, paper auto-SIP, paper→real, Kite CSV.
import os
import re
import math
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import yfinance as yf
import pandas as pd
import pymupdf
from google import genai
from supabase import create_client, Client
from datetime import date, timedelta
from collections import Counter
import requests as _requests
import verdict_engine


# ══════════════════════════════════════════════
# LIGHTWEIGHT BOOK RAG (no ChromaDB needed)
# ══════════════════════════════════════════════
def load_books_simple():
    """Load investment books into text chunks for keyword search."""
    books = {
        "Graham": "The Intelligent Investor.pdf",
        "Greenblatt": "The Little Book That Still Beats the Market.pdf",
        "Dorsey": "The Five Rules for Successful Stock Investing.pdf",
    }
    chunks = []
    for author, filename in books.items():
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found. Skipping.")
            continue
        try:
            doc = pymupdf.open(filename)
            full_text = "\n".join(page.get_text() for page in doc)
            doc.close()

            paragraphs = full_text.split("\n\n")
            current = ""
            for para in paragraphs:
                para = para.strip()
                if not para or len(para) < 50:
                    continue
                if len(current) + len(para) < 1200:
                    current = current + "\n" + para if current else para
                else:
                    if len(current) >= 100:
                        chunks.append({"author": author, "text": current})
                    current = para
            if current and len(current) >= 100:
                chunks.append({"author": author, "text": current})
        except Exception as e:
            print(f"Warning: Could not load {filename}: {e}")

    print(f"Loaded {len(chunks)} book passages from {len(books)} books.")
    return chunks


def search_passages(chunks, query, n=3):
    """Simple keyword search over book chunks. Returns top N passages."""
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                  "have", "has", "do", "does", "did", "will", "would", "could",
                  "should", "may", "might", "shall", "can", "to", "of", "in",
                  "for", "on", "with", "at", "by", "from", "and", "or", "not",
                  "but", "if", "that", "this", "it", "its", "as", "about"}

    keywords = [w.lower() for w in re.split(r'\W+', query) if w.lower() not in stop_words and len(w) > 2]
    if not keywords:
        return []

    scored = []
    for chunk in chunks:
        text_lower = chunk["text"].lower()
        score = sum(text_lower.count(kw) for kw in keywords)
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda x: -x[0])
    return [s[1] for s in scored[:n]]


# Alert type → book search query mapping
ALERT_BOOK_QUERIES = {
    "score_drop": "deteriorating fundamentals declining competitive position when to sell warning signs",
    "quality_fail": "earnings quality non-recurring income value traps artificial profits cash flow",
    "price_crash": "Mr Market irrational prices holding through declines margin of safety buying opportunity panic",
    "opportunity": "buying undervalued stocks discount intrinsic value margin of safety quality companies",
    "review_due": "periodic review discipline portfolio maintenance rebalancing intelligent investor patience",
    "overvalued": "selling overpriced stocks taking profits margin of safety disappearing valuation stretched",
    "goal_drift": "falling behind investment goals compounding patience increasing contributions discipline",
    "sector_headwind": "sector rotation industry downturn diversification concentration risk cyclical",
    "new_entry": "new investment opportunities emerging companies quality discovery fresh screening",
    "watchlist_score_up": "improving fundamentals rising quality score upgrade strengthening competitive position",
    "watchlist_score_down": "deteriorating fundamentals declining score weakening position watch carefully",
    "watchlist_quality_flip": "earnings quality change cash flow quality reversal accounting red flags",
    "watchlist_near_low": "buying near 52-week low discount margin of safety patience value opportunity",
}

# Sector → Nifty sectoral index (yfinance tickers)
SECTOR_INDEX_MAP = {
    "Technology": "^CNXIT",
    "Financial Services": "^NSEBANK",
    "Industrials": "^CNXINFRA",
    "Basic Materials": "^CNXMETAL",
    "Consumer Cyclical": "^CNXAUTO",
    "Consumer Defensive": "^CNXFMCG",
    "Healthcare": "^CNXPHARMA",
    "Energy": "^CNXENERGY",
    "Real Estate": "^CNXREALTY",
    "Communication Services": "^CNXMEDIA",
}

def _html_esc(text):
    """Escape HTML special chars for Telegram."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
 
def send_telegram(chat_id, text, bot_token):
    """Send a message via Telegram Bot API (HTML parse mode). Non-blocking."""
    try:
        _requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        print(f"  Telegram send failed: {e}")

def send_review_email(recipient, subject, body, smtp_user, smtp_pass):
    """Send a plain-text review reminder email. Non-blocking."""
    try:
        msg = MIMEMultipart()
        msg["From"] = smtp_user
        msg["To"] = recipient
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        print(f"  ✓ Review reminder sent to {recipient}")
    except Exception as e:
        print(f"  ✗ Review email failed for {recipient}: {e}")

def compute_xirr_standalone(supabase, portfolio_id, current_value, nifty_shadow_value=None):
    """Compute XIRR for a portfolio and its Nifty shadow. Standalone — no Streamlit dependency.
    Returns (port_xirr_pct, nifty_xirr_pct) as percentages, or (None, None)."""
    try:
        from pyxirr import xirr
    except ImportError:
        return None, None
    try:
        txn_resp = supabase.table("sip_transactions").select(
            "transaction_date, amount_inr, transaction_type"
        ).eq("portfolio_id", str(portfolio_id)).order("transaction_date").execute()
        txns = txn_resp.data or []
    except Exception:
        return None, None
    if not txns:
        return None, None
    dates = []
    amounts = []
    for t in txns:
        d = date.fromisoformat(t["transaction_date"])
        amt = float(t.get("amount_inr") or 0)
        if t["transaction_type"] == "buy":
            amounts.append(-amt)
        else:
            amounts.append(amt)
        dates.append(d)
    # XIRR meaningless under 90 days
    if (date.today() - dates[0]).days < 90:
        return None, None
    today = date.today()
    dates.append(today)
    amounts.append(float(current_value))
    try:
        port_xirr = xirr(dates, amounts)
    except Exception:
        port_xirr = None
    port_xirr_pct = round(port_xirr * 100, 2) if port_xirr is not None else None
    nifty_xirr_pct = None
    if nifty_shadow_value and nifty_shadow_value > 0:
        nifty_amounts = amounts[:-1] + [float(nifty_shadow_value)]
        try:
            n_xirr = xirr(dates, nifty_amounts)
            nifty_xirr_pct = round(n_xirr * 100, 2) if n_xirr is not None else None
        except Exception:
            pass
    return port_xirr_pct, nifty_xirr_pct

def compute_diversification_score(holdings, universe_df=None):
    """Compute portfolio diversification score (0-100).
    Reilly & Brown Ch 6: HHI sector spread, concentration, holdings count, cap distribution."""
    if not holdings or len(holdings) == 0:
        return 0

    total_value = sum(h.get("current_value", 0) or h.get("sip_amount_inr", 0) or 0 for h in holdings)
    if total_value <= 0:
        return 0

    def _val(h):
        return h.get("current_value", 0) or h.get("sip_amount_inr", 0) or 0

    # ── 1. Sector HHI inverted (40%) ──
    sector_weights = {}
    for h in holdings:
        sec = h.get("sector", "Unknown") or "Unknown"
        sector_weights[sec] = sector_weights.get(sec, 0) + _val(h)
    sector_fracs = [v / total_value for v in sector_weights.values()]
    hhi = sum(f ** 2 for f in sector_fracs)
    n_sectors = max(len(sector_fracs), 1)
    hhi_min = 1.0 / n_sectors
    hhi_score = max(0, (1.0 - hhi) / (1.0 - hhi_min) * 100) if hhi_min < 1.0 else 0

    # ── 2. Single-stock concentration (20%) ──
    stock_fracs = [_val(h) / total_value for h in holdings]
    max_weight = max(stock_fracs) if stock_fracs else 1.0
    conc_score = max(0, min(100, (1.0 - max_weight) / 0.9 * 100))

    # ── 3. Holdings adequacy vs book minimum of 12 (20%) ──
    BOOK_MIN = 12
    adequacy_score = min(100, (len(holdings) / BOOK_MIN) * 100)

    # ── 4. Cap distribution (20%) ──
    cap_score = 50
    if universe_df is not None and not universe_df.empty:
        try:
            large_w, small_w = 0.0, 0.0
            for h in holdings:
                row = universe_df[universe_df["ticker"] == h.get("ticker", "")]
                tier = row.iloc[0].get("risk_tier", "Unknown") if not row.empty else "Unknown"
                frac = _val(h) / total_value
                if tier == "Large":
                    large_w += frac
                elif tier == "Small":
                    small_w += frac
            if small_w > 0.5:
                cap_score = max(0, 100 - (small_w - 0.5) * 200)
            elif large_w < 0.1 and len(holdings) >= 5:
                cap_score = max(0, large_w * 1000)
            else:
                cap_score = 80
        except Exception:
            pass

    final = (hhi_score * 0.4) + (conc_score * 0.2) + (adequacy_score * 0.2) + (cap_score * 0.2)
    return round(max(0, min(100, final)))

def _json_safe(obj):
    """Strip NaN/inf from any payload before it reaches Supabase.

    PostgREST's JSON encoder rejects NaN and Infinity. Python's json.dumps emits
    them happily, so the failure surfaces only at the HTTP boundary — where it
    kills the ENTIRE daily run, after earlier portfolios have already written.
    2026-07-10: one NaN from NIFTYBEES.NS took down the whole tracker.

    NULL is the honest representation of "we could not compute this". A zero
    would be a lie, and the risk metrics would propagate it into the Alpha Report.

    Three subtleties, all load-bearing:
      - bool is a subclass of int, so it must be matched BEFORE the numeric cases.
      - np.float64('nan') is NOT a Python float. isinstance(obj, float) misses it
        entirely. That is the version of this function that looks right and does
        nothing. Hence the .item() branch.
      - inf, not just nan. Every ratio in compute_portfolio_risk_metrics divides
        by something that can be zero (sigma, beta, tracking error).
    """
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if hasattr(obj, "item"):  # numpy scalar
        try:
            return _json_safe(obj.item())
        except Exception:
            return obj
    return obj


def _usable_close(hist, label):
    """Last FINITE, positive close from a yfinance history frame, else None.

    `hist.empty` is False when yfinance returns today's partially-formed row.
    For an INDEX (^NSEI) that row has a value the moment the session opens,
    because an index is computed. For an ETF (NIFTYBEES.NS) it is NaN until
    somebody actually trades. The old guard tested the container, not the value.

    Take the last row with a real close — not simply the last row.
    """
    try:
        if hist is None or hist.empty or "Close" not in hist:
            return None
        closes = hist["Close"].dropna()
        closes = closes[closes > 0]
        if closes.empty:
            print(f"Warning: {label} returned rows but no usable close.")
            return None
        px = float(closes.iloc[-1])
        if not math.isfinite(px) or px <= 0:
            return None
        return round(px, 2)
    except Exception as e:
        print(f"Warning: Could not read close for {label}: {e}")
        return None


def compute_portfolio_risk_metrics(holdings, universe_df=None, nifty_history=None):
    """Compute portfolio-level risk and performance metrics from Reilly & Brown.
    Ch 7: CAPM, Beta, Alpha (Jensen). Ch 18: Sharpe, Treynor, Sortino, IR.

    Returns dict with all computed metrics, or empty dict on failure."""
    import yfinance as yf
    from datetime import datetime, timedelta

    if not holdings or len(holdings) == 0:
        return {}

    RFR = 0.07  # India 10Y govt bond rate — TODO: fetch dynamically via get_web_context
    result = {}

    total_value = sum(h.get("current_value", 0) or h.get("sip_amount_inr", 0) or 0 for h in holdings)
    if total_value <= 0:
        return {}

    # ── 1. Portfolio Beta = Σ wi × βi (Ch 7) ──
    weighted_beta = 0.0
    beta_available = 0
    for h in holdings:
        ticker = h.get("ticker", "")
        weight = (h.get("current_value", 0) or h.get("sip_amount_inr", 0) or 0) / total_value
        beta = None
        # Try universe CSV first (faster)
        if universe_df is not None and not universe_df.empty:
            row = universe_df[universe_df["ticker"] == ticker]
            if not row.empty and "beta" in row.columns:
                beta = row.iloc[0].get("beta")
        # Fallback to yfinance
        if beta is None or (hasattr(beta, '__float__') and str(beta) == 'nan'):
            try:
                info = yf.Ticker(ticker).info
                beta = info.get("beta")
            except Exception:
                pass
        if beta is not None and str(beta) != 'nan':
            weighted_beta += weight * float(beta)
            beta_available += 1

    if beta_available > 0:
        result["portfolio_beta"] = round(weighted_beta, 3)

    # ── 2. Portfolio returns from price history (1 year) ──
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        tickers = [h["ticker"] for h in holdings]
        weights = [(h.get("current_value", 0) or h.get("sip_amount_inr", 0) or 0) / total_value for h in holdings]

        # Fetch daily returns
        hist = yf.download(tickers, start=start_date.strftime("%Y-%m-%d"),
                           end=end_date.strftime("%Y-%m-%d"), progress=False)
        if hist.empty:
            return result

        # Handle single vs multiple tickers
        if len(tickers) == 1:
            daily_prices = hist["Close"].to_frame(tickers[0])
        else:
            daily_prices = hist["Close"]

        daily_returns = daily_prices.pct_change().dropna()
        if daily_returns.empty:
            return result

        # Portfolio daily returns (weighted)
        port_returns = None
        for i, ticker in enumerate(tickers):
            if ticker in daily_returns.columns:
                col = daily_returns[ticker] * weights[i]
                port_returns = col if port_returns is None else port_returns + col

        if port_returns is None:
            return result

        # Annualize
        trading_days = 252
        port_annual_return = port_returns.mean() * trading_days
        port_annual_std = port_returns.std() * (trading_days ** 0.5)

        # Written unconditionally before; round(nan, 4) is nan.
        if math.isfinite(port_annual_return):
            result["annual_return"] = round(port_annual_return, 4)
        if math.isfinite(port_annual_std):
            result["annual_std"] = round(port_annual_std, 4)

        # ── 3. Sharpe Ratio = (Rp - RFR) / σp (Ch 18) ──
        if port_annual_std > 0:
            result["sharpe_ratio"] = round((port_annual_return - RFR) / port_annual_std, 3)

        # ── 4. Treynor Ratio = (Rp - RFR) / βp (Ch 18) ──
        # `nan != 0` is True. Without isfinite(), a NaN beta propagates straight
        # into treynor and jensen_alpha. The other three ratios are already
        # guarded by `> 0` comparisons, which are False for NaN.
        _beta = result.get("portfolio_beta")
        if _beta is not None and math.isfinite(_beta) and _beta != 0:
            result["treynor_ratio"] = round((port_annual_return - RFR) / result["portfolio_beta"], 4)

        # ── 5. Jensen's Alpha = Rp - RFR - β(Rm - RFR) (Ch 18) ──
        # Need market return (Nifty 50)
        try:
            nifty = yf.download("^NSEI", start=start_date.strftime("%Y-%m-%d"),
                                end=end_date.strftime("%Y-%m-%d"), progress=False)
            if not nifty.empty:
                nifty_returns = nifty["Close"].pct_change().dropna()
                market_annual_return = nifty_returns.mean() * trading_days
                result["market_return"] = round(market_annual_return, 4)

                if "portfolio_beta" in result:
                    alpha = port_annual_return - RFR - result["portfolio_beta"] * (market_annual_return - RFR)
                    result["jensen_alpha"] = round(alpha, 4)

                # ── 6. Information Ratio = (Rp - Rb) / σ(Rp - Rb) (Ch 18) ──
                # Align dates
                aligned = port_returns.to_frame("port").join(nifty_returns.to_frame("nifty"), how="inner")
                if not aligned.empty:
                    tracking_diff = aligned["port"] - aligned["nifty"]
                    tracking_error = tracking_diff.std() * (trading_days ** 0.5)
                    if tracking_error > 0:
                        ir = (port_annual_return - market_annual_return) / tracking_error
                        result["information_ratio"] = round(ir, 3)
        except Exception:
            pass

        # ── 7. Sortino Ratio = (Rp - τ) / DRp (Ch 18) ──
        # τ = target return, use RFR; DR = downside deviation
        daily_rfr = RFR / trading_days
        downside = port_returns[port_returns < daily_rfr]
        if len(downside) > 0:
            downside_dev = ((downside - daily_rfr) ** 2).mean() ** 0.5 * (trading_days ** 0.5)
            if downside_dev > 0:
                result["sortino_ratio"] = round((port_annual_return - RFR) / downside_dev, 3)

        # ── 8. Semi-deviation (Ch 6) ──
        below_mean = port_returns[port_returns < port_returns.mean()]
        if len(below_mean) > 0:
            semi_dev = below_mean.std() * (trading_days ** 0.5)
            result["semi_deviation"] = round(semi_dev, 4)

        # ── 9. Max drawdown ──
        cumulative = (1 + port_returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        result["max_drawdown"] = round(drawdown.min(), 4)

        # ── 10. CAPM expected return = RFR + β(Rm - RFR) (Ch 7) ──
        if "portfolio_beta" in result and "market_return" in result:
            expected = RFR + result["portfolio_beta"] * (result["market_return"] - RFR)
            result["capm_expected_return"] = round(expected, 4)

    except Exception as e:
        print(f"  Risk metrics computation error (non-blocking): {e}")

    return result
 
def run_daily_tracker():
    print("Initiating Kordent Daily Portfolio Audit...")

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY environment variables.")

    supabase: Client = create_client(url, key)

    # ── Load books for RAG ──
    book_chunks = load_books_simple()

    # ── Load fresh universe CSV ──
    universe_df = None
    if os.path.exists("universe_scored.csv"):
        universe_df = pd.read_csv("universe_scored.csv")
        print(f"Loaded universe: {len(universe_df)} stocks")
    else:
        print("Warning: universe_scored.csv not found.")

    # ── Fetch portfolios, holdings, profiles ──
    portfolios_resp = supabase.table("portfolios").select("*").execute()
    portfolios = portfolios_resp.data

    if not portfolios:
        print("No portfolios found. Exiting.")
        return

    holdings_resp = supabase.table("holdings").select("*").execute()
    holdings = holdings_resp.data

    # ── Fetch Nifty 50 close price once ──
    nifty_close = None
    try:
        nifty_close = _usable_close(yf.Ticker("^NSEI").history(period="5d"), "^NSEI")
        if nifty_close is not None:
            print(f"Nifty 50 close: {nifty_close:,.2f}")
    except Exception as e:
        print(f"Warning: Could not fetch Nifty 50: {e}")

    # ── Fetch Nifty BeES for shadow portfolio ──
    nifty_bees_price = None
    try:
        nifty_bees_price = _usable_close(
            yf.Ticker("NIFTYBEES.NS").history(period="5d"), "NIFTYBEES.NS")
    except Exception as e:
        print(f"Warning: Could not fetch Nifty BeES: {e}")

    if nifty_bees_price is None:
        # Do NOT silently substitute a stale price. Every nifty-relative metric
        # (shadow value, nifty_units, nifty_xirr) is skipped for this run and
        # written as NULL. One missing day is recoverable; a ledger with a wrong
        # entry price is not.
        print("Warning: no usable NIFTYBEES close. Nifty shadow metrics are NULL "
              "for this run; the ledger will NOT be written with a guessed price.")
    else:
        print(f"Nifty BeES close: {nifty_bees_price:,.2f}")

    # ── Fetch all transactions once ──
    txn_resp = supabase.table("sip_transactions").select("portfolio_id, amount_inr, transaction_type, nifty_units").execute()
    all_txns = txn_resp.data or []

    # ── Bootstrap: create genesis transactions for portfolios with holdings but no transactions ──
    _txn_port_ids = set(t["portfolio_id"] for t in all_txns)
    holdings_resp_all = supabase.table("holdings").select("portfolio_id, ticker, shares, price_at_entry, entry_date").execute()
    _all_h = holdings_resp_all.data or []
    _ports_needing_bootstrap = set()
    for h in _all_h:
        if h["portfolio_id"] not in _txn_port_ids:
            _ports_needing_bootstrap.add(h["portfolio_id"])
    if _ports_needing_bootstrap:
        _bootstrap_today = date.today().isoformat()
        print(f"Bootstrapping {len(_ports_needing_bootstrap)} portfolios with genesis transactions...")
        for h in _all_h:
            if h["portfolio_id"] in _ports_needing_bootstrap:
                _h_shares = float(h.get("shares") or 0)
                _h_price = float(h.get("price_at_entry") or 0)
                if _h_shares <= 0 or _h_price <= 0:
                    continue
                _h_amt = round(_h_shares * _h_price, 2)
                _h_date = (h.get("entry_date") or _bootstrap_today)[:10]
                # `nifty_bees_price and ...` was TRUE for NaN — bool(nan) is True.
                # This wrote nifty_units=NaN into the permanent ledger. The twin
                # of this line at the paper-SIP site has `nifty_bees_price > 0`,
                # which is False for NaN, and was saved by that accident.
                _nifty_u = (round(_h_amt / nifty_bees_price, 6)
                            if (nifty_bees_price and nifty_bees_price > 0 and _h_amt > 0)
                            else None)
                _port_user = next((p["user_id"] for p in portfolios if p["id"] == h["portfolio_id"]), None)
                if _port_user:
                    supabase.table("sip_transactions").insert(_json_safe({
                        "portfolio_id": h["portfolio_id"],
                        "user_id": _port_user,
                        "ticker": h["ticker"],
                        "shares": _h_shares,
                        "price": _h_price,
                        "amount_inr": _h_amt,
                        "transaction_type": "buy",
                        "transaction_date": _h_date,
                        "nifty_price": nifty_bees_price,
                        "nifty_units": _nifty_u,
                    })).execute()
        # Refresh transactions after bootstrap
        txn_resp = supabase.table("sip_transactions").select("portfolio_id, amount_inr, transaction_type, nifty_units").execute()
        all_txns = txn_resp.data or []
        print("Bootstrap complete.")

    price_cache = {}
    today_str = date.today().isoformat()
    all_alerts = []
    _port_values = {}  # port_id -> (value, return_pct) for Telegram digest
 
    # ── Pre-load user profiles (Telegram, email, name) ──
    _tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    _smtp_user = os.environ.get("ALERT_EMAIL", "")
    _smtp_pass = os.environ.get("ALERT_EMAIL_PASSWORD", "")
    _tg_map = {}  # user_id -> chat_id
    _user_profiles = {}  # user_id -> {email, full_name}
    try:
        _prof_resp = supabase.table("profiles").select("id, email, full_name, telegram_chat_id").execute()
        for p in (_prof_resp.data or []):
            _user_profiles[p["id"]] = {"email": p.get("email"), "full_name": p.get("full_name")}
            if _tg_token and p.get("telegram_chat_id"):
                _tg_map[p["id"]] = p["telegram_chat_id"]
        if _tg_map:
            print(f"Telegram: {len(_tg_map)} users with linked accounts.")
    except Exception as e:
        print(f"Profile pre-load failed: {e}")

    for port in portfolios:
        port_id = port["id"]
        user_id = port["user_id"]
        port_holdings = [h for h in holdings if h["portfolio_id"] == port_id]

        if not port_holdings:
            continue

        total_invested = 0.0
        current_total_value = 0.0

        for holding in port_holdings:
            ticker = holding["ticker"]
            shares = holding["shares"]
            invested_inr = holding["sip_amount_inr"]

            if ticker not in price_cache:
                try:
                    info = yf.Ticker(ticker).fast_info
                    price_cache[ticker] = info.last_price
                except Exception as e:
                    print(f"Warning: Failed to fetch {ticker}: {e}")
                    price_cache[ticker] = holding["price_at_entry"]

            live_price = price_cache[ticker]
            total_invested += invested_inr
            current_total_value += (shares * live_price)

        return_pct = ((current_total_value - total_invested) / total_invested) * 100 if total_invested > 0 else 0.0

        # ── 1. Update leaderboard snapshot ──
        supabase.table("portfolios").update(_json_safe({
            "current_value": round(current_total_value, 2),
            "current_return_pct": round(return_pct, 2)
        })).eq("id", port_id).execute()

        # ── 2. Compute cumulative invested & Nifty shadow from transaction ledger ──
        port_txns = [t for t in all_txns if t["portfolio_id"] == port_id]
        cumulative_invested = 0.0
        total_nifty_units = 0.0
        for t in port_txns:
            amt = float(t.get("amount_inr") or 0)
            if t.get("transaction_type") == "buy":
                cumulative_invested += amt
            else:
                cumulative_invested -= amt
            total_nifty_units += float(t.get("nifty_units") or 0)

        nifty_shadow = (round(total_nifty_units * nifty_bees_price, 2)
                        if (nifty_bees_price and nifty_bees_price > 0 and total_nifty_units > 0)
                        else None)
        if nifty_shadow is not None and not math.isfinite(nifty_shadow):
            nifty_shadow = None

        # ── 3. Log history ──
        history_row = {
            "portfolio_id": port_id,
            "date": today_str,
            "total_value": round(current_total_value, 2),
            "daily_return_pct": round(return_pct, 2),
        }
        if cumulative_invested > 0:
            history_row["cumulative_invested"] = round(cumulative_invested, 2)
        if nifty_shadow is not None:
            history_row["nifty_shadow_value"] = nifty_shadow
        if nifty_close is not None:
            history_row["nifty_value"] = nifty_close

        supabase.table("portfolio_history").upsert(
            _json_safe(history_row), on_conflict="portfolio_id,date"
        ).execute()

        # ── Compute & store XIRR ──
        _p_xirr, _n_xirr = compute_xirr_standalone(supabase, port_id, current_total_value, nifty_shadow)
        _xirr_update = {}
        if _p_xirr is not None:
            _xirr_update["xirr_pct"] = _p_xirr
        if _n_xirr is not None:
            _xirr_update["nifty_xirr_pct"] = _n_xirr
        if _xirr_update:
            try:
                supabase.table("portfolios").update(_json_safe(_xirr_update)).eq("id", port_id).execute()
            except Exception as e:
                print(f"  XIRR store failed (non-blocking): {e}")

        _xirr_str = f" | XIRR {_p_xirr:+.1f}%" if _p_xirr is not None else ""
        # Diversification score
        _div_score = compute_diversification_score(port_holdings, universe_df)
        try:
            supabase.table("portfolios").update(_json_safe({
                "diversification_score": _div_score
            })).eq("id", port_id).execute()
        except Exception as e:
            print(f"  Diversification score store failed (non-blocking): {e}")

        _div_label = "🟢" if _div_score >= 70 else "🟡" if _div_score >= 40 else "🔴"
        print(f"Updated [{port['name']}]: Value {current_total_value:,.2f} | Return {return_pct:+.2f}%{_xirr_str} | Div {_div_label}{_div_score}")
        # Sprint 11: Portfolio risk & performance metrics (Reilly & Brown Ch 7, 18)
        try:
            _risk = compute_portfolio_risk_metrics(port_holdings, universe_df)
            if _risk:
                _risk_update = {}
                for k in ["portfolio_beta", "sharpe_ratio", "sortino_ratio", "jensen_alpha",
                           "treynor_ratio", "information_ratio", "max_drawdown",
                           "capm_expected_return", "semi_deviation", "annual_return", "annual_std"]:
                    if k in _risk:
                        _risk_update[k] = _risk[k]
                if _risk_update:
                    supabase.table("portfolios").update(_json_safe(_risk_update)).eq("id", port_id).execute()
                _beta_str = f" | β={_risk.get('portfolio_beta', '?')}"
                _sharpe_str = f" | Sharpe={_risk.get('sharpe_ratio', '?')}"
                _alpha_str = f" | α={_risk.get('jensen_alpha', '?')}"
                print(f"  Risk metrics: {_beta_str}{_sharpe_str}{_alpha_str}")
        except Exception as e:
            print(f"  Risk metrics failed (non-blocking): {e}")
        _port_values[port_id] = (round(current_total_value, 2), round(return_pct, 2), _p_xirr, _n_xirr)

        # ── SIP budget management (30% cap for mid-cycle opportunities) ──
        sip_amount = port.get("sip_amount") or 0
        opp_budget = sip_amount * 0.3
        budget_reset_date = port.get("sip_budget_reset_date")
        sip_budget = port.get("sip_budget_remaining")

        needs_reset = False
        if sip_budget is None or budget_reset_date is None:
            needs_reset = True
        else:
            try:
                reset_dt = date.fromisoformat(str(budget_reset_date))
                if reset_dt.month != date.today().month or reset_dt.year != date.today().year:
                    needs_reset = True
            except (ValueError, TypeError):
                needs_reset = True

        if needs_reset:
            sip_budget = opp_budget
            supabase.table("portfolios").update(_json_safe({
                "sip_budget_remaining": round(sip_budget, 2),
                "sip_budget_reset_date": today_str,
            })).eq("id", port_id).execute()
            if sip_amount > 0:
                print(f"  Budget reset for [{port['name']}]: ₹{sip_budget:,.0f}")

        # ══════════════════════════════════════
        # 3. ALERT DETECTION (with book passages)
        # ══════════════════════════════════════

        def make_alert(alert_type, ticker, headline, detail, book_query_key=None):
            """Helper to build alert dict with book passage attached."""
            passages = []
            if book_chunks and book_query_key:
                query = ALERT_BOOK_QUERIES.get(book_query_key, "")
                if query:
                    results = search_passages(book_chunks, query, n=2)
                    passages = [{"author": r["author"], "text": r["text"][:400]} for r in results]

            return {
                "portfolio_id": port_id,
                "user_id": user_id,
                "alert_type": alert_type,
                "ticker": ticker,
                "headline": headline,
                "detail": {**detail, "book_passages": passages},
                "alert_date": today_str,
            }

        # ── 3a. Review due / upcoming reminders ──
        review_date = port.get("next_review_date")
        if review_date:
            try:
                rd = date.fromisoformat(str(review_date))
                _tomorrow = date.today() + timedelta(days=1)
                _prof = _user_profiles.get(user_id, {})
                _uname = _prof.get("full_name") or (_prof.get("email") or "").split("@")[0].title()
                _uemail = _prof.get("email")
                _pname = port.get("name", "Portfolio")

                if rd == _tomorrow:
                    # Day-before reminder
                    _msg = f"Tomorrow is your review day for {_pname}. Prepare to assess performance, rebalance, and decide on any changes."
                    if _tg_token and user_id in _tg_map:
                        send_telegram(_tg_map[user_id], f"📅 <b>{_html_esc(_uname)}, review tomorrow</b>\n\n{_html_esc(_msg)}\n\n<a href='https://kordent.streamlit.app'>Open Kordent</a>", _tg_token)
                    if _uemail and _smtp_user and _smtp_pass:
                        send_review_email(_uemail, f"📅 {_uname}, your {_pname} review is tomorrow", f"{_uname},\n\n{_msg}\n\nOpen Kordent: https://kordent.streamlit.app\n\n— Kordent", _smtp_user, _smtp_pass)

                elif rd == date.today():
                    # Day-of reminder
                    _msg = f"Your review for {_pname} is due today. Open Kordent to run your review."
                    if _tg_token and user_id in _tg_map:
                        send_telegram(_tg_map[user_id], f"🔔 <b>{_html_esc(_uname)}, review day!</b>\n\n{_html_esc(_msg)}\n\n<a href='https://kordent.streamlit.app'>Open Kordent</a>", _tg_token)
                    if _uemail and _smtp_user and _smtp_pass:
                        send_review_email(_uemail, f"🔔 {_uname}, your {_pname} review is today", f"{_uname},\n\n{_msg}\n\nOpen Kordent: https://kordent.streamlit.app\n\n— Kordent", _smtp_user, _smtp_pass)

                elif rd < date.today():
                    # Overdue — store as alert for weekly mentor
                    all_alerts.append(make_alert(
                        "review_due", "_review",
                        f"Portfolio review overdue — was due {review_date}",
                        {"days_overdue": (date.today() - rd).days},
                        "review_due"
                    ))
            except (ValueError, TypeError):
                pass

        if universe_df is None:
            continue

        # ── 3b. Danger alerts for holdings ──
        held_tickers = set()
        held_sectors = []
        for holding in port_holdings:
            ticker = holding["ticker"]
            held_tickers.add(ticker)
            held_sectors.append(holding.get("sector", ""))

            entry_score = holding.get("score_at_entry") or 0
            entry_price = holding.get("price_at_entry") or 0
            live_price = price_cache.get(ticker, entry_price)

            row = universe_df[universe_df["ticker"] == ticker]
            if row.empty:
                continue

            current_score = int(row["score"].iloc[0]) if pd.notna(row["score"].iloc[0]) else 0
            quality_pass = bool(row["quality_pass"].iloc[0]) if "quality_pass" in row.columns and pd.notna(row["quality_pass"].iloc[0]) else True

            if entry_score - current_score >= 2:
                _dd_headline = f"{holding.get('name', ticker)} score dropped {entry_score} -> {current_score}"
                all_alerts.append(make_alert(
                    "danger", ticker, _dd_headline,
                    {"name": holding.get("name", ticker), "entry_score": entry_score,
                     "current_score": current_score, "reason": "score_drop"},
                    "score_drop"
                ))
                if _tg_token and user_id in _tg_map:
                    send_telegram(_tg_map[user_id], f"⚠️ <b>{_html_esc(_dd_headline)}</b>\n\n<a href='https://kordent.streamlit.app'>Open Kordent</a>", _tg_token)

            if not quality_pass:
                _qf_headline = f"{holding.get('name', ticker)} flagged as potential value trap"
                all_alerts.append(make_alert(
                    "danger", ticker, _qf_headline,
                    {"name": holding.get("name", ticker), "reason": "quality_fail"},
                    "quality_fail"
                ))
                if _tg_token and user_id in _tg_map:
                    send_telegram(_tg_map[user_id], f"⚠️ <b>{_html_esc(_qf_headline)}</b>\n\n<a href='https://kordent.streamlit.app'>Open Kordent</a>", _tg_token)

            if entry_price > 0:
                stock_return = ((live_price - entry_price) / entry_price) * 100
                if stock_return < -20:
                    _pc_headline = f"{holding.get('name', ticker)} down {stock_return:.0f}% from entry"
                    all_alerts.append(make_alert(
                        "danger", ticker, _pc_headline,
                        {"name": holding.get("name", ticker), "reason": "price_crash",
                         "entry_price": entry_price, "current_price": round(live_price, 2),
                         "return_pct": round(stock_return, 1)},
                        "price_crash"
                    ))
                    if _tg_token and user_id in _tg_map:
                        send_telegram(_tg_map[user_id], f"⚠️ <b>{_html_esc(_pc_headline)}</b>\n\n<a href='https://kordent.streamlit.app'>Open Kordent</a>", _tg_token)

            # ── Overvalued: Graham margin of safety eroding ──
            current_pe = float(row["pe"].iloc[0]) if pd.notna(row["pe"].iloc[0]) else None
            current_pb = float(row["pb"].iloc[0]) if "pb" in row.columns and pd.notna(row["pb"].iloc[0]) else None

            overvalued_reasons = []
            if current_pe and current_pe > 18:
                overvalued_reasons.append(f"PE {current_pe:.1f} > 18")
            if current_pb and current_pb > 1.8:
                overvalued_reasons.append(f"PB {current_pb:.1f} > 1.8")

            if overvalued_reasons:
                all_alerts.append(make_alert(
                    "overvalued", ticker,
                    f"{holding.get('name', ticker)} may be overvalued — {', '.join(overvalued_reasons)}",
                    {"name": holding.get("name", ticker), "reason": "overvalued",
                     "pe": current_pe, "pb": current_pb},
                    "overvalued"
                ))

        # ── 3c. Opportunity alerts ──
        investor_type = port.get("investor_type", "balanced")

        opps = universe_df[
            (universe_df["score"] >= 4) &
            (universe_df["quality_pass"] == True) &
            (~universe_df["ticker"].isin(held_tickers)) &
            (universe_df["pe"] > 0) &
            (pd.notna(universe_df["pe"]))
        ].copy()

        if investor_type == "defensive":
            opps = opps[opps["graham_pass"] == True]
        elif investor_type == "enterprising":
            opps = opps[opps["trajectory_pass"] == True]
        else:
            opps = opps[(opps["greenblatt_pass"] == True) | (opps["dorsey_pass"] == True)]

        sector_counts = Counter(held_sectors)
        full_sectors = [s for s, c in sector_counts.items() if c >= 2]
        if full_sectors:
            opps = opps[~opps["sector"].isin(full_sectors)]

        opps = opps.sort_values("pe").head(3)

        for _, opp_row in opps.iterrows():
            stock_price = round(float(opp_row["price"]), 2) if pd.notna(opp_row.get("price")) else 0
            can_afford = sip_budget >= stock_price and stock_price > 0
            act_now = can_afford and sip_amount > 0
            suggested_shares = int(sip_budget // stock_price) if can_afford else 0
            suggested_amount = round(suggested_shares * stock_price, 2) if suggested_shares > 0 else 0

            opp_score = int(opp_row["score"]) if pd.notna(opp_row.get("score")) else 0
            opp_pass_dict = {
                "graham_pass": bool(opp_row.get("graham_pass")) if pd.notna(opp_row.get("graham_pass")) else False,
                "greenblatt_pass": bool(opp_row.get("greenblatt_pass")) if pd.notna(opp_row.get("greenblatt_pass")) else False,
                "dorsey_pass": bool(opp_row.get("dorsey_pass")) if pd.notna(opp_row.get("dorsey_pass")) else False,
                "trajectory_pass": bool(opp_row.get("trajectory_pass")) if pd.notna(opp_row.get("trajectory_pass")) else False,
                "lynch_pass": bool(opp_row.get("lynch_pass")) if pd.notna(opp_row.get("lynch_pass")) else False,
            }
            opp_verdict = verdict_engine.get_verdict_tier(opp_score, True, opp_pass_dict)
            opp_emoji = verdict_engine.VERDICT_EMOJI.get(opp_verdict, "")

            all_alerts.append(make_alert(
                "opportunity", opp_row["ticker"],
                f"{opp_emoji} {opp_row.get('name', opp_row['ticker'])} — {opp_verdict} ({opp_score}/5) — fits your {investor_type} profile",
                {"name": str(opp_row.get("name", opp_row["ticker"])),
                 "sector": str(opp_row.get("sector", "N/A")),
                 "price": stock_price,
                 "pe": round(float(opp_row["pe"]), 2) if pd.notna(opp_row.get("pe")) else 0,
                 "roe_pct": round(float(opp_row["roe_pct"]), 2) if pd.notna(opp_row.get("roe_pct")) else 0,
                 "score": opp_score,
                 "verdict": opp_verdict,
                 "act_now": act_now,
                 "suggested_shares": suggested_shares,
                 "suggested_amount": suggested_amount,
                 "budget_remaining": round(sip_budget, 2)},
                "opportunity"
            ))
        # ── 3d. Portfolio-level health warnings ──
        if len(port_holdings) >= 3:
            # Sector concentration
            sector_weights = Counter(held_sectors)
            total_h = len(held_sectors)
            for sector, count in sector_weights.items():
                weight = count / total_h
                if weight > 0.4:
                    all_alerts.append(make_alert(
                        "danger", "_portfolio",
                        f"Portfolio {port['name']}: {sector} is {weight*100:.0f}% of holdings (>40%)",
                        {"reason": "sector_concentration", "sector": sector, "weight_pct": round(weight * 100)},
                        "review_due"
                    ))

            # Diversification score (HHI)
            weights = [count / total_h for count in sector_weights.values()]
            hhi = sum(w ** 2 for w in weights)
            div_score = round((1 - hhi) * 100)
            if div_score < 50:
                all_alerts.append(make_alert(
                    "danger", "_portfolio",
                    f"Portfolio {port['name']}: diversification score is {div_score}/100 (critical)",
                    {"reason": "low_diversification", "score": div_score},
                    "review_due"
                ))

    # ── Sector headwind: top-weighted sector index dropped >10% in 30 days ──
        if held_sectors:
            sector_weights = Counter(held_sectors)
            top_sector = sector_weights.most_common(1)[0][0]
            index_ticker = SECTOR_INDEX_MAP.get(top_sector)

            if index_ticker:
                try:
                    idx_hist = yf.Ticker(index_ticker).history(period="1mo")
                    if len(idx_hist) >= 2:
                        idx_start = float(idx_hist["Close"].iloc[0])
                        idx_end = float(idx_hist["Close"].iloc[-1])
                        idx_return = ((idx_end - idx_start) / idx_start) * 100
                        if idx_return < -10:
                            alloc_pct = (sector_weights[top_sector] / len(held_sectors)) * 100
                            all_alerts.append(make_alert(
                                "sector_headwind", "_portfolio",
                                f"{top_sector} index down {idx_return:.1f}% this month — {alloc_pct:.0f}% of {port['name']}",
                                {"reason": "sector_headwind", "sector": top_sector,
                                 "index_return_pct": round(idx_return, 1),
                                 "portfolio_weight_pct": round(alloc_pct, 1)},
                                "sector_headwind"
                            ))
                except Exception as e:
                    print(f"Sector index check failed for {top_sector}: {e}")

            # ── Goal drift: trailing CAGR < 80% of needed CAGR ──
            target_amount = port.get("target_amount")
            target_date_str = port.get("target_date")
            if target_amount and target_date_str:
                try:
                    target_dt = date.fromisoformat(str(target_date_str))
                    months_remaining = max(1, (target_dt - date.today()).days / 30.44)
    
                    # Need 6+ months of history before judging trajectory
                    hist_resp = supabase.table("portfolio_history").select("date, total_value").eq(
                        "portfolio_id", port_id
                    ).order("date").execute()
                    hist_rows = hist_resp.data
    
                    if len(hist_rows) >= 180:  # ~6 months of weekday entries
                        first_val = float(hist_rows[0]["total_value"])
                        first_date = date.fromisoformat(hist_rows[0]["date"])
                        days_active = max(1, (date.today() - first_date).days)
    
                        if first_val > 0:
                            actual_cagr = (current_total_value / first_val) ** (365 / days_active) - 1
    
                            sip_monthly = port.get("sip_amount", 0) or 0
                            # Approximate needed CAGR (ignoring SIP for simplicity — full math in goal tracker)
                            if current_total_value > 0:
                                needed_cagr = (float(target_amount) / current_total_value) ** (12 / months_remaining) - 1
    
                                if needed_cagr > 0 and actual_cagr < (0.8 * needed_cagr):
                                    severity = "danger" if actual_cagr < (0.5 * needed_cagr) else "goal_drift"
                                    all_alerts.append(make_alert(
                                        severity, "_portfolio",
                                        f"{port['name']} trailing behind goal — actual {actual_cagr*100:.1f}% vs needed {needed_cagr*100:.1f}%",
                                        {"reason": "goal_drift",
                                         "actual_cagr_pct": round(actual_cagr * 100, 1),
                                         "needed_cagr_pct": round(needed_cagr * 100, 1),
                                         "target_amount": float(target_amount),
                                         "months_remaining": round(months_remaining)},
                                        "goal_drift"
                                    ))
                except (ValueError, TypeError) as e:
                    print(f"Goal drift check failed for {port['name']}: {e}")

    # ══════════════════════════════════════
    # 3d. WATCHLIST MONITORING ALERTS
    # ══════════════════════════════════════
    if universe_df is not None:
        try:
            wl_resp = supabase.table("watchlist").select("*").execute()
            wl_items = wl_resp.data or []
        except Exception as e:
            print(f"Warning: Could not fetch watchlist: {e}")
            wl_items = []

        if wl_items:
            # Get yesterday's scores from score_history for change detection
            prev_scores = {}
            try:
                latest_hist = supabase.table("score_history").select("date").order(
                    "date", desc=True
                ).limit(1).execute()
                if latest_hist.data:
                    prev_date = latest_hist.data[0]["date"]
                    if prev_date != today_str:
                        prev_resp = supabase.table("score_history").select(
                            "ticker,score,quality_pass"
                        ).eq("date", prev_date).execute()
                        for row in (prev_resp.data or []):
                            prev_scores[row["ticker"]] = {
                                "score": row.get("score"),
                                "quality_pass": row.get("quality_pass"),
                            }
            except Exception as e:
                print(f"Warning: Could not fetch previous scores: {e}")

            wl_alert_count = 0
            for wl in wl_items:
                wl_ticker = wl["ticker"]
                wl_user_id = wl["user_id"]

                row = universe_df[universe_df["ticker"] == wl_ticker]
                if row.empty:
                    continue

                cur_score = int(row["score"].iloc[0]) if pd.notna(row["score"].iloc[0]) else None
                cur_quality = bool(row["quality_pass"].iloc[0]) if "quality_pass" in row.columns and pd.notna(row["quality_pass"].iloc[0]) else None

                # Determine previous score: prefer yesterday's score_history, fall back to score_when_added
                prev = prev_scores.get(wl_ticker)
                if prev and prev.get("score") is not None:
                    prev_score = prev["score"]
                    prev_quality = prev.get("quality_pass")
                else:
                    prev_score = wl.get("score_when_added")
                    prev_quality = wl.get("quality_when_added")

                wl_name = wl.get("name") or wl_ticker

                def wl_alert(alert_type, headline, detail, book_key):
                    passages = []
                    if book_chunks and book_key:
                        query = ALERT_BOOK_QUERIES.get(book_key, "")
                        if query:
                            results = search_passages(book_chunks, query, n=2)
                            passages = [{"author": r["author"], "text": r["text"][:400]} for r in results]
                    return {
                        "portfolio_id": None,
                        "user_id": wl_user_id,
                        "alert_type": alert_type,
                        "ticker": wl_ticker,
                        "headline": headline,
                        "detail": {**detail, "book_passages": passages},
                        "alert_date": today_str,
                    }

                # Score up
                if cur_score is not None and prev_score is not None and cur_score > prev_score:
                    all_alerts.append(wl_alert(
                        "watchlist_score_up",
                        f"👁 {wl_name} score improved {prev_score} → {cur_score}/5",
                        {"prev_score": prev_score, "current_score": cur_score, "source": "watchlist"},
                        "watchlist_score_up"
                    ))
                    wl_alert_count += 1

                # Score down
                if cur_score is not None and prev_score is not None and cur_score < prev_score:
                    all_alerts.append(wl_alert(
                        "watchlist_score_down",
                        f"👁 {wl_name} score dropped {prev_score} → {cur_score}/5",
                        {"prev_score": prev_score, "current_score": cur_score, "source": "watchlist"},
                        "watchlist_score_down"
                    ))
                    wl_alert_count += 1

                # Quality flip
                if cur_quality is not None and prev_quality is not None and cur_quality != prev_quality:
                    flip_dir = "PASS" if cur_quality else "FAIL"
                    all_alerts.append(wl_alert(
                        "watchlist_quality_flip",
                        f"👁 {wl_name} quality flipped to {flip_dir}",
                        {"previous": prev_quality, "current": cur_quality, "source": "watchlist"},
                        "watchlist_quality_flip"
                    ))
                    wl_alert_count += 1

                # Near 52-week low (within 5%)
                try:
                    w52_low = float(row["week52_low"].iloc[0]) if pd.notna(row.get("week52_low", pd.Series([None])).iloc[0]) else None
                    cur_price = price_cache.get(wl_ticker)
                    if cur_price is None:
                        try:
                            cur_price = yf.Ticker(wl_ticker).fast_info.last_price
                            price_cache[wl_ticker] = cur_price
                        except Exception:
                            cur_price = None

                    if w52_low and cur_price and w52_low > 0:
                        pct_above_low = ((cur_price - w52_low) / w52_low) * 100
                        if pct_above_low <= 5:
                            all_alerts.append(wl_alert(
                                "watchlist_near_low",
                                f"👁 {wl_name} within {pct_above_low:.1f}% of 52-week low",
                                {"current_price": round(cur_price, 2), "week52_low": round(w52_low, 2),
                                 "pct_above_low": round(pct_above_low, 1), "source": "watchlist"},
                                "watchlist_near_low"
                            ))
                            wl_alert_count += 1
                except Exception:
                    pass

            print(f"Watchlist monitoring: {len(wl_items)} items, {wl_alert_count} alerts generated.")

    # ══════════════════════════════════════
    # 3e. SCORE HISTORY TRACKING
    # ══════════════════════════════════════
    if universe_df is not None:
        all_held = set()
        for port in portfolios:
            port_holdings = [h for h in holdings if h["portfolio_id"] == port["id"]]
            for h in port_holdings:
                all_held.add(h["ticker"])

        # Also track watched tickers so score_history covers them even if score drops below 3
        all_watched = set()
        try:
            _wl_all = supabase.table("watchlist").select("ticker").execute()
            all_watched = {w["ticker"] for w in (_wl_all.data or [])}
        except Exception:
            pass

        trackable = universe_df[
            (universe_df["ticker"].isin(all_held | all_watched)) |
            (universe_df["score"] >= 3)
        ].copy()

        score_rows = []
        for _, row in trackable.iterrows():
            score_rows.append({
                "ticker": row["ticker"],
                "date": today_str,
                "score": int(row["score"]) if pd.notna(row.get("score")) else 0,
                "graham_pass": bool(row["graham_pass"]) if pd.notna(row.get("graham_pass")) else None,
                "greenblatt_pass": bool(row["greenblatt_pass"]) if pd.notna(row.get("greenblatt_pass")) else None,
                "dorsey_pass": bool(row["dorsey_pass"]) if pd.notna(row.get("dorsey_pass")) else None,
                "trajectory_pass": bool(row["trajectory_pass"]) if pd.notna(row.get("trajectory_pass")) else None,
                "pe": round(float(row["pe"]), 2) if pd.notna(row.get("pe")) else None,
                "roe_pct": round(float(row["roe_pct"]), 2) if pd.notna(row.get("roe_pct")) else None,
                "quality_pass": bool(row["quality_pass"]) if pd.notna(row.get("quality_pass")) else None,
            })

        written_scores = 0
        for i in range(0, len(score_rows), 100):
            batch = score_rows[i:i+100]
            try:
                supabase.table("score_history").upsert(
                    _json_safe(batch), on_conflict="ticker,date"
                ).execute()
                written_scores += len(batch)
            except Exception as e:
                print(f"Score history batch failed: {e}")

        print(f"Logged {written_scores} score history rows.")
        # ── New entry detection: score ≥ 3 stocks appearing for the first time ──
        try:
            latest_hist = supabase.table("score_history").select("date").order("date", desc=True).limit(1).execute()
            if latest_hist.data:
                prev_date = latest_hist.data[0]["date"]
                if prev_date != today_str:
                    prev_resp = supabase.table("score_history").select("ticker").eq("date", prev_date).execute()
                    prev_tickers = set(row["ticker"] for row in prev_resp.data)

                    new_high_scorers = universe_df[
                        (universe_df["score"] >= 3) &
                        (universe_df["quality_pass"] == True) &
                        (~universe_df["ticker"].isin(prev_tickers))
                    ]

                    for _, nr in new_high_scorers.iterrows():
                        all_alerts.append({
                            "portfolio_id": None,
                            "user_id": None,  # broadcast — weekly_mentor sends to all users
                            "alert_type": "new_entry",
                            "ticker": nr["ticker"],
                            "headline": f"{nr.get('name', nr['ticker'])} new to radar at score {int(nr['score'])}/5",
                            "detail": {
                                "name": str(nr.get("name", nr["ticker"])),
                                "score": int(nr["score"]),
                                "sector": str(nr.get("sector", "N/A")),
                                "pe": round(float(nr["pe"]), 2) if pd.notna(nr.get("pe")) else None,
                                "book_passages": [],
                            },
                            "alert_date": today_str,
                        })

                    if len(new_high_scorers) > 0:
                        print(f"Detected {len(new_high_scorers)} new high-scoring entries.")
        except Exception as e:
            print(f"New entry detection failed: {e}")
            
 
    # ══════════════════════════════════════
    # 4. WRITE ALERTS TO SUPABASE
    # ══════════════════════════════════════
    try:
        cutoff = (date.today() - timedelta(days=7)).isoformat()
        supabase.table("portfolio_alerts").delete().lt("alert_date", cutoff).eq("is_read", False).execute()
    except Exception as e:
        print(f"Warning: Could not clean old alerts: {e}")

    written = 0
    for alert in all_alerts:
        try:
            # Opportunity/new_entry collected silently — weekly_mentor activates the best ones Monday
            if alert.get("alert_type") in ("opportunity", "new_entry"):
                alert["is_read"] = True
            supabase.table("portfolio_alerts").upsert(
                _json_safe(alert), on_conflict="portfolio_id,ticker,alert_type,alert_date"
            ).execute()
            written += 1
        except Exception as e:
            print(f"Alert write failed: {e}")

    print(f"Wrote {written} alerts.")


    # ══════════════════════════════════════
    # 4a. PAPER PORTFOLIO AUTO-SIP (1st of month)
    # ══════════════════════════════════════
    if date.today().day == 1:
        _paper_ports = [p for p in portfolios if p.get("is_paper") and (p.get("sip_amount") or 0) > 0]
        if _paper_ports:
            print(f"\n── Paper Auto-SIP: {len(_paper_ports)} paper portfolio(s) with active SIP ──")
            for pp in _paper_ports:
                pp_id = pp["id"]
                pp_user = pp["user_id"]
                pp_sip = float(pp["sip_amount"])
                pp_holdings = [h for h in holdings if h["portfolio_id"] == pp_id]
                if not pp_holdings:
                    print(f"  [{pp.get('name')}] No holdings — skipping auto-SIP")
                    continue
                # Guard: check if auto-SIP already ran this month
                _month_start = date.today().replace(day=1).isoformat()
                try:
                    _dup_check = supabase.table("sip_transactions").select("id").eq(
                        "portfolio_id", pp_id
                    ).eq("transaction_type", "paper_sip").gte(
                        "transaction_date", _month_start
                    ).limit(1).execute()
                    if _dup_check.data:
                        print(f"  [{pp.get('name')}] Auto-SIP already ran this month — skipping")
                        continue
                except Exception:
                    pass  # Proceed if check fails

                # Compute allocation per holding
                _total_alloc = sum(float(h.get("allocation_pct") or 0) for h in pp_holdings)
                _sip_spent = 0
                _sip_log = []
                for h in pp_holdings:
                    _alloc_pct = float(h.get("allocation_pct") or 0)
                    if _total_alloc > 0:
                        _h_budget = pp_sip * (_alloc_pct / _total_alloc)
                    else:
                        _h_budget = pp_sip / len(pp_holdings)
                    _ticker = h["ticker"]
                    _cur_price = price_cache.get(_ticker)
                    if not _cur_price or _cur_price <= 0:
                        continue
                    _new_shares = int(_h_budget / _cur_price)
                    if _new_shares <= 0:
                        continue
                    _new_amt = round(_new_shares * _cur_price, 2)
                    _nifty_u = round(_new_amt / nifty_bees_price, 6) if nifty_bees_price and nifty_bees_price > 0 else None
                    # Record transaction
                    try:
                        supabase.table("sip_transactions").insert(_json_safe({
                            "portfolio_id": pp_id,
                            "user_id": pp_user,
                            "ticker": _ticker,
                            "shares": _new_shares,
                            "price": round(_cur_price, 2),
                            "amount_inr": _new_amt,
                            "transaction_type": "paper_sip",
                            "transaction_date": today_str,
                            "nifty_price": nifty_bees_price,
                            "nifty_units": _nifty_u,
                        })).execute()
                    except Exception as e:
                        print(f"  Paper SIP txn failed for {_ticker}: {e}")
                        continue
                    # Update holding
                    _old_shares = float(h.get("shares") or 0)
                    _old_invested = float(h.get("sip_amount_inr") or 0)
                    try:
                        supabase.table("holdings").update(_json_safe({
                            "shares": _old_shares + _new_shares,
                            "sip_amount_inr": round(_old_invested + _new_amt, 2),
                        })).eq("id", h["id"]).execute()
                    except Exception as e:
                        print(f"  Holding update failed for {_ticker}: {e}")
                    _sip_spent += _new_amt
                    _sip_log.append(f"{_ticker}×{_new_shares}")

                if _sip_log:
                    print(f"  [{pp.get('name')}] Auto-SIP ₹{_sip_spent:,.0f}: {', '.join(_sip_log)}")
                else:
                    print(f"  [{pp.get('name')}] No shares affordable at current prices")
        else:
            print("\nPaper Auto-SIP: No paper portfolios with active SIP.")

    # ══════════════════════════════════════
    # 5. TELEGRAM DAILY DIGEST
    # ══════════════════════════════════════
    if _tg_token and _tg_map:
        _user_ports = {}
        for port in portfolios:
            if not port.get("is_paper"):
                _user_ports.setdefault(port["user_id"], []).append(port)
 
        _tg_sent = 0
        for uid, chat_id in _tg_map.items():
            ports = _user_ports.get(uid, [])
            if not ports:
                continue
 
            lines = ["<b>📊 Kordent Daily Update</b>\n"]
            for p in ports:
                val, ret, p_xirr, n_xirr = _port_values.get(p["id"], (0, 0, None, None))
                _xirr_line = ""
                if p_xirr is not None:
                    _xirr_line = f"\nXIRR: {p_xirr:+.1f}%"
                    if n_xirr is not None:
                        _alpha = round(p_xirr - n_xirr, 1)
                        _xirr_line += f" (Alpha: {_alpha:+.1f}%)"
                lines.append(f"<b>{_html_esc(p.get('name', 'Portfolio'))}</b>\nRs. {val:,.0f} ({ret:+.1f}%){_xirr_line}\n")
 
            # SIP reminder on 1st of month
            if date.today().day == 1:
                for p in ports:
                    sip = p.get("sip_amount", 0)
                    if sip > 0:
                        lines.append(f"📅 SIP due: Rs. {sip:,.0f} for <b>{_html_esc(p['name'])}</b>")
 
            lines.append(f"\n<a href='https://kordent.streamlit.app'>Open Kordent</a>")
            send_telegram(chat_id, "\n".join(lines), _tg_token)
            _tg_sent += 1
 
        if _tg_sent:
            print(f"Sent {_tg_sent} Telegram daily digests.")
 
    print("Kordent Daily Audit Complete.")

if __name__ == "__main__":
    run_daily_tracker()
