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
import economics
# Sprint 16: the READ side of the macro series. Stdlib-only, no network — see
# the risk-free rate block below for why this is a hard import.
import macro_read
# Pure: no Streamlit, no network, no LLM. That purity is what makes the W1
# drift classification cheap enough to run inside the daily alert loop.
import selector


# ══════════════════════════════════════════════
# RISK-FREE RATE — one rate, one place, LIVE since Sprint 16
# ══════════════════════════════════════════════
# India 10Y G-Sec yield, as a decimal. Feeds Sharpe/Treynor/Jensen/Sortino/CAPM.
#
# WHAT CHANGED AND WHY THE OLD REASONING NO LONGER HOLDS. This was a hardcoded
# constant on two grounds recorded in 2026-07: that RFR is the smallest term in
# every ratio it enters, and that no free, unauthenticated, stable India-10Y
# feed existed. The first is still true and is why nothing here is urgent. The
# second stopped being true on 2026-07-27, when macro_fetch.py began writing a
# dated India-10Y series every weekday — and then nothing read it for seven
# weeks. This is not an unauthenticated third-party feed with a daily failure
# mode; it is a local JSON file we already own, medianed over five readings,
# freshness-checked and band-checked by macro_read, with a documented fallback.
#
# The original bug this replaced was DUPLICATION, not staleness: the rate was
# hardcoded here AND in the Graham scorer, invisible in both. That fix stands.
# deep_metrics still does NOT import this — see the block at deep_metrics.py:22.
# A live RFR in a daily-recomputed portfolio statistic is correct; a live rate
# in a frozen scoring constant would re-score the universe on G-Sec noise and
# break archive comparability. Different objects, different rules.
#
# THE FALLBACK LIVES IN macro_read, NOT HERE. A second 0.07 in this file would
# recreate exactly the bug the original note described: one rate hardcoded in
# two places, invisible in both, free to drift apart. macro_read owns the
# number, the plausibility band and the freshness rule; this module owns nothing
# but the question.
#
# macro_read is imported HARD at the top of this file, like archetype in
# deep_metrics. It is a stdlib-only module sitting in this repo, so its absence
# is a broken deployment rather than a runtime condition to degrade around —
# and degrading around it would mean inventing a rate here, which is the
# duplication all over again. Fail at import, not at portfolio three.
_RFR_CACHE = None


def get_india_rfr_status():
    """(rate_as_decimal, status). Cached per process — the tracker loops over
    portfolios and must not re-read the series for each one.

    status is macro_read's: "ok" for a live operative value, or UNAVAILABLE /
    INSUFFICIENT / STALE / OUT_OF_BAND alongside macro_read's fallback rate.
    Callers that display or store the rate MUST carry the status with it:
    "7.0% because the G-Sec is 7.0%" and "7.0% because the macro job has been
    down for a week" are not the same number and must not render alike.
    """
    global _RFR_CACHE
    if _RFR_CACHE is None:
        _RFR_CACHE = macro_read.india_rfr()
    return _RFR_CACHE


def get_india_rfr():
    """India 10Y G-Sec yield as a decimal (0.07 == 7%). Single source of truth.

    Returns the number only. Anything that SHOWS the rate should call
    get_india_rfr_status() instead and carry the status with it.
    """
    return get_india_rfr_status()[0]


# ══════════════════════════════════════════════
# HONEST RANGES — a ratio from few days is an INTERVAL, not a point
# ══════════════════════════════════════════════
# Two kinds of range, because two kinds of uncertainty:
#   - series ratios (Sharpe, Sortino, IR, semi-dev): bootstrap the daily returns,
#     then FLOOR the width by 1/sqrt(n) so a calm short sample cannot fake
#     precision. Narrows continuously as history accrues — no step function.
#   - beta ratios (Treynor, Jensen, CAPM): beta is ESTIMATED in-house (Sprint
#     17) by regressing the portfolio's daily returns on the market, and its
#     Newey-West 95% band is propagated through each formula. The band narrows
#     as history accrues. It replaced a fixed ±0.15 on a yfinance beta of
#     unknown window and reference index.
MIN_DAYS_SERIES = 20          # below this a series ratio is withheld entirely
NW_LAGS = 5                   # Newey-West lags for the beta band
_TD = 252


def _series_ratio_range(daily_returns, ratio_fn, n_boot=400, block=5,
                        floor_rel=0.9):
    """10th-90th block-bootstrap band for a series ratio, width-floored by
    1/sqrt(n). Returns (low, high, n_days) or None if history too short."""
    import numpy as np
    r = np.asarray(daily_returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < MIN_DAYS_SERIES:
        return None
    point = ratio_fn(r)
    if not (isinstance(point, float) and math.isfinite(point)):
        try:
            point = float(point)
            if not math.isfinite(point):
                return None
        except Exception:
            return None
    rng = np.random.default_rng(12345)   # fixed seed: stable day-to-day numbers
    n_blocks = int(np.ceil(n / block))
    boot = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        boot[b] = ratio_fn(r[idx[:n]])
    boot = boot[np.isfinite(boot)]
    if len(boot) < 20:
        return None
    lo, hi = np.percentile(boot, [10, 90])
    scale = math.sqrt(MIN_DAYS_SERIES / n)
    min_half = floor_rel * max(abs(point), 0.25) * scale
    mid = 0.5 * (lo + hi)
    half = max(0.5 * (hi - lo), min_half)
    return (round(float(mid - half), 3), round(float(mid + half), 3), int(n))


def _regression_beta(port_returns, market_returns, lags=NW_LAGS):
    """In-house beta: OLS slope of portfolio on market daily returns, with a
    Newey-West 95% band. Returns (beta, low, high, n_days) or None below
    MIN_DAYS_SERIES aligned days.

    Newey-West, not plain OLS errors: daily portfolio returns are mildly
    autocorrelated, and plain errors would print a band narrower than the data
    supports — the same false precision the old fixed ±0.15 was meant to avoid.
    """
    import numpy as np
    j = pd.concat([port_returns.rename("p"), market_returns.rename("m")],
                  axis=1, join="inner").dropna()
    n = len(j)
    if n < MIN_DAYS_SERIES:
        return None
    X = np.column_stack([np.ones(n), j["m"].to_numpy()])
    y = j["p"].to_numpy()
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    S = (X * e[:, None]).T @ (X * e[:, None])
    for l in range(1, min(lags, n - 1) + 1):
        G = (X[l:] * e[l:, None]).T @ (X[:-l] * e[:-l, None])
        S += (1 - l / (lags + 1)) * (G + G.T)
    se = float(np.sqrt(max((XtX_inv @ S @ XtX_inv)[1, 1], 0.0)))
    beta = float(b[1])
    if not math.isfinite(beta):
        return None
    return beta, beta - 1.96 * se, beta + 1.96 * se, n


def _beta_ratio_range(point_fn, beta_low, beta_high, divides_by_beta=False):
    """Range for a beta-dependent ratio: evaluate the formula at both ends of
    the ESTIMATED beta band. Returns (low, high) or None.

    divides_by_beta: for Treynor, a band that contains zero makes the ratio
    unbounded, and a min/max of the two ends would print a finite range that
    is false. Return None then."""
    if beta_low is None or beta_high is None:
        return None
    if not (math.isfinite(beta_low) and math.isfinite(beta_high)):
        return None
    if divides_by_beta and beta_low <= 0 <= beta_high:
        return None
    vals = []
    for b in (beta_low, beta_high):
        try:
            v = point_fn(b)
        except Exception:
            v = None
        if v is not None and math.isfinite(v):
            vals.append(v)
    if len(vals) < 2:
        return None
    return (round(min(vals), 4), round(max(vals), 4))


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

# Alert severity vocabulary. MODULE level, deliberately: make_alert lives inside
# `for port in portfolios:` while wl_alert lives after that loop ends. Both are
# locals of run_daily_tracker, so a constant defined in the loop body is visible
# afterwards ONLY IF the loop ran at least once. A user with no portfolios but a
# non-empty watchlist would hit NameError on the first watchlist alert. A
# constant has no business being a loop-body local.
_SEVERITIES = ("danger", "warning", "info")

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

def compute_xirr_standalone(econ, nifty_shadow_value=None):
    """XIRR from EXTERNAL cash flows only (economics model (a)).

    Buys and sells are internal transfers between cash and securities, not flows
    into or out of the investor's pocket. Signing every buy negative and every
    sell positive — the old behaviour — turned a same-day sell-and-rebuy into a
    spurious round trip and, once sells started reaching the ledger, would have
    reported a return on gross turnover. The only real flows are contributions;
    the terminal value is total_assets (holdings + uninvested cash).

    econ: output of economics.portfolio_economics(). No DB access.
    Returns (port_xirr_pct, nifty_xirr_pct), or (None, None).
    """
    try:
        from pyxirr import xirr
    except ImportError:
        return None, None
    dates, amounts = economics.xirr_flows(econ)
    if not dates or len(dates) < 2:
        return None, None
    # XIRR meaningless under 90 days
    if (date.today() - dates[0]).days < 90:
        return None, None
    try:
        port_xirr = xirr(dates, amounts)
    except Exception:
        port_xirr = None
    port_xirr_pct = round(port_xirr * 100, 2) if port_xirr is not None else None
    nifty_xirr_pct = None
    if nifty_shadow_value and nifty_shadow_value > 0:
        nifty_amounts = amounts[:-1] + [float(nifty_shadow_value) + float(econ.get("cash_balance") or 0.0)]
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

    def _val(h):
        return h.get("current_value", 0) or h.get("sip_amount_inr", 0) or 0

    # allocate_shares deliberately keeps unfunded names in `holdings` with
    # shares=0 so their allocation_pct target survives to the next SIP cycle.
    # They are TARGETS, not POSITIONS. Counting them here inflated adequacy
    # (a 9-stock portfolio with 6 pending targets scored as 15/12 = 100) and
    # deflated the HHI score by padding n_sectors with zero-weight sectors.
    holdings = [h for h in holdings if _val(h) > 0]
    if not holdings:
        return 0

    total_value = sum(_val(h) for h in holdings)
    if total_value <= 0:
        return 0

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


# Every field compute_own_risk_metrics owns. Written on EVERY run — as None when
# a metric is withheld — so a value computed by the retired simulated method, or
# on an earlier day's longer history, can never linger in the row.
OWN_RISK_KEYS = [
    "portfolio_beta", "beta_low", "beta_high", "beta_days",
    "sharpe_ratio", "sharpe_low", "sharpe_high",
    "sortino_ratio", "sortino_low", "sortino_high",
    "treynor_ratio", "treynor_low", "treynor_high",
    "jensen_alpha", "information_ratio", "capm_expected_return",
    "max_drawdown", "max_drawdown_provisional", "semi_deviation",
    "annual_return", "annual_std", "coefficient_of_variation",
    "market_return", "tracking_error",
    "bench_annual_return", "bench_annual_std",
    "benchmark_stats_ticker", "benchmark_coverage", "benchmark_status",
    "metrics_history_days", "rfr_used", "rfr_status",
]
DRAWDOWN_PROVISIONAL_DAYS = 126    # ~6 months; the label app.py already renders


def compute_own_risk_metrics(history_rows, benchmark_ticker=None):
    """Risk and performance from the portfolio's OWN daily history (Sprint 17).

    Replaces compute_portfolio_risk_metrics, which priced the CURRENT holdings
    over the PRIOR year — hindsight: today's basket passed today's screens
    partly because of that year. Every figure here is this portfolio's record,
    time-weighted net of SIP flows (economics.daily_twr_returns), and each is
    withheld until the history reaches economics.RISK_MIN_DAYS for it.

    The "market" for beta, Jensen, Treynor and CAPM is the portfolio's own
    benchmark statistics series (selector.benchmark_stats_ticker). Measuring a
    midcap portfolio against the Nifty 50 booked a year of midcap-over-largecap
    as selection skill (+25% "alpha" on 2026-09-30) — the benchmark
    specification error the Sprint 17 plan names.
    """
    import numpy as np
    import yfinance as yf

    RFR, RFR_STATUS = get_india_rfr_status()
    out = {k: None for k in OWN_RISK_KEYS}
    out["rfr_used"], out["rfr_status"] = round(RFR, 6), RFR_STATUS

    twr = economics.daily_twr_returns(history_rows)
    n = len(twr)
    out["metrics_history_days"] = n
    if n == 0:
        return out
    r = pd.Series([x for _, x in twr],
                  index=pd.to_datetime([d for d, _ in twr])).sort_index()

    def ok(metric):
        return economics.risk_days_left(metric, n) == 0

    td = 252
    # ── Realised: worst fall so far (a fact, shown from day one) ──
    cum = (1 + r).cumprod()
    dd = float((cum / cum.cummax() - 1).min())
    if math.isfinite(dd):
        out["max_drawdown"] = round(min(dd, 0.0), 4)
        out["max_drawdown_provisional"] = n < DRAWDOWN_PROVISIONAL_DAYS

    # ── Benchmark over the SAME intervals ──
    # Priced at each history date, so a missing tracker day spans the same two
    # sessions on both sides.
    b = None
    stats_t = selector.benchmark_stats_ticker(benchmark_ticker) if benchmark_ticker else "^NSEI"
    out["benchmark_stats_ticker"] = stats_t
    try:
        dates = pd.to_datetime(sorted(str(x["date"]) for x in history_rows if x.get("date")))
        h = yf.download(stats_t, start=(dates.min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
                        end=(dates.max() + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
                        progress=False, auto_adjust=True, group_by="column")
        c = h["Close"] if not h.empty else pd.Series(dtype=float)
        if hasattr(c, "columns"):
            c = c.iloc[:, 0]
        c = c.dropna()
        c.index = pd.to_datetime(c.index).tz_localize(None).normalize()
        exact = dates.isin(c.index)
        cov = float(exact.mean()) if len(dates) else 0.0
        out["benchmark_coverage"] = round(cov, 3)
        if cov < selector.BENCHMARK_MIN_COVERAGE:
            out["benchmark_status"] = "INCOMPLETE"
            print(f"  Benchmark {stats_t} priced on {cov:.0%} of this portfolio's "
                  f"days — benchmark-relative figures withheld.")
        else:
            out["benchmark_status"] = "ok"
            px = c.reindex(c.index.union(dates)).ffill().reindex(dates)
            b = (px / px.shift(1) - 1).reindex(r.index)
    except Exception as e:
        out["benchmark_status"] = "INCOMPLETE"
        print(f"  Benchmark fetch failed (non-blocking): {type(e).__name__}: {e}")

    # ── SHAPE tier: how bumpy ──
    std = float(r.std() * np.sqrt(td))
    if ok("annual_std") and math.isfinite(std):
        out["annual_std"] = round(std, 4)
    below = r[r < r.mean()]
    if ok("semi_deviation") and len(below) > 1:
        out["semi_deviation"] = round(float(below.std() * np.sqrt(td)), 4)
    if b is not None and ok("portfolio_beta"):
        fit = _regression_beta(r, b)
        if fit:
            out["portfolio_beta"] = round(fit[0], 3)
            out["beta_low"], out["beta_high"] = round(fit[1], 3), round(fit[2], 3)
            out["beta_days"] = fit[3]
    if b is not None and ok("tracking_error"):
        te = float((r - b).dropna().std() * np.sqrt(td))
        if math.isfinite(te) and te > 0:
            out["tracking_error"] = round(te, 4)

    # ── RETURN tier: was it worth it ──
    if ok("annual_return"):
        ann = float(r.mean() * td)
        out["annual_return"] = round(ann, 4)
        if std > 0:
            out["sharpe_ratio"] = round((ann - RFR) / std, 3)
        rng_ = _series_ratio_range(
            r.to_numpy(), lambda x: ((x.mean() * td - RFR) / (x.std() * np.sqrt(td))
                                     if x.std() > 0 else float("nan")))
        if rng_:
            out["sharpe_low"], out["sharpe_high"] = rng_[0], rng_[1]
        d_rfr = RFR / td
        dn = r[r < d_rfr]
        if len(dn):
            ddev = float(((dn - d_rfr) ** 2).mean() ** 0.5 * np.sqrt(td))
            if ddev > 0:
                out["sortino_ratio"] = round((ann - RFR) / ddev, 3)

        def _sortino(x):
            d = x[x < d_rfr]
            if not len(d):
                return float("nan")
            v = ((d - d_rfr) ** 2).mean() ** 0.5 * np.sqrt(td)
            return (x.mean() * td - RFR) / v if v > 0 else float("nan")
        rng_ = _series_ratio_range(r.to_numpy(), _sortino)
        if rng_:
            out["sortino_low"], out["sortino_high"] = rng_[0], rng_[1]
        # CV = std / mean explodes as the mean nears zero (a synthetic check
        # returned 1,940 on a ~0% year). Shown only while the annualised return
        # is at least two standard errors above zero, i.e. reliably positive.
        _se = std / np.sqrt(n / td) if n else float("inf")
        if std > 0 and ann > 2 * _se:
            out["coefficient_of_variation"] = round(std / ann, 3)

        if b is not None:
            bm = float(b.dropna().mean() * td)
            bs = float(b.dropna().std() * np.sqrt(td))
            out["market_return"] = round(bm, 4)
            out["bench_annual_return"] = round(bm, 4)
            out["bench_annual_std"] = round(bs, 4)
            if out["tracking_error"]:
                out["information_ratio"] = round((ann - bm) / out["tracking_error"], 3)
            beta = out["portfolio_beta"]
            if beta is not None:
                out["jensen_alpha"] = round(ann - RFR - beta * (bm - RFR), 4)
                out["capm_expected_return"] = round(RFR + beta * (bm - RFR), 4)
                if beta != 0:
                    out["treynor_ratio"] = round((ann - RFR) / beta, 4)
                    tr = _beta_ratio_range(lambda bb: (ann - RFR) / bb if bb else None,
                                           out["beta_low"], out["beta_high"],
                                           divides_by_beta=True)
                    if tr:
                        out["treynor_low"], out["treynor_high"] = tr
    return out
 
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

    # ── Fetch a current close for EACH benchmark ETF in use ──
    # Portfolios no longer share one benchmark: each was locked at registration
    # to the ETF its IPS mandate implied (Nifty 50 / Midcap 150 / Smallcap 250).
    # Price every portfolio's shadow against the SAME ETF its nifty_units were
    # bought in. The NaN discipline is unchanged, only applied per ticker: a
    # missing close means that benchmark's shadow is NULL this run, never a guess.
    _port_by_id = {p["id"]: p for p in portfolios}
    bench_px = {}
    for _bt in {(p.get("benchmark_ticker") or "NIFTYBEES.NS") for p in portfolios}:
        try:
            _px = _usable_close(yf.Ticker(_bt).history(period="5d"), _bt)
        except Exception as e:
            _px = None
            print(f"Warning: Could not fetch benchmark {_bt}: {e}")
        bench_px[_bt] = _px
        if _px is None:
            print(f"Warning: no usable {_bt} close. Shadow metrics are NULL this "
                  f"run for portfolios on {_bt}; the ledger will NOT be written "
                  f"with a guessed price.")
        else:
            print(f"{_bt} close: {_px:,.2f}")

    def _bench_price_for(port):
        """Current close of a portfolio's assigned benchmark ETF, or None."""
        return bench_px.get((port or {}).get("benchmark_ticker") or "NIFTYBEES.NS")

    # ── Fetch all transactions once ──
    txn_resp = supabase.table("sip_transactions").select("id, created_at, portfolio_id, ticker, shares, price, amount_inr, transaction_type, transaction_date, nifty_units, nifty_price").execute()
    all_txns = txn_resp.data or []

    # ── Bootstrap: create genesis transactions for portfolios with holdings but no transactions ──
    _txn_port_ids = set(t["portfolio_id"] for t in all_txns)
    holdings_resp_all = supabase.table("holdings").select("portfolio_id, ticker, shares, price_at_entry, entry_date").execute()
    _all_h = holdings_resp_all.data or []
    _ports_needing_bootstrap = set()
    for h in _all_h:
        if h["portfolio_id"] not in _txn_port_ids:
            _ports_needing_bootstrap.add(h["portfolio_id"])
    # nifty_units is a LEDGER quantity, not a metric. Metrics are recomputed daily;
    # a NULL costs one day. Ledger rows are summed forever (see L564:
    # `float(t.get("nifty_units") or 0)`), so a NULL genesis row silently and
    # permanently understates the Nifty shadow. Skip and retry tomorrow.
    # Only bootstrap portfolios whose OWN benchmark has a usable close today; a
    # genesis row with NULL nifty_units would permanently understate that
    # portfolio's shadow. The rest retry next run.
    if _ports_needing_bootstrap:
        _skip_boot = {pid for pid in _ports_needing_bootstrap
                      if not _bench_price_for(_port_by_id.get(pid))}
        if _skip_boot:
            print(f"Skipping bootstrap of {len(_skip_boot)} portfolios: no usable "
                  f"benchmark close today. Will retry next run.")
        _ports_needing_bootstrap = _ports_needing_bootstrap - _skip_boot

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
                # Price the genesis row off THIS portfolio's benchmark ETF, not a
                # shared NIFTYBEES. `_bp and _bp > 0` also guards NaN — bool(nan)
                # is True, so `_bp and ...` alone once wrote nifty_units=NaN into
                # the permanent ledger; the `> 0` is what actually rejects NaN.
                _pf = _port_by_id.get(h["portfolio_id"])
                _bp = _bench_price_for(_pf)
                _bt = (_pf or {}).get("benchmark_ticker") or "NIFTYBEES.NS"
                _nifty_u = (round(_h_amt / _bp, 6)
                            if (_bp and _bp > 0 and _h_amt > 0)
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
                        "nifty_price": _bp,
                        "nifty_units": _nifty_u,
                        "benchmark_ticker": _bt,
                    })).execute()
        # Refresh transactions after bootstrap
        txn_resp = supabase.table("sip_transactions").select("id, created_at, portfolio_id, ticker, shares, price, amount_inr, transaction_type, transaction_date, nifty_units, nifty_price").execute()
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

        # ── Portfolio economics: ONE computation, model (a). ──
        # total_invested above is the cost basis of SURVIVING holdings and is no
        # longer a return denominator: it shrinks when a position is sold, which
        # deletes the loss along with the cost and makes realising a loser look
        # like a gain. External capital never shrinks on a sale.
        port_txns = [t for t in all_txns if t["portfolio_id"] == port_id]
        _bp = _bench_price_for(port)
        econ = economics.portfolio_economics(port_txns, current_total_value, _bp)
        return_pct = econ["return_pct"] if econ["return_pct"] is not None else 0.0

        # ── 1. Update leaderboard snapshot ──
        supabase.table("portfolios").update(_json_safe({
            "current_value": round(econ["total_assets"], 2),
            "current_return_pct": econ["return_pct"],
            "cash_balance": econ["cash_balance"],
            "realized_pnl": econ["realized_pnl"],
            "withdrawn": econ["withdrawn"],
        })).eq("id", port_id).execute()

        # ── 2. Cumulative invested (= external capital) & benchmark shadow ──
        # The shadow is now derived from EXTERNAL flows inside replay_ledger,
        # priced at each row's own nifty_price. Summing the stored nifty_units
        # column (+amt on buys, -amt on everything else) made a sale look like a
        # withdrawal: sell, hold the cash, and the shadow fell even though no
        # money had left. With withdrawals in the ledger it double-counted too.
        cumulative_invested = econ["external_capital"]
        nifty_shadow = econ["shadow_value"]
        if nifty_shadow is not None and not math.isfinite(nifty_shadow):
            nifty_shadow = None
        if econ["shadow_incomplete"]:
            print(f"  WARNING [{port['name']}]: a contribution had no benchmark "
                  f"price on its ledger row; the shadow understates.")
        if econ["unreconciled_withdrawal"] > 0:
            print(f"  WARNING [{port['name']}]: withdrawals exceed known cash by "
                  f"{econ['unreconciled_withdrawal']:,.2f} - a sale or contribution "
                  f"is missing from the ledger.")

        # ── 3. Log history ──
        history_row = {
            "portfolio_id": port_id,
            "date": today_str,
            # total_value stays "market value of holdings" so the risk series is
            # unchanged; cash_balance is stored alongside it and the chart plots
            # total_value + cash_balance. Without this the equity line drops on
            # every sale date and never recovers, which is a fictitious loss.
            "total_value": round(current_total_value, 2),
            "cash_balance": econ["cash_balance"],
            "withdrawn": econ["withdrawn"],
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
        _p_xirr, _n_xirr = compute_xirr_standalone(econ, nifty_shadow)
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
        _cash_str = f" | Cash {econ['cash_balance']:,.2f}" if econ["cash_balance"] > 0 else ""
        _cash_str += f" | Withdrawn {econ['withdrawn']:,.2f}" if econ["withdrawn"] > 0 else ""
        print(f"Updated [{port['name']}]: Assets {econ['total_assets']:,.2f}{_cash_str} "
              f"| Ext capital {econ['external_capital']:,.2f} | Return {return_pct:+.2f}%"
              f"{_xirr_str} | Div {_div_label}{_div_score}")
        # Sprint 11: Portfolio risk & performance metrics (Reilly & Brown Ch 7, 18)
        try:
            # Sprint 17: the portfolio's OWN history (net of SIP flows), not the
            # current holdings priced over the prior year. Today's row was
            # upserted above, so it is included.
            _hist = supabase.table("portfolio_history").select(
                "date, total_value, cash_balance, cumulative_invested, withdrawn"
            ).eq("portfolio_id", port_id).order("date").execute().data or []
            _risk = compute_own_risk_metrics(
                _hist, benchmark_ticker=port.get("benchmark_ticker"))
            if _risk:
                # EVERY owned key, None included: a withheld metric CLEARS its
                # column instead of leaving yesterday's — or the retired
                # simulated method's — number standing.
                _risk_update = {k: _risk.get(k) for k in OWN_RISK_KEYS}
                if _risk_update:
                    supabase.table("portfolios").update(_json_safe(_risk_update)).eq("id", port_id).execute()
                _rfr_str = (f"  RFR: {_risk.get('rfr_used', 0)*100:.3f}% "
                            f"[{_risk.get('rfr_status', '?')}]")
                print(_rfr_str)
                _beta_str = f" | β={_risk.get('portfolio_beta', '?')}"
                _sharpe_str = f" | Sharpe={_risk.get('sharpe_ratio', '?')}"
                _alpha_str = f" | α={_risk.get('jensen_alpha', '?')}"
                _hd = _risk.get("metrics_history_days") or 0
                print(f"  Risk metrics ({_hd} days of own history; ratios need "
                      f"{economics.RISK_SHAPE_MIN_DAYS}/{economics.RISK_RETURN_MIN_DAYS}):"
                      f"{_beta_str}{_sharpe_str}{_alpha_str}")
        except Exception as e:
            print(f"  Risk metrics failed (non-blocking): {e}")
        _port_values[port_id] = (round(econ["total_assets"], 2), round(return_pct, 2), _p_xirr, _n_xirr)

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

        def make_alert(alert_type, ticker, headline, detail,
                       book_query_key=None, *, severity):
            """Build an alert dict with book passage attached.

            alert_type is WHAT HAPPENED (score_drop, quality_fail, goal_drift).
            severity is HOW BAD (danger/warning/info). These were one column
            until now, which cost us twice:

              1. The unique key is (portfolio_id, ticker, alert_type,
                 alert_date) and the write is an UPSERT on it. When score_drop,
                 quality_fail and price_crash all wrote alert_type='danger', a
                 ticker triggering two of them on one day had the second
                 silently overwrite the first — no exception, and the `written`
                 counter incremented for both. Separating the columns makes the
                 key meaningful and the collision disappears.
              2. It forced nonsense like `severity = 'danger' if ... else
                 'goal_drift'`, a ternary choosing between a severity and a type.

            severity is KEYWORD-ONLY and has NO DEFAULT deliberately. Inserting
            it positionally would let an un-updated call site pass `ticker` as
            severity and write a wrong row that still satisfies the CHECK. This
            way a missed site raises TypeError on the first run instead.
            """
            if severity not in _SEVERITIES:
                # The DB CHECK on severity lands in step 5. Until then this is
                # the only thing standing between a typo and a bad row.
                raise ValueError(
                    f"severity must be one of {_SEVERITIES}, got {severity!r} "
                    f"(alert_type={alert_type!r}, ticker={ticker!r})")
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
                "severity": severity,
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
                        "review_due", severity="info"
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
            # If the universe carries duplicate rows for a ticker (an earlier
            # partial run left a zero-scored HBLENGINE that a later concat didn't
            # overwrite; an NSE/BSE key collision), iloc[0] can grab the STALE
            # copy. Prefer the best-scored, non-stale, freshest row so we never
            # read a phantom 0 for a name that actually scored 5.
            if len(row) > 1:
                _r = row.copy()
                if "is_stale" in _r.columns:
                    _r = _r.sort_values("is_stale")            # False (fresh) first
                _r = _r.sort_values("score", ascending=False, kind="stable")
                row = _r
                print(f"  [DEDUP] {ticker}: {len(row)} universe rows; using score="
                      f"{row['score'].iloc[0]}")

            _raw_score = row["score"].iloc[0]
            # A missing/blank score is NOT a zero — it's "no reading today". Skip
            # the drop check entirely rather than manufacture a 5 -> 0 sell signal
            # on a data gap.
            if pd.isna(_raw_score):
                continue
            current_score = int(_raw_score)
            _is_stale = bool(row["is_stale"].iloc[0]) if "is_stale" in row.columns and pd.notna(row["is_stale"].iloc[0]) else False
            quality_pass = bool(row["quality_pass"].iloc[0]) if "quality_pass" in row.columns and pd.notna(row["quality_pass"].iloc[0]) else True

            # Never fire a sell/defend alert off a carried-forward (stale) row:
            # a throttle-day refill is for continuity, not for triggering trades.
            _cmp = selector.comparable_score_drop(holding.get("entry_trace"), row.iloc[0], entry_score, current_score)
            if _cmp["fires"] and not _is_stale:
                # W1: WHY it fell, not just that it fell. A drop we can prove
                # was price-only is a different event from one we can prove was
                # the business. The classification is ADVISORY — it never gates
                # firing, only volume.
                try:
                    _drift = selector.classify_score_drop(
                        holding.get("entry_trace"), row.iloc[0])
                except Exception as _e:
                    # Classification must never be able to suppress or soften
                    # an alert. Any failure falls back to the loudest reading
                    # and says so in the log.
                    print(f"  [W1] drift classification failed for {ticker}: {_e}")
                    _drift = {"reason": "unknown_inputs", "severity": "danger",
                              "newly_failing": [], "per_framework": {}}
                _dd_headline = selector.score_drop_headline(holding.get('name', ticker), _cmp, entry_score, current_score)
                all_alerts.append(make_alert(
                    "score_drop", ticker, _dd_headline,
                    {"name": holding.get("name", ticker), "entry_score": entry_score,
                     "current_score": current_score, "reason": "score_drop",
                     "comparable_frameworks": _cmp["n_common"], "comparable_delta": _cmp["delta"],
                     "drift_reason": _drift["reason"],
                     "drift_newly_failing": _drift["newly_failing"],
                     "drift_per_framework": _drift["per_framework"]},
                    "score_drop", severity=_drift["severity"]
                ))
                if _tg_token and user_id in _tg_map:
                    send_telegram(_tg_map[user_id], f"⚠️ <b>{_html_esc(_dd_headline)}</b>\n\n<a href='https://kordent.streamlit.app'>Open Kordent</a>", _tg_token)

            if not quality_pass:
                _qf_headline = f"{holding.get('name', ticker)} flagged as potential value trap"
                all_alerts.append(make_alert(
                    "quality_fail", ticker, _qf_headline,
                    {"name": holding.get("name", ticker), "reason": "quality_fail"},
                    "quality_fail", severity="danger"
                ))
                if _tg_token and user_id in _tg_map:
                    send_telegram(_tg_map[user_id], f"⚠️ <b>{_html_esc(_qf_headline)}</b>\n\n<a href='https://kordent.streamlit.app'>Open Kordent</a>", _tg_token)

            if entry_price > 0:
                stock_return = ((live_price - entry_price) / entry_price) * 100
                if stock_return < -20:
                    _pc_headline = f"{holding.get('name', ticker)} down {stock_return:.0f}% from entry"
                    all_alerts.append(make_alert(
                        "price_crash", ticker, _pc_headline,
                        {"name": holding.get("name", ticker), "reason": "price_crash",
                         "entry_price": entry_price, "current_price": round(live_price, 2),
                         "return_pct": round(stock_return, 1)},
                        "price_crash", severity="danger"
                    ))
                    if _tg_token and user_id in _tg_map:
                        send_telegram(_tg_map[user_id], f"⚠️ <b>{_html_esc(_pc_headline)}</b>\n\n<a href='https://kordent.streamlit.app'>Open Kordent</a>", _tg_token)

        # `overvalued` removed. It fired on naked PE>18 / PB>1.8 — IPS-blind
        # thresholds that ignore sector, growth and the user's own mandate, and
        # it duplicated the portfolio review, which is the single place a
        # holding gets a hold/sell verdict. It had ALSO been unreachable: the
        # write loop discarded alert_type 'overvalued' before the DB, so this
        # block ran a book-passage search every day and threw the result away.

        # ── 3c. Opportunity alerts ──
        investor_type = port.get("investor_type", "balanced")

        opps = universe_df[
            selector.meets_score_mask(universe_df, 4) &
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
            # n_applicable, not the default 5: Greenblatt abstains on
            # financials/utilities and Lynch on an unclassifiable business, so
            # judging this row against five tests it never took would understate
            # the verdict — and this one goes out by email, unreviewed.
            _opp_n = len(selector._applicable_frameworks(opp_row))
            opp_verdict = verdict_engine.get_verdict_tier(
                opp_score, True, opp_pass_dict, n_applicable=_opp_n)
            opp_emoji = verdict_engine.VERDICT_EMOJI.get(opp_verdict, "")

            all_alerts.append(make_alert(
                "opportunity", opp_row["ticker"],
                # Headlines are STORED and rendered verbatim downstream, so a
                # hardcoded /5 here outlives every render-side fix.
                f"{opp_emoji} {opp_row.get('name', opp_row['ticker'])} — {opp_verdict} "
                f"({selector.score_label(opp_row, opp_score)}) — fits your {investor_type} profile",
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
                "opportunity", severity="info"
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
                        "sector_concentration", "_portfolio",
                        f"Portfolio {port['name']}: {sector} is {weight*100:.0f}% of holdings (>40%)",
                        {"reason": "sector_concentration", "sector": sector, "weight_pct": round(weight * 100)},
                        "review_due", severity="danger"
                    ))

            # Diversification score (HHI)
            weights = [count / total_h for count in sector_weights.values()]
            hhi = sum(w ** 2 for w in weights)
            div_score = round((1 - hhi) * 100)
            if div_score < 50:
                all_alerts.append(make_alert(
                    "low_diversification", "_portfolio",
                    f"Portfolio {port['name']}: diversification score is {div_score}/100 (critical)",
                    {"reason": "low_diversification", "score": div_score},
                    "review_due", severity="danger"
                ))

        # `sector_headwind` removed. It fired on a raw >10% monthly drop in a
        # sector index — market timing, not an IPS judgment, and nothing the
        # user could act on within their mandate. It had ALSO been unreachable:
        # the write loop discarded alert_type 'sector_headwind' before the DB,
        # so it made a yfinance call and a book-passage search every single day
        # and threw both results away.
        #
        # NOTE: `if held_sectors:` below is now vestigial — it gates only the
        # goal-drift check, which never reads held_sectors. Left in place
        # because removing it CHANGES BEHAVIOUR: goal drift would start firing
        # for portfolios with no sector data. Separate decision, not this one.
        if held_sectors:
            # ── Goal drift: actual rate vs the rate still needed (Sprint 17) ──
            # Both sides were wrong before, in opposite directions: ACTUAL
            # counted every SIP instalment as growth (portfolio 109: "+26,000%
            # a year" on a -4.8% record, so the alert could never fire), and
            # NEEDED ignored every future instalment (a Rs 1L portfolio with a
            # Rs 10k SIP over 10 years "needed" 38% a year; the truth is 12%).
            #   actual — the portfolio's own time-weighted rate, SIP flows
            #            removed (economics.daily_twr_returns), compounded.
            #   needed — the rate at which today's assets plus the remaining
            #            monthly SIPs reach the target on the target date
            #            (economics.required_annual_return).
            # Judged only once the history clears the same gate as every other
            # return figure (economics.RISK_RETURN_MIN_DAYS): a trajectory read
            # off six months is ±27 points of noise, and an alert built on noise
            # teaches the user to ignore alerts.
            target_amount = port.get("target_amount")
            target_date_str = port.get("target_date")
            if target_amount and target_date_str:
                try:
                    target_dt = date.fromisoformat(str(target_date_str))
                    months_remaining = max(1, round((target_dt - date.today()).days / 30.44))

                    hist_rows = supabase.table("portfolio_history").select(
                        "date, total_value, cash_balance, cumulative_invested, withdrawn"
                    ).eq("portfolio_id", port_id).order("date").execute().data or []
                    _twr = economics.daily_twr_returns(hist_rows)

                    if economics.risk_days_left("annual_return", len(_twr)) == 0:
                        actual_rate = economics.twr_annualised(_twr)
                        sip_monthly = float(port.get("sip_amount") or 0)
                        needed_rate, reachable = economics.required_annual_return(
                            float(econ["total_assets"]), sip_monthly,
                            months_remaining, float(target_amount))

                        behind = (needed_rate is not None and needed_rate > 0
                                  and actual_rate is not None
                                  and (not reachable or actual_rate < 0.8 * needed_rate))
                        if behind:
                            # Type constant, severity varies (see make_alert).
                            _sev = ("danger" if (not reachable or actual_rate < 0.5 * needed_rate)
                                    else "warning")
                            _needed_txt = (f"more than {needed_rate*100:.0f}%" if not reachable
                                           else f"{needed_rate*100:.1f}%")
                            all_alerts.append(make_alert(
                                "goal_drift", "_portfolio",
                                f"{port['name']} trailing behind goal — earning "
                                f"{actual_rate*100:.1f}% a year vs {_needed_txt} needed "
                                f"from here (with Rs {sip_monthly:,.0f}/month SIP)",
                                # Field names kept: app.py renders actual_cagr_pct
                                # and needed_cagr_pct. Only their meaning is fixed.
                                {"reason": "goal_drift",
                                 "actual_cagr_pct": round(actual_rate * 100, 1),
                                 "needed_cagr_pct": round(needed_rate * 100, 1),
                                 "needed_reachable": reachable,
                                 "actual_basis": "time-weighted, own history, SIP flows removed",
                                 "needed_basis": "includes remaining monthly SIPs",
                                 "history_days": len(_twr),
                                 "sip_monthly": sip_monthly,
                                 "target_amount": float(target_amount),
                                 "months_remaining": months_remaining},
                                "goal_drift", severity=_sev
                            ))
                except (ValueError, TypeError, KeyError) as e:
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

                # W1: why the score moved, BOTH directions, computed once — the
                # up and down blocks below both read it.
                #
                # BASELINE IS ADD TIME. entry_trace is what we recorded when the
                # stock was watched, so the label answers "is this better than
                # when I flagged it" — the question a watchlist actually serves.
                # The alert TRIGGERS are narrower (vs last-notified for up, vs
                # yesterday for down), so headline and label can measure
                # different spans. detail carries drift_baseline='added' so
                # nothing downstream has to infer which.
                # W2: archetype for the severity floor below. Own try/except —
                # a missing column must never suppress the drift call that follows.
                try:
                    _wl_archetype = str(row.iloc[0].get("lynch_category") or "")
                except Exception:
                    _wl_archetype = ""
                try:
                    _wl_drift = selector.classify_score_change(
                        wl.get("entry_trace"), row.iloc[0])
                except Exception as _e:
                    # Never let classification suppress or soften an alert.
                    print(f"  [W1] watchlist drift failed for {wl_ticker}: {_e}")
                    _wl_blank = {"reason": "unknown_inputs", "frameworks": [],
                                 "per_framework": {}}
                    _wl_drift = {"traceable": False,
                                 "down": {**_wl_blank, "severity": "warning"},
                                 "up": dict(_wl_blank)}

                def wl_alert(alert_type, headline, detail, book_key, *, severity):
                    if severity not in _SEVERITIES:
                        raise ValueError(
                            f"severity must be one of {_SEVERITIES}, got "
                            f"{severity!r} (alert_type={alert_type!r})")
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
                        "severity": severity,
                        "ticker": wl_ticker,
                        "headline": headline,
                        "detail": {**detail, "book_passages": passages},
                        "alert_date": today_str,
                    }

                # Score up — fire ONLY when the score sets a NEW high above what we
                # last told the user about (last_notified_score), not every day it
                # merely sits above the entry score. This is what kills the daily
                # "KOTHARIA improved" repeat.
                _last_notified = wl.get("last_notified_score")
                if _last_notified is None:
                    _last_notified = wl.get("score_when_added")
                if cur_score is not None and _last_notified is not None and cur_score > _last_notified:
                    all_alerts.append(wl_alert(
                        "watchlist_score_up",
                        # Denominator from the live row — headlines are STORED and
                        # rendered verbatim downstream, so a hardcoded /5 here
                        # outlives every render-side fix.
                        f"👁 {wl_name} score improved {_last_notified} → "
                        f"{selector.score_label(row.iloc[0], cur_score)}",
                        {"prev_score": _last_notified, "current_score": cur_score, "source": "watchlist",
                         # reason may be None: the score rose against
                         # last-notified while the pass-set is unchanged since
                         # ADD time. That is "nothing changed versus when you
                         # flagged it", which is a real answer — not the same as
                         # "we could not tell", so it is not coerced to unclear.
                         "drift_reason": _wl_drift["up"]["reason"],
                         "drift_frameworks": _wl_drift["up"]["frameworks"],
                         "drift_per_framework": _wl_drift["up"]["per_framework"],
                         "drift_baseline": "added"},
                        # Severity stays info. There is no level above info for
                        # good news, so here the label shapes the MESSAGE and
                        # decides nothing.
                        "watchlist_score_up", severity="info"
                    ))
                    wl_alert_count += 1
                    # Remember we told them, stamp the date, reset the reason rotation
                    # so the email leads with THIS fresh improvement, then rotates.
                    try:
                        supabase.table("watchlist").update({
                            "last_notified_score": int(cur_score),
                            "score_improved_on": today_str,
                            "reasons_shown": []
                        }).eq("id", wl["id"]).execute()
                    except Exception as _e:
                        print(f"  watchlist state update failed for {wl_ticker}: {_e}")

                # Score down
                if cur_score is not None and prev_score is not None and cur_score < prev_score:
                    all_alerts.append(wl_alert(
                        "watchlist_score_down",
                        f"👁 {wl_name} score dropped {prev_score} → "
                        f"{selector.score_label(row.iloc[0], cur_score)}",
                        {"prev_score": prev_score, "current_score": cur_score, "source": "watchlist",
                         # None here means the add-time comparison shows no net
                         # framework loss even though the score fell against
                         # yesterday. We cannot explain that from what we track,
                         # so it is genuinely unclear — which keeps warning.
                         "drift_reason": _wl_drift["down"]["reason"] or "unclear",
                         "drift_frameworks": _wl_drift["down"]["frameworks"],
                         "drift_per_framework": _wl_drift["down"]["per_framework"],
                         "drift_baseline": "added"},
                        "watchlist_score_down",
                        # W2: archetype conditions SEVERITY, never the label.
                        # For a cyclical, "valuation" inverts — a falling PE with
                        # fundamentals intact is the peak-earnings signature, not
                        # a discount. HARDEN-ONLY, and the watchlist ceiling
                        # still caps it at warning (a watched stock is not owned).
                        severity=selector.apply_archetype_severity(
                            selector.WATCHLIST_DRIFT_SEVERITY.get(
                                _wl_drift["down"]["reason"] or "unclear", "warning"),
                            _wl_archetype,
                            _wl_drift["down"]["reason"] or "unclear",
                            ceiling="warning")
                    ))
                    wl_alert_count += 1

                # Quality flip
                if cur_quality is not None and prev_quality is not None and cur_quality != prev_quality:
                    flip_dir = "PASS" if cur_quality else "FAIL"
                    all_alerts.append(wl_alert(
                        "watchlist_quality_flip",
                        f"👁 {wl_name} quality flipped to {flip_dir}",
                        {"previous": prev_quality, "current": cur_quality, "source": "watchlist"},
                        "watchlist_quality_flip",
                        # Direction matters: a flip TO pass is good news, a flip
                        # to fail is an accounting red flag. One type, two
                        # severities — exactly what the split column is for.
                        severity=("info" if cur_quality else "warning")
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
                                "watchlist_near_low", severity="info"
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
            selector.meets_score_mask(universe_df, 3)
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
                # Recorded, not derived. It WAS recoverable as score minus the
                # other four (deep_metrics.py:1426 defines score as the sum of
                # exactly these five), and rows up to 2026-07-24 were backfilled
                # that way. But that identity is an undocumented invariant: if
                # score is ever redefined — to count only APPLICABLE frameworks,
                # say, which selector already models — new rows would follow one
                # rule and old rows another, with nothing recording which. An
                # append-only audit trail should store what it observed.
                "lynch_pass": bool(row["lynch_pass"]) if pd.notna(row.get("lynch_pass")) else None,
                "pe": round(float(row["pe"]), 2) if pd.notna(row.get("pe")) else None,
                "roe_pct": round(float(row["roe_pct"]), 2) if pd.notna(row.get("roe_pct")) else None,
                "quality_pass": bool(row["quality_pass"]) if pd.notna(row.get("quality_pass")) else None,
                # W2 abstention is INVISIBLE in the five booleans above: a False
                # lynch_pass means BOTH "failed Lynch" and "Lynch does not apply".
                # GREENPOWER.NS recorded lynch_pass false on 2026-07-27 having
                # passed it on 07-24 — nothing about the business changed, only
                # lynch_category flipping to unknown. This is the backtest's
                # input, and it would read that as deterioration.
                # Same shape as entry_trace: `pass == False AND framework not in
                # applicable` is an abstention, not a failure. One convention.
                # Rows before this column existed carry NULL and stay unknowable
                # — historical lynch_category was never stored, so there is
                # nothing honest to backfill from.
                "applicable": list(selector._applicable_frameworks(row)),
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
                        selector.meets_score_mask(universe_df, 3) &
                        (universe_df["quality_pass"] == True) &
                        (~universe_df["ticker"].isin(prev_tickers))
                    ]

                    for _, nr in new_high_scorers.iterrows():
                        all_alerts.append({
                            "portfolio_id": None,
                            "user_id": None,  # broadcast — weekly_mentor sends to all users
                            "alert_type": "new_entry",
                            "severity": "info",
                            "ticker": nr["ticker"],
                            "headline": (f"{nr.get('name', nr['ticker'])} new to radar at "
                                         f"score {selector.score_label(nr, int(nr['score']))}"),
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
    _failed_alerts = []
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
            _failed_alerts.append((alert.get("alert_type"), alert.get("ticker")))
            print(f"Alert write failed: {e}")

    print(f"Wrote {written} alerts.")
    if _failed_alerts:
        # A CHECK-constraint violation is a SCHEMA BUG, not a transient error:
        # it recurs for every row, every day, and prints into a log nobody
        # reads. On 2026-07-09 nine alerts were rejected because the constraint
        # allowed only 3 of the 10 types this file emits. `goal_drift` and all
        # four `watchlist_*` types had NEVER been writable — the watchlist alert
        # UI in app.py:6688 was reading a table the writer could not populate.
        from collections import Counter as _C
        _by_type = _C(t for t, _ in _failed_alerts)
        print(f"\n!! {len(_failed_alerts)} ALERTS REJECTED — by type: {dict(_by_type)}")
        print("!! If this is a CHECK-constraint violation, the schema is behind "
              "the code. Fix the constraint; do not silence this.")


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
                    _bp = _bench_price_for(pp)
                    _bt = pp.get("benchmark_ticker") or "NIFTYBEES.NS"
                    if not _bp or _bp <= 0:
                        print(f"  Skipping paper SIP for {_ticker}: no {_bt} close. "
                              f"A ledger row with NULL nifty_units would permanently "
                              f"understate the shadow portfolio.")
                        continue
                    _nifty_u = round(_new_amt / _bp, 6)
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
                            "nifty_price": _bp,
                            "nifty_units": _nifty_u,
                            "benchmark_ticker": _bt,
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
