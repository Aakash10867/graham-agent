"""
UNIVERSE UPDATER
================
Run monthly. Pulls all NSE + BSE tickers, fetches fundamentals for each,
scores every stock against all 4 frameworks, and saves a ready-to-use CSV.

The app reads this CSV directly — no live scanning needed.

Usage:
    python universe_updater.py

Output:
    universe_scored.csv — complete pre-processed universe with framework verdicts
"""

import requests
import pandas as pd
import yfinance as yf
import time
import io
import os
import sys
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import deep_metrics

# ──────────────────────────────────────────────
# SCHEMA VERSION — bump whenever a scorer threshold, formula, or the raw-input
# column set changes. Stamped onto every row of universe_scored.csv so a future
# re-score of an archived snapshot knows which scorer vintage produced it.
# History:
#   1  2026-07-10  dividend_yield fraction->percent fix; trajectory_pass >= 8
# ──────────────────────────────────────────────
SCHEMA_VERSION = 1

import random

# --- BROWSER FINGERPRINT POOL ---
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.3; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/122.0.0.0"
]

def get_rotated_session():
    """Returns a fresh session with a random modern user-agent to drop tracking cookies."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive"
    })
    return session

# ──────────────────────────────────────────────
# NSE LAST-KNOWN-GOOD CACHE
# ──────────────────────────────────────────────
# NSE blocks the Actions runner IP often enough that we keep the last FRESH
# equity list on disk (committed to the repo) and fall back to it when a fetch
# fails. Same dict shape fetch_nse_tickers() returns, so dedup is identical.
LASTGOOD_NSE = "nse_tickers_lastgood.csv"


def save_lastgood_nse(tickers):
    """Persist a fresh, floor-passing NSE list as last-known-good. Never raises."""
    try:
        pd.DataFrame(tickers).to_csv(LASTGOOD_NSE, index=False)
        print(f"[NSE] cached {len(tickers)} tickers -> {LASTGOOD_NSE} (last-known-good)")
    except Exception as e:
        print(f"[NSE] non-fatal: could not write cache: {type(e).__name__}: {e}")


def load_lastgood_nse():
    """Read the last committed good NSE list. Returns [] if absent/unreadable."""
    if not os.path.exists(LASTGOOD_NSE):
        print(f"[NSE] no {LASTGOOD_NSE} on disk — cannot fall back.")
        return []
    try:
        df = pd.read_csv(LASTGOOD_NSE)
        tickers = []
        for _, row in df.iterrows():
            symbol = str(row.get("symbol", "")).strip()
            if symbol and symbol.lower() != "nan":
                _isin = row.get("isin")
                tickers.append({
                    "symbol": symbol,
                    "name": str(row.get("name", "")).strip(),
                    "exchange": "NSE",
                    "isin": str(_isin).strip() if pd.notna(_isin) else "",
                })
        return tickers
    except Exception as e:
        print(f"[NSE] non-fatal: could not read cache: {type(e).__name__}: {e}")
        return []


# ──────────────────────────────────────────────
# NSE FETCHER
# ──────────────────────────────────────────────
def fetch_nse_tickers():
    """Fetch all equity tickers from NSE India."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
    })

    tickers = []

    # Method 1: CSV from NSE archives
    try:
        print("[NSE] Attempting CSV download from archives...")
        session.get("https://www.nseindia.com", timeout=10)
        time.sleep(1)

        # NSE migrated the archive to nsearchives.nseindia.com; the old
        # archives.nseindia.com host now intermittently 503s (esp. from CI IPs).
        # Try the current host first, fall back to the legacy one.
        csv_hosts = [
            "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
            "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
        ]
        resp = None
        for _u in csv_hosts:
            try:
                _r = session.get(_u, timeout=15)
                _r.raise_for_status()
                resp = _r
                print(f"[NSE] archive host OK: {_u.split('/')[2]}")
                break
            except Exception as _he:
                print(f"[NSE] archive host failed ({_u.split('/')[2]}): {_he}")
        if resp is None:
            raise RuntimeError("all NSE archive hosts failed")

        df = pd.read_csv(io.StringIO(resp.text))

        if "SERIES" in df.columns:
            df = df[df["SERIES"].isin(["EQ", "BE"])]

        symbol_col = [c for c in df.columns if "SYMBOL" in c.upper()][0]
        name_col = [c for c in df.columns if "NAME" in c.upper()][0]
        isin_cols = [c for c in df.columns if "ISIN" in c.upper()]
        isin_col = isin_cols[0] if isin_cols else None

        for _, row in df.iterrows():
            symbol = str(row[symbol_col]).strip()
            name = str(row[name_col]).strip()
            isin = str(row[isin_col]).strip() if isin_col and pd.notna(row.get(isin_col)) else ""
            if symbol and symbol != "nan":
                tickers.append({
                    "symbol": symbol,
                    "name": name,
                    "exchange": "NSE",
                    "isin": isin,
                })

        print(f"[NSE] CSV method: got {len(tickers)} tickers")
        return tickers

    except Exception as e:
        print(f"[NSE] CSV method failed: {e}")

    # Method 2: NSE API
    try:
        print("[NSE] Attempting API method...")
        session.get("https://www.nseindia.com", timeout=10)
        time.sleep(2)

        api_url = "https://www.nseindia.com/api/market-data-pre-open?key=ALL"
        resp = session.get(api_url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        for item in data.get("data", []):
            meta = item.get("metadata", {})
            symbol = meta.get("symbol", "")
            name = meta.get("companyName", "")
            if symbol:
                tickers.append({
                    "symbol": symbol,
                    "name": name,
                    "exchange": "NSE",
                    "isin": "",
                })

        print(f"[NSE] API method: got {len(tickers)} tickers")
        return tickers

    except Exception as e:
        print(f"[NSE] API method failed: {e}")

    return tickers


# ──────────────────────────────────────────────
# BSE FETCHER
# ──────────────────────────────────────────────
def fetch_bse_tickers():
    """Fetch all active equity tickers from BSE India, with a CI-safe mirror fallback."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
    })

    tickers = []

    try:
        print("[BSE] Attempting API method...")
        session.headers.update({"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"})
        session.get("https://www.bseindia.com/", timeout=15)
        time.sleep(2)
        
        session.headers.update({
            "Accept": "application/json",
            "Referer": "https://www.bseindia.com/",
        })
        
        url = (
            "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w"
            "?Group=&Scripcode=&industry=&segment=Equity&status=Active"
        )
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if data and len(data) > 0:
            print(f"[BSE] API returned {len(data)} items.")

        for item in data:
            scrip_code = str(item.get("SCRIP_CD") or "").strip()
            name = str(item.get("Issuer_Name") or item.get("Scrip_Name") or "").strip()
            group = str(item.get("GROUP") or "").strip()
            industry = str(item.get("INDUSTRY") or "").strip()
            isin = str(item.get("ISIN_NUMBER") or "").strip()

            if scrip_code and scrip_code not in ("", "nan", "None"):
                tickers.append({
                    "scrip_code": scrip_code,
                    "name": name,
                    "exchange": "BSE",
                    "group": group,
                    "industry": industry,
                    "isin": isin,
                })

        print(f"[BSE] API method: got {len(tickers)} tickers")
        return tickers

    except Exception as e:
        print(f"[BSE] API method blocked by WAF ({e}).")
        print("[BSE] Falling back to CI-safe GitHub mirror...")
        
        # GitHub Actions IP bypass: Fetch from a daily-updated community mirror
        try:
            mirror_url = "https://raw.githubusercontent.com/RuchiTanmay/bseindia/main/bseindia/bse_security_list.csv"
            df = pd.read_csv(mirror_url)
            
            # Normalize columns to handle minor upstream schema changes
            df.columns = [str(c).strip().lower() for c in df.columns]
            
            code_col = next((c for c in df.columns if 'code' in c), None)
            name_col = next((c for c in df.columns if 'name' in c or 'id' in c), None)
            grp_col = next((c for c in df.columns if 'group' in c), None)
            ind_col = next((c for c in df.columns if 'industry' in c), None)
            isin_col = next((c for c in df.columns if 'isin' in c), None)
            status_col = next((c for c in df.columns if 'status' in c), None)

            if status_col:
                df = df[df[status_col].astype(str).str.contains('Active', case=False, na=False)]

            for _, row in df.iterrows():
                scrip_code = str(row.get(code_col, "")).strip() if code_col else ""
                name = str(row.get(name_col, "")).strip() if name_col else ""
                
                if scrip_code and scrip_code not in ("", "nan", "None"):
                    tickers.append({
                        "scrip_code": scrip_code,
                        "name": name,
                        "exchange": "BSE",
                        "group": str(row.get(grp_col, "")).strip() if grp_col else "",
                        "industry": str(row.get(ind_col, "")).strip() if ind_col else "",
                        "isin": str(row.get(isin_col, "")).strip() if isin_col else "",
                    })
            print(f"[BSE] Mirror fallback: got {len(tickers)} tickers")
        except Exception as mirror_err:
            print(f"[BSE] Mirror method also failed: {mirror_err}")

    return tickers


# ──────────────────────────────────────────────
# COMBINER & DEDUPLICATOR
# ──────────────────────────────────────────────
def combine_and_deduplicate(nse_tickers, bse_tickers):
    """Combine NSE + BSE, dedup on ISIN, fallback to name."""
    combined = []
    nse_isins = set()
    nse_names_clean = set()

    for t in nse_tickers:
        yf_ticker = f"{t['symbol']}.NS"
        combined.append({"ticker": yf_ticker, "name": t["name"], "exchange": "NSE"})

        isin = t.get("isin", "").strip()
        if isin:
            nse_isins.add(isin)

        clean = (
            t["name"].lower()
            .replace(" ltd.", "").replace(" ltd", "")
            .replace(" limited", "").replace(" inc.", "")
            .replace(".", "").replace(",", "")
            .strip()
        )
        if clean:
            nse_names_clean.add(clean)

    bse_only_count = 0
    skipped_isin = 0
    skipped_name = 0

    for t in bse_tickers:
        isin = t.get("isin", "").strip()
        if isin and isin in nse_isins:
            skipped_isin += 1
            continue

        clean_name = (
            t["name"].lower()
            .replace(" ltd.", "").replace(" ltd", "")
            .replace(" limited", "").replace(" inc.", "")
            .replace(".", "").replace(",", "")
            .strip()
        )
        if clean_name and clean_name in nse_names_clean:
            skipped_name += 1
            continue

        yf_ticker = f"{t['scrip_code']}.BO"
        combined.append({"ticker": yf_ticker, "name": t["name"], "exchange": "BSE"})
        bse_only_count += 1

    print(f"[DEDUP] Matched by ISIN: {skipped_isin} | Matched by name: {skipped_name}")
    print(f"[COMBINED] NSE: {len(nse_tickers)} | BSE-only: {bse_only_count} | Total: {len(combined)}")
    return combined


# ──────────────────────────────────────────────
# FUNDAMENTALS FETCHER (per stock)
# ──────────────────────────────────────────────
# Every ticker that fails is recorded with WHY. "No data: 606" mixes delisted
# shells with tickers Yahoo throttled today. Only the second kind is
# non-deterministic, and only the second kind makes the universe — and
# therefore backtest_runner.py — irreproducible.
FAILURES = []   # (ticker, reason)


def fetch_fundamentals(ticker, retries=3):
    """Fetch all metrics needed for the 4 frameworks. Returns dict or None. Includes backoff."""
    for attempt in range(retries):
        try:
            # 1. CRITICAL: Let modern yfinance handle the session and crumb natively
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info or not info.get("regularMarketPrice"):
                FAILURES.append((ticker, "no_price"))
                return None, None
            # Compute years listed for Graham 7-year guard
            first_trade = info.get("firstTradeDateEpochUtc")
            if first_trade:
                first_date = datetime.fromtimestamp(first_trade, tz=timezone.utc)
                calc_years_listed = round((datetime.now(tz=timezone.utc) - first_date).days / 365.25, 1)
            else:
                calc_years_listed = None
            def _f(x):
                # yfinance's info dict occasionally returns numeric fields as
                # strings ('123.4', 'Infinity', 'N/A'). Those pass the truthy
                # guards below and then raise "'>' not supported between str and
                # int" on a "> 0" comparison — the TypeError that silently failed
                # 8 tickers. Coerce every numeric info field to float-or-None here.
                try:
                    v = float(x)
                    return v if (v == v and v not in (float("inf"), float("-inf"))) else None
                except (TypeError, ValueError):
                    return None

            pe = _f(info.get("trailingPE"))

            data = {
                "ticker": ticker,
                "years_listed": calc_years_listed,
                "name": info.get("longName") or info.get("shortName", ticker),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "price": _f(info.get("regularMarketPrice") or info.get("currentPrice")),
                "pe": pe,
                "pb": _f(info.get("priceToBook")),
                "roe": _f(info.get("returnOnEquity")),
                "de": _f(info.get("debtToEquity")),
                # yfinance changed convention: dividendYield now returns PERCENT
                # (3.29), not the fraction (0.0329) every consumer in this
                # codebase assumes. Normalize once, here, at ingest.
                #   deep_metrics:393  dy_pct = div_yield * 100
                #   deep_metrics:676  total_divs = div_yield * market_cap
                #   deep_metrics:1098 if dy > 0.04
                #   universe_updater:616 dividend_yield_pct = dy * 100
                # All four are correct against a fraction. Fixing them
                # individually would be four places to get wrong next time.
                "dividend_yield": (_f(info.get("dividendYield")) / 100.0
                                   if _f(info.get("dividendYield")) is not None else None),
                "eps": _f(info.get("trailingEps")),
                "earnings_yield": round(1.0 / pe * 100, 2) if pe and pe > 0 else None,
                "profit_margin": _f(info.get("profitMargins")),
                "market_cap": _f(info.get("marketCap")),
                "current_ratio": _f(info.get("currentRatio")),
                "beta": _f(info.get("beta")),
                "week52_high": _f(info.get("fiftyTwoWeekHigh")),
                "week52_low": _f(info.get("fiftyTwoWeekLow")),
                "pct_from_high": None,
                "pct_from_low": None,
                "pe_4y_avg": None,
                "pe_vs_avg": None,
                "revenue_cagr_3y": None,
                "ni_cagr_3y": None,
                "rev_growth": None,
                "ni_growth": None,
                "debt_growth": None,
                # Daily Tracking & Momentum Metrics
                "price_1d_pct": None,
                "price_5d_pct": None,
                "rsi_14": None,
                "vol_spike_flag": False,
                # Historical (y0 = most recent year, y3 = oldest)
                "years_of_data": 0,
                "revenue_y0": None, "revenue_y1": None, "revenue_y2": None, "revenue_y3": None,
                "net_income_y0": None, "net_income_y1": None, "net_income_y2": None, "net_income_y3": None,
                "total_debt_y0": None, "total_debt_y1": None, "total_debt_y2": None, "total_debt_y3": None,
                "equity_y0": None, "equity_y1": None, "equity_y2": None, "equity_y3": None,
                "roe_y0": None, "roe_y1": None, "roe_y2": None, "roe_y3": None,
                "de_y0": None, "de_y1": None, "de_y2": None, "de_y3": None,
            }
            # ── 52-Week Proximity ──
            _price = data["price"]
            _w52h = data["week52_high"]
            _w52l = data["week52_low"]
            if _price and _w52h and _w52h > 0:
                data["pct_from_high"] = round((_price / _w52h - 1) * 100, 2)
            if _price and _w52l and _w52l > 0:
                data["pct_from_low"] = round((_price / _w52l - 1) * 100, 2)

            # ── Daily Momentum & Tracking Data ──
            try:
                hist = stock.history(period="1mo")
                if not hist.empty and len(hist) >= 2:
                    closes = hist["Close"]
                    vols = hist["Volume"]

                    # 1D and 5D Returns
                    data["price_1d_pct"] = round((closes.iloc[-1] / closes.iloc[-2] - 1) * 100, 2)
                    if len(closes) >= 6:
                        data["price_5d_pct"] = round((closes.iloc[-1] / closes.iloc[-6] - 1) * 100, 2)
                    
                    # Volume Spike (>300% of average)
                    avg_vol = vols.mean()
                    if avg_vol > 0:
                        data["vol_spike_flag"] = bool(vols.iloc[-1] > (3 * avg_vol))
                        data["avg_daily_volume"] = round(float(avg_vol), 0)

                    # 14-day RSI
                    if len(closes) > 14:
                        delta = closes.diff()
                        gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
                        loss = -1 * delta.clip(upper=0).ewm(span=14, adjust=False).mean()
                        rs = gain / loss
                        rsi = 100 - (100 / (1 + rs))
                        data["rsi_14"] = round(float(rsi.iloc[-1]), 2)
            except Exception:
                pass 

            # Historical data extraction (up to 4 years)
            try:
                income_stmt = stock.financials
                if income_stmt is not None and not income_stmt.empty:
                    cols = sorted(income_stmt.columns)  
                    data["years_of_data"] = len(cols)

                    for i, col in enumerate(cols[-4:]):  
                        idx = len(cols[-4:]) - 1 - i  
                        try:
                            val = income_stmt.loc["Total Revenue", col]
                            if pd.notna(val): data[f"revenue_y{idx}"] = round(float(val), 2)
                        except KeyError: pass
                        try:
                            val = income_stmt.loc["Net Income", col]
                            if pd.notna(val): data[f"net_income_y{idx}"] = round(float(val), 2)
                        except KeyError: pass

                    if len(cols) >= 2:
                        last2 = sorted(cols)[-2:]
                        try:
                            rev = [income_stmt.loc["Total Revenue", c] for c in last2]
                            if all(pd.notna(v) and v > 0 for v in rev):
                                data["rev_growth"] = round((rev[1] / rev[0] - 1) * 100, 2)
                        except (KeyError, ZeroDivisionError): pass
                        try:
                            ni = [income_stmt.loc["Net Income", c] for c in last2]
                            if all(pd.notna(v) for v in ni) and ni[0] != 0:
                                data["ni_growth"] = round((ni[1] / ni[0] - 1) * 100, 2)
                        except (KeyError, ZeroDivisionError): pass
            except Exception:
                pass

            try:
                balance_sheet = stock.balance_sheet
                if balance_sheet is not None and not balance_sheet.empty:
                    cols = sorted(balance_sheet.columns)
                    for i, col in enumerate(cols[-4:]):
                        idx = len(cols[-4:]) - 1 - i
                        try:
                            val = balance_sheet.loc["Total Debt", col]
                            if pd.notna(val): data[f"total_debt_y{idx}"] = round(float(val), 2)
                        except KeyError: pass
                        try:
                            eq = balance_sheet.loc["Stockholders Equity", col]
                            if pd.notna(eq):
                                data[f"equity_y{idx}"] = round(float(eq), 2)
                                ni_key = f"net_income_y{idx}"
                                if data.get(ni_key) and float(eq) > 0:
                                    data[f"roe_y{idx}"] = round(data[ni_key] / float(eq) * 100, 2)
                                debt_key = f"total_debt_y{idx}"
                                if data.get(debt_key) and float(eq) > 0:
                                    data[f"de_y{idx}"] = round(data[debt_key] / float(eq) * 100, 2)
                        except KeyError: pass

                    if len(cols) >= 2:
                        last2 = sorted(cols)[-2:]
                        try:
                            debt = [balance_sheet.loc["Total Debt", c] for c in last2]
                            if all(pd.notna(v) for v in debt) and debt[0] > 0:
                                data["debt_growth"] = round((debt[1] / debt[0] - 1) * 100, 2)
                        except (KeyError, ZeroDivisionError): pass
            except Exception:
                pass

            # ── Historical PE & Growth Rates ──
            try:
                shares_out = info.get("sharesOutstanding")
                if shares_out and shares_out > 0:
                    pe_history = []
                    for yr in range(4):
                        ni = data.get(f"net_income_y{yr}")
                        if ni and ni > 0:
                            hist_eps = ni / shares_out
                            hist_pe = data["price"] / hist_eps if hist_eps > 0 else None
                            if hist_pe and 0 < hist_pe < 200:  # sanity bounds
                                pe_history.append(hist_pe)
                    if pe_history:
                        data["pe_4y_avg"] = round(sum(pe_history) / len(pe_history), 2)
                        if pe and pe > 0 and data["pe_4y_avg"] > 0:
                            data["pe_vs_avg"] = round((pe / data["pe_4y_avg"] - 1) * 100, 2)
            except Exception:
                pass

            try:
                rev_y0 = data.get("revenue_y0")
                rev_y3 = data.get("revenue_y3")
                if rev_y0 and rev_y3 and rev_y3 > 0 and rev_y0 > 0:
                    data["revenue_cagr_3y"] = round(((rev_y0 / rev_y3) ** (1/3) - 1) * 100, 2)

                ni_y0 = data.get("net_income_y0")
                ni_y3 = data.get("net_income_y3")
                if ni_y0 and ni_y3 and ni_y3 > 0 and ni_y0 > 0:
                    data["ni_cagr_3y"] = round(((ni_y0 / ni_y3) ** (1/3) - 1) * 100, 2)
            except Exception:
                pass

            # ── Earnings quality checks ──
            try:
                ni_y0 = data.get("net_income_y0")
                ni_y1 = data.get("net_income_y1")
                ni_y2 = data.get("net_income_y2")

                cashflow = stock.cashflow
                if cashflow is not None and not cashflow.empty:
                    cf_cols = sorted(cashflow.columns)
                    ocf = None
                    for row_name in ["Operating Cash Flow", "Total Cash From Operating Activities"]:
                        if row_name in cashflow.index:
                            val = cashflow.loc[row_name, cf_cols[-1]]
                            if pd.notna(val):
                                ocf = float(val)
                                break
                    if ocf is not None and ni_y0 and ni_y0 > 0:
                        data["cash_conversion"] = round(ocf / ni_y0, 2)
                    else:
                        data["cash_conversion"] = None
                else:
                    data["cash_conversion"] = None

                prior_ni = [v for v in [ni_y1, ni_y2] if v is not None and v > 0]
                if ni_y0 and len(prior_ni) >= 1:
                    prior_avg = sum(prior_ni) / len(prior_ni)
                    if prior_avg > 0:
                        data["earnings_spike"] = round(ni_y0 / prior_avg, 2)
                    else:
                        data["earnings_spike"] = None
                else:
                    data["earnings_spike"] = None

                if income_stmt is not None and not income_stmt.empty:
                    latest_col = sorted(income_stmt.columns)[-1]
                    op_income = None
                    for row_name in ["Operating Income", "EBIT"]:
                        if row_name in income_stmt.index:
                            val = income_stmt.loc[row_name, latest_col]
                            if pd.notna(val):
                                op_income = float(val)
                                break
                    if op_income and op_income > 0 and ni_y0 and ni_y0 > 0:
                        data["non_op_pct"] = round((ni_y0 - op_income) / ni_y0 * 100, 2)
                    else:
                        data["non_op_pct"] = None
                else:
                    data["non_op_pct"] = None

                quality_pass = True
                cc = data.get("cash_conversion")
                spike = data.get("earnings_spike")
                non_op = data.get("non_op_pct")

                if cc is not None and cc < 0.5 and ni_y0 and ni_y0 > 0: quality_pass = False
                if spike is not None and spike > 3.0: quality_pass = False
                if non_op is not None and non_op > 40: quality_pass = False

                data["quality_pass"] = quality_pass

            except Exception:
                data["cash_conversion"] = None
                data["earnings_spike"] = None
                data["non_op_pct"] = None
                data["quality_pass"] = True  

            # ── Sprint 13 §4: guard an IMPOSSIBLE dividend yield ──
            # info["dividendYield"] roughly doubles after a bonus/split (e.g.
            # Wipro 1:1, Dec 2024): trailing DPS over a post-action price. A
            # payout above 150% of a PROFITABLE company's earnings is
            # arithmetically implausible, so rather than feed a value we know is
            # wrong into Lynch / Graham payout / Dorsey, we FLAG it and NULL the
            # yield for scoring — "unknown" beats "wrong". Payout is derived from
            # the same numbers already in `data` (yield x market_cap / net income),
            # so flag and null can't disagree, and only the handful of
            # provably-broken rows are touched — clean yields, including
            # legitimate >100% special/reserve-funded payers, are left alone.
            data["dividend_yield_unreliable"] = False
            _dy = data.get("dividend_yield")
            _mc = data.get("market_cap")
            _ni0 = data.get("net_income_y0")
            if _dy and _dy > 0 and _mc and _mc > 0 and _ni0 and _ni0 > 0:
                if (_dy * _mc) / _ni0 > 1.5:          # payout > 150% on a profit
                    data["dividend_yield_unreliable"] = True
                    data["dividend_yield"] = None      # do not trust it in scoring

            # 2. Be nice to Yahoo: Small delay between successful requests
            time.sleep(0.5) 
            return data, stock

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "Too Many Requests" in error_str:
                # 3. EXPONENTIAL BACKOFF: If blocked, sleep for 5s, 10s, 20s
                sleep_time = (2 ** attempt) * 5
                print(f"[{ticker}] Rate limited. Sleeping {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                if "404" in error_str:
                    _reason = "404_not_found"
                elif "delisted" in error_str.lower():
                    _reason = "delisted"
                else:
                    _reason = f"error:{type(e).__name__}"
                    print(f"[{ticker}] Failed: {e}")
                FAILURES.append((ticker, _reason))
                return None, None

    # Exhausted all retries — this only happens under 429 backoff. The old code
    # returned None here silently, making a throttled ticker indistinguishable
    # from a delisted one in the `failed` counter.
    FAILURES.append((ticker, "rate_limited"))
    print(f"[{ticker}] DROPPED after {retries} retries (rate limited).")
    return None, None


# ──────────────────────────────────────────────
# BULK PROCESSOR
# ──────────────────────────────────────────────
def process_universe(ticker_list, max_workers=2):
    """
    Fetch fundamentals for all tickers in parallel, score frameworks.
    Prints live progress. Returns list of scored dicts.
    """
    total = len(ticker_list)
    results = []
    failed = 0
    completed = 0

    print(f"\n[SCAN] Processing {total} tickers with {max_workers} workers...")
    print(f"[SCAN] Estimated time: {total // max_workers * 2 // 60} - {total // max_workers * 3 // 60} minutes\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_fundamentals, t["ticker"]): t
            for t in ticker_list
        }

        for future in as_completed(futures):
            completed += 1
            ticker_info = futures[future]

            try:
                result = future.result()
                if result and result[0]:
                    data, stock_obj = result
                    try:
                        deep_metrics.compute_all_deep_metrics(data, stock_obj)
                    except Exception as dm_err:
                        print(f"  [DEEP] {data.get('ticker', '?')}: {dm_err}")
                    results.append(data)
                else:
                    failed += 1
            except Exception:
                failed += 1

            # Progress update every 100 stocks
            if completed % 100 == 0 or completed == total:
                pct = completed / total * 100
                print(
                    f"  [{completed:>5}/{total}] {pct:5.1f}%  |  "
                    f"Valid: {len(results)}  |  No data: {failed}",
                    flush=True,
                )

    # ── Retry the throttled ──────────────────────────────────────────────
    # Rate-limited tickers are the ONLY non-deterministic failure class: a
    # rerun would drop a different 88. They fail in bursts, so by the time the
    # scan ends the window has long since cleared. 88 x 0.5s is ~45 seconds,
    # and it is the difference between a reproducible universe and a universe
    # that is a function of Yahoo's mood that morning.
    _throttled = [t for t, r in FAILURES if r == "rate_limited"]
    if _throttled:
        print(f"\n[RETRY] {len(_throttled)} rate-limited tickers. Cooling 60s...")
        time.sleep(60)
        FAILURES[:] = [(t, r) for t, r in FAILURES if r != "rate_limited"]
        _recovered = 0
        for _t in _throttled:
            _data, _stock = fetch_fundamentals(_t, retries=5)
            if _data:
                try:
                    deep_metrics.compute_all_deep_metrics(_data, _stock)
                    results.append(_data)
                    _recovered += 1
                except Exception:
                    FAILURES.append((_t, "retry_scoring_failed"))
            time.sleep(1.0)
        print(f"[RETRY] Recovered {_recovered}/{len(_throttled)}. "
              f"Still lost: {len(_throttled) - _recovered}")

    # Greenblatt universe-level ranking (requires all stocks)
    deep_metrics.compute_greenblatt_ranks(results)
    return results


# ──────────────────────────────────────────────
# ARCHIVE COMPLETENESS GUARD
# ──────────────────────────────────────────────
def verify_archive_completeness(df):
    """Refuse to commit a snapshot that a future scorer could not re-derive.

    The whole point of the archive is that any snapshot can be re-scored by a
    later version of deep_metrics. That only holds if the RAW INPUTS the scorer
    reads are all present in the saved columns. This guard proves it, per commit,
    by re-running the two pure verdict functions against the saved columns and
    asserting the result matches what was written.

    A mismatch means a saved score can NOT be reproduced from saved inputs —
    i.e. the archive is silently un-rescoreable. That is a red CI run today, not
    a discovery in 2027.

    Cheap: pure Python over the rows, no network, ~1-2s on 4,400 rows.
    """
    import math

    # Inputs read by compute_trajectory_score + compute_framework_verdicts.
    # If deep_metrics grows a new input, add it here AND to the column list —
    # this list existing is what forces that discipline.
    REQUIRED_INPUTS = [
        # trajectory raw inputs
        "ni_cagr_3y", "revenue_cagr_3y", "rev_growth", "debt_growth", "de",
        "revenue_y0", "revenue_y1", "revenue_y2", "revenue_y3",
        "net_income_y0", "net_income_y1", "net_income_y2", "net_income_y3",
        # sub-scores the verdict function thresholds
        "graham_defensive_score", "greenblatt_score", "dorsey_buffett_score",
        "trajectory_score", "lynch_score", "lynch_category",
    ]
    missing = [c for c in REQUIRED_INPUTS if c not in df.columns]
    if missing:
        print("[ARCHIVE] FATAL: raw inputs missing from the saved column set:")
        for c in missing:
            print(f"[ARCHIVE]   {c}")
        print("[ARCHIVE] These snapshots could not be re-scored later. Add the "
              "column(s) to base_columns/deep_columns before committing.")
        sys.exit(1)

    # Re-derive on a sample and confirm reproducibility. Full-universe is fine
    # too, but a 200-row sample catches drift just as reliably and stays instant.
    sample = df.head(200)
    mismatches = 0
    for _, row in sample.iterrows():
        d = row.to_dict()
        before = (d.get("trajectory_score"), d.get("score"))
        deep_metrics.compute_trajectory_score(d)
        deep_metrics.compute_framework_verdicts(d)
        after = (d.get("trajectory_score"), d.get("score"))
        # trajectory_score is recomputed from raw inputs; score from sub-scores.
        for b, a in zip(before, after):
            if b is None or (isinstance(b, float) and math.isnan(b)):
                continue
            if int(b) != int(a):
                mismatches += 1
                break
    if mismatches:
        pct = mismatches / len(sample)
        print(f"[ARCHIVE] FATAL: {mismatches}/{len(sample)} sampled rows "
              f"({pct:.0%}) could not be reproduced from their own saved inputs.")
        print("[ARCHIVE] The saved scores and the saved inputs disagree. Either a "
              "scorer changed without a re-run, or an input column is stale. Fix "
              "before committing — this snapshot is not re-scoreable.")
        sys.exit(1)
    print(f"[ARCHIVE] OK: {len(sample)} rows re-derived from saved inputs, "
          f"schema_version={SCHEMA_VERSION}.")


# ──────────────────────────────────────────────
# ARCHIVE MANIFEST — the clock the "Does It Work?" page counts
# ──────────────────────────────────────────────
# One record per COMMITTED, GUARD-PASSED snapshot. Written by the same process
# that produces the archive, at the moment it produces it — so the counter and
# the archive are the same object, not a proxy. Carries schema_version so the
# page can count only clean-scorer snapshots (>= 1). Committed to the repo, so
# the deployed Streamlit app reads it directly with no git history or DB call.
#
# CLOCK START: 2026-07-10, the day the dividend + trajectory_pass fixes landed.
# Snapshots before that ran a scorer with wrong flags and do NOT count.
ARCHIVE_MANIFEST = "archive_manifest.json"
CLOCK_START = "2026-07-10"


def append_archive_manifest(df, manifest_path=ARCHIVE_MANIFEST):
    """Record this snapshot in the clock. Idempotent per date: re-running on the
    same day overwrites that day's entry rather than double-counting. Never
    raises — a manifest failure must not fail the universe commit."""
    import json
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        entry = {
            "date": today,
            "schema_version": SCHEMA_VERSION,
            "n_universe": int(len(df)),
            "n_investable": int(len(df[df["quality_pass"] == True]))
                            if "quality_pass" in df.columns else None,
        }
        records = []
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path) as f:
                    records = json.load(f)
                if not isinstance(records, list):
                    records = []
            except Exception:
                records = []
        # idempotent: drop any existing entry for today, then append
        records = [r for r in records if r.get("date") != today]
        records.append(entry)
        records.sort(key=lambda r: r.get("date", ""))
        with open(manifest_path, "w") as f:
            json.dump(records, f, indent=2)

        # count clean-scorer snapshots since the clock start
        clean = [r for r in records
                 if r.get("schema_version", 0) >= 1 and r.get("date", "") >= CLOCK_START]
        print(f"[MANIFEST] recorded {today}; clean snapshots since "
              f"{CLOCK_START}: {len(clean)}")
    except Exception as e:
        print(f"[MANIFEST] non-fatal: could not update manifest: "
              f"{type(e).__name__}: {e}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("ALPHACONSENSUS UNIVERSE UPDATER")
    print(f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # ── Step 1: Fetch ticker lists ──
    # Composition floors: an exchange returning below these is a STRUCTURAL
    # break (its fetch was blocked), not a light delisting day. Set well below
    # known-good counts (NSE ~2,384, BSE mirror ~4,330), well above any real
    # delisting day.
    _NSE_FLOOR = 1500
    _BSE_FLOOR = 3000
    _NSE_HEALTHY = 2000   # only (re)cache last-known-good from a clearly-complete fetch

    nse_tickers = fetch_nse_tickers()

    # ── NSE last-known-good fallback ──
    # NSE routinely IP-blocks the Actions runner (503 on the archive + timeout on
    # the API), and it's a fresh runner IP each run, so retrying the same blocked
    # IP in-process can't win. Instead of losing the whole day, fall back to the
    # last FRESH NSE list we committed. The equity master moves slowly, so a
    # days-stale list keeps composition correct and dedup identical (ISIN kept) —
    # far better than a .BO-only universe.
    _nse_from_cache = False
    if len(nse_tickers) < _NSE_FLOOR:
        print(f"[NSE] fresh fetch returned {len(nse_tickers)} (< floor "
              f"{_NSE_FLOOR}); falling back to last-known-good committed list.")
        _cached = load_lastgood_nse()
        if len(_cached) >= _NSE_FLOOR:
            nse_tickers = _cached
            _nse_from_cache = True
            print(f"[NSE] using {len(nse_tickers)} CACHED tickers (stale but "
                  f"present — NSE was unreachable this run).")
    print()

    bse_tickers = fetch_bse_tickers()
    print()

    # ── Fail-fast on composition, before the 90-min scan ──
    # Fatal now only when NSE is truly unrecoverable: fresh fetch failed AND no
    # usable cache. A blocked-NSE morning WITH a cache proceeds on the cached
    # list, so the downstream watchlist email still fires.
    if len(nse_tickers) < _NSE_FLOOR:
        print(f"FATAL: NSE unavailable — fresh fetch failed and no usable "
              f"last-known-good list (have {len(nse_tickers)}, floor {_NSE_FLOOR}). "
              f"The universe would be .BO-only, breaking every consumer that "
              f"assumes .NS. Not scanning; rerun later. NOTE: the cache is seeded "
              f"by the first fully-successful run — until one lands, a blocked NSE "
              f"still fatals here.")
        sys.exit(1)
    if len(bse_tickers) < _BSE_FLOOR:
        print(f"FATAL: BSE returned {len(bse_tickers)} tickers (floor {_BSE_FLOOR}). "
              f"BSE API WAF + GitHub mirror both failed. Not scanning; rerun later.")
        sys.exit(1)

    # Refresh last-known-good — but ONLY from a fresh, clearly-complete fetch.
    # Never re-cache a fallback (resets its apparent age, snowballs staleness);
    # never cache a marginal fetch (poisons the floor for future fallbacks).
    if _nse_from_cache:
        print("[NSE] ran on cached list; leaving last-known-good untouched.")
    elif len(nse_tickers) >= _NSE_HEALTHY:
        save_lastgood_nse(nse_tickers)
    else:
        print(f"[NSE] fresh list {len(nse_tickers)} < healthy {_NSE_HEALTHY}; "
              f"NOT refreshing cache (avoids poisoning last-known-good).")

    combined = combine_and_deduplicate(nse_tickers, bse_tickers)

    # Save raw ticker list too (for reference)
    raw_df = pd.DataFrame(combined).sort_values("ticker").reset_index(drop=True)
    raw_df.to_csv("universe_tickers.csv", index=False)
    print(f"\nSaved raw ticker list: universe_tickers.csv ({len(raw_df)} tickers)")

    # ── Step 2: Fetch fundamentals & score ──
    print("\n--- STEP 2: Fetching fundamentals & scoring frameworks ---\n")
    scored_results = process_universe(combined, max_workers=2)

    # ── Step 3: Save scored universe ──
    print("\n--- STEP 3: Saving scored universe ---\n")

    if not scored_results:
        print("ERROR: No valid data was retrieved for any ticker. Exiting to prevent DataFrame crash.")
        sys.exit(1)

    # Convert ROE and dividend_yield from decimal to percentage for readability
    for r in scored_results:
        if r.get("roe") is not None:
            r["roe_pct"] = round(r["roe"] * 100, 2)
        else:
            r["roe_pct"] = None
        if r.get("dividend_yield") is not None:
            r["dividend_yield_pct"] = round(r["dividend_yield"] * 100, 2)
        else:
            r["dividend_yield_pct"] = None

    # ── Dividend-yield convention guard ──
    # If yfinance flips convention again, this fires loudly instead of
    # silently corrupting lynch_score, graham_payout_ratio, and dorsey_pass.
    # As fractions, a universe median should sit near 0.006 (0.6%). A median
    # above 0.30 means we are reading percents as fractions; below 0.0001
    # means the opposite.
    _dys = [r["dividend_yield"] for r in scored_results
            if r.get("dividend_yield") is not None and r["dividend_yield"] > 0]
    if _dys:
        _dys.sort()
        _med = _dys[len(_dys) // 2]
        _mx = _dys[-1]
        print(f"\n[GUARD] dividend_yield: n={len(_dys)} median={_med:.5f} max={_mx:.4f}")
        if not (0.00005 < _med < 0.30):
            print("[GUARD] FATAL: dividend_yield convention has changed upstream.")
            print("[GUARD] Expected a FRACTION (0.033 == 3.3%). Fix the ingest "
                  "normalization in fetch_fundamentals before trusting this CSV.")
            sys.exit(1)
        if _mx > 0.50:
            print(f"[GUARD] WARN: max yield {_mx:.1%} — verify against screener.in.")

    # ── Lender exclusion (declared, not accidental) ──
    # Kordent has no framework for net interest margin, asset quality, or
    # capital adequacy. Scoring a bank on manufacturing ratios and printing
    # a verdict is worse than declining to score it. Note this is emitted as
    # a COLUMN, not a filter — the selector decides, and discloses.
    # Verified 2026-07 against yfinance industry strings on 4,461 stocks.
    # HDFCBANK/ICICIBANK/SBIN/DCBBANK -> "Banks - Regional"
    # LICHSGFIN -> "Mortgage Finance"      BAJFINANCE -> "Credit Services"
    # BSE/CRISIL -> "Financial Data & Stock Exchanges"   CDSL -> "Capital Markets"
    # The last three are NOT lenders and must pass through.
    _UNEVALUABLE_INDUSTRIES = {
        # Lenders: net interest margin, asset quality, capital adequacy.
        "Banks - Regional", "Banks - Diversified", "Banks—Regional",
        "Banks—Diversified", "Credit Services", "Mortgage Finance",
        # Insurers: float, combined ratio, embedded value.
        "Insurance - Life", "Insurance - Diversified", "Insurance - Reinsurance",
        "Insurance - Property & Casualty", "Insurance Brokers",
        "Insurance - Specialty",
        # Shells: no business to evaluate.
        "Shell Companies",
    }
    for r in scored_results:
        _ind = (r.get("industry") or "")
        r["is_unevaluable"] = bool(_ind in _UNEVALUABLE_INDUSTRIES)
        r["unevaluable_reason"] = _ind if _ind in _UNEVALUABLE_INDUSTRIES else ""

    # ── Existing columns ──
    base_columns = [
        "ticker", "name", "sector", "industry",
        "is_unevaluable", "unevaluable_reason",
        "price", "market_cap", "years_listed",
        "pe", "pb", "roe", "roe_pct", "de", "eps",
        "dividend_yield", "dividend_yield_pct", "dividend_yield_unreliable", "profit_margin",
        "current_ratio", "beta",
        "week52_high", "week52_low", "pct_from_high", "pct_from_low",
        "pe_4y_avg", "pe_vs_avg",
        "rev_growth", "ni_growth", "debt_growth",
        "revenue_cagr_3y", "ni_cagr_3y",
        "price_1d_pct", "price_5d_pct", "rsi_14", "vol_spike_flag", "avg_daily_volume",
        "risk_tier", "liquidity_flag",
        "years_of_data",
        "revenue_y0", "revenue_y1", "revenue_y2", "revenue_y3",
        "net_income_y0", "net_income_y1", "net_income_y2", "net_income_y3",
        "total_debt_y0", "total_debt_y1", "total_debt_y2", "total_debt_y3",
        "equity_y0", "equity_y1", "equity_y2", "equity_y3",
        "roe_y0", "roe_y1", "roe_y2", "roe_y3",
        "de_y0", "de_y1", "de_y2", "de_y3",
        "earnings_spike", "non_op_pct",
    ]
    # ── Sprint 6: Deep Framework Columns ──
    deep_columns = [
        # Balance Sheet Health
        "graham_adequate_size", "graham_current_ratio_pass", "graham_ltd_vs_nca",
        "graham_net_current_assets", "graham_ncav_per_share", "graham_ncav_ratio",
        "graham_bvps", "graham_price_to_ntav", "graham_net_cash",
        "lynch_net_cash_per_share", "dorsey_financial_leverage", "dorsey_interest_coverage",
        "dorsey_quick_ratio", "dorsey_clean_balance_sheet",
        # Earnings Quality
        "graham_earnings_stable_4y", "graham_avg_eps_4y", "graham_eps_cv",
        "graham_eps_growth_pct_4y", "schilit_accruals_ratio", "schilit_cfo_ni_ratio",
        "schilit_fcf_ni_ratio", "mulford_ecm", "mulford_ecm_trend",
        "mulford_cash_margin", "mulford_ocf_oi_ratio", "dorsey_consistent_cfo",
        # Valuation
        "graham_pe_3y_avg", "graham_pe_pb_composite", "graham_number",
        "graham_earnings_yield_spread", "graham_intrinsic_value",
        "graham_margin_of_safety_pct", "greenblatt_earnings_yield",
        "greenblatt_combined_rank", "lynch_peg", "lynch_peg_adjusted",
        "lynch_cash_adjusted_pe", "dorsey_cash_return",
        "buffett_intrinsic_value", "buffett_margin_of_safety_pct",
        # Growth Trajectory
        "lynch_growth_flag", "lynch_growth_acceleration",
        "buffett_value_creating_growth", "greenblatt_ebit_avg_4y",
        "graham_ent_earnings_growing",
        # Moat Durability
        "greenblatt_roic", "greenblatt_roic_trend", "dorsey_roic",
        "dorsey_fcf_margin", "dorsey_roe_consistent", "dorsey_roa",
        "dorsey_pb_roe_signal", "buffett_roe_unleveraged",
        # Dividend Quality
        "dividend_consecutive_years", "graham_payout_ratio",
        "graham_ent_has_dividend", "graham_deep_value_flag",
        # Management Quality
        "buffett_owner_earnings_ps", "buffett_one_dollar_test",
        "buffett_rational_allocation", "dorsey_share_dilution_pct",
        "dorsey_has_operating_profit",
        # Manipulation Flags
        "schilit_dso", "schilit_ar_revenue_divergence", "schilit_capex_depr_ratio",
        "schilit_dsi", "schilit_inventory_revenue_div", "schilit_wc_cffo_pct",
        "schilit_leverage_trend", "schilit_serial_acquirer", "goodwill_pct",
        "dorsey_cfo_ni_divergence", "dorsey_ar_growth_flag", "lynch_inventory_flag",
        "greenblatt_sector_excluded", "greenblatt_low_pe_flag", "mulford_fcf_consistent",
        # Classification
        "lynch_category", "lynch_debt_healthy", "mulford_lifecycle_stage",
        "graham_ent_financial_pass",
        # Spectrum Scores
        "graham_defensive_score", "graham_enterprising_score", "greenblatt_score",
        "dorsey_buffett_score", "dorsey_10min_score", "lynch_score",
        "schilit_manipulation_score", "mulford_cashflow_quality_score",
        # Quality Gate & Framework Verdicts
        "quality_pass",
        "trajectory_score",
        "graham_pass", "greenblatt_pass", "dorsey_pass", "trajectory_pass",
        "lynch_pass", "score",
    ]
    columns = base_columns + deep_columns

    df = pd.DataFrame(scored_results)

    # Only keep columns that exist
    columns = [c for c in columns if c in df.columns]
    df = df[columns].sort_values("ticker").reset_index(drop=True)

    # ── Risk tier and liquidity flag ──
    def _risk_tier(mcap):
        if pd.isna(mcap) or mcap is None:
            return "Unknown"
        if mcap >= 2e11:      # ≥ ₹20,000 Cr
            return "Large"
        elif mcap >= 5e10:    # ≥ ₹5,000 Cr
            return "Mid"
        elif mcap >= 5e9:     # ≥ ₹500 Cr
            return "Small"
        else:
            return "Micro"    # < ₹500 Cr — the universe had no word for this

    if "market_cap" in df.columns:
        df["risk_tier"] = df["market_cap"].apply(_risk_tier)
    else:
        df["risk_tier"] = "Unknown"

    if "avg_daily_volume" in df.columns:
        df["liquidity_flag"] = df["avg_daily_volume"].apply(
            lambda v: "illiquid" if pd.notna(v) and v < 50000 else "liquid"
        )
    else:
        df["liquidity_flag"] = "Unknown"

    # Add metadata
    df["updated_date"] = datetime.now().strftime("%Y-%m-%d")
    df["schema_version"] = SCHEMA_VERSION

    # ── Failure breakdown ──────────────────────────────────────────────
    from collections import Counter as _Counter
    if FAILURES:
        _reasons = _Counter(r for _, r in FAILURES)
        print(f"\n[SCAN] {len(FAILURES)} tickers produced no data:")
        for _r, _n in _reasons.most_common():
            print(f"         {_r:<20s} {_n:>5d}")
        _rate_limited = _reasons.get("rate_limited", 0)
        pd.DataFrame(FAILURES, columns=["ticker", "reason"]).to_csv(
            "universe_failures.csv", index=False)
        print("[SCAN] Wrote universe_failures.csv")
        if _rate_limited:
            print(f"[SCAN] WARNING: {_rate_limited} tickers lost to Yahoo throttling.")
            print("[SCAN] These are NON-DETERMINISTIC — a rerun would return a "
                  "different universe. This is the reproducibility ceiling.")

    # ── Regression guard ───────────────────────────────────────────────
    # Refuse to overwrite a good universe with a throttled one. The CSV is
    # committed to git and read by app.py, portfolio_tracker.py and
    # backtest_runner.py. A silent 10% shrink corrupts all three.
    output_file = "universe_scored.csv"
    if os.path.exists(output_file):
        try:
            _prev = len(pd.read_csv(output_file))
            _delta = (len(df) - _prev) / _prev
            # Reason-based, not a raw % floor. The universe breathes daily (real
            # delistings, new listings) so a fixed threshold blocks legitimate
            # delisting days and forces a manual rerun. What we must NOT do is
            # overwrite a good universe with one missing LIVE stocks because Yahoo
            # timed out. So we ask WHY it shrank: TRANSIENT failures (rate limits /
            # network errors) come back on a rerun; REAL losses (delisted / 404 /
            # no price) are genuine and should overwrite. A catastrophic drop is
            # blocked regardless — that is a systemic break, not a normal day.
            _transient = sum(1 for _t, r in FAILURES
                             if r == "rate_limited" or r.startswith("error:"))
            _best = len(df) + _transient          # if every transient had succeeded
            _delta_best = (_best - _prev) / _prev
            print(f"\n[GUARD] universe size: {_prev} -> {len(df)} ({_delta:+.1%}); "
                  f"transient failures {_transient} -> best-case {_best} "
                  f"({_delta_best:+.1%})")
            _CATASTROPHIC = -0.15    # systemic break — block whatever the labels say
            _FLOOR = -0.02
            if _delta < _CATASTROPHIC:
                print(f"[GUARD] FATAL: catastrophic shrink ({_delta:+.1%}). Systemic "
                      f"problem, not a normal day. Not overwriting.")
                sys.exit(1)
            elif _delta < _FLOOR and _delta_best >= _FLOOR:
                print(f"[GUARD] FATAL: the shrink is TRANSIENT — recovering the "
                      f"{_transient} timed-out/throttled tickers would restore the "
                      f"universe. Not overwriting; rerun when Yahoo is healthy.")
                sys.exit(1)
            else:
                print("[GUARD] shrink is real (delistings / no-price dominate; a "
                      "rerun would not recover them) — accepting the overwrite.")
        except SystemExit:
            raise
        except Exception as _e:
            print(f"[GUARD] Could not compare to previous universe: {_e}")

    # Protect the clock: a snapshot we cannot re-score later is worthless as
    # an archive. Prove re-scoreability before committing.
    verify_archive_completeness(df)

    df.to_csv(output_file, index=False)

    # Record this guard-passed, committed snapshot in the clock the
    # "Does It Work?" page counts. Non-fatal on failure.
    append_archive_manifest(df)

    # ── Summary ──
    total_scored = len(df)
    tier5 = len(df[df["score"] == 5]) if "score" in df.columns else 0
    tier4 = len(df[df["score"] == 4]) if "score" in df.columns else 0
    tier3 = len(df[df["score"] == 3]) if "score" in df.columns else 0
    tier2 = len(df[df["score"] == 2]) if "score" in df.columns else 0
    quality_failed = len(df[df["quality_pass"] == False]) if "quality_pass" in df.columns else 0
    tier5_traps = len(df[(df["score"] == 5) & (df["quality_pass"] == False)]) if "quality_pass" in df.columns else 0

    print(f"Saved {total_scored} scored stocks to {output_file}")
    print(f"\n  Score 5/5 (Perfect):  {tier5} stocks")
    print(f"  Score 4/5 (Strong):   {tier4} stocks")
    print(f"  Score 3/5 (Moderate): {tier3} stocks")
    print(f"  Score 2/5 (Watch):    {tier2} stocks")
    print(f"  Score 0-1/5:          {total_scored - tier5 - tier4 - tier3 - tier2} stocks")
    print(f"\n  Quality check failed: {quality_failed} stocks (value traps stripped)")
    print(f"  5/5 stocks that are actually traps: {tier5_traps}")

    if tier5 > 0:
        print(f"\n  Top 5/5 stocks:")
        top5 = df[df["score"] == 5].sort_values("pe").head(10)
        for _, row in top5.iterrows():
            print(f"    {row['ticker']:20s} {str(row['name'])[:35]:35s} P/E: {row['pe']}")
    elif tier4 > 0:
        print(f"\n  Top 4/5 stocks:")
        top4 = df[df["score"] == 4].sort_values("pe").head(10)
        for _, row in top4.iterrows():
            print(f"    {row['ticker']:20s} {str(row['name'])[:35]:35s} P/E: {row['pe']}")

    print(f"\nDone. Commit {output_file} to your repo and redeploy.")
    print("The app will read this file directly — no live scanning needed.")


if __name__ == "__main__":
    main()
