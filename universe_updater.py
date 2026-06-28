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
import sys
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import deep_metrics

# --- ADD THIS GLOBAL SESSION BLOCK HERE ---
global_session = requests.Session()
global_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive"
})


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

        csv_url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        resp = session.get(csv_url, timeout=15)
        resp.raise_for_status()

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
    """Fetch all active equity tickers from BSE India."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
        "Referer": "https://www.bseindia.com/",
    })

    tickers = []

    try:
        print("[BSE] Attempting API method...")
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
        print(f"[BSE] API method failed: {e}")

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
def fetch_fundamentals(ticker, retries=3):
    """Fetch all metrics needed for the 4 frameworks. Returns dict or None. Includes backoff."""
    for attempt in range(retries):
        try:
            # 1. CRITICAL: Explicitly pass the global session so Yahoo doesn't block us
            stock = yf.Ticker(ticker, session=global_session)
            info = stock.info
            
            if not info or not info.get("regularMarketPrice"):
                return None, None
            # Compute years listed for Graham 7-year guard
            first_trade = info.get("firstTradeDateEpochUtc")
            if first_trade:
                first_date = datetime.fromtimestamp(first_trade, tz=timezone.utc)
                calc_years_listed = round((datetime.now(tz=timezone.utc) - first_date).days / 365.25, 1)
            else:
                calc_years_listed = None
            pe = info.get("trailingPE")

            data = {
                "ticker": ticker,
                "years_listed": calc_years_listed,
                "name": info.get("longName") or info.get("shortName", ticker),
                "sector": info.get("sector", ""),
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
                "current_ratio": info.get("currentRatio"),
                "beta": info.get("beta"),
                "week52_high": info.get("fiftyTwoWeekHigh"),
                "week52_low": info.get("fiftyTwoWeekLow"),
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
                print(f"[{ticker}] Failed: {e}")
                return None, None
            
    # Failed all retries
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

    # Greenblatt universe-level ranking (requires all stocks)
    deep_metrics.compute_greenblatt_ranks(results)
    return results


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("ALPHACONSENSUS UNIVERSE UPDATER")
    print(f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # ── Step 1: Fetch ticker lists ──
    print("\n--- STEP 1: Fetching ticker lists ---\n")
    nse_tickers = fetch_nse_tickers()
    print()
    bse_tickers = fetch_bse_tickers()
    print()

    if not nse_tickers and not bse_tickers:
        print("ERROR: Could not fetch from either exchange. Check internet connection.")
        return

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

    # ── Existing columns ──
    base_columns = [
        "ticker", "name", "sector", "price", "market_cap", "years_listed",
        "pe", "pb", "roe_pct", "de", "eps",
        "dividend_yield_pct", "profit_margin",
        "current_ratio", "beta",
        "week52_high", "week52_low", "pct_from_high", "pct_from_low",
        "pe_4y_avg", "pe_vs_avg",
        "rev_growth", "ni_growth", "debt_growth",
        "revenue_cagr_3y", "ni_cagr_3y",
        "price_1d_pct", "price_5d_pct", "rsi_14", "vol_spike_flag",
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
        "graham_pass", "greenblatt_pass", "dorsey_pass", "trajectory_pass",
        "lynch_pass", "score",
    ]
    columns = base_columns + deep_columns

    df = pd.DataFrame(scored_results)

    # Only keep columns that exist
    columns = [c for c in columns if c in df.columns]
    df = df[columns].sort_values("ticker").reset_index(drop=True)

    # Add metadata
    df["updated_date"] = datetime.now().strftime("%Y-%m-%d")

    output_file = "universe_scored.csv"
    df.to_csv(output_file, index=False)

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
