"""
BACKTEST RUNNER
===============
Sprint 6, Phase 4 — Retrospective simulation.

Uses current yfinance 4Y financials to reconstruct framework scores at
quarterly decision dates, then tracks portfolio performance vs Nifty 50.

Universe: Nifty 200 constituents (current list as proxy)
Decision dates: Quarterly (Jan/Apr/Jul/Oct), from earliest usable date
Publication lag: 6 months (Indian FY ends March; results published ~Sep)
Strategy: Top 15 stocks by composite score, equal-weight, quarterly rebalance
Sell rule: Score < 2 OR quality_pass flipped
Benchmark: Nifty 50 buy-and-hold
Starting capital: ₹10,00,000

Honest framing: This is a RETROSPECTIVE reconstruction using current financials.
It tells us whether the deep metrics would have been useful, not what we would
have actually done (survivorship bias from using current Nifty 200 list).

Output: backtest_results.csv
"""

import os
import sys
import math
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import deep_metrics

# ─── Config ───
STARTING_CAPITAL = 10_00_000  # ₹10 lakh
TOP_N = 15
MIN_SCORE_TO_HOLD = 2
NIFTY50_TICKER = "^NSEI"
NIFTY200_URL = "https://archives.nseindia.com/content/indices/ind_nifty200list.csv"

# Quarterly decision dates (1st trading day of each quarter)
# We go back as far as our 4Y financials allow with 6-month lag
DECISION_MONTHS = [1, 4, 7, 10]

global_session = requests.Session()
global_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "*/*",
})


# ─── Nifty 200 Fetcher ───

def fetch_nifty200_tickers():
    """Fetch current Nifty 200 constituent tickers."""
    tickers = []

    # Try NSE CSV
    try:
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "text/html,application/xhtml+xml",
        })
        session.get("https://www.nseindia.com", timeout=10)
        time.sleep(1)
        resp = session.get(NIFTY200_URL, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(pd.io.common.StringIO(resp.text))
        symbol_col = [c for c in df.columns if "symbol" in c.lower()]
        if symbol_col:
            for sym in df[symbol_col[0]]:
                tickers.append(f"{sym.strip()}.NS")
        print(f"[NIFTY200] Fetched {len(tickers)} tickers from NSE")
    except Exception as e:
        print(f"[NIFTY200] CSV fetch failed: {e}")

    if not tickers:
        # Fallback: use universe_scored.csv if available, take top 200 by market cap
        try:
            df = pd.read_csv("universe_scored.csv")
            df = df.dropna(subset=["market_cap"]).sort_values("market_cap", ascending=False)
            tickers = df["ticker"].head(200).tolist()
            print(f"[NIFTY200] Fallback: top {len(tickers)} from universe_scored.csv by market cap")
        except Exception:
            print("[NIFTY200] ERROR: No source for Nifty 200 tickers")
            sys.exit(1)

    return tickers


# ─── Financial Data Fetcher ───

def fetch_stock_data(ticker):
    """
    Fetch 4Y financials + 4Y price history for a single stock.
    Returns dict with raw financial columns per fiscal year + price series.
    """
    try:
        stock = yf.Ticker(ticker, session=global_session)
        info = stock.info or {}

        if not info.get("regularMarketPrice"):
            return None

        income_stmt = stock.financials
        balance_sheet = stock.balance_sheet
        cashflow = stock.cashflow

        if income_stmt is None or income_stmt.empty:
            return None

        # Get price history (4+ years)
        hist = stock.history(period="5y")
        if hist is None or hist.empty or len(hist) < 250:
            return None

        is_cols = sorted(income_stmt.columns)
        bs_cols = sorted(balance_sheet.columns) if balance_sheet is not None and not balance_sheet.empty else []
        cf_cols = sorted(cashflow.columns) if cashflow is not None and not cashflow.empty else []

        # Build per-fiscal-year snapshots
        fy_data = {}
        for i, col in enumerate(is_cols):
            fy_end = col  # Timestamp of fiscal year end
            fy_key = fy_end.strftime("%Y-%m")

            snapshot = {"ticker": ticker, "fy_end": fy_end, "sector": info.get("sector", "")}
            shares = info.get("sharesOutstanding") or 1

            # Income statement
            for row, key in [("Total Revenue", "revenue"), ("Net Income", "net_income"),
                             ("EBIT", "ebit"), ("Operating Income", "operating_income")]:
                val = deep_metrics._bs_row(income_stmt, [row], col)
                snapshot[key] = val

            # Balance sheet (match by closest date)
            if bs_cols:
                # Find the BS column closest to this FY end
                bs_col = min(bs_cols, key=lambda x: abs((x - col).days))
                for row, key in [("Total Assets", "total_assets"),
                                 ("Stockholders Equity", "equity"),
                                 ("Total Debt", "total_debt"),
                                 ("Current Assets", "current_assets"),
                                 ("Current Liabilities", "current_liabilities"),
                                 ("Goodwill", "goodwill")]:
                    val = deep_metrics._bs_row(balance_sheet, [row, f"Total {row}"], bs_col)
                    snapshot[key] = val

            # Cash flow
            if cf_cols:
                cf_col = min(cf_cols, key=lambda x: abs((x - col).days))
                for row, key in [("Operating Cash Flow", "ocf"),
                                 ("Capital Expenditure", "capex")]:
                    val = deep_metrics._bs_row(cashflow,
                        [row, "Total Cash From Operating Activities",
                         "Cash Flow From Continuing Operating Activities",
                         "Purchase Of PPE"], cf_col)
                    snapshot[key] = val

            snapshot["shares"] = shares
            snapshot["dividend_yield"] = info.get("dividendYield")
            snapshot["book_value"] = info.get("bookValue")
            fy_data[fy_key] = snapshot

        return {
            "ticker": ticker,
            "info": info,
            "fy_data": fy_data,
            "price_history": hist["Close"],
            "fy_dates": sorted(fy_data.keys()),
        }

    except Exception as e:
        return None


# ─── Score Reconstruction ───

def reconstruct_score(stock_data, decision_date, available_fy_key):
    """
    Reconstruct deep metrics for a stock at a specific decision date
    using the fiscal year data that would have been available.
    """
    ticker = stock_data["ticker"]
    info = stock_data["info"]
    fy = stock_data["fy_data"].get(available_fy_key)

    if fy is None:
        return None

    # Get price at decision date
    prices = stock_data["price_history"]
    # Find closest trading day to decision date
    target = pd.Timestamp(decision_date)
    mask = prices.index <= target
    if mask.sum() == 0:
        return None
    price_at_date = float(prices[mask].iloc[-1])

    shares = fy.get("shares") or 1
    revenue = fy.get("revenue")
    ni = fy.get("net_income")
    ebit = fy.get("ebit") or fy.get("operating_income")
    total_assets = fy.get("total_assets")
    equity = fy.get("equity")
    total_debt = fy.get("total_debt") or 0
    ocf = fy.get("ocf")
    capex = fy.get("capex")
    ca = fy.get("current_assets")
    cl = fy.get("current_liabilities")

    if ni is None or revenue is None:
        return None

    eps = ni / shares if shares > 0 else 0
    bvps = equity / shares if equity and shares > 0 else None
    pe = price_at_date / eps if eps > 0 else None
    pb = price_at_date / bvps if bvps and bvps > 0 else None
    roe = ni / equity if equity and equity > 0 else None
    de = (total_debt / equity * 100) if equity and equity > 0 else None
    market_cap = price_at_date * shares

    # Compute key metrics for scoring
    result = {
        "ticker": ticker,
        "decision_date": decision_date.strftime("%Y-%m-%d"),
        "price": price_at_date,
        "sector": fy.get("sector", ""),
        "market_cap": market_cap,
        "pe": pe,
        "pb": pb,
        "roe": roe,
        "de": de,
        "eps": eps,
        "revenue": revenue,
        "net_income": ni,
        "profit_margin": ni / revenue if revenue and revenue > 0 else None,
        "current_ratio": (ca / cl) if ca and cl and cl > 0 else None,
        "dividend_yield": fy.get("dividend_yield"),
    }

    # ── Simplified scoring for backtest (key metrics only) ──
    score = 0

    # Graham: PE ≤ 15, PB ≤ 1.5, PE×PB ≤ 22.5
    graham_points = 0
    if revenue and revenue >= 2_000_000_000: graham_points += 1
    if result["current_ratio"] and result["current_ratio"] >= 2.0: graham_points += 1
    if ni and ni > 0: graham_points += 1  # Earnings positive
    if pe and 0 < pe <= 15: graham_points += 1
    if pb and 0 < pb <= 1.5: graham_points += 1
    if pe and pb and 0 < pe * pb <= 22.5: graham_points += 1
    result["graham_defensive_score"] = graham_points
    if graham_points >= 4: score += 1

    # Greenblatt: EBIT/EV and ROIC
    if ebit and ebit > 0 and market_cap > 0:
        ev = market_cap + total_debt - (fy.get("total_cash") or 0)
        if ev and ev > 0:
            result["greenblatt_ey"] = ebit / ev * 100
            if ca and cl:
                tangible_cap = (ca - cl) + (total_assets - ca if total_assets else 0) - (fy.get("goodwill") or 0)
                if tangible_cap and tangible_cap > 0:
                    result["greenblatt_roic"] = ebit / tangible_cap * 100
    # Greenblatt pass is rank-based; simplified here to top 30% EY
    if result.get("greenblatt_ey") and result["greenblatt_ey"] > 10:
        score += 1

    # Dorsey+Buffett: ROE > 15%, margin > 10%, low debt
    db_points = 0
    if roe and roe > 0.15: db_points += 1
    if result["profit_margin"] and result["profit_margin"] > 0.10: db_points += 1
    if de is not None and de < 100: db_points += 1
    if ocf and ni and ni > 0 and ocf / ni > 0.8: db_points += 1
    if ocf and capex:
        fcf = ocf + capex
        if revenue and revenue > 0 and fcf / revenue > 0.05: db_points += 1
    result["dorsey_buffett_score"] = db_points
    if db_points >= 3: score += 1

    # Trajectory: growth
    if ni > 0 and revenue > 0:
        score += 1  # Simplified: positive earnings = trajectory pass for backtest

    # Lynch: PEG
    lynch_points = 0
    ni_cagr = None  # Can't compute multi-year CAGR from single FY snapshot
    # Use basic growth heuristics
    if pe and 0 < pe < 20: lynch_points += 2
    if de is not None and de <= 33: lynch_points += 2
    if ni and ni > 0: lynch_points += 1
    result["lynch_score"] = lynch_points
    if lynch_points >= 4: score += 1

    # Quality gate (simplified)
    quality = True
    if ocf and ni and ni > 0:
        accruals = (ni - ocf) / total_assets if total_assets and total_assets > 0 else None
        if accruals is not None and accruals > 0.10: quality = False
        cfo_ni = ocf / ni
        if cfo_ni < 0.5: quality = False
    result["quality_pass"] = quality
    result["score"] = score

    return result


# ─── Backtest Engine ───

def run_backtest(all_stock_data):
    """Run the quarterly rebalancing backtest."""

    # Determine decision dates
    # Start from Jan 2023 (need FY2022 data, published ~Sep 2022, 6-month lag OK by Jan 2023)
    today = datetime.now()
    decision_dates = []
    for year in range(2023, today.year + 1):
        for month in DECISION_MONTHS:
            dt = datetime(year, month, 1)
            if dt < today - timedelta(days=30):  # At least 1 month of forward data needed
                decision_dates.append(dt)

    print(f"\n[BACKTEST] {len(decision_dates)} decision dates: {decision_dates[0].strftime('%b %Y')} → {decision_dates[-1].strftime('%b %Y')}")

    # Map decision dates to available fiscal year
    # Indian FY ends March. Results published by Sep. With 6-month lag:
    # Jan 2023 → use FY ending ≤ Jul 2022 → FY Mar 2022
    # Oct 2023 → use FY ending ≤ Apr 2023 → FY Mar 2023
    def get_usable_fy(decision_dt):
        cutoff = decision_dt - timedelta(days=180)  # 6 month lag
        # Find latest FY end before cutoff
        # Most Indian companies: March FY
        for fy_year in range(decision_dt.year, decision_dt.year - 5, -1):
            fy_end = datetime(fy_year, 3, 31)
            if fy_end <= cutoff:
                return fy_end.strftime("%Y-%m")
        return None

    # Track portfolio
    portfolio = {}  # ticker → {shares, buy_price}
    capital = STARTING_CAPITAL
    portfolio_values = []  # [{date, portfolio_value, benchmark_value}]

    # Get Nifty 50 prices for benchmark
    print("[BACKTEST] Fetching Nifty 50 benchmark...")
    try:
        nifty = yf.Ticker(NIFTY50_TICKER, session=global_session)
        nifty_hist = nifty.history(period="5y")["Close"]
    except Exception as e:
        print(f"[BACKTEST] Nifty 50 fetch failed: {e}")
        nifty_hist = pd.Series(dtype=float)

    nifty_start = None
    all_trades = []

    for dd_idx, decision_date in enumerate(decision_dates):
        fy_key = get_usable_fy(decision_date)
        if not fy_key:
            continue

        print(f"\n[{decision_date.strftime('%b %Y')}] Using FY {fy_key} data")

        # Score all stocks at this decision date
        scored = []
        for sd in all_stock_data:
            if sd is None:
                continue
            # Check if this stock has the needed FY data
            if fy_key not in sd.get("fy_data", {}):
                # Try adjacent FY
                available = sd.get("fy_dates", [])
                usable = [k for k in available if k <= fy_key]
                if usable:
                    fy_key_use = usable[-1]
                else:
                    continue
            else:
                fy_key_use = fy_key

            result = reconstruct_score(sd, decision_date, fy_key_use)
            if result and result.get("quality_pass") and result.get("score", 0) > 0:
                scored.append(result)

        # Rank and select top N
        scored.sort(key=lambda x: x.get("score", 0), reverse=True)
        # Among equal scores, prefer lower PE
        scored.sort(key=lambda x: (-(x.get("score", 0)), x.get("pe") or 999))
        picks = scored[:TOP_N]

        if not picks:
            print(f"  No qualifying stocks at {decision_date.strftime('%Y-%m-%d')}")
            continue

        print(f"  Qualified: {len(scored)} stocks | Selected top {len(picks)}")
        print(f"  Score distribution: " +
              ", ".join(f"{s}/5: {sum(1 for p in picks if p['score']==s)}" for s in range(5, 0, -1)))

        # ── Rebalance ──
        # Sell everything, buy new picks equal-weight
        # First, compute current portfolio value
        if portfolio:
            # Get prices at this decision date
            port_val = 0
            for t, holding in portfolio.items():
                sd_match = next((sd for sd in all_stock_data if sd and sd["ticker"] == t), None)
                if sd_match:
                    prices = sd_match["price_history"]
                    target = pd.Timestamp(decision_date)
                    mask = prices.index <= target
                    if mask.sum() > 0:
                        current_price = float(prices[mask].iloc[-1])
                        port_val += holding["shares"] * current_price

            if port_val > 0:
                capital = port_val
                all_trades.append({
                    "date": decision_date.strftime("%Y-%m-%d"),
                    "action": "REBALANCE",
                    "portfolio_value": round(capital, 2),
                })

        # Buy new portfolio
        per_stock = capital / len(picks)
        portfolio = {}
        for pick in picks:
            price = pick["price"]
            if price and price > 0:
                num_shares = per_stock / price
                portfolio[pick["ticker"]] = {
                    "shares": num_shares,
                    "buy_price": price,
                    "score": pick["score"],
                }
                all_trades.append({
                    "date": decision_date.strftime("%Y-%m-%d"),
                    "action": "BUY",
                    "ticker": pick["ticker"],
                    "price": round(price, 2),
                    "score": pick["score"],
                    "amount": round(per_stock, 2),
                })

        # Record portfolio value and benchmark
        if not nifty_hist.empty:
            target = pd.Timestamp(decision_date)
            mask = nifty_hist.index <= target
            if mask.sum() > 0:
                nifty_price = float(nifty_hist[mask].iloc[-1])
                if nifty_start is None:
                    nifty_start = nifty_price
                benchmark_val = STARTING_CAPITAL * (nifty_price / nifty_start)
            else:
                benchmark_val = STARTING_CAPITAL
        else:
            benchmark_val = STARTING_CAPITAL

        portfolio_values.append({
            "date": decision_date.strftime("%Y-%m-%d"),
            "portfolio_value": round(capital, 2),
            "benchmark_value": round(benchmark_val, 2),
            "num_holdings": len(portfolio),
        })

    # ── Final valuation (today) ──
    if portfolio:
        final_val = 0
        for t, holding in portfolio.items():
            sd_match = next((sd for sd in all_stock_data if sd and sd["ticker"] == t), None)
            if sd_match:
                prices = sd_match["price_history"]
                if not prices.empty:
                    current_price = float(prices.iloc[-1])
                    final_val += holding["shares"] * current_price

        if final_val > 0:
            capital = final_val

        if not nifty_hist.empty and nifty_start:
            benchmark_final = STARTING_CAPITAL * (float(nifty_hist.iloc[-1]) / nifty_start)
        else:
            benchmark_final = STARTING_CAPITAL

        portfolio_values.append({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "portfolio_value": round(capital, 2),
            "benchmark_value": round(benchmark_final, 2),
            "num_holdings": len(portfolio),
        })

    return portfolio_values, all_trades


# ─── Main ───

def main():
    print("=" * 60)
    print("KORDENT BACKTEST RUNNER")
    print(f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    print(f"\nStarting capital: ₹{STARTING_CAPITAL:,.0f}")
    print(f"Strategy: Top {TOP_N} by composite score, equal-weight, quarterly rebalance")
    print(f"Benchmark: Nifty 50 buy-and-hold")
    print(f"Publication lag: 6 months (no look-ahead bias)")
    print()

    # Step 1: Get Nifty 200 tickers
    print("--- STEP 1: Fetching Nifty 200 constituents ---")
    tickers = fetch_nifty200_tickers()

    # Step 2: Fetch all stock data
    print(f"\n--- STEP 2: Fetching financial data for {len(tickers)} stocks ---")
    all_stock_data = []
    completed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(fetch_stock_data, t): t for t in tickers}
        for future in as_completed(futures):
            completed += 1
            try:
                result = future.result()
                if result:
                    all_stock_data.append(result)
                else:
                    failed += 1
            except Exception:
                failed += 1

            if completed % 20 == 0:
                print(f"  [{completed}/{len(tickers)}] Valid: {len(all_stock_data)} | Failed: {failed}")

    print(f"\n  Total usable: {len(all_stock_data)} stocks")

    # Step 3: Run backtest
    print("\n--- STEP 3: Running quarterly backtest ---")
    portfolio_values, all_trades = run_backtest(all_stock_data)

    # Step 4: Save results
    print("\n--- STEP 4: Saving results ---")

    if portfolio_values:
        pv_df = pd.DataFrame(portfolio_values)
        pv_df.to_csv("backtest_results.csv", index=False)
        print(f"Saved {len(pv_df)} data points to backtest_results.csv")

        trades_df = pd.DataFrame(all_trades)
        trades_df.to_csv("backtest_trades.csv", index=False)
        print(f"Saved {len(trades_df)} trades to backtest_trades.csv")

        # Summary stats
        start_val = STARTING_CAPITAL
        end_val = pv_df["portfolio_value"].iloc[-1]
        bench_end = pv_df["benchmark_value"].iloc[-1]
        start_date = pv_df["date"].iloc[0]
        end_date = pv_df["date"].iloc[-1]

        years = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365.25
        if years > 0:
            strategy_cagr = ((end_val / start_val) ** (1 / years) - 1) * 100
            benchmark_cagr = ((bench_end / start_val) ** (1 / years) - 1) * 100
            alpha = strategy_cagr - benchmark_cagr
        else:
            strategy_cagr = benchmark_cagr = alpha = 0

        total_return = (end_val / start_val - 1) * 100
        bench_return = (bench_end / start_val - 1) * 100

        print(f"\n{'='*50}")
        print(f"BACKTEST RESULTS ({start_date} → {end_date})")
        print(f"{'='*50}")
        print(f"  Strategy:  ₹{start_val:>12,.0f} → ₹{end_val:>12,.0f}  ({total_return:+.1f}%)")
        print(f"  Nifty 50:  ₹{start_val:>12,.0f} → ₹{bench_end:>12,.0f}  ({bench_return:+.1f}%)")
        print(f"  CAGR:      Strategy {strategy_cagr:.1f}% | Benchmark {benchmark_cagr:.1f}%")
        print(f"  Alpha:     {alpha:+.1f}% annualized")
        print(f"  Period:    {years:.1f} years")
        print(f"\n  ⚠ RETROSPECTIVE RECONSTRUCTION — not a live track record.")
        print(f"  Uses current Nifty 200 list (survivorship bias present).")
        print(f"  Live prospective tracking begins with score_history.")
    else:
        print("  No backtest results generated.")

    print("\nDone.")


if __name__ == "__main__":
    main()
