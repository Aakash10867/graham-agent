import os
import re
import yfinance as yf
import pandas as pd
import pymupdf
from google import genai
from supabase import create_client, Client
from datetime import date, timedelta
from collections import Counter


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
        nifty = yf.Ticker("^NSEI")
        hist = nifty.history(period="5d")
        if not hist.empty:
            nifty_close = round(float(hist["Close"].iloc[-1]), 2)
            print(f"Nifty 50 close: {nifty_close:,.2f}")
    except Exception as e:
        print(f"Warning: Could not fetch Nifty 50: {e}")

    # ── Fetch Nifty BeES for shadow portfolio ──
    nifty_bees_price = None
    try:
        _bees = yf.Ticker("NIFTYBEES.NS")
        _bees_hist = _bees.history(period="5d")
        if not _bees_hist.empty:
            nifty_bees_price = round(float(_bees_hist["Close"].iloc[-1]), 2)
            print(f"Nifty BeES close: {nifty_bees_price:,.2f}")
    except Exception as e:
        print(f"Warning: Could not fetch Nifty BeES: {e}")

    # ── Fetch all transactions once ──
    txn_resp = supabase.table("sip_transactions").select("portfolio_id, amount_inr, transaction_type, nifty_units").execute()
    all_txns = txn_resp.data or []

    # ── Bootstrap: create genesis transactions for portfolios with holdings but no transactions ──
    _txn_port_ids = set(t["portfolio_id"] for t in all_txns)
    holdings_resp_all = supabase.table("holdings").select("portfolio_id, ticker, shares, price_at_entry, created_at").execute()
    _all_h = holdings_resp_all.data or []
    _ports_needing_bootstrap = set()
    for h in _all_h:
        if h["portfolio_id"] not in _txn_port_ids:
            _ports_needing_bootstrap.add(h["portfolio_id"])
    if _ports_needing_bootstrap:
        print(f"Bootstrapping {len(_ports_needing_bootstrap)} portfolios with genesis transactions...")
        for h in _all_h:
            if h["portfolio_id"] in _ports_needing_bootstrap:
                _h_shares = float(h.get("shares") or 0)
                _h_price = float(h.get("price_at_entry") or 0)
                _h_amt = round(_h_shares * _h_price, 2)
                _h_date = (h.get("created_at") or today_str)[:10]
                _nifty_u = round(_h_amt / nifty_bees_price, 6) if nifty_bees_price and _h_amt > 0 else None
                _port_user = next((p["user_id"] for p in portfolios if p["id"] == h["portfolio_id"]), None)
                if _port_user and _h_amt > 0:
                    supabase.table("sip_transactions").insert({
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
                    }).execute()
        # Refresh transactions after bootstrap
        txn_resp = supabase.table("sip_transactions").select("portfolio_id, amount_inr, transaction_type, nifty_units").execute()
        all_txns = txn_resp.data or []
        print("Bootstrap complete.")

    price_cache = {}
    today_str = date.today().isoformat()
    all_alerts = []

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
        supabase.table("portfolios").update({
            "current_value": round(current_total_value, 2),
            "current_return_pct": round(return_pct, 2)
        }).eq("id", port_id).execute()

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

        nifty_shadow = round(total_nifty_units * nifty_bees_price, 2) if nifty_bees_price and total_nifty_units > 0 else None

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
            history_row, on_conflict="portfolio_id,date"
        ).execute()

        print(f"Updated [{port['name']}]: Value {current_total_value:,.2f} | Return {return_pct:+.2f}%")

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
            supabase.table("portfolios").update({
                "sip_budget_remaining": round(sip_budget, 2),
                "sip_budget_reset_date": today_str,
            }).eq("id", port_id).execute()
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

        # ── 3a. Review due ──
        review_date = port.get("next_review_date")
        if review_date:
            try:
                rd = date.fromisoformat(str(review_date))
                if rd <= date.today():
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
                all_alerts.append(make_alert(
                    "danger", ticker,
                    f"{holding.get('name', ticker)} score dropped {entry_score} -> {current_score}",
                    {"name": holding.get("name", ticker), "entry_score": entry_score,
                     "current_score": current_score, "reason": "score_drop"},
                    "score_drop"
                ))

            if not quality_pass:
                all_alerts.append(make_alert(
                    "danger", ticker,
                    f"{holding.get('name', ticker)} flagged as potential value trap",
                    {"name": holding.get("name", ticker), "reason": "quality_fail"},
                    "quality_fail"
                ))

            if entry_price > 0:
                stock_return = ((live_price - entry_price) / entry_price) * 100
                if stock_return < -20:
                    all_alerts.append(make_alert(
                        "danger", ticker,
                        f"{holding.get('name', ticker)} down {stock_return:.0f}% from entry",
                        {"name": holding.get("name", ticker), "reason": "price_crash",
                         "entry_price": entry_price, "current_price": round(live_price, 2),
                         "return_pct": round(stock_return, 1)},
                        "price_crash"
                    ))

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
            (universe_df["score"] == 4) &
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

            all_alerts.append(make_alert(
                "opportunity", opp_row["ticker"],
                f"{opp_row.get('name', opp_row['ticker'])} hit 4/4 — fits your {investor_type} profile",
                {"name": str(opp_row.get("name", opp_row["ticker"])),
                 "sector": str(opp_row.get("sector", "N/A")),
                 "price": stock_price,
                 "pe": round(float(opp_row["pe"]), 2) if pd.notna(opp_row.get("pe")) else 0,
                 "roe_pct": round(float(opp_row["roe_pct"]), 2) if pd.notna(opp_row.get("roe_pct")) else 0,
                 "score": 4,
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
                        f"👁 {wl_name} score improved {prev_score} → {cur_score}/4",
                        {"prev_score": prev_score, "current_score": cur_score, "source": "watchlist"},
                        "watchlist_score_up"
                    ))
                    wl_alert_count += 1

                # Score down
                if cur_score is not None and prev_score is not None and cur_score < prev_score:
                    all_alerts.append(wl_alert(
                        "watchlist_score_down",
                        f"👁 {wl_name} score dropped {prev_score} → {cur_score}/4",
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
                    batch, on_conflict="ticker,date"
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
                            "headline": f"{nr.get('name', nr['ticker'])} new to radar at score {int(nr['score'])}/4",
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
        from datetime import timedelta
        cutoff = (date.today() - timedelta(days=7)).isoformat()
        supabase.table("portfolio_alerts").delete().lt("alert_date", cutoff).eq("is_read", False).execute()
    except Exception as e:
        print(f"Warning: Could not clean old alerts: {e}")

    written = 0
    for alert in all_alerts:
        try:
            supabase.table("portfolio_alerts").upsert(
                alert, on_conflict="portfolio_id,ticker,alert_type,alert_date"
            ).execute()
            written += 1
        except Exception as e:
            print(f"Alert write failed: {e}")

    print(f"Wrote {written} alerts.")

    print("Kordent Daily Audit Complete.")

if __name__ == "__main__":
    run_daily_tracker()
