import os
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import yfinance as yf
import pandas as pd
import pymupdf
from google import genai
from supabase import create_client, Client
from datetime import date
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
}


def run_daily_tracker():
    print("Initiating Kordent Daily Portfolio Audit...")

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")

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

    profiles_resp = supabase.table("profiles").select("id, email").execute()
    profiles = {p["id"]: p.get("email", "") for p in profiles_resp.data}

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

        # ── 2. Log history ──
        history_row = {
            "portfolio_id": port_id,
            "date": today_str,
            "total_value": round(current_total_value, 2),
            "daily_return_pct": round(return_pct, 2),
        }
        if nifty_close is not None:
            history_row["nifty_value"] = nifty_close

        supabase.table("portfolio_history").upsert(
            history_row, on_conflict="portfolio_id,date"
        ).execute()

        print(f"Updated [{port['name']}]: Value {current_total_value:,.2f} | Return {return_pct:+.2f}%")

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
            all_alerts.append(make_alert(
                "opportunity", opp_row["ticker"],
                f"{opp_row.get('name', opp_row['ticker'])} hit 4/4 — fits your {investor_type} profile",
                {"name": str(opp_row.get("name", opp_row["ticker"])),
                 "sector": str(opp_row.get("sector", "N/A")),
                 "price": round(float(opp_row["price"]), 2) if pd.notna(opp_row.get("price")) else 0,
                 "pe": round(float(opp_row["pe"]), 2) if pd.notna(opp_row.get("pe")) else 0,
                 "roe_pct": round(float(opp_row["roe_pct"]), 2) if pd.notna(opp_row.get("roe_pct")) else 0,
                 "score": 4},
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

    # ══════════════════════════════════════
    # 3e. SCORE HISTORY TRACKING
    # ══════════════════════════════════════
    if universe_df is not None:
        all_held = set()
        for port in portfolios:
            port_holdings = [h for h in holdings if h["portfolio_id"] == port["id"]]
            for h in port_holdings:
                all_held.add(h["ticker"])

        trackable = universe_df[
            (universe_df["ticker"].isin(all_held)) |
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

        # Clean up rows older than 90 days
        try:
            from datetime import timedelta
            cutoff_90 = (date.today() - timedelta(days=90))
            supabase.table("score_history").delete().lt("date", cutoff_90.isoformat()).execute()
        except Exception as e:
            print(f"Score history cleanup failed: {e}")

        print(f"Logged {written_scores} score history rows. Cleaned >90 days.")
 
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

    # ══════════════════════════════════════
    # 5. LLM-WRITTEN EMAIL ALERTS
    # ══════════════════════════════════════
    smtp_user = os.environ.get("ALERT_EMAIL")
    smtp_pass = os.environ.get("ALERT_EMAIL_PASSWORD")

    if not smtp_user or not smtp_pass:
        print("Email credentials not configured. Skipping.")
    elif not all_alerts:
        print("No alerts to email.")
    elif not gemini_key:
        print("GEMINI_API_KEY not set. Sending plain email fallback.")
        _send_plain_emails(all_alerts, profiles, smtp_user, smtp_pass)
    else:
        _send_llm_emails(all_alerts, profiles, smtp_user, smtp_pass, gemini_key)

    print("Kordent Daily Audit Complete.")


def _build_llm_email(alerts, gemini_key):
    """Use Gemini to write a cohesive, book-grounded email from raw alerts."""
    client = genai.Client(api_key=gemini_key)

    # Build context block with alerts and their book passages
    alerts_block = ""
    for a in alerts:
        detail = a.get("detail", {})
        passages = detail.get("book_passages", [])
        passage_text = ""
        if passages:
            passage_text = "\n".join(
                f"  [{p['author']}]: {p['text']}" for p in passages
            )

        alerts_block += f"""
ALERT: {a['headline']}
Type: {a['alert_type']} | Ticker: {a.get('ticker', 'N/A')}
Data: {', '.join(f'{k}={v}' for k, v in detail.items() if k != 'book_passages')}
Book context:
{passage_text}
---
"""

    prompt = f"""You are Kordent's Chief Investment Analyst writing a daily email alert.

Today's date: {date.today().strftime('%B %d, %Y')}

ALERTS TO COVER:
{alerts_block}

Write a professional, concise investment note email. Rules:
1. Open with a one-line summary of the day's findings
2. Group by priority: dangers first, then opportunities, then review reminders
3. For each alert, explain what happened AND what the investment books say about it
4. Reference specific authors and concepts naturally (e.g., "Graham's margin of safety principle suggests..." or "Dorsey would flag this as potential moat erosion because...")
5. End with a clear action summary: what the investor should consider doing
6. Keep the entire email under 500 words
7. Use plain text formatting, no HTML or markdown
8. Sign off as "Kordent Daily Intelligence"

Do NOT be generic. Use the actual book passages provided to make specific, grounded recommendations."""

    models = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"]
    for model in models:
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            return response.text
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                continue
            print(f"Gemini error: {e}")
            return None
    return None


def _send_llm_emails(all_alerts, profiles, smtp_user, smtp_pass, gemini_key):
    """Group alerts by user, generate LLM narrative, send email."""
    user_alerts = {}
    for alert in all_alerts:
        uid = alert["user_id"]
        if uid not in user_alerts:
            user_alerts[uid] = []
        user_alerts[uid].append(alert)

    for uid, alerts in user_alerts.items():
        recipient = profiles.get(uid)
        if not recipient:
            print(f"No email for user {uid}. Skipping.")
            continue

        # Generate narrative via Gemini
        narrative = _build_llm_email(alerts, gemini_key)

        if not narrative:
            print(f"LLM failed for {recipient}. Falling back to plain.")
            _send_single_plain_email(alerts, recipient, smtp_user, smtp_pass)
            continue

        # Subject line
        danger_count = sum(1 for a in alerts if a["alert_type"] == "danger")
        opp_count = sum(1 for a in alerts if a["alert_type"] == "opportunity")
        parts = []
        if danger_count:
            parts.append(f"{danger_count} danger")
        if opp_count:
            parts.append(f"{opp_count} opportunity")
        if any(a["alert_type"] == "review_due" for a in alerts):
            parts.append("review due")
        subject = f"Kordent Daily: {', '.join(parts)}" if parts else "Kordent Daily Intelligence"

        try:
            msg = MIMEMultipart()
            msg["From"] = smtp_user
            msg["To"] = recipient
            msg["Subject"] = subject
            msg.attach(MIMEText(narrative, "plain"))

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(smtp_user, smtp_pass)
                server.send_message(msg)

            print(f"LLM email sent to {recipient}: {subject}")
        except Exception as e:
            print(f"Email failed for {recipient}: {e}")


def _send_plain_emails(all_alerts, profiles, smtp_user, smtp_pass):
    """Fallback: plain text emails without LLM."""
    user_alerts = {}
    for alert in all_alerts:
        uid = alert["user_id"]
        if uid not in user_alerts:
            user_alerts[uid] = []
        user_alerts[uid].append(alert)

    for uid, alerts in user_alerts.items():
        recipient = profiles.get(uid)
        if not recipient:
            continue
        _send_single_plain_email(alerts, recipient, smtp_user, smtp_pass)


def _send_single_plain_email(alerts, recipient, smtp_user, smtp_pass):
    """Send a single plain text email for one user."""
    danger_alerts = [a for a in alerts if a["alert_type"] == "danger"]
    opp_alerts = [a for a in alerts if a["alert_type"] == "opportunity"]
    review_alerts = [a for a in alerts if a["alert_type"] == "review_due"]

    body_parts = [f"Kordent Daily Alert — {date.today().strftime('%B %d, %Y')}\n{'=' * 40}\n"]

    if danger_alerts:
        body_parts.append("DANGER ALERTS\n")
        for a in danger_alerts:
            body_parts.append(f"  >> {a['headline']}\n")
        body_parts.append("")

    if opp_alerts:
        body_parts.append("OPPORTUNITIES\n")
        for a in opp_alerts:
            d = a.get("detail", {})
            body_parts.append(f"  >> {a['headline']}\n")
        body_parts.append("")

    if review_alerts:
        body_parts.append("REVIEW DUE\n")
        for a in review_alerts:
            body_parts.append(f"  >> {a['headline']}\n")
        body_parts.append("")

    body_parts.append("--\nLog in to Kordent to take action.\nNot financial advice.")

    parts = []
    if danger_alerts:
        parts.append(f"{len(danger_alerts)} danger")
    if opp_alerts:
        parts.append(f"{len(opp_alerts)} opportunity")
    if review_alerts:
        parts.append("review due")
    subject = f"Kordent: {', '.join(parts)}" if parts else "Kordent Daily"

    try:
        msg = MIMEMultipart()
        msg["From"] = smtp_user
        msg["To"] = recipient
        msg["Subject"] = subject
        msg.attach(MIMEText("\n".join(body_parts), "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)

        print(f"Plain email sent to {recipient}: {subject}")
    except Exception as e:
        print(f"Email failed for {recipient}: {e}")


if __name__ == "__main__":
    run_daily_tracker()
