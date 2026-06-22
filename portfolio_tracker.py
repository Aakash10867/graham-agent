import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import yfinance as yf
import pandas as pd
from supabase import create_client, Client
from datetime import date
from collections import Counter


def run_daily_tracker():
    print("Initiating DeepMoat Daily Portfolio Audit...")

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY environment variables.")

    supabase: Client = create_client(url, key)

    # ── Load fresh universe CSV (updated in prior workflow step) ──
    universe_df = None
    if os.path.exists("universe_scored.csv"):
        universe_df = pd.read_csv("universe_scored.csv")
        print(f"Loaded universe: {len(universe_df)} stocks")
    else:
        print("Warning: universe_scored.csv not found. Score checks disabled.")

    # ── Fetch all portfolios and holdings ──
    portfolios_resp = supabase.table("portfolios").select("*").execute()
    portfolios = portfolios_resp.data

    if not portfolios:
        print("No portfolios found. Exiting.")
        return

    holdings_resp = supabase.table("holdings").select("*").execute()
    holdings = holdings_resp.data

    # ── Fetch user emails from profiles table ──
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
        # 3. ALERT DETECTION
        # ══════════════════════════════════════

        # ── 3a. Review due ──
        review_date = port.get("next_review_date")
        if review_date:
            try:
                rd = date.fromisoformat(str(review_date))
                if rd <= date.today():
                    all_alerts.append({
                        "portfolio_id": port_id,
                        "user_id": user_id,
                        "alert_type": "review_due",
                        "ticker": "_review",
                        "headline": f"Portfolio review overdue — was due {review_date}",
                        "detail": {"days_overdue": (date.today() - rd).days},
                        "alert_date": today_str,
                    })
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

            # Score dropped by 2+
            if entry_score - current_score >= 2:
                all_alerts.append({
                    "portfolio_id": port_id,
                    "user_id": user_id,
                    "alert_type": "danger",
                    "ticker": ticker,
                    "headline": f"{holding.get('name', ticker)} score dropped {entry_score} -> {current_score}",
                    "detail": {
                        "name": holding.get("name", ticker),
                        "entry_score": entry_score,
                        "current_score": current_score,
                        "reason": "score_drop",
                    },
                    "alert_date": today_str,
                })

            # Quality trap
            if not quality_pass:
                all_alerts.append({
                    "portfolio_id": port_id,
                    "user_id": user_id,
                    "alert_type": "danger",
                    "ticker": ticker,
                    "headline": f"{holding.get('name', ticker)} flagged as potential value trap",
                    "detail": {
                        "name": holding.get("name", ticker),
                        "reason": "quality_fail",
                    },
                    "alert_date": today_str,
                })

            # Price crash > 20%
            if entry_price > 0:
                stock_return = ((live_price - entry_price) / entry_price) * 100
                if stock_return < -20:
                    all_alerts.append({
                        "portfolio_id": port_id,
                        "user_id": user_id,
                        "alert_type": "danger",
                        "ticker": ticker,
                        "headline": f"{holding.get('name', ticker)} down {stock_return:.0f}% from entry",
                        "detail": {
                            "name": holding.get("name", ticker),
                            "reason": "price_crash",
                            "entry_price": entry_price,
                            "current_price": round(live_price, 2),
                            "return_pct": round(stock_return, 1),
                        },
                        "alert_date": today_str,
                    })

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
            all_alerts.append({
                "portfolio_id": port_id,
                "user_id": user_id,
                "alert_type": "opportunity",
                "ticker": opp_row["ticker"],
                "headline": f"{opp_row.get('name', opp_row['ticker'])} hit 4/4 — fits your {investor_type} profile",
                "detail": {
                    "name": str(opp_row.get("name", opp_row["ticker"])),
                    "sector": str(opp_row.get("sector", "N/A")),
                    "price": round(float(opp_row["price"]), 2) if pd.notna(opp_row.get("price")) else 0,
                    "pe": round(float(opp_row["pe"]), 2) if pd.notna(opp_row.get("pe")) else 0,
                    "roe_pct": round(float(opp_row["roe_pct"]), 2) if pd.notna(opp_row.get("roe_pct")) else 0,
                    "score": 4,
                },
                "alert_date": today_str,
            })

    # ══════════════════════════════════════
    # 4. WRITE ALERTS TO SUPABASE
    # ══════════════════════════════════════
    try:
        cutoff = date.today().replace(day=max(1, date.today().day - 7)).isoformat()
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
    # 5. SEND EMAIL ALERTS
    # ══════════════════════════════════════
    smtp_user = os.environ.get("ALERT_EMAIL")
    smtp_pass = os.environ.get("ALERT_EMAIL_PASSWORD")

    if not smtp_user or not smtp_pass:
        print("Email credentials not configured. Skipping email alerts.")
    elif not all_alerts:
        print("No alerts to email.")
    else:
        user_alerts = {}
        for alert in all_alerts:
            uid = alert["user_id"]
            if uid not in user_alerts:
                user_alerts[uid] = []
            user_alerts[uid].append(alert)

        for uid, alerts in user_alerts.items():
            recipient = profiles.get(uid)
            if not recipient:
                print(f"No email found for user {uid}. Skipping.")
                continue

            danger_alerts = [a for a in alerts if a["alert_type"] == "danger"]
            opp_alerts = [a for a in alerts if a["alert_type"] == "opportunity"]
            review_alerts = [a for a in alerts if a["alert_type"] == "review_due"]

            body_parts = ["DeepMoat Daily Alert\n" + "=" * 40 + "\n"]

            if danger_alerts:
                body_parts.append("DANGER ALERTS\n")
                for a in danger_alerts:
                    body_parts.append(f"  >> {a['headline']}\n")
                body_parts.append("")

            if opp_alerts:
                body_parts.append("OPPORTUNITIES\n")
                for a in opp_alerts:
                    d = a.get("detail", {})
                    body_parts.append(
                        f"  >> {a['headline']}\n"
                        f"     Price: {d.get('price', 'N/A')} | "
                        f"P/E: {d.get('pe', 'N/A')} | "
                        f"ROE: {d.get('roe_pct', 'N/A')}%\n"
                    )
                body_parts.append("")

            if review_alerts:
                body_parts.append("REVIEW DUE\n")
                for a in review_alerts:
                    body_parts.append(f"  >> {a['headline']}\n")
                body_parts.append("")

            body_parts.append(
                "--\n"
                "Log in to DeepMoat to take action.\n"
                "This is an automated alert, not financial advice."
            )

            subject = "DeepMoat: "
            parts = []
            if danger_alerts:
                parts.append(f"{len(danger_alerts)} danger")
            if opp_alerts:
                parts.append(f"{len(opp_alerts)} opportunity")
            if review_alerts:
                parts.append("review due")
            subject += ", ".join(parts)

            try:
                msg = MIMEMultipart()
                msg["From"] = smtp_user
                msg["To"] = recipient
                msg["Subject"] = subject
                msg.attach(MIMEText("\n".join(body_parts), "plain"))

                with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                    server.login(smtp_user, smtp_pass)
                    server.send_message(msg)

                print(f"Email sent to {recipient}: {subject}")
            except Exception as e:
                print(f"Email failed for {recipient}: {e}")

    print("DeepMoat Daily Audit Complete.")


if __name__ == "__main__":
    run_daily_tracker()
