"""
WEEKLY MENTOR EMAIL
==================
Runs every Monday at 3:30 UTC (9:00 AM IST) via GitHub Actions.
Reads alerts written by the daily engine, validates, deduplicates,
synthesizes via Gemini into one mentor-style email per user.

Zero alerts are generated here. This script only consumes.

Usage:  python weekly_mentor.py
Cron:   '30 3 * * 1'
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import date, timedelta
from collections import defaultdict

import pandas as pd
import yfinance as yf
from google import genai
from supabase import create_client, Client


APP_URL = "https://kordent.streamlit.app"
GEMINI_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-pro",
    "gemini-3.1-pro-preview",
]


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def fetch_nifty_weekly():
    """Get Nifty 50 weekly return percentage."""
    try:
        hist = yf.Ticker("^NSEI").history(period="7d")
        if len(hist) >= 2:
            return round((float(hist["Close"].iloc[-1]) / float(hist["Close"].iloc[0]) - 1) * 100, 2)
    except Exception as e:
        print(f"Nifty fetch failed: {e}")
    return None


def validate_alerts(alerts, universe_df):
    """
    Re-check each alert against current universe data.
    Drop opportunities/new_entries whose score or quality flipped.
    Portfolio-level alerts (ticker starts with _) pass through.
    """
    if universe_df is None:
        return alerts

    valid = []
    for a in alerts:
        ticker = a.get("ticker", "")
        a_type = a.get("alert_type", "")

        # Portfolio-level alerts — no ticker to validate
        if ticker.startswith("_") or not ticker:
            valid.append(a)
            continue

        row = universe_df[universe_df["ticker"] == ticker]
        if row.empty:
            continue  # vanished from universe — skip silently

        cur_score = int(row["score"].iloc[0]) if pd.notna(row["score"].iloc[0]) else 0
        cur_quality = bool(row["quality_pass"].iloc[0]) if "quality_pass" in row.columns and pd.notna(row["quality_pass"].iloc[0]) else True

        # Opportunities and new entries must still pass quality + score gate
        if a_type in ("opportunity", "new_entry"):
            if cur_score < 3 or not cur_quality:
                continue

        valid.append(a)

    dropped = len(alerts) - len(valid)
    if dropped:
        print(f"  Validation dropped {dropped} stale alerts.")
    return valid


def deduplicate_alerts(alerts):
    """
    Same stock + same alert type across multiple portfolios → keep one.
    Attach list of portfolio_ids it appeared in for context.
    """
    groups = defaultdict(list)
    for a in alerts:
        key = f"{a.get('ticker', '')}:{a.get('alert_type', '')}"
        groups[key].append(a)

    deduped = []
    for key, group in groups.items():
        rep = group[0]  # keep first occurrence
        if len(group) > 1:
            rep["_appeared_in_count"] = len(group)
        deduped.append(rep)

    return deduped


def get_portfolio_summaries(ports, supabase, nifty_weekly_pct):
    """Build a summary dict for each portfolio."""
    cutoff = (date.today() - timedelta(days=7)).isoformat()
    summaries = []

    for p in ports:
        pid = p["id"]
        current_val = float(p.get("current_value") or 0)
        ret_pct = float(p.get("current_return_pct") or 0)
        sip = float(p.get("sip_amount") or 0)
        budget = float(p.get("sip_budget_remaining") or 0)
        opp_budget_total = sip * 0.3

        # Weekly return from portfolio_history
        weekly_ret = None
        try:
            hist = supabase.table("portfolio_history").select("total_value").eq(
                "portfolio_id", pid
            ).gte("date", cutoff).order("date").limit(1).execute()
            if hist.data:
                week_start_val = float(hist.data[0]["total_value"])
                if week_start_val > 0:
                    weekly_ret = round((current_val / week_start_val - 1) * 100, 2)
        except Exception:
            pass

        summaries.append({
            "name": p.get("name", "Unnamed"),
            "id": pid,
            "value": current_val,
            "return_pct": ret_pct,
            "weekly_return_pct": weekly_ret,
            "nifty_weekly_pct": nifty_weekly_pct,
            "sip": sip,
            "budget_remaining": budget,
            "budget_total": opp_budget_total,
            "goal_amount": p.get("target_amount"),
            "goal_date": p.get("target_date"),
            "link": f"{APP_URL}/?portfolio={pid}",
            "is_paper": p.get("is_paper", False),
        })

    return summaries


def build_gemini_prompt(name, summaries, alerts):
    """
    Build the LLM prompt for the weekly email.
    Book passages come from the alerts themselves — no book loading needed.
    """
    # Portfolio summary block
    port_block = ""
    for s in summaries:
        weekly = f"{s['weekly_return_pct']:+.2f}%" if s['weekly_return_pct'] is not None else "N/A"
        nifty = f"{s['nifty_weekly_pct']:+.2f}%" if s['nifty_weekly_pct'] is not None else "N/A"
        goal_line = ""
        if s["goal_amount"]:
            goal_line = f"\n  Goal: ₹{float(s['goal_amount']):,.0f} by {s['goal_date']}"

        budget_line = ""
        if s["sip"] > 0:
            used = s["budget_total"] - s["budget_remaining"]
            budget_line = f"\n  Opportunity budget: ₹{used:,.0f} used of ₹{s['budget_total']:,.0f}"

        _paper_tag = " 👁 (Paper — not yet invested)" if s.get("is_paper") else ""
        port_block += f"""
Portfolio: {s['name']}{_paper_tag}
  Value: ₹{s['value']:,.0f} | Overall return: {s['return_pct']:+.1f}%
  This week: {weekly} (Nifty: {nifty}){goal_line}{budget_line}
  View: {s['link']}
"""

    # Alerts block with book passages
    danger_block = ""
    opp_block = ""
    info_block = ""

    for a in alerts:
        detail = a.get("detail") or {}
        if isinstance(detail, str):
            import json
            try:
                detail = json.loads(detail)
            except Exception:
                detail = {}

        passages = detail.get("book_passages", [])
        passage_text = ""
        if passages:
            passage_text = " | Book context: " + "; ".join(
                f"[{p['author']}] {p['text'][:200]}" for p in passages[:2]
            )

        appeared = ""
        if a.get("_appeared_in_count", 0) > 1:
            appeared = f" (flagged in {a['_appeared_in_count']} portfolios)"

        line = f"- {a['headline']}{appeared}{passage_text}\n"

        a_type = a.get("alert_type", "")
        if a_type in ("danger", "overvalued", "goal_drift"):
            danger_block += line
        elif a_type in ("opportunity", "new_entry"):
            opp_block += line
        else:
            info_block += line

    alerts_block = ""
    if danger_block:
        alerts_block += f"NEEDS ATTENTION:\n{danger_block}\n"
    if opp_block:
        alerts_block += f"OPPORTUNITIES:\n{opp_block}\n"
    if info_block:
        alerts_block += f"ALSO NOTING:\n{info_block}\n"

    if not alerts_block.strip():
        alerts_block = "No significant alerts this week. Quiet weeks are good weeks.\n"

    prompt = f"""You are Kordent's Chief Investment Officer writing a weekly email to {name}.

TODAY: {date.today().strftime('%A, %B %d, %Y')}

PORTFOLIO DATA:
{port_block}

THIS WEEK'S ALERTS:
{alerts_block}

Write a warm, personal weekly email. Rules:

1. Address {name} by name. First line should be a one-sentence summary of their week — not a greeting.
2. Portfolio summary: mention each portfolio's value and weekly change vs Nifty. Be honest — if they underperformed, say so plainly.
3. For each alert, explain WHY it matters using the book context provided. Reference Graham, Greenblatt, or Dorsey naturally — "Graham would say..." not "According to Benjamin Graham's The Intelligent Investor..."
4. If there are opportunities with budget remaining, mention the suggested amount. If budget is used up, say so matter-of-factly.
5. If a goal is set, give a one-line status: on track, behind, or ahead.
6. If a portfolio is marked "Paper", note it's a watchlist portfolio — tracking performance without real money. Be encouraging about what they're learning from the simulation.
7. End with a patience reminder — one sentence, not preachy. Vary it each week.
8. Include the portfolio link(s) so they can take action.
9. Sign off as "Kordent"

TONE: You're a wise friend who happens to be great with money. Not a robot, not a salesperson, not a professor. Use simple language. If you must use a financial term, define it in parentheses.
LENGTH: Under 400 words. Shorter is better.
FORMAT: Plain text. Use emojis sparingly — 2-3 max in the whole email, only where they add warmth.
DO NOT: Use bullet points. Use "Dear". Use "I hope this finds you well". Use "Best regards". Use corporate jargon."""

    return prompt


def call_gemini(prompt, gemini_key):
    """Call Gemini with model fallback. Returns narrative string or None."""
    client = genai.Client(api_key=gemini_key)
    for model in GEMINI_MODELS:
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            if response.text:
                return response.text.strip()
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print(f"  {model} rate-limited, trying next...")
                continue
            print(f"  {model} error: {e}")
            return None
    return None


def build_plain_fallback(name, summaries, alerts):
    """Decent plain-text email when Gemini is unavailable."""
    lines = [f"{name}, here's your week at a glance.\n"]

    for s in summaries:
        weekly = f"{s['weekly_return_pct']:+.2f}%" if s['weekly_return_pct'] is not None else "no data"
        nifty = f"Nifty {s['nifty_weekly_pct']:+.2f}%" if s['nifty_weekly_pct'] is not None else ""
        lines.append(f"📊 {s['name']}: ₹{s['value']:,.0f} ({weekly} this week, {nifty})")
        lines.append(f"   {s['link']}")

        if s["sip"] > 0:
            used = s["budget_total"] - s["budget_remaining"]
            lines.append(f"   Opportunity budget: ₹{s['budget_remaining']:,.0f} remaining of ₹{s['budget_total']:,.0f}")
        lines.append("")

    dangers = [a for a in alerts if a.get("alert_type") in ("danger", "overvalued", "goal_drift")]
    opps = [a for a in alerts if a.get("alert_type") in ("opportunity", "new_entry")]
    others = [a for a in alerts if a.get("alert_type") not in ("danger", "overvalued", "goal_drift", "opportunity", "new_entry")]

    if dangers:
        lines.append("⚠️ Needs your attention:")
        for a in dangers:
            lines.append(f"  {a['headline']}")
        lines.append("")

    if opps:
        lines.append("⚡ Opportunities:")
        for a in opps:
            lines.append(f"  {a['headline']}")
        lines.append("")

    if others:
        lines.append("📋 Also noting:")
        for a in others:
            lines.append(f"  {a['headline']}")
        lines.append("")

    lines.append("Wealth is built in the waiting. See you next week.")
    lines.append("— Kordent")

    return "\n".join(lines)


def build_subject(name, summaries, alerts):
    """Subject line with emotional context — a mentor who feels what you feel."""
    if summaries:
        weighted_ret = 0
        total_weight = 0
        for s in summaries:
            if s.get("weekly_return_pct") is not None and s["value"] > 0:
                weighted_ret += s["weekly_return_pct"] * s["value"]
                total_weight += s["value"]

        if total_weight > 0:
            r = weighted_ret / total_weight
            pct = f"({r:+.1f}%)"

            if r >= 3:
                return f"🟢 {name}, weeks like this are why you stay patient {pct}"
            elif r >= 1:
                return f"🟢 {name}, steady progress — your portfolio grew {pct}"
            elif r >= 0:
                return f"📊 {name}, quiet week {pct} — that's how compounding feels"
            elif r >= -2:
                return f"📊 {name}, a small dip {pct} — nothing to worry about"
            elif r >= -5:
                return f"🟡 {name}, bumpy week {pct} — let's look at what matters"
            else:
                return f"🔴 {name}, tough week {pct} — here's the bigger picture"

    danger_count = sum(1 for a in alerts if a.get("alert_type") in ("danger", "overvalued", "goal_drift"))
    opp_count = sum(1 for a in alerts if a.get("alert_type") in ("opportunity", "new_entry"))

    if danger_count and opp_count:
        return f"📊 {name}, a few things to look at this week"
    elif danger_count:
        return f"📊 {name}, something needs your attention"
    elif opp_count:
        return f"📊 {name}, spotted something interesting for you"

    return f"📊 {name}, your week in a minute"


def send_email(recipient, subject, body, smtp_user, smtp_pass):
    """Send a single plain-text email."""
    try:
        msg = MIMEMultipart()
        msg["From"] = smtp_user
        msg["To"] = recipient
        msg["Subject"] = subject

        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)

        print(f"  ✓ Sent to {recipient}: {subject}")
        return True
    except Exception as e:
        print(f"  ✗ Failed for {recipient}: {e}")
        return False


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def run_weekly_mentor():
    print(f"Kordent Weekly Mentor — {date.today().strftime('%A, %B %d, %Y')}")
    print("=" * 50)

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    smtp_user = os.environ.get("ALERT_EMAIL")
    smtp_pass = os.environ.get("ALERT_EMAIL_PASSWORD")

    if not url or not key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY")
    if not smtp_user or not smtp_pass:
        raise ValueError("Missing ALERT_EMAIL or ALERT_EMAIL_PASSWORD")

    supabase: Client = create_client(url, key)

    # ── Load universe for validation ──
    universe_df = None
    if os.path.exists("universe_scored.csv"):
        universe_df = pd.read_csv("universe_scored.csv")
        print(f"Universe loaded: {len(universe_df)} stocks")
    else:
        print("Warning: universe_scored.csv not found. Skipping validation.")

    # ── Nifty weekly return ──
    nifty_weekly = fetch_nifty_weekly()
    if nifty_weekly is not None:
        print(f"Nifty 50 weekly: {nifty_weekly:+.2f}%")

    # ── Fetch all users ──
    profiles = supabase.table("profiles").select("id, email, full_name").execute().data or []
    if not profiles:
        print("No users. Exiting.")
        return

    user_map = {p["id"]: p for p in profiles}

    # ── Fetch all portfolios ──
    all_portfolios = supabase.table("portfolios").select("*").execute().data or []
    user_portfolios = defaultdict(list)
    for p in all_portfolios:
        user_portfolios[p["user_id"]].append(p)

    # ── Fetch alerts from past 7 days ──
    cutoff = (date.today() - timedelta(days=7)).isoformat()
    all_alerts = supabase.table("portfolio_alerts").select("*").gte(
        "alert_date", cutoff
    ).execute().data or []

    # Group alerts by user (broadcast alerts go to everyone)
    user_alerts = defaultdict(list)
    for a in all_alerts:
        uid = a.get("user_id")
        if uid:
            user_alerts[uid].append(a)
        else:
            for p in profiles:
                user_alerts[p["id"]].append(a)

    print(f"\nProcessing {len(profiles)} users, {len(all_alerts)} raw alerts...\n")

    sent = 0
    skipped = 0

    for uid, profile in user_map.items():
        email = profile.get("email")
        if not email:
            skipped += 1
            continue

        name = profile.get("full_name") or email.split("@")[0].title()
        ports = user_portfolios.get(uid, [])
        raw = user_alerts.get(uid, [])

        # Skip users with no portfolios AND no alerts
        if not ports and not raw:
            skipped += 1
            continue

        print(f"[{name}] {len(ports)} portfolios, {len(raw)} raw alerts")

        # ── Pipeline: validate → deduplicate → cap ──
        validated = validate_alerts(raw, universe_df)
        deduped = deduplicate_alerts(validated)

        # Cap opportunities at 3 total (across all portfolios, post-dedup)
        opps = [a for a in deduped if a.get("alert_type") in ("opportunity", "new_entry")]
        non_opps = [a for a in deduped if a.get("alert_type") not in ("opportunity", "new_entry")]
        opps = opps[:3]
        final_alerts = non_opps + opps

        print(f"  After pipeline: {len(final_alerts)} alerts ({len(validated)} valid, {len(deduped)} deduped)")

        # ── Portfolio summaries ──
        summaries = get_portfolio_summaries(ports, supabase, nifty_weekly)

        # ── Generate email ──
        subject = build_subject(name, summaries, final_alerts)

        body = None
        if gemini_key:
            prompt = build_gemini_prompt(name, summaries, final_alerts)
            body = call_gemini(prompt, gemini_key)
            if body:
                print(f"  LLM narrative generated ({len(body)} chars)")
            else:
                print(f"  LLM failed — using fallback")

        if not body:
            body = build_plain_fallback(name, summaries, final_alerts)

        # ── Send ──
        if send_email(email, subject, body, smtp_user, smtp_pass):
            sent += 1

    print(f"\n{'=' * 50}")
    print(f"Done. Sent: {sent} | Skipped: {skipped}")


if __name__ == "__main__":
    run_weekly_mentor()
