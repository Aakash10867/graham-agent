"""
Kordent
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
import math

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

def enrich_holdings_live(holdings, cache_key=None):
    """Compute live allocation_pct from current market prices.

    Overwrites the stored allocation_pct with:
        (shares × current_price) / total_portfolio_value × 100
    Adds 'current_price' and 'current_value' to each holding dict.
    Caches prices in session state for 5 min to avoid re-fetching on Streamlit reruns.
    """
    import time as _time

    price_cache = {}
    if cache_key:
        cached = st.session_state.get(f"_price_cache_{cache_key}")
        if cached and _time.time() - cached.get("_ts", 0) < 300:
            price_cache = dict(cached)
            price_cache.pop("_ts", None)

    for h in holdings:
        t = h.get("ticker", "")
        if t and t not in price_cache:
            try:
                price_cache[t] = yf.Ticker(t).fast_info.last_price or h.get("price_at_entry", 0)
            except Exception:
                price_cache[t] = h.get("price_at_entry", 0)

    if cache_key:
        st.session_state[f"_price_cache_{cache_key}"] = {**price_cache, "_ts": _time.time()}

    enriched = []
    for h in holdings:
        h = dict(h)  # shallow copy — don't mutate originals
        t = h.get("ticker", "")
        h["current_price"] = round(price_cache.get(t, h.get("price_at_entry", 0)), 2)
        h["current_value"] = round(h.get("shares", 0) * h["current_price"], 2)
        enriched.append(h)

    total_val = sum(h["current_value"] for h in enriched)
    for h in enriched:
        h["allocation_pct"] = round(h["current_value"] / total_val * 100, 1) if total_val > 0 else 0

    return enriched

def generate_portfolio_pdf(portfolio, holdings, history_data=None, alerts=None, chart_buf=None, narrative=None):
    """Generate a professional PDF report for a portfolio."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    import io

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=20*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    story = []

    # Custom styles
    brand_blue = colors.HexColor("#1D4ED8")
    light_gray = colors.HexColor("#F3F4F6")
    dark_text = colors.HexColor("#111827")

    title_style = ParagraphStyle("PDFTitle", parent=styles["Title"], fontSize=22,
                                  textColor=brand_blue, spaceAfter=4)
    heading_style = ParagraphStyle("PDFHeading", parent=styles["Heading2"], fontSize=13,
                                    textColor=brand_blue, spaceBefore=14, spaceAfter=6)
    body_style = ParagraphStyle("PDFBody", parent=styles["Normal"], fontSize=10,
                                 textColor=dark_text, leading=14)
    small_style = ParagraphStyle("PDFSmall", parent=styles["Normal"], fontSize=8,
                                  textColor=colors.HexColor("#6B7280"), leading=10)

    # ── Header ──
    story.append(Paragraph("Kordent", title_style))
    story.append(Paragraph("Portfolio Report", body_style))
    story.append(Spacer(1, 4*mm))

    today_str = datetime.date.today().strftime("%B %d, %Y")
    story.append(Paragraph(f"Generated: {today_str}", small_style))
    story.append(Spacer(1, 6*mm))

    # ── Portfolio Summary ──
    story.append(Paragraph("Portfolio Summary", heading_style))

    sip = portfolio.get("sip_amount", 0)
    current_val = portfolio.get("current_value", 0)
    return_pct = portfolio.get("current_return_pct", 0)

    summary_data = [
        ["Name", portfolio.get("name", "—")],
        ["Investor Type", portfolio.get("investor_type", "—")],
        ["Time Horizon", portfolio.get("time_horizon", "—")],
        ["Monthly SIP", f"Rs. {sip:,.0f}"],
        ["Current Value", f"Rs. {current_val:,.2f}"],
        ["Return", f"{return_pct:+.2f}%"],
        ["Review Frequency", f"Every {portfolio.get('review_freq', 90)} days"],
        ["Next Review", portfolio.get("next_review_date", "—")],
    ]

    summary_table = Table(summary_data, colWidths=[45*mm, 110*mm])
    summary_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), brand_blue),
        ("TEXTCOLOR", (1, 0), (1, -1), dark_text),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 6*mm))

    # ── Holdings Table ──
    if holdings:
        story.append(Paragraph("Holdings", heading_style))

        header = ["Stock", "Ticker", "Shares", "Entry Price", "Invested", "Alloc %"]
        rows = [header]
        total_invested = 0

        cell_style = ParagraphStyle("CellWrap", parent=styles["Normal"], fontSize=9,
                                     textColor=dark_text, leading=11)
        for h in holdings:
            invested = h.get("sip_amount_inr", 0)
            total_invested += invested
            rows.append([
                Paragraph(h.get("name", "—"), cell_style),
                Paragraph(h.get("ticker", "—"), cell_style),
                str(h.get("shares", 0)),
                f"Rs. {h.get('price_at_entry', 0):,.2f}",
                f"Rs. {invested:,.0f}",
                f"{h.get('allocation_pct', 0)}%",
            ])

        rows.append(["", "", "", "", f"Rs. {total_invested:,.0f}", ""])

        col_widths = [48*mm, 30*mm, 15*mm, 25*mm, 25*mm, 17*mm]
        holdings_table = Table(rows, colWidths=col_widths)
        holdings_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 0), (-1, 0), brand_blue),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("TEXTCOLOR", (0, 1), (-1, -1), dark_text),
            ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, light_gray]),
            ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
            ("LINEABOVE", (0, -1), (-1, -1), 1, brand_blue),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E5E7EB")),
        ]))
        story.append(holdings_table)
        story.append(Spacer(1, 4*mm))

        # P&L note
        if current_val > 0 and total_invested > 0:
            pnl = current_val - total_invested
            pnl_str = f"Rs. {pnl:+,.2f}"
            story.append(Paragraph(
                f"Total Invested: Rs. {total_invested:,.0f} | "
                f"Current Value: Rs. {current_val:,.2f} | "
                f"P&amp;L: {pnl_str} ({return_pct:+.2f}%)", body_style
            ))
            story.append(Spacer(1, 4*mm))

    # ── Growth Chart ──
    if chart_buf:
        from reportlab.platypus import Image as RLImage
        story.append(Paragraph("Growth vs Market", heading_style))
        chart_buf.seek(0)
        chart_img = RLImage(chart_buf, width=160*mm, height=70*mm)
        story.append(chart_img)
        story.append(Spacer(1, 4*mm))

    # ── Performance vs Nifty ──
    if history_data and len(history_data) >= 2:
        story.append(Paragraph("Performance", heading_style))
        first = history_data[0]
        last = history_data[-1]
        port_growth = ((last["total_value"] - first["total_value"]) / first["total_value"]) * 100 if first["total_value"] > 0 else 0
        days = (datetime.date.fromisoformat(str(last["date"])) - datetime.date.fromisoformat(str(first["date"]))).days

        perf_text = f"Portfolio grew {port_growth:+.1f}% over {days} days."

        if first.get("nifty_value") and last.get("nifty_value"):
            nifty_growth = ((last["nifty_value"] - first["nifty_value"]) / first["nifty_value"]) * 100
            alpha = port_growth - nifty_growth
            perf_text += f" Nifty 50: {nifty_growth:+.1f}%. Alpha: {alpha:+.1f}%."

        story.append(Paragraph(perf_text, body_style))
        story.append(Spacer(1, 4*mm))

    # ── Active Alerts ──
    if alerts:
        story.append(Paragraph("Active Alerts", heading_style))
        for a in alerts:
            icon = {"danger": "DANGER", "opportunity": "OPPORTUNITY", "review_due": "REVIEW DUE"}.get(a["alert_type"], "ALERT")
            story.append(Paragraph(f"[{icon}] {a['headline']}", body_style))
        story.append(Spacer(1, 4*mm))

    # ── Investment Analysis ──
    if narrative:
        from reportlab.platypus import PageBreak, HRFlowable
        story.append(PageBreak())
        story.append(Paragraph("Investment Analysis", heading_style))
        story.append(Spacer(1, 2*mm))

        section_style = ParagraphStyle("SectionHead", parent=body_style,
                                        fontName="Helvetica-Bold", fontSize=12,
                                        textColor=brand_blue, spaceBefore=10, spaceAfter=4)
        stock_style = ParagraphStyle("StockHead", parent=body_style,
                                      fontName="Helvetica-Bold", fontSize=10,
                                      textColor=dark_text, spaceBefore=8, spaceAfter=2)
        label_style = ParagraphStyle("LabelLine", parent=body_style, fontSize=9.5,
                                      textColor=dark_text, leading=13, leftIndent=8)

        for line in narrative.split("\n"):
            line = line.strip()
            if not line:
                story.append(Spacer(1, 2*mm))
                continue

            safe_line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

            # Numbered section headers like "1. PORTFOLIO THESIS"
            if line.isupper() and len(line) > 3 and not line.startswith("Selection") and not line.startswith("Strength") and not line.startswith("Risk"):
                story.append(HRFlowable(width="40%", thickness=0.5, color=brand_blue, spaceAfter=4, spaceBefore=6))
                story.append(Paragraph(safe_line, section_style))
            # Stock name headers (ALL CAPS with sector in parens)
            elif line.isupper() and "(" in line:
                story.append(Paragraph(safe_line, stock_style))
            # Labeled lines: Selection:, Strength:, Risk:
            elif line.startswith("Selection:") or line.startswith("Strength:") or line.startswith("Risk:"):
                label, _, rest = line.partition(":")
                rest = rest.strip().replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                story.append(Paragraph(f"<b>{label}:</b> {rest}", label_style))
            else:
                story.append(Paragraph(safe_line, body_style))

        story.append(Spacer(1, 6*mm))

    # ── Disclaimer ──
    story.append(Spacer(1, 10*mm))
    story.append(Paragraph(
        "This report is auto-generated by Kordent for informational purposes only. "
        "It does not constitute financial advice. Past performance does not guarantee future results. "
        "Consult a qualified financial advisor before making investment decisions.",
        small_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def generate_portfolio_chart(history_data):
    """Render portfolio growth chart as PNG bytes for PDF embedding."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import io

    if not history_data or len(history_data) < 2:
        return None

    df = pd.DataFrame(history_data)
    df["date"] = pd.to_datetime(df["date"])

    # Normalize to % return from day 1
    port_base = df["total_value"].iloc[0]
    df["portfolio_pct"] = ((df["total_value"] / port_base) - 1) * 100 if port_base > 0 else 0

    has_nifty = "nifty_value" in df.columns and df["nifty_value"].notna().sum() >= 2
    if has_nifty:
        nifty_base = df["nifty_value"].dropna().iloc[0]
        df["nifty_pct"] = ((df["nifty_value"] / nifty_base) - 1) * 100 if nifty_base > 0 else 0

    fig, ax = plt.subplots(figsize=(6.5, 2.8), dpi=150)

    ax.plot(df["date"], df["portfolio_pct"], color="#1D4ED8", linewidth=2, label="Portfolio")
    if has_nifty:
        ax.plot(df["date"], df["nifty_pct"], color="#9CA3AF", linewidth=1.5, linestyle="--", label="Nifty 50")

    ax.set_ylabel("Return %", fontsize=9, color="#374151")
    ax.tick_params(labelsize=8, colors="#6B7280")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.axhline(y=0, color="#E5E7EB", linewidth=0.8)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#E5E7EB")
    ax.spines["bottom"].set_color("#E5E7EB")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_portfolio_narrative(portfolio, holdings, collection):
    """Generate LLM-written investment thesis using book RAG."""
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

    # ── Gather book passages for each holding ──
    stock_contexts = []
    for h in holdings:
        sector = h.get("sector", "unknown")
        name = h.get("name", h.get("ticker", ""))
        query = f"{sector} stock investment analysis valuation"

        try:
            passages = collection.query(query_texts=[query], n_results=3)
            book_text = "\n".join(passages["documents"][0][:2])  # top 2 passages
        except Exception:
            book_text = ""

        stock_contexts.append({
            "name": name,
            "ticker": h.get("ticker"),
            "sector": sector,
            "entry_price": h.get("price_at_entry"),
            "shares": h.get("shares"),
            "allocation_pct": h.get("allocation_pct"),
            "invested": h.get("sip_amount_inr"),
            "score": h.get("score_at_entry"),
            "book_passages": book_text,
        })

    # ── Portfolio-level book passages ──
    portfolio_type = portfolio.get("investor_type", "balanced")
    try:
        port_passages = collection.query(
            query_texts=[f"{portfolio_type} investor portfolio construction diversification sector allocation"],
            n_results=3
        )
        portfolio_book_text = "\n".join(port_passages["documents"][0][:2])
    except Exception:
        portfolio_book_text = ""

    # ── Sector distribution ──
    from collections import Counter
    sector_dist = Counter(h.get("sector", "Unknown") for h in holdings)
    sector_summary = ", ".join(f"{s}: {c} stocks" for s, c in sector_dist.most_common())

    # ── Build prompt ──
    stocks_block = ""
    for s in stock_contexts:
        stocks_block += f"""
--- {s['name']} ({s['ticker']}) ---
Sector: {s['sector']} | Entry: Rs.{s['entry_price']} | Shares: {s['shares']} | Allocation: {s['allocation_pct']}% | Score: {s['score']}/4
Relevant book passages:
{s['book_passages'][:800]}
"""

    prompt = f"""You are Kordent's Chief Investment Analyst writing a portfolio report.
Portfolio: {portfolio.get('name')} | Type: {portfolio_type} | Horizon: {portfolio.get('time_horizon')} | SIP: Rs.{portfolio.get('sip_amount', 0):,}/month
Sector distribution: {sector_summary}

HOLDINGS:
{stocks_block}

PORTFOLIO-LEVEL BOOK CONTEXT:
{portfolio_book_text[:800]}

Write a professional investment analysis with these sections:

1. PORTFOLIO THESIS (2-3 sentences: what this portfolio is designed to do)

2. STOCK-BY-STOCK ANALYSIS (for each stock):
   - Why it was selected (connect to its score and the investment frameworks)
   - Key strength (grounded in book principles — cite which book/concept)
   - Key risk (be honest about what could go wrong — cite relevant book warnings)

3. PORTFOLIO-LEVEL ASSESSMENT:
   - Sector concentration risk (is it too concentrated? what do the books say?)
   - Diversification quality
   - Alignment with the stated investor type and time horizon
   - One specific improvement recommendation grounded in the books

Be direct. No filler. Use Rs. for currency.

FORMAT RULES (follow exactly):
- Section headers: write them in ALL CAPS on their own line (e.g., PORTFOLIO THESIS)
- Stock names: write as "STOCK NAME (Sector)" in ALL CAPS on their own line
- Under each stock, write exactly three labeled lines:
  Selection: why it was picked
  Strength: key advantage with book reference
  Risk: key danger with book reference
- Leave a blank line between sections
- No markdown, no bullets, no asterisks, no dashes"""

    try:
        for model_name in FREE_MODELS:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                return response.text
            except Exception as e:
                error_msg = str(e).upper()
                if any(err in error_msg for err in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500"]):
                    continue
                raise e
        return "Analysis unavailable — all models rate-limited."
    except Exception as e:
        return f"Analysis unavailable: {e}"

def generate_health_check(portfolio, holdings, universe_df, collection):
    """Portfolio-level diagnostic: diversification, risk, valuation, book-grounded assessment."""
    if not holdings:
        return None

    # ── Look up each holding in universe CSV ──
    enriched = []
    sectors = []
    betas = []
    pe_vs_avgs = []
    pct_from_highs = []
    scores = []

    for h in holdings:
        ticker = h.get("ticker", "")
        alloc = h.get("allocation_pct", 0)
        row = universe_df[universe_df["ticker"] == ticker]

        sector = h.get("sector", "Unknown")
        beta = None
        pe_vs_avg = None
        pct_from_high = None
        score = h.get("score_at_entry", 0)

        if not row.empty:
            r = row.iloc[0]
            sector = r.get("sector", sector) if pd.notna(r.get("sector")) else sector
            beta = round(float(r["beta"]), 2) if pd.notna(r.get("beta")) else None
            pe_vs_avg = round(float(r["pe_vs_avg"]), 2) if pd.notna(r.get("pe_vs_avg")) else None
            pct_from_high = round(float(r["pct_from_high"]), 2) if pd.notna(r.get("pct_from_high")) else None
            score = int(r["score"]) if pd.notna(r.get("score")) else score

        sectors.append(sector)
        if beta is not None:
            betas.append((beta, alloc))
        if pe_vs_avg is not None:
            pe_vs_avgs.append(pe_vs_avg)
        if pct_from_high is not None:
            pct_from_highs.append(pct_from_high)
        scores.append(score)

        enriched.append({
            "name": h.get("name", ticker), "ticker": ticker, "sector": sector,
            "alloc": alloc, "beta": beta, "pe_vs_avg": pe_vs_avg,
            "pct_from_high": pct_from_high, "score": score,
        })

    # ── Sector concentration (HHI) ──
    from collections import Counter
    sector_counts = Counter(sectors)
    total = len(sectors)
    sector_weights = {s: c / total for s, c in sector_counts.items()}
    hhi = sum(w ** 2 for w in sector_weights.values())
    diversification_score = round((1 - hhi) * 100)

    # ── Weighted average beta ──
    if betas:
        total_alloc = sum(a for _, a in betas)
        avg_beta = round(sum(b * a for b, a in betas) / total_alloc, 2) if total_alloc > 0 else None
    else:
        avg_beta = None

    # ── Valuation positioning ──
    avg_pe_vs_avg = round(sum(pe_vs_avgs) / len(pe_vs_avgs), 1) if pe_vs_avgs else None
    avg_pct_from_high = round(sum(pct_from_highs) / len(pct_from_highs), 1) if pct_from_highs else None

    # ── Quality distribution ──
    score_dist = Counter(scores)

    # ── Concentration warnings ──
    warnings = []
    for sector, weight in sector_weights.items():
        if weight > 0.3:
            warnings.append(f"{sector} is {weight*100:.0f}% of portfolio (>30%)")
    if avg_beta and avg_beta > 1.3:
        warnings.append(f"High portfolio beta ({avg_beta}) — amplifies market swings")
    if avg_pe_vs_avg and avg_pe_vs_avg > 20:
        warnings.append(f"Holdings trading {avg_pe_vs_avg}% above their historical PE — possible overvaluation")

    metrics = {
        "diversification_score": diversification_score,
        "sector_distribution": dict(sector_counts),
        "avg_beta": avg_beta,
        "avg_pe_vs_historical": avg_pe_vs_avg,
        "avg_pct_from_52w_high": avg_pct_from_high,
        "score_distribution": dict(score_dist),
        "warnings": warnings,
        "holdings_detail": enriched,
    }

    # ── LLM narrative ──
    investor_type = portfolio.get("investor_type", "balanced")
    time_horizon = portfolio.get("time_horizon", "medium")

    # Book passages for portfolio construction
    try:
        passages = collection.query(
            query_texts=[
                f"{investor_type} portfolio construction sector diversification concentration risk",
                "margin of safety portfolio level risk management number of holdings",
            ],
            n_results=3
        )
        book_text = "\n".join(passages["documents"][0][:2]) if passages["documents"] else ""
    except Exception:
        book_text = ""

    holdings_summary = "\n".join(
        f"  {e['name']} ({e['ticker']}) — Sector: {e['sector']}, Alloc: {e['alloc']}%, "
        f"Beta: {e['beta'] or 'N/A'}, PE vs Avg: {e['pe_vs_avg'] or 'N/A'}%, "
        f"From 52w High: {e['pct_from_high'] or 'N/A'}%, Score: {e['score']}/4"
        for e in enriched
    )
    # ── Find complementary stocks from universe ──
    complement_candidates = []
    try:
        overweight_sectors = [s for s, w in sector_weights.items() if w > 0.25]
        candidates = universe_df[
            (universe_df["score"] >= 3) &
            (universe_df["quality_pass"] == True) &
            (~universe_df["ticker"].isin([h.get("ticker") for h in holdings])) &
            (universe_df["pe"] > 0) &
            (pd.notna(universe_df["pe"]))
        ].copy()

        # Prefer sectors not already overweight
        if overweight_sectors:
            underweight = candidates[~candidates["sector"].isin(overweight_sectors)]
            if len(underweight) >= 5:
                candidates = underweight

        # Top 5 by score then lowest PE
        candidates = candidates.sort_values(["score", "pe"], ascending=[False, True]).head(5)

        for _, r in candidates.iterrows():
            complement_candidates.append({
                "ticker": r["ticker"],
                "name": str(r.get("name", r["ticker"])),
                "sector": str(r.get("sector", "N/A")),
                "score": int(r["score"]),
                "pe": round(float(r["pe"]), 2) if pd.notna(r.get("pe")) else None,
                "roe_pct": round(float(r["roe_pct"]), 2) if pd.notna(r.get("roe_pct")) else None,
                "pct_from_high": round(float(r["pct_from_high"]), 2) if pd.notna(r.get("pct_from_high")) else None,
                "price": round(float(r["price"]), 2) if pd.notna(r.get("price")) else None,
            })
    except Exception:
        pass

    # ── User Decision Context ──
    user_context = ""
    _profile = portfolio.get("portfolio_profile") or {}
    if isinstance(_profile, str):
        try: _profile = json.loads(_profile)
        except: _profile = {}
    if _profile.get("decision_context"):
        user_context = f"\nUSER DECISION CONTEXT:\n{_profile.get('decision_context')}\n(CRITICAL: Do not penalize the portfolio for risks or sector concentrations that the user explicitly accepted during portfolio creation.)\n"

    prompt = f"""You are Kordent's Chief Risk Officer diagnosing a portfolio's health.

Portfolio: {portfolio.get('name')} | Type: {investor_type} | Horizon: {time_horizon}
Holdings: {total} stocks
{user_context}
    prompt = f"""You are Kordent's Chief Risk Officer diagnosing a portfolio's health.

Portfolio: {portfolio.get('name')} | Type: {investor_type} | Horizon: {time_horizon}
Holdings: {total} stocks

PORTFOLIO METRICS:
Diversification Score: {diversification_score}/100 (based on sector HHI)
Sector Distribution: {dict(sector_counts)}
Average Beta: {avg_beta or 'N/A'}
Average PE vs Historical Average: {avg_pe_vs_avg or 'N/A'}% (negative = discount, positive = premium)
Average Distance from 52-Week High: {avg_pct_from_high or 'N/A'}%
Score Distribution: {dict(score_dist)}
Warnings: {warnings if warnings else 'None'}

HOLDINGS:
{holdings_summary}

BOOK CONTEXT:
{book_text[:800]}

Write a diagnostic with these sections:
1. VERDICT (one line: is this portfolio healthy, needs attention, or at risk?)
2. STRENGTHS (what is working well — cite book principles)
3. RISKS (what could go wrong — cite book warnings, be specific about which holdings)
4. ACTION ITEMS — ONLY if the portfolio has real problems. If diversification score is above 80, no sector exceeds 30%, and quality scores are 3+, then state "Portfolio is well-constructed. No changes recommended." and output ACTIONS_JSON: []. Do NOT recommend changes just to have something to say. A good portfolio deserves acknowledgment, not perpetual tinkering. Graham explicitly warns against excessive trading and over-optimization.

Be direct and specific. Reference actual holdings by name. Under 300 words total.

COMPLEMENTARY CANDIDATES (stocks not in portfolio that could improve diversification):
{chr(10).join(f"  {c['ticker']} — {c['name']} | Sector: {c['sector']} | Score: {c['score']}/4 | PE: {c['pe']} | ROE: {c['roe_pct']}%" for c in complement_candidates) if complement_candidates else "None available"}
If the portfolio needs more holdings or sector diversity, recommend specific stocks from the candidates above in your ACTION ITEMS.

After the narrative, on a new line, output a JSON block starting with ACTIONS_JSON: followed by a JSON array.
Each action object must have:
- "type": one of "sell", "reduce", "investigate"
- "ticker": the stock ticker
- "reason": one line explanation
- For "reduce": include "target_alloc_pct" (the new target allocation percentage)
- For "sell": include "shares" (number of shares to sell, 0 means all)
- For "add": include "name", "sector", "score", "pe", and "suggested_alloc_pct"

Example: ACTIONS_JSON: [{{"type": "reduce", "ticker": "CHENNPETRO.NS", "target_alloc_pct": 10, "reason": "Trim cyclical concentration"}}, {{"type": "add", "ticker": "HDFCBANK.NS", "name": "HDFC Bank", "sector": "Financial Services", "score": 4, "pe": 18.5, "suggested_alloc_pct": 10, "reason": "Adds financial sector exposure, improves diversification"}}]

Only include actions for holdings that need changes. Do not include "investigate" for more than 2 stocks."""

    narrative = None
    try:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        last_good = st.session_state.get("last_working_model")
        models = [last_good] + [m for m in FREE_MODELS if m != last_good] if last_good else FREE_MODELS
        for model in models:
            try:
                response = client.models.generate_content(model=model, contents=prompt)
                raw_text = response.text
                # Parse structured actions from LLM response
                actions = []
                narrative = raw_text
                if "ACTIONS_JSON:" in raw_text:
                    parts = raw_text.split("ACTIONS_JSON:", 1)
                    narrative = parts[0].strip()
                    try:
                        actions_str = parts[1].strip()
                        # Handle markdown code fences
                        actions_str = actions_str.replace("```json", "").replace("```", "").strip()
                        actions = json.loads(actions_str)
                    except Exception:
                        actions = []
                st.session_state.last_working_model = model
                break
            except Exception as e:
                error_msg = str(e).upper()
                if any(err in error_msg for err in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500"]):
                    continue
                break
    except Exception:
        pass

    metrics["narrative"] = narrative
    metrics["actions"] = actions if 'actions' in dir() else []
    metrics["complement_candidates"] = complement_candidates
    return metrics


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
        live_red = any("RED FLAG" in f for f in quality_flags) if isinstance(quality_flags, list) else False
        
        # CSV quality_pass is deterministic — computed monthly, doesn't shift between calls
        csv_red = False
        if len(urow) and "quality_pass" in urow.columns:
            qp_val = urow["quality_pass"].iloc[0]
            if pd.notna(qp_val) and qp_val == False:
                csv_red = True
                if not live_red:
                    quality_flags.append(
                        "RED FLAG: Monthly pre-screen flagged quality failure "
                        "(live check returned clean — yfinance data inconsistency)."
                    )
        
        has_red_flags = live_red or csv_red

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


def generate_review_recommendations(enriched_holdings, investor_type, time_horizon, portfolio):
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

    user_context = ""
    _profile = portfolio.get("portfolio_profile") or {}
    if isinstance(_profile, str):
        try: _profile = json.loads(_profile)
        except: _profile = {}
    if _profile.get("decision_context"):
        user_context = f"\nUSER DECISION CONTEXT:\n{_profile.get('decision_context')}\n(CRITICAL: Honor these preferences. Do not recommend selling a stock solely for a trait the user explicitly accepted, such as sector volatility.)\n"

    review_prompt = (
        f"You are the Kordent Investment Committee reviewing a {investor_type} investor's "
        f"portfolio with a {time_horizon}-term horizon.\n\n"
        f"{user_context}\n"
    
    review_prompt = (
        f"You are the Kordent Investment Committee reviewing a {investor_type} investor's "
        f"portfolio with a {time_horizon}-term horizon.\n\n"
        f"For each stock below, provide a recommendation.\n\n"
        f"DECISION FRAMEWORK (apply in order):\n"
        f"1. RED FLAGS OVERRIDE: If earnings quality has RED FLAGS, recommend SELL ALL. Cite Graham on value traps.\n"
        f"2. MOAT EROSION: If ROE declined for 3+ years AND stock underperformed market, recommend SELL HALF. Cite Dorsey.\n"
        f"3. MARKET EFFECT: If stock dropped BUT Nifty also dropped similarly (within 5%), recommend HOLD. "
        f"Cite Graham on Mr. Market. The business hasn't changed.\n"
        f"4. NO THESIS: If current score = 0 (no framework passes), recommend SELL ALL regardless of "
        f"other signals. A score of 0 means no investment thesis exists.\n"
        f"5. THESIS INTACT: If score >= 1 AND score stable or improved AND no red flags AND cash conversion > 0.5, "
        f"recommend HOLD or BUY MORE. Cite the relevant framework.\n"
        f"6. OVERVALUATION: If stock gained >30% and score dropped, recommend HOLD but note reduced margin of safety.\n"
        f"7. INVESTOR PROFILE: "
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
        except json.JSONDecodeError:
            continue
        except Exception as e:
            error_msg = str(e).upper()
            if any(err in error_msg for err in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500"]):
                continue
            break
    return None



def register_portfolio(portfolio_name: str, investor_type: str, sip_amount: int, time_horizon: str, review_days: int = 90, stocks_json: str = "[]", portfolio_profile: str = "{}", target_amount: float = 0, target_date: str = "", decision_context: str = "") -> dict:
    """Register a finalized SIP portfolio so the user can save it to their account.
    Call this ONLY after you have presented the final portfolio table with all stocks and allocations.

    Args:
        ... (keep existing args) ...
        decision_context: A brief summary of any specific preferences, trade-offs, or choices the user made during the Phase 1 clarification questions (e.g., 'User explicitly accepted volatility in Basic Materials for deep value'). Pass an empty string if no questions were asked.
    """

    Args:
        portfolio_name: Short descriptive name, e.g. 'Conservative Growth SIP - June 2026'
        investor_type: The investor profile - defensive, balanced, or enterprising
        sip_amount: Monthly SIP amount in INR
        time_horizon: Investment time horizon from the builder profile
        review_days: Number of days between portfolio reviews from the builder profile.
        stocks_json: A JSON string representing a list of stock objects. Each object must have keys: ticker (str), name (str), sector (str), allocation_pct (number).
        portfolio_profile: JSON string of the full investor profile from the builder form. Pass through from the [BUILDER_PROFILE] message.
        target_amount: Savings goal in INR. 0 if no goal set.
        target_date: Goal deadline as ISO date string (YYYY-MM-DD). Empty string if no goal.
    """
    try:
        stocks = json.loads(stocks_json) if isinstance(stocks_json, str) else stocks_json
    except json.JSONDecodeError:
        return {"error": f"Could not parse stocks_json: {stocks_json[:200]}"}

    if not stocks:
        return {"error": "No stocks provided."}

    # --- FIX: Bypass LLM amnesia by pulling directly from secure session state ---
    _profile = st.session_state.get("builder_profile") or {}
    
    # INJECT CONVERSATION CONTEXT
    if decision_context:
        _profile["decision_context"] = decision_context
    
    final_target = _profile.get("target_amount") or (target_amount if target_amount > 0 else None)
    final_date = _profile.get("target_date") or (target_date if target_date else None)

    st.session_state.pending_portfolio = {
        "name": portfolio_name,
        "investor_type": investor_type,
        "sip_amount": sip_amount,
        "time_horizon": time_horizon,
        "review_days": int(review_days),
        "stocks": stocks,
        "portfolio_profile": _profile if _profile else None,
        "target_amount": final_target,
        "target_date": final_date,
        "is_paper": _profile.get("is_paper", False),
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

def fuzzy_search_universe(query: str, df, max_results: int = 6):
    """
    Fuzzy-match a user query against universe_df name + ticker columns.
    Returns list of {ticker, name, match_score, score, quality_pass} sorted desc.
    """
    from difflib import SequenceMatcher

    if df is None or df.empty:
        return []

    q = query.lower().strip()
    if not q:
        return []

    noise = {
        # pronouns / determiners
        "i", "me", "my", "you", "your", "we", "our", "it", "its", "a", "an",
        "the", "this", "that", "these", "those", "any", "some", "all", "each",
        # verbs / auxiliaries
        "is", "are", "was", "were", "be", "been", "am", "do", "does", "did",
        "will", "would", "could", "should", "shall", "may", "might", "can",
        "have", "has", "had", "get", "got", "make", "let", "go", "going",
        # common action verbs in finance queries
        "buy", "sell", "hold", "invest", "investing", "invested", "analyse",
        "analyze", "analysis", "review", "check", "show", "tell", "give",
        "find", "look", "looking", "think", "want", "need", "know", "see",
        "compare", "pick", "choose", "suggest", "recommend", "evaluate",
        # prepositions / conjunctions / adverbs
        "in", "on", "at", "to", "of", "for", "with", "from", "by", "about",
        "into", "between", "through", "after", "before", "up", "down", "out",
        "and", "or", "but", "not", "nor", "so", "if", "then", "than", "also",
        "just", "only", "very", "really", "how", "what", "why", "when", "where",
        "which", "who", "whom", "whose", "whether", "now", "still", "yet",
        # finance generic words
        "stock", "stocks", "share", "shares", "company", "companies", "price",
        "market", "investment", "portfolio", "sector", "industry", "worth",
        "good", "bad", "best", "worst", "top", "right", "safe", "risky",
        "value", "valued", "undervalued", "overvalued", "growth", "income",
        "dividend", "return", "returns", "profit", "loss", "money", "rupee",
        "long", "short", "term", "today", "currently", "recent", "recently",
        # filler
        "please", "thanks", "hey", "hi", "hello", "ok", "okay",
    }
    q_words = [w for w in q.split() if w not in noise]
    q_clean = " ".join(q_words).strip()

    if not q_clean:
        return []

    candidates = []
    for _, row in df.iterrows():
        ticker = str(row.get("ticker", ""))
        name = str(row.get("name", ""))
        t_bare = ticker.lower().replace(".ns", "").replace(".bo", "")
        n_lower = name.lower()

        score = 0.0

        # 1. Exact ticker match
        if q_clean == t_bare or q_clean == ticker.lower():
            score = 1.0
        # 2. Ticker appears as a word in cleaned query
        elif t_bare in q_words:
            score = 0.95
        # 3. Full company name is substring of cleaned query
        elif n_lower in q_clean:
            score = 0.92
        # 4. Cleaned query is substring of company name (min 3 chars)
        elif len(q_clean) >= 3 and q_clean in n_lower:
            score = 0.85
        # 5. All cleaned query words appear in company name
        elif len(q_words) >= 2 and all(w in n_lower for w in q_words):
            score = 0.80
        else:
            # 6. Fuzzy match via SequenceMatcher
            if len(q_clean) >= 3:
                best = max(
                    SequenceMatcher(None, q_clean, n_lower).ratio(),
                    SequenceMatcher(None, q_clean, t_bare).ratio()
                )
                if best > 0.55:
                    score = best * 0.75

        if score > 0.4:
            candidates.append({
                "ticker": ticker,
                "name": name,
                "match_score": round(score, 3),
                "score": int(row["score"]) if pd.notna(row.get("score")) else 0,
                "quality_pass": bool(row["quality_pass"]) if pd.notna(row.get("quality_pass")) else False,
            })

    candidates.sort(key=lambda x: x["match_score"], reverse=True)
    return candidates[:max_results]


# ══════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="Kordent",
    page_icon="logo.svg",
    layout="centered",
)


if "sb_view_mode" not in st.session_state:
    st.session_state.sb_view_mode = "chat"

# ══════════════════════════════════════════════
# CRITICAL INITIALIZATION (Prevents Crashes)
# ══════════════════════════════════════════════
default_state = {
    "sb_view_mode": "chat",
    "messages": [],
    "chat_history": [],
    "sb_access_token": None,
    "sb_refresh_token": None,
    "sb_user_email": None,
    "sb_user_id": None,
    "pending_portfolio": None,
    "pending_retry": None,
    "pending_disambiguation": None,
    "pending_watch_tickers": None,
    "builder_profile": None
}

for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value



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
    ("🔎 Screen Indian Stocks",
     "Find the best Indian stocks to invest in right now. Show me which stocks pass all 4 frameworks and which pass 3 out of 4 and which pass 2 out of 4, with upto top 10 from each tier. Explain why each tier is a good investment using the book philosophies."),
    ("💎 Find Hidden Gems",
     "Find hidden gem stocks — small and mid cap Indian companies outside the Nifty 50 that pass at least 3 out of 4 frameworks. Show top 10 with key metrics. Explain why each is a good investment using book philosophies."),
]

# ══════════════════════════════════════════════
# CSS — INSTITUTIONAL LIGHT THEME
# ══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto+Slab:wght@700;900&family=Inter:wght@400;500;600&display=swap');

/* ── Fix 1: Nuke the dark chat container ── */
[data-testid="stChatInput"] {
    background-color: #FFFFFF !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 4px !important;
    box-shadow: 2px 2px 0px rgba(0,0,0,0.05) !important;
}

[data-testid="stChatInput"] > div {
    background-color: transparent !important;
}

/* Ensure the send arrow icon is visible */
[data-testid="stChatInput"] button svg {
    fill: #FFFFFF !important;
}

/* ── Fix 2: Institutionalize the st.info alerts ── */
[data-testid="stAlert"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-left: 4px solid #1D4ED8 !important; /* Trust Blue Accent */
    color: #374151 !important;
    border-radius: 4px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
}


/* ── Fix 5: Force All Standard Buttons to Light Theme ── */
.stButton > button, 
div[data-testid="stButton"] > button,
[data-testid="baseButton-secondary"],
[data-testid="baseButton-primary"] {
    background-color: #FFFFFF !important;
    color: #111827 !important;
    border: 1px solid #D1D5DB !important;
    box-shadow: 2px 2px 0px rgba(0,0,0,0.05) !important;
    font-weight: 600 !important;
    transition: all 0.1s ease !important;
}

.stButton > button:hover,
div[data-testid="stButton"] > button:hover,
[data-testid="baseButton-secondary"]:hover,
[data-testid="baseButton-primary"]:hover {
    background-color: #F8F9FA !important;
    border-color: #1D4ED8 !important; /* Trust Blue on hover */
    color: #1D4ED8 !important;
}

/* Ensure text/markdown inside the button inherits the correct color */
.stButton > button * {
    color: inherit !important; 
}

/* ── Fix 6: Eradicate the dotted outline inside the chat text area ── */
[data-testid="stChatInput"] textarea,
[data-testid="stChatInputContainer"] textarea {
    outline: none !important;
    border: none !important;
    box-shadow: none !important;
    outline-style: none !important; /* Kills the browser default dotted line */
}

[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInputContainer"] textarea:focus {
    outline: none !important;
    border: none !important;
    box-shadow: none !important;
    outline-style: none !important;
}

/* ── Fix 7: Force the Info Box background to solid white ── */
div[data-testid="stAlert"] {
    background-color: transparent !important;
}
div[data-testid="stAlert"] > div {
    background-color: #FFFFFF !important;
}

/* ── Fix 8: Portfolio Boundary Cards ── */
/* Targets the st.container(border=True) wrappers */
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 6px !important;
    padding: 1.5rem !important; /* Gives the text and tables breathing room */
    box-shadow: 2px 2px 0px rgba(0,0,0,0.03) !important; /* Subtle institutional weight */
    margin-bottom: 2rem !important; /* Space between different portfolios */
}

/* Make sure the portfolio title stands out inside the card */
[data-testid="stVerticalBlockBorderWrapper"] h3 {
    margin-top: 0 !important;
    padding-top: 0 !important;
    border-bottom: 1px solid #F3F4F6 !important;
    padding-bottom: 10px !important;
    margin-bottom: 15px !important;
}

/* ── Fix 9: Fallback Table Styling ── */
/* Forces any native HTML/Markdown tables into the light theme */
.stTable {
    background-color: #FFFFFF !important;
}
.stTable > div > table {
    border: 1px solid #E5E7EB !important;
    border-radius: 4px !important;
}
.stTable th {
    background-color: #F9FAFB !important;
    color: #374151 !important;
    border-bottom: 2px solid #D1D5DB !important;
    font-weight: 600 !important;
}
.stTable td {
    color: #111827 !important;
    border-bottom: 1px solid #E5E7EB !important;
}

/* ── Fix 10: Force Symmetric Rounded Corners on DataFrames ── */
[data-testid="stDataFrame"] {
    border-radius: 6px !important;
    overflow: hidden !important; /* This acts like a cookie-cutter, clipping sharp inner corners */
    border: 1px solid #E5E7EB !important;
}

[data-testid="stDataFrame"] > div {
    border-radius: 6px !important;
}

/* Ensure the header row doesn't break the top-right curve */
[data-testid="stDataFrame"] [data-baseweb="table"] {
    border-radius: 6px !important;
}


/* ── Base ── */
.stApp {
    background-color: #F8F9FA !important; /* Concrete Off-White */
}

.stApp, .stApp * {
    font-family: 'Inter', sans-serif !important;
    color: #374151 !important; /* Dark Slate */
}

/* Exclude icons and code blocks from the universal font override */
.stApp *:not(code):not(.material-symbols-rounded):not(i):not(svg) {
    font-family: 'Inter', sans-serif !important;
}

/* Explicitly protect the expander toggle icons */
[data-testid="stExpanderToggleIcon"], 
.material-symbols-rounded {
    font-family: "Material Symbols Rounded" !important;
    color: #6B7280 !important;
}

[data-testid="stAppViewContainer"] {
    background: transparent !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E5E7EB !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdown"] p {
    color: #6B7280 !important;
    font-size: 0.85rem !important;
}

/* ── Heavy, Carved Headers ── */
[data-testid="stSidebar"] h1, .stApp h1 {
    font-family: 'Roboto Slab', serif !important;
    color: #111827 !important;
    font-weight: 900 !important;
    letter-spacing: -0.5px !important;
    /* 3D Debossed 'stamped concrete' effect */
    text-shadow: 1px 1px 0px #ffffff, 2px 2px 0px rgba(0,0,0,0.08) !important;
}

[data-testid="stSidebar"] h1 {
    font-size: 1.3rem !important;
}

.stApp h1 {
    font-size: 2.2rem !important;
    padding-bottom: 2px;
    text-transform: none !important;
}

[data-testid="stSidebar"] h3 {
    font-family: 'Roboto Slab', serif !important;
    color: #4B5563 !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    margin-top: 1.5rem !important;
}

[data-testid="stSidebar"] hr, .stApp hr {
    border-color: #E5E7EB !important;
}

.stApp .stCaption, .stApp [data-testid="stCaptionContainer"] p {
    color: #6b7280 !important;
    font-size: 0.88rem !important;
}

/* ── Chat bubbles ── */
[data-testid="stChatMessage"] {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 4px !important; /* Sharper, institutional corners */
    padding: 1rem 1.2rem !important;
    margin-bottom: 10px !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {
    color: #1F2937 !important;
    line-height: 1.7 !important;
    font-size: 0.95rem !important;
}

[data-testid="stChatMessage"] strong {
    color: #1D4ED8 !important; /* Trust Blue */
}

[data-testid="stChatMessage"] code {
    background: #F3F4F6 !important;
    color: #1D4ED8 !important;
    border-radius: 2px !important;
    padding: 2px 6px !important;
    border: 1px solid #E5E7EB !important;
}

[data-testid="stChatMessage"] [data-testid="stAvatar"] {
    border: 1px solid #E5E7EB !important;
    border-radius: 4px !important; /* Square avatar */
    background: #F8F9FA !important;
}

/* ── Chat input: Institutional Single-Box Design ── */
[data-testid="stChatInput"],
[data-testid="stChatInputContainer"] {
    background: transparent !important;
}

/* 1. Nuke the outer Streamlit wrapper and its red focus ring */
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div:focus-within {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    background-color: transparent !important;
}

/* 2. Style the Base Web input container to act as the main box */
[data-testid="stChatInput"] [data-baseweb="base-input"] {
    background: #FFFFFF !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 4px !important;
    box-shadow: 2px 2px 0px rgba(0,0,0,0.05) !important;
    padding: 4px !important; /* Space for the button */
    transition: all 0.2s ease;
}

/* Trust Blue focus state for the whole container */
[data-testid="stChatInput"] [data-baseweb="base-input"]:focus-within {
    border-color: #1D4ED8 !important;
    box-shadow: 0 0 0 1px rgba(29, 78, 216, 0.2) !important;
    outline: none !important;
}

/* 3. Strip all borders from the raw textarea so it blends in */
[data-testid="stChatInput"] textarea,
[data-testid="stChatInputContainer"] textarea {
    background: transparent !important;
    border: none !important; /* Removes the inner blue box */
    box-shadow: none !important;
    outline: none !important;
    color: #111827 !important;
    font-size: 0.95rem !important;
    padding: 8px 12px !important;
}

[data-testid="stChatInput"] textarea:focus {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #9CA3AF !important;
}

/* 4. Button styling to match */
[data-testid="stChatInput"] button {
    background: #111827 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 4px !important;
    margin-top: 4px !important;
}

[data-testid="stChatInput"] button:hover {
    background: #374151 !important;
}

[data-testid="stChatInput"] button svg {
    fill: #FFFFFF !important;
}

/* Kill generic focus outlines */
*:focus, *:active, *:focus-visible { outline: none !important; }
div[data-baseweb] [aria-invalid] { box-shadow: none !important; }


/* ── Text input ── */
.stTextInput > div > div > input {
    background: #FFFFFF !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 4px !important;
    color: #111827 !important;
    font-family: 'Roboto Slab', serif !important;
    font-size: 0.95rem !important;
    padding: 10px 14px !important;
    text-align: center !important;
}

.stTextInput > div > div > input::placeholder {
    color: #9CA3AF !important;
}

.stTextInput > div > div > input:focus {
    border-color: #1D4ED8 !important;
    box-shadow: 0 0 0 1px rgba(29, 78, 216, 0.2) !important;
    outline: none !important;
}

.stTextInput label {
    color: #4B5563 !important;
    font-size: 0.75rem !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
}

/* ── Bottom dock ── */
[data-testid="stBottom"] {
    background: #F8F9FA !important;
    background-color: #F8F9FA !important;
    border-top: 1px solid #E5E7EB !important;
}

[data-testid="stBottom"] > div {
    background: transparent !important;
    background-color: transparent !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #1D4ED8 !important; }
[data-testid="stSpinnerContainer"] { color: #6B7280 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #D1D5DB; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #9CA3AF; }

/* ── Tables ── */
.stDataFrame, .stTable {
    max-width: 100% !important;
    overflow-x: auto !important;
}

[data-testid="stChatMessage"] table {
    display: block !important;
    overflow-x: auto !important;
    white-space: nowrap !important;
    max-width: 100% !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 4px !important;
}

[data-testid="stChatMessage"] th {
    background-color: #F3F4F6 !important;
    color: #111827 !important;
    font-weight: 600 !important;
}

[data-testid="stChatMessage"] td {
    border-top: 1px solid #E5E7EB !important;
}

/* ── Responsive ── */
@media (max-width: 768px) {
    .stApp h1 { font-size: 1.8rem !important; }
}

</style>
""", unsafe_allow_html=True)

# ── Google OAuth callback handler ──
_oa_access = st.query_params.get("access_token")
_oa_refresh = st.query_params.get("refresh_token")
if _oa_access and not st.session_state.sb_user_email:
    try:
        sb = get_supabase()
        resp = sb.auth.set_session(_oa_access, _oa_refresh)
        st.session_state.sb_access_token = _oa_access
        st.session_state.sb_refresh_token = _oa_refresh
        st.session_state.sb_user_email = resp.user.email
        st.session_state.sb_user_id = str(resp.user.id)
        meta = resp.user.user_metadata or {}
        st.session_state["_profile_name"] = meta.get("full_name") or meta.get("name") or ""
        st.session_state["_profile_name_checked"] = True
        try:
            sb.table("profiles").upsert({
                "id": st.session_state.sb_user_id,
                "full_name": st.session_state["_profile_name"],
                "email": resp.user.email
            }, on_conflict="id").execute()
        except Exception:
            pass
    except Exception as e:
        st.error(f"Google login failed: {e}")
    st.query_params.clear()
    st.rerun()



# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    # ── Auth ──
    if st.session_state.sb_user_email is None:
        _goog_url = f"{st.secrets['SUPABASE_URL']}/auth/v1/authorize?provider=google&redirect_to=https://aakash10867.github.io/graham-agent/auth-callback.html"
        st.link_button("🔵 Sign in with Google", _goog_url, use_container_width=True)
        st.divider()
        auth_mode = st.radio(
            "Account", ["Login", "Sign Up"],
            horizontal=True, label_visibility="collapsed"
        )
        if auth_mode == "Sign Up":
            auth_full_name = st.text_input("Full Name", key="auth_name_input")
        auth_email = st.text_input("Email", key="auth_email_input")
        auth_password = st.text_input("Password", type="password", key="auth_password_input")

        if auth_mode == "Login":
            if st.button("Log In", width="stretch"):
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
            if st.button("Sign Up", width="stretch"):
                if not auth_full_name or not auth_full_name.strip():
                    st.warning("Enter your full name.")
                elif not auth_email or not auth_password:
                    st.warning("Enter email and password.")
                elif len(auth_password) < 6:
                    st.warning("Password must be at least 6 characters.")
                else:
                    try:
                        sb = get_supabase()
                        resp = sb.auth.sign_up({
                            "email": auth_email,
                            "password": auth_password,
                            "options": {"data": {"full_name": auth_full_name.strip()}}
                        })
                        st.session_state.sb_access_token = resp.session.access_token
                        st.session_state.sb_refresh_token = resp.session.refresh_token
                        st.session_state.sb_user_email = resp.user.email
                        st.session_state.sb_user_id = str(resp.user.id)
                        try:
                            sb.table("profiles").upsert({
                                "id": st.session_state.sb_user_id,
                                "full_name": auth_full_name.strip()
                            }, on_conflict="id").execute()
                        except Exception:
                            pass  # non-blocking — name also lives in user_metadata
                        st.rerun()
                    except Exception as e:
                        st.error(f"Sign up failed: {e}")

    else:
        _display_name = st.session_state.get("_profile_name") or st.session_state.sb_user_email
        st.caption(f"Logged in as {_display_name}")
        sb = get_supabase()

        # ── One-time name collection for existing users ──
        if not st.session_state.get("_profile_name_checked"):
            try:
                _prof = sb.table("profiles").select("full_name").eq(
                    "id", st.session_state.sb_user_id
                ).execute()
                _existing_name = (_prof.data[0].get("full_name") or "") if _prof.data else ""
                st.session_state["_profile_name_checked"] = True
                st.session_state["_profile_name"] = _existing_name
            except Exception:
                _existing_name = ""
                st.session_state["_profile_name_checked"] = True
                st.session_state["_profile_name"] = ""
        
        if st.session_state.get("_profile_name_checked") and not st.session_state.get("_profile_name"):
            with st.container(border=True):
                st.caption("👋 Add your name for personalized reports & emails")
                _name_input = st.text_input("Full Name", key="profile_name_fill")
                if st.button("Save", key="save_profile_name") and _name_input and _name_input.strip():
                    try:
                        sb.table("profiles").upsert({
                            "id": st.session_state.sb_user_id,
                            "full_name": _name_input.strip()
                        }, on_conflict="id").execute()
                        st.session_state["_profile_name"] = _name_input.strip()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")
        
        if st.session_state.sb_view_mode != "builder":
            if st.button("🏗️ Build Portfolio", width="stretch"):
                st.session_state.sb_view_mode = "builder"
                st.rerun()

        if st.session_state.sb_view_mode != "import":
            if st.button("📥 Import Existing Portfolio", width="stretch"):
                st.session_state.sb_view_mode = "import"
                st.rerun()
                
        if st.session_state.sb_view_mode != "portfolios":
            try:
                _all_ports = sb.table("portfolios").select("id, is_paper").eq(
                    "user_id", st.session_state.sb_user_id
                ).execute().data or []
                _port_count = len([p for p in _all_ports if not p.get("is_paper")])
            except Exception:
                _port_count = 0
            
            _port_label = f"📁 My Portfolios ({_port_count})" if _port_count else "📁 My Portfolios"
            
            if st.button(_port_label, width="stretch"):
                st.session_state.sb_view_mode = "portfolios"
                st.rerun()
                
        if st.session_state.sb_view_mode != "chat":
            if st.button("← Back to Chat", width="stretch"):
                st.session_state.sb_view_mode = "chat"
                st.rerun()

        # ── My Watchlist nav ──
        if st.session_state.sb_view_mode != "watchlist":
            try:
                _wl_stocks = len((sb.table("watchlist").select("id").eq(
                    "user_id", st.session_state.sb_user_id
                ).execute()).data or [])
                _all_ports_wl = sb.table("portfolios").select("id, is_paper").eq(
                    "user_id", st.session_state.sb_user_id
                ).execute().data or []
                _wl_paper_ports = len([p for p in _all_ports_wl if p.get("is_paper")])
                _wl_count = _wl_stocks + _wl_paper_ports
            except Exception:
                _wl_count = 0
            _wl_label = f"👁 My Watchlist ({_wl_count})" if _wl_count else "👁 My Watchlist"
            if st.button(_wl_label, width="stretch"):
                st.session_state.sb_view_mode = "watchlist"
                st.rerun()

        st.divider()

        if st.button("Log Out", width="stretch", key="logout_btn"):
            try:
                sb = get_supabase()
                sb.auth.sign_out()
            except Exception:
                pass
            st.session_state.sb_access_token = None
            st.session_state.sb_refresh_token = None
            st.session_state.sb_user_email = None
            st.session_state.sb_user_id = None
            st.session_state.sb_view_mode = "chat" # Resets view on logout
            st.rerun()

    st.divider()
    st.markdown("Multi-framework investment analysis powered by Graham, Greenblatt, Dorsey, and momentum scoring.")

    if st.button("🔄 New Chat", width="stretch"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.sb_view_mode = "chat" 
        if "pending_portfolio" in st.session_state:
            st.session_state.pending_portfolio = None
            st.session_state.pop("pending_watch_tickers", None)
        st.rerun()


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
    st.markdown("<h1 style='margin-top: -15px; padding-bottom: 0px;'>Kordent</h1>", unsafe_allow_html=True)

st.markdown("---")
# ──────────────────────────────────────────────
# PUBLIC LEADERBOARD (Landing Page Only)
# ──────────────────────────────────────────────
if st.session_state.sb_view_mode == "chat" and not st.session_state.messages:
    try:
        sb = get_supabase()
        # Fetch the top 3 public portfolios by current return
        leaderboard_resp = sb.table("portfolios").select(
            "name, investor_type, time_horizon, current_return_pct"
        ).order("current_return_pct", desc=True).limit(3).execute()
        
        top_portfolios = leaderboard_resp.data
        
        if top_portfolios:
            st.markdown("### 🏆 Top Performing Portfolios")
            l_cols = st.columns(3)
            for i, port in enumerate(top_portfolios):
                with l_cols[i]:
                    with st.container(border=True):
                        st.metric(
                            label=port["name"], 
                            value=f"{port.get('current_return_pct', 0):+.2f}%", 
                            delta=str(port.get("investor_type", "balanced")).title()
                        )
                        st.caption(f"Horizon: {str(port.get('time_horizon', 'medium')).title()}")
            st.markdown("---")
    except Exception as e:
        pass # Fail silently if database is unreachable or empty



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

            st.altair_chart(chart, width="stretch")

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
        # Graham requires 7+ years of earnings track record
        first_trade = info.get("firstTradeDateEpochUtc")
        if first_trade:
            first_date = datetime.datetime.fromtimestamp(first_trade, tz=datetime.timezone.utc)
            years_listed = (datetime.datetime.now(tz=datetime.timezone.utc) - first_date).days / 365.25
            if years_listed < 7:
                return {
                    "ticker": resolved,
                    "graham_applicable": False,
                    "years_listed": round(years_listed, 1),
                    "error": (
                        f"Graham analysis not applicable: {resolved} has only "
                        f"~{round(years_listed, 1)} years of trading history. "
                        f"Graham required a minimum of 7 years of consistent earnings "
                        f"data before trusting any valuation formula. Companies with "
                        f"shorter track records lack the earnings stability evidence "
                        f"his intrinsic value formula assumes."
                    ),
                }

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
        # Stocks near 52-week lows rank higher (more negative pct_from_high = better value)
        t["_high_sort"] = t["pct_from_high"].apply(lambda x: x if pd.notna(x) else 0)
        t = t.sort_values(["_pe_sort", "_high_sort", "_roe_sort", "_rev_sort"])
        return t.drop(columns=["_pe_sort", "_roe_sort", "_rev_sort", "_high_sort"])

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
                "pct_from_52w_high": round(row["pct_from_high"], 1) if pd.notna(row.get("pct_from_high")) else "N/A",
                "pct_from_52w_low": round(row["pct_from_low"], 1) if pd.notna(row.get("pct_from_low")) else "N/A",
                "pe_vs_historical": round(row["pe_vs_avg"], 1) if pd.notna(row.get("pe_vs_avg")) else "N/A",
                "beta": round(row["beta"], 2) if pd.notna(row.get("beta")) else "N/A",
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

def get_sip_candidates(sip_amount: int, time_horizon: str, investor_type: str, review_freq: str, avoid_sectors: str = "[]") -> dict:
    """Filter the pre-scored universe to SIP-suitable candidates based on investor profile.
    Returns a min/max stock count range computed from the SIP amount. The LLM decides the
    exact count within that range based on candidate quality, not a hardcoded number.

    Use this when building a portfolio from a [BUILDER_PROFILE] message.

    Args:
        sip_amount: Monthly SIP amount in INR (e.g. 5000, 25000, 50000)
        time_horizon: Investment duration. Must be one of:
                      short  - 1 to 3 years
                      medium - 3 to 7 years
                      long   - 7+ years
        investor_type: Risk profile. Must be one of:
                       defensive    - wants to beat FD returns with safety
                       balanced     - wants to build wealth steadily over time
                       enterprising - wants maximum growth, patient through volatility
        review_freq: How often the investor wants to monitor. Must be one of:
                     passive  - set it and forget for years
                     moderate - review every few months
                     active   - likes staying informed and adjusting
        avoid_sectors: JSON string list of sector names to exclude, e.g. '["Energy", "Basic Materials"]'.
                       Pass '[]' for no exclusions.
    """
    df = universe_df.copy()

    # ── Base quality filter (all profiles) ──
    if "quality_pass" in df.columns:
        df = df[df["quality_pass"] != False]
    df = df[df["years_of_data"] >= 2]
    df = df[pd.notna(df["pe"]) & pd.notna(df["roe_pct"]) & pd.notna(df["de"])]
    df = df[df["pe"] > 0]  # Exclude negative P/E (loss-making)

    # ── Sector exclusions from builder profile ──
    try:
        _excluded = json.loads(avoid_sectors) if isinstance(avoid_sectors, str) else avoid_sectors
        if _excluded:
            df = df[~df["sector"].isin(_excluded)]
    except (json.JSONDecodeError, TypeError):
        pass

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
            # Enriched columns for LLM judgment
            "pe_4y_avg": round(row["pe_4y_avg"], 2) if pd.notna(row.get("pe_4y_avg")) else None,
            "pe_vs_avg": round(row["pe_vs_avg"], 2) if pd.notna(row.get("pe_vs_avg")) else None,
            "pct_from_high": round(row["pct_from_high"], 2) if pd.notna(row.get("pct_from_high")) else None,
            "pct_from_low": round(row["pct_from_low"], 2) if pd.notna(row.get("pct_from_low")) else None,
            "current_ratio": round(row["current_ratio"], 2) if pd.notna(row.get("current_ratio")) else None,
            "beta": round(row["beta"], 2) if pd.notna(row.get("beta")) else None,
            "revenue_cagr_3y": round(row["revenue_cagr_3y"], 2) if pd.notna(row.get("revenue_cagr_3y")) else None,
            "ni_cagr_3y": round(row["ni_cagr_3y"], 2) if pd.notna(row.get("ni_cagr_3y")) else None,
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

    max_stocks = min(20, sip_amount // 750)
    min_stocks = max(5, max_stocks // 3)

    fringe_candidates = []
    try:
        # Find 5 highly-scored stocks (3 or 4) that were excluded due to sector restrictions
        _exc = json.loads(avoid_sectors) if isinstance(avoid_sectors, str) else avoid_sectors
        if _exc:
            fringe_df = universe_df[(universe_df["score"] >= 3) & (universe_df["sector"].isin(_exc))].head(5)
            for _, r in fringe_df.iterrows():
                fringe_candidates.append({
                    "ticker": r["ticker"], "name": str(r.get("name", r["ticker"])),
                    "sector": str(r.get("sector", "N/A")), "score": int(r["score"]),
                    "reason_excluded": f"Sector ({r.get('sector')}) was excluded by user."
                })
    except Exception:
        pass

    return {
        "investor_profile": {
            "sip_amount_inr": sip_amount,
            "time_horizon": time_horizon,
            "investor_type": investor_type,
            "review_frequency": review_freq,
        },
        "portfolio_sizing": {
            "min_stocks": min_stocks,
            "max_stocks": max_stocks,
            "note": "These are hard bounds from the SIP amount. You decide the exact count based on candidate quality.",
        },
        "candidates_count": len(candidates),
        "candidates": candidates,
        "fringe_candidates": fringe_candidates, # Inject fringe list here
        "selection_instruction": (
            f"You have {len(candidates)} pre-filtered candidates. "
            f"Pick between {min_stocks} and {max_stocks} stocks. YOU decide the exact count — "
            f"it depends on how many candidates are genuinely strong, not a formula. "
            f"Research shows ~15 stocks eliminates ~85% of unsystematic risk. But 8 excellent picks beat 15 mediocre ones. Never pad. "
            f"Use search_book to pull Graham/Greenblatt/Dorsey wisdom relevant to this {investor_type} profile with {time_horizon}-term horizon. "
            f"Use pe_vs_avg to check if a stock is cheap relative to its own history (negative = discount, positive = premium). "
            f"Use pct_from_high to spot stocks near 52-week lows (potential value) vs near highs (potential overvaluation). "
            f"Use revenue_cagr_3y and ni_cagr_3y to assess growth trajectory beyond single-year noise. "
            f"Use beta to assess how much each stock moves with the market — relevant for portfolio-level risk. "
            f"Apply qualitative moat assessment (Dorsey) — check ROE trends to see if moat is stable or eroding. "
            f"Enforce max 2 stocks per sector for diversification. "
            f"Before finalizing, compute: (1) how many sectors you cover — aim for at least 4, "
            f"(2) whether any single sector exceeds 30% allocation — if so, rebalance, "
            f"(3) whether the portfolio beta is balanced — avoid loading up on all high-beta or all low-beta stocks. "
            f"If the portfolio fails these checks, revise your selection before outputting. "
            f"Allocate the monthly SIP of INR {sip_amount} across selected stocks. "
            f"For each pick, be prepared to explain WHY it fits this investor using book philosophy. "
            f"CRITICAL: If the user's message started with [BUILDER_PROFILE], you are in PHASE 1. DO NOT output the portfolio table yet. DO NOT call register_portfolio. Use this data ONLY to formulate your 1-3 clarification questions."
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
    Returns macro context: sector, Nifty 5-day performance, India VIX (fear gauge),
    and sector-level index performance where available.
    """
    resolved = _resolve_ticker(ticker)
    try:
        stock = yf.Ticker(resolved)
        sector = stock.info.get('sector', 'Unknown Sector')
        
        result = {"ticker": resolved, "sector": sector}

        # Nifty 50 momentum
        try:
            nifty = yf.Ticker("^NSEI")
            hist = nifty.history(period="1mo")
            if not hist.empty and len(hist) >= 2:
                result["nifty_50_1d_pct"] = round(((hist['Close'].iloc[-1] / hist['Close'].iloc[-2]) - 1) * 100, 2)
                result["nifty_50_5d_pct"] = round(((hist['Close'].iloc[-1] / hist['Close'].iloc[-6 if len(hist) >= 6 else 0]) - 1) * 100, 2)
                result["nifty_50_1mo_pct"] = round(((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100, 2)
                result["nifty_50_close"] = round(float(hist['Close'].iloc[-1]), 2)
        except Exception:
            result["nifty_50_5d_pct"] = "N/A"

        # India VIX (fear gauge)
        try:
            vix = yf.Ticker("^INDIAVIX")
            vix_hist = vix.history(period="5d")
            if not vix_hist.empty:
                vix_val = round(float(vix_hist['Close'].iloc[-1]), 2)
                result["india_vix"] = vix_val
                if vix_val < 13:
                    result["vix_interpretation"] = "Low fear — market is complacent, potential for sudden correction"
                elif vix_val < 20:
                    result["vix_interpretation"] = "Normal range — market pricing moderate uncertainty"
                elif vix_val < 30:
                    result["vix_interpretation"] = "Elevated fear — market expects significant moves"
                else:
                    result["vix_interpretation"] = "Panic levels — historically a contrarian buy signal per Graham"
        except Exception:
            result["india_vix"] = "N/A"

        # Sector-specific index (map common sectors to Nifty sector indices)
        SECTOR_INDICES = {
            "Technology": "^CNXIT",
            "Financial Services": "^CNXFIN",
            "Energy": "^CNXENERGY",
            "Consumer Cyclical": "^CNXCONSUMER",
            "Healthcare": "^CNXPHARMA",
            "Basic Materials": "^CNXMETAL",
            "Industrials": "^CNXINFRA",
            "Consumer Defensive": "^CNXFMCG",
            "Real Estate": "^CNXREALTY",
            "Utilities": "^CNXPSUBANK",
        }

        sector_idx = SECTOR_INDICES.get(sector)
        if sector_idx:
            try:
                sidx = yf.Ticker(sector_idx)
                sidx_hist = sidx.history(period="1mo")
                if not sidx_hist.empty and len(sidx_hist) >= 2:
                    result["sector_index"] = sector_idx
                    result["sector_1d_pct"] = round(((sidx_hist['Close'].iloc[-1] / sidx_hist['Close'].iloc[-2]) - 1) * 100, 2)
                    result["sector_5d_pct"] = round(((sidx_hist['Close'].iloc[-1] / sidx_hist['Close'].iloc[-6 if len(sidx_hist) >= 6 else 0]) - 1) * 100, 2)
                    result["sector_1mo_pct"] = round(((sidx_hist['Close'].iloc[-1] / sidx_hist['Close'].iloc[0]) - 1) * 100, 2)
            except Exception:
                pass

        return result

    except Exception as e:
        return {"error": f"Error fetching macro context: {str(e)}"}


def _sanitize_for_json(obj):
    """Replace NaN/Inf with None so Gemini gets valid JSON."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    # Catch Python float, numpy.float64, numpy.float32, and any numeric type
    try:
        if math.isnan(obj) or math.isinf(obj):
            return None
    except (TypeError, ValueError, OverflowError):
        pass
    # Convert stray numpy scalars to Python native so json.dumps never chokes
    if hasattr(obj, 'item'):
        return obj.item()
    return obj
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
16. get_sip_candidates — Build a SIP portfolio from a builder profile. Takes sip_amount, time_horizon, investor_type, review_freq, and avoid_sectors (a JSON string list of sector names to exclude, e.g. '["Energy"]'; pass '[]' for no exclusions). Returns pre-filtered candidates with a min/max stock count range. You decide the exact count based on candidate quality.
17. register_portfolio — After presenting your finalized SIP portfolio, call this to register it for saving. Pass portfolio_name, investor_type, sip_amount, time_horizon, review_days (integer), stocks_json (JSON string list with ticker/name/sector/allocation_pct per item), portfolio_profile (JSON string of the full builder profile), target_amount (number, 0 if no goal), and target_date (ISO date string, empty if no goal). ALWAYS call this after presenting the final portfolio table.

SIP PORTFOLIO PROTOCOL:
Portfolio building uses the embedded Builder form (🏗️ Build Portfolio sidebar button). When you receive a message starting with [BUILDER_PROFILE], you must execute a strict 2-Phase process.

PHASE 1: DRAFT & INTERROGATE (DO NOT SHOW THE PORTFOLIO YET)
1. Call get_sip_candidates with the profile parameters.
2. Silently construct a "V1" portfolio in your mind. Do NOT output a table, do NOT list the stocks, and do NOT call register_portfolio.
3. Call register_portfolio with all fields including portfolio_profile, target_amount, and target_date. CRITICAL: Use the `decision_context` parameter to summarize the user's answers to your Phase 1 questions so the system remembers their accepted trade-offs (e.g. "User accepted volatility in Industrials for higher growth").
4. Output a brief, layman-friendly summary of the strategy you are considering.
5. Ask the user 1 to 3 targeted questions to refine the build. 
   - RULE: Speak to them as a layman. Do NOT use jargon like "Graham", "Dorsey", "moat", "beta", or "PE expansion". 
   - Example (Fringe Candidate): "I found a highly profitable company that fits your goals perfectly, but it's in the Energy sector which you asked to avoid. Are you open to making an exception for a top-tier performer?"
   - Example (Risk Trade-off): "To hit your target, we need a bit more growth. Would you prefer adding a fast-growing but bumpier stock, or stick to steady, slow-moving giants?"
6. Stop and wait for the user's reply.

PHASE 2: FINALIZE & REGISTER (TRIGGERED ONLY AFTER USER REPLIES)
1. When the user answers your questions, finalize the stock selection.
2. Output the final portfolio table. NOW you may explain the quantitative reasoning using book philosophies (Graham/Greenblatt/Dorsey) to educate them on why these picks were made.
3. CRITICAL: Generate this textual explanation FIRST.
4. Call register_portfolio with all fields including portfolio_profile, target_amount, and target_date. Do not ask for permission to save, just call the tool.

If someone asks to build a portfolio WITHOUT a [BUILDER_PROFILE] prefix, direct them to click the 🏗️ Build Portfolio button in the sidebar. If they insist or provide enough info inline, you may proceed by mapping their inputs to the profile parameters.

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
        last_good = st.session_state.get("last_working_model")
        if last_good and last_good in FREE_MODELS:
            models_to_try = [last_good] + [m for m in FREE_MODELS if m != last_good]
        else:
            models_to_try = FREE_MODELS
        for model_name in models_to_try:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=router_prompt,
                )
                return f"SYSTEM DIRECTIVE (Translated Intent): {response.text}"
            except Exception as inner_e:
                error_msg = str(inner_e).upper()
                if any(err in error_msg for err in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500"]):
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
    # Prioritize the model that worked last turn
    last_good = st.session_state.get("last_working_model")
    if last_good and last_good in FREE_MODELS:
        models_to_try = [last_good] + [m for m in FREE_MODELS if m != last_good]
    else:
        models_to_try = FREE_MODELS
    for model_name in models_to_try:
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
                        # Execute the tool function, then immediately sanitize the dictionary output
                        raw_tool_output = tool_functions[fc.name](**fc.args)
                        result = _sanitize_for_json(raw_tool_output)
                    else:
                        result = {"error": f"Unknown tool: {fc.name}"}
                        
                    function_responses.append(
                        types.Part.from_function_response(name=fc.name, response=result)
                    )
                # Send the sanitized parameters back to the chat manager
                analyst_response = analyst_chat.send_message(function_responses)

            final_chunk = _extract_text(analyst_response)
            if final_chunk:
                all_text_parts.append(final_chunk)
            
            clean_parts = [p.strip() for p in all_text_parts if p.strip()]
            draft_text = "\n\n".join(clean_parts).strip()

            if not draft_text:
                recovery_prompt = (
                    "You successfully executed the register_portfolio tool, but you provided zero text to the user. "
                    "You MUST reply now with a stock-by-stock explanation of why you selected each company, "
                    "grounding your reasoning in the Graham, Greenblatt, and Dorsey frameworks. Do not output any more tool calls."
                )
                recovery_response = analyst_chat.send_message(recovery_prompt)
                draft_text = _extract_text(recovery_response).strip()

            # --- PHASE 2: AUDITOR REVIEWS DRAFT (with independent data) -----
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
                correction_prompt = f"The Chief Risk Officer REJECTED your draft with the following feedback:\n\n{audit_result}\n\nRewrite your entire analysis to comply with this feedback. CRITICAL: If you are building a portfolio, you MUST call the register_portfolio tool AGAIN with your updated stock list to overwrite the rejected database entry."
                final_response = analyst_chat.send_message(correction_prompt)
                
                # --- FIX: We must process tool calls during the correction phase too! ---
                corr_text_parts = []
                while final_response.function_calls:
                    text_chunk = _extract_text(final_response)
                    if text_chunk:
                        corr_text_parts.append(text_chunk)
                        
                    function_responses = []
                    for fc in final_response.function_calls:
                        if fc.name in tool_functions:
                            raw_tool_output = tool_functions[fc.name](**fc.args)
                            result = _sanitize_for_json(raw_tool_output)
                        else:
                            result = {"error": f"Unknown tool: {fc.name}"}
                            
                        function_responses.append(
                            types.Part.from_function_response(name=fc.name, response=result)
                        )
                    final_response = analyst_chat.send_message(function_responses)

                final_chunk = _extract_text(final_response)
                if final_chunk:
                    corr_text_parts.append(final_chunk)
                
                clean_corr_parts = [p.strip() for p in corr_text_parts if p.strip()]
                final_text = "\n\n".join(clean_corr_parts).strip()

                st.session_state.chat_history = analyst_chat.get_history()
                
                # Append an internal note to the UI so the user sees the system working
                st.session_state.last_working_model = model_name
                return f"*(Internal Audit Triggered: Adjusted thesis based on earnings quality)*\n\n{final_text}", model_name
            else:
                # Auditor approved
                st.session_state.chat_history = analyst_chat.get_history()
                st.session_state.last_working_model = model_name
                return draft_text, model_name

        except Exception as e:
            last_error = str(e)
            error_upper = last_error.upper()
            if any(err in error_upper for err in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500"]):
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

    st.markdown("")
    st.caption("Market screeners")
    scr_cols = st.columns(len(SCREENER_PRESETS))
    for i, (label, template) in enumerate(SCREENER_PRESETS):
        with scr_cols[i]:
            if st.button(label, key=f"screener_{i}", width="stretch"):
                st.session_state.pending_prompt = template
                st.rerun()

    prompt = st.chat_input("Ask about any stock, or type a question...")

    if not prompt and "pending_prompt" in st.session_state:
        prompt = st.session_state.pop("pending_prompt")
        if prompt and st.session_state.get("pending_disambiguation"):
            st.session_state.pending_disambiguation = None

    with chat_area:
        if not st.session_state.messages:
            st.markdown("")
            st.info("Type a company name or question below to get started, or use the screeners below.")

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
                st.dataframe(pd.DataFrame(preview_data), hide_index=True, width="stretch")
                _paper_tag = " · 👁 Paper Portfolio" if portfolio.get("is_paper") else ""
                _goal_tag = f" · Goal: ₹{portfolio['target_amount']:,.0f}" if portfolio.get("target_amount") else ""
                st.caption(f"Total SIP: ₹{portfolio['sip_amount']:,}/month · {portfolio.get('investor_type', '')} · {portfolio.get('time_horizon', '')} horizon{_goal_tag}{_paper_tag}")
                if st.button("💾 Save Portfolio", width="stretch"):
                    try:
                        sb = get_supabase()
                        review_days = portfolio.get("review_days", 90)
                        next_review = (datetime.date.today() + datetime.timedelta(days=review_days)).isoformat()
                        next_sip = (datetime.date.today() + datetime.timedelta(days=30)).isoformat()
                        
                        port_resp = sb.table("portfolios").insert({
                            "user_id": st.session_state.sb_user_id,
                            "name": portfolio["name"],
                            "investor_type": portfolio["investor_type"],
                            "sip_amount": portfolio["sip_amount"],
                            "time_horizon": portfolio["time_horizon"],
                            "review_freq": str(review_days),
                            "next_review_date": next_review,
                            "next_sip_date": next_sip,
                            "is_paper": portfolio.get("is_paper", False),
                            "portfolio_profile": portfolio.get("portfolio_profile", {})
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
                        st.dataframe(pd.DataFrame(breakdown_data), hide_index=True, width="stretch")
                        st.session_state.pending_portfolio = None
                        if portfolio.get("is_paper"):
                            st.session_state._paper_just_saved = True
                            st.session_state.sb_view_mode = "watchlist"
                            st.rerun()
                    except Exception as e:
                        st.error(f"Save failed: {e}")

        # ── Disambiguation UI (shown when awaiting user's pick) ──
        if not prompt and st.session_state.get("pending_disambiguation"):
            pd_data = st.session_state.pending_disambiguation
            with st.chat_message("user", avatar=USER_AVATAR):
                st.markdown(pd_data["original_query"])
            st.info("🔍 Multiple matches found. Which company did you mean?")
            _matches = pd_data["matches"]
            _ncols = min(len(_matches), 4)
            for _row_start in range(0, len(_matches), _ncols):
                _row_items = _matches[_row_start:_row_start + _ncols]
                _btn_cols = st.columns(len(_row_items))
                for _j, _m in enumerate(_row_items):
                    _i = _row_start + _j
                    with _btn_cols[_j]:
                        _lbl = f"{_m['name']} ({_m['ticker'].replace('.NS','').replace('.BO','')})"
                        if st.button(_lbl, key=f"disambig_{_i}", use_container_width=True):
                            _resolved = f"{pd_data['original_query']} (company: {_m['name']}, ticker: {_m['ticker']})"
                            st.session_state.pending_prompt = _resolved
                            st.session_state.pending_disambiguation = None
                            st.rerun()

        if prompt:
            # ── Fuzzy search: disambiguate before LLM call ──
            _is_builder = prompt.startswith("[BUILDER_PROFILE]")
            _is_disambiguated = "(company:" in prompt and "ticker:" in prompt
            if not _is_disambiguated and not _is_builder:
                _fz = fuzzy_search_universe(prompt, universe_df)
                _good = [m for m in _fz if m["match_score"] > 0.4]
                # Only skip disambiguation if there's EXACTLY one strong match
                _exact = [m for m in _good if m["match_score"] >= 0.95]
                if len(_exact) == 1 and len(_good) <= 1:
                    pass  # single exact match — go straight to LLM
                elif len(_good) >= 2:
                    st.session_state.pending_disambiguation = {
                        "original_query": prompt,
                        "matches": _good[:10],
                    }
                    st.rerun()

            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar=USER_AVATAR):
                st.markdown(prompt)
            with st.chat_message("assistant", avatar=AGENT_AVATAR):
                response_placeholder = st.empty()
                answer = None
                model_used = None
                with st.spinner("Routing & Analyzing..."):
                    try:
                        answer, model_used = agent_turn(prompt)
                    except Exception as e:
                        error_msg = str(e)
                        error_upper = error_msg.upper()
                        if any(err in error_upper for err in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "ALL MODELS RATE-LIMITED"]):
                            st.warning("API experiencing high demand. Using fallback system...")
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
                    st.caption(f"⚡ {model_used}")
                    st.session_state.messages.append({"role": "assistant", "content": answer, "model": model_used})
                    if st.session_state.get("pending_portfolio"):
                        st.rerun()

                    # ── Chat → Watchlist bridge: store YES tickers for buttons ──
                    if st.session_state.sb_user_id and "YES" in answer.upper():
                        _resp_tickers = set(re.findall(r'\b[A-Z][A-Z0-9&]+\.(?:NS|BO)\b', answer))
                        _disambig = re.search(r'ticker:\s*([A-Z][A-Z0-9&]+\.(?:NS|BO))', prompt)
                        if _disambig:
                            _resp_tickers.add(_disambig.group(1))

                        _answer_upper = answer.upper()
                        _yes_tickers = []
                        for _t in _resp_tickers:
                            _t_up = _t.upper()
                            if re.search(
                                rf'(?:{re.escape(_t_up)}.{{0,300}}VERDICT.*?YES)|(?:YES.{{0,300}}{re.escape(_t_up)})',
                                _answer_upper
                            ):
                                _yes_tickers.append(_t)
                        if not _yes_tickers and len(_resp_tickers) == 1:
                            _yes_tickers = list(_resp_tickers)

                        if _yes_tickers:
                            st.session_state.pending_watch_tickers = _yes_tickers

        if st.session_state.get("pending_retry"):
            if st.button("🔄 Retry last query", width="stretch"):
                st.session_state.pending_prompt = st.session_state.pop("pending_retry")
                st.rerun()

        # ── Watchlist bridge buttons (persistent across reruns) ──
        if st.session_state.get("pending_watch_tickers") and st.session_state.sb_user_id:
            _pw_sb = get_supabase()
            _pw_tickers = st.session_state.pending_watch_tickers

            try:
                _pw_ports = _pw_sb.table("portfolios").select("id").eq(
                    "user_id", st.session_state.sb_user_id
                ).execute().data or []
                _pw_port_ids = [p["id"] for p in _pw_ports]
                if _pw_port_ids:
                    _pw_held = {h["ticker"] for h in (_pw_sb.table("holdings").select(
                        "ticker"
                    ).in_("portfolio_id", _pw_port_ids).execute().data or [])}
                else:
                    _pw_held = set()
            except Exception:
                _pw_held = set()

            try:
                _pw_watched = {w["ticker"] for w in (_pw_sb.table("watchlist").select(
                    "ticker"
                ).eq("user_id", st.session_state.sb_user_id).execute().data or [])}
            except Exception:
                _pw_watched = set()

            _any_actionable = False
            for _yt in _pw_tickers:
                _bare = _yt.replace(".NS", "").replace(".BO", "")
                if _yt in _pw_held:
                    st.caption(f"✅ {_bare} — already in your portfolio")
                elif _yt in _pw_watched:
                    st.caption(f"👁 {_bare} — already on your watchlist")
                else:
                    _any_actionable = True
                    if st.button(f"👁 Watch {_bare}", key=f"watch_{_yt}", use_container_width=True):
                        _wl_row = universe_df[universe_df["ticker"] == _yt]
                        _wl_data = {
                            "user_id": st.session_state.sb_user_id,
                            "ticker": _yt,
                            "name": str(_wl_row["name"].iloc[0]) if not _wl_row.empty else _bare,
                            "score_when_added": int(_wl_row["score"].iloc[0]) if not _wl_row.empty and pd.notna(_wl_row["score"].iloc[0]) else None,
                            "quality_when_added": bool(_wl_row["quality_pass"].iloc[0]) if not _wl_row.empty and "quality_pass" in _wl_row.columns and pd.notna(_wl_row["quality_pass"].iloc[0]) else None,
                        }
                        try:
                            _pw_sb.table("watchlist").insert(_wl_data).execute()
                            st.session_state.pending_watch_tickers = [t for t in _pw_tickers if t != _yt]
                            if not st.session_state.pending_watch_tickers:
                                del st.session_state["pending_watch_tickers"]
                            st.rerun()
                        except Exception as _we:
                            st.error(f"Failed: {_we}")

            if not _any_actionable:
                st.session_state.pop("pending_watch_tickers", None)
elif st.session_state.sb_view_mode == "watchlist":
    st.markdown("### 👁 My Watchlist")

    if st.session_state.sb_user_id is None:
        st.warning("Please log in to view your watchlist.")
    else:
        _wl_tab_stocks, _wl_tab_paper = st.tabs(["📊 Stocks", "📁 Paper Portfolios"])

        with _wl_tab_stocks:
            _w_sb = get_supabase()
            try:
                _w_resp = _w_sb.table("watchlist").select("*").eq(
                    "user_id", st.session_state.sb_user_id
                ).order("added_date", desc=True).execute()
                _w_items = _w_resp.data or []
            except Exception as _w_err:
                st.error(f"Failed to load watchlist: {_w_err}")
                _w_items = []
    
            # Fetch today's watchlist alerts (auto-expire after 24h)
            _wl_alerts_by_ticker = {}
            if _w_items:
                try:
                    _wl_alert_resp = _w_sb.table("portfolio_alerts").select("*").eq(
                        "user_id", st.session_state.sb_user_id
                    ).eq("alert_date", datetime.date.today().isoformat()).in_(
                        "alert_type", ["watchlist_score_up", "watchlist_score_down",
                                       "watchlist_quality_flip", "watchlist_near_low"]
                    ).execute()
                    for _wa in (_wl_alert_resp.data or []):
                        _wl_alerts_by_ticker.setdefault(_wa["ticker"], []).append(_wa)
                except Exception:
                    pass
    
            if not _w_items:
                st.info("Your watchlist is empty. Analyze a stock in chat — if it gets a YES verdict, you'll see a Watch button.")
            else:
                for _w in _w_items:
                    _w_ticker = _w["ticker"]
                    _w_name = _w.get("name") or _w_ticker
                    _w_bare = _w_ticker.replace(".NS", "").replace(".BO", "")
                    _w_added_score = _w.get("score_when_added")
                    _w_added_quality = _w.get("quality_when_added")
                    _w_note = _w.get("note") or ""
                    _w_added_date = _w.get("added_date", "")
                    _w_days = 0
                    if _w_added_date:
                        try:
                            _w_days = (datetime.date.today() - datetime.date.fromisoformat(str(_w_added_date))).days
                        except Exception:
                            pass
    
                    # Current data from universe_df
                    _w_cur_score = "?"
                    _w_cur_quality = None
                    _w_sector = "—"
                    _w_pe = "—"
                    _w_pb = "—"
                    try:
                        _w_row = universe_df[universe_df["ticker"] == _w_ticker]
                        if not _w_row.empty:
                            if pd.notna(_w_row["score"].iloc[0]):
                                _w_cur_score = int(_w_row["score"].iloc[0])
                            if "quality_pass" in _w_row.columns and pd.notna(_w_row["quality_pass"].iloc[0]):
                                _w_cur_quality = bool(_w_row["quality_pass"].iloc[0])
                            _w_sector = str(_w_row["sector"].iloc[0]) if pd.notna(_w_row.get("sector", pd.Series([None])).iloc[0]) else "—"
                            _w_pe = round(float(_w_row["pe"].iloc[0]), 1) if pd.notna(_w_row.get("pe", pd.Series([None])).iloc[0]) else "—"
                            _w_pb = round(float(_w_row["pb"].iloc[0]), 2) if pd.notna(_w_row.get("pb", pd.Series([None])).iloc[0]) else "—"
                    except NameError:
                        pass
    
                    with st.container(border=True):
                        _wh1, _wh2 = st.columns([4, 1])
                        with _wh1:
                            # Score delta
                            _w_delta_str = ""
                            if _w_added_score is not None and _w_cur_score != "?":
                                _w_diff = _w_cur_score - _w_added_score
                                if _w_diff > 0:
                                    _w_delta_str = f"  ↑{_w_diff} since added"
                                elif _w_diff < 0:
                                    _w_delta_str = f"  ↓{abs(_w_diff)} since added"
                            st.markdown(f"**{_w_name}** ({_w_bare})")
                            st.caption(f"Score: {_w_cur_score}/4{_w_delta_str} · {_w_sector} · PE {_w_pe} · PB {_w_pb} · Watching {_w_days}d")
    
                            # Quality flip warning
                            if _w_added_quality is not None and _w_cur_quality is not None and _w_added_quality != _w_cur_quality:
                                if _w_cur_quality:
                                    st.success("Quality flipped to PASS since you added this.")
                                else:
                                    st.warning("Quality flipped to FAIL since you added this.")
    
                        with _wh2:
                            if st.button("✕ Remove", key=f"wl_rm_{_w['id']}", use_container_width=True):
                                try:
                                    _w_sb.table("watchlist").delete().eq("id", _w["id"]).execute()
                                    st.rerun()
                                except Exception as _e:
                                    st.error(f"Failed: {_e}")
                        # Daily alerts (auto-expire after 24h)
                        _card_alerts = _wl_alerts_by_ticker.get(_w_ticker, [])
                        for _ca in _card_alerts:
                            _ca_type = _ca["alert_type"]
                            _ca_headline = _ca.get("headline", "")
                            if _ca_type in ("watchlist_score_up", "watchlist_near_low"):
                                st.success(f"{_ca_headline}")
                            elif _ca_type == "watchlist_score_down":
                                st.warning(f"{_ca_headline}")
                            elif _ca_type == "watchlist_quality_flip":
                                _ca_detail = _ca.get("detail") or {}
                                if isinstance(_ca_detail, str):
                                    import json as _json
                                    try:
                                        _ca_detail = _json.loads(_ca_detail)
                                    except Exception:
                                        _ca_detail = {}
                                if _ca_detail.get("current"):
                                    st.success(f"{_ca_headline}")
                                else:
                                    st.warning(f"{_ca_headline}")
    
                        
                        # Editable note
                        _w_new_note = st.text_input(
                            "Note", value=_w_note, key=f"wl_note_{_w['id']}",
                            placeholder="e.g. Waiting for PE to drop below 12",
                            label_visibility="collapsed"
                        )
                        if _w_new_note != _w_note:
                            try:
                                _w_sb.table("watchlist").update({"note": _w_new_note}).eq("id", _w["id"]).execute()
                            except Exception:
                                pass


        with _wl_tab_paper:
            if st.session_state.get("_paper_just_saved"):
                st.success("Paper portfolio saved! Track its performance here.")
                st.session_state.pop("_paper_just_saved", None)

            _pp_sb = get_supabase()
            try:
                _pp_resp = _pp_sb.table("portfolios").select("*").eq(
                    "user_id", st.session_state.sb_user_id
                ).eq("is_paper", True).order("created_at", desc=True).execute()
                _pp_ports = _pp_resp.data or []
            except Exception as _pp_err:
                st.error(f"Failed to load paper portfolios: {_pp_err}")
                _pp_ports = []

            if not _pp_ports:
                st.info("No paper portfolios yet. Use 🏗️ Build Portfolio and check 'Watch only' to create one.")
            else:
                for _pp in _pp_ports:
                    with st.container(border=True):
                        st.markdown(f"**👁 {_pp['name']}**")

                        try:
                            _pp_h_resp = _pp_sb.table("holdings").select("*").eq(
                                "portfolio_id", _pp["id"]
                            ).execute()
                            _pp_holdings = _pp_h_resp.data or []
                        except Exception:
                            _pp_holdings = []

                        if _pp_holdings:
                            _pp_enriched = enrich_holdings_live(_pp_holdings, cache_key=f"paper_{_pp['id']}")
                            _pp_invested = sum(h.get("shares", 0) * h.get("price_at_entry", 0) for h in _pp_enriched)
                            _pp_current = sum(h.get("current_value", 0) for h in _pp_enriched)
                            _pp_ret = ((_pp_current - _pp_invested) / _pp_invested * 100) if _pp_invested > 0 else 0

                            _pm1, _pm2, _pm3 = st.columns(3)
                            with _pm1:
                                st.metric("Invested", f"₹{_pp_invested:,.0f}")
                            with _pm2:
                                st.metric("Current Value", f"₹{_pp_current:,.0f}")
                            with _pm3:
                                st.metric("Return", f"{_pp_ret:+.1f}%")

                            _pp_rows = []
                            for _h in _pp_enriched:
                                _h_entry = _h.get("price_at_entry", 0)
                                _h_now = _h.get("current_price", 0)
                                _h_sh = _h.get("shares", 0)
                                _h_pnl = (_h_now - _h_entry) * _h_sh
                                _h_ret = ((_h_now - _h_entry) / _h_entry * 100) if _h_entry > 0 else 0
                                _pp_rows.append({
                                    "Stock": _h.get("name") or _h.get("ticker", ""),
                                    "Shares": _h_sh,
                                    "Entry": f"₹{_h_entry:,.2f}",
                                    "Now": f"₹{_h_now:,.2f}",
                                    "P&L": f"₹{_h_pnl:,.0f}",
                                    "Return": f"{_h_ret:+.1f}%",
                                })
                            st.dataframe(pd.DataFrame(_pp_rows), hide_index=True, use_container_width=True)

                            _pp_profile = _pp.get("portfolio_profile") or {}
                            if isinstance(_pp_profile, str):
                                try:
                                    _pp_profile = json.loads(_pp_profile)
                                except Exception:
                                    _pp_profile = {}
                            _pp_cap_parts = [_pp.get("created_at", "")[:10]]
                            if _pp.get("investor_type"):
                                _pp_cap_parts.append(_pp["investor_type"])
                            if _pp.get("time_horizon"):
                                _pp_cap_parts.append(f"{_pp['time_horizon']} horizon")
                            if _pp.get("target_amount"):
                                _pp_cap_parts.append(f"Goal: ₹{_pp['target_amount']:,.0f}")
                            st.caption(" · ".join(_pp_cap_parts))
                        else:
                            st.caption("No holdings recorded.")

                        # ── Action buttons ──
                        _pp_c1, _pp_c2 = st.columns(2)
                        with _pp_c1:
                            if st.button("🚀 Make This Real", key=f"make_real_{_pp['id']}", use_container_width=True):
                                st.session_state[f"confirm_real_{_pp['id']}"] = True
                        with _pp_c2:
                            if st.button("🗑️ Delete", key=f"del_paper_{_pp['id']}", use_container_width=True, type="secondary"):
                                st.session_state[f"confirm_del_paper_{_pp['id']}"] = True

                        # ── Make This Real confirmation ──
                        if st.session_state.get(f"confirm_real_{_pp['id']}"):
                            st.warning("This converts to a real portfolio at **current market prices** (not original paper prices). Tracking restarts from today.")
                            _rc1, _rc2 = st.columns(2)
                            with _rc1:
                                if st.button("Yes, make it real", key=f"real_yes_{_pp['id']}", use_container_width=True):
                                    try:
                                        for _rh in _pp_holdings:
                                            try:
                                                _rh_price = yf.Ticker(_rh.get("ticker", "")).fast_info.last_price or _rh.get("price_at_entry", 0)
                                            except Exception:
                                                _rh_price = _rh.get("price_at_entry", 0)
                                            _pp_sb.table("holdings").update({
                                                "price_at_entry": round(_rh_price, 2),
                                                "sip_amount_inr": round(_rh.get("shares", 0) * _rh_price, 2),
                                            }).eq("id", _rh["id"]).execute()
                                        _pp_sb.table("portfolios").update({"is_paper": False}).eq("id", _pp["id"]).execute()
                                        try:
                                            _pp_sb.table("portfolio_history").delete().eq("portfolio_id", _pp["id"]).execute()
                                        except Exception:
                                            pass
                                        st.session_state.pop(f"confirm_real_{_pp['id']}", None)
                                        st.success("Portfolio is now real! Find it in My Portfolios.")
                                        st.rerun()
                                    except Exception as _re:
                                        st.error(f"Failed: {_re}")
                            with _rc2:
                                if st.button("Cancel", key=f"real_no_{_pp['id']}", use_container_width=True):
                                    st.session_state.pop(f"confirm_real_{_pp['id']}", None)
                                    st.rerun()

                        # ── Delete confirmation ──
                        if st.session_state.get(f"confirm_del_paper_{_pp['id']}"):
                            st.warning("Delete this paper portfolio? This cannot be undone.")
                            _dc1, _dc2 = st.columns(2)
                            with _dc1:
                                if st.button("Yes, delete", key=f"del_yes_{_pp['id']}", use_container_width=True):
                                    try:
                                        _pp_sb.table("holdings").delete().eq("portfolio_id", _pp["id"]).execute()
                                        _pp_sb.table("portfolios").delete().eq("id", _pp["id"]).execute()
                                        st.session_state.pop(f"confirm_del_paper_{_pp['id']}", None)
                                        st.rerun()
                                    except Exception as _de:
                                        st.error(f"Failed: {_de}")
                            with _dc2:
                                if st.button("Cancel", key=f"del_no_{_pp['id']}", use_container_width=True):
                                    st.session_state.pop(f"confirm_del_paper_{_pp['id']}", None)
                                    st.rerun()

elif st.session_state.sb_view_mode == "import":
    st.markdown("### 📥 Import Your Existing Portfolio")
    st.caption("Onboard your current holdings to analyze them using the Kordent framework.")
    
    if st.session_state.sb_user_id is None:
        st.warning("Please log in via the sidebar to save and analyze an existing portfolio.")
    else:
        # 1. Meta Information Gathering
        with st.container(border=True):
            st.markdown("#### ⚙️ Portfolio Metadata & Goals")
            p_name = st.text_input("Portfolio Name", placeholder="e.g., My Main Brokerage")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                total_invested = st.number_input("Total Amount Invested Till Date (INR)", min_value=0, value=100000, step=5000)
                sip_amt = st.number_input("Current Monthly SIP Amount (INR)", min_value=0, value=10000, step=1000)
            with col_m2:
                inv_type = st.selectbox("Investment Goal Profile", ["defensive", "balanced", "enterprising"], index=1)
                horizon = st.selectbox("Time Horizon", ["short", "medium", "long"], index=1)
        
        # 2. Dynamic Asset Adder (Searchable Dropdown)
        if "import_holding_pool" not in st.session_state:
            st.session_state.import_holding_pool = []

        with st.container(border=True):
            st.markdown("#### 📊 Add Your Stock Holdings")
            
            # Create a list of "Company Name (TICKER)" from your existing universe_df
            stock_options = [
                f"{row.get('name', row['ticker'])} ({row['ticker']})" 
                for _, row in universe_df.iterrows()
            ]
            

            selected_stock = st.selectbox("🔍 Search & Select Company (Type to filter)", stock_options)
            
            # Preview CSV price as default, but let user override with actual buy price
            _ticker_preview = selected_stock.split("(")[-1].replace(")", "").strip()
            _row_preview = universe_df[universe_df["ticker"] == _ticker_preview]
            _default_price = float(_row_preview["price"].iloc[0]) if len(_row_preview) and pd.notna(_row_preview["price"].iloc[0]) else 0.0
            
            col_sh, col_px = st.columns(2)
            with col_sh:
                shares_to_add = st.number_input("Shares Owned", min_value=1, value=10, step=1)
            with col_px:
                price_paid = st.number_input("Avg Buy Price (₹)", min_value=0.01, value=_default_price, format="%.2f")
            
            if st.button("➕ Add to List", width="stretch"):
                ticker_resolved = _ticker_preview
                company_name = selected_stock.split(" (")[0]
                
                st.session_state.import_holding_pool.append({
                    "ticker": ticker_resolved,
                    "name": company_name,
                    "shares": shares_to_add,
                    "price": price_paid
                })
                st.success(f"Added {ticker_resolved} to your staging list.")
                st.rerun()

        # 3. Present Staging List Table
        if st.session_state.import_holding_pool:
            st.markdown("#### Staging Review Table")
            staging_df = pd.DataFrame(st.session_state.import_holding_pool)
            st.dataframe(staging_df[["name", "ticker", "shares", "price"]], hide_index=True, width="stretch")
            
            if st.button("🗑️ Clear List"):
                st.session_state.import_holding_pool = []
                st.rerun()
                
            # 4. Save and Trigger Instant Review
            if st.button("💾 Save & Run Instant Analysis", width="stretch"):
                if not p_name:
                    st.error("Please provide a name for this portfolio.")
                else:
                    try:
                        sb = get_supabase()
                        review_days = 90 if horizon == "medium" else (180 if horizon == "long" else 30)
                        next_review = (datetime.date.today() + datetime.timedelta(days=review_days)).isoformat()
                        next_sip = (datetime.date.today() + datetime.timedelta(days=30)).isoformat()
                        
                        _port_data = {
                            "user_id": st.session_state.sb_user_id,
                            "name": p_name,
                            "investor_type": inv_type,
                            "sip_amount": sip_amt,
                            "time_horizon": horizon,
                            "review_freq": str(review_days),
                            "next_review_date": next_review,
                            "next_sip_date": next_sip,
                            "is_paper": False
                        }
                        
                        port_resp = sb.table("portfolios").insert(_port_data).execute()
                        portfolio_id = port_resp.data[0]["id"]
                        
                        for s in st.session_state.import_holding_pool:
                            row = universe_df[universe_df["ticker"] == s["ticker"]]
                            pe = float(row["pe"].iloc[0]) if len(row) and pd.notna(row["pe"].iloc[0]) else None
                            roe = float(row["roe_y0"].iloc[0]) if len(row) and "roe_y0" in row.columns and pd.notna(row["roe_y0"].iloc[0]) else None
                            score = int(row["score"].iloc[0]) if len(row) and pd.notna(row["score"].iloc[0]) else None
                            sect = str(row["sector"].iloc[0]) if len(row) and "sector" in row.columns and pd.notna(row["sector"].iloc[0]) else "Unknown"
                            
                            sb.table("holdings").insert({
                                "portfolio_id": portfolio_id, 
                                "ticker": s["ticker"], 
                                "name": s["name"],
                                "sector": sect, 
                                "allocation_pct": 0, 
                                "shares": s["shares"],
                                "sip_amount_inr": round(s["shares"] * s["price"], 2), 
                                "price_at_entry": s["price"],
                                "pe_at_entry": pe, 
                                "roe_at_entry": roe, 
                                "score_at_entry": score
                            }).execute()
                        
                        st.session_state.import_holding_pool = []
                        st.session_state[f"auto_trigger_review_{portfolio_id}"] = True
                        st.session_state.sb_view_mode = "portfolios"
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Failed to onboard portfolio: {e}")

elif st.session_state.sb_view_mode == "builder":
    st.markdown("### 🏗️ Build Your Portfolio")
    st.caption("Answer a few simple questions — no financial jargon, we promise.")

    if st.session_state.sb_user_id is None:
        st.warning("Please log in via the sidebar to build a portfolio.")
    else:
        with st.form("portfolio_builder_form"):
            # Q1 — Monthly investment
            _b_sip = st.number_input(
                "💰 How much can you invest every month? (₹)",
                min_value=500, max_value=10000000, value=5000, step=500,
                help="Start small — you can always increase later.",
            )

            st.divider()

            # Q2 — Time horizon
            _b_horizon_label = st.radio(
                "⏳ How long do you plan to keep investing?",
                options=["1–3 years", "3–7 years", "7+ years"],
                index=1,
                help="Longer horizons let compounding do the heavy lifting.",
            )

            st.divider()

            # Q3 — Goal (optional)
            st.markdown("**🎯 Do you have a specific savings target?** *(optional — leave amount at 0 to skip)*")
            _b_goal_cols = st.columns(2)
            with _b_goal_cols[0]:
                _b_target_amt = st.number_input(
                    "Target amount (₹)", min_value=0, value=0, step=100000,
                    help="Leave at 0 if you don't have a number in mind.",
                )
            with _b_goal_cols[1]:
                _b_target_dt = st.date_input(
                    "By when?",
                    value=datetime.date.today() + datetime.timedelta(days=365 * 5),
                    min_value=datetime.date.today() + datetime.timedelta(days=180),
                    help="Only matters if you set a target amount.",
                )

            st.divider()

            # Q4 — Risk tolerance
            _b_risk_resp = st.radio(
                "📉 If your portfolio dropped 20% in a month, would you:",
                options=[
                    "Buy more — it's on sale!",
                    "Hold and wait for recovery",
                    "Sell some to sleep better",
                ],
                index=1,
            )

            st.divider()

            # Q5 — Sector exclusions
            _b_avoid = st.multiselect(
                "🚫 Any industries you want to avoid?",
                options=[
                    "Energy", "Basic Materials", "Utilities",
                    "Real Estate", "Financial Services", "Industrials",
                ],
                default=[],
                help="Select sectors to stay away from. Leave empty for no preference.",
            )

            st.divider()

            # Q6 — Income vs growth
            _b_pref_resp = st.radio(
                "🎚️ What matters more to you?",
                options=[
                    "Steady dividends and low risk",
                    "Maximum growth, even if it's bumpy",
                ],
                index=1,
            )

            st.divider()

            # Q7 — Paper portfolio toggle
            _b_is_paper = st.checkbox(
                "👁 Watch only — don't invest yet (paper portfolio)",
                value=False,
                help="Track how this portfolio would perform without putting real money in.",
            )

            _b_submitted = st.form_submit_button("Build My Portfolio →", use_container_width=True)

        if _b_submitted:
            # ── Map responses to profile dict ──
            _risk_map = {
                "Buy more — it's on sale!": "aggressive",
                "Hold and wait for recovery": "moderate",
                "Sell some to sleep better": "conservative",
            }
            _pref_map = {
                "Steady dividends and low risk": "income",
                "Maximum growth, even if it's bumpy": "growth",
            }
            _horizon_map = {"1–3 years": "short", "3–7 years": "medium", "7+ years": "long"}

            _b_risk = _risk_map.get(_b_risk_resp, "moderate")
            _b_pref = _pref_map.get(_b_pref_resp, "growth")
            _b_time = _horizon_map.get(_b_horizon_label, "medium")

            # investor_type from risk × preference
            if _b_risk == "conservative" or _b_pref == "income":
                _b_inv_type = "defensive"
            elif _b_risk == "aggressive" and _b_pref == "growth":
                _b_inv_type = "enterprising"
            else:
                _b_inv_type = "balanced"

            # review cadence from investor_type
            _rev_map = {"defensive": ("passive", 180), "balanced": ("moderate", 90), "enterprising": ("active", 60)}
            _b_rev_freq, _b_rev_days = _rev_map.get(_b_inv_type, ("moderate", 90))

            # Override time_horizon from target_date when goal is set
            if _b_target_amt > 0:
                _yrs = ((_b_target_dt - datetime.date.today()).days) / 365.25
                if _yrs <= 3:
                    _b_time = "short"
                elif _yrs <= 7:
                    _b_time = "medium"
                else:
                    _b_time = "long"

            _b_profile = {
                "sip_amount": _b_sip,
                "target_amount": _b_target_amt if _b_target_amt > 0 else None,
                "target_date": _b_target_dt.isoformat() if _b_target_amt > 0 else None,
                "risk": _b_risk,
                "avoid_sectors": _b_avoid,
                "preference": _b_pref,
                "investor_type": _b_inv_type,
                "time_horizon": _b_time,
                "review_freq": _b_rev_freq,
                "review_days": _b_rev_days,
                "is_paper": _b_is_paper,
            }
            st.session_state.builder_profile = _b_profile

            # ── Build the chat prompt that triggers stock selection ──
            _goal_line = f"\n- Goal: ₹{_b_target_amt:,.0f} by {_b_target_dt.isoformat()}" if _b_target_amt > 0 else ""
            _avoid_line = f"\n- Avoid sectors: {', '.join(_b_avoid)}" if _b_avoid else ""
            _paper_line = "\n- Mode: Paper portfolio (watch only)" if _b_is_paper else ""

            st.session_state.pending_prompt = (
                f"[BUILDER_PROFILE]\n"
                f"Build me a portfolio with these preferences:\n"
                f"- Monthly SIP: ₹{_b_sip:,}\n"
                f"- Time horizon: {_b_time} ({_b_horizon_label})\n"
                f"- Investor type: {_b_inv_type}\n"
                f"- Risk tolerance: {_b_risk}\n"
                f"- Preference: {_b_pref}\n"
                f"- Review: every {_b_rev_days} days ({_b_rev_freq})"
                f"{_goal_line}{_avoid_line}{_paper_line}"
            )
            st.session_state.sb_view_mode = "chat"
            st.rerun()

elif st.session_state.sb_view_mode == "portfolios":
    st.markdown("### 📁 My Portfolios")
    sb = get_supabase()
    try:
        port_resp = sb.table("portfolios").select("*").eq(
            "user_id", st.session_state.sb_user_id
        ).order("created_at", desc=True).execute()
        portfolios = [p for p in (port_resp.data or []) if not p.get("is_paper")]
    except Exception as e:
        st.error(f"Failed to load portfolios: {e}")
        portfolios = []

    if not portfolios:
        st.info("No saved portfolios yet. Click 🏗️ Build Portfolio in the sidebar to get started!")
    else:
        for port in portfolios:
            with st.container(border=True):
                st.markdown(f"**{port['name']}**")

                # ── Alert Banner ──
                try:
                    alerts_resp = sb.table("portfolio_alerts").select("*").eq(
                        "portfolio_id", port["id"]
                    ).eq("is_read", False).order("created_at", desc=True).execute()
                    port_alerts = alerts_resp.data

                    # Also fetch broadcast alerts (new_entry — no portfolio_id)
                    try:
                        broadcast_resp = sb.table("portfolio_alerts").select("*").is_(
                            "portfolio_id", "null"
                        ).eq("is_read", False).order("created_at", desc=True).limit(5).execute()
                        if broadcast_resp.data:
                            seen_tickers = {a["ticker"] for a in port_alerts}
                            for ba in broadcast_resp.data:
                                if ba["ticker"] not in seen_tickers:
                                    port_alerts.append(ba)
                    except Exception:
                        pass
                except Exception:
                    port_alerts = []

                if port_alerts:
                    for alert in port_alerts:
                        a_type = alert["alert_type"]
                        a_id = alert["id"]
                        detail = alert.get("detail") or {}
                        if isinstance(detail, str):
                            import json as _json
                            try:
                                detail = _json.loads(detail)
                            except Exception:
                                detail = {}

                        if a_type == "danger":
                            st.error(f"🛡️ **{alert['headline']}**")
                            
                            # Replaced expander with a permanently open, bordered container
                            with st.container(border=True):
                                st.markdown("##### Defend Position")
                                
                                ticker = alert.get("ticker", "")
                                # Fetch holding for this portfolio directly from Supabase
                                try:
                                    h_resp = sb.table("holdings").select("*").eq("portfolio_id", port["id"]).eq("ticker", ticker).execute()
                                    h_match = h_resp.data[0] if h_resp.data else None
                                except Exception:
                                    h_match = None

                                if h_match:
                                    max_shares = h_match.get("shares", 0)
                                    sell_qty = st.number_input(
                                        f"Shares to sell (of {max_shares})",
                                        min_value=0, max_value=max_shares, value=max_shares,
                                        key=f"defend_qty_{a_id}"
                                    )
                                    c1, c2 = st.columns(2)
                                    with c1:
                                        if st.button("🛡️ Confirm Sell", key=f"defend_confirm_{a_id}", width="stretch"):
                                            if sell_qty > 0:
                                                new_shares = max_shares - sell_qty
                                                if new_shares <= 0:
                                                    sb.table("holdings").delete().eq("id", h_match["id"]).execute()
                                                else:
                                                    new_invested = new_shares * h_match.get("price_at_entry", 0)
                                                    sb.table("holdings").update({
                                                        "shares": new_shares,
                                                        "sip_amount_inr": round(new_invested, 2)
                                                    }).eq("id", h_match["id"]).execute()
                                                sb.table("portfolio_alerts").update({"is_read": True}).eq("id", a_id).execute()
                                                st.success(f"Sold {sell_qty} shares of {ticker}.")
                                                st.rerun()
                                    with c2:
                                        if st.button("Dismiss", key=f"defend_dismiss_{a_id}", width="stretch"):
                                            sb.table("portfolio_alerts").update({"is_read": True}).eq("id", a_id).execute()
                                            st.rerun()
                                else:
                                    st.caption("Holding not found — may have been sold already.")
                                    if st.button("Dismiss", key=f"defend_dismiss_nf_{a_id}"):
                                        sb.table("portfolio_alerts").update({"is_read": True}).eq("id", a_id).execute()
                                        st.rerun()

                        elif a_type == "opportunity":
                            st.success(f"⚡ **{alert['headline']}**")

                            with st.container(border=True):
                                ticker = alert.get("ticker", "")
                                opp_name = detail.get("name", ticker)
                                live_price = float(detail.get("price", 0)) if detail.get("price") else 0.0
                                act_now = detail.get("act_now", False)
                                budget_left = float(port.get("sip_budget_remaining") or 0)

                                suggested_qty = int(budget_left // live_price) if live_price > 0 and budget_left > 0 else 0

                                if act_now and suggested_qty > 0:
                                    st.markdown(f"Budget left this month: **₹{budget_left:,.0f}** · Price: **₹{live_price:,.2f}** · Suggested: **{suggested_qty} shares** (~₹{suggested_qty * live_price:,.0f})")
                                elif live_price > budget_left and budget_left > 0:
                                    st.info(f"One share costs ₹{live_price:,.0f} but only ₹{budget_left:,.0f} left in this month's opportunity budget. Consider this at your next review.")
                                elif budget_left <= 0:
                                    st.info("This month's opportunity budget is used up. Noted for your weekly summary.")
                                else:
                                    st.markdown(f"Price: **₹{live_price:,.2f}**")

                                col_sh, col_px = st.columns(2)
                                with col_sh:
                                    buy_qty = st.number_input(
                                        "Shares to buy",
                                        min_value=0, value=suggested_qty, key=f"buy_qty_{a_id}"
                                    )
                                with col_px:
                                    buy_price = st.number_input(
                                        "Price per share (₹)",
                                        min_value=0.0, value=live_price,
                                        format="%.2f", key=f"buy_price_{a_id}"
                                    )

                                c1, c2 = st.columns(2)
                                with c1:
                                    if st.button("✅ Bought", key=f"buy_confirm_{a_id}", use_container_width=True):
                                        if buy_qty > 0 and buy_price > 0:
                                            try:
                                                invested = round(buy_qty * buy_price, 2)
                                                sb.table("holdings").insert({
                                                    "portfolio_id": port["id"],
                                                    "ticker": ticker,
                                                    "name": opp_name,
                                                    "sector": detail.get("sector", ""),
                                                    "allocation_pct": 0,
                                                    "shares": buy_qty,
                                                    "sip_amount_inr": invested,
                                                    "price_at_entry": round(buy_price, 2),
                                                    "score_at_entry": detail.get("score"),
                                                }).execute()
                                                new_budget = max(0, budget_left - invested)
                                                sb.table("portfolios").update({
                                                    "sip_budget_remaining": round(new_budget, 2)
                                                }).eq("id", port["id"]).execute()
                                                sb.table("portfolio_alerts").update({"is_read": True}).eq("id", a_id).execute()
                                                st.success(f"Tracked {buy_qty} shares of {opp_name}. Budget remaining: ₹{new_budget:,.0f}")
                                                st.rerun()
                                            except Exception as e:
                                                st.error(f"Failed: {e}")
                                        else:
                                            st.warning("Enter shares and price.")
                                with c2:
                                    if st.button("✗ Skip", key=f"buy_skip_{a_id}", use_container_width=True):
                                        sb.table("portfolio_alerts").update({"is_read": True}).eq("id", a_id).execute()
                                        st.rerun()

                        
                        elif a_type == "overvalued":
                            st.warning(f"📈 **{alert['headline']}**")
                            with st.container(border=True):
                                pe_val = detail.get("pe")
                                pb_val = detail.get("pb")
                                metrics = []
                                if pe_val: metrics.append(f"PE {pe_val:.1f}")
                                if pb_val: metrics.append(f"PB {pb_val:.1f}")
                                st.markdown(f"Graham's margin of safety is thinning ({', '.join(metrics)}). This doesn't mean sell — but it's worth reviewing whether the price still makes sense.")
                                c1, c2 = st.columns(2)
                                with c1:
                                    if st.button("📋 Review Now", key=f"overval_review_{a_id}", use_container_width=True):
                                        sb.table("portfolio_alerts").update({"is_read": True}).eq("id", a_id).execute()
                                        st.session_state["active_tab"] = "review"
                                        st.rerun()
                                with c2:
                                    if st.button("✗ Dismiss", key=f"overval_dismiss_{a_id}", use_container_width=True):
                                        sb.table("portfolio_alerts").update({"is_read": True}).eq("id", a_id).execute()
                                        st.rerun()

                        elif a_type == "sector_headwind":
                            st.warning(f"🌊 **{alert['headline']}**")
                            with st.container(border=True):
                                idx_ret = detail.get("index_return_pct", 0)
                                weight = detail.get("portfolio_weight_pct", 0)
                                sector = detail.get("sector", "")
                                st.markdown(f"The {sector} sector index dropped {idx_ret:.1f}% this month and makes up {weight:.0f}% of this portfolio. Keep an eye on it — if fundamentals hold, sector dips can be buying opportunities.")
                                if st.button("✗ Dismiss", key=f"headwind_dismiss_{a_id}", use_container_width=True):
                                    sb.table("portfolio_alerts").update({"is_read": True}).eq("id", a_id).execute()
                                    st.rerun()

                        elif a_type == "goal_drift":
                            st.error(f"🎯 **{alert['headline']}**")
                            with st.container(border=True):
                                actual = detail.get("actual_cagr_pct", 0)
                                needed = detail.get("needed_cagr_pct", 0)
                                months = detail.get("months_remaining", 0)
                                st.markdown(f"Your portfolio is growing at {actual:.1f}% but needs {needed:.1f}% to hit your goal in {months} months. Consider increasing your SIP or reviewing your picks — but don't chase risk.")
                                if st.button("✗ Dismiss", key=f"goaldrift_dismiss_{a_id}", use_container_width=True):
                                    sb.table("portfolio_alerts").update({"is_read": True}).eq("id", a_id).execute()
                                    st.rerun()

                        elif a_type == "new_entry":
                            st.info(f"🆕 **{alert['headline']}**")
                            with st.container(border=True):
                                ne_name = detail.get("name", alert.get("ticker", ""))
                                ne_score = detail.get("score", 0)
                                ne_sector = detail.get("sector", "N/A")
                                ne_pe = detail.get("pe")
                                pe_str = f" · PE {ne_pe:.1f}" if ne_pe else ""
                                st.markdown(f"**{ne_name}** just appeared on our radar with a score of {ne_score}/4. Sector: {ne_sector}{pe_str}. Worth a closer look if it fits your portfolio.")
                                if st.button("✗ Dismiss", key=f"newentry_dismiss_{a_id}", use_container_width=True):
                                    sb.table("portfolio_alerts").update({"is_read": True}).eq("id", a_id).execute()
                                    st.rerun()

                
                # Check if this specific portfolio is in "SIP edit mode"
                is_editing_sip = st.session_state.get(f"edit_sip_{port['id']}", False)

                if is_editing_sip:
                    # Edit Mode UI
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        new_sip = st.number_input(
                            "New SIP Amount (₹)", 
                            value=int(port.get('sip_amount', 0)), 
                            step=1000, 
                            key=f"new_sip_input_{port['id']}", 
                            label_visibility="collapsed"
                        )
                    with col2:
                        if st.button("💾 Save", key=f"save_sip_{port['id']}", width="stretch"):
                            try:
                                sb.table("portfolios").update({"sip_amount": new_sip}).eq("id", port["id"]).execute()
                                st.session_state[f"edit_sip_{port['id']}"] = False
                                st.success("Updated!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed: {e}")
                    with col3:
                        if st.button("❌", key=f"cancel_sip_{port['id']}", width="stretch"):
                            st.session_state[f"edit_sip_{port['id']}"] = False
                            st.rerun()
                    
                    # Show the rest of the caption without the SIP amount while editing
                    st.caption(
                        f"Created: {port['created_at'][:10]} · "
                        f"{port.get('investor_type', '—')} · "
                        f"{port.get('time_horizon', '—')} horizon · "
                        f"Review: every {port.get('review_freq', '90')} days · "
                        f"Next: {port.get('next_review_date', '—')}"
                    )
                else:
                    # Normal Mode UI with Edit Button
                    col_cap, col_btn = st.columns([11, 1])
                    with col_cap:
                        st.caption(
                            f"Created: {port['created_at'][:10]} · "
                            f"{port.get('investor_type', '—')} · "
                            f"**₹{port.get('sip_amount', 0):,}/mo** · "
                            f"{port.get('time_horizon', '—')} horizon · "
                            f"Review: every {port.get('review_freq', '90')} days · "
                            f"Next: {port.get('next_review_date', '—')}"
                        )
                    with col_btn:
                        if st.button("✏️", key=f"trigger_edit_sip_{port['id']}", help="Edit SIP Amount"):
                            st.session_state[f"edit_sip_{port['id']}"] = True
                            st.rerun()

                try:
                    hold_resp = sb.table("holdings").select("*").eq("portfolio_id", port["id"]).execute()
                    holdings = hold_resp.data
                except Exception:
                    holdings = []

                if holdings:
                    display_holdings = enrich_holdings_live(holdings, cache_key=str(port["id"]))
                    hold_df = pd.DataFrame(display_holdings)
                    display_cols = {
                        "name": "Stock", "ticker": "Ticker", "sector": "Sector", "shares": "Shares",
                        "price_at_entry": "Entry ₹", "current_price": "CMP ₹",
                        "sip_amount_inr": "Invested", "current_value": "Value",
                        "allocation_pct": "Alloc %", "score_at_entry": "Score",
                    }
                    available = {k: v for k, v in display_cols.items() if k in hold_df.columns}
                    st.dataframe(hold_df[list(available.keys())].rename(columns=available), hide_index=True, width="stretch")
                else:
                    st.caption("No holdings found.")

                # ── Portfolio Growth Chart ──
                try:
                    hist_resp = sb.table("portfolio_history").select(
                        "date, total_value, daily_return_pct, nifty_value"
                    ).eq("portfolio_id", port["id"]).order("date").execute()
                    hist_data = hist_resp.data

                    if hist_data and len(hist_data) >= 2:
                        hist_df = pd.DataFrame(hist_data)
                        hist_df["date"] = pd.to_datetime(hist_df["date"])
                        hist_df = hist_df.set_index("date")

                        # Normalize both to % return from day 1
                        port_base = hist_df["total_value"].iloc[0]
                        chart_data = pd.DataFrame(index=hist_df.index)
                        chart_data["Portfolio"] = ((hist_df["total_value"] / port_base) - 1) * 100 if port_base > 0 else 0

                        has_nifty = "nifty_value" in hist_df.columns and hist_df["nifty_value"].notna().sum() >= 2
                        if has_nifty:
                            nifty_base = hist_df["nifty_value"].dropna().iloc[0]
                            chart_data["Nifty 50"] = ((hist_df["nifty_value"] / nifty_base) - 1) * 100 if nifty_base > 0 else 0

                        st.markdown("**Growth vs Market (%)**")
                        st.line_chart(chart_data, width="stretch", color=["#1D4ED8", "#9CA3AF"][:len(chart_data.columns)])

                        # Summary
                        first_val = hist_df["total_value"].iloc[0]
                        last_val = hist_df["total_value"].iloc[-1]
                        port_growth = ((last_val - first_val) / first_val) * 100 if first_val > 0 else 0
                        days_tracked = (hist_df.index[-1] - hist_df.index[0]).days

                        summary = f"Portfolio: ₹{first_val:,.0f} → ₹{last_val:,.0f} ({port_growth:+.1f}%)"
                        if has_nifty:
                            nifty_first = hist_df["nifty_value"].dropna().iloc[0]
                            nifty_last = hist_df["nifty_value"].dropna().iloc[-1]
                            nifty_growth = ((nifty_last - nifty_first) / nifty_first) * 100 if nifty_first > 0 else 0
                            alpha = port_growth - nifty_growth
                            summary += f" · Nifty: {nifty_growth:+.1f}% · Alpha: {alpha:+.1f}%"
                        summary += f" · {days_tracked} days"
                        st.caption(summary)
                    elif hist_data and len(hist_data) == 1:
                        st.caption("📈 Growth chart available after 2+ days of tracking.")
                except Exception:
                    pass  # Fail silently if history table doesn't exist yet

                # ── PDF Export (two-step: generate then download) ──
                report_key = f"report_ready_{port['id']}"

                if st.session_state.get(report_key):
                    st.download_button(
                        label="⬇️ Download Report",
                        data=st.session_state[report_key],
                        file_name=f"Kordent_{re.sub(r'[^a-zA-Z0-9]', '_', port.get('name', 'portfolio'))}_{datetime.date.today().isoformat()}.pdf",
                        mime="application/pdf",
                        key=f"pdf_download_{port['id']}",
                        width="stretch",
                    )
                    if st.button("✕ Clear", key=f"pdf_clear_{port['id']}"):
                        del st.session_state[report_key]
                        st.rerun()
                else:
                    if st.button("📄 Generate Report", key=f"pdf_gen_{port['id']}", width="stretch"):
                        with st.spinner("Building report — analyzing holdings against book principles..."):
                            try:
                                hold_for_pdf = sb.table("holdings").select("*").eq("portfolio_id", port["id"]).execute().data or []
                                hold_for_pdf = enrich_holdings_live(hold_for_pdf, cache_key=f"pdf_{port['id']}")
                                hist_for_pdf = sb.table("portfolio_history").select("*").eq("portfolio_id", port["id"]).order("date").execute().data or []
                                alerts_for_pdf = sb.table("portfolio_alerts").select("*").eq("portfolio_id", port["id"]).eq("is_read", False).execute().data or []

                                chart_buf = generate_portfolio_chart(hist_for_pdf)
                                narrative = generate_portfolio_narrative(port, hold_for_pdf, collection)
                                pdf_bytes = generate_portfolio_pdf(port, hold_for_pdf, hist_for_pdf, alerts_for_pdf, chart_buf, narrative)
                                st.session_state[report_key] = pdf_bytes
                                st.rerun()
                            except Exception as e:
                                st.error(f"Report generation failed: {e}")

                # ── Standalone Health Check (when not in review) ──
                _review_imminent = False
                if port.get("next_review_date"):
                    try:
                        _rd = datetime.date.fromisoformat(str(port["next_review_date"]))
                        _review_imminent = (_rd - datetime.date.today()).days <= 7
                    except (ValueError, TypeError):
                        pass

                if not st.session_state.get(f"review_data_{port['id']}") and not _review_imminent:
                    hc_key = f"health_check_{port['id']}"
                    if st.session_state.get(hc_key):
                        hc = st.session_state[hc_key]
                        with st.container(border=True):
                            st.markdown("**Health Check Results**")
                            d_score = hc["diversification_score"]
                            d_color = "🟢" if d_score >= 70 else "🟡" if d_score >= 40 else "🔴"
                            st.metric("Diversification Score", f"{d_color} {d_score}/100")
                            m1, m2, m3 = st.columns(3)
                            with m1:
                                st.metric("Avg Beta", hc["avg_beta"] or "N/A")
                            with m2:
                                pe_val = hc["avg_pe_vs_historical"]
                                pe_label = f"{pe_val:+.1f}%" if pe_val is not None else "N/A"
                                st.metric("PE vs History", pe_label)
                            with m3:
                                high_val = hc["avg_pct_from_52w_high"]
                                high_label = f"{high_val:.1f}%" if high_val is not None else "N/A"
                                st.metric("From 52w High", high_label)
                            sector_dist = hc["sector_distribution"]
                            if sector_dist:
                                sector_df = pd.DataFrame([
                                    {"Sector": s, "Stocks": c, "Weight": f"{c/sum(sector_dist.values())*100:.0f}%"}
                                    for s, c in sorted(sector_dist.items(), key=lambda x: -x[1])
                                ])
                                st.dataframe(sector_df, hide_index=True, width="stretch")
                            for w in hc.get("warnings", []):
                                st.warning(w)
                            if hc.get("narrative"):
                                st.markdown("---")
                                st.markdown(hc["narrative"])

                        # ── Actionable recommendations ──
                        hc_actions = hc.get("actions", [])
                        if hc_actions:
                            st.markdown("---")
                            st.markdown("**Execute Recommendations**")
                            for ai, act in enumerate(hc_actions):
                                act_type = act.get("type", "")
                                act_ticker = act.get("ticker", "")
                                act_reason = act.get("reason", "")

                                if act_type == "reduce":
                                    target_pct = act.get("target_alloc_pct", 0)
                                    if st.button(
                                        f"📉 Reduce {act_ticker} to {target_pct}% — {act_reason}",
                                        key=f"hc_reduce_{port['id']}_{ai}",
                                        width="stretch"
                                    ):
                                        try:
                                            sb.table("holdings").update(
                                                {"allocation_pct": target_pct}
                                            ).eq("portfolio_id", port["id"]).eq("ticker", act_ticker).execute()
                                            st.success(f"Updated {act_ticker} allocation to {target_pct}%.")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Failed: {e}")

                                elif act_type == "sell":
                                    sell_shares = act.get("shares", 0)
                                    label = f"🔴 Sell all {act_ticker}" if sell_shares == 0 else f"🔴 Sell {sell_shares} shares of {act_ticker}"
                                    if st.button(
                                        f"{label} — {act_reason}",
                                        key=f"hc_sell_{port['id']}_{ai}",
                                        width="stretch"
                                    ):
                                        try:
                                            if sell_shares == 0:
                                                sb.table("holdings").delete().eq(
                                                    "portfolio_id", port["id"]
                                                ).eq("ticker", act_ticker).execute()
                                                st.success(f"Removed {act_ticker} from portfolio.")
                                            else:
                                                h_resp = sb.table("holdings").select("*").eq(
                                                    "portfolio_id", port["id"]
                                                ).eq("ticker", act_ticker).execute()
                                                if h_resp.data:
                                                    h = h_resp.data[0]
                                                    new_shares = max(0, h["shares"] - sell_shares)
                                                    if new_shares == 0:
                                                        sb.table("holdings").delete().eq("id", h["id"]).execute()
                                                    else:
                                                        new_invested = new_shares * h.get("price_at_entry", 0)
                                                        sb.table("holdings").update({
                                                            "shares": new_shares,
                                                            "sip_amount_inr": round(new_invested, 2)
                                                        }).eq("id", h["id"]).execute()
                                                st.success(f"Sold {sell_shares} shares of {act_ticker}.")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Failed: {e}")

                                elif act_type == "add":
                                    btn_key = f"hc_action_{port['id']}_{ai}"
                                    action_msg_key = f"hc_action_msg_{port['id']}"
                                    act_name = act.get("name", act_ticker)
                                    act_sector = act.get("sector", "")
                                    act_score = act.get("score", 0)
                                    suggested_pct = act.get("suggested_alloc_pct", 10)
                                    add_state_key = f"hc_add_form_{port['id']}_{ai}"

                                    if st.button(
                                        f"Add {act_name} ({act_ticker}) — {act_reason}",
                                        key=btn_key, width="stretch"
                                    ):
                                        st.session_state[add_state_key] = True

                                    if st.session_state.get(add_state_key):
                                        with st.container(border=True):
                                            # Fetch live price
                                            try:
                                                _live = yf.Ticker(act_ticker).fast_info
                                                _live_price = round(float(_live.last_price), 2)
                                            except Exception:
                                                _live_price = 100.0

                                            # Suggest shares from SIP and allocation
                                            _sip = port.get("sip_amount", 10000)
                                            _budget = _sip * suggested_pct / 100
                                            _suggested_qty = max(1, int(_budget / _live_price)) if _live_price > 0 else 1

                                            st.caption(
                                                f"Sector: {act_sector} · Score: {act_score}/4 · "
                                                f"PE: {act.get('pe', 'N/A')} · Price: ₹{_live_price:,.2f} · "
                                                f"Budget ({suggested_pct}% of ₹{_sip:,}): ₹{_budget:,.0f}"
                                            )
                                            ac1, ac2 = st.columns(2)
                                            with ac1:
                                                add_qty = st.number_input(
                                                    "Shares to buy", min_value=1, value=_suggested_qty,
                                                    key=f"hc_add_qty_{port['id']}_{ai}"
                                                )
                                            with ac2:
                                                add_price = st.number_input(
                                                    "Price per share (₹)", min_value=0.01,
                                                    value=_live_price,
                                                    format="%.2f", key=f"hc_add_price_{port['id']}_{ai}"
                                                )
                                            bc1, bc2 = st.columns(2)
                                            with bc1:
                                                if st.button("Confirm Add", key=f"hc_add_confirm_{port['id']}_{ai}", width="stretch"):
                                                    try:
                                                        invested = round(add_qty * add_price, 2)
                                                        sb.table("holdings").insert({
                                                            "portfolio_id": port["id"],
                                                            "ticker": act_ticker,
                                                            "name": act_name,
                                                            "sector": act_sector,
                                                            "allocation_pct": suggested_pct,
                                                            "shares": add_qty,
                                                            "sip_amount_inr": invested,
                                                            "price_at_entry": round(add_price, 2),
                                                            "score_at_entry": act_score,
                                                        }).execute()

                                                        # Normalize all allocations to sum to 100%
                                                        all_h = sb.table("holdings").select("id, allocation_pct").eq(
                                                            "portfolio_id", port["id"]
                                                        ).execute().data or []
                                                        if all_h:
                                                            raw_total = sum(h["allocation_pct"] for h in all_h)
                                                            non_zero = [h for h in all_h if h["allocation_pct"] > 0]
                                                            if len(non_zero) < len(all_h) / 2:
                                                                # Most are zero — reset to equal allocation
                                                                equal_pct = round(100 / len(all_h), 1)
                                                                for h in all_h:
                                                                    sb.table("holdings").update(
                                                                        {"allocation_pct": equal_pct}
                                                                    ).eq("id", h["id"]).execute()
                                                            elif raw_total > 0:
                                                                for h in all_h:
                                                                    normalized = round(h["allocation_pct"] / raw_total * 100, 1)
                                                                    sb.table("holdings").update(
                                                                        {"allocation_pct": normalized}
                                                                    ).eq("id", h["id"]).execute()

                                                        st.session_state[action_msg_key] = f"Added {act_name}. All allocations normalized to 100%."
                                                        del st.session_state[add_state_key]
                                                        st.rerun()
                                                    except Exception as e:
                                                        st.error(f"Failed: {e}")
                                            with bc2:
                                                if st.button("Cancel", key=f"hc_add_cancel_{port['id']}_{ai}", width="stretch"):
                                                    del st.session_state[add_state_key]
                                                    st.rerun()

                                elif act_type == "investigate":
                                    btn_key = f"hc_action_{port['id']}_{ai}"
                                    inv_key = f"hc_inv_result_{port['id']}_{ai}"
                                    if st.button(
                                        f"Investigate {act_ticker} — {act_reason}",
                                        key=btn_key, width="stretch"
                                    ):
                                        with st.spinner(f"Investigating {act_ticker}..."):
                                            try:
                                                client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
                                                stock_data = get_stock_data(act_ticker)
                                                book_data = search_book(f"{act_reason} investment risk")
                                                inv_prompt = (
                                                    f"You are Kordent's analyst investigating a specific concern about {act_ticker}.\n\n"
                                                    f"CONCERN: {act_reason}\n\n"
                                                    f"STOCK DATA:\n{json.dumps(stock_data, indent=2, default=str)}\n\n"
                                                    f"BOOK CONTEXT:\n{book_data.get('passages', '')[:800]}\n\n"
                                                    f"Write a focused 150-word investigation: what does the data show about this concern? "
                                                    f"Is the concern valid? What should the investor do? Cite book principles."
                                                )
                                                last_good = st.session_state.get("last_working_model")
                                                models = [last_good] + [m for m in FREE_MODELS if m != last_good] if last_good else FREE_MODELS
                                                for model in models:
                                                    try:
                                                        resp = client.models.generate_content(model=model, contents=inv_prompt)
                                                        st.session_state[inv_key] = resp.text
                                                        st.session_state.last_working_model = model
                                                        break
                                                    except Exception as e:
                                                        error_msg = str(e).upper()
                                                        if any(err in error_msg for err in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500"]):
                                                            continue
                                                        break
                                            except Exception as e:
                                                st.session_state[inv_key] = f"Investigation failed: {e}"
                                        st.rerun()

                                    if st.session_state.get(inv_key):
                                        with st.container(border=True):
                                            st.markdown(st.session_state[inv_key])
                                            if st.button("Dismiss", key=f"inv_dismiss_{port['id']}_{ai}"):
                                                del st.session_state[inv_key]
                                                st.rerun()
                        if st.button("✕ Close", key=f"hc_close_{port['id']}"):
                            del st.session_state[hc_key]
                            st.rerun()
                    else:
                        if st.button("🩺 Health Check", key=f"hc_btn_{port['id']}", width="stretch"):
                            with st.spinner("Diagnosing portfolio against book principles..."):
                                try:
                                    hc_holdings = sb.table("holdings").select("*").eq("portfolio_id", port["id"]).execute().data or []
                                    hc_holdings = enrich_holdings_live(hc_holdings, cache_key=str(port["id"]))
                                    hc_result = generate_health_check(port, hc_holdings, universe_df, collection)
                                    if hc_result:
                                        st.session_state[hc_key] = hc_result
                                        st.rerun()
                                except Exception as e:
                                    st.error(f"Health check failed: {e}")

                
                # --- NEW COLLISION PROTOCOL & SIP MODULE ---
                today = datetime.date.today()
                _auto_key = f"auto_trigger_review_{port['id']}"
                _auto_run = st.session_state.pop(_auto_key, False)
                
                # 1. Calculate Review Clock
                review_date = None
                rev_due_days = 0
                if port.get("next_review_date"):
                    try:
                        review_date = datetime.date.fromisoformat(str(port["next_review_date"]))
                        rev_due_days = (review_date - today).days
                    except (ValueError, TypeError):
                        pass

                # 2. Calculate SIP Clock
                sip_date_str = port.get("next_sip_date")
                sip_due_days = 0
                if sip_date_str:
                    try:
                        sip_due_days = (datetime.date.fromisoformat(str(sip_date_str)) - today).days
                    except (ValueError, TypeError):
                        pass

                _review_clicked = False
                
                if holdings:
                    if _auto_run or rev_due_days <= 0:
                        # STATE 1: PRIORITY OVERRIDE (Review is Due)
                        if not _auto_run:
                            st.warning(f"📅 Review overdue by {abs(rev_due_days)} days! You must evaluate fundamentals before deploying more capital.")
                        _review_clicked = st.button("🔄 Review Portfolio", key=f"review_{port['id']}", width="stretch")
                    
                    elif sip_due_days <= 0:
                        # STATE 2: MECHANICAL SIP DEPLOYMENT
                        st.success(f"💰 Monthly SIP of ₹{port.get('sip_amount', 0):,} is due!")
                        if st.button("💵 Deploy SIP", key=f"deploy_sip_{port['id']}", width="stretch"):
                            st.session_state[f"active_sip_{port['id']}"] = True
                        
                        if st.session_state.get(f"active_sip_{port['id']}"):
                            with st.container(border=True):
                                st.markdown(f"**Mechanical SIP Deployment (₹{port.get('sip_amount', 0):,})**")
                                sip_stocks = []
                                for h in display_holdings:
                                    if h.get("allocation_pct", 0) > 0:
                                        sip_stocks.append({
                                            "ticker": h["ticker"],
                                            "name": h.get("name", h["ticker"]),
                                            "allocation_pct": h["allocation_pct"],
                                            "price": h.get("current_price", h.get("price_at_entry", 1)),
                                            "id": h["id"],
                                            "old_shares": h["shares"],
                                            "old_entry": h.get("price_at_entry", 1)
                                        })
                                
                                if sip_stocks:
                                    allocated, unallocated = allocate_shares(sip_stocks, port.get("sip_amount", 0))
                                    for a in allocated:
                                        c1, c2, c3 = st.columns([2,1,1])
                                        with c1: st.markdown(f"**{a['name']}**")
                                        with c2: st.number_input("Shares", value=a["shares"], key=f"sip_q_{port['id']}_{a['ticker']}")
                                        with c3: st.number_input("Price (₹)", value=float(a["price"]), key=f"sip_p_{port['id']}_{a['ticker']}")
                                    
                                    st.caption(f"Unallocated: ₹{unallocated:,.0f} (not enough for another full share)")
                                    
                                    bc1, bc2 = st.columns(2)
                                    with bc1:
                                        if st.button("✅ Confirm Purchase", key=f"conf_sip_{port['id']}", width="stretch"):
                                            try:
                                                for a in allocated:
                                                    buy_q = st.session_state[f"sip_q_{port['id']}_{a['ticker']}"]
                                                    buy_p = st.session_state[f"sip_p_{port['id']}_{a['ticker']}"]
                                                    if buy_q > 0:
                                                        new_total_shares = a["old_shares"] + buy_q
                                                        old_value = a["old_shares"] * a["old_entry"]
                                                        new_value = buy_q * buy_p
                                                        new_avg_price = (old_value + new_value) / new_total_shares
                                                        sb.table("holdings").update({
                                                            "shares": new_total_shares,
                                                            "price_at_entry": round(new_avg_price, 2),
                                                            "sip_amount_inr": round(new_total_shares * new_avg_price, 2)
                                                        }).eq("id", a["id"]).execute()
                                                
                                                # Bump the SIP timer by 30 days
                                                new_sip_date = (today + datetime.timedelta(days=30)).isoformat()
                                                sb.table("portfolios").update({"next_sip_date": new_sip_date}).eq("id", port["id"]).execute()
                                                
                                                del st.session_state[f"active_sip_{port['id']}"]
                                                st.success("SIP Deployed Successfully!")
                                                st.rerun()
                                            except Exception as e:
                                                st.error(f"Failed: {e}")
                                    with bc2:
                                        if st.button("Cancel", key=f"canc_sip_{port['id']}", width="stretch"):
                                            del st.session_state[f"active_sip_{port['id']}"]
                                            st.rerun()
                                else:
                                    st.warning("No active holdings found to allocate to.")

                    else:
                        # STATE 3: NO ACTION REQUIRED (Anti-Tinkering)
                        st.caption(f"📅 Next Review due in {rev_due_days} days ({review_date.isoformat() if review_date else '—'})")
                        if sip_date_str:
                            st.caption(f"💰 Next SIP due in {sip_due_days} days ({sip_date_str})")
                        else:
                            st.caption(f"💰 Next SIP due in N/A")

                        if _review_clicked or _auto_run:
                            with st.spinner("Analyzing holdings with market context and book philosophy..."):
                                enriched = build_review_context(holdings, port)
                                llm_recs = generate_review_recommendations(
                                    enriched, port.get("investor_type", "balanced"),
                                    port.get("time_horizon", "medium")
                                    port
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

                                    
                                    # ── DETERMINISTIC RED FLAG OVERRIDE ──
                                    # The LLM is not trusted on quality failures.
                                    # If has_red_flags is True, force SELL ALL regardless.
                                    if h["has_red_flags"] and "SELL ALL" not in action:
                                        action = f"🔴 SELL ALL ({h['shares']})"
                                        sell_qty = h["shares"]
                                        reasoning = (
                                            f"OVERRIDE: Earnings quality RED FLAGS detected — "
                                            f"{', '.join(h['quality_flags'])}. "
                                            f"Graham warns against value traps where reported earnings "
                                            f"are inflated by non-recurring items. Forced SELL."
                                        )
                                        confidence = "high"

                                    # ── DETERMINISTIC BELOW-THRESHOLD OVERRIDE ──
                                    # Score 0 = no thesis. Score 1 without Graham = below buy threshold.
                                    # Both warrant exit. Score 1 WITH Graham = deep value exception, hold.
                                    if h["now_score"] == 0 and "SELL" not in action:
                                        action = f"🔴 SELL ALL ({h['shares']})"
                                        sell_qty = h["shares"]
                                        reasoning = (
                                            f"Score is 0/4 — all frameworks fail. "
                                            f"No investment thesis exists. Redeploy capital."
                                        )
                                        confidence = "high"
                                    elif h["now_score"] == 1 and "SELL" not in action:
                                        # Check if the lone pass is Graham (deep value exception)
                                        urow_check = universe_df[universe_df["ticker"] == h["ticker"]]
                                        graham_alive = (
                                            len(urow_check) > 0 
                                            and "graham_pass" in urow_check.columns 
                                            and urow_check["graham_pass"].iloc[0] == True
                                        )
                                        if not graham_alive:
                                            action = f"🔴 SELL ALL ({h['shares']})"
                                            sell_qty = h["shares"]
                                            reasoning = (
                                                f"Score dropped to 1/4 without Graham pass — "
                                                f"below the 2/4 buy threshold and no deep value "
                                                f"exception applies. Thesis has eroded."
                                            )
                                            confidence = "high"

                                    
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
                                        "Score": f"{h['entry_score']}→{h['now_score']}", "Trend": h.get("score_trend", "—"), "Action": action,
                                        "_reasoning": reasoning, "_confidence": confidence,
                                        "_market_note": mkt_note, "_book_passage": h["book_passage"],
                                        "_sell_qty": sell_qty, "_holding_id": h["holding_id"],
                                        "_ticker": h["ticker"], "_sector": h["sector"],
                                        "_entry_price": h["entry_price"], "_now_price": h["now_price"],
                                    })

                                # Auto-run health check during review
                                hc_result = generate_health_check(port, enrich_holdings_live(holdings, cache_key=str(port["id"])), universe_df, collection)

                                st.session_state[f"review_data_{port['id']}"] = {
                                    "rows": review_rows, "total_entry": total_entry,
                                    "total_current": total_current, "holdings": holdings,
                                    "enriched": enriched, "health_check": hc_result,
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
                    st.dataframe(display_df, hide_index=True, width="stretch")


                    # Per-stock reasoning with quality data
                    enriched_data = review_state.get("enriched", [])
                    for r in review_rows:
                        if "SELL" in r["Action"]:
                            st.error(f"**{r['Stock']}** — {r['Action']}\n\n{r['_reasoning']}")
                        elif "BUY" in r["Action"]:
                            st.success(f"**{r['Stock']}** — {r['Action']}\n\n{r['_reasoning']}")
                        else:
                            st.info(f"**{r['Stock']}** — {r['Action']}\n\n{r['_reasoning']}")
                        matching = next((h for h in enriched_data if h.get("ticker") == r["_ticker"]), None)
                        if matching:
                            flags = matching.get('quality_flags', 'N/A')
                            ccr = matching.get('cash_conversion', 'N/A')
                            red = matching.get('has_red_flags', False)
                            roe_t = matching.get('roe_trend', [])
                            st.markdown(
                                f"<details><summary style='cursor:pointer;color:#6B7280;font-size:0.82rem;'>"
                                f"Quality Data: {r['Stock']}</summary>"
                                f"<p style='color:#6B7280;font-size:0.8rem;margin:4px 0;'>"
                                f"Flags: {flags}<br>"
                                f"Cash conversion: {ccr} · Red flags: {red}<br>"
                                f"ROE trend: {roe_t}</p></details>",
                                unsafe_allow_html=True
                            )

                    # ── Health Check (inline during review) ──
                    hc = review_state.get("health_check")
                    if hc:
                        with st.container(border=True):
                            st.markdown("**Health Check Results**")
                            d_score = hc["diversification_score"]
                            d_color = "🟢" if d_score >= 70 else "🟡" if d_score >= 40 else "🔴"
                            st.metric("Diversification Score", f"{d_color} {d_score}/100")

                            m1, m2, m3 = st.columns(3)
                            with m1:
                                st.metric("Avg Beta", hc["avg_beta"] or "N/A")
                            with m2:
                                pe_val = hc["avg_pe_vs_historical"]
                                pe_label = f"{pe_val:+.1f}%" if pe_val is not None else "N/A"
                                st.metric("PE vs History", pe_label)
                            with m3:
                                high_val = hc["avg_pct_from_52w_high"]
                                high_label = f"{high_val:.1f}%" if high_val is not None else "N/A"
                                st.metric("From 52w High", high_label)

                            sector_dist = hc["sector_distribution"]
                            if sector_dist:
                                sector_df = pd.DataFrame([
                                    {"Sector": s, "Stocks": c, "Weight": f"{c/sum(sector_dist.values())*100:.0f}%"}
                                    for s, c in sorted(sector_dist.items(), key=lambda x: -x[1])
                                ])
                                st.dataframe(sector_df, hide_index=True, width="stretch")

                            for w in hc.get("warnings", []):
                                st.warning(w)

                            if hc.get("narrative"):
                                st.markdown("---")
                                st.markdown(hc["narrative"])

                        # ── Actionable recommendations ──
                        hc_actions = hc.get("actions", [])
                        if hc_actions:
                            st.markdown("---")
                            st.markdown("**Execute Recommendations**")
                            for ai, act in enumerate(hc_actions):
                                act_type = act.get("type", "")
                                act_ticker = act.get("ticker", "")
                                act_reason = act.get("reason", "")

                                if act_type == "reduce":
                                    target_pct = act.get("target_alloc_pct", 0)
                                    if st.button(
                                        f"📉 Reduce {act_ticker} to {target_pct}% — {act_reason}",
                                        key=f"hc_reduce_{port['id']}_{ai}",
                                        width="stretch"
                                    ):
                                        try:
                                            sb.table("holdings").update(
                                                {"allocation_pct": target_pct}
                                            ).eq("portfolio_id", port["id"]).eq("ticker", act_ticker).execute()
                                            st.success(f"Updated {act_ticker} allocation to {target_pct}%.")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Failed: {e}")

                                elif act_type == "sell":
                                    sell_shares = act.get("shares", 0)
                                    label = f"🔴 Sell all {act_ticker}" if sell_shares == 0 else f"🔴 Sell {sell_shares} shares of {act_ticker}"
                                    if st.button(
                                        f"{label} — {act_reason}",
                                        key=f"hc_sell_{port['id']}_{ai}",
                                        width="stretch"
                                    ):
                                        try:
                                            if sell_shares == 0:
                                                sb.table("holdings").delete().eq(
                                                    "portfolio_id", port["id"]
                                                ).eq("ticker", act_ticker).execute()
                                                st.success(f"Removed {act_ticker} from portfolio.")
                                            else:
                                                h_resp = sb.table("holdings").select("*").eq(
                                                    "portfolio_id", port["id"]
                                                ).eq("ticker", act_ticker).execute()
                                                if h_resp.data:
                                                    h = h_resp.data[0]
                                                    new_shares = max(0, h["shares"] - sell_shares)
                                                    if new_shares == 0:
                                                        sb.table("holdings").delete().eq("id", h["id"]).execute()
                                                    else:
                                                        new_invested = new_shares * h.get("price_at_entry", 0)
                                                        sb.table("holdings").update({
                                                            "shares": new_shares,
                                                            "sip_amount_inr": round(new_invested, 2)
                                                        }).eq("id", h["id"]).execute()
                                                st.success(f"Sold {sell_shares} shares of {act_ticker}.")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Failed: {e}")

                                elif act_type == "investigate":
                                    btn_key = f"hc_action_{port['id']}_{ai}"
                                    inv_key = f"hc_inv_result_{port['id']}_{ai}"
                                    if st.button(
                                        f"Investigate {act_ticker} — {act_reason}",
                                        key=btn_key, width="stretch"
                                    ):
                                        with st.spinner(f"Investigating {act_ticker}..."):
                                            try:
                                                client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
                                                stock_data = get_stock_data(act_ticker)
                                                book_data = search_book(f"{act_reason} investment risk")
                                                inv_prompt = (
                                                    f"You are Kordent's analyst investigating a specific concern about {act_ticker}.\n\n"
                                                    f"CONCERN: {act_reason}\n\n"
                                                    f"STOCK DATA:\n{json.dumps(stock_data, indent=2, default=str)}\n\n"
                                                    f"BOOK CONTEXT:\n{book_data.get('passages', '')[:800]}\n\n"
                                                    f"Write a focused 150-word investigation: what does the data show about this concern? "
                                                    f"Is the concern valid? What should the investor do? Cite book principles."
                                                )
                                                last_good = st.session_state.get("last_working_model")
                                                models = [last_good] + [m for m in FREE_MODELS if m != last_good] if last_good else FREE_MODELS
                                                for model in models:
                                                    try:
                                                        resp = client.models.generate_content(model=model, contents=inv_prompt)
                                                        st.session_state[inv_key] = resp.text
                                                        st.session_state.last_working_model = model
                                                        break
                                                    except Exception as e:
                                                        error_msg = str(e).upper()
                                                        if any(err in error_msg for err in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500"]):
                                                            continue
                                                        break
                                            except Exception as e:
                                                st.session_state[inv_key] = f"Investigation failed: {e}"
                                        st.rerun()

                                    if st.session_state.get(inv_key):
                                        with st.container(border=True):
                                            st.markdown(st.session_state[inv_key])
                                            if st.button("Dismiss", key=f"inv_dismiss_{port['id']}_{ai}"):
                                                del st.session_state[inv_key]
                                                st.rerun()

                    # ── Update form — ALL holdings ──
                    sell_stocks = [(i, r) for i, r in enumerate(review_rows) if "SELL" in r["Action"]]

                    st.markdown("---")

                    # Review is now strictly analytical. SIP is handled mechanically elsewhere.
                    # We only deploy freed capital from sells during a review.
                    cycle_amount = 0

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
                    unallocated_sip = cycle_amount
                    if sip_stocks and cycle_amount > 0:
                        allocated, unallocated_sip = allocate_shares(sip_stocks, cycle_amount)
                        for a in allocated:
                            sip_alloc[a["ticker"]] = a["shares"]
                        st.caption(f"💰 This cycle ({review_days} days): ₹{cycle_amount:,} to invest — suggested shares pre-filled below")
                        if unallocated_sip > 0:
                            st.caption(f"₹{unallocated_sip:,.0f} unallocatable (not enough for another share)")
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
                            st.number_input(
                                f"🔻 {r['Stock']} — shares sold (of {r['Shares']})",
                                min_value=0, max_value=r["Shares"], value=0,
                                key=f"manual_sold_{port['id']}_{h_id}"
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
                            total_repl_budget = freed + unallocated_sip
                            st.markdown(f"**Replacement candidates** (₹{freed:,.0f} freed + ₹{unallocated_sip:,.0f} SIP = **₹{total_repl_budget:,.0f}** to deploy)")
                            cand_df = pd.DataFrame(candidates)
                            cand_display = cand_df[["name", "ticker", "sector", "price", "score", "pe", "roe_pct"]].rename(columns={
                                "name": "Stock", "ticker": "Ticker", "sector": "Sector",
                                "price": "Price", "score": "Score", "pe": "P/E", "roe_pct": "ROE %"
                            })
                            st.dataframe(cand_display, hide_index=True, width="stretch")
                            st.caption("Shares pre-filled from total budget. Set to 0 to skip a stock.")
                            per_slot = total_repl_budget / len(candidates) if candidates else 0
                            for c in candidates:
                                suggested = max(1, int(per_slot // c["price"])) if c["price"] > 0 else 0
                                col_name, col_qty, col_px = st.columns([2, 1, 1])
                                with col_name:
                                    st.markdown(f"**{c['name'].strip()}** ({c['ticker']})")
                                with col_qty:
                                    st.number_input("Shares", min_value=0, value=suggested, key=f"repl_qty_{port['id']}_{c['ticker']}")
                                with col_px:
                                    st.number_input("Price (₹)", min_value=0.0, value=float(c["price"]), format="%.2f", key=f"repl_px_{port['id']}_{c['ticker']}")
                            # ── Budget tracker ──
                            spent = 0
                            for c in candidates:
                                rq = st.session_state.get(f"repl_qty_{port['id']}_{c['ticker']}", 0)
                                rp = st.session_state.get(f"repl_px_{port['id']}_{c['ticker']}", 0.0)
                                if rq > 0:
                                    spent += rq * rp
                            remaining = total_repl_budget - spent
                            if remaining >= 0:
                                st.caption(f"💰 Budget: ₹{total_repl_budget:,.0f} — Allocated: ₹{spent:,.0f} = ₹{remaining:,.0f} remaining")
                            else:
                                st.warning(f"Over-allocated by ₹{abs(remaining):,.0f}. Budget: ₹{total_repl_budget:,.0f}, Allocated: ₹{spent:,.0f}")

                    # ── Single update button ──
                    if st.button("✅ Portfolio Updated", key=f"apply_{port['id']}", width="stretch"):
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
                                manual_sold = st.session_state.get(f"manual_sold_{port['id']}_{h_id}", 0)
                                if manual_sold > 0:
                                    new_shares = r["Shares"] - manual_sold
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
                                qty = st.session_state.get(f"repl_qty_{port['id']}_{c['ticker']}", 0)
                                px = st.session_state.get(f"repl_px_{port['id']}_{c['ticker']}", 0.0)
                                if qty > 0 and px > 0:
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
