"""
DEEP METRICS MODULE
===================
Computes all 90 new columns from the 7-book extraction (Sprint 6).
Called by universe_updater.py after basic fundamentals are fetched.

Books: Graham, Greenblatt, Dorsey, Lynch, Buffett, Schilit, Mulford
Categories: Balance Sheet, Earnings Quality, Valuation, Growth, Moat,
            Dividend, Management, Manipulation Flags, Classification
"""

import math
import pandas as pd
from datetime import datetime, timezone

# ─── Constants ───
INDIA_10Y_BOND_RATE = 7.0        # Hardcoded, stable
ADEQUATE_SIZE_INR = 2_000_000_000  # ₹200Cr (PPP-adjusted from Graham's $100M)
COST_OF_CAPITAL_PROXY = 12.0     # Rough WACC for Indian equities


# ─── Helpers ───

def _sf(val, default=None):
    """Safely convert to float. Returns default for NaN/None/inf."""
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _bs_row(sheet, names, col, default=None):
    """Try multiple row names on a balance sheet / financials DataFrame."""
    if sheet is None or sheet.empty:
        return default
    for name in names:
        if name in sheet.index:
            val = _sf(sheet.loc[name, col])
            if val is not None:
                return val
    return default


def _get_yearly_values(sheet, row_names, cols):
    """Extract a list of yearly values for a given row, most-recent-first."""
    vals = []
    for col in cols:
        vals.append(_bs_row(sheet, row_names, col))
    return vals


def _pct_growth(new, old):
    """Compute percentage growth. Returns None if not computable."""
    if new is None or old is None or old == 0:
        return None
    return (new / old - 1) * 100


# ─── 1. BALANCE SHEET HEALTH (14 columns) ───

def compute_balance_sheet(data, info, bs, shares):
    """Compute balance sheet health metrics."""
    price = _sf(data.get("price"))
    revenue_y0 = _sf(data.get("revenue_y0"))

    # D1: Adequate Size
    data["graham_adequate_size"] = bool(revenue_y0 and revenue_y0 >= ADEQUATE_SIZE_INR)

    # D2: Current Ratio Pass (≥ 2.0 for defensive)
    cr = _sf(data.get("current_ratio"))
    data["graham_current_ratio_pass"] = bool(cr and cr >= 2.0)

    if bs is None or bs.empty or shares is None or shares <= 0:
        # Set all BS-dependent columns to None
        for col in ["graham_ltd_vs_nca", "graham_net_current_assets", "graham_ncav_per_share",
                     "graham_ncav_ratio", "graham_bvps", "graham_price_to_ntav",
                     "graham_net_cash", "lynch_net_cash_per_share",
                     "dorsey_financial_leverage", "dorsey_interest_coverage",
                     "dorsey_quick_ratio", "dorsey_clean_balance_sheet"]:
            data[col] = None
        return

    cols = sorted(bs.columns)
    latest = cols[-1]

    ca = _bs_row(bs, ["Current Assets", "Total Current Assets"], latest)
    cl = _bs_row(bs, ["Current Liabilities", "Total Current Liabilities"], latest)
    total_assets = _bs_row(bs, ["Total Assets"], latest)
    total_debt = _sf(info.get("totalDebt"))
    total_cash = _sf(info.get("totalCash"))
    equity = _bs_row(bs, ["Stockholders Equity", "Total Stockholder Equity", "Total Equity Gross Minority Interest"], latest)
    ltd = _bs_row(bs, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"], latest)
    inventory = _bs_row(bs, ["Inventory"], latest)
    intangibles = _bs_row(bs, ["Intangible Assets", "Other Intangible Assets"], latest, 0)
    goodwill = _bs_row(bs, ["Goodwill"], latest, 0)
    bvps = _sf(info.get("bookValue"))

    nca = (ca - cl) if (ca is not None and cl is not None) else None

    # D3: Long-Term Debt vs NCA
    if ltd is not None and nca is not None:
        data["graham_ltd_vs_nca"] = bool(ltd <= nca)
    else:
        data["graham_ltd_vs_nca"] = None

    # F1: Net Current Assets
    data["graham_net_current_assets"] = round(nca, 2) if nca is not None else None

    # N1: NCAV per share
    total_liab = _bs_row(bs, ["Total Liabilities Net Minority Interest", "Total Liab"], latest)
    if ca is not None and total_liab is not None:
        ncav = ca - total_liab
        data["graham_ncav_per_share"] = round(ncav / shares, 2)
    else:
        data["graham_ncav_per_share"] = None

    # N2: NCAV Ratio
    ncav_ps = _sf(data.get("graham_ncav_per_share"))
    if ncav_ps and ncav_ps > 0 and price:
        data["graham_ncav_ratio"] = round(price / ncav_ps, 2)
    else:
        data["graham_ncav_ratio"] = None

    # F3: Book Value per Share
    data["graham_bvps"] = round(bvps, 2) if bvps else None

    # E5: Price to Net Tangible Assets
    if total_assets is not None and total_liab is not None:
        ntav = total_assets - (intangibles or 0) - (goodwill or 0) - total_liab
        ntav_ps = ntav / shares if shares > 0 else None
        if ntav_ps and ntav_ps > 0 and price:
            data["graham_price_to_ntav"] = round(price / ntav_ps, 2)
        else:
            data["graham_price_to_ntav"] = None
    else:
        data["graham_price_to_ntav"] = None

    # I6: Net Cash
    if total_cash is not None and total_debt is not None:
        net_cash = total_cash - total_debt
        data["graham_net_cash"] = round(net_cash, 2)
        data["lynch_net_cash_per_share"] = round(net_cash / shares, 2)
    else:
        data["graham_net_cash"] = None
        data["lynch_net_cash_per_share"] = None

    # F1 Dorsey: Financial Leverage
    if total_assets and equity and equity > 0:
        data["dorsey_financial_leverage"] = round(total_assets / equity, 2)
    else:
        data["dorsey_financial_leverage"] = None

    # F4: Quick Ratio
    if ca is not None and cl is not None and cl > 0:
        inv = inventory or 0
        data["dorsey_quick_ratio"] = round((ca - inv) / cl, 2)
    else:
        data["dorsey_quick_ratio"] = None

    # T5: Clean Balance Sheet
    de = _sf(data.get("de"))
    data["dorsey_clean_balance_sheet"] = bool(de is not None and de <= 100)  # de is in %, ≤100% = D/E ≤ 1.0

    # F2: Interest Coverage — computed in earnings quality (needs EBIT from income stmt)
    data["dorsey_interest_coverage"] = None  # placeholder, filled later


# ─── 2. EARNINGS QUALITY (12 columns) ───

def compute_earnings_quality(data, info, income_stmt, cashflow, bs, shares):
    """Compute earnings quality metrics."""
    # Gather NI values
    ni = [_sf(data.get(f"net_income_y{i}")) for i in range(4)]
    valid_ni = [v for v in ni if v is not None]

    # D4: Earnings Stability
    if len(valid_ni) >= 4:
        data["graham_earnings_stable_4y"] = all(v > 0 for v in valid_ni)
    else:
        data["graham_earnings_stable_4y"] = False  # Fail conservatively

    # Q1: Average EPS
    if valid_ni and shares and shares > 0:
        avg_ni = sum(valid_ni) / len(valid_ni)
        data["graham_avg_eps_4y"] = round(avg_ni / shares, 2)
    else:
        data["graham_avg_eps_4y"] = None

    # Q2: EPS Coefficient of Variation
    if len(valid_ni) >= 3 and shares and shares > 0:
        eps_vals = [v / shares for v in valid_ni]
        mean_eps = sum(eps_vals) / len(eps_vals)
        if mean_eps != 0:
            std_eps = (sum((e - mean_eps) ** 2 for e in eps_vals) / len(eps_vals)) ** 0.5
            data["graham_eps_cv"] = round(abs(std_eps / mean_eps), 4)
        else:
            data["graham_eps_cv"] = None
    else:
        data["graham_eps_cv"] = None

    # D6: EPS Growth (3Y-avg method)
    if ni[0] is not None and ni[1] is not None and ni[2] is not None and ni[3] is not None:
        avg_recent = (ni[0] + ni[1]) / 2
        avg_older = (ni[2] + ni[3]) / 2
        if avg_older > 0:
            data["graham_eps_growth_pct_4y"] = round((avg_recent / avg_older - 1) * 100, 2)
        else:
            data["graham_eps_growth_pct_4y"] = None
    else:
        data["graham_eps_growth_pct_4y"] = None

    # ── Cash flow based quality metrics ──
    if cashflow is not None and not cashflow.empty:
        cf_cols = sorted(cashflow.columns)
        latest_cf = cf_cols[-1]

        ocf_vals = _get_yearly_values(cashflow,
            ["Operating Cash Flow", "Total Cash From Operating Activities", "Cash Flow From Continuing Operating Activities"],
            cf_cols[-4:] if len(cf_cols) >= 4 else cf_cols)
        capex_vals = _get_yearly_values(cashflow,
            ["Capital Expenditure", "Purchase Of PPE"],
            cf_cols[-4:] if len(cf_cols) >= 4 else cf_cols)

        ocf_y0 = ocf_vals[-1] if ocf_vals else None
        capex_y0 = capex_vals[-1] if capex_vals else None
        fcf_y0 = (ocf_y0 + capex_y0) if (ocf_y0 is not None and capex_y0 is not None) else None  # CapEx is negative in yfinance

        total_assets = _bs_row(bs, ["Total Assets"], sorted(bs.columns)[-1]) if (bs is not None and not bs.empty) else None

        # S6: Accruals Ratio (REPLACES cash_conversion)
        if ni[0] is not None and ocf_y0 is not None and total_assets and total_assets > 0:
            data["schilit_accruals_ratio"] = round((ni[0] - ocf_y0) / total_assets, 4)
        else:
            data["schilit_accruals_ratio"] = None

        # S7: CFO / NI ratio
        if ni[0] is not None and ni[0] > 0 and ocf_y0 is not None:
            data["schilit_cfo_ni_ratio"] = round(ocf_y0 / ni[0], 2)
        else:
            data["schilit_cfo_ni_ratio"] = None

        # S8: FCF / NI ratio
        if ni[0] is not None and ni[0] > 0 and fcf_y0 is not None:
            data["schilit_fcf_ni_ratio"] = round(fcf_y0 / ni[0], 2)
        else:
            data["schilit_fcf_ni_ratio"] = None

        # M1: Excess Cash Margin
        oi_y0 = None
        if income_stmt is not None and not income_stmt.empty:
            is_cols = sorted(income_stmt.columns)
            oi_y0 = _bs_row(income_stmt, ["Operating Income", "EBIT"], is_cols[-1])
        rev_y0 = _sf(data.get("revenue_y0"))
        if ocf_y0 is not None and oi_y0 is not None and rev_y0 and rev_y0 > 0:
            data["mulford_ecm"] = round((ocf_y0 - oi_y0) / rev_y0 * 100, 2)
        else:
            data["mulford_ecm"] = None

        # M2: ECM Trend
        if len(cf_cols) >= 2 and income_stmt is not None and not income_stmt.empty:
            is_cols = sorted(income_stmt.columns)
            prev_ocf = _bs_row(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities", "Cash Flow From Continuing Operating Activities"], cf_cols[-2])
            prev_oi = _bs_row(income_stmt, ["Operating Income", "EBIT"], is_cols[-2]) if len(is_cols) >= 2 else None
            rev_y1 = _sf(data.get("revenue_y1"))
            if prev_ocf is not None and prev_oi is not None and rev_y1 and rev_y1 > 0:
                ecm_prev = (prev_ocf - prev_oi) / rev_y1 * 100
                ecm_curr = _sf(data.get("mulford_ecm"))
                if ecm_curr is not None:
                    data["mulford_ecm_trend"] = round(ecm_curr - ecm_prev, 2)
                else:
                    data["mulford_ecm_trend"] = None
            else:
                data["mulford_ecm_trend"] = None
        else:
            data["mulford_ecm_trend"] = None

        # M3: Cash Margin
        if ocf_y0 is not None and rev_y0 and rev_y0 > 0:
            data["mulford_cash_margin"] = round(ocf_y0 / rev_y0 * 100, 2)
        else:
            data["mulford_cash_margin"] = None

        # M4: OCF / Operating Income
        if ocf_y0 is not None and oi_y0 is not None and oi_y0 > 0:
            data["mulford_ocf_oi_ratio"] = round(ocf_y0 / oi_y0, 2)
        else:
            data["mulford_ocf_oi_ratio"] = None

        # T2: Consistent CFO
        valid_ocf = [v for v in ocf_vals if v is not None]
        data["dorsey_consistent_cfo"] = bool(len(valid_ocf) >= 3 and all(v > 0 for v in valid_ocf))

        # Interest Coverage (needs EBIT from income_stmt)
        if income_stmt is not None and not income_stmt.empty:
            is_cols = sorted(income_stmt.columns)
            ebit = _bs_row(income_stmt, ["EBIT", "Operating Income"], is_cols[-1])
            int_exp = _bs_row(income_stmt, ["Interest Expense", "Interest Expense Non Operating", "Net Interest Income"], is_cols[-1])
            if ebit is not None and int_exp is not None and int_exp != 0:
                data["dorsey_interest_coverage"] = round(abs(ebit / int_exp), 2)

    else:
        for col in ["schilit_accruals_ratio", "schilit_cfo_ni_ratio", "schilit_fcf_ni_ratio",
                     "mulford_ecm", "mulford_ecm_trend", "mulford_cash_margin",
                     "mulford_ocf_oi_ratio", "dorsey_consistent_cfo"]:
            data[col] = None


# ─── 3. VALUATION (14 columns) ───

def compute_valuation(data, info, income_stmt, cashflow, bs, shares):
    """Compute valuation metrics."""
    price = _sf(data.get("price"))
    pe = _sf(data.get("pe"))
    pb = _sf(data.get("pb"))
    eps = _sf(data.get("eps"))
    avg_eps = _sf(data.get("graham_avg_eps_4y"))
    bvps = _sf(data.get("graham_bvps"))
    ni_cagr = _sf(data.get("ni_cagr_3y"))
    div_yield = _sf(data.get("dividend_yield"))
    market_cap = _sf(info.get("marketCap"))
    total_debt = _sf(info.get("totalDebt"))
    total_cash = _sf(info.get("totalCash"))

    # D7a: PE on 3Y average EPS
    if price and avg_eps and avg_eps > 0:
        data["graham_pe_3y_avg"] = round(price / avg_eps, 2)
    else:
        data["graham_pe_3y_avg"] = None

    # D7c: PE × PB composite
    if pe and pe > 0 and pb and pb > 0:
        data["graham_pe_pb_composite"] = round(pe * pb, 2)
    else:
        data["graham_pe_pb_composite"] = None

    # D8: Graham Number
    if eps and eps > 0 and bvps and bvps > 0:
        data["graham_number"] = round((22.5 * eps * bvps) ** 0.5, 2)
    else:
        data["graham_number"] = None

    # M1: Earnings Yield Spread vs bond rate
    if pe and pe > 0:
        ey = 1.0 / pe * 100
        data["graham_earnings_yield_spread"] = round(ey - INDIA_10Y_BOND_RATE, 2)
    else:
        data["graham_earnings_yield_spread"] = None

    # M2: Graham Intrinsic Value
    g = min(ni_cagr, 25) if (ni_cagr is not None and ni_cagr > 0) else 0
    if avg_eps and avg_eps > 0:
        data["graham_intrinsic_value"] = round(avg_eps * (8.5 + 2 * g), 2)
    else:
        data["graham_intrinsic_value"] = None

    # M3: Graham Margin of Safety
    giv = _sf(data.get("graham_intrinsic_value"))
    if giv and giv > 0 and price and price > 0:
        data["graham_margin_of_safety_pct"] = round((giv - price) / price * 100, 2)
    else:
        data["graham_margin_of_safety_pct"] = None

    # G2: Greenblatt Earnings Yield (EBIT / EV)
    ebit = None
    if income_stmt is not None and not income_stmt.empty:
        is_cols = sorted(income_stmt.columns)
        ebit = _bs_row(income_stmt, ["EBIT", "Operating Income"], is_cols[-1])
    ev = None
    if market_cap is not None:
        td = total_debt or 0
        tc = total_cash or 0
        ev = market_cap + td - tc
    if ebit and ebit > 0 and ev and ev > 0:
        data["greenblatt_earnings_yield"] = round(ebit / ev * 100, 2)
    else:
        data["greenblatt_earnings_yield"] = None

    # G3: Combined rank — computed at universe level in score_all_stocks()
    data["greenblatt_combined_rank"] = None  # Placeholder

    # P1: Lynch PEG
    if pe and pe > 0 and ni_cagr and ni_cagr > 0:
        data["lynch_peg"] = round(pe / ni_cagr, 2)
    else:
        data["lynch_peg"] = None

    # P2: Lynch PEG Adjusted (dividend-adjusted)
    dy_pct = (div_yield * 100) if div_yield else 0
    if pe and pe > 0 and ni_cagr and ni_cagr > 0:
        data["lynch_peg_adjusted"] = round((ni_cagr + dy_pct) / pe, 2)
    else:
        data["lynch_peg_adjusted"] = None

    # C3: Lynch Cash-Adjusted PE
    net_cash_ps = _sf(data.get("lynch_net_cash_per_share"))
    if price and eps and eps > 0 and net_cash_ps is not None:
        adj_price = price - max(net_cash_ps, 0)  # Only subtract if net cash positive
        if adj_price > 0:
            data["lynch_cash_adjusted_pe"] = round(adj_price / eps, 2)
        else:
            data["lynch_cash_adjusted_pe"] = None
    else:
        data["lynch_cash_adjusted_pe"] = None

    # V1 Dorsey: Cash Return (FCF / EV)
    if cashflow is not None and not cashflow.empty and ev and ev > 0:
        cf_cols = sorted(cashflow.columns)
        ocf = _bs_row(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities", "Cash Flow From Continuing Operating Activities"], cf_cols[-1])
        capex = _bs_row(cashflow, ["Capital Expenditure", "Purchase Of PPE"], cf_cols[-1])
        if ocf is not None and capex is not None:
            fcf = ocf + capex  # capex is negative
            data["dorsey_cash_return"] = round(fcf / ev * 100, 2)
        else:
            data["dorsey_cash_return"] = None
    else:
        data["dorsey_cash_return"] = None

    # BM1: Buffett Intrinsic Value (DCF perpetuity)
    ocf_val = None
    capex_val = None
    if cashflow is not None and not cashflow.empty:
        cf_cols = sorted(cashflow.columns)
        ocf_val = _bs_row(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities", "Cash Flow From Continuing Operating Activities"], cf_cols[-1])
        capex_val = _bs_row(cashflow, ["Capital Expenditure", "Purchase Of PPE"], cf_cols[-1])

    if ocf_val is not None and capex_val is not None and shares and shares > 0:
        oe_ps = (ocf_val + capex_val) / shares  # Owner earnings per share
        g_rate = min(ni_cagr / 100, 0.15) if (ni_cagr and ni_cagr > 0) else 0.03
        r_rate = INDIA_10Y_BOND_RATE / 100
        if r_rate > g_rate:
            data["buffett_intrinsic_value"] = round(oe_ps * (1 + g_rate) / (r_rate - g_rate), 2)
        else:
            data["buffett_intrinsic_value"] = None
    else:
        data["buffett_intrinsic_value"] = None

    # BM2: Buffett Margin of Safety
    biv = _sf(data.get("buffett_intrinsic_value"))
    if biv and biv > 0 and price and price > 0:
        data["buffett_margin_of_safety_pct"] = round((biv - price) / biv * 100, 2)
    else:
        data["buffett_margin_of_safety_pct"] = None


# ─── 4. GROWTH TRAJECTORY (5 columns) ───

def compute_growth(data, income_stmt, shares):
    """Compute growth trajectory metrics."""
    ni_cagr = _sf(data.get("ni_cagr_3y"))

    # P3: Lynch Growth Flag
    if ni_cagr is None:
        data["lynch_growth_flag"] = "unknown"
    elif ni_cagr > 50:
        data["lynch_growth_flag"] = "dangerous"
    elif ni_cagr > 25:
        data["lynch_growth_flag"] = "suspicious"
    elif ni_cagr >= 20:
        data["lynch_growth_flag"] = "ideal"
    elif ni_cagr >= 12:
        data["lynch_growth_flag"] = "acceptable"
    else:
        data["lynch_growth_flag"] = "slow"

    # I7: Growth Acceleration
    ni = [_sf(data.get(f"net_income_y{i}")) for i in range(4)]
    if ni[0] is not None and ni[1] is not None and ni[1] > 0 and ni[2] is not None and ni[2] > 0:
        recent_growth = (ni[0] / ni[1] - 1) * 100
        prior_growth = (ni[1] / ni[2] - 1) * 100
        data["lynch_growth_acceleration"] = round(recent_growth - prior_growth, 2)
    else:
        data["lynch_growth_acceleration"] = None

    # I6 Buffett: Value-creating growth
    ni_cagr_val = ni_cagr if ni_cagr else 0
    # We need ROIC — use greenblatt_roic if already computed, else use ROE as proxy
    roic = _sf(data.get("greenblatt_roic")) or _sf(data.get("dorsey_roic"))
    roe = _sf(data.get("roe"))
    best_return = roic or (roe * 100 if roe else None)  # roe is decimal in data
    if best_return and ni_cagr_val > 0:
        data["buffett_value_creating_growth"] = bool(best_return > COST_OF_CAPITAL_PROXY)
    else:
        data["buffett_value_creating_growth"] = None

    # GI3: Average EBIT (4Y)
    if income_stmt is not None and not income_stmt.empty:
        is_cols = sorted(income_stmt.columns)
        ebit_vals = []
        for col in is_cols[-4:]:
            ebit = _bs_row(income_stmt, ["EBIT", "Operating Income"], col)
            if ebit is not None:
                ebit_vals.append(ebit)
        if ebit_vals:
            data["greenblatt_ebit_avg_4y"] = round(sum(ebit_vals) / len(ebit_vals), 2)
        else:
            data["greenblatt_ebit_avg_4y"] = None
    else:
        data["greenblatt_ebit_avg_4y"] = None

    # E4: Enterprising earnings growing
    if ni[0] is not None and ni[1] is not None:
        data["graham_ent_earnings_growing"] = bool(ni[0] > ni[1])
    else:
        data["graham_ent_earnings_growing"] = None


# ─── 5. MOAT DURABILITY (8 columns) ───

def compute_moat(data, info, income_stmt, bs, cashflow, shares):
    """Compute moat durability metrics."""
    if income_stmt is None or income_stmt.empty:
        for col in ["greenblatt_roic", "greenblatt_roic_trend", "dorsey_roic",
                     "dorsey_fcf_margin", "dorsey_roe_consistent", "dorsey_roa",
                     "dorsey_pb_roe_signal", "buffett_roe_unleveraged"]:
            data[col] = None
        return

    is_cols = sorted(income_stmt.columns)
    latest_is = is_cols[-1]
    ebit = _bs_row(income_stmt, ["EBIT", "Operating Income"], latest_is)
    revenue = _bs_row(income_stmt, ["Total Revenue"], latest_is)

    # G1: Greenblatt ROIC = EBIT / (NWC + Net Fixed Assets)
    if bs is not None and not bs.empty and ebit is not None:
        bs_cols = sorted(bs.columns)
        latest_bs = bs_cols[-1]
        ca = _bs_row(bs, ["Current Assets", "Total Current Assets"], latest_bs)
        cl = _bs_row(bs, ["Current Liabilities", "Total Current Liabilities"], latest_bs)
        cash = _sf(info.get("totalCash")) or 0
        st_debt = _bs_row(bs, ["Current Debt", "Short Long Term Debt", "Current Debt And Capital Lease Obligation"], latest_bs, 0)
        ppe = _bs_row(bs, ["Net PPE", "Property Plant Equipment Net", "Net Property Plant And Equipment"], latest_bs, 0)
        gw = _bs_row(bs, ["Goodwill"], latest_bs, 0)
        intang = _bs_row(bs, ["Intangible Assets", "Other Intangible Assets"], latest_bs, 0)

        if ca is not None and cl is not None:
            nwc = (ca - cash) - (cl - st_debt)
            tangible_capital = nwc + ppe
            if tangible_capital and tangible_capital > 0:
                data["greenblatt_roic"] = round(ebit / tangible_capital * 100, 2)
            else:
                data["greenblatt_roic"] = None
        else:
            data["greenblatt_roic"] = None
    else:
        data["greenblatt_roic"] = None

    # GI4: ROIC Trend (latest vs average)
    roic_vals = []
    if bs is not None and not bs.empty:
        bs_cols = sorted(bs.columns)
        for i, bc in enumerate(bs_cols[-4:]):
            ic = is_cols[-(4-i)] if len(is_cols) >= (4-i) else None
            if ic is None:
                continue
            e = _bs_row(income_stmt, ["EBIT", "Operating Income"], ic)
            ca_i = _bs_row(bs, ["Current Assets", "Total Current Assets"], bc)
            cl_i = _bs_row(bs, ["Current Liabilities", "Total Current Liabilities"], bc)
            ppe_i = _bs_row(bs, ["Net PPE", "Property Plant Equipment Net", "Net Property Plant And Equipment"], bc, 0)
            if e is not None and ca_i is not None and cl_i is not None:
                tc = (ca_i - cl_i) + ppe_i
                if tc and tc > 0:
                    roic_vals.append(e / tc * 100)
    if len(roic_vals) >= 2:
        avg_roic = sum(roic_vals) / len(roic_vals)
        if avg_roic != 0:
            data["greenblatt_roic_trend"] = round(roic_vals[-1] / avg_roic, 2)
        else:
            data["greenblatt_roic_trend"] = None
    else:
        data["greenblatt_roic_trend"] = None

    # D5 Dorsey: ROIC (after-tax)
    if ebit is not None and bs is not None and not bs.empty:
        bs_cols = sorted(bs.columns)
        latest_bs = bs_cols[-1]
        ta = _bs_row(bs, ["Total Assets"], latest_bs)
        # Approximate tax rate from NI / pretax income, or default 25%
        ni_y0 = _sf(data.get("net_income_y0"))
        pretax = _bs_row(income_stmt, ["Pretax Income", "Income Before Tax"], latest_is)
        tax_rate = 0.25  # default
        if ni_y0 and pretax and pretax > 0:
            tax_rate = max(0, 1 - ni_y0 / pretax)
        nopat = ebit * (1 - tax_rate)
        # Invested Capital = Total Assets - non-interest-bearing CL - excess cash
        cl = _bs_row(bs, ["Current Liabilities", "Total Current Liabilities"], latest_bs)
        cash = _sf(info.get("totalCash")) or 0
        if ta and cl:
            invested_cap = ta - cl  # Simplified
            if invested_cap > 0:
                data["dorsey_roic"] = round(nopat / invested_cap * 100, 2)
            else:
                data["dorsey_roic"] = None
        else:
            data["dorsey_roic"] = None
    else:
        data["dorsey_roic"] = None

    # D1: FCF Margin
    if cashflow is not None and not cashflow.empty and revenue and revenue > 0:
        cf_cols = sorted(cashflow.columns)
        ocf = _bs_row(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities", "Cash Flow From Continuing Operating Activities"], cf_cols[-1])
        capex = _bs_row(cashflow, ["Capital Expenditure", "Purchase Of PPE"], cf_cols[-1])
        if ocf is not None and capex is not None:
            fcf = ocf + capex
            data["dorsey_fcf_margin"] = round(fcf / revenue * 100, 2)
        else:
            data["dorsey_fcf_margin"] = None
    else:
        data["dorsey_fcf_margin"] = None

    # D3: ROE Consistent (≥15% in ≥3 of 4 years)
    roe_vals = [_sf(data.get(f"roe_y{i}")) for i in range(4)]
    valid_roe = [v for v in roe_vals if v is not None]
    if len(valid_roe) >= 3:
        data["dorsey_roe_consistent"] = sum(1 for v in valid_roe if v >= 15) >= 3
    else:
        data["dorsey_roe_consistent"] = None

    # D4: ROA
    data["dorsey_roa"] = _sf(info.get("returnOnAssets"))
    if data["dorsey_roa"] is not None:
        data["dorsey_roa"] = round(data["dorsey_roa"] * 100, 2)

    # V3: P/B + ROE signal
    pb = _sf(data.get("pb"))
    roe = _sf(data.get("roe"))
    if pb and roe:
        data["dorsey_pb_roe_signal"] = bool(pb < 1.5 and roe > 0.15)
    else:
        data["dorsey_pb_roe_signal"] = None

    # BF1: Buffett ROE Unleveraged
    de = _sf(data.get("de"))
    if roe and de is not None:
        data["buffett_roe_unleveraged"] = bool(roe >= 0.15 and de <= 50)  # de in %, 50% = D/E 0.5
    else:
        data["buffett_roe_unleveraged"] = None


# ─── 6. DIVIDEND QUALITY (4 columns) ───

def compute_dividends(data, stock, income_stmt, shares):
    """Compute dividend quality metrics."""
    # Consecutive dividend years
    try:
        divs = stock.dividends
        if divs is not None and not divs.empty:
            years_with_divs = set()
            for dt in divs.index:
                if divs[dt] > 0:
                    years_with_divs.add(dt.year)
            if years_with_divs:
                current_year = datetime.now().year
                consec = 0
                for yr in range(current_year, current_year - 50, -1):
                    if yr in years_with_divs:
                        consec += 1
                    else:
                        break
                data["dividend_consecutive_years"] = consec
            else:
                data["dividend_consecutive_years"] = 0
        else:
            data["dividend_consecutive_years"] = 0
    except Exception:
        data["dividend_consecutive_years"] = 0

    # Payout Ratio
    ni_y0 = _sf(data.get("net_income_y0"))
    div_yield = _sf(data.get("dividend_yield"))
    market_cap = _sf(data.get("market_cap"))
    if ni_y0 and ni_y0 > 0 and div_yield and market_cap:
        total_divs = div_yield * market_cap
        data["graham_payout_ratio"] = round(total_divs / ni_y0 * 100, 2)
    else:
        data["graham_payout_ratio"] = None

    # E3: Has Dividend
    data["graham_ent_has_dividend"] = bool(div_yield and div_yield > 0)

    # I9: Deep Value Flag
    pb = _sf(data.get("pb"))
    data["graham_deep_value_flag"] = bool(pb and pb <= 0.67 and ni_y0 and ni_y0 > 0)


# ─── 7. MANAGEMENT QUALITY (5 columns) ───

def compute_management(data, info, income_stmt, cashflow, bs, shares, stock):
    """Compute management quality metrics (Buffett tenets)."""
    price = _sf(data.get("price"))
    market_cap = _sf(info.get("marketCap"))

    # BF2: Owner Earnings per share
    if cashflow is not None and not cashflow.empty and shares and shares > 0:
        cf_cols = sorted(cashflow.columns)
        ocf = _bs_row(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities", "Cash Flow From Continuing Operating Activities"], cf_cols[-1])
        capex = _bs_row(cashflow, ["Capital Expenditure", "Purchase Of PPE"], cf_cols[-1])
        if ocf is not None and capex is not None:
            data["buffett_owner_earnings_ps"] = round((ocf + capex) / shares, 2)
        else:
            data["buffett_owner_earnings_ps"] = None
    else:
        data["buffett_owner_earnings_ps"] = None

    # BF4: One-Dollar Premise
    # ΔMarketCap / ΣRetainedEarnings over available years
    ni_vals = [_sf(data.get(f"net_income_y{i}")) for i in range(4)]
    valid_ni = [v for v in ni_vals if v is not None]
    div_yield = _sf(data.get("dividend_yield"))
    payout = _sf(data.get("graham_payout_ratio"))

    if len(valid_ni) >= 3 and market_cap and price:
        # Estimate retained earnings: NI × (1 - payout_ratio)
        retention = 1.0 - (payout / 100 if payout else 0.3)  # Default 30% payout
        retained = sum(v * retention for v in valid_ni)
        # Estimate market cap change: approximate from price change
        # Use earliest available year's NI/equity to approximate old price
        try:
            hist = stock.history(period="4y")
            if hist is not None and not hist.empty and len(hist) > 250:
                old_price = float(hist["Close"].iloc[0])
                delta_mcap = (price - old_price) * shares
                if retained > 0:
                    data["buffett_one_dollar_test"] = round(delta_mcap / retained, 2)
                else:
                    data["buffett_one_dollar_test"] = None
            else:
                data["buffett_one_dollar_test"] = None
        except Exception:
            data["buffett_one_dollar_test"] = None
    else:
        data["buffett_one_dollar_test"] = None

    # BG1: Rational Allocation
    roic = _sf(data.get("greenblatt_roic")) or _sf(data.get("dorsey_roic"))
    dilution = _sf(data.get("dorsey_share_dilution_pct"))
    payout_r = _sf(data.get("graham_payout_ratio"))
    if roic is not None:
        if roic > COST_OF_CAPITAL_PROXY:
            # High ROIC — reinvesting is rational
            data["buffett_rational_allocation"] = True
        elif payout_r and payout_r > 40:
            # Low ROIC but returning cash — rational
            data["buffett_rational_allocation"] = True
        elif dilution is not None and dilution < 0:
            # Low ROIC but buying back shares — rational
            data["buffett_rational_allocation"] = True
        else:
            data["buffett_rational_allocation"] = False
    else:
        data["buffett_rational_allocation"] = None

    # T8 Dorsey: Share Dilution
    if bs is not None and not bs.empty:
        bs_cols = sorted(bs.columns)
        # Try to get shares from balance sheet across years
        shares_latest = shares
        shares_oldest = None
        # Approximate: use equity ratio as proxy if shares not directly available
        if len(bs_cols) >= 3:
            # Use yfinance sharesOutstanding as current, try to back-calculate
            # For now, set to None — complex to compute without historical shares
            data["dorsey_share_dilution_pct"] = None  # TODO: improve
        else:
            data["dorsey_share_dilution_pct"] = None
    else:
        data["dorsey_share_dilution_pct"] = None

    # T1: Has Operating Profit
    if income_stmt is not None and not income_stmt.empty:
        is_cols = sorted(income_stmt.columns)
        has_op = False
        for col in is_cols:
            oi = _bs_row(income_stmt, ["Operating Income", "EBIT"], col)
            if oi is not None and oi > 0:
                has_op = True
                break
        data["dorsey_has_operating_profit"] = has_op
    else:
        data["dorsey_has_operating_profit"] = None


# ─── 8. MANIPULATION FLAGS (15 columns) ───

def compute_manipulation_flags(data, income_stmt, bs, cashflow):
    """Compute Schilit + Dorsey + Lynch manipulation red flags."""

    # Initialize all to None
    flag_cols = ["schilit_dso", "schilit_ar_revenue_divergence", "schilit_capex_depr_ratio",
                 "schilit_dsi", "schilit_inventory_revenue_div", "schilit_wc_cffo_pct",
                 "schilit_leverage_trend", "schilit_serial_acquirer", "goodwill_pct",
                 "dorsey_cfo_ni_divergence", "dorsey_ar_growth_flag", "lynch_inventory_flag",
                 "greenblatt_sector_excluded", "greenblatt_low_pe_flag", "mulford_fcf_consistent"]
    for col in flag_cols:
        data[col] = None

    sector = data.get("sector", "")
    pe = _sf(data.get("pe"))

    # Greenblatt exclusions (Greenblatt-only, not universe-wide)
    data["greenblatt_sector_excluded"] = bool(sector in ("Financial Services", "Utilities"))
    data["greenblatt_low_pe_flag"] = bool(pe is not None and 0 < pe < 5)

    if bs is None or bs.empty:
        return

    bs_cols = sorted(bs.columns)
    latest_bs = bs_cols[-1]

    # Goodwill %
    total_assets = _bs_row(bs, ["Total Assets"], latest_bs)
    gw = _bs_row(bs, ["Goodwill"], latest_bs)
    if gw is not None and total_assets and total_assets > 0:
        data["goodwill_pct"] = round(gw / total_assets * 100, 2)
    else:
        data["goodwill_pct"] = 0.0

    # A/R and Inventory analysis (need at least 2 years)
    if len(bs_cols) >= 2:
        prev_bs = bs_cols[-2]
        ar_now = _bs_row(bs, ["Accounts Receivable", "Net Receivables"], latest_bs)
        ar_prev = _bs_row(bs, ["Accounts Receivable", "Net Receivables"], prev_bs)
        inv_now = _bs_row(bs, ["Inventory"], latest_bs)
        inv_prev = _bs_row(bs, ["Inventory"], prev_bs)
        rev_y0 = _sf(data.get("revenue_y0"))
        rev_y1 = _sf(data.get("revenue_y1"))

        # S1: DSO
        if ar_now and rev_y0 and rev_y0 > 0:
            data["schilit_dso"] = round(ar_now / (rev_y0 / 365), 1)

        # S2: A/R vs Revenue divergence
        ar_growth = _pct_growth(ar_now, ar_prev)
        rev_growth = _pct_growth(rev_y0, rev_y1)
        if ar_growth is not None and rev_growth is not None:
            data["schilit_ar_revenue_divergence"] = round(ar_growth - rev_growth, 2)
            data["dorsey_ar_growth_flag"] = bool(ar_growth > rev_growth + 5)

        # S4: DSI
        if inv_now and income_stmt is not None and not income_stmt.empty:
            is_cols = sorted(income_stmt.columns)
            cogs = _bs_row(income_stmt, ["Cost Of Revenue", "Cost Of Goods Sold"], is_cols[-1])
            if cogs and cogs > 0:
                data["schilit_dsi"] = round(inv_now / (cogs / 365), 1)

        # S5: Inventory vs Revenue divergence
        inv_growth = _pct_growth(inv_now, inv_prev)
        if inv_growth is not None and rev_growth is not None:
            data["schilit_inventory_revenue_div"] = round(inv_growth - rev_growth, 2)
            data["lynch_inventory_flag"] = bool(inv_growth > rev_growth + 5)

    # S3: CapEx / Depreciation ratio
    if cashflow is not None and not cashflow.empty and income_stmt is not None and not income_stmt.empty:
        cf_cols = sorted(cashflow.columns)
        is_cols = sorted(income_stmt.columns)
        capex = _bs_row(cashflow, ["Capital Expenditure", "Purchase Of PPE"], cf_cols[-1])
        da = _bs_row(cashflow, ["Depreciation And Amortization", "Depreciation Amortization Depletion"], cf_cols[-1])
        if da is None:
            da = _bs_row(income_stmt, ["Reconciled Depreciation", "Depreciation And Amortization In Income Statement"], is_cols[-1])
        if capex is not None and da is not None and da != 0:
            data["schilit_capex_depr_ratio"] = round(abs(capex) / abs(da), 2)

    # S9: Working Capital as % of CFO
    if cashflow is not None and not cashflow.empty:
        cf_cols = sorted(cashflow.columns)
        wc_change = _bs_row(cashflow, ["Change In Working Capital", "Changes In Working Capital"], cf_cols[-1])
        ocf = _bs_row(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities", "Cash Flow From Continuing Operating Activities"], cf_cols[-1])
        if wc_change is not None and ocf is not None and ocf != 0:
            data["schilit_wc_cffo_pct"] = round(wc_change / ocf * 100, 2)

    # S11: Leverage Trend (debt growth - equity growth)
    if len(bs_cols) >= 2:
        debt_now = _bs_row(bs, ["Total Debt"], latest_bs)
        debt_prev = _bs_row(bs, ["Total Debt"], bs_cols[-2])
        eq_now = _bs_row(bs, ["Stockholders Equity", "Total Stockholder Equity", "Total Equity Gross Minority Interest"], latest_bs)
        eq_prev = _bs_row(bs, ["Stockholders Equity", "Total Stockholder Equity", "Total Equity Gross Minority Interest"], bs_cols[-2])
        debt_g = _pct_growth(debt_now, debt_prev)
        eq_g = _pct_growth(eq_now, eq_prev)
        if debt_g is not None and eq_g is not None:
            data["schilit_leverage_trend"] = round(debt_g - eq_g, 2)

    # S12: Serial Acquirer
    if len(bs_cols) >= 2:
        gw_prev = _bs_row(bs, ["Goodwill"], bs_cols[-2], 0)
        gw_now = gw or 0
        rev_growth = _pct_growth(_sf(data.get("revenue_y0")), _sf(data.get("revenue_y1")))
        gw_growth = _pct_growth(gw_now, gw_prev) if gw_prev and gw_prev > 0 else None
        if gw_growth is not None and rev_growth is not None:
            data["schilit_serial_acquirer"] = bool(gw_growth > 20 and rev_growth < 10)

    # R1 Dorsey: CFO/NI Divergence
    ni = [_sf(data.get(f"net_income_y{i}")) for i in range(3)]
    cfo_ni = _sf(data.get("schilit_cfo_ni_ratio"))
    if ni[0] is not None and ni[1] is not None and ni[0] > ni[1] and cfo_ni is not None and cfo_ni < 0.8:
        data["dorsey_cfo_ni_divergence"] = True
    else:
        data["dorsey_cfo_ni_divergence"] = False

    # M8: FCF Consistent
    if cashflow is not None and not cashflow.empty:
        cf_cols = sorted(cashflow.columns)
        fcf_positive_count = 0
        total_years = 0
        for col in cf_cols[-4:]:
            ocf = _bs_row(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities", "Cash Flow From Continuing Operating Activities"], col)
            capex = _bs_row(cashflow, ["Capital Expenditure", "Purchase Of PPE"], col)
            if ocf is not None and capex is not None:
                total_years += 1
                if (ocf + capex) > 0:
                    fcf_positive_count += 1
        data["mulford_fcf_consistent"] = bool(total_years >= 3 and fcf_positive_count == total_years)


# ─── 9. STOCK CLASSIFICATION (4 columns) ───

def compute_classification(data):
    """Compute Lynch category, debt health, lifecycle, and enterprising financial pass."""
    ni_cagr = _sf(data.get("ni_cagr_3y"))
    eps_cv = _sf(data.get("graham_eps_cv"))
    ni = [_sf(data.get(f"net_income_y{i}")) for i in range(4)]
    pb = _sf(data.get("pb"))
    de = _sf(data.get("de"))
    market_cap = _sf(data.get("market_cap"))
    net_cash = _sf(data.get("graham_net_cash"))

    # Lynch Category
    has_recent_loss = any(v is not None and v < 0 for v in ni[:2])
    has_older_loss = any(v is not None and v < 0 for v in ni[2:])
    recovering = has_older_loss and ni[0] is not None and ni[0] > 0

    if recovering:
        data["lynch_category"] = "turnaround"
    elif eps_cv is not None and eps_cv > 0.5:
        data["lynch_category"] = "cyclical"
    elif pb and pb <= 0.67 and net_cash and net_cash > 0:
        data["lynch_category"] = "asset_play"
    elif ni_cagr is not None and ni_cagr > 15:
        data["lynch_category"] = "fast_grower"
    elif ni_cagr is not None and ni_cagr >= 5:
        data["lynch_category"] = "stalwart"
    elif ni_cagr is not None:
        data["lynch_category"] = "slow_grower"
    else:
        data["lynch_category"] = "unknown"

    # Lynch Debt Health
    if de is None:
        data["lynch_debt_healthy"] = "unknown"
    elif de <= 33:
        data["lynch_debt_healthy"] = "normal"
    elif de <= 100:
        data["lynch_debt_healthy"] = "acceptable"
    else:
        data["lynch_debt_healthy"] = "risky"

    # Mulford Lifecycle Stage
    ni_y0 = _sf(data.get("net_income_y0"))
    ocf_ni_ratio = _sf(data.get("schilit_cfo_ni_ratio"))
    consistent_cfo = data.get("dorsey_consistent_cfo")

    if ni_y0 is not None and ni_y0 < 0 and not consistent_cfo:
        data["mulford_lifecycle_stage"] = "startup"
    elif ni_y0 is not None and ni_y0 > 0 and ocf_ni_ratio is not None and ocf_ni_ratio < 0.5:
        data["mulford_lifecycle_stage"] = "growth"
    elif ni_y0 is not None and ni_y0 > 0 and ocf_ni_ratio is not None and ocf_ni_ratio >= 1.0:
        data["mulford_lifecycle_stage"] = "mature"
    elif ni_cagr is not None and ni_cagr < -5:
        data["mulford_lifecycle_stage"] = "decline"
    else:
        data["mulford_lifecycle_stage"] = "growth"  # default

    # E1: Enterprising Financial Pass
    cr = _sf(data.get("current_ratio"))
    data["graham_ent_financial_pass"] = bool(cr and cr >= 1.5 and de is not None and de <= 110)


# ─── 10. SPECTRUM SCORES (9 columns) ───

def compute_spectrum_scores(data):
    """Compute all spectrum scores from the layer-1 metrics."""

    # ── Graham Defensive Score (X/8) ──
    graham_d = 0
    if data.get("graham_adequate_size"): graham_d += 1
    if data.get("graham_current_ratio_pass"): graham_d += 1
    if data.get("graham_ltd_vs_nca"): graham_d += 1
    if data.get("graham_earnings_stable_4y"): graham_d += 1
    if (data.get("dividend_consecutive_years") or 0) >= 5: graham_d += 1
    eps_growth = _sf(data.get("graham_eps_growth_pct_4y"))
    if eps_growth is not None and eps_growth >= 33: graham_d += 1
    pe_pb = _sf(data.get("graham_pe_pb_composite"))
    if pe_pb is not None and 0 < pe_pb <= 22.5: graham_d += 1
    gn = _sf(data.get("graham_number"))
    price = _sf(data.get("price"))
    if gn and price and price <= gn: graham_d += 1
    data["graham_defensive_score"] = graham_d

    # ── Graham Enterprising Score (X/6) ──
    graham_e = 0
    if data.get("graham_ent_financial_pass"): graham_e += 1
    if data.get("graham_earnings_stable_4y"): graham_e += 1  # same check, relaxed = same with 4Y
    if data.get("graham_ent_has_dividend"): graham_e += 1
    if data.get("graham_ent_earnings_growing"): graham_e += 1
    ntav = _sf(data.get("graham_price_to_ntav"))
    if ntav is not None and 0 < ntav <= 1.2: graham_e += 1
    pe = _sf(data.get("pe"))
    if pe is not None and 0 < pe <= 10: graham_e += 1
    data["graham_enterprising_score"] = graham_e

    # ── Greenblatt Score (percentile bucket 1-10) ──
    # NOTE: This requires universe-level ranking. Set placeholder here.
    # Actual ranking computed in score_all_stocks() after all stocks processed.
    data["greenblatt_score"] = None  # Filled by rank pass

    # ── Dorsey+Buffett Combined Score (X/10) ──
    db_score = 0
    # Dorsey 5 moat criteria
    fcf_m = _sf(data.get("dorsey_fcf_margin"))
    if fcf_m is not None and fcf_m >= 5: db_score += 1
    pm = _sf(data.get("profit_margin"))
    if pm is not None and pm >= 0.15: db_score += 1
    if data.get("dorsey_roe_consistent"): db_score += 1
    roa = _sf(data.get("dorsey_roa"))
    if roa is not None and roa >= 6: db_score += 1
    droic = _sf(data.get("dorsey_roic"))
    if droic is not None and droic > COST_OF_CAPITAL_PROXY: db_score += 1
    # Buffett 5 tenets
    if data.get("buffett_roe_unleveraged"): db_score += 1
    odt = _sf(data.get("buffett_one_dollar_test"))
    if odt is not None and odt >= 1.0: db_score += 1
    if data.get("buffett_rational_allocation"): db_score += 1
    if data.get("graham_earnings_stable_4y") and data.get("dorsey_consistent_cfo"):
        db_score += 1  # Consistent operations
    if data.get("buffett_value_creating_growth"): db_score += 1
    data["dorsey_buffett_score"] = db_score

    # ── Dorsey 10-Minute Score (X/8) ──
    t10 = 0
    if data.get("dorsey_has_operating_profit"): t10 += 1
    if data.get("dorsey_consistent_cfo"): t10 += 1
    # ROE > 10% in 3/4 years
    roe_vals = [_sf(data.get(f"roe_y{i}")) for i in range(4)]
    valid_roe = [v for v in roe_vals if v is not None]
    if len(valid_roe) >= 3 and sum(1 for v in valid_roe if v >= 10) >= 3: t10 += 1
    # Earnings consistency (low CV)
    cv = _sf(data.get("graham_eps_cv"))
    if cv is not None and cv < 0.5: t10 += 1
    if data.get("dorsey_clean_balance_sheet"): t10 += 1
    if fcf_m is not None and fcf_m >= 5: t10 += 1
    # Low share dilution
    dil = _sf(data.get("dorsey_share_dilution_pct"))
    if dil is not None and dil < 2: t10 += 1
    elif dil is None: t10 += 1  # No data = benefit of doubt
    # FCF consistent
    if data.get("mulford_fcf_consistent"): t10 += 1
    data["dorsey_10min_score"] = t10

    # ── Lynch Score (category-branching) ──
    cat = data.get("lynch_category", "unknown")
    l_score = 0
    peg = _sf(data.get("lynch_peg"))
    peg_adj = _sf(data.get("lynch_peg_adjusted"))
    debt_h = data.get("lynch_debt_healthy")

    if cat == "fast_grower":
        if peg is not None and peg < 1: l_score += 3
        elif peg is not None and peg < 1.5: l_score += 2
        elif peg is not None and peg < 2: l_score += 1
        growth_f = data.get("lynch_growth_flag")
        if growth_f == "ideal": l_score += 3
        elif growth_f == "acceptable": l_score += 2
        if debt_h == "normal": l_score += 2
        elif debt_h == "acceptable": l_score += 1
        if data.get("graham_earnings_stable_4y"): l_score += 2
        # Max 10

    elif cat == "stalwart":
        if pe is not None and 0 < pe <= 15: l_score += 3
        elif pe is not None and 0 < pe <= 20: l_score += 2
        if peg_adj is not None and peg_adj >= 2: l_score += 3
        elif peg_adj is not None and peg_adj >= 1: l_score += 2
        if debt_h in ("normal", "acceptable"): l_score += 2
        if data.get("graham_earnings_stable_4y"): l_score += 2
        # Max 10

    elif cat == "slow_grower":
        consec = data.get("dividend_consecutive_years", 0) or 0
        if consec >= 10: l_score += 3
        elif consec >= 5: l_score += 2
        payout = _sf(data.get("graham_payout_ratio"))
        if payout and 30 <= payout <= 75: l_score += 3
        elif payout and payout < 30: l_score += 1
        dy = _sf(data.get("dividend_yield"))
        if dy and dy > 0.04: l_score += 2
        elif dy and dy > 0.02: l_score += 1
        if data.get("dorsey_clean_balance_sheet"): l_score += 2
        # Max 10

    elif cat == "cyclical":
        # For cyclicals: low PE = expensive (peak), high PE = cheap (trough)
        # So we DON'T reward low PE. Instead reward:
        if debt_h == "normal": l_score += 3
        elif debt_h == "acceptable": l_score += 2
        accel = _sf(data.get("lynch_growth_acceleration"))
        if accel is not None and accel > 0: l_score += 3  # Recovering
        inv_flag = data.get("lynch_inventory_flag")
        if inv_flag is False: l_score += 2  # No inventory buildup
        if data.get("dorsey_consistent_cfo"): l_score += 2
        # Max 10

    elif cat == "turnaround":
        net_cash_ps = _sf(data.get("lynch_net_cash_per_share"))
        if net_cash_ps is not None and net_cash_ps > 0: l_score += 3
        if debt_h == "normal": l_score += 3
        elif debt_h == "acceptable": l_score += 2
        ni_y0 = _sf(data.get("net_income_y0"))
        if ni_y0 and ni_y0 > 0: l_score += 2  # Now profitable
        if data.get("dorsey_consistent_cfo"): l_score += 2
        # Max 10

    elif cat == "asset_play":
        ncav_r = _sf(data.get("graham_ncav_ratio"))
        if ncav_r is not None and ncav_r <= 0.67: l_score += 4
        elif ncav_r is not None and ncav_r <= 1.0: l_score += 2
        if debt_h == "normal": l_score += 3
        if _sf(data.get("graham_net_cash")) and data["graham_net_cash"] > 0: l_score += 3
        # Max 10

    data["lynch_score"] = l_score

    # ── Schilit Manipulation Score (X/10, INVERTED: higher = worse) ──
    manip = 0
    ar_div = _sf(data.get("schilit_ar_revenue_divergence"))
    if ar_div is not None and ar_div > 10: manip += 1
    inv_div = _sf(data.get("schilit_inventory_revenue_div"))
    if inv_div is not None and inv_div > 10: manip += 1
    accruals = _sf(data.get("schilit_accruals_ratio"))
    if accruals is not None and accruals > 0.10: manip += 1
    cfo_ni = _sf(data.get("schilit_cfo_ni_ratio"))
    if cfo_ni is not None and cfo_ni < 0.5: manip += 1
    fcf_ni = _sf(data.get("schilit_fcf_ni_ratio"))
    if fcf_ni is not None and fcf_ni < 0: manip += 1
    capex_d = _sf(data.get("schilit_capex_depr_ratio"))
    if capex_d is not None and capex_d > 2.0: manip += 1
    wc_pct = _sf(data.get("schilit_wc_cffo_pct"))
    if wc_pct is not None and abs(wc_pct) > 50: manip += 1
    lev_trend = _sf(data.get("schilit_leverage_trend"))
    if lev_trend is not None and lev_trend > 20: manip += 1
    if data.get("schilit_serial_acquirer"): manip += 1
    if data.get("dorsey_cfo_ni_divergence"): manip += 1
    data["schilit_manipulation_score"] = manip

    # ── Mulford Cash Flow Quality Score (X/5) ──
    mq = 0
    ecm = _sf(data.get("mulford_ecm"))
    if ecm is not None and ecm > 0: mq += 1
    ecm_trend = _sf(data.get("mulford_ecm_trend"))
    if ecm_trend is not None and ecm_trend >= 0: mq += 1
    cm = _sf(data.get("mulford_cash_margin"))
    if cm is not None and cm > 5: mq += 1
    ocf_oi = _sf(data.get("mulford_ocf_oi_ratio"))
    if ocf_oi is not None and ocf_oi >= 1.0: mq += 1
    if data.get("mulford_fcf_consistent"): mq += 1
    data["mulford_cashflow_quality_score"] = mq


# ─── 11. QUALITY GATE (rewritten) ───

def compute_quality_gate(data):
    """Rewritten quality gate using manipulation flags + ECM."""
    quality = True

    accruals = _sf(data.get("schilit_accruals_ratio"))
    if accruals is not None and accruals > 0.10:
        quality = False

    manip_score = data.get("schilit_manipulation_score", 0) or 0
    if manip_score > 3:
        quality = False

    ecm_trend = _sf(data.get("mulford_ecm_trend"))
    if ecm_trend is not None and ecm_trend < -5.0:
        quality = False

    # Retain existing checks
    spike = _sf(data.get("earnings_spike"))
    if spike is not None and spike > 3.0:
        quality = False

    non_op = _sf(data.get("non_op_pct"))
    if non_op is not None and non_op > 40:
        quality = False

    data["quality_pass"] = quality


# ─── 12. FRAMEWORK PASS VERDICTS (5 dimensions) ───

def compute_framework_verdicts(data):
    """Compute the 5-framework pass/fail and 0-5 composite score."""
    graham_pass = (data.get("graham_defensive_score", 0) or 0) >= 6
    greenblatt_pass = (data.get("greenblatt_score") or 0) >= 7
    dorsey_buff_pass = (data.get("dorsey_buffett_score", 0) or 0) >= 6

    # Trajectory (existing logic preserved)
    rev_g = _sf(data.get("rev_growth"))
    ni_g = _sf(data.get("ni_growth"))
    de = _sf(data.get("de"))
    debt_g = _sf(data.get("debt_growth"))
    growth_ok = (rev_g is not None and rev_g > 0) or (ni_g is not None and ni_g > 0)
    debt_ok = (debt_g is not None and debt_g < 0) or (de is not None and de < 50)
    trajectory_pass = bool(growth_ok and debt_ok)

    # Lynch pass (category-dependent threshold)
    lynch_s = data.get("lynch_score", 0) or 0
    cat = data.get("lynch_category", "unknown")
    # Threshold: ≥6 out of 10 for most categories
    lynch_threshold = 6
    if cat == "slow_grower":
        lynch_threshold = 5  # Easier to pass for dividend payers
    lynch_pass = lynch_s >= lynch_threshold

    data["graham_pass"] = graham_pass
    data["greenblatt_pass"] = greenblatt_pass
    data["dorsey_pass"] = dorsey_buff_pass  # Keep column name for backward compat
    data["trajectory_pass"] = trajectory_pass
    data["lynch_pass"] = lynch_pass
    data["score"] = sum([graham_pass, greenblatt_pass, dorsey_buff_pass, trajectory_pass, lynch_pass])


# ═══════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════

def compute_all_deep_metrics(data, stock):
    """
    Main entry point. Computes all 90 new columns.

    Args:
        data: dict with basic fundamentals (from fetch_fundamentals)
        stock: yfinance Ticker object (for dividend history, balance sheet detail)
    """
    info = stock.info or {}
    shares = _sf(info.get("sharesOutstanding"))

    # Fetch raw data from yfinance
    try:
        income_stmt = stock.financials
        if income_stmt is None or income_stmt.empty:
            income_stmt = None
    except Exception:
        income_stmt = None

    try:
        balance_sheet = stock.balance_sheet
        if balance_sheet is None or balance_sheet.empty:
            balance_sheet = None
    except Exception:
        balance_sheet = None

    try:
        cashflow = stock.cashflow
        if cashflow is None or cashflow.empty:
            cashflow = None
    except Exception:
        cashflow = None

    # Compute all categories
    compute_balance_sheet(data, info, balance_sheet, shares)
    compute_earnings_quality(data, info, income_stmt, cashflow, balance_sheet, shares)
    compute_moat(data, info, income_stmt, balance_sheet, cashflow, shares)
    compute_valuation(data, info, income_stmt, cashflow, balance_sheet, shares)
    compute_growth(data, income_stmt, shares)
    compute_dividends(data, stock, income_stmt, shares)
    compute_management(data, info, income_stmt, cashflow, balance_sheet, shares, stock)
    compute_manipulation_flags(data, income_stmt, balance_sheet, cashflow)
    compute_classification(data)
    compute_spectrum_scores(data)
    compute_quality_gate(data)

    return data


def compute_greenblatt_ranks(all_data):
    """
    Universe-level computation: rank all stocks on ROIC and Earnings Yield,
    compute combined rank and percentile score.
    Must be called AFTER all stocks have individual metrics computed.
    """
    # Filter to non-excluded stocks with valid ROIC and EY
    rankable = []
    for d in all_data:
        if d.get("greenblatt_sector_excluded"):
            continue
        roic = _sf(d.get("greenblatt_roic"))
        ey = _sf(d.get("greenblatt_earnings_yield"))
        if roic is not None and ey is not None:
            rankable.append(d)

    if not rankable:
        return

    # Sort by ROIC (descending = best gets rank 1)
    rankable.sort(key=lambda x: _sf(x.get("greenblatt_roic")) or 0, reverse=True)
    for i, d in enumerate(rankable):
        d["_roic_rank"] = i + 1

    # Sort by Earnings Yield (descending = best gets rank 1)
    rankable.sort(key=lambda x: _sf(x.get("greenblatt_earnings_yield")) or 0, reverse=True)
    for i, d in enumerate(rankable):
        d["_ey_rank"] = i + 1

    # Combined rank
    for d in rankable:
        d["greenblatt_combined_rank"] = d.get("_roic_rank", 0) + d.get("_ey_rank", 0)

    # Convert to percentile score (1-10)
    rankable.sort(key=lambda x: x.get("greenblatt_combined_rank", 9999))
    n = len(rankable)
    for i, d in enumerate(rankable):
        percentile = (1 - i / n) * 10
        d["greenblatt_score"] = max(1, min(10, int(round(percentile))))

    # Now compute framework verdicts for ALL stocks (including greenblatt)
    for d in all_data:
        compute_framework_verdicts(d)
