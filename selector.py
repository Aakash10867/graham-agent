"""
selector.py — deterministic portfolio construction.

PURE. No Streamlit. No network. No LLM. Takes data in, returns picks out.
That purity is not hygiene: it is what lets backtest_runner.py call the REAL
selector with a point-in-time price slice instead of reimplementing it and
backtesting a lookalike.

    select_portfolio(universe_df, policy, price_history) -> SelectionResult

WHAT THIS REPLACES, AND WHY
---------------------------
The old get_sip_candidates ended with:

    candidates.sort(key=lambda c: c.get("diversification_rank", 999))

Every ranking computed above that line was discarded. diversification_rank came
from a greedy minimum-variance loop over a covariance matrix estimated from one
year of daily closes. For a thinly traded stock, days with no trade produce a
flat close, hence a ZERO return, hence a downward-biased sigma; and
non-synchronous trading biases every correlation downward too (Scholes-Williams).
So argmin(portfolio variance) is mechanically argmax(staleness). The loop was a
staleness detector wearing a Markowitz costume, and it was choosing the stocks.

That is why every portfolio was penny stocks. It is also why every portfolio was
IDENTICAL regardless of the questionnaire: diversification_rank contains no user
information at all.

THE HIERARCHY, RESTORED
----------------------
  Tier 1  investability floor    — existence, not preference
  Tier 2  the gate               — Q8, hard, abstention-adjusted
  Tier 3  the ranking            — Q7 x Q9, continuous, WITHIN SECTOR
  Struct  integer quotas         — resolved up front, reservation not repair
  Covar   staleness + tiebreak   — never security selection

Reilly & Brown builds the efficient frontier FROM the assets you are willing to
hold. It never asks covariance to choose them.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd

# ── Tier 1: investability floor ────────────────────────────────────────────
# Calibrated 2026-07 against the measured distribution of the 4+ pool.
# ₹500cr x ₹50L keeps 100 of 113. It removes, among others:
#   505685.BO  ₹7cr    ₹0.0 lakh/day   (name field corrupted)
#   Comfort Fincap  ₹54cr   ₹3.6 lakh/day
#   PATINTLOG.NS    ₹110cr  ₹26 lakh/day   score 4   <- junk reaches 4/5
MIN_MARKET_CAP = 500e7          # ₹500 crore. Equivalently: risk_tier != "Micro".
MIN_TURNOVER = 50e5             # ₹50 lakh/day = price x avg_daily_volume.

# Turnover is NOT an exit-liquidity gate — a ₹5,000 SIP holding ₹333 per name
# could exit anything. It is a PRICE INTEGRITY gate. A stock trading ₹10 lakh a
# day has a few hundred prints, prices stale by construction, and a tape an
# operator can move. That is what corrupts the covariance matrix and what makes
# the cheapness flags fire on prices nobody transacted at.
#
# Which makes turnover a coarse prefilter and staleness the precise instrument
# for the same underlying property. Hence both, at modest thresholds.
MIN_NONZERO_RETURN_FRAC = 0.90
MIN_RETURN_OBSERVATIONS = 120

# ── Struct: the ruin floor ─────────────────────────────────────────────────
# NOT Evans & Archer. Their 12-18 was measured on RANDOMLY SELECTED portfolios,
# where stock #15 has the same expected return as stock #1 and diversification is
# free. Under a ranking, stock #15 is your fifteenth-best idea and costs expected
# return. Importing their number into a selected portfolio is a category error.
#
# This floor is a RUIN constraint, not a variance constraint, and it is derived
# rather than invented: generate_ips caps any single stock at 10% (SEBI). Equal
# weight is 100/n, which exceeds 10% whenever n < 10. Below n=10 the IPS
# literally contradicts itself.
MIN_STOCKS_RUIN_FLOOR = 10

FRAMEWORKS = ("graham", "greenblatt", "dorsey_buffett", "trajectory", "lynch")

# The boolean gates and the continuous sub-scores they threshold.
# Four of the five booleans are literally `subscore >= k` (compute_framework_verdicts).
# So the sub-score IS the native continuous underlying — no proxy, no judgement call.
# Five booleans give 32 orderings across 4,461 stocks. Five 0-10 sub-scores give ~10^4.
# THAT is what lets the questionnaire answers move the portfolio at all.
PASS_FLAG = {
    "graham": "graham_pass",
    "greenblatt": "greenblatt_pass",
    "dorsey_buffett": "dorsey_pass",
    "trajectory": "trajectory_pass",
    "lynch": "lynch_pass",
}
SUBSCORE = {
    "graham": "graham_defensive_score",
    "greenblatt": "greenblatt_score",
    "dorsey_buffett": "dorsey_buffett_score",
    "trajectory": "trajectory_score",
    "lynch": "lynch_score",
}

# Q9. The user names what they VALUE, not what they will WAIVE. So it tilts the
# ranking; it never opens a side door in the gate. A 5/5 stock with the best
# trajectory in its sector must win through the front door, not be displaced by
# a 4/5 admitted through an exception.
TRADEOFF_TILT = {
    "ok_fail_graham": ("graham",),
    "ok_fail_trajectory_lynch": ("trajectory", "lynch"),
    "ok_fail_dorsey_buffett": ("dorsey_buffett",),
    "any": (),
}

# Q7. Continuous tiebreak when two stocks tie on the weighted sub-score rank.
# (column, higher_is_better)
PHILOSOPHY_TIEBREAK = {
    "deep_value": ("graham_margin_of_safety_pct", True),
    "contrarian": ("pct_from_low", False),
    "quality_compounder": ("dorsey_roic", True),
    # lynch_peg_adjusted = (ni_cagr_3y + dividend_yield_pct) / pe, higher better.
    # This column was garbage until Sprint 11.4: dividend_yield arrived from
    # yfinance as a percent, was multiplied by 100 again, and the dividend term
    # ran 3.4x the growth term at the median. The ingest fix repaired it.
    "growth_at_fair_price": ("lynch_peg_adjusted", True),
}

RANK_BAND = 0.05  # percentile points within which covariance may reorder


# ══════════════════════════════════════════════════════════════════════════
# TIER 1 — INVESTABILITY FLOOR
# ══════════════════════════════════════════════════════════════════════════
def _tier1(df: pd.DataFrame, sip_amount: float, rejects: dict) -> pd.DataFrame:
    """Existence, not preference. A stock failing these is un-exitable,
    un-priceable, or un-assessable — not merely low quality."""
    n0 = len(df)

    def cut(mask, reason):
        nonlocal df
        removed = int((~mask).sum())
        if removed:
            rejects[reason] = rejects.get(reason, 0) + removed
        df = df[mask]

    cut(df["quality_pass"] != False, "quality_gate")            # noqa: E712
    cut(df["years_of_data"] >= 2, "insufficient_history")
    cut(df["pe"].notna() & (df["pe"] > 0), "loss_making_or_no_pe")

    # NOT notna(roe_pct) / notna(de). Those were vestigial: once we rank on
    # sub-scores, nothing downstream reads either column, and they cost us
    # CRISIL (roe_pct NaN) and every lender (de NaN). Missing data already has
    # an honest path — it lowers the sub-score, which fails the boolean, which
    # lowers `score`, which the gate rejects. A blanket notna() DOUBLE-COUNTS
    # that penalty, converting "we could not measure it" into "excluded".

    # Structural: no sector => cannot be checked against max_same_sector or
    # min_sectors. Do not let these become an "Unknown" sector that spuriously
    # satisfies the breadth requirement.
    cut(df["sector"].notna(), "no_sector")

    if "is_unevaluable" in df.columns:
        # Declared, not accidental. Lenders stayed out before only because
        # `notna(de)` happened to drop them — banks report no debtToEquity.
        cut(~df["is_unevaluable"].fillna(False).astype(bool), "unevaluable_business_model")

    cut(df["market_cap"] >= MIN_MARKET_CAP, "below_market_cap_floor")

    _turnover = df["price"] * df["avg_daily_volume"]
    cut(_turnover.fillna(0) >= MIN_TURNOVER, "below_turnover_floor")

    # Affordability. The user's rule, unchanged: one share must fit inside one
    # SIP installment. The old code silently DROPPED this filter when fewer
    # than 10 stocks survived, handing back a portfolio the user cannot buy.
    # Fail loudly instead.
    cut(df["price"] <= float(sip_amount), "unaffordable_at_sip")

    rejects["_tier1_in"] = n0
    rejects["_tier1_out"] = len(df)
    return df


def _staleness_filter(df: pd.DataFrame, price_history: pd.DataFrame | None,
                      rejects: dict) -> pd.DataFrame:
    """A stock whose price does not move on 10%+ of trading days is UNPRICEABLE
    for covariance. Chapter 6's variance algebra presupposes continuously priced
    securities; feed it stale prices and it ranks tickers by deadness.

    This runs BEFORE any covariance is computed, so the staleness attractor never
    gets a chance to operate."""
    if price_history is None or price_history.empty:
        return df

    cols = [t for t in df["ticker"] if t in price_history.columns]
    if not cols:
        return df

    # fill_method=None pins pandas 3.0's semantics. Under pandas 2.x the default
    # was 'pad', which forward-filled a stale price into a ZERO return — exactly
    # the artefact this filter exists to catch.
    rets = price_history[cols].pct_change(fill_method=None)

    obs = rets.count()
    nonzero = (rets.fillna(0) != 0).sum()
    frac = (nonzero / obs.replace(0, np.nan)).fillna(0)

    ok = set(obs[(obs >= MIN_RETURN_OBSERVATIONS) & (frac >= MIN_NONZERO_RETURN_FRAC)].index)
    # Tickers absent from price_history are not penalised; we simply have no
    # basis to judge them and the covariance step will skip them.
    absent = set(df["ticker"]) - set(cols)
    keep = df["ticker"].isin(ok | absent)

    removed = int((~keep).sum())
    if removed:
        rejects["stale_prices"] = removed
    return df[keep]


# ══════════════════════════════════════════════════════════════════════════
# TIER 2 — THE GATE (Q8)
# ══════════════════════════════════════════════════════════════════════════
def _applicable_frameworks(row) -> tuple:
    """Greenblatt's formula uses ROIC and earnings yield, which are meaningless
    for a levered balance sheet. He says so himself: do not apply it to
    financials or utilities. Scoring them as though they FAILED it is applying
    it to them.

    Result today: exactly ONE stock in Financial Services reaches 4/5, out of a
    universe where financials are a third of the index. Not a market fact — an
    arithmetic ceiling of 4/5, judged against a 4+ gate."""
    if bool(row.get("greenblatt_sector_excluded", False)):
        return tuple(f for f in FRAMEWORKS if f != "greenblatt")
    return FRAMEWORKS


def _effective_gate(min_score: int, n_applicable: int) -> int:
    """Hold every stock to the same FRACTION of applicable tests.

    Subtracting one from the score works at a 4+ gate and quietly becomes a
    SUBSIDY at 2+: a bank would need 1 of 4 (25%) against everyone else's
    2 of 5 (40%). Proportional does not have that failure mode.

        80% of 4 tests = 3.2. You cannot pass 3.2 tests. Nearest whole = 3.

    int(x + 0.5) rather than round(), because round() is banker's rounding and
    round(2.5) == 2. Not reachable with these inputs, but not worth relying on.
    """
    if n_applicable >= len(FRAMEWORKS):
        return int(min_score)
    return max(1, int(min_score * n_applicable / len(FRAMEWORKS) + 0.5))


def _tier2(df: pd.DataFrame, policy: dict, rejects: dict) -> pd.DataFrame:
    min_score = int(policy.get("min_acceptable_score", 3))
    avoid = set(policy.get("avoid_sectors") or [])

    if avoid:
        n = len(df)
        df = df[~df["sector"].isin(avoid)]
        rejects["sector_excluded_by_user"] = n - len(df)

    keep, applicable_col, score_col, gate_col = [], [], [], []
    for _, row in df.iterrows():
        app = _applicable_frameworks(row)
        s = sum(1 for f in app if bool(row.get(PASS_FLAG[f], False)))
        g = _effective_gate(min_score, len(app))
        keep.append(s >= g)
        applicable_col.append(app)
        score_col.append(s)
        gate_col.append(g)

    df = df.assign(_applicable=applicable_col, _score_applicable=score_col,
                   _effective_gate=gate_col)
    rejects["below_score_gate"] = int((~pd.Series(keep, index=df.index)).sum())
    return df[pd.Series(keep, index=df.index)]


# ══════════════════════════════════════════════════════════════════════════
# TIER 3 — THE RANKING (Q7 x Q9), WITHIN SECTOR
# ══════════════════════════════════════════════════════════════════════════
def _resolve_weights(policy: dict) -> dict:
    """Q7 sets the weights. Q9 tilts them. Neither touches the gate."""
    w = dict(policy.get("framework_weights") or {})
    if not w:
        w = {f: 20 for f in FRAMEWORKS}
    for f in TRADEOFF_TILT.get(policy.get("acceptable_tradeoff", "any"), ()):
        w[f] = w.get(f, 0) / 2.0
    total = sum(w.values()) or 1.0
    return {f: 100.0 * w.get(f, 0) / total for f in FRAMEWORKS}


def _tier3(df: pd.DataFrame, policy: dict) -> pd.DataFrame:
    """Percentile-rank each sub-score WITHIN SECTOR, then weight.

    Within sector, because a 20% ROIC means something different in Utilities
    than in Software — and because a global rank would silently become a bet on
    whichever sector happens to be cheap this quarter. It also produces exactly
    the per-sector queues the quota filler consumes.

    Percentile rather than z-score: bounded, NaN-safe, and immune to the fat
    tails that riddle Indian small-cap fundamentals.
    """
    w = _resolve_weights(policy)
    df = df.copy()

    for f in FRAMEWORKS:
        col = SUBSCORE[f]
        if col not in df.columns:
            df[f"_pct_{f}"] = 0.0
            continue
        df[f"_pct_{f}"] = (df.groupby("sector")[col]
                             .rank(pct=True, method="average", na_option="bottom")
                             .fillna(0.0))

    # Weight only over frameworks that APPLY to each stock, renormalized.
    # A financial ranked on 4 frameworks must not be penalised for the absent
    # fifth — that is the same abstention principle as the gate, applied to rank.
    def _rank(row):
        app = row["_applicable"]
        wt = sum(w[f] for f in app) or 1.0
        return sum(w[f] * row[f"_pct_{f}"] for f in app) / wt

    df["_rank_score"] = df.apply(_rank, axis=1)

    tb_col, tb_high = PHILOSOPHY_TIEBREAK.get(
        policy.get("philosophy", "growth_at_fair_price"), (None, True))
    if tb_col and tb_col in df.columns:
        v = pd.to_numeric(df[tb_col], errors="coerce")
        df["_tiebreak"] = (v if tb_high else -v).fillna(-np.inf)
        df["_tiebreak_metric"] = tb_col
        df["_tiebreak_value"] = v
    else:
        df["_tiebreak"] = 0.0
        df["_tiebreak_metric"] = None
        df["_tiebreak_value"] = np.nan

    df["_rank_in_sector"] = (df.groupby("sector")["_rank_score"]
                               .rank(ascending=False, method="first").astype(int))
    df["_sector_depth"] = df.groupby("sector")["ticker"].transform("size")
    return df.sort_values(["_rank_score", "_tiebreak"], ascending=False)


# ══════════════════════════════════════════════════════════════════════════
# STRUCT — INTEGER QUOTAS, RESERVATION NOT REPAIR
# ══════════════════════════════════════════════════════════════════════════
def _affordable_n(prices: list[float], sip_amount: float) -> int:
    """Largest k such that the k cheapest names cost <= one SIP installment,
    one share each. This is what allocate_shares actually does breadth-first.

    generate_ips uses `sip_amount // 250`; get_sip_candidates used `// 500`.
    Same quantity, two numbers. Derive it from real prices instead."""
    total, k = 0.0, 0
    for p in sorted(prices):
        if total + p > sip_amount:
            break
        total += p
        k += 1
    return k


def _quotas(n: int, alloc: dict) -> dict:
    per_sector = max(1, min(int(alloc.get("max_same_sector", 3)),
                            int(alloc.get("max_sector_pct", 25) * n / 100)))
    return {
        "n": n,
        "large_min": math.ceil(alloc.get("large_cap_min_pct", 30) * n / 100),
        "mid_min": math.ceil(alloc.get("mid_cap_min_pct", 20) * n / 100),
        "small_max": math.floor(alloc.get("small_cap_max_pct", 25) * n / 100),
        "micro_max": 0,
        "per_sector": per_sector,
        "min_sectors": int(alloc.get("min_sectors", 3)),
    }


def _feasible(pool: pd.DataFrame, q: dict) -> tuple[bool, str]:
    """Check BEFORE filling, not by discovering mid-loop.

    The old code discovered infeasibility inside a greedy loop and repaired it
    with a swap that was free to evict your best stock for a worse large-cap."""
    if q["large_min"] + q["mid_min"] > q["n"]:
        return False, "large_min + mid_min exceeds n"
    for tier, need in (("Large", q["large_min"]), ("Mid", q["mid_min"])):
        by_sec = pool[pool["risk_tier"] == tier].groupby("sector").size()
        ceiling = sum(min(v, q["per_sector"]) for v in by_sec.values)
        if ceiling < need:
            return False, f"only {ceiling} {tier} reachable, need {need}"
    if pool["sector"].nunique() < q["min_sectors"]:
        return False, f"{pool['sector'].nunique()} sectors, need {q['min_sectors']}"
    return True, ""


def _corr_tiebreak(cands: pd.DataFrame, chosen: list, corr: pd.DataFrame | None):
    """Covariance's ONLY role in selection: reorder within a quality band.

    Take candidates whose _rank_score is within RANK_BAND of the best remaining,
    then prefer the one least correlated with what we already hold. It can never
    leapfrog a band. It never performs security selection.
    """
    if cands.empty:
        return None
    best = cands["_rank_score"].iloc[0]
    band = cands[cands["_rank_score"] >= best - RANK_BAND]
    if len(band) <= 1 or corr is None or not chosen:
        return cands.index[0]

    held = [c for c in chosen if c in corr.columns]
    if not held:
        return band.index[0]

    scores = {}
    for idx, row in band.iterrows():
        t = row["ticker"]
        scores[idx] = corr.loc[t, held].abs().mean() if t in corr.columns else 0.5
    return min(scores, key=scores.get)


def _fill(pool: pd.DataFrame, q: dict, corr) -> tuple[list, dict]:
    chosen, tickers = [], []
    sec_count, tier_count = defaultdict(int), defaultdict(int)
    slot_type = {}

    def can_take(row) -> bool:
        t, s = row["risk_tier"], row["sector"]
        if sec_count[s] >= q["per_sector"]:
            return False
        if t == "Micro" and tier_count["Micro"] >= q["micro_max"]:
            return False
        if t == "Small" and tier_count["Small"] >= q["small_max"]:
            return False
        return True

    def take(idx, why):
        row = pool.loc[idx]
        chosen.append(idx)
        tickers.append(row["ticker"])
        sec_count[row["sector"]] += 1
        tier_count[row["risk_tier"]] += 1
        slot_type[idx] = why

    def remaining(mask=None):
        m = ~pool.index.isin(chosen)
        if mask is not None:
            m &= mask
        sub = pool[m]
        return sub[sub.apply(can_take, axis=1)] if len(sub) else sub

    # Pass 1: reserve the large-cap quota. Reservation, not post-hoc repair.
    while tier_count["Large"] < q["large_min"] and len(chosen) < q["n"]:
        c = remaining(pool["risk_tier"] == "Large")
        if c.empty:
            break
        take(_corr_tiebreak(c, tickers, corr), "cap_quota_large")

    # Pass 2: reserve the mid-cap quota.
    while tier_count["Mid"] < q["mid_min"] and len(chosen) < q["n"]:
        c = remaining(pool["risk_tier"] == "Mid")
        if c.empty:
            break
        take(_corr_tiebreak(c, tickers, corr), "cap_quota_mid")

    # Pass 3: breadth. One name per uncovered sector until min_sectors.
    while len(sec_count) < q["min_sectors"] and len(chosen) < q["n"]:
        c = remaining(~pool["sector"].isin(sec_count.keys()))
        if c.empty:
            break
        take(c.index[0], "breadth")

    # Pass 4: free fill, pure merit order.
    while len(chosen) < q["n"]:
        c = remaining()
        if c.empty:
            break
        take(_corr_tiebreak(c, tickers, corr), "free")

    return chosen, slot_type


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════
def select_portfolio(universe_df: pd.DataFrame, policy: dict,
                     price_history: pd.DataFrame | None = None) -> dict:
    """Deterministic. Same inputs, same portfolio. The LLM's only remaining job
    is to phrase the trace in plain English — it makes no structural decision."""
    rejects: dict = {}
    sip = float(policy.get("sip_amount", 5000))
    alloc = dict(policy.get("allocation_policy") or {})
    warnings: list[str] = []

    df = _tier1(universe_df.copy(), sip, rejects)
    df = _staleness_filter(df, price_history, rejects)
    df = _tier2(df, policy, rejects)

    if df.empty:
        return {"holdings": [], "warnings": ["No stock clears the gate you chose."],
                "rejects": rejects, "diagnostics": {}}

    pool = _tier3(df, policy).reset_index(drop=True)

    corr = None
    if price_history is not None and not price_history.empty:
        cols = [t for t in pool["ticker"] if t in price_history.columns]
        if len(cols) >= 2:
            r = price_history[cols].pct_change(fill_method=None).dropna(how="all")
            # dropna(how="all"), not dropna(): a row-wise dropna across 200
            # tickers lets ONE gappy ticker delete that date for every other one.
            corr = r.corr(min_periods=MIN_RETURN_OBSERVATIONS)

    # ── n is endogenous. No padding, no truncation. ──
    ips_target = int((policy.get("portfolio_sizing") or {}).get("ips_target", 15))
    aff_n = _affordable_n(pool["price"].tolist(), sip)
    n = min(len(pool), aff_n, ips_target)

    if n < MIN_STOCKS_RUIN_FLOOR:
        warnings.append(
            f"Only {n} holdings are possible ({len(pool)} clear your "
            f"{policy.get('min_acceptable_score')}+ gate; {aff_n} fit one SIP "
            f"installment). Below {MIN_STOCKS_RUIN_FLOOR}, equal weight puts more "
            f"than 10% behind each name — above the SEBI single-stock cap this "
            f"IPS applies. Relax the score gate, drop a sector exclusion, or "
            f"increase the SIP.")

    # Infeasible => shrink n. NEVER lower the gate, never inject a stock that
    # failed Tier 2 to satisfy a percentage.
    q = _quotas(n, alloc)
    ok, why = _feasible(pool, q)
    while not ok and n > 1:
        n -= 1
        q = _quotas(n, alloc)
        ok, why = _feasible(pool, q)
    if n < min(len(pool), aff_n, ips_target):
        warnings.append(f"Portfolio shrunk to {n} holdings: {why}.")

    chosen, slot_type = _fill(pool, q, corr)

    pct = round(100.0 / max(len(chosen), 1), 1)
    holdings = []
    for idx in chosen:
        r = pool.loc[idx]
        app = r["_applicable"]
        abstained = [f for f in FRAMEWORKS if f not in app]
        holdings.append({
            "ticker": r["ticker"], "name": r.get("name", ""),
            "sector": r["sector"], "score": int(r.get("score", 0)),
            "price": float(r["price"]), "risk_tier": r["risk_tier"],
            "allocation_pct": pct,
            "pe": r.get("pe"), "roe_pct": r.get("roe_pct"), "beta": r.get("beta"),
            "_trace": {
                # Two ORTHOGONAL facts. Conflating them is why every stock used
                # to read "diversifier": the old role was a function of
                # diversification_rank and nothing else.
                "slot_type": slot_type[idx],           # why it got a seat
                "gate_cleared": "merit",               # how it passed Tier 2
                "sector": r["sector"],
                "rank_in_sector": int(r["_rank_in_sector"]),
                "sector_depth": int(r["_sector_depth"]),
                "rank_score": round(float(r["_rank_score"]), 4),
                "applicable": list(app),
                "abstained": abstained,
                "score_applicable": int(r["_score_applicable"]),
                "effective_gate": int(r["_effective_gate"]),
                "passed": [f for f in app if bool(r.get(PASS_FLAG[f], False))],
                "failed": [f for f in app if not bool(r.get(PASS_FLAG[f], False))],
                "tiebreak_metric": r["_tiebreak_metric"],
                "tiebreak_value": (None if pd.isna(r["_tiebreak_value"])
                                   else round(float(r["_tiebreak_value"]), 3)),
            },
        })

    # Rejections build more trust than the picks do.
    # "HDFCBANK ranked #1 in Financials; sector already at 3/3."
    rejections = []
    for sec, grp in pool[~pool.index.isin(chosen)].groupby("sector"):
        for _, r in grp.nsmallest(2, "_rank_in_sector").iterrows():
            rejections.append({
                "ticker": r["ticker"], "sector": sec,
                "rank_in_sector": int(r["_rank_in_sector"]),
                "reason": "sector_full" if any(h["sector"] == sec for h in holdings)
                          else "outranked_globally",
            })

    return {
        "holdings": holdings,
        "warnings": warnings,
        "rejects": rejects,
        "rejections": rejections[:12],
        "diagnostics": {
            "pool_size": len(pool),
            "affordable_n": aff_n,
            "ips_target": ips_target,
            "n_selected": len(chosen),
            "quotas": q,
            "weights": _resolve_weights(policy),
            "philosophy": policy.get("philosophy"),
            "min_acceptable_score": policy.get("min_acceptable_score"),
            "acceptable_tradeoff": policy.get("acceptable_tradeoff"),
            "covariance_used": corr is not None,
            "sector_counts": {h["sector"]: sum(1 for x in holdings if x["sector"] == h["sector"])
                              for h in holdings},
            "tier_counts": {t: sum(1 for x in holdings if x["risk_tier"] == t)
                            for t in ("Large", "Mid", "Small", "Micro")},
        },
    }
