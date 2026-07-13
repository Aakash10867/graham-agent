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

# ── The conviction sleeve ──────────────────────────────────────────────────
# Measured 2026-07: gate 3+ and gate 2+ produced IDENTICAL portfolios under all
# four philosophies. Widening the gate admits 557 stocks and none can reach the
# top fifteen of their sector, because _rank_score = Σ wᵢ·pctᵢ and the dominant
# weight is capped at 35%:
#
#   specialist (99th on Graham, 40th elsewhere)  0.35(0.99) + 0.65(0.40) = 0.607
#   generalist (70th on everything)              1.00(0.70)              = 0.700
#
# The specialist loses. A gate is a floor; a floor removes, it never surfaces.
# Middle-tier value has to come from the RANKING, and the ranking punishes
# specialization by construction.
#
# So: reserve k slots for the top-decile specialist under the user's dominant
# framework, gate-exempt but Tier-1 bound, and make the trace say so out loud:
#   "Fails three of our five frameworks. 99th percentile of Graham value in its
#    sector. You told us that's what you're hunting."
#
# PROVISIONAL. This encodes an unmeasured belief — that one framework's signal
# justifies holding a stock that fails three others. Sprint 12 measures forward
# return per flag and sets k from data. graham_pass is rarest (5.6%) and most
# orthogonal (max |phi| = 0.26), so it is the one most likely to carry alpha.
# Gate-respecting, so safe to enable at 3+: a specialist who clears the user's
# own bar is precisely who this is for. Left at 0 for 4+ — "only the best"
# already means broad agreement, and a buried 4/5 specialist is a rounding error.
CONVICTION_SLOTS = {4: 0, 3: 1, 2: 2}
CONVICTION_MIN_PCT = 0.90


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

    # A row whose `name` is blank or a comma-mangled fragment is a corrupt CSV
    # record, not a company. The universe has held `505685.BO,0P0000CFCT,0` and
    # a broken TRANSRAILL row. Never render one to a user.
    _nm = df["name"].astype(str).str.strip()
    cut(_nm.str.len() > 2, "corrupt_name")

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


def _tier2(df: pd.DataFrame, policy: dict, rejects: dict):
    """Annotate EVERY investable stock with its gate arithmetic, then return
    (gated_pool, annotated_frame). The conviction sleeve needs the arithmetic
    for stocks that FAILED the gate — that is the whole point of the sleeve.

    avoid_sectors moved to select_portfolio: a user's sector exclusion must bind
    on conviction candidates too, and _tier2 no longer sees them all.
    """
    min_score = int(policy.get("min_acceptable_score", 3))

    keep, applicable_col, score_col, gate_col = [], [], [], []
    for _, row in df.iterrows():
        app = row["_applicable"] if "_applicable" in row else _applicable_frameworks(row)
        s = sum(1 for f in app if bool(row.get(PASS_FLAG[f], False)))
        g = _effective_gate(min_score, len(app))
        keep.append(s >= g)
        applicable_col.append(app)
        score_col.append(s)
        gate_col.append(g)

    df = df.assign(_score_applicable=score_col, _effective_gate=gate_col)
    mask = pd.Series(keep, index=df.index)
    rejects["below_score_gate"] = int((~mask).sum())
    return df[mask], df


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


def _attach_applicable(df: pd.DataFrame) -> pd.DataFrame:
    """Which frameworks can even evaluate this stock. Needed by rank AND gate."""
    return df.assign(_applicable=[_applicable_frameworks(r) for _, r in df.iterrows()])


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
    df = _attach_applicable(df.copy())

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

    # No _rank_in_sector / _sector_depth, no sort here. Those are TRACE fields —
    # "#1 of 14 in Technology" must count the stocks that actually competed for
    # the slot, i.e. the POST-gate pool. select_portfolio computes them after
    # _tier2. The _rank_score percentiles above are deliberately computed
    # PRE-gate, against everything investable.
    return df


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

def _mark_conviction(gated: pd.DataFrame, weights: dict) -> tuple[pd.DataFrame, str]:
    """Flag top-decile specialists WITHIN the gated pool.

    The first version of this drew from stocks that FAILED the gate. At a 2+
    gate, failing means score_applicable < 2, so every conviction pick was
    necessarily a 1/5 — we handed a user who asked for "2+ with a compelling
    reason" a stock passing one framework. The sleeve was structurally unable
    to surface 2/5 and 3/5 at all; it surfaced the bottom tier.

    The gate is the user's stated minimum and is inviolable. The problem was
    never the gate — it is that _rank_score BURIES specialists inside the pool.
    A 90th-percentile-Graham stock at 3/5 clears a 2+ gate easily and then loses
    every slot to broad 4/5 names. It is right there, unreachable.

    So the sleeve reorders within the gate instead of reaching beneath it.

    Three guards, each load-bearing:
      1. The dominant framework's BOOLEAN must pass. 99th percentile in a sector
         where nobody clears Graham is a fact about the sector, not the stock.
      2. The framework must APPLY (a utility cannot have Greenblatt conviction).
      3. Percentiles come from the PRE-gate frame, so "top decile" means top
         decile of everything investable, not of whatever survived the gate.
    """
    dom = max(weights, key=weights.get)
    g = gated.copy()
    if g.empty:
        g["_conviction_pct"] = np.nan
        g["_conviction_rank"] = np.nan
        g["_conviction_eligible"] = False
        return g, dom

    applies = g["_applicable"].apply(lambda a: dom in a)
    passes = g[PASS_FLAG[dom]].fillna(False).astype(bool)
    pct = g[f"_pct_{dom}"]

    g["_conviction_pct"] = pct.where(applies & passes)
    g["_conviction_eligible"] = applies & passes & (pct >= CONVICTION_MIN_PCT)

    # A conviction pick was chosen on the DOMINANT FRAMEWORK's percentile, not on
    # _rank_score. Reporting its composite rank is therefore misleading, and
    # occasionally unsayable: an early run surfaced BUILDPRO at greenblatt_pct
    # 0.936 and rank_in_sector #74 of 74. Both numbers were true. "Ranked last of
    # 74, and we bought it" is not a sentence any user accepts, however
    # well-founded. Give the trace the rank that actually did the choosing.
    g["_conviction_rank"] = (g.groupby("sector")[f"_pct_{dom}"]
                              .rank(ascending=False, method="first")
                              .where(applies & passes))
    return g, dom


def _fill(pool: pd.DataFrame, q: dict, corr, k_conviction: int = 0) -> tuple[list, dict]:
    """pool carries a boolean `_conviction` column. Conviction rows are invisible
    to the merit passes and are admitted only in their own pass, AFTER every IPS
    minimum is satisfied. They consume FREE slots, never quota slots."""
    # Conviction rows are NOT a separate population — they live in the same
    # gated pool and may well be taken on merit. The sleeve only fires for
    # specialists the merit passes left behind. That is the entire point.
    eligible = (pool["_conviction_eligible"] if "_conviction_eligible" in pool.columns
                else pd.Series(False, index=pool.index))
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

    # Pass 4: the conviction sleeve. AFTER every IPS minimum is met, so a
    # long-shot can never displace the large-cap floor or sector breadth.
    # Ordered by the dominant framework's percentile, not by _rank_score —
    # _rank_score is precisely the number that buried these stocks.
    taken_conv = 0
    while taken_conv < k_conviction and len(chosen) < q["n"]:
        c = remaining(eligible)
        if c.empty:
            break
        # Ordered by the dominant framework's percentile, NOT by _rank_score.
        # _rank_score is precisely the number that buried these stocks.
        take(c.sort_values("_conviction_pct", ascending=False).index[0], "conviction")
        taken_conv += 1

    # Pass 5: free fill, pure merit order.
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

    # RANK FIRST, GATE SECOND.
    #
    # Percentiles must be computed against everything INVESTABLE, not against
    # the gated subset. Otherwise "top decile" means something different at a
    # 4+ gate (≈100 peers) than at 2+ (≈450), every survivor is renormalized
    # upward, and the gate silently cancels its own effect: a distinctive 2/5
    # Graham name loses exactly the distinctiveness that admitting it was
    # supposed to surface.
    #
    # A stock's standing is a property of the stock, not of the filter it
    # happened to pass through.
    # avoid_sectors binds on conviction candidates too, so it applies here —
    # before ranking, before the gate — not inside _tier2.
    avoid = set(policy.get("avoid_sectors") or [])
    if avoid:
        _n = len(df)
        df = df[~df["sector"].isin(avoid)]
        rejects["sector_excluded_by_user"] = _n - len(df)

    ranked = _tier3(df, policy)
    gated, annotated = _tier2(ranked, policy, rejects)

    weights = _resolve_weights(policy)
    k_conv = CONVICTION_SLOTS.get(int(policy.get("min_acceptable_score", 3)), 0)
    pool, dom_framework = _mark_conviction(gated, weights)
    if not k_conv:
        pool["_conviction_eligible"] = False
        dom_framework = None

    if pool.empty:
        return {"holdings": [], "warnings": ["No stock clears the gate you chose."],
                "rejects": rejects, "diagnostics": {}}

    # rank_in_sector / sector_depth are TRACE fields — "#1 of 14 in Technology"
    # must count the stocks that actually competed, i.e. post-gate.
    pool = pool.copy()
    pool["_rank_in_sector"] = (pool.groupby("sector")["_rank_score"]
                                   .rank(ascending=False, method="first").astype(int))
    pool["_sector_depth"] = pool.groupby("sector")["ticker"].transform("size")
    pool = pool.sort_values(["_rank_score", "_tiebreak"], ascending=False).reset_index(drop=True)

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

    chosen, slot_type = _fill(pool, q, corr, k_conviction=k_conv)

    # Rounded for display, but the validator re-normalizes off this value, so a
    # 1-decimal round at n=12 (8.3) reintroduces the epsilon it was meant to
    # avoid: 12 x 8.3 = 99.6, not 100. Keep enough precision that the sum is
    # exact to well inside the tolerance. Renderers should format, not the model.
    pct = round(100.0 / max(len(chosen), 1), 4)
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
                # Two ORTHOGONAL facts, and both are now informative.
                "slot_type": slot_type[idx],           # why it got a seat
                "gate_cleared": ("conviction_sleeve" if slot_type[idx] == "conviction"
                                 else "abstention_adjusted" if abstained
                                 else "merit"),        # how it cleared Tier 2
                "conviction_framework": (dom_framework if slot_type[idx] == "conviction"
                                         else None),
                "conviction_pct": (round(float(r["_conviction_pct"]), 3)
                                   if slot_type[idx] == "conviction"
                                   and not pd.isna(r.get("_conviction_pct"))
                                   else None),
                # The rank that CHOSE this stock, not the composite rank that
                # buried it. "#3 of 56 on Greenblatt in Basic Materials."
                "conviction_rank": (int(r["_conviction_rank"])
                                    if slot_type[idx] == "conviction"
                                    and not pd.isna(r.get("_conviction_rank"))
                                    else None),
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
            "conviction_slots": k_conv,
            "conviction_framework": dom_framework,
            "conviction_candidates": int(pool["_conviction_eligible"].sum()),
            "sector_counts": {h["sector"]: sum(1 for x in holdings if x["sector"] == h["sector"])
                              for h in holdings},
            "tier_counts": {t: sum(1 for x in holdings if x["risk_tier"] == t)
                            for t in ("Large", "Mid", "Small", "Micro")},
        },
    }

def investable_tickers(universe_df: pd.DataFrame, sip_amount: float,
                       avoid_sectors=None, limit: int = 250) -> list[str]:
    """Tier-1 survivors, best-scored first. Callers use this to decide which
    price histories to download — WITHOUT reimplementing the floor.

    Single source of truth: if MIN_TURNOVER changes, this changes with it.
    """
    df = _tier1(universe_df.copy(), sip_amount, {})
    if avoid_sectors:
        df = df[~df["sector"].isin(set(avoid_sectors))]
    return (df.sort_values("score", ascending=False)
              .head(limit)["ticker"].tolist())

# ══════════════════════════════════════════════════════════════════════════
# LANDING SCREEN — improving businesses, not perfect scores
# ══════════════════════════════════════════════════════════════════════════
# NOT the 5-of-5 list. A watchlist exists to be WATCHED, and the alert that
# brings a user back is watchlist_score_up. A perfect score has nowhere to go
# but down. A 3-of-5 that fails only Graham becomes a 4-of-5 the day its price
# falls — which teaches the correct reflex: a price drop is GOOD news, because
# it widens margin of safety.
#
# We do NOT claim "the market hasn't noticed yet". We measured that claim and
# could not support it. `pe_vs_avg` is a percent deviation from a stock's own
# 4-year average P/E, and P/E has EARNINGS in the denominator — so any stock we
# selected on trajectory and sorted by earnings CAGR shows a depressed P/E
# versus its own history *by construction*. Bharti Airtel reads -55 while
# sitting at an all-time high. The metric measured the growth we conditioned on.
#
# The honest claim is the failure pattern: graham appeared in 65 of the 73
# three-of-five stocks, exactly as its 5.6% base rate predicts.
#     "These businesses are measurably improving. Graham would tell you they
#      aren't cheap. Both are true, and which matters is up to you."
MIN_BASE_NET_INCOME = 10e7   # ₹10 crore


def improving_businesses(universe_df: pd.DataFrame, limit: int = 25) -> pd.DataFrame:
    """Investable, quality-passing, improving, and NOT unanimous."""
    df = _tier1(universe_df.copy(), sip_amount=float("inf"), rejects={})
    df = _attach_applicable(df)

    df["attainable"] = df["_applicable"].apply(len)
    df["score_applicable"] = [
        sum(1 for f in r["_applicable"] if bool(r.get(PASS_FLAG[f], False)))
        for _, r in df.iterrows()
    ]
    df["abstained"] = df["_applicable"].apply(
        lambda a: ", ".join(f for f in FRAMEWORKS if f not in a))
    df["failed"] = [
        ", ".join(f for f in r["_applicable"] if not bool(r.get(PASS_FLAG[f], False)))
        for _, r in df.iterrows()
    ]

    m = (df["trajectory_pass"].fillna(False).astype(bool)
         # 3 .. attainable-1: strong, and not unanimous. Excludes perfect scores.
         & (df["score_applicable"] >= 3)
         & (df["score_applicable"] <= df["attainable"] - 1)
         # Base-effect guard. Nobody compounds earnings at 226% for three years:
         # SAILIFE, WABAG and PRIVISCL are recoveries off a near-zero base, and
         # sorting on ni_cagr_3y would put the noisiest names on the front page.
         & (df["net_income_y3"].fillna(0) >= MIN_BASE_NET_INCOME))

    # revenue_cagr_3y, never ni_cagr_3y. Revenue has no zero-base pathology.
    return (df[m]
            .sort_values(["trajectory_score", "revenue_cagr_3y", "pe"],
                         ascending=[False, False, True])
            .head(limit))

# ══════════════════════════════════════════════════════════════════════════
# THESIS DRIFT  (Sprint 13 §2)
#
# A review must show what CHANGED since purchase, not re-explain the holding.
# diff_thesis compares the recorded reason a stock was bought (its entry _trace,
# stored on holdings at registration) against its status in a fresh
# select_portfolio run over today's universe. Pure: no Streamlit, no network,
# no LLM. The caller renders the diff; the LLM, if used at all, only phrases
# these deterministic facts into prose — it never decides the classification.
# ══════════════════════════════════════════════════════════════════════════

# The one slot_type that means "held only because the conviction sleeve fired" —
# a specialist the merit passes left behind. The other four (cap_quota_large,
# cap_quota_mid, breadth, free) all earned a seat without the sleeve.
_CONVICTION_SLOTS = {"conviction"}


def _on_conviction(trace: dict) -> bool:
    return bool(trace) and trace.get("slot_type") in _CONVICTION_SLOTS


def _trace_facts(trace: dict) -> dict:
    """The subset of a _trace used to render a thesis line. Defensive to
    partial/older traces and to manual holdings with no trace at all."""
    if not trace:
        return {}
    return {
        "slot_type": trace.get("slot_type"),
        "gate_cleared": trace.get("gate_cleared"),
        "conviction_framework": trace.get("conviction_framework"),
        "conviction_rank": trace.get("conviction_rank"),
        "conviction_pct": trace.get("conviction_pct"),
        "sector": trace.get("sector"),
        "rank_in_sector": trace.get("rank_in_sector"),
        "sector_depth": trace.get("sector_depth"),
        "score_applicable": trace.get("score_applicable"),
        "effective_gate": trace.get("effective_gate"),
        "passed": list(trace.get("passed", [])),
        "failed": list(trace.get("failed", [])),
    }


def _thesis_changes(entry: dict, current: dict) -> list:
    """Ordered, deterministic list of what moved between entry and today.
    Each item is {"field", "from", "to"}. Empty when nothing tracked changed."""
    changes = []
    e, c = _trace_facts(entry), _trace_facts(current)

    # Rank within the sector pool — the margin narrowing or widening.
    if (e.get("rank_in_sector"), e.get("sector_depth")) != \
       (c.get("rank_in_sector"), c.get("sector_depth")):
        changes.append({"field": "rank_in_sector",
                        "from": (e.get("rank_in_sector"), e.get("sector_depth")),
                        "to": (c.get("rank_in_sector"), c.get("sector_depth"))})

    # How many applicable frameworks it passes now.
    if e.get("score_applicable") != c.get("score_applicable"):
        changes.append({"field": "score_applicable",
                        "from": e.get("score_applicable"),
                        "to": c.get("score_applicable")})

    # Which specific frameworks flipped, in each direction.
    e_pass, c_pass = set(e.get("passed", [])), set(c.get("passed", []))
    if sorted(c_pass - e_pass):
        changes.append({"field": "newly_passing", "from": None,
                        "to": sorted(c_pass - e_pass)})
    if sorted(e_pass - c_pass):
        changes.append({"field": "newly_failing", "from": None,
                        "to": sorted(e_pass - c_pass)})

    # Conviction rank drift, when both entry and today were conviction picks.
    if e.get("conviction_rank") is not None and c.get("conviction_rank") is not None \
       and e.get("conviction_rank") != c.get("conviction_rank"):
        changes.append({"field": "conviction_rank",
                        "from": e.get("conviction_rank"),
                        "to": c.get("conviction_rank")})

    # The seat itself changed character (e.g. large-cap floor -> pure merit).
    if e.get("slot_type") != c.get("slot_type"):
        changes.append({"field": "slot_type",
                        "from": e.get("slot_type"), "to": c.get("slot_type")})

    return changes


def diff_thesis(entry_trace: dict | None, current_trace: dict | None,
                still_investable: bool = True) -> dict:
    """
    Classify how a holding's thesis has drifted since purchase.

    entry_trace     : _trace stored at registration, or None if never recorded
                      (manual holding, or bought before trace capture shipped).
    current_trace   : the same ticker's _trace in a fresh selection over today's
                      universe, or None if it was not selected today.
    still_investable: is the ticker still in the Tier-1 pool today? Only consulted
                      when current_trace is None, to tell "outranked" (still
                      investable, just not top-N) apart from "fell out of the
                      pool". Defaults True so a caller that cannot check pool
                      membership never falsely claims the turnover floor failed.

    Returns {"drift", "entry", "current", "changes"} where drift is one of:
      no_trace              no entry thesis to compare against.
      no_longer_investable  fell out of the Tier-1 pool (turnover/quality floor);
                            would NOT be bought today.
      outranked             still investable, but other names now rank above it;
                            not in today's portfolio.
      now_merit             was held on conviction, now clears the gate on merit
                            (thesis strengthened).
      now_conviction        was a merit pick, now survives only via the
                            conviction sleeve (thesis weakened).
      still_selected        same basis; see `changes` for the delta.
    """
    if not entry_trace:
        return {"drift": "no_trace", "entry": {},
                "current": _trace_facts(current_trace), "changes": []}

    if not current_trace:
        # Not selected in today's re-run. Distinguish two very different
        # realities: still in the pool but outranked (soft) vs fell out of the
        # pool entirely — the turnover/quality floor, the real sell signal.
        drift = "outranked" if still_investable else "no_longer_investable"
        return {"drift": drift,
                "entry": _trace_facts(entry_trace), "current": None, "changes": []}

    was_conv, now_conv = _on_conviction(entry_trace), _on_conviction(current_trace)
    drift = ("now_merit" if (was_conv and not now_conv)
             else "now_conviction" if (not was_conv and now_conv)
             else "still_selected")

    return {"drift": drift,
            "entry": _trace_facts(entry_trace),
            "current": _trace_facts(current_trace),
            "changes": _thesis_changes(entry_trace, current_trace)}
