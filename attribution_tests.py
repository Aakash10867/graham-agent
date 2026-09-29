"""
attribution_tests.py — Sprint 17, step 3: the tests. READ-ONLY; reads
attribution_out/, writes attribution_out/test_log.txt and results files.

Every expectation this script is judged against was locked BEFORE it existed,
in PREREGISTRATION_SPRINT17.md (2026-09-23). The script prints the prediction
beside each result and a verdict of CONSISTENT / CONTRADICTED / UNDETERMINED —
UNDETERMINED whenever the 95% band contains zero, which on ~40 days of data
will be most of them. That is the expected, honest outcome, not a failure.

Tests
  1. Factor regressions — 4-factor (Carhart) and 5-factor (+ low-vol) on the
     calendar-time portfolio of Kordent's picks, plus the score-0/1 portfolio
     and the high-minus-low spread. v8 window headline, v1+ sensitivity.
  2. In-house beta of the picks portfolio against each benchmark ETF — the
     number that will replace yfinance's `beta` and BETA_UNCERTAINTY.
  3. Event study — cumulative market-adjusted returns +/-20 trading days
     around score changes, split BUSINESS-driven vs PRICE-driven.
  4. Holdings-based style — size and cheapness percentile of the picks on
     every snapshot.

Run:  py attribution_tests.py      (no network; under a minute)

STATISTICAL CHOICES, stated once:
  * OLS with Newey-West standard errors (5 lags). The calendar-time portfolio
    averages overlapping 63-day cohorts, so consecutive days share most of
    their holdings; plain OLS errors would be too narrow.
  * 95% bands use the t distribution on the residual degrees of freedom.
  * Alpha is reported annualised (x252) because that is the unit the
    pre-registered display gate (+/-5% a year) is written in.
  * Event-study bands are cross-sectional and assume independent events. They
    are not: events cluster on the same dates. The bands are therefore too
    narrow, and are labelled so.
"""
import json
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

import attribution_factors as af

OUT = "attribution_out"
LOG_FILE = os.path.join(OUT, "test_log.txt")
NW_LAGS = 5
EVENT_WIN = 20
EVENT_MCAP_FLOOR = 500e7          # selector.MIN_MARKET_CAP — the buyable size range
DISPLAY_MIN_DAYS = 250            # pre-registered gate
DISPLAY_MAX_ALPHA_HALFWIDTH = 0.05
BUSINESS_COLS = ["revenue_y0", "revenue_y1", "net_income_y0", "net_income_y1",
                 "equity_y0", "total_debt_y0", "eps"]
BENCHMARKS = ["NIFTYBEES.NS", "MID150BEES.NS", "SMALLCAP.NS", "^CRSLDX"]

_log = []


def log(m=""):
    print(m)
    _log.append(str(m))


try:
    from scipy import stats as _st

    def tcrit(df):
        return float(_st.t.ppf(0.975, max(df, 1)))
except Exception:                                     # pragma: no cover
    def tcrit(df):
        return 1.96


# ══════════════════════════════════════════════════════════════════════════
# OLS with Newey-West errors
# ══════════════════════════════════════════════════════════════════════════
def ols_nw(y, X, lags=NW_LAGS):
    """Returns dict(coef, se, t, lo, hi, r2, n, k) with X including a constant."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    S = (X * e[:, None]).T @ (X * e[:, None])
    for l in range(1, min(lags, n - 1) + 1):
        w = 1 - l / (lags + 1)
        G = (X[l:] * e[l:, None]).T @ (X[:-l] * e[:-l, None])
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    tc = tcrit(n - k)
    ss_tot = ((y - y.mean()) ** 2).sum()
    return {"coef": b, "se": se, "t": b / np.where(se > 0, se, np.nan),
            "lo": b - tc * se, "hi": b + tc * se,
            "r2": 1 - (e ** 2).sum() / ss_tot if ss_tot > 0 else np.nan,
            "n": n, "k": k}


def verdict(lo, hi, expect):
    """expect: '+', '-', '0'. Returns CONSISTENT / CONTRADICTED / UNDETERMINED."""
    if expect == "0":
        return "CONSISTENT" if lo <= 0 <= hi else "CONTRADICTED"
    if lo <= 0 <= hi:
        return "UNDETERMINED"
    positive = lo > 0
    return "CONSISTENT" if (positive == (expect == "+")) else "CONTRADICTED"


# ══════════════════════════════════════════════════════════════════════════
# 1. Factor regressions
# ══════════════════════════════════════════════════════════════════════════
PREDICTIONS = {        # PREREGISTRATION_SPRINT17.md section 2
    "SMB": ("+", "positive and large"),
    "HML": ("+", "positive, smaller than a raw value screen"),
    "WML": ("-", "negative"),
    "LMH": ("+", "positive"),
    "alpha": ("0", "indistinguishable from zero"),
}


def regressions(kor, fac):
    results = {}
    for window in ("v8", "v1plus"):
        k = kor[kor["window"] == window]
        j = k.join(fac, how="inner").dropna(subset=["high", "low", "MKT_RF", "SMB",
                                                    "HML", "WML", "LMH"])
        for model, cols in (("4F", ["MKT_RF", "SMB", "HML", "WML"]),
                            ("5F", ["MKT_RF", "SMB", "HML", "WML", "LMH"])):
            for series in ("high", "low", "spread"):
                y = j[series] - (0 if series == "spread" else j["RF"])
                X = np.column_stack([np.ones(len(j))] + [j[c] for c in cols])
                if len(j) < len(cols) + 5:
                    continue
                r = ols_nw(y, X)
                names = ["alpha"] + [c.replace("MKT_RF", "MKT") for c in cols]
                results[(window, model, series)] = (r, names, len(j),
                                                    j.index.min(), j.index.max())
    return results


def print_regressions(results):
    log("\n" + "=" * 78)
    log("1. FACTOR REGRESSIONS — Newey-West(5), 95% bands, alpha annualised")
    log("=" * 78)
    for (window, model, series), (r, names, n, d0, d1) in results.items():
        if series != "high" and not (window == "v8" and model == "5F"):
            continue
        log(f"\n[{window} | {model} | {series}] {n} days {d0.date()}..{d1.date()} "
            f"| R2 {r['r2']:.2f}")
        for i, nm in enumerate(names):
            c, lo, hi, t = r["coef"][i], r["lo"][i], r["hi"][i], r["t"][i]
            if nm == "alpha":
                log(f"  {nm:6s} {c*252:+8.1%} /yr   [{lo*252:+8.1%}, {hi*252:+8.1%}]  t={t:+5.2f}")
            else:
                log(f"  {nm:6s} {c:+8.3f}        [{lo:+8.3f}, {hi:+8.3f}]  t={t:+5.2f}")


def prediction_table(results):
    log("\n" + "=" * 78)
    log("PRE-REGISTERED PREDICTIONS vs RESULT — picks portfolio, v8 window, 5-factor")
    log("=" * 78)
    key = ("v8", "5F", "high")
    if key not in results:
        log("  (v8 5-factor regression could not be run)")
        return {}
    r, names, n, _, _ = results[key]
    out = {}
    for nm, (expect, text) in PREDICTIONS.items():
        i = names.index(nm)
        lo, hi, c = r["lo"][i], r["hi"][i], r["coef"][i]
        v = verdict(lo, hi, expect)
        scale = 252 if nm == "alpha" else 1
        fmt = "{:+.1%}" if nm == "alpha" else "{:+.3f}"
        log(f"  {nm:5s} predicted {text:45s} got {fmt.format(c*scale)} "
            f"[{fmt.format(lo*scale)}, {fmt.format(hi*scale)}]  -> {v}")
        out[nm] = {"predicted": text, "coef": c * scale, "lo": lo * scale,
                   "hi": hi * scale, "verdict": v}

    i = names.index("alpha")
    half = (r["hi"][i] - r["lo"][i]) / 2 * 252
    gate_days = n >= DISPLAY_MIN_DAYS
    gate_band = half < DISPLAY_MAX_ALPHA_HALFWIDTH
    log(f"\n  DISPLAY GATE: days {n} >= {DISPLAY_MIN_DAYS}? {gate_days} | "
        f"alpha half-width {half:.1%} < {DISPLAY_MAX_ALPHA_HALFWIDTH:.0%}? {gate_band} "
        f"-> {'MAY DISPLAY' if gate_days and gate_band else 'PROVISIONAL — do not show users'}")
    out["_gate"] = {"days": n, "alpha_halfwidth": half,
                    "display": bool(gate_days and gate_band)}
    return out


# ══════════════════════════════════════════════════════════════════════════
# 2. In-house beta
# ══════════════════════════════════════════════════════════════════════════
def inhouse_beta(kor, prices):
    log("\n" + "=" * 78)
    log("2. IN-HOUSE BETA — picks portfolio vs each benchmark (replaces yfinance beta)")
    log("=" * 78)
    out = {}
    k = kor[kor["window"] == "v8"]["high"].dropna()
    log("  benchmark data check — a stale benchmark breaks every portfolio that uses it:")
    for b in BENCHMARKS:
        if b in prices.columns:
            s_ = prices[b]
            recent = s_.loc[s_.index >= k.index.min()]
            log(f"    {b:14s} last price {s_.last_valid_index().date()} | "
                f"priced {int(recent.notna().sum())}/{len(recent)} days in the window")
    for b in BENCHMARKS:
        if b not in prices.columns:
            log(f"  {b:14s} not in price file")
            continue
        rb = prices[b].pct_change(fill_method=None).rename("b")
        j = pd.concat([k.rename("p"), rb], axis=1, join="inner").dropna()
        if len(j) < 20:
            log(f"  {b:14s} only {len(j)} overlapping days")
            continue
        r = ols_nw(j["p"], np.column_stack([np.ones(len(j)), j["b"]]))
        te = (j["p"] - j["b"]).std() * np.sqrt(252)
        corr = j.corr().iloc[0, 1]
        log(f"  {b:14s} beta {r['coef'][1]:.2f} [{r['lo'][1]:.2f}, {r['hi'][1]:.2f}] "
            f"| corr {corr:.2f} | tracking error {te:.1%}/yr | {len(j)} days")
        out[b] = {"beta": r["coef"][1], "lo": r["lo"][1], "hi": r["hi"][1],
                  "corr": corr, "tracking_error": te, "days": len(j)}
    log("  This band comes from the data and narrows as days accrue — the property")
    log("  the fixed +/-0.15 BETA_UNCERTAINTY on an unexplained yfinance beta lacked.")
    return out


# ══════════════════════════════════════════════════════════════════════════
# 3. Event study
# ══════════════════════════════════════════════════════════════════════════
def build_events(panel, prices):
    v8 = panel[panel["schema_version"] >= 8].copy()
    v8 = v8[pd.to_numeric(v8["market_cap"], errors="coerce") >= EVENT_MCAP_FLOOR]
    snaps = sorted(v8["snap_date"].unique())
    rows = []
    for a, b in zip(snaps[:-1], snaps[1:]):
        A = v8[v8["snap_date"] == a].drop_duplicates("ticker").set_index("ticker")
        B = v8[v8["snap_date"] == b].drop_duplicates("ticker").set_index("ticker")
        common = A.index.intersection(B.index).intersection(prices.columns)
        sa, sb = A.loc[common, "score"], B.loc[common, "score"]
        ch = common[(sa != sb) & sa.notna() & sb.notna()]
        for t in ch:
            biz = False
            for c in BUSINESS_COLS:
                if c not in A.columns:
                    continue
                x, y = pd.to_numeric(A.at[t, c], errors="coerce"), \
                    pd.to_numeric(B.at[t, c], errors="coerce")
                if (pd.isna(x) != pd.isna(y)) or (pd.notna(x) and pd.notna(y)
                                                  and abs(x - y) > 1e-9 * max(1, abs(x))):
                    biz = True
                    break
            rows.append({"ticker": t, "date": pd.Timestamp(b),
                         "direction": "up" if sb[t] > sa[t] else "down",
                         "driver": "business" if biz else "price",
                         "from": int(sa[t]), "to": int(sb[t])})
    return pd.DataFrame(rows)


def event_study(panel, prices, fac):
    log("\n" + "=" * 78)
    log(f"3. EVENT STUDY — size-matched CAR, -{EVENT_WIN}..+{EVENT_WIN} trading days, "
        f"v8 snapshots, market cap >= Rs {EVENT_MCAP_FLOOR/1e7:.0f} Cr")
    log("=" * 78)
    ev = build_events(panel, prices)
    if ev.empty:
        log("  no events")
        return {}, pd.DataFrame()
    rets = prices.pct_change(fill_method=None)
    cal = rets.index
    # SIZE-MATCHED benchmark (fixed 2026-09-30). The first run adjusted by the
    # value-weighted market, which is large-cap dominated. Over this window
    # small and mid caps beat large caps by ~0.09%/day (the SMB mean), so every
    # event group — upgrades AND downgrades — showed the same +2% post-event
    # drift. That was size, not the score. Each event is now measured against
    # the equal-weighted return of stocks in its own size tercile (>= Rs 500 Cr
    # universe, terciles fixed on the event's own snapshot).
    v8 = panel[(panel["schema_version"] >= 8)
               & (pd.to_numeric(panel["market_cap"], errors="coerce") >= EVENT_MCAP_FLOOR)]
    bench = {}
    for d, g in v8.groupby("snap_date"):
        g = g[g["ticker"].isin(rets.columns)].drop_duplicates("ticker")
        terc = pd.qcut(g["market_cap"].rank(method="first"), 3, labels=False)
        for q in range(3):
            members = g.loc[terc == q, "ticker"].tolist()
            bench[(pd.Timestamp(d), q)] = (rets[members].mean(axis=1, skipna=True), set(members))
    paths = {}
    kept = []
    for _, e in ev.iterrows():
        pos = cal.searchsorted(e["date"], side="right") - 1     # day 0 = snapshot day
        if pos - EVENT_WIN < 0 or pos + EVENT_WIN >= len(cal):
            continue
        win = cal[pos - EVENT_WIN: pos + EVENT_WIN + 1]
        bq = next(((ser, m) for (d, q), (ser, m) in bench.items()
                   if d == e["date"] and e["ticker"] in m), None)
        if bq is None:
            continue
        ar = (rets.loc[win, e["ticker"]] - bq[0].loc[win]).to_numpy()
        if np.isnan(ar).mean() > 0.2:
            continue
        paths[len(kept)] = np.nan_to_num(ar)
        kept.append(e)
    kept = pd.DataFrame(kept).reset_index(drop=True)
    log(f"  events: {len(ev)} score changes | with a complete window: {len(kept)}")
    if kept.empty:
        return {}, ev
    M = np.vstack([paths[i] for i in range(len(kept))])
    rel = np.arange(-EVENT_WIN, EVENT_WIN + 1)
    out = {}
    for (drv, dirn), g in kept.groupby(["driver", "direction"]):
        A = M[g.index]
        car = A.cumsum(axis=1)
        pre = car[:, EVENT_WIN - 1]                       # through day -1
        day0 = A[:, EVENT_WIN]
        post = car[:, -1] - car[:, EVENT_WIN]             # days +1..+20
        n = len(g)

        def band(x):
            m, s = x.mean(), x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
            return m, m - 1.96 * s, m + 1.96 * s
        p, d0, po = band(pre), band(day0), band(post)
        log(f"\n  {drv.upper():8s} {dirn:4s} n={n:4d}")
        log(f"    CAR -20..-1  {p[0]:+7.2%}  [{p[1]:+7.2%}, {p[2]:+7.2%}]")
        log(f"    AR  day 0    {d0[0]:+7.2%}  [{d0[1]:+7.2%}, {d0[2]:+7.2%}]")
        log(f"    CAR +1..+20  {po[0]:+7.2%}  [{po[1]:+7.2%}, {po[2]:+7.2%}]")
        out[f"{drv}_{dirn}"] = {"n": n, "pre": p, "day0": d0, "post": po,
                                "mean_car_path": car.mean(axis=0).tolist(),
                                "rel_days": rel.tolist()}
    log("\n  Reading it (prediction 6 is about BUSINESS rows only):")
    log("    PRICE rows are the control. A price-driven upgrade happens BECAUSE the")
    log("    price fell (and a downgrade because it rose), so a negative day-0 return")
    log("    on price upgrades and a positive one on price downgrades are built in.")
    log("    Seeing them confirms the business/price split works; they are not findings.")
    log("    BUSINESS upgrade with positive pre-CAR = the market moved first; Kordent")
    log("    read the news late. Flat pre-CAR then a jump or drift after = Kordent early.")
    log("    Day 0 of a BUSINESS row can still carry a same-day price move (results day),")
    log("    so read prediction 6 from the PRE-event CAR, which day 0 cannot contaminate.")
    log("    Bands assume independent events; events share dates, so bands are too narrow.")
    return out, kept


# ══════════════════════════════════════════════════════════════════════════
# 4. Holdings-based style
# ══════════════════════════════════════════════════════════════════════════
def style_drift(panel):
    log("\n" + "=" * 78)
    log("4. HOLDINGS-BASED STYLE — picks' average percentile vs the broad market "
        "(>= Rs 100 Cr), per snapshot")
    log("=" * 78)
    rows = []
    for d, g in panel[panel["schema_version"] >= 1].groupby("snap_date"):
        g = g[pd.to_numeric(g["market_cap"], errors="coerce") >= af.MCAP_FLOOR].copy()
        if g.empty:
            continue
        g["size_pct"] = g["market_cap"].rank(pct=True)
        bm = (1 / g["pb"]).where(g["pb"] > 0)
        g["value_pct"] = bm.rank(pct=True)
        picks = g[g["investable"] & (g["score"] >= 4)]
        if len(picks) < 5:
            continue
        rows.append({"date": d, "n_picks": len(picks),
                     "size_pct": picks["size_pct"].mean(),
                     "value_pct": picks["value_pct"].mean(),
                     "median_mcap_cr": picks["market_cap"].median() / 1e7,
                     "schema": int(g["schema_version"].mode().iloc[0])})
    s = pd.DataFrame(rows).set_index("date")
    s.to_csv(os.path.join(OUT, "style_drift.csv"))
    if s.empty:
        log("  no snapshots")
        return {}
    for w, sub in (("v8", s[s["schema"] >= 8]), ("v1plus", s)):
        log(f"  {w:7s} {len(sub)} snapshots | size pct {sub['size_pct'].mean():.2f} "
            f"(range {sub['size_pct'].min():.2f}-{sub['size_pct'].max():.2f}) | "
            f"cheapness pct {sub['value_pct'].mean():.2f} "
            f"(range {sub['value_pct'].min():.2f}-{sub['value_pct'].max():.2f}) | "
            f"median mcap Rs {sub['median_mcap_cr'].median():,.0f} Cr")
    log("  0.50 = the broad-market middle. Above 0.5 on size = bigger than typical;")
    log("  above 0.5 on cheapness = cheaper than typical. Written to style_drift.csv.")
    return {"v8_size_pct": float(s[s["schema"] >= 8]["size_pct"].mean()),
            "v8_value_pct": float(s[s["schema"] >= 8]["value_pct"].mean())}


def main():
    log(f"attribution_tests.py — run {date.today()} — PROVISIONAL, not for users")
    for f in ("factors_daily.csv", "kordent_daily.csv"):
        if not os.path.exists(os.path.join(OUT, f)):
            sys.exit(f"Missing {OUT}/{f}. Run py attribution_factors.py first.")
    fac = pd.read_csv(os.path.join(OUT, "factors_daily.csv"), index_col=0, parse_dates=True)
    kor = pd.read_csv(os.path.join(OUT, "kordent_daily.csv"), index_col=0, parse_dates=True)
    panel, prices = af.load()

    res = regressions(kor, fac)
    print_regressions(res)
    preds = prediction_table(res)
    beta = inhouse_beta(kor, prices)
    events, kept = event_study(panel, prices, fac)
    if len(kept):
        kept.to_csv(os.path.join(OUT, "events.csv"), index=False)
    style = style_drift(panel)

    summary = {"run": str(date.today()), "provisional": True,
               "predictions_v8_5F": preds,
               "inhouse_beta_v8": beta,
               "event_study": {k: {kk: vv for kk, vv in v.items()
                                   if kk not in ("mean_car_path", "rel_days")}
                               for k, v in events.items()},
               "style": style}
    with open(os.path.join(OUT, "attribution_results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: float(o) if hasattr(o, "__float__") else str(o))
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(_log))
    print(f"\nWrote {LOG_FILE}. Paste it back.")


if __name__ == "__main__":
    main()
