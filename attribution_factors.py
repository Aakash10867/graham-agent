"""
attribution_factors.py — Sprint 17, step 2: build the scoreboards and Kordent's
own return series. READ-ONLY; reads attribution_out/, writes attribution_out/.

No regression, no alpha, no event study here — those live in step 3, and the
expectations they are judged against are locked in PREREGISTRATION_SPRINT17.md.

Outputs
  attribution_out/factors_daily.csv     date, MKT_RF, SMB, HML, WML, LMH, RF (+ breadth)
  attribution_out/kordent_daily.csv     date, high, low, spread, cohorts, names, window
  attribution_out/factors_hist.csv      2021-2025 price-only factors, for the IIM-A check
  attribution_out/factor_log.txt        construction record and the IIM-A comparison

Run:  py attribution_factors.py      (no network; ~1 minute)

CONSTRUCTION DECISIONS — the reasoning, so a later reader does not re-litigate.

* MONTHLY formation, not daily. Size and book-to-market both move with today's
  price, so a daily-rebalanced sort buys whatever fell today and sells whatever
  rose — the sort picks up bid-ask bounce and reports it as a factor return.
  Fama-French rebalance annually (size/value) and monthly (momentum); monthly is
  the compromise this 3-month archive can support.

* ONE-DAY SKIP between formation and entry, for BOTH the factors and Kordent's
  portfolios. The snapshot is stamped at a close; entering at that same close is
  a look-ahead, and entering at the next close removes the same-day bounce that
  would otherwise flatter anything sorted on price. Kordent's own score contains
  price terms, so it inherits exactly this bias if entry is not lagged.

* VALUE-WEIGHTED factor legs, EQUAL-WEIGHTED Kordent legs. The factors have to
  match the published definition to be comparable to IIM-A. Kordent equal-weights
  because `_equal_pct` is what the product actually does — which is itself a size
  tilt, and one the regression should therefore attribute to SMB rather than hide.

* STALENESS, TWO TIERS. A stock that does not move on 10%+ of days is unpriceable
  (selector.MIN_NONZERO_RETURN_FRAC, the covariance lesson from the dropped
  diversification_rank). Size/value legs need 40 observations; momentum and
  volatility legs need 200, because a 12-1 lookback with gaps is not a lookback.
  The 560 BSE tickers that survive only through backtest_price_cache.csv (from
  2026-07-06) therefore enter SMB/HML and cannot enter WML/LMH. Recorded, not
  papered over.

* B/M = 1/pb, and pb <= 0 is EXCLUDED from the value sort rather than ranked
  last. Negative book value is a different object from expensive.

* THE MARKET LEG is the value-weighted return of the factor universe itself, not
  an index. Self-consistent with the other legs, and it includes dividends
  through adjusted closes, which the price indices do not.
"""
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

OUT = "attribution_out"
PANEL_FILE = os.path.join(OUT, "panel.csv.gz")
PRICE_FILE = os.path.join(OUT, "prices.csv.gz")
IIMA_FILE = os.path.join(OUT, "iima_daily.csv")
SOURCES_FILE = os.path.join(OUT, "price_sources.csv")
LOG_FILE = os.path.join(OUT, "factor_log.txt")

MCAP_FLOOR = 100e7              # Rs 100 Cr — the broad-market floor
V8_FROM = pd.Timestamp("2026-07-30")   # today's scorer; the primary window
HOLD_DAYS = 63                  # backtest_runner.HORIZON_TRADING_DAYS
MOM_LOOKBACK, MOM_SKIP = 252, 21       # 12-1 momentum
VOL_WINDOW = 252
MIN_OBS_SIZE, MIN_OBS_PRICE_FACTOR = 40, 200
NONZERO_FRAC = 0.90             # selector.MIN_NONZERO_RETURN_FRAC
FALLBACK_RF = 0.07

_log = []


def log(m=""):
    print(m)
    _log.append(str(m))


def load():
    if not (os.path.exists(PANEL_FILE) and os.path.exists(PRICE_FILE)):
        sys.exit(f"Missing {PANEL_FILE} or {PRICE_FILE}. Unzip the "
                 f"attribution_out artifact into this folder first.")
    panel = pd.read_csv(PANEL_FILE, low_memory=False)
    panel["snap_date"] = pd.to_datetime(panel["snap_date"])
    for c in ("market_cap", "pb", "score", "schema_version"):
        panel[c] = pd.to_numeric(panel[c], errors="coerce")
    panel["investable"] = panel["investable"].astype("string").str.lower().isin(["true", "1"])

    # Dual-listed duplicates. Snapshots from 2026-09-28 until the symbol-based
    # dedup landed carry some companies twice: as NAME.NS and as a numeric .BO
    # whose yf_symbol is NAME.BO. Counting both would double a company's weight
    # in every value-weighted leg. Drop the .BO copy when the .NS twin is in the
    # same snapshot.
    # Older snapshots carry no yf_symbol, so a second key: Yahoo's own company
    # name (the `name` column is Yahoo's longName, not the truncated BSE list),
    # which is identical for both listings of one company.
    is_bo = panel["ticker"].astype(str).str.endswith(".BO")
    is_ns = panel["ticker"].astype(str).str.endswith(".NS")
    ns_keys = set(zip(panel["snap_date"], panel["ticker"].astype(str)))
    nm = panel["name"].astype("string").str.strip().str.lower()
    ns_names = set(zip(panel.loc[is_ns, "snap_date"], nm[is_ns]))
    dup = is_bo & pd.Series([(d, n) in ns_names for d, n in zip(panel["snap_date"], nm)],
                            index=panel.index)
    if "yf_symbol" in panel.columns:
        stem = panel["yf_symbol"].astype("string").str.replace(".BO", "", regex=False)
        twin = stem + ".NS"
        dup |= is_bo & pd.Series([(d, t) in ns_keys for d, t in zip(panel["snap_date"], twin)],
                                 index=panel.index)
    if dup.any():
        log(f"[DATA] dropped {int(dup.sum())} dual-listed .BO duplicates "
            f"across {panel.loc[dup, 'snap_date'].nunique()} snapshots")
    panel = panel[~dup]
    prices = pd.read_csv(PRICE_FILE, index_col=0, parse_dates=True).sort_index()
    return panel, prices


def rf_series(index):
    """Daily risk-free, from the macro series when it is readable, else the
    documented fallback. One rate, one place (Sprint 16)."""
    try:
        import macro_read
        vals, ok = [], 0
        for d in index:
            r, status = macro_read.india_rfr(as_of=d.date())
            vals.append(r / 252.0)
            ok += status == "ok"
        log(f"[RF] macro_read supplied {ok}/{len(index)} days; rest fell back")
        return pd.Series(vals, index=index)
    except Exception as e:
        log(f"[RF] macro_read unavailable ({type(e).__name__}); "
            f"flat {FALLBACK_RF:.1%} used throughout")
        return pd.Series(FALLBACK_RF / 252.0, index=index)


# ══════════════════════════════════════════════════════════════════════════
# Sorting machinery
# ══════════════════════════════════════════════════════════════════════════
def quality_mask(rets, upto, min_obs):
    """Tickers priced well enough to be sorted: enough observations, and moving
    on at least NONZERO_FRAC of them."""
    w = rets.loc[:upto].tail(max(min_obs, 120))
    obs = w.count()
    nz = (w.fillna(0) != 0).sum()
    frac = (nz / obs.replace(0, np.nan)).fillna(0)
    return (obs >= min_obs) & (frac >= NONZERO_FRAC)


def vw_return(rets_slice, weights):
    """Value-weighted portfolio return per day. Weights are fixed at formation;
    within the holding month a name that stops pricing simply drops out."""
    w = weights.reindex(rets_slice.columns).astype(float)
    w = w[w > 0]
    if w.empty:
        return pd.Series(np.nan, index=rets_slice.index)
    r = rets_slice[w.index]
    ok = r.notna()
    wm = ok.mul(w, axis=1)
    denom = wm.sum(axis=1).replace(0, np.nan)
    return (r.fillna(0) * wm).sum(axis=1) / denom


def two_by_three(size, signal, rets_slice, weights, lo=0.30, hi=0.70):
    """FF-style 2x3 sort. Returns (high_leg - low_leg) averaged across size
    halves, plus the leg count, or (nan series, 0) if any leg is empty."""
    common = size.index.intersection(signal.dropna().index)
    if len(common) < 20:
        return pd.Series(np.nan, index=rets_slice.index), 0
    size, signal = size.loc[common], signal.loc[common]
    small = size <= size.median()
    q_lo, q_hi = signal.quantile(lo), signal.quantile(hi)
    legs = {}
    for s_name, s_mask in (("S", small), ("B", ~small)):
        for g_name, g_mask in (("L", signal <= q_lo), ("H", signal >= q_hi)):
            idx = signal.index[s_mask & g_mask]
            legs[s_name + g_name] = vw_return(rets_slice, weights.reindex(idx))
    n = min(int((small & (signal <= q_lo)).sum()), int((small & (signal >= q_hi)).sum()),
            int((~small & (signal <= q_lo)).sum()), int((~small & (signal >= q_hi)).sum()))
    if n == 0:
        return pd.Series(np.nan, index=rets_slice.index), 0
    high = (legs["SH"] + legs["BH"]) / 2
    low = (legs["SL"] + legs["BL"]) / 2
    return high - low, n


def smb_leg(size, signal, rets_slice, weights):
    """SMB from the same 2x3 size/value sort: small minus big, averaged across
    the three value groups."""
    common = size.index.intersection(signal.dropna().index)
    if len(common) < 20:
        return pd.Series(np.nan, index=rets_slice.index)
    size, signal = size.loc[common], signal.loc[common]
    small = size <= size.median()
    q_lo, q_hi = signal.quantile(0.30), signal.quantile(0.70)
    groups = {"L": signal <= q_lo, "M": (signal > q_lo) & (signal < q_hi), "H": signal >= q_hi}
    s_legs, b_legs = [], []
    for g in groups.values():
        s_legs.append(vw_return(rets_slice, weights.reindex(signal.index[small & g])))
        b_legs.append(vw_return(rets_slice, weights.reindex(signal.index[~small & g])))
    return sum(s_legs) / 3 - sum(b_legs) / 3


def smb_simple(size, rets_slice, weights):
    """Small minus big on size alone, value-weighted within each half. Used for
    the historical IIM-A check, where no point-in-time book value exists — the
    2x3 size/value SMB cannot be built without a value signal, and sorting size
    against itself produces empty legs."""
    if len(size) < 40:
        return pd.Series(np.nan, index=rets_slice.index)
    small = size.index[size <= size.median()]
    big = size.index[size > size.median()]
    return vw_return(rets_slice, weights.reindex(small)) - \
        vw_return(rets_slice, weights.reindex(big))


def momentum(prices, f, cal):
    """12-1: cumulative return from t-252 to t-21 trading days before formation."""
    hist = prices.loc[:f]
    if len(hist) < MOM_LOOKBACK:
        return pd.Series(dtype=float)
    start, end = hist.iloc[-MOM_LOOKBACK], hist.iloc[-MOM_SKIP]
    return (end / start - 1).replace([np.inf, -np.inf], np.nan)


# ══════════════════════════════════════════════════════════════════════════
# Live-window factors (the archive period)
# ══════════════════════════════════════════════════════════════════════════
def build_factors(panel, prices, rets, cal):
    snaps = sorted(panel["snap_date"].unique())
    # First snapshot of each month is the formation date.
    forms = []
    seen = set()
    for d in snaps:
        key = (d.year, d.month)
        if key not in seen:
            seen.add(key)
            forms.append(pd.Timestamp(d))
    log(f"[FACTORS] formations: {[str(f.date()) for f in forms]}")

    rows = []
    for i, f in enumerate(forms):
        entry = cal[cal > f]
        if not len(entry):
            continue
        t0 = entry[0]
        t1 = cal[cal > forms[i + 1]][0] if i + 1 < len(forms) else cal[-1] + pd.Timedelta(days=1)
        window = rets.loc[(rets.index >= t0) & (rets.index < t1)]
        if window.empty:
            continue

        snap = panel[panel["snap_date"] == f]
        snap = snap[snap["market_cap"] >= MCAP_FLOOR].dropna(subset=["ticker"])
        snap = snap.drop_duplicates("ticker").set_index("ticker")
        snap = snap[snap.index.isin(prices.columns)]

        ok_size = quality_mask(rets, f, MIN_OBS_SIZE)
        ok_price = quality_mask(rets, f, MIN_OBS_PRICE_FACTOR)
        mcap = snap["market_cap"]

        # size/value universe
        u = snap.index[ok_size.reindex(snap.index).fillna(False)]
        bm = (1.0 / snap.loc[u, "pb"]).where(snap.loc[u, "pb"] > 0)
        hml, n_hml = two_by_three(mcap.loc[u], bm, window, mcap)
        smb = smb_leg(mcap.loc[u], bm, window, mcap)
        mkt = vw_return(window, mcap.loc[u])

        # momentum / volatility universe (needs a year of history)
        p = snap.index[ok_price.reindex(snap.index).fillna(False)]
        mom = momentum(prices[p], f, cal)
        vol = rets[p].loc[:f].tail(VOL_WINDOW).std()
        wml, n_wml = two_by_three(mcap.loc[p], mom, window, mcap)
        lmh, n_lmh = two_by_three(mcap.loc[p], -vol, window, mcap)   # LOW minus high

        log(f"  {f.date()} -> {t0.date()}..{window.index[-1].date()}  "
            f"size/value universe {len(u)} (legs {n_hml})  "
            f"price-factor universe {len(p)} (mom legs {n_wml}, vol legs {n_lmh})")

        rows.append(pd.DataFrame({
            "MKT": mkt, "SMB": smb, "HML": hml, "WML": wml, "LMH": lmh,
            "n_size_value": len(u), "n_price_factor": len(p),
        }))

    fac = pd.concat(rows).sort_index()
    fac["RF"] = rf_series(fac.index)
    fac["MKT_RF"] = fac["MKT"] - fac["RF"]
    return fac


# ══════════════════════════════════════════════════════════════════════════
# Kordent's own returns — calendar-time cohorts
# ══════════════════════════════════════════════════════════════════════════
def cohort_series(panel, rets, cal, lo, hi, label):
    """Calendar-time portfolio: every snapshot forms an equal-weighted cohort of
    investable names scoring in [lo, hi], entered one trading day later and held
    HOLD_DAYS. The daily series is the mean across live cohorts — the same
    overlapping-cohort design as backtest_runner, which is why the regression can
    use it."""
    daily = {}
    counts = {}
    for f, g in panel.groupby("snap_date"):
        sel = g[(g["investable"]) & (g["score"] >= lo) & (g["score"] <= hi)]
        names = [t for t in sel["ticker"] if t in rets.columns]
        after = cal[cal > pd.Timestamp(f)]
        if len(names) < 5 or not len(after):
            continue
        t0 = after[0]
        hold = cal[(cal >= t0)][:HOLD_DAYS]
        r = rets.loc[hold, names].mean(axis=1, skipna=True)
        for d, v in r.items():
            daily.setdefault(d, []).append(v)
            counts.setdefault(d, []).append(len(names))
    s = pd.Series({d: np.nanmean(v) for d, v in daily.items()}).sort_index()
    n_coh = pd.Series({d: len(v) for d, v in daily.items()}).sort_index()
    n_nm = pd.Series({d: float(np.mean(v)) for d, v in counts.items()}).sort_index()
    log(f"[KORDENT] {label}: {len(s)} days, "
        f"{n_coh.max()} cohorts at peak, {n_nm.mean():.0f} names per cohort")
    return s, n_coh, n_nm


def build_kordent(panel, rets, cal):
    out = {}
    for tag, sub in (("v8", panel[panel["schema_version"] >= 8]),
                     ("v1plus", panel[panel["schema_version"] >= 1])):
        hi_s, n_coh, n_nm = cohort_series(sub, rets, cal, 4, 5, f"{tag} high (score 4-5)")
        lo_s, _, _ = cohort_series(sub, rets, cal, 0, 1, f"{tag} low (score 0-1)")
        df = pd.DataFrame({"high": hi_s, "low": lo_s})
        df["spread"] = df["high"] - df["low"]
        df["cohorts"] = n_coh
        df["names_per_cohort"] = n_nm
        df["window"] = tag
        out[tag] = df
    return pd.concat(out.values()).sort_index()


# ══════════════════════════════════════════════════════════════════════════
# The IIM-A check — price-only factors, 2021-2025, on today's survivors
# ══════════════════════════════════════════════════════════════════════════
def build_hist_and_compare(panel, prices, rets, cal):
    if not os.path.exists(IIMA_FILE):
        log("[IIMA] file missing — comparison skipped")
        return None
    iima = pd.read_csv(IIMA_FILE)
    iima["Date"] = pd.to_datetime(iima["Date"])
    iima = iima.set_index("Date").apply(pd.to_numeric, errors="coerce") / 100.0

    latest = (panel.sort_values("snap_date").drop_duplicates("ticker", keep="last")
              .set_index("ticker"))
    mcap_now = latest["market_cap"].dropna()
    mcap_now = mcap_now[mcap_now.index.isin(prices.columns)]
    px_now = prices[mcap_now.index].ffill().iloc[-1]

    hist_cal = cal[(cal >= pd.Timestamp("2021-07-01")) & (cal <= pd.Timestamp("2025-12-31"))]
    forms = pd.Series(hist_cal).groupby([hist_cal.year, hist_cal.month]).first()
    rows = []
    for i, f in enumerate(forms):
        t0 = cal[cal > f][0]
        t1 = forms.iloc[i + 1] if i + 1 < len(forms) else hist_cal[-1]
        window = rets.loc[(rets.index >= t0) & (rets.index <= t1)]
        if window.empty:
            continue
        ok = quality_mask(rets, f, MIN_OBS_PRICE_FACTOR)
        u = [t for t in mcap_now.index if ok.get(t, False)]
        if len(u) < 100:
            continue
        # Share counts are not in the archive, so size is approximated by
        # scaling today's market cap by the price ratio. Splits and issuance
        # break this; it is a validation aid, never an input to a result.
        px_then = prices.loc[:f, u].ffill().iloc[-1]
        mcap_then = (mcap_now[u] * px_then / px_now[u]).dropna()
        mom = momentum(prices[u], f, cal)
        vol = rets[u].loc[:f].tail(VOL_WINDOW).std()
        rows.append(pd.DataFrame({
            "MKT": vw_return(window, mcap_then),
            "SMB": smb_simple(mcap_then, window, mcap_then),
            "WML": two_by_three(mcap_then, mom, window, mcap_then)[0],
            "LMH": two_by_three(mcap_then, -vol, window, mcap_then)[0],
        }))
    if not rows:
        log("[IIMA] no historical window could be built")
        return None
    hist = pd.concat(rows).sort_index()

    log("\n[IIMA] correlation of our price-only factors with IIM-A, daily, "
        "2021-07 to 2025-12 (today's survivors only — survivorship-biased):")
    pairs = [("MKT", "MF"), ("SMB", "SMB"), ("WML", "WML")]
    for ours, theirs in pairs:
        j = pd.concat([hist[ours], iima[theirs]], axis=1, join="inner").dropna()
        if len(j) > 60:
            log(f"  {ours:4s} vs IIM-A {theirs:4s}: corr {j.corr().iloc[0, 1]:+.3f} "
                f"on {len(j)} days | ours {j.iloc[:, 0].std()*np.sqrt(252):.1%} vol, "
                f"theirs {j.iloc[:, 1].std()*np.sqrt(252):.1%} vol")
        else:
            log(f"  {ours:4s} vs IIM-A {theirs:4s}: too few overlapping days ({len(j)})")
    log("  LMH has no IIM-A counterpart — the library publishes no volatility factor.")
    log("  SMB here sorts on size alone (no book-to-market), so a weaker "
        "correlation than MKT/WML is expected, not a defect.")
    return hist


def main():
    log(f"attribution_factors.py — built {date.today()}")
    panel, prices = load()
    if os.path.exists(SOURCES_FILE):
        src = pd.read_csv(SOURCES_FILE)
        log("[DATA] price sources: " + ", ".join(
            f"{k}={v}" for k, v in src["source"].value_counts().items()))
    rets = prices.pct_change(fill_method=None)
    cal = prices.index[prices.get("NIFTYBEES.NS", prices.iloc[:, 0]).notna()]
    log(f"[DATA] panel {panel.shape}, prices {prices.shape}, "
        f"calendar {len(cal)} days to {cal[-1].date()}")

    fac = build_factors(panel, prices, rets, cal)
    fac.to_csv(os.path.join(OUT, "factors_daily.csv"))
    log(f"\n[FACTORS] wrote factors_daily.csv: {len(fac)} days "
        f"({fac.index.min().date()} -> {fac.index.max().date()})")
    log(fac[["MKT_RF", "SMB", "HML", "WML", "LMH"]]
        .describe().loc[["count", "mean", "std"]].to_string())

    kor = build_kordent(panel, rets, cal)
    kor.to_csv(os.path.join(OUT, "kordent_daily.csv"))
    v8 = kor[kor["window"] == "v8"]
    log(f"\n[KORDENT] wrote kordent_daily.csv | v8 window: {len(v8)} days "
        f"({v8.index.min().date()} -> {v8.index.max().date()})")

    hist = build_hist_and_compare(panel, prices, rets, cal)
    if hist is not None:
        hist.to_csv(os.path.join(OUT, "factors_hist.csv"))

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(_log))
    print(f"\nWrote {LOG_FILE}. Paste it back.")


if __name__ == "__main__":
    main()
