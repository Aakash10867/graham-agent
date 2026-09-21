"""
inspect_attribution_inputs.py — Sprint 17, step 0. READ-ONLY.

Checks what data the attribution tests can stand on, before any of them is
written. Touches nothing: reads git history and the CSVs, writes no file.

Run from the Kordent repo folder:
    py inspect_attribution_inputs.py > inspect_out.txt
then paste inspect_out.txt back.
"""
import os
import subprocess
import sys
from io import StringIO

import pandas as pd

ARCHIVE = "universe_scored.csv"
KEY_COLS = ["ticker", "name", "score", "schema_version", "market_cap", "pb", "pe",
            "price", "sector", "industry", "quality_pass", "is_stale", "beta",
            "years_of_data", "years_listed", "data_as_of"]


def git(args):
    try:
        r = subprocess.run(["git"] + args, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        return r.stdout if r.returncode == 0 else None
    except Exception as e:
        print(f"  git failed: {e}")
        return None


def hr(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ── 1. Git archive ────────────────────────────────────────────────────────
hr("1. GIT ARCHIVE OF universe_scored.csv")
print("shallow clone:", (git(["rev-parse", "--is-shallow-repository"]) or "?").strip())
raw = git(["log", "--format=%H|%cI", "--", ARCHIVE]) or ""
commits = []
for line in raw.strip().splitlines():
    if "|" in line:
        sha, iso = line.split("|", 1)
        commits.append((sha.strip(), pd.Timestamp(iso.strip())))
print("commits touching the CSV:", len(commits))

# last commit per calendar date (same rule as backtest_runner.load_cohorts)
by_date = {}
for sha, ts in commits:                       # newest first
    d = ts.date()
    if d not in by_date:
        by_date[d] = (sha, ts)
print("distinct snapshot dates:", len(by_date))

rows = []
prev = None
for d in sorted(by_date):
    sha, ts = by_date[d]
    txt = git(["show", f"{sha}:{ARCHIVE}"])
    if not txt:
        rows.append({"date": d, "err": "unreadable"})
        continue
    try:
        df = pd.read_csv(StringIO(txt), low_memory=False)
    except Exception as e:
        rows.append({"date": d, "err": str(e)[:40]})
        continue
    sv = None
    if "schema_version" in df.columns:
        s = pd.to_numeric(df["schema_version"], errors="coerce").dropna()
        sv = int(s.mode().iloc[0]) if len(s) else None
    mc = pd.to_numeric(df.get("market_cap"), errors="coerce") if "market_cap" in df else None
    pb = pd.to_numeric(df.get("pb"), errors="coerce") if "pb" in df else None
    sc = df.set_index("ticker")["score"] if {"ticker", "score"} <= set(df.columns) else None
    n_changed = None
    if sc is not None and prev is not None:
        common = sc.index.intersection(prev.index)
        a = pd.to_numeric(sc.loc[common], errors="coerce")
        b = pd.to_numeric(prev.loc[common], errors="coerce")
        a = a[~a.index.duplicated()]
        b = b[~b.index.duplicated()]
        n_changed = int(((a != b) & a.notna() & b.notna()).sum())
    rows.append({
        "date": d, "utc_time": ts.strftime("%H:%M"), "schema": sv, "rows": len(df),
        "n_cols": df.shape[1],
        "mcap_ok": int(mc.notna().sum()) if mc is not None else None,
        "mcap>=100cr": int((mc >= 100e7).sum()) if mc is not None else None,
        "mcap>=500cr": int((mc >= 500e7).sum()) if mc is not None else None,
        "pb>0": int((pb > 0).sum()) if pb is not None else None,
        "score>=4": int((pd.to_numeric(df["score"], errors="coerce") >= 4).sum())
                    if "score" in df else None,
        "score_changes_vs_prev": n_changed,
    })
    prev = sc
arch = pd.DataFrame(rows)
with pd.option_context("display.width", 200, "display.max_rows", 200,
                       "display.max_columns", 20):
    print(arch.to_string(index=False))

# ── 2. Latest universe CSV ────────────────────────────────────────────────
hr("2. CURRENT universe_scored.csv")
if os.path.exists(ARCHIVE):
    u = pd.read_csv(ARCHIVE, low_memory=False)
    print("shape:", u.shape)
    print("ALL COLUMNS:\n" + ", ".join(u.columns))
    print("\nkey columns (dtype | non-null | example):")
    for c in KEY_COLS:
        if c in u.columns:
            nn = u[c].notna().sum()
            ex = u[c].dropna().iloc[0] if nn else None
            print(f"  {c:16s} {str(u[c].dtype):8s} {nn:6d}  {str(ex)[:40]}")
        else:
            print(f"  {c:16s} MISSING")
    print("\nticker suffixes:", u["ticker"].astype(str).str[-3:].value_counts().to_dict())
    if "score" in u:
        print("score distribution:", u["score"].value_counts().sort_index().to_dict())
else:
    print("NOT FOUND")

# ── 3. Price cache ────────────────────────────────────────────────────────
hr("3. backtest_price_cache.csv")
if os.path.exists("backtest_price_cache.csv"):
    p = pd.read_csv("backtest_price_cache.csv", parse_dates=["date"])
    print("columns:", list(p.columns), "| rows:", len(p))
    print("tickers:", p["ticker"].nunique(), "| dates:", p["date"].min().date(),
          "->", p["date"].max().date(), "| trading days:", p["date"].nunique())
    first = p.groupby("ticker")["date"].min()
    print("first-date spread per ticker:", first.min().date(), "..", first.max().date())
    print("benchmarks present:",
          {b: b in set(p["ticker"]) for b in ["NIFTYBEES.NS", "^NSEI", "^CRSLDX",
                                              "MID150BEES.NS", "^NSEMDCP50"]})
else:
    print("NOT FOUND")

# ── 4. Other files ────────────────────────────────────────────────────────
for f in ["backtest_results.csv", "universe_tickers.csv", "nse_tickers_lastgood.csv"]:
    hr(f"4. {f}")
    if os.path.exists(f):
        try:
            x = pd.read_csv(f, low_memory=False)
            print("shape:", x.shape)
            print("columns:", list(x.columns))
            print(x.head(3).to_string(index=False)[:600])
        except Exception as e:
            print("unreadable:", e)
    else:
        print("NOT FOUND")

# ── 5. Environment ────────────────────────────────────────────────────────
hr("5. ENVIRONMENT")
print("python", sys.version.split()[0], "| pandas", pd.__version__)
for mod in ["numpy", "yfinance", "statsmodels", "scipy"]:
    try:
        m = __import__(mod)
        print(f"{mod} {getattr(m, '__version__', '?')}")
    except Exception:
        print(f"{mod} NOT INSTALLED")
print("\nDone.")
