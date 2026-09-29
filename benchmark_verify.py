"""
benchmark_verify.py — READ-ONLY, ~1 minute, laptop. Run BEFORE committing the
selector.py hunk that makes HDFCMID150.NS the default Midcap 150 ETF.

Criteria fixed 2026-09-30, before any candidate was measured:
  1. HISTORY    first price on or before 2024-06-30 (two years for statistics)
  2. COVERAGE   priced on >= 95% of NIFTYBEES.NS trading days since its start
  3. LIQUIDITY  median daily traded value >= Rs 1 crore (a SIP must fill at
                a fair price; the shadow must be a thing a user could buy)
  4. TRACKING   daily-return correlation with the NIFTYMIDCAP150.NS index
                >= 0.98 over the last year (it must actually be the index)
First candidate passing all four, in this order, becomes the default.

    py benchmark_verify.py > benchmark_verify.txt
"""
import time

import pandas as pd
import yfinance as yf

CANDIDATES = ["HDFCMID150.NS", "MID150CASE.NS", "GROWWMC150.NS"]
INDEX = "NIFTYMIDCAP150.NS"
REF = "NIFTYBEES.NS"
HISTORY_BY = pd.Timestamp("2024-06-30")


def get(t):
    h = yf.download(t, start="2021-01-01", progress=False, auto_adjust=True, threads=False)
    if h.empty:
        return None, None
    c, v = h["Close"], h["Volume"]
    if hasattr(c, "columns"):
        c, v = c.iloc[:, 0], v.iloc[:, 0]
    return c.dropna(), v


ref, _ = get(REF)
idx, _ = get(INDEX)
idx_r = idx.pct_change(fill_method=None).dropna() if idx is not None else None
print(f"reference {REF}: {len(ref)} days | index {INDEX}: "
      f"{len(idx) if idx is not None else 0} days\n")

chosen = None
for t in CANDIDATES:
    c, v = get(t)
    time.sleep(1)
    if c is None or c.empty:
        print(f"{t}: NO DATA")
        continue
    first = c.index.min()
    ref_since = ref[ref.index >= first]
    cover = c.index.normalize().isin(ref_since.index.normalize()).sum() / max(len(ref_since), 1)
    tv = (c * v.reindex(c.index)).dropna()
    med_tv_cr = float(tv.tail(250).median()) / 1e7 if len(tv) else 0.0
    r = c.pct_change(fill_method=None).dropna()
    j = pd.concat([r.rename("etf"), idx_r.rename("idx")], axis=1, join="inner").dropna().tail(250)
    corr = float(j.corr().iloc[0, 1]) if len(j) > 60 else float("nan")
    te = float((j["etf"] - j["idx"]).std() * 252 ** 0.5) if len(j) > 60 else float("nan")
    checks = {"history": first <= HISTORY_BY, "coverage": cover >= 0.95,
              "liquidity": med_tv_cr >= 1.0, "tracking": corr >= 0.98}
    ok = all(checks.values())
    print(f"{t}: first {first.date()} | coverage {cover:.1%} | median traded "
          f"Rs {med_tv_cr:,.2f} Cr/day | corr with index {corr:.3f} | "
          f"tracking error {te:.2%}/yr")
    print("    " + "  ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in checks.items())
          + f"  ->  {'PASSES' if ok else 'fails'}")
    if ok and chosen is None:
        chosen = t

print(f"\nDEFAULT MIDCAP 150 ETF: {chosen or 'NONE PASSED — keep the selector hunk out and report back'}")
