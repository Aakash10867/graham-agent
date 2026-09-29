"""
benchmark_probe.py — READ-ONLY, ~1 minute, runs on the laptop (.NS works there).

Found 2026-09-30 by attribution_tests.py: MID150BEES.NS has prices on 5 of the
last 43 trading days (NIFTYBEES.NS: 42/43). It is selector.choose_benchmark's
default for a typical Kordent portfolio, so every consumer of its daily
history is running on almost nothing. Which Midcap 150 / Smallcap 250 series
does Yahoo actually keep complete?

    py benchmark_probe.py > benchmark_probe.txt
"""
import time

import yfinance as yf

REFERENCE = "NIFTYBEES.NS"
CANDIDATES = ["MID150BEES.NS", "SMALLCAP.NS", "NIFTYBEES.NS", "^CRSLDX", "^NSEI"]
SEARCHES = ["Nifty Midcap 150", "Midcap 150 ETF", "Nifty Smallcap 250", "Smallcap 250 ETF"]

for q in SEARCHES:
    try:
        for r in yf.Search(q, max_results=10).quotes:
            s = r.get("symbol", "")
            if s.endswith(".NS") or s.startswith("^"):
                if s not in CANDIDATES:
                    CANDIDATES.append(s)
                print(f"  search '{q}': {s:20s} {r.get('quoteType',''):8s} {r.get('shortname','')}")
    except Exception as e:
        print(f"  search '{q}' failed: {e}")
    time.sleep(1)

ref = yf.download(REFERENCE, period="6mo", progress=False, auto_adjust=True, threads=False)
ref_days = set(ref.index.normalize())
print(f"\nreference {REFERENCE}: {len(ref_days)} trading days in 6 months\n")
print(f"{'symbol':20s} {'days':>5s} {'of ref':>7s} {'last 43':>8s}  last date   first date")
for s in CANDIDATES:
    try:
        h = yf.download(s, period="6mo", progress=False, auto_adjust=True, threads=False)
        c = h["Close"]
        if hasattr(c, "columns"):
            c = c.iloc[:, 0]
        c = c.dropna()
        days = set(c.index.normalize())
        recent = sorted(ref_days)[-43:]
        print(f"{s:20s} {len(days):5d} {len(days & ref_days)/max(len(ref_days),1):7.0%} "
              f"{sum(d in days for d in recent):5d}/43  "
              f"{c.index.max().date() if len(c) else '-'}  {c.index.min().date() if len(c) else '-'}")
    except Exception as e:
        print(f"{s:20s} FAILED {type(e).__name__}: {str(e)[:60]}")
    time.sleep(1)
print("\nDone.")
