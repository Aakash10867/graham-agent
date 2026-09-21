"""
yf_probe.py — is Yahoo blocking this machine, or are those tickers really absent?
READ-ONLY. Takes ~30 seconds.   py yf_probe.py > probe_out.txt
"""
import os
import time
import traceback

import pandas as pd
import yfinance as yf

print("yfinance", yf.__version__)
try:
    import curl_cffi
    print("curl_cffi", curl_cffi.__version__)
except Exception:
    print("curl_cffi NOT INSTALLED")

FAILED_SAMPLE = ["511066.BO", "509525.BO", "512018.BO"]
KNOWN_GOOD = ["RELIANCE.NS", "500325.BO", "NIFTYBEES.NS"]

# Did GitHub Actions manage to price the failing ones? The backtest cache was
# built there, so this says whether Yahoo serves these tickers at all.
if os.path.exists("backtest_price_cache.csv"):
    c = pd.read_csv("backtest_price_cache.csv", usecols=["ticker"])
    have = set(c["ticker"])
    for t in FAILED_SAMPLE:
        print(f"in backtest cache (priced by Actions): {t} -> {t in have}")
    bo = [t for t in have if str(t).endswith(".BO")]
    print(f".BO tickers in backtest cache: {len(bo)}")


def probe(label, fn):
    print(f"\n--- {label}")
    try:
        r = fn()
        if isinstance(r, pd.DataFrame):
            cl = r["Close"] if "Close" in r.columns.get_level_values(0) else r
            print("shape", r.shape, "| non-empty columns:",
                  [str(c) for c in cl.columns if cl[c].notna().any()]
                  if hasattr(cl, "columns") else "series")
        else:
            print(r)
    except Exception as e:
        print(f"EXCEPTION {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)
    time.sleep(2)


probe("A. one known-good ticker, last 5 days",
      lambda: yf.download(["RELIANCE.NS"], period="5d", progress=False,
                          auto_adjust=True, threads=False))
probe("B. known-good, full history from 2021",
      lambda: yf.download(KNOWN_GOOD, start="2021-01-01", progress=False,
                          auto_adjust=True, group_by="column", threads=False))
probe("C. the failing BSE codes, full history",
      lambda: yf.download(FAILED_SAMPLE, start="2021-01-01", progress=False,
                          auto_adjust=True, group_by="column", threads=False))
probe("D. one failing code via Ticker.history",
      lambda: yf.Ticker("511066.BO").history(period="1mo"))
probe("E. same failing code as NSE-style search",
      lambda: yf.Search("511066", max_results=3).quotes)
print("\nDone.")
