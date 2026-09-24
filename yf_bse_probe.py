"""
yf_bse_probe.py — why does Yahoo serve BSE FUNDAMENTALS but not BSE PRICES?

READ-ONLY, ~2 minutes, no writes outside stdout. Run on Actions (the laptop
cannot reach .BO at all, so a laptop result proves nothing).

Evidence being explained (2026-09-21/22/23):
  * attribution_data.py: 645 of 646 price failures are .BO, on two separate runs.
  * update-universe 2026-09-22: BSE FUNDAMENTALS mostly fine (no-data ~2.5%),
    while the price path logs "No data found, symbol may be delisted" for .BO,
    plus two `HTTP Error 401 ... Invalid Crumb`.

Hypotheses, one test each:
  H1  The chart (price) service has dropped NUMERIC BSE symbols. Then the same
      company under a NAME-based symbol still has history, and the fix is a
      symbol mapping in universe_updater's BSE branch.
  H2  Session/crumb authentication. Then a fresh Ticker object, or period=
      instead of start/end, behaves differently from yf.download.
  H3  The history is simply gone for these listings (genuinely delisted or
      dropped by Yahoo). Then nothing works and we stop trying.
"""
import time
import traceback

import pandas as pd
import yfinance as yf

# Failures seen on Actions, plus two dual-listed BSE codes as controls.
FAILING = ["511066.BO", "509525.BO", "512018.BO", "511700.BO"]
CONTROL_BSE = ["500325.BO", "532540.BO"]       # Reliance, TCS — numeric too
CONTROL_NSE = ["RELIANCE.NS"]

print("yfinance", yf.__version__)
try:
    import curl_cffi
    print("curl_cffi", curl_cffi.__version__)
except Exception:
    pass


def show(label, fn):
    try:
        r = fn()
        if isinstance(r, pd.DataFrame):
            print(f"    {label}: rows={len(r)}"
                  + (f" {r.index.min().date()}..{r.index.max().date()}" if len(r) else ""))
        else:
            print(f"    {label}: {r}")
    except Exception as e:
        print(f"    {label}: EXCEPTION {type(e).__name__}: {str(e)[:120]}")
    time.sleep(1)


for t in FAILING + CONTROL_BSE + CONTROL_NSE:
    print(f"\n=== {t}")
    tk = yf.Ticker(t)

    # H2 — does the price path work at all, in any of its forms?
    show("download(start=2021)",
         lambda: yf.download(t, start="2021-01-01", progress=False,
                             auto_adjust=True, threads=False))
    show("download(period=1y)",
         lambda: yf.download(t, period="1y", progress=False,
                             auto_adjust=True, threads=False))
    show("Ticker.history(period=1mo)", lambda: tk.history(period="1mo"))

    # Does the FUNDAMENTALS path work for the same symbol? The universe run says
    # it should. If info works while every price call fails, the split is in the
    # chart service, not in authentication or in the symbol being unknown.
    def info_bits():
        i = tk.info or {}
        return {k: i.get(k) for k in
                ("symbol", "shortName", "exchange", "quoteType",
                 "regularMarketPrice", "marketCap")}
    show("info", info_bits)

    def fast_bits():
        f = tk.fast_info
        return {"last_price": f.get("last_price"), "exchange": f.get("exchange"),
                "currency": f.get("currency")}
    show("fast_info", fast_bits)

    # H1 — what symbol does Yahoo itself offer for this company? If it returns a
    # name-based .BO symbol, that is the mapping universe_updater needs.
    def search_syms():
        i = {}
        try:
            i = tk.info or {}
        except Exception:
            pass
        q = i.get("shortName") or t.split(".")[0]
        res = yf.Search(q, max_results=6).quotes
        return [(r.get("symbol"), r.get("exchange"), r.get("shortname")) for r in res]
    show("search by name", search_syms)


# If a name-based symbol exists, does IT have history? One explicit check, so
# the answer is not inferred from the search result alone.
print("\n=== name-based symbol check")
for q in ["Ashika Credit", "Nagreeka Capital"]:
    try:
        res = yf.Search(q, max_results=5).quotes
        cands = [r["symbol"] for r in res if str(r.get("symbol", "")).endswith(".BO")]
        print(f"  {q}: candidates {cands}")
        for c in cands[:2]:
            h = yf.download(c, period="1y", progress=False, auto_adjust=True, threads=False)
            print(f"    {c}: rows={len(h)}")
            time.sleep(1)
    except Exception as e:
        print(f"  {q}: EXCEPTION {type(e).__name__}: {str(e)[:120]}")
        traceback.print_exc(limit=1)

print("\nDone.")
