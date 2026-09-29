"""
check_bse_investable.py — READ-ONLY. Did the BSE symbol fix make BSE-only
listings buyable?

Before 2026-09-24 every BSE-only row had no sector (Yahoo never classified
numeric-code listings), and selector._tier1's no_sector cut removed all of
them. Name-based symbols may now carry a sector. If so, BSE-only names can
reach a portfolio — and app.py builds their Kite order as tradingsymbol
"531205", which Kite does not recognise.

Run:  py check_bse_investable.py
"""
import pandas as pd

import selector

u = pd.read_csv("universe_scored.csv", low_memory=False)
bo = u["ticker"].astype(str).str.endswith(".BO")
has_sector = u["sector"].notna() & (u["sector"].astype(str).str.strip() != "")
score = pd.to_numeric(u["score"], errors="coerce")

print(f"BSE-only rows: {bo.sum()} | with a sector: {(bo & has_sector).sum()}")
print(f"score >= 4: NSE {(~bo & (score >= 4)).sum()} | BSE-only {(bo & (score >= 4)).sum()}")

inv = selector._tier1(u.copy(), float("inf"), {})
inv_bo = inv["ticker"].astype(str).str.endswith(".BO")
print(f"\ninvestable (tier 1, affordability off): {len(inv)} | of which BSE-only: {inv_bo.sum()}")
picks = inv[pd.to_numeric(inv["score"], errors="coerce") >= 4]
pb = picks[picks["ticker"].astype(str).str.endswith(".BO")]
print(f"investable & score >= 4: {len(picks)} | of which BSE-only: {len(pb)}")
if len(pb):
    cols = [c for c in ("ticker", "yf_symbol", "name", "sector", "score", "market_cap") if c in pb.columns]
    print("\nBSE-only names that could now be picked:")
    print(pb[cols].sort_values("market_cap", ascending=False).head(30).to_string(index=False))

# ── Duplicate check (added 2026-09-29) ────────────────────────────────────
# The `name` column is Yahoo's full company name, identical for both listings
# of one company. A BSE-only row sharing a name with an NSE row is a dual
# listing the dedup let through.
nm = u["name"].astype(str).str.strip().str.lower()
ns_names = set(nm[~bo])
leak = u[bo & nm.isin(ns_names)]
print(f"\nDUPLICATES: BSE-only rows whose Yahoo name matches an NSE row: {len(leak)}")
if len(leak):
    cols = [c for c in ("ticker", "yf_symbol", "name", "score", "market_cap") if c in leak.columns]
    print(leak[cols].sort_values("market_cap", ascending=False).head(25).to_string(index=False))
