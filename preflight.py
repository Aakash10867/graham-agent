"""
preflight.py — wiring checks. Read-only: imports, runs, asserts, prints.

Supersedes preflight_rank.py (which covered only the W2 rank change). Delete
that file once this is in place.

`ast.parse` passed on three runtime errors in W2 and would have passed on the
harness bug this sprint found. These are the assertions it cannot make.

    py preflight.py --csv universe_scored.csv

Exit code 1 on any failure, so it can gate a workflow step.

COVERAGE
  A. abstention denominators in _tier3        (rank side, shipped)
  B. forced-exit ceiling                      (sell trigger, shipped)
  C. floor + partition helpers                (13 threshold sites, shipped)
  D. comparable score-drop guard              (delta site, shipped)
  E. cross-cutting invariants
"""

import argparse
import inspect
import json
import sys

import numpy as np
import pandas as pd

import selector

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def section(t):
    print(f"\n{t}\n" + "-" * len(t))


# ── A. rank-side abstention ───────────────────────────────────────────────
def a_rank(df):
    section("A. abstention denominators (_tier3)")
    ok = all(hasattr(selector, n) for n in ("_rank_population", "_framework_percentile"))
    check("helpers exist", ok)
    if not ok:
        return None
    src = inspect.getsource(selector._tier3)
    check("_tier3 calls _framework_percentile", "_framework_percentile" in src)
    check("_tier3 calls _rank_population", "_rank_population" in src)
    check("old na_option='bottom' gone from _tier3",
          'na_option="bottom"' not in src and "na_option='bottom'" not in src)

    r = {}
    t1 = selector._tier1(df.copy(), 5000, r)
    ranked = selector._tier3(t1, {"philosophy": "deep_value", "demand_tilt": {}})
    lynch_out = ranked["lynch_category"].astype(str).str.strip() == "unknown"
    grn_out = ranked.get("greenblatt_sector_excluded",
                         pd.Series(False, index=ranked.index)).fillna(False).astype(bool)
    nanfrac = ranked["greenblatt_frac"].isna() & ~grn_out
    check("lynch abstainers hold NO percentile",
          bool(ranked.loc[lynch_out, "_pct_lynch"].isna().all()), f"{int(lynch_out.sum())} rows")
    check("greenblatt excluded hold NO percentile",
          bool(ranked.loc[grn_out, "_pct_greenblatt"].isna().all()), f"{int(grn_out.sum())} rows")
    check("uncomputable greenblatt hold NO percentile",
          bool(ranked.loc[nanfrac, "_pct_greenblatt"].isna().all()), f"{int(nanfrac.sum())} rows")
    for f in selector.FRAMEWORKS:
        live = int(ranked[f"_pct_{f}"].notna().sum())
        check(f"{f}: population non-empty", live > 0, f"{live} rows")
    bad = sum(1 for _, row in ranked.iterrows()
              if not set(row["_ranked_on"]) <= set(row["_applicable"]))
    check("ranked_on subset of applicable (frame)", bad == 0, f"{bad} violations")
    return ranked


# ── B. forced-exit ceiling ────────────────────────────────────────────────
def b_exit(df):
    section("B. forced-exit ceiling")
    t = {n: selector.forced_exit_threshold(n) for n in (5, 4, 3, 2, 1)}
    check("thresholds are floor(n/5)", t == {5: 1, 4: 0, 3: 0, 2: 0, 1: 0}, str(t))
    # A ceiling must never be HARSHER than the 5-denominator fraction it encodes.
    worst = max((t[n] / n) for n in (5, 4, 3, 2, 1) if t[n])
    check("no denominator exits at a harsher fraction than 1/5", worst <= 0.2 + 1e-9,
          f"max effective exit fraction {worst:.0%}")
    tick = df["ticker"].iloc[0]
    check("forced_exit_applies runs on a real row",
          isinstance(selector.forced_exit_applies(df, tick, 0), bool))
    check("score 0 exits at every denominator",
          all(selector.forced_exit_threshold(n) >= 0 for n in (5, 4, 3)))


# ── C. floor + partition helpers ──────────────────────────────────────────
def c_thresholds(df):
    section("C. floor + partition helpers")
    n, s = selector.score_applicable(df)
    check("score == score_applicable (numerator premise)",
          int((df["score"] != s).sum()) == 0,
          f"{int((df['score'] != s).sum())} mismatches")
    check("n_applicable within 1..5", bool(((n >= 1) & (n <= 5)).all()),
          ", ".join(f"n={k}:{v}" for k, v in sorted(n.value_counts().items())))

    # A FLOOR normalisation may only ADD. If it removes, the numerator premise
    # is broken — these two checks are linked, not independent.
    for k in (2, 3, 4):
        m = selector.meets_score_mask(df, k)
        removed = int(((df["score"] >= k) & ~m).sum())
        check(f"floor k={k} never removes", removed == 0,
              f"added {int((m & ~(df['score'] >= k)).sum())}, removed {removed}")

    tiers = selector.score_tiers(df)
    check("tiers only 4/3/2/None",
          set(tiers.dropna().unique()) <= {2, 3, 4}, str(sorted(set(tiers.dropna().unique()))))
    perfect = (n == s) & (n > 0)
    check("every full-marks row reaches tier 4",
          bool((tiers[perfect] == 4).all()), f"{int(perfect.sum())} full-marks rows")
    # The defect the equality partition carried independently of abstention.
    five = df["score"] == 5
    check("score==5 rows are tiered (were invisible before)",
          bool(tiers[five].notna().all()), f"{int(five.sum())} rows")


# ── D. comparable score drop ──────────────────────────────────────────────
def d_drop(df):
    section("D. comparable score-drop guard")
    th = {n: selector.score_drop_threshold(n) for n in (5, 4, 3, 2)}
    check("drop thresholds are ceil(2n/5)", th == {5: 2, 4: 2, 3: 2, 2: 1}, str(th))

    row = df.iloc[0]
    full = list(selector.FRAMEWORKS)
    # Artifact case: a framework leaves, nothing else moves -> delta 0.
    tr = {"applicable": full, "passed": ["greenblatt", "dorsey_buffett", "trajectory", "lynch"]}
    c = selector.comparable_score_drop(tr, row, 4, 3)
    check("comparable_score_drop returns the full contract",
          {"comparable", "n_common", "entry_common", "current_common", "delta", "fires"}
          <= set(c), str(sorted(c)))
    # Legacy trace with no applicable/passed must FALL BACK, never suppress.
    leg = selector.comparable_score_drop({}, row, 4, 1)
    check("legacy trace falls back to raw and still fires",
          leg["comparable"] is False and leg["fires"] is True, str(leg))
    check("headline names its basis when comparable",
          "comparable to entry" in selector.score_drop_headline("X", c, 4, 3))
    check("headline falls back cleanly when not",
          "comparable to entry" not in selector.score_drop_headline("X", leg, 4, 1))


# ── E. cross-cutting ──────────────────────────────────────────────────────
def e_wiring(df):
    section("E. cross-cutting")
    pol = {"sip_amount": 5000, "min_acceptable_score": 3, "philosophy": "deep_value",
           "demand_tilt": {}, "allocation_policy": {}, "portfolio_sizing": {"ips_target": 15}}
    res = selector.select_portfolio(df.copy(), pol, None)
    hs = res.get("holdings", [])
    check("selection produces holdings", len(hs) > 0, str(len(hs)))
    if not hs:
        return
    check("every trace carries ranked_on", all("ranked_on" in h["_trace"] for h in hs))
    check("ranked_on subset of applicable (trace)",
          all(set(h["_trace"]["ranked_on"]) <= set(h["_trace"]["applicable"]) for h in hs))
    try:
        json.dumps([h["_trace"] for h in hs])
        check("trace is JSON-serialisable", True)
    except (TypeError, ValueError) as e:
        check("trace is JSON-serialisable", False, str(e))

    # No consumer should be gating on the raw integer any more.
    # A check that cannot read its evidence must FAIL, not pass. The first
    # version of this swallowed OSError and reported PASS on zero files read.
    import re
    leftover, read = 0, []
    for f in ("app.py", "portfolio_tracker.py"):
        try:
            leftover += len(re.findall(r"\[.score.\]\s*(>=|==)\s*[0-9]",
                                       open(f, encoding="utf-8").read()))
            read.append(f)
        except OSError:
            pass
    check("no raw score thresholds left in consumers",
          len(read) == 2 and leftover == 0,
          f"{leftover} found in {read or 'NO FILES READ — run from the repo root'}")

    div = [h["ticker"] for h in hs
           if set(h["_trace"]["ranked_on"]) != set(h["_trace"]["applicable"])]
    print(f"\n  note: {len(div)}/{len(hs)} holdings rank on fewer frameworks than apply"
          + (f": {', '.join(div)}" if div else ""))
    print("        (the gate still counts `applicable` — that divergence is C1's)")


# ── F. drift constants ────────────────────────────────────────────────────
def f_drift():
    section("F. drift constants (two denominators)")
    import re
    from pathlib import Path

    # The old shared constant must be gone everywhere. A check that cannot read
    # its evidence must FAIL, not pass — so a zero-file walk is a failure, not
    # a silent PASS on nothing.
    files = sorted(Path(".").glob("*.py"))
    hits = []
    # Split so this file's own source does not match the pattern it searches
    # for. Written whole, the check reported preflight.py as a live use of the
    # dead constant -- a grep that cannot tell "still in use" from "mentioned
    # by the checker" is a false positive generator.
    needle = "DRIFT_MATERIAL" + "_CONTINUOUS"
    for p in files:
        try:
            if needle in p.read_text(encoding="utf-8"):
                hits.append(p.name)
        except OSError:
            pass
    check("old shared constant is gone",
          len(files) > 0 and not hits,
          f"{len(files)} files read"
          + (f", still present in {hits}" if hits else "")
          + ("" if files else " — run from the repo root"))

    # Assert the RELATION, not the value 0.25. When 3b's per-framework
    # measurement moves the floor, this check keeps holding without an edit —
    # a check pinned to the literal would have to be edited in lockstep, which
    # is how the two constants drift apart again.
    check("total bound is derived from the framework floor",
          abs(selector.DRIFT_MATERIAL_TOTAL
              - len(selector.FRAMEWORKS) * selector.DRIFT_FLOOR_FRAMEWORK) < 1e-9,
          f"{selector.DRIFT_MATERIAL_TOTAL} vs "
          f"{len(selector.FRAMEWORKS)}x{selector.DRIFT_FLOOR_FRAMEWORK}")

    # Behavioural, not textual: the two sites must read DIFFERENT constants.
    base = {f: 0.5 for f in selector.FRAMEWORKS}
    mk = lambda fr, sc: {"fracs": fr, "score_continuous": sc}

    # (i) five sub-floor moves. Total 0.20 < 0.25 -> must NOT reach the alert,
    #     and crowns no mover. This was the genuine orphan: "your score moved,
    #     cause unattributable", with unattributed = 0.
    wob = selector._continuous_drift(
        mk(base, 2.5), mk({k: v + 0.04 for k, v in base.items()}, 2.7))
    check("five sub-floor moves stay below the total bound",
          abs(wob["delta"]) < selector.DRIFT_MATERIAL_TOTAL
          and wob["largest_move"] is None
          and abs(wob["unattributed"]) < 1e-9,
          f"delta={wob['delta']} largest={wob['largest_move']}")

    # (ii) one clearly material move -> fires AND names its mover.
    big = dict(base); big["graham"] = 0.5 + 0.30
    mv = selector._continuous_drift(mk(base, 2.5), mk(big, 2.8))
    check("a material single move fires and names its mover",
          abs(mv["delta"]) >= selector.DRIFT_MATERIAL_TOTAL
          and mv["largest_move"] == "graham",
          f"delta={mv['delta']} largest={mv['largest_move']}")

    # (iii) a framework becomes scoreable. largest_move is None here and that
    #     is CORRECT, not an orphan — the cause is named via `unattributed`,
    #     which app.py renders in the same sentence as the number. Asserting
    #     this pins the distinction so a later "fix" cannot collapse the two.
    ent = dict(base); ent[selector.FRAMEWORKS[0]] = None
    became = selector._continuous_drift(mk(ent, 2.0), mk(base, 2.5))
    check("became-scoreable reports unattributed, not a false mover",
          became["largest_move"] is None
          and abs(became["unattributed"]) >= selector.DRIFT_FLOOR_FRAMEWORK,
          f"unattributed={became['unattributed']} "
          f"unmeasured={became['unmeasured']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="universe_scored.csv")
    args = ap.parse_args()
    df = pd.read_csv(args.csv, low_memory=False)
    print(f"preflight — {len(df)} rows")
    a_rank(df)
    b_exit(df)
    c_thresholds(df)
    d_drop(df)
    e_wiring(df)
    f_drift()
    print("\n" + ("ALL CHECKS PASSED" if not FAILS
                  else f"{len(FAILS)} FAILED: " + "; ".join(FAILS)))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
