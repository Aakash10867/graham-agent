"""
stats.py — descriptive statistics about the SCORER, not its performance.

Half A of the Sprint 12 statistics layer. Everything here is computable today,
with zero forward returns, because it describes the shape of the universe as the
scorer sees it right now:

  - flag base rates          how often each of the 5 frameworks passes
  - phi orthogonality        whether the flags measure different things
  - score distribution       how the universe spreads across 0..5
  - trajectory cliff         the pile of stocks at exactly 7 (gate is >= 8)

HONESTY BOUNDARY: none of this says a flag PREDICTS anything. A base rate is not
a win rate. Orthogonality means "independent", not "useful". Those claims need
forward returns, which do not exist yet. Everything here is labelled as
descriptive so the "Does It Work?" page cannot be misread as a performance claim.

Single source of truth for the flag set: FLAGS. If a sixth framework is added,
add it here and every stat updates.
"""

import math
import pandas as pd

FLAGS = ["graham_pass", "greenblatt_pass", "dorsey_pass",
         "trajectory_pass", "lynch_pass"]

# Display names, kept out of the app so the page and the LLM agree on labels.
FLAG_LABELS = {
    "graham_pass": "Graham",
    "greenblatt_pass": "Greenblatt",
    "dorsey_pass": "Dorsey+Buffett",
    "trajectory_pass": "Trajectory",
    "lynch_pass": "Lynch",
}

TRAJECTORY_GATE = 8   # deep_metrics: trajectory_pass = trajectory_score >= 8


def _as_bool(series):
    """CSV round-trips booleans as strings ('True'/'False') or 0/1. Coerce."""
    if series.dtype == bool:
        return series.fillna(False)
    return (series.astype(str).str.strip().str.lower()
            .isin(["true", "1", "1.0", "yes"]))


def _phi(a, b):
    """Phi coefficient (Pearson correlation between two booleans).

    phi = (n11*n00 - n10*n01) / sqrt(row/col marginals). Range -1..1.
    Returns None if either variable is constant (marginal is zero -> undefined).
    """
    n11 = int((a & b).sum())
    n10 = int((a & ~b).sum())
    n01 = int((~a & b).sum())
    n00 = int((~a & ~b).sum())
    num = n11 * n00 - n10 * n01
    den = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    if den == 0:
        return None
    return num / den


def compute_universe_stats(universe_df):
    """Descriptive stats over the full scored universe.

    Parameters
    ----------
    universe_df : DataFrame with the 5 *_pass flags, `score`, `quality_pass`,
                  `trajectory_score`. Missing columns degrade gracefully — the
                  stat that needs them is omitted, never faked.

    Returns a dict ready for the "Does It Work?" page and LLM context.
    """
    df = universe_df
    n = len(df)
    out = {"n_universe": n, "descriptive_only": True}
    if n == 0:
        return out

    present = [f for f in FLAGS if f in df.columns]
    bools = {f: _as_bool(df[f]) for f in present}

    # ── 1. Base rates ────────────────────────────────────────────────────
    base = {}
    for f in present:
        base[f] = {
            "label": FLAG_LABELS.get(f, f),
            "pass_rate": round(float(bools[f].mean()), 4),
            "pass_count": int(bools[f].sum()),
        }
    out["base_rates"] = dict(sorted(base.items(),
                                    key=lambda kv: kv[1]["pass_rate"]))
    if base:
        rates = [v["pass_rate"] for v in base.values() if v["pass_rate"] > 0]
        if rates:
            out["base_rate_spread"] = {
                "min": round(min(rates), 4),
                "max": round(max(rates), 4),
                "ratio": round(max(rates) / min(rates), 1),
                "rarest": min(base, key=lambda k: base[k]["pass_rate"]),
                "commonest": max(base, key=lambda k: base[k]["pass_rate"]),
            }

    # ── 2. Phi orthogonality ─────────────────────────────────────────────
    # rare AND independent is the signature of a signal the others miss.
    phi = {}
    max_abs = {}
    for i, fi in enumerate(present):
        for fj in present[i + 1:]:
            p = _phi(bools[fi], bools[fj])
            phi[f"{fi}|{fj}"] = None if p is None else round(p, 3)
    for f in present:
        others = []
        for f2 in present:
            if f2 == f:
                continue
            key = f"{f}|{f2}" if f"{f}|{f2}" in phi else f"{f2}|{f}"
            v = phi.get(key)
            if v is not None:
                others.append(abs(v))
        max_abs[f] = round(max(others), 3) if others else None
    out["phi_matrix"] = phi
    out["phi_max_abs"] = dict(sorted(
        max_abs.items(),
        key=lambda kv: (kv[1] is None, kv[1])))
    # the most orthogonal flag: lowest max|phi| among those defined
    _defined = {k: v for k, v in max_abs.items() if v is not None}
    if _defined:
        out["most_orthogonal"] = min(_defined, key=_defined.get)

    # The real story lives in the PAIRS, not per-flag maxima. If three flags
    # travel together (high |phi|), a 3/5 that passes all three is a weaker
    # signal than a 3/5 spread across independent flags — they are ~1.5 votes,
    # not 3. Surface the most- and least-entangled pairs by name so the page
    # leads with the finding instead of burying it in the matrix.
    _pairs = [(k, v) for k, v in phi.items() if v is not None]
    if _pairs:
        _by_abs = sorted(_pairs, key=lambda kv: abs(kv[1]))
        def _pair_dict(key, val):
            fi, fj = key.split("|")
            return {
                "pair": key,
                "labels": [FLAG_LABELS.get(fi, fi), FLAG_LABELS.get(fj, fj)],
                "phi": val,
            }
        out["least_orthogonal_pair"] = _pair_dict(*_by_abs[-1])   # most entangled
        out["most_orthogonal_pair"] = _pair_dict(*_by_abs[0])     # most independent
        # the entangled cluster: every pair above a redundancy-worth-noting bar
        CLUSTER_PHI = 0.30
        out["correlated_cluster"] = [
            _pair_dict(k, v) for k, v in
            sorted(_pairs, key=lambda kv: abs(kv[1]), reverse=True)
            if abs(v) >= CLUSTER_PHI
        ]

    # ── 3. Score distribution ────────────────────────────────────────────
    if "score" in df.columns:
        sc = pd.to_numeric(df["score"], errors="coerce").dropna().astype(int)
        dist = {int(k): int(v) for k, v in sc.value_counts().sort_index().items()}
        # ensure 0..5 all present
        out["score_distribution"] = {s: dist.get(s, 0) for s in range(6)}
        out["score_pct"] = {s: round(dist.get(s, 0) / n, 4) for s in range(6)}

    if "quality_pass" in df.columns:
        qp = _as_bool(df["quality_pass"])
        out["quality_gate"] = {
            "pass": int(qp.sum()),
            "fail": int((~qp).sum()),
            "pass_rate": round(float(qp.mean()), 4),
        }

    # ── 4. Trajectory cliff ──────────────────────────────────────────────
    # The gate is a hard >= 8. If a large mass sits at exactly 7, the boundary
    # is a cliff, not a slope — a design fact worth showing, not hiding.
    if "trajectory_score" in df.columns:
        ts = pd.to_numeric(df["trajectory_score"], errors="coerce").dropna().astype(int)
        hist = {int(k): int(v) for k, v in ts.value_counts().sort_index().items()}
        out["trajectory_hist"] = {s: hist.get(s, 0) for s in range(11)}
        _at_boundary = hist.get(TRAJECTORY_GATE - 1, 0)
        _passing = int((ts >= TRAJECTORY_GATE).sum())
        out["trajectory_cliff"] = {
            "gate": TRAJECTORY_GATE,
            "at_boundary": _at_boundary,               # stuck at gate-1
            "boundary_score": TRAJECTORY_GATE - 1,
            "passing": _passing,
            # how big the just-missed pile is relative to those who passed
            "boundary_vs_passing_ratio": (round(_at_boundary / _passing, 2)
                                          if _passing else None),
        }

    return out
