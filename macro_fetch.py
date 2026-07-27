"""macro_fetch.py — daily macro OBSERVATION layer.

BUILT, NOT WIRED. Nothing in app.py, selector.py or deep_metrics.py reads this
file or its output. It accumulates a dated series so that, for the first time,
the extraction noise on these scalars becomes MEASURABLE rather than arguable.
The decision about whether any of them should reach a recommendation is a
separate, still-open question and must not be answered by this script existing.

WHY IT EXISTS
The previous path called a web search per portfolio review, at review time.
Consequences, all removed here:
  1. Two users reviewing four hours apart could get different numbers, so
     `as_of` labelled when someone CLICKED, not the vintage of the fact.
  2. On CSE 429 the old code fell through to Gemini google_search grounding —
     a DIFFERENT retrieval stack. A diff between two snapshots could be partly
     a diff between two search engines, and nothing recorded which ran.
  3. Nothing was stored, so a real RBI revision and a bad parse were
     indistinguishable: you only ever saw two points.

NO FALLBACK IS DELIBERATE. On CSE failure this records the failure. Falling
back to a second retrieval system is the defect being removed, not resilience.

HARD BOUNDARY — india_10y_yield_pct is OBSERVE-ONLY and must not be wired to
INDIA_10Y_BOND_RATE while deep_metrics.py:496-503 stands. That is a Gordon
growth model, oe_ps*(1+g)/(r-g), with r = the bond rate and g = min(ni_cagr,
15%). At r = 7% every company with NI CAGR >= 7% yields None, and one at 6.9%
yields ~1000x owner earnings. A live rate there would flip companies between
"no value" and "absurd value" on a macro input. Fix the model first.
"""

import hashlib
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from statistics import median

import requests

try:
    from google import genai
except ImportError:
    print("FATAL: google-genai not installed", file=sys.stderr)
    sys.exit(1)


SERIES_PATH = "macro_series.json"
SCHEMA_VERSION = 1

# Extraction models, cheapest first. Same ladder as the app.
EXTRACT_MODELS = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"]

# The snippet assembler writes "- {title} ({source}): {snippet}" lines prefixed
# with U+2022. Requiring that marker is a STRUCTURAL whitelist: text that did
# not come from the snippet assembler cannot pass. The old guard blacklisted
# three exact sentinel strings, so "Search API returned 403" sailed through and
# was handed to the LLM as the evidence to extract a CPI projection from.
# Blacklists fail open. Whitelists fail closed.
SNIPPET_MARKER = "\u2022"

# One query per SCALAR. Specificity is the whole point: a query that names one
# number is far likelier to surface a snippet stating that number outright than
# a broad "latest tax and inflation" sweep whose top hits are explainers.
FIELDS = [
    {
        "name": "ltcg_pct",
        "query": "India long term capital gains tax rate listed equity shares percent",
        "prompt_key": "ltcg_pct (long-term capital gains rate on listed equity as a "
                      "percentage number, e.g. 12.5)",
        "lo": 0.0, "hi": 40.0, "kind": "float",
    },
    {
        "name": "stcg_pct",
        "query": "India short term capital gains tax rate listed equity shares percent",
        "prompt_key": "stcg_pct (short-term capital gains rate on listed equity as a "
                      "percentage number, e.g. 20.0)",
        "lo": 0.0, "hi": 40.0, "kind": "float",
    },
    {
        "name": "ltcg_holding_months",
        "query": "India listed equity long term capital gains holding period months",
        "prompt_key": "ltcg_holding_months (months a listed equity must be held to "
                      "qualify as long-term, e.g. 12)",
        "lo": 1.0, "hi": 60.0, "kind": "int",
    },
    {
        "name": "ltcg_exemption_inr",
        "query": "India LTCG annual exemption limit rupees listed equity",
        "prompt_key": "ltcg_exemption_inr (annual LTCG exemption in rupees, e.g. 125000)",
        "lo": 0.0, "hi": 10_000_000.0, "kind": "float",
    },
    {
        "name": "rbi_cpi_projection_pct",
        "query": "RBI monetary policy statement CPI inflation projection percent",
        "prompt_key": "rbi_cpi_projection_pct (RBI's FORWARD projected CPI inflation as "
                      "a percentage number, e.g. 4.5 - NOT a fraction like 0.045; prefer "
                      "the official forward projection over a spot or current print), "
                      "and target_fy (the fiscal year that projection targets, e.g. "
                      "'FY2027')",
        "lo": 0.0, "hi": 15.0, "kind": "float",
        "extra": "target_fy",
    },
    {
        "name": "india_10y_yield_pct",
        "query": "India 10 year government bond yield percent today",
        "prompt_key": "india_10y_yield_pct (the current India 10-year government bond "
                      "G-Sec yield as a percentage number, e.g. 6.8)",
        "lo": 0.0, "hi": 15.0, "kind": "float",
    },
]


# ── Retrieval ─────────────────────────────────────────────────────────────
def fetch_snippets(query: str, key: str, cx: str) -> tuple[str | None, str]:
    """Return (context_text, status). No fallback to any other retrieval stack.

    dateRestrict=d90 constrains the INDEX to recent documents. CSE ranks on
    relevance and authority, not recency, so without it a well-linked 2024
    article outranks a fresh thin one and the snippet set is temporally
    unordered. This does not fix truncation -- snippets are still ~160-char
    fragments -- but it removes the worst of the staleness.
    """
    try:
        resp = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "key": key,
                "cx": cx,
                "q": query,
                "num": 5,
                "dateRestrict": "d90",
            },
            timeout=20,
        )
    except Exception as exc:
        print(f"    retrieval exception: {type(exc).__name__}", file=sys.stderr)
        return None, "retrieval_failed"

    if resp.status_code != 200:
        # Log the BODY, not just the code. Google returns 403 for at least four
        # distinct causes — quota exhausted, key restricted by IP/referrer, API
        # not enabled on the project, key/cx mismatch — and only the body tells
        # them apart. Printing the code alone is a check that discards its own
        # evidence. The body carries no credentials.
        detail = (resp.text or "")[:400].replace("\n", " ")
        print(f"    CSE HTTP {resp.status_code}: {detail}", file=sys.stderr)
        return None, "retrieval_failed"

    items = resp.json().get("items", [])
    if not items:
        return None, "retrieval_failed"

    lines = []
    for item in items:
        lines.append(
            f"{SNIPPET_MARKER} {item.get('title', '')} "
            f"({item.get('displayLink', '')}): "
            f"{item.get('snippet', '').replace(chr(10), ' ')}"
        )
    return "\n".join(lines), "ok"


# ── Extraction ────────────────────────────────────────────────────────────
def extract(context_text: str, field: dict, client) -> tuple[dict, str]:
    """One scalar out of snippet prose. Returns (values, status)."""
    if not context_text or SNIPPET_MARKER not in context_text:
        # Structural whitelist. Not retrieved snippets -> not evidence.
        return {}, "extraction_failed"

    prompt = (
        "From the text below, extract the following as JSON. Use null if the "
        "value is not clearly and explicitly stated - do NOT infer, estimate, "
        "or carry over a value you happen to know.\n"
        f"Keys: {field['prompt_key']}.\n"
        "Output ONLY the JSON object, no prose, no markdown.\n\nTEXT:\n"
        + context_text
    )

    raw = None
    for model_name in EXTRACT_MODELS:
        try:
            resp = client.models.generate_content(model=model_name, contents=prompt)
            raw = (resp.text or "").strip()
            if raw:
                break
        except Exception:
            continue
    if not raw:
        return {}, "extraction_failed"

    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}, "extraction_failed"

    val = parsed.get(field["name"])
    try:
        val = float(val)
    except (TypeError, ValueError):
        return {}, "extraction_failed"

    # A model asked for a percentage sometimes returns a fraction.
    if field["name"].endswith("_pct") and 0 < val < 1:
        val = val * 100

    if not (field["lo"] <= val <= field["hi"]):
        return {"value": val}, "out_of_range"

    out = {"value": int(val) if field["kind"] == "int" else round(val, 3)}
    if field.get("extra"):
        extra_val = parsed.get(field["extra"])
        out[field["extra"]] = str(extra_val) if extra_val else None
    return out, "ok"


# ── Series I/O ────────────────────────────────────────────────────────────
def load_series() -> dict:
    if not os.path.exists(SERIES_PATH):
        return {"schema": SCHEMA_VERSION, "readings": []}
    with open(SERIES_PATH, encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("schema") != SCHEMA_VERSION:
        print(f"FATAL: {SERIES_PATH} schema {data.get('schema')} != "
              f"{SCHEMA_VERSION}", file=sys.stderr)
        sys.exit(1)
    return data


def operative_value(readings: list, field_name: str, window: int = 5,
                    max_span_days: int = 10):
    """Rolling median of the last `window` OK readings, within max_span_days.

    BUILT, NOT WIRED -- no production caller. Here so the 10-weekday
    falsification report can use the same definition the eventual consumer
    would, rather than a second copy that drifts.

    Median, not mean: the failure mode is one wild parse, not drift. RBI's
    projection is a step function with ~6 steps a year, so a 5-reading median
    lags a genuine step by 2-3 days and rejects everything else.

    Re-runs on the same date are collapsed to the LAST reading for that date,
    so a manual re-trigger does not double-weight a day.
    """
    by_date = {}
    for r in readings:
        if r.get("field") == field_name and r.get("status") == "ok" \
                and r.get("value") is not None:
            by_date[r["date"]] = r["value"]

    if not by_date:
        return None, "INSUFFICIENT"

    dates = sorted(by_date)[-window:]
    newest = datetime.strptime(dates[-1], "%Y-%m-%d").date()
    oldest = datetime.strptime(dates[0], "%Y-%m-%d").date()
    if (newest - oldest).days > max_span_days:
        dates = [d for d in dates
                 if (newest - datetime.strptime(d, "%Y-%m-%d").date()).days
                 <= max_span_days]

    if len(dates) < 3:
        return None, "INSUFFICIENT"
    return median(by_date[d] for d in dates), "ok"


# ── Main ──────────────────────────────────────────────────────────────────
def main() -> int:
    key = os.environ.get("GOOGLE_SEARCH_API_KEY")
    cx = os.environ.get("GOOGLE_CSE_ID")
    gem = os.environ.get("GEMINI_API_KEY")
    missing = [n for n, v in (("GOOGLE_SEARCH_API_KEY", key),
                              ("GOOGLE_CSE_ID", cx),
                              ("GEMINI_API_KEY", gem)) if not v]
    if missing:
        # A job that cannot read its evidence must FAIL, not report success on
        # zero readings.
        print(f"FATAL: missing secrets: {', '.join(missing)}", file=sys.stderr)
        return 1

    client = genai.Client(api_key=gem)
    today = datetime.now(timezone.utc).date().isoformat()
    series = load_series()

    new_rows = []
    for field in FIELDS:
        print(f"  {field['name']}", flush=True)
        context, status = fetch_snippets(field["query"], key, cx)
        source_hash = (hashlib.sha256(context.encode("utf-8")).hexdigest()[:16]
                       if context else None)

        row = {
            "date": today,
            "field": field["name"],
            "value": None,
            "status": status,
            "source_hash": source_hash,
            "retrieval": "cse",
        }

        if status == "ok":
            vals, status = extract(context, field, client)
            row["status"] = status
            row["value"] = vals.get("value")
            if field.get("extra"):
                row[field["extra"]] = vals.get(field["extra"])

        # An explicit null row, ALWAYS. A missing date must mean "the job did
        # not run", never "the job ran and found nothing" -- those have to stay
        # distinguishable or the null-rate falsifier is unmeasurable.
        new_rows.append(row)
        print(f"    {row['status']}: {row['value']}", flush=True)

    if not new_rows:
        print("FATAL: zero rows produced", file=sys.stderr)
        return 1

    if all(r["status"] == "retrieval_failed" for r in new_rows):
        # Every single retrieval failing is an infrastructure fact (expired
        # key, CSE disabled), not a data outcome. Write the rows so the record
        # is honest, then fail the build so it is noticed.
        series["readings"].extend(new_rows)
        with open(SERIES_PATH, "w", encoding="utf-8") as fh:
            json.dump(series, fh, indent=2, ensure_ascii=False)
        print("FATAL: all retrievals failed", file=sys.stderr)
        return 1

    series["readings"].extend(new_rows)
    with open(SERIES_PATH, "w", encoding="utf-8") as fh:
        json.dump(series, fh, indent=2, ensure_ascii=False)

    ok = sum(1 for r in new_rows if r["status"] == "ok")
    print(f"wrote {len(new_rows)} rows ({ok} ok), "
          f"{len(series['readings'])} total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
