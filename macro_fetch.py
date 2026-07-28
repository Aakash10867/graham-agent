"""macro_fetch.py — daily macro OBSERVATION layer.

BUILT, NOT WIRED. Nothing in app.py, selector.py or deep_metrics.py reads this
file or its output. It accumulates a dated series so that, for the first time,
the extraction noise on these scalars becomes MEASURABLE rather than arguable.
Whether any of them should reach a recommendation is a separate, still-open
question and must not be answered by this script existing.

WHY IT EXISTS
The previous path called a web search per portfolio review, at review time:
  1. Two users reviewing four hours apart could get different numbers, so
     `as_of` labelled when someone CLICKED, not the vintage of the fact.
  2. On quota exhaustion it fell through to Gemini google_search grounding — a
     DIFFERENT retrieval stack. A diff between two snapshots could be partly a
     diff between two search engines, and nothing recorded which ran.
  3. Nothing was stored, so a real RBI revision and a bad parse were
     indistinguishable: you only ever saw two points.

WHY TAVILY AND NOT GOOGLE CUSTOM SEARCH  (decided 2026-07-28)
Google CSE returns ~160-character SERP snippets ranked by relevance, not
recency. Two consequences no amount of scheduling fixes: fragments truncate
mid-sentence so a number arrives without its scope ("...projection at 4.0% for
FY2027, revising Q3 to..."), and the top-5 set is temporally unordered so the
extractor cannot tell a current projection from a 2024 one.

Tavily addresses both directly. search_depth="advanced" returns parsed page
content rather than a fragment, and include_domains lets the CPI query be
restricted to rbi.org.in — which is much of what "replace with a sourced
series" wanted, without a scraper.

Separately and decisively: Google will not serve Custom Search JSON API on a
project with no billing account. Confirmed across two projects, four key
configurations, and a 13-hour propagation wait; the console reports the API
Enabled while the API returns PERMISSION_DENIED at project level. Do not
re-litigate this without a billing account attached.

COST. 6 fields x ~22 weekdays x 2 credits (advanced) = ~264 credits/month
against a 1,000/month free tier. Basic search would be ~132 but returns
fragments, which is the thing being fixed.

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
from datetime import datetime, timezone
from statistics import median

import requests

try:
    from google import genai
except ImportError:
    print("FATAL: google-genai not installed", file=sys.stderr)
    sys.exit(1)


SERIES_PATH = "macro_series.json"
SCHEMA_VERSION = 1

TAVILY_URL = "https://api.tavily.com/search"

# Extraction models, cheapest first. Same ladder as the app.
EXTRACT_MODELS = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"]

# This script assembles the context text itself, one marker-prefixed line per
# result. Requiring that marker downstream is a STRUCTURAL whitelist: text that
# did not come from the assembler cannot pass. The old app-side guard
# blacklisted three exact sentinel strings, so "Search API returned 403" sailed
# through and was handed to the LLM as the evidence to extract a CPI projection
# from. Blacklists fail open. Whitelists fail closed.
SNIPPET_MARKER = "\u2022"

# One query per SCALAR. Specificity is the point: a query naming one number is
# far likelier to surface a source stating that number than a broad sweep whose
# top hits are explainers.
#
# time_range is set ONLY where recency genuinely matters. Tax rates are stable
# legislative facts and the authoritative page may be old but still current, so
# restricting them would exclude the best source. The bond yield moves daily.
FIELDS = [
    {
        "name": "ltcg_pct",
        "query": "India long term capital gains tax rate listed equity shares percent",
        "prompt_key": "ltcg_pct (long-term capital gains rate on listed equity as a "
                      "percentage number, e.g. 12.5)",
        "lo": 0.0, "hi": 40.0, "kind": "float",
        "topic": "general",
    },
    {
        "name": "stcg_pct",
        "query": "India short term capital gains tax rate listed equity shares percent",
        "prompt_key": "stcg_pct (short-term capital gains rate on listed equity as a "
                      "percentage number, e.g. 20.0)",
        "lo": 0.0, "hi": 40.0, "kind": "float",
        "topic": "general",
    },
    {
        "name": "ltcg_holding_months",
        "query": "India listed equity long term capital gains holding period months",
        "prompt_key": "ltcg_holding_months (months a listed equity must be held to "
                      "qualify as long-term, e.g. 12)",
        "lo": 1.0, "hi": 60.0, "kind": "int",
        "topic": "general",
    },
    {
        "name": "ltcg_exemption_inr",
        "query": "India LTCG annual exemption limit rupees listed equity",
        "prompt_key": "ltcg_exemption_inr (annual LTCG exemption in rupees, e.g. 125000)",
        "lo": 0.0, "hi": 10_000_000.0, "kind": "float",
        "topic": "general",
    },
    {
        "name": "rbi_cpi_projection_pct",
        "query": "RBI monetary policy statement CPI inflation projection",
        "prompt_key": "rbi_cpi_projection_pct (RBI's FORWARD projected CPI inflation as "
                      "a percentage number, e.g. 4.5 - NOT a fraction like 0.045; prefer "
                      "the official forward projection over a spot or current print), "
                      "and target_fy (the fiscal year that projection targets, e.g. "
                      "'FY2027')",
        "lo": 0.0, "hi": 15.0, "kind": "float",
        "extra": "target_fy",
        # Three filters at once returned zero results on 2026-07-28. Keeping
        # the domain restriction (it is the sourced-series property worth
        # having) and dropping topic/time_range, which fought it: MPC
        # statements are not indexed as recent news.
        "topic": "general",
        "include_domains": ["rbi.org.in"],
    },
    {
        "name": "india_10y_yield_pct",
        "query": "India 10 year government bond G-Sec yield",
        "prompt_key": "india_10y_yield_pct (the current India 10-year government bond "
                      "G-Sec yield as a percentage number, e.g. 6.8)",
        "lo": 0.0, "hi": 15.0, "kind": "float",
        "topic": "finance",
        "time_range": "week",
    },
]


# ── Retrieval ─────────────────────────────────────────────────────────────
def fetch_snippets(field: dict, key: str) -> tuple[str | None, str]:
    """Return (context_text, status). No fallback to any other retrieval stack.

    Falling back to a second provider is the defect being removed, not
    resilience: it made a diff between two snapshots partly a diff between two
    search engines, undetectably. On failure this records the failure.
    """
    payload = {
        "query": field["query"],
        # Parsed page content, not a 160-char fragment. 2 credits, not 1.
        "search_depth": "advanced",
        "max_results": 5,
        "topic": field.get("topic", "general"),
        "include_answer": False,
        "include_raw_content": False,
    }
    if field.get("time_range"):
        payload["time_range"] = field["time_range"]
    if field.get("include_domains"):
        payload["include_domains"] = field["include_domains"]

    try:
        resp = requests.post(
            TAVILY_URL,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=45,
        )
    except Exception as exc:
        print(f"    retrieval exception: {type(exc).__name__}", file=sys.stderr)
        return None, "retrieval_failed"

    if resp.status_code != 200:
        # Log the BODY, not just the code. A status code alone is a check that
        # discards its own evidence -- that cost several rounds of misdiagnosis
        # against the previous provider. The body carries no credentials.
        detail = (resp.text or "")[:400].replace("\n", " ")
        print(f"    Tavily HTTP {resp.status_code}: {detail}", file=sys.stderr)
        return None, "retrieval_failed"

    results = resp.json().get("results", [])
    if not results:
        # 200 with zero results is a FILTER outcome, not an API failure. Name
        # the filters so the next reader does not have to guess which one bit.
        print(f"    0 results (topic={payload['topic']} "
              f"time_range={payload.get('time_range', '-')} "
              f"domains={payload.get('include_domains', '-')})", file=sys.stderr)
        return None, "retrieval_failed"

    lines = []
    for item in results:
        content = (item.get("content") or "").replace("\n", " ").strip()
        if not content:
            continue
        lines.append(
            f"{SNIPPET_MARKER} {item.get('title', '')} "
            f"({item.get('url', '')}): {content}"
        )
    if not lines:
        return None, "retrieval_failed"
    return "\n".join(lines), "ok"


# ── Extraction ────────────────────────────────────────────────────────────
def extract(context_text: str, field: dict, client) -> tuple[dict, str]:
    """One scalar out of retrieved prose. Returns (values, status)."""
    if not context_text or SNIPPET_MARKER not in context_text:
        # Structural whitelist. Not assembled results -> not evidence.
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
    falsification report uses the same definition the eventual consumer would,
    rather than a second copy that drifts.

    Median, not mean: the failure mode is one wild parse, not drift. RBI's
    projection is a step function with ~6 steps a year, so a 5-reading median
    lags a genuine step by 2-3 days and rejects everything else.

    Re-runs on the same date collapse to the LAST reading for that date, so a
    manual re-trigger does not double-weight a day.
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
    tav = os.environ.get("TAVILY_API_KEY")
    gem = os.environ.get("GEMINI_API_KEY")
    missing = [n for n, v in (("TAVILY_API_KEY", tav),
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
        context, status = fetch_snippets(field, tav)
        source_hash = (hashlib.sha256(context.encode("utf-8")).hexdigest()[:16]
                       if context else None)

        row = {
            "date": today,
            "field": field["name"],
            "value": None,
            "status": status,
            "source_hash": source_hash,
            "retrieval": "tavily",
        }

        if status == "ok":
            vals, status = extract(context, field, client)
            row["status"] = status
            row["value"] = vals.get("value")
            if field.get("extra"):
                row[field["extra"]] = vals.get(field["extra"])
            if status != "ok":
                # Same rule as the HTTP body: a failure that discards its own
                # evidence cannot be diagnosed. Retrieved snippets are public
                # web text and carry no credentials.
                print(f"    context sample: {context[:300]}", file=sys.stderr)

        # An explicit null row, ALWAYS. A missing date must mean "the job did
        # not run", never "the job ran and found nothing" -- those have to stay
        # distinguishable or the null-rate falsifier is unmeasurable.
        new_rows.append(row)
        print(f"    {row['status']}: {row['value']}", flush=True)

    if not new_rows:
        print("FATAL: zero rows produced", file=sys.stderr)
        return 1

    if all(r["status"] == "retrieval_failed" for r in new_rows):
        # Every retrieval failing is an infrastructure fact (bad key, credits
        # exhausted, outage), not a data outcome. Write the rows so the record
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
