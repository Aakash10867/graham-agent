"""
economics.py — single source of truth for portfolio-level money.

MODEL (a), decided Sprint 14. Sale proceeds stay INSIDE the portfolio as cash.
"Invested" means external capital the user actually paid in from outside; a buy
funded by earlier sale proceeds contributes nothing to it. Return is measured on
external capital, never on gross turnover, and never on the surviving cost basis.

    external_capital = sum over buys of max(0, buy_amount - cash_before)
    cash             = external_capital + sum(sells) - sum(buys)     [>= 0 always]
    market_value     = sum(shares * live_price)          [caller supplies]
    total_assets     = market_value + cash
    total_pnl        = total_assets - external_capital
    return_pct       = total_pnl / external_capital * 100

Why this and not "invested = surviving cost basis": reducing the denominator on
sale deletes the cost basis and the outcome of the sold position together, so
realising a loss RAISES reported return. A return series that improves when you
lose money is not a return series.

realized_pnl uses weighted-average cost per ticker, which is exactly what
holdings.price_at_entry already represents (every top-up site re-averages it).

Identity guaranteed by construction: unrealized_pnl == total_pnl - realized_pnl
== market_value - cost_basis_of_surviving_shares.

This module does NO I/O. Callers pass transaction rows in. Rows are dicts with
keys: transaction_date, created_at, id, ticker, shares, price, amount_inr,
transaction_type.
"""

import datetime


def _sort_key(t):
    """Chronological replay order. transaction_date is the economic date;
    created_at breaks same-day ties by real insertion order, which is what
    makes 'sell then rebuy on the same day' fund itself from cash rather than
    drawing fresh external capital. id is the final deterministic tiebreak."""
    d = t.get("transaction_date") or "0001-01-01"
    c = t.get("created_at") or ""
    return (str(d)[:10], str(c), str(t.get("id") or ""))


def replay_ledger(txns):
    """Replay a portfolio's sip_transactions in chronological order.

    Returns dict with external_capital, cash, realized_pnl, total_buys,
    total_sells, and external_flows [(date, amount)] for XIRR.
    """
    cash = 0.0
    external = 0.0
    realized = 0.0
    total_buys = 0.0
    total_sells = 0.0
    external_flows = []
    lots = {}  # ticker -> [shares, avg_cost]

    for t in sorted(txns or [], key=_sort_key):
        ttype = str(t.get("transaction_type") or "buy").lower()
        try:
            amt = float(t.get("amount_inr") or 0.0)
            sh = float(t.get("shares") or 0.0)
            px = float(t.get("price") or 0.0)
        except (TypeError, ValueError):
            continue
        if amt <= 0:
            continue
        tk = t.get("ticker") or ""
        d = str(t.get("transaction_date") or "")[:10]

        if ttype == "buy":
            total_buys += amt
            shortfall = amt - cash
            if shortfall > 0:
                external += shortfall
                cash += shortfall
                external_flows.append((d, round(shortfall, 2)))
            cash -= amt
            lot = lots.setdefault(tk, [0.0, 0.0])
            new_sh = lot[0] + sh
            if new_sh > 0:
                lot[1] = ((lot[0] * lot[1]) + amt) / new_sh
            lot[0] = new_sh
        else:
            total_sells += amt
            cash += amt
            lot = lots.get(tk)
            if lot and lot[0] > 0:
                sold = min(sh, lot[0])
                realized += sold * (px - lot[1])
                lot[0] -= sold
            # A sell with no recorded lot (e.g. holding predates the ledger)
            # contributes cash but no realized P&L. Silently dropping it would
            # be worse; the cash is real either way.

    return {
        "external_capital": external,
        "cash": max(0.0, cash),
        "realized_pnl": realized,
        "total_buys": total_buys,
        "total_sells": total_sells,
        "external_flows": external_flows,
    }


def portfolio_economics(txns, market_value):
    """The one function every display site calls. market_value is the live
    market value of surviving holdings (sum of shares * current price).

    return_pct is None — not 0.0 — when there is no external capital to
    measure against. A wrong number is worse than no number.
    """
    led = replay_ledger(txns)
    ext = led["external_capital"]
    cash = led["cash"]
    mv = float(market_value or 0.0)
    total_assets = mv + cash
    total_pnl = total_assets - ext
    realized = led["realized_pnl"]
    return {
        "external_capital": round(ext, 2),
        "cash_balance": round(cash, 2),
        "market_value": round(mv, 2),
        "total_assets": round(total_assets, 2),
        "realized_pnl": round(realized, 2),
        "unrealized_pnl": round(total_pnl - realized, 2),
        "total_pnl": round(total_pnl, 2),
        "return_pct": round(total_pnl / ext * 100, 2) if ext > 0 else None,
        "external_flows": led["external_flows"],
        "has_ledger": bool(led["total_buys"] > 0),
    }


def xirr_flows(econ, as_of=None):
    """Convert portfolio_economics output into (dates, amounts) for pyxirr.

    Under model (a) the only true external flows are contributions (negative)
    and the terminal total_assets (positive). Buys and sells are internal
    transfers between cash and securities and must NOT appear, or a same-day
    sell-and-rebuy shows up as a spurious round trip.
    """
    dates, amounts = [], []
    for d, amt in econ.get("external_flows", []):
        try:
            dates.append(datetime.date.fromisoformat(d))
        except (ValueError, TypeError):
            continue
        amounts.append(-float(amt))
    if not dates:
        return None, None
    dates.append(as_of or datetime.date.today())
    amounts.append(float(econ.get("total_assets") or 0.0))
    return dates, amounts
