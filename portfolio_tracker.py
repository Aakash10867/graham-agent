import os
import yfinance as yf
from supabase import create_client, Client
from datetime import date

def run_daily_tracker():
    print("Initiating DeepMoat Daily Portfolio Audit...")
    
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    
    if not url or not key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY environment variables.")
        
    supabase: Client = create_client(url, key)
    
    # Fetch active portfolios and holdings
    portfolios_resp = supabase.table("portfolios").select("*").execute()
    portfolios = portfolios_resp.data
    
    if not portfolios:
        print("No portfolios found. Exiting.")
        return

    holdings_resp = supabase.table("holdings").select("*").execute()
    holdings = holdings_resp.data

    # ── Fetch Nifty 50 close price once for the entire run ──
    nifty_close = None
    try:
        nifty = yf.Ticker("^NSEI")
        hist = nifty.history(period="5d")
        if not hist.empty:
            nifty_close = round(float(hist["Close"].iloc[-1]), 2)
            print(f"Nifty 50 close: ₹{nifty_close:,.2f}")
    except Exception as e:
        print(f"Warning: Could not fetch Nifty 50: {e}")
    
    price_cache = {}
    today_str = date.today().isoformat()

    for port in portfolios:
        port_id = port["id"]
        port_holdings = [h for h in holdings if h["portfolio_id"] == port_id]
        
        if not port_holdings:
            continue
            
        total_invested = 0.0
        current_total_value = 0.0
        
        for holding in port_holdings:
            ticker = holding["ticker"]
            shares = holding["shares"]
            invested_inr = holding["sip_amount_inr"]
            
            # Fetch live price via yfinance (cached per run)
            if ticker not in price_cache:
                try:
                    info = yf.Ticker(ticker).fast_info
                    price_cache[ticker] = info.last_price
                except Exception as e:
                    print(f"Warning: Failed to fetch {ticker}: {e}")
                    price_cache[ticker] = holding["price_at_entry"]
            
            live_price = price_cache[ticker]
            
            total_invested += invested_inr
            current_total_value += (shares * live_price)
            
        # Calculate portfolio-level return
        if total_invested > 0:
            return_pct = ((current_total_value - total_invested) / total_invested) * 100
        else:
            return_pct = 0.0
            
        # 1. Update Leaderboard Data (Live Snapshot)
        supabase.table("portfolios").update({
            "current_value": round(current_total_value, 2),
            "current_return_pct": round(return_pct, 2)
        }).eq("id", port_id).execute()
        
        # 2. Log History (Time Series) — with Nifty for comparison chart
        history_row = {
            "portfolio_id": port_id,
            "date": today_str,
            "total_value": round(current_total_value, 2),
            "daily_return_pct": round(return_pct, 2),
        }
        if nifty_close is not None:
            history_row["nifty_value"] = nifty_close

        supabase.table("portfolio_history").upsert(
            history_row, on_conflict="portfolio_id,date"
        ).execute()
        
        print(f"Updated [{port['name']}]: Value ₹{current_total_value:,.2f} | Return {return_pct:+.2f}%")

    print("DeepMoat Daily Audit Complete.")

if __name__ == "__main__":
    run_daily_tracker()
