import yfinance as yf
import datetime
import os
from supabase import create_client

# --- Configuration ---
SUPABASE_URL = os.environ.get("SUPABASE_URL", "YOUR_SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "YOUR_SUPABASE_KEY")

# 👉 PASTE YOUR SUPABASE USER UUID HERE 👈
TARGET_USER_ID = "0b7e7531-6831-4569-92fa-1d28bec4f5c3"

sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# 10 Thematic Baskets mapping to the core frameworks
DECORATIVE_PORTFOLIOS = [
    # --- Graham Framework ---
    {
        "name": "Graham Deep Value",
        "investor_type": "defensive",
        "time_horizon": "long",
        "tickers": ["ITC.NS", "NTPC.NS", "COALINDIA.NS", "ONGC.NS", "POWERGRID.NS"]
    },
    {
        "name": "Graham Asset Heavy",
        "investor_type": "defensive",
        "time_horizon": "long",
        "tickers": ["TATASTEEL.NS", "HINDALCO.NS", "M&M.NS", "BPCL.NS", "IOC.NS"]
    },
    
    # --- Greenblatt Framework ---
    {
        "name": "Greenblatt Capital Compounders",
        "investor_type": "enterprising",
        "time_horizon": "medium",
        "tickers": ["TATAMOTORS.NS", "TRENT.NS", "HAL.NS", "VBL.NS", "BEL.NS"]
    },
    {
        "name": "Greenblatt ROCE Champions",
        "investor_type": "enterprising",
        "time_horizon": "medium",
        "tickers": ["BAJFINANCE.NS", "NESTLEIND.NS", "BRITANNIA.NS", "TITAN.NS", "PIDILITIND.NS"]
    },

    # --- Dorsey Framework ---
    {
        "name": "Dorsey Wide Moat",
        "investor_type": "balanced",
        "time_horizon": "long",
        "tickers": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "L&T.NS", "ASIANPAINT.NS"]
    },
    {
        "name": "Dorsey Niche Dominators",
        "investor_type": "balanced",
        "time_horizon": "medium",
        "tickers": ["EICHERMOT.NS", "MARUTI.NS", "SUNPHARMA.NS", "DIVISLAB.NS", "BAJAJ-AUTO.NS"]
    },

    # --- Trajectory Framework ---
    {
        "name": "Trajectory Breakouts",
        "investor_type": "aggressive",
        "time_horizon": "short",
        "tickers": ["ZOMATO.NS", "DIXON.NS", "KPITTECH.NS", "POLYCAB.NS", "CGPOWER.NS"]
    },
    {
        "name": "Trajectory Alpha",
        "investor_type": "aggressive",
        "time_horizon": "short",
        "tickers": ["CHOLAFIN.NS", "TVSMOTOR.NS", "BSE.NS", "ANGELONE.NS", "APOLLOTYRE.NS"]
    },

    # --- Blended ---
    {
        "name": "High Conviction Blend",
        "investor_type": "balanced",
        "time_horizon": "long",
        "tickers": ["ICICIBANK.NS", "INFY.NS", "BHARTIARTL.NS", "HCLTECH.NS", "AXISBANK.NS"]
    },
    {
        "name": "Defensive Income Blend",
        "investor_type": "defensive",
        "time_horizon": "long",
        "tickers": ["HINDUNILVR.NS", "WIPRO.NS", "TECHM.NS", "CIPLA.NS", "DRREDDY.NS"]
    }
]

def calculate_4y_return(tickers):
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=4 * 365)
    
    total_return = 0
    valid_tickers = 0

    for ticker in tickers:
        try:
            hist = yf.Ticker(ticker).history(start=start_date, end=end_date)
            if not hist.empty and len(hist) > 10:
                start_price = float(hist["Close"].iloc[0])
                end_price = float(hist["Close"].iloc[-1])
                
                pct_return = ((end_price - start_price) / start_price) * 100
                total_return += pct_return
                valid_tickers += 1
                print(f"  [{ticker}] 4Y Return: {pct_return:+.2f}%")
        except Exception as e:
            print(f"  Failed to fetch {ticker}: {e}")

    if valid_tickers == 0:
        return 0
        
    return round(total_return / valid_tickers, 2)

def seed_database():
    print("Calculating 4-Year historical returns based on real yfinance data...\n")
    
    for port in DECORATIVE_PORTFOLIOS:
        print(f"--- Processing {port['name']} ---")
        real_return = calculate_4y_return(port["tickers"])
        print(f">> Aggregate Basket Return: {real_return:+.2f}%\n")
        
        # The payload now maps these directly to your account
        payload = {
            "user_id": TARGET_USER_ID,
            "name": port["name"],
            "investor_type": port["investor_type"],
            "time_horizon": port["time_horizon"],
            "sip_amount": 10000, 
            "review_freq": "90",
            "current_return_pct": real_return,
            "created_at": (datetime.date.today() - datetime.timedelta(days=4 * 365)).isoformat()
        }
        
        try:
            sb.table("portfolios").insert(payload).execute()
            print(f"Successfully injected: {port['name']} (Assigned to {TARGET_USER_ID})\n")
        except Exception as e:
            print(f"Database injection failed for {port['name']}: {e}\n")

if __name__ == "__main__":
    seed_database()
