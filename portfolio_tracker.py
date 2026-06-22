import os
import yfinance as yf
from supabase import create_client, Client
from datetime import date
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- ALERT THRESHOLDS ---
PORTFOLIO_DROP_ALERT_PCT = -5.0

def send_alert_email(subject, body):
    sender_email = os.environ.get("ALERT_EMAIL_SENDER")
    sender_password = os.environ.get("ALERT_EMAIL_PASSWORD")
    recipient_email = os.environ.get("ALERT_EMAIL_RECIPIENT") # You can default to sender_email

    if not all([sender_email, sender_password, recipient_email]):
        print("Email credentials missing. Skipping alert.")
        return

    msg = MIMEMultipart()
    msg['From'] = f"DeepMoat Risk Engine <{sender_email}>"
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        print(f"📧 Alert email sent: {subject}")
    except Exception as e:
        print(f"Failed to send email: {e}")

def run_daily_tracker():
    print("Initiating DeepMoat Daily Portfolio Audit...")
    
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    
    if not url or not key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY environment variables.")
        
    supabase: Client = create_client(url, key)
    
    # Fetch active portfolios and holdings
    portfolios = supabase.table("portfolios").select("*").execute().data
    holdings = supabase.table("holdings").select("*").execute().data
    
    if not portfolios:
        print("No portfolios found. Exiting.")
        return

    price_cache = {}
    today_str = date.today().isoformat()

    for port in portfolios:
        port_id = port["id"]
        port_holdings = [h for h in holdings if h["portfolio_id"] == port_id]
        
        if not port_holdings:
            continue
            
        total_invested = 0.0
        current_total_value = 0.0
        red_flags = []
        
        for holding in port_holdings:
            ticker = holding["ticker"]
            shares = holding["shares"]
            invested_inr = holding["sip_amount_inr"]
            
            # Fetch live price
            if ticker not in price_cache:
                try:
                    info = yf.Ticker(ticker).fast_info
                    price_cache[ticker] = info.last_price
                except Exception:
                    price_cache[ticker] = holding["price_at_entry"]
            
            live_price = price_cache[ticker]
            total_invested += invested_inr
            current_total_value += (shares * live_price)
            
            # Check for sudden drops on individual holdings (>10%)
            holding_return = ((live_price - holding["price_at_entry"]) / holding["price_at_entry"]) * 100
            if holding_return < -10.0:
                red_flags.append(f"- {ticker} is down {holding_return:.2f}% from entry.")

        # Calculate portfolio-level return
        if total_invested > 0:
            return_pct = ((current_total_value - total_invested) / total_invested) * 100
        else:
            return_pct = 0.0
            
        # Update Leaderboard Data
        supabase.table("portfolios").update({
            "current_value": round(current_total_value, 2),
            "current_return_pct": round(return_pct, 2)
        }).eq("id", port_id).execute()
        
        # Log History
        supabase.table("portfolio_history").insert({
            "portfolio_id": port_id,
            "date": today_str,
            "total_value": round(current_total_value, 2),
            "daily_return_pct": round(return_pct, 2)
        }).execute()
        
        print(f"Updated [{port['name']}]: Value ₹{current_total_value:,.2f} | Return {return_pct:+.2f}%")
        
        # --- ALERT LOGIC ---
        if return_pct <= PORTFOLIO_DROP_ALERT_PCT:
            subject = f"⚠️ DeepMoat Alert: Portfolio '{port['name']}' Down {return_pct:.2f}%"
            body = (
                f"Your DeepMoat portfolio '{port['name']}' has triggered a risk alert.\n\n"
                f"Current Value: ₹{current_total_value:,.2f}\n"
                f"Total Return: {return_pct:.2f}%\n\n"
                f"Holding Warnings:\n" + "\n".join(red_flags) + "\n\n"
                f"Recommendation: Log into DeepMoat and run a review cycle on this portfolio using the Dorsey framework to check for moat erosion."
            )
            send_alert_email(subject, body)

    print("DeepMoat Daily Audit Complete.")

if __name__ == "__main__":
    run_daily_tracker()
