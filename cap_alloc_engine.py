import os
import math
import argparse
import urllib.request
import json
import glob
from dotenv import load_dotenv
from SmartApi import SmartConnect
import pyotp

# 1. Load Secure Credentials
load_dotenv()
API_KEY = os.environ.get("ANGEL_API_KEY")
CLIENT_CODE = os.environ.get("ANGEL_CLIENT_ID")
PASSWORD = os.environ.get("ANGEL_PIN")          
TOTP_SECRET = os.environ.get("ANGEL_TOTP_SECRET") 

FEE_BUFFER = 0.005  # 0.5% buffer for AngelOne STT, DP charges, and slippage

def auto_detect_signal_file():
    """Scans the directory for the latest ensemble results file."""
    files = glob.glob("ensemble_results_*.txt")
    
    if not files:
        raise FileNotFoundError("❌ Could not find any file matching 'ensemble_results_*.txt' in the current directory.")
    
    latest_file = max(files, key=os.path.getmtime)
    print(f"🔍 Auto-detected signal file: {latest_file}")
    
    return latest_file

def fetch_nse_tokens():
    """
    Downloads AngelOne's daily Master Scrip list to map Ticker names to Symbol Tokens.
    """
    print("📥 Downloading Angel One NSE Master Token List...")
    url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    token_map = {}
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
        
        for item in data:
            if item.get('exch_seg') == 'NSE' and item.get('symbol', '').endswith('-EQ'):
                base_symbol = item['symbol'].split('-')[0]
                token_map[base_symbol] = item['token']
                
        print(f"✅ Loaded {len(token_map)} NSE Equity Tokens.")
        return token_map
    except Exception as e:
        print(f"❌ Failed to fetch tokens: {e}")
        return {}

def parse_signals(filepath):
    """Reads the ensemble_results.txt file and parses the tickers and signals."""
    bot_signals = {}
    try:
        with open(filepath, 'r') as file:
            for line in file:
                if '->' in line:
                    parts = line.split('->')
                    ticker = parts[0].strip().replace('.NS', '')
                    signal = parts[1].strip().upper()
                    if ticker and signal:
                        bot_signals[ticker] = signal
        return bot_signals
    except FileNotFoundError:
        raise FileNotFoundError(f"Signal file not found at: {filepath}")

def calculate_allocations(live_trade, principal_amount, signal_file, target_tickers=None):
    print("="*45)
    print("🚀 STARTING CAPITAL ALLOCATION SCRIPT")
    print("="*45)
    
    # 1. Ensure credentials exist
    if not all([API_KEY, CLIENT_CODE, PASSWORD, TOTP_SECRET]):
        raise ValueError("Missing credentials! Check your .env file or GitHub Secrets.")

    # 2. Load and Filter Signals
    bot_signals = parse_signals(signal_file)
    if not bot_signals:
        print("❌ No valid signals found. Exiting.")
        return
        
    print(f"📡 Loaded {len(bot_signals)} total signals from {signal_file}")

    # Apply Ticker Subset Filter
    if target_tickers:
        target_tickers = [t.upper() for t in target_tickers]
        bot_signals = {ticker: sig for ticker, sig in bot_signals.items() if ticker in target_tickers}
        print(f"🎯 Applied Subset Filter. Processing ONLY {len(bot_signals)} targeted tickers: {', '.join(bot_signals.keys())}")
        
        if not bot_signals:
            print("❌ None of the target tickers were found in the ensemble file. Exiting.")
            return

    TOTAL_BUCKETS = len(bot_signals)
    
    available_cash = 0.0
    current_holdings = []
    smartApi = None
    
    # 3. GLOBAL API LOGIN
    print("\n🔐 Authenticating with Angel One API...")
    smartApi = SmartConnect(api_key=API_KEY)
    try:
        totp = pyotp.TOTP(TOTP_SECRET).now()
        login_data = smartApi.generateSession(CLIENT_CODE, PASSWORD, totp)
        if not login_data['status']:
            print("❌ Login Failed:", login_data['message'])
            return
        print("✅ Authenticated Successfully.")
    except Exception as e:
        print("❌ Authentication Error:", str(e))
        return

    # 4. Mode Execution: Portfolio & Cash Setup
    if live_trade == 1:
        print("\n🟢 MODE: LIVE TRADE (Using Actual Brokerage Account Data)")
        try:
            rms_data = smartApi.rmsLimit()
            if rms_data['status']:
                available_cash = float(rms_data['data'].get('availablecash', 0.0))
            else:
                print("❌ Failed to fetch balance.")
                return
            
            holdings_response = smartApi.holding()
            if holdings_response['status'] and holdings_response['data']:
                current_holdings = [item['tradingsymbol'].split('-')[0] for item in holdings_response['data']]
        except Exception as e:
            print("❌ API Execution Error fetching portfolio:", str(e))
            return
    else:
        print("\n🟡 MODE: SIMULATION / OFFLINE (Using manual cash, pulling LIVE prices)")
        if principal_amount <= 0:
            print("❌ ERROR: For live_trade=0, principal_amount must be greater than 0.")
            return
        available_cash = principal_amount
        current_holdings = [] 
        print(f"ℹ️ Using manual principal amount: ₹ {available_cash:,.2f}")

    # 5. Get Token Map for Live Pricing
    print("")
    nse_tokens = fetch_nse_tokens()
    if not nse_tokens:
        print("❌ Cannot proceed without Token List. Exiting.")
        return

    # 6. Calculate Capital Allocation
    bucket_limit = available_cash / TOTAL_BUCKETS
    investable_per_bucket = bucket_limit * (1 - FEE_BUFFER)
    
    print("\n" + "="*45)
    print("📈 DAILY CAPITAL ALLOCATION REPORT")
    print("="*45)
    print(f"Total Settled Cash   : ₹ {available_cash:,.2f}")
    print(f"Total Target Tickers : {TOTAL_BUCKETS}")
    print(f"Max Bucket Limit     : ₹ {bucket_limit:,.2f} per ticker")
    print(f"Investable (w/ fees) : ₹ {investable_per_bucket:,.2f} per ticker")
    print("-" * 45)

    # 7. Process Signals and Calculate Real Quantity (Q)
    for ticker, signal in bot_signals.items():
        if signal == "BUY":
            if ticker in current_holdings:
                print(f"⚠️ {ticker:<12} : IGNORED (Already held - No Double Dip)")
                continue
            
            token = nse_tokens.get(ticker)
            if not token:
                print(f"❌ {ticker:<12} : ERROR - Token not found in NSE Master List.")
                continue
                
            try:
                ltp_resp = smartApi.ltpData("NSE", f"{ticker}-EQ", token)
                if ltp_resp['status'] and ltp_resp['data']:
                    live_price = float(ltp_resp['data']['ltp'])
                else:
                    print(f"❌ {ticker:<12} : ERROR - Failed to fetch live price from API.")
                    continue
            except Exception as e:
                print(f"❌ {ticker:<12} : ERROR fetching price API - {str(e)}")
                continue

            quantity = math.floor(investable_per_bucket / live_price)
            
            if quantity > 0:
                capital_used = quantity * live_price
                print(f"✅ BUY  {ticker:<10} : {quantity} shares @ ₹{live_price:,.2f} (Cost: ₹{capital_used:,.2f})")
            else:
                print(f"❌ BUY  {ticker:<10} : INSUFFICIENT FUNDS (Price ₹{live_price:,.2f} > Bucket ₹{investable_per_bucket:,.2f})")
        
        elif signal == "SELL":
            if live_trade == 1 and ticker not in current_holdings:
                print(f"ℹ️ SELL {ticker:<10} : Signal was SELL, but no active position held.")
            else:
                print(f"🔴 SELL {ticker:<10} : Liquidate active position at market open.")
        
        elif signal == "HOLD":
             print(f"⏸️ HOLD {ticker:<10} : Hold active position / Idle cash.")

    print("="*45)

    if smartApi:
        smartApi.terminateSession(CLIENT_CODE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading Bot Capital Allocation Script")
    
    parser.add_argument("--signal_file", type=str, required=False, 
                        help="Optional: Path to the ensemble_results.txt file. Auto-detects if left blank.")
                        
    parser.add_argument("--live_trade", type=int, choices=[0, 1], required=True, 
                        help="0 = Simulation/Manual Cash, 1 = Live API execution")
                        
    parser.add_argument("--principal_amount", type=float, default=0.0, 
                        help="Required if live_trade=0. The manual cash amount to use.")
                        
    parser.add_argument("--target_tickers", type=str, nargs='*', default=None,
                        help="Optional: Space-separated list of tickers to strictly process.")
    
    args = parser.parse_args()
    
    if args.live_trade == 0 and args.principal_amount <= 0:
        parser.error("--principal_amount must be provided and > 0 when --live_trade is 0")

    # Auto-detect the file if one isn't explicitly provided
    actual_signal_file = args.signal_file
    if not actual_signal_file:
        actual_signal_file = auto_detect_signal_file()

    calculate_allocations(
        live_trade=args.live_trade, 
        principal_amount=args.principal_amount, 
        signal_file=actual_signal_file,
        target_tickers=args.target_tickers
    )
