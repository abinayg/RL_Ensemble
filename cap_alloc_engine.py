import os
import math
import argparse
import urllib.request
import json
import glob
import datetime as dt
from dotenv import load_dotenv
from SmartApi import SmartConnect
import pyotp

# Google Drive API Imports
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# 1. Load Secure Credentials
load_dotenv()
API_KEY = os.environ.get("ANGEL_API_KEY")
CLIENT_CODE = os.environ.get("ANGEL_CLIENT_ID")
PASSWORD = os.environ.get("ANGEL_PIN")          
TOTP_SECRET = os.environ.get("ANGEL_TOTP_SECRET") 

FEE_BUFFER = 0.005  # 0.5% buffer for AngelOne STT, DP charges, and slippage

# --- GOOGLE DRIVE FUNCTIONS ---
SCOPES = ['https://www.googleapis.com/auth/drive']
TRADING_BOT_FOLDER_ID = "13aEoVnHq9s5WRffwjdz71Ztw4KOCz6Fw"

def authenticate_drive():
    """Authenticates using an OAuth User Token from a GitHub Secret."""
    token_json_str = os.environ.get('DRIVE_TOKEN_JSON')
    if not token_json_str:
        raise ValueError("DRIVE_TOKEN_JSON environment variable is missing!")
    
    creds_dict = json.loads(token_json_str)
    creds = Credentials.from_authorized_user_info(creds_dict, SCOPES)
    return build('drive', 'v3', credentials=creds)

def get_or_create_folder(service, folder_name, parent_id=None):
    """Creates a folder. If parent_id is provided, creates it INSIDE that folder."""
    query = f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    
    results = service.files().list(
        q=query, fields="files(id, name)",
        supportsAllDrives=True, includeItemsFromAllDrives=True
    ).execute()
    
    items = results.get('files', [])
    if items:
        return items[0]['id']
    else:
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        if parent_id:
            file_metadata['parents'] = [parent_id]

        folder = service.files().create(
            body=file_metadata, fields='id', supportsAllDrives=True
        ).execute()
        return folder.get('id')

def upload_file(service, filename, folder_id, mime_type):
    """Uploads a file. Overwrites if it exists, creates if it does not."""
    name = os.path.basename(filename)
    query = f"name = '{name}' and '{folder_id}' in parents and trashed = false"
    
    results = service.files().list(
        q=query, fields="files(id)",
        supportsAllDrives=True, includeItemsFromAllDrives=True
    ).execute()
    
    files = results.get('files', [])
    media = MediaFileUpload(filename, mimetype=mime_type)

    if files:
        file_id = files[0]['id']
        service.files().update(
            fileId=file_id, media_body=media, supportsAllDrives=True
        ).execute()
        print(f"☁️ Overwrote existing file on Drive: {name}")
    else:
        file_metadata = {'name': name, 'parents': [folder_id]}
        service.files().create(
            body=file_metadata, media_body=media, fields='id', supportsAllDrives=True
        ).execute()
        print(f"☁️ Uploaded new file to Drive: {name}")

# --- TRADING BOT FUNCTIONS ---

def auto_detect_signal_file():
    """Scans the directory for the latest ensemble results file."""
    files = glob.glob("ensemble_results_*.txt")
    if not files:
        raise FileNotFoundError("❌ Could not find any file matching 'ensemble_results_*.txt' in the current directory.")
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def fetch_nse_tokens():
    """Downloads AngelOne's daily Master Scrip list to map Ticker names to Symbol Tokens."""
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
    report_lines = []
    
    def log(msg):
        """Helper to print to console AND save to the text file array"""
        print(msg)
        report_lines.append(msg)

    log("="*45)
    log("🚀 STARTING CAPITAL ALLOCATION SCRIPT")
    log("="*45)
    
    if not all([API_KEY, CLIENT_CODE, PASSWORD, TOTP_SECRET]):
        raise ValueError("Missing credentials! Check your .env file or GitHub Secrets.")

    bot_signals = parse_signals(signal_file)
    if not bot_signals:
        log("❌ No valid signals found. Exiting.")
        return
        
    log(f"📡 Loaded {len(bot_signals)} total signals from {signal_file}")

    if target_tickers:
        target_tickers = [t.upper() for t in target_tickers]
        bot_signals = {ticker: sig for ticker, sig in bot_signals.items() if ticker in target_tickers}
        log(f"🎯 Applied Subset Filter. Processing ONLY {len(bot_signals)} targeted tickers: {', '.join(bot_signals.keys())}")
        
        if not bot_signals:
            log("❌ None of the target tickers were found in the ensemble file. Exiting.")
            return

    TOTAL_BUCKETS = len(bot_signals)
    available_cash = 0.0
    current_holdings = []
    smartApi = None
    
    log("\n🔐 Authenticating with Angel One API...")
    smartApi = SmartConnect(api_key=API_KEY)
    try:
        totp = pyotp.TOTP(TOTP_SECRET).now()
        login_data = smartApi.generateSession(CLIENT_CODE, PASSWORD, totp)
        if not login_data['status']:
            log("❌ Login Failed: " + login_data['message'])
            return
        log("✅ Authenticated Successfully.")
    except Exception as e:
        log("❌ Authentication Error: " + str(e))
        return

    if live_trade == 1:
        log("\n🟢 MODE: LIVE TRADE (Using Actual Brokerage Account Data)")
        try:
            rms_data = smartApi.rmsLimit()
            if rms_data['status']:
                available_cash = float(rms_data['data'].get('availablecash', 0.0))
            else:
                log("❌ Failed to fetch balance.")
                return
            
            holdings_response = smartApi.holding()
            if holdings_response['status'] and holdings_response['data']:
                current_holdings = [item['tradingsymbol'].split('-')[0] for item in holdings_response['data']]
        except Exception as e:
            log("❌ API Execution Error fetching portfolio: " + str(e))
            return
    else:
        log("\n🟡 MODE: SIMULATION / OFFLINE (Using manual cash, pulling LIVE prices)")
        if principal_amount <= 0:
            log("❌ ERROR: For live_trade=0, principal_amount must be greater than 0.")
            return
        available_cash = principal_amount
        current_holdings = [] 
        log(f"ℹ️ Using manual principal amount: ₹ {available_cash:,.2f}")

    log("\n📥 Downloading Angel One NSE Master Token List...")
    nse_tokens = fetch_nse_tokens()
    if not nse_tokens:
        log("❌ Cannot proceed without Token List. Exiting.")
        return

    bucket_limit = available_cash / TOTAL_BUCKETS
    investable_per_bucket = bucket_limit * (1 - FEE_BUFFER)
    
    log("\n" + "="*45)
    log("📈 DAILY CAPITAL ALLOCATION REPORT")
    log("="*45)
    log(f"Total Settled Cash   : ₹ {available_cash:,.2f}")
    log(f"Total Target Tickers : {TOTAL_BUCKETS}")
    log(f"Max Bucket Limit     : ₹ {bucket_limit:,.2f} per ticker")
    log(f"Investable (w/ fees) : ₹ {investable_per_bucket:,.2f} per ticker")
    log("-" * 45)

    for ticker, signal in bot_signals.items():
        if signal == "BUY":
            if ticker in current_holdings:
                log(f"⚠️ {ticker:<12} : IGNORED (Already held - No Double Dip)")
                continue
            
            token = nse_tokens.get(ticker)
            if not token:
                log(f"❌ {ticker:<12} : ERROR - Token not found in NSE Master List.")
                continue
                
            try:
                ltp_resp = smartApi.ltpData("NSE", f"{ticker}-EQ", token)
                if ltp_resp['status'] and ltp_resp['data']:
                    live_price = float(ltp_resp['data']['ltp'])
                else:
                    log(f"❌ {ticker:<12} : ERROR - Failed to fetch live price from API.")
                    continue
            except Exception as e:
                log(f"❌ {ticker:<12} : ERROR fetching price API - {str(e)}")
                continue

            quantity = math.floor(investable_per_bucket / live_price)
            
            if quantity > 0:
                capital_used = quantity * live_price
                log(f"✅ BUY  {ticker:<10} : {quantity} shares @ ₹{live_price:,.2f} (Cost: ₹{capital_used:,.2f})")
            else:
                log(f"❌ BUY  {ticker:<10} : INSUFFICIENT FUNDS (Price ₹{live_price:,.2f} > Bucket ₹{investable_per_bucket:,.2f})")
        
        elif signal == "SELL":
            if live_trade == 1 and ticker not in current_holdings:
                log(f"ℹ️ SELL {ticker:<10} : Signal was SELL, but no active position held.")
            else:
                log(f"🔴 SELL {ticker:<10} : Liquidate active position at market open.")
        
        elif signal == "HOLD":
             log(f"⏸️ HOLD {ticker:<10} : Hold active position / Idle cash.")

    log("="*45)

    if smartApi:
        smartApi.terminateSession(CLIENT_CODE)

    # --- SAVE TEXT FILE AND UPLOAD TO GOOGLE DRIVE ---
    filename = "cap_alloc.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print("\n----- Uploading Allocation Report to Google Drive -----")
    try:
        service = authenticate_drive()
        today_str = str(dt.date.today())
        sub_folder_id = get_or_create_folder(service, today_str, parent_id=TRADING_BOT_FOLDER_ID)
        upload_file(service, filename, sub_folder_id, "text/plain")
        print("✅ cap_alloc.txt successfully uploaded to Drive!")
    except Exception as e:
        print(f"❌ Failed to upload to Google Drive: {e}")

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

    actual_signal_file = args.signal_file
    if not actual_signal_file:
        actual_signal_file = auto_detect_signal_file()

    calculate_allocations(
        live_trade=args.live_trade, 
        principal_amount=args.principal_amount, 
        signal_file=actual_signal_file,
        target_tickers=args.target_tickers
    )
