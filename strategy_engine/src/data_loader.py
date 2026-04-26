import os
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

try:
    from config_loader import load_config
except ImportError:
    from .config_loader import load_config

def get_latest_tiingo_date(api_keys):
    """
    Fetches the most recent trading date available on Tiingo using SPY.
    This ensures we sync perfectly with the provider's update schedule.
    """
    safe_ticker = ticker.replace("/", "-")
    url = f"https://api.tiingo.com/tiingo/daily/{safe_ticker}/prices"
    
    # Check the last 10 days to guarantee we catch the latest trading day
    start_check = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
    
    for key in api_keys:
        headers = {'Content-Type': 'application/json', 'Authorization': f'Token {key}'}
        params = {'startDate': start_check, 'format': 'json', 'resampleFreq': 'daily'}
        
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            if data:
                # Get the date of the very last entry
                latest_date = data[-1]['date'][:10] # Extract 'YYYY-MM-DD'
                return latest_date
    
    raise Exception("Failed to fetch the latest market date from Tiingo.")

def download_ticker_data(ticker, api_keys, data_dir):
    """
    Downloads the FULL historical daily data for a ticker, rotating API keys on failure.
    """
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
    success = False
    data = None
    
    for key in api_keys:
        headers = {'Content-Type': 'application/json', 'Authorization': f'Token {key}'}
        # Using 1900-01-01 to ensure we get the absolute maximum history available
        params = {'startDate': '1900-01-01', 'format': 'json', 'resampleFreq': 'daily'}
        
        print(f"[{ticker}] Downloading full history using key ending in ...{key[-4:]}")
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            success = True
            break
        else:
            print(f"[{ticker}] Key failed. Status: {response.status_code}. Rotating...")
            continue
            
    if not success or not data:
        raise Exception(f"[{ticker}] Failed to download data. API keys exhausted.")

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df = df[['date', 'adjClose']].rename(columns={'adjClose': 'close'})
    
    os.makedirs(data_dir, exist_ok=True)
    file_path = data_dir / f"{safe_ticker}.csv"
    df.to_csv(file_path, index=False)
    print(f"[{ticker}] Successfully saved {len(df)} rows to {file_path}")
    return True

def check_freshness_and_update(tickers, api_keys, data_dir):
    """
    Checks each ticker against the latest market date.
    Rebuilds the entire history if the CSV is missing or outdated.
    """
    print("Checking dataset freshness...")
    latest_market_date = get_latest_tiingo_date(api_keys)
    print(f"Latest US trading day on Tiingo: {latest_market_date}")
    
    for ticker in tickers:
        file_path = data_dir / f"{ticker.replace('/', '-')}.csv"
        needs_rebuild = True
        
        if file_path.exists():
            # Read CSV to find the max date
            df = pd.read_csv(file_path)
            if not df.empty:
                latest_csv_date = df['date'].max()
                if latest_csv_date >= latest_market_date:
                    needs_rebuild = False
                    print(f"[{ticker}] Data is up to date (Latest: {latest_csv_date}). Skipping.")
                else:
                    print(f"[{ticker}] Data outdated (CSV: {latest_csv_date} < Market: {latest_market_date}).")
        else:
            print(f"[{ticker}] CSV not found.")
            
        if needs_rebuild:
            download_ticker_data(ticker, api_keys, data_dir)

# --- TEST ---
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    data_directory = base_dir / "data"
    
    try:
        cfg, keys = load_config()
        
        # Test freshness check logic
        test_tickers =["SPY", "QQQ"]
        
        print("--- RUN 1: Should trigger downloads ---")
        check_freshness_and_update(test_tickers, keys, data_directory)
        
        print("\n--- RUN 2: Should skip downloads ---")
        check_freshness_and_update(test_tickers, keys, data_directory)
        
        print("\nPASS")
    except Exception as e:
        print(f"FAIL: {e}")