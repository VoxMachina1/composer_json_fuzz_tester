import os
import time
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
    safe_ticker = "SPY"
    url = f"https://api.tiingo.com/tiingo/daily/{safe_ticker}/prices"

    # Check the last 10 days to guarantee we catch the latest trading day
    start_check = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')

    for key in api_keys:
        headers = {'Content-Type': 'application/json', 'Authorization': f'Token {key}'}
        params = {'startDate': start_check, 'format': 'json', 'resampleFreq': 'daily'}

        response = requests.get(url, headers=headers, params=params, timeout=30)
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
    # Tiingo encodes share-class tickers with a hyphen (e.g. BRK/B -> BRK-B).
    # The sanitized symbol must go in the URL; the raw ticker lets the '/' split
    # the URL path so Tiingo returns 404.
    safe_ticker = ticker.replace("/", "-").replace(".", "-")
    url = f"https://api.tiingo.com/tiingo/daily/{safe_ticker}/prices"
    success = False
    data = None
    last_status = None

    for key in api_keys:
        headers = {'Content-Type': 'application/json', 'Authorization': f'Token {key}'}
        # Using 1900-01-01 to ensure we get the absolute maximum history available
        params = {'startDate': '1900-01-01', 'format': 'json', 'resampleFreq': 'daily'}

        print(f"[{ticker}] Downloading full history using key ending in ...{key[-4:]}")
        retries = 5
        backoff_s = 1.0
        response = None
        for _ in range(retries):
            response = requests.get(url, headers=headers, params=params, timeout=45)
            if response.status_code != 429:
                break
            print(f"[{ticker}] 429 rate limit hit. Retrying in {backoff_s:.1f}s...")
            time.sleep(backoff_s)
            backoff_s = min(backoff_s * 2, 16.0)

        last_status = response.status_code if response is not None else None

        if last_status == 200:
            data = response.json()
            success = True
            break

        # A 404 means Tiingo does not recognize the symbol itself. Rotating keys
        # cannot fix that, so fail fast with a clear message instead of burning
        # every key and reporting a misleading "keys exhausted" error.
        if last_status == 404:
            raise Exception(
                f"[{ticker}] Tiingo returned 404 for symbol '{safe_ticker}'. "
                f"The ticker is unsupported or misspelled; rotating keys will not help."
            )

        print(f"[{ticker}] Key failed. Status: {last_status}. Rotating...")
        continue

    if not success or not data:
        raise Exception(
            f"[{ticker}] Failed to download data after trying {len(api_keys)} key(s). "
            f"Last HTTP status: {last_status}."
        )

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

    download_count = 0
    for ticker in tickers:
        safe_ticker = ticker.replace('/', '-').replace('.', '-')
        file_path = data_dir / f"{safe_ticker}.csv"
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
            # Round-robin which key is tried first to spread load across all keys.
            offset = download_count % len(api_keys)
            rotated_keys = api_keys[offset:] + api_keys[:offset]
            download_ticker_data(ticker, rotated_keys, data_dir)
            download_count += 1
            # Free-tier Tiingo is rate-limited; small delay helps avoid 429s.
            time.sleep(0.4)

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
