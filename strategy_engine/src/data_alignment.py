import os
import pandas as pd
from pathlib import Path

def load_ticker_csv(ticker, data_dir):
    """
    Reads the price CSV for a given ticker.
    Converts 'date' to a pandas datetime object and sorts chronologically.
    """
    file_path = data_dir / f"{ticker.replace('/', '-')}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Data file for {ticker} not found at {file_path}")
        
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df[['date', 'close']]

def build_master_dataframe(signal_ticker, target_ticker, benchmark_ticker, data_dir, filter_assets=None):
    """
    Loads signal, target, benchmark, and any global filter CSVs.
    Renames columns according to their roles or exact ticker names.
    Merges them into a single aligned DataFrame, dropping any missing dates.
    """
    if filter_assets is None:
        filter_assets =[]
        
    # 1. Load base 3 assets
    df_signal = load_ticker_csv(signal_ticker, data_dir).rename(columns={'close': 'signal_close'})
    df_target = load_ticker_csv(target_ticker, data_dir).rename(columns={'close': 'target_close'})
    df_bench = load_ticker_csv(benchmark_ticker, data_dir).rename(columns={'close': 'benchmark_close'})
    
    # Merge base 3
    master_df = pd.merge(df_signal, df_target, on='date', how='inner')
    master_df = pd.merge(master_df, df_bench, on='date', how='inner')
    
    # 2. Load and merge filter assets
    # We use set() to remove duplicates just in case a ticker was listed twice
    for ticker in set(filter_assets):
        # We explicitly name these columns by their ticker (e.g., SPY_close)
        df_filter = load_ticker_csv(ticker, data_dir).rename(columns={'close': f'{ticker}_close'})
        master_df = pd.merge(master_df, df_filter, on='date', how='inner')
    
    # Drop any rows with missing data across all merged assets
    master_df = master_df.dropna().reset_index(drop=True)
    
    return master_df

# --- TEST ---
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    data_directory = base_dir / "data"
    
    try:
        print("Testing Multi-Asset Merge with Filter Assets...")
        
        # Base assets
        test_signal = "XLF" # Or QQQ if you don't have XLF downloaded
        test_target = "VIXY"
        test_bench = "SPY"
        
        # New Filter Assets!
        test_filters = ["SPY", "QQQ"]
        
        master = build_master_dataframe(
            test_signal, test_target, test_bench, data_directory, filter_assets=test_filters
        )
        
        print("\nColumns successfully generated:")
        print(master.columns.tolist())
        
        print("\nMaster DataFrame Head (First 3 rows):")
        pd.set_option('display.max_columns', None)
        print(master.head(3))
        
        # Verify the custom named columns exist
        if 'SPY_close' in master.columns and 'QQQ_close' in master.columns:
            print("\n-> Filter asset columns cleanly added.")
            print("\nPASS")
        else:
            print("\nFAIL: Filter columns missing.")
        
    except Exception as e:
        print(f"FAIL: {e}")