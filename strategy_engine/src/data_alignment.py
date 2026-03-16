import os
import pandas as pd
from pathlib import Path

def load_ticker_csv(ticker, data_dir):
    """
    Reads the price CSV for a given ticker.
    Converts 'date' to a pandas datetime object and sorts chronologically.
    """
    file_path = data_dir / f"{ticker}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Data file for {ticker} not found at {file_path}")
        
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df[['date', 'close']]

def build_master_dataframe(signal_ticker, target_ticker, benchmark_ticker, data_dir):
    """
    Loads signal, target, and benchmark CSVs.
    Renames columns according to the specification.
    Merges them into a single aligned DataFrame, dropping any missing dates.
    """
    # Load individual datasets
    df_signal = load_ticker_csv(signal_ticker, data_dir)
    df_target = load_ticker_csv(target_ticker, data_dir)
    df_bench = load_ticker_csv(benchmark_ticker, data_dir)
    
    # Rename columns to prevent overlap and match Master DataFrame spec
    df_signal = df_signal.rename(columns={'close': 'signal_close'})
    df_target = df_target.rename(columns={'close': 'target_close'})
    df_bench = df_bench.rename(columns={'close': 'benchmark_close'})
    
    # Merge datasets on 'date' using inner join to align perfectly
    # This automatically sets the start date to max(first_available_date_of_each_ticker)
    master_df = pd.merge(df_signal, df_target, on='date', how='inner')
    master_df = pd.merge(master_df, df_bench, on='date', how='inner')
    
    # Drop any rows with missing data
    master_df = master_df.dropna().reset_index(drop=True)
    
    return master_df

# --- TEST ---
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    data_directory = base_dir / "data"
    
    try:
        print("Testing Multi-Asset Merge...")
        
        # We use QQQ as signal, and SPY as both target and benchmark just to test the merge logic
        # since we know you have QQQ and SPY downloaded from Step 4.
        test_signal = "QQQ"
        test_target = "SPY"
        test_bench = "SPY"
        
        master = build_master_dataframe(test_signal, test_target, test_bench, data_directory)
        
        print(f"Successfully merged {test_signal}, {test_target}, and {test_bench}.")
        print(f"Total Aligned Rows: {len(master)}")
        print(f"Effective Start Date: {master['date'].min().date()}")
        print(f"Effective End Date: {master['date'].max().date()}")
        print("\nMaster DataFrame Head:")
        print(master.head())
        print("\nPASS")
        
    except Exception as e:
        print(f"FAIL: {e}")