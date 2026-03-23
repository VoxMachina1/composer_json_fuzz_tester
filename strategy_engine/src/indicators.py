import pandas as pd
import numpy as np
from pathlib import Path

# Try importing the alignment module for testing purposes
try:
    from data_alignment import build_master_dataframe
except ImportError:
    from .data_alignment import build_master_dataframe

def calculate_sma(series, period):
    """Simple Moving Average"""
    return series.rolling(window=period).mean()

def calculate_ema(series, period):
    """Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()

def calculate_rsi(series, period):
    """
    Relative Strength Index (Wilder's Smoothing)
    """
    delta = series.diff()
    
    # Separate gains and losses
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    
    # Wilder's Smoothing uses an EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Handle edge case where average loss is 0 (RSI = 100)
    rsi = rsi.replace(np.inf, 100)
    
    # Set the first 'period' rows to NaN to match standard indicator behavior
    rsi.iloc[:period] = np.nan
    
    return rsi

def calculate_cumret(series, period):
    """Cumulative Return over a rolling window (percentage, not decimal)."""
    return series.pct_change(periods=period) * 100

def add_indicator(df, asset_role, indicator_name, period):
    """
    Calculates a technical indicator and appends it to the Master DataFrame.
    """
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    price_col = f"{asset_role}_close"
    if price_col not in df.columns:
        raise ValueError(f"Required price column '{price_col}' not found in DataFrame.")
        
    indicator_col = f"{asset_role}_{indicator_name}_{period}"
    
    if indicator_name.upper() == "RSI":
        df[indicator_col] = calculate_rsi(df[price_col], period)
    elif indicator_name.upper() == "SMA":
        df[indicator_col] = calculate_sma(df[price_col], period)
    elif indicator_name.upper() == "EMA":
        df[indicator_col] = calculate_ema(df[price_col], period)
    elif indicator_name.upper() == "CUMRET":
        df[indicator_col] = calculate_cumret(df[price_col], period)
    else:
        raise ValueError(f"Unsupported indicator: {indicator_name}")
        
    return df

# --- TEST ---
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    data_directory = base_dir / "data"
    
    try:
        print("Testing Pure Pandas Indicator Calculations...")
        
        # Build a temporary master dataframe using QQQ and SPY
        df_master = build_master_dataframe("QQQ", "SPY", "SPY", data_directory)
        
        # 1. Add Signal RSI (10)
        df_master = add_indicator(df_master, "signal", "RSI", 10)
        
        # 2. Add Benchmark SMA (200)
        df_master = add_indicator(df_master, "benchmark", "SMA", 200)
        
        print("\nColumns after indicators added:")
        print(df_master.columns.tolist())
        
        # Drop NaNs created by the 200-day warmup period
        df_clean = df_master.dropna().reset_index(drop=True)
        
        print(f"\nRows remaining after dropping warmup period: {len(df_clean)}")
        print("\nCleaned Master DataFrame Head (first 3 rows):")
        # Format pandas output for readability
        pd.set_option('display.max_columns', None)
        print(df_clean[['date', 'signal_close', 'signal_RSI_10', 'benchmark_SMA_200']].head(3))
        
        print("\nPASS")
        
    except Exception as e:
        print(f"FAIL: {e}")