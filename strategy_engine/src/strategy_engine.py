import numpy as np
import pandas as pd
from pathlib import Path

# Try importing dependencies for testing purposes
try:
    from config_loader import load_config
    from data_alignment import build_master_dataframe
    from indicators import add_indicator
    from preconditions import evaluate_preconditions
    from signals import generate_signals
except ImportError:
    from .config_loader import load_config
    from .data_alignment import build_master_dataframe
    from .indicators import add_indicator
    from .preconditions import evaluate_preconditions
    from .signals import generate_signals

def calculate_asset_returns(df):
    """
    Calculates the forward 1-day return for the target and benchmark assets.
    """
    df = df.copy()
    df['target_return'] = df['target_close'].pct_change().shift(-1)
    df['benchmark_return'] = df['benchmark_close'].pct_change().shift(-1)
    df = df.dropna(subset=['target_return', 'benchmark_return']).reset_index(drop=True)
    return df

def calculate_strategy_returns(df):
    """
    Combines signals with asset returns to generate the strategy's daily returns.
    100% Target Asset when Signal is 1.
    100% Benchmark Asset when Signal is 0.
    """
    df = df.copy()
    
    # Vectorized conditional assignment: if signal_active == 1, target_return, else benchmark_return
    df['strategy_return'] = np.where(
        df['signal_active'] == 1,
        df['target_return'],
        df['benchmark_return']
    )
    return df

def filter_date_range(df, start_date, end_date):
    """
    Filters the dataframe to the specified date range for the backtest window.
    """
    df = df.copy()
    if start_date:
        df = df[df['date'] >= start_date]
    if end_date:
        df = df[df['date'] <= end_date]
        
    return df.reset_index(drop=True)

def calculate_equity_curves(df, initial_capital=1.0):
    """
    Calculates cumulative compounding equity curves.
    Starts from the beginning of the filtered dataset.
    """
    df = df.copy()
    
    # (1 + Return) cumulative product
    df['strategy_equity'] = initial_capital * (1 + df['strategy_return']).cumprod()
    df['benchmark_equity'] = initial_capital * (1 + df['benchmark_return']).cumprod()
    
    return df

# --- TEST ---
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    data_directory = base_dir / "data"
    
    try:
        print("Testing Full Strategy Engine...")
        
        # 1. Load config
        cfg, _ = load_config()
        preconds = cfg.get('preconditions',[])
        sig_op = cfg.get('signal_operator', '>=')
        threshold = cfg.get('threshold_start', 50.0)
        ind_period = cfg.get('indicator_period', 10)
        ind_name = cfg.get('indicator', 'RSI')
        start_date = cfg.get('date_range', {}).get('start', '2020-01-01')
        end_date = cfg.get('date_range', {}).get('end', '2026-01-01')
        
        sig_col = f"signal_{ind_name}_{ind_period}"
        
        # 2. Build Pipeline
        df = build_master_dataframe("QQQ", "SPY", "SPY", data_directory)
        df = add_indicator(df, "signal", ind_name, ind_period)
        df = add_indicator(df, "benchmark", "SMA", 200)
        df = df.dropna().reset_index(drop=True)
        
        df = evaluate_preconditions(df, preconds)
        df = generate_signals(df, sig_col, sig_op, threshold)
        
        # 3. Apply Strategy Engine Logic
        df = calculate_asset_returns(df)
        df = filter_date_range(df, start_date, end_date)
        df = calculate_strategy_returns(df)
        df = calculate_equity_curves(df)
        
        # 4. Results Output
        print("\nAll Required Columns Present:")
        print(df.columns.tolist())
        
        print(f"\nBacktest Window: {df['date'].min()} to {df['date'].max()}")
        print(f"Total Trading Days: {len(df)}")
        
        final_strat_eq = df.iloc[-1]['strategy_equity']
        final_bench_eq = df.iloc[-1]['benchmark_equity']
        
        print(f"\nFinal Strategy Equity Multiplier:  {final_strat_eq:.4f}x")
        print(f"Final Benchmark Equity Multiplier: {final_bench_eq:.4f}x")
        
        print("\nSample Output (Last 3 Rows):")
        pd.set_option('display.max_columns', None)
        print(df[['date', 'signal_active', 'strategy_return', 'strategy_equity', 'benchmark_equity']].tail(3))
        
        print("\nPASS")
        
    except Exception as e:
        print(f"FAIL: {e}")