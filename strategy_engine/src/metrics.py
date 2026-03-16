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
    from strategy_engine import calculate_asset_returns, calculate_strategy_returns, filter_date_range, calculate_equity_curves
except ImportError:
    from .config_loader import load_config
    from .data_alignment import build_master_dataframe
    from .indicators import add_indicator
    from .preconditions import evaluate_preconditions
    from .signals import generate_signals
    from .strategy_engine import calculate_asset_returns, calculate_strategy_returns, filter_date_range, calculate_equity_curves

def calculate_metrics(df, strategy_params):
    """
    Computes performance metrics from the backtest DataFrame.
    Returns a dictionary representing a single row of results.
    """
    metrics = strategy_params.copy()
    
    # Handle empty dataframe edge case
    if df.empty:
        return metrics
        
    total_days = len(df)
    active_days = df[df['signal_active'] == 1]
    
    # 1. Trade & Holding Stats
    total_trades = len(active_days)
    
    # Win rate strictly based on active days
    if total_trades > 0:
        win_rate = len(active_days[active_days['strategy_return'] > 0]) / total_trades
        avg_return = active_days['strategy_return'].mean()
        
        # Calculate streaks for average hold days
        # A streak starts when signal is 1 and previous day is 0
        streak_starts = ((df['signal_active'] == 1) & (df['signal_active'].shift(1) != 1)).sum()
        avg_hold_days = total_trades / streak_starts if streak_starts > 0 else 0
    else:
        win_rate = 0.0
        avg_return = 0.0
        avg_hold_days = 0.0

    # 2. Return Stats
    final_equity = df['strategy_equity'].iloc[-1]
    total_return = final_equity - 1.0
    annualized_return = (final_equity ** (252 / total_days)) - 1.0 if total_days > 0 else 0.0
    
    # 3. Risk Metrics (using 3% annualized risk-free rate)
    rf_daily = 0.03 / 252
    excess_returns = df['strategy_return'] - rf_daily
    strat_std = df['strategy_return'].std()
    
    sharpe_ratio = 0.0
    if strat_std > 0:
        sharpe_ratio = (excess_returns.mean() / strat_std) * np.sqrt(252)
        
    downside_returns = df['strategy_return'][df['strategy_return'] < 0]
    downside_std = downside_returns.std()
    
    sortino_ratio = 0.0
    if len(downside_returns) > 0 and downside_std > 0:
        sortino_ratio = (excess_returns.mean() / downside_std) * np.sqrt(252)
        
    # Drawdown
    rolling_max = df['strategy_equity'].cummax()
    drawdown = (df['strategy_equity'] / rolling_max) - 1.0
    max_drawdown = drawdown.min()
    
    calmar_ratio = 0.0
    if max_drawdown < 0:
        calmar_ratio = annualized_return / abs(max_drawdown)
        
    # 4. Benchmark Stats
    benchmark_avg_return = df['benchmark_return'].mean()
    benchmark_median_return = df['benchmark_return'].median()
    
    # Populate dictionary
    metrics.update({
        'Total_Trades': total_trades,
        'Win_Rate': round(win_rate, 4),
        'Avg_Return': round(avg_return, 6),
        'Total_Return': round(total_return, 4),
        'Annualized_Return': round(annualized_return, 4),
        'Sharpe_Ratio': round(sharpe_ratio, 4),
        'Sortino_Ratio': round(sortino_ratio, 4),
        'Calmar_Ratio': round(calmar_ratio, 4),
        'Max_Drawdown': round(max_drawdown, 4),
        'Final_Equity': round(final_equity, 4),
        'Avg_Hold_Days': round(avg_hold_days, 2),
        'Benchmark_Avg_Return': round(benchmark_avg_return, 6),
        'Benchmark_Median_Return': round(benchmark_median_return, 6)
    })
    
    return metrics

# --- TEST ---
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    data_directory = base_dir / "data"
    
    try:
        print("Testing Metrics Calculation...")
        
        # 1. Load config (Avoiding the bugged character sequence)
        cfg, _ = load_config()
        if "preconditions" in cfg:
            preconds = cfg["preconditions"]
        else:
            preconds = list()
            
        sig_op = cfg.get('signal_operator', '>=')
        threshold = cfg.get('threshold_start', 50.0)
        ind_period = cfg.get('indicator_period', 10)
        ind_name = cfg.get('indicator', 'RSI')
        
        # safely get dates
        date_cfg = cfg.get('date_range', {})
        start_date = date_cfg.get('start', '2020-01-01')
        end_date = date_cfg.get('end', '2026-01-01')
        
        sig_col = f"signal_{ind_name}_{ind_period}"
        
        # 2. Build Pipeline
        df = build_master_dataframe("QQQ", "SPY", "SPY", data_directory)
        df = add_indicator(df, "signal", ind_name, ind_period)
        df = add_indicator(df, "benchmark", "SMA", 200)
        df = df.dropna().reset_index(drop=True)
        
        df = evaluate_preconditions(df, preconds)
        df = generate_signals(df, sig_col, sig_op, threshold)
        
        df = calculate_asset_returns(df)
        df = filter_date_range(df, start_date, end_date)
        df = calculate_strategy_returns(df)
        df = calculate_equity_curves(df)
        
        # 3. Calculate Metrics
        # Pass a dummy params dictionary so we can see it merge
        dummy_params = {
            "Signal_Asset": "QQQ",
            "Target_Asset": "SPY",
            "Threshold": threshold
        }
        
        results = calculate_metrics(df, dummy_params)
        
        # 4. Results Output
        print("\nMetrics Output:")
        for key, value in results.items():
            print(f"{key}: {value}")
            
        print("\nPASS")
        
    except Exception as e:
        print(f"FAIL: {e}")