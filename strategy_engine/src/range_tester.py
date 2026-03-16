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
    from strategy_engine import calculate_asset_returns, filter_date_range, calculate_strategy_returns, calculate_equity_curves
    from metrics import calculate_metrics
except ImportError:
    from .config_loader import load_config
    from .data_alignment import build_master_dataframe
    from .indicators import add_indicator
    from .preconditions import evaluate_preconditions
    from .signals import generate_signals
    from .strategy_engine import calculate_asset_returns, filter_date_range, calculate_strategy_returns, calculate_equity_curves
    from .metrics import calculate_metrics

def generate_threshold_range(start, end, step):
    """
    Generates an inclusive list of float thresholds to test.
    """
    if step <= 0:
        raise ValueError("Step size must be greater than 0")
    thresholds = np.arange(start, end + (step / 2), step)
    return [round(float(x), 4) for x in thresholds]

def generate_asset_combinations(signal_assets, target_assets, benchmark_asset):
    """
    Generates all valid combinations of signal and target assets.
    Enforces constraint: signal_asset != target_asset.
    """
    combinations =[]
    for sig in signal_assets:
        for tgt in target_assets:
            if sig != tgt:
                combinations.append({
                    "signal_asset": sig,
                    "target_asset": tgt,
                    "benchmark_asset": benchmark_asset
                })
    return combinations

def run_threshold_range_tests(df, base_params, thresholds, sig_col, sig_op, start_date, end_date):
    """
    Loops through all thresholds for a single asset combination.
    Returns a list of metric dictionaries (one per threshold).
    """
    results =[]
    
    for thresh in thresholds:
        # 1. Update params for this specific run
        run_params = base_params.copy()
        run_params['threshold'] = thresh
        
        # 2. Generate Signals
        test_df = generate_signals(df, sig_col, sig_op, thresh)
        
        # 3. Filter to the backtest window
        test_df = filter_date_range(test_df, start_date, end_date)
        
        # 4. Run Strategy Engine
        test_df = calculate_strategy_returns(test_df)
        test_df = calculate_equity_curves(test_df)
        
        # 5. Calculate Metrics
        run_metrics = calculate_metrics(test_df, run_params)
        results.append(run_metrics)
        
    return results

# --- TEST ---
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    data_directory = base_dir / "data"
    
    try:
        print("Testing Strategy Discovery Loop...")
        
        # 1. Load config safely
        cfg, _ = load_config()
        if "preconditions" in cfg:
            preconds = cfg["preconditions"]
        else:
            preconds = list()
            
        sig_op = cfg.get('signal_operator', '>=')
        ind_period = cfg.get('indicator_period', 10)
        ind_name = cfg.get('indicator', 'RSI')
        date_cfg = cfg.get('date_range', {})
        start_date = date_cfg.get('start', '2020-01-01')
        end_date = date_cfg.get('end', '2026-01-01')
        sig_col = f"signal_{ind_name}_{ind_period}"
        
        # 2. Test Asset Combinations constraint
        sig_assets =["SPY", "QQQ"]
        tgt_assets = ["SPY", "VIXY"]
        combinations = generate_asset_combinations(sig_assets, tgt_assets, "SPY")
        
        print("\nGenerated Asset Combinations:")
        for combo in combinations:
            print(f"Signal: {combo['signal_asset']} -> Target: {combo['target_asset']}")
            if combo['signal_asset'] == combo['target_asset']:
                raise ValueError("Constraint Failed: Identical pair found!")
        print("-> Constraint rule (signal != target) enforced perfectly.")
        
        # 3. Test Range Loop Execution
        print("\nExecuting Strategy Loop for QQQ -> SPY using 3 thresholds (50, 60, 70)...")
        test_thresholds = [50.0, 60.0, 70.0]
        base_params = {
            "signal_asset": "QQQ",
            "target_asset": "SPY",
            "indicator": ind_name
        }
        
        # Build Master Dataframe ONCE
        df = build_master_dataframe("QQQ", "SPY", "SPY", data_directory)
        df = add_indicator(df, "signal", ind_name, ind_period)
        df = add_indicator(df, "benchmark", "SMA", 200)
        df = evaluate_preconditions(df, preconds)
        df = calculate_asset_returns(df) # Pre-calculate base asset returns
        df = df.dropna().reset_index(drop=True)
        
        # Run the loop
        loop_results = run_threshold_range_tests(
            df=df,
            base_params=base_params,
            thresholds=test_thresholds,
            sig_col=sig_col,
            sig_op=sig_op,
            start_date=start_date,
            end_date=end_date
        )
        
        # 4. Output validation
        print("\nLoop Results:")
        for res in loop_results:
            print(f"Threshold: {res['threshold']} | Win Rate: {res['Win_Rate']} | Sharpe: {res['Sharpe_Ratio']}")
            
        print(f"\nTotal tests executed in loop: {len(loop_results)}")
        print("\nPASS")
        
    except Exception as e:
        print(f"FAIL: {e}")