import sys
import re
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

# Ensure Python can find our src directory regardless of where the script is run from
base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))

from src.config_loader import load_config
from src.data_loader import check_freshness_and_update
from src.data_alignment import build_master_dataframe
from src.indicators import add_indicator
from src.preconditions import evaluate_preconditions
from src.strategy_engine import calculate_asset_returns
from src.range_tester import generate_threshold_range, generate_asset_combinations, run_threshold_range_tests

def save_results(results_list, results_dir, run_id, mode="per_strategy"):
    """Saves the list of strategy metrics to a CSV file."""
    if not results_list:
        print("Warning: No results to save.")
        return
        
    df = pd.DataFrame(results_list)
    df['run_id'] = run_id
    
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if mode == "master":
        master_file = results_dir / "results.csv"
        if master_file.exists():
            df.to_csv(master_file, mode='a', header=False, index=False)
        else:
            df.to_csv(master_file, index=False)
        print(f" -> Appended to master: {master_file.name}")
    else:
        strategy_file = results_dir / f"{run_id}.csv"
        df.to_csv(strategy_file, index=False)
        print(f" -> Batch results saved to: {strategy_file.name}")
        
    return True

def parse_and_add_precondition_indicators(df, precondition_string):
    """
    Dynamically scans the boolean string for indicator patterns like 'SPY_RSI_10' or 'signal_SMA_200'.
    Automatically calculates and adds them to the dataframe.
    """
    if not precondition_string or str(precondition_string).strip().lower() in ["", "none", "[]"]:
        return df
        
    # Regex pattern looks for: [Letters]_[SMA/EMA/RSI]_[Numbers]
    pattern = r'([a-zA-Z]+)_(SMA|EMA|RSI)_(\d+)'
    matches = re.findall(pattern, str(precondition_string), re.IGNORECASE)
    
    for role, ind_name, period_str in matches:
        try:
            period = int(period_str)
            col_name = f"{role}_{ind_name.upper()}_{period}"
            close_col = f"{role}_close"
            
            # Warn if they typo'd a ticker or forgot to put it in filter_assets
            if close_col not in df.columns:
                print(f" [!] Warning: Cannot calculate {col_name} because '{close_col}' is missing. Did you add {role} to filter_assets?")
                continue
                
            # Only calculate if it's not already in the dataframe
            if col_name not in df.columns:
                df = add_indicator(df, role, ind_name.upper(), period)
        except Exception as e:
            print(f" [!] Error calculating precondition indicator {role}_{ind_name}_{period_str}: {e}")
            
    return df

def main():
    parser = argparse.ArgumentParser(description="Run the Strategy Discovery Engine")
    parser.add_argument(
        "--config", "-c", 
        type=str, 
        default="config/strategy_config.yaml", 
        help="Path to the config YAML file (relative to strategy_engine/)"
    )
    args = parser.parse_args()

    print("="*50)
    print(" STARTING STRATEGY DISCOVERY ENGINE ")
    print("="*50)
    
    data_dir = base_dir / "data"
    results_dir = base_dir / "results"
    
    # 1. Load Configuration
    print(f"\n[1] Loading Configuration from {args.config}...")
    cfg, api_keys = load_config(args.config)
    
    sig_assets = [t.upper() for t in cfg.get("signal_assets", [])]
    tgt_assets = [t.upper() for t in cfg.get("target_assets", [])]
    bench_asset = cfg.get("benchmark_asset", "SPY").upper()
    
    # Load our new filter_assets list
    filter_assets = [t.upper() for t in cfg.get("filter_assets", [])]
    
    ind_name = cfg.get("indicator", "RSI").upper()
    ind_period = cfg.get("indicator_period", 10)
    sig_op = cfg.get("signal_operator", ">")
    
    t_start = cfg.get("threshold_start", 50.0)
    t_end = cfg.get("threshold_end", 80.0)
    t_step = cfg.get("threshold_step", 0.5)
    
    # Load our new boolean string
    preconds_str = cfg.get("preconditions", "")
    
    start_date = cfg.get("date_range", {}).get("start", "2020-01-01")
    end_date = cfg.get("date_range", {}).get("end", "2026-01-01")
    
    results_mode = cfg.get("results_mode", "per_strategy")
    
    config_name = Path(args.config).stem
    now_str = datetime.now().strftime("%Y-%m-%d_%H%M")
    batch_run_id = f"Batch_{config_name}_{ind_name}_{now_str}"
    
    # 2. Update Price Data (Now includes filter_assets!)
    print("\n[2] Checking Data Freshness...")
    all_tickers = list(set(sig_assets + tgt_assets + [bench_asset] + filter_assets))
    check_freshness_and_update(all_tickers, api_keys, data_dir)
    
    # 3. Generate Combinations & Thresholds
    print("\n[3] Generating Test Matrices...")
    combinations = generate_asset_combinations(sig_assets, tgt_assets, bench_asset)
    thresholds = generate_threshold_range(t_start, t_end, t_step)
    sig_col = f"signal_{ind_name}_{ind_period}"
    
    print(f" -> Found {len(combinations)} valid asset combinations.")
    print(f" -> Testing {len(thresholds)} thresholds per combination.")
    print(f" -> Total strategy runs queued: {len(combinations) * len(thresholds)}")
    if preconds_str:
        print(f" -> Preconditions: '{preconds_str}'")
    
    # 4. Strategy Discovery Loop
    print("\n[4] Executing Strategy Engine...")
    
    all_batch_results = []
    
    for combo in combinations:
        sig = combo["signal_asset"]
        tgt = combo["target_asset"]
        ben = combo["benchmark_asset"]
        
        print(f"\nEvaluating: Signal={sig} | Target={tgt} | Benchmark={ben}")
        
        try:
            # Pass filter_assets down to the alignment script
            df = build_master_dataframe(sig, tgt, ben, data_dir, filter_assets=filter_assets)
            
            # Primary signal indicator
            df = add_indicator(df, "signal", ind_name, ind_period)
            
            # Dynamically parse and add ANY indicators required by the boolean string!
            df = parse_and_add_precondition_indicators(df, preconds_str)
            
            # Evaluate the boolean string
            df = evaluate_preconditions(df, preconds_str)
            
            df = calculate_asset_returns(df)
            df = df.dropna().reset_index(drop=True)
            
            base_params = {
                "signal_asset": sig,
                "target_asset": tgt,
                "benchmark_asset": ben,
                "indicator": ind_name,
                "indicator_period": ind_period,
                "slippage_bps": cfg.get("slippage_bps", 1.0),
                "risk_free_rate": cfg.get("risk_free_rate", 0.0)
            }
            
            results = run_threshold_range_tests(
                df=df,
                base_params=base_params,
                thresholds=thresholds,
                sig_col=sig_col,
                sig_op=sig_op,
                start_date=start_date,
                end_date=end_date
            )
            
            all_batch_results.extend(results)
            
        except Exception as e:
            print(f" [!] Error processing {sig}->{tgt}: {e}")
            
    # 5. Save all aggregated results to one file
    print("\n[5] Saving Final Batch Results...")
    save_results(all_batch_results, results_dir, batch_run_id, mode=results_mode)
            
    print("\n" + "="*50)
    print(" STRATEGY DISCOVERY COMPLETE ")
    print("="*50)

if __name__ == "__main__":
    main()