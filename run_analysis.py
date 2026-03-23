"""
run_analysis.py
===============
Orchestrates the full pipeline from a Composer/VOXPORT strategy JSON
to a combined results CSV.

Workflow:
  1. Parse the strategy JSON using strategy_paths to extract all paths
  2. For each path, build a config dict in memory
  3. Run the engine's range tester against that config
  4. Collect all results into a single CSV

Usage:
    python run_analysis.py strategy.json

Output:
    results/STRATEGY_NAME_YYYYMMDD_HHMMSS.csv
"""

import sys
import json
import re
import csv
import os
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Hardcoded constants — edit these as needed
# ---------------------------------------------------------------------------

EXTRA_SIGNAL_ASSETS = [
    "SPY", "SPYV", "IOO", "VTV", "QQQ", "QQQE",
    "XLF", "XLK", "XLE", "XLY", "XLP", "TLT",
    "USO", "CORP", "GLD"
]

RESULTS_DIR = r"C:\Python Projects\rsi_tester\strategy_engine\results"

# Engine settings — match your Composer-validated template
INDICATOR       = "RSI"
INDICATOR_PERIOD = 10
SIGNAL_OPERATOR = ">"
THRESHOLD_START = 50.0
THRESHOLD_END   = 80.0
THRESHOLD_STEP  = 0.5
SLIPPAGE_BPS    = 1.0
RISK_FREE_RATE  = 0.0
BENCHMARK_ASSET = "SPY"
DATE_START      = "2015-01-01"
DATE_END        = "2026-03-15"

# ---------------------------------------------------------------------------
# Path setup — locate the engine's src/ directory relative to this script
# ---------------------------------------------------------------------------

SCRIPT_DIR  = Path(__file__).resolve().parent
ENGINE_SRC  = SCRIPT_DIR / "strategy_engine" / "src"
DATA_DIR    = SCRIPT_DIR / "strategy_engine" / "data"

if not ENGINE_SRC.exists():
    print(f"ERROR: Engine src/ not found at {ENGINE_SRC}", file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, str(ENGINE_SRC))
PATHFINDER_DIR = SCRIPT_DIR / "pathfinder"
sys.path.insert(0, str(PATHFINDER_DIR))

# ---------------------------------------------------------------------------
# Imports — engine modules and path extractor
# ---------------------------------------------------------------------------

from config_loader   import load_config
from data_loader     import check_freshness_and_update
from data_alignment  import build_master_dataframe
from indicators      import add_indicator
from preconditions   import evaluate_preconditions
from signals         import generate_signals
from strategy_engine import calculate_asset_returns, filter_date_range, \
                            calculate_strategy_returns, calculate_equity_curves
from metrics         import calculate_metrics
from range_tester    import generate_threshold_range
import strategy_paths as sp


# ---------------------------------------------------------------------------
# Helper: extract tickers mentioned in an engine_precondition string
# e.g. "SPY_close > SPY_SMA_200 and TLT_RSI_20 > PSQ_RSI_20"
#   -> {"SPY", "TLT", "PSQ"}
# ---------------------------------------------------------------------------

def extract_tickers_from_precondition(engine_precondition):
    """
    Pull every ticker referenced in an engine precondition string.
    Tickers are the prefix before the first underscore in each column name.
    Skips fixed numeric values.
    """
    if not engine_precondition:
        return set()

    tickers = set()
    # Match tokens that look like column names: WORD_WORD or WORD_WORD_NUMBER
    tokens = re.findall(r'\b([A-Z][A-Z0-9]*)_[A-Za-z_0-9]+', engine_precondition)
    for t in tokens:
        tickers.add(t)
    return tickers


# ---------------------------------------------------------------------------
# Helper: build a config dict for one path + one signal asset
# ---------------------------------------------------------------------------

def build_config(path, signal_asset, filter_assets):
    return {
        "signal_assets":    [signal_asset],
        "target_assets":    [path["endpoint"]],
        "benchmark_asset":  BENCHMARK_ASSET,
        "filter_assets":    list(filter_assets),
        "indicator":        INDICATOR,
        "indicator_period": INDICATOR_PERIOD,
        "signal_operator":  SIGNAL_OPERATOR,
        "threshold_start":  THRESHOLD_START,
        "threshold_end":    THRESHOLD_END,
        "threshold_step":   THRESHOLD_STEP,
        "slippage_bps":     SLIPPAGE_BPS,
        "risk_free_rate":   RISK_FREE_RATE,
        "preconditions":    path["engine_precondition"] or None,
        "date_range": {
            "start": DATE_START,
            "end":   DATE_END,
        },
    }


# ---------------------------------------------------------------------------
# Helper: run the engine pipeline for one config, return list of result dicts
# ---------------------------------------------------------------------------

def run_pipeline(config, api_keys, path_meta):
    """
    Runs the full engine pipeline for a single config dict.
    Returns a list of result rows (one per threshold).
    path_meta: dict with sub_strategy, conditions, engine_precondition, endpoint
    """
    signal_asset  = config["signal_assets"][0]
    target_asset  = config["target_assets"][0]
    filter_assets = config.get("filter_assets", [])
    preconds      = config.get("preconditions")
    ind_name      = config["indicator"]
    ind_period    = config["indicator_period"]
    sig_op        = config["signal_operator"]
    start_date    = config["date_range"]["start"]
    end_date      = config["date_range"]["end"]
    sig_col       = f"signal_{ind_name}_{ind_period}"

    thresholds = generate_threshold_range(
        config["threshold_start"],
        config["threshold_end"],
        config["threshold_step"]
    )

    # Build master dataframe once for this signal/target/filter combination
    df = build_master_dataframe(
        signal_asset, target_asset, BENCHMARK_ASSET,
        DATA_DIR, filter_assets=filter_assets
    )
    df = add_indicator(df, "signal", ind_name, ind_period)

    # Add indicators for any filter assets referenced in preconditions
    # The precondition string tells us exactly which columns are needed
    if preconds:
        # Find all {TICKER}_{INDICATOR}_{PERIOD} patterns in precondition
        col_patterns = re.findall(
            r'\b([A-Z][A-Z0-9]+)_(RSI|SMA|EMA|CumRet)_(\d+)\b',
            preconds
        )
        for ticker, indicator, period in col_patterns:
            col_name = f"{ticker}_{indicator}_{period}"
            if col_name not in df.columns:
                df = add_indicator(df, ticker, indicator, int(period))

        # Also add plain _close columns if referenced (current-price conditions)
        # These are already present from build_master_dataframe for filter_assets
        # so no action needed — just drop NaNs after all indicators are added

    df = df.dropna().reset_index(drop=True)
    df = evaluate_preconditions(df, preconds)
    df = calculate_asset_returns(df)

    results = []
    for thresh in thresholds:
        test_df = generate_signals(df, sig_col, sig_op, thresh)
        test_df = filter_date_range(test_df, start_date, end_date)
        test_df = calculate_strategy_returns(test_df, slippage_bps=config["slippage_bps"])
        test_df = calculate_equity_curves(test_df)

        base_params = {
            "sub_strategy":        path_meta["sub_strategy"],
            "conditions":          " AND ".join(path_meta["conditions"]),
            "engine_precondition": path_meta["engine_precondition"],
            "endpoint":            path_meta["endpoint"],
            "signal_asset":        signal_asset,
            "threshold":           thresh,
            "risk_free_rate":      config["risk_free_rate"],
            "slippage_bps":        config["slippage_bps"],
        }

        metrics = calculate_metrics(test_df, base_params)
        results.append(metrics)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_analysis.py <strategy.json>", file=sys.stderr)
        sys.exit(1)

    strategy_file = Path(sys.argv[1])
    if not strategy_file.exists():
        print(f"ERROR: Strategy file not found: {strategy_file}", file=sys.stderr)
        sys.exit(1)

    # --- Step 1: Extract all paths from the strategy JSON ---
    print(f"\nParsing strategy: {strategy_file.name}")
    with open(strategy_file, "r", encoding="utf-8") as f:
        tree = json.load(f)

    path_results = []
    sp.walk(tree, conditions=[], engine_conds=[], sub_strategy=None, results=path_results)
    print(f"  Found {len(path_results)} paths across all sub-strategies")

    # --- Step 2: Load API keys (config_dict mode — no YAML file needed) ---
    _, api_keys = load_config(config_dict={"_dummy": True})

    # --- Step 3: Ensure all required data is fresh ---
    # Collect every ticker that will be needed across all paths
    all_tickers = set()
    all_tickers.add(BENCHMARK_ASSET)
    for path in path_results:
        all_tickers.add(path["endpoint"])
        all_tickers.update(extract_tickers_from_precondition(path["engine_precondition"]))
    all_tickers.update(t.upper() for t in EXTRA_SIGNAL_ASSETS)

    print(f"\nChecking data freshness for {len(all_tickers)} tickers...")
    check_freshness_and_update(list(all_tickers), api_keys, DATA_DIR)

    # --- Step 4: Run engine for each path x signal combination ---
    all_rows = []
    total_runs = 0
    failed_runs = 0

    for path_idx, path in enumerate(path_results):
        endpoint = path["endpoint"]

        # Build signal asset list: tickers from precondition + extra list
        # Remove the endpoint itself (can't be its own signal)
        precond_tickers = extract_tickers_from_precondition(path["engine_precondition"])
        signal_assets = list(
            (precond_tickers | {t.upper() for t in EXTRA_SIGNAL_ASSETS})
            - {endpoint.upper()}
        )
        signal_assets.sort()

        # Filter assets are whatever appears in the precondition
        filter_assets = list(precond_tickers)

        print(f"\n[{path_idx + 1}/{len(path_results)}] "
              f"{path['sub_strategy']} -> {endpoint} "
              f"({len(signal_assets)} signals)")

        for signal in signal_assets:
            total_runs += 1
            try:
                config = build_config(path, signal, filter_assets)
                rows = run_pipeline(config, api_keys, path)
                all_rows.extend(rows)
                print(f"  {signal} -> {len(rows)} threshold results")
            except Exception as e:
                failed_runs += 1
                print(f"  {signal} -> FAILED: {e}", file=sys.stderr)

    # --- Step 5: Write combined CSV ---
    if not all_rows:
        print("\nNo results to write.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"{strategy_file.stem}_{timestamp}.csv"
    output_path = Path(RESULTS_DIR) / output_name

    # Determine column order — path metadata first, then metrics
    meta_cols = [
        "sub_strategy", "conditions", "engine_precondition",
        "endpoint", "signal_asset", "threshold",
        "slippage_bps", "risk_free_rate"
    ]
    metric_cols = [
        "Total_Trades", "Win_Rate", "Avg_Return", "Median_Return",
        "Benchmark_Avg_Return", "Benchmark_Median_Return",
        "Total_Return", "Annualized_Return",
        "Sharpe_Ratio", "Sortino_Ratio", "Calmar_Ratio",
        "Max_Drawdown", "Final_Equity", "Avg_Hold_Days",
    ]
    all_cols = meta_cols + metric_cols

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n{'='*60}")
    print(f"  Complete.")
    print(f"  Total runs:   {total_runs}")
    print(f"  Failed runs:  {failed_runs}")
    print(f"  Result rows:  {len(all_rows)}")
    print(f"  Output:       {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()