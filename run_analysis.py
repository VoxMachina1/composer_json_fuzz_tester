"""
run_analysis.py
===============
Orchestrates the RSI backtest pipeline from strategy JSON to filtered results.

Two modes, selected at startup:

  [1] Full pipeline (default)
      Runs two complete backtest passes automatically:
        Pass 1: signal_operator '>'  thresholds 50.0 -> 99.0
        Pass 2: signal_operator '<'  thresholds  1.0 -> 50.0
      Then automatically runs filter_results.py to produce filtered.csv.
      threshold_step and all other settings come from template.yaml.
      signal_operator and threshold_start/end in template.yaml are ignored.

  [2] Single operator
      Runs exactly one pass using all settings from template.yaml as-is.
      Outputs a raw timestamped CSV. No filtering. Useful for exploration
      and manual analysis.

Canonical input:  pathfinder/strategy.json
Canonical config: strategy_engine/config/template.yaml
Canonical output: strategy_engine/results/STRATEGY_NAME_YYYYMMDD_HHMMSS.csv

Usage:
    python run_analysis.py
"""

import sys
import json
import re
import csv
import os
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Canonical paths
# ---------------------------------------------------------------------------

SCRIPT_DIR     = Path(__file__).resolve().parent
STRATEGY_JSON  = SCRIPT_DIR / "pathfinder" / "strategy.json"
ENGINE_SRC     = SCRIPT_DIR / "strategy_engine" / "src"
DATA_DIR       = SCRIPT_DIR / "strategy_engine" / "data"
TEMPLATE_PATH  = SCRIPT_DIR / "strategy_engine" / "config" / "template.yaml"
RESULTS_DIR    = SCRIPT_DIR / "strategy_engine" / "results"
PATHFINDER_DIR = SCRIPT_DIR / "pathfinder"
FILTER_SCRIPT  = SCRIPT_DIR / "strategy_filter" / "filter_results.py"

# ---------------------------------------------------------------------------
# Mode 1 hardcoded threshold ranges
# ---------------------------------------------------------------------------

MODE1_GT_START = 50.0
MODE1_GT_END   = 99.0
MODE1_LT_START =  1.0
MODE1_LT_END   = 50.0

# ---------------------------------------------------------------------------
# Extra signal assets — always included regardless of strategy content
# ---------------------------------------------------------------------------

EXTRA_SIGNAL_ASSETS = [
    "SPY", "SPYV", "IOO", "VTV", "QQQ", "QQQE",
    "XLF", "XLK", "XLE", "XLY", "XLP", "TLT",
    "USO", "CORP", "GLD"
]

# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------

for _path, _label in [
    (STRATEGY_JSON, "Strategy JSON"),
    (ENGINE_SRC,    "Engine src/"),
    (TEMPLATE_PATH, "Template config"),
]:
    if not _path.exists():
        print(f"ERROR: {_label} not found at {_path}", file=sys.stderr)
        sys.exit(1)

sys.path.insert(0, str(ENGINE_SRC))
sys.path.insert(0, str(PATHFINDER_DIR))

# ---------------------------------------------------------------------------
# Load engine settings from template.yaml
# ---------------------------------------------------------------------------

import yaml as _yaml
with open(TEMPLATE_PATH, "r") as _f:
    _tmpl = _yaml.safe_load(_f)

TEMPLATE_TARGETS  = [t.upper() for t in _tmpl.get("target_assets", [])]
INDICATOR         = _tmpl.get("indicator", "RSI")
INDICATOR_PERIOD  = _tmpl.get("indicator_period", 10)
THRESHOLD_STEP    = _tmpl.get("threshold_step", 0.5)
SLIPPAGE_BPS      = _tmpl.get("slippage_bps", 1.0)
RISK_FREE_RATE    = _tmpl.get("risk_free_rate", 0.0)
DATE_START        = _tmpl.get("date_range", {}).get("start", "2015-01-01")
DATE_END          = _tmpl.get("date_range", {}).get("end", "2026-03-15")

# These are only used in Mode 2
TEMPLATE_OPERATOR       = _tmpl.get("signal_operator", ">")
TEMPLATE_THRESHOLD_START = _tmpl.get("threshold_start", 50.0)
TEMPLATE_THRESHOLD_END   = _tmpl.get("threshold_end", 80.0)

# ---------------------------------------------------------------------------
# Imports
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
from strategy_paths  import extract_paths


# ---------------------------------------------------------------------------
# Helper: extract tickers from engine_precondition string
# ---------------------------------------------------------------------------

def extract_tickers_from_precondition(engine_precondition):
    if not engine_precondition:
        return set()
    tickers = set()
    tokens = re.findall(r'\b([A-Z][A-Z0-9]*)_[A-Za-z_0-9]+', engine_precondition)
    for t in tokens:
        tickers.add(t)
    return tickers


# ---------------------------------------------------------------------------
# Helper: build config dict for one pass
# ---------------------------------------------------------------------------

def build_config(path, signal_asset, target_asset, filter_assets,
                 benchmark_asset, operator, threshold_start,
                 threshold_end):
    return {
        "signal_assets":    [signal_asset],
        "target_assets":    [target_asset],
        "benchmark_asset":  benchmark_asset,
        "filter_assets":    list(filter_assets),
        "indicator":        INDICATOR,
        "indicator_period": INDICATOR_PERIOD,
        "signal_operator":  operator,
        "threshold_start":  threshold_start,
        "threshold_end":    threshold_end,
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
# Helper: run engine pipeline for one config, return list of result dicts
# ---------------------------------------------------------------------------

def run_pipeline(config, path_meta):
    signal_asset    = config["signal_assets"][0]
    target_asset    = config["target_assets"][0]
    benchmark_asset = config["benchmark_asset"]
    filter_assets   = config.get("filter_assets", [])
    preconds        = config.get("preconditions")
    ind_name        = config["indicator"]
    ind_period      = config["indicator_period"]
    sig_op          = config["signal_operator"]
    start_date      = config["date_range"]["start"]
    end_date        = config["date_range"]["end"]
    sig_col         = f"signal_{ind_name}_{ind_period}"

    thresholds = generate_threshold_range(
        config["threshold_start"],
        config["threshold_end"],
        config["threshold_step"]
    )

    df = build_master_dataframe(
        signal_asset, target_asset, benchmark_asset,
        DATA_DIR, filter_assets=filter_assets
    )
    df = add_indicator(df, "signal", ind_name, ind_period)

    if preconds:
        col_patterns = re.findall(
            r'\b([A-Z][A-Z0-9]+)_(RSI|SMA|EMA|CumRet)_(\d+)\b',
            preconds
        )
        for ticker, indicator, period in col_patterns:
            col_name = f"{ticker}_{indicator}_{period}"
            if col_name not in df.columns:
                df = add_indicator(df, ticker, indicator, int(period))

    df = df.dropna().reset_index(drop=True)
    df = evaluate_preconditions(df, preconds)
    df = calculate_asset_returns(df)

    results = []
    for thresh in thresholds:
        test_df = generate_signals(df, sig_col, sig_op, thresh)
        test_df = filter_date_range(test_df, start_date, end_date)
        test_df = calculate_strategy_returns(
            test_df, slippage_bps=config["slippage_bps"]
        )
        test_df = calculate_equity_curves(test_df)

        base_params = {
            "sub_strategy":        path_meta["sub_strategy"],
            "conditions":          " AND ".join(path_meta["conditions"]),
            "engine_precondition": path_meta["engine_precondition"],
            "endpoint":            path_meta["endpoint"],
            "benchmark_asset":     benchmark_asset,
            "target_asset":        target_asset,
            "signal_asset":        signal_asset,
            "signal_operator":     sig_op,
            "threshold":           thresh,
            "risk_free_rate":      config["risk_free_rate"],
            "slippage_bps":        config["slippage_bps"],
        }

        metrics = calculate_metrics(test_df, base_params)
        results.append(metrics)

    return results


# ---------------------------------------------------------------------------
# CSV column order
# ---------------------------------------------------------------------------

META_COLS = [
    "sub_strategy", "conditions", "engine_precondition",
    "endpoint", "benchmark_asset", "target_asset", "signal_asset",
    "signal_operator", "threshold", "slippage_bps", "risk_free_rate",
]
METRIC_COLS = [
    "Total_Trades", "Win_Rate", "Avg_Return", "Median_Return",
    "Benchmark_Avg_Return", "Benchmark_Median_Return",
    "Total_Return", "Annualized_Return",
    "Sharpe_Ratio", "Sortino_Ratio", "Calmar_Ratio",
    "Max_Drawdown", "Final_Equity", "Avg_Hold_Days",
]
ALL_COLS = META_COLS + METRIC_COLS


# ---------------------------------------------------------------------------
# Core backtest runner — one pass for one operator
# ---------------------------------------------------------------------------

def run_pass(path_results, api_keys, all_target_assets,
             operator, threshold_start, threshold_end, label):
    """
    Run the full backtest for all paths x targets x signals for one operator.
    Returns (all_rows, total_runs, failed_runs, output_path).
    """
    print(f"\n{'='*60}")
    print(f"  PASS: {label}  "
          f"(operator={operator}, "
          f"thresholds={threshold_start}–{threshold_end}, "
          f"step={THRESHOLD_STEP})")
    print(f"{'='*60}\n")

    all_rows    = []
    total_runs  = 0
    failed_runs = 0

    for path_idx, path in enumerate(path_results):
        endpoint = path["endpoint"]

        precond_tickers = extract_tickers_from_precondition(
            path["engine_precondition"]
        )
        signal_assets = sorted(
            (precond_tickers | {t.upper() for t in EXTRA_SIGNAL_ASSETS})
            - {endpoint.upper()}
        )
        filter_assets = list(precond_tickers | {endpoint.upper()})

        print(f"[{path_idx + 1}/{len(path_results)}] "
              f"{path['sub_strategy']} -> {endpoint} "
              f"({len(signal_assets)} signals)")

        target_candidates = [
            t for t in all_target_assets if t != endpoint.upper()
        ]

        for target in target_candidates:
            for signal in signal_assets:
                total_runs += 1
                try:
                    config = build_config(
                        path, signal, target,
                        filter_assets, benchmark_asset=path["endpoint"],
                        operator=operator,
                        threshold_start=threshold_start,
                        threshold_end=threshold_end,
                    )
                    rows = run_pipeline(config, path)
                    all_rows.extend(rows)
                    print(f"  {target} | {signal} -> {len(rows)} results")
                except Exception as e:
                    failed_runs += 1
                    print(f"  {target} | {signal} -> FAILED: {e}",
                          file=sys.stderr)

    # Write CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"{STRATEGY_JSON.stem}_{timestamp}.csv"
    output_path = RESULTS_DIR / output_name

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n  Pass complete.")
    print(f"  Total runs:  {total_runs}")
    print(f"  Failed runs: {failed_runs}")
    print(f"  Result rows: {len(all_rows)}")
    print(f"  Output:      {output_path}\n")

    return all_rows, total_runs, failed_runs, output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Mode selection ---
    print(f"\n{'='*60}")
    print(f"  RSI SEARCH — FRONTRUNNER DISCOVERY PIPELINE")
    print(f"{'='*60}")
    print(f"\n  [1] Full pipeline  — both operators, auto-filter (default)")
    print(f"  [2] Single operator — from template.yaml, raw CSV only\n")

    raw = input("  Enter choice (1/2, default 1): ").strip()
    mode = 2 if raw == "2" else 1
    print()

    # --- Parse strategy ---
    print(f"Parsing strategy: {STRATEGY_JSON}")
    with open(STRATEGY_JSON, "r", encoding="utf-8") as f:
        tree = json.load(f)

    path_results = extract_paths(tree)
    print(f"  Found {len(path_results)} paths\n")

    # --- API keys ---
    _, api_keys = load_config(config_dict={"_dummy": True})

    # --- Data freshness ---
    all_tickers = set()
    all_tickers.add("SPY")
    for path in path_results:
        all_tickers.add(path["endpoint"])
        all_tickers.update(
            extract_tickers_from_precondition(path["engine_precondition"])
        )
    all_tickers.update(t.upper() for t in EXTRA_SIGNAL_ASSETS)

    print(f"Checking data freshness for {len(all_tickers)} tickers...")
    check_freshness_and_update(list(all_tickers), api_keys, DATA_DIR)

    strategy_endpoints = {p["endpoint"].upper() for p in path_results}
    all_target_assets  = sorted(strategy_endpoints | set(TEMPLATE_TARGETS))

    # --- Mode 1: full pipeline ---
    if mode == 1:
        # Pass 1: overbought
        _, runs_gt, fails_gt, path_gt = run_pass(
            path_results, api_keys, all_target_assets,
            operator       = ">",
            threshold_start = MODE1_GT_START,
            threshold_end   = MODE1_GT_END,
            label          = "Overbought",
        )

        # Pass 2: oversold
        _, runs_lt, fails_lt, path_lt = run_pass(
            path_results, api_keys, all_target_assets,
            operator       = "<",
            threshold_start = MODE1_LT_START,
            threshold_end   = MODE1_LT_END,
            label          = "Oversold",
        )

        print(f"\n{'='*60}")
        print(f"  BOTH PASSES COMPLETE")
        print(f"  Total runs:   {runs_gt + runs_lt}")
        print(f"  Failed runs:  {fails_gt + fails_lt}")
        print(f"{'='*60}\n")

        # Auto-filter
        print("Running filter pipeline automatically...\n")

        import importlib.util as _ilu
        _fr_spec = _ilu.spec_from_file_location("filter_results", FILTER_SCRIPT)
        _fr_mod  = _ilu.module_from_spec(_fr_spec)
        _fr_spec.loader.exec_module(_fr_mod)

        filtered_csv = _fr_mod.run_filter(
            path_gt=path_gt,
            path_lt=path_lt,
        )

        print(f"\n{'='*60}")
        print(f"  PIPELINE COMPLETE")
        print(f"  Filtered CSV: {filtered_csv}")
        print(f"\n  Next step:")
        print(f"    python strategy_inserter/strategy_inserter.py --dry-run")
        print(f"{'='*60}\n")

    # --- Mode 2: single operator ---
    else:
        print(f"Mode 2 — single operator from template.yaml")
        print(f"  Operator:  {TEMPLATE_OPERATOR}")
        print(f"  Thresholds: {TEMPLATE_THRESHOLD_START} -> "
              f"{TEMPLATE_THRESHOLD_END}  (step {THRESHOLD_STEP})\n")

        _, total_runs, failed_runs, output_path = run_pass(
            path_results, api_keys, all_target_assets,
            operator        = TEMPLATE_OPERATOR,
            threshold_start = TEMPLATE_THRESHOLD_START,
            threshold_end   = TEMPLATE_THRESHOLD_END,
            label           = f"Single ({TEMPLATE_OPERATOR})",
        )

        print(f"\n{'='*60}")
        print(f"  COMPLETE")
        print(f"  Output: {output_path}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()