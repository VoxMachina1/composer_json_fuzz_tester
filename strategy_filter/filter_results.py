"""
filter_results.py
=================
Automatically finds the two most recent CSVs in strategy_engine/results/,
validates that one contains all '>' signal_operator rows and the other all '<',
applies performance filters to both, and outputs a combined filtered.csv.

Filter criteria (hardcoded — see memory note for future configurability):
  - Win_Rate > 0.75
  - Total_Trades > 20
  - Benchmark_Median_Return < 0

Can be run standalone or imported by run_analysis.py (Mode 1).

Canonical inputs:  two most recent CSVs in strategy_engine/results/
Canonical output:  strategy_filter/filtered.csv

Usage:
    python strategy_filter/filter_results.py
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Canonical paths
# ---------------------------------------------------------------------------

_HERE       = Path(__file__).resolve().parent        # strategy_filter/
_ROOT       = _HERE.parent                           # project root
RESULTS_DIR = _ROOT / "strategy_engine" / "results"
OUTPUT_CSV  = _HERE / "filtered.csv"
SUMMARY_TXT = _HERE / "filter_summary.txt"

# ---------------------------------------------------------------------------
# Hardcoded filter criteria
# (TODO: make configurable — see project memory note)
# ---------------------------------------------------------------------------

WIN_RATE_MIN                = 0.75
TOTAL_TRADES_MIN            = 20
BENCHMARK_MEDIAN_RETURN_MAX = 0.0


# ---------------------------------------------------------------------------
# CSV discovery
# ---------------------------------------------------------------------------

def find_two_most_recent_csvs(results_dir):
    """
    Scan results_dir for CSV files, excluding filtered.csv and lock files.
    Return the two most recently modified, or error if fewer than 2 exist.
    """
    candidates = [
        f for f in results_dir.glob("*.csv")
        if not f.name.startswith("filtered")
        and not f.name.startswith(".~lock")
        and f.name != "filtered.csv"
    ]

    if len(candidates) < 2:
        print(f"ERROR: Need at least 2 CSVs in {results_dir}, "
              f"found {len(candidates)}.", file=sys.stderr)
        sys.exit(1)

    candidates.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return candidates[0], candidates[1]


def find_two_specific_csvs(path_gt, path_lt):
    """
    Accept two explicit Path objects (already validated by run_analysis.py
    Mode 1). Returns them as (df_gt, df_lt, path_gt, path_lt) directly,
    bypassing file discovery and operator validation since the caller
    already knows which is which.
    """
    df_gt = pd.read_csv(path_gt)
    df_lt = pd.read_csv(path_lt)
    return df_gt, df_lt, path_gt, path_lt


# ---------------------------------------------------------------------------
# Operator validation
# ---------------------------------------------------------------------------

def validate_and_assign(csv_a, csv_b):
    """
    Load both CSVs, check that each has a signal_operator column,
    and that one is exclusively '>' and the other exclusively '<'.
    Returns (df_gt, df_lt, path_gt, path_lt).
    """
    df_a = pd.read_csv(csv_a)
    df_b = pd.read_csv(csv_b)

    for df, path in [(df_a, csv_a), (df_b, csv_b)]:
        if "signal_operator" not in df.columns:
            print(
                f"ERROR: '{path.name}' has no signal_operator column.\n"
                f"  Re-run run_analysis.py to generate CSVs with this column.",
                file=sys.stderr
            )
            sys.exit(1)

    ops_a = set(df_a["signal_operator"].unique())
    ops_b = set(df_b["signal_operator"].unique())

    for ops, path in [(ops_a, csv_a), (ops_b, csv_b)]:
        if len(ops) > 1:
            print(
                f"ERROR: '{path.name}' contains mixed operators: {ops}.\n"
                f"  Each CSV should come from a single run_analysis.py run.",
                file=sys.stderr
            )
            sys.exit(1)

    op_a = next(iter(ops_a))
    op_b = next(iter(ops_b))

    if not ({op_a, op_b} == {">", "<"}):
        print(
            f"ERROR: Expected one '>' CSV and one '<' CSV, "
            f"got '{op_a}' and '{op_b}'.\n"
            f"  Run run_analysis.py twice with different operators.",
            file=sys.stderr
        )
        sys.exit(1)

    if op_a == ">":
        return df_a, df_b, csv_a, csv_b
    else:
        return df_b, df_a, csv_b, csv_a


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def apply_filters(df, label):
    """
    Apply hardcoded performance filters. Returns filtered DataFrame.
    """
    before = len(df)

    mask = (
        (df["Win_Rate"]                > WIN_RATE_MIN)               &
        (df["Total_Trades"]            > TOTAL_TRADES_MIN)           &
        (df["Benchmark_Median_Return"] < BENCHMARK_MEDIAN_RETURN_MAX)
    )

    filtered = df[mask].copy()
    after = len(filtered)

    print(f"  [{label}] {before:,} rows -> {after:,} passed "
          f"({before - after:,} removed)")

    return filtered


# ---------------------------------------------------------------------------
# Core pipeline — callable by both main() and run_analysis.py Mode 1
# ---------------------------------------------------------------------------

def run_filter(path_gt=None, path_lt=None):
    """
    Execute the full filter pipeline.

    If path_gt and path_lt are provided (Path objects), use them directly —
    this is the Mode 1 path called from run_analysis.py after both passes
    complete, bypassing file discovery and operator validation.

    If neither is provided, auto-discover the two most recent CSVs in
    strategy_engine/results/ and validate their operators — this is the
    standalone usage path.

    Returns the output CSV path on success.
    """
    print(f"\n{'='*60}")
    print(f"  FILTER RESULTS PIPELINE")
    print(f"{'='*60}\n")

    if path_gt is not None and path_lt is not None:
        # Called from run_analysis.py Mode 1 — paths already known
        df_gt = pd.read_csv(path_gt)
        df_lt = pd.read_csv(path_lt)
        print(f"Overbought ('>'):  {path_gt.name}  ({len(df_gt):,} rows)")
        print(f"Oversold  ('<'):   {path_lt.name}  ({len(df_lt):,} rows)\n")
    else:
        # Standalone — auto-discover and validate
        csv_recent, csv_older = find_two_most_recent_csvs(RESULTS_DIR)
        print(f"Most recent CSV:   {csv_recent.name}")
        print(f"Second recent CSV: {csv_older.name}\n")

        df_gt, df_lt, path_gt, path_lt = validate_and_assign(csv_recent, csv_older)
        print(f"Overbought ('>'):  {path_gt.name}  ({len(df_gt):,} rows)")
        print(f"Oversold  ('<'):   {path_lt.name}  ({len(df_lt):,} rows)\n")

    # Apply filters
    print(f"Applying filters:")
    print(f"  Win_Rate > {WIN_RATE_MIN}")
    print(f"  Total_Trades > {TOTAL_TRADES_MIN}")
    print(f"  Benchmark_Median_Return < {BENCHMARK_MEDIAN_RETURN_MAX}\n")

    filtered_gt = apply_filters(df_gt, "overbought >")
    filtered_lt = apply_filters(df_lt, "oversold  <")

    # Combine
    combined = pd.concat([filtered_gt, filtered_lt], ignore_index=True)
    print(f"\n  Combined: {len(combined):,} rows total")

    if combined.empty:
        print("\nWARNING: No rows passed filters. filtered.csv will be empty.",
              file=sys.stderr)

    # Write output
    _HERE.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Output written to: {OUTPUT_CSV}")

    # Write summary
    summary_lines = [
        f"Filter Results Summary",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"Inputs:",
        f"  Overbought: {path_gt.name}  ({len(df_gt):,} rows)",
        f"  Oversold:   {path_lt.name}  ({len(df_lt):,} rows)",
        f"",
        f"Filter criteria:",
        f"  Win_Rate > {WIN_RATE_MIN}",
        f"  Total_Trades > {TOTAL_TRADES_MIN}",
        f"  Benchmark_Median_Return < {BENCHMARK_MEDIAN_RETURN_MAX}",
        f"",
        f"Results:",
        f"  Overbought passed: {len(filtered_gt):,}",
        f"  Oversold passed:   {len(filtered_lt):,}",
        f"  Combined total:    {len(combined):,}",
        f"",
        f"Output: {OUTPUT_CSV}",
    ]
    with open(SUMMARY_TXT, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"  Summary written to: {SUMMARY_TXT}")

    print(f"\n{'='*60}\n")
    return OUTPUT_CSV


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    run_filter()


if __name__ == "__main__":
    main()