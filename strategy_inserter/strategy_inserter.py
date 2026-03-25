"""
strategy_inserter.py
====================
Takes a Composer strategy JSON and a filtered results CSV (containing both
overbought '>' and oversold '<' rows), and inserts frontrunner logic at the
correct leaf nodes in the tree in a single pass.

For each unique (sub_strategy, conditions, endpoint) group, the matching
leaf asset node is replaced with a wt-cash-equal block. Each unique
(signal_asset, operator, most_inclusive_threshold) triple becomes one if node:

  wt-cash-equal
    IF signal_A_RSI > threshold_1  ->  [target_1, target_2]
      ELSE -> original_endpoint
    IF signal_A_RSI < threshold_2  ->  [target_3]
      ELSE -> original_endpoint
    IF signal_B_RSI > threshold_3  ->  [target_1]
      ELSE -> original_endpoint

Threshold deduplication (most inclusive per signal_asset+operator+target):
  '>' or '>=' : keep minimum threshold (fires most often)
  '<' or '<=' : keep maximum threshold (fires most often)

Tautology detection: if the same signal_asset appears with both '>' and '<'
at thresholds that together always evaluate true (thresh_gt < thresh_lt),
that signal_asset is skipped with a warning.

Canonical input (strategy):  pathfinder/strategy.json
Canonical input (CSV):        strategy_filter/filtered.csv (default)
Canonical output:             strategy_inserter/strategy_modified.json
                              strategy_inserter/insertion_log.json

Usage:
    python strategy_inserter/strategy_inserter.py
    python strategy_inserter/strategy_inserter.py path/to/custom_filtered.csv
    python strategy_inserter/strategy_inserter.py --dry-run
"""

import json
import sys
import uuid
import copy
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd

# ---------------------------------------------------------------------------
# Canonical paths
# ---------------------------------------------------------------------------

_HERE          = Path(__file__).resolve().parent         # strategy_inserter/
_ROOT          = _HERE.parent                            # project root
STRATEGY_JSON  = _ROOT / "pathfinder" / "strategy.json"
DEFAULT_CSV    = _ROOT / "strategy_filter" / "filtered.csv"
OUTPUT_JSON    = _HERE / "strategy_modified.json"
OUTPUT_LOG     = _HERE / "insertion_log.json"
PATHFINDER_DIR = _ROOT / "pathfinder"

# ---------------------------------------------------------------------------
# Import strategy_paths from pathfinder/
# ---------------------------------------------------------------------------

import importlib.util as _ilu
_sp_path = PATHFINDER_DIR / "strategy_paths.py"
if not _sp_path.exists():
    print(f"ERROR: strategy_paths.py not found at {_sp_path}", file=sys.stderr)
    sys.exit(1)

_spec   = _ilu.spec_from_file_location("strategy_paths", _sp_path)
_sp_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_sp_mod)
extract_paths = _sp_mod.extract_paths

# ---------------------------------------------------------------------------
# UUID helper
# ---------------------------------------------------------------------------

def new_id():
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Composer function name maps
# ---------------------------------------------------------------------------

FN_MAP = {
    "RSI":    "relative-strength-index",
    "SMA":    "moving-average-price",
    "CumRet": "cumulative-return",
    "Price":  "current-price",
}

COMPARATOR_MAP = {
    ">":  "gt",
    "<":  "lt",
    ">=": "gte",
    "<=": "lte",
    "==": "eq",
    "!=": "neq",
}


# ---------------------------------------------------------------------------
# Threshold deduplication
# ---------------------------------------------------------------------------

def most_inclusive_threshold(thresholds, operator):
    """
    Return the most inclusive threshold for a given operator.
      '>' or '>=' : minimum threshold (fires most often)
      '<' or '<=' : maximum threshold (fires most often)
    """
    if operator in (">", ">="):
        return min(thresholds)
    elif operator in ("<", "<="):
        return max(thresholds)
    else:
        return thresholds[0]


# ---------------------------------------------------------------------------
# Tautology detection
# ---------------------------------------------------------------------------

def is_tautology(thresh_gt, thresh_lt):
    """
    Returns True if RSI > thresh_gt OR RSI < thresh_lt is always true.
    This happens when thresh_gt < thresh_lt — the two conditions overlap
    and together cover all possible RSI values.

    Example: RSI > 16 OR RSI < 18.5 is always true since 16 < 18.5.
    Example: RSI > 80 OR RSI < 20 is NOT always true since 80 > 20.
    """
    return thresh_gt < thresh_lt


# ---------------------------------------------------------------------------
# Composer node builders
# ---------------------------------------------------------------------------

def build_asset_node(ticker):
    return {
        "id":                           new_id(),
        "step":                         "asset",
        "name":                         ticker,
        "suppress_incomplete_warnings": False,
        "ticker":                       ticker,
    }


def _fmt_threshold(value):
    return str(int(value)) if value == int(value) else str(value)


def build_if_node(signal_asset, operator, threshold, ind_period,
                  target_tickers, original_asset_node):
    """
    Build a single flat-style if/else node for one
    (signal_asset, operator, threshold) combination.

      IF signal_asset_RSI_period OP threshold -> [target_1, target_2, ...]
      ELSE -> original_asset_node
    """
    lhs_fn_raw = FN_MAP.get("RSI", "relative-strength-index")
    comp       = COMPARATOR_MAP.get(operator, "gt")
    thresh_str = _fmt_threshold(threshold)

    target_nodes = [build_asset_node(t) for t in target_tickers]

    if_child_true = {
        "id":                           new_id(),
        "step":                         "if-child",
        "name":                         "If",
        "is-else-condition?":           False,
        "suppress_incomplete_warnings": False,
        "lhs-fn":                       lhs_fn_raw,
        "lhs-fn-params":                {"window": ind_period},
        "lhs-val":                      signal_asset,
        "comparator":                   comp,
        "rhs-fixed-value?":             True,
        "rhs-val":                      thresh_str,
        "children":                     target_nodes,
    }

    if_child_else = {
        "id":                           new_id(),
        "step":                         "if-child",
        "name":                         "Else",
        "is-else-condition?":           True,
        "suppress_incomplete_warnings": True,
        "children":                     [copy.deepcopy(original_asset_node)],
    }

    return {
        "id":                           new_id(),
        "step":                         "if",
        "name":                         "Condition",
        "suppress_incomplete_warnings": False,
        "children":                     [if_child_true, if_child_else],
    }


def build_wt_cash_equal_wrapper(if_nodes):
    return {
        "id":                           new_id(),
        "step":                         "wt-cash-equal",
        "name":                         "Weight",
        "suppress_incomplete_warnings": False,
        "children":                     if_nodes,
    }


# ---------------------------------------------------------------------------
# Core grouping logic
# ---------------------------------------------------------------------------

def build_if_nodes_for_group(group_df, original_asset_node, ind_period):
    """
    For one (sub_strategy, conditions, endpoint) group, build the list of
    if nodes for the wt-cash-equal wrapper.

    Algorithm:
    1. For each (signal_asset, operator, target_asset) triple, find the most
       inclusive threshold
    2. Tautology check: if the same signal_asset has both '>' and '<' conditions
       whose thresholds overlap, skip that signal_asset entirely with a warning
    3. Group targets by (signal_asset, operator, most_inclusive_threshold)
    4. Each unique (signal_asset, operator, threshold) becomes one if node
    5. Sort: most targets first, then by most inclusive threshold
    """
    # Step 1: find most inclusive threshold per (signal_asset, operator, target)
    trio_threshold = {}
    for (sig, op, tgt), trio_df in group_df.groupby(
            ["signal_asset", "signal_operator", "target_asset"]):
        thresholds = trio_df["threshold"].tolist()
        trio_threshold[(sig, op, tgt)] = most_inclusive_threshold(thresholds, op)

    # Step 2: tautology check per signal_asset
    # Collect best (most inclusive) threshold per (signal_asset, operator)
    sig_op_best = defaultdict(list)
    for (sig, op, tgt), thresh in trio_threshold.items():
        sig_op_best[(sig, op)].append(thresh)

    # Find most inclusive per (sig, op) across all targets
    sig_op_inclusive = {}
    for (sig, op), thresholds in sig_op_best.items():
        sig_op_inclusive[(sig, op)] = most_inclusive_threshold(thresholds, op)

    # Check for tautologies
    tautology_signals = set()
    signals_with_gt = {sig for (sig, op) in sig_op_inclusive if op == ">"}
    signals_with_lt = {sig for (sig, op) in sig_op_inclusive if op == "<"}
    shared_signals  = signals_with_gt & signals_with_lt

    for sig in shared_signals:
        thresh_gt = sig_op_inclusive.get((sig, ">"))
        thresh_lt = sig_op_inclusive.get((sig, "<"))
        if thresh_gt is not None and thresh_lt is not None:
            if is_tautology(thresh_gt, thresh_lt):
                print(f"  [TAUTOLOGY] Skipping signal_asset '{sig}': "
                      f"RSI > {thresh_gt} OR RSI < {thresh_lt} is always true")
                tautology_signals.add(sig)

    # Step 3: group targets by (signal_asset, operator, threshold)
    # key: (signal_asset, operator, threshold) -> set of target_tickers
    key_targets = defaultdict(set)
    for (sig, op, tgt), thresh in trio_threshold.items():
        if sig in tautology_signals:
            continue
        key_targets[(sig, op, thresh)].add(tgt)

    if not key_targets:
        return []

    # Step 4: build one if node per (signal_asset, operator, threshold)
    if_nodes = []
    for (sig, op, thresh), targets in key_targets.items():
        target_list = sorted(
            targets,
            key=lambda t: group_df[group_df["target_asset"] == t]["Median_Return"].max(),
            reverse=True
        )
        if_node = build_if_node(
            signal_asset=sig,
            operator=op,
            threshold=thresh,
            ind_period=ind_period,
            target_tickers=target_list,
            original_asset_node=original_asset_node,
        )
        if_nodes.append((sig, op, thresh, len(targets), if_node))

    # Step 5: sort — most targets first, then most inclusive threshold
    def sort_key(item):
        sig, op, thresh, n_targets, _ = item
        # More targets = higher priority
        # Within same target count, most inclusive threshold first
        inclusive_score = -thresh if op in (">", ">=") else thresh
        return (-n_targets, inclusive_score)

    if_nodes.sort(key=sort_key)
    return [n for _, _, _, _, n in if_nodes]


# ---------------------------------------------------------------------------
# Node index builders
# ---------------------------------------------------------------------------

def build_node_index(tree):
    index = {}
    def _walk(node):
        nid = node.get("id")
        if nid:
            index[nid] = node
        for child in node.get("children", []):
            _walk(child)
    _walk(tree)
    return index


def build_parent_index(tree):
    parent_index = {}
    def _walk(node):
        for i, child in enumerate(node.get("children", [])):
            cid = child.get("id")
            if cid:
                parent_index[cid] = (node, i)
            _walk(child)
    _walk(tree)
    return parent_index


# ---------------------------------------------------------------------------
# Conditions string normaliser
# ---------------------------------------------------------------------------

def normalise_conditions(cond_str):
    return " ".join(cond_str.strip().upper().split())


# ---------------------------------------------------------------------------
# Path matcher
# ---------------------------------------------------------------------------

def build_path_to_nodeid_map(strategy_tree):
    paths  = extract_paths(strategy_tree)
    lookup = {}
    for p in paths:
        if not p.get("node_id") or p["node_id"] == "UNKNOWN":
            continue
        cond_str = " AND ".join(p["conditions"])
        key = (
            p["sub_strategy"].strip(),
            normalise_conditions(cond_str),
            p["endpoint"].upper(),
        )
        lookup[key] = p["node_id"]
    return lookup


# ---------------------------------------------------------------------------
# CSV loader and grouper
# ---------------------------------------------------------------------------

def load_and_group_candidates(csv_path):
    """
    Load filtered CSV (must have signal_operator column) and group by
    (sub_strategy, conditions, endpoint).

    Validates that signal_operator column exists and contains only
    recognised operators.
    """
    df = pd.read_csv(csv_path)

    if "signal_operator" not in df.columns:
        print(
            f"ERROR: CSV has no signal_operator column: {csv_path}\n"
            f"  Re-run run_analysis.py and filter_results.py to generate "
            f"a CSV with this column.",
            file=sys.stderr
        )
        sys.exit(1)

    df["endpoint"]        = df["endpoint"].str.upper()
    df["target_asset"]    = df["target_asset"].str.upper()
    df["signal_asset"]    = df["signal_asset"].str.upper()
    df["signal_operator"] = df["signal_operator"].str.strip()

    valid_ops = {">", "<", ">=", "<="}
    bad_ops = set(df["signal_operator"].unique()) - valid_ops
    if bad_ops:
        print(f"ERROR: Unknown signal_operator values: {bad_ops}", file=sys.stderr)
        sys.exit(1)

    specs = []
    for keys, group_df in df.groupby(["sub_strategy", "conditions", "endpoint"]):
        sub_strategy, conditions, endpoint = keys
        specs.append({
            "sub_strategy": sub_strategy,
            "conditions":   conditions,
            "endpoint":     endpoint,
            "group_df":     group_df.copy(),
        })
    return specs


# ---------------------------------------------------------------------------
# Main insertion logic
# ---------------------------------------------------------------------------

def insert_frontrunners(strategy_tree, csv_path, ind_period=10, dry_run=False):
    """
    Main pipeline. Deep-copies the tree, applies all insertions.
    Returns (modified_tree, log_list).
    """
    tree    = copy.deepcopy(strategy_tree)
    log     = []
    skipped = []

    # Always build path map from original canonical JSON for stable node IDs
    with open(STRATEGY_JSON, "r", encoding="utf-8") as f:
        original_tree = json.load(f)

    path_map     = build_path_to_nodeid_map(original_tree)
    parent_index = build_parent_index(tree)
    node_index   = build_node_index(tree)

    specs = load_and_group_candidates(csv_path)

    # Count operator breakdown for info
    all_df = pd.concat([s["group_df"] for s in specs], ignore_index=True)
    n_gt   = (all_df["signal_operator"] == ">").sum()
    n_lt   = (all_df["signal_operator"] == "<").sum()

    print(f"\n{'='*70}")
    print(f"  FRONTRUNNER INSERTION PIPELINE  (single pass)")
    print(f"  {len(specs)} unique insertion point(s)")
    print(f"  Signal rows: {n_gt} overbought ('>'),  {n_lt} oversold ('<')")
    print(f"  RSI period:  {ind_period}")
    print(f"{'='*70}\n")

    for spec in specs:
        sub   = spec["sub_strategy"]
        conds = spec["conditions"]
        endpt = spec["endpoint"]
        gdf   = spec["group_df"]

        lookup_key = (sub.strip(), normalise_conditions(conds), endpt.upper())
        node_id    = path_map.get(lookup_key)

        if not node_id:
            msg = f"[SKIP] No node_id match for ({sub} | {conds} | {endpt})"
            print(msg)
            skipped.append(msg)
            continue

        if node_id not in parent_index:
            msg = f"[SKIP] node_id {node_id} not in parent index ({endpt})"
            print(msg)
            skipped.append(msg)
            continue

        original_node          = node_index[node_id]
        parent_node, child_idx = parent_index[node_id]

        if_nodes = build_if_nodes_for_group(
            gdf, copy.deepcopy(original_node), ind_period
        )

        if not if_nodes:
            msg = (f"[SKIP] All signal assets were tautologies for "
                   f"({sub} | {endpt})")
            print(msg)
            skipped.append(msg)
            continue

        wrapper = build_wt_cash_equal_wrapper(if_nodes)

        unique_signals  = sorted(gdf["signal_asset"].unique().tolist())
        unique_targets  = sorted(gdf["target_asset"].unique().tolist())
        ops_present     = sorted(gdf["signal_operator"].unique().tolist())

        print(f"[INSERT] {sub} | {endpt}")
        print(f"         Conditions: {conds}")
        print(f"         Node ID:    {node_id}")
        print(f"         Targets:    {', '.join(unique_targets)}")
        print(f"         Signals:    {', '.join(unique_signals)}")
        print(f"         Operators:  {', '.join(ops_present)}")
        print(f"         If nodes:   {len(if_nodes)}")

        if not dry_run:
            parent_node["children"][child_idx] = wrapper

        log.append({
            "sub_strategy":  sub,
            "conditions":    conds,
            "endpoint":      endpt,
            "node_id":       node_id,
            "target_assets": unique_targets,
            "signal_assets": unique_signals,
            "operators":     ops_present,
            "if_node_count": len(if_nodes),
        })

    print(f"\n{'='*70}")
    print(f"  DONE: {len(log)} insertion(s) applied, {len(skipped)} skipped")
    if skipped:
        print(f"\n  Skipped paths:")
        for s in skipped:
            print(f"    {s}")
    print(f"{'='*70}\n")

    return tree, log


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Insert frontrunner logic into a Composer strategy JSON."
    )
    parser.add_argument(
        "filtered_csv",
        nargs="?",
        default=None,
        help=f"Path to filtered results CSV with signal_operator column. "
             f"Defaults to {DEFAULT_CSV}"
    )
    parser.add_argument(
        "--input", "-i", default=None,
        help="Input strategy JSON. Defaults to pathfinder/strategy.json."
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output strategy JSON. Defaults to strategy_inserter/strategy_modified.json."
    )
    parser.add_argument(
        "--period", "-p", type=int, default=10,
        help="RSI indicator period. Default: 10"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Match and log insertions without modifying the JSON."
    )
    args = parser.parse_args()

    csv_path    = Path(args.filtered_csv) if args.filtered_csv else DEFAULT_CSV
    input_json  = Path(args.input)  if args.input  else STRATEGY_JSON
    output_json = Path(args.output) if args.output else OUTPUT_JSON

    if not input_json.exists():
        print(f"ERROR: Input JSON not found at {input_json}", file=sys.stderr)
        sys.exit(1)
    if not csv_path.exists():
        print(f"ERROR: CSV not found at {csv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Input JSON:  {input_json}")
    print(f"Input CSV:   {csv_path}")
    print(f"Output JSON: {output_json}")

    with open(input_json, "r", encoding="utf-8") as f:
        strategy_tree = json.load(f)

    modified_tree, log = insert_frontrunners(
        strategy_tree,
        csv_path,
        ind_period=args.period,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("[DRY RUN] No output files written.")
        return

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(modified_tree, f, indent=4)
    print(f"Modified strategy written to: {output_json}")

    if log:
        log_path = output_json.parent / "insertion_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)
        print(f"Insertion log written to:     {log_path}")


if __name__ == "__main__":
    main()