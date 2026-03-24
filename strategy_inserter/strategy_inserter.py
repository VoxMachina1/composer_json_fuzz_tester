"""
strategy_inserter.py
====================
Takes a Composer strategy JSON and a filtered backtest results CSV,
and inserts frontrunner logic at the correct leaf nodes in the tree.

For each unique (sub_strategy, conditions, endpoint) group in the CSV,
the matching leaf asset node is replaced with a wt-cash-equal block
containing one if node per unique (signal_asset, most_inclusive_threshold) pair:

  wt-cash-equal
    IF signal_A_RSI > threshold_1  ->  [target_1, target_2]  (same threshold)
      ELSE -> original_endpoint
    IF signal_A_RSI > threshold_2  ->  [target_3]            (different threshold)
      ELSE -> original_endpoint
    IF signal_B_RSI > threshold_3  ->  [target_1, target_3]  (different signal)
      ELSE -> original_endpoint

Threshold deduplication (most inclusive per signal_asset per target):
  - '>' operator: keep minimum threshold (fires most often)
  - '<' operator: keep maximum threshold (fires most often)

Targets sharing the exact same (signal_asset, most_inclusive_threshold)
are grouped as equal-weight siblings in one if node's true branch.
Targets with different thresholds get separate if nodes.

Canonical input (strategy):  pathfinder/strategy.json
Canonical output:             strategy_inserter/strategy_modified.json
                              strategy_inserter/insertion_log.json

Usage:
    python strategy_inserter/strategy_inserter.py --dry-run
    python strategy_inserter/strategy_inserter.py strategy_engine/results/filtered_overbought.csv
    python strategy_inserter/strategy_inserter.py strategy_engine/results/filtered_oversold.csv
    python strategy_inserter/strategy_inserter.py path/to/any_filtered.csv
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
    Return the single most inclusive threshold for a given operator.
      '>' or '>=' : minimum threshold (fires most often)
      '<' or '<=' : maximum threshold (fires most often)
    """
    if operator in (">", ">="):
        return min(thresholds)
    elif operator in ("<", "<="):
        return max(thresholds)
    else:
        # For == or != just take the first
        return thresholds[0]


# ---------------------------------------------------------------------------
# Composer node builders
# ---------------------------------------------------------------------------

def build_asset_node(ticker):
    """Build a minimal Composer asset node."""
    return {
        "id":                           new_id(),
        "step":                         "asset",
        "name":                         ticker,
        "suppress_incomplete_warnings": False,
        "ticker":                       ticker,
    }


def build_if_node(signal_asset, operator, threshold, ind_period,
                  target_tickers, original_asset_node):
    """
    Build a single flat-style if/else node:

      IF signal_asset_RSI_period OP threshold -> [target_1, target_2, ...]
      ELSE -> original_asset_node

    Uses flat lhs-fn / rhs-fn style (no compound block) since it's
    always a single condition.
    """
    lhs_fn_raw = FN_MAP.get("RSI", "relative-strength-index")
    comp       = COMPARATOR_MAP.get(operator, "gt")

    # Format threshold: int if whole number, float otherwise
    thresh_str = str(int(threshold)) if threshold == int(threshold) else str(threshold)

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
    """
    Wrap a list of if nodes in a wt-cash-equal node.
    """
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

def build_if_nodes_for_group(group_df, original_asset_node, operator, ind_period):
    """
    For one (sub_strategy, conditions, endpoint) group, build the list of
    if nodes to populate the wt-cash-equal wrapper.

    Algorithm:
    1. For each (signal_asset, target_asset) pair, find the most inclusive threshold
    2. Group targets by (signal_asset, most_inclusive_threshold)
    3. Each unique (signal_asset, threshold) pair becomes one if node
    4. Targets sharing the same (signal_asset, threshold) are siblings in that node
    5. Sort if nodes: most targets first, then by threshold (most inclusive first)
    """
    # Step 1: find most inclusive threshold per (signal_asset, target_asset)
    pair_threshold = {}
    for (sig, tgt), pair_df in group_df.groupby(["signal_asset", "target_asset"]):
        thresholds = pair_df["threshold"].tolist()
        pair_threshold[(sig, tgt)] = most_inclusive_threshold(thresholds, operator)

    # Step 2: group targets by (signal_asset, threshold)
    # key: (signal_asset, threshold) -> set of target_tickers
    sig_thresh_targets = defaultdict(set)
    for (sig, tgt), thresh in pair_threshold.items():
        sig_thresh_targets[(sig, thresh)].add(tgt)

    # Step 3: build one if node per (signal_asset, threshold) pair
    if_nodes = []
    for (sig, thresh), targets in sig_thresh_targets.items():
        # Sort targets by best Median_Return descending for consistent ordering
        target_list = sorted(
            targets,
            key=lambda t: group_df[group_df["target_asset"] == t]["Median_Return"].max(),
            reverse=True
        )
        if_node = build_if_node(
            signal_asset=sig,
            operator=operator,
            threshold=thresh,
            ind_period=ind_period,
            target_tickers=target_list,
            original_asset_node=original_asset_node,
        )
        if_nodes.append((sig, thresh, len(targets), if_node))

    # Step 4: sort if nodes — most targets first, then by most inclusive threshold
    if operator in (">", ">="):
        if_nodes.sort(key=lambda x: (-x[2], x[1]))   # most targets, then lowest thresh
    else:
        if_nodes.sort(key=lambda x: (-x[2], -x[1]))  # most targets, then highest thresh

    return [n for _, _, _, n in if_nodes]


# ---------------------------------------------------------------------------
# Node index builders
# ---------------------------------------------------------------------------

def build_node_index(tree):
    """Flat dict: {node_id: node_dict} — live references into tree."""
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
    """Flat dict: {child_node_id: (parent_node, child_index)}"""
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
    Load filtered CSV and group by (sub_strategy, conditions, endpoint).

    Returns list of specs:
    {
        "sub_strategy": str,
        "conditions":   str,
        "endpoint":     str,
        "group_df":     DataFrame,   # all rows for this group
    }
    """
    df = pd.read_csv(csv_path)

    df["endpoint"]     = df["endpoint"].str.upper()
    df["target_asset"] = df["target_asset"].str.upper()
    df["signal_asset"] = df["signal_asset"].str.upper()

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

def insert_frontrunners(strategy_tree, csv_path, operator=">",
                        ind_period=10, dry_run=False, index_tree=None):
    """
    Main pipeline. Deep-copies the tree, applies all insertions.
    Returns (modified_tree, log_list).

    index_tree: optional separate tree to build the path->node_id map from.
                Use this when strategy_tree is a modified version (pass 2+)
                so the path map is built from the original stable node IDs.
                If None, uses strategy_tree for both indexing and insertion.
    """
    tree    = copy.deepcopy(strategy_tree)
    log     = []
    skipped = []

    # Build path map from original tree if provided, otherwise from working tree
    map_tree     = index_tree if index_tree is not None else tree
    path_map     = build_path_to_nodeid_map(map_tree)
    parent_index = build_parent_index(tree)
    node_index   = build_node_index(tree)

    specs = load_and_group_candidates(csv_path)

    print(f"\n{'='*70}")
    print(f"  FRONTRUNNER INSERTION PIPELINE")
    print(f"  {len(specs)} unique insertion point(s) found in CSV")
    print(f"  Operator: {operator} | RSI period: {ind_period}")
    print(f"{'='*70}\n")

    for spec in specs:
        sub     = spec["sub_strategy"]
        conds   = spec["conditions"]
        endpt   = spec["endpoint"]
        gdf     = spec["group_df"]

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

        # Build the if nodes for this group
        if_nodes = build_if_nodes_for_group(
            gdf, copy.deepcopy(original_node), operator, ind_period
        )

        # Wrap in wt-cash-equal
        wrapper = build_wt_cash_equal_wrapper(if_nodes)

        # Summarise for logging
        unique_signals  = sorted(gdf["signal_asset"].unique().tolist())
        unique_targets  = sorted(gdf["target_asset"].unique().tolist())
        total_if_nodes  = len(if_nodes)

        print(f"[INSERT] {sub} | {endpt}")
        print(f"         Conditions: {conds}")
        print(f"         Node ID:    {node_id}")
        print(f"         Targets:    {', '.join(unique_targets)}")
        print(f"         Signals:    {', '.join(unique_signals)}")
        print(f"         If nodes:   {total_if_nodes}")

        if not dry_run:
            parent_node["children"][child_idx] = wrapper

        log.append({
            "sub_strategy":  sub,
            "conditions":    conds,
            "endpoint":      endpt,
            "node_id":       node_id,
            "target_assets": unique_targets,
            "signal_assets": unique_signals,
            "if_node_count": total_if_nodes,
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
        help="Path to filtered results CSV. "
             "Defaults to strategy_engine/results/filtered_overbought.csv"
    )
    parser.add_argument(
        "--input", "-i", default=None,
        help="Path to input strategy JSON. "
             "Defaults to pathfinder/strategy.json. "
             "Use strategy_inserter/strategy_modified.json to chain passes."
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Path to output strategy JSON. "
             "Defaults to strategy_inserter/strategy_modified.json."
    )
    parser.add_argument(
        "--log", "-l", default=None,
        help="Path to output insertion log JSON. "
             "Defaults to strategy_inserter/insertion_log.json."
    )
    parser.add_argument(
        "--operator", "-op", default=">",
        choices=[">", "<", ">=", "<="],
        help="RSI signal operator. Use '>' for overbought CSV, '<' for oversold CSV. "
             "Default: '>'"
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

    # Resolve paths
    csv_path    = Path(args.filtered_csv) if args.filtered_csv \
                  else _ROOT / "strategy_engine" / "results" / "filtered_overbought.csv"
    input_json  = Path(args.input)  if args.input  else STRATEGY_JSON
    output_json = Path(args.output) if args.output else OUTPUT_JSON
    log_path    = Path(args.log)    if args.log    else OUTPUT_LOG

    # Validate inputs
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

    # Always build the path->node_id index from the canonical original JSON
    # so node IDs are stable across chained passes (pass 1 and pass 2+)
    if not STRATEGY_JSON.exists():
        print(f"ERROR: Canonical strategy JSON not found at {STRATEGY_JSON}", file=sys.stderr)
        sys.exit(1)
    with open(STRATEGY_JSON, "r", encoding="utf-8") as f:
        original_tree = json.load(f)

    modified_tree, log = insert_frontrunners(
        strategy_tree,
        csv_path,
        operator=args.operator,
        ind_period=args.period,
        dry_run=args.dry_run,
        index_tree=original_tree,
    )

    if args.dry_run:
        print("[DRY RUN] No output files written.")
        return

    # Write modified JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(modified_tree, f, indent=4)
    print(f"Modified strategy written to: {output_json}")

    # Write insertion log
    if log:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)
        print(f"Insertion log written to:     {log_path}")


if __name__ == "__main__":
    main()