"""
strategy_inserter.py
====================
Takes a Composer strategy JSON and a filtered backtest results CSV,
and inserts frontrunner logic at the correct leaf nodes in the tree.

For each unique (sub_strategy, conditions, endpoint) group in the CSV,
the matching leaf asset node is wrapped in a new if/else block:

  if ANY(all thresholds across all signal assets across all targets)
    -> [target_1, target_2, target_3, ...]  (equal weight, wt-cash-equal parent)
  else
    -> original endpoint asset (unchanged)

All signal conditions are pooled into a single `any` compound block
regardless of threshold count. Targets are listed as equal-weight siblings.

Canonical input (strategy):  pathfinder/strategy.json
Canonical output:             strategy_inserter/strategy_modified.json
                              strategy_inserter/insertion_log.json

Usage:
    # Dry run — match and log without modifying JSON
    python strategy_inserter/strategy_inserter.py --dry-run

    # Run with default filtered CSV
    python strategy_inserter/strategy_inserter.py strategy_engine/results/filtered_overbought.csv

    # Run with any CSV
    python strategy_inserter/strategy_inserter.py path/to/my_filtered.csv

    # Run both overbought and oversold CSVs
    python strategy_inserter/strategy_inserter.py strategy_engine/results/filtered_overbought.csv
    python strategy_inserter/strategy_inserter.py strategy_engine/results/filtered_oversold.csv
"""

import json
import sys
import uuid
import copy
import argparse
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Canonical paths — resolved relative to this script's location
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
# Engine condition string parser
# ---------------------------------------------------------------------------

def _parse_engine_condition(engine_cond_str):
    """
    Parse an engine condition string back into its components.

    Handles:
      ticker_FN_period OP value            e.g. "SPY_RSI_10 > 79"
      ticker_FN_period OP ticker2_FN_period e.g. "TLT_RSI_20 > PSQ_RSI_20"
      ticker_close OP ticker2_SMA_200      e.g. "SPY_close > SPY_SMA_200"

    Returns a dict with keys:
      lhs_ticker, lhs_fn, lhs_window,
      op,
      rhs_fixed, rhs_value, rhs_ticker, rhs_fn, rhs_window
    """
    op = None
    for candidate in (">=", "<=", "!=", ">", "<", "=="):
        if f" {candidate} " in engine_cond_str:
            op = candidate
            break
    if op is None:
        raise ValueError(f"Cannot parse operator from: {engine_cond_str!r}")

    lhs_str, rhs_str = [s.strip() for s in engine_cond_str.split(f" {op} ", 1)]

    def _parse_side(s):
        parts = s.split("_")
        try:
            int(parts[-1])
            window = parts[-1]
            fn_key = parts[-2].upper()
            ticker = "_".join(parts[:-2])
            return ticker, fn_key, window
        except (ValueError, IndexError):
            pass
        if parts[-1].lower() == "close":
            ticker = "_".join(parts[:-1])
            return ticker, "Price", None
        return None, None, None

    lhs_ticker, lhs_fn, lhs_window = _parse_side(lhs_str)

    try:
        rhs_value = float(rhs_str)
        return {
            "lhs_ticker": lhs_ticker, "lhs_fn": lhs_fn, "lhs_window": lhs_window,
            "op": op,
            "rhs_fixed": True, "rhs_value": rhs_value,
            "rhs_ticker": None, "rhs_fn": None, "rhs_window": None,
        }
    except ValueError:
        pass

    rhs_ticker, rhs_fn, rhs_window = _parse_side(rhs_str)
    return {
        "lhs_ticker": lhs_ticker, "lhs_fn": lhs_fn, "lhs_window": lhs_window,
        "op": op,
        "rhs_fixed": False, "rhs_value": None,
        "rhs_ticker": rhs_ticker, "rhs_fn": rhs_fn, "rhs_window": rhs_window,
    }


# ---------------------------------------------------------------------------
# Composer node builders
# ---------------------------------------------------------------------------

def _build_binary_compound(parsed):
    """
    Build a Composer binary-compound condition node from a parsed engine condition.
    Used inside compound any condition blocks.
    """
    lhs_fn_raw = FN_MAP.get(parsed["lhs_fn"], parsed["lhs_fn"])
    comp       = COMPARATOR_MAP.get(parsed["op"], "gt")

    lhs_node = {"fn": lhs_fn_raw, "ticker": "%"}
    if parsed["lhs_window"]:
        lhs_node["params"] = {"window": int(parsed["lhs_window"])}
    else:
        lhs_node["params"] = {}

    node = {
        "condition-type": "binary-compound",
        "operator":       "any",
        "tickers":        [parsed["lhs_ticker"]],
        "lhs":            lhs_node,
        "comparator":     comp,
    }

    if parsed["rhs_fixed"]:
        node["rhs"] = {"constant": parsed["rhs_value"]}
    else:
        rhs_fn_raw = FN_MAP.get(parsed["rhs_fn"], parsed["rhs_fn"])
        rhs_node   = {"fn": rhs_fn_raw, "ticker": parsed["rhs_ticker"]}
        if parsed["rhs_window"]:
            rhs_node["params"] = {"window": int(parsed["rhs_window"])}
        else:
            rhs_node["params"] = {}
        node["rhs"] = rhs_node

    return node


def build_asset_node(ticker):
    """Build a minimal Composer asset node."""
    return {
        "id":     new_id(),
        "step":   "asset",
        "ticker": ticker,
    }


def build_frontrunner_wrapper(group_rows, target_tickers, original_asset_node, ind_period=10):
    """
    Build a single if/else wrapper node for one insertion point.

    Structure:
      if ANY(all thresholds across all signal assets across all targets)
        -> [target_1, target_2, ...]  (equal weight siblings in children)
      else
        -> original_asset_node (unchanged)

    Parameters
    ----------
    group_rows       : DataFrame — all CSV rows for this (sub_strategy, conditions, endpoint)
    target_tickers   : list of str — all unique target assets for this group, 
                       sorted by best Median_Return descending
    original_asset_node : dict — the original Composer asset node being wrapped
    ind_period       : int — RSI period used in signal column names
    """
    # Build one binary-compound condition per row (signal_asset + threshold)
    conditions = []
    for _, row in group_rows.iterrows():
        cond_str = f"{row['signal_asset']}_RSI_{ind_period} > {row['threshold']}"
        parsed   = _parse_engine_condition(cond_str)
        conditions.append(_build_binary_compound(parsed))

    # True branch: all target assets as equal-weight siblings
    target_asset_nodes = [build_asset_node(t) for t in target_tickers]

    if_child_true = {
        "id":                 new_id(),
        "step":               "if-child",
        "name":               "If",
        "is-else-condition?": False,
        "suppress_incomplete_warnings": False,
        "condition": {
            "condition-type": "compound",
            "operator":       "any",
            "conditions":     conditions,
        },
        "children": target_asset_nodes,
    }

    # Else branch: original asset node, untouched
    if_child_else = {
        "id":                           new_id(),
        "step":                         "if-child",
        "name":                         "Else",
        "is-else-condition?":           True,
        "suppress_incomplete_warnings": True,
        "children":                     [original_asset_node],
    }

    return {
        "id":       new_id(),
        "step":     "if",
        "name":     "Condition",
        "suppress_incomplete_warnings": False,
        "children": [if_child_true, if_child_else],
    }


# ---------------------------------------------------------------------------
# Node index builders
# ---------------------------------------------------------------------------

def build_node_index(tree):
    """Flat dict: {node_id: node_dict} — references into the live tree."""
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
    """Normalise for matching: strip, collapse spaces, uppercase AND."""
    return " ".join(cond_str.strip().upper().split())


# ---------------------------------------------------------------------------
# Path matcher
# ---------------------------------------------------------------------------

def build_path_to_nodeid_map(strategy_tree):
    """
    Build lookup: (sub_strategy, normalised_conditions, endpoint) -> node_id
    """
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
    Load the filtered CSV and group by (sub_strategy, conditions, endpoint).

    All target assets and all signal thresholds are pooled per group.

    Returns a list of insertion specs:
    {
        "sub_strategy":   str,
        "conditions":     str,
        "endpoint":       str,
        "target_tickers": [str, ...],   # unique targets, sorted by best Median_Return
        "all_rows":       DataFrame,    # all rows for this group (for condition building)
        "total_thresholds": int,
        "signal_assets":  [str, ...],   # unique signal assets in this group
    }
    """
    df = pd.read_csv(csv_path)

    df["endpoint"]     = df["endpoint"].str.upper()
    df["target_asset"] = df["target_asset"].str.upper()
    df["signal_asset"] = df["signal_asset"].str.upper()

    specs      = []
    group_keys = ["sub_strategy", "conditions", "endpoint"]

    for keys, group_df in df.groupby(group_keys):
        sub_strategy, conditions, endpoint = keys

        # Rank unique target assets by their best Median_Return across all rows
        target_best = (
            group_df.groupby("target_asset")["Median_Return"]
            .max()
            .sort_values(ascending=False)
        )
        target_tickers = list(target_best.index)

        signal_assets = sorted(group_df["signal_asset"].unique().tolist())

        specs.append({
            "sub_strategy":     sub_strategy,
            "conditions":       conditions,
            "endpoint":         endpoint,
            "target_tickers":   target_tickers,
            "all_rows":         group_df,
            "total_thresholds": len(group_df),
            "signal_assets":    signal_assets,
        })

    return specs


# ---------------------------------------------------------------------------
# Main insertion logic
# ---------------------------------------------------------------------------

def insert_frontrunners(strategy_tree, csv_path, ind_period=10, dry_run=False):
    """
    Main pipeline. Deep-copies the tree, applies all insertions, returns result.
    Returns (modified_tree, log_list).
    """
    tree    = copy.deepcopy(strategy_tree)
    log     = []
    skipped = []

    path_map     = build_path_to_nodeid_map(tree)
    parent_index = build_parent_index(tree)
    node_index   = build_node_index(tree)

    specs = load_and_group_candidates(csv_path)

    print(f"\n{'='*70}")
    print(f"  FRONTRUNNER INSERTION PIPELINE")
    print(f"  {len(specs)} unique insertion point(s) found in CSV")
    print(f"{'='*70}\n")

    for spec in specs:
        sub     = spec["sub_strategy"]
        conds   = spec["conditions"]
        endpt   = spec["endpoint"]
        targets = spec["target_tickers"]
        n_thresh = spec["total_thresholds"]
        signals  = spec["signal_assets"]

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

        print(f"[INSERT] {sub} | {endpt} -> [{', '.join(targets)}]")
        print(f"         Conditions:  {conds}")
        print(f"         Node ID:     {node_id}")
        print(f"         Signals:     {', '.join(signals)}")
        print(f"         Thresholds:  {n_thresh} total | operator: any")

        wrapper = build_frontrunner_wrapper(
            spec["all_rows"],
            targets,
            copy.deepcopy(original_node),
            ind_period=ind_period,
        )

        if not dry_run:
            parent_node["children"][child_idx] = wrapper

        log.append({
            "sub_strategy":     sub,
            "conditions":       conds,
            "endpoint":         endpt,
            "target_assets":    targets,
            "signal_assets":    signals,
            "node_id":          node_id,
            "total_thresholds": n_thresh,
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
        "--dry-run", action="store_true",
        help="Match and log insertions without modifying the JSON."
    )
    args = parser.parse_args()

    # Resolve CSV path
    if args.filtered_csv:
        csv_path = Path(args.filtered_csv)
    else:
        csv_path = _ROOT / "strategy_engine" / "results" / "filtered_overbought.csv"

    # Validate inputs
    if not STRATEGY_JSON.exists():
        print(f"ERROR: Strategy JSON not found at {STRATEGY_JSON}", file=sys.stderr)
        sys.exit(1)
    if not csv_path.exists():
        print(f"ERROR: CSV not found at {csv_path}", file=sys.stderr)
        sys.exit(1)

    with open(STRATEGY_JSON, "r", encoding="utf-8") as f:
        strategy_tree = json.load(f)

    modified_tree, log = insert_frontrunners(
        strategy_tree, csv_path, dry_run=args.dry_run
    )

    if args.dry_run:
        print("[DRY RUN] No output files written.")
        return

    # Write modified JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(modified_tree, f, indent=4)
    print(f"Modified strategy written to: {OUTPUT_JSON}")

    # Write insertion log
    if log:
        with open(OUTPUT_LOG, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)
        print(f"Insertion log written to:     {OUTPUT_LOG}")


if __name__ == "__main__":
    main()