"""
strategy_paths.py
=================
Walks a Composer/VOXPORT strategy JSON tree and emits every possible
boolean path that leads to a leaf endpoint (asset or filter pick).

Each path is stored with two parallel condition representations:
  - conditions       : human-readable list  e.g. "RSI(TLT, 20) > RSI(PSQ, 20)"
  - engine_conds     : engine-compatible list e.g. "TLT_RSI_20 > PSQ_RSI_20"

The engine_precondition field on each result is the engine_conds joined
with ' and ', ready to drop directly into a YAML preconditions field.

The node_id field records the JSON node ID of the leaf asset node,
used by strategy_inserter.py to locate the exact insertion point.

Canonical input:  pathfinder/strategy.json  (relative to project root)
Canonical output: pathfinder/paths.txt      (written automatically + stdout)

Usage:
    python pathfinder/strategy_paths.py
"""

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Canonical paths — resolved relative to this script's location (pathfinder/)
# ---------------------------------------------------------------------------

_HERE         = Path(__file__).resolve().parent          # pathfinder/
STRATEGY_JSON = _HERE / "strategy.json"
PATHS_TXT     = _HERE / "paths.txt"

# ---------------------------------------------------------------------------
# Formatting helpers — human-readable
# ---------------------------------------------------------------------------

FN_LABELS = {
    "relative-strength-index": "RSI",
    "cumulative-return":        "CumRet",
    "moving-average-price":     "MA",
    "current-price":            "Price",
}

COMPARATOR_LABELS = {
    "gt":  ">",
    "lt":  "<",
    "gte": ">=",
    "lte": "<=",
    "eq":  "==",
    "neq": "!=",
}

COMPARATOR_NEGATIONS = {
    "gt":  "<=",
    "lt":  ">=",
    "gte": "<",
    "lte": ">",
    "eq":  "!=",
    "neq": "==",
}

# Engine column name for each function (current-price has no period)
FN_ENGINE = {
    "relative-strength-index": "RSI",
    "cumulative-return":        "CumRet",
    "moving-average-price":     "SMA",
    "current-price":            None,   # special case: {TICKER}_close
}


def _get_window(node, prefix):
    params = node.get(f"{prefix}-fn-params", {})
    if "window" in params:
        return str(params["window"])
    wd = node.get(f"{prefix}-window-days")
    if wd is not None:
        return str(wd)
    return None


# ---------------------------------------------------------------------------
# Human-readable side formatters
# ---------------------------------------------------------------------------

def _format_side(node, prefix):
    if prefix == "rhs" and node.get("rhs-fixed-value?"):
        return str(node.get("rhs-val", "?"))
    fn_raw = node.get(f"{prefix}-fn", "")
    fn     = FN_LABELS.get(fn_raw, fn_raw)
    ticker = node.get(f"{prefix}-val", "?")
    window = _get_window(node, prefix)
    return f"{fn}({ticker}, {window})" if window else f"{fn}({ticker})"


def _format_condition(if_child):
    lhs  = _format_side(if_child, "lhs")
    comp = COMPARATOR_LABELS.get(if_child.get("comparator", "?"), "?")
    rhs  = _format_side(if_child, "rhs")
    return f"{lhs} {comp} {rhs}"


def _format_negated_condition(if_child):
    lhs  = _format_side(if_child, "lhs")
    comp = COMPARATOR_NEGATIONS.get(if_child.get("comparator", "?"), "?")
    rhs  = _format_side(if_child, "rhs")
    return f"{lhs} {comp} {rhs}"


# ---------------------------------------------------------------------------
# Engine-compatible side formatters
# ---------------------------------------------------------------------------

def _engine_side(node, prefix):
    """
    Format one side of a condition as an engine DataFrame column name.
      RSI(TLT, 20)    ->  TLT_RSI_20
      MA(SPY, 200)    ->  SPY_SMA_200
      CumRet(BND, 60) ->  BND_CumRet_60
      Price(SPY)      ->  SPY_close
    Fixed values (rhs only) are returned as-is.
    """
    if prefix == "rhs" and node.get("rhs-fixed-value?"):
        return str(node.get("rhs-val", "?"))

    fn_raw  = node.get(f"{prefix}-fn", "")
    ticker  = node.get(f"{prefix}-val", "?")
    window  = _get_window(node, prefix)
    eng_fn  = FN_ENGINE.get(fn_raw)

    if eng_fn is None:
        return f"{ticker}_close"
    else:
        return f"{ticker}_{eng_fn}_{window}"


def _engine_condition(if_child):
    lhs  = _engine_side(if_child, "lhs")
    comp = COMPARATOR_LABELS.get(if_child.get("comparator", "?"), "?")
    rhs  = _engine_side(if_child, "rhs")
    return f"{lhs} {comp} {rhs}"


def _engine_negated_condition(if_child):
    lhs  = _engine_side(if_child, "lhs")
    comp = COMPARATOR_NEGATIONS.get(if_child.get("comparator", "?"), "?")
    rhs  = _engine_side(if_child, "rhs")
    return f"{lhs} {comp} {rhs}"


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

def _filter_sort_fn(node):
    fn_raw   = node.get("sort-by-fn", "")
    sort_win = node.get("sort-by-window-days") or (
                   node.get("sort-by-fn-params") or {}
               ).get("window")
    return fn_raw, str(sort_win) if sort_win else None


def _is_portfolio_filter(node):
    children = node.get("children", [])
    return children and all(c.get("step") == "group" for c in children)


def _expand_filter(node, conditions, engine_conds, sub_strategy, results):
    """
    Expand a top-1 asset-selection filter into one branch per child asset.
    Each branch gets a pairwise condition: winner_RSI > loser_RSI.
    Raises NotImplementedError for select-n > 1 or non-2-asset filters.
    """
    select_n = int(node.get("select-n", 1))
    if select_n > 1:
        raise NotImplementedError(
            f"Filter with select-n={select_n} is not supported. "
            f"Only top-1 filters are currently handled."
        )

    fn_raw, window = _filter_sort_fn(node)
    fn_human  = FN_LABELS.get(fn_raw, fn_raw)
    fn_engine = FN_ENGINE.get(fn_raw, fn_raw)

    asset_children = [c for c in node.get("children", []) if c.get("step") == "asset"]

    if len(asset_children) != 2:
        raise NotImplementedError(
            f"Top-1 filter with {len(asset_children)} assets is not supported. "
            f"Only 2-asset top-1 filters are currently handled."
        )

    for i, child in enumerate(asset_children):
        winner = child.get("ticker", "?")
        loser  = asset_children[1 - i].get("ticker", "?")

        if window:
            human_cond  = f"{fn_human}({winner}, {window}) > {fn_human}({loser}, {window})"
            engine_cond = f"{winner}_{fn_engine}_{window} > {loser}_{fn_engine}_{window}"
        else:
            human_cond  = f"{fn_human}({winner}) > {fn_human}({loser})"
            engine_cond = f"{winner}_{fn_engine} > {loser}_{fn_engine}"

        results.append({
            "sub_strategy":        sub_strategy or "(root)",
            "conditions":          list(conditions) + [human_cond],
            "engine_conds":        list(engine_conds) + [engine_cond],
            "engine_precondition": " and ".join(list(engine_conds) + [engine_cond]),
            "endpoint":            winner,
            "node_id":             child.get("id", "UNKNOWN"),
        })


# ---------------------------------------------------------------------------
# Core recursive walker
# ---------------------------------------------------------------------------

def walk(node, conditions, engine_conds, sub_strategy, results):
    step = node.get("step")

    if step in ("root", "wt-cash-equal", "wt-cash-specified"):
        for child in node.get("children", []):
            walk(child, conditions, engine_conds, sub_strategy, results)
        return

    if step == "group":
        name    = node.get("name")
        new_sub = name if (name and sub_strategy is None) else sub_strategy
        for child in node.get("children", []):
            walk(child, conditions, engine_conds, new_sub, results)
        return

    if step == "asset":
        results.append({
            "sub_strategy":        sub_strategy or "(root)",
            "conditions":          list(conditions),
            "engine_conds":        list(engine_conds),
            "engine_precondition": " and ".join(engine_conds),
            "endpoint":            node.get("ticker", "UNKNOWN"),
            "node_id":             node.get("id", "UNKNOWN"),
        })
        return

    if step == "filter":
        if _is_portfolio_filter(node):
            for child in node.get("children", []):
                walk(child, conditions, engine_conds, sub_strategy, results)
        else:
            _expand_filter(node, conditions, engine_conds, sub_strategy, results)
        return

    if step == "if":
        if_children       = node.get("children", [])
        positive_branches = [c for c in if_children if not c.get("is-else-condition?")]
        else_branches     = [c for c in if_children if     c.get("is-else-condition?")]

        for pos in positive_branches:
            human_cond = _format_condition(pos)
            eng_cond   = _engine_condition(pos)
            new_conds  = conditions   + [human_cond]
            new_eng    = engine_conds + [eng_cond]
            for grandchild in pos.get("children", []):
                walk(grandchild, new_conds, new_eng, sub_strategy, results)

        if else_branches:
            pos_node   = positive_branches[0] if positive_branches else None
            human_neg  = _format_negated_condition(pos_node) if pos_node else "ELSE"
            eng_neg    = _engine_negated_condition(pos_node) if pos_node else "ELSE"
            new_conds  = conditions   + [human_neg]
            new_eng    = engine_conds + [eng_neg]
            for els in else_branches:
                for grandchild in els.get("children", []):
                    walk(grandchild, new_conds, new_eng, sub_strategy, results)
        else:
            results.append({
                "sub_strategy":        sub_strategy or "(root)",
                "conditions":          list(conditions) + ["[condition false, no else]"],
                "engine_conds":        list(engine_conds) + ["[condition false, no else]"],
                "engine_precondition": "",
                "endpoint":            "UNALLOCATED",
                "node_id":             None,
            })
        return

    if step == "if-child":
        for child in node.get("children", []):
            walk(child, conditions, engine_conds, sub_strategy, results)
        return

    print(f"  [WARN] Unknown step type: '{step}' (id={node.get('id', '?')})",
          file=sys.stderr)
    for child in node.get("children", []):
        walk(child, conditions, engine_conds, sub_strategy, results)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_path(path):
    sub   = path["sub_strategy"]
    conds = path["conditions"]
    end   = path["endpoint"]
    nid   = path.get("node_id", "?")
    if conds:
        return f"[{sub}] IF {' AND '.join(conds)} -> {end}  (node_id: {nid})"
    else:
        return f"[{sub}] (no conditions) -> {end}  (node_id: {nid})"


# ---------------------------------------------------------------------------
# Public API — used by run_analysis.py and strategy_inserter.py
# ---------------------------------------------------------------------------

def extract_paths(strategy_json):
    """
    Walk a loaded strategy JSON dict and return the full list of path dicts.
    Each dict contains: sub_strategy, conditions, engine_conds,
    engine_precondition, endpoint, node_id.
    """
    results = []
    walk(strategy_json, conditions=[], engine_conds=[], sub_strategy=None, results=results)
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if not STRATEGY_JSON.exists():
        print(f"ERROR: Strategy JSON not found at {STRATEGY_JSON}", file=sys.stderr)
        sys.exit(1)

    with open(STRATEGY_JSON, "r", encoding="utf-8") as f:
        tree = json.load(f)

    results = extract_paths(tree)

    # Build output lines
    lines = []
    results_sorted = sorted(results, key=lambda r: r["endpoint"])
    current_end = None
    for path in results_sorted:
        if path["endpoint"] != current_end:
            current_end = path["endpoint"]
            lines.append(f"\n{'='*80}")
            lines.append(f"  ENDPOINT: {current_end}")
            lines.append(f"{'='*80}")
        lines.append(format_path(path))

    lines.append(f"\n{'='*80}")
    lines.append(f"  TOTAL PATHS: {len(results)}")
    lines.append(f"{'='*80}\n")

    output = "\n".join(lines)

    # Print to stdout
    print(output)

    # Write to paths.txt alongside this script
    with open(PATHS_TXT, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"\n[Written to {PATHS_TXT}]", file=sys.stderr)


if __name__ == "__main__":
    main()