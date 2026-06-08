"""
fuzz_tester.py
==============
Reads a Composer strategy JSON, extracts every IF condition,
and runs a 2D parameter sweep (period × threshold/window) for each one.

For each parameter combination, measures win rate of the endpoint asset
vs BIL on days the condition fires. Outputs a self-contained HTML report
with heatmaps and a logic-tree summary ranked by condition fragility.

Usage:
    python fuzz_tester.py                              # interactive prompts
    python fuzz_tester.py pathfinder/bestsignals3.json # non-interactive, all defaults
"""

import json
import sys
import math
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Locate project root and import shared modules
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
STRATEGY_ENGINE_DIR = SCRIPT_DIR / "strategy_engine" / "src"
sys.path.insert(0, str(STRATEGY_ENGINE_DIR))

from data_loader import check_freshness_and_update
from data_alignment import load_ticker_csv
from indicators import calculate_rsi, calculate_sma, calculate_ema, calculate_cumret
from config_loader import load_config

DATA_DIR = SCRIPT_DIR / "strategy_engine" / "data"


def calculate_maxdd(series, period):
    """Rolling strict peak-to-trough max drawdown over `period` days (positive %)."""
    def _maxdd(window):
        arr = np.asarray(window, dtype=float)
        if arr.size == 0:
            return 0.0
        running_peaks = np.maximum.accumulate(arr)
        valid = running_peaks > 0
        if not valid.any():
            return 0.0
        drawdowns = np.zeros_like(arr, dtype=float)
        drawdowns[valid] = (running_peaks[valid] - arr[valid]) / running_peaks[valid]
        return float(drawdowns.max() * 100.0)
    return series.rolling(window=period).apply(_maxdd, raw=True)


def calculate_mareturn(series, period):
    """Rolling mean of daily returns over `period` days (decimal return)."""
    return series.pct_change().rolling(window=period).mean()

# ---------------------------------------------------------------------------
# Strategy JSON condition extractor
# ---------------------------------------------------------------------------

FN_LABELS = {
    "relative-strength-index":          "RSI",
    "cumulative-return":                "CumRet",
    "moving-average-price":             "MA",
    "current-price":                    "Price",
    "max-drawdown":                     "MaxDD",
    "moving-average-return":            "MAReturn",
    "exponential-moving-average-price": "EMA",
}

EXTRACTION_DEBUG = False

COMPARATOR_LABELS = {
    "gt": ">", "lt": "<", "gte": ">=", "lte": "<=", "eq": "==", "neq": "!=",
}

COMPARATOR_NEGATIONS = {
    "gt": "<=", "lt": ">=", "gte": "<", "lte": ">", "eq": "!=", "neq": "==",
}


def _get_window(node, prefix):
    params = node.get(f"{prefix}-fn-params", {})
    if "window" in params:
        return params["window"]
    wd = node.get(f"{prefix}-window-days")
    return wd


def _parse_side(node, prefix):
    """Returns dict with fn_type, ticker, window, fixed_value."""
    if prefix == "rhs" and node.get("rhs-fixed-value?"):
        return {"type": "fixed", "value": float(node.get("rhs-val", 0))}
    fn_raw = node.get(f"{prefix}-fn", "")
    ticker = node.get(f"{prefix}-val", "?")
    window = _get_window(node, prefix)
    fn_label = FN_LABELS.get(fn_raw, fn_raw)
    return {"type": "indicator", "fn_raw": fn_raw, "fn_label": fn_label,
            "ticker": ticker, "window": int(window) if window is not None else None}


def _parse_side_from_condition_spec(side_spec, ticker_override=None):
    """Parses condition side from Composer condition payload format."""
    if not isinstance(side_spec, dict):
        return {"type": "indicator", "fn_raw": "", "fn_label": "", "ticker": "?", "window": None}
    if "constant" in side_spec:
        return {"type": "fixed", "value": float(side_spec.get("constant", 0))}
    fn_raw = side_spec.get("fn", "")
    params = side_spec.get("params", {}) or {}
    window = params.get("window")
    ticker = side_spec.get("ticker", ticker_override if ticker_override else "?")
    if ticker == "%":
        ticker = ticker_override if ticker_override else "?"
    fn_label = FN_LABELS.get(fn_raw, fn_raw)
    return {"type": "indicator", "fn_raw": fn_raw, "fn_label": fn_label,
            "ticker": ticker, "window": int(window) if window is not None else None}


def _extract_atomic_conditions(if_child):
    """
    Returns list of atomic conditions normalized as:
      {"lhs": dict, "rhs": dict, "comparator": str}
    Supports direct if-child shape plus condition payload shapes:
      compound, binary-compound, binary.
    """
    comparator = if_child.get("comparator")
    has_direct_shape = if_child.get("lhs-fn") and comparator
    if has_direct_shape:
        return [{
            "lhs": _parse_side(if_child, "lhs"),
            "rhs": _parse_side(if_child, "rhs"),
            "comparator": comparator,
        }]

    payload = if_child.get("condition")
    if not isinstance(payload, dict):
        return []

    out = []

    def _walk(cond_payload, inherited_ticker=None):
        ctype = cond_payload.get("condition-type")
        if ctype == "compound":
            for inner in cond_payload.get("conditions", []):
                _walk(inner, inherited_ticker=inherited_ticker)
            return

        if ctype in ("binary-compound", "binary"):
            comp = cond_payload.get("comparator")
            lhs_spec = cond_payload.get("lhs", {}) or {}
            rhs_spec = cond_payload.get("rhs", {}) or {}
            tickers = cond_payload.get("tickers") or []
            if not tickers:
                tickers = [inherited_ticker] if inherited_ticker else [None]

            for t in tickers:
                lhs = _parse_side_from_condition_spec(lhs_spec, ticker_override=t)
                rhs = _parse_side_from_condition_spec(rhs_spec, ticker_override=t)
                out.append({"lhs": lhs, "rhs": rhs, "comparator": comp})
            return

    _walk(payload)
    return [x for x in out if x.get("comparator")]


def extract_conditions_from_tree(node, depth=0, path_conditions=None, sub_strategy=None, results=None, _stats=None):
    """
    Recursively walks the strategy tree. For every IF node, extracts
    the condition with full context (depth, sub_strategy, endpoint path).
    Returns a flat list of condition dicts.
    """
    if path_conditions is None:
        path_conditions = []
    if results is None:
        results = []
    if _stats is None:
        _stats = {"visited": 0, "extracted": 0, "skipped": {}}

    _stats["visited"] += 1
    if EXTRACTION_DEBUG:
        node_id   = node.get("id", "?")
        node_name = node.get("name")
        node_step = node.get("step", "?")
        print(f"[Extractor] step={node_step} id={node_id} name={node_name} outcome=visiting")

    step = node.get("step")

    if step == "group":
        name = node.get("name")
        new_sub = name if name else sub_strategy
        for child in node.get("children", []):
            extract_conditions_from_tree(child, depth, path_conditions, new_sub, results, _stats=_stats)
        return results

    if step in ("root", "wt-cash-equal", "wt-cash-specified"):
        for child in node.get("children", []):
            extract_conditions_from_tree(child, depth, path_conditions, sub_strategy, results, _stats=_stats)
        return results

    if step == "asset":
        return results

    if step == "filter":
        # Check if this is an asset-selection sort (top-N by indicator)
        select_n = int(node.get("select-n", 0))
        asset_children = [c for c in node.get("children", []) if c.get("step") == "asset"]
        fn_raw   = node.get("sort-by-fn", "")
        sort_win = node.get("sort-by-window-days") or (node.get("sort-by-fn-params") or {}).get("window")
        fn_label = FN_LABELS.get(fn_raw, fn_raw)

        if select_n == 1 and len(asset_children) == 2 and fn_raw:
            # Expand into pairwise comparisons: winner > loser for each asset
            for i, child in enumerate(asset_children):
                winner = child.get("ticker", "?")
                loser  = asset_children[1 - i].get("ticker", "?")
                w      = str(sort_win) if sort_win else None
                human  = f"{fn_label}({winner}, {w}) > {fn_label}({loser}, {w})" if w else f"{fn_label}({winner}) > {fn_label}({loser})"
                lhs = {"type": "indicator", "fn_raw": fn_raw, "fn_label": fn_label,
                       "ticker": winner, "window": int(sort_win) if sort_win else None}
                rhs = {"type": "indicator", "fn_raw": fn_raw, "fn_label": fn_label,
                       "ticker": loser,  "window": int(sort_win) if sort_win else None}
                category = _categorize_condition(lhs, rhs)
                results.append({
                    "id":                len(results),
                    "sub_strategy":      sub_strategy or "(root)",
                    "depth":             depth,
                    "human":             human,
                    "comparator":        "gt",
                    "comp_label":        ">",
                    "lhs":               lhs,
                    "rhs":               rhs,
                    "category":          category,
                    "path_so_far":       list(path_conditions),
                    "children_endpoints": [winner],
                })
                _stats["extracted"] += 1
                if EXTRACTION_DEBUG:
                    node_id   = node.get("id", "?")
                    node_name = node.get("name")
                    node_step = node.get("step", "?")
                    print(f"[Extractor] step={node_step} id={node_id} name={node_name} outcome=extracted")
            return results

        # Portfolio filter or unsupported sort — log and walk children
        skip_reason = f"portfolio filter (select-n={select_n})"
        _stats["skipped"][skip_reason] = _stats["skipped"].get(skip_reason, 0) + 1
        if EXTRACTION_DEBUG:
            node_id   = node.get("id", "?")
            node_name = node.get("name")
            print(f"[Extractor] step=filter id={node_id} name={node_name} outcome=skipped: {skip_reason}")
        for child in node.get("children", []):
            extract_conditions_from_tree(child, depth, path_conditions, sub_strategy, results, _stats=_stats)
        return results

    if step == "if":
        if_children = node.get("children", [])
        positive_branches = [c for c in if_children if not c.get("is-else-condition?")]
        else_branches     = [c for c in if_children if     c.get("is-else-condition?")]

        for pos in positive_branches:
            atomic_conditions = _extract_atomic_conditions(pos)
            if not atomic_conditions:
                continue

            # Build human-readable label
            def side_label(s):
                if s["type"] == "fixed":
                    return str(s["value"])
                w = s["window"]
                return f"{s['fn_label']}({s['ticker']}, {w})" if w else f"{s['fn_label']}({s['ticker']})"

            for atom in atomic_conditions:
                lhs = atom["lhs"]
                rhs = atom["rhs"]
                comparator = atom.get("comparator", "?")
                comp_label = COMPARATOR_LABELS.get(comparator, "?")
                human = f"{side_label(lhs)} {comp_label} {side_label(rhs)}"
                category = _categorize_condition(lhs, rhs)

                results.append({
                    "id":           len(results),
                    "sub_strategy": sub_strategy or "(root)",
                    "depth":        depth,
                    "human":        human,
                    "comparator":   comparator,
                    "comp_label":   comp_label,
                    "lhs":          lhs,
                    "rhs":          rhs,
                    "category":     category,
                    "path_so_far":  list(path_conditions),
                    "children_endpoints": _collect_endpoints(pos),
                })
                _stats["extracted"] += 1
                if EXTRACTION_DEBUG:
                    node_id   = node.get("id", "?")
                    node_name = node.get("name")
                    node_step = node.get("step", "?")
                    print(f"[Extractor] step={node_step} id={node_id} name={node_name} outcome=extracted")

                new_path = path_conditions + [human]
                for grandchild in pos.get("children", []):
                    extract_conditions_from_tree(grandchild, depth + 1, new_path, sub_strategy, results, _stats=_stats)

                if else_branches:
                    neg_label = COMPARATOR_NEGATIONS.get(comparator, "?")
                    neg_human = f"{side_label(lhs)} {neg_label} {side_label(rhs)}"
                    new_path_neg = path_conditions + [neg_human]
                    for els in else_branches:
                        for grandchild in els.get("children", []):
                            extract_conditions_from_tree(grandchild, depth + 1, new_path_neg, sub_strategy, results, _stats=_stats)

    if step == "if-child":
        for child in node.get("children", []):
            extract_conditions_from_tree(child, depth, path_conditions, sub_strategy, results, _stats=_stats)

    return results


def extract_conditions(node):
    """
    Public wrapper around extract_conditions_from_tree.
    Always prints an extraction summary line, returns the conditions list.
    """
    _stats = {"visited": 0, "extracted": 0, "skipped": {}}
    results = extract_conditions_from_tree(node, _stats=_stats)
    skip_parts = ", ".join(
        f"{reason}: {count}" for reason, count in _stats["skipped"].items()
    ) or "none"
    print(
        f"[Extractor] Visited {_stats['visited']} nodes | "
        f"Extracted {_stats['extracted']} conditions | "
        f"Skipped: {skip_parts}"
    )
    return results


def _categorize_condition(lhs, rhs):
    """Returns category string for fuzzing: RSI_fixed, RSI_vs_RSI, MA_price, CumRet_fixed, etc."""
    if lhs["type"] != "indicator":
        return "unknown"
    fn = lhs["fn_label"]
    if rhs["type"] == "fixed":
        return f"{fn}_fixed"
    elif rhs["type"] == "indicator":
        rhs_fn = rhs["fn_label"]
        # Price vs EMA cross
        if fn == "Price" and rhs_fn == "EMA":
            return "Price_vs_EMA"
        # EMA vs MA cross
        if fn == "EMA" and rhs_fn in ("MA", "SMA"):
            return "EMA_vs_MA"
        # EMA vs EMA
        if fn == "EMA" and rhs_fn == "EMA":
            return "EMA_vs_EMA"
        return f"{fn}_vs_{rhs_fn}"
    return "unknown"


def _collect_endpoints(node):
    """Collect unique asset tickers reachable from this node (order-preserving)."""
    endpoints = []
    seen = set()
    def _walk(n):
        if n.get("step") == "asset":
            t = n.get("ticker", "?")
            if t not in seen:
                seen.add(t)
                endpoints.append(t)
        for c in n.get("children", []):
            _walk(c)
    _walk(node)
    return endpoints


# ---------------------------------------------------------------------------
# Prompting helpers
# ---------------------------------------------------------------------------

def prompt(label, default=None, cast=str):
    default_str = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"  {label}{default_str}: ").strip()
        if raw == "" and default is not None:
            return cast(default)
        try:
            return cast(raw)
        except (ValueError, TypeError):
            print(f"    Invalid input. Expected {cast.__name__}.")


def gather_inputs():
    print("\n=== Fuzz Tester Configuration ===\n")

    non_interactive = len(sys.argv) > 1
    if non_interactive:
        json_path = Path(sys.argv[1])
        if not json_path.exists():
            print(f"  ERROR: File not found: {json_path}")
            sys.exit(1)
        print(f"  Strategy JSON path: {json_path}  (from command line)")
        rsi_fuzz = ma_fuzz = cumret_fuzz = price_fuzz = maxdd_fuzz = 30.0
        thresh_step = 0.5
        period_step = 1
        primary_asset = "TQQQ"
        start_date = "2015-01-01"
        end_date = "2026-03-15"
        print("  Using all defaults.\n")
    else:
        json_path = prompt("Strategy JSON path", default="pathfinder/strategy.json")
        json_path = Path(json_path)
        if not json_path.exists():
            print(f"  ERROR: File not found: {json_path}")
            sys.exit(1)

        print()
        rsi_fuzz    = prompt("RSI fuzz range % (e.g. 30 = ±30% on period AND threshold)", default=30, cast=float)
        ma_fuzz     = prompt("MA fuzz range %", default=30, cast=float)
        cumret_fuzz = prompt("CumRet fuzz range %", default=30, cast=float)
        price_fuzz  = prompt("Price/MA cross fuzz range % (window only)", default=30, cast=float)
        maxdd_fuzz  = prompt("MaxDD fuzz range % (window and threshold)", default=30, cast=float)

        print()
        thresh_step = prompt("Threshold step size", default=0.5, cast=float)
        period_step = prompt("Period step size (integer)", default=1, cast=int)

        print()
        primary_asset = prompt("Primary strategy asset (benchmark for vs-primary stat)", default="TQQQ")
        start_date    = prompt("Start date", default="2015-01-01")
        end_date      = prompt("End date",   default="2026-03-15")

    return {
        "json_path":      json_path,
        "primary_asset":  primary_asset.upper(),
        "fuzz_pct": {
            "RSI":    rsi_fuzz / 100,
            "MA":     ma_fuzz / 100,
            "CumRet": cumret_fuzz / 100,
            "Price":  price_fuzz / 100,
            "MaxDD":  maxdd_fuzz / 100,
        },
        "thresh_step":  thresh_step,
        "period_step":  period_step,
        "start_date":   start_date,
        "end_date":     end_date,
    }


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
_PRICE_CACHE = {}

def load_price_series(ticker):
    """Loads price series from CSV, caching it in memory to prevent duplicate I/O."""
    ticker_upper = ticker.upper()
    if ticker_upper not in _PRICE_CACHE:
        df = load_ticker_csv(ticker_upper, DATA_DIR)
        df = df.set_index("date").sort_index()
        _PRICE_CACHE[ticker_upper] = df["close"]
    return _PRICE_CACHE[ticker_upper]


def compute_indicator(series, fn_label, period):
    if fn_label == "RSI":
        return calculate_rsi(series, period)
    elif fn_label in ("MA", "SMA"):
        return calculate_sma(series, period)
    elif fn_label == "EMA":
        return calculate_ema(series, period)
    elif fn_label == "CumRet":
        return calculate_cumret(series, period)
    elif fn_label == "MaxDD":
        return calculate_maxdd(series, period)
    elif fn_label == "MAReturn":
        return calculate_mareturn(series, period)
    else:
        return series  # Price fallback


def get_bil_daily_returns(start_date, end_date):
    """Returns BIL daily returns aligned to a date index."""
    s = load_price_series("BIL")
    s = s[(s.index >= start_date) & (s.index <= end_date)]
    return s.pct_change().fillna(0)


def get_primary_daily_returns(ticker, start_date, end_date):
    """Returns primary asset daily returns for vs-primary stat."""
    s = load_price_series(ticker)
    s = s[(s.index >= start_date) & (s.index <= end_date)]
    return s.pct_change().fillna(0)


# ---------------------------------------------------------------------------
# Single condition sweep
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Single condition sweep
# ---------------------------------------------------------------------------

def _apply_comparator(comp, lhs_vals, rhs_vals):
    """Helper to apply the string comparator to pandas Series or scalars."""
    if comp == "gt":   return lhs_vals > rhs_vals
    if comp == "lt":   return lhs_vals < rhs_vals
    if comp == "gte":  return lhs_vals >= rhs_vals
    if comp == "lte":  return lhs_vals <= rhs_vals
    if comp == "eq":   return lhs_vals == rhs_vals
    if comp == "neq":  return lhs_vals != rhs_vals
    return lhs_vals > rhs_vals  # fallback

def _evaluate_signal(combined, fired_mask, bil_returns, all_returns, period_val, param_val):
    """Helper to calculate standard sweep metrics for a fired condition."""
    fired_idx = combined.index[fired_mask]
    if len(fired_idx) < 2:
        return None

    ep_returns    = combined["ep"].pct_change().shift(-1)
    fired_returns = ep_returns.loc[fired_idx].dropna()
    if fired_returns.empty:
        return None

    bil_aligned  = bil_returns.reindex(fired_returns.index, fill_value=0)

    wins  = (fired_returns.values > bil_aligned.values).sum()
    total = len(fired_returns)
    if total == 0:
        return None

    win_rate = wins / total
    score    = win_rate * math.log(max(total, 1))

    gains  = fired_returns[fired_returns > 0].sum()
    losses = abs(fired_returns[fired_returns < 0].sum())
    pf     = gains / losses if losses > 0 else (2.0 if gains > 0 else 0.0)

    beat_rates = {}
    for ticker, returns in all_returns.items():
        aligned = returns.reindex(fired_returns.index, fill_value=0)
        ticker_wins = (fired_returns.values > aligned.values).sum()
        beat_rates[ticker] = round(ticker_wins / total, 4)

    return {
        "period":             period_val,
        "param":              param_val,
        "win_rate":           round(win_rate, 4),
        "total_trades":       total,
        "score":              round(score, 4),
        "profit_factor":      round(min(pf, 99.0), 4),
        "beat_rates":         beat_rates,
    }

def sweep_condition(cond, config, bil_returns, all_returns, endpoint=None):
    """
    Runs a 2D sweep over (period, threshold/window) for a single condition.
    Returns a DataFrame with columns: period, param, win_rate, total_trades, score, etc.
    """
    lhs, rhs, cat = cond["lhs"], cond["rhs"], cond["category"]
    start, end, fuzz = config["start_date"], config["end_date"], config["fuzz_pct"]
    comp = cond["comparator"]

    # Use explicit endpoint or fall back to first child
    if endpoint is None:
        endpoints = cond["children_endpoints"]
        if not endpoints: return None, "No endpoint found"
        endpoint = endpoints[0]

    try:
        ep_price = load_price_series(endpoint)
    except FileNotFoundError:
        return None, f"No data for endpoint {endpoint}"

    results = []
    # --- 1. RSI / CumRet / MaxDD / MAReturn vs Fixed Threshold ---
    if cat in ("RSI_fixed", "CumRet_fixed", "MaxDD_fixed", "MAReturn_fixed"):
        base_period, base_thresh = lhs["window"], rhs["value"]
        f_val = fuzz.get(cat.split('_')[0], 0.2)

        p_lo, p_hi = max(2, round(base_period * (1 - f_val))), max(3, round(base_period * (1 + f_val)))
        t_lo, t_hi = min(base_thresh * (1 - f_val), base_thresh * (1 + f_val)), max(base_thresh * (1 - f_val), base_thresh * (1 + f_val))

        try: price = load_price_series(lhs["ticker"])
        except FileNotFoundError as e: return None, str(e)

        thresholds = [round(float(t), 4) for t in np.arange(t_lo, t_hi + config["thresh_step"] / 2, config["thresh_step"])] or [round(base_thresh, 4)]
        periods = range(p_lo, p_hi + 1, config["period_step"])

        for period in periods:
            metric_vals = compute_indicator(price, lhs["fn_label"], period)
            for thresh in thresholds:
                combined = pd.DataFrame({"metric": metric_vals, "ep": ep_price}).dropna()
                combined = combined[(combined.index >= start) & (combined.index <= end)]
                if len(combined) < 20: continue

                fired = _apply_comparator(comp, combined["metric"], thresh)
                res = _evaluate_signal(combined, fired, bil_returns, all_returns, period, thresh)
                if res: results.append(res)

    # --- 2. Indicator vs Indicator (RSI vs RSI, MA vs MA, MaxDD vs MaxDD, etc.) ---
    elif cat in ("RSI_vs_RSI", "CumRet_vs_CumRet", "MaxDD_vs_MaxDD", "MAReturn_vs_MAReturn", "MaxDD_vs_MAReturn", "MAReturn_vs_MaxDD", "MA_vs_MA", "EMA_vs_EMA"):
        p_l, p_r = lhs.get("window") or 10, rhs.get("window") or 10
        f_val = fuzz.get(cat.split('_')[0], 0.2)

        p_l_lo, p_l_hi = max(2, round(p_l * (1 - f_val))), max(3, round(p_l * (1 + f_val)))
        p_r_lo, p_r_hi = max(2, round(p_r * (1 - f_val))), max(3, round(p_r * (1 + f_val)))

        try:
            price_l = load_price_series(lhs["ticker"])
            price_r = load_price_series(rhs["ticker"])
        except FileNotFoundError as e: return None, str(e)

        periods_l = range(p_l_lo, p_l_hi + 1, config["period_step"])
        periods_r = range(p_r_lo, p_r_hi + 1, config["period_step"])

        for period_l in periods_l:
            metric_l = compute_indicator(price_l, lhs["fn_label"], period_l)
            for period_r in periods_r:
                metric_r = compute_indicator(price_r, rhs["fn_label"], period_r)
                combined = pd.DataFrame({"lhs_m": metric_l, "rhs_m": metric_r, "ep": ep_price}).dropna()
                combined = combined[(combined.index >= start) & (combined.index <= end)]
                if len(combined) < 20: continue

                fired = _apply_comparator(comp, combined["lhs_m"], combined["rhs_m"])
                res = _evaluate_signal(combined, fired, bil_returns, all_returns, period_l, period_r)
                if res: results.append(res)

    # --- 3. Price vs MAReturn (1D sweep on MAReturn window) ---
    elif cat == "Price_vs_MAReturn":
        rhs_ticker = rhs.get("ticker", lhs["ticker"])
        base_window = rhs.get("window") or 200
        f_val = fuzz.get("MAReturn", fuzz.get("MA", 0.3))
        win_lo = max(2, round(base_window * (1 - f_val)))
        win_hi = max(3, round(base_window * (1 + f_val)))
        try:
            price_series = load_price_series(lhs["ticker"])
            rhs_price_series = load_price_series(rhs_ticker)
        except FileNotFoundError as e:
            return None, str(e)
        windows = range(win_lo, win_hi + 1, max(1, config["period_step"]))
        for window in windows:
            rhs_vals = compute_indicator(rhs_price_series, "MAReturn", window)
            combined = pd.DataFrame({"price": price_series, "rhs_metric": rhs_vals, "ep": ep_price}).dropna()
            combined = combined[(combined.index >= start) & (combined.index <= end)]
            if len(combined) < 20:
                continue
            fired = _apply_comparator(comp, combined["price"], combined["rhs_metric"])
            res = _evaluate_signal(combined, fired, bil_returns, all_returns, window, "win_rate")
            if res:
                results.append(res)

    # --- 4. Price vs Moving Average (1D Sweep on the RHS window) ---
    elif cat in ("Price_vs_MA", "Price_vs_EMA", "MA_fixed", "Price_fixed"):
        rhs_fn = rhs.get("fn_label", "MA") if rhs["type"] == "indicator" else "MA"
        rhs_ticker = rhs.get("ticker", lhs["ticker"]) if rhs["type"] == "indicator" else lhs["ticker"]
        base_window = rhs.get("window") or lhs.get("window") or 200
        f_val = fuzz.get("MA", 0.3)

        win_lo, win_hi = max(2, round(base_window * (1 - f_val))), max(3, round(base_window * (1 + f_val)))

        try:
            price_series = load_price_series(lhs["ticker"])
            rhs_price_series = load_price_series(rhs_ticker)
        except FileNotFoundError as e: return None, str(e)

        windows = range(win_lo, win_hi + 1, max(1, config["period_step"]))

        for window in windows:
            rhs_vals = compute_indicator(rhs_price_series, rhs_fn, window)
            combined = pd.DataFrame({"price": price_series, "rhs_metric": rhs_vals, "ep": ep_price}).dropna()
            combined = combined[(combined.index >= start) & (combined.index <= end)]
            if len(combined) < 20: continue

            fired = _apply_comparator(comp, combined["price"], combined["rhs_metric"])
            res = _evaluate_signal(combined, fired, bil_returns, all_returns, window, "win_rate")
            if res: results.append(res)

    # --- 5. EMA vs MA cross (2D Sweep on both windows) ---
    elif cat == "EMA_vs_MA":
        b_ema, b_ma = lhs.get("window") or 8, rhs.get("window") or 70
        f_val = fuzz.get("MA", 0.3)

        ema_lo, ema_hi = max(2, round(b_ema * (1 - f_val))), max(3, round(b_ema * (1 + f_val)))
        ma_lo, ma_hi   = max(2, round(b_ma * (1 - f_val))), max(3, round(b_ma * (1 + f_val)))

        try: price_series = load_price_series(lhs["ticker"])
        except FileNotFoundError as e: return None, str(e)

        for ema_w in range(ema_lo, ema_hi + 1, max(1, config["period_step"])):
            ema_vals = compute_indicator(price_series, "EMA", ema_w)
            for ma_w in range(ma_lo, ma_hi + 1, max(1, config["period_step"])):
                ma_vals = compute_indicator(price_series, "SMA", ma_w)
                combined = pd.DataFrame({"ema": ema_vals, "ma": ma_vals, "ep": ep_price}).dropna()
                combined = combined[(combined.index >= start) & (combined.index <= end)]
                if len(combined) < 20: continue

                fired = _apply_comparator(comp, combined["ema"], combined["ma"])
                res = _evaluate_signal(combined, fired, bil_returns, all_returns, ema_w, ma_w)
                if res: results.append(res)

    else:
        return None, f"Unsupported category: {cat}"

    if not results:
        return None, "Condition never fired in date range"

    return pd.DataFrame(results), None

def compute_fragility(df):
    """
    Returns a fragility score 0–1 for a sweep result DataFrame.
    Higher = more fragile (spiky). Lower = more robust (plateau).
    Based on coefficient of variation of win_rate across the sweep.
    """
    if df is None or df.empty:
        return 1.0
    wr = df["win_rate"].values
    mean = wr.mean()
    if mean == 0:
        return 1.0
    cv = wr.std() / mean   # coefficient of variation
    # Normalize to 0–1 range (cv > 1 = very fragile)
    return round(min(cv, 1.0), 4)


def compute_tail_metrics(fired_returns, bil_returns):
    """
    Compute tail dependency metrics for signal-day endpoint returns.
    Higher tail_score means the condition's edge is driven by outlier days.
    Returns a dict with 6 keys: tail_score, tail_concentration, excess_kurtosis,
    base_win_rate, stripped_win_rate, wr_delta.
    """
    _zero = {"tail_score": 0.0, "tail_concentration": 0.0, "excess_kurtosis": 0.0,
             "base_win_rate": 0.0, "stripped_win_rate": 0.0, "wr_delta": 0.0}
    if fired_returns is None or len(fired_returns) < 5:
        return _zero

    bil_aligned = bil_returns.reindex(fired_returns.index, fill_value=0)
    n = len(fired_returns)

    # Base win rate vs BIL
    base_wins = (fired_returns.values > bil_aligned.values).sum()
    base_win_rate = base_wins / n

    # Stripped win rate: remove top 5% days by absolute return magnitude
    cutoff = fired_returns.abs().quantile(0.95)
    keep = fired_returns.abs() < cutoff
    stripped = fired_returns[keep]
    bil_stripped = bil_aligned[keep]
    if len(stripped) >= 5:
        stripped_wins = (stripped.values > bil_stripped.values).sum()
        stripped_win_rate = stripped_wins / len(stripped)
    else:
        stripped_win_rate = base_win_rate
    wr_delta = max(base_win_rate - stripped_win_rate, 0.0)

    # Tail concentration: fraction of total gains from top 5% gain days
    gains = fired_returns[fired_returns > 0]
    if len(gains) > 1 and gains.sum() > 0:
        top_cutoff = gains.quantile(0.95)
        tail_concentration = float(gains[gains >= top_cutoff].sum() / gains.sum())
    else:
        tail_concentration = 0.0

    # Excess kurtosis (numpy — no scipy needed)
    vals = fired_returns.values
    if len(vals) >= 4:
        mean, std = np.mean(vals), np.std(vals)
        excess_kurtosis = float(np.mean(((vals - mean) / std) ** 4) - 3) if std > 0 else 0.0
    else:
        excess_kurtosis = 0.0

    # Tail score: blend tail concentration and WR sensitivity to outlier removal
    norm_wr_delta = min(wr_delta / 0.15, 1.0)  # 0.15 WR-point drop -> fully tail-driven
    tail_score = round(min(0.5 * tail_concentration + 0.5 * norm_wr_delta, 1.0), 4)

    return {
        "tail_score":         tail_score,
        "tail_concentration": round(float(tail_concentration), 4),
        "excess_kurtosis":    round(float(excess_kurtosis), 4),
        "base_win_rate":      round(float(base_win_rate), 4),
        "stripped_win_rate":  round(float(stripped_win_rate), 4),
        "wr_delta":           round(float(wr_delta), 4),
    }


def _get_base_fired_returns(cond, config, bil_returns, all_returns, endpoint):
    """
    Run signal generation for the base parameter cell only and return
    the endpoint asset's daily returns on signal-fired days.
    Used to compute tail metrics without modifying the sweep loop.
    """
    lhs, rhs, cat = cond["lhs"], cond["rhs"], cond["category"]
    start, end = config["start_date"], config["end_date"]
    comp = cond["comparator"]

    if endpoint is None:
        return None
    try:
        ep_price = load_price_series(endpoint)
    except FileNotFoundError:
        return None

    fired_idx = None
    combined = None

    if cat in ("RSI_fixed", "CumRet_fixed", "MaxDD_fixed", "MAReturn_fixed"):
        try:
            price = load_price_series(lhs["ticker"])
        except FileNotFoundError:
            return None
        metric_vals = compute_indicator(price, lhs["fn_label"], lhs["window"])
        combined = pd.DataFrame({"metric": metric_vals, "ep": ep_price}).dropna()
        combined = combined[(combined.index >= start) & (combined.index <= end)]
        if len(combined) < 2:
            return None
        fired_idx = combined.index[_apply_comparator(comp, combined["metric"], rhs["value"])]

    elif cat in ("RSI_vs_RSI", "CumRet_vs_CumRet", "MaxDD_vs_MaxDD", "MAReturn_vs_MAReturn",
                 "MaxDD_vs_MAReturn", "MAReturn_vs_MaxDD", "MA_vs_MA", "EMA_vs_EMA"):
        try:
            price_l = load_price_series(lhs["ticker"])
            price_r = load_price_series(rhs["ticker"])
        except FileNotFoundError:
            return None
        metric_l = compute_indicator(price_l, lhs["fn_label"], lhs.get("window") or 10)
        metric_r = compute_indicator(price_r, rhs["fn_label"], rhs.get("window") or 10)
        combined = pd.DataFrame({"lhs_m": metric_l, "rhs_m": metric_r, "ep": ep_price}).dropna()
        combined = combined[(combined.index >= start) & (combined.index <= end)]
        if len(combined) < 2:
            return None
        fired_idx = combined.index[_apply_comparator(comp, combined["lhs_m"], combined["rhs_m"])]

    elif cat == "Price_vs_MAReturn":
        rhs_ticker = rhs.get("ticker", lhs["ticker"])
        window = rhs.get("window") or 200
        try:
            price_series = load_price_series(lhs["ticker"])
            rhs_price_series = load_price_series(rhs_ticker)
        except FileNotFoundError:
            return None
        rhs_vals = compute_indicator(rhs_price_series, "MAReturn", window)
        combined = pd.DataFrame({"price": price_series, "rhs_metric": rhs_vals, "ep": ep_price}).dropna()
        combined = combined[(combined.index >= start) & (combined.index <= end)]
        if len(combined) < 2:
            return None
        fired_idx = combined.index[_apply_comparator(comp, combined["price"], combined["rhs_metric"])]

    elif cat in ("Price_vs_MA", "Price_vs_EMA", "MA_fixed", "Price_fixed"):
        rhs_fn = rhs.get("fn_label", "MA") if rhs["type"] == "indicator" else "MA"
        rhs_ticker = rhs.get("ticker", lhs["ticker"]) if rhs["type"] == "indicator" else lhs["ticker"]
        window = rhs.get("window") or lhs.get("window") or 200
        try:
            price_series = load_price_series(lhs["ticker"])
            rhs_price_series = load_price_series(rhs_ticker)
        except FileNotFoundError:
            return None
        rhs_vals = compute_indicator(rhs_price_series, rhs_fn, window)
        combined = pd.DataFrame({"price": price_series, "rhs_metric": rhs_vals, "ep": ep_price}).dropna()
        combined = combined[(combined.index >= start) & (combined.index <= end)]
        if len(combined) < 2:
            return None
        fired_idx = combined.index[_apply_comparator(comp, combined["price"], combined["rhs_metric"])]

    elif cat == "EMA_vs_MA":
        try:
            price_series = load_price_series(lhs["ticker"])
        except FileNotFoundError:
            return None
        ema_vals = compute_indicator(price_series, "EMA", lhs.get("window") or 8)
        ma_vals = compute_indicator(price_series, "SMA", rhs.get("window") or 70)
        combined = pd.DataFrame({"ema": ema_vals, "ma": ma_vals, "ep": ep_price}).dropna()
        combined = combined[(combined.index >= start) & (combined.index <= end)]
        if len(combined) < 2:
            return None
        fired_idx = combined.index[_apply_comparator(comp, combined["ema"], combined["ma"])]

    else:
        return None

    ep_returns = combined["ep"].pct_change().shift(-1)
    return ep_returns.loc[fired_idx].dropna()


# ---------------------------------------------------------------------------
# HTML report generator
# ---------------------------------------------------------------------------

FRAGILITY_COLOR = [
    (0.0,  "#2ecc71"),   # green
    (0.25, "#f1c40f"),   # yellow
    (0.5,  "#e67e22"),   # orange
    (0.75, "#e74c3c"),   # red
    (1.0,  "#8e44ad"),   # purple (extreme)
]

def fragility_color(score):
    # Linear interpolation across breakpoints
    for i in range(len(FRAGILITY_COLOR) - 1):
        lo_v, lo_c = FRAGILITY_COLOR[i]
        hi_v, hi_c = FRAGILITY_COLOR[i + 1]
        if score <= hi_v:
            t = (score - lo_v) / (hi_v - lo_v)
            # Hex interpolation (simplified — just pick nearest)
            return lo_c if t < 0.5 else hi_c
    return FRAGILITY_COLOR[-1][1]


def fragility_label(score):
    if score < 0.15:  return "Robust"
    if score < 0.35:  return "Stable"
    if score < 0.55:  return "Moderate"
    if score < 0.75:  return "Fragile"
    return "Very Fragile"



def df_to_heatmap_data(df, cond):
    """Convert sweep DataFrame to JSON-serializable heatmap data."""
    if df is None or isinstance(df, str):
        return None
    periods = sorted(df["period"].unique())
    params  = sorted(df["param"].unique())
    matrix  = []
    for period in periods:
        row = []
        for param in params:
            cell = df[(df["period"] == period) & (df["param"] == param)]
            if cell.empty:
                row.append(None)
            else:
                row.append({
                    "wr": cell.iloc[0]["win_rate"],
                    "n":  int(cell.iloc[0]["total_trades"]),
                    "s":  cell.iloc[0]["score"],
                    "pf": cell.iloc[0].get("profit_factor", 0),
                    "pb": cell.iloc[0].get("beat_rates", {}),
                })
        matrix.append(row)
    return {
        "periods": [str(p) for p in periods],
        "params":  [str(p) for p in params],
        "matrix":  matrix,
        "is_1d":   len(params) == 1,
    }


# ---------------------------------------------------------------------------
# Signal data builder (for JS-side equity curve + signal overlay charts)
# ---------------------------------------------------------------------------

def _normalize_prices(prices):
    """Rebase a price series so the first non-None value equals 100."""
    first = next((v for v in prices if v is not None), None)
    if not first:
        return prices
    return [round(v / first * 100, 4) if v is not None else None for v in prices]


def build_signal_data(conditions, config):
    """
    Builds pre-computed signal data for browser-side rendering.
    Returns a dict with:
      dates:   list of date strings (unified spine)
      prices:  dict ticker -> list of close prices aligned to dates (forward-filled)
      signals: dict "condId:alloc" -> {
        price:     [price, ...] (endpoint price, forward-filled),
        lhs_vals:  [val, ...] (lhs indicator at base params),
        signal:    [bool, ...] (pre-computed signal mask),
        strat_curve:  [float, ...] (strategy cumret, 100-based),
        asset_curve:  [float, ...] (buy-hold cumret, 100-based),
        bil_curve:    [float, ...] (BIL cumret, 100-based),
        label: str (human-readable condition label)
      }
    """
    import pandas as pd
    import numpy as np

    start = config["start_date"]
    end = config["end_date"]

    all_tickers = set()
    for cond in conditions:
        lhs = cond["lhs"]
        rhs = cond["rhs"]
        if lhs.get("ticker"):
            all_tickers.add(lhs["ticker"].upper())
        if rhs.get("type") == "indicator" and rhs.get("ticker"):
            all_tickers.add(rhs["ticker"].upper())
        for ep in cond.get("children_endpoints", []):
            all_tickers.add(ep.upper())
    all_tickers.add("BIL")

    price_series = {}
    for t in sorted(all_tickers):
        try:
            price_series[t] = load_price_series(t)
        except (FileNotFoundError, Exception):
            continue

    date_set = set()
    for t, s in price_series.items():
        s_filt = s[(s.index >= start) & (s.index <= end)]
        for dt in s_filt.index:
            date_set.add(dt)
    global_dates = sorted(date_set)
    global_dates_str = [d.strftime("%Y-%m-%d") for d in global_dates]
    dt_index = pd.DatetimeIndex(global_dates)

    prices_fwd = {}
    for t, s in price_series.items():
        s_filt = s[(s.index >= start) & (s.index <= end)]
        s_aligned = s_filt.reindex(dt_index).ffill()
        prices_fwd[t] = [round(float(v), 4) if pd.notna(v) else None for v in s_aligned.values]

    returns_fwd = {}
    for t, s in price_series.items():
        s_filt = s[(s.index >= start) & (s.index <= end)]
        s_aligned = s_filt.reindex(dt_index).ffill()
        ret = s_aligned.pct_change().fillna(0)
        returns_fwd[t] = [round(float(v), 6) for v in ret.values]

    signals = {}
    for cond in conditions:
        allocs = cond.get("children_endpoints") or [None]
        lhs = cond["lhs"]
        rhs = cond["rhs"]
        comp = cond.get("comparator", "gt")

        for alloc in allocs:
            key = f"{cond['id']}:{alloc if alloc is not None else '(none)'}"
            ep_ticker = (alloc or "").upper()
            lhs_ticker = (lhs.get("ticker") or "").upper()

            price_l = prices_fwd.get(lhs_ticker, [None] * len(global_dates))
            price_ep = prices_fwd.get(ep_ticker, [None] * len(global_dates))
            price_bil = prices_fwd.get("BIL", [None] * len(global_dates))

            ep_series = pd.Series(price_ep, dtype=float)
            bil_series = pd.Series(price_bil, dtype=float)
            ep_ret = ep_series.pct_change().fillna(0).values
            bil_ret = bil_series.pct_change().fillna(0).values

            lhs_series = pd.Series(price_l, dtype=float)
            lhs_window = lhs.get("window")
            if lhs_window and lhs["fn_label"]:
                try:
                    metric = compute_indicator(lhs_series, lhs["fn_label"], int(lhs_window))
                except Exception:
                    metric = pd.Series([np.nan] * len(global_dates))
            else:
                metric = ep_series

            if rhs["type"] == "fixed":
                rhs_val = float(rhs.get("value", 0))
                fired = _apply_comparator(comp, metric, rhs_val)
            else:
                rhs_ticker = (rhs.get("ticker") or "").upper()
                price_r = prices_fwd.get(rhs_ticker, [None] * len(global_dates))
                rhs_series = pd.Series(price_r, dtype=float)
                rhs_window = rhs.get("window")
                if rhs_window and rhs["fn_label"]:
                    try:
                        rhs_metric = compute_indicator(rhs_series, rhs["fn_label"], int(rhs_window))
                    except Exception:
                        rhs_metric = pd.Series([np.nan] * len(global_dates))
                else:
                    rhs_metric = rhs_series
                fired = _apply_comparator(comp, metric, rhs_metric)

            signal_arr = fired.astype(bool).values.tolist()
            lhs_vals_arr = [round(float(v), 4) if pd.notna(v) else None for v in metric.values]

            strat = 100.0
            asset = 100.0
            bil_c = 100.0
            strat_curve = [100.0]
            asset_curve = [100.0]
            bil_curve = [100.0]
            for i in range(1, len(global_dates)):
                sr = ep_ret[i] if signal_arr[i] else bil_ret[i]
                strat *= (1 + sr)
                asset *= (1 + ep_ret[i])
                bil_c *= (1 + bil_ret[i])
                strat_curve.append(round(strat, 4))
                asset_curve.append(round(asset, 4))
                bil_curve.append(round(bil_c, 4))

            lhs_label = f"{lhs['fn_label']}({lhs_ticker}, {lhs_window})" if lhs_window else f"{lhs['fn_label']}({lhs_ticker})"
            signals[key] = {
                "price": [round(float(v), 4) if pd.notna(v) else None for v in ep_series.values],
                "lhs_vals": lhs_vals_arr,
                "signal": signal_arr,
                "strat_curve": strat_curve,
                "asset_curve": asset_curve,
                "bil_curve": bil_curve,
                "label": lhs_label,
                "human": cond.get("human", ""),
                "ep":   _normalize_prices(prices_fwd.get(ep_ticker, [None] * len(global_dates))),
                "lhs":  prices_fwd.get(lhs_ticker, [None] * len(global_dates)),
                "rhs":  prices_fwd.get((rhs.get("ticker") or "").upper(), [None] * len(global_dates)) if rhs["type"] == "indicator" else None,
                "fn_l": lhs.get("fn_label") or None,
                "fn_r": rhs.get("fn_label") if rhs["type"] == "indicator" else None,
                "wl":   lhs.get("window"),
                "wr":   rhs.get("window") if rhs["type"] == "indicator" else None,
                "fv":   rhs.get("value") if rhs["type"] == "fixed" else None,
                "cmp":  comp,
                "bp":   lhs.get("window"),
                "bp2":  rhs.get("window") if rhs["type"] == "indicator" else (rhs.get("value") if rhs["type"] == "fixed" else None),
                "ep_t": ep_ticker,
            }

    return {
        "dates": global_dates_str,
        "prices": prices_fwd,
        "returns": returns_fwd,
        "signals": signals,
    }


def generate_html(conditions, sweep_results, reliability_scores, tail_detail, config, signal_data=None):
    """Generate the full HTML report by injecting data into the external template."""
    import json as jsonmod

    timestamp    = datetime.now().strftime("%Y-%m-%d %H:%M")
    json_name    = Path(config["json_path"]).name
    primary_asset = config.get("primary_asset", "TQQQ")

    # Build per-(cond, alloc) heatmap data dict keyed as "condId:alloc"
    heatmap_data = {}
    for cond in conditions:
        allocs = cond.get("children_endpoints") or []
        if not allocs:
            allocs = ["(none)"]
        for alloc in allocs:
            key = f"{cond['id']}:{alloc}"
            df  = sweep_results.get((cond["id"], alloc))
            heatmap_data[key] = df_to_heatmap_data(df, cond)

    # Conditions metadata for JS — include per-alloc error map
    conds_for_js = []
    for c in conditions:
        allocs = c.get("children_endpoints") or []
        alloc_errors = {}
        for a in allocs:
            v = sweep_results.get((c["id"], a))
            if isinstance(v, str):
                alloc_errors[a] = v
        conds_for_js.append({
            "id":           c["id"],
            "human":        c["human"],
            "sub_strategy": c["sub_strategy"],
            "depth":        c["depth"],
            "category":     c["category"],
            "lhs":          c.get("lhs"),
            "rhs":          c.get("rhs"),
            "allocations":  allocs,
            "alloc_errors": alloc_errors,
        })

    heatmap_json      = jsonmod.dumps(heatmap_data)
    reliability_json  = jsonmod.dumps({str(k): v for k, v in reliability_scores.items()})
    tail_metrics_json = jsonmod.dumps({str(k): v for k, v in tail_detail.items()})
    conditions_json   = jsonmod.dumps(conds_for_js)

    # Load the external HTML template
    template_path = SCRIPT_DIR / "report_template.html"
    if not template_path.exists():
        return "<html><body><h1>Error: report_template.html not found!</h1><p>Please ensure the template file is in the same directory as fuzz_tester.py</p></body></html>"

    html = template_path.read_text(encoding="utf-8")

    # Inject variables using simple string replacement (safe for JS/CSS)
    html = html.replace("__JSON_NAME__", json_name)
    html = html.replace("__TIMESTAMP__", timestamp)
    html = html.replace("__COND_COUNT__", str(len(conditions)))
    html = html.replace("__PRIMARY_ASSET__", primary_asset)
    html = html.replace("__HEATMAP_JSON__", heatmap_json)
    html = html.replace("__RELIABILITY_JSON__", reliability_json)
    html = html.replace("__TAIL_METRICS_JSON__", tail_metrics_json)
    html = html.replace("__CONDITIONS_JSON__", conditions_json)

    available_assets = config.get("available_assets", [primary_asset])
    html = html.replace("__AVAILABLE_ASSETS__", jsonmod.dumps(available_assets))

    # Inject signal data for JS-side equity curve + signal overlay charts
    if signal_data is not None:
        signal_json = jsonmod.dumps(signal_data)
    else:
        signal_json = jsonmod.dumps({"dates": [], "prices": {}, "returns": {}, "signals": {}})
    html = html.replace("__SIGNAL_DATA__", signal_json)

    # Inject uPlot library (cache locally after first fetch)
    uplot_cache = SCRIPT_DIR / "uplot.min.js"
    if uplot_cache.exists():
        uplot_js = uplot_cache.read_text(encoding="utf-8")
    else:
        import requests as _req
        print("  Downloading uPlot library...")
        resp = _req.get("https://cdn.jsdelivr.net/npm/uplot@1.6.31/dist/uPlot.iife.min.js", timeout=30)
        resp.raise_for_status()
        uplot_js = resp.text
        uplot_cache.write_text(uplot_js, encoding="utf-8")
        print("  uPlot cached to uplot.min.js")
    html = html.replace("__UPLOT_JS__", uplot_js)

    # Inject uPlot CSS (required for .u-wrap position:relative and canvas sizing)
    uplot_css_cache = SCRIPT_DIR / "uplot.min.css"
    if uplot_css_cache.exists():
        uplot_css = uplot_css_cache.read_text(encoding="utf-8")
    else:
        import requests as _req
        print("  Downloading uPlot CSS...")
        resp = _req.get("https://cdn.jsdelivr.net/npm/uplot@1.6.31/dist/uPlot.min.css", timeout=30)
        resp.raise_for_status()
        uplot_css = resp.text
        uplot_css_cache.write_text(uplot_css, encoding="utf-8")
        print("  uPlot CSS cached to uplot.min.css")
    html = html.replace("__UPLOT_CSS__", uplot_css)

    return html




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _PRICE_CACHE.clear()

    config = gather_inputs()

    # Load API keys
    print("\n  Loading config and API keys...")
    _, api_keys = load_config(config_dict={"dummy": True})

    # Parse strategy JSON
    print(f"  Parsing {config['json_path']}...")
    with open(config["json_path"], "r", encoding="utf-8") as f:
        tree = json.load(f)

    conditions = extract_conditions(tree)
    if not conditions:
        print("  ERROR: No IF conditions found in strategy JSON.")
        sys.exit(1)

    print(f"  Found {len(conditions)} conditions.\n")

    # Collect all tickers needed
    all_tickers = set(["BIL", config["primary_asset"]])
    for cond in conditions:
        lhs = cond["lhs"]
        rhs = cond["rhs"]
        if lhs.get("ticker"):
            all_tickers.add(lhs["ticker"].upper())
        if rhs.get("type") == "indicator" and rhs.get("ticker"):
            all_tickers.add(rhs["ticker"].upper())
        for ep in cond.get("children_endpoints", []):
            all_tickers.add(ep.upper())

    # Download/update price data
    print("  Checking price data freshness...")
    check_freshness_and_update(list(all_tickers), api_keys, DATA_DIR)
    print()

    # Load daily returns for all tickers (beat rates for every comparison asset)
    print("  Loading daily returns for all tickers...")
    all_returns = {}
    for t in sorted(all_tickers):
        try:
            all_returns[t] = get_primary_daily_returns(t, config["start_date"], config["end_date"])
        except (FileNotFoundError, Exception) as e:
            print(f"    Warning: no return data for {t}, skipping: {e}")
    bil_returns = all_returns.get("BIL")
    if bil_returns is None:
        bil_returns = get_bil_daily_returns(config["start_date"], config["end_date"])

    # Run sweeps — one per (condition × allocation)
    # sweep_results keyed by (cond_id, allocation_ticker)
    # reliability_scores keyed by cond_id — {fragility, tail_score, combined}
    sweep_results      = {}   # (cond_id, alloc) -> df or error string
    reliability_scores = {}   # cond_id -> {fragility, tail_score, combined}
    tail_detail        = {}   # cond_id -> compute_tail_metrics result

    total = len(conditions)
    for i, cond in enumerate(conditions):
        allocs = cond.get("children_endpoints") or []
        if not allocs:
            allocs = [None]   # will fail gracefully inside sweep_condition

        label = cond["human"][:55]
        alloc_str = ",".join(str(a) for a in allocs[:3])
        print(f"  [{i+1}/{total}] {label}... [{alloc_str}]")

        worst_fragility = 0.0
        for alloc in allocs:
            key = (cond["id"], alloc)
            df, err = sweep_condition(cond, config, bil_returns, all_returns, endpoint=alloc)
            if err:
                print(f"    ⚠  {alloc}: {err}")
                sweep_results[key] = err
            else:
                sweep_results[key] = df
                fs = compute_fragility(df)
                worst_fragility = max(worst_fragility, fs)
                print(f"    -> {alloc}: Fragility {fs:.3f} ({fragility_label(fs)})")

        fragility = worst_fragility if allocs != [None] else 1.0

        # Tail metrics from base parameter cell (first alloc)
        base_fired = _get_base_fired_returns(cond, config, bil_returns, all_returns, allocs[0])
        tm = compute_tail_metrics(base_fired, bil_returns)
        tail_detail[cond["id"]] = tm
        combined = round(0.6 * fragility + 0.4 * tm["tail_score"], 4)
        reliability_scores[cond["id"]] = {"fragility": fragility, "tail_score": tm["tail_score"], "combined": combined}
        print(f"    Tail: {tm['tail_score']:.3f} · Combined: {combined:.3f}")

    # Build signal data for JS-side charts
    print("  Building signal data...")
    signal_data = build_signal_data(conditions, config)

    # Generate HTML
    config["available_assets"] = sorted(all_returns.keys())
    print("\n  Generating HTML report...")
    html = generate_html(conditions, sweep_results, reliability_scores, tail_detail, config, signal_data=signal_data)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_stem = Path(config["json_path"]).stem
    out_path = SCRIPT_DIR / f"fuzz_report_{json_stem}_{ts}.html"
    out_path.write_text(html, encoding="utf-8")

    print(f"\n  ✓ Report saved to: {out_path}\n")


if __name__ == "__main__":
    main()
