"""
fuzz_tester.py
==============
Reads a Composer/VOXPORT strategy JSON, extracts every IF condition,
and runs a 2D parameter sweep (period × threshold/window) for each one.

For each parameter combination, measures win rate of the endpoint asset
vs BIL on days the condition fires. Outputs a self-contained HTML report
with heatmaps and a logic-tree summary ranked by condition fragility.

Usage:
    python fuzz_tester.py
    (prompts for all inputs interactively)
"""

import json
import sys
import os
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


def extract_conditions_from_tree(node, depth=0, path_conditions=None, sub_strategy=None, results=None):
    """
    Recursively walks the strategy tree. For every IF node, extracts
    the condition with full context (depth, sub_strategy, endpoint path).
    Returns a flat list of condition dicts.
    """
    if path_conditions is None:
        path_conditions = []
    if results is None:
        results = []

    step = node.get("step")

    if step == "group":
        name = node.get("name")
        new_sub = name if name else sub_strategy
        for child in node.get("children", []):
            extract_conditions_from_tree(child, depth, path_conditions, new_sub, results)
        return results

    if step in ("root", "wt-cash-equal", "wt-cash-specified"):
        for child in node.get("children", []):
            extract_conditions_from_tree(child, depth, path_conditions, sub_strategy, results)
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
            return results

        # Portfolio filter or unsupported sort — just walk children
        for child in node.get("children", []):
            extract_conditions_from_tree(child, depth, path_conditions, sub_strategy, results)
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

                new_path = path_conditions + [human]
                for grandchild in pos.get("children", []):
                    extract_conditions_from_tree(grandchild, depth + 1, new_path, sub_strategy, results)

                if else_branches:
                    neg_label = COMPARATOR_NEGATIONS.get(comparator, "?")
                    neg_human = f"{side_label(lhs)} {neg_label} {side_label(rhs)}"
                    new_path_neg = path_conditions + [neg_human]
                    for els in else_branches:
                        for grandchild in els.get("children", []):
                            extract_conditions_from_tree(grandchild, depth + 1, new_path_neg, sub_strategy, results)

    if step == "if-child":
        for child in node.get("children", []):
            extract_conditions_from_tree(child, depth, path_conditions, sub_strategy, results)

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

    json_path = prompt("Strategy JSON path", default="pathfinder/strategy.json")
    json_path = Path(json_path)
    if not json_path.exists():
        print(f"  ERROR: File not found: {json_path}")
        sys.exit(1)

    print()
    rsi_fuzz    = prompt("RSI fuzz range % (e.g. 10 = ±10% on period AND threshold)", default=10, cast=float)
    ma_fuzz     = prompt("MA fuzz range %", default=30, cast=float)
    cumret_fuzz = prompt("CumRet fuzz range %", default=20, cast=float)
    price_fuzz  = prompt("Price/MA cross fuzz range % (window only)", default=30, cast=float)
    maxdd_fuzz  = prompt("MaxDD fuzz range % (window and threshold)", default=20, cast=float)

    print()
    thresh_step = prompt("Threshold step size", default=1.0, cast=float)
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

def load_price_series(ticker):
    df = load_ticker_csv(ticker.upper(), DATA_DIR)
    df = df.set_index("date").sort_index()
    return df["close"]


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

def sweep_condition(cond, config, bil_returns, primary_returns, endpoint=None):
    """
    Runs a 2D sweep over (period, threshold/window) for a single condition.
    endpoint: the allocation asset to measure. If None, uses first child endpoint.
    Returns a DataFrame with columns: period, param, win_rate, total_trades, score
    """
    lhs = cond["lhs"]
    rhs = cond["rhs"]
    cat = cond["category"]
    start = config["start_date"]
    end   = config["end_date"]
    fuzz  = config["fuzz_pct"]

    # Use explicit endpoint or fall back to first child
    if endpoint is None:
        endpoints = cond["children_endpoints"]
        if not endpoints:
            return None, "No endpoint found"
        endpoint = endpoints[0]

    results = []

    # --- RSI vs fixed threshold ---
    if cat == "RSI_fixed":
        base_period    = lhs["window"]
        base_threshold = rhs["value"]
        fuzz_r         = fuzz.get("RSI", 0.1)

        period_lo = max(2, round(base_period    * (1 - fuzz_r)))
        period_hi = max(3, round(base_period    * (1 + fuzz_r)))
        thresh_lo =        base_threshold * (1 - fuzz_r)
        thresh_hi =        base_threshold * (1 + fuzz_r)

        ticker = lhs["ticker"]
        try:
            price = load_price_series(ticker)
        except FileNotFoundError:
            return None, f"No data for {ticker}"

        try:
            ep_price = load_price_series(endpoint)
        except FileNotFoundError:
            return None, f"No data for endpoint {endpoint}"

        thresholds = np.arange(thresh_lo, thresh_hi + config["thresh_step"] / 2, config["thresh_step"])
        thresholds = [round(float(t), 4) for t in thresholds]
        periods    = range(period_lo, period_hi + 1, config["period_step"])

        comp = cond["comparator"]

        for period in periods:
            rsi_vals = calculate_rsi(price, period)
            for thresh in thresholds:
                # Align all series
                combined = pd.DataFrame({
                    "rsi": rsi_vals,
                    "ep":  ep_price,
                })
                combined = combined.dropna()
                combined = combined[(combined.index >= start) & (combined.index <= end)]
                if len(combined) < 20:
                    continue

                # Fire condition
                if comp == "gt":
                    fired = combined["rsi"] > thresh
                elif comp == "lt":
                    fired = combined["rsi"] < thresh
                elif comp == "gte":
                    fired = combined["rsi"] >= thresh
                elif comp == "lte":
                    fired = combined["rsi"] <= thresh
                else:
                    continue

                fired_idx = combined.index[fired]
                if len(fired_idx) < 2:
                    continue

                # Compute next-day returns on fired days
                ep_returns = combined["ep"].pct_change().shift(-1)
                fired_returns = ep_returns.loc[fired_idx].dropna()

                # Get BIL returns for same days
                bil_aligned = bil_returns.reindex(fired_returns.index, fill_value=0)
                pri_aligned = primary_returns.reindex(fired_returns.index, fill_value=0)

                wins = (fired_returns.values > bil_aligned.values).sum()
                total = len(fired_returns)
                win_rate = wins / total if total > 0 else 0.0
                score = win_rate * math.log(max(total, 1))

                gross_profit = fired_returns[fired_returns > 0].sum()
                gross_loss   = abs(fired_returns[fired_returns < 0].sum())
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                profit_factor = round(min(profit_factor, 99.0), 4)

                primary_beat = (fired_returns.values > pri_aligned.values).sum()
                primary_beat_rate = primary_beat / total if total > 0 else 0.0

                results.append({
                    "period":             period,
                    "param":              thresh,
                    "win_rate":           round(win_rate, 4),
                    "total_trades":       total,
                    "score":              round(score, 4),
                    "profit_factor":      profit_factor,
                    "primary_beat_rate":  round(primary_beat_rate, 4),
                })

    # --- RSI vs RSI (relative strength) ---
    elif cat == "RSI_vs_RSI":
        base_period_l = lhs.get("window") or 10
        base_period_r = rhs.get("window") or 10
        fuzz_r      = fuzz.get("RSI", 0.1)
        period_l_lo = max(2, round(base_period_l * (1 - fuzz_r)))
        period_l_hi = max(3, round(base_period_l * (1 + fuzz_r)))
        period_r_lo = max(2, round(base_period_r * (1 - fuzz_r)))
        period_r_hi = max(3, round(base_period_r * (1 + fuzz_r)))

        ticker_l = lhs["ticker"]
        ticker_r = rhs["ticker"]

        try:
            price_l = load_price_series(ticker_l)
            price_r = load_price_series(ticker_r)
        except FileNotFoundError as e:
            return None, str(e)

        try:
            ep_price = load_price_series(endpoint)
        except FileNotFoundError:
            return None, f"No data for endpoint {endpoint}"

        comp = cond["comparator"]
        periods_l = range(period_l_lo, period_l_hi + 1, config["period_step"])
        periods_r = range(period_r_lo, period_r_hi + 1, config["period_step"])

        for period_l in periods_l:
            rsi_l = calculate_rsi(price_l, period_l)
            for period_r in periods_r:
                rsi_r = calculate_rsi(price_r, period_r)
                combined = pd.DataFrame({"rsi_l": rsi_l, "rsi_r": rsi_r, "ep": ep_price}).dropna()
                combined = combined[(combined.index >= start) & (combined.index <= end)]
                if len(combined) < 20:
                    continue

                if comp == "gt":
                    fired = combined["rsi_l"] > combined["rsi_r"]
                elif comp == "lt":
                    fired = combined["rsi_l"] < combined["rsi_r"]
                else:
                    fired = combined["rsi_l"] > combined["rsi_r"]

                fired_idx = combined.index[fired]
                if len(fired_idx) < 2:
                    continue

                ep_returns    = combined["ep"].pct_change().shift(-1)
                fired_returns = ep_returns.loc[fired_idx].dropna()
                bil_aligned   = bil_returns.reindex(fired_returns.index, fill_value=0)

                wins      = (fired_returns.values > bil_aligned.values).sum()
                total     = len(fired_returns)
                win_rate  = wins / total if total > 0 else 0.0
                score     = win_rate * math.log(max(total, 1))

                gross_profit = fired_returns[fired_returns > 0].sum()
                gross_loss   = abs(fired_returns[fired_returns < 0].sum())
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                profit_factor = round(min(profit_factor, 99.0), 4)

                pri_aligned2 = primary_returns.reindex(fired_returns.index, fill_value=0)
                primary_beat_rate = (fired_returns.values > pri_aligned2.values).sum() / total if total > 0 else 0.0

                results.append({
                    "period":             period_l,
                    "param":              period_r,
                    "win_rate":           round(win_rate, 4),
                    "total_trades":       total,
                    "score":              round(score, 4),
                    "profit_factor":      profit_factor,
                    "primary_beat_rate":  round(primary_beat_rate, 4),
                })

    # --- MA / Price cross (Price > MA) ---
    elif cat in ("Price_vs_MA", "MA_fixed", "Price_fixed", "Price_vs_MAReturn"):
        # Fuzz the indicator window on the rhs side (MA / MAReturn variants)
        rhs_fn_label = rhs.get("fn_label", "MA") if rhs["type"] == "indicator" else "MA"
        rhs_ticker = rhs.get("ticker", lhs["ticker"]) if rhs["type"] == "indicator" else lhs["ticker"]
        base_window = rhs.get("window") or lhs.get("window") or 200
        fuzz_r = fuzz.get("MA", 0.3)
        win_lo  = max(2, round(base_window * (1 - fuzz_r)))
        win_hi  = max(3, round(base_window * (1 + fuzz_r)))

        try:
            price_series = load_price_series(lhs["ticker"])
        except FileNotFoundError as e:
            return None, str(e)
        try:
            rhs_price_series = load_price_series(rhs_ticker)
        except FileNotFoundError as e:
            return None, str(e)

        try:
            ep_price = load_price_series(endpoint)
        except FileNotFoundError:
            return None, f"No data for endpoint {endpoint}"

        comp = cond["comparator"]
        windows = range(win_lo, win_hi + 1, max(1, config["period_step"]))

        for window in windows:
            rhs_vals = compute_indicator(rhs_price_series, rhs_fn_label, window)
            combined = pd.DataFrame({"price": price_series, "rhs_metric": rhs_vals, "ep": ep_price}).dropna()
            combined = combined[(combined.index >= start) & (combined.index <= end)]
            if len(combined) < 20:
                continue

            if comp == "gt":
                fired = combined["price"] > combined["rhs_metric"]
            elif comp == "lt":
                fired = combined["price"] < combined["rhs_metric"]
            else:
                fired = combined["price"] > combined["rhs_metric"]

            fired_idx = combined.index[fired]
            if len(fired_idx) < 2:
                continue

            ep_returns    = combined["ep"].pct_change().shift(-1)
            fired_returns = ep_returns.loc[fired_idx].dropna()
            bil_aligned   = bil_returns.reindex(fired_returns.index, fill_value=0)

            wins     = (fired_returns.values > bil_aligned.values).sum()
            total    = len(fired_returns)
            win_rate = wins / total if total > 0 else 0.0
            score    = win_rate * math.log(max(total, 1))

            gross_profit = fired_returns[fired_returns > 0].sum()
            gross_loss   = abs(fired_returns[fired_returns < 0].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            profit_factor = round(min(profit_factor, 99.0), 4)

            pri_aligned3 = primary_returns.reindex(fired_returns.index, fill_value=0)
            primary_beat_rate = (fired_returns.values > pri_aligned3.values).sum() / total if total > 0 else 0.0

            results.append({
                "period":             window,
                "param":              "win_rate",
                "win_rate":           round(win_rate, 4),
                "total_trades":       total,
                "score":              round(score, 4),
                "profit_factor":      profit_factor,
                "primary_beat_rate":  round(primary_beat_rate, 4),
            })

    # --- CumRet vs fixed ---
    elif cat == "CumRet_fixed":
        base_period    = lhs["window"]
        base_threshold = rhs["value"]
        fuzz_r         = fuzz.get("CumRet", 0.2)

        period_lo = max(2, round(base_period    * (1 - fuzz_r)))
        period_hi = max(3, round(base_period    * (1 + fuzz_r)))
        thresh_lo =        base_threshold * (1 - fuzz_r)
        thresh_hi =        base_threshold * (1 + fuzz_r)

        ticker = lhs["ticker"]
        try:
            price = load_price_series(ticker)
        except FileNotFoundError as e:
            return None, str(e)

        try:
            ep_price = load_price_series(endpoint)
        except FileNotFoundError:
            return None, f"No data for endpoint {endpoint}"

        comp       = cond["comparator"]
        t_lo       = min(thresh_lo, thresh_hi)
        t_hi       = max(thresh_lo, thresh_hi)
        thresholds = np.arange(t_lo, t_hi + config["thresh_step"] / 2, config["thresh_step"])
        thresholds = [round(float(t), 4) for t in thresholds]
        if not thresholds:
            thresholds = [round(base_threshold, 4)]
        periods    = range(period_lo, period_hi + 1, config["period_step"])

        for period in periods:
            cumret_vals = calculate_cumret(price, period)
            for thresh in thresholds:
                combined = pd.DataFrame({"cr": cumret_vals, "ep": ep_price}).dropna()
                combined = combined[(combined.index >= start) & (combined.index <= end)]
                if len(combined) < 20:
                    continue

                if comp == "gt":
                    fired = combined["cr"] > thresh
                elif comp == "lt":
                    fired = combined["cr"] < thresh
                else:
                    fired = combined["cr"] > thresh

                fired_idx = combined.index[fired]
                if len(fired_idx) < 2:
                    continue

                ep_returns    = combined["ep"].pct_change().shift(-1)
                fired_returns = ep_returns.loc[fired_idx].dropna()
                bil_aligned   = bil_returns.reindex(fired_returns.index, fill_value=0)

                wins     = (fired_returns.values > bil_aligned.values).sum()
                total    = len(fired_returns)
                win_rate = wins / total if total > 0 else 0.0
                score    = win_rate * math.log(max(total, 1))

                gross_profit = fired_returns[fired_returns > 0].sum()
                gross_loss   = abs(fired_returns[fired_returns < 0].sum())
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                profit_factor = round(min(profit_factor, 99.0), 4)

                pri_aligned4 = primary_returns.reindex(fired_returns.index, fill_value=0)
                primary_beat_rate = (fired_returns.values > pri_aligned4.values).sum() / total if total > 0 else 0.0

                results.append({
                    "period":             period,
                    "param":              thresh,
                    "win_rate":           round(win_rate, 4),
                    "total_trades":       total,
                    "score":              round(score, 4),
                    "profit_factor":      profit_factor,
                    "primary_beat_rate":  round(primary_beat_rate, 4),
                })

    # ---- Price vs EMA cross (fuzz EMA window) ----
    elif cat == "Price_vs_EMA":
        base_window = rhs.get("window") or lhs.get("window") or 50
        fuzz_r      = fuzz.get("MA", 0.3)
        win_lo      = max(2, round(base_window * (1 - fuzz_r)))
        win_hi      = max(3, round(base_window * (1 + fuzz_r)))

        try:
            price_series = load_price_series(lhs["ticker"])
        except FileNotFoundError as e:
            return None, str(e)

        try:
            ep_price = load_price_series(endpoint)
        except FileNotFoundError:
            return None, f"No data for endpoint {endpoint}"

        comp    = cond["comparator"]
        windows = range(win_lo, win_hi + 1, max(1, config["period_step"]))

        for window in windows:
            ema_vals = calculate_ema(price_series, window)
            combined = pd.DataFrame({"price": price_series, "ema": ema_vals, "ep": ep_price}).dropna()
            combined = combined[(combined.index >= start) & (combined.index <= end)]
            if len(combined) < 20:
                continue

            if comp == "gt":   fired = combined["price"] > combined["ema"]
            elif comp == "lt": fired = combined["price"] < combined["ema"]
            else:              fired = combined["price"] > combined["ema"]

            fired_idx     = combined.index[fired]
            if len(fired_idx) < 2: continue
            ep_returns    = combined["ep"].pct_change().shift(-1)
            fired_returns = ep_returns.loc[fired_idx].dropna()
            bil_aligned   = bil_returns.reindex(fired_returns.index, fill_value=0)
            prim_aligned  = primary_returns.reindex(fired_returns.index, fill_value=0)

            wins      = (fired_returns.values > bil_aligned.values).sum()
            prim_wins = (fired_returns.values > prim_aligned.values).sum()
            total     = len(fired_returns)
            win_rate  = wins / total if total > 0 else 0.0
            gains     = fired_returns[fired_returns > 0].sum()
            losses    = abs(fired_returns[fired_returns < 0].sum())
            pf        = gains / losses if losses > 0 else (2.0 if gains > 0 else 0.0)
            score     = win_rate * math.log(max(total, 1))

            results.append({"period": window, "param": "win_rate",
                "win_rate": round(win_rate, 4), "total_trades": total,
                "score": round(score, 4), "profit_factor": round(min(pf, 99.0), 4),
                "primary_beat_rate": round(prim_wins / total if total > 0 else 0, 4)})

    # ---- EMA vs MA cross (fuzz both windows independently — 2D sweep) ----
    elif cat == "EMA_vs_MA":
        base_ema = lhs.get("window") or 8
        base_ma  = rhs.get("window") or 70
        fuzz_r   = fuzz.get("MA", 0.3)

        ema_lo = max(2, round(base_ema * (1 - fuzz_r)))
        ema_hi = max(3, round(base_ema * (1 + fuzz_r)))
        ma_lo  = max(2, round(base_ma  * (1 - fuzz_r)))
        ma_hi  = max(3, round(base_ma  * (1 + fuzz_r)))

        try:
            price_series = load_price_series(lhs["ticker"])
        except FileNotFoundError as e:
            return None, str(e)

        try:
            ep_price = load_price_series(endpoint)
        except FileNotFoundError:
            return None, f"No data for endpoint {endpoint}"

        comp     = cond["comparator"]
        ema_wins = range(ema_lo, ema_hi + 1, max(1, config["period_step"]))
        ma_wins  = range(ma_lo,  ma_hi  + 1, max(1, config["period_step"]))

        for ema_w in ema_wins:
            ema_vals = calculate_ema(price_series, ema_w)
            for ma_w in ma_wins:
                ma_vals  = calculate_sma(price_series, ma_w)
                combined = pd.DataFrame({"ema": ema_vals, "ma": ma_vals, "ep": ep_price}).dropna()
                combined = combined[(combined.index >= start) & (combined.index <= end)]
                if len(combined) < 20: continue

                if comp == "gt":   fired = combined["ema"] > combined["ma"]
                elif comp == "lt": fired = combined["ema"] < combined["ma"]
                else:              fired = combined["ema"] > combined["ma"]

                fired_idx     = combined.index[fired]
                if len(fired_idx) < 2: continue
                ep_returns    = combined["ep"].pct_change().shift(-1)
                fired_returns = ep_returns.loc[fired_idx].dropna()
                bil_aligned   = bil_returns.reindex(fired_returns.index, fill_value=0)
                prim_aligned  = primary_returns.reindex(fired_returns.index, fill_value=0)

                wins      = (fired_returns.values > bil_aligned.values).sum()
                prim_wins = (fired_returns.values > prim_aligned.values).sum()
                total     = len(fired_returns)
                win_rate  = wins / total if total > 0 else 0.0
                gains     = fired_returns[fired_returns > 0].sum()
                losses    = abs(fired_returns[fired_returns < 0].sum())
                pf        = gains / losses if losses > 0 else (2.0 if gains > 0 else 0.0)
                score     = win_rate * math.log(max(total, 1))

                results.append({"period": ema_w, "param": ma_w,
                    "win_rate": round(win_rate, 4), "total_trades": total,
                    "score": round(score, 4), "profit_factor": round(min(pf, 99.0), 4),
                    "primary_beat_rate": round(prim_wins / total if total > 0 else 0, 4)})

    # ---- EMA vs EMA (2D period sweep: lhs window x rhs window) ----
    elif cat == "EMA_vs_EMA":
        base_period_l = lhs.get("window") or 10
        base_period_r = rhs.get("window") or 10
        fuzz_r      = fuzz.get("MA", 0.3)
        period_l_lo = max(2, round(base_period_l * (1 - fuzz_r)))
        period_l_hi = max(3, round(base_period_l * (1 + fuzz_r)))
        period_r_lo = max(2, round(base_period_r * (1 - fuzz_r)))
        period_r_hi = max(3, round(base_period_r * (1 + fuzz_r)))

        try:
            price_l = load_price_series(lhs["ticker"])
            price_r = load_price_series(rhs["ticker"])
        except FileNotFoundError as e:
            return None, str(e)

        try:
            ep_price = load_price_series(endpoint)
        except FileNotFoundError:
            return None, f"No data for endpoint {endpoint}"

        comp    = cond["comparator"]
        periods_l = range(period_l_lo, period_l_hi + 1, config["period_step"])
        periods_r = range(period_r_lo, period_r_hi + 1, config["period_step"])

        for period_l in periods_l:
            ema_l = calculate_ema(price_l, period_l)
            for period_r in periods_r:
                ema_r = calculate_ema(price_r, period_r)
                combined = pd.DataFrame({"ema_l": ema_l, "ema_r": ema_r, "ep": ep_price}).dropna()
                combined = combined[(combined.index >= start) & (combined.index <= end)]
                if len(combined) < 20: continue

                if comp == "gt":   fired = combined["ema_l"] > combined["ema_r"]
                elif comp == "lt": fired = combined["ema_l"] < combined["ema_r"]
                else:              fired = combined["ema_l"] > combined["ema_r"]

                fired_idx     = combined.index[fired]
                if len(fired_idx) < 2: continue
                ep_returns    = combined["ep"].pct_change().shift(-1)
                fired_returns = ep_returns.loc[fired_idx].dropna()
                bil_aligned   = bil_returns.reindex(fired_returns.index, fill_value=0)
                prim_aligned  = primary_returns.reindex(fired_returns.index, fill_value=0)

                wins      = (fired_returns.values > bil_aligned.values).sum()
                prim_wins = (fired_returns.values > prim_aligned.values).sum()
                total     = len(fired_returns)
                win_rate  = wins / total if total > 0 else 0.0
                gains     = fired_returns[fired_returns > 0].sum()
                losses    = abs(fired_returns[fired_returns < 0].sum())
                pf        = gains / losses if losses > 0 else (2.0 if gains > 0 else 0.0)
                score     = win_rate * math.log(max(total, 1))

                results.append({"period": period_l, "param": period_r,
                    "win_rate": round(win_rate, 4), "total_trades": total,
                    "score": round(score, 4), "profit_factor": round(min(pf, 99.0), 4),
                    "primary_beat_rate": round(prim_wins / total if total > 0 else 0, 4)})

    # ---- MaxDD vs fixed threshold ----
    elif cat in ("MaxDD_fixed", "MAReturn_fixed"):
        base_period    = lhs["window"]
        base_threshold = rhs["value"]
        fuzz_r         = fuzz.get("MaxDD", 0.2)
        lhs_fn         = lhs["fn_label"]

        period_lo = max(2, round(base_period    * (1 - fuzz_r)))
        period_hi = max(3, round(base_period    * (1 + fuzz_r)))
        t_lo      = min(base_threshold * (1 - fuzz_r), base_threshold * (1 + fuzz_r))
        t_hi      = max(base_threshold * (1 - fuzz_r), base_threshold * (1 + fuzz_r))

        ticker = lhs["ticker"]
        try:
            price = load_price_series(ticker)
        except FileNotFoundError as e:
            return None, str(e)

        try:
            ep_price = load_price_series(endpoint)
        except FileNotFoundError:
            return None, f"No data for endpoint {endpoint}"

        comp       = cond["comparator"]
        thresholds = np.arange(t_lo, t_hi + config["thresh_step"] / 2, config["thresh_step"])
        thresholds = [round(float(t), 4) for t in thresholds]
        if not thresholds:
            thresholds = [round(base_threshold, 4)]
        periods    = range(period_lo, period_hi + 1, config["period_step"])

        for period in periods:
            metric_vals = compute_indicator(price, lhs_fn, period)
            for thresh in thresholds:
                combined = pd.DataFrame({"metric": metric_vals, "ep": ep_price}).dropna()
                combined = combined[(combined.index >= start) & (combined.index <= end)]
                if len(combined) < 20:
                    continue
                if comp == "gt":   fired = combined["metric"] > thresh
                elif comp == "lt": fired = combined["metric"] < thresh
                else:              fired = combined["metric"] > thresh

                fired_idx = combined.index[fired]
                if len(fired_idx) < 2:
                    continue

                ep_returns    = combined["ep"].pct_change().shift(-1)
                fired_returns = ep_returns.loc[fired_idx].dropna()
                bil_aligned   = bil_returns.reindex(fired_returns.index, fill_value=0)
                prim_aligned  = primary_returns.reindex(fired_returns.index, fill_value=0)

                wins     = (fired_returns.values > bil_aligned.values).sum()
                prim_wins= (fired_returns.values > prim_aligned.values).sum()
                total    = len(fired_returns)
                win_rate = wins / total if total > 0 else 0.0
                gains    = fired_returns[fired_returns > 0].sum()
                losses   = abs(fired_returns[fired_returns < 0].sum())
                pf       = gains / losses if losses > 0 else (2.0 if gains > 0 else 0.0)
                score    = win_rate * math.log(max(total, 1))

                results.append({"period": period, "param": thresh,
                    "win_rate": round(win_rate, 4), "total_trades": total,
                    "score": round(score, 4), "profit_factor": round(pf, 4),
                    "primary_beat_rate": round(prim_wins / total if total > 0 else 0, 4)})

    # ---- Drawdown/return family comparisons (2D window sweep) ----
    elif cat in ("MaxDD_vs_MaxDD", "MAReturn_vs_MAReturn", "MaxDD_vs_MAReturn", "MAReturn_vs_MaxDD"):
        base_period_l = lhs.get("window") or 10
        base_period_r = rhs.get("window") or 10
        fuzz_r      = fuzz.get("MaxDD", 0.2)
        period_l_lo = max(2, round(base_period_l * (1 - fuzz_r)))
        period_l_hi = max(3, round(base_period_l * (1 + fuzz_r)))
        period_r_lo = max(2, round(base_period_r * (1 - fuzz_r)))
        period_r_hi = max(3, round(base_period_r * (1 + fuzz_r)))
        lhs_fn      = lhs["fn_label"]
        rhs_fn      = rhs["fn_label"]

        ticker_l = lhs["ticker"]
        ticker_r = rhs["ticker"]
        try:
            price_l = load_price_series(ticker_l)
            price_r = load_price_series(ticker_r)
        except FileNotFoundError as e:
            return None, str(e)

        try:
            ep_price = load_price_series(endpoint)
        except FileNotFoundError:
            return None, f"No data for endpoint {endpoint}"

        comp      = cond["comparator"]
        periods_l = range(period_l_lo, period_l_hi + 1, config["period_step"])
        periods_r = range(period_r_lo, period_r_hi + 1, config["period_step"])

        for period_l in periods_l:
            metric_l = compute_indicator(price_l, lhs_fn, period_l)
            for period_r in periods_r:
                metric_r = compute_indicator(price_r, rhs_fn, period_r)
                combined = pd.DataFrame({"metric_l": metric_l, "metric_r": metric_r, "ep": ep_price}).dropna()
                combined = combined[(combined.index >= start) & (combined.index <= end)]
                if len(combined) < 20:
                    continue

                if comp == "gt":   fired = combined["metric_l"] > combined["metric_r"]
                elif comp == "lt": fired = combined["metric_l"] < combined["metric_r"]
                else:              fired = combined["metric_l"] < combined["metric_r"]

                fired_idx = combined.index[fired]
                if len(fired_idx) < 2:
                    continue

                ep_returns    = combined["ep"].pct_change().shift(-1)
                fired_returns = ep_returns.loc[fired_idx].dropna()
                bil_aligned   = bil_returns.reindex(fired_returns.index, fill_value=0)
                prim_aligned  = primary_returns.reindex(fired_returns.index, fill_value=0)

                wins      = (fired_returns.values > bil_aligned.values).sum()
                prim_wins = (fired_returns.values > prim_aligned.values).sum()
                total     = len(fired_returns)
                win_rate  = wins / total if total > 0 else 0.0
                gains     = fired_returns[fired_returns > 0].sum()
                losses    = abs(fired_returns[fired_returns < 0].sum())
                pf        = gains / losses if losses > 0 else (2.0 if gains > 0 else 0.0)
                score     = win_rate * math.log(max(total, 1))

                results.append({"period": period_l, "param": period_r,
                    "win_rate": round(win_rate, 4), "total_trades": total,
                    "score": round(score, 4), "profit_factor": round(pf, 4),
                    "primary_beat_rate": round(prim_wins / total if total > 0 else 0, 4)})
    elif cat == "CumRet_vs_CumRet":
        base_period_l = lhs.get("window") or 10
        base_period_r = rhs.get("window") or 10
        fuzz_r      = fuzz.get("CumRet", 0.2)
        period_l_lo = max(2, round(base_period_l * (1 - fuzz_r)))
        period_l_hi = max(3, round(base_period_l * (1 + fuzz_r)))
        period_r_lo = max(2, round(base_period_r * (1 - fuzz_r)))
        period_r_hi = max(3, round(base_period_r * (1 + fuzz_r)))
        ticker_l = lhs["ticker"]
        ticker_r = rhs["ticker"]
        try:
            price_l = load_price_series(ticker_l)
            price_r = load_price_series(ticker_r)
        except FileNotFoundError as e:
            return None, str(e)
        try:
            ep_price = load_price_series(endpoint)
        except FileNotFoundError:
            return None, f"No data for endpoint {endpoint}"
        comp    = cond["comparator"]
        periods_l = range(period_l_lo, period_l_hi + 1, config["period_step"])
        periods_r = range(period_r_lo, period_r_hi + 1, config["period_step"])
        for period_l in periods_l:
            cr_l = calculate_cumret(price_l, period_l)
            for period_r in periods_r:
                cr_r = calculate_cumret(price_r, period_r)
                combined = pd.DataFrame({"cr_l": cr_l, "cr_r": cr_r, "ep": ep_price}).dropna()
                combined = combined[(combined.index >= start) & (combined.index <= end)]
                if len(combined) < 20:
                    continue
                if comp == "gt":   fired = combined["cr_l"] > combined["cr_r"]
                elif comp == "lt": fired = combined["cr_l"] < combined["cr_r"]
                else:              fired = combined["cr_l"] > combined["cr_r"]
                fired_idx     = combined.index[fired]
                if len(fired_idx) < 2: continue
                ep_returns    = combined["ep"].pct_change().shift(-1)
                fired_returns = ep_returns.loc[fired_idx].dropna()
                bil_aligned   = bil_returns.reindex(fired_returns.index, fill_value=0)
                prim_aligned  = primary_returns.reindex(fired_returns.index, fill_value=0)
                wins      = (fired_returns.values > bil_aligned.values).sum()
                prim_wins = (fired_returns.values > prim_aligned.values).sum()
                total     = len(fired_returns)
                win_rate  = wins / total if total > 0 else 0.0
                gains     = fired_returns[fired_returns > 0].sum()
                losses    = abs(fired_returns[fired_returns < 0].sum())
                pf        = gains / losses if losses > 0 else (2.0 if gains > 0 else 0.0)
                score     = win_rate * math.log(max(total, 1))
                results.append({"period": period_l, "param": period_r,
                    "win_rate": round(win_rate, 4), "total_trades": total,
                    "score": round(score, 4), "profit_factor": round(min(pf, 99.0), 4),
                    "primary_beat_rate": round(prim_wins / total if total > 0 else 0, 4)})

    elif cat == "MA_vs_MA":
        base_period_l = lhs.get("window") or 10
        base_period_r = rhs.get("window") or 10
        fuzz_r      = fuzz.get("MA", 0.3)
        period_l_lo = max(2, round(base_period_l * (1 - fuzz_r)))
        period_l_hi = max(3, round(base_period_l * (1 + fuzz_r)))
        period_r_lo = max(2, round(base_period_r * (1 - fuzz_r)))
        period_r_hi = max(3, round(base_period_r * (1 + fuzz_r)))
        ticker_l = lhs["ticker"]
        ticker_r = rhs["ticker"]
        try:
            price_l = load_price_series(ticker_l)
            price_r = load_price_series(ticker_r)
        except FileNotFoundError as e:
            return None, str(e)
        try:
            ep_price = load_price_series(endpoint)
        except FileNotFoundError:
            return None, f"No data for endpoint {endpoint}"
        comp      = cond["comparator"]
        periods_l = range(period_l_lo, period_l_hi + 1, config["period_step"])
        periods_r = range(period_r_lo, period_r_hi + 1, config["period_step"])
        for period_l in periods_l:
            ma_l = calculate_sma(price_l, period_l)
            for period_r in periods_r:
                ma_r = calculate_sma(price_r, period_r)
                combined = pd.DataFrame({"ma_l": ma_l, "ma_r": ma_r, "ep": ep_price}).dropna()
                combined = combined[(combined.index >= start) & (combined.index <= end)]
                if len(combined) < 20:
                    continue
                if comp == "gt":   fired = combined["ma_l"] > combined["ma_r"]
                elif comp == "lt": fired = combined["ma_l"] < combined["ma_r"]
                else:              fired = combined["ma_l"] > combined["ma_r"]
                fired_idx     = combined.index[fired]
                if len(fired_idx) < 2: continue
                ep_returns    = combined["ep"].pct_change().shift(-1)
                fired_returns = ep_returns.loc[fired_idx].dropna()
                bil_aligned   = bil_returns.reindex(fired_returns.index, fill_value=0)
                prim_aligned  = primary_returns.reindex(fired_returns.index, fill_value=0)
                wins      = (fired_returns.values > bil_aligned.values).sum()
                prim_wins = (fired_returns.values > prim_aligned.values).sum()
                total     = len(fired_returns)
                win_rate  = wins / total if total > 0 else 0.0
                gains     = fired_returns[fired_returns > 0].sum()
                losses    = abs(fired_returns[fired_returns < 0].sum())
                pf        = gains / losses if losses > 0 else (2.0 if gains > 0 else 0.0)
                score     = win_rate * math.log(max(total, 1))
                results.append({"period": period_l, "param": period_r,
                    "win_rate": round(win_rate, 4), "total_trades": total,
                    "score": round(score, 4), "profit_factor": round(min(pf, 99.0), 4),
                    "primary_beat_rate": round(prim_wins / total if total > 0 else 0, 4)})
    
    else:
        return None, f"Unsupported category: {cat}"

    if not results:
        return None, "No results generated (insufficient data)"

    df = pd.DataFrame(results)
    return df, None


# ---------------------------------------------------------------------------
# Fragility scoring
# ---------------------------------------------------------------------------

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
                    "pb": cell.iloc[0].get("primary_beat_rate", 0),
                })
        matrix.append(row)
    return {
        "periods": [str(p) for p in periods],
        "params":  [str(p) for p in params],
        "matrix":  matrix,
        "is_1d":   len(params) == 1,
    }


def fragility_label(score):
    if score < 0.15: return "Robust"
    if score < 0.35: return "Stable"
    if score < 0.55: return "Moderate"
    if score < 0.75: return "Fragile"
    return "Very Fragile"


def generate_html(conditions, sweep_results, fragility_scores, config):
    """Generate the full HTML report."""
    import json as jsonmod

    timestamp    = datetime.now().strftime("%Y-%m-%d %H:%M")
    json_name    = Path(config["json_path"]).name
    primary_asset = config.get("primary_asset", "TQQQ")

    # Build per-(cond, alloc) heatmap data dict  keyed as  "condId:alloc"
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
            "allocations":  allocs,
            "alloc_errors": alloc_errors,
        })

    heatmap_json   = jsonmod.dumps(heatmap_data)
    fragility_json = jsonmod.dumps({str(k): v for k, v in fragility_scores.items()})
    conditions_json = jsonmod.dumps(conds_for_js)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Strategy Fuzz Report — {json_name}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
  :root {{
    --bg:#0d0f14; --surface:#141720; --surface2:#1c2030; --border:#2a2f3e;
    --text:#c8cfe0; --muted:#5a6180; --accent:#4f8ef7;
    --green:#2ecc71; --yellow:#f1c40f; --orange:#e67e22; --red:#e74c3c; --purple:#8e44ad;
    --font-mono:'IBM Plex Mono',monospace; --font-sans:'IBM Plex Sans',sans-serif;
  }}
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:var(--bg);color:var(--text);font-family:var(--font-sans);font-size:14px;line-height:1.6;}}

  .layout{{display:grid;grid-template-columns:280px 1fr 260px;min-height:100vh;}}

  /* Sidebar */
  .sidebar{{background:var(--surface);border-right:1px solid var(--border);position:sticky;top:0;height:100vh;overflow-y:auto;display:flex;flex-direction:column;}}
  .sidebar-header{{padding:16px 16px 14px;border-bottom:1px solid var(--border);flex-shrink:0;}}
  .sidebar-title{{font-family:var(--font-mono);font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.12em;margin-bottom:3px;}}
  .sidebar-file{{font-family:var(--font-mono);font-size:12px;color:var(--accent);}}
  .sidebar-meta{{font-size:10px;color:var(--muted);margin-top:3px;}}
  .sidebar-body{{flex:1;overflow-y:auto;padding:8px 0;}}

  /* Help button */
  .sidebar-help{{padding:10px 16px;border-top:1px solid var(--border);flex-shrink:0;}}
  .help-btn{{width:100%;background:var(--surface2);border:1px solid var(--border);border-radius:4px;
    color:var(--muted);font-family:var(--font-mono);font-size:10px;padding:6px;cursor:pointer;text-align:left;}}
  .help-btn:hover{{color:var(--text);border-color:var(--accent);}}

  /* Tree — lazy groups */
  .tree-group-header{{
    display:flex;align-items:center;justify-content:space-between;
    padding:7px 16px;cursor:pointer;user-select:none;
    font-family:var(--font-mono);font-size:10px;color:var(--muted);
    text-transform:uppercase;letter-spacing:.1em;
    border-left:2px solid transparent;
    min-width:0;
  }}
  .tree-group-header:hover{{background:var(--surface2);color:var(--text);}}
  .tree-group-header.open{{color:var(--text);border-left-color:var(--accent);}}
  .tree-group-name{{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}}
  .tree-group-count{{font-size:9px;color:var(--muted);flex-shrink:0;margin-left:6px;}}
  .tree-group-body{{display:none;}}
  .tree-group-body.open{{display:block;}}

  .tree-node{{display:flex;align-items:center;gap:6px;padding:5px 16px;cursor:pointer;border-left:2px solid transparent;transition:background .12s;}}
  .tree-node:hover{{background:var(--surface2);}}
  .tree-node.active{{background:var(--surface2);border-left-color:var(--accent);}}
  .tree-indent{{display:inline-block;width:10px;flex-shrink:0;color:#2a2f3e;font-size:10px;}}
  .tree-dot{{width:7px;height:7px;border-radius:50%;flex-shrink:0;}}
  .tree-label{{font-family:var(--font-mono);font-size:10px;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:1;min-width:0;}}
  .tree-badge{{font-family:var(--font-mono);font-size:8px;padding:1px 4px;border-radius:3px;flex-shrink:0;}}

  /* Main */
  .main{{padding:28px 32px;overflow-x:hidden;border-right:1px solid var(--border);}}
  .page-header{{border-bottom:1px solid var(--border);padding-bottom:16px;margin-bottom:24px;}}
  .page-title{{font-family:var(--font-mono);font-size:18px;color:var(--text);font-weight:600;letter-spacing:-.02em;}}
  .page-subtitle{{font-size:11px;color:var(--muted);margin-top:4px;font-family:var(--font-mono);}}

  .cond-panel{{display:none;animation:fadeIn .18s ease;}}
  .cond-panel.visible{{display:block;}}
  @keyframes fadeIn{{from{{opacity:0;transform:translateY(5px);}}to{{opacity:1;transform:translateY(0);}}}}

  .cond-header{{display:flex;align-items:flex-start;gap:10px;margin-bottom:18px;}}
  .cond-badge{{font-family:var(--font-mono);font-size:10px;padding:3px 8px;border-radius:4px;white-space:nowrap;flex-shrink:0;margin-top:2px;}}
  .cond-title{{font-family:var(--font-mono);font-size:15px;color:var(--text);line-height:1.4;font-weight:600;}}
  .cond-meta{{font-size:11px;color:var(--muted);margin-top:3px;}}

  /* Compare bar */
  .compare-bar{{display:flex;align-items:center;gap:8px;background:var(--surface2);border:1px solid var(--border);border-radius:5px;padding:7px 12px;margin-bottom:18px;font-family:var(--font-mono);font-size:10px;}}
  .compare-bar.hidden{{display:none;}}
  .compare-slot{{flex:1;padding:3px 7px;border-radius:3px;background:var(--surface);border:1px solid var(--border);color:var(--text);min-height:22px;display:flex;align-items:center;gap:5px;font-size:10px;}}
  .compare-vs{{color:var(--muted);}}
  .compare-clear{{background:none;border:none;color:var(--muted);cursor:pointer;font-size:15px;padding:0 3px;}}
  .compare-clear:hover{{color:var(--red);}}

  /* Allocation tabs */
  .alloc-tabs{{display:flex;gap:4px;margin-bottom:16px;flex-wrap:wrap;}}
  .alloc-tab{{font-family:var(--font-mono);font-size:10px;padding:4px 10px;border-radius:4px;cursor:pointer;
    background:var(--surface2);border:1px solid var(--border);color:var(--muted);transition:all .12s;}}
  .alloc-tab:hover{{color:var(--text);border-color:var(--muted);}}
  .alloc-tab.active{{color:var(--text);border-color:var(--accent);background:var(--surface);}}
  .alloc-tab.error{{color:var(--red);border-color:rgba(231,76,60,.3);}}

  /* Charts */
  .chart-section{{margin-bottom:24px;}}
  .section-label{{font-family:var(--font-mono);font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px;}}
  .heatmap-wrap{{background:var(--surface);border:1px solid var(--border);border-radius:5px;padding:14px;overflow-x:auto;}}
  .heatmap-grid{{display:inline-grid;gap:2px;}}
  .heatmap-cell{{width:26px;height:26px;border-radius:3px;cursor:pointer;transition:transform .1s;border:2px solid transparent;}}
  .heatmap-cell:hover{{transform:scale(1.25);z-index:10;}}
  .heatmap-cell.pinned-a{{border-color:#4f8ef7!important;}}
  .heatmap-cell.pinned-b{{border-color:#f1c40f!important;}}
  .heatmap-cell.no-data{{background:var(--surface2)!important;opacity:.3;cursor:default;}}
  .heatmap-axis-label{{font-family:var(--font-mono);font-size:8px;color:var(--muted);display:flex;align-items:center;justify-content:center;}}
  .heatmap-legend{{display:flex;align-items:center;gap:8px;margin-top:8px;font-family:var(--font-mono);font-size:9px;color:var(--muted);}}
  .legend-gradient{{width:90px;height:6px;border-radius:3px;}}

  /* Right info panel */
  .info-panel{{background:var(--surface);padding:20px 14px;position:sticky;top:0;height:100vh;overflow-y:auto;}}
  .info-panel-title{{font-family:var(--font-mono);font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;margin-bottom:14px;}}
  .info-empty{{color:var(--muted);font-size:11px;font-family:var(--font-mono);text-align:center;margin-top:32px;line-height:2;}}
  .info-slot{{margin-bottom:16px;}}
  .info-slot-header{{display:flex;align-items:center;gap:7px;font-family:var(--font-mono);font-size:10px;font-weight:600;margin-bottom:8px;padding-bottom:6px;border-bottom:1px solid var(--border);}}
  .info-slot-dot{{width:7px;height:7px;border-radius:50%;flex-shrink:0;}}
  .info-stat{{display:flex;justify-content:space-between;align-items:baseline;padding:3px 0;border-bottom:1px solid #1c2030;}}
  .info-stat-label{{font-family:var(--font-mono);font-size:10px;color:var(--muted);}}
  .info-stat-value{{font-family:var(--font-mono);font-size:11px;font-weight:600;color:var(--text);}}
  .info-divider{{height:1px;background:var(--border);margin:12px 0;}}

  /* Help overlay */
  .help-overlay{{display:none;position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:100;align-items:center;justify-content:center;}}
  .help-overlay.visible{{display:flex;}}
  .help-box{{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:28px 32px;max-width:540px;width:90vw;max-height:80vh;overflow-y:auto;}}
  .help-box h2{{font-family:var(--font-mono);font-size:14px;color:var(--text);margin-bottom:18px;}}
  .help-box h3{{font-family:var(--font-mono);font-size:11px;color:var(--accent);text-transform:uppercase;letter-spacing:.1em;margin:16px 0 6px;}}
  .help-box p,.help-box li{{font-size:12px;color:var(--muted);line-height:1.7;}}
  .help-box li{{margin-left:16px;margin-bottom:3px;}}
  .help-box .metric{{display:flex;gap:10px;padding:5px 0;border-bottom:1px solid var(--border);}}
  .help-box .metric-name{{font-family:var(--font-mono);font-size:10px;color:var(--text);width:100px;flex-shrink:0;}}
  .help-box .metric-desc{{font-size:11px;color:var(--muted);}}
  .help-close{{float:right;background:none;border:none;color:var(--muted);font-size:20px;cursor:pointer;}}
  .help-close:hover{{color:var(--text);}}

  .error-box{{background:rgba(231,76,60,.1);border:1px solid rgba(231,76,60,.3);border-radius:5px;padding:12px;font-family:var(--font-mono);font-size:11px;color:var(--red);}}
  .welcome{{display:flex;flex-direction:column;align-items:center;justify-content:center;height:55vh;text-align:center;color:var(--muted);}}
  .welcome-icon{{font-size:36px;margin-bottom:14px;opacity:.3;}}
  .welcome-title{{font-family:var(--font-mono);font-size:14px;color:var(--text);margin-bottom:6px;}}
  .welcome-sub{{font-size:12px;}}
</style>
</head>
<body>
<div class="layout">

  <aside class="sidebar">
    <div class="sidebar-header">
      <div class="sidebar-title">Fuzz Report</div>
      <div class="sidebar-file">{json_name}</div>
      <div class="sidebar-meta">{timestamp} · {len(conditions)} conditions · primary: {primary_asset}</div>
    </div>
    <div class="sidebar-body" id="logic-tree"></div>
    <div class="sidebar-help">
      <button class="help-btn" onclick="document.getElementById('help-overlay').classList.add('visible')">
        ? How to read this report
      </button>
    </div>
  </aside>

  <main class="main">
    <div class="page-header">
      <div class="page-title">Parameter Robustness</div>
      <div class="page-subtitle">Win rate &amp; profit factor across ±fuzz sweep · click cell to pin · click second to compare</div>
    </div>
    <div class="compare-bar hidden" id="compare-bar">
      <div class="compare-slot" id="slot-a"><span style="color:#4f8ef7;margin-right:4px">▪</span><span id="slot-a-label">Pin a cell</span></div>
      <span class="compare-vs">vs</span>
      <div class="compare-slot" id="slot-b"><span style="color:#f1c40f;margin-right:4px">▪</span><span id="slot-b-label">Pin second</span></div>
      <button class="compare-clear" id="compare-clear">×</button>
    </div>
    <div id="content">
      <div class="welcome">
        <div class="welcome-icon">⬡</div>
        <div class="welcome-title">Select a condition</div>
        <div class="welcome-sub">Click any node in the sidebar tree</div>
      </div>
    </div>
  </main>

  <aside class="info-panel">
    <div class="info-panel-title">Cell Detail</div>
    <div id="info-content"><div class="info-empty">Hover or click<br>a cell to see<br>its stats here</div></div>
  </aside>
</div>

<!-- Help overlay -->
<div class="help-overlay" id="help-overlay">
  <div class="help-box">
    <button class="help-close" onclick="document.getElementById('help-overlay').classList.remove('visible')">×</button>
    <h2>How to Read This Report</h2>
    <h3>What this shows</h3>
    <p>Each condition in your strategy (RSI check, price/MA cross, etc.) is tested across a range of slightly different parameter values — the "fuzz". If the condition only works at one very specific value, it may be overfit. If it works across a broad range, it's likely robust.</p>

    <h3>The charts</h3>
    <p><b>Heatmap (2D conditions):</b> rows = indicator period, columns = threshold value. Each cell's colour shows win rate: red = low, green = high. A healthy condition shows a broad green plateau. A spike of green surrounded by red is a warning sign.</p>
    <p style="margin-top:6px"><b>Line chart (1D conditions):</b> Used for MA crosses and RSI-vs-RSI where there's no threshold to sweep — only the window length varies. A flat line = robust. A sharp peak = fragile.</p>

    <h3>Allocation tabs</h3>
    <p>When a condition gates multiple possible holdings, each gets its own tab. The sweep for each tab measures whether <em>that specific asset</em> outperforms BIL on days the condition fires.</p>

    <h3>Metrics</h3>
    <div class="metric"><div class="metric-name">Win Rate</div><div class="metric-desc">% of days the condition fired where the allocated asset returned more than BIL. &gt;55% is meaningful; &gt;65% is strong.</div></div>
    <div class="metric"><div class="metric-name">Profit Factor</div><div class="metric-desc">Sum of gains ÷ sum of losses on signal days. &gt;1.0 means the wins outweigh the losses in dollar terms. &gt;1.5 is solid.</div></div>
    <div class="metric"><div class="metric-name">Trades (n)</div><div class="metric-desc">How many days the condition fired. Low n (under ~20) means the win rate is unreliable — small samples swing wildly.</div></div>
    <div class="metric"><div class="metric-name">Score</div><div class="metric-desc">Win Rate × log(n). Penalises high win rates that come from tiny samples. Useful for ranking conditions against each other.</div></div>
    <div class="metric"><div class="metric-name">vs {primary_asset}</div><div class="metric-desc">% of signal days where the allocated asset beat {primary_asset} (your primary asset). Relevant when the condition is supposed to be an improvement over just holding {primary_asset}.</div></div>

    <h3>Fragility score (sidebar colours)</h3>
    <div class="metric"><div class="metric-name" style="color:#2ecc71">Robust</div><div class="metric-desc">Win rate is very consistent across the sweep. High confidence the condition is real.</div></div>
    <div class="metric"><div class="metric-name" style="color:#f1c40f">Stable</div><div class="metric-desc">Minor variation. Probably fine; worth a quick look.</div></div>
    <div class="metric"><div class="metric-name" style="color:#e67e22">Moderate</div><div class="metric-desc">Noticeable variation. The exact parameter values matter more than you'd like.</div></div>
    <div class="metric"><div class="metric-name" style="color:#e74c3c">Fragile</div><div class="metric-desc">Win rate spikes sharply at the fitted value. Consider widening the parameter or removing the condition.</div></div>
    <div class="metric"><div class="metric-name" style="color:#8e44ad">Very Fragile</div><div class="metric-desc">The condition appears heavily overfit. Treat with scepticism.</div></div>

    <h3>Comparing cells</h3>
    <p>Click any cell to pin it (blue border). Click a second cell to pin it (yellow border). The right panel shows both sets of stats plus a Δ row showing the difference.</p>
  </div>
</div>

<script>
const HEATMAP_DATA   = {heatmap_json};
const FRAGILITY      = {fragility_json};
const CONDITIONS     = {conditions_json};
const PRIMARY_ASSET  = "{primary_asset}";

let pinnedA = null, pinnedB = null, hoveredCell = null;
let currentCondId = null, currentAlloc = null;

// ── Colour helpers ──────────────────────────────────────────────────────────
function wrColor(wr) {{
  if (wr==null) return '#1c2030';
  const r=Math.round(231+(46-231)*wr), g=Math.round(76+(204-76)*wr), b=Math.round(60+(113-60)*wr);
  return `rgb(${{r}},${{g}},${{b}})`;
}}
function pfColor(pf) {{
  const t=Math.min(Math.max((pf-0.5)/1.5,0),1);
  const r=Math.round(231+(46-231)*t), g=Math.round(76+(204-76)*t), b=Math.round(60+(113-60)*t);
  return `rgb(${{r}},${{g}},${{b}})`;
}}
function fragColor(s) {{
  if(s<.15)return'#2ecc71'; if(s<.35)return'#f1c40f';
  if(s<.55)return'#e67e22'; if(s<.75)return'#e74c3c'; return'#8e44ad';
}}
function fragLabel(s) {{
  if(s<.15)return'Robust'; if(s<.35)return'Stable';
  if(s<.55)return'Moderate'; if(s<.75)return'Fragile'; return'Very Fragile';
}}
function pct(v){{return(v*100).toFixed(1)+'%';}}
function f2(v){{return(v==null||v===undefined)?'—':v.toFixed(2);}}

// ── Lazy sidebar tree ───────────────────────────────────────────────────────
function buildTree() {{
  const container = document.getElementById('logic-tree');
  const groups = {{}};
  CONDITIONS.forEach(c => {{
    const s = c.sub_strategy;
    if(!groups[s]) groups[s]=[];
    groups[s].push(c);
  }});

Object.entries(groups).forEach(([sub, conds]) => {{
    if (sub === '(root)' || sub === null) return;
    if (sub.length <= 2) return;

    const groupEl = document.createElement('div');

    const header = document.createElement('div');
    header.className = 'tree-group-header';
    const shortSub = sub.length > 32 ? sub.slice(0,30)+'…' : sub;
    header.innerHTML = `<span class="tree-group-name" title="${{sub}}">${{shortSub}}</span><span class="tree-group-count">${{conds.length}}</span>`;
    groupEl.appendChild(header);

    const body = document.createElement('div');
    body.className = 'tree-group-body';
    let rendered = false;

    header.addEventListener('click', () => {{
      const open = body.classList.toggle('open');
      header.classList.toggle('open', open);
      // Lazy render nodes only on first open
      if(open && !rendered) {{
        rendered = true;
        conds.forEach(c => body.appendChild(makeNode(c)));
      }}
    }});

    groupEl.appendChild(body);
    container.appendChild(groupEl);
  }});

}}

function makeNode(c) {{
  const fs = parseFloat(FRAGILITY[c.id] ?? 1.0);
  const color = fragColor(fs);
  const node = document.createElement('div');
  node.className = 'tree-node';
  node.dataset.id = c.id;

  for(let i=0;i<c.depth;i++) {{
    const sp=document.createElement('span');
    sp.className='tree-indent'; sp.textContent='·'; node.appendChild(sp);
  }}
  const dot=document.createElement('div');
  dot.className='tree-dot'; dot.style.background=color; node.appendChild(dot);
  const lbl=document.createElement('div');
  lbl.className='tree-label';
  lbl.title=c.human;
  lbl.textContent=c.human.length>26?c.human.slice(0,24)+'…':c.human;
  node.appendChild(lbl);
  const badge=document.createElement('div');
  badge.className='tree-badge';
  badge.style.cssText=`background:${{color}}22;color:${{color}}`;
  badge.textContent=fragLabel(fs); node.appendChild(badge);
  node.addEventListener('click', ()=>showCondition(c.id));
  return node;
}}

// ── Show condition ──────────────────────────────────────────────────────────
function showCondition(id) {{
  document.querySelectorAll('.tree-node').forEach(n=>n.classList.remove('active'));
  const an=document.querySelector(`.tree-node[data-id="${{id}}"]`);
  if(an) an.classList.add('active');

  const cond  = CONDITIONS.find(c=>c.id===id);
  const fs    = parseFloat(FRAGILITY[id]??1.0);
  const color = fragColor(fs);
  const allocs= cond.allocations||[];
  currentCondId = id;
  currentAlloc  = allocs[0]||null;

  const content = document.getElementById('content');
  const allocTabsHtml = allocs.length > 1
    ? `<div class="alloc-tabs" id="alloc-tabs">
        ${{allocs.map((a,i)=>{{
          const hasErr = cond.alloc_errors&&cond.alloc_errors[a];
          const noData = !HEATMAP_DATA[id+':'+a];
          return `<div class="alloc-tab${{hasErr||noData?' error':''}}${{i===0?' active':''}}" data-alloc="${{a}}" onclick="switchAlloc(${{id}},'${{a}}')">${{a}}</div>`;
        }}).join('')}}
      </div>`
    : '';

  content.innerHTML = `
    <div class="cond-panel visible" id="panel-${{id}}">
      <div class="cond-header">
        <div class="cond-badge" style="background:${{color}}22;color:${{color}}">${{fragLabel(fs)}} (${{(fs*100).toFixed(0)}}%)</div>
        <div>
          <div class="cond-title">${{cond.human}}</div>
          <div class="cond-meta">${{cond.sub_strategy}} · ${{cond.category}} · allocates to: ${{allocs.join(', ')||'—'}}</div>
        </div>
      </div>
      ${{allocTabsHtml}}
      <div id="charts-area"></div>
    </div>`;

  renderChartsForAlloc(id, currentAlloc, cond);
}}

function switchAlloc(condId, alloc) {{
  document.querySelectorAll('.alloc-tab').forEach(t=>t.classList.toggle('active', t.dataset.alloc===alloc));
  currentAlloc = alloc;
  const cond = CONDITIONS.find(c=>c.id===condId);
  renderChartsForAlloc(condId, alloc, cond);
}}

function renderChartsForAlloc(condId, alloc, cond) {{
  const area = document.getElementById('charts-area');
  if(!area) return;
  const key  = condId+':'+(alloc||'(none)');
  const hd   = HEATMAP_DATA[key];

  if(!hd) {{
    const err = cond.alloc_errors&&cond.alloc_errors[alloc] || 'No data available';
    area.innerHTML = `<div class="error-box">⚠ ${{alloc}}: ${{err}}</div>`;
    return;
  }}

  const is1d = hd.is_1d;
  area.innerHTML = `
    <div class="chart-section">
      <div class="section-label">${{is1d?'Win Rate by Window':'Win Rate · Period (rows) × Threshold (cols)'}}</div>
      <div class="heatmap-wrap"><div id="wr-c"></div>
        ${{is1d?'':'<div class="heatmap-legend"><span>Low</span><div class="legend-gradient" style="background:linear-gradient(to right,#e74c3c,#f1c40f,#2ecc71)"></div><span>High</span></div>'}}
      </div>
    </div>
    <div class="chart-section">
      <div class="section-label">${{is1d?'Profit Factor by Window':'Profit Factor · Period (rows) × Threshold (cols)'}}</div>
      <div class="heatmap-wrap"><div id="pf-c"></div>
        ${{is1d?'':'<div class="heatmap-legend"><span>Low</span><div class="legend-gradient" style="background:linear-gradient(to right,#e74c3c,#f1c40f,#2ecc71)"></div><span>High</span></div>'}}
      </div>
    </div>`;

  if(is1d) {{
    renderLine('wr-c', hd, 'wr', wrColor, v=>pct(v));
    renderLine('pf-c', hd, 'pf', pfColor, v=>f2(v));
  }} else {{
    renderHeatmap('wr-c', hd, 'wr', condId);
    renderHeatmap('pf-c', hd, 'pf', condId);
  }}
}}

// ── Info panel ──────────────────────────────────────────────────────────────
function renderInfo() {{
  const el = document.getElementById('info-content');
  if(!pinnedA && !hoveredCell) {{
    el.innerHTML='<div class="info-empty">Hover or click<br>a cell to see<br>its stats here</div>';
    return;
  }}
  function slotHtml(pin, label, dotColor) {{
    if(!pin) return '';
    const c=pin.cell;
    return `<div class="info-slot">
      <div class="info-slot-header"><div class="info-slot-dot" style="background:${{dotColor}}"></div>${{label}} · P:${{pin.period}} T:${{pin.param}}</div>
      <div class="info-stat"><span class="info-stat-label">Win Rate</span><span class="info-stat-value" style="color:${{wrColor(c.wr)}}">${{pct(c.wr)}}</span></div>
      <div class="info-stat"><span class="info-stat-label">Profit Factor</span><span class="info-stat-value" style="color:${{pfColor(c.pf)}}">${{f2(c.pf)}}</span></div>
      <div class="info-stat"><span class="info-stat-label">Trades (n)</span><span class="info-stat-value">${{c.n}}</span></div>
      <div class="info-stat"><span class="info-stat-label">Score</span><span class="info-stat-value">${{f2(c.s)}}</span></div>
      <div class="info-stat"><span class="info-stat-label">vs ${{PRIMARY_ASSET}}</span><span class="info-stat-value" style="color:${{wrColor(c.pb)}}">${{pct(c.pb)}}</span></div>
    </div>`;
  }}
  let html='';
  if(hoveredCell && !pinnedA) {{
    html=slotHtml(hoveredCell,'Hover','#888');
  }} else if(pinnedA && !pinnedB) {{
    html=slotHtml(pinnedA,'Pinned','#4f8ef7');
    if(hoveredCell){{ html+='<div class="info-divider"></div>'+slotHtml(hoveredCell,'Hover','#888'); }}
  }} else if(pinnedA && pinnedB) {{
    html=slotHtml(pinnedA,'A','#4f8ef7')+'<div class="info-divider"></div>'+slotHtml(pinnedB,'B','#f1c40f');
    const dw=pinnedB.cell.wr-pinnedA.cell.wr, dp=pinnedB.cell.pf-pinnedA.cell.pf, db=pinnedB.cell.pb-pinnedA.cell.pb;
    const s=v=>v>=0?'+':'';
    html+=`<div class="info-divider"></div><div class="info-slot">
      <div class="info-slot-header" style="color:var(--muted)">Δ B minus A</div>
      <div class="info-stat"><span class="info-stat-label">Win Rate</span><span class="info-stat-value" style="color:${{dw>=0?'#2ecc71':'#e74c3c'}}">${{s(dw)}}${{pct(dw)}}</span></div>
      <div class="info-stat"><span class="info-stat-label">Profit Factor</span><span class="info-stat-value" style="color:${{dp>=0?'#2ecc71':'#e74c3c'}}">${{s(dp)}}${{f2(dp)}}</span></div>
      <div class="info-stat"><span class="info-stat-label">vs ${{PRIMARY_ASSET}}</span><span class="info-stat-value" style="color:${{db>=0?'#2ecc71':'#e74c3c'}}">${{s(db)}}${{pct(db)}}</span></div>
    </div>`;
  }}
  el.innerHTML=html;
}}

// ── Cell pin ────────────────────────────────────────────────────────────────
function handlePin(period, param, cell) {{
  if(!pinnedA) {{
    pinnedA={{period,param,cell}}; updateCompareBar(); renderInfo(); return;
  }}
  if(pinnedA.period===period&&pinnedA.param===param) {{
    pinnedA=null; pinnedB=null; updateCompareBar(); renderInfo(); return;
  }}
  if(!pinnedB) {{
    pinnedB={{period,param,cell}}; updateCompareBar(); renderInfo(); return;
  }}
  pinnedA=pinnedB; pinnedB={{period,param,cell}}; updateCompareBar(); renderInfo();
}}

function updateCompareBar() {{
  const bar=document.getElementById('compare-bar');
  document.getElementById('slot-a-label').textContent=pinnedA?`P:${{pinnedA.period}} T:${{pinnedA.param}}`:'Pin a cell';
  document.getElementById('slot-b-label').textContent=pinnedB?`P:${{pinnedB.period}} T:${{pinnedB.param}}`:'Pin second';
  bar.classList.toggle('hidden',!pinnedA);
}}

document.addEventListener('DOMContentLoaded',()=>{{
  document.getElementById('compare-clear').addEventListener('click',()=>{{
    pinnedA=null; pinnedB=null; updateCompareBar(); renderInfo();
  }});
}});

// ── Heatmap renderer ────────────────────────────────────────────────────────
function renderHeatmap(cid, hd, metric, condId) {{
  const container=document.getElementById(cid);
  if(!container) return;
  const colorFn=metric==='wr'?wrColor:pfColor;
  const grid=document.createElement('div');
  grid.className='heatmap-grid';
  grid.style.gridTemplateColumns=`24px ${{hd.params.map(()=>'26px').join(' ')}}`;

  // Header row
  grid.appendChild(document.createElement('div'));
  hd.params.forEach(p=>{{
    const el=document.createElement('div');
    el.className='heatmap-axis-label'; el.textContent=p; grid.appendChild(el);
  }});

  hd.matrix.forEach((row,ri)=>{{
    const rl=document.createElement('div');
    rl.className='heatmap-axis-label';
    rl.style.cssText='justify-content:flex-end;padding-right:3px;';
    rl.textContent=hd.periods[ri]; grid.appendChild(rl);

    row.forEach((cell,ci)=>{{
      const el=document.createElement('div');
      el.className='heatmap-cell'+(cell?'':' no-data');
      if(cell){{
        const val=metric==='wr'?cell.wr:cell.pf;
        el.style.background=colorFn(val);
        const p=hd.periods[ri], pa=hd.params[ci];
        if(pinnedA&&pinnedA.period==p&&pinnedA.param==pa) el.classList.add('pinned-a');
        if(pinnedB&&pinnedB.period==p&&pinnedB.param==pa) el.classList.add('pinned-b');
        el.addEventListener('mouseenter',()=>{{hoveredCell={{period:p,param:pa,cell}};renderInfo();}});
        el.addEventListener('mouseleave',()=>{{hoveredCell=null;renderInfo();}});
        el.addEventListener('click',()=>{{
          handlePin(p,pa,cell);
          // Refresh pin borders in all current charts
          document.querySelectorAll('.heatmap-cell.pinned-a,.heatmap-cell.pinned-b')
            .forEach(e=>e.classList.remove('pinned-a','pinned-b'));
          if(pinnedA) document.querySelectorAll(`.heatmap-cell[data-p="${{pinnedA.period}}"][data-pa="${{pinnedA.param}}"]`).forEach(e=>e.classList.add('pinned-a'));
          if(pinnedB) document.querySelectorAll(`.heatmap-cell[data-p="${{pinnedB.period}}"][data-pa="${{pinnedB.param}}"]`).forEach(e=>e.classList.add('pinned-b'));
        }});
        el.dataset.p=p; el.dataset.pa=pa;
      }}
      grid.appendChild(el);
    }});
  }});
  container.appendChild(grid);
}}

// ── Line chart renderer ─────────────────────────────────────────────────────
function renderLine(cid, hd, metric, colorFn, fmtFn) {{
  const container=document.getElementById(cid);
  if(!container) return;
  const pts=hd.matrix.map((row,i)=>row[0]?{{
    x:parseFloat(hd.periods[i]),v:metric==='wr'?row[0].wr:row[0].pf,
    wr:row[0].wr,pf:row[0].pf,pb:row[0].pb,n:row[0].n,s:row[0].s,
    period:hd.periods[i],param:hd.params[0],
  }}:null).filter(Boolean);
  if(!pts.length) return;

  const W=520,H=170,PAD={{t:12,r:14,b:30,l:42}};
  const cW=W-PAD.l-PAD.r, cH=H-PAD.t-PAD.b;
  const xs=pts.map(p=>p.x), vs=pts.map(p=>p.v);
  const xMin=Math.min(...xs),xMax=Math.max(...xs);
  const yMin=Math.max(0,Math.min(...vs)*.96), yMax=Math.min(metric==='wr'?1:99,Math.max(...vs)*1.04);
  const tx=x=>PAD.l+((x-xMin)/(xMax-xMin||1))*cW;
  const ty=v=>PAD.t+(1-(v-yMin)/(yMax-yMin||1))*cH;

  const canvas=document.createElement('canvas');
  canvas.width=W; canvas.height=H;
  canvas.style.cssText=`width:${{W}}px;height:${{H}}px;display:block;`;
  container.appendChild(canvas);
  const ctx=canvas.getContext('2d');

  // Grid lines
  ctx.strokeStyle='#2a2f3e'; ctx.lineWidth=1;
  [.25,.5,.75].forEach(f=>{{ const y=PAD.t+f*cH; ctx.beginPath();ctx.moveTo(PAD.l,y);ctx.lineTo(PAD.l+cW,y);ctx.stroke(); }});

  // Reference line (50% for WR, 1.0 for PF)
  const ref=metric==='wr'?.5:1.0, yRef=ty(ref);
  if(yRef>=PAD.t&&yRef<=PAD.t+cH){{
    ctx.strokeStyle='#4f8ef744';ctx.setLineDash([4,4]);
    ctx.beginPath();ctx.moveTo(PAD.l,yRef);ctx.lineTo(PAD.l+cW,yRef);ctx.stroke();
    ctx.setLineDash([]);
  }}

  // Fill
  const grad=ctx.createLinearGradient(0,PAD.t,0,PAD.t+cH);
  grad.addColorStop(0,'rgba(79,142,247,.18)');grad.addColorStop(1,'rgba(79,142,247,.01)');
  ctx.fillStyle=grad;
  ctx.beginPath();ctx.moveTo(tx(pts[0].x),PAD.t+cH);
  pts.forEach(p=>ctx.lineTo(tx(p.x),ty(p.v)));
  ctx.lineTo(tx(pts[pts.length-1].x),PAD.t+cH);ctx.closePath();ctx.fill();

  // Line
  ctx.strokeStyle='#4f8ef7';ctx.lineWidth=2;ctx.lineJoin='round';
  ctx.beginPath();pts.forEach((p,i)=>i===0?ctx.moveTo(tx(p.x),ty(p.v)):ctx.lineTo(tx(p.x),ty(p.v)));ctx.stroke();

  // Dots
  pts.forEach(p=>{{ctx.beginPath();ctx.arc(tx(p.x),ty(p.v),3,0,Math.PI*2);ctx.fillStyle=colorFn(p.v);ctx.fill();}});

  // Y labels
  ctx.fillStyle='#5a6180';ctx.font='9px IBM Plex Mono,monospace';ctx.textAlign='right';
  [yMin,(yMin+yMax)/2,yMax].forEach(v=>ctx.fillText(fmtFn(v),PAD.l-4,ty(v)+3));

  // X labels
  ctx.textAlign='center';
  const step=Math.ceil(pts.length/7);
  pts.forEach((p,i)=>{{if(i%step===0||i===pts.length-1) ctx.fillText(p.x,tx(p.x),PAD.t+cH+12);}});
  ctx.fillText('Window',PAD.l+cW/2,H-2);

  canvas.addEventListener('mousemove',e=>{{
    const r=canvas.getBoundingClientRect(), mx=e.clientX-r.left;
    let cl=null,md=Infinity;
    pts.forEach(p=>{{const d=Math.abs(tx(p.x)-mx);if(d<md){{md=d;cl=p;}}}});
    if(cl&&md<28){{hoveredCell={{period:cl.period,param:cl.param,cell:{{wr:cl.wr,pf:cl.pf,pb:cl.pb,n:cl.n,s:cl.s}}}};renderInfo();}}
    else{{hoveredCell=null;renderInfo();}}
  }});
  canvas.addEventListener('mouseleave',()=>{{hoveredCell=null;renderInfo();}});
  canvas.addEventListener('click',e=>{{
    const r=canvas.getBoundingClientRect(),mx=e.clientX-r.left;
    let cl=null,md=Infinity;
    pts.forEach(p=>{{const d=Math.abs(tx(p.x)-mx);if(d<md){{md=d;cl=p;}}}});
    if(cl&&md<28) handlePin(cl.period,cl.param,{{wr:cl.wr,pf:cl.pf,pb:cl.pb,n:cl.n,s:cl.s}});
  }});
}}

// ── Init ────────────────────────────────────────────────────────────────────
buildTree();
</script>
</body>
</html>"""

    return html




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config = gather_inputs()

    # Load API keys
    print("\n  Loading config and API keys...")
    _, api_keys = load_config(config_dict={"dummy": True})

    # Parse strategy JSON
    print(f"  Parsing {config['json_path']}...")
    with open(config["json_path"], "r", encoding="utf-8") as f:
        tree = json.load(f)

    conditions = extract_conditions_from_tree(tree)
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

    # Load BIL and primary returns once
    bil_returns     = get_bil_daily_returns(config["start_date"], config["end_date"])
    primary_returns = get_primary_daily_returns(config["primary_asset"], config["start_date"], config["end_date"])

    # Run sweeps — one per (condition × allocation)
    # sweep_results keyed by (cond_id, allocation_ticker)
    # fragility_scores keyed by cond_id — worst fragility across allocations
    sweep_results    = {}   # (cond_id, alloc) -> df or error string
    fragility_scores = {}   # cond_id -> float

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
            df, err = sweep_condition(cond, config, bil_returns, primary_returns, endpoint=alloc)
            if err:
                print(f"    ⚠  {alloc}: {err}")
                sweep_results[key] = err
            else:
                sweep_results[key] = df
                fs = compute_fragility(df)
                worst_fragility = max(worst_fragility, fs)
                print(f"    → {alloc}: Fragility {fs:.3f} ({fragility_label(fs)})")

        fragility_scores[cond["id"]] = worst_fragility if allocs != [None] else 1.0

    # Generate HTML
    print("\n  Generating HTML report...")
    html = generate_html(conditions, sweep_results, fragility_scores, config)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_stem = Path(config["json_path"]).stem
    out_path = SCRIPT_DIR / f"fuzz_report_{json_stem}_{ts}.html"
    out_path.write_text(html, encoding="utf-8")

    print(f"\n  ✓ Report saved to: {out_path}\n")


if __name__ == "__main__":
    main()