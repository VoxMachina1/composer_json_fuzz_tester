# HANDOFF.md — composer_json_fuzz_tester

---

## Project Brief

**Goal:** Given a Composer.Trade strategy exported as JSON, extract every IF condition and measure how robust each one is to parameter changes. Output is a single self-contained HTML report with interactive heatmaps, a fragility-ranked sidebar, and a per-cell stat panel. This list of elements will grow as the end user interacts with it.

**Current status:** Working and in active use. Core sweep engine is complete. Several bugs were fixed in the most recent session. A small number of known issues remain (see below).

**What "done" looks like:** Every condition type in any Composer JSON is correctly extracted, swept, and rendered in the report with no silent failures. Several weeks/months of usser input has been parsed and  used to inform changes and/or new features.

---

## Architecture

```
fuzz_tester.py              ← single entrypoint, ~1913 lines
│
├── extract_conditions_from_tree()   walks strategy JSON, emits condition dicts
├── sweep_condition()                runs the parameter sweep for one condition
├── compute_fragility()              scores sweep result 0–1
├── generate_html()                  builds the full self-contained HTML report
└── main()                           prompts user, orchestrates everything

strategy_engine/src/        ← shared engine modules (also used by sibling project rsi_search)
├── config_loader.py         loads .env API keys
├── data_loader.py           Tiingo download + freshness checks
├── data_alignment.py        multi-asset DataFrame builder
└── indicators.py            RSI, SMA, EMA, CumRet implementations

pathfinder/
├── strategy.json            user's strategy export (gitignored)
└── strategy_paths.py        strategy tree walker (used by sibling project, not fuzz_tester directly)

strategy_engine/data/        downloaded price CSVs, gitignored, auto-generated on first run
```

**Data flow:**
1. User runs `python fuzz_tester.py`, prompted for config
2. Strategy JSON parsed → conditions extracted as flat list of dicts
3. All tickers collected → Tiingo downloads any missing CSVs
4. For each condition × allocation: `sweep_condition()` runs grid, returns DataFrame
5. `compute_fragility()` scores each result
6. `generate_html()` serializes everything to JSON, embeds in HTML template
7. Report written to `fuzz_report_{json_stem}_{timestamp}.html`

---

## Condition Extraction

Each condition dict has these keys:
```python
{
    "id":                 int,           # index in results list
    "sub_strategy":       str,           # group name from JSON tree
    "depth":              int,           # nesting depth
    "human":              str,           # e.g. "RSI(QQQ, 10) > 79.0"
    "comparator":         str,           # "gt", "lt", "gte", etc.
    "lhs":                dict,          # {"type": "indicator", "fn_label": "RSI", "ticker": "QQQ", "window": 10}
    "rhs":                dict,          # {"type": "fixed", "value": 79.0} or same as lhs
    "category":           str,           # "RSI_fixed", "RSI_vs_RSI", "MA_vs_MA", etc.
    "children_endpoints": list[str],     # allocation tickers reachable if condition fires
}
```

**Supported categories and their sweep type:**

| Category | Sweep |
|---|---|
| RSI_fixed | 2D: period × threshold |
| RSI_vs_RSI | 2D: lhs_period × rhs_period |
| CumRet_fixed | 2D: period × threshold |
| CumRet_vs_CumRet | 2D: lhs_period × rhs_period |
| MaxDD_fixed | 2D: period × threshold |
| MaxDD_vs_MaxDD | 2D: lhs_period × rhs_period |
| Price_vs_MA | 1D: MA window |
| MA_vs_MA | 2D: short window × long window |
| Price_vs_EMA | 1D: EMA window |
| EMA_vs_MA | 2D: EMA window × MA window |
| EMA_vs_EMA | 2D: lhs_period × rhs_period |
| MAReturn_fixed / MAReturn_vs_MAReturn | 2D for vs-family; MAReturn has dedicated calc |

---

## Current Priorities

1. **2D sweep coverage expansion (HIGH)** — end-state target is "everything 2D where feasible." Today, mixed categories still include 1D sweeps (e.g. `*_vs_*` same-window comparisons). Extend sweep logic to independent left/right window grids where this is mathematically and computationally practical.

2. **Validate extraction parity on large real-world strategies (HIGH)** — run condition-count and category audits against representative Composer exports to verify no shape-dependent drops and no malformed placeholder conditions.

3. **Refine drawdown/return-family UX labels (MEDIUM)** — now that `MaxDD` and `MAReturn` are computed separately, ensure labels, legends, and help text clearly distinguish units/interpretation.

4. **Rate-limit resilience tuning (MEDIUM)** — current Tiingo retry/backoff and pacing are in place; monitor with larger ticker sets and tune retries/sleep if needed.

### Status Snapshot (2026-04-27)

- Condition extraction now supports direct `if-child` conditions and nested payload conditions (`compound`, `binary-compound`, `binary`) without malformed placeholder entries.
- `MaxDD` now uses strict rolling peak-to-trough logic (order-aware), not simple max/min range.
- `MAReturn` now has its own calculation (`pct_change().rolling(period).mean()`) and no longer routes to max drawdown.
- Tiingo data loader fixed for production use (`SPY` market-date probe + `safe_ticker` filename handling), with 429 exponential backoff and post-download pacing.
- Verified against `pathfinder/any_all_example.json` and `pathfinder/bestsignals3.json` with clean extraction output (no unknown/empty conditions).

---

## Known Pain Points / Bugs

- **1D sweep coverage remains in several `*_vs_*` categories** — roadmap is 2D everywhere feasible.
- **`HANDOFF.md` drift risk** — this file can become stale quickly after code sessions; keep `Status Snapshot` authoritative and update this section in the same pass as code changes.
- **Tiingo free-tier sensitivity still exists at high ticker counts** — retries/pacing are in place, but large cold-start runs may still need tuning.
- **Report labeling for return-family metrics may be ambiguous** — `MaxDD` and `MAReturn` are now distinct calculations, but UI text should keep emphasizing unit/interpretation differences.

---

## Recent Decisions (Decision Log)

- **Single file architecture** — `fuzz_tester.py` is intentionally one large file. The HTML report generator is embedded as a Python f-string template. Don't split it unless there's a strong reason; the self-contained report depends on this.
- **BIL as win-rate baseline** — all win rates measure the allocated asset beating BIL (cash proxy), not SPY. This is intentional — we want to know if the condition identifies days worth being invested at all, not just days that beat the market.
- **Fragility = CV of win rate** — coefficient of variation (std/mean) of win rate across sweep points, capped at 1.0. Higher = more fragile.
- **Deduplicate allocations** — `_collect_endpoints` walks the full if-child subtree but uses a `seen` set to avoid duplicates. Without this, a condition gating 8 pairs would produce 16 allocation tabs (TQQQ appearing for each losing side).
- **Short group names hidden in sidebar** — groups with names ≤ 2 chars are skipped in `buildTree()` JS. Some Composer strategies have unnamed groups that get assigned numeric IDs ("1", "2") as names.
- **Ticker sanitization** — tickers with `/` (e.g. `BRK/A`) are normalized to `-` (`BRK-A`) before any file path or API URL construction. Applied in both `data_loader.py` and `data_alignment.py`.
- **Output filename includes JSON stem** — report files are named `fuzz_report_{json_stem}_{timestamp}.html` so runs on different strategies don't all look the same.

---

## Constraints / Don't Touch

- The HTML report is a single f-string in `generate_html()`. All JS uses `{{}}` for literal braces (Python f-string escaping). Don't introduce regular `{}` in the JS without escaping.
- `fuzz_tester.py` imports from `strategy_engine/src/` by inserting it into `sys.path`. Don't restructure imports without testing.
- `pathfinder/strategy_paths.py` is NOT used by `fuzz_tester.py` directly — it's used by the sibling `rsi_search` project. `fuzz_tester.py` has its own condition extractor (`extract_conditions_from_tree`).
# Dev note - we need to confirm the truth of the above point about strategy_paths.py. The current project is forked from a separate project (which Claude calls its "sibling"). There may be crossover and leftover code, but removing things from /fuzz_tester will not break /rsi_tester.

---

## Run Commands

```bash
# Setup (first time)
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Mac/Linux
pip install -r requirements.txt

# Create API key file
# Create strategy_engine/.env with:
# TIINGO_API_KEYS=your_key_here

# Run
python fuzz_tester.py
```

---

## Key Files to Read First

1. `README.md` — setup and usage
2. `fuzz_tester.py` — entire project (entrypoint + engine + report generator)
3. `strategy_engine/src/data_loader.py` — Tiingo download logic

---

## Suggested First Task

Run a full-strategy regression pass (large Composer exports) and review runtime impact now that all major `*_vs_*` families have 2D sweeps. If runtime is too high, add bounded-grid controls or adaptive sampling.


## This document *may* not be complete, as Claude ran out of tokens and failed to output a succsess message.