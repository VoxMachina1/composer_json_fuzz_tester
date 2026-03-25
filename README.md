# RSI Search — Frontrunner Discovery & Strategy Insertion Engine

A modular, zero-lookahead bias, multi-asset Python backtesting engine built for discovering alpha across large RSI parameter grids. Includes a full automated pipeline for extracting boolean paths from Composer.Trade strategies, backtesting frontrunner candidates against each path, filtering results, and inserting validated logic back into the strategy JSON.

---

## Project Structure

```
rsi_search/
├── run_analysis.py              # Step 1 — runs the full backtest pipeline
│
├── pathfinder/
│   ├── strategy.json            # Your Composer strategy export (gitignored)
│   ├── strategy_paths.py        # Extracts all boolean paths + node IDs
│   └── paths.txt                # Auto-generated path output
│
├── strategy_filter/
│   ├── filter_results.py        # Step 2 — filters and combines result CSVs
│   ├── filtered.csv             # Combined filtered output (gitignored)
│   └── filter_summary.txt       # Summary of last filter run (gitignored)
│
├── strategy_inserter/
│   ├── strategy_inserter.py     # Step 3 — inserts frontrunner logic into JSON
│   ├── strategy_modified.json   # Modified strategy output (gitignored)
│   └── insertion_log.json       # Log of all insertions made (gitignored)
│
└── strategy_engine/
    ├── .env                     # Tiingo API keys (gitignored, never commit)
    ├── config/
    │   └── template.yaml        # All engine settings
    ├── data/                    # Auto-managed price CSVs (gitignored)
    ├── results/                 # Timestamped result CSVs (gitignored)
    └── src/
        ├── config_loader.py
        ├── data_alignment.py
        ├── data_loader.py
        ├── indicators.py
        ├── metrics.py
        ├── preconditions.py
        ├── range_tester.py
        ├── signals.py
        └── strategy_engine.py
```

---

## Setup & Installation

1. Clone the repository and navigate to the project root.
2. Create and activate a Python virtual environment:

   **Windows (PowerShell):**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **Linux / macOS:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file inside `strategy_engine/` with your Tiingo API keys:
   ```
   TIINGO_API_KEYS=key1,key2,key3
   ```

5. Place your Composer strategy export at `pathfinder/strategy.json`.

---

## Full Pipeline

The pipeline runs in three steps. Each step produces output that feeds the next.

### Step 1 — Backtest (`run_analysis.py`)

Run twice — once for overbought signals (`>`), once for oversold (`<`).

**Run 1 — Overbought:**

In `strategy_engine/config/template.yaml`, set:
```yaml
signal_operator: ">"
threshold_start: 50.0
threshold_end: 80.0
```
Then run:
```bash
python run_analysis.py
```

**Run 2 — Oversold:**

Flip the operator and threshold range in `template.yaml`:
```yaml
signal_operator: "<"
threshold_start: 20.0
threshold_end: 50.0
```
Then run:
```bash
python run_analysis.py
```

Each run produces a timestamped CSV in `strategy_engine/results/` with a `signal_operator` column recording which operator was used.

---

### Step 2 — Filter (`strategy_filter/filter_results.py`)

Automatically finds the two most recent CSVs in `strategy_engine/results/`, validates that one is all `>` and one is all `<`, applies performance filters, and combines them into a single `filtered.csv`.

```bash
python strategy_filter/filter_results.py
```

**Filter criteria (hardcoded):**
- `Win_Rate > 0.75`
- `Total_Trades > 20`
- `Benchmark_Median_Return < 0`

Output: `strategy_filter/filtered.csv` and `strategy_filter/filter_summary.txt`

---

### Step 3 — Insert (`strategy_inserter/strategy_inserter.py`)

Reads `strategy_filter/filtered.csv` and inserts frontrunner logic into `pathfinder/strategy.json` at the exact validated nodes. Both overbought and oversold signals are processed in a single pass.

```bash
# Dry run first — no files written
python strategy_inserter/strategy_inserter.py --dry-run

# Full run
python strategy_inserter/strategy_inserter.py
```

Output: `strategy_inserter/strategy_modified.json` and `strategy_inserter/insertion_log.json`

Import `strategy_modified.json` into Composer.Trade to test the modified strategy.

---

## How the Inserter Works

For each unique `(sub_strategy, conditions, endpoint)` group in the filtered CSV, the matching leaf asset node in the strategy JSON is replaced with a `wt-cash-equal` block. Each unique `(signal_asset, operator, most_inclusive_threshold)` triple becomes one `if` node:

```
wt-cash-equal
  IF XLF_RSI_10 > 24    ->  [TQQQ, QQQ]     (overbought, shared threshold)
    ELSE -> PSQ
  IF XLF_RSI_10 < 25.5  ->  [TQQQ, QQQ]     (oversold, shared threshold)
    ELSE -> PSQ
  IF IOO_RSI_10 > 31    ->  [SOXL, UPRO]
    ELSE -> PSQ
```

**Threshold deduplication** — for each `(signal_asset, operator, target)` triple, only the most inclusive threshold is kept:
- `>` operator: minimum threshold (fires most often)
- `<` operator: maximum threshold (fires most often)

**Tautology detection** — if the same signal asset appears with both `>` and `<` at thresholds that together always evaluate true (e.g. RSI > 16 OR RSI < 18.5), that signal asset is skipped with a warning.

**Target grouping** — targets sharing the same `(signal_asset, operator, threshold)` are listed as equal-weight siblings in the same `if` node's true branch.

---

## Path Extractor — `pathfinder/strategy_paths.py`

Parses a Composer strategy JSON and emits every possible boolean path to every leaf endpoint, grouped by ticker. Also writes `pathfinder/paths.txt`.

```bash
python pathfinder/strategy_paths.py
```

Each path shows the sub-strategy sleeve, full condition chain, endpoint ticker, and the JSON node ID of the leaf asset node:
```
[Bond Compares] IF Price(SPY) > MA(SPY, 200) AND RSI(TLT, 20) > RSI(PSQ, 20) -> TQQQ  (node_id: a3c0604a-...)
[FTLT] IF Price(SPY) > MA(SPY, 200) AND RSI(QQQ, 10) <= 79 -> TQQQ  (node_id: beb628c6-...)
```

Asset-selection filters are expanded into explicit pairwise conditions:
```
RSI(SQQQ, 10) > RSI(TLT, 10) -> SQQQ
RSI(TLT, 10) > RSI(SQQQ, 10) -> TLT
```

---

## Output CSV Columns

| Column | Description |
|---|---|
| `sub_strategy` | Strategy sleeve (Bond Compares, KMLM, etc.) |
| `conditions` | Human-readable boolean condition chain |
| `engine_precondition` | pandas-compatible precondition string |
| `endpoint` | Strategy endpoint asset (serves as benchmark) |
| `benchmark_asset` | Same as endpoint |
| `target_asset` | Candidate frontrunner asset being tested |
| `signal_asset` | Asset whose RSI drives the signal |
| `signal_operator` | RSI comparison operator (`>` or `<`) |
| `threshold` | RSI threshold being tested |
| `slippage_bps` | Slippage assumption in basis points |
| `risk_free_rate` | Risk-free rate for Sharpe/Sortino |
| `Total_Trades` | Days signal was active |
| `Win_Rate` | % of signal-active days target beat benchmark |
| `Avg_Return` | Mean daily return on signal-active days |
| `Median_Return` | Median daily return on signal-active days |
| `Benchmark_Avg_Return` | Mean daily benchmark return on signal-active days |
| `Benchmark_Median_Return` | Median daily benchmark return on signal-active days |
| `Total_Return` | Cumulative return over backtest window |
| `Annualized_Return` | Annualized return |
| `Sharpe_Ratio` | Sharpe ratio |
| `Sortino_Ratio` | Sortino ratio |
| `Calmar_Ratio` | Calmar ratio |
| `Max_Drawdown` | Maximum drawdown |
| `Final_Equity` | Final equity multiplier (1.0 = no change) |
| `Avg_Hold_Days` | Average consecutive days signal stays active |

---

## template.yaml Reference

```yaml
# Candidate frontrunner target assets
target_assets: ["BIL", "UVXY"]

benchmark_asset: "SPY"       # Manual runs only
indicator: "RSI"
indicator_period: 10

# Flip between runs: ">" for overbought, "<" for oversold
signal_operator: ">"
threshold_start: 50.0        # Use 20.0 for oversold run
threshold_end: 80.0          # Use 50.0 for oversold run
threshold_step: 0.5

slippage_bps: 1.0
risk_free_rate: 0.0

date_range:
  start: "2015-01-01"
  end: "2026-03-15"
```

---

## Matching Composer.Trade Settings

| Setting | Value | Reason |
|---|---|---|
| `slippage_bps` | `1.0` | Composer's default slippage assumption |
| `risk_free_rate` | `0.0` | Composer assumes 0% risk-free rate |
| `date_range.start` | match Composer chart | Align backtest window exactly |

---

## Supported Indicators

| Name | Key | Notes |
|---|---|---|
| RSI (Wilder's smoothing) | `RSI` | Primary signal indicator |
| Simple Moving Average | `SMA` | Used in preconditions |
| Exponential Moving Average | `EMA` | Available |
| Cumulative Return | `CumRet` | Rolling % return over N days |